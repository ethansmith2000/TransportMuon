"""Experimental leverage-balanced Transport Muon for tall matrices.

Aurora changes the tall-matrix Muon objective by asking for both orthonormal
columns and uniform row leverage. ``MuonWarm`` transposes tall matrices before
tracking their polar factor, so uniform row leverage becomes uniform column
norms in the prepared representation used here.

This module keeps a diagonal column scaling for the momentum matrix. Full
anchors refine that scaling with multiple polar projections; warm steps update
it from the cached factor and use the usual rectangular tangent transport.
"""

from __future__ import annotations

import torch

from muon_warm import (
    MuonWarm,
    _polar_solve_prepared,
    _row_ns_retract,
)


def _column_squares(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.square().sum(dim=0, dtype=torch.float32)


def _normalize_scale(scale: torch.Tensor, limit: float) -> torch.Tensor:
    """Remove irrelevant global scale and bound extreme diagonal ratios."""
    scale = scale.float().clamp_min(1e-12)
    scale = scale / scale.log().mean().exp()
    if limit > 0.0:
        scale = scale.clamp(min=1.0 / float(limit), max=float(limit))
    return scale


def _initial_column_scale(
    work_matrix: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    scale = _column_squares(work_matrix).clamp_min(1e-24).rsqrt()
    return _normalize_scale(scale, limit)


def _update_column_scale(
    scale: torch.Tensor,
    polar_factor: torch.Tensor,
    power: float,
    limit: float,
) -> torch.Tensor:
    if power == 0.0:
        return scale
    target_column_square = polar_factor.shape[0] / polar_factor.shape[1]
    column_squares = _column_squares(polar_factor).clamp_min(1e-24)
    correction = (target_column_square / column_squares).pow(float(power))
    return _normalize_scale(scale * correction, limit)


def _apply_column_scale(
    work_matrix: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    scaled = work_matrix * scale.to(work_matrix.dtype).unsqueeze(0)
    return scaled / (scaled.norm() + float(eps))


def _record_leverage_metrics(state: dict, polar_factor: torch.Tensor) -> None:
    column_squares = _column_squares(polar_factor)
    mean = column_squares.mean().clamp_min(1e-12)
    state["aurora_leverage_cv_tensor"] = (
        column_squares.std(unbiased=False) / mean
    )
    state["aurora_min_relative_leverage_tensor"] = column_squares.amin() / mean
    state["aurora_max_relative_leverage_tensor"] = column_squares.amax() / mean


class MuonWarmAurora(MuonWarm):
    """Transport Muon with Aurora-style leverage balancing on tall 2D tensors.

    Extra arguments:
        aurora_anchor_iterations: Alternating diagonal-scale/polar refinements
            performed on a full anchor.
        aurora_balance_power: Diagonal correction exponent between anchor
            refinements. Aurora's reference value is ``0.5``.
        aurora_warm_balance_power: Correction exponent applied from the cached
            factor before a warm transport step. Smaller values change the
            target more smoothly over optimizer steps.
        aurora_balance_every: Warm steps between diagonal-scale corrections.
        aurora_scale_limit: Maximum diagonal scale or inverse scale after
            removing its irrelevant geometric mean.
        aurora_eps: Numerical floor used for column norms and normalization.
        aurora_min_aspect_ratio: Apply balancing only when the original matrix
            has at least this many rows per column.
        aurora_apply_to_convs: Also balance 4D tensors after Muon's flattening.
    """

    def __init__(
        self,
        params,
        *,
        aurora_anchor_iterations: int = 2,
        aurora_balance_power: float = 0.5,
        aurora_warm_balance_power: float = 0.25,
        aurora_balance_every: int = 1,
        aurora_scale_limit: float = 1e3,
        aurora_eps: float = 1e-7,
        aurora_min_aspect_ratio: float = 1.0,
        aurora_apply_to_convs: bool = False,
        **kwargs,
    ):
        if int(aurora_anchor_iterations) < 1:
            raise ValueError("aurora_anchor_iterations must be >= 1")
        if not 0.0 < float(aurora_balance_power) <= 1.0:
            raise ValueError("aurora_balance_power must be in (0, 1]")
        if not 0.0 <= float(aurora_warm_balance_power) <= 1.0:
            raise ValueError("aurora_warm_balance_power must be in [0, 1]")
        if int(aurora_balance_every) < 1:
            raise ValueError("aurora_balance_every must be >= 1")
        if float(aurora_scale_limit) < 1.0:
            raise ValueError("aurora_scale_limit must be >= 1")
        if float(aurora_eps) <= 0.0:
            raise ValueError("aurora_eps must be positive")
        if float(aurora_min_aspect_ratio) < 1.0:
            raise ValueError("aurora_min_aspect_ratio must be >= 1")

        self.aurora_anchor_iterations = int(aurora_anchor_iterations)
        self.aurora_balance_power = float(aurora_balance_power)
        self.aurora_warm_balance_power = float(aurora_warm_balance_power)
        self.aurora_balance_every = int(aurora_balance_every)
        self.aurora_scale_limit = float(aurora_scale_limit)
        self.aurora_eps = float(aurora_eps)
        self.aurora_min_aspect_ratio = float(aurora_min_aspect_ratio)
        self.aurora_apply_to_convs = bool(aurora_apply_to_convs)
        super().__init__(params, **kwargs)

    def _muon_update_warm(self, grad, state, group):
        rows, columns = grad.shape[0], grad.numel() // grad.shape[0]
        eligible_ndim = grad.ndim == 2 or (
            grad.ndim == 4 and self.aurora_apply_to_convs
        )
        state["aurora_balance_enabled"] = bool(
            eligible_ndim
            and rows > columns
            and rows / columns >= self.aurora_min_aspect_ratio
        )
        return super()._muon_update_warm(grad, state, group)

    @staticmethod
    def _uses_aurora(state: dict, transposed: bool) -> bool:
        return bool(transposed and state.get("aurora_balance_enabled", False))

    def _prepare_tracking_matrix(
        self,
        work_matrix,
        q_prev,
        state,
        transposed,
    ):
        if not self._uses_aurora(state, transposed):
            return work_matrix

        scale = state.get("aurora_column_scale")
        if scale is None or scale.shape[0] != work_matrix.shape[1]:
            scale = _initial_column_scale(
                work_matrix,
                self.aurora_scale_limit,
            )
        elif state["step"] % self.aurora_balance_every == 0:
            scale = _update_column_scale(
                scale,
                q_prev,
                self.aurora_warm_balance_power,
                self.aurora_scale_limit,
            )
        state["aurora_column_scale"] = scale
        return _apply_column_scale(
            work_matrix,
            scale,
            self.aurora_eps,
        )

    def _compute_anchor_direction(self, work_matrix, state, transposed):
        if not self._uses_aurora(state, transposed):
            return super()._compute_anchor_direction(
                work_matrix,
                state,
                transposed,
            )

        scale = _initial_column_scale(work_matrix, self.aurora_scale_limit)
        q_next = None
        for iteration in range(self.aurora_anchor_iterations):
            tracking_matrix = _apply_column_scale(
                work_matrix,
                scale,
                self.aurora_eps,
            )
            q_next = _polar_solve_prepared(
                tracking_matrix,
                self.muon_ns_steps,
                self.muon_polar_method,
            )
            q_next = _row_ns_retract(
                q_next,
                self.muon_warm_anchor_retract_steps,
                self.muon_warm_anchor_retract_method,
                False,
            )
            if iteration + 1 < self.aurora_anchor_iterations:
                scale = _update_column_scale(
                    scale,
                    q_next,
                    self.aurora_balance_power,
                    self.aurora_scale_limit,
                )

        state["aurora_column_scale"] = scale
        _record_leverage_metrics(state, q_next)
        return q_next

    def _compute_warm_direction(
        self,
        tracking_matrix,
        q_prev,
        alignment_matrix,
        state,
        transposed,
    ):
        q_next = super()._compute_warm_direction(
            tracking_matrix,
            q_prev,
            alignment_matrix,
            state,
            transposed,
        )
        if self._uses_aurora(state, transposed):
            _record_leverage_metrics(state, q_next)
        return q_next


TransportAurora = MuonWarmAurora


__all__ = ["MuonWarmAurora", "TransportAurora"]
