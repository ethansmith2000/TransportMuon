"""Transport Muon with an inner Procrustes rotation and trust-capped drift.

For a cached row-orthonormal factor Q and current matrix M, the best update that
stays in Q's row space is OQ, where O = polar(M Q.T) = polar((Q M.T).T).
This replaces the diagonal Jacobi approximation for the rotational component
with a polar solve on the much smaller square alignment matrix.

The remaining complementary-row-space correction still uses a damped diagonal
stretch approximation. Its step can be capped using a device-side estimate of
the proposed RMS motion, keeping large low-stretch corrections inside a trust
region without a host synchronization.
"""

from __future__ import annotations

import math

import torch

from muon_warm import (
    MuonWarm,
    _muon_ns5_prepared,
    _row_ns_retract,
)


def _small_polar(
    matrix: torch.Tensor,
    ns_steps: int,
    retract_steps: int,
    retract_method: str,
    eps: float,
) -> torch.Tensor:
    prepared = matrix / (matrix.norm() + float(eps))
    polar = _muon_ns5_prepared(prepared, ns_steps)
    # The normalized Muon polynomial maps singular values in [0, 1] below
    # 1.203, so the conservative Gershgorin cap is unnecessary here.
    return _row_ns_retract(
        polar,
        retract_steps,
        retract_method,
        False,
    )


def _positive_diagonal_inverse(
    diagonal: torch.Tensor,
    relative_damping: float,
    damping: str,
) -> torch.Tensor:
    floor = float(relative_damping) * (diagonal.abs().mean() + 1e-8)
    if damping == "tikhonov":
        return diagonal / (
            diagonal.square() + floor.square()
        ).clamp_min(1e-30)
    magnitude = diagonal.abs().clamp_min(floor)
    return torch.copysign(magnitude, diagonal).reciprocal()


@torch.compile
def _procrustes_transport_step(
    work_matrix: torch.Tensor,
    q_prev: torch.Tensor,
    alignment_matrix: torch.Tensor | None,
    eta: float,
    jacobi_eps: float,
    jacobi_damping: str,
    retract_method: str,
    retract_steps: int,
    inner_retract_method: str,
    inner_ns_steps: int,
    inner_retract_steps: int,
    max_normal_rms: float,
    alignment_tolerance: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q_prev.shape[0]
    r = (
        q_prev @ work_matrix.mT
        if alignment_matrix is None
        else alignment_matrix
    )
    # M Q.T = R.T. Its polar is the exact best rotation for the component of
    # M visible inside Q's current row space.
    rotation = _small_polar(
        r.mT,
        inner_ns_steps,
        inner_retract_steps,
        inner_retract_method,
        jacobi_eps,
    )
    rotated_alignment = 0.5 * (
        rotation @ r + (rotation @ r).mT
    )
    inverse_diagonal = _positive_diagonal_inverse(
        torch.diagonal(rotated_alignment),
        jacobi_eps,
        jacobi_damping,
    )

    # For row-orthonormal Q, diag(M(I-Q.T Q)M.T) can be obtained without
    # explicitly forming the residual or another large matrix product.
    residual_row_squares = (
        work_matrix.square().sum(dim=1, dtype=torch.float32)
        - r.mT.square().sum(dim=1, dtype=torch.float32)
    ).clamp_min(0.0)
    normal_rms = (
        inverse_diagonal.float().square() * residual_row_squares
    ).sum().sqrt() / math.sqrt(max(1, rows))

    effective_eta = work_matrix.new_tensor(float(eta), dtype=torch.float32)
    if max_normal_rms > 0.0:
        cap = effective_eta.new_tensor(float(max_normal_rms))
        effective_eta = torch.minimum(
            effective_eta,
            cap / normal_rms.clamp_min(1e-12),
        )
    eta_compute = effective_eta.to(work_matrix.dtype)

    identity = torch.eye(rows, device=work_matrix.device, dtype=work_matrix.dtype)
    coefficient = (
        identity
        - eta_compute
        * inverse_diagonal.unsqueeze(1)
        * rotated_alignment.mT
    ) @ rotation
    q_tilde = (
        eta_compute * inverse_diagonal.unsqueeze(1) * work_matrix
        + coefficient @ q_prev
    )
    q_next = _row_ns_retract(q_tilde, retract_steps, retract_method)

    accepted = effective_eta.new_tensor(1.0)
    if alignment_tolerance >= 0.0:
        old_alignment = (q_prev.float() * work_matrix.float()).sum()
        new_alignment = (q_next.float() * work_matrix.float()).sum()
        accept = new_alignment + (
            float(alignment_tolerance) * old_alignment.abs()
        ) >= old_alignment
        q_next = torch.where(accept, q_next, q_prev)
        accepted = accept.to(torch.float32)
    stats = torch.stack(
        (
            effective_eta,
            normal_rms,
            effective_eta * normal_rms,
            accepted,
        )
    )
    return q_next, stats


class MuonWarmProcrustes(MuonWarm):
    """MuonWarm whose in-space transport uses a small Procrustes solve.

    Extra arguments:
        muon_warm_inner_ns_steps: Newton--Schulz steps for the square alignment
            polar solve. This work scales with the cube of the smaller matrix
            dimension rather than with the full rectangular matrix.
        muon_warm_inner_retract_steps: Polishing steps for that square polar.
        muon_warm_max_normal_rms: Maximum normalized Frobenius motion from the
            complementary row-space correction. Zero disables the trust cap.

    ``muon_warm_lr`` controls the complementary correction. The Procrustes
    rotation is applied fully because it is the exact optimum within the current
    row space rather than an Euler approximation.
    """

    def __init__(
        self,
        params,
        *,
        muon_warm_inner_ns_steps: int = 5,
        muon_warm_inner_retract_steps: int = 2,
        muon_warm_max_normal_rms: float = 0.0,
        **kwargs,
    ):
        if int(muon_warm_inner_ns_steps) < 1:
            raise ValueError("muon_warm_inner_ns_steps must be >= 1")
        if int(muon_warm_inner_retract_steps) < 1:
            raise ValueError("muon_warm_inner_retract_steps must be >= 1")
        if float(muon_warm_max_normal_rms) < 0.0:
            raise ValueError("muon_warm_max_normal_rms must be non-negative")
        self.muon_warm_inner_ns_steps = int(muon_warm_inner_ns_steps)
        self.muon_warm_inner_retract_steps = int(
            muon_warm_inner_retract_steps
        )
        self.muon_warm_max_normal_rms = float(muon_warm_max_normal_rms)
        super().__init__(params, **kwargs)

    def _compute_warm_direction(
        self,
        tracking_matrix,
        q_prev,
        alignment_matrix,
        state,
        transposed,
    ):
        q_next, stats = _procrustes_transport_step(
            tracking_matrix,
            q_prev,
            alignment_matrix,
            float(self.muon_warm_lr),
            float(self.muon_warm_jacobi_eps),
            self.muon_warm_jacobi_damping,
            self.muon_warm_retract_method,
            max(1, int(self.muon_warm_retract_steps)),
            self.muon_warm_anchor_retract_method,
            self.muon_warm_inner_ns_steps,
            self.muon_warm_inner_retract_steps,
            self.muon_warm_max_normal_rms,
            self.muon_warm_alignment_tolerance,
        )
        state["muon_warm_effective_eta_tensor"] = stats[0]
        state["muon_warm_normal_rms_tensor"] = stats[1]
        state["muon_warm_applied_normal_rms_tensor"] = stats[2]
        state["muon_warm_step_accepted_tensor"] = stats[3]
        return q_next


TransportProcrustes = MuonWarmProcrustes


__all__ = ["MuonWarmProcrustes", "TransportProcrustes"]
