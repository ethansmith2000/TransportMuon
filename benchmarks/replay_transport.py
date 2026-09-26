"""Deterministic replay for fixed and adaptive Transport Muon variants.

This isolates polar tracking from model-training noise. It reports reference
error, angular and row-space error, orthogonality, anchors, warm-step acceptance,
and elapsed update time for controlled matrix trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import muon_warm as warm  # noqa: E402
import muon_warm_procrustes as procrustes  # noqa: E402


def _eager(function):
    return getattr(function, "_torchdynamo_orig_callable", function)


def _orthonormal_rows(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    return torch.linalg.qr(
        torch.randn(columns, rows, generator=generator), mode="reduced"
    ).Q.T


def build_trajectory(
    kind: str,
    rows: int,
    columns: int,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    """Construct full-rank matrices with independently controlled polar drift."""
    if columns < 2 * rows:
        raise ValueError("replay requires columns >= 2 * rows for controlled row-space drift")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    q_zero = _orthonormal_rows(rows, columns, generator)
    normal_seed = torch.randn(rows, columns, generator=generator)
    normal_seed -= (normal_seed @ q_zero.T) @ q_zero
    q_normal = torch.linalg.qr(normal_seed.T, mode="reduced").Q.T
    skew_seed = torch.randn(rows, rows, generator=generator)
    skew = skew_seed - skew_seed.T
    skew /= skew.norm().clamp_min(1e-12)
    stretch_basis = torch.linalg.qr(
        torch.randn(rows, rows, generator=generator)
    ).Q

    trajectory: list[torch.Tensor] = []
    for step in range(steps):
        progress = step / max(1, steps - 1)
        if kind == "slow":
            row_angle = 0.35 * progress
            angular_angle = 0.25 * progress
            minimum_stretch = 0.35
        elif kind == "angular":
            row_angle = 0.0
            angular_angle = 0.35 * progress
            minimum_stretch = 0.35
        elif kind == "normal":
            row_angle = 0.35 * progress
            angular_angle = 0.0
            minimum_stretch = 0.35
        elif kind == "shock":
            row_angle = 0.08 * progress + (0.35 if progress >= 0.5 else 0.0)
            angular_angle = 0.06 * progress + (0.30 if progress >= 0.5 else 0.0)
            minimum_stretch = 0.35
        elif kind == "rank_stress":
            row_angle = 0.20 * progress
            angular_angle = 0.15 * progress
            minimum_stretch = max(0.01, 0.6 * (1.0 - progress))
        else:
            raise ValueError(f"unknown trajectory: {kind!r}")

        row_space = math.cos(row_angle) * q_zero + math.sin(row_angle) * q_normal
        left_rotation = torch.matrix_exp(angular_angle * skew)
        polar = left_rotation @ row_space
        singular_values = torch.linspace(minimum_stretch, 1.8, rows)
        stretch = stretch_basis @ torch.diag(singular_values) @ stretch_basis.T
        trajectory.append(stretch @ polar)
    return trajectory


def _reference_polar(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    return left @ right


def _anchor(
    matrix: torch.Tensor,
    ns_steps: int,
    retract_steps: int,
    method: str,
) -> torch.Tensor:
    prepared = matrix.to(torch.bfloat16)
    prepared = prepared / (prepared.norm() + 1e-7)
    if method == "muon":
        candidate = _eager(warm._muon_ns5_prepared)(prepared, ns_steps)
    elif method == "polar_express":
        candidate = _eager(warm._polar_express_prepared)(prepared, ns_steps)
    else:
        raise ValueError(f"unknown anchor method: {method!r}")
    return warm._row_ns_retract(
        candidate, retract_steps, "higham_cubic", False
    )


def _tracking_errors(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    qf = q.float()
    alignment = qf @ reference.T
    skew = 0.5 * (alignment - alignment.T)
    normal = reference - (reference @ qf.T) @ qf
    identity = torch.eye(qf.shape[0], device=qf.device)
    scale = math.sqrt(max(1, qf.shape[0]))
    return torch.stack(
        (
            (qf - reference).norm() / scale,
            skew.norm() / alignment.norm().clamp_min(1e-12),
            normal.norm() / scale,
            (qf @ qf.T - identity).norm() / scale,
        )
    )


def run_method(
    matrices: list[torch.Tensor],
    method: str,
    device: torch.device,
    anchor_every: int,
    adaptive_error: float,
    adaptive_min_stretch: float,
    adaptive_max_age: int,
    adaptive_check_every: int,
    warm_retract_steps: int,
    anchor_ns_steps: int,
    anchor_retract_steps: int,
    anchor_method: str,
    tangent_cap: float,
    normal_cap: float,
    alignment_tolerance: float,
    angular_scale: float,
    normal_scale: float,
    warm_retract_every: int,
) -> dict:
    q = None
    age = 0
    anchors = 0
    accepted_values: list[torch.Tensor] = []
    warm_steps = 0
    warm_retractions = 0
    effective_etas: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    anchor_reasons = {"initial": 0, "schedule": 0, "error": 0, "age": 0}
    event_pairs = []
    cpu_update_ms = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()

    for step, host_matrix in enumerate(matrices):
        matrix = host_matrix.to(device)
        reference = _reference_polar(matrix)
        if device.type == "cuda":
            update_start = torch.cuda.Event(enable_timing=True)
            update_end = torch.cuda.Event(enable_timing=True)
            update_start.record()
        else:
            update_start = time.perf_counter()
        prepared = matrix.to(torch.bfloat16)
        prepared = prepared / (prepared.norm() + 1e-7)
        reason = None
        if q is None:
            reason = "initial"
        elif method == "fresh_matched":
            reason = "schedule"
        elif method.endswith("fixed") and anchor_every > 0 and step % anchor_every == 0:
            reason = "schedule"
        elif method.endswith("adaptive"):
            if step % adaptive_check_every == 0:
                _, metrics = _eager(warm._warm_alignment_metrics)(prepared, q)
                if bool(
                    (torch.maximum(metrics[0], metrics[1]) > adaptive_error)
                    | (metrics[3] < adaptive_min_stretch)
                ):
                    reason = "error"
            if reason is None and adaptive_max_age > 0 and age >= adaptive_max_age:
                reason = "age"

        if reason is not None:
            q = _anchor(
                matrix.to(device), anchor_ns_steps, anchor_retract_steps, anchor_method
            )
            anchors += 1
            age = 0
            anchor_reasons[reason] += 1
        else:
            warm_steps += 1
            age += 1
            current_retract_steps = (
                warm_retract_steps if age % warm_retract_every == 0 else 0
            )
            warm_retractions += int(current_retract_steps > 0)
            if method.startswith("jacobi"):
                q, stats = _eager(warm._warm_polar_jacobi_step_with_stats)(
                    prepared,
                    q,
                    1.0,
                    1e-3,
                    "higham_cubic",
                    current_retract_steps,
                    True,
                    None,
                    "tikhonov",
                    tangent_cap,
                    alignment_tolerance,
                    None,
                    angular_scale,
                    normal_scale,
                )
                effective_etas.append(stats[0])
                accepted_values.append(stats[2])
            elif method.startswith("procrustes"):
                q, stats = _eager(procrustes._procrustes_transport_step)(
                    prepared,
                    q,
                    None,
                    normal_scale,
                    1e-3,
                    "tikhonov",
                    "higham_cubic",
                    current_retract_steps,
                    "higham_cubic",
                    5,
                    2,
                    True,
                    normal_cap,
                    alignment_tolerance,
                )
                effective_etas.append(stats[0])
                accepted_values.append(stats[3])
            else:
                raise ValueError(f"unknown method: {method!r}")
        if device.type == "cuda":
            update_end.record()
            event_pairs.append((update_start, update_end))
        else:
            cpu_update_ms += 1000.0 * (time.perf_counter() - update_start)
        errors.append(_tracking_errors(q, reference))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        update_elapsed_ms = sum(
            start.elapsed_time(end) for start, end in event_pairs
        )
    else:
        update_elapsed_ms = cpu_update_ms
    wall_elapsed_ms = 1000.0 * (time.perf_counter() - wall_start)
    error_tensor = torch.stack(errors).cpu()
    eta = torch.stack(effective_etas).float().mean().item() if effective_etas else 1.0
    accepted_fraction = (
        torch.stack(accepted_values).float().mean().item()
        if accepted_values
        else 1.0
    )
    labels = ("polar_error", "angular_error", "normal_error", "orthogonality_error")
    result = {
        "method": method,
        "update_elapsed_ms": update_elapsed_ms,
        "update_milliseconds_per_step": update_elapsed_ms / len(matrices),
        "wall_elapsed_ms": wall_elapsed_ms,
        "wall_milliseconds_per_step": wall_elapsed_ms / len(matrices),
        "anchors": anchors,
        "anchor_reasons": anchor_reasons,
        "warm_steps": warm_steps,
        "warm_retractions": warm_retractions,
        "accepted_fraction": accepted_fraction,
        "mean_effective_eta": eta,
        "retraction_iterations": (
            warm_retractions * warm_retract_steps + anchors * anchor_retract_steps
        ),
        "full_ns_iterations": anchors * anchor_ns_steps,
    }
    for index, label in enumerate(labels):
        result[f"mean_{label}"] = float(error_tensor[:, index].mean())
        result[f"max_{label}"] = float(error_tensor[:, index].max())
        result[f"final_{label}"] = float(error_tensor[-1, index])
    return result


def replay(args: argparse.Namespace) -> dict:
    if args.adaptive_check_every < 1:
        raise ValueError("adaptive_check_every must be >= 1")
    if args.warm_retract_every < 1:
        raise ValueError("warm_retract_every must be >= 1")
    device = torch.device(args.device)
    kinds = ("slow", "shock", "rank_stress") if args.trajectory == "all" else (args.trajectory,)
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "trajectories": [],
    }
    # The Procrustes inner solve imports the compiled anchor function by name.
    # Use its eager body so timing excludes first-use compilation from every run.
    procrustes._muon_ns5_prepared = _eager(warm._muon_ns5_prepared)
    for kind in kinds:
        matrices = build_trajectory(
            kind, args.rows, args.columns, args.steps, args.seed
        )
        # Initialize CUDA libraries and each update path before recording the
        # first method. This avoids assigning one-time startup to a baseline.
        if device.type == "cuda":
            matrix = matrices[0].to(device)
            reference = _reference_polar(matrix)
            prepared = matrix.to(torch.bfloat16)
            prepared = prepared / (prepared.norm() + 1e-7)
            q = _anchor(
                matrix,
                args.anchor_ns_steps,
                args.anchor_retract_steps,
                args.anchor_method,
            )
            _eager(warm._warm_alignment_metrics)(prepared, q)
            _eager(warm._warm_polar_jacobi_step_with_stats)(
                prepared,
                q,
                1.0,
                1e-3,
                "higham_cubic",
                args.warm_retract_steps,
                True,
                None,
                "tikhonov",
                args.tangent_cap,
                args.alignment_tolerance,
                None,
                args.angular_scale,
                args.normal_scale,
            )
            _eager(procrustes._procrustes_transport_step)(
                prepared,
                q,
                None,
                args.normal_scale,
                1e-3,
                "tikhonov",
                "higham_cubic",
                args.warm_retract_steps,
                "higham_cubic",
                5,
                2,
                True,
                args.normal_cap,
                args.alignment_tolerance,
            )
            _tracking_errors(q, reference)
            torch.cuda.synchronize(device)
        methods = []
        for method in (
            "fresh_matched",
            "jacobi_fixed",
            "jacobi_adaptive",
            "procrustes_fixed",
            "procrustes_adaptive",
        ):
            methods.append(
                run_method(
                    matrices,
                    method,
                    device,
                    args.anchor_every,
                    args.adaptive_error,
                    args.adaptive_min_stretch,
                    args.adaptive_max_age,
                    args.adaptive_check_every,
                    args.warm_retract_steps,
                    args.anchor_ns_steps,
                    args.anchor_retract_steps,
                    args.anchor_method,
                    args.tangent_cap,
                    args.normal_cap,
                    args.alignment_tolerance,
                    args.angular_scale,
                    args.normal_scale,
                    args.warm_retract_every,
                )
            )
        output["trajectories"].append({"name": kind, "methods": methods})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--trajectory",
        choices=("all", "slow", "angular", "normal", "shock", "rank_stress"),
        default="all",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--anchor-every", type=int, default=8)
    parser.add_argument("--adaptive-error", type=float, default=0.08)
    parser.add_argument("--adaptive-min-stretch", type=float, default=0.05)
    parser.add_argument("--adaptive-max-age", type=int, default=32)
    parser.add_argument("--adaptive-check-every", type=int, default=1)
    parser.add_argument("--warm-retract-steps", type=int, default=1)
    parser.add_argument("--warm-retract-every", type=int, default=1)
    parser.add_argument("--anchor-ns-steps", type=int, default=5)
    parser.add_argument("--anchor-retract-steps", type=int, default=2)
    parser.add_argument(
        "--anchor-method", choices=("muon", "polar_express"), default="muon"
    )
    parser.add_argument("--tangent-cap", type=float, default=0.0)
    parser.add_argument("--normal-cap", type=float, default=0.0)
    parser.add_argument("--angular-scale", type=float, default=1.0)
    parser.add_argument("--normal-scale", type=float, default=1.0)
    parser.add_argument("--alignment-tolerance", type=float, default=-1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = replay(arguments)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n")
