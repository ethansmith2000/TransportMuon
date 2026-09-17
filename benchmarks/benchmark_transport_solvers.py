"""Compare diagonal-Jacobi and inner-Procrustes warm transports."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "muon" not in sys.modules:
    stub = types.ModuleType("muon")
    stub.adam_update = lambda *args, **kwargs: args[0]
    sys.modules["muon"] = stub

from muon_warm import _warm_polar_jacobi_step  # noqa: E402
from muon_warm_procrustes import _procrustes_transport_step  # noqa: E402


def _elapsed_ms(function, repetitions: int, device: torch.device) -> float:
    for _ in range(2):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / repetitions


def _pair_rotation(size: int, angle: float, device: torch.device) -> torch.Tensor:
    rotation = torch.eye(size, device=device)
    first = torch.arange(0, size - 1, 2, device=device)
    second = first + 1
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation[first, first] = cosine
    rotation[second, second] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    return rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--angle", type=float, default=0.05)
    parser.add_argument("--condition", type=float, default=10.0)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--normal-cap", type=float, default=0.0)
    parser.add_argument("--inner-retract-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(89)

    for rows in args.rows:
        identity = torch.eye(rows, device=device)
        q_previous = torch.cat((identity, torch.zeros_like(identity)), dim=1)
        row_rotation = _pair_rotation(rows, args.angle, device)
        subspace_rotation = torch.cat(
            (
                math.cos(args.angle) * identity,
                math.sin(args.angle) * identity,
            ),
            dim=1,
        )
        target = row_rotation @ subspace_rotation
        eigenvectors = torch.linalg.qr(torch.randn(rows, rows, device=device)).Q
        eigenvalues = torch.logspace(
            0.0,
            -math.log10(args.condition),
            rows,
            device=device,
        )
        stretch = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        work_matrix = stretch @ target
        work_matrix = (work_matrix / work_matrix.norm()).to(torch.bfloat16)
        q_previous = q_previous.to(torch.bfloat16)

        def jacobi():
            return _warm_polar_jacobi_step(
                work_matrix,
                q_previous,
                1.0,
                1e-3,
                "higham_cubic",
                1,
                True,
                None,
                "tikhonov",
            )

        def procrustes():
            return _procrustes_transport_step(
                work_matrix,
                q_previous,
                None,
                1.0,
                1e-3,
                "tikhonov",
                "higham_cubic",
                1,
                "higham_cubic",
                5,
                args.inner_retract_steps,
                args.normal_cap,
                -1.0,
            )

        jacobi_result = jacobi().float()
        procrustes_result, stats = procrustes()
        target_norm = target.norm()
        print(
            json.dumps(
                {
                    "rows": rows,
                    "columns": 2 * rows,
                    "condition": args.condition,
                    "angle": args.angle,
                    "inner_retract_steps": args.inner_retract_steps,
                    "jacobi_ms": _elapsed_ms(
                        jacobi, args.repetitions, device
                    ),
                    "procrustes_ms": _elapsed_ms(
                        procrustes, args.repetitions, device
                    ),
                    "jacobi_relative_error": float(
                        (jacobi_result - target).norm() / target_norm
                    ),
                    "procrustes_relative_error": float(
                        (procrustes_result.float() - target).norm()
                        / target_norm
                    ),
                    "effective_eta": float(stats[0]),
                    "normal_rms": float(stats[1]),
                    "applied_normal_rms": float(stats[2]),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
