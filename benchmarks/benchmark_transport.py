"""Microbenchmark the two warm transports against a fresh NS anchor."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# This standalone repository imports adam_update from a surrounding Muon
# implementation. The benchmark only exercises polar routines.
if "muon" not in sys.modules:
    stub = types.ModuleType("muon")
    stub.adam_update = lambda *args, **kwargs: args[0]
    sys.modules["muon"] = stub

from muon_warm import (  # noqa: E402
    _muon_ns5_prepared,
    _polar_express_prepared,
    _prepare_muon_matrix,
    _row_ns_retract,
    _warm_polar_jacobi_step,
    _warm_polar_jacobi_step_with_stats,
)


def _elapsed_ms(function, repetitions: int, device: torch.device) -> float:
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--angle", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--retract-method",
        choices=("higham_cubic", "quadratic"),
        default="higham_cubic",
    )
    parser.add_argument("--warm-retract-steps", type=int, default=1)
    parser.add_argument("--anchor-retract-steps", type=int, default=2)
    parser.add_argument(
        "--polar-method",
        choices=("muon", "polar_express"),
        default="muon",
    )
    parser.add_argument("--polar-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    polar_steps = args.polar_steps
    if polar_steps is None:
        polar_steps = 5 if args.polar_method == "muon" else 6
    polar_function = (
        _muon_ns5_prepared
        if args.polar_method == "muon"
        else _polar_express_prepared
    )
    device = torch.device(args.device)

    for rows in args.rows:
        identity = torch.eye(rows, device=device, dtype=torch.bfloat16)
        q_previous = torch.cat((identity, torch.zeros_like(identity)), dim=1)
        target = torch.cat(
            (
                math.cos(args.angle) * identity,
                math.sin(args.angle) * identity,
            ),
            dim=1,
        )
        work_matrix, _ = _prepare_muon_matrix(target)

        def warm(track_subspace: bool):
            return _warm_polar_jacobi_step(
                work_matrix,
                q_previous,
                1.0,
                1e-3,
                args.retract_method,
                args.warm_retract_steps,
                track_subspace,
                None,
                "floor",
            )

        def controlled_warm():
            return _warm_polar_jacobi_step_with_stats(
                work_matrix,
                q_previous,
                1.0,
                1e-3,
                args.retract_method,
                args.warm_retract_steps,
                True,
                None,
                "floor",
                0.05,
                1e-4,
            )

        rotation_only = warm(False).float()
        full_tangent = warm(True).float()
        fresh_anchor = _row_ns_retract(
            polar_function(work_matrix, polar_steps),
            args.anchor_retract_steps,
            args.retract_method,
            False,
        ).float()
        target_float = target.float()
        record = {
            "rows": rows,
            "columns": 2 * rows,
            "retract_method": args.retract_method,
            "warm_retract_steps": args.warm_retract_steps,
            "anchor_retract_steps": args.anchor_retract_steps,
            "polar_method": args.polar_method,
            "polar_steps": polar_steps,
            "rotation_only_relative_error": float(
                (rotation_only - target_float).norm() / target_float.norm()
            ),
            "full_tangent_relative_error": float(
                (full_tangent - target_float).norm() / target_float.norm()
            ),
            "fresh_anchor_relative_error": float(
                (fresh_anchor - target_float).norm() / target_float.norm()
            ),
            "rotation_only_ms": _elapsed_ms(
                lambda: warm(False), args.repetitions, device
            ),
            "full_tangent_ms": _elapsed_ms(
                lambda: warm(True), args.repetitions, device
            ),
            "controlled_tangent_ms": _elapsed_ms(
                controlled_warm, args.repetitions, device
            ),
            "fresh_anchor_ms": _elapsed_ms(
                lambda: _row_ns_retract(
                    polar_function(work_matrix, polar_steps),
                    args.anchor_retract_steps,
                    args.retract_method,
                    False,
                ),
                args.repetitions,
                device,
            ),
        }
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
