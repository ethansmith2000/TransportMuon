"""Compare Transport Muon and leverage-balanced warm/anchor directions."""

import argparse
import json
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

from muon_warm import MuonWarm, _prepare_muon_matrix  # noqa: E402
from muon_warm_aurora import MuonWarmAurora  # noqa: E402


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


def _leverage_cv(prepared_factor: torch.Tensor) -> float:
    leverage = prepared_factor.float().square().sum(dim=0)
    return float(leverage.std(unbiased=False) / leverage.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(73)

    for rows in args.rows:
        columns = min(args.columns, rows - 1)
        parameter = torch.nn.Parameter(
            torch.zeros(rows, columns, device=device, dtype=torch.bfloat16)
        )
        base = MuonWarm(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_full_ns_steps=0,
        )
        aurora = MuonWarmAurora(
            [{"params": [parameter], "use_muon": True}],
            muon_warm_full_ns_steps=0,
            aurora_anchor_iterations=2,
        )

        gradient = torch.randn_like(parameter)
        gradient[: rows // 4].mul_(0.05)
        work_matrix, transposed = _prepare_muon_matrix(gradient)
        base_anchor = base._compute_anchor_direction(
            work_matrix, {}, transposed
        )
        aurora_state = {"aurora_balance_enabled": True}
        aurora_anchor = aurora._compute_anchor_direction(
            work_matrix, aurora_state, transposed
        )

        perturbed = gradient + 0.01 * torch.randn_like(gradient)
        next_matrix, _ = _prepare_muon_matrix(perturbed)

        def base_warm():
            return base._compute_warm_direction(
                next_matrix, base_anchor, None, {}, transposed
            )

        def aurora_warm():
            state = {
                "step": 2,
                "aurora_balance_enabled": True,
                "aurora_column_scale": aurora_state[
                    "aurora_column_scale"
                ].clone(),
            }
            tracking = aurora._prepare_tracking_matrix(
                next_matrix, aurora_anchor, state, transposed
            )
            return aurora._compute_warm_direction(
                tracking, aurora_anchor, None, state, transposed
            )

        record = {
            "rows": rows,
            "columns": columns,
            "muon_anchor_ms": _elapsed_ms(
                lambda: base._compute_anchor_direction(
                    work_matrix, {}, transposed
                ),
                args.repetitions,
                device,
            ),
            "aurora_anchor_ms": _elapsed_ms(
                lambda: aurora._compute_anchor_direction(
                    work_matrix,
                    {"aurora_balance_enabled": True},
                    transposed,
                ),
                args.repetitions,
                device,
            ),
            "muon_warm_ms": _elapsed_ms(
                base_warm, args.repetitions, device
            ),
            "aurora_warm_ms": _elapsed_ms(
                aurora_warm, args.repetitions, device
            ),
            "muon_anchor_leverage_cv": _leverage_cv(base_anchor),
            "aurora_anchor_leverage_cv": _leverage_cv(aurora_anchor),
            "aurora_warm_leverage_cv": _leverage_cv(aurora_warm()),
        }
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
