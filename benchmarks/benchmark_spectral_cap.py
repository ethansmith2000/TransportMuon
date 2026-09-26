"""Stress the heuristic power cap against exact singular values."""

import argparse
import json
import math
from pathlib import Path
import sys
import types

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "muon" not in sys.modules:
    stub = types.ModuleType("muon")
    stub.adam_update = lambda *args, **kwargs: args[0]
    sys.modules["muon"] = stub

from muon_warm import (  # noqa: E402
    ROW_NS_MAX_SINGULAR,
    _power_spectral_cap,
)


def _orthogonal(size: int, generator: torch.Generator, device: torch.device):
    matrix = torch.randn(size, size, generator=generator, device="cpu")
    return torch.linalg.qr(matrix).Q.to(device)


def run_trajectory(
    rows: int,
    steps: int,
    power_steps: int,
    safety_factor: float,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(seed + rows)
    basis = _orthogonal(rows, generator, device)
    right = torch.linalg.qr(
        torch.randn(2 * rows, rows, generator=generator, device="cpu")
    ).Q.T.to(device)
    skew_seed = torch.randn(rows, rows, generator=generator, device="cpu")
    skew = (skew_seed - skew_seed.T).to(device)
    skew = 0.03 * skew / skew.norm().clamp_min(1e-12)
    slow_rotation = torch.linalg.matrix_exp(skew)
    cached = None
    records = []

    for step in range(steps):
        shock = step == steps // 2
        if shock:
            basis = _orthogonal(rows, generator, device)
        else:
            basis = slow_rotation @ basis
        phase = 2.0 * math.pi * step / max(1, steps - 1)
        top = 1.05 + 0.75 * (0.5 + 0.5 * math.sin(phase))
        singular_values = torch.linspace(0.65, top, rows, device=device)
        candidate = (
            basis @ torch.diag(singular_values) @ right
        ).to(torch.bfloat16)
        gram = candidate @ candidate.mT
        exact = torch.linalg.svdvals(candidate.float())[0]
        gershgorin = gram.abs().sum(dim=-1).amax().float().sqrt()
        _, cached, estimate, scale = _power_spectral_cap(
            gram,
            cached,
            power_steps,
            safety_factor,
        )
        records.append(
            {
                "step": step,
                "shock": shock,
                "exact_top_singular": float(exact),
                "gershgorin_bound": float(gershgorin),
                "power_estimate": float(estimate),
                "power_safe_estimate": float(safety_factor * estimate),
                "power_input_scale": float(scale),
                "post_cap_top_singular": float(exact * scale),
                "violates_retraction_limit": bool(
                    exact * scale > ROW_NS_MAX_SINGULAR * 1.001
                ),
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--power-steps", type=int, default=2)
    parser.add_argument("--safety-factor", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)

    output = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "retraction_limit": ROW_NS_MAX_SINGULAR,
        "sizes": {},
    }
    for rows in args.rows:
        records = run_trajectory(
            rows,
            args.steps,
            args.power_steps,
            args.safety_factor,
            device,
            args.seed,
        )
        ratios = [
            record["power_safe_estimate"] / record["exact_top_singular"]
            for record in records
        ]
        output["sizes"][str(rows)] = {
            "steps": args.steps,
            "violations": sum(
                record["violates_retraction_limit"] for record in records
            ),
            "minimum_safe_estimate_ratio": min(ratios),
            "mean_safe_estimate_ratio": sum(ratios) / len(ratios),
            "maximum_post_cap_top_singular": max(
                record["post_cap_top_singular"] for record in records
            ),
            "mean_gershgorin_over_exact": sum(
                record["gershgorin_bound"] / record["exact_top_singular"]
                for record in records
            )
            / len(records),
            "records": records,
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
