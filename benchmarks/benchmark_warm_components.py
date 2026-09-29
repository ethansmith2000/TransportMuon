"""Attribute retained Transport Muon warm-path cost at model-sized shapes."""

import argparse
import json
import math
from pathlib import Path
import sys
import types

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# This benchmark exercises only the polar kernels. Keep it runnable when the
# surrounding Muon package is not installed.
if "muon" not in sys.modules:
    stub = types.ModuleType("muon")
    stub.adam_update = lambda *args, **kwargs: args[0]
    sys.modules["muon"] = stub

from muon_warm import (  # noqa: E402
    _muon_ns5_prepared,
    _row_ns_retract,
    _skew_ratio_from_terms,
    _warm_polar_jacobi_step_power_step_cap,
    _warm_polar_jacobi_step_power_step_cap_with_adaptive_power_skew_ratio_signal_no_cache_retract,
    _warm_polar_jacobi_step_power_step_cap_with_angular_signal_no_cache_retract,
    _warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract,
    _warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract,
    _warm_polar_jacobi_step_power_step_cap_with_skew_terms_no_cache_retract,
)


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, columns = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must look like 768x2048") from error
    if rows < 1 or columns < rows:
        raise argparse.ArgumentTypeError(
            "prepared shape requires 1 <= rows <= columns"
        )
    return rows, columns


def _elapsed_ms(function, repetitions: int, device: torch.device) -> float:
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            function()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repetitions

    import time

    start_time = time.perf_counter()
    for _ in range(repetitions):
        function()
    return 1e3 * (time.perf_counter() - start_time) / repetitions


def _direction_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    candidate_float = candidate.float()
    reference_float = reference.float()
    candidate_norm = candidate_float.norm()
    reference_norm = reference_float.norm().clamp_min(1e-30)
    return {
        "cosine_to_power2": float(
            (candidate_float * reference_float).sum()
            / candidate_norm.clamp_min(1e-30)
            / reference_norm
        ),
        "rms_ratio_to_power2": float(candidate_norm / reference_norm),
        "relative_error_to_power2": float(
            (candidate_float - reference_float).norm() / reference_norm
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        dest="shapes",
        help="prepared row-orthogonal shape, repeatable (default: 768x768, 768x2048)",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--drift", type=float, default=0.08)
    parser.add_argument("--power-safety-factor", type=float, default=1.05)
    parser.add_argument("--normal-inv-cap", type=float, default=4.0)
    parser.add_argument(
        "--power-refine-threshold",
        action="append",
        type=float,
        dest="power_refine_thresholds",
        help="adaptive second-power direction-change threshold, repeatable",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.drift < 0.0:
        raise ValueError("drift must be non-negative")
    power_refine_thresholds = args.power_refine_thresholds or [0.01, 0.02, 0.05]
    if any(value <= 0.0 for value in power_refine_thresholds):
        raise ValueError("power refine thresholds must be positive")

    shapes = args.shapes or [(768, 768), (768, 2048)]
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []

    for rows, columns in shapes:
        # QR on a columns-by-rows matrix produces an orthonormal column basis;
        # transpose it to the row-orthogonal orientation tracked by Muon.
        raw = torch.randn(
            columns, rows, device=device, dtype=torch.float32, generator=generator
        )
        q_previous = torch.linalg.qr(raw, mode="reduced").Q.mT.to(torch.bfloat16)
        innovation = torch.randn(
            rows, columns, device=device, dtype=torch.float32, generator=generator
        ) / math.sqrt(columns)
        target = q_previous.float() + args.drift * innovation
        work_matrix = (target / target.norm().clamp_min(1e-30)).to(torch.bfloat16)

        common = (
            work_matrix,
            q_previous,
            1.0,
            1e-3,
            "higham_cubic",
            0,
            True,
            None,
            "tikhonov",
        )
        quadratic_common = (*common[:4], "quadratic", *common[5:])
        # Prime a stable probe vector once, then feed the same cached vector to
        # every timed call. This models an ordinary warm optimizer step without
        # mixing Python state mutation into the measurement.
        primed = _warm_polar_jacobi_step_power_step_cap(
            *common,
            None,
            2,
            args.power_safety_factor,
            1.0,
            1.0,
            args.normal_inv_cap,
        )
        power_vector = primed[1]

        def fresh_anchor():
            return _row_ns_retract(
                _muon_ns5_prepared(work_matrix, 5),
                2,
                "higham_cubic",
                False,
            )

        def warm_geometry():
            return _warm_polar_jacobi_step_power_step_cap(
                *common,
                power_vector,
                2,
                args.power_safety_factor,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def warm_output(power_steps: int = 2):
            return _warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract(
                *common,
                power_vector,
                power_steps,
                args.power_safety_factor,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def warm_output_quadratic(power_steps: int = 1):
            return _warm_polar_jacobi_step_power_step_cap_with_legacy_output_no_cache_retract(
                *quadratic_common,
                power_vector,
                power_steps,
                args.power_safety_factor,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def warm_angular():
            return _warm_polar_jacobi_step_power_step_cap_with_angular_signal_no_cache_retract(
                *common,
                power_vector,
                2,
                args.power_safety_factor,
                True,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def warm_skew(power_steps: int = 2, quadratic: bool = False):
            return _warm_polar_jacobi_step_power_step_cap_with_skew_ratio_signal_no_cache_retract(
                *(quadratic_common if quadratic else common),
                power_vector,
                power_steps,
                args.power_safety_factor,
                True,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def warm_skew_terms(power_steps: int = 1):
            return _warm_polar_jacobi_step_power_step_cap_with_skew_terms_no_cache_retract(
                *common,
                power_vector,
                power_steps,
                args.power_safety_factor,
                True,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        def reduce_skew_terms():
            result = warm_skew_terms()
            return _skew_ratio_from_terms(result[2], result[3])

        def warm_adaptive(threshold: float):
            return _warm_polar_jacobi_step_power_step_cap_with_adaptive_power_skew_ratio_signal_no_cache_retract(
                *common,
                power_vector,
                2,
                args.power_safety_factor,
                threshold,
                True,
                1.0,
                1.0,
                args.normal_inv_cap,
            )

        variants = {
            "fresh_anchor": fresh_anchor,
            "warm_geometry_power2": warm_geometry,
            "warm_output_power2": warm_output,
            "warm_output_power1": lambda: warm_output(1),
            "warm_output_power1_quadratic": warm_output_quadratic,
            "warm_angular_signal": warm_angular,
            "warm_skew_signal": warm_skew,
            "warm_skew_power1": lambda: warm_skew(1),
            "warm_skew_terms_power1": warm_skew_terms,
            "warm_skew_terms_power1_with_reduction": reduce_skew_terms,
            "warm_skew_power1_quadratic": lambda: warm_skew(1, True),
        }
        variants.update(
            {
                f"warm_skew_adaptive_{threshold:g}": (
                    lambda threshold=threshold: warm_adaptive(threshold)
                )
                for threshold in power_refine_thresholds
            }
        )
        timings = {
            name: _elapsed_ms(function, args.repetitions, device)
            for name, function in variants.items()
        }

        def warm_skew_power1_period4_cycle():
            # The retained controller checks every fourth optimizer step. Three
            # signal-free calls followed by one measured call reproduce the
            # optimized cadence without mixing model/trainer work into this
            # component benchmark.
            warm_output(1)
            warm_output(1)
            warm_output(1)
            return warm_skew(1)

        timings["warm_skew_power1_check_every4_amortized"] = (
            _elapsed_ms(
                warm_skew_power1_period4_cycle,
                args.repetitions,
                device,
            )
            / 4.0
        )

        def warm_skew_power1_anchor8_check4_cycle():
            # A period-eight scheduled anchor makes the step-seven signal
            # irrelevant. Among the seven warm steps, only step three needs the
            # skew measurement that feeds the step-four controller check.
            for _ in range(6):
                warm_output(1)
            return warm_skew(1)

        timings["warm_skew_power1_anchor8_check4_amortized"] = (
            _elapsed_ms(
                warm_skew_power1_anchor8_check4_cycle,
                args.repetitions,
                device,
            )
            / 7.0
        )

        def warm_skew_terms_power1_anchor8_check4_cycle():
            # The raw-term direction graph is identical for all seven warm
            # steps. Only step three reduces its returned terms to a scalar.
            for index in range(7):
                result = warm_skew_terms()
                if index == 2:
                    _skew_ratio_from_terms(result[2], result[3])
            return result

        timings[
            "warm_skew_terms_power1_anchor8_check4_amortized"
        ] = (
            _elapsed_ms(
                warm_skew_terms_power1_anchor8_check4_cycle,
                args.repetitions,
                device,
            )
            / 7.0
        )
        reference_output = warm_output(2)[1]
        power1_output = warm_output(1)[1]
        quadratic_output = warm_output_quadratic()[1]
        angular_output = warm_angular()[1]
        skew_output = warm_skew()[1]
        skew_terms_output = warm_skew_terms()[1]
        adaptive_outputs = {
            f"{threshold:g}": warm_adaptive(threshold)
            for threshold in power_refine_thresholds
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        record = {
            "rows": rows,
            "columns": columns,
            "dtype": "bfloat16",
            "drift": args.drift,
            "repetitions": args.repetitions,
            "power_safety_factor": args.power_safety_factor,
            "normal_inv_cap": args.normal_inv_cap,
            "timing_milliseconds": timings,
            "relative_timing": {
                name: value / timings["fresh_anchor"]
                for name, value in timings.items()
            },
            "power1_output_metrics": _direction_metrics(
                power1_output, reference_output
            ),
            "power1_quadratic_output_metrics": _direction_metrics(
                quadratic_output, reference_output
            ),
            "angular_output_metrics": _direction_metrics(
                angular_output, reference_output
            ),
            "skew_output_metrics": _direction_metrics(skew_output, reference_output),
            "skew_terms_output_metrics": _direction_metrics(
                skew_terms_output, reference_output
            ),
            "angular_signal": float(warm_angular()[2]),
            "skew_ratio_signal": float(warm_skew()[2]),
            "skew_terms_ratio_signal": float(reduce_skew_terms()),
            "adaptive_power": {
                threshold: {
                    "output_metrics": _direction_metrics(
                        output[1], reference_output
                    ),
                    "refined": bool(output[8]),
                    "uncertainty": float(output[9]),
                }
                for threshold, output in adaptive_outputs.items()
            },
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    result = {
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "records": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
