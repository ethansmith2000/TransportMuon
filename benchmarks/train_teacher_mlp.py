"""Short teacher-student training validation for Transport Muon variants."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muon import get_muon_param_groups  # noqa: E402
from muon_warm import MuonWarm  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def _evaluate(model, inputs, targets, batch_size: int) -> float:
    total = torch.zeros((), device=inputs.device)
    count = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            prediction = model(inputs[start : start + batch_size]).float()
            target = targets[start : start + batch_size]
            total += (prediction - target).square().sum()
            count += prediction.numel()
    return float(total / count)


def _optimizer_diagnostics(optimizer: MuonWarm) -> dict:
    anchors = Counter()
    warm_steps = 0
    retractions = 0
    rejections = 0.0
    max_age = 0
    max_tangent_rms = 0.0
    max_applied_tangent_rms = 0.0
    min_effective_eta = 1.0
    for state in optimizer.state.values():
        anchors.update(state.get("muon_warm_anchor_counts", {}))
        warm_steps += int(state.get("muon_warm_warm_steps", 0))
        retractions += int(state.get("muon_warm_warm_retractions", 0))
        max_age = max(max_age, int(state.get("muon_warm_age", 0)))
        value = state.get("muon_warm_rejections_tensor")
        if value is not None:
            rejections += float(value)
        value = state.get("muon_warm_max_tangent_rms_tensor")
        if value is not None:
            max_tangent_rms = max(max_tangent_rms, float(value))
        value = state.get("muon_warm_max_applied_tangent_rms_tensor")
        if value is not None:
            max_applied_tangent_rms = max(
                max_applied_tangent_rms, float(value)
            )
        value = state.get("muon_warm_min_effective_eta_tensor")
        if value is not None:
            min_effective_eta = min(min_effective_eta, float(value))
    return {
        "anchor_counts": dict(sorted(anchors.items())),
        "warm_steps": warm_steps,
        "warm_retractions": retractions,
        "rejections": rejections,
        "maximum_final_age": max_age,
        "maximum_proposed_tangent_rms": max_tangent_rms,
        "maximum_applied_tangent_rms": max_applied_tangent_rms,
        "minimum_effective_eta": min_effective_eta,
    }


def train_variant(
    name: str,
    optimizer_options: dict,
    initial_state: dict,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    device = train_inputs.device
    model = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(
        device=device, dtype=train_inputs.dtype
    )
    model.load_state_dict(initial_state)
    groups = get_muon_param_groups(
        model,
        muon_lr=args.muon_lr,
        muon_momentum=args.momentum,
        adam_lr=args.adam_lr,
        adam_betas=(0.9, 0.99),
        muon_weight_decay=0.0,
        adam_weight_decay=0.0,
    )
    optimizer_options = dict(optimizer_options)
    optimizer = MuonWarm(
        groups,
        muon_lr=args.muon_lr,
        muon_momentum=args.momentum,
        adam_lr=args.adam_lr,
        muon_ns_steps=5,
        muon_warm_full_ns_steps=optimizer_options.pop(
            "muon_warm_full_ns_steps", 0
        ),
        muon_warm_jacobi_damping="tikhonov",
        **optimizer_options,
    )
    checkpoints = {0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps - 1}
    curve = []
    step_events = []
    optimizer_events = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    for step in range(args.steps):
        batch_start = (step * args.batch_size) % (
            train_inputs.shape[0] - args.batch_size + 1
        )
        inputs = train_inputs[batch_start : batch_start + args.batch_size]
        targets = train_targets[batch_start : batch_start + args.batch_size]
        timed = step >= args.timing_warmup
        if timed and device.type == "cuda":
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            optimizer_start = torch.cuda.Event(enable_timing=True)
            optimizer_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs).float()
        loss = (prediction - targets).square().mean()
        loss.backward()
        if timed and device.type == "cuda":
            optimizer_start.record()
        optimizer.step()
        if timed and device.type == "cuda":
            optimizer_end.record()
            step_end.record()
            step_events.append((step_start, step_end))
            optimizer_events.append((optimizer_start, optimizer_end))
        if step in checkpoints:
            curve.append(
                {"step": step + 1, "train_loss": float(loss.detach())}
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        training_ms = sum(start.elapsed_time(end) for start, end in step_events)
        optimizer_ms = sum(
            start.elapsed_time(end) for start, end in optimizer_events
        )
    else:
        training_ms = math.nan
        optimizer_ms = math.nan
    wall_ms = 1000.0 * (time.perf_counter() - wall_start)
    measured = max(1, args.steps - args.timing_warmup)
    return {
        "variant": name,
        "validation_loss": _evaluate(
            model, validation_inputs, validation_targets, args.batch_size
        ),
        "final_train_loss": curve[-1]["train_loss"],
        "loss_curve": curve,
        "training_milliseconds_per_step": training_ms / measured,
        "optimizer_milliseconds_per_step": optimizer_ms / measured,
        "wall_milliseconds_per_step": wall_ms / args.steps,
        **_optimizer_diagnostics(optimizer),
    }


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(seed)
    teacher = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    teacher.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    total_examples = args.train_examples + args.validation_examples
    inputs_float = torch.randn(
        total_examples,
        args.input_dim,
        device=device,
        generator=generator,
    )
    with torch.no_grad():
        targets = teacher(inputs_float).detach()
    inputs = inputs_float.to(dtype)
    del teacher, inputs_float

    torch.manual_seed(seed + 1000)
    initial_model = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(
        device=device, dtype=dtype
    )
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }
    del initial_model
    train_inputs = inputs[: args.train_examples]
    validation_inputs = inputs[args.train_examples :]
    train_targets = targets[: args.train_examples]
    validation_targets = targets[args.train_examples :]
    variants = (
        (
            "fresh_matched",
            {
                "muon_warm_anchor_every": 1,
                "muon_warm_retract_every": 1,
            },
        ),
        (
            "transport_fixed8",
            {
                "muon_warm_anchor_every": 8,
                "muon_warm_retract_every": 1,
            },
        ),
        (
            "transport_fixed8_normal_inv_cap4",
            {
                "muon_warm_anchor_every": 8,
                "muon_warm_retract_every": 1,
                "muon_warm_normal_inv_cap": 4.0,
            },
        ),
        (
            "transport_fixed4",
            {
                "muon_warm_anchor_every": 4,
                "muon_warm_retract_every": 1,
            },
        ),
        (
            "transport_fixed2",
            {
                "muon_warm_anchor_every": 2,
                "muon_warm_retract_every": 1,
            },
        ),
        (
            "transport_fixed16",
            {
                "muon_warm_anchor_every": 16,
                "muon_warm_retract_every": 1,
            },
        ),
        (
            "transport_fixed8_lazy2_cap005",
            {
                "muon_warm_anchor_every": 8,
                "muon_warm_full_ns_steps": 16,
                "muon_warm_max_tangent_rms": 0.05,
                "muon_warm_retract_every": 2,
            },
        ),
        (
            "transport_guarded_adaptive",
            {
                "muon_warm_anchor_every": 0,
                "muon_warm_full_ns_steps": 16,
                "muon_warm_max_age": 7,
                "muon_warm_max_tracking_error": 0.08,
                "muon_warm_min_stretch": 0.05,
                "muon_warm_check_every": 4,
                "muon_warm_retract_every": 1,
            },
        ),
    )
    if args.variants:
        requested = set(args.variants)
        available = {name for name, _ in variants}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        variants = tuple(item for item in variants if item[0] in requested)
    return {
        "seed": seed,
        "results": [
            train_variant(
                name,
                options,
                initial_state,
                train_inputs,
                train_targets,
                validation_inputs,
                validation_targets,
                args,
            )
            for name, options in variants
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--timing-warmup", type=int, default=40)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument(
        "--variants",
        nargs="+",
        help="run only the named variants (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= args.timing_warmup:
        raise ValueError("steps must exceed timing_warmup")

    runs = [run_seed(seed, args) for seed in args.seeds]
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else "cpu"
        ),
        "runs": runs,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
