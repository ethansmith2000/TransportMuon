"""Self-contained causal-language-model training harness.

The same file is used in TransportMuon and TurboSOAP. It discovers the local
optimizer implementation, keeps optional dataset dependencies lazy, and writes a
machine-readable summary suitable for the shared optimizer research journal.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from transformer import Transformer


try:
    from muon import get_muon_param_groups
    from muon_warm import MuonWarm

    HAS_TRANSPORT_MUON = True
except ImportError:
    HAS_TRANSPORT_MUON = False

try:
    from soap import SOAP

    HAS_TURBO_SOAP = True
except ImportError:
    HAS_TURBO_SOAP = False


class SyntheticSequenceDataset(Dataset):
    """Learnable arithmetic token sequences for offline integration tests."""

    def __init__(
        self,
        samples: int,
        sequence_length: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        if samples < 1 or sequence_length < 2 or vocab_size < 16:
            raise ValueError("synthetic samples/length must be positive and vocab >= 16")
        generator = torch.Generator().manual_seed(int(seed))
        starts = torch.randint(vocab_size, (samples, 1), generator=generator)
        strides = torch.randint(1, min(9, vocab_size), (samples, 1), generator=generator)
        positions = torch.arange(sequence_length + 1).unsqueeze(0)
        self.tokens = (starts + strides * positions).remainder(vocab_size).long()

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        tokens = self.tokens[index]
        return tokens[:-1], tokens[1:]


class PackedTokenDataset(Dataset):
    """Non-overlapping causal blocks from one packed token stream."""

    def __init__(self, tokens: Tensor, sequence_length: int) -> None:
        self.tokens = tokens.long().contiguous()
        self.sequence_length = int(sequence_length)
        self.blocks = (self.tokens.numel() - 1) // self.sequence_length
        if self.blocks < 1:
            raise ValueError("token stream is shorter than one sequence")

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.sequence_length
        block = self.tokens[start : start + self.sequence_length + 1]
        return block[:-1], block[1:]


def _compact_token_ids(tokens: Tensor) -> Tensor:
    """Use a portable compact dtype for persistent token storage."""
    if tokens.numel() and (
        int(tokens.min()) < 0 or int(tokens.max()) > torch.iinfo(torch.int32).max
    ):
        raise ValueError("token IDs cannot be represented as int32")
    return tokens.to(torch.int32)


def _append_tokenized(
    pieces: list[Tensor],
    encoded: list[list[int]],
    eos: int,
    maximum_tokens: int,
    total_tokens: int,
) -> tuple[int, bool]:
    for token_ids in encoded:
        if not token_ids:
            continue
        piece = torch.tensor([*token_ids, eos], dtype=torch.long)
        if maximum_tokens > 0:
            remaining = maximum_tokens - total_tokens
            if remaining <= 0:
                return total_tokens, True
            piece = piece[:remaining]
        if piece.numel() > 0:
            pieces.append(piece)
            total_tokens += piece.numel()
        if maximum_tokens > 0 and total_tokens >= maximum_tokens:
            return total_tokens, True
    return total_tokens, False


def _tokenize_split(
    dataset,
    tokenizer,
    text_column: str,
    maximum_documents: int,
    maximum_tokens: int = 0,
) -> Tensor:
    if maximum_documents > 0:
        dataset = dataset.select(range(min(maximum_documents, len(dataset))))
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pieces: list[Tensor] = []
    total_tokens = 0
    chunk_size = 256
    for start in range(0, len(dataset), chunk_size):
        texts = [str(value) for value in dataset[start : start + chunk_size][text_column]]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        total_tokens, done = _append_tokenized(
            pieces, encoded, eos, maximum_tokens, total_tokens
        )
        if done:
            break
    if not pieces:
        raise ValueError("the selected dataset split produced no tokens")
    return torch.cat(pieces)


def _tokenize_iterable_split(
    dataset,
    tokenizer,
    text_column: str,
    maximum_documents: int,
    maximum_tokens: int,
) -> Tensor:
    """Materialize a bounded token sample without downloading the full dataset."""
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pieces: list[Tensor] = []
    texts: list[str] = []
    total_tokens = 0
    documents = 0

    def flush() -> bool:
        nonlocal total_tokens
        if not texts:
            return False
        encoded = tokenizer(
            texts, add_special_tokens=False, truncation=False, verbose=False
        )["input_ids"]
        total_tokens, done = _append_tokenized(
            pieces, encoded, eos, maximum_tokens, total_tokens
        )
        texts.clear()
        return done

    iterator = iter(dataset)
    try:
        for document in iterator:
            if maximum_documents > 0 and documents >= maximum_documents:
                break
            if text_column not in document:
                raise ValueError(f"streamed document has no {text_column!r} field")
            texts.append(str(document[text_column]))
            documents += 1
            if len(texts) >= 256 and flush():
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if texts:
        flush()
    if not pieces:
        raise ValueError("the selected dataset stream produced no tokens")
    return torch.cat(pieces)


def build_datasets(args: argparse.Namespace):
    if args.dataset == "synthetic":
        return (
            SyntheticSequenceDataset(
                args.synthetic_train_samples,
                args.sequence_length,
                args.synthetic_vocab_size,
                args.seed,
            ),
            SyntheticSequenceDataset(
                args.synthetic_validation_samples,
                args.sequence_length,
                args.synthetic_vocab_size,
                args.seed + 1,
            ),
            args.synthetic_vocab_size,
        )

    cache_metadata = {
        "version": 2,
        "storage_dtype": "int32",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer_name": args.tokenizer_name,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "validation_fraction": args.validation_fraction,
        "text_column": args.text_column,
        "max_train_documents": args.max_train_documents,
        "max_validation_documents": args.max_validation_documents,
        "max_train_tokens": args.max_train_tokens,
        "max_validation_tokens": args.max_validation_tokens,
        "hf_streaming": args.hf_streaming,
        "streaming_validation_documents": args.streaming_validation_documents,
        "split_seed": None if args.hf_streaming else args.seed,
    }
    if args.token_cache is not None and args.token_cache.exists():
        cached = torch.load(args.token_cache, map_location="cpu", weights_only=True)
        if cached.get("metadata") != cache_metadata:
            raise ValueError(
                f"token cache metadata does not match this run: {args.token_cache}; "
                "use a different path or remove the stale cache"
            )
        return (
            PackedTokenDataset(cached["train_tokens"], args.sequence_length),
            PackedTokenDataset(cached["validation_tokens"], args.sequence_length),
            int(cached["vocab_size"]),
        )

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "--dataset hf requires the datasets and transformers packages"
        ) from error

    openwebtext = args.dataset_name.casefold() == "skylion007/openwebtext"
    if openwebtext and not args.hf_streaming and not args.allow_large_hf_download:
        raise ValueError(
            "eager OpenWebText loading can consume about 64 GB; use "
            "--hf-streaming or explicitly pass --allow-large-hf-download"
        )

    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
        streaming=args.hf_streaming,
    )
    if args.train_split not in raw:
        raise ValueError(f"dataset has no {args.train_split!r} split")
    if args.validation_split in raw:
        train_split = raw[args.train_split]
        validation_split = raw[args.validation_split]
    elif args.hf_streaming:
        if args.streaming_validation_documents < 1:
            raise ValueError("streaming validation documents must be positive")
        source = raw[args.train_split]
        validation_split = source.take(args.streaming_validation_documents)
        train_split = source.skip(args.streaming_validation_documents)
    else:
        divided = raw[args.train_split].train_test_split(
            test_size=args.validation_fraction,
            seed=args.seed,
        )
        train_split, validation_split = divided["train"], divided["test"]
    column_names = getattr(train_split, "column_names", None)
    if column_names and args.text_column not in column_names:
        raise ValueError(
            f"text column {args.text_column!r} not in {train_split.column_names}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    tokenizer_fn = _tokenize_iterable_split if args.hf_streaming else _tokenize_split
    train_tokens = tokenizer_fn(
        train_split,
        tokenizer,
        args.text_column,
        args.max_train_documents,
        args.max_train_tokens,
    )
    validation_tokens = tokenizer_fn(
        validation_split,
        tokenizer,
        args.text_column,
        args.max_validation_documents,
        args.max_validation_tokens,
    )
    if args.token_cache is not None:
        args.token_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.token_cache.with_suffix(
            args.token_cache.suffix + f".{os.getpid()}.tmp"
        )
        torch.save(
            {
                "metadata": cache_metadata,
                # Model inputs become int64 in PackedTokenDataset, but token IDs
                # only need int32 on disk. This halves persistent cache size.
                "train_tokens": _compact_token_ids(train_tokens),
                "validation_tokens": _compact_token_ids(validation_tokens),
                "vocab_size": len(tokenizer),
            },
            temporary,
        )
        os.replace(temporary, args.token_cache)
    return (
        PackedTokenDataset(train_tokens, args.sequence_length),
        PackedTokenDataset(validation_tokens, args.sequence_length),
        len(tokenizer),
    )


def _embedding_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }


def _adamw_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    embedding_ids = _embedding_parameter_ids(model)
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = (
            decay
            if parameter.ndim >= 2 and id(parameter) not in embedding_ids
            else no_decay
        )
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _soap_groups(model: nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    embedding_ids = _embedding_parameter_ids(model)
    preconditioned: list[Tensor] = []
    fallback: list[Tensor] = []
    for name, parameter in model.named_parameters():
        use_preconditioner = (
            parameter.ndim >= 2
            and id(parameter) not in embedding_ids
            and not name.startswith("lm_head.")
        )
        (preconditioned if use_preconditioner else fallback).append(parameter)
    groups: list[dict[str, Any]] = []
    if preconditioned:
        groups.append(
            {
                "params": preconditioned,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_precond_dim": args.soap_max_precond_dim,
                "normalize_grads": args.soap_normalize_grads,
            }
        )
    if fallback:
        groups.append(
            {
                "params": fallback,
                "lr": args.soap_fallback_lr,
                "weight_decay": 0.0,
                "max_precond_dim": 1,
                "normalize_grads": False,
            }
        )
    return groups


def build_optimizer(model: nn.Module, args: argparse.Namespace, device: torch.device):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            _adamw_groups(model, args.weight_decay),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            fused=device.type == "cuda",
        )
    if args.optimizer == "transport_muon":
        if not HAS_TRANSPORT_MUON:
            raise RuntimeError("Transport Muon is not available in this repository")
        if args.muon_reference_lr > 0.0 and args.muon_lr <= 0.0:
            raise ValueError("muon_reference_lr requires a positive muon_lr")

        def use_muon(name: str, parameter: Tensor) -> bool:
            return (
                parameter.ndim >= 2
                and not name.startswith("lm_head.")
                and max(parameter.shape) <= args.muon_large_tensor_threshold
            )

        def muon_split_count(name: str, _parameter: Tensor) -> int:
            if args.muon_split_qkv and name.endswith(".attn.to_qkv.weight"):
                return 3
            if args.muon_split_swiglu and name.endswith(".ff.proj_in.weight"):
                return 2
            return 1

        groups = get_muon_param_groups(
            model,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_weight_decay=args.weight_decay,
            adam_lr=args.learning_rate,
            adam_betas=(args.beta1, args.beta2),
            adam_eps=args.eps,
            adam_weight_decay=0.0,
            large_tensor_threshold=args.muon_large_tensor_threshold,
            muon_predicate=use_muon,
            muon_split_predicate=muon_split_count,
        )
        return MuonWarm(
            groups,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_ns_steps=args.muon_ns_steps,
            muon_nesterov=args.muon_nesterov,
            muon_warm_anchor_every=args.muon_anchor_every,
            muon_warm_max_age=args.muon_max_age,
            muon_warm_check_every=args.muon_check_every,
            muon_warm_full_ns_steps=args.muon_full_ns_steps,
            muon_warm_retract_steps=args.muon_retract_steps,
            muon_warm_retract_every=args.muon_retract_every,
            muon_warm_retract_method=args.muon_retract_method,
            muon_warm_jacobi_damping=args.muon_jacobi_damping,
            muon_warm_max_tangent_rms=args.muon_max_tangent_rms,
            muon_warm_record_stats=args.muon_record_stats,
            muon_warm_normal_inv_cap=args.muon_normal_inv_cap,
            muon_warm_normal_inv_ema_ratio=args.muon_normal_inv_ema_ratio,
            muon_warm_normal_inv_ema_beta=args.muon_normal_inv_ema_beta,
            muon_warm_max_angular_rms=args.muon_max_angular_rms,
            muon_warm_max_skew_ratio=args.muon_max_skew_ratio,
            muon_warm_signal_check_only=args.muon_signal_check_only,
            muon_warm_separate_skew_signal=args.muon_separate_skew_signal,
            muon_warm_async_checks=args.muon_async_checks,
            muon_warm_update_stats_every=args.muon_update_stats_every,
            muon_warm_reference_lr_ratio=(
                args.muon_reference_lr / args.muon_lr
                if args.muon_reference_lr > 0.0
                else 1.0
            ),
            muon_warm_normalize_output=args.muon_normalize_output,
            muon_warm_output_scale=args.muon_output_scale,
            muon_warm_output_scale_start=args.muon_output_scale_start,
            muon_warm_output_scale_decay=args.muon_output_scale_decay,
            muon_warm_legacy_retraction_output=args.muon_legacy_retraction_output,
            muon_warm_spectral_cap_mode=args.muon_spectral_cap_mode,
            muon_warm_power_steps=args.muon_power_steps,
            muon_warm_power_safety_factor=args.muon_power_safety_factor,
            muon_warm_power_refine_threshold=args.muon_power_refine_threshold,
        )
    if args.optimizer == "soap":
        if not HAS_TURBO_SOAP:
            raise RuntimeError("TurboSOAP is not available in this repository")
        return SOAP(
            _soap_groups(model, args),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.soap_shampoo_beta,
            eps=args.eps,
            precondition_frequency=args.soap_precondition_frequency,
            precondition_frequency_after_warmup=(
                args.soap_precondition_frequency_after_warmup
            ),
            precondition_frequency_warmup_steps=(
                args.soap_precondition_frequency_warmup_steps
            ),
            basis_residual_threshold=args.soap_residual_threshold,
            basis_residual_max_age=args.soap_residual_max_age,
            basis_residual_warmup_steps=args.soap_residual_warmup_steps,
            precondition_mode=args.soap_precondition_mode,
            precondition_aspect_ratio=args.soap_precondition_aspect_ratio,
            normalize_grads=args.soap_normalize_grads,
            basis_lr_age_compensation=args.soap_basis_lr_age_compensation,
            basis_lr_reference_age=args.soap_basis_lr_reference_age,
            basis_reset_frequency=args.soap_reset_frequency,
            basis_reset_max_age=args.soap_reset_max_age,
            basis_reset_stagger=args.soap_reset_stagger,
            basis_reset_method=args.soap_reset_method,
            basis_track_stats=args.soap_track_stats,
            covariance_compute_dtype=args.soap_covariance_dtype,
        )
    raise ValueError(f"unknown optimizer {args.optimizer!r}")


def _tree_tensor_bytes(value: Any, seen: set[int]) -> int:
    if isinstance(value, Tensor):
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tree_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_tensor_bytes(item, seen) for item in value)
    return 0


def optimizer_diagnostics(optimizer) -> dict[str, Any]:
    anchors = Counter()
    angular_checks = 0
    skew_ratio_checks = 0
    angular_signal_evaluations = 0
    skew_ratio_signal_evaluations = 0
    async_checks_submitted = async_checks_completed = 0
    async_enqueue_skips = async_stale_checks = 0
    warm_steps = warm_retractions = output_retractions = 0
    basis_refreshes = basis_active_refreshes = active_factors = 0
    basis_residual_checks = basis_residual_skips = 0
    basis_residual_threshold_refreshes = basis_residual_max_age_refreshes = 0
    basis_residual_ratio_sum = basis_residual_ratio_max = 0.0
    basis_residual_ratio_samples: list[float] = []
    basis_effective_lr_samples = 0
    basis_effective_lr_sum = basis_effective_lr_max = 0.0
    hard_reset_events = hard_reset_factors = 0
    reset_first_refreshes: list[int] = []
    reset_first_steps: list[int] = []
    seen: set[int] = set()
    state_bytes = 0
    update_elements = update_samples = 0
    reference_elements = reference_samples = 0
    power_cap_samples = 0
    power_sigma_sum = power_cap_scale_sum = 0.0
    power_cap_min_scale = 1.0
    power_step_samples = 0
    power_probe_sigma_sum = power_step_scale_sum = 0.0
    power_step_min_scale = 1.0
    power_refinement_samples = 0
    power_refinement_sum = power_uncertainty_sum = 0.0
    power_uncertainty_max = 0.0
    controller_samples = alignment_condition_samples = 0
    controller_sums: dict[str, float] = Counter()
    controller_maxima: dict[str, float] = {}
    alignment_condition_sums: dict[str, float] = Counter()
    relative_min_abs_diag = math.inf
    relative_max_abs_inv = 0.0
    normal_inv_ema_samples = normal_inv_dynamic_cap_samples = 0
    normal_inv_observation_sum = normal_inv_dynamic_cap_sum = 0.0
    normal_inv_effective_cap_sum = normal_inv_ema_active_sum = 0.0
    normal_inv_dynamic_cap_min = math.inf
    refresh_predictor_names = (
        "momentum_innovation_ratio",
        "tangent_rms",
        "angular_rms",
        "normal_rms",
        "relative_max_abs_inverse",
        "normal_inverse_clipped_fraction",
    )
    refresh_predictor_suffixes = (
        "sample",
        "value",
        "value_sq",
        "error",
        "error_sq",
        "value_error",
    )
    shape_predictor_sums: dict[str, dict[str, Counter]] = {}
    statistic_sums: dict[str, float] = Counter()
    statistic_maxima: dict[str, float] = {}
    age_statistics: dict[int, dict[str, Any]] = {}
    statistic_keys = (
        "muon_update_direction_sq_sum_tensor",
        "muon_update_momentum_sq_sum_tensor",
        "muon_update_direction_momentum_dot_sum_tensor",
        "muon_update_applied_sq_sum_tensor",
        "muon_update_orthogonality_error_sum_tensor",
        "muon_reference_direction_sq_sum_tensor",
        "muon_reference_direction_dot_sum_tensor",
        "muon_reference_difference_sq_sum_tensor",
        "muon_reference_candidate_applied_sq_sum_tensor",
        "muon_reference_applied_sq_sum_tensor",
    )
    states = []
    for parameter_state in optimizer.state.values():
        block_states = parameter_state.get("muon_block_states")
        states.extend(block_states if block_states is not None else (parameter_state,))
    for state in states:
        state_bytes += _tree_tensor_bytes(state, seen)
        anchors.update(state.get("muon_warm_anchor_counts", {}))
        angular_checks += int(state.get("muon_warm_angular_checks", 0))
        skew_ratio_checks += int(
            state.get("muon_warm_skew_ratio_checks", 0)
        )
        angular_signal_evaluations += int(
            state.get("muon_warm_angular_signal_evaluations", 0)
        )
        skew_ratio_signal_evaluations += int(
            state.get("muon_warm_skew_ratio_signal_evaluations", 0)
        )
        async_checks_submitted += int(
            state.get("muon_warm_async_checks_submitted", 0)
        )
        async_checks_completed += int(
            state.get("muon_warm_async_checks_completed", 0)
        )
        async_enqueue_skips += int(
            state.get("muon_warm_async_enqueue_skips", 0)
        )
        async_stale_checks += int(
            state.get("muon_warm_async_stale_checks", 0)
        )
        warm_steps += int(state.get("muon_warm_warm_steps", 0))
        warm_retractions += int(state.get("muon_warm_warm_retractions", 0))
        output_retractions += int(
            state.get("muon_warm_output_retractions", 0)
        )
        basis_refreshes += int(state.get("basis_refreshes", 0))
        basis_active_refreshes += int(
            state.get("basis_active_refreshes", 0)
        )
        state_effective_lr_samples = int(
            state.get("basis_effective_lr_samples", 0)
        )
        if not any(factor is not None for factor in state.get("Q", ())):
            # Fallback Adam parameters share the cadence bookkeeping but do
            # not execute a basis step. Exclude their nominal eta samples.
            state_effective_lr_samples = 0
        basis_effective_lr_samples += state_effective_lr_samples
        if state_effective_lr_samples > 0:
            basis_effective_lr_sum += float(
                state.get("basis_effective_lr_sum", 0.0)
            )
            basis_effective_lr_max = max(
                basis_effective_lr_max,
                float(state.get("basis_effective_lr_max", 0.0)),
            )
        state_residual_checks = int(state.get("basis_residual_checks", 0))
        basis_residual_checks += state_residual_checks
        basis_residual_skips += int(state.get("basis_residual_skips", 0))
        basis_residual_threshold_refreshes += int(
            state.get("basis_residual_threshold_refreshes", 0)
        )
        basis_residual_max_age_refreshes += int(
            state.get("basis_residual_max_age_refreshes", 0)
        )
        if state_residual_checks > 0:
            basis_residual_ratio_sum += float(
                state.get("basis_residual_ratio_sum", 0.0)
            )
            basis_residual_ratio_max = max(
                basis_residual_ratio_max,
                float(state.get("basis_residual_ratio_max", 0.0)),
            )
            basis_residual_ratio_samples.extend(
                float(value)
                for value in state.get("basis_residual_ratio_samples", ())
            )
        state_active_factors = sum(
            value is not None for value in state.get("Q", ())
        )
        active_factors += state_active_factors
        hard_reset_events += int(state.get("basis_hard_reset_events", 0))
        hard_reset_factors += int(state.get("basis_hard_reset_factors", 0))
        if state_active_factors > 0 and "basis_reset_first_refresh" in state:
            reset_first_refreshes.append(
                int(state["basis_reset_first_refresh"])
            )
        if state_active_factors > 0 and "basis_reset_first_step" in state:
            reset_first_steps.append(int(state["basis_reset_first_step"]))
        update_elements += int(state.get("muon_update_elements", 0))
        update_samples += int(state.get("muon_update_samples", 0))
        reference_elements += int(state.get("muon_reference_elements", 0))
        reference_samples += int(state.get("muon_reference_samples", 0))
        state_power_samples = int(state.get("muon_warm_power_cap_samples", 0))
        if state_power_samples > 0:
            power_cap_samples += state_power_samples
            power_sigma_sum += float(
                state["muon_warm_power_sigma_sum_tensor"]
            )
            power_cap_scale_sum += float(
                state["muon_warm_spectral_cap_scale_sum_tensor"]
            )
            power_cap_min_scale = min(
                power_cap_min_scale,
                float(state["muon_warm_spectral_cap_min_scale_tensor"]),
            )
        state_power_step_samples = int(
            state.get("muon_warm_power_step_samples", 0)
        )
        if state_power_step_samples > 0:
            power_step_samples += state_power_step_samples
            power_probe_sigma_sum += float(
                state["muon_warm_power_probe_sigma_sum_tensor"]
            )
            power_step_scale_sum += float(
                state["muon_warm_spectral_step_scale_sum_tensor"]
            )
            power_step_min_scale = min(
                power_step_min_scale,
                float(state["muon_warm_spectral_step_min_scale_tensor"]),
            )
        state_refinement_samples = int(
            state.get("muon_warm_power_refinement_samples", 0)
        )
        if state_refinement_samples > 0:
            power_refinement_samples += state_refinement_samples
            power_refinement_sum += float(
                state["muon_warm_power_refinement_sum_tensor"]
            )
            power_uncertainty_sum += float(
                state["muon_warm_power_uncertainty_sum_tensor"]
            )
            power_uncertainty_max = max(
                power_uncertainty_max,
                float(state["muon_warm_power_uncertainty_max_tensor"]),
            )
        state_controller_samples = int(
            state.get("muon_warm_controller_samples", 0)
        )
        if state_controller_samples > 0:
            controller_samples += state_controller_samples
            for name in (
                "effective_eta",
                "tangent_rms",
                "angular_rms",
                "normal_rms",
                "skew_ratio",
            ):
                controller_sums[name] += float(
                    state[f"muon_warm_{name}_sum_tensor"]
                )
                controller_maxima[name] = max(
                    controller_maxima.get(name, 0.0),
                    float(state[f"muon_warm_{name}_max_tensor"]),
                )
        state_alignment_samples = int(
            state.get("muon_warm_alignment_condition_samples", 0)
        )
        if state_alignment_samples > 0:
            alignment_condition_samples += state_alignment_samples
            for name in (
                "relative_min_abs_diag",
                "relative_max_abs_inv",
                "normal_inv_clipped_fraction",
            ):
                alignment_condition_sums[name] += float(
                    state[f"muon_warm_{name}_sum_tensor"]
                )
            relative_min_abs_diag = min(
                relative_min_abs_diag,
                float(state["muon_warm_relative_min_abs_diag_min_tensor"]),
            )
            relative_max_abs_inv = max(
                relative_max_abs_inv,
                float(state["muon_warm_relative_max_abs_inv_max_tensor"]),
            )
        state_normal_inv_ema_samples = int(
            state.get("muon_warm_normal_inv_ema_samples", 0)
        )
        if state_normal_inv_ema_samples > 0:
            normal_inv_ema_samples += state_normal_inv_ema_samples
            normal_inv_observation_sum += float(
                state["muon_warm_normal_inv_observation_sum_tensor"]
            )
        state_dynamic_cap_samples = int(
            state.get("muon_warm_normal_inv_dynamic_cap_samples", 0)
        )
        if state_dynamic_cap_samples > 0:
            normal_inv_dynamic_cap_samples += state_dynamic_cap_samples
            normal_inv_dynamic_cap_sum += float(
                state["muon_warm_normal_inv_dynamic_cap_sum_tensor"]
            )
            normal_inv_effective_cap_sum += float(
                state["muon_warm_normal_inv_effective_cap_sum_tensor"]
            )
            normal_inv_ema_active_sum += float(
                state["muon_warm_normal_inv_ema_active_sum_tensor"]
            )
            normal_inv_dynamic_cap_min = min(
                normal_inv_dynamic_cap_min,
                float(state["muon_warm_normal_inv_dynamic_cap_min_tensor"]),
            )
        for key in statistic_keys:
            value = state.get(key)
            if value is not None:
                statistic_sums[key] += float(value)
        value = state.get("muon_update_max_orthogonality_error_tensor")
        if value is not None:
            statistic_maxima["muon_update_max_orthogonality_error_tensor"] = max(
                statistic_maxima.get(
                    "muon_update_max_orthogonality_error_tensor", 0.0
                ),
                float(value),
            )
        for age, bucket in state.get("muon_update_stats_by_age", {}).items():
            age = int(age)
            total = age_statistics.setdefault(
                age,
                {
                    "sums": Counter(),
                    "maximum_orthogonality_error": 0.0,
                    "elements": 0,
                    "samples": 0,
                    "reference_elements": 0,
                    "reference_samples": 0,
                },
            )
            for key, value in bucket.items():
                if key.endswith("_tensor") and key != "max_orthogonality_error_tensor":
                    total["sums"][key] += float(value)
            value = bucket.get("max_orthogonality_error_tensor")
            if value is not None:
                total["maximum_orthogonality_error"] = max(
                    total["maximum_orthogonality_error"], float(value)
                )
            for key in (
                "elements",
                "samples",
                "reference_elements",
                "reference_samples",
            ):
                total[key] += int(bucket.get(key, 0))
        cached_q = state.get("muon_warm_q")
        if cached_q is not None:
            shape_key = "x".join(str(dimension) for dimension in cached_q.shape)
            shape_sums = shape_predictor_sums.setdefault(
                shape_key,
                {name: Counter() for name in refresh_predictor_names},
            )
            for bucket in state.get("muon_update_stats_by_age", {}).values():
                for name in refresh_predictor_names:
                    prefix = f"refresh_predictor_{name}_"
                    for suffix in refresh_predictor_suffixes:
                        key = prefix + suffix + "_sum_tensor"
                        value = bucket.get(key)
                        if value is not None:
                            shape_sums[name][key] += float(value)
    diagnostics = {
        "state_megabytes": state_bytes / 2**20,
        "anchor_counts": dict(sorted(anchors.items())),
        "angular_checks": angular_checks,
        "skew_ratio_checks": skew_ratio_checks,
        "angular_signal_evaluations": angular_signal_evaluations,
        "skew_ratio_signal_evaluations": skew_ratio_signal_evaluations,
        "async_drift_checks": {
            "submitted": async_checks_submitted,
            "completed": async_checks_completed,
            "enqueue_skips": async_enqueue_skips,
            "stale_after_anchor": async_stale_checks,
        },
        "warm_steps": warm_steps,
        "warm_retractions": warm_retractions,
        "output_retractions": output_retractions,
        "basis_refreshes": basis_refreshes,
        "basis_active_refreshes": basis_active_refreshes,
        "active_factors": active_factors,
        "hard_reset_events": hard_reset_events,
        "hard_reset_factors": hard_reset_factors,
    }
    if reset_first_refreshes:
        diagnostics["reset_first_refresh_range"] = [
            min(reset_first_refreshes),
            max(reset_first_refreshes),
        ]
    if reset_first_steps:
        diagnostics["reset_first_step_range"] = [
            min(reset_first_steps),
            max(reset_first_steps),
        ]
    if basis_residual_checks > 0 or basis_residual_max_age_refreshes > 0:
        diagnostics["basis_residual_gate"] = {
            "checks": basis_residual_checks,
            "skips": basis_residual_skips,
            "threshold_refreshes": basis_residual_threshold_refreshes,
            "max_age_refreshes": basis_residual_max_age_refreshes,
            "mean_checked_residual": (
                basis_residual_ratio_sum / basis_residual_checks
                if basis_residual_checks > 0
                else 0.0
            ),
            "maximum_checked_residual": basis_residual_ratio_max,
        }
        if basis_residual_ratio_samples:
            diagnostics["basis_residual_gate"]["checked_residual_percentiles"] = {
                "p50": _percentile(basis_residual_ratio_samples, 0.50),
                "p75": _percentile(basis_residual_ratio_samples, 0.75),
                "p90": _percentile(basis_residual_ratio_samples, 0.90),
                "p95": _percentile(basis_residual_ratio_samples, 0.95),
            }
    if basis_effective_lr_samples > 0:
        diagnostics["basis_tracker_step"] = {
            "samples": basis_effective_lr_samples,
            "mean_effective_lr": (
                basis_effective_lr_sum / basis_effective_lr_samples
            ),
            "maximum_effective_lr": basis_effective_lr_max,
        }
    if update_elements > 0:
        direction_sq = statistic_sums["muon_update_direction_sq_sum_tensor"]
        momentum_sq = statistic_sums["muon_update_momentum_sq_sum_tensor"]
        diagnostics["update_stats"] = {
            "samples": update_samples,
            "elements": update_elements,
            "direction_rms": math.sqrt(direction_sq / update_elements),
            "applied_direction_rms": math.sqrt(
                statistic_sums["muon_update_applied_sq_sum_tensor"]
                / update_elements
            ),
            "direction_momentum_cosine": (
                statistic_sums[
                    "muon_update_direction_momentum_dot_sum_tensor"
                ]
                / math.sqrt(max(direction_sq * momentum_sq, 1e-30))
            ),
            "mean_orthogonality_error": statistic_sums[
                "muon_update_orthogonality_error_sum_tensor"
            ]
            / update_samples,
            "maximum_orthogonality_error": statistic_maxima.get(
                "muon_update_max_orthogonality_error_tensor", 0.0
            ),
        }
    if power_cap_samples > 0 or power_step_samples > 0:
        power_cap_stats = {}
        if power_cap_samples > 0:
            power_cap_stats.update(
                {
                    "samples": power_cap_samples,
                    "mean_estimated_top_singular": (
                        power_sigma_sum / power_cap_samples
                    ),
                    "mean_input_scale": (
                        power_cap_scale_sum / power_cap_samples
                    ),
                    "minimum_input_scale": power_cap_min_scale,
                }
            )
        if power_step_samples > 0:
            power_cap_stats.update(
                {
                    "step_samples": power_step_samples,
                    "mean_probe_top_singular": (
                        power_probe_sigma_sum / power_step_samples
                    ),
                    "mean_transport_step_scale": (
                        power_step_scale_sum / power_step_samples
                    ),
                    "minimum_transport_step_scale": power_step_min_scale,
                }
            )
        if power_refinement_samples > 0:
            power_cap_stats.update(
                {
                    "adaptive_power_samples": power_refinement_samples,
                    "adaptive_power_refinements": power_refinement_sum,
                    "adaptive_power_refinement_fraction": (
                        power_refinement_sum / power_refinement_samples
                    ),
                    "mean_power_uncertainty": (
                        power_uncertainty_sum / power_refinement_samples
                    ),
                    "maximum_power_uncertainty": power_uncertainty_max,
                }
            )
        diagnostics["power_cap_stats"] = power_cap_stats
    if controller_samples > 0:
        diagnostics["transport_step_stats"] = {
            "samples": controller_samples,
            "mean_effective_eta": (
                controller_sums["effective_eta"] / controller_samples
            ),
            "mean_proposed_tangent_rms": (
                controller_sums["tangent_rms"] / controller_samples
            ),
            "maximum_proposed_tangent_rms": controller_maxima["tangent_rms"],
            "mean_angular_rms": (
                controller_sums["angular_rms"] / controller_samples
            ),
            "maximum_angular_rms": controller_maxima["angular_rms"],
            "mean_normal_rms": (
                controller_sums["normal_rms"] / controller_samples
            ),
            "maximum_normal_rms": controller_maxima["normal_rms"],
            "mean_skew_ratio": (
                controller_sums["skew_ratio"] / controller_samples
            ),
            "maximum_skew_ratio": controller_maxima["skew_ratio"],
        }
    if alignment_condition_samples > 0:
        diagnostics["alignment_condition_stats"] = {
            "samples": alignment_condition_samples,
            "mean_relative_min_abs_diag": (
                alignment_condition_sums["relative_min_abs_diag"]
                / alignment_condition_samples
            ),
            "minimum_relative_min_abs_diag": relative_min_abs_diag,
            "mean_relative_max_abs_inverse": (
                alignment_condition_sums["relative_max_abs_inv"]
                / alignment_condition_samples
            ),
            "maximum_relative_max_abs_inverse": relative_max_abs_inv,
            "mean_normal_inverse_clipped_fraction": (
                alignment_condition_sums["normal_inv_clipped_fraction"]
                / alignment_condition_samples
            ),
        }
    if normal_inv_ema_samples > 0:
        diagnostics["normal_inverse_ema_stats"] = {
            "samples": normal_inv_ema_samples,
            "mean_fixed_cap_inverse_rms": (
                normal_inv_observation_sum / normal_inv_ema_samples
            ),
            "dynamic_cap_samples": normal_inv_dynamic_cap_samples,
            "mean_dynamic_cap": (
                normal_inv_dynamic_cap_sum
                / max(1, normal_inv_dynamic_cap_samples)
            ),
            "mean_effective_cap": (
                normal_inv_effective_cap_sum
                / max(1, normal_inv_dynamic_cap_samples)
            ),
            "ema_active_fraction": (
                normal_inv_ema_active_sum
                / max(1, normal_inv_dynamic_cap_samples)
            ),
            "minimum_dynamic_cap": (
                normal_inv_dynamic_cap_min
                if normal_inv_dynamic_cap_samples > 0
                else None
            ),
        }
    if reference_elements > 0:
        direction_sq = statistic_sums["muon_update_direction_sq_sum_tensor"]
        reference_sq = statistic_sums["muon_reference_direction_sq_sum_tensor"]
        # Direction statistics include sampled anchors as well, while reference
        # statistics intentionally include only warm steps. Reconstruct the
        # candidate terms over warm samples from the exact identities involving
        # the reference dot product and difference norm.
        reference_dot = statistic_sums["muon_reference_direction_dot_sum_tensor"]
        difference_sq = statistic_sums[
            "muon_reference_difference_sq_sum_tensor"
        ]
        warm_direction_sq = max(
            0.0,
            difference_sq - reference_sq + 2.0 * reference_dot,
        )
        diagnostics["fresh_reference_stats"] = {
            "samples": reference_samples,
            "elements": reference_elements,
            "direction_cosine": reference_dot
            / math.sqrt(max(warm_direction_sq * reference_sq, 1e-30)),
            "direction_rms_ratio": math.sqrt(
                warm_direction_sq / max(reference_sq, 1e-30)
            ),
            "relative_direction_error": math.sqrt(
                difference_sq / max(reference_sq, 1e-30)
            ),
            "reference_direction_rms": math.sqrt(
                reference_sq / reference_elements
            ),
            "candidate_applied_direction_rms": math.sqrt(
                statistic_sums[
                    "muon_reference_candidate_applied_sq_sum_tensor"
                ]
                / reference_elements
            ),
            "reference_applied_direction_rms": math.sqrt(
                statistic_sums["muon_reference_applied_sq_sum_tensor"]
                / reference_elements
            ),
        }
        anchor_elements = update_elements - reference_elements
        anchor_samples = update_samples - reference_samples
        if anchor_elements > 0:
            total_applied_sq = statistic_sums[
                "muon_update_applied_sq_sum_tensor"
            ]
            warm_applied_sq = statistic_sums[
                "muon_reference_candidate_applied_sq_sum_tensor"
            ]
            diagnostics["anchor_update_stats"] = {
                "samples": anchor_samples,
                "elements": anchor_elements,
                "direction_rms": math.sqrt(
                    max(0.0, direction_sq - warm_direction_sq)
                    / anchor_elements
                ),
                "applied_direction_rms": math.sqrt(
                    max(0.0, total_applied_sq - warm_applied_sq)
                    / anchor_elements
                ),
            }
    def refresh_predictor_summary(sums, name):
        prefix = f"refresh_predictor_{name}_"
        count = sums[prefix + "sample_sum_tensor"]
        if count <= 0.0:
            return None
        value_sum = sums[prefix + "value_sum_tensor"]
        value_sq_sum = sums[prefix + "value_sq_sum_tensor"]
        error_sum = sums[prefix + "error_sum_tensor"]
        error_sq_sum = sums[prefix + "error_sq_sum_tensor"]
        cross_sum = sums[prefix + "value_error_sum_tensor"]
        value_variance_sum = max(
            0.0, value_sq_sum - value_sum * value_sum / count
        )
        error_variance_sum = max(
            0.0, error_sq_sum - error_sum * error_sum / count
        )
        centered_cross = cross_sum - value_sum * error_sum / count
        denominator = math.sqrt(value_variance_sum * error_variance_sum)
        return {
            "samples": int(round(count)),
            "mean_value": value_sum / count,
            "standard_deviation": math.sqrt(value_variance_sum / count),
            "mean_relative_reference_error": error_sum / count,
            "error_standard_deviation": math.sqrt(
                error_variance_sum / count
            ),
            "pearson_error_correlation": (
                centered_cross / denominator if denominator > 1e-30 else 0.0
            ),
        }

    if age_statistics:
        global_predictor_sums = {
            name: Counter() for name in refresh_predictor_names
        }
        by_age = {}
        for age, total in sorted(age_statistics.items()):
            sums = total["sums"]
            elements = total["elements"]
            samples = total["samples"]
            direction_sq = sums["direction_sq_sum_tensor"]
            momentum_sq = sums["momentum_sq_sum_tensor"]
            entry = {
                "kind": "anchor" if age == 0 else "warm",
                "samples": samples,
                "elements": elements,
                "direction_rms": math.sqrt(direction_sq / elements),
                "applied_direction_rms": math.sqrt(
                    sums["applied_sq_sum_tensor"] / elements
                ),
                "direction_momentum_cosine": (
                    sums["direction_momentum_dot_sum_tensor"]
                    / math.sqrt(max(direction_sq * momentum_sq, 1e-30))
                ),
                "mean_orthogonality_error": (
                    sums["orthogonality_error_sum_tensor"] / samples
                ),
                "maximum_orthogonality_error": total[
                    "maximum_orthogonality_error"
                ],
            }
            reference_elements = total["reference_elements"]
            if reference_elements > 0:
                reference_sq = sums["reference_direction_sq_sum_tensor"]
                reference_dot = sums["reference_direction_dot_sum_tensor"]
                difference_sq = sums["reference_difference_sq_sum_tensor"]
                candidate_sq = max(
                    0.0,
                    difference_sq - reference_sq + 2.0 * reference_dot,
                )
                entry["fresh_reference"] = {
                    "samples": total["reference_samples"],
                    "elements": reference_elements,
                    "direction_cosine": reference_dot
                    / math.sqrt(max(candidate_sq * reference_sq, 1e-30)),
                    "direction_rms_ratio": math.sqrt(
                        candidate_sq / max(reference_sq, 1e-30)
                    ),
                    "relative_direction_error": math.sqrt(
                        difference_sq / max(reference_sq, 1e-30)
                    ),
                    "candidate_applied_direction_rms": math.sqrt(
                        sums["candidate_applied_sq_sum_tensor"]
                        / reference_elements
                    ),
                    "reference_applied_direction_rms": math.sqrt(
                        sums["reference_applied_sq_sum_tensor"]
                        / reference_elements
                    ),
                }
                age_predictors = {}
                for name in refresh_predictor_names:
                    predictor = refresh_predictor_summary(sums, name)
                    if predictor is None:
                        continue
                    age_predictors[name] = predictor
                    prefix = f"refresh_predictor_{name}_"
                    for suffix in refresh_predictor_suffixes:
                        key = prefix + suffix + "_sum_tensor"
                        global_predictor_sums[name][key] += sums[key]
                if age_predictors:
                    entry["refresh_predictors"] = age_predictors
            by_age[str(age)] = entry
        diagnostics["update_stats_by_age"] = by_age
        predictor_diagnostics = {}
        for name, sums in global_predictor_sums.items():
            predictor = refresh_predictor_summary(sums, name)
            if predictor is not None:
                predictor_diagnostics[name] = predictor
        if predictor_diagnostics:
            diagnostics["refresh_predictor_stats"] = predictor_diagnostics
        predictors_by_shape = {}
        for shape, predictor_sums in sorted(shape_predictor_sums.items()):
            shape_diagnostics = {}
            for name, sums in predictor_sums.items():
                predictor = refresh_predictor_summary(sums, name)
                if predictor is not None:
                    shape_diagnostics[name] = predictor
            if shape_diagnostics:
                predictors_by_shape[shape] = shape_diagnostics
        if predictors_by_shape:
            diagnostics["refresh_predictor_stats_by_shape"] = (
                predictors_by_shape
            )
    return diagnostics


def _autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=dtype != torch.float32,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    maximum_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for index, (inputs, targets) in enumerate(loader):
        if maximum_batches > 0 and index >= maximum_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast_context(device, dtype):
            loss = model(inputs, targets)
        tokens = targets.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("evaluation loader produced no batches")
    return total_loss / total_tokens


def _infinite(loader: DataLoader) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


class StatefulBatchStream:
    """Infinite DataLoader stream with an exactly replayable shuffle position."""

    def __init__(
        self,
        loader: DataLoader,
        generator: torch.Generator,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.loader = loader
        self.generator = generator
        self.epoch_generator_state = torch.empty(0, dtype=torch.uint8)
        self.batches_consumed = 0
        self.iterator: Iterator[tuple[Tensor, Tensor]]
        if state is None:
            self._start_epoch()
        else:
            self.generator.set_state(state["epoch_generator_state"])
            self._start_epoch()
            expected = int(state["batches_consumed"])
            if not 0 <= expected <= len(loader):
                raise ValueError("checkpoint has an invalid training-loader position")
            for _ in range(expected):
                try:
                    next(self.iterator)
                except StopIteration as error:
                    raise ValueError(
                        "checkpoint training-loader position exceeds the epoch"
                    ) from error
                self.batches_consumed += 1

    def _start_epoch(self) -> None:
        # DataLoader and RandomSampler both consume this dedicated generator.
        # Saving its state before iterator creation reproduces the same worker
        # seed and permutation, after which replay only has to skip delivered
        # batches within the epoch.
        self.epoch_generator_state = self.generator.get_state().clone()
        self.iterator = iter(self.loader)
        self.batches_consumed = 0

    def __next__(self) -> tuple[Tensor, Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._start_epoch()
            batch = next(self.iterator)
        self.batches_consumed += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch_generator_state": self.epoch_generator_state.clone(),
            "batches_consumed": self.batches_consumed,
        }


def _schedule_multiplier(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    if args.scheduler == "constant":
        return 1.0
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            arguments,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": command("git", "rev-parse", "HEAD") or None,
        "dirty": bool(command("git", "status", "--porcelain")),
    }


_OPERATIONAL_CONFIG_KEYS = {
    "checkpoint",
    "checkpoint_every",
    "config",
    "output",
    "resume",
    "stop_after_step",
}


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
        if key not in _OPERATIONAL_CONFIG_KEYS
    }


def _load_resume_checkpoint(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.resume is None:
        return None
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "resume checkpoint is missing the exact-training format version"
        )
    saved_config = _comparable_config(checkpoint["config"])
    current_config = _comparable_config(vars(args))
    mismatches = {
        key: (saved_config.get(key), current_config.get(key))
        for key in saved_config.keys() | current_config.keys()
        if saved_config.get(key) != current_config.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}={old!r}->{new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(f"resume configuration does not match checkpoint: {details}")
    return checkpoint


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _move_optimizer_value_exact(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _move_optimizer_value_exact(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_optimizer_value_exact(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_value_exact(item, device) for item in value)
    return copy.deepcopy(value)


def _load_optimizer_state_exact(optimizer, saved: dict[str, Any]) -> None:
    """Load optimizer state without casting BF16 caches to parameter dtype."""
    optimizer.load_state_dict(saved)
    saved_ids = [
        parameter_id
        for group in saved["param_groups"]
        for parameter_id in group["params"]
    ]
    current_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(saved_ids) != len(current_parameters):
        raise ValueError("optimizer checkpoint parameter count does not match")
    parameter_by_saved_id = dict(zip(saved_ids, current_parameters))
    optimizer.state.clear()
    for parameter_id, state in saved["state"].items():
        parameter = parameter_by_saved_id[parameter_id]
        optimizer.state[parameter] = _move_optimizer_value_exact(
            state,
            parameter.device,
        )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("steps and gradient_accumulation_steps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if args.checkpoint_every > 0 and args.checkpoint is None:
        raise ValueError("checkpoint_every requires --checkpoint")
    if args.stop_after_step < 0 or args.stop_after_step > args.steps:
        raise ValueError("stop_after_step must be between zero and steps")
    if args.stop_after_step > 0 and args.checkpoint is None:
        raise ValueError("stop_after_step requires --checkpoint")
    if (args.checkpoint is not None or args.resume is not None) and args.muon_async_checks:
        raise ValueError(
            "exact checkpoints do not support pending asynchronous Muon checks"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    resume_checkpoint = _load_resume_checkpoint(args)

    train_dataset, validation_dataset, vocab_size = build_datasets(args)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    if len(train_loader) == 0:
        raise ValueError("training dataset is smaller than one full batch")

    model = Transformer(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=args.ff_multiplier,
        ff_hidden_dim=args.ff_hidden_dim,
        vocab_size=vocab_size,
        max_seq_len=args.sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
        use_rope=args.rope,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        norm_eps=args.norm_eps,
        tie_embeddings=args.tie_embeddings,
        input_projection=args.input_projection,
        bias=args.bias,
        dropout=args.dropout,
        fused_qkv=args.fused_qkv,
        fused_swiglu=args.fused_swiglu,
    ).to(device)
    optimizer = build_optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _schedule_multiplier(step, args),
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        _load_optimizer_state_exact(optimizer, resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
    training_model = torch.compile(
        model,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
    ) if args.compile else model

    start_step = int(resume_checkpoint["step"]) if resume_checkpoint else 0
    if not 0 <= start_step <= args.steps:
        raise ValueError(
            f"resume step {start_step} must not exceed target steps {args.steps}"
        )
    execution_end = (
        min(args.steps, args.stop_after_step)
        if args.stop_after_step > 0
        else args.steps
    )
    if execution_end <= start_step and not (
        start_step == execution_end == args.steps and resume_checkpoint is not None
    ):
        raise ValueError(
            f"stop_after_step {execution_end} must exceed resume step {start_step}"
        )
    history: list[dict[str, Any]] = (
        list(resume_checkpoint["history"]) if resume_checkpoint else []
    )
    stream = StatefulBatchStream(
        train_loader,
        generator,
        resume_checkpoint["data_stream"] if resume_checkpoint else None,
    )
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    if resume_checkpoint is not None:
        _restore_rng_state(resume_checkpoint["rng"])
    wall_start = time.perf_counter()
    saved_timing = resume_checkpoint.get("timing", {}) if resume_checkpoint else {}
    timed_tokens = int(saved_timing.get("timed_tokens", 0))
    prior_wall_seconds = float(saved_timing.get("wall_seconds", 0.0))
    prior_peak_memory = int(saved_timing.get("peak_memory_bytes", 0))
    checkpoint_io_seconds = 0.0
    step_events = []
    optimizer_events = []
    step_measurements: list[float] = list(
        saved_timing.get("step_measurements", [])
    )
    optimizer_measurements: list[float] = list(
        saved_timing.get("optimizer_measurements", [])
    )
    optimizer_event_kinds: list[str] = list(
        saved_timing.get("optimizer_event_kinds", [])
    )
    last_loss = float(resume_checkpoint.get("last_loss", math.nan)) if resume_checkpoint else math.nan
    last_checkpoint_step = start_step

    def drain_cuda_timings() -> None:
        if device.type != "cuda" or not step_events:
            return
        torch.cuda.synchronize(device)
        step_measurements.extend(
            start.elapsed_time(end) for start, end in step_events
        )
        optimizer_measurements.extend(
            start.elapsed_time(end) for start, end in optimizer_events
        )
        step_events.clear()
        optimizer_events.clear()

    def save_checkpoint(step: int) -> None:
        nonlocal checkpoint_io_seconds, last_checkpoint_step
        if args.checkpoint is None:
            return
        drain_cuda_timings()
        peak_memory_bytes = prior_peak_memory
        if device.type == "cuda":
            peak_memory_bytes = max(
                peak_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
        checkpoint_start = time.perf_counter()
        _atomic_torch_save(
            {
                "format_version": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": vars(args),
                "step": step,
                "history": history,
                "last_loss": last_loss,
                "data_stream": stream.state_dict(),
                "rng": _rng_state(),
                "timing": {
                    "timed_tokens": timed_tokens,
                    "step_measurements": step_measurements,
                    "optimizer_measurements": optimizer_measurements,
                    "optimizer_event_kinds": optimizer_event_kinds,
                    "wall_seconds": prior_wall_seconds
                    + (time.perf_counter() - wall_start - checkpoint_io_seconds),
                    "peak_memory_bytes": peak_memory_bytes,
                },
            },
            Path(args.checkpoint),
        )
        checkpoint_io_seconds += time.perf_counter() - checkpoint_start
        last_checkpoint_step = step

    for step in range(start_step + 1, execution_end + 1):
        timed = step > args.timing_warmup
        if timed and device.type == "cuda":
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            optimizer_start = torch.cuda.Event(enable_timing=True)
            optimizer_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
        accumulated_loss = torch.zeros((), device=device)
        for _ in range(args.gradient_accumulation_steps):
            inputs, targets = next(stream)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device, dtype):
                loss = training_model(inputs, targets)
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += loss.detach()
            if timed:
                timed_tokens += targets.numel()
        if args.max_grad_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
            grad_norm = torch.zeros((), device=device)
        if timed and device.type == "cuda":
            optimizer_start.record()
        optimizer.step()
        if timed and device.type == "cuda":
            optimizer_end.record()
            if int(getattr(optimizer, "last_hard_reset_events", 0)) > 0:
                optimizer_event_kinds.append("hard_reset")
            elif int(getattr(optimizer, "last_basis_refreshes", 0)) > 0:
                optimizer_event_kinds.append("warm_refresh")
            else:
                optimizer_event_kinds.append("ordinary")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if timed and device.type == "cuda":
            step_end.record()
            step_events.append((step_start, step_end))
            optimizer_events.append((optimizer_start, optimizer_end))

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        should_evaluate = step % args.eval_every == 0 or step == args.steps
        if should_log or should_evaluate:
            last_loss = float(accumulated_loss / args.gradient_accumulation_steps)
            if not math.isfinite(last_loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
        record: dict[str, Any] = {}
        if should_log:
            record = {
                "step": step,
                "train_loss": last_loss,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "grad_norm": float(grad_norm),
            }
        if should_evaluate:
            record.setdefault("step", step)
            record["validation_loss"] = evaluate(
                training_model,
                validation_loader,
                device,
                dtype,
                args.eval_batches,
            )
        if record:
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if (
            args.checkpoint_every > 0
            and step % args.checkpoint_every == 0
        ):
            save_checkpoint(step)

    if args.checkpoint is not None and last_checkpoint_step != execution_end:
        save_checkpoint(execution_end)

    if device.type == "cuda":
        drain_cuda_timings()
        step_ms = sum(step_measurements)
        optimizer_ms = sum(optimizer_measurements)
        optimizer_timing_by_kind = {}
        for kind in ("ordinary", "warm_refresh", "hard_reset"):
            measurements = [
                value
                for value, event_kind in zip(
                    optimizer_measurements, optimizer_event_kinds
                )
                if event_kind == kind
            ]
            if measurements:
                optimizer_timing_by_kind[kind] = {
                    "samples": len(measurements),
                    "mean_milliseconds": sum(measurements) / len(measurements),
                    "p50_milliseconds": _percentile(measurements, 0.50),
                    "p95_milliseconds": _percentile(measurements, 0.95),
                    "maximum_milliseconds": max(measurements),
                }
        peak_memory = max(
            prior_peak_memory,
            int(torch.cuda.max_memory_allocated(device)),
        )
    else:
        step_measurements = []
        optimizer_measurements = []
        step_ms = math.nan
        optimizer_ms = math.nan
        optimizer_timing_by_kind = {}
        peak_memory = 0
    wall_seconds = prior_wall_seconds + (
        time.perf_counter() - wall_start - checkpoint_io_seconds
    )
    measured_steps = max(1, len(step_measurements))
    final_validation = evaluate(
        training_model,
        validation_loader,
        device,
        dtype,
        args.eval_batches,
    )
    result = {
        "config": vars(args),
        "git": _git_metadata(Path(__file__).resolve().parent),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "vocab_size": vocab_size,
            "architecture": (
                "embedding->input_projection->pre_norm_blocks->final_norm->lm_head"
                if args.input_projection
                else "embedding->pre_norm_blocks->final_norm->lm_head"
            ),
            "fused_qkv": args.fused_qkv,
            "fused_swiglu": args.fused_swiglu,
            "qk_norm": args.qk_norm,
        },
        "data": {
            "train_sequences": len(train_dataset),
            "validation_sequences": len(validation_dataset),
            "train_tokens": int(train_dataset.tokens.numel()),
            "validation_tokens": int(validation_dataset.tokens.numel()),
            "streaming_source": bool(args.dataset == "hf" and args.hf_streaming),
            "token_cache_megabytes": (
                args.token_cache.stat().st_size / 2**20
                if args.token_cache is not None and args.token_cache.exists()
                else 0.0
            ),
        },
        "progress": {
            "completed": execution_end == args.steps,
            "completed_steps": execution_end,
            "target_steps": args.steps,
            "resumed_from_step": start_step,
        },
        "final": {
            "train_loss": last_loss,
            "validation_loss": final_validation,
            "perplexity": math.exp(min(20.0, final_validation)),
            "step_milliseconds": step_ms / measured_steps,
            "optimizer_milliseconds": optimizer_ms / measured_steps,
            "optimizer_p50_milliseconds": _percentile(
                optimizer_measurements, 0.50
            ),
            "optimizer_p95_milliseconds": _percentile(
                optimizer_measurements, 0.95
            ),
            "optimizer_p99_milliseconds": _percentile(
                optimizer_measurements, 0.99
            ),
            "optimizer_max_milliseconds": max(
                optimizer_measurements, default=math.nan
            ),
            "optimizer_timing_by_kind": optimizer_timing_by_kind,
            "tokens_per_second": (
                timed_tokens / (step_ms / 1000.0) if step_ms > 0.0 else math.nan
            ),
            "wall_seconds": wall_seconds,
            "peak_memory_megabytes": peak_memory / 2**20,
        },
        "optimizer": optimizer_diagnostics(optimizer),
        "history": history,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    default_optimizer = "transport_muon" if HAS_TRANSPORT_MUON else "soap"
    default_learning_rate = 3e-4 if HAS_TRANSPORT_MUON else 2.5e-3
    default_beta1 = 0.9 if HAS_TRANSPORT_MUON else 0.95
    choices = ["adamw"]
    if HAS_TRANSPORT_MUON:
        choices.append("transport_muon")
    if HAS_TURBO_SOAP:
        choices.append("soap")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON object of argument defaults")
    parser.add_argument("--optimizer", choices=choices, default=default_optimizer)
    parser.add_argument("--dataset", choices=("synthetic", "hf"), default="synthetic")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-train-documents", type=int, default=10_000)
    parser.add_argument("--max-validation-documents", type=int, default=1_000)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-validation-tokens", type=int, default=0)
    parser.add_argument("--hf-streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-validation-documents", type=int, default=256)
    parser.add_argument(
        "--allow-large-hf-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--synthetic-train-samples", type=int, default=4096)
    parser.add_argument("--synthetic-validation-samples", type=int, default=512)
    parser.add_argument("--synthetic-vocab-size", type=int, default=256)

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=float, default=8.0 / 3.0)
    parser.add_argument("--ff-hidden-dim", type=int)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fused-qkv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=default_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=default_beta1)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--timing-warmup", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="atomically updated exact-training checkpoint",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help="stop this invocation early while retaining --steps as the schedule horizon",
    )

    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-split-qkv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-split-swiglu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-anchor-every", type=int, default=8)
    parser.add_argument("--muon-max-age", type=int, default=0)
    parser.add_argument("--muon-check-every", type=int, default=1)
    parser.add_argument("--muon-full-ns-steps", type=int, default=0)
    parser.add_argument("--muon-retract-steps", type=int, default=1)
    parser.add_argument("--muon-retract-every", type=int, default=1)
    parser.add_argument(
        "--muon-retract-method",
        choices=("higham_cubic", "quadratic"),
        default="higham_cubic",
    )
    parser.add_argument("--muon-jacobi-damping", choices=("floor", "tikhonov"), default="tikhonov")
    parser.add_argument("--muon-max-tangent-rms", type=float, default=0.0)
    parser.add_argument("--muon-record-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-normal-inv-cap", type=float, default=0.0)
    parser.add_argument("--muon-normal-inv-ema-ratio", type=float, default=0.0)
    parser.add_argument("--muon-normal-inv-ema-beta", type=float, default=0.95)
    parser.add_argument("--muon-max-angular-rms", type=float, default=0.0)
    parser.add_argument("--muon-max-skew-ratio", type=float, default=0.0)
    parser.add_argument(
        "--muon-signal-check-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--muon-separate-skew-signal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--muon-async-checks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--muon-update-stats-every", type=int, default=0)
    parser.add_argument("--muon-reference-lr", type=float, default=0.0)
    parser.add_argument("--muon-normalize-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-output-scale", type=float, default=1.0)
    parser.add_argument("--muon-output-scale-start", type=float)
    parser.add_argument("--muon-output-scale-decay", type=float, default=1.0)
    parser.add_argument("--muon-legacy-retraction-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-spectral-cap-mode", choices=("gershgorin", "power", "power_step"), default="gershgorin")
    parser.add_argument("--muon-power-steps", type=int, default=2)
    parser.add_argument("--muon-power-safety-factor", type=float, default=1.25)
    parser.add_argument("--muon-power-refine-threshold", type=float, default=0.0)
    parser.add_argument("--muon-large-tensor-threshold", type=int, default=16_384)

    parser.add_argument("--soap-shampoo-beta", type=float, default=0.999)
    parser.add_argument("--soap-fallback-lr", type=float, default=3e-4)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument(
        "--soap-precondition-frequency-after-warmup", type=int, default=0
    )
    parser.add_argument(
        "--soap-precondition-frequency-warmup-steps", type=int, default=0
    )
    parser.add_argument("--soap-residual-threshold", type=float, default=0.0)
    parser.add_argument("--soap-residual-max-age", type=int, default=40)
    parser.add_argument("--soap-residual-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--soap-precondition-mode",
        choices=("all", "smaller_side", "aspect_ratio"),
        default="smaller_side",
    )
    parser.add_argument("--soap-precondition-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--soap-max-precond-dim", type=int, default=10_000)
    parser.add_argument("--soap-normalize-grads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--soap-basis-lr-age-compensation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--soap-basis-lr-reference-age", type=int, default=0)
    parser.add_argument("--soap-reset-frequency", type=int, default=20)
    parser.add_argument("--soap-reset-max-age", type=int, default=0)
    parser.add_argument(
        "--soap-reset-stagger",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--soap-reset-method", choices=("qr", "eigh"), default="qr")
    parser.add_argument("--soap-track-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--soap-covariance-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config = json.loads(preliminary.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("config must contain a JSON object")
        known = {action.dest for action in parser._actions}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.output is not None:
        args.output = Path(args.output)
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
    if args.resume is not None:
        args.resume = Path(args.resume)
    if args.cache_dir is not None:
        args.cache_dir = Path(args.cache_dir)
    if args.token_cache is not None:
        args.token_cache = Path(args.token_cache)
    return args


def main() -> None:
    args = parse_args()
    result = train(args)
    serialized = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized + "\n")
        os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
