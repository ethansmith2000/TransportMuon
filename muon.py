"""Shared Muon/Adam helpers used by the experimental optimizers.

The optimizer implementations in this repository import ``adam_update`` and the
README uses ``get_muon_param_groups``. Keeping these small helpers locally makes
the repository runnable without relying on an untracked sibling module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def adam_update(
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> Tensor:
    """Update Adam moments in place and return the bias-corrected direction."""
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**int(step)
    bias_correction2 = 1.0 - beta2**int(step)
    denominator = exp_avg_sq.sqrt() / bias_correction2**0.5
    return (exp_avg / bias_correction1) / denominator.add(float(eps))


def get_muon_param_groups(
    model: nn.Module,
    *,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adam_lr: float = 3e-4,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-10,
    adam_weight_decay: float = 0.0,
    large_tensor_threshold: int = 16384,
    muon_predicate: Callable[[str, Tensor], bool] | None = None,
) -> list[dict]:
    """Split trainable parameters into Muon and Adam groups.

    By default, matrices and higher-rank tensors use Muon when every dimension
    is at most ``large_tensor_threshold``. Embedding tables, vectors, scalars,
    and larger axes use Adam. ``muon_predicate`` can override this decision for
    non-embedding parameters using their ``named_parameters`` name and tensor.
    """
    if int(large_tensor_threshold) < 1:
        raise ValueError("large_tensor_threshold must be >= 1")

    embedding_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    muon_parameters: list[Tensor] = []
    adam_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_muon = (
            parameter.ndim >= 2
            and max(parameter.shape) <= int(large_tensor_threshold)
            and id(parameter) not in embedding_parameters
        )
        if muon_predicate is not None and id(parameter) not in embedding_parameters:
            use_muon = bool(muon_predicate(name, parameter))
        (muon_parameters if use_muon else adam_parameters).append(parameter)

    groups: list[dict] = []
    if muon_parameters:
        groups.append(
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": float(muon_lr),
                "momentum": float(muon_momentum),
                "weight_decay": float(muon_weight_decay),
            }
        )
    if adam_parameters:
        groups.append(
            {
                "params": adam_parameters,
                "use_muon": False,
                "lr": float(adam_lr),
                "betas": (float(adam_betas[0]), float(adam_betas[1])),
                "eps": float(adam_eps),
                "weight_decay": float(adam_weight_decay),
            }
        )
    return groups


__all__ = ["adam_update", "get_muon_param_groups"]
