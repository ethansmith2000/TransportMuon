# MuonWarm Optimizer

Blog post: [https://ethansmith2000.substack.com/p/transport-muon-beating-muon-in-speed](https://ethansmith2000.substack.com/p/transport-muon-beating-muon-in-speed)

`muon_warm.py` defines `MuonWarm`, a PyTorch optimizer for matrix-shaped neural network parameters. It is a Muon-style optimizer that orthogonalizes momentum updates with Newton-Schulz iterations, then reuses a cached polar direction between full refreshes to reduce per-step work. Non-matrix parameters are handled with an Adam update.

## What It Does

`MuonWarm` splits parameters into two update paths:

- Muon path: for 2D+ tensors such as linear weights and convolution kernels.
- Adam path: for 1D tensors and other parameters you explicitly route away from Muon.

For Muon parameters, each step:

1. Updates a momentum buffer.
2. Forms the Nesterov or standard momentum update.
3. Flattens 4D convolution kernels into matrices.
4. Computes or refreshes a near-orthogonal update direction.
5. Applies decoupled weight decay and the parameter update.

The "warm" part keeps `state["muon_warm_q"]`, a cached orthogonalized direction. Early steps and periodic anchor steps use full Newton-Schulz orthogonalization. Intermediate steps update the cached direction with a Jacobi-scaled tangent step and retract it back toward the row-orthonormal manifold.

## Minimal Usage

Use the `get_muon_param_groups` helper from `muon.py` to route matrix parameters to Muon and biases, norms, embeddings, or very large tensors to Adam.

```python
from muon import get_muon_param_groups
from muon_warm import MuonWarm

param_groups = get_muon_param_groups(
    model,
    muon_lr=0.02,
    muon_momentum=0.95,
    muon_weight_decay=weight_decay,
    adam_lr=3e-4,
    adam_betas=(0.9, 0.95),
    adam_weight_decay=0.0,
    large_tensor_threshold=16384,
)

optimizer = MuonWarm(
    param_groups,
    muon_lr=0.02,
    muon_momentum=0.95,
    muon_ns_steps=5,
    muon_warm_anchor_every=200,
    muon_warm_lr=1.0,
    muon_warm_jacobi_eps=1e-3,
    muon_warm_retract_method="higham_cubic",
    muon_warm_retract_steps=1,
    muon_warm_full_ns_steps=250,
)

loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

## Main Arguments

### Adam Arguments

- `adam_lr`: learning rate for Adam-routed parameters. Default: `3e-4`.
- `adam_betas`: Adam beta values. Default: `(0.9, 0.999)`.
- `adam_eps`: Adam epsilon. Default: `1e-10`.
- `adam_weight_decay`: decoupled weight decay for Adam-routed parameters. Default: `0`.

### Muon Arguments

- `muon_lr`: learning rate for Muon-routed parameters. Default: `0.02`.
- `muon_momentum`: Muon momentum beta. Default: `0.95`.
- `muon_weight_decay`: decoupled weight decay for Muon-routed parameters. Default: `0`.
- `muon_ns_steps`: Newton-Schulz iterations used on full anchor refreshes. Default: `5`.
- `muon_nesterov`: whether to use Nesterov momentum before orthogonalization. Default: `True`.

### Warm-Start Arguments

- `muon_warm_anchor_every`: run a full Newton-Schulz anchor every N steps. Set to `0` to disable periodic anchors after initialization and full-NS warmup. Default: `8`.
- `muon_warm_lr`: step size for the warm Jacobi tangent update between anchors. Default: `1.0` yields full correction.
- `muon_warm_jacobi_eps`: denominator floor used in the Jacobi scaling. Larger values are more conservative near small or ill-conditioned diagonals. Default: `1e-3`.
- `muon_warm_retract_method`: retraction used after warm updates. Supported values are `"higham_cubic"` and `"quadratic"`. Default: `"higham_cubic"`.
- `muon_warm_retract_steps`: retraction iterations for warm intermediate steps. Default: `1`.
- `muon_warm_full_ns_steps`: always use full Newton-Schulz anchors for the first N optimizer steps. Default: `2500`.

## Tuning Notes

- Start from `muon_lr=0.02`, `muon_momentum=0.95`, and `muon_ns_steps=5` if you are matching common Muon settings.
- Increase `muon_warm_full_ns_steps` when early training is unstable or when the cached direction needs more time to settle.
- Decrease `muon_warm_anchor_every` to refresh more often. This is more expensive but keeps the cached direction closer to a full Muon update.
- Increase `muon_warm_jacobi_eps` or decrease `muon_warm_lr` if warm updates are too aggressive.
- Use `"higham_cubic"` first. Try `"quadratic"` if you want a simpler, potentially cheaper retraction.

## Implementation Notes

- The optimizer stores one momentum buffer per Muon parameter and one cached `muon_warm_q` direction.
- Full anchor steps use the same quintic Newton-Schulz coefficients as baseline Muon.
- Warm steps update `q_prev` with a skew-symmetric Jacobi correction, then retract rows back toward orthonormality.
- Missing gradients are replaced with zero tensors inside `step()`, which can force synchronization in distributed setups.
- The optimizer expects param groups with a `use_muon` boolean. Groups with `use_muon=True` use MuonWarm; groups with `use_muon=False` use Adam.
