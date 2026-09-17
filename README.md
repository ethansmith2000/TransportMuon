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
- `muon_polar_method`: full-anchor polynomial schedule. `"muon"` keeps the
  fixed quintic baseline; `"polar_express"` uses iteration-specific minimax
  quintics. For the latter, start with `muon_ns_steps=6` and
  `muon_warm_anchor_retract_steps=0`.
- `muon_nesterov`: whether to use Nesterov momentum before orthogonalization. Default: `True`.

### Warm-Start Arguments

- `muon_warm_anchor_every`: run a full Newton-Schulz anchor every N steps. Set to `0` to disable periodic anchors after initialization and full-NS warmup. Default: `8`.
- `muon_warm_lr`: step size for the warm Jacobi tangent update between anchors. Default: `1.0` yields full correction.
- `muon_warm_jacobi_eps`: denominator floor used in the Jacobi scaling. Larger values are more conservative near small or ill-conditioned diagonals. Default: `1e-3`.
- `muon_warm_retract_method`: retraction used after warm updates. Supported values are `"higham_cubic"` and `"quadratic"`. Default: `"higham_cubic"`.
- `muon_warm_retract_steps`: retraction iterations for warm intermediate steps. Default: `1`.
- `muon_warm_anchor_retract_method`: retraction used after full NS anchors.
  Default: `"higham_cubic"`, independent of the warm-step method.
- `muon_warm_anchor_retract_steps`: polishing iterations after a full anchor.
  Default: `2`.
- `muon_warm_full_ns_steps`: always use full Newton-Schulz anchors for the first N optimizer steps. Default: `2500`.
- `muon_warm_track_subspace`: include the complementary tangent component needed
  to follow row-space motion in rectangular matrices. Default: `True`; set it to
  `False` to reproduce the original rotation-only transport.
- `muon_warm_max_tracking_error`: adaptively anchor when the rotation or row-space
  residual exceeds this value. `0` disables the check. The check synchronizes
  the device, so combine it with `muon_warm_check_every > 1` for large models.
- `muon_warm_min_alignment`: adaptively anchor if the smallest diagonal alignment,
  relative to the mean absolute diagonal, falls below this value.
- `muon_warm_check_every`: interval for adaptive tracking checks. Default: `1`.
- `muon_warm_stagger_anchors`: distribute scheduled anchors across parameters to
  smooth optimizer latency. Default: `False`.
- `muon_warm_jacobi_damping`: `"floor"` preserves the original signed-floor
  inverse; `"tikhonov"` smoothly suppresses poorly determined corrections.
- `muon_warm_max_tangent_rms`: cap the normalized Frobenius magnitude of a
  proposed warm tangent step by reducing its effective `muon_warm_lr`. `0`
  disables the cap. This device-side trust region has no host synchronization.
- `muon_warm_alignment_tolerance`: accept a warm step only when its alignment
  with the current momentum does not fall by more than this relative tolerance.
  Negative values disable the gate. The decision and fallback stay on-device.
- `muon_warm_record_stats`: use the instrumented warm path and record proposed
  tangent RMS even when both controllers are disabled. Default: `False`, which
  preserves the fastest path.
- `muon_warm_max_tangent_ratio`: tensor-wide scalar preconditioner that limits
  the current tangent RMS to this multiple of its historical EMA. `0` disables
  it. Unlike coordinatewise scaling, this preserves the transport direction and
  its tangency. A value around `2` is a reasonable experimental spike limiter.
- `muon_warm_tangent_ema_beta`: decay for that scalar tangent-RMS history.
  Default: `0.95`.

## Tuning Notes

- Start from `muon_lr=0.02`, `muon_momentum=0.95`, and `muon_ns_steps=5` if you are matching common Muon settings.
- Increase `muon_warm_full_ns_steps` when early training is unstable or when the cached direction needs more time to settle.
- Decrease `muon_warm_anchor_every` to refresh more often. This is more expensive but keeps the cached direction closer to a full Muon update.
- Keep `muon_warm_lr=1.0` as the local full-correction step and use
  `muon_warm_max_tangent_rms` to shrink only unusually large moves. This is a
  more targeted experiment than globally lowering the warm learning rate.
- Use a small non-negative `muon_warm_alignment_tolerance` as a safety valve;
  pair it with periodic adaptive-anchor checks so repeated rejected steps cause
  a fresh solve instead of leaving a stale direction indefinitely.
- A useful experiment is `muon_warm_retract_method="quadratic"` with one warm
  step while retaining the default two cubic anchor polishes. The quadratic
  warm path was roughly 17--22% faster in the included microbenchmark; its
  training stability still needs measurement.
- Retractions evaluate their polynomials in residual form around `G = I`. This
  is algebraically unchanged but avoids BF16 cancellation when the candidate is
  already close to orthogonal. Full anchors also skip the loose Gershgorin cap:
  normalized Muon NS keeps their largest singular value below the retraction's
  safety limit. Warm candidates retain the cap because their spectrum is not
  bounded by the Muon polynomial.

## Implementation Notes

- The optimizer stores one momentum buffer per Muon parameter and one cached `muon_warm_q` direction.
- `muon_warm_did_anchor` and `muon_warm_age` in each parameter's optimizer
  state expose the actual refresh pattern for training diagnostics. Enabled
  trust controllers also record their effective step and acceptance as scalar
  device tensors.
- Full anchor steps use the same quintic Newton-Schulz coefficients as baseline Muon.
- Warm steps update `q_prev` with a skew-symmetric Jacobi correction, then retract rows back toward orthonormality.
- Missing gradients are replaced with zero tensors inside `step()`, which can force synchronization in distributed setups.
- The optimizer expects param groups with a `use_muon` boolean. Groups with `use_muon=True` use MuonWarm; groups with `use_muon=False` use Adam.

Run `python benchmarks/benchmark_transport.py` to compare rotation-only warm
transport, full rectangular tangent transport, the guarded warm path, and a
fresh solver-matched anchor.
Pass `--retract-method`, `--warm-retract-steps`, and
`--anchor-retract-steps` to compare the quadratic and cubic maps directly.
Use `--polar-method polar_express --polar-steps 6 --anchor-retract-steps 0`
to benchmark the dynamic-coefficient anchor.

For a solver-matched training baseline, set `muon_warm_anchor_every=1` and
`muon_warm_full_ns_steps=0`. This runs the same five quintic Newton--Schulz steps
and two cubic polishing steps used by `MuonWarm` anchors on every update. Compare
that run with a fixed-schedule warm run, then enable the trust cap and alignment
gate separately. This separates gains from transport reuse from gains caused by
the sharper anchor solve or a changed effective update scale.

## Experimental leverage-balanced variant

`muon_warm_aurora.py` provides `MuonWarmAurora`, which combines warm polar
transport with Aurora-style leverage balancing for tall 2D parameters:

```python
from muon_warm_aurora import MuonWarmAurora

optimizer = MuonWarmAurora(
    param_groups,
    muon_warm_track_subspace=True,
    muon_warm_jacobi_damping="tikhonov",
    aurora_anchor_iterations=2,
    aurora_balance_power=0.5,
    aurora_warm_balance_power=0.25,
)
```

Tall matrices are transposed internally by MuonWarm. Consequently, uniform row
leverage in the returned update is implemented as uniform column norms in the
cached prepared factor. Square and wide parameters continue to use the standard
Transport Muon path. The optimizer records leverage diagnostics as scalar
tensors in `aurora_leverage_cv_tensor`, `aurora_min_relative_leverage_tensor`,
and `aurora_max_relative_leverage_tensor` without synchronizing the device.

Run `python benchmarks/benchmark_aurora_transport.py` to measure the anchor and
warm-step overhead alongside the resulting leverage uniformity.

References: [Aurora method](https://blog.tilderesearch.com/blog/aurora) and
[reference implementation](https://github.com/tilde-research/aurora-release).

The optional dynamic anchor follows [The Polar Express](https://arxiv.org/abs/2505.16932)
and its [reference implementation](https://github.com/NoahAmsel/PolarExpress).

## Experimental Procrustes transport

`muon_warm_procrustes.py` provides `MuonWarmProcrustes`. It replaces the
diagonal Jacobi approximation for rotation inside the cached row space with a
small orthogonal Procrustes solve:

```text
R = Q @ M.T
O = polar(R.T)
Q_rotated = O @ Q
```

The polar solve is only on the smaller square dimension. The complementary
row-space correction retains a damped stretch approximation and can be bounded
without host synchronization:

```python
from muon_warm_procrustes import MuonWarmProcrustes

optimizer = MuonWarmProcrustes(
    param_groups,
    muon_warm_inner_ns_steps=5,
    muon_warm_inner_retract_steps=2,
    muon_warm_max_normal_rms=0.02,
)
```

`muon_warm_max_normal_rms=0` disables the trust cap. When enabled, the optimizer
records the proposed normal motion, effective step size, and applied motion as
scalar device tensors. The `0.02` value above is an experimental starting point,
not a tuned default. The base `muon_warm_alignment_tolerance` gate also applies
to this variant.

Two residual-form inner polishing steps are the default. In the controlled
benchmark, a third did not improve tracking error, while one left the small
polar solve under-converged.

Run `python benchmarks/benchmark_transport_solvers.py` to compare the Jacobi and
Procrustes paths on a controlled dense-stretch and row-space drift problem.
