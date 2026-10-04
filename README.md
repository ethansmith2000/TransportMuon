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
- `muon_warm_retract_steps`: retraction iterations for warm intermediate steps.
  Default: `1`; `0` now means an exact no-op retraction for controlled lazy or
  no-retraction experiments.
- `muon_warm_retract_every`: apply the configured warm retraction every N warm
  steps. Default: `1`. Values above one are experimental and expose
  `state["muon_warm_did_retract"]` for diagnostics.
- `muon_warm_spectral_cap_mode`: spectral guard applied before a warm
  retraction. `"gershgorin"` uses the certified absolute-row-sum bound and is
  the default. `"power"` uses a cached power vector plus a fresh
  largest-diagonal coordinate seed. `"power_step"` probes the full correction,
  reduces its effective transport step while retaining the cached basis, and
  checks the adjusted candidate before retraction. Both power modes are
  heuristic experiments.
- `muon_warm_power_steps`: square-Gram power iterations for the heuristic cap.
  Default: `2`.
- `muon_warm_power_safety_factor`: multiplier on the estimated top singular
  value before clipping to the retraction limit. Default: `1.25`. Smaller
  factors preserve more magnitude but can underestimate abrupt rotations.
- `muon_warm_power_refine_threshold`: experimental device-side uncertainty
  threshold for conditionally running the remaining power iterations. `0`
  disables it. The width-768 component benchmark found that `torch.cond`
  increased the complete skew path by about 21% even when the second iteration
  never ran, so this is retained only to reproduce that rejected experiment.
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
- `muon_warm_min_stretch`: adaptively anchor if the smallest eigenvalue of the
  symmetric alignment, relative to mean absolute eigenvalue, falls below this
  value. This detects a symmetric but indefinite wrong polar branch that skew
  and subspace checks miss. `0` disables it.
- `muon_warm_max_age`: anchor after this many consecutive warm steps even when
  no diagnostic threshold fires. `0` disables the age limit.
- `muon_warm_max_angular_rms`: anchor when the previous warm step's proposed
  in-row-space angular correction exceeds this RMS threshold. `0` disables the
  controller. Checks are batched into one device-to-host transfer every
  `muon_warm_check_every` steps. The controller supports `"gershgorin"` and
  `"power_step"`; the latter uses a dedicated angular-only compiled path rather
  than the full statistics kernel. With the local output filter, threshold `2`
  checked every four steps passed the three-seed OpenWebText gate and repaired
  the batch-480 teacher stress failure, then matched the ungated profile at
  2,000 steps. The raw threshold remains experimental because it is sensitive
  to dtype and compiled arithmetic.
- `muon_warm_max_skew_ratio`: anchor when
  `||skew(Q M^T)||_F / ||Q M^T||_F` from the previous warm step exceeds this
  dimensionless threshold. `0` disables it. It is mutually exclusive with the
  raw angular-RMS controller and currently requires `"power_step"`. A dedicated
  compiled path reuses the alignment product already needed by transport, then
  batches the scalar checks at `muon_warm_check_every` just like the angular
  controller. Threshold `0.52`, checked every four steps, passed the CUDA
  batch-480 teacher stress screen, the three-seed 1,000-step OpenWebText gate,
  and the 2,000-step survivor. It is the current dimensionless quality profile:
  its 1,000-step mean loss (`6.54368`) nearly matches raw angular threshold 2
  (`6.54325`), while its 2,000-step loss improves from `6.22535` to `6.21645`.
  It uses about twice as many adaptive anchors and roughly `0.97` ms more
  optimizer time at 1,000 steps, so the raw angular controller remains the
  cheaper robust profile.
- `muon_warm_async_checks`: replace the periodic blocking scalar read with a
  nonblocking copy to reusable pinned host memory. A CUDA event is polled on
  later optimizer steps, so a triggered refresh is deliberately delayed. If a
  scheduled anchor occurs in between, the stale result is discarded. Default:
  `False`; CPU execution keeps the synchronous path. Diagnostics report
  submitted, completed, skipped, and stale checks. A matched threshold-0.50
  CUDA stress screen regressed from `0.00024543` synchronously to `0.00027585`
  asynchronously despite similar refresh counts. The delayed path therefore
  remains a research option; removing host synchronization needs a device-side
  or predictive policy rather than a late replay of the same decision.
- `muon_warm_max_rejection_streak`: anchor at the next periodic check after this
  many alignment-gate rejections. It requires a non-negative
  `muon_warm_alignment_tolerance`; `0` disables it. The streak stays on-device
  between amortized checks.
- `muon_warm_check_every`: interval for adaptive tracking checks. Default: `1`.
  Diagnostics distinguish signal evaluations from checks.
- `muon_warm_signal_check_only`: opt-in rejected throughput experiment. It
  evaluates an angular/skew signal only on the warm step whose value the next
  check consumes and omits checks before a fixed scheduled anchor. This saves
  about 16% optimizer time at the target shape, but regressed three-seed mean
  validation loss by `0.00555`; default: `False`.
- `muon_warm_separate_skew_signal`: with check-only cadence, return raw skew and
  alignment terms from one consistent compiled direction path and reduce them
  only before useful checks. This avoids both an extra Gram product and the
  rejected alternating direction kernels. At equal step count it trades a small
  loss increase for higher throughput; at matched wall time the extra steps
  improve three-seed mean validation loss. It requires
  `muon_warm_signal_check_only=True`; default: `False`.
  `state["muon_warm_anchor_reason"]` records `initial`, `state_reset`, `warmup`,
  `schedule`, `max_age`, `rejection_streak`, `tracking_error`, `alignment`,
  `stretch`, `angular_rms`, `skew_ratio`, or `warm`.
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
  total, angular, and normal tangent RMS, normalized skew ratio, relative
  diagonal conditioning, raw inverse magnitude, and cap activation even when
  controllers are disabled.
  Default: `False`, which preserves the fastest path.
- `muon_warm_max_tangent_ratio`: tensor-wide scalar preconditioner that limits
  the current tangent RMS to this multiple of its historical EMA. `0` disables
  it. Unlike coordinatewise scaling, this preserves the transport direction and
  its tangency. A value around `2` is a reasonable experimental spike limiter.
- `muon_warm_tangent_ema_beta`: decay for that scalar tangent-RMS history.
  Default: `0.95`.
- `muon_warm_update_stats_every`: every N steps, accumulate update RMS,
  momentum cosine, and cached-basis orthogonality on device. On warm steps it
  also runs an opt-in shadow fresh solve and records direction cosine, relative
  error, warm/anchor magnitudes, and correlations between fresh-reference error
  and candidate refresh signals. The final trainer report separates these
  measurements by warm age and matrix shape, with age zero denoting an anchor.
  `0` disables it. This is a diagnostic run, not a timing benchmark, because the
  shadow solve performs full anchor work.
- `muon_warm_reference_lr_ratio`: scale applied shadow-reference statistics to
  a comparison learning rate. It does not change parameters.
- `muon_warm_normalize_output`: rescale the returned direction to the Frobenius
  norm expected from an orthogonal Muon update while leaving the cached basis
  unchanged. Default: `False`; the included OpenWebText screen removed magnitude
  oscillation but worsened validation loss, so this remains an ablation.
- `muon_warm_output_scale`: multiply warm-step updates by a fixed scalar after
  retraction and output normalization, while leaving anchor updates and the
  cached transported basis unchanged. Default: `1.0`. This is an attribution
  control for testing whether the retained single-retraction method benefits
  from its implicit magnitude filter. An RMS-matched scale of `0.2067`
  recovered about 63% of the seed-123 loss gap between full-size
  geometry-preserving transport and the retained update, but did not match the
  retained loss.
- `muon_warm_output_scale_start` and `muon_warm_output_scale_decay`: optionally
  decay the warm output scale exponentially from `start` at warm age one toward
  `muon_warm_output_scale`. The schedule uses the existing Python-side age and
  adds no device reduction or synchronization. The default `start=None` keeps
  the fixed-scale behavior above. The fitted schedule improved seed-123 loss
  from `6.58420` to `6.58107`, still well behind the retained `6.54938`, so it
  remains an attribution option.
- `muon_warm_legacy_retraction_output`: with `power_step`, cache the
  geometry-preserving direction but return the legacy capped one-retraction
  direction computed from the same full candidate. This exact output-filter
  path reuses the Gram already formed by the power-step probe. Cache retraction
  can be disabled with `muon_warm_retract_steps=0`; the emitted direction still
  receives its one local cubic filter. That no-cache profile improved all three
  1,000-step OpenWebText seeds and the 2,000-step seed-123 gate. Default:
  `False`.
- `muon_warm_angular_scale`: relative scale for rotation inside the cached row
  space. Default: `1.0`.
- `muon_warm_normal_scale`: relative scale for complementary row-space motion.
  Default: `1.0`. These two scales are applied before the shared step size and
  tangent trust cap, preserving a one-matmul combined correction.
- `muon_warm_normal_inv_cap`: scale-invariant bound on the diagonal inverse used
  by rectangular normal transport. The optimizer clamps
  `abs(D^-1) * mean(abs(D))` to this value before constructing the correction.
  `0` disables it. The bound adds no matrix multiply or optimizer-sized state;
  `4` is the retained experimental profile.
- `muon_warm_normal_inv_ema_ratio`: optional tensor-scalar controller for the
  rectangular normal inverse. It keeps an EMA of the inverse RMS after the
  absolute cap and limits the next step to this multiple of that history. It
  requires a positive `muon_warm_normal_inv_cap`; `0` disables it. Ratio `3`
  improved the three-seed 1,000-step mean but lost to fixed cap 4 at 2,000
  steps, so it remains an attribution setting rather than the retained profile.
- `muon_warm_normal_inv_ema_beta`: EMA decay for that controller. Default:
  `0.95`. The state is one control scalar per rectangular tensor plus small
  diagnostic scalars, and all decisions remain on device.

## Tuning Notes

- Start from `muon_lr=0.02`, `muon_momentum=0.95`, and `muon_ns_steps=5` if you are matching common Muon settings.
- Increase `muon_warm_full_ns_steps` when early training is unstable or when the cached direction needs more time to settle.
- Decrease `muon_warm_anchor_every` to refresh more often. This is more expensive but keeps the cached direction closer to a full Muon update.
- For the experimental local-filter profile, start with period-eight anchors,
  `power_step`, safety factor `1.05`, normal-inverse cap `4`, no cache
  retraction, and the exact local output filter. Choose a check every four
  steps with either raw angular threshold `2` for the lower-cost robust profile
  or normalized skew-ratio threshold `0.52` for the dimensionless quality and
  portability profile. Both remain experimental pending validation on the
  reported nanoGPT batch-480 regime; the normalized controller is more
  portable across scale and shape but currently refreshes about twice as often.
- Keep `muon_warm_lr=1.0` as the local full-correction step and use
  `muon_warm_max_tangent_rms` to shrink only unusually large moves. This is a
  more targeted experiment than globally lowering the warm learning rate.
- Use a small non-negative `muon_warm_alignment_tolerance` as a safety valve;
  pair it with periodic adaptive-anchor checks so repeated rejected steps cause
  a fresh solve instead of leaving a stale direction indefinitely.
- `muon_warm_retract_method="quadratic"` remains an attribution control while
  full anchors keep their two cubic polishes. At the target `768x768` and
  `768x2048` logical shapes it cut the complete skew warm path by 7--8%, but
  contracted emitted update RMS to about `0.91` of cubic. Multiplying warm
  outputs by `1.1` repaired the magnitude and passed the three-seed 200-step
  screen, then regressed both matched 1,000-step seeds by `0.00887` loss on
  average for only a 2.1% optimizer-time saving. Keep `higham_cubic` as the
  training default.
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
  device tensors. Instrumented runs accumulate the maximum proposed and applied
  tangent RMS and the minimum effective transport step without a host sync.
- `muon_warm_warm_retractions` counts cache repairs, while
  `muon_warm_output_retractions` counts local filters applied to emitted warm
  updates. Keeping these counters separate makes the no-cache profile explicit.
- The angular refresh controller reuses the correction already formed by the
  warm update. It adds one scalar reduction per active matrix and batches all
  decisions on a device into one periodic synchronization; it does not build an
  extra Gram matrix.
- Opt-in update/reference statistics remain as device tensors during training;
  `train_llm.py` transfers and aggregates them only for the final JSON report.
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
to this variant. `muon_warm_track_subspace=False` omits the complementary
correction. The generic `muon_warm_max_tangent_rms` and
`muon_warm_max_tangent_ratio` options are rejected by this variant because its
exact in-space rotation does not have the same scalar-step semantics. Its exact
rotation also requires `muon_warm_angular_scale=1`; `muon_warm_normal_scale` and
`muon_warm_retract_every` are supported.

Two residual-form inner polishing steps are the default. In the controlled
benchmark, a third did not improve tracking error, while one left the small
polar solve under-converged.

Run `python benchmarks/benchmark_transport_solvers.py` to compare the Jacobi and
Procrustes paths on a controlled dense-stretch and row-space drift problem.

## Deterministic replay

`benchmarks/replay_transport.py` replays slow drift, an abrupt shock, and a
rank-stress trajectory through a fresh matched Muon solve plus fixed and adaptive
Jacobi/Procrustes transports. It reports angular, normal, total polar, and
orthogonality errors alongside anchor reasons, work counts, update-device time,
and wall time.

```bash
python benchmarks/replay_transport.py --device cuda:0 --trajectory all \
  --rows 16 --columns 64 --steps 64 --anchor-every 8 \
  --output transport_replay.json
```

This is the preferred first check for controller, anchor, and retraction changes.
It deliberately uses the same anchor solve for fresh and transported variants so
the effect of transport is visible separately from a stronger orthogonalizer.
Use `--adaptive-check-every` to amortize adaptive diagnostics. Compare anchor
solvers with `--anchor-method muon --anchor-ns-steps 5` and
`--anchor-method polar_express --anchor-ns-steps 6 --anchor-retract-steps 0`.

`benchmarks/train_teacher_mlp.py` runs a deterministic short teacher-student
training comparison of fresh matched Muon, fixed-interval transport, and the
adaptive cadence-2 candidate. It records validation loss, steady-state optimizer
time, anchors by reason, warm retractions, and rejected steps.

## LLM training harness

`transformer.py` is a compact modern decoder: token embeddings feed the residual
stream directly, followed by pre-norm RMSNorm blocks, RoPE attention with optional
QK normalization, bias-free SwiGLU, a final RMSNorm, and a tied LM head. The
nonstandard input projection from the imported trainer is disabled by default;
`--input-projection` retains it as an explicit ablation. RMSNorm uses an explicit
`1e-5` epsilon under BF16. QK normalization is per head dimension and is applied
before RoPE. QKV and the two SwiGLU input projections remain fused by default for
the forward pass; `--no-fused-qkv` and `--no-fused-swiglu` expose physical-split
architecture controls.

For Muon, `--muon-split-qkv` and `--muon-split-swiglu` preserve those fused
forward GEMMs but orthogonalize the three Q/K/V and two value/gate gradient row
blocks independently. Every logical block gets its own momentum, transported
polar cache, and refresh decision. This matches the optimizer topology without
paying for separate model projections. Existing configurations keep whole-tensor
behavior unless the flags are set.

CUDA AdamW already uses PyTorch's fused implementation. `--compile` enables
`torch.compile`, and `--compile-fullgraph` requests the full-graph behavior used
by the larger target profile.

`train_llm.py` is self-contained and defaults to a structured synthetic dataset,
so the complete model/optimizer path can be checked offline:

```bash
/venv/main/bin/python train_llm.py \
  --optimizer transport_muon --steps 1000 --output transport_llm.json
```

For a packed Hugging Face text run:

```bash
/venv/main/bin/python train_llm.py --dataset hf \
  --dataset-name Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
  --tokenizer-name gpt2 --optimizer transport_muon \
  --output transport_wikitext.json
```

For OpenWebText, use streaming plus explicit token limits. The supplied 1,000-step
configs reserve a deterministic validation prefix, materialize 2.5 million train
tokens and 262,144 validation tokens, and share a roughly 11 MB int32 token cache:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_transport_1000.json \
  --output transport_openwebtext_1000.json
```

### Width-768 target screen

The first target-regime screen uses a 95.2M-parameter decoder at width 768,
depth 8, 12 heads, batch 32, and sequence length 1024. All runs use BF16,
full-graph compilation, QK normalization, fused QKV/SwiGLU forward projections,
and independent logical Q/K/V and SwiGLU Muon states. Each 50-step run consumes
1.638M training tokens, so it fits within the 2.5M-token int32 cache without
repeating a sequence.

The seed-123 rate sweep selected Muon LR `0.06`: losses improved monotonically
from `0.01` through `0.06`, then regressed at `0.08`. A matched three-seed check
gave:

| Profile | Mean validation loss | Mean step ms | Mean optimizer ms | State MiB | Mean adaptive anchors |
|---|---:|---:|---:|---:|---:|
| Fused AdamW, LR `4e-4` | 7.04498 | **134.67** | **1.76** | 726.58 | 0 |
| Angular threshold 2, Muon LR `0.06` | 6.60473 | 171.95 | 42.58 | **618.67** | 226.7 |
| Skew ratio 0.52, Muon LR `0.06` | **6.59310** | 173.21 | 43.87 | **618.67** | **104.0** |
| Smaller-side SOAP, LR `1.25e-4` | 6.89174 | 146.15 | 14.55 | 870.58 | 0 |

Skew ratio 0.52 slightly improves the mean over angular threshold 2 and cuts
adaptive anchors by 54%. It is nevertheless about 1.25 ms slower per step:
fewer anchors leave more warm transport and output-retraction steps. This makes
the warm path, especially its power probe and checked retraction, the next
performance target; reducing refresh count alone does not reduce wall time in
this regime.

These are learning-rate and integration screens, not long-run convergence
claims. The complete rate sweep, seed rows, timing, memory, and source artifact
names are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_50step_summary.json`.
The next gate will use a larger int32-only token cache so the frozen candidates
can run longer without recycling training sequences.

The cache has now been expanded to 40M training tokens plus the 262,144-token
validation prefix. It occupies 153.6 MiB and retains only int32 token IDs. A
matched 200-step, three-seed gate consumes 6.554M distinct training tokens per
run and gives:

| Profile | Validation loss, mean ± sd | Mean step ms | Mean optimizer ms | State MiB | Mean adaptive anchors |
|---|---:|---:|---:|---:|---:|
| Fused AdamW, LR `4e-4` | 6.09585 ± 0.03048 | **135.75** | **1.76** | 726.58 | 0 |
| Smaller-side SOAP, LR `1.25e-4` | 5.90796 ± 0.01387 | 146.81 | 13.54 | 870.58 | 0 |
| Angular threshold 2, Muon LR `0.06` | 5.67133 ± 0.00135 | 172.71 | 42.25 | **618.58** | 666.3 |
| Skew ratio 0.52, Muon LR `0.06` | **5.66059 ± 0.00539** | 174.44 | 43.73 | **618.58** | **356.0** |

Skew leads angular at every averaged validation checkpoint and improves final
mean loss by `0.01074`. It cuts adaptive anchors by 46.6%, but executes 310 more
warm output retractions per run and remains 1.0% slower. The same conclusion
therefore holds over the longer gate: optimize the warm transport kernel before
spending effort on still fewer refreshes. Relative to AdamW, skew improves mean
loss by `0.43526`, costs 28.5% more per step, and uses 108.0 MiB less optimizer
state.

The complete curves and per-seed evidence are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_200step_three_seed_summary.json`.

A shape-matched component benchmark at logical matrix sizes `768x768` and
`768x2048` then separated the warm geometry, local output filter, controller
signal, and full anchor. At drift `0.50`, the two-power-iteration warm output
took `0.505/0.539` ms, versus `0.834/0.857` ms for a full anchor. Reducing the
power estimate to one iteration lowered the warm output to `0.454/0.509` ms.
The resulting BF16 direction was identical to the two-iteration direction in
these prepared probes. The skew signal itself cost `0.624/0.668` ms, slightly
more than the angular signal's `0.595/0.633` ms. Component artifacts for drifts
`0.08`, `0.25`, and `0.50` are stored as
`../optimizer_replay_results/transport_warm_components_768*.json`.

The one-power-iteration change also passes the full three-seed 200-step gate:

| Skew 0.52 profile | Validation loss, mean ± sd | Step ms | Optimizer ms | Tokens/s | Adaptive anchors |
|---|---:|---:|---:|---:|---:|
| Two power iterations | 5.66059 ± 0.00539 | 174.44 | 43.73 | 187,846 | 356.0 |
| One power iteration | 5.66084 ± 0.01749 | **170.67** | **40.70** | **191,996** | **351.3** |

One iteration changes mean validation loss by only `+0.00025`, reduces optimizer
time by 6.9% and total step time by 2.2%, and improves throughput by 2.2%. Its
mean transport step scale is larger (`0.01370` versus `0.00920`), so the longer
gate must continue reporting the minimum scale, refresh causes, and stability.
It is the current Transport efficiency candidate; the two-iteration profile is
retained as the matched control.

The next training gate can run 1,000 steps without sequence reuse from the same
cache. Exact resumable checkpoint support comes first.

The optional two-retraction profile is a geometry/quality candidate:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_transport_retract2_1000.json \
  --output transport_openwebtext_retract2_1000.json
```

Age-resolved diagnostics showed that the one-retraction warm direction contracts
on the first warm step: its RMS was `0.441` of fresh at age one and about
`0.16--0.22` at later ages, while orthogonality error rose from `0.052` at the
anchor to `0.837` at age one. Two retractions improved the age-one RMS ratio to
`0.599`, cosine to `0.797`, and orthogonality error to `0.685`. With its matrix
rate retuned to `0.0075`, it matched fresh Muon's three-seed 1,000-step loss and
slightly improved on the existing transport profile. A focused kernel benchmark
found that the second retraction increased full-tangent warm-step cost by roughly
`26--34%`; end-to-end timing differences between training runs were noisy. It is
therefore kept as an explicit candidate rather than replacing the faster
one-retraction profile.

The cached-power profile replaces the loose row-sum bound while retaining one
retraction:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_transport_power_cap_1000.json \
  --output transport_openwebtext_power_cap_1000.json
```

Its `1.05` safety factor is intentionally an aggressive experiment. Across
three 1,000-step seeds it improved mean validation loss from `6.55822` to
`6.55452`, while optimizer time rose `16.9%` and total step time rose `7.1%`.
At seed 123 it raised warm/fresh RMS ratio from `0.243` to `0.301`, but mean
orthogonality error remained high (`0.818`). A rotating-spectrum stress test
found 3--4 retraction-limit violations per 64 steps at safety `1.05`; the
default `1.25` safety factor had none but removed most of the quality gain. The
profile is therefore an attribution experiment rather than a recommended
default.

The bounded-normal profile attacks the source of the large candidate before
the spectral guard:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_transport_normal_inv_cap4_1000.json \
  --output transport_openwebtext_normal_inv_cap4_1000.json
```

In the uncapped seed-123 diagnostic, normal correction RMS averaged `107.8`
and reached `9436`, while the dimensionless inverse factor averaged `47.0` and
reached the Tikhonov maximum near `500`. Cap `4` clipped only `3.8%` of row
inverses, reduced mean normal RMS to `8.73`, and reduced the maximum proposed
tangent RMS from `9423` to `594`. Across three 1,000-step seeds it improved
mean validation loss from `6.55822` to `6.55448`. Isolated 64-by-128 and
128-by-256 kernels showed no measurable cost beyond timing noise. The cap stays
opt-in: it also improved the 2,000-step seed-123 gate (`6.23355` to `6.23110`)
and both seeds in the batch-480 teacher-MLP stress check, but anchor periods 12
and 16 regressed to `6.56584` and `6.58126`. It controls denominator outliers
without making a longer fixed warm lifetime safe.

An EMA-relative scalar was tested on top of cap 4. Ratio 2 over-damped the
seed-123 run (`6.55078`). Ratio 3 improved all three 1,000-step seeds, moving
their mean from `6.55448` to `6.55376` with similar measured optimizer time.
The gain did not survive the longer gate: at 2,000 steps it reached `6.23242`
versus `6.23110` for fixed cap 4. The implementation is available in
`llm_openwebtext_transport_normal_inv_cap4_ema_ratio3_1000.json`, but the
supplied retained profile continues to use only the fixed cap.

The spectral-step profile avoids whole-candidate shrinkage:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_transport_power_step_1000.json \
  --output transport_openwebtext_power_step_1000.json
```

It repairs the intended geometry: in the seed-123 period-eight diagnostic,
warm/fresh RMS was `1.006`, cosine was `0.692`, and mean orthogonality error was
`0.030`. The final whole-input guard averaged `0.99996`, confirming that it
nearly always accepted the adjusted candidate unchanged. The cost is substantial
because it evaluates a probe Gram: a representative 64-by-128 warm kernel took
`0.626` ms versus `0.314` ms for the default. At the matched `0.0025` rate,
period eight reached loss `6.58633`; shortening the period to four improved this
to `6.56562`, still behind fresh Muon's `6.55023`. This profile is retained to
separate geometric correctness from training utility.

`benchmarks/benchmark_spectral_cap.py` compares the heuristic estimate with
exact singular values under slow rotations and an abrupt basis shock. The
ordinary transport microbenchmark accepts `--spectral-cap-mode`,
`--power-steps`, `--power-safety-factor`, and `--normal-inv-cap` for isolated
cost measurement.

The modern width-768 gate now extends to 1,000 steps and seeds 123, 456, and
789. With batch 32, sequence length 1024, BF16, and full-graph compilation, the
matched results are:

| Profile | Validation loss, mean ± sd | Step ms | Optimizer ms | Tokens/s | State MiB |
|---|---:|---:|---:|---:|---:|
| AdamW `4e-4` | 4.75432 ± 0.00774 | **137.92** | **1.76** | **237,607** | 726.58 |
| Smaller-side SOAP `1.25e-4` | 4.61910 ± 0.01213 | 148.34 | 13.52 | 220,899 | 870.58 |
| Transport skew 0.52, one power iteration, `0.06` | **4.23854 ± 0.00144** | 171.27 | 40.65 | 191,325 | **618.58** |

One-power Transport averages 2,314.7 adaptive anchors and 46,629.3 warm output
retractions. Its three final losses span only 0.00286. In the matched seed-123
control it improves validation loss by 0.00499 over two power iterations while
reducing optimizer time by 6.8% and total step time by 1.6%. This promotes one
power iteration as the next-scale efficiency candidate while retaining the
two-power profile as its numerical control. The complete curves and raw source
paths are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_1000step_three_seed_summary.json`.

Two follow-up attempts targeted the remaining warm-path cost. A conditional
second power iteration was rejected at the component stage: thresholds
`0.01`, `0.02`, and `0.05` requested no refinement on the prepared drift-0.50
matrices, yet the device-side branch raised complete skew-path time by 21--22%.
A quadratic output retraction was more promising in isolation, reducing that
path by 7--8%. Its approximately 9% contraction was compensated with a warm-only
output scale of `1.1`. The compensated profile matched cubic over three 200-step
seeds (`-0.00107` mean loss, `-3.7%` optimizer time), but lost both matched
1,000-step gates: mean validation loss rose from `4.23929` to `4.24816` while
optimizer time fell only 2.1% and total step time 0.9%. Cubic therefore remains
the retained output retraction. The component measurements, curves, source
paths, and decisions are in
`../optimizer_replay_results/transport_output_retraction_768_summary.json`.

The next experiment amortized the skew controller itself. In the retained
period-eight-anchor, period-four-check schedule, only the step-three signal can
request a controller-only anchor; step seven precedes the fixed step-eight
anchor. The opt-in fast path uses the ordinary local-output kernel on the other
six warm steps. At `768x768` and `768x2048`, this reduced the exact warm-cycle
component cost by 16.8% and 18.5%. Across three 1,000-step seeds it reduced mean
optimizer time by 15.9% and total step time by 3.9%, but validation loss rose
from `4.23854` to `4.24409` (`+0.00555`). It therefore remains disabled and is
recorded as a rejected speed/quality tradeoff. A follow-up should keep one
compiled direction path on every warm step and move only the diagnostic
reduction behind the cadence gate.

That follow-up is now implemented behind
`muon_warm_separate_skew_signal=True`. The direction graph returns its existing
raw skew and alignment intermediates on every warm step; only the useful
period-four check reduces them to an FP32 scalar. This keeps one direction
kernel and does not rebuild the Gram product. At `768x768` and `768x2048`, its
exact warm-cycle savings are 13.4% and 12.6%. Across three matched 1,000-step
seeds, mean optimizer time falls from `40.65` to `33.44` ms (17.7%), total step
time falls from `171.27` to `164.93` ms (3.7%), and throughput rises 3.8%.
Validation loss is worse on all three paired seeds and rises from `4.23854` to
`4.24322` (`+0.00468` mean). It therefore remains an opt-in throughput tradeoff
rather than the quality-neutral default. The complete aggregate is
`../optimizer_replay_results/transport_skew_signal_terms_cadence_768_summary.json`.

A matched-wall-clock follow-up gives the faster profile 1,038 steps versus
1,000 for the retained controller. Mean end-to-end time is `181.61` versus
`181.92` seconds (`-0.17%`), while validation improves on every seed and falls
from `4.23854` to `4.21781` (`-0.02073`). Raw-term cadence therefore advances
as the wall-clock throughput profile, while every-warm evaluation remains the
conservative fixed-token profile. The aggregate is
`../optimizer_replay_results/transport_skew_signal_terms_wallclock_768_summary.json`.

A full-horizon seed-123 threshold screen does not repair the equal-step gap.
Raw-term thresholds `0.50`, `0.52`, and `0.54` finish at `4.25345`, `4.24671`,
and `4.25587`, with 2,920, 2,385, and 1,854 adaptive anchors. Threshold `0.52`
is retained; neither neighboring setting merits replication. Results are in
`../optimizer_replay_results/transport_skew_signal_terms_threshold_screen_768_summary.json`.

Long runs support atomic rolling checkpoints with exact shuffled-data position,
model, optimizer, scheduler, and RNG state:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_modern_768x8_transport_skew052_power1_muonlr006_batch32_seq1024_1000.json \
  --checkpoint /workspace/optimizer_checkpoints/transport.pt \
  --checkpoint-every 100

/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_modern_768x8_transport_skew052_power1_muonlr006_batch32_seq1024_1000.json \
  --resume /workspace/optimizer_checkpoints/transport.pt
```

Keep the original target `--steps` and schedule when resuming. CPU controls are
bitwise exact across an interruption, including an epoch boundary. The original torch 2.12.0+cu130 compiled CUDA
controls reproduced every logged loss and controller event, with FP32 state
differences around `1e-9`. This is environment-specific evidence, not a bitwise
CUDA guarantee; revalidate numerical replay after software or hardware changes. Exact checkpointing rejects asynchronous Muon checks because a
pending CUDA event cannot be serialized faithfully.

`--hf-streaming` reads only the bounded sample. `--max-train-tokens` and
`--max-validation-tokens` cap RAM and token-cache usage, while `--token-cache`
stores token IDs as int32 and avoids repeated network reads and tokenization
across seeds. The packed dataset currently expands the token stream to int64 in CPU RAM;
the persistent cache remains int32. Eager loading of
`Skylion007/openwebtext` is rejected unless `--allow-large-hf-download` is passed,
because the complete download plus generated dataset can occupy about 64 GB.

The trainer supports JSON configuration defaults, gradient accumulation,
checkpoint output, cosine/constant schedules, mixed precision, optional model
compilation, validation, CUDA timing, peak memory, optimizer-state size, and
Transport Muon anchor diagnostics. Embeddings and the tied LM head use the Adam
branch; internal matrix projections use Transport Muon.

### Receiving-device audit (2026-09-30)

Independent probes confirm tall/wide orientation and complementary subspace
motion. A new CPU regression verifies exact local-optimizer resume with logical
QKV/SwiGLU state across anchors and a shuffled-epoch boundary. The matched
full-anchor baseline remains `muon_warm_anchor_every=1` with identical anchor
polishing and output normalization. Adaptive signal checks still synchronize
once per checked device batch; the power estimate is heuristic.

The completed-checkpoint recovery path can now regenerate missing result JSON
without training again. SOAP gate completion metadata and active-only basis eta statistics are also
covered by the shared audit. Shared trainer/model/test
files remain byte-identical between repositories. Final CPU suites: 97
TransportMuon tests and 66 TurboSOAP tests. No model weights are retained.

### Current LLM evaluation check (2026-10-01)

The shared 768x8 trainer passed same-weight evaluation checks at steps 100,
200 and 1,000 on torch 2.11.0+cu128: maximum compiled/eager BF16 validation CE
difference 2.9e-5, identical repeated compiled losses, and identical eager
first-batch grad/no-grad losses. Model hashes were unchanged by diagnostics.
This supports the tested current trajectory; unsaved historical weights and
other architectures/software were not validated. The run used SOAP; it tests
the shared model/evaluator, not every optimizer trajectory.
See [the foundation report](../optimizer_replay_results/soap_budget_20261001/README.md).
