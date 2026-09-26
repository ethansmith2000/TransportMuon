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
  `muon_warm_check_every` steps and currently require the default
  `"gershgorin"` spectral-cap mode. This is an experimental diagnostic control;
  the included OpenWebText threshold screen did not beat fixed period eight.
- `muon_warm_max_rejection_streak`: anchor at the next periodic check after this
  many alignment-gate rejections. It requires a non-negative
  `muon_warm_alignment_tolerance`; `0` disables it. The streak stays on-device
  between amortized checks.
- `muon_warm_check_every`: interval for adaptive tracking checks. Default: `1`.
  `state["muon_warm_anchor_reason"]` records `initial`, `state_reset`, `warmup`,
  `schedule`, `max_age`, `rejection_streak`, `tracking_error`, `alignment`,
  `stretch`, `angular_rms`, or `warm`.
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
  total, angular, and normal tangent RMS, relative diagonal conditioning, raw
  inverse magnitude, and cap activation even when controllers are disabled.
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
  device tensors. Instrumented runs accumulate the maximum proposed and applied
  tangent RMS and the minimum effective transport step without a host sync.
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
`1e-5` epsilon under BF16.

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

`--hf-streaming` reads only the bounded sample. `--max-train-tokens` and
`--max-validation-tokens` cap RAM and token-cache usage, while `--token-cache`
stores token IDs as int32 and avoids repeated network reads and tokenization
across seeds. Batches are promoted to int64 only when loaded for embedding
lookup. Eager loading of
`Skylion007/openwebtext` is rejected unless `--allow-large-hf-download` is passed,
because the complete download plus generated dataset can occupy about 64 GB.

The trainer supports JSON configuration defaults, gradient accumulation,
checkpoint output, cosine/constant schedules, mixed precision, optional model
compilation, validation, CUDA timing, peak memory, optimizer-state size, and
Transport Muon anchor diagnostics. Embeddings and the tied LM head use the Adam
branch; internal matrix projections use Transport Muon.
