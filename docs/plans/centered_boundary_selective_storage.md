# Selective storage after accurate centered-boundary preparation

Research record, updated 28 October 2026. Follow-up to [PR 979](https://github.com/openghg/openghg_inversions/pull/979)
and the [precision research brief now owned by PR 986](https://github.com/openghg/openghg_inversions/pull/986#issuecomment-6470193042).
Controlled numerical, posterior-sensitivity and bounded runtime/memory experiments
are recorded below. Scientific acceptance and production integration remain pending;
the float64 production guard is unchanged. The prototype began at 979 revision
`2fcd3cf5`; the research commits were restacked onto the reviewed 979 revision
`0c948b5f` for the 28 October experiment. PR 980 remains the separate codec change.
This record does not establish supported precision modes or change a specification.

## Concrete first slice

`scripts/probe_centered_storage.py` uses 979's full-operand projection and existing
fixed-OU cache. It rounds only the active complete boundary design `B = A + cM`
once, then uses those identical values in a float32 shared tensor and float64
numerical cache. Physical coefficients and `b - 1`, fixed intercepts, shifts,
covariance and caches remain float64. Direct `pytensor.shared` registration avoids
`pm.Data` converting the chosen storage dtype to global `floatX`; this is an
isolated graph experiment, not a new production registration API.

The script checks the compiled mean/state gradient, uncached likelihood and cached
quadratic against an independent dense Gaussian with cross-block covariance. It
retains nonunit fixed values and a zero complete-design column with nonzero anomaly.
It covers CO2/O2 reference scales, nonpartition mass, all-active/mixed/all-fixed
policies and tiny through broad deterministic deviations. Errors against the
original float64 target are reported separately from internal consistency asserts.
No production function is patched, no existing assertion is weakened, and the
float64 construction guard is unchanged.

From the repository root, in the normal development environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTENSOR_FLAGS=floatX=float64,mode=CVM uv run python -m scripts.probe_centered_storage --dtype float64 > /tmp/centered-storage64.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTENSOR_FLAGS=floatX=float64,mode=CVM uv run python -m scripts.probe_centered_storage --dtype float32 > /tmp/centered-storage32.json
```

If existing PyTensor flags are needed, retain them while selecting float64/CVM;
on the cluster also use a writable node-local `base_compiledir`. Run each dtype
in a fresh process. JSON records versions, graph operations, selected retained
array sizes, total process peak and all deterministic points. This small probe
is not a full memory inventory or representative memory benchmark. In particular,
casts may allocate float64 execution temporaries and the numerical design remains64.
It does not exercise physical priors/transforms, NUTS, amplitude-step transitions,
linked channel alignment, period expansion, or saved-artifact codecs.

Initial checks passed for both storage dtypes: 48 points each, using Python 3.12,
NumPy 2.2.6 and PyTensor 3.3.2/CVM with one BLAS thread. Existing
`tests/models/test_cached_sigma.py` and `tests/test_centered_model_guards.py`
also passed (38 tests), as did focused Ruff. This is bounded draft validation,
not final CI-equivalent or scientific acceptance.

For the new probe's active O2 case at sigma=0.2 ppm, deviations are
`b_i - 1 = i * grid_value` for the three active columns:

| Grid value | Maximum mean error (ppm) | D | Delta total log likelihood | Maximum baseline residual (ppm) |
| --- | ---: | ---: | ---: | ---: |
| 1e-8 | 1.31e-10 | 1.88e-18 | -4.39e-10 | 0.326 |
| 1e-4 | 1.37e-6 | 1.90e-10 | 8.37e-4 | 45.1 |
| 1e-2 | 1.37e-4 | 1.90e-6 | 8.42 | 4489 |

The broadest point has baseline log likelihood about -7.84e8. Its sizeable
likelihood change, despite small D, comes from the residual/error cross term.
It illustrates why tiny matrix errors alone cannot establish a posterior result;
these arbitrary points are not representative posterior draws. At b=1 and for
all-fixed cases the design-storage error is exactly zero. The float64 control
has zero approximation error throughout. No memory-saving claim is made from
this tiny fixture's process peak.

## Interpretation of prior research

Earlier isolated tests at 979 `2012f37a` found a coherent active-B32 graph/cache
target with tiny differences from the original target: up to `2.14e-7 ppm` in a
linked synthetic mean (on CO2 rows), and `6.11e-8` in total log likelihood.
Broader synthetic prior draws produced `1.56e-4 ppm` product differences.
These are bounded fixture results, neither scientific rejection nor acceptance.
They are historical measurements, not outputs of the new probe above.

The `0.0184 ppm` reference-shift diagnostic did not propagate an error through
inference. A consistent common shift cancels in `(y-q) - (mu-q)`. Neither ambient
O2 nor daily concentration variability alone establishes posterior impact.

Compared options in the initial research assessment (superseded for performance
recommendations by the 28 October measurements below):

| Option | Initial interpretation |
| --- | --- |
| Active B32, accurate fixed terms and small states | Preferred candidate for further validation; preserve one rounded target across all conditionals. |
| Raw A/M32 | Adds error to fixed contributions too; less attractive than quantizing only active B. |
| Direct deviations or `expm1` for log states | Can preserve tiny deviations; prior support/Jacobians and optimizer behavior require care. No general reparameterization proposed. |
| Blocked float64 RHS construction | Useful control/possible independent optimization; historical synthetic peak saving about 3%. |
| Covariance32 | Separate investigation; no requirement or adequacy conclusion established. |

## Validation before production support

1. **Corrected real operands.** Start with MHD July 2021, 129 matched rows per
   tracer. Regenerate directional A/M from pinned raw curtains/footprints, with
   validated units and joint curtain normalization. The old boundary pilot
   predates the correction: its O2 baseline differs by up to 14.78 ppm; dividing
   its rounded absolute result by mass still leaves about 0.02466 ppm difference.
   These preparation differences must not contaminate the storage comparison.
   Use the corrected scalar OPE231 boundary as the b=1 oracle, not as a source
   from which directional operands can be recovered. Record hashes/labels/times.
   Mean-only deterministic screens need no new likelihood-error assumption.
2. **Freeze the scientific target.** Agree physical BC priors/support/widths,
   independent errors per tracer, aggregation representation/rank, OU policy,
   observation noise and scientific output budgets. Construct controlled UOB
   observations from matching native enhancement plus one corrected boundary.
   Start with one period and four directional coefficients per tracer, inherited
   retained flux geometry/prior and exact-zero activity. Do not silently adopt
   broad fixture BC priors or legacy 0.2 ppm errors as campaign policy.
3. **Matching float64 baseline.** Inspected existing posterior traces are BC-free;
   they cannot be reweighted into this sampled-BC target. Adding independent BC
   prior draws does not produce the required joint posterior. Obtain a converged
   matching baseline before claiming posterior-wide accuracy.
4. **Paired likelihood evaluation.** At each identical state use the same full
   `C(theta)`, including that draw's amplitudes. With `e = mu32 - mu64` and
   `r = y - mu64`, evaluate `D = e.T solve(C,e)` and
   `delta_logL = r.T solve(C,e) - D/2`. Verify normalized likelihood differences.
   Chunk mean evaluation if needed, but retain the global covariance operation;
   independent row/chunk/channel likelihoods can discard cross-covariance.
5. **Posterior summaries.** Start with 50 reproducible draws per chain and explicit
   tail points for screening. Use all suitable posterior draws for final weights
   proportional to `exp(delta_logL)` when priors/support are unchanged. Extreme-
   selected points are not an unbiased posterior sample. Check overlap and MCMC
   autocorrelation separately. Compare concentration/BC/OU summaries and inner-
   PARIS GPP, TER and net biosphere with matched masks, state mappings and units.
   A 279-state functional map cannot be reused for a 304-state target.
   Use chain-aware paired uncertainty bounds against agreed budgets; noisy or
   statistically insignificant differences alone are not a pass. Possible
   discussion values (not adopted): mean shifts <0.01 posterior SD, interval
   endpoints <0.02 SD, SD change <1%, plus per-channel absolute ppm limits.
6. **Candidate sampling when informative.** Run separate candidate chains if
   overlap is inadequate, shifts need confirmation or sampler dynamics are the
   question. Validate priors/Jacobians, amplitude/state ordering, rejected updates,
   chain isolation, products and save/reopen without double restoration. Use
   980 when complete linked prepared codecs are needed. Preserve strict float64
   regressions alongside separate approximate-storage acceptance.
7. **Actual memory.** Compare float64, selective B32, blocked64 and their combination
   using the same compiled recipe/workload in fresh processes. Measure preparation,
   construction, compilation, refresh, sampling and save. Include one chain and
   the actual four-chain mode; use whole-job/PSS accounting rather than summed
   RSS that double-counts shared pages. Agree useful savings before interpreting
   results. Extend to four July sites (the actual retained selection is 618 rows/channel,
   as measured below), then full scale and a
   period transition as needed. Larger runs use Slurm and wakeup monitoring.

Covariance precision remains separate: changing C requires logdet/solve/gradient,
conditioning and full delta-log-density checks; the fixed-C identity is insufficient.

## Controlled MHD posterior experiment, 27 October 2026

Four float64 baseline chains completed on SLURM array `19761123`, with 1,000
tuning iterations and 1,000 saved draws each, using revision `e51f0a7e` plus an
isolated research runner. All 4,000 draws were evaluated against the once-rounded
active-B32 target. This is a controlled technical benchmark with real transport
operands and synthetic observations, not an accepted campaign configuration.

The corrected July MHD inputs have 129 rows per tracer, 304 retained flux states,
four NESW boundary coefficients per tracer and one boundary period. The original
correlated flux prior and full joint aggregation covariance were retained,
including cross-channel covariance. Boundary coefficients have independent
Normal(1, 1/reference) priors (approximately 1 ppm per pure curtain), with
references 420 ppm CO2 and 209300 ppm O2. Independent error SD is 0.2 ppm per
tracer; OU time scale is five hours and amplitude priors are HalfNormal(0.75 ppm).
Controlled observations use unit flux/boundary coefficients, OU amplitude
0.5 ppm and one joint Gaussian noise draw (seed 20261027). Sampling seeds are
20261027 through 20261030. These choices do not establish scientific policy.

The corrected O2 boundary closes against the independent scalar preparation to
`2.91e-11 ppm`. The stored legacy CO2 native enhancement differs from the accurate
model contraction by up to `5.60e-6 ppm`; controlled observations use the matching
model contraction and preserve the legacy enhancement separately. This difference
is not attributed to selective boundary storage.

Both research targets retain physical states, fixed contributions, shifts,
covariance and numerical caches in float64. The float64 research recipe agrees
with the unpatched baseline to `7.11e-15 ppm` in mean and `1.14e-13` in log
likelihood in preflight checks. Save/reopen and chain provenance checks passed.
Post-run likelihood reconstruction at 200 draws agrees to `2.28e-13`.

| Measurement | Result |
| --- | ---: |
| Maximum R-hat, including whitened latent states | 1.00565 |
| Minimum bulk / tail MCMC effective sample size | 3464 / 2050 |
| Divergences / tree-depth limit hits, both update steps | 0 / 0 |
| Maximum per-draw concentration mean error | 5.55e-8 ppm |
| Maximum full-covariance D | 5.45e-13 |
| Delta total log likelihood range | -1.264e-6 to 8.172e-7 |
| Raw importance effective sample size | approximately 4000 of 4000 |
| Maximum posterior concentration mean change, CO2 / O2 | 5.96e-8 / 5.37e-8 ppm |
| Maximum daily concentration mean change, CO2 / O2 | 5.69e-8 / 4.98e-8 ppm |

Reweighting uses the full global covariance at each draw's amplitudes, with
unchanged priors. Candidate concentration summaries include both reweighting and
the direct change in model output. Stable paired mean estimates avoid subtracting
absolute O2 means. A circular moving-block bootstrap within each chain used 500
replicates at block lengths 25, 50 and 100. At length 50, the largest absolute
endpoint of the componentwise 95% mean-change intervals is `8.19e-8 ppm` for CO2
and `8.20e-8 ppm` for O2. These are descriptive maxima of pointwise intervals,
not simultaneous coverage or general model-error bounds. Posterior SD and 95%
interval endpoint changes are also recorded, as point estimates only. Importance
ESS assesses overlap separately from MCMC ESS; no separate candidate chains were
needed to resolve overlap in this experiment.

This evidence supports continued selective-storage development; it supplies no
numerical reason to reject the candidate in this setup. Scientific acceptance
budgets and a verified 304-state inner-PARIS GPP/TER/net-biosphere functional map
remain missing. Broader priors, multiple periods, candidate sampler dynamics and
covariance32 have not been assessed here. Each baseline task took 3:09--3:22 with
467--488 MiB maximum RSS. The two shared boundary designs save only 8,256 bytes
when stored in float32 in this case; there is no measured process-memory saving.

Run scripts, frozen inputs, checksums, traces and analysis remain on BP1 under
`build/pr986-inversions/` in worktree
`/group/chem/acrg/brendan_for_codex/.codex-worktrees/d4f4/openghg_inversions`.
They are local research artifacts, not files distributed by this PR. The evidence
entry point is `results/analysis/report.md`, alongside `analysis-summary.json`,
`paired_mean_changes.csv` and the executable `analyse_precision.py`.
Input fingerprint: `0289e5bd091f076ce3a48abf8b6109418a3c5b72513e526e8586eb8db973e9e7`;
configuration SHA256: `f491a0a87b864e98fd05796e8c3434dfb324f8cdb96662049d6496abcff3baba`.

## Bounded compiled memory and runtime experiment, 28 October 2026

All sixteen measurement tasks in Slurm array `19775806` completed with exit code
zero after the separate eight-case correctness preflight `19775696` passed.
The experiment demonstrates an exact, small reduction in retained boundary-design
storage. It does **not** demonstrate a useful whole-process memory or runtime
benefit for B32, blocked construction, or their combination on these workloads.
This is one run per case, with no repeated-trial uncertainty estimate.

### Workloads, target and measurement method

The workloads use actual July 2021 UOB BASE transport rows: MHD has 129 rows per
tracer; MHD/BIR/BSD/TAC has 618 per tracer (129/154/180/155). The joint targets
therefore have 258 or 1,236 observations. Both retain 304 flux states and four
NESW boundary coefficients per tracer in one boundary period: the cache has
312 affine coefficients. No rows were replicated. The complete selected joint
aggregation covariance, including cross-site and cross-tracer entries, is retained.

The observation-space simulation target is the accurate retained-model forward
concentration at unit flux and boundary coefficients, with one corrected boundary
baseline and one Gaussian noise draw (seed 20261027). The covariance is the
retained aggregation covariance plus independent error variance `(0.2 ppm)^2`
and independent OU blocks by **tracer and site**, with five-hour time scales and
0.5 ppm generating amplitudes. Inference uses the same fixed aggregation/error
inputs, inferred OU amplitudes with HalfNormal(0.75 ppm) priors, and the flux and
boundary priors described in the 27 October experiment. This is a conditional
observation-space technical target, not accepted campaign data, flux-truth scoring
or simulation-based calibration.

The four variants share the same collapsed research boundary recipe: `64` keeps
active complete B in float64; `B32` rounds active B once and stores the prediction
graph's shared design as float32 while widening those rounded values for numerical
work; `blocked64` keeps float64 and transforms the residual plus blocks of 32
design columns into a preallocated float64 RHS; `combined` uses both changes.
The measurement runner temporarily replaces callable boundary/likelihood hooks
and, for blocked variants, the numerical target constructor **inside its research
process only**. It does not change the production API or precision guard. The
float64 research recipe was checked against the unmodified production builder;
all variants passed mean, gradient and direct/cache likelihood consistency checks.

Each case runs in a fresh process with its own node-local PyTensor cache. Every
case requests four CPUs and 8 GiB; actual PyMC execution uses one chain/core or
four chains/cores with `mp_ctx="spawn"`. BLAS/OpenMP/NumExpr thread counts are one,
with two Dask workers. Versions are Python 3.13.7, NumPy 2.2.6, PyMC 6.3.2 and
PyTensor 3.3.2; global `floatX=float64` and optimizer `o4` are unchanged. Each chain
uses 250 tuning iterations and 250 retained draws. These short runs benchmark
execution and artifact writing; they do not establish posterior convergence or
scientific acceptance.

Memory is sampled whole-process-tree **proportional set size (PSS)**, which
apportions shared pages, rather than summed RSS. The monitor waits a nominal
100 ms between samples, then scans processes and flushes a JSONL record; its
actual interval is longer and short-lived allocations can be missed. Telemetry
streams to disk with bounded in-memory summaries. All cases use the same monitor
policy; its overhead is included. Sampling-phase memory also includes temporary
compiler children during worker startup. Independent cgroup accounting records
current and cumulative peak charged memory in each task's own memory cgroup.
Those kernel peaks are cumulative, not attributable to individual phases, and
are a different metric from PSS. No dedicated-node isolation or repeated trials
were established, so scheduling, page sharing and allocator effects can vary.

### Process-memory peaks

Every row is one measured workload/variant/chain-count case. Values are nominally
sampled process-tree peak PSS in **MiB (2^20 bytes)** during each named phase.
The refresh phase contains five cache refreshes, followed by restoration of the
initial cache; save includes saving, reopening and checking the posterior artifact.

| Workload / variant / chains | Prepare | Construct | Compile | Refresh | Sampling | Save/reopen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MHD / 64 / 1 | 285.1 | 341.6 | 483.4 | 420.3 | 609.5 | 610.4 |
| MHD / 64 / 4 | 282.3 | 342.3 | 477.6 | 421.3 | 2298.6 | 650.0 |
| MHD / B32 / 1 | 282.0 | 343.0 | 475.2 | 414.4 | 612.4 | 613.3 |
| MHD / B32 / 4 | 287.0 | 345.8 | 477.3 | 417.4 | 2326.3 | 664.9 |
| MHD / blocked64 / 1 | 281.5 | 342.4 | 492.5 | 421.2 | 610.4 | 611.4 |
| MHD / blocked64 / 4 | 286.7 | 342.4 | 481.9 | 421.1 | 2315.4 | 664.5 |
| MHD / combined / 1 | 282.9 | 345.5 | 487.9 | 414.9 | 625.3 | 629.9 |
| MHD / combined / 4 | 281.9 | 343.5 | 488.8 | 414.6 | 2326.5 | 661.9 |
| 4 sites / 64 / 1 | 319.3 | 442.4 | 542.0 | 522.8 | 1073.4 | 1035.0 |
| 4 sites / 64 / 4 | 318.8 | 458.3 | 541.3 | 522.5 | 2994.1 | 1234.3 |
| 4 sites / B32 / 1 | 362.4 | 492.7 | 579.0 | 553.8 | 1275.9 | 1205.8 |
| 4 sites / B32 / 4 | 344.7 | 491.8 | 582.3 | 558.3 | 3022.5 | 1203.7 |
| 4 sites / blocked64 / 1 | 342.3 | 495.9 | 587.0 | 559.1 | 1136.0 | 1097.6 |
| 4 sites / blocked64 / 4 | 316.6 | 443.3 | 531.7 | 507.4 | 2940.5 | 1184.7 |
| 4 sites / combined / 1 | 327.1 | 441.1 | 516.0 | 501.3 | 1202.3 | 1132.2 |
| 4 sites / combined / 4 | 311.7 | 439.5 | 517.6 | 506.1 | 2936.4 | 1234.8 |

The next table records whole-run evidence separately: maximum sampled PSS across
all phases, the largest observed kernel cgroup cumulative peak, and Slurm's raw
`.batch` MaxRSS field converted from KiB to MiB. Slurm MaxRSS is corroborating
scheduler accounting; it is not substituted for PSS or treated as proof of summed
chain memory. All task and `.batch` records report `COMPLETED`, `0:0`.

| Array task | Workload / variant / chains | Sampled PSS MiB | Cgroup peak MiB | Slurm MaxRSS MiB | Slurm elapsed |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | MHD / 64 / 1 | 610.5 | 646.6 | 649.6 | 00:01:38 |
| 1 | MHD / 64 / 4 | 2298.6 | 2284.8 | 2284.8 | 00:02:34 |
| 2 | MHD / B32 / 1 | 613.3 | 648.3 | 649.2 | 00:01:36 |
| 3 | MHD / B32 / 4 | 2326.3 | 2314.0 | 2314.0 | 00:02:25 |
| 4 | MHD / blocked64 / 1 | 611.4 | 651.4 | 652.2 | 00:01:38 |
| 5 | MHD / blocked64 / 4 | 2315.4 | 2298.6 | 2298.6 | 00:02:30 |
| 6 | MHD / combined / 1 | 629.9 | 700.4 | 701.4 | 00:01:30 |
| 7 | MHD / combined / 4 | 2326.5 | 2281.2 | 2281.2 | 00:03:09 |
| 8 | 4 sites / 64 / 1 | 1073.4 | 1133.5 | 1133.5 | 00:14:57 |
| 9 | 4 sites / 64 / 4 | 2994.1 | 3016.8 | 3016.8 | 00:27:56 |
| 10 | 4 sites / B32 / 1 | 1275.9 | 1305.9 | 1305.9 | 00:14:26 |
| 11 | 4 sites / B32 / 4 | 3022.5 | 3135.6 | 3135.6 | 00:26:24 |
| 12 | 4 sites / blocked64 / 1 | 1136.0 | 1184.7 | 1184.7 | 00:15:42 |
| 13 | 4 sites / blocked64 / 4 | 2940.5 | 2969.1 | 2969.1 | 00:26:56 |
| 14 | 4 sites / combined / 1 | 1202.3 | 1248.4 | 1248.4 | 00:14:21 |
| 15 | 4 sites / combined / 4 | 2936.4 | 2980.5 | 2980.5 | 00:27:35 |

There are sixteen distinct recorded cgroup paths, each containing the corresponding
Slurm job ID. The machine-readable summary preserves these paths, scheduler rows
and telemetry hashes. All eight four-chain cases contain four posterior chains and
four distinct simultaneous spawn command lines/pipe handles in the telemetry;
all eight one-chain cases have no spawn workers during sampling. Process counts
also include the parent, resource tracker and temporary compilers. One four-site
combined sample contains a fifth spawn-looking PID with an identical command line
to a persistent worker; a transient fork before exec is the plausible explanation,
not evidence of a fifth chain. The raw PID count and worker identities are preserved.

### Phase wall times

Values below are elapsed **seconds** measured inside the research process. Refresh
is the total for **five** refreshes, not one. Sampling is the full sampling helper:
it includes worker startup/compilation, NUTS execution and trace/likelihood work,
not a pure steady-state kernel measurement. Slurm elapsed time above also includes
interpreter imports, environment setup and output outside these timed calls.

| Workload / variant / chains | Prepare | Construct | Compile | Five refreshes | Sampling | Save/reopen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MHD / 64 / 1 | 1.46 | 5.50 | 9.57 | 0.04 | 45.34 | 1.07 |
| MHD / 64 / 4 | 1.46 | 5.49 | 9.56 | 0.05 | 101.43 | 1.04 |
| MHD / B32 / 1 | 1.64 | 5.30 | 8.56 | 0.04 | 43.95 | 1.18 |
| MHD / B32 / 4 | 1.63 | 5.30 | 8.57 | 0.04 | 94.04 | 0.92 |
| MHD / blocked64 / 1 | 1.71 | 5.25 | 9.68 | 0.04 | 44.56 | 1.04 |
| MHD / blocked64 / 4 | 1.71 | 5.25 | 9.67 | 0.04 | 97.94 | 0.98 |
| MHD / combined / 1 | 1.45 | 5.41 | 8.37 | 0.04 | 39.29 | 0.91 |
| MHD / combined / 4 | 1.46 | 5.54 | 10.70 | 0.05 | 134.19 | 1.21 |
| 4 sites / 64 / 1 | 2.26 | 5.10 | 9.16 | 0.71 | 839.41 | 1.48 |
| 4 sites / 64 / 4 | 2.26 | 5.10 | 9.16 | 0.72 | 1622.93 | 1.04 |
| 4 sites / B32 / 1 | 2.11 | 4.76 | 8.54 | 0.70 | 810.63 | 1.30 |
| 4 sites / B32 / 4 | 2.08 | 4.90 | 8.56 | 0.67 | 1531.88 | 1.49 |
| 4 sites / blocked64 / 1 | 1.74 | 5.26 | 9.50 | 0.74 | 887.80 | 1.04 |
| 4 sites / blocked64 / 4 | 2.02 | 5.45 | 9.29 | 0.71 | 1562.74 | 1.14 |
| 4 sites / combined / 1 | 2.02 | 5.45 | 8.83 | 0.63 | 803.95 | 1.48 |
| 4 sites / combined / 4 | 2.02 | 5.45 | 8.90 | 0.64 | 1601.83 | 1.12 |

### Retained arrays and the supplemental inventory

The original measurements used revision `3dfc2b96`. Their array walker missed
borrowed prepared inputs stored in slotted dataclasses; this affects the reported
array subtotal, **not** the whole-process PSS or measured times. A separate
post-hoc eight-case audit, Slurm `19777205` at revision `310ef7c7`, traverses
`dataclasses.fields` and keeps references alive during traversal to prevent
transient object-ID reuse. It rebuilds and compiles each same frozen recipe,
then refreshes once, without sampling. These are parent-process retained-array
inventories, not replacement timings or new four-worker memory measurements.
All eight audits completed with exit code zero and matching input/configuration
fingerprints, measured software versions and original preflight hashes.

The audit includes observations, fixed contributions, native sensitivities,
independent errors, full joint covariance, retained-prior moments/factors and
both tracers' raw anomaly A, mass M and scalar reference arrays. All of these
scientific payloads remain float64 in every variant. In particular, the four raw
A/M arrays retain 16,512 bytes total for MHD and 79,104 bytes for four sites. They
are common retained inputs, not savings attributable to B32. Coordinates and
integer index arrays are also counted with their actual dtypes. The numerical
design contains the widened boundary columns in float64; the B32 audit finds no
separate joint-B-shaped float64 array duplicating the shared graph designs.

The table shows selected actual float64 storage retained by **all four** variants;
shapes are MHD / four-site. CSR mode transforms additionally retain integer index
arrays, which the machine-readable audit inventories identify separately.

| Array | Shape, MHD / four-site | Bytes, MHD / four-site |
| --- | --- | ---: |
| Prepared joint covariance | 258×258 / 1236×1236 | 532,512 / 12,221,568 |
| Numerical covariance factor, each of original and transformed factors | 258×258 / 1236×1236 | 532,512 / 12,221,568 each |
| Target design | 258×312 / 1236×312 | 643,968 / 3,085,056 |
| Transformed RHS | 258×313 / 1236×313 | 646,032 / 3,094,944 |
| OU site-block Cholesky arrays, total | 2 blocks / 8 blocks | 266,256 / 1,548,512 |
| CSR mode-transform floating values | 33,282 / 193,564 | 266,256 / 1,548,512 |
| Cache precision, each initial and shared matrix | 312×312 / 312×312 | 778,752 each |

The shared active boundary matrices have joint-row shapes `[258,4]` or `[1236,4]`
**per tracer**. Their combined storage changes from 16,512 to 8,256 bytes for MHD,
and 79,104 to 39,552 bytes for four sites. This is the exact B32 saving. The next
table gives the corrected audit's unique NumPy-backing-array subtotals in **bytes**;
`64 = blocked64` and `B32 = combined` at every phase. Compiled and refreshed totals
are identical within each audit. The reductions are less than 0.1% of the compiled
parent subtotal, before Python/native overhead and spawned workers.

| Workload | Constructed 64 | Constructed B32 | Compiled 64 | Compiled B32 | Exact reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHD | 8,824,756 | 8,816,500 | 9,638,532 | 9,630,276 | 8,256 |
| 4 sites | 57,284,556 | 57,245,004 | 58,200,716 | 58,161,164 | 39,552 |

The inventory deduplicates ultimate NumPy `.base` objects and follows shared graph
inputs, compiled storage and sampler state, with a twelve-object-level traversal
limit. Distinct wrappers around an external buffer may still be counted twice;
it is not exhaustive allocator/native-memory accounting. Full audit artifacts
retain array paths, shapes, dtypes and backing owners. The original incomplete
subtotals are preserved as historical evidence, not used for the corrected savings.
The supplemental audit also removes a 2,432-byte coordinate-traversal discrepancy
in some original subtotals; that discrepancy was not a blocked-construction saving.

### Interpretation and reproducibility

B32's shared-array saving is real but small relative to these processes. Whole-tree
PSS does not consistently fall: MHD one-chain B32 has a 613.3 MiB save peak versus
610.4 MiB for 64; four-site one-chain sampling peaks are 1,275.9 versus 1,073.4 MiB.
These observed increases are not attributed causally to precision. Likewise,
blocked construction leaves retained arrays unchanged; it removes the full
stacked-RHS temporary but still allocates a block output. The constructor interval
has other allocations, and nominal 100 ms sampling can miss this short-lived change.
The measured phase peaks do not establish a useful transient-memory reduction.

Runtime order also varies. Combined MHD sampling takes 39.29 versus 45.34 seconds
for one chain, but 134.19 versus 101.43 seconds for four chains. Four-site sampling
times span roughly 804–888 seconds for one chain and 1,532–1,623 seconds for four.
NUTS trajectories/step counts vary across the changed targets, and these are not
matched kernel-operation counts. With one trial per case, these observations do
not establish a general speedup, slowdown or parallel-scaling claim. Larger-scale
or repeated performance work should start from an agreed useful-saving threshold.
The current evidence supports retaining the float64 production restriction;
scientific precision budgets, approved flux-functional comparisons and production
sampler/product/handoff integration remain separate unfinished work.

The durable numeric record is [the machine-readable benchmark summary](centered_boundary_memory_20261028.json).
Raw inputs, preflight results, original measurement manifests/JSONL streams,
posteriors, scheduler evidence and supplemental audits are local BP1 artifacts at
`build/pr979-review-20261028/` in the originating worktree
`/group/chem/acrg/brendan_for_codex/.codex-worktrees/dbdf/openghg_inversions`.
The summary preserves input-manifest and result/telemetry/audit/scheduler hashes;
private data and full traces are not shipped by this PR. Reproduce the summary from the research worktree root with
[the standard-library analysis script](../../scripts/research/centered_boundary_memory/analyse.py):

```bash
python scripts/research/centered_boundary_memory/analyse.py \
  --run ../pr979-review-20261028 \
  --sacct ../pr979-review-20261028/sacct-19775806.psv \
  --audit-sacct ../pr979-review-20261028/sacct-19777205.psv \
  --out /tmp/centered_boundary_memory_20261028.json
```

The script verifies frozen input payloads, configurations, preflight/results/audit
fingerprints, versioned source blobs, worker evidence, distinct cgroups and
completed scheduler records before writing the summary. Exact configuration and
input hashes for each workload, and measured/audit source hashes, are in that
record. No new inference is performed during analysis.

## Remaining draft work

- [x] Isolated once-rounded B32 prototype with compiled/dense/cache checks.
- [x] Corrected real-input mean screens and frozen provenance.
- [x] Controlled matching float64 posterior and paired concentration/BC/OU comparison.
- [ ] Agreed scientific target, priors, errors and acceptance budgets.
- [ ] Campaign-approved posterior and paired scientific flux-functional comparison.
- [x] Bounded MHD/four-site compiled and actual multiple-chain memory comparison, including blocked64.
- [ ] Any justified full-scale/period-transition performance follow-up, with agreed useful-saving criteria.
- [ ] Local production integration, full sampler/product/handoff coverage and CI.

Verification Games owns the campaign configuration and independent scientific
acceptance; OpenGHG Inversions owns any eventual reusable implementation.
This draft is not ready to merge or to enable lower-precision production inference.
