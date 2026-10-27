# Selective storage after accurate centered-boundary preparation

Research draft, 27 October 2026. Follow-up to [PR 979](https://github.com/openghg/openghg_inversions/pull/979)
and its [research brief](https://github.com/openghg/openghg_inversions/pull/979#issuecomment-6444397846).
This record proposes validation; it does not establish supported precision modes
or change a specification. The branch starts at 979 revision `2fcd3cf5`.
PR 980 remains the separate codec change; this draft does not reparent it or
change the existing stack's review order. Retarget/rebase when ready for integration.

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

Compared options:

| Option | Current interpretation |
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
   results. Extend to four July sites (622 rows/channel), then full scale and a
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

## Remaining draft work

- [x] Isolated once-rounded B32 prototype with compiled/dense/cache checks.
- [x] Corrected real-input mean screens and frozen provenance.
- [x] Controlled matching float64 posterior and paired concentration/BC/OU comparison.
- [ ] Agreed scientific target, priors, errors and acceptance budgets.
- [ ] Campaign-approved posterior and paired scientific flux-functional comparison.
- [ ] Representative compiled/multiple-chain memory comparison, including blocked64.
- [ ] Local production integration, full sampler/product/handoff coverage and CI.

Verification Games owns the campaign configuration and independent scientific
acceptance; OpenGHG Inversions owns any eventual reusable implementation.
This draft is not ready to merge or to enable lower-precision production inference.
