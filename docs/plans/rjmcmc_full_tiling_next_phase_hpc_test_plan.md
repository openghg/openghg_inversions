# Full-tiling next-phase real-data HPC test plan

## Purpose and status

This plan validates the next fixed-\(K\), construction-history-free full-tiling
sampler on the frozen May 2014 PARIS Stage C input. It is the follow-up to the
[first native-data smoke](rjmcmc_full_tiling_native_hpc_test_plan.md), not a
replacement for that baseline record.

The candidate branch is `codex/rjmcmc-full-tiling-next-phase`. It combines four
changes that must be tested together:

1. an \(O(K)\), sibling-indexed and cache-backed merge catalogue;
2. an exact slice update in \(z=\log T\), with width 1, a finite 100-step
   outward budget (including the initial bracket), and a 1,000-draw shrink cap;
3. optional, output-only per-transition movement and cost diagnostics; and
4. strict, no-pickle durable checkpoint/restart.

The run is a correctness, restart, mobility, and performance test. It does
**not** establish convergence, irreducibility, adequate posterior exploration,
or a scientific emissions result.

## Frozen comparison contract

Use the same reviewed input and outer-label order as the first full-tiling
smoke:

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
```

The input has 1,382 observations, a \(183\times128\) inner grid, and six
ordered outer columns. The semantic SHA-256 of `nominal_weight` is
`bd3e014e042dc35f847c0aec44ffe88c843dbb82cc7da43bea4981aa835feb26`.
No weight flooring, input substitution, variable discovery, or outer-label
reordering is allowed.

The performance baseline is commit `6e123ae` (`Validate full-tiling real-data
smoke`) on this input:

| Fixed \(K\) | Likelihood power | Transitions/s | Peak RSS | Root refresh acceptance |
|---:|---:|---:|---:|---:|
| 50 | 1 | 280.08 | 1.54 GiB | 6% |
| 250 | 1 | 28.39 | 1.54 GiB | 3% |

The old independent-prior root refresh and the new slice update are different
kernels. The 6% and 3% figures are context for the reason for the change, not
acceptance targets for the slice kernel. A valid slice update is accepted by
construction; its useful diagnostics are displacement, bracket work,
shrinkage work, and conditional-density evaluations.

## Driver interface readiness gate

The existing driver remains
`examples/rjmcmc/full_tiling_native_smoke.py`, and all commands below retain
its current frozen-input and scientific arguments. Before launching HPC jobs,
its `--help` must additionally expose these exact next-phase controls:

```text
exactly one of --cycles N or --iterations N
--resume-checkpoint CHECKPOINT
--chain-id ID
--collect-movement-diagnostics
--root-slice-width 1
--root-slice-max-steps 100
--root-slice-max-shrink-steps 1000
```

These are the final flag names. Record the exact `--help` output in
`preflight/driver-help.txt`. Do not emulate an atomic-transition boundary by
rounding to `--cycles`: the restart test deliberately stops inside compound
cycles. Do not run this plan until the driver can:

- persist a `checkpoint.npz` without object arrays or pickle;
- resume from it at an arbitrary schedule phase;
- turn movement diagnostics on and off without changing the transition
  kernel or PCG64 stream; and
- preserve one `--chain-id` across all segments while writing each segment to
  a new output directory.

Retain `--cycles` for whole-cycle matrix jobs if that is the final interface;
100 cycles is exactly 1,400 transitions with the six-column PARIS schedule.

## Preflight on BP1

Run inside a Slurm allocation or site-standard batch wrapper, never on the
login node. Do not hard-code a partition. Request one CPU, 8 GiB, and 60
minutes initially, then adjust walltime from a completed \(K=50\) run. Pin
numerical libraries to that CPU:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Resolve a clean candidate revision and create a revision-specific,
non-overwriting run root:

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
git fetch origin
git switch codex/rjmcmc-full-tiling-next-phase
git pull --ff-only
test -z "$(git status --porcelain)"
export CODE_REVISION="$(git rev-parse HEAD)"
export DRIVER=examples/rjmcmc/full_tiling_native_smoke.py
export RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_next_phase/"$CODE_REVISION"
export BASELINE_CHECKOUT=/group/chem/acrg/brendan_for_codex/openghg_inversions-baseline-6e123ae
export BASELINE_RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_next_phase/6e123ae
mkdir -p "$RUN_ROOT"/{preflight,matrix,root-calibration,diagnostics-parity,restart,rejections,report}
mkdir -p "$BASELINE_RUN_ROOT"/matrix
```

Record software and machine provenance:

```bash
git rev-parse HEAD > "$RUN_ROOT/preflight/code-revision.txt"
git show -s --format=fuller HEAD > "$RUN_ROOT/preflight/code-commit.txt"
git diff --exit-code
git diff --cached --exit-code
pixi run -e dev --frozen python "$DRIVER" --help \
  > "$RUN_ROOT/preflight/driver-help.txt"
pixi run -e dev --frozen python -VV > "$RUN_ROOT/preflight/python.txt" 2>&1
pixi list > "$RUN_ROOT/preflight/pixi-list.txt"
uname -a > "$RUN_ROOT/preflight/uname.txt"
lscpu > "$RUN_ROOT/preflight/lscpu.txt"
sha256sum "$FROZEN_INPUT" > "$RUN_ROOT/preflight/input.sha256"
test "$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')" = "$FROZEN_INPUT_SHA"
```

Confirm the required controls are present:

```bash
for FLAG in cycles iterations resume-checkpoint chain-id \
  collect-movement-diagnostics root-slice-width \
  root-slice-max-steps root-slice-max-shrink-steps; do
  rg -- "--$FLAG" "$RUN_ROOT/preflight/driver-help.txt"
done
```

Run focused repository checks before using the real input:

```bash
pixi run -e dev --frozen pytest -q tests/experimental/rjmcmc \
  > "$RUN_ROOT/preflight/pytest.txt" 2>&1
pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_io.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/ruff.txt" 2>&1
pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_io.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/pyright.txt" 2>&1
```

Finally, perform both PARIS-profile dry runs. The output directory arguments
are intentionally unused by `--dry-run`:

```bash
for K in 50 250; do
  if [ "$K" -eq 50 ]; then KAPPA=100; else KAPPA=500; fi
  pixi run -e dev --frozen python "$DRIVER" \
    --input "$FROZEN_INPUT" \
    --output-directory "$RUN_ROOT/preflight/dry-k${K}-unused" \
    --k "$K" --cycles 1 --seed 812 \
    --concentration "$KAPPA" --root-variance 0.25 \
    --likelihood-power 1 \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision "$CODE_REVISION" \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" \
    --require-paris-profile \
    --root-slice-width 1 --root-slice-max-steps 100 \
    --root-slice-max-shrink-steps 1000 \
    --dry-run > "$RUN_ROOT/preflight/dry-k${K}.json"
done
```

Both dry reports must record a finite initial target, cycle length 14, the
versioned root-slice schedule, root prior shape/rate 4/4, exact input SHA,
closure within the existing \(10^{-12}\) relative/absolute tolerance, and the
three root-slice limits 1/100/1000.

## Real-scale checkpoint gate

Commit `c1a6944a` failed after otherwise successful candidate sampling because
checkpoint reconstruction used a fixed \(5\times10^{-13}\) ppb absolute cache
tolerance. Incremental and canonical reconstruction sums differed by at most
\(4.77\times10^{-12}\) ppb after 1,400 transitions, without any coordinate
disagreement. The revised audit:

- continues to require exact topology, leaf masses, fixed coefficients,
  hashes, schedule phase, kernel settings, and PCG64 state;
- persists the original incremental caches for bitwise continuation;
- independently rebuilds canonical caches from the exact coordinates;
- uses cache tolerance
  \(\max(5\times10^{-13}, 512\,\mathrm{ULP}(S))\), where \(S\) is the largest
  absolute observation or cache value, with a floor of one; and
- separately reconstructs prediction, residual, and all target components
  from the exact persisted dynamic and fixed prediction caches. Dependent
  caches and target components must then agree exactly.

The focused checkpoint tests include PARIS-scale six-fixed-position sweeps at
14, 1,400, and 14,000 transitions. Locally the last case reproduces more than
\(10^{-9}\) benign raw-Gaussian drift from canonical cache reconstruction.
Deterministic 256/1,024-ULP cases test the cache-audit boundary; deliberate
\(10^{-8}\) ppb stale-cache corruption and one-ULP target corruption must still
fail closed.

Before submitting the four-cell matrix, reproduce the old transition boundary
on the frozen input:

```bash
for N in 13 14; do
  OUT="$RUN_ROOT/preflight/checkpoint-gate-k50-beta0-t${N}"
  pixi run -e dev --frozen python "$DRIVER" \
    --input "$FROZEN_INPUT" --output-directory "$OUT" \
    --k 50 --iterations "$N" --seed 31050 \
    --chain-id checkpoint-gate-k50-beta0 \
    --concentration 100 --root-variance 0.25 --likelihood-power 0 \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision "$CODE_REVISION" \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" --require-paris-profile \
    --root-slice-width 1 --root-slice-max-steps 100 \
    --root-slice-max-shrink-steps 1000
  test -f "$OUT/checkpoint.npz"
  test -f "$OUT/complete.json"
  pixi run -e dev --frozen python -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); c=json.loads((p/"complete.json").read_text()); assert "checkpoint.npz" in c["sha256"]' \
    "$OUT"
done
```

Both boundaries are hard gates. Transition 14 must now publish a complete
bundle. Keep these new-commit artifacts separate from the immutable
`c1a6944a` failure record.

## Hard-gate debugging protocol

When a later hard gate fails, stop downstream expensive stages but continue
bounded read-only diagnosis where possible:

1. Preserve the failed revision-specific run directory and all incomplete
   artifacts; never repair or overwrite them in place.
2. Find the shortest failing transition boundary by running new output
   directories, including the immediately preceding successful boundary.
3. Record the global transition, schedule phase, move, validity/acceptance,
   exact coordinate equality, per-cache maximum absolute difference, error in
   units of the reported audit tolerance, and target-component differences.
4. Diagnostic wrappers or monkeypatches may live under the run's `jobs/` or
   `diagnostic/` directory, but must not edit the frozen candidate source or
   publish diagnostic output as a successful result.
5. Launcher/environment defects may be corrected and rerun with the correction
   recorded. A sampler, target, checkpoint, or scientific-input defect remains
   a hard stop pending a new reviewed commit.
6. After a fix, use a new commit-addressed run root. Rerun preflight and the
   smallest reproduction first, then the matrix and awkward restart. Submit
   calibration and diagnostics stages only after those gates pass.

Failure messages from the cache audit now report the worst cache, maximum
absolute discrepancy, permitted scale/ULP-aware tolerance, and
observation-space scale; retain the complete message in the run report.

## Reference-oracle semantic parity

There is deliberately no production runtime flag for the slow catalogue
oracle. Semantic parity is established before HPC by the exhaustive/reference
tests, while HPC supplies same-commit real-data evidence. The optimized
sibling index and caches must change lookup cost only; the slice kernel is
separately checked against its exact log-density target.

At the clean candidate revision, retain successful output from these focused
tests:

```bash
pixi run -e dev --frozen pytest -q \
  tests/experimental/rjmcmc/test_full_tiling.py \
  tests/experimental/rjmcmc/test_full_tiling_posterior.py \
  tests/experimental/rjmcmc/test_full_tiling_compound_sampling.py \
  tests/experimental/rjmcmc/test_full_tiling_movement_diagnostics.py \
  tests/experimental/rjmcmc/test_full_tiling_io.py \
  tests/experimental/rjmcmc/test_full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/reference-oracle-tests.txt" 2>&1
```

The retained test evidence must cover:

- sibling-indexed merge choices equal the exhaustive/reference catalogue,
  including no-merge and awkward-boundary tilings;
- cached and full-rebuild design columns, predictions, targets, and proposal
  accounting agree at floating-point oracle tolerances;
- seeded optimized transitions agree with the reference construction wherever
  both consume the declared same draw sequence;
- log-root slice density and rescaling agree with direct target
  recomputation; and
- output-only diagnostics leave the ordinary trace and PCG64 checkpoint
  exactly unchanged.

Missing coverage or any oracle disagreement is a hard failure. The old commit
and new commit are not expected to produce identical traces because the root
kernel changed.

## \(K50/K250\), \(\beta0/\beta1\) performance matrix

Create a detached, clean baseline checkout once and keep its output separate:

```bash
if [ ! -d "$BASELINE_CHECKOUT" ]; then
  git worktree add --detach "$BASELINE_CHECKOUT" 6e123ae
fi
test "$(git -C "$BASELINE_CHECKOUT" rev-parse HEAD)" = \
  "$(git rev-parse 6e123ae^{commit})"
test -z "$(git -C "$BASELINE_CHECKOUT" status --porcelain)"
```

Run both commit `6e123ae` and the clean candidate revision for each matrix
cell. The old driver uses its original independent-prior root refresh; the new
driver uses the exact log-root slice. This is a whole-sampler performance
comparison, not seeded semantic parity.

| \(K\) | \(\beta\) (`--likelihood-power`) | concentration | seed |
|---:|---:|---:|---:|
| 50 | 0 | 100 | 31050 |
| 50 | 1 | 100 | 31150 |
| 250 | 0 | 500 | 31250 |
| 250 | 1 | 500 | 31350 |

Candidate command template:

```bash
K=50
BETA=0
KAPPA=100
SEED=31050
OUT="$RUN_ROOT/matrix/candidate-k${K}-beta${BETA}"
/usr/bin/time -v pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUT" \
  --k "$K" --cycles 100 --seed "$SEED" --chain-id "candidate-k${K}-beta${BETA}" \
  --concentration "$KAPPA" --root-variance 0.25 \
  --likelihood-power "$BETA" \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile \
  --root-slice-width 1 --root-slice-max-steps 100 \
  --root-slice-max-shrink-steps 1000 \
  > "$OUT.stdout.json" 2> "$OUT.resource.txt"
```

Baseline command template, run from its checkout:

```bash
BASELINE_DRIVER="$BASELINE_CHECKOUT/examples/rjmcmc/full_tiling_native_smoke.py"
OUT="$BASELINE_RUN_ROOT/matrix/baseline-k${K}-beta${BETA}"
(
  cd "$BASELINE_CHECKOUT"
  /usr/bin/time -v pixi run -e dev --frozen python "$BASELINE_DRIVER" \
    --input "$FROZEN_INPUT" --output-directory "$OUT" \
    --k "$K" --cycles 100 --seed "$SEED" \
    --concentration "$KAPPA" --root-variance 0.25 \
    --likelihood-power "$BETA" \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision 6e123ae \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" \
    --require-paris-profile
) > "$OUT.stdout.json" 2> "$OUT.resource.txt"
```

Repeat both templates with the other three table rows. If filesystem cache
effects appear material, make one unreported warm-up per \(K\), then run three
new non-overwriting replicates per commit/cell and report the median. Never
overwrite or cherry-pick the fastest replicate.

Collect:

- setup, sampling, and total wall time;
- atomic transitions/s;
- `/usr/bin/time -v` maximum RSS;
- attempts, valid candidates, and accepted moves by kernel;
- invalid-reason counts and distinct retained tilings;
- source merge count and design-cache miss totals from a separate
  diagnostics-on run, not from the timed diagnostics-off result;
- root-slice left/right outward steps, shrink draws, and density evaluations;
  and
- CPU model, Slurm allocation/job ID, numerical-library thread settings, and
  output filesystem.

For the full-likelihood candidate runs, compare with commit `6e123ae`:

| Measure | Pass | Warning | Failure |
|---|---:|---:|---:|
| \(K=50\) throughput | \(\ge252.07\)/s (90% of baseline) | 224.06--252.06/s | \(<224.06\)/s |
| \(K=250\) throughput | \(\ge25.55\)/s (90% of baseline) | 22.71--25.54/s | \(<22.71\)/s |
| Peak RSS | \(\le1.85\) GiB (120% of baseline) | 1.86--2.31 GiB | \(>2.31\) GiB |

Also report speedup relative to the paired `6e123ae` run and state that the
root kernel differs. Catalogue work that grows approximately as \(K^2\), repeated
whole-grid design reconstruction after cache warm-up, or unbounded cache
growth is a hard failure even when wall-clock thresholds happen to pass.

The \(\beta=0\) rows isolate geometry/catalogue and prior-kernel costs. The
\(\beta=1\) rows show the production-shaped likelihood cost. Do not compare
absolute log targets between \(K=50\) and \(K=250\).

## Prior-only Gamma(4,4) root calibration

The driver setting `--root-variance 0.25` with prior mean 1 implies
\(T\sim\operatorname{Gamma}(4,4)\) in shape/rate notation. Run the ordinary
compound schedule with \(\beta=0\). Because the declared prior factorizes, the
root update has the Gamma(4,4) target even though structural, share, and outer
updates occur between root slots. Use 50,000 complete cycles at \(K=50\), seed
41050, width 1, outward budget 100, and shrink cap 1,000. This yields 50,000
post-start root-slice updates:

```bash
OUT="$RUN_ROOT/root-calibration/gamma-4-4-k50"
/usr/bin/time -v pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUT" \
  --k 50 --cycles 50000 --seed 41050 --chain-id gamma-4-4-k50 \
  --concentration 100 --root-variance 0.25 --likelihood-power 0 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile \
  --collect-movement-diagnostics \
  --root-slice-width 1 --root-slice-max-steps 100 \
  --root-slice-max-shrink-steps 1000 \
  > "$OUT.stdout.json" 2> "$OUT.resource.txt"
```

Pass criteria are:

- all 50,000 root updates are valid and accepted;
- every retained total is finite and strictly positive;
- no update exceeds the configured outward or shrink cap;
- empirical mean is in [0.98, 1.02] and variance in [0.23, 0.27];
- the empirical 5%, 50%, and 95% quantiles are each within 0.03 of quantiles
  calculated independently with `scipy.stats.gamma(a=4, scale=0.25).ppf`;
- rerunning the same seed reproduces the ordinary trace and final PCG64 state
  exactly (timings excluded).

Exclude the deterministic retained state at transition zero from these
calibration summaries. Non-root state is expected to move under the compound
schedule and is excluded from this marginal calibration. Independently retain
the focused unit-test evidence that a single root slice rescales only the
total-dependent state and leaves tiling, shares, and fixed coefficients
unchanged.

A mean/variance/quantile miss is a hard calibration failure, not evidence to
tune width from this one chain. Width changes require a separate declared
experiment.

## Posterior root diagnostics

For both \(\beta=1\) candidate matrix runs, report:

- root-total minimum, median, maximum, and finite/non-positive counts;
- absolute and absolute-log displacement quantiles (0, 25, 50, 75, 95, 99,
  and 100%);
- left/right outward-step, shrink-draw, and density-evaluation quantiles and
  maxima;
- lag-1, lag-5, and lag-10 autocorrelation as descriptive diagnostics;
- zero-displacement count; and
- whether either finite guard was approached.

Hard failures are non-finite/non-positive totals, a cap exception, any
reported count beyond a cap, or disagreement with direct target recomputation
in the focused reference-oracle tests.
Warnings are more than 5% zero root displacement, a maximum shrink count above
500, a maximum outward extension count above 80, or lag-1 autocorrelation
above 0.99. These warning levels trigger inspection; they are not convergence
thresholds and do not support an inference claim.

## Diagnostics off/on parity and overhead

Run the candidate \(K=50,\beta=1\) and \(K=250,\beta=1\) cases again with the
same respective seeds and `--chain-id`, once with diagnostics off (omit
`--collect-movement-diagnostics`) and once on (supply it). Require exact
equality of every ordinary trace field and the final checkpoint scientific
state, transition/schedule coordinates, and PCG64 state. The diagnostics-on
run must add one aligned row per atomic transition without altering the
manifest payload that defines the scientific kernel.

Validate movement fields by move:

- invalid attempts have zero movement;
- structural rows record sibling merge counts and relocation catalogue size
  \(2(K-1)\) where applicable;
- cache misses are non-negative and fall to zero for already-cached rectangle
  columns;
- root rows alone have root displacement and slice-work counters;
- allocation rows alone have share \(L_1\) displacement;
- fixed rows alone have a valid fixed-position index and coefficient
  displacement; and
- all movement values are finite and non-negative.

`design_cache_misses` is deliberately segment-local: the lazy performance
cache is not checkpointed Markov state. Do not require it to match between an
uninterrupted process and a process-boundary restart.

Calculate overhead from three paired, non-overwriting off/on replicates after
one warm-up, using median sampling time:

| Diagnostic overhead | Result |
|---:|---|
| \(\le10\%\) | pass |
| \(>10\%\) and \(\le25\%\) | warning |
| \(>25\%\) | failure |

Also report RSS delta. An RSS increase above 10% is a warning and above 25% is
a failure. Timing fields are observational and must never be included in
scientific replay hashes.

## Durable awkward-boundary restart

At \(K=250,\beta=1\), seed 51250, compare a direct 1,400-transition run with
four durable segments of 5, 11, 137, and 1,247 transitions. The cumulative
coordinates are 5, 16, 153, and 1,400, exercising schedule phases 5, 2, 13,
and 0 for the 14-transition compound cycle.

Direct run:

```bash
OUT="$RUN_ROOT/restart/direct-1400"
/usr/bin/time -v pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUT" \
  --k 250 --iterations 1400 --seed 51250 --chain-id restart-k250-beta1 \
  --concentration 500 --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" --require-paris-profile \
  --root-slice-width 1 --root-slice-max-steps 100 \
  --root-slice-max-shrink-steps 1000
```

Segmented run:

```bash
PREVIOUS=
INDEX=0
for LENGTH in 5 11 137 1247; do
  OUT="$RUN_ROOT/restart/segment-$(printf '%03d' "$INDEX")"
  RESUME_ARGS=()
  if [ -n "$PREVIOUS" ]; then
    RESUME_ARGS=(--resume-checkpoint "$PREVIOUS/checkpoint.npz")
  fi
  pixi run -e dev --frozen python "$DRIVER" \
    --input "$FROZEN_INPUT" --output-directory "$OUT" \
    --k 250 --iterations "$LENGTH" --seed 51250 \
    --chain-id restart-k250-beta1 \
    --concentration 500 --root-variance 0.25 --likelihood-power 1 \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision "$CODE_REVISION" \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" --require-paris-profile \
    --root-slice-width 1 --root-slice-max-steps 100 \
    --root-slice-max-shrink-steps 1000 \
    "${RESUME_ARGS[@]}"
  PREVIOUS="$OUT"
  INDEX=$((INDEX + 1))
done
```

The current interface requires `--seed 51250` on every segment because seed is
part of the immutable manifest identity. Resumed segments nevertheless obtain
the exact active PCG64 state from `--resume-checkpoint`; the driver must not
reinitialize the stream from that repeated CLI seed.

The checkpoint is the authoritative recovery object; `complete.json`
certifies a complete reporting bundle. Each resumed segment must record the
SHA-256 of its parent checkpoint and its independently reconstructed
segment-start state fingerprint so the four-segment lineage is auditable.

Concatenate segment ordinary traces by their global transition/state
coordinates, removing only duplicated boundary states explicitly documented
by the schema. They must equal the direct ordinary trace exactly. The last
segmented and direct checkpoints must have identical scientific state, cache
semantics, global transition 1,400, schedule phase 0, kernel settings, and
PCG64 state. Each segment must retain its own immutable manifest, checkpoint,
trace, summary, and completion hashes.

Inspect the checkpoint without pickle:

```bash
pixi run -e dev --frozen python -c \
  'import numpy as np, pathlib, sys; p=pathlib.Path(sys.argv[1]); z=np.load(p, allow_pickle=False); assert all(a.dtype.kind != "O" for a in z.values()); print(sorted(z.files))' \
  "$RUN_ROOT/restart/segment-003/checkpoint.npz"
```

Any requirement for `allow_pickle=True`, any object array, missing schema or
problem/manifest identity, restart difference, or partially accepted
checkpoint is a hard failure.

## Fail-closed rejection tests

Perform rejection checks on copies under `rejections/`; never alter a
successful run or the frozen input.

1. **Corruption:** copy a checkpoint, flip one non-header byte, and attempt a
   resume. It must reject before sampling and create no `complete.json`.
2. **Input mismatch:** resume a valid checkpoint while supplying an incorrect
   `--expected-input-sha256`, then repeat with a byte-different copied NetCDF
   and its actual SHA. Both must reject before state construction/sampling.
3. **Manifest mismatch:** resume while changing one item at a time: \(K\),
   likelihood power, concentration, root variance, outer-label order,
   root-slice limits, chain/schedule identity, and frozen input ID.
   Every scientific/kernel mismatch must reject explicitly.
4. **Checkpoint/schema mismatch:** remove or alter a required metadata member,
   change its schema version, and add an object-dtype member in separate
   copies. Loading with `allow_pickle=False` and application validation must
   reject each copy.

Each expected failure gets `stdout.txt`, `stderr.txt`, and `exit-status.txt`.
A zero exit status, sampling output, a completion marker, an unhandled
traceback without a specific validation message, or acceptance based only on
matching filenames is a hard failure.

## Artifacts, hashes, and final report

Every successful sampling directory must contain, at minimum:

- immutable `manifest.json`;
- strict no-pickle `checkpoint.npz`;
- ordinary `trace.nc`;
- `summary.json`;
- optional movement-diagnostic variables in `trace.nc`, identified by the
  trace attribute `movement_diagnostics_collected`;
- `complete.json`, written last, with hashes for every preceding durable
  artifact; and
- captured stdout plus `/usr/bin/time -v` stderr outside the immutable run
  directory or included in a higher-level run index.

At the run-root level write:

- `preflight/` records listed above;
- a CSV/JSON matrix with revision, input digest, \(K\), \(\beta\), seed,
  commit, diagnostics setting, transitions, wall time, transitions/s,
  RSS, and artifact paths;
- reference-oracle test evidence plus exact diagnostics off/on and
  direct/restarted parity reports;
- root calibration and posterior root diagnostic tables;
- rejection-test results;
- `report/sha256sums.txt` covering all retained artifacts; and
- `report/RESULTS.md` stating what passed, warned, or failed against every
  threshold in this plan.

Generate and immediately verify the run-root digest inventory:

```bash
cd "$RUN_ROOT"
fd -t f -0 --exclude sha256sums.txt . | sort -z | xargs -0 sha256sum \
  > report/sha256sums.txt
sha256sum --check report/sha256sums.txt
```

`report/sha256sums.txt` is deliberately excluded because a file cannot
consistently hash itself. Do not hash only filenames or JSON metadata: hash
artifact bytes.

## Overall decision

Promotion to a longer diagnostic phase requires all semantic parity, strict
restart, no-pickle, input/manifest rejection, closure, finite-value, and
completion-hash checks to pass. Performance and diagnostic-overhead failures
also block promotion. Warnings require an explanation and a targeted follow-up
but do not by themselves invalidate exactness.

Even a complete pass remains conditional on the communication component
reachable from the deterministic fixed-\(K\) start. It makes no claim about:

- convergence, effective posterior sample size, or calibrated uncertainty;
- scientific emissions inference or agreement with another inversion;
- fixed-\(K\) structural irreducibility or mixing across all tilings;
- inference over \(K\), split/merge RJ correctness, or a cross-\(K\)
  structural normalizer;
- correlated observation error, parallel tempering, Numba, or production
  deployment; or
- tuning the root-slice width or scientific priors from this diagnostic run.
