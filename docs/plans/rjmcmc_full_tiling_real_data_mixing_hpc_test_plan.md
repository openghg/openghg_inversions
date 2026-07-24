# Full-tiling PARIS mixing and start-sensitivity HPC plan

## Decision and purpose

The numerical-correctness phase is complete. Commit `0c4be63` passed the
endpoint, exact-restart, checkpoint, corruption, diagnostics-parity, and
Gamma-calibration gates in
[the next-phase plan](rjmcmc_full_tiling_next_phase_hpc_test_plan.md).
The diagnostics-overhead results, +16.7% at \(K=50\) and +11.4% at \(K=250\),
were warnings rather than scientific or durability failures.

No further short numerical gate is required before a longer real-data test.
The next uncertainty is mixing. This plan therefore tests whether the local
fixed-\(K\) full-tiling kernel forgets substantially different starting
tilings on the frozen May 2014 PARIS data.

This is still a diagnostic experiment. It must not be reported as a converged
emissions inversion unless the convergence gates below pass.

## Model being tested

Keep the successful `0c4be63` scientific contract:

- 1,382 May 2014 PARIS methane observations;
- \(183\times128=23{,}424\) inner cells and six ordered InTEM outer columns;
- independent Gaussian observation errors from the frozen input;
- fixed \(K\), tested separately at 50 and 250;
- Gamma(4, 4) prior for the globally additive inner total \(T\);
- additive-alpha allocation with concentration \(\kappa=2K\);
- arithmetic-mean/SD \(1/1\) lognormal priors for the six outer coefficients;
- likelihood power \(\beta=1\);
- two structural attempts, one exact log-\(T\) slice update, five unordered
  pair-allocation refreshes, and six fixed-coefficient updates per 14-slot
  cycle.

This is not yet the full Lunt model: \(K\) is fixed, there is no AR(1)/OU
model-data mismatch covariance in this full-tiling track, and there are no
Lunt/Ganesan hyperpriors.

The structural target is uniform over unique canonical tilings at a given
\(K\), conditional on the communication component reached by the implemented
edge-flip and resolution-relocation moves. Production-grid connectivity is not
proved. The structural normalizer across different values of \(K\) is not in
the target, so:

- do not compare absolute log targets between \(K=50\) and \(K=250\);
- do not treat this run as inference on \(K\);
- do not pool the two fixed-\(K\) models;
- predictive or fixed-projection stability may be compared across \(K\), but
  only as a sensitivity analysis.

## New dispersed-start control

The driver now supports:

```text
--initialization {largest-nominal,random-recursive}
--initialization-seed SEED
```

`random-recursive` starts at the whole-domain rectangle and repeatedly draws
uniformly from the current canonical split catalogue until it reaches the
requested fixed \(K\). It uses a fresh, dedicated PCG64 stream, separate from
the sampler PCG64 state. The initializer is path-biased and is **not** the structural
prior or part of the MCMC target.

Every initializer assigns prior-mean leaf masses proportional to nominal
emissions and sets the six fixed coefficients to one. Consequently all starts
represent the same initial native scaling field (one everywhere), the same
root total, and the same initial forward prediction. They differ only in
tiling geometry. This makes separation between chains a direct test of the
structural kernel rather than an artefact of different initial emissions.

The manifest schema is
`openghg_inversions.full_tiling_native_smoke_manifest.v3`. It records the
initialization strategy, initialization seed, and initial-topology SHA-256.
Changing any of them when resuming a durable chain must fail closed.
Version 3 is an intentional checkpoint-identity boundary: a version-2
checkpoint cannot be resumed, even for the deterministic initializer.

## Frozen input and run root

Use the exact reviewed Stage C input:

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
```

Resolve a clean pushed revision on a compute node:

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
module load git/2.45.1-pqk5
git fetch origin
git switch codex/rjmcmc-full-tiling-next-phase
git pull --ff-only
test -z "$(git status --porcelain)"
export CODE_REVISION="$(git rev-parse HEAD)"
export DRIVER=examples/rjmcmc/full_tiling_native_smoke.py
export RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_mixing/"$CODE_REVISION"
mkdir -p "$RUN_ROOT"/{preflight,stage-r1,stage-r2,analysis,report}
```

Pin all numerical libraries to the one allocated CPU:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Never write intermediate or failed results to `PARIS_inversions`.

## Stage R0: preflight

Run the experimental suite only, plus focused static checks:

```bash
pixi run -e dev --frozen pytest -q tests/experimental/rjmcmc \
  > "$RUN_ROOT/preflight/pytest.txt" 2>&1
pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  tests/experimental/rjmcmc/test_full_tiling_posterior.py \
  tests/experimental/rjmcmc/test_full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/ruff.txt" 2>&1
pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/pyright.txt" 2>&1
```

Record provenance and check the new interface:

```bash
git rev-parse HEAD > "$RUN_ROOT/preflight/code-revision.txt"
git show -s --format=fuller HEAD > "$RUN_ROOT/preflight/code-commit.txt"
pixi run -e dev --frozen python "$DRIVER" --help \
  > "$RUN_ROOT/preflight/driver-help.txt"
rg -- "--initialization" "$RUN_ROOT/preflight/driver-help.txt"
rg -- "--initialization-seed" "$RUN_ROOT/preflight/driver-help.txt"
sha256sum "$FROZEN_INPUT" > "$RUN_ROOT/preflight/input.sha256"
test "$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')" = "$FROZEN_INPUT_SHA"
```

For each \(K\), run one deterministic and three random-recursive dry runs.
Require finite targets, exact closure, identical initial likelihoods within
floating point, four distinct topology hashes, and exact replay of one
repeated initialization seed:

| \(K\) | \(\kappa\) | strategy | initialization seed |
|---:|---:|---|---:|
| 50 | 100 | largest-nominal | none |
| 50 | 100 | random-recursive | 51051 |
| 50 | 100 | random-recursive replay | 51051 |
| 50 | 100 | random-recursive | 51052 |
| 50 | 100 | random-recursive | 51053 |
| 250 | 500 | largest-nominal | none |
| 250 | 500 | random-recursive | 51251 |
| 250 | 500 | random-recursive replay | 51251 |
| 250 | 500 | random-recursive | 51252 |
| 250 | 500 | random-recursive | 51253 |

Use the common scientific arguments from Stage R1 below and add `--dry-run`.
Dry runs do not create their output directory.

Before launching the matrix, perform one random-start awkward restart:

1. direct run: 14 transitions;
2. first segment: 5 transitions;
3. resumed segment: 9 transitions from the five-transition checkpoint;
4. require byte-identical direct/resumed final checkpoint arrays and
   transition traces;
5. repeat the resume with a different initialization seed and require explicit
   manifest rejection before sampling or output publication.

Any initializer replay, closure, checkpoint, or manifest failure is a hard
stop. The HPC agent may diagnose and patch an isolated test-harness or
round-off defect on a new commit, but must not weaken a scientific invariant,
edit the frozen input, overwrite the failure record, or continue expensive
jobs against uncommitted source.

## Stage R1: dispersed-start mobility screen

Run eight independent chains:

| \(K\) | chain | start | initialization seed | sampler seed |
|---:|---:|---|---:|---:|
| 50 | 0 | largest-nominal | none | 61050 |
| 50 | 1 | random-recursive | 51051 | 61051 |
| 50 | 2 | random-recursive | 51052 | 61052 |
| 50 | 3 | random-recursive | 51053 | 61053 |
| 250 | 0 | largest-nominal | none | 61250 |
| 250 | 1 | random-recursive | 51251 | 61251 |
| 250 | 2 | random-recursive | 51252 | 61252 |
| 250 | 3 | random-recursive | 51253 | 61253 |

Each chain runs 20,000 complete cycles, or 280,000 atomic transitions. Enable
movement diagnostics for this screen. Based on the successful `0c4be63`
throughputs, one chain should take roughly 9 minutes at \(K=50\) and 46
minutes at \(K=250\), plus diagnostics overhead and filesystem variability.
Request one CPU and 4 GiB per array member; 8 GiB is conservative if it is
easier to retain the already-tested resource request.

The common invocation is:

```bash
pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUTPUT" \
  --k "$K" --cycles 20000 --seed "$SAMPLER_SEED" \
  --chain-id "full-tiling-k${K}-chain${CHAIN}" \
  --concentration "$KAPPA" --root-variance 0.25 \
  --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --root-slice-width 1 --root-slice-max-steps 100 \
  --root-slice-max-shrink-steps 1000 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile \
  --collect-movement-diagnostics \
  "${INITIALIZATION_ARGUMENTS[@]}"
```

For chain zero, `INITIALIZATION_ARGUMENTS=(--initialization largest-nominal)`.
For chains one to three:

```bash
INITIALIZATION_ARGUMENTS=(
  --initialization random-recursive
  --initialization-seed "$INITIALIZATION_SEED"
)
```

Write each chain to
`$RUN_ROOT/stage-r1/k${K}/chain${CHAIN}/segment000`. Capture stdout and
`/usr/bin/time -v` stderr outside the immutable bundle. Submit the eight cells
as a Slurm array or separate jobs; do not run several timing cells
concurrently on the same allocated CPU.

Every successful directory must contain `manifest.json`, `trace.nc`,
`summary.json`, `checkpoint.npz`, and last-written `complete.json`. Verify all
hashes before analysis. The four initial topology hashes at each \(K\) must be
distinct.

## Stage R1 analysis

Analyze each \(K\) independently. Preserve every cycle-boundary draw for
diagnostics; thinning is allowed only for plotting or storage copies.

### Common-coordinate convergence

Before opening any chain trace, write and hash a projection-definition JSON.
Use exactly 24 rectangles from the Cartesian product of row edges
`[0, 30, 61, 91, 122, 152, 183]` and column edges
`[0, 32, 64, 96, 128]`. Compute rank-normalized split-\(\hat R\), bulk ESS,
and tail ESS for:

- root total \(T\);
- log likelihood;
- each of the six outer coefficients;
- the 24 fixed spatial projections of the native scaling field defined above,
  identically for every chain and both values of \(K\).

Reconstruct a draw's native scaling by assigning each leaf

\[
s_\ell = m_\ell W / w_\ell,
\]

where \(m_\ell\) is its additive mass, \(w_\ell\) is the sum of raw frozen
nominal weights in that rectangle, and \(W\) is the raw whole-domain weight
sum recorded as the manifest's `weight_normalization_factor`. Equivalently,
divide by the normalized leaf weight \(w_\ell/W\). Omitting \(W\) gives the
wrong scaling units.

The coarse projection must be nominal-emission-weighted:

\[
\bar s_B = \frac{\sum_{c\in B} w_c s_c}{\sum_{c\in B} w_c}.
\]

Do not compare leaf arrays by their canonical index: a leaf index has no
common physical meaning across tilings. For pairwise chain-mean field
distances, use the predeclared nominal-mass-weighted RMS metric

\[
d(a,b)=\sqrt{\frac{\sum_c w_c(s_{a,c}-s_{b,c})^2}{\sum_c w_c}}.
\]

Absolute log target may be diagnosed across chains only within the same
fixed-\(K\) model. It must not be compared between \(K=50\) and \(K=250\).

### Structural communication and movement

For each chain and half-chain report:

- distinct topology hashes and topology revisit rate;
- accepted edge flips and resolution relocations;
- valid and accepted fractions by structural move;
- changed cell count, changed nominal mass, and standardized prediction
  displacement distributions;
- a clearly labelled cycle-boundary return/cancellation proxy for the two
  structural slots, not an exact path-level immediate-reversal rate;
- slice displacement and slice work;
- pair-allocation displacement;
- wall time, transitions/s, peak RSS, and diagnostics overhead context.

Plot:

1. root total, log likelihood, and outer-coefficient traces;
2. selected coarse spatial projections;
3. cumulative distinct topologies;
4. structural displacement and acceptance by chain;
5. pairwise distances between chain mean native scaling fields through time.

### R1 decision

R1 is a screen, not a convergence claim.

- **Hard failure:** corrupt/incomplete artifacts, non-finite states,
  restart/hash mismatch, literal-zero diagnostic ownership failure, or no
  valid/accepted structural moves.
- **Stop as a mixing failure:** different initial topologies remain in
  clearly separated bands in common spatial projections and log likelihood,
  with no reduction in separation during the second half. Do not pool.
- **Extend to R2:** chains show overlap or movement toward overlap, structural
  moves explore new topologies, and no numerical/durability problem appears.
  A provisional \(\hat R>1.01\) or low ESS after R1 is expected and is a
  reason to extend, not by itself a failure.

Record the decision separately for \(K=50\) and \(K=250\). One value of \(K\)
may proceed even if the other fails.

## Stage R2: exact continuation to the old opportunity budget

For each value of \(K\) promoted from R1, resume the same four logical chains
for five additional 20,000-cycle segments. This gives 120,000 cycles or
1,680,000 atomic transitions per chain, matching the previous nominal
opportunity budget.

Each segment must:

- use the preceding segment's `checkpoint.npz`;
- retain the same chain ID, sampler seed, initialization strategy/seed,
  scientific arguments, and kernel settings;
- write to a new non-existing `segmentNNN` directory;
- validate its parent checkpoint hash and exact schedule phase;
- preserve the source segment and checkpoint unchanged.

Movement diagnostics may be disabled after `segment000`: they are output-only,
their trace/checkpoint parity has passed, and disabling them saves the measured
11--17% overhead. Keep diagnostics enabled in at least one full chain if the
storage and CPU budget permit so movement can be assessed over the longer
trajectory.

Treat `segment000` (280,000 transitions) as the predeclared warmup for the
first convergence report. This is an analysis convention, not sampler
adaptation. Concatenate later segments in exact global-transition order and
repeat every R1 analysis.

The minimum standard for evidence that the chain forgets its initial geometry
at fixed \(K\) is:

- rank-normalized split-\(\hat R\le1.01\) for every reported common scalar and
  spatial projection;
- bulk and tail ESS at least 400 for each reported quantity;
- no persistent clustering by initial topology;
- stable chain means and uncertainty bands over the final two segments;
- continued structural mobility without one-step reversal domination.

Failure of these gates is a scientific result about the tested local kernel.
Report it directly; do not pool chains, select the visually preferred start,
or substitute longer runtime for demonstrated communication. Passing them
shows structural start-insensitivity from a common physical state; it is not
alone sufficient to claim a converged pooled posterior because \(T\), leaf
scalings, and outer coefficients were not initialized overdispersedly. A
separate continuously overdispersed ensemble or a validated prior-draw
initializer is required before that stronger claim.

## Optional follow-ups, not part of the launch

Do not add these before R1:

- \(K=100\), which is useful only after the K=50/250 start-sensitivity screen;
- variable-\(K\) moves or an RJ structural normalizer;
- parallel tempering;
- likelihood-informed structural selection;
- a fixed-geometry comparator;
- AR(1)/OU model-data mismatch covariance.

A fixed-geometry or standard inversion is still needed later to quantify the
cost and scientific value of averaging over tilings. The current compound
schedule has no supported “disable structural slots” mode, so such a
comparator must be implemented and tested explicitly rather than emulated by
discarding structural proposals.

## Required report and provenance

Write:

- `analysis/chain-matrix.csv` with all seeds, strategies, hashes, segment
  ranges, timings, RSS, and artifact paths;
- `analysis/common-diagnostics.csv`;
- `analysis/structural-diagnostics.csv`;
- all plot inputs in machine-readable NetCDF/CSV/JSON;
- `report/RESULTS.md`, stating pass/warn/fail for every gate;
- `report/summary.json`;
- `report/sha256sums.txt` covering every retained byte artifact.

The report must begin with one of:

1. “fixed-\(K\) chains passed the declared structural start-sensitivity
   gates”;
2. “chains were mobile but did not pass the declared structural
   start-sensitivity gates”; or
3. “the run is invalid because a numerical/durability gate failed.”

It must also state explicitly that \(K\) was not inferred and that the
random-recursive initialization law was not used as a structural prior.
