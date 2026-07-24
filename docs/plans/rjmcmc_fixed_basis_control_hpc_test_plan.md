# Fixed-basis versus mobile-tiling PARIS HPC comparison

## Purpose

The dispersed-start mobile full-tiling screen at commit `548fa41` was
structurally active but failed its predeclared mixing gate:

- \(K=50\): worst \(\hat R=1.522\), minimum bulk ESS 7.2;
- \(K=250\): worst \(\hat R=1.948\), minimum bulk ESS 5.5;
- log-likelihood bands remained disjoint and separated further in the second
  half;
- native-field distances contracted and thousands of distinct tilings were
  visited, but this did not produce likelihood mixing.

The headline report did not name the variables associated with the worst
\(\hat R\) and ESS. Do not infer those names from the summary. This plan first
extracts them from the archived `common-diagnostics.csv`, then performs a
matched control which changes only whether the deterministic tiling can move.

The primary question is:

> With the same deterministic basis, likelihood, priors, initial physical
> state, and continuous update opportunities, do the continuous parameters
> mix when structural tiling moves are disabled?

This is a diagnostic comparison, not a production inversion. In particular,
it is a fixed-geometry Gamma--Dirichlet control, not the conventional
independent-coefficient “standard inversion”.

## Compared targets and schedules

Both arms use the current full-tiling continuous model:

- May 2014 PARIS methane data with 1,382 observations;
- the exact deterministic `largest-nominal` tiling at each fixed \(K\);
- Gamma(4,4) prior for the globally additive inner total \(T\);
- additive-alpha allocation with \(\kappa=2K\);
- arithmetic-mean/SD \(1/1\) lognormal priors for six outer coefficients;
- independent Gaussian likelihood with the frozen observation errors;
- likelihood power one;
- prior-mean initial physical state: native scaling one and outer
  coefficients one.

The arms differ only in structural support and schedule:

| Arm | Structural support | Cycle |
|---|---|---|
| `mobile` | reachable canonical fixed-\(K\) tilings | 2 structural + 1 root slice + 5 pair allocation + 6 outer = 14 slots |
| `fixed-basis` | point mass at the recorded deterministic tiling | 1 root slice + 5 pair allocation + 6 outer = 12 slots |

One complete cycle in either arm therefore gives exactly:

- one root-total update;
- five unordered pair-allocation updates; and
- one update of each of the six outer coefficients.

Match cycles and useful proposal opportunities, not raw atomic-transition
counts. Report cycles/s, useful proposals/s, ESS/cycle, and ESS/wall-hour.
Raw transitions/s is not a fair cross-schedule metric.

The fixed-basis structural density is a normalized point mass. The mobile
structural target remains uniform over its reachable fixed-\(K\) component
with an omitted within-\(K\) constant. Never compare absolute log targets
between values of \(K\) **or between the fixed and mobile arms**, even at the
same \(K\).

## Interpretation limits

All four chains in an arm start from the same physical state. The matched
mobile arm is necessary because the previous screen used four different
initial geometries; only one previous chain used the deterministic basis.

Agreement among these same-start chains can identify a strong structural
penalty, but cannot by itself establish full posterior convergence. If the
fixed-basis arm appears well mixed, a later continuously overdispersed
fixed-basis ensemble is required before reporting its posterior scientifically.

The deterministic bases are controls, not optimized or preferred bases. A
greedy or independently constructed fixed basis is a later sensitivity test.

## Frozen inputs and archived comparison

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
export MOBILE_R1_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_mixing/548fa41f1fef8b8cab93a6afe8717fdb562f689f
```

The archived R1 root is immutable. Verify its existing inventory before using
its diagnostics:

```bash
cd "$MOBILE_R1_ROOT"
sha256sum --check report/sha256sums.txt
```

## Stage C0: checkout and provenance

Run on compute nodes, not a login node. Do not specify a partition unless the
site requires it. Request one CPU and initially 4 GiB per chain; previous peak
RSS was 2.24 GiB.

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
module load git/2.45.1-pqk5
git fetch origin
git switch codex/rjmcmc-fixed-basis-control
git pull --ff-only
test -z "$(git status --porcelain)"

export CODE_REVISION="$(git rev-parse HEAD)"
export DRIVER=examples/rjmcmc/full_tiling_native_smoke.py
export RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_control/"$CODE_REVISION"
mkdir -p "$RUN_ROOT"/{preflight,prior-r1,matrix,analysis,report}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Record:

```bash
git rev-parse HEAD > "$RUN_ROOT/preflight/code-revision.txt"
git show -s --format=fuller HEAD > "$RUN_ROOT/preflight/code-commit.txt"
git diff --exit-code
git diff --cached --exit-code
pixi run -e dev --frozen python -VV > "$RUN_ROOT/preflight/python.txt" 2>&1
pixi list > "$RUN_ROOT/preflight/pixi-list.txt"
uname -a > "$RUN_ROOT/preflight/uname.txt"
lscpu > "$RUN_ROOT/preflight/lscpu.txt"
sha256sum "$FROZEN_INPUT" > "$RUN_ROOT/preflight/input.sha256"
test "$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')" = "$FROZEN_INPUT_SHA"
```

## Stage C1: identify the previous worst variables

Read
`$MOBILE_R1_ROOT/analysis/common-diagnostics.csv`. For each \(K\), write:

- the variable with maximum rank-normalized split-\(\hat R\);
- the variable with minimum bulk ESS;
- the variable with minimum tail ESS;
- the ten worst rows ranked by \(\hat R\), with bulk/tail ESS and chain means;
- the diagnostic class: root, likelihood, outer coefficient, or spatial
  projection.

Write both CSV and JSON under `prior-r1/`. The report must give names and
values, not only “worst R-hat” or “minimum ESS”.

This is also where the report should explain that overlapping coarse
projections do not imply likelihood mixing. The 24 projections are marginal,
nominal-base-measure-weighted linear summary; for this frozen input that base
measure is spherical grid-cell area. In contrast, log likelihood is a
quadratic over 1,382 error-standardized residuals. Small correlated
observation-space differences can separate likelihood without separating one
coarse projection.

## Stage C2: local and real-input preflight

Run only the experimental test suite and focused static checks:

```bash
pixi run -e dev --frozen pytest -q tests/experimental/rjmcmc \
  > "$RUN_ROOT/preflight/pytest.txt" 2>&1
pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_io.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  tests/experimental/rjmcmc/test_full_tiling_compound_sampling.py \
  tests/experimental/rjmcmc/test_full_tiling_io.py \
  tests/experimental/rjmcmc/test_full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/ruff.txt" 2>&1
pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_io.py \
  examples/rjmcmc/full_tiling_native_smoke.py \
  > "$RUN_ROOT/preflight/pyright.txt" 2>&1
```

Capture the interface and require the comparison control:

```bash
pixi run -e dev --frozen python "$DRIVER" --help \
  > "$RUN_ROOT/preflight/driver-help.txt"
rg -- "--structure-mode" "$RUN_ROOT/preflight/driver-help.txt"
```

For \(K=50\) and \(K=250\), run mobile and fixed-basis dry runs with
`--initialization largest-nominal`. Require:

- identical input SHA, topology SHA, initial likelihood, initial continuous
  coordinates, closure, and prior settings within each \(K\);
- mobile manifest: schedule version for 14 slots, two structural slots,
  uniform reachable structural target, connectivity not proved;
- fixed manifest: schedule version for 12 slots, zero structural slots,
  normalized singleton point-mass support, connectivity true;
- manifest schema v4 and exact mode/schedule identity.

The common arguments are those in Stage C4, with `--dry-run`.

Perform exact direct/restart tests on the real input:

- fixed basis: direct 12 versus 5+7 transitions;
- mobile: direct 14 versus 5+9 transitions;
- exact final coordinates, caches, target components, PCG64 state, and
  non-elapsed trace fields;
- fixed trace contains one root, five pair-allocation, and six fixed rows,
  with no structural slot or structural move;
- fixed rectangle bounds equal the deterministic initial bounds at every
  retained draw;
- resuming a fixed checkpoint under a mobile manifest, or vice versa, fails
  before sampling and creates no completion marker;
- resuming fixed mode with any non-deterministic initializer fails before data
  sampling.

Checkpoint archive schema remains v1 because schedule identity was already a
persisted field. Existing mobile v1 checkpoint semantics must still pass.
Driver manifest v4 is an intentional run-identity boundary; old v3 output
bundles are not resumed by this comparison driver.

Any mismatch is a hard stop before the matrix.

## Stage C3: short prior-only schedule calibration

For both \(K\), run four fixed-basis chains for 10,000 cycles with likelihood
power zero. This is a descriptive scheduler/wiring calibration, not a new
proof of the already validated proposal mathematics and not a formal
finite-sample moment gate.

Check:

- root \(T\) against Gamma(4,4);
- fixed leaf allocation against the declared additive-alpha Dirichlet
  marginals and selected pair contrasts;
- outer coefficients against their arithmetic mean/SD \(1/1\) lognormal
  priors;
- exact opportunity counts and unchanged topology;
- no non-finite values or slice guard failures.

Discard the first 2,000 cycles for the descriptive moment table, report ESS
and Monte Carlo standard errors, and express differences from analytic moments
in MCSE units. Do not pass or fail the matrix from a noisy empirical moment
alone. A target, opportunity-count, topology, non-finite-state, or restart
mismatch is a hard failure.

## Stage C4: matched real-data matrix

Run 16 independent jobs:

| \(K\) | mode | chains | cycles/chain | atomic transitions/chain |
|---:|---|---:|---:|---:|
| 50 | mobile | 4 | 20,000 | 280,000 |
| 50 | fixed-basis | 4 | 20,000 | 240,000 |
| 250 | mobile | 4 | 20,000 | 280,000 |
| 250 | fixed-basis | 4 | 20,000 | 240,000 |

Use sampler seeds:

- \(K=50\): 71050, 71051, 71052, 71053;
- \(K=250\): 71250, 71251, 71252, 71253.

Use the same four numeric seeds in both arms as stable matched labels. They are
not common-random-number trajectories because the mobile structural kernels
consume additional conditional RNG draws.

Use this exact array mapping:

| Array tasks | mode | \(K\) | chain | sampler seed |
|---:|---|---:|---:|---:|
| 0--3 | mobile | 50 | task modulo 4 | 71050 + chain |
| 4--7 | fixed-basis | 50 | task modulo 4 | 71050 + chain |
| 8--11 | mobile | 250 | task modulo 4 | 71250 + chain |
| 12--15 | fixed-basis | 250 | task modulo 4 | 71250 + chain |

Common invocation:

```bash
pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUTPUT" \
  --structure-mode "$MODE" \
  --initialization largest-nominal \
  --k "$K" --cycles 20000 --seed "$SAMPLER_SEED" \
  --chain-id "${MODE}-k${K}-chain${CHAIN}" \
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
  --collect-movement-diagnostics
```

Set \(\kappa=100\) at \(K=50\) and 500 at \(K=250\). Write to:

```text
$RUN_ROOT/matrix/<mode>/k<K>/chain<CHAIN>/segment000
```

Save the following as `$RUN_ROOT/preflight/run-comparison-array.sbatch`:

```bash
#!/bin/bash
set -u

TASK=${SLURM_ARRAY_TASK_ID:?}
CHAIN=$((TASK % 4))
GROUP=$((TASK / 4))

case "$GROUP" in
  0) MODE=mobile;      K=50;  KAPPA=100; SAMPLER_SEED=$((71050 + CHAIN)) ;;
  1) MODE=fixed-basis; K=50;  KAPPA=100; SAMPLER_SEED=$((71050 + CHAIN)) ;;
  2) MODE=mobile;      K=250; KAPPA=500; SAMPLER_SEED=$((71250 + CHAIN)) ;;
  3) MODE=fixed-basis; K=250; KAPPA=500; SAMPLER_SEED=$((71250 + CHAIN)) ;;
  *) echo "Unsupported array task $TASK" >&2; exit 2 ;;
esac

PARENT="$RUN_ROOT/matrix/$MODE/k$K/chain$CHAIN"
OUTPUT="$PARENT/segment000"
LOG_PREFIX="$PARENT/segment000-run"
mkdir -p "$PARENT"

status=0
/usr/bin/time -v pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUTPUT" \
  --structure-mode "$MODE" \
  --initialization largest-nominal \
  --k "$K" --cycles 20000 --seed "$SAMPLER_SEED" \
  --chain-id "${MODE}-k${K}-chain${CHAIN}" \
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
  > "$LOG_PREFIX.stdout.json" 2> "$LOG_PREFIX.stderr.txt" || status=$?

if [ "$status" -eq 0 ]; then
  pixi run -e dev --frozen python - "$OUTPUT" <<'PY' || status=$?
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
completion = json.loads((root / "complete.json").read_text())
for name, expected in completion["sha256"].items():
    digest = sha256((root / name).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"hash mismatch for {root / name}")
PY
fi

printf '%s\n' "$status" > "$LOG_PREFIX.exit-status.txt"
exit "$status"
```

Submit it with one CPU per array member and no explicit partition:

```bash
mkdir -p "$RUN_ROOT/matrix/slurm"
sbatch \
  --array=0-15 \
  --cpus-per-task=1 \
  --mem=4G \
  --time=02:00:00 \
  --output="$RUN_ROOT/matrix/slurm/%A_%a.stdout.txt" \
  --error="$RUN_ROOT/matrix/slurm/%A_%a.stderr.txt" \
  "$RUN_ROOT/preflight/run-comparison-array.sbatch"
```

The submission environment must export `RUN_ROOT`, `DRIVER`, `FROZEN_INPUT`,
`FROZEN_INPUT_ID`, `FROZEN_INPUT_SHA`, `CODE_REVISION`, `WEIGHT_POLICY`, and
`OUTER_LABELS`; Slurm's default environment export is expected. Capture
stdout and `/usr/bin/time -v` stderr outside each immutable artifact bundle.
Do not co-locate multiple timing cells on one allocated CPU.

Every output must pass its recorded hashes and contain `manifest.json`,
`trace.nc`, `summary.json`, `checkpoint.npz`, and last-written
`complete.json`.

## Stage C5: comparison diagnostics

Analyze each \(K\) and arm separately. Preserve every cycle-boundary draw for
diagnostics; thin only plot copies.

For every diagnostic table, include explicit columns for the variable name,
class, maximum \(\hat R\), minimum bulk ESS, minimum tail ESS, and the chain
responsible for extreme means. Produce a ranked worst-ten table.

### Common diagnostics

Use:

- root total \(T\) and \(\log T\);
- normalized Gaussian log likelihood;
- six outer coefficients;
- exactly 24 predeclared spatial projections from the Cartesian product of row
  edges `[0, 30, 61, 91, 122, 152, 183]` and column edges
  `[0, 32, 64, 96, 128]`;
- nominal-base-measure/area-weighted distances between chain-mean native
  scaling fields;
- error-standardized observation-space distance between chain mean
  predictions:

\[
d_\mathrm{obs}(a,b)=
\sqrt{\frac1N\sum_{i=1}^N
\left(\frac{\bar\mu_{a,i}-\bar\mu_{b,i}}{s_i}\right)^2};
\]

For a block \(B\), use the spherical-area/base-measure-weighted projection

\[
\bar s_B=\frac{\sum_{c\in B}w_cs_c}{\sum_{c\in B}w_c}.
\]

Reconstruct predictions in a streaming calculation; do not materialize a
chain-by-draw-by-observation cube unless storage has been checked.

### Fixed-basis diagnostics

Because leaf identity and bounds are common across all fixed chains, also
compute diagnostics for every:

- leaf mass;
- leaf scaling;
- centered log share relative to its nominal share.

For leaf \(\ell\), define normalized base mass
\(q_\ell=\sum_{c\in\ell}w_c/\sum_cw_c\), scaling
\(s_\ell=m_\ell/q_\ell\), and centered log share

\[
\log\left(\frac{m_\ell/T}{q_\ell}\right)
=\log(s_\ell/T).
\]

Labels must include the exact rectangle bounds. This identifies whether one
specific high-sensitivity or low-base-measure region is the bottleneck.

### Schedule and performance diagnostics

Require exact per-cycle opportunities. Report:

- cycles/s and wall time;
- root, pair-allocation, and per-outer-coefficient opportunities/s;
- ESS per 1,000 cycles;
- ESS per wall-hour;
- peak RSS;
- acceptance and displacement by continuous move.

Do not use raw transitions/s as the headline fixed/mobile comparison.

## Decision table

Interpret the results as follows:

| Result | Evidence |
|---|---|
| Fixed mixes at both K; mobile fails | Supports geometry or geometry-continuous coupling as the likely bottleneck for this basis and schedule |
| Fixed mixes at K=50 but fails at K=250 | Supports continuous allocation dimension, five-pair schedule, or K-specific basis geometry as the likely degradation |
| Fixed fails at both K | Root/share/outer likelihood geometry or continuous proposals are inadequate even without structural moves |
| Same-start mobile mixes but prior dispersed-start mobile failed | Initial-topology basins or communication between them are the main issue |
| Same-start fixed appears mixed but later overdispersed fixed fails | Same-start diagnostics masked conditional posterior modes or ridges |

Use rank-normalized split-\(\hat R\le1.01\) and bulk/tail ESS at least 400 as
the same formal thresholds as the prior R1 plan, but retain the same-start
caveat. Failure must be reported without pooling.

Do not automatically extend to a longer run. First diagnose the named worst
variables. A follow-up may change pair-update count, add blocked allocation
updates, or introduce continuously overdispersed initial coordinates,
depending on this comparison.

## Artifacts and final report

Write:

- `prior-r1/worst-common-diagnostics.csv` and `.json`;
- `analysis/chain-matrix.csv`;
- `analysis/common-diagnostics.csv`;
- `analysis/fixed-leaf-diagnostics.csv`;
- `analysis/prediction-space-diagnostics.csv`;
- `analysis/schedule-performance.csv`;
- plot inputs in machine-readable NetCDF/CSV/JSON;
- `report/RESULTS.md`;
- `report/summary.json`;
- `report/sha256sums.txt`.

The first page of `RESULTS.md` must explicitly state:

1. the exact worst-\(\hat R\), minimum-bulk-ESS, and minimum-tail-ESS variables
   in the archived mobile R1 run;
2. the exact worst variables in every new \(K\)/mode cell;
3. whether fixed geometry changed likelihood mixing;
4. that the fixed basis was the deterministic control, not an optimized
   basis;
5. that same-start agreement is not complete convergence evidence; and
6. that \(K\) was fixed and absolute targets were not compared across \(K\) or
   structure mode.

All output roots are revision-specific and non-overwriting. Nothing is written
to `PARIS_inversions`.
