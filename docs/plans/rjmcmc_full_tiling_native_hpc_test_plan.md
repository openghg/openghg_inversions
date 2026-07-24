# Fixed-K full-tiling native-data HPC smoke test

## Purpose and current status

This is the first real-data test of the construction-history-free full-tiling
prototype. It uses the reconstructed modern PARIS May 2014 frozen native-grid
input, but it does **not** reproduce either the moving-Voronoi model or the
fixed-direction Gamma--Beta tree model.

The executable driver is
`examples/rjmcmc/full_tiling_native_smoke.py`. The first run is deliberately:

- fixed \(K\);
- one uninterrupted NumPy chain;
- independent fixed Gaussian observation errors;
- fixed archived `YaprioriBC`;
- six inferred outer coefficients;
- diagnostic only, with no convergence or posterior-summary claim; and
- restricted to the communication component reachable from the deterministic
  starting tiling until connectivity of the production move graph is proved.

The smoke is ready when the focused experimental tests and the local synthetic
NetCDF integration test pass at the committed revision. Durable restart,
variable-\(K\) moves, Numba, correlated errors, and production inference are
separate follow-ups.

## Scientific target

For a canonical recursive midpoint-bisection tiling
\(P=\{R_1,\ldots,R_K\}\), the active coordinates are:

\[
T>0,\qquad
(s_1,\ldots,s_K)\in\Delta_{K-1},\qquad
c_1,\ldots,c_6>0.
\]

With normalized nominal native-cell weights \(w_j\), define

\[
W_R=\sum_{j\in R}w_j,\qquad
X_R=T s_R,\qquad
a_R=X_R/W_R.
\]

The inner field has one scaling \(a_R\) within each active rectangle, so the
prediction is

\[
\hat y =
Y_{\mathrm{aprioriBC}}
+\sum_{R\in P}\frac{X_R}{W_R}\sum_{j\in R}G_j
+H_{\mathrm{outer}}c.
\]

This is a genuine partition-dependent reduced model: unresolved within-leaf
scaling contrasts are removed rather than exactly marginalized.

The prior is:

\[
T\sim\operatorname{Gamma}(a,b),\qquad
s\mid P\sim\operatorname{Dirichlet}(\kappa W_{R_1},\ldots,\kappa W_{R_K}),
\]

plus independent arithmetic-moment lognormal priors for the six outer
coefficients. The fixed-\(K\) structural target is uniform over unique
canonical tilings. Its unknown normalizing count cancels within a connected
fixed-\(K\) component; it cannot be ignored for variable-\(K\) inference.

The numerical target uses the \(T+\)shares chart. The root Gamma target and
independence-proposal terms therefore cancel without a
\(T^{K-1}\) Jacobian. Structural moves use the explicitly verified augmented
maps and the resolution-relocation Jacobian.

## Schedule

With six outer coefficients and five pair-allocation slots, one cycle contains
14 atomic transitions:

1. two structural slots, each independently choosing edge flip or resolution
   relocation with probability \(1/2\);
2. one Gamma-prior root-total refresh;
3. five uniformly selected unordered active-leaf pair allocation refreshes;
4. six fixed-coefficient random walks in deterministic outer-column order.

Every slot consumes an acceptance uniform. Unavailable or numerically
out-of-support proposals remain explicit self-transitions.

The smoke sampler never enumerates all tilings, edge-flip paths, or
relocation paths. It enumerates only currently mergeable friend pairs, and a
relocation selects from a fixed \(2(K-1)\) intermediate leaf-by-axis
catalogue. Full rebuilds remain the correctness oracle for incremental
prediction updates.

## Prior calibration for the smoke

The allocation concentration is a scientific setting, not a tuning
parameter. For a leaf with nominal mass \(W_R=w\), root mean one, and root
variance \(v\), the prior variance of its scaling is

\[
\operatorname{Var}(a_R)
=v+(1+v)\frac{1-w}{w(\kappa+1)}.
\]

The driver records the minimum, median, and maximum initial-leaf scaling
standard deviations implied by the supplied \(v\), \(\kappa\), and actual
nominal weights.

For the first diagnostic comparison only, use:

- root variance \(v=0.25\);
- \(\kappa=2K\);
- outer arithmetic mean one and arithmetic SD one; and
- fixed-coefficient random-walk SD 0.4.

For equal-mass leaves, \(\kappa=2K\) gives a leaf scaling SD near 0.93 for
moderate or large \(K\), close to but not identical to an arithmetic
mean-one/SD-one scaling prior. Unequal nominal masses deliberately have
different relative widths. These settings are an auditable smoke profile, not
a recovered Lunt or Ganesan prior and not a calibrated production choice.

## Frozen input contract

The reviewed PARIS file must contain:

| Variable | Dimensions | Meaning |
|---|---|---|
| `fp_x_flux` | `nmeasure, lat, lon` | response to unit native-cell scaling |
| `mf` | `nmeasure` | 1,382 filtered observations |
| `mf_error` | `nmeasure` | fixed positive diagonal errors |
| `nominal_weight` | `lat, lon` | reviewed strictly positive base measure |
| `outer_design` | `nmeasure, outer_region` | six inferred outer columns |
| `YaprioriBC` | `nmeasure` | fixed archived row-aligned BC contribution |

The PARIS profile additionally requires a \(183\times128\) inner grid, six
reviewed `outer_region` labels in exact order, and unique measurement labels.
No weight flooring, variable discovery, or boundary inference occurs.

Before sampling the driver checks, row by row,

\[
S w=\sum_jG_j
\]

and checks the complete prior-mean prediction against raw `fp_x_flux`, the
fixed BC contribution, and the six outer prior means.

## HPC handoff

### 1. Resolve and verify the exact revision and input

On BP1, use the checkout that owns the test:

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
git fetch origin
git switch codex/rjmcmc-full-tiling-real-data-smoke
git pull --ff-only
git status --porcelain
CODE_REVISION="$(git rev-parse HEAD)"
```

`git status --porcelain` must be empty. The previous Gamma--Beta Stage C run
independently recorded the following frozen-input contract in its run record,
input sidecar, launch script, Stage A dry run, and Stage C manifest:

```bash
FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
WEIGHT_POLICY=spherical-grid-cell-area-v1
RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_smoke/"$CODE_REVISION"
mkdir -p "$RUN_ROOT"
```

Verify the file against the independently pinned digest before running:

```bash
ACTUAL_INPUT_SHA="$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')"
test "$ACTUAL_INPUT_SHA" = "$FROZEN_INPUT_SHA"
```

The reviewed `nominal_weight` is normalized spherical grid-cell area under
`spherical-grid-cell-area-v1`. It is strictly positive with no epsilon floor
(recorded zero count 0, minimum `1.5552283732582164e-05`, maximum
`6.588256807625427e-05`, and sum `0.9999999999999999`). The 96 zero cells in
`prior_flux` are intentionally irrelevant to this base measure. Do not
silently substitute a different input, label order, or weight policy.

The authoritative audit trail is under
`/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa`:
`RUN_RECORD.md`, `input/paris_may_2014_gamma_beta_native.json`,
`jobs/common.sh`, `stage1/dry_low.json`, and
`stage3/chain_0/segment_000/manifest.json`. The recorded semantic SHA-256 of
the `nominal_weight` variable is
`bd3e014e042dc35f847c0aec44ffe88c843dbb82cc7da43bea4981aa835feb26`.

Pin numerical libraries to the requested single CPU:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Run the dry runs and sampling stages inside an interactive Slurm allocation or
a site-standard batch wrapper, not on the login node. Do not hard-code a
partition in the repository instructions. Start with one CPU, 8 GB, and
60 minutes; revise the sampling walltime only from a completed short report.

### 2. Run the focused code checks

Do not run the repository-wide tox matrix for this handoff. Run the
experimental RJMCMC tests and focused static checks:

```bash
pixi run -e dev --frozen pytest -q tests/experimental/rjmcmc
pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  examples/rjmcmc/full_tiling_native_smoke.py
pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling_posterior.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_compound_sampling.py \
  examples/rjmcmc/full_tiling_native_smoke.py
```

### 3. Dry-run both fixed dimensions

The dry run performs input hashing, PARIS-profile validation, problem
construction, and closure checks but creates no output directory:

```bash
pixi run -e dev --frozen python examples/rjmcmc/full_tiling_native_smoke.py \
  --input "$FROZEN_INPUT" \
  --output-directory "$RUN_ROOT/dry-k50-unused" \
  --k 50 --cycles 1 --seed 812 \
  --concentration 100 --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile --dry-run \
  > "$RUN_ROOT/dry-k50.json"

pixi run -e dev --frozen python examples/rjmcmc/full_tiling_native_smoke.py \
  --input "$FROZEN_INPUT" \
  --output-directory "$RUN_ROOT/dry-k250-unused" \
  --k 250 --cycles 1 --seed 1812 \
  --concentration 500 --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile --dry-run \
  > "$RUN_ROOT/dry-k250.json"
```

Both reports must show a finite initial target, cycle length 14, and closure at
the documented floating-point tolerance.

### 4. Prior-only mobility check

Run 20 cycles at each \(K\) with likelihood power zero. This tests structural
validity, prior/proposal accounting, geometry cost, and artifact persistence
without a data-likelihood barrier:

```bash
for K in 50 250; do
  if [ "$K" -eq 50 ]; then KAPPA=100; SEED=912; else KAPPA=500; SEED=1912; fi
  pixi run -e dev --frozen python examples/rjmcmc/full_tiling_native_smoke.py \
    --input "$FROZEN_INPUT" \
    --output-directory "$RUN_ROOT/prior-k${K}" \
    --k "$K" --cycles 20 --seed "$SEED" \
    --concentration "$KAPPA" --root-variance 0.25 --likelihood-power 0 \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision "$CODE_REVISION" \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" \
    --require-paris-profile
done
```

Do not rerun into an existing output directory. Use a new `RUN_ROOT` or new
suffix after any failed or interrupted attempt.

### 5. Full-likelihood real-data smoke

If both prior-only jobs complete, run 100 cycles at each fixed \(K\):

```bash
for K in 50 250; do
  if [ "$K" -eq 50 ]; then KAPPA=100; SEED=1012; else KAPPA=500; SEED=2012; fi
  /usr/bin/time -v pixi run -e dev --frozen python examples/rjmcmc/full_tiling_native_smoke.py \
    --input "$FROZEN_INPUT" \
    --output-directory "$RUN_ROOT/posterior-k${K}" \
    --k "$K" --cycles 100 --seed "$SEED" \
    --concentration "$KAPPA" --root-variance 0.25 --likelihood-power 1 \
    --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
    --input-id "$FROZEN_INPUT_ID" \
    --expected-input-sha256 "$FROZEN_INPUT_SHA" \
    --code-revision "$CODE_REVISION" \
    --nominal-weight-policy "$WEIGHT_POLICY" \
    --expected-outer-labels "$OUTER_LABELS" \
    --require-paris-profile \
    > "$RUN_ROOT/posterior-k${K}.stdout.json" \
    2> "$RUN_ROOT/posterior-k${K}.resource.txt"
done
```

The local smoke is not a reliable BP1 runtime prediction. Increase walltime
only after examining the first completed \(K=50\) report.

## Pass criteria and report

Each successful sampling directory must contain:

- `manifest.json`;
- `trace.nc`;
- `summary.json`; and
- `complete.json` written last.

Recompute every recorded completion hash, then inspect the summaries:

```bash
for RUN_DIRECTORY in "$RUN_ROOT/posterior-k50" "$RUN_ROOT/posterior-k250"; do
  pixi run -e dev --frozen python -c \
    'from hashlib import sha256; import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); c=json.loads((p/"complete.json").read_text()); actual={n:sha256((p/n).read_bytes()).hexdigest() for n in c["sha256"]}; assert actual == c["sha256"], (actual, c["sha256"]); print(p, "hashes OK")' \
    "$RUN_DIRECTORY"
done
python -m json.tool "$RUN_ROOT/posterior-k50/complete.json"
python -m json.tool "$RUN_ROOT/posterior-k50/summary.json"
python -m json.tool "$RUN_ROOT/posterior-k250/summary.json"
```

Report:

- closure errors;
- actual initial leaf scaling-SD ranges;
- setup and sampling time;
- atomic transitions per second and peak RSS;
- attempts, valid proposals, and accepted proposals by move;
- invalid-reason counts;
- distinct retained tilings;
- initial/final log target; and
- whether the prior-only and full-likelihood runs differ materially in
  structural validity or acceptance.

Hard failures are:

- input SHA/profile/closure failure;
- NaN or positive-infinite target/acceptance values;
- missing or invalid completion hashes;
- schedule phase other than zero;
- an attempted-transition count other than 280 for 20 cycles or 1,400 for
  100 cycles;
- no valid structural proposal at either \(K\); or
- incremental/full-rebuild disagreement in the focused tests.

Warnings, not automatic correctness failures, are:

- no accepted edge flip or no accepted relocation in 100 cycles;
- only one retained topology;
- extreme initial-leaf prior scaling SD caused by a small nominal-mass leaf;
- a large prior-only versus full-likelihood acceptance collapse; or
- K-dependent runtime dominated by recursive admissibility or merge discovery.

These short chains cannot establish mixing or convergence. Because \(K\) is
fixed, they also say nothing about inference over \(K\). Their purpose is to
establish that the full-tiling target, local moves, native PARIS likelihood,
and output boundary operate together on real data. The structural normalizer
of the reachable fixed-\(K\) component is unknown and omitted, so absolute log
targets must not be compared between the \(K=50\) and \(K=250\) runs.
