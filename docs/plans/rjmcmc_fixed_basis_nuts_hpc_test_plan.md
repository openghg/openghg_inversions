# Fixed-basis PyMC/NumPyro NUTS reference and grouped-root roadmap

## Status and purpose

The matched PARIS fixed-basis experiment showed that holding the deterministic
tiling fixed largely resolves likelihood mixing but does not fully resolve the
continuous-coordinate diagnostics:

| \(K\) | Geometry | Worst \(\hat R\) | Minimum bulk ESS | Minimum tail ESS |
|---:|---|---:|---:|---:|
| 50 | mobile | likelihood 1.1820 | likelihood 15.31 | `fixed_intem_label_2` 53.13 |
| 50 | fixed | `root_total` 1.1003 | `root_total` 31.34 | share 019 68.33 |
| 250 | mobile | likelihood 1.3173 | likelihood 9.85 | likelihood 28.01 |
| 250 | fixed | `root_total` 1.0976 | `root_total` 30.93 | `leaf_mass_124` 43.24 |

The fixed-\(K=50\) likelihood itself mixed well
(\(\hat R=1.0084\), bulk ESS 676.6, tail ESS 1403.3). This is evidence that
geometry or geometry--continuous coupling is the main likelihood bottleneck.
The remaining slow global root may be either:

1. a limitation of the current local root/share transition schedule;
2. difficult posterior geometry even at fixed basis; or
3. a scientific-model problem caused by assigning one coherent scaling mode
   to the heterogeneous inner European domain.

This plan introduces an independent, gradient-based reference for the **same
fixed-basis target**. PyMC defines the model and NumPyro NUTS samples its
continuous posterior. No reversible-jump or tiling move is involved.

This is a diagnostic reference, not a production inversion and not yet the
multi-root model.

## Exact target

For the recorded deterministic tiling in canonical leaf order,

\[
T\sim\operatorname{Gamma}(a,b),\qquad
p\sim\operatorname{Dirichlet}(\alpha),\qquad
m=Tp,
\]

\[
\hat y=Dm+o+Bc,\qquad
c_j\sim\operatorname{LogNormal}(\mu_j,\sigma_j),\qquad
y_i\sim N(\hat y_i,s_i).
\]

The implementation takes:

- \(a,b\) directly from the existing Gamma root prior;
- \(\alpha\) from the exact additive-alpha leaf measure;
- \(D\) from the existing cached rectangle design columns;
- \(o\) from the archived row-aligned fixed boundary contribution;
- \(B\) and the outer prior arithmetic moments from the fixed design block;
- \((\mu_j,\sigma_j)\) from the existing arithmetic-moment-to-lognormal
  conversion; and
- the Gaussian errors from the frozen PARIS input.

The target density is defined in scientific coordinates \((T,p,c)\). There is
no \(T^{K-1}\) leaf-mass Jacobian. The fixed tiling has a point-mass structural
prior and therefore contributes only a constant.

For this first reference, likelihood power must equal one. Powered likelihoods
and prior-only sampling are separate experiments.

The normalized nominal weight is spherical grid-cell area in this frozen
input. Consequently `root_total` is an area-weighted aggregate inner-domain
scaling, not a literal emissions total.

## What the comparison can establish

- If NUTS mixes the fixed target well from dispersed starts, the remaining
  fixed-basis failure is mainly a transition-kernel problem.
- If NUTS has the same root/likelihood separation, divergences, or extreme
  posterior curvature, the fixed target itself is difficult. This supports
  examining the scientific meaning of one global root.
- A NUTS success does not imply that NUTS can replace reversible jump. NUTS
  samples only the continuous coordinates conditional on one tiling.
- A NUTS failure at \(K=250\) is not conclusive if diagonal mass adaptation
  also fails at \(K=50\). A predeclared dense-mass \(K=50\) follow-up separates
  adaptation from scientific-model problems.

Do not compare NumPyro `sample_stats.lp` directly with the custom sampler's
`log_target`. With the pinned PyMC/NumPyro stack it is the negative density in
NumPyro's unconstrained transformed chart and includes transform Jacobians.
The sign and chart therefore both differ from the custom scientific-coordinate
target. Compare pointwise Gaussian log likelihood and independently recomputed
scientific-coordinate target components instead.

## Frozen input and immutable prior result

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
export FIXED_CONTROL_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_control/d1c673eb7eae4ee8bf18a15050898b4b6bb78d5c
```

The fixed-control result is read-only comparison evidence. Verify its checksum
manifest before extracting values; do not modify it.

Nothing in this plan writes to `PARIS_inversions`.

## N0: checkout and compute-node environment

Run on a compute node. Use one Slurm array task per chain. Initially request
one CPU, 16 GiB, and enough wall time for JAX compilation plus sampling. Memory
is deliberately generous for this first reference and should be reduced only
after measuring peak RSS.

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
module load git/2.45.1-pqk5
git fetch origin
git switch codex/rjmcmc-fixed-basis-nuts-reference
git pull --ff-only
test -z "$(git status --porcelain)"

export CODE_REVISION="$(git rev-parse HEAD)"
export DRIVER=examples/rjmcmc/full_tiling_fixed_basis_nuts.py
export RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_nuts/"$CODE_REVISION"
mkdir -p "$RUN_ROOT"/{preflight,smoke,matrix,analysis,report}

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export RUN_CACHE_ROOT="${SLURM_TMPDIR}/rjmcmc-nuts-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-single}"
mkdir -p "$RUN_CACHE_ROOT"/{xdg,matplotlib,pytensor}
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTENSOR_FLAGS="floatX=float64,base_compiledir=$RUN_CACHE_ROOT/pytensor"
```

Do not import `openghg_inversions.models` or the normal RHIME PyMC
configuration in the driver process: that path currently selects PyTensor
float32. The experimental driver must fail before sampling unless PyTensor
and JAX are both in 64-bit mode.

Record:

```bash
git rev-parse HEAD > "$RUN_ROOT/preflight/code-revision.txt"
git show -s --format=fuller HEAD > "$RUN_ROOT/preflight/code-commit.txt"
test "$CODE_REVISION" = "$(git rev-parse HEAD)"
git diff --exit-code
git diff --cached --exit-code
pixi run -e dev --frozen python -VV > "$RUN_ROOT/preflight/python.txt" 2>&1
pixi list > "$RUN_ROOT/preflight/pixi-list.txt"
uname -a > "$RUN_ROOT/preflight/uname.txt"
lscpu > "$RUN_ROOT/preflight/lscpu.txt"
sha256sum "$FROZEN_INPUT" > "$RUN_ROOT/preflight/input.sha256"
test "$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')" = "$FROZEN_INPUT_SHA"
```

`--code-revision` is recorded provenance supplied by this external preflight;
the driver does not infer or validate Git state. Before every driver call,
require `CODE_REVISION` to equal `git rev-parse HEAD`, require the two clean
diff checks above to pass, and archive those command results. A mismatch is a
hard stop.

## N1: code, precision, and target-parity preflight

Run only the experimental suite and focused static checks:

```bash
pixi run -e dev --frozen pytest -q tests/experimental/rjmcmc \
  > "$RUN_ROOT/preflight/pytest.txt" 2>&1
pixi run -e dev --frozen pytest -q \
  tests/test_rhime.py::test_build_rhime_model_contains_expected_variables \
  tests/experimental/rjmcmc/test_fixed_basis_nuts.py::test_pymc_model_has_expected_float64_variables_and_coordinates \
  'tests/experimental/rjmcmc/test_full_tiling_fixed_basis_nuts.py::test_dry_run_closes_and_matches_target_without_writing[prior-mean-None]' \
  > "$RUN_ROOT/preflight/pytest-mixed-pytensor-config.txt" 2>&1
pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/fixed_basis_nuts.py \
  examples/rjmcmc/full_tiling_fixed_basis_nuts.py \
  tests/experimental/rjmcmc/test_fixed_basis_nuts.py \
  tests/experimental/rjmcmc/test_full_tiling_fixed_basis_nuts.py \
  > "$RUN_ROOT/preflight/ruff.txt" 2>&1
pixi run -e dev --frozen ruff format --check \
  openghg_inversions/experimental/rjmcmc/fixed_basis_nuts.py \
  examples/rjmcmc/full_tiling_fixed_basis_nuts.py \
  tests/experimental/rjmcmc/test_fixed_basis_nuts.py \
  tests/experimental/rjmcmc/test_full_tiling_fixed_basis_nuts.py \
  > "$RUN_ROOT/preflight/format.txt" 2>&1
pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/fixed_basis_nuts.py \
  examples/rjmcmc/full_tiling_fixed_basis_nuts.py \
  > "$RUN_ROOT/preflight/pyright.txt" 2>&1
```

Verify the compute-node JAX installation and actual default dtype:

```bash
pixi run -e dev --frozen python - <<'PY' \
  > "$RUN_ROOT/preflight/jax.txt" 2>&1
import jax
import jax.numpy as jnp
import numpyro
import pymc
import pytensor

x = jnp.array([1.0, 2.0])
gradient = jax.grad(lambda value: jnp.sum(value * value))(x)
print("pymc", pymc.__version__)
print("numpyro", numpyro.__version__)
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("jax_x64", jax.config.x64_enabled)
print("pytensor_floatX", pytensor.config.floatX)
print("array_dtype", x.dtype)
print("gradient_dtype", gradient.dtype)
assert jax.default_backend() == "cpu"
assert jax.config.x64_enabled
assert pytensor.config.floatX == "float64"
assert str(x.dtype) == "float64"
assert str(gradient.dtype) == "float64"
PY
```

Run `--dry-run` for \(K=50\) and \(K=250\). The driver must:

- check the complete frozen-input SHA and reviewed PARIS dimensions/labels;
- build the deterministic largest-nominal tiling;
- verify prior-mean forward-model closure;
- verify every model value variable is float64;
- evaluate the PyMC normalized log density with transform Jacobians disabled;
- compare it with the independently assembled custom target in
  \((T,p,c)\) coordinates;
- record the maximum prediction discrepancy, log-target discrepancy, backend,
  precision, rectangle bounds, and topology hash; and
- create no output directory.

Use the common model arguments from N3 with `--dry-run`. Any wrong hash,
float32 variable, non-finite density, topology discrepancy, or scientific-logp
difference larger than the documented float64 tolerance is a hard stop.

The phrase “same fixed basis” is an experimental identity requirement, not
only a shared initializer name. Before N2, extract the topology SHA-256 from
every archived fixed-basis manifest under `$FIXED_CONTROL_ROOT`. Require one
unique hash within each \(K\), then require the new \(K=50\) and \(K=250\)
dry-run hashes to equal their corresponding archived hashes. Write the
archived paths, hashes, new hashes, and Boolean comparisons to
`preflight/topology-identity.json`. A mismatch is a hard stop until the input,
initializer, or hashing difference is understood.

Perform the same machine-readable equality audit for every target-defining
field shared by the two implementations:

- frozen input SHA and variable contract;
- fixed \(K\);
- concentration \(2K\);
- Gamma root shape/rate \(4/4\);
- likelihood power one;
- six outer coefficients in the same order;
- outer lognormal arithmetic means/SDs \(1/1\);
- fixed-offset and fixed-design variable names; and
- observation/error vector identity through the frozen input SHA and explicit
  observation/error variable names.

Write both source values and equality flags to
`preflight/fixed-target-identity.json`. The NUTS run is not an
apples-to-apples kernel diagnostic if any field differs, even if each model is
individually valid.

## N2: bounded real-data execution smoke

Before the full matrix, run one \(K=50\) chain with 100 tuning and 100 retained
draws. This is an execution and artifact test, not a convergence test:

```bash
pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" \
  --output-directory "$RUN_ROOT/smoke/k50-chain0" \
  --k 50 --draws 100 --tune 100 \
  --seed 83050 --chain-id k50-chain0 \
  --continuous-initialization prior-mean \
  --concentration 100 --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 \
  --target-accept 0.9 --max-tree-depth 12 --no-dense-mass \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile
```

Require:

- `manifest.json`, `trace.nc`, `summary.json`, and `complete.json`;
- completion written last and containing verified SHA-256 values for the first
  three files;
- one chain and exactly 100 retained draws;
- finite root, share, leaf-mass, leaf-scaling, outer, mean-prediction, and
  pointwise-log-likelihood values;
- normalized shares within a float64 tolerance;
- fixed rectangle coordinates equal the manifest bounds;
- no output pretending to provide custom-sampler checkpoint/restart
  equivalence; and
- reported total wall time, NUTS sampling time when available, peak RSS from
  Slurm, divergences, step count, tree depth, acceptance, and step size.

An interrupted monolithic NUTS chain is rerun in a new empty output directory.
This first reference does not claim exact mid-warmup or mid-chain restart.

## N3: dispersed-start real-data matrix

Run eight independent one-chain jobs:

| Tasks | \(K\) | chain | continuous start | initialization seed | sampler seed |
|---:|---:|---:|---|---:|---:|
| 0 | 50 | 0 | prior mean | none | 84050 |
| 1--3 | 50 | 1--3 | prior draw | 94051--94053 | 84051--84053 |
| 4 | 250 | 0 | prior mean | none | 84250 |
| 5--7 | 250 | 1--3 | prior draw | 94251--94253 | 84251--84253 |

Each prior draw uses a dedicated NumPy PCG64 stream and the exact Gamma,
Dirichlet, and lognormal priors. It must be finite and strictly interior. The
initialization stream is separate from the NumPyro sampler seed. NumPyro
transformed-space jitter remains disabled, so the recorded constrained start
is the actual start.

Use 2,000 tuning steps and 2,000 retained draws, diagonal mass adaptation,
target acceptance 0.9, and maximum tree depth 12:

```bash
TASK="${SLURM_ARRAY_TASK_ID}"
if [ "$TASK" -lt 4 ]; then
  K=50
  CHAIN="$TASK"
else
  K=250
  CHAIN=$((TASK - 4))
fi
SAMPLER_SEED=$((84000 + K + CHAIN))
INITIALIZATION_SEED=$((94000 + K + CHAIN))
KAPPA=$((2 * K))
OUTPUT="$RUN_ROOT/matrix/k${K}-chain${CHAIN}"
INIT_ARGS=(--continuous-initialization prior-mean)
if [ "$CHAIN" -ne 0 ]; then
  INIT_ARGS=(
    --continuous-initialization prior-draw
    --initialization-seed "$INITIALIZATION_SEED"
  )
fi

pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUTPUT" \
  --k "$K" --draws 2000 --tune 2000 \
  --seed "$SAMPLER_SEED" --chain-id "k${K}-chain${CHAIN}" \
  "${INIT_ARGS[@]}" \
  --concentration "$KAPPA" --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 \
  --target-accept 0.9 --max-tree-depth 12 --no-dense-mass \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile
```

Do not shorten chains to equal wall time. If a job fails, preserve its
incomplete directory, diagnose the cause, fix code only on a new commit, and
rerun the complete matrix under the new commit-addressed run root. The HPC
agent is authorized to debug environment, driver, and artifact issues and to
report a minimal source fix, but must not silently weaken scientific or
durability gates.

## N4: combine and diagnose

Verify every completion hash before reading traces. Concatenate the four
single-chain `InferenceData` objects for each \(K\) along `chain`; do not
concatenate \(K=50\) and \(K=250\).

Primary diagnostics:

- summed pointwise Gaussian log likelihood;
- `root_total` and `log(root_total)`;
- all six outer coefficients;
- the same 24 native-field projections used in the mobile/fixed comparison;
- posterior fitted-mean residual RMSE from `mean_observation`.

The 24 projections are not left to analyst choice. Use the Cartesian product
of row edges `[0, 30, 61, 91, 122, 152, 183]` and column edges
`[0, 32, 64, 96, 128]`. For normalized native nominal weights \(q_c\), fixed
leaf scalings \(s_\ell\), and block \(B\), compute

\[
\bar s_B =
\frac{\sum_{c\in B} q_c s_c}{\sum_{c\in B}q_c}.
\]

Each fixed leaf has one stored scaling, so the block calculation may intersect
the manifest's canonical rectangle bounds with \(B\); it does not need to
materialize a draw-by-native-cell array. Write and hash one
`projection-definition.json` containing the edges, ordering, weight policy,
and formula before opening chain traces.

The trace's `observation` and `fixed` coordinates are positional. Resolve
their scientific labels only from the exact frozen input: `nmeasure` order for
observations and the manifest-verified `outer_region` order for fixed
coefficients. Record that mapping in analysis output. Never infer labels from
position without the frozen input SHA and manifest label check.

Secondary diagnostics:

- leaf shares, masses, and scalings;
- selected coarse aggregations;
- energy/BFMI;
- divergences;
- mean and maximum tree depth and fraction at the depth limit;
- mean and maximum leapfrog step count;
- acceptance rate and adapted step size;
- bulk ESS, tail ESS, MCSE, ESS/retained draw, ESS/wall-hour, and
  ESS/CPU-hour.

Do not fail the scientific screen solely because the single worst tiny leaf
has a low ESS. Report the worst leaf, but apply the primary gate to likelihood,
root, outers, and predeclared coarse spatial summaries.

Suggested primary success criteria:

- rank-normalized split-\(\hat R\leq1.01\);
- bulk ESS at least 400 and tail ESS at least 200;
- zero divergences;
- no more than 1% of draws at maximum tree depth;
- per-chain BFMI at least 0.3; and
- no persistent start-separated likelihood band.

These are diagnostic thresholds, not proof of convergence. Report every
quantity even when a gate fails.

Compare against the archived custom fixed-basis result using scientific
coordinates and likelihood, not sampler `lp`. Report:

- whether NUTS resolves the custom sampler's slow root;
- posterior means/intervals and Monte Carlo uncertainty for common summaries;
- total wall and CPU cost;
- compile/warmup time separately from retained sampling when available; and
- the ratio of ESS/hour for the common primary variables.

## N5: predeclared follow-ups

1. If diagonal-mass NUTS has poor geometry at \(K=50\), repeat only the
   \(K=50\) four-chain matrix with dense mass adaptation. Keep all other
   settings and starts fixed and use a new output root.
2. If NUTS mixes the single-root target but the custom sampler does not,
   improve continuous within-tiling transitions before changing the
   scientific prior. A compound RJ/NUTS design is then technically justified,
   although it still needs correct RJ accounting across tilings.
3. If NUTS also shows a slow or strongly coupled global root, proceed to the
   grouped experiments below.

Parallel tempering is not part of this plan. It has been tried historically
and would not diagnose whether the fixed continuous parameterization or the
single-root prior is responsible.

## Grouped Gamma--Beta groundwork

Two grouped models must not be conflated.

### A. Exact grouped factorization of the current prior

For fixed, predeclared leaf groups \(g\),

\[
U\sim\operatorname{Dirichlet}(\alpha_g),\qquad
p_g\sim\operatorname{Dirichlet}(\alpha_{\ell\mid g}),\qquad
m_\ell=T\,U_g\,p_{\ell\mid g}.
\]

Here \(\alpha_g=\sum_{\ell\in g}\alpha_\ell\). By Dirichlet aggregation,
this is exactly the same global Dirichlet prior and retains the same global
Gamma root \(T\). It is:

- an exact target-preserving NUTS reparameterization experiment;
- natural Gamma--Beta groundwork for later split/merge proposals; and
- a way to test whether multiscale coordinates improve geometry.

It cannot answer the scientific objection to one coherent domain-wide root.
The \(G=1\) case and the grouped case must match the current target and prior
predictive distribution within Monte Carlo error.

### B. Scientifically distinct coherent regional roots

Let \(Q_g\) be the normalized nominal mass in group \(g\), let \(r_g\) be a
group scaling, and define

\[
p_g\sim\operatorname{Dirichlet}
\left(\kappa_g q_\ell/Q_g\right),\qquad
m_\ell=Q_g r_g p_{\ell\mid g}.
\]

This replaces the global root with coherent regional modes and is a new prior,
not a reparameterization. The groups should be fixed independently of the
observations for the first comparison, for example from a predeclared coarse
tree frontier or scientifically defined inner regions.

Independent mean-one roots with common variance \(v\) imply aggregate
variance \(v\sum_g Q_g^2\), which decreases as the domain is subdivided.
Subdivision must not manufacture certainty. Before real-data inference, use
prior-predictive aggregation tests to calibrate either:

- a shared common factor plus group contrasts; or
- an explicit partially pooled hierarchy with a separately identified common
  mode.

Do not include both an unconstrained global total and unconstrained group
totals. Use a one-to-one coordinate system. The partially pooled model is a
separate versioned target and should begin with a fixed-basis NUTS
implementation before entering reversible-jump proposals.

## Reporting and promotion

The final report must include:

- exact code/input/environment provenance;
- all preflight and hash results;
- scientific-coordinate target-parity results;
- initialization coordinates and seeds;
- \(K=50\) and \(K=250\) diagnostics and resource use;
- comparison with the archived fixed-basis sampler;
- a clear disposition: continuous-kernel problem, posterior-geometry problem,
  inconclusive, or numerical failure; and
- the recommended next grouped experiment.

Do not promote a posterior result if the primary diagnostics fail. A failed
screen remains useful diagnostic evidence and must be reported without writing
to production result directories.
