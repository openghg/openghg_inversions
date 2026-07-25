# Full-tiling PyMC HMC HPC test plan

## Purpose

Test whether a joint gradient transition removes the likelihood
start-separation seen with the local continuous sampler while the fixed-\(K\)
full-tiling topology remains mobile.

The experimental sweep is:

```text
one edge-flip/resolution-relocation attempt
-> one static PyMC HamiltonianMC transition
```

The HMC transition runs after every structural outcome, including rejection
and an invalid self-transition. This experiment changes the transition kernel,
not the scientific target.

Do not use its posterior summaries unless the predeclared convergence gates
pass.

## Code and environment identity

Before submission, resolve and record the pushed revision and the H2
calibration artifact:

```text
branch: codex/rjmcmc-compound-hmc
commit: $CODE_REVISION from git rev-parse HEAD after pull --ff-only
calibration file: $CALIBRATION_FILE created beneath $RUN_ROOT/calibration
calibration SHA-256: $CALIBRATION_SHA computed from the final file bytes
```

Use the exact reviewed Stage C input and archived comparison bundles:

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
export MOBILE_FIXED_COMPARISON_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_control/d1c673eb7eae4ee8bf18a15050898b4b6bb78d5c
export EARLIER_MOBILE_SCREEN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_mixing/548fa41f1fef8b8cab93a6afe8717fdb562f689f
export NUTS_REFERENCE_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_nuts/c5f908ce51eac452df2ea7f9db0cbf015fff8ef4
```

The scientific controls are fixed before calibration:

- \(K=50,\kappa=100\) and \(K=250,\kappa=500\);
- Gamma root mean one and variance 0.25, hence shape/rate \(4/4\);
- likelihood power one;
- six ordered InTEM outer coefficients with arithmetic lognormal mean/SD
  \(1/1\);
- normalized spherical-grid-cell-area nominal weights;
- input variables `fp_x_flux`, `mf`, `mf_error`, `nominal_weight`,
  `outer_design`, and fixed `YaprioriBC`;
- the fixed-\(K\) construction-history-free tiling target restricted to the
  communication component reachable from the recorded initializer.

Before calibration, verify the checksum manifests under all archived roots.
Write `preflight/target-identity.json` containing the archived and candidate
values plus a Boolean equality flag for every item above, the frozen input
digest, ordered outer labels, variable contract, initial topology digest, and
scientific target normalization statement. A false or missing equality flag
is a hard stop. Absolute fixed-\(K\) targets must not be compared across
\(K=50\) and \(K=250\).

Use one CPU per chain. Pin numerical libraries to one thread:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTENSOR_FLAGS=floatX=float64
```

Record:

- Python minor version;
- NumPy, PyMC, and PyTensor versions;
- PyTensor `floatX`;
- platform and architecture;
- Git revision and clean-worktree status;
- frozen input path, size, and SHA-256;
- calibration path and SHA-256.

Use a commit-addressed run root and never reuse it after a source change:

```bash
export DRIVER=examples/rjmcmc/full_tiling_pymc_hmc_native.py
export CODE_REVISION="$(git rev-parse HEAD)"
export RUN_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/"$CODE_REVISION"
mkdir -p "$RUN_ROOT"/{preflight,calibration,restart,smoke,matrix,analysis,report}
test -z "$(git status --porcelain)"
```

The native-PyTensor kernel is required. The PyMC NumPyro/JAX sampler bridge
must not be used for this compound transition.

## Stage H0: source and synthetic gates

Run only the experimental checks, not the repository-wide tox matrix:

```bash
pytest -q \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_io.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_native.py

uv run ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc_io.py \
  examples/rjmcmc/full_tiling_pymc_hmc_native.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_io.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_native.py

uv run pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc_io.py \
  examples/rjmcmc/full_tiling_pymc_hmc_native.py
```

Hard gates:

- the compiled transformed density equals the independent scientific target
  plus both declared chart Jacobians;
- exact static position-scale/momentum-precision semantics reach PyMC;
- accepted, rejected, and invalid structural attempts each produce one HMC
  transition;
- HMC leaves topology unchanged;
- every endpoint closes through a full scientific state rebuild;
- direct execution and awkward split continuation are byte-identical for all
  non-timing state, trace, PCG64, HMC-seed, and checkpoint fields;
- unrepresentable exponentiated coordinates are rejected by the PyMC target;
- corrupted or incompatible durable checkpoints fail closed.

Do not submit real-data jobs if any H0 gate fails.

## Stage H1: real-input dry runs

Execute Stage H2 first, because the production driver deliberately refuses
even a dry run unless it can verify the final calibration artifact. H1 is
numbered as an input/target gate but is submitted only after calibration.

Use the same frozen reconstructed PARIS May 2014 input and target as the
earlier full-tiling comparison:

- 1,382 observations from 12 sites;
- 23,424 native inner cells;
- six inferred fixed InTEM outer coefficients;
- fixed archived row-aligned boundary contribution;
- Gamma(4, rate 4) root and globally additive Dirichlet allocation prior
  with \(\kappa=2K\);
- fixed \(K=50\) and \(K=250\);
- likelihood power one;
- outer arithmetic-lognormal mean/SD \(1/1\);
- the exact normalization, variable contract, and outer-label order declared
  above.

Run no-output dry runs for the deterministic initializer and the three
random-recursive initializers declared in Stage H5 at each \(K\). It must
verify:

- input SHA-256 before and after eager loading;
- PARIS dimensions, outer labels, and closure;
- finite scientific and transformed initial targets;
- transformed-target parity;
- all PyMC continuous value variables are float64;
- resolved position-covariance diagonal, dimension, and coordinate ordering
  from the verified calibration identity; H0 separately proves that this
  exact ordering reaches PyMC's momentum precision;
- initial topology and state fingerprints plus the canonical
  `manifest_payload_sha256` that binds the complete input/model/sampler
  identity.

The repeated dry run for each random initializer must reproduce the exact
topology and state fingerprints. The four starts at each \(K\) must have four
distinct topology fingerprints. The dry run must also emit the values needed
for `preflight/target-identity.json`.

## Stage H2: bounded static-HMC calibration

Calibration is discarded and must finish before retained mobile sampling.
There is no online adaptation.

The production driver requires an already selected calibration certificate,
so do not fabricate decision statistics to run these pilots. Use a
run-specific calibration harness that calls
`sample_full_tiling_pymc_hmc()` with the same frozen-input adapter and target
construction as the driver, writes only beneath `$RUN_ROOT/calibration`, and
records its own source SHA-256, clean repository revision, input digest,
resolved target, topology fingerprint, settings, trace, and runtime identity.
Archive the harness source and hash it as one of
`source_artifact_sha256` entries in the final calibration file. The HPC agent
may debug this harness, but any harness change requires a new
content-digest-addressed calibration subroot and complete repetition of H2;
any repository source change requires a new commit-addressed run root.

### Initial metric

Use the checksum-verified fixed-basis NUTS reference draws at the same \(K\)
and target:

1. verify the NUTS bundle checksum manifest, then transform each retained
   leaf mass and fixed coefficient to the compound
   kernel's authoritative coordinates,
   \(x_i=\log m_i\) and \(y_j=\log c_j\);
2. define each robust variance as
   \([1.4826\,\operatorname{median}|z-\operatorname{median}(z)|]^2\);
3. use a single scalar leaf position scale equal to the median robust
   variance over leaf coordinates;
4. use one robust log-coordinate variance per fixed coefficient;
5. clip every resolved position scale to \([10^{-4},10^2]\);
6. set leaf/fixed cross terms to zero;
7. record that PyMC uses this position scale as momentum precision, then write
   the source artifact hashes, estimator, clipping rule, resolved diagonal,
   and coordinate-layout ID to `calibration.json`.

Do not transfer a dense leaf metric by canonical leaf position. Leaf identity
changes with topology.

### Step and path search

For each \(K\):

1. start at step size 0.1 with 10 leapfrog steps and halve the step size at
   most eight times until both pilot topologies have finite states and zero
   divergences;
2. evaluate exactly the in-range members of
   \(\{\epsilon/2,\epsilon,2\epsilon\}\times\{5,10,20\}\), where \(\epsilon\)
   is the first zero-divergence halving result;
3. use exactly 100 sweeps from the largest-nominal topology and 100 sweeps
   from random-recursive initializer seed 51051 at \(K=50\), or 51251 at
   \(K=250\), for every candidate;
4. require mean HMC acceptance between 0.6 and 0.9 for every pilot topology;
5. reject any candidate with a non-finite state, divergence, or acceptance
   outside the band;
6. choose the surviving candidate with greatest median Euclidean displacement
   in log coordinates per reported leapfrog step, breaking ties first toward
   fewer leapfrog steps and then toward the smaller step size.

The calibration search is bounded before results are inspected. Do not widen
it opportunistically. Hash the final calibration file and bind that digest
into every production manifest.

`calibration.json` must use exactly the driver-enforced v1 schema documented
by `python "$DRIVER" --help` and the driver module docstring. Its root keys
are `schema`, `calibration_id`, `fixed_k`, `input_sha256`, `target`, `kernel`,
and `evidence`; extra keys are rejected. The exact `evidence` object contains
the code revision, robust estimator, clipping bounds, two pilot
strategy/seed/sweep records, candidate grid, one decision row per candidate,
and a nonempty source-artifact SHA-256 map. The driver verifies file bytes,
all target/kernel identities, one selected requested candidate, finite
zero-divergence evidence, and both pilot acceptance means in the frozen band.
Pass the actual file path and independently computed digest to the driver;
caller-supplied ID or digest text is not sufficient.

The strict schema intentionally does not claim to prove the search procedure
from summary rows alone. Write a separate
`calibration-search-audit.json` that records the two pilot topology
fingerprints, clean-worktree check, initial epsilon and every halving result,
the derived adjacent candidate grid, source NUTS paths and verified hashes,
all raw per-topology diagnostics, the score ordering, and the declared
tie-break calculation. Include the audit file and calibration-harness source
hashes in `source_artifact_sha256`. An independent H2 analysis must recompute
these fields and emit an all-true decision before H1/H3/H4/H5; the production
driver enforces the selected candidate's identity and basic gates, while this
external audit enforces bounded-search derivation and optimal selection.
The H2 decision script must resolve every source-artifact ID to an immutable
path recorded in the audit, recompute the file SHA-256, and require equality
with `calibration.json["evidence"]["source_artifact_sha256"]`. In particular,
the map must contain `calibration-search-audit`,
`calibration-harness-source`, and the K-specific NUTS trace/checksum manifest.
Write these path/hash/equality results to `calibration-source-audit.json` and
require all true before any production-driver invocation.

## Stage H3: exact real-input restart gate

At both \(K=50\) and \(K=250\), run a short chain in two ways:

```text
direct:       N sweeps
segmented:    a sweeps + fresh-process restart + (N-a) sweeps
```

Choose \(a\) so it is not half of \(N\). Recompile PyTensor in a fresh process
for the resumed segment.

Require exact equality for:

- authoritative log leaf masses and log fixed coefficients;
- decoded scientific masses and coefficients;
- rectangle bounds;
- every posterior cache and target component;
- structural diagnostics;
- HMC accepted/divergent/acceptance/energy-error/step-size/step-count fields;
- per-sweep uint64 HMC seeds;
- final PCG64 state;
- final checkpoint arrays and metadata.

Compilation and elapsed-time fields are explicitly excluded. Reject wrong
input, manifest, \(K\), calibration digest, requested step size, leapfrog
count, metric,
schedule, coordinate layout, precision, and backend version.

## Stage H4: execution smoke

Run:

- one 100-sweep chain at \(K=50\);
- one 50-to-100-sweep chain at \(K=250\).

Report separately:

- input/setup time;
- PyTensor compilation time;
- sampling time;
- sweeps/s;
- leapfrog steps/s;
- peak RSS;
- structural validity and acceptance;
- HMC acceptance and divergences;
- energy-error quantiles;
- likelihood and root displacement.

Stop before the comparison matrix if throughput or memory makes the proposed
budget infeasible, any divergence occurs, or acceptance leaves the calibrated
band.

For H1, H3, H4, and H5, the launcher must read the calibration ID and kernel
fields from the verified JSON rather than transcribing them independently.
The common production invocation is:

```bash
pixi run -e dev --frozen python "$DRIVER" \
  --input "$FROZEN_INPUT" --output-directory "$OUTPUT" \
  --k "$K" --sweeps "$SEGMENT_SWEEPS" \
  --seed "$SAMPLER_SEED" --chain-id "k${K}-chain${CHAIN}" \
  "${INITIALIZATION_ARGUMENTS[@]}" "${RESUME_ARGUMENTS[@]}" \
  --step-size "$STEP_SIZE" --leapfrog-steps "$LEAPFROG_STEPS" \
  --leaf-position-scale "$LEAF_POSITION_SCALE" \
  --fixed-coefficient-position-scale "$FIXED_POSITION_SCALES" \
  --calibration-file "$CALIBRATION_FILE" \
  --calibration-id "$CALIBRATION_ID" \
  --calibration-sha256 "$CALIBRATION_SHA" \
  --concentration "$KAPPA" --root-variance 0.25 \
  --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --require-paris-profile \
  --input-netcdf-engine h5netcdf --netcdf-engine h5netcdf
```

Use `INITIALIZATION_ARGUMENTS=(--initialization largest-nominal)` for chain
zero. For chains one through three use
`--initialization random-recursive --initialization-seed <declared seed>`.
Use an empty `RESUME_ARGUMENTS` array for a fresh segment and
`--resume-checkpoint <certified parent bundle>/checkpoint.npz` thereafter.
The driver must reject a parent whose sibling `complete.json` or any
certificate hash is missing or stale.

## Stage H5: real-data comparison

For each \(K=50\) and \(K=250\):

- four independently seeded chains;
- exactly one largest-nominal and three random-recursive topology starts;
- 2,500 compound sweeps per chain;
- immutable 500-sweep segments;
- discard the first 500 sweeps only during analysis;
- retain every post-warmup sweep.

This is 20,000 HMC sweeps total across both \(K\) values. Do not extend the
chains until the planned analysis has been generated.

Use the exact established topology and sampler streams:

| \(K\) | chain | start | initialization seed | master sampler seed |
|---:|---:|---|---:|---:|
| 50 | 0 | largest-nominal | none | 61050 |
| 50 | 1 | random-recursive | 51051 | 61051 |
| 50 | 2 | random-recursive | 51052 | 61052 |
| 50 | 3 | random-recursive | 51053 | 61053 |
| 250 | 0 | largest-nominal | none | 61250 |
| 250 | 1 | random-recursive | 51251 | 61251 |
| 250 | 2 | random-recursive | 51252 | 61252 |
| 250 | 3 | random-recursive | 51253 | 61253 |

Every 500-sweep segment contains its boundary state. Before analysis, require
the final state/checkpoint of segment \(j\) to equal the initial state of
segment \(j+1\) exactly, concatenate all per-sweep diagnostics, and concatenate
state traces after dropping draw zero from every segment after the first.
Then remove states with global `state_sweep <= 500`; never remove 500 rows
independently from every segment.

The primary same-target control is the earlier mobile local sampler. The
fixed-basis local and fixed-basis NUTS runs are conditional diagnostics, not
direct mobile-posterior competitors. Reuse archived controls only when their
input, prior, topology/\(K\), and model manifests match exactly.
Select the `mobile` \(K=50\) and \(K=250\) cells beneath
`$MOBILE_FIXED_COMPARISON_ROOT` as the primary control because those cells
were run in the same checked matrix as the deterministic fixed-basis
comparison. Record their exact bundle paths and artifact hashes in
`analysis/control-identity.json`, and require every target-defining equality
from `preflight/target-identity.json`. The separately pinned
`$EARLIER_MOBILE_SCREEN_ROOT` is secondary continuity evidence only; do not
silently substitute it or a fixed-basis cell for the declared primary
comparison.

## Diagnostics

Common scientific coordinates:

- log likelihood;
- root total;
- six outer coefficients;
- the existing 24 predeclared native-field projections;
- native-field between-chain distances.

Sampler/structure coordinates:

- structural validity and acceptance;
- unique tilings;
- edge-flip versus relocation outcomes;
- HMC acceptance probability;
- divergence count;
- energy-error quantiles;
- BFMI where the recorded energy sequence supports it;
- step size and leapfrog count;
- scientific displacement per sweep and per leapfrog step.

Report rank-normalized split \(\hat R\), bulk and tail ESS, MCSE,
ESS/sweep, ESS/leapfrog-step, ESS/wall-hour, CPU-hour, throughput, and peak
RSS. A fixed leaf coordinate is not a valid mobile-chain convergence
coordinate because its geometric meaning changes with topology.

## Decision gates

The mobile HMC result passes only if all of the following hold:

- zero restart, hash, schema, or fail-closed errors;
- finite retained scientific coordinates and target components;
- zero HMC divergences;
- mean HMC acceptance in 0.6--0.9 for every chain;
- structural movement is accepted and more than one tiling is visited;
- \(\hat R\le 1.01\), bulk ESS at least 400, and tail ESS at least 200 for
  likelihood, root, all six outers, and all 24 projections;
- BFMI at least 0.3 when evaluable;
- no persistent start-separated likelihood band.

If the continuous diagnostics pass but topology-sensitive likelihood bands
remain separated, conclude that the continuous bottleneck was repaired but
the topology kernel still mixes poorly. If HMC itself fails, revise the
coordinate/metric/calibration scheme before drawing conclusions about the
topology model.

No output is written to `PARIS_inversions` unless all scientific convergence
gates pass and a separate promotion decision is made.
