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

Before submission, resolve and record the pushed revision and the H2c
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

### Completed H2/H2b history and H2c boundary

The first H2 attempt stopped at boundary draw 0, before any topology or HMC
transition. The initializer's physical leaf masses failed the exact
`exp(log(m)) == m` replay audit by 3--7 ULP. This was a real representation
defect, not evidence about HMC calibration, and the exact audit was not
relaxed. Commit `fe9e546` corrected it.

The corrected diagonal H2 selected a valid \(K=50\) calibration but found no
candidate satisfying both development topologies at \(K=250\). The separately
predeclared H2b interior grid selected
\(\epsilon=0.08409,L=5\); its untouched held-out random topology had
acceptance 0.031 and 42 divergences over 500 discarded sweeps. H2b therefore
stopped without a \(K=250\) calibration or production output.

The immutable evidence is under
`/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/fe9e546ab57a6b0ff852057e0e6afa13725a5419`,
including `report/H2B_RESULTS.md` and
`calibration/H2B_HARD_STOP.json`. Preserve it unchanged.

H2c changes the metric contract. The earlier \(K=50\) calibration remains
valid evidence for commit `fe9e546`, but it is a version-1 diagonal
calibration and cannot be used with the version-2 total/contrast driver.
Both \(K\) values require new H2c calibration certificates before H1 or
H3--H5.

The agent may diagnose and repair run harnesses, Slurm scripts, module loads,
and analysis scripts beneath the new run root. Preserve failed artifacts for
provenance. A calibration-harness change requires a new
content-digest-addressed calibration subroot and complete H2c repetition. A
repository-source change must be committed and pushed, and requires a new
commit-addressed run root. Do not work around a gate by editing an artifact,
loosening an exact comparison, or continuing from a failed checkpoint.

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
- the leaf block has the declared total and contrast eigenvalues, is
  symmetric positive definite, and is invariant under arbitrary leaf
  permutations;
- the fixed block is exactly diagonal and all leaf/fixed cross terms are
  exactly zero;
- accepted, rejected, and invalid structural attempts each produce one HMC
  transition;
- HMC leaves topology unchanged;
- every endpoint closes through a full scientific state rebuild;
- fresh draw 0 is the canonical scientific boundary: physical leaf masses
  and fixed coefficients equal the exponentials of the retained
  authoritative log coordinates bit for bit, the manifest initial-state hash
  equals the segment-start hash, and the draw-0 target equals both recorded
  initial targets;
- direct execution and awkward split continuation are byte-identical for all
  non-timing state, trace, PCG64, HMC-seed, and checkpoint fields;
- unrepresentable exponentiated coordinates are rejected by the PyMC target;
- corrupted or incompatible durable checkpoints fail closed.

Do not submit real-data jobs if any H0 gate fails.

## Stage H1: real-input dry runs

Execute Stage H2c first, because the production driver deliberately refuses
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

Run no-output dry runs for the four random-recursive initializers declared in
Stage H5 at each \(K\). It must verify:

- input SHA-256 before and after eager loading;
- PARIS dimensions, outer labels, and closure;
- finite scientific and transformed initial targets;
- transformed-target parity;
- all PyMC continuous value variables are float64;
- resolved position-covariance matrix, dimension, and coordinate ordering
  from the verified calibration identity; H0 separately proves that this
  exact ordering reaches PyMC's momentum precision;
- initial topology and state fingerprints plus the canonical
  `manifest_payload_sha256` that binds the complete input/model/sampler
  identity.

The repeated dry run for each random initializer must reproduce the exact
topology and state fingerprints. The four starts at each \(K\) must have four
distinct topology fingerprints, and none may equal the calibration
metric-source, development, or held-out topology hashes. The dry run must
also emit the values needed for `preflight/target-identity.json`.

## Stage H2c: bounded total/contrast static-HMC calibration

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
content-digest-addressed calibration subroot and complete repetition of H2c;
any repository source change requires a new commit-addressed run root.

### Initial metric

Use only checksum-verified, post-warmup fixed-basis NUTS draws at the same
\(K\) and target. The archived NUTS reference uses the deterministic
largest-nominal topology; record its canonical topology hash as the
calibration metric source, and exclude that topology from H5. Let
\(X_{di}=\log m_{di}\) and
\(Y_{dj}=\log c_{dj}\). Define the normalized common coordinate and centered
contrasts by

\[
z_d=\frac{1}{\sqrt K}\sum_i X_{di},
\qquad
R_{di}=X_{di}-\frac1K\sum_l X_{dl}.
\]

For any scalar series \(a\), define

\[
V_{\rm MAD}(a)
=
\left[1.4826\,\operatorname{median}
|a-\operatorname{median}(a)|\right]^2.
\]

Construct the frozen metric as follows:

1. verify the NUTS bundle checksum manifest, then transform each retained
   masses and fixed coefficients to \(X\) and \(Y\);
2. set
   \(g_{\rm total}=V_{\rm MAD}(z)\);
3. set
   \[
   g_{\rm contrast}
   =
   \frac{\operatorname{median}_i V_{\rm MAD}(R_{\cdot i})}
        {1-1/K};
   \]
4. set each fixed scale to \(V_{\rm MAD}(Y_{\cdot j})\);
5. require every raw estimate to be finite and strictly positive before
   clipping it independently to \([10^{-4},10^2]\);
6. require
   \(\max(g_{\rm total},g_{\rm contrast})/
     \min(g_{\rm total},g_{\rm contrast})\le10^4\);
7. assemble
   \(G_{\rm leaf}=g_{\rm contrast}P_\perp+g_{\rm total}P_1\),
   retain the fixed diagonal, and set leaf/fixed cross terms to zero;
8. record raw and clipped values, clipping flags, source hashes, estimator ID
   `normalized_common_and_centered_contrast_scaled_mad_v1`, normalized
   direction, \(1-1/K\) correction, and coordinate-layout ID.

This dense leaf block is permitted because it is invariant under every leaf
permutation. Do not transfer an arbitrary leaf-by-leaf covariance learned in
one fixed basis.

### Step and path search

For each \(K\):

1. start at step size 0.1 with 10 leapfrog steps and halve the step size at
   most eight times until both pilot topologies have finite states and zero
   divergences;
2. evaluate exactly the in-range members of
   \(\{\epsilon/2,\epsilon,2\epsilon\}\times\{5,10,20\}\), where \(\epsilon\)
   is the first zero-divergence halving result;
3. use exactly 200 sweeps from each of two random-recursive development
   topologies for every candidate, with the frozen topology/master-PCG64
   seeds in the table below;
4. require mean HMC acceptance between 0.6 and 0.9 for every pilot topology;
5. reject any candidate with a non-finite state, divergence, or acceptance
   outside the band;
6. for each sweep, concatenate
   `log_leaf_mass[post-HMC] - hmc_start_log_leaf_mass` and
   `log_fixed_coefficient[post-HMC] -
   hmc_start_log_fixed_coefficient`, then take its Euclidean norm; rejected
   HMC transitions therefore contribute zero and accepted structural motion
   contributes nothing;
7. pool the 400 HMC-only displacement values from the two equally sized
   development runs, divide each by the candidate's reported leapfrog count,
   and choose the surviving candidate with greatest binary64 median;
8. break an exactly equal score first toward fewer leapfrog steps and then
   toward the smaller step size.

The development identities are:

| \(K\) | role | topology seed | master PCG64 seed |
|---:|---|---:|---:|
| 50 | development-a | 41050 | 71050 |
| 50 | development-b | 41051 | 71051 |
| 250 | development-a | 41250 | 71250 |
| 250 | development-b | 41251 | 71251 |

For each role, restart the same master PCG64 seed for every candidate. This is
the frozen common-random-number policy for candidate comparison. Do not
advance one shared stream sequentially across candidates, and do not choose
new seeds after inspecting results.

After selecting one candidate, freeze its metric and HMC controls and write a
selection-lock hash before running validation. Validate exactly 500 discarded
sweeps from each frozen topology/master-PCG64 pair:

| \(K\) | role | topology seed | validation master PCG64 seed |
|---:|---|---:|---:|
| 50 | development-a | 41050 | 72050 |
| 50 | development-b | 41051 | 72051 |
| 50 | held-out | 41052 | 72052 |
| 250 | development-a | 41250 | 72250 |
| 250 | development-b | 41251 | 72251 |
| 250 | held-out | 41252 | 72252 |

All three must be finite, have zero divergences, and have mean HMC acceptance
in \([0.6,0.9]\). Failure of the held-out topology is a hard stop: do not use
its result to retune the metric, step size, path length, estimator, or gates.
A new attempt requires a new predeclared metric or parameterization and a new
content-addressed calibration root.

All calibration topologies and master streams are disjoint from the H5
retained-production starts. The deterministic largest-nominal topology is the
fixed-basis NUTS metric source and is therefore calibration-exposed; it is not
used by H5.

The calibration search is bounded before results are inspected. Do not widen
it opportunistically. Hash the final calibration file and bind that digest
into every production manifest.

`calibration.json` must use exactly the driver-enforced v2 schema documented
by `python "$DRIVER" --help` and the driver module docstring. Its root keys
are `schema`, `calibration_id`, `fixed_k`, `input_sha256`, `target`, `kernel`,
and `evidence`; extra keys are rejected. The exact evidence binds the code
revision, robust and leaf-metric estimator IDs, clipping bounds, two
development initializer records including topology and master-PCG64 seeds,
candidate grid, one decision row per candidate, the three-case 500-sweep
selected validation with separate master seeds, and a nonempty source-artifact
SHA-256 map. It also binds four distinct excluded topology hashes: the
fixed-basis NUTS metric source, development-a, development-b, and held-out.
The driver verifies file bytes, all target/kernel identities, role-specific
seeds, the selected requested candidate, development gates, and all three
validation gates. It rejects retained-production random-recursive starts that
reuse a calibration topology seed or whose actual canonical topology hash
collides with any excluded hash. Pass the actual file path and independently
computed digest to the driver; caller-supplied ID or digest text is not
sufficient.

The strict schema intentionally does not claim to prove metric derivation or
the search procedure from summary rows alone. Write a separate
`calibration-search-audit.json` that records the metric projections and raw
estimates, clipping and condition calculation, development and held-out
topology fingerprints, topology/master seeds, the common-random-number
policy, disjoint production/calibration topology sets, clean-worktree check,
initial epsilon and every halving result, the derived adjacent candidate grid,
source NUTS paths and verified hashes, all raw diagnostics, HMC-start and
post-HMC coordinates used by the displacement score, the selection-lock hash,
score pooling/order, and tie-break calculation. Include the audit file and
calibration-harness source hashes in `source_artifact_sha256`. An independent
H2c analysis must recompute these fields and emit an all-true decision before
H1/H3/H4/H5; the production driver enforces the selected candidate and basic
gates, while the external audit enforces metric derivation, bounded search,
held-out isolation, common random numbers, and optimal selection. The H2c
decision script must
resolve every source-artifact ID to an immutable
path recorded in the audit, recompute the file SHA-256, and require equality
with `calibration.json["evidence"]["source_artifact_sha256"]`. In particular,
the map must contain `calibration-search-audit`,
`calibration-harness-source`, and the K-specific NUTS trace/checksum manifest.
Write these path/hash/equality results to `calibration-source-audit.json` and
require all true before any production-driver invocation.

The audit must reconstruct and hash the fixed-basis NUTS source topology and
all three mobile calibration topologies. Those exact four digests populate
`excluded_production_topology_sha256`; seed inequality alone is not accepted
as proof of topology disjointness.

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
- post-structure/pre-HMC authoritative log coordinates;
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
  --leaf-contrast-position-scale "$LEAF_CONTRAST_POSITION_SCALE" \
  --leaf-total-position-scale "$LEAF_TOTAL_POSITION_SCALE" \
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

For every chain use
`INITIALIZATION_ARGUMENTS=(--initialization random-recursive
--initialization-seed <declared seed>)`.
Use an empty `RESUME_ARGUMENTS` array for a fresh segment and
`--resume-checkpoint <certified parent bundle>/checkpoint.npz` thereafter.
The driver must reject a parent whose sibling `complete.json` or any
certificate hash is missing or stale.

## Stage H5: real-data comparison

For each \(K=50\) and \(K=250\):

- four independently seeded chains;
- four random-recursive topology starts disjoint from calibration;
- 2,500 compound sweeps per chain;
- immutable 500-sweep segments;
- discard the first 500 sweeps only during analysis;
- retain every post-warmup sweep.

This is 20,000 HMC sweeps total across both \(K\) values. Do not extend the
chains until the planned analysis has been generated.

Use the exact established topology and sampler streams:

| \(K\) | chain | start | initialization seed | master sampler seed |
|---:|---:|---|---:|---:|
| 50 | 0 | random-recursive | 51050 | 61050 |
| 50 | 1 | random-recursive | 51051 | 61051 |
| 50 | 2 | random-recursive | 51052 | 61052 |
| 50 | 3 | random-recursive | 51053 | 61053 |
| 250 | 0 | random-recursive | 51250 | 61250 |
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
