# Topology-conditioned HMC HPC test plan

## 2026-07-26 handover

The minimum repaired source revision is
`e6199150e680d43e6e3c1388db45773c5337802a`. Its attempted BP1 run was
interrupted when VPN access ended and supplies no D0 or D1 conclusion. Treat
the following as incomplete evidence:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/e619915
/group/chem/acrg/brendan_for_codex/rjmcmc_h2d_worker_e619915
```

Commit `7f7b1509bf032d04c9839ec9fa4d7be69b03e1ab` failed D0 and must
never be resumed. Its evidence remains at
`/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/7f7b150`.

Use the later full `origin/codex/rjmcmc-topology-conditioned-hmc` SHA,
including handover documentation, as the candidate. Start from D0 in a fresh
full-SHA run root. See
[`rjmcmc_bp1_handover.md`](rjmcmc_bp1_handover.md).

## Status

This is the executable follow-up to the certified H2c hard stop. It tests the
topology-conditioned, static Euclidean precision described in
[`rjmcmc_topology_conditioned_hmc_next_phase.md`](rjmcmc_topology_conditioned_hmc_next_phase.md).

Do not reuse an H2, H2b, or H2c calibration. The metric semantics, checkpoint
schema, and calibration evidence are intentionally incompatible.

Commit `7f7b150` also stopped at D0 and must not be resumed. Its
log/exp-rebuilt physical split proposal admitted forward paths whose reverse
fraction rounded to a binary64 endpoint. The replacement schedule uses an
exact log-coordinate involution and has a new identity. Preserve the failed
`7f7b150` run root as evidence; use a new commit-addressed root below.

Fill in the pushed candidate revision before launch:

```text
branch: codex/rjmcmc-topology-conditioned-hmc
candidate revision: <git rev-parse HEAD>
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/<candidate revision>
```

Create an immutable clean worktree and frozen development environment:

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
git fetch origin
export CANDIDATE_REVISION="$(git rev-parse origin/codex/rjmcmc-topology-conditioned-hmc)"
export CANDIDATE_WORKTREE="/group/chem/acrg/brendan_for_codex/rjmcmc_candidate_${CANDIDATE_REVISION}"
git worktree add --detach "${CANDIDATE_WORKTREE}" "${CANDIDATE_REVISION}"
cd "${CANDIDATE_WORKTREE}"
test "$(git rev-parse HEAD)" = "${CANDIDATE_REVISION}"
test -z "$(git status --porcelain)"
pixi install -e dev --frozen
```

The agent may repair launch, analysis, or reporting scripts beneath the
commit-addressed run root. Preserve every failed artifact. A repository source
change requires a new pushed commit and a new run root. Do not weaken an exact
scientific, replay, checkpoint, or held-out-topology gate to continue.
Record and hash every run-root script and command. D1--D4 prose defines gates,
not a hidden executable driver; a stage passes only when a committed or
run-root script produces the required machine-readable evidence.

Run only the experimental tests. Do not run the repository-wide tox matrix.
Nothing from this experiment may be written to `PARIS_inversions`.

## Frozen scientific identity

Use the same reconstructed PARIS May 2014 target as the earlier full-tiling
experiments:

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
```

The immutable target controls are:

- 1,382 observations and 23,424 inner-domain cells;
- six ordered, inferred InTEM coefficients;
- fixed archived row-aligned boundary contribution;
- fixed \(K=50\) or \(K=250\);
- Gamma root shape/rate \(4/4\);
- globally additive Dirichlet allocation with \(\kappa=2K\);
- arithmetic-lognormal outer mean/SD \(1/1\);
- likelihood power one and the archived fixed diagonal errors; and
- the existing fixed-\(K\), construction-history-free tiling target.

Record the exact candidate and archived comparison identities:

```bash
export H2C_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/7a1a1cc673a4b6a6ce0ed7b5123494ebd205b467
export NUTS_REFERENCE_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_nuts/c5f908ce51eac452df2ea7f9db0cbf015fff8ef4
export FIXED_CONTROL_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_fixed_basis_control/d1c673eb7eae4ee8bf18a15050898b4b6bb78d5c
export MOBILE_CONTROL_ROOT=/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_mixing/548fa41f1fef8b8cab93a6afe8717fdb562f689f
```

Verify every available archived SHA-256 manifest before using a comparison.

Use one numerical thread per process:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTENSOR_FLAGS=floatX=float64
```

The BP1 login node may be used for bounded, sequential preflight and
calibration work only while it is quiet. Check load and available memory
first, keep aggregate resident memory below 200 GB, and do not start a
multi-chain matrix there. Use Slurm for the retained multi-chain screen. Stop
remote work if BP1 becomes unreachable.

## Stage D0: source and exact synthetic gates

Run:

```bash
pixi run -e dev --frozen pytest -q \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_io.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_native.py \
  tests/experimental/rjmcmc/test_aggregation_error.py \
  tests/experimental/rjmcmc/test_aggregation_error_low_rank.py

pixi run -e dev --frozen ruff check \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc_io.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error_low_rank.py \
  examples/rjmcmc/full_tiling_pymc_hmc_native.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_io.py \
  tests/experimental/rjmcmc/test_full_tiling_pymc_hmc_native.py \
  tests/experimental/rjmcmc/test_aggregation_error.py \
  tests/experimental/rjmcmc/test_aggregation_error_low_rank.py

pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc.py \
  openghg_inversions/experimental/rjmcmc/full_tiling_pymc_hmc_io.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error_low_rank.py \
  examples/rjmcmc/full_tiling_pymc_hmc_native.py
```

Hard gates:

- the transformed PyMC target still equals the independent scientific target
  plus the declared log-coordinate Jacobians;
- the reference precision equals the independent prior-plus-whitened-Jacobian
  oracle;
- leaf permutation produces the corresponding precision permutation, while
  fixed-coefficient order remains unchanged;
- the precision is finite, exactly symmetric, strictly positive definite, and
  deterministic;
- PyMC receives the precision with `is_cov=False`, so velocity and kinetic
  energy agree with solves against that precision;
- topology data, potential, and leapfrog integrator change atomically after
  every accepted, rejected, or invalid structural outcome;
- a valid structural candidate is an exact involutive permutation of the
  authoritative log-mass coordinates passed to HMC, while geometrically
  unchanged leaves retain their coordinate bits;
- edge flips transfer the two old child coordinates to the two new
  perpendicular children, and relocations exchange the old destination
  coordinate with the merged-parent coordinate while transferring the old
  merge pair to the new destination children;
- the structural MH ratio uses the exact transformed PyMC target plus
  reverse-minus-forward discrete selection probability, with no Beta
  auxiliary draw and no rounded physical split;
- a bounded structural oracle traverses at least 10,000 valid moves at each
  tested \(K\), including deliberately extreme log-mass contrasts, and
  requires every recorded reverse to recover exact topology, coordinate bits,
  decoded masses, RNG accounting, and the negated forward log ratio;
- unequal forward/reverse merge-catalogue sizes and every invalid relocation
  catalogue case are represented in that oracle, and no unsupported reverse
  path may be skipped;
- metric construction consumes no sampler RNG and depends on no current
  continuous coordinates;
- direct and awkward-boundary continuation are exact;
- checkpoint loading reconstructs and verifies the topology precision hash;
- retired checkpoint and calibration schemas fail closed; and
- the exact aggregation oracle shows coarse/fine evidence equality while its
  nominal-fill sentinel does not.

Do not proceed if any gate fails.

The aggregation-oracle checks are whole-branch source preflight. Their failure
blocks candidate promotion, but is not evidence that the H2d metric
calibration itself failed; report the failing track by name.

## Stage D1: frozen-input precision audit

Reproduce the four audited H2c post-structure boundaries and their exact
topology fingerprints. For each boundary:

1. construct the topology-reference precision twice in independent processes;
2. require byte-identical float64 arrays and hashes;
3. record dimension, minimum/maximum eigenvalue, condition number, Cholesky
   success, block norms, build time, factorization time, and peak RSS;
4. verify the reference shares sum to one and the reference root and fixed
   coefficients equal their declared arithmetic prior means; and
5. compare with the archived Phase 0 curvature report.

Also time 100 sequential full rebuilds at each \(K\). Report medians and
95th percentiles; do not hide warm-up or first-build costs. This stage has no
sampling and may run on an idle login node.

The precision hash is intentionally exact and includes the binary64
`J.T @ J` result. Verify one save/load/continue boundary on a different
intended BP1 compute node with the same pinned environment. A hash mismatch is
a portability hard stop, even if matrices agree approximately; report the
BLAS implementation and CPU model before deciding whether a later checkpoint
schema should persist the validated precision bytes.

The precision builder is a hard failure if any reconstructed hash changes
between processes, a matrix is not SPD, or the builder uses more than 10 GiB
peak RSS. A slow but correct full rebuild is not a hard failure; record it as
the baseline for a later incremental implementation.

## Stage D2: predeclared discarded calibration

There is no online adaptation in retained sampling. Every calibration sweep is
discarded and uses the production compound schedule:

```text
one fixed-K structural attempt
-> install the resulting topology's deterministic precision
-> one non-adapting PyMC HamiltonianMC transition
```

Accepted, rejected, and invalid structural outcomes all receive the HMC
transition.

### Frozen topology and master-stream seeds

The development roles are:

| \(K\) | role | initializer | topology seed | master PCG64 seed |
|---:|---|---|---:|---:|
| 50 | development-nominal | largest nominal | n/a | 73050 |
| 50 | development-a | random recursive | 42050 | 73051 |
| 50 | development-b | random recursive | 42051 | 73052 |
| 250 | development-nominal | largest nominal | n/a | 73250 |
| 250 | development-a | random recursive | 42250 | 73251 |
| 250 | development-b | random recursive | 42251 | 73252 |

The untouched validation roles are:

| \(K\) | role | initializer | topology seed | master PCG64 seed |
|---:|---|---|---:|---:|
| 50 | held-out-a | random recursive | 42052 | 74050 |
| 50 | held-out-b | random recursive | 42053 | 74051 |
| 250 | held-out-a | random recursive | 42252 | 74250 |
| 250 | held-out-b | random recursive | 42253 | 74251 |

Freeze and publish the resulting initial topology hashes before inspecting any
HMC result. The two deterministic largest-nominal geometries are intentionally
known development cases and may match an earlier development topology. Require
every random-recursive development and held-out hash to be distinct from the
H2/H2b/H2c calibration topologies, from each other, and from later
retained-production starts.

### Candidate grid and selection

Use the same requested grid at both \(K\) values:

```text
step size epsilon: 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75
leapfrog steps L: 3, 5, 8
```

For every candidate, run 200 discarded sweeps from each of the three
development roles, restarting the declared master stream for each candidate.
Record acceptance, divergences, non-finite endpoints, energy error, structural
acceptance, mean Mahalanobis squared displacement per gradient evaluation,
precision condition-number strata, precision-build time, gradient time, and
total throughput. Also report fixed-window and lower-tail acceptance/energy
diagnostics so a good overall mean cannot hide a catastrophic topology-local
interval.

A candidate is development-admissible only if every development role has:

- finite scientific and transformed endpoints;
- zero divergences;
- mean HMC acceptance in \([0.60,0.95]\); and
- at least one accepted nonzero continuous displacement.

Among admissible candidates, maximize the minimum, across the three
development roles, of mean Mahalanobis squared displacement per gradient
evaluation. Break ties by smaller maximum absolute deviation of the
three acceptance rates from 0.75, then higher minimum throughput, then smaller
step size. Reject an unresolved exact tie rather than depending on JSON row
order.

The calibration certificate must embed the complete ordered 21-candidate grid
and all 63 development trajectories. The driver must recompute admissibility
and the selection rule from those rows; a digest of an external grid or an
unverified claimed winner is not sufficient.

Lock the selected candidate and its complete evidence hash. Rerun it for 500
discarded sweeps from all three development roles with fresh validation
master seeds obtained by adding 10,000 to the table values, and for 500
discarded sweeps from both held-out roles using the seeds in the validation
table. Every one of the five trajectories must satisfy the same gates.

Failure of either held-out topology is a certified H2d hard stop. Do not add
an unplanned finer grid or average across topology-specific failures.

## Stage D3: durability and short real-input smoke

After D2 passes:

- run direct and awkward split continuations that cross a non-cycle segment
  boundary and require exact state, trace, PCG64, HMC-seed, metric-hash, and
  checkpoint replay;
- test accepted, rejected, and invalid structure outcomes;
- inject corrupt state, topology, metric-builder identity, reference identity,
  precision hash, runtime identity, and manifest values and require
  fail-closed loading;
- publish checkpoints at every smoke segment;
- independently reopen every artifact and verify recorded SHA-256 digests; and
- write `complete.json` last.

Run 2,000 mobile compound sweeps from one new random-recursive start at each
\(K\). Report precision-build, factorization, topology, gradient, checkpoint,
and diagnostics time separately, plus peak RSS. This is a performance and
durability smoke, not a convergence run.

If full precision reconstruction consumes more than half of total wall time,
retain the correct full-rebuild result and propose an exactly equivalent
incremental row/column update. Do not silently replace the metric with a
diagonal or low-rank approximation.

## Stage D4: retained start-sensitivity screen

Only after D0--D3 pass, submit four one-CPU Slurm chains at each \(K\), with
four newly seeded random-recursive initial tilings. The deterministic
largest-nominal geometry has already served as a development case and must not
be relabelled as retained validation. Use new topology and sampler seeds
recorded before submission and exclude every calibration topology hash.

The initial bounded budget is:

```text
K=50:  12,000 compound sweeps per chain
K=250:  8,000 compound sweeps per chain
warmup excluded from diagnostics: first 20%
retention: every compound sweep after warmup; no thinning for convenience
checkpoint segments: no more than 2,000 sweeps
```

The matrix is a screen, not a production posterior. Diagnose at minimum:

- log likelihood, log target, root total, six fixed coefficients, and the 24
  predeclared native-field projections;
- split rank-normalized \(\hat R\), bulk ESS, tail ESS, and ESS/wall-hour;
- between-start likelihood-band overlap in both halves;
- native-field distance contraction;
- topology hashes visited, structural acceptance, return proxies, and
  continuous HMC acceptance/divergences; and
- time spent in precision construction versus HMC gradients.

The main success gate is removal of persistent likelihood start separation:

- log-likelihood \(\hat R\leq1.05\);
- log-likelihood bulk and tail ESS at least 100;
- overlapping second-half likelihood bands; and
- zero HMC divergences.

Report all common-coordinate failures even if the likelihood gate passes.
Spatial summaries remain diagnostic unless every predeclared common
projection also meets its declared convergence gate.

Compare with fixed-basis NUTS, fixed-basis local sampling, the earlier mobile
local sampler, and H2c. Do not compare raw sweep counts without reporting
structural opportunities, gradient evaluations, and wall time.

## Disposition

- If D2 fails across held-out topologies, the static topology-reference metric
  has exhausted its bounded test. Localize failure to metric conditioning,
  within-topology position curvature, or topology/continuous landing before
  choosing a richer method.
- If D2 passes but D4 retains topology-dependent likelihood separation,
  prototype the source-HMC/structural-map/destination-HMC joint acceptance
  construction using the same verified metric.
- Use position-dependent/Riemannian HMC only if repeated positions within the
  same frozen topology show materially different curvature or energy error.
- In parallel, retain the exact aggregation-error oracle. If all partitions
  are meant to represent one common native model, the exact marginal limit
  must leave the structural posterior equal to its prior. An NLE may
  approximate that normalized marginal likelihood only after the exact small
  oracle and normalization gates pass.

## Required report

Publish a readable report, machine-readable summary, job inventory, source and
environment record, and complete SHA-256 inventory. The report must identify
the worst diagnostic by name, not only its value. It must distinguish hard
failures, warnings, and stages withheld by an earlier gate.
