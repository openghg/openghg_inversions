# BP1 exploratory plan: resolution-SMC aggregation likelihood

## Purpose

This plan asks whether progressive Gamma--Beta refinement with martingale
Gaussian look-ahead gives a more stable fixed-root marginal-likelihood
estimator than drawing complete allocations from the prior.

It is an exploratory numerical programme, not a production certification
protocol.  Correctness, target identity, normalization, replay, and protected
data boundaries are hard requirements.  Performance thresholds are
diagnostic.  A poor first configuration should be debugged and documented,
not converted prematurely into a terminal architecture stop.

Background and equations are in
[`rjmcmc_resolution_smc_martingale.md`](rjmcmc_resolution_smc_martingale.md).

## Branch and execution model

Planning branch:

```text
codex/rjmcmc-resolution-smc-plan
```

The HPC agent may implement on this branch or create a child branch named
`codex/rjmcmc-resolution-smc-<short-description>`.  Before any Slurm launch:

1. commit and push every source, test, driver, and frozen configuration;
2. resolve the complete 40-character remote SHA;
3. create a clean detached worktree at that SHA; and
4. create a new run root:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc/<full-SHA>
```

Every scientific source change gets a new commit and run root.  Iteration is
allowed; retain a ledger linking every attempt to its SHA, commands, Slurm
jobs, result, and disposition.

Do not run the repository-wide tox matrix.  Run focused experimental tests,
focused Ruff, formatting, and focused Pyright.

## Frozen PARIS input

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/
dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/
paris_may_2014_gamma_beta_native.nc

schema:
paris-may-2014-gamma-beta-native-v1

SHA-256:
24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044

shape:
1382 observations x 23,424 native cells
```

For the PARIS comparison use the existing observation-blind G4 calibration:

```text
native concentration eta: 528.618161317525
root variance: 0.022861001527515423
root CV: 0.15119855001790006
```

Reuse the G4 observation-blind validation catalogue and authenticated
spectrum where possible.  The preserved G4 source run is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/
189427d5ccca9187618ab8be1cc2cf7d7105b216
```

Do not read realized PARIS `mf` during R0--R4.  Do not access any protected
catalogue and do not write `PARIS_inversions`.

## Intended implementation surface

Add an opt-in module:

```text
openghg_inversions/experimental/rjmcmc/
aggregation_error_resolution_smc.py
```

and focused tests:

```text
tests/experimental/rjmcmc/
test_aggregation_error_resolution_smc.py
```

Do not change existing experimental serialization or public exports.

The minimum reusable types should represent:

- immutable tree nodes and stable scientific cell identities;
- a deterministic parent-first resolution schedule;
- a particle frontier containing revealed masses and accumulated
  observation prediction;
- one Gaussian-guide specification;
- one SMC configuration with particle count, ESS threshold, seed, and
  resampling policy;
- per-level diagnostics;
- an exact restart boundary; and
- a normalizing-constant result with linear- and log-scale representations.

Use NumPy/SciPy first.  No new dependency is needed for the baseline.

## Required numerical contracts

1. Float64 throughout.
2. NumPy PCG64 with recorded independent streams for propagation and
   resampling.
3. Stable log weights and `logsumexp`.
4. Multinomial resampling for the first exact audit because its unbiasedness
   and replay contract are simplest.  Stratified or systematic resampling may
   be a later, separately labelled variance-reduction comparison.
5. No resampling when ESS is above the configured threshold.
6. Prefix masses remain positive and conserve their parent mass.
7. Native terminal allocations sum to the retained root mass within a
   scale-aware float64 tolerance.
8. The terminal likelihood is the ordinary normalized
   \(\mathcal N(y;b+HX,D)\), not a closure.
9. Intermediate Gaussian closure may guide particles but may not be retained
   as an unlabelled exact result.
10. Checkpoints contain the current level, particles, log weights,
    normalizing-constant accumulator, ancestry needed for diagnostics, RNG
    states, schedule identity, input identity, and source identity.
11. With resampling disabled and identical complete allocation paths, prior
    proposal SMC must reproduce direct IID likelihood averaging.
12. Swapping child identities and applying the matching scientific
    permutation must leave the calculation equivariant.

## R0: focused preflight

Run on a quiet login node:

1. authenticate the clean full-SHA worktree;
2. link/install the pinned Pixi environment;
3. run the existing aggregation-error and Gamma--Beta tests touched by the
   implementation;
4. run the new focused tests;
5. run focused Ruff format/check and Pyright;
6. run one two-cell SMC smoke;
7. restart that smoke at every resolution boundary and require exact replay;
8. deliberately corrupt schedule, tree, input, seed, and particle metadata
   and require fail-closed loading; and
9. publish an R0 summary after all checks complete.

Hard stop only for target, conservation, normalization, replay, non-finite,
or provenance failures.

## R1: exact tiny-oracle matrix

### R1a: smallest decisive experiment

Run this before launching the full matrix.  It is the recommended first
scientific experiment because it can distinguish a correctness failure from
the hoped-for variance reduction while every likelihood remains independently
checkable.

Use exactly three cases:

1. the two-cell near-Gaussian case;
2. one skewed two-cell case from the G1 screen; and
3. one boundary-heavy four-cell case with the row/column observation
   operator, using an independently converged quadrature oracle.

For each case compare:

1. direct IID complete allocations;
2. prior-proposal SMC with resampling disabled, using exactly the same
   allocation paths as the IID calculation;
3. bootstrap resolution-SMC with multinomial resampling at ESS fraction
   \(0.5\); and
4. bootstrap resolution-SMC with resampling at every nonterminal refinement,
   retained mainly as a stress test.

Use particle counts \(64,256,1024,4096\) and 64 independent replicates.  Freeze
the tree, reveal schedule, input, and seed derivation before launching.  For
the four-cell case repeat the SMC calculation under one compatible alternative
tree chart.

The first correctness requirements are:

- direct IID and no-resampling SMC are pathwise identical;
- the replicate mean of each non-negative likelihood estimator is consistent
  with the oracle on the linear scale;
- normalization, mass conservation, exact replay, and child-swap equivariance
  hold; and
- compatible charts agree in expectation within replicate uncertainty.

The central performance comparison is relative variance times measured cost,
not error in one log-likelihood estimate.  A useful signal is a reproducible
twofold reduction relative to IID for the boundary-heavy case without loss of
oracle agreement.  This is a target, not a hard gate.  If SMC is merely equal
or worse, complete and report R1a before changing the guide.

Only after R1a is understood should the agent expand to the cases, estimators,
and orderings below.

### Cases

Reuse or reconstruct the existing exact targets:

| Cells | Allocation shape | Observation geometry | Oracle |
|---:|---|---|---|
| 2 | near-Gaussian Beta | visible child contrast | high-order Gauss--Jacobi |
| 2 | skewed Beta | visible child contrast | high-order Gauss--Jacobi |
| 2 | boundary-heavy Beta | visible child contrast | independently converged adaptive quadrature and Gauss--Jacobi |
| 4 | balanced Dirichlet | row/column and generic contrasts | tensor Gauss--Jacobi |
| 4 | boundary-heavy Dirichlet | row/column and generic contrasts | increasing-order tensor Gauss--Jacobi |
| 16 | balanced and skewed | fixed synthetic \(H\) | replicated large-IID reference, not labelled exact |

Do not reuse an “exact” boundary-heavy quadrature order without demonstrating
order convergence for this experiment.

### Estimators

Compare:

1. IID complete-allocation Monte Carlo;
2. existing blocked scrambled Sobol where supported;
3. bootstrap resolution-SMC with breadth-first levels; and
4. bootstrap resolution-SMC with fixed observation-energy ordering.

Do not add local guidance until the bootstrap result is reported.

### Frozen exploratory grid

```text
particle/sample counts: 64, 256, 1024, 4096
independent replicates: 32 after the 64-replicate R1a screen
resampling ESS fractions: 0.25, 0.50
tree charts: balanced, valid chain, row-first, column-first where applicable
```

The sample counts may be reduced for an expensive tensor oracle, but record
the matched work actually used.  Do not compare methods using nominal sample
count alone: report Beta draws, forward updates, likelihood evaluations, wall
time, and peak RSS.

### Required outputs

For each cell:

- mean and standard error of \(\widehat Z\) on the linear scale;
- relative bias and relative RMSE against the oracle;
- median and quantiles of log-likelihood error;
- between-replicate variance before taking logs;
- ESS and coefficient of variation of incremental weights at every level;
- maximum normalized weight, Shannon perplexity, and unique ancestor count;
- number and location of resampling events;
- variance and cost of
  \(\widetilde L_\ell-\widetilde L_{\ell-1}\);
- closure error at selected prefixes where exact \(L_\ell\) is available;
- tree/chart differences with Monte Carlo uncertainty; and
- exact replay and checkpoint identities.

### Interpretation

The following are hard failures:

- empirical linear-scale means exclude the exact value by a discrepancy that
  cannot be explained by recorded Monte Carlo uncertainty;
- terminal likelihood or proposal corrections are misnormalized;
- different restart boundaries change results; or
- equivalent cell permutations change the frozen estimator distribution.

These are diagnostic, not automatic hard stops:

- SMC is slower than IID;
- ESS becomes small;
- one tree ordering is poor; or
- no variance reduction appears at one particle count.

If bootstrap SMC collapses but target/replay checks pass, proceed to the
bounded guided-proposal experiment rather than terminating the whole track.

## R2: locally guided Beta proposal

Run only after R1 bootstrap results are complete.

For one eligible split at a time, construct a continuous proposal
approximating

\[
q(\rho_v)
\propto
\operatorname{Beta}_v(\rho_v)
\widetilde L_{\ell+1}(\rho_v).
\]

Allowed first implementations are:

- a normalized piecewise-polynomial/log-density approximation built from
  Gauss--Jacobi evaluations;
- a beta-mixture proposal with an explicitly evaluable density; or
- another continuous one-dimensional proposal with exact sampling and density
  evaluation.

A finite set of Gauss--Jacobi atoms is not an exact continuous proposal.  If
used, it defines a separately labelled finite-quadrature target.

Repeat the smallest informative subset of R1.  Compare:

- proposal-normalizer accuracy against direct one-dimensional integration;
- incremental-weight CV and ESS;
- total likelihood variance per wall time;
- extra proposal construction cost; and
- the same replay and chart audits.

The agent may try a small documented proposal-parameter ladder.  Keep every
attempt; do not select using realized PARIS data.

## R3: medium synthetic scaling

Construct balanced native trees with:

```text
native cells: 16, 64, 256, 1024
observation dimensions: 8, 32
allocation regimes: balanced, skewed, boundary-heavy
particle counts: 256, 1024, 4096
replicates: at least 16
```

Include:

- one \(H\) whose sensitivity is concentrated in upper-tree contrasts;
- one whose energy is spread over many descendants; and
- one deliberately adversarial ordering.

The decisive diagnostic is whether either

\[
\operatorname{Var}
\{\widetilde L_\ell-\widetilde L_{\ell-1}\}
\]

or the coefficient of variation of incremental SMC weights decreases as the
tree is refined.  Estimate cost-versus-variance scaling; do not fit an
asymptotic rate from fewer than three useful sizes.

Record when the unresolved-tail Gaussian closure becomes accurate enough that
early stopping would be scientifically plausible, but do not promote an
early-stopped likelihood in this plan.

## R4: observation-blind PARIS screen

Run only after R1 correctness and at least one viable R3 configuration.

### Stage R4a: engineering canary

Use 2--4 G4 prior-predictive validation states and:

```text
particles: 16, 64
summary-guide ranks: 16, 32
schedule: breadth-first depth batches
replicates: 4
```

Request enough memory to retain the full frozen \(H\), particles, frontier
masses, and the fixed summary workspace.  Start with 32 GiB and one CPU task;
increase resources if measured evidence warrants it.  Memory is not the
scientific gate.

The intermediate guide may use
`AdditiveDirichletAggregation` in the frozen
`RootResidualSpectrum` basis.  At the terminal level, score the actual
complete \(HX\) under diagonal \(D\).  The summary guide is allowed to be
imperfect because it is not the terminal target.

### Stage R4b: matched estimator comparison

After the canary:

```text
validation states: 16 observation-blind G4 states
particles/samples: 64, 256, 1024 as resources permit
replicates: at least 8 independent seeds
SMC schedules:
  - breadth-first depth batches
  - fixed equal-prior-energy batches if implemented before launch
baselines:
  - IID complete allocations
  - blocked scrambled Sobol complete allocations
```

Compare at matched measured work.  Required diagnostics are those from R1,
plus:

- fraction of terminal likelihood mass carried by the top 1, 5, and 10
  particles;
- between-run log-normalizer dispersion;
- node/depth at first severe ESS loss;
- forward-model throughput;
- terminal allocation and \(HX\) batch timings;
- peak RSS and output size; and
- agreement across equivalent fixed tree charts within replicate
  uncertainty.

Do not use finite estimator differences to weight a partition.  The purpose
is estimator stability for one fixed root.

### Advancement signal

Resolution-SMC is promising if, across several states rather than one:

- its linear-scale normalizing constant is consistent across independent
  replicates;
- it materially improves variance per wall time over IID;
- ESS does not collapse irreversibly before informative refinement;
- results agree across compatible tree charts within Monte Carlo uncertainty;
  and
- the result improves predictably with particle count.

No single numerical factor is a hard gate.  A useful provisional target is at
least a twofold variance-per-cost improvement over IID with no chart
discrepancy beyond replicate uncertainty.

## R5: decision report

Publish:

```text
report/RESULTS.md
report/summary.json
report/attempt-ledger.csv
report/jobs.csv
report/sha256sums.txt
```

The report must distinguish:

- exact mathematical identities;
- approximate intermediate guides;
- finite-particle Monte Carlo error;
- any deliberately finite-quadrature target;
- observations from completed runs;
- implementation defects and repairs;
- negative results for particular configurations; and
- untested future ideas.

Include readable figures for:

1. incremental ESS by resolution;
2. correction variance by resolution;
3. likelihood RMSE versus measured work;
4. tree-order sensitivity; and
5. particle ancestry or terminal weight concentration.

No protected catalogue access and no `PARIS_inversions` writes are permitted.

## Debugging and iteration policy

This plan deliberately avoids a rigid all-or-nothing progression during
method development.

The HPC agent is authorized to:

- localize and fix implementation, serialization, checkpoint, launcher, and
  numerical-stability defects;
- add diagnostics needed to understand an observed failure;
- adjust Slurm memory, wall time, and array shape from measured resource use;
- try the bounded R2 proposal alternatives; and
- reduce an oversized exploratory matrix while preserving representative
  cases.

For every change:

1. preserve the failed attempt;
2. explain whether the change is scientific, numerical, or operational;
3. commit and push source changes;
4. use a new full-SHA run root; and
5. keep prior results in the attempt ledger.

Do not silently relax target, normalization, replay, provenance, protected
data, or partition-invariance requirements.

## Handoff completion condition

Work is ready to hand back when:

- R0 and R1 have complete reports;
- either bootstrap SMC is characterized or R2 has been tried;
- at least one R3 scaling matrix is complete;
- R4 is either complete or has a well-localized resource/scientific failure;
- all source and planning changes are pushed;
- every job is terminal or explicitly listed as active; and
- the final report recommends continue, redesign, or stop with evidence.
