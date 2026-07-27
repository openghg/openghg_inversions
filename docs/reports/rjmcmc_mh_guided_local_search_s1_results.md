# MH-guided local partition search: S1 atmospheric-like synthetic result

## Decision

The bounded S1 experiment **failed**. Its first failed predeclared gate was
`oracle_learnability_edge-one`; the relocation oracle-learnability gate and
all three mobile utility gates also failed. The operational, structural
mobility, and fixed-topology continuous-reference gates passed.

The result is therefore a scientific hard stop for this experiment. S2 and
the proposed real-CH4-footprint synthetic experiment were not run. No
factor-four extension was authorized because every conditional local-versus-
NUTS reference passed, while utility results were explicitly forbidden from
authorizing a retry.

The immutable completed run is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_s1/4848316d62502ef26abefa1aa175005af9cdacfa/harness-c5b6180cace94b2d7c301c1b3a38c12e6baeab17db7067d06c74ba9486d020f8
```

Scientific source revision:
`4848316d62502ef26abefa1aa175005af9cdacfa`.

This is a finite-budget stochastic-local-search result. It is not evidence
about convergence of a partition posterior, correct marginalization over
partitions, partition probabilities, or connectivity of the fixed-\(K\)
tiling graph.

## What was tested

The experiment used a positive dimensionless scaling field on an \(8\times8\)
native grid with \(K=8\) rectangular regions. The initial practical partition
\(P_0\) was the deterministic balanced `largest-nominal` basis. It was neither
a random tiling nor a draw from a structural prior.

“Truth” here means a planted native-grid scaling field, constant within the
leaves of its planted topology \(P_\star\), normalized to native-grid mean
one, and used to generate the synthetic observations. Three cases were
predeclared:

1. `aligned`, with \(P_\star=P_0\);
2. `edge-one`, with \(P_\star\) exactly one witnessed edge flip from \(P_0\);
3. `relocation-one`, with \(P_\star\) exactly one witnessed resolution
   relocation from \(P_0\).

The independently seeded Gaussian-footprint operators contained 96 training
rows and 48 held-out rows. Four paired Gaussian-noise replicates were used per
case. The likelihood had fixed independent error
\(R=0.08^2I\), in scaling units; its standard deviation was not inferred.
The held-out observations, held-out operator, truth, \(P_\star\), and witness
were unavailable to practical fixed/mobile sampling.

The fitted model used:

- \(T\sim\operatorname{Gamma}(4,4)\);
- globally additive Dirichlet allocation with total concentration \(2K=16\);
- likelihood power one;
- no outer coefficients, boundary or baseline term, hierarchy, aggregation
  error, or inferred error parameter.

Three matched arms were compared:

- `fixed`, conditioned and sampled on \(P_0\);
- `mobile`, forked from the same conditioned \(P_0\) state and allowed two
  local structural MH slots per cycle; and
- `oracle`, separately conditioned and fixed on the sealed \(P_\star\).

The primary metric was the held-out noiseless prediction RMSE of the
all-production-cycle mean, in dimensionless scaling units. The native score
was native-grid scaling RMSE. Every mobile/fixed or oracle/fixed ratio divides
by the paired fixed-\(P_0\) score, so values below one favor the numerator.

## Sampling parameters

Each practical cell used one fixed-\(P_0\) conditioning segment before its
fixed/mobile fork; each oracle cell used a separate fixed-\(P_\star\)
conditioning segment. The declared counts were:

- 10,000 conditioning cycles, excluded from reported averages;
- 50,000 production cycles;
- all 50,000 complete-cycle states retained;
- thinning interval one cycle.

Thus “burned” or excluded work was 10,000 conditioning cycles and the retained
sample count was 50,000 per arm and replicate. The conditioning segment was a
common-state warm-start operation, not evidence that a Markov chain had
reached stationarity.

A fixed or oracle cycle contained one root slice and five allocation-pair
updates: 60,000 excluded conditioning transition slots and 300,000 production
slots. A mobile production cycle contained the same six continuous updates
plus two structural MH slots: 400,000 production slots. Mobile and fixed
therefore had matched continuous-update opportunities; equal-wall-time
prefixes were also scored.

Each of five fixed-topology local-reference cells used four local chains with
10,000 conditioning and 50,000 retained cycles per chain, thinning one. Its
NUTS reference used four chains, 1,000 warm-up draws and 1,000 retained draws
per chain, no thinning, diagonal mass, and target acceptance 0.9. The primary
NUTS attempt passed in every cell, so the declared NUTS retry was not used.

## Key results

The table reports medians across four paired noise replicates. Fixed and
mobile RMSE columns are dimensionless scaling RMSE; the ratios are medians of
the four paired ratios rather than ratios of the displayed medians.

| Scenario | Fixed held-out RMSE | Mobile held-out RMSE | Mobile/fixed held-out | Mobile/fixed native | Equal-wall mobile/fixed held-out | Oracle/fixed held-out | Mobile replicates below one |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aligned | 0.021971 | 0.028265 | 1.454480 | 1.983898 | 1.452200 | 0.997966 | 0/4 |
| Edge-one | 0.024481 | 0.024794 | 1.123784 | 1.171817 | 1.157434 | 0.992032 | 1/4 |
| Relocation-one | 0.023558 | 0.025356 | 1.046796 | 1.202538 | 1.050650 | 0.947016 | 1/4 |

The oracle-learnability ratios were:

- `edge-one`: 1.0608, 1.0271, 0.9255, and 0.9569. Only two of four were below
  one and the median was 0.9920, versus the required three of four and median
  at most 0.90.
- `relocation-one`: 0.9182, 0.9758, 0.9986, and 0.7913. All four were below
  one, but the median was 0.9470 rather than at most 0.90.

The aligned result is the clearest negative result: structural search made
held-out prediction worse in all four replicates even though \(P_0\) was the
planted topology. Equal-wall-time and native-grid results support the same
conclusion. One relocation replicate improved substantially, with held-out
and native mobile/fixed ratios 0.8005 and 0.7242, but this was not reproduced
by the other three seeds.

## Structural movement

The mobile kernel was not stuck. Across the 12 chains:

- 1,200,000 structural slots produced 19,223 accepted moves;
- 11,268 of 484,570 valid edge flips were accepted (2.33%);
- 7,955 of 379,357 valid relocations were accepted (2.10%);
- each chain saw valid proposals of both types and accepted structural moves;
- chains visited 56--224 unique topologies, with median 136;
- \(P_\star\) was present or reached in 10 of 12 chains: all four aligned
  chains by identity at cycle one, all four `edge-one` chains, and two
  `relocation-one` chains; and
- median \(P_0\) residence was 0.952%.

Aligned chains spent only 0.882--7.166% of retained cycles at their planted
correct \(P_0\). All four edge chains reached \(P_\star\), yet only one
improved its held-out score. The residence-weighted result therefore cannot
be explained by an absence of valid proposals or by total structural
immobility.

## Continuous-reference results

All five fixed-topology local-versus-NUTS reference cells passed:

- zero NUTS divergences;
- worst \(\hat R=1.002178\), for `leaf_mass[r6:8_c0:4]`;
- minimum bulk ESS \(=1873.82\), for `leaf_mass[r0:1_c0:8]`;
- minimum tail ESS \(=1574.48\), for the same leaf mass;
- worst local MCSE/posterior-SD ratio \(=0.01671\), for `top_right`;
- worst half-window difference/posterior-SD ratio \(=0.05915\), for
  `top_right`; and
- worst local-versus-NUTS tolerance use \(=0.71004\), for `top_half`.

These checks make a fixed-topology continuous-sampler failure an implausible
explanation for the S1 decision. They do not establish convergence of the
mobile structural process.

## Operational history and integrity

Two earlier full-SHA run roots are preserved as failed or incomplete evidence
and were not promoted:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_s1/83361b8c993aca1f81f190efd562c59a6c74f50f/harness-c5b6180cace94b2d7c301c1b3a38c12e6baeab17db7067d06c74ba9486d020f8
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_s1/1450ab9476892bf5fdf241e2033c3b90269c25bc/harness-c5b6180cace94b2d7c301c1b3a38c12e6baeab17db7067d06c74ba9486d020f8
```

The first exposed non-canonical target caches when a conditioned branch was
reloaded; the second exposed one-to-two-ULP target-output drift in the
retained local-reference trace. The reviewed fixes canonicalized persisted
branch states and local-reference output from authoritative coordinates,
without changing the sampler, MH accounting, random streams, practical
mobile/oracle arms, or retry rules. Utility was not inspected in either
failed run.

The completed run used primary jobs `18188730`--`18188739`, recovered
conditional-reference array `18188876`, recovered index job `18188984`, and
aggregate/final-audit job `18188988`. Immutable dependency launchers attempted
to attach `afterok` to already-terminal arrays, which BP1 rejected. A
premature aggregate job, `18188981`, then failed because the index did not
yet exist. Those records remain preserved. Recovery submitted the unchanged
jobs only after read-only Slurm checks proved their local parents had
completed successfully. This was a launcher defect, not a scientific-job
failure.

The final audit found a complete, finite 12-cell matrix, replayed all indexed
hashes, reproduced the aggregate completion byte-for-byte, and wrote root
`complete.json` last. Focused validation of the final scientific revision
passed 33 tests across the two affected experimental modules, two standalone
end-to-end regressions, Ruff, formatting, and Pyright. No repository-wide tox
matrix was run.

Key hashes are:

| Artifact | SHA-256 |
|---|---|
| Frozen S1 definition | `24099340c5e192bbd258e32270e61247bbad33769da277c39d143f15702f819d` |
| Frozen harness | `c5b6180cace94b2d7c301c1b3a38c12e6baeab17db7067d06c74ba9486d020f8` |
| Pixi lock | `4ed1244c33ffb7ef929bad73d8bd9944e49ed9b36b51fa05163b59b2a5b2f564` |
| Final index | `2dbc44daea52bd40133408d3be49b7190ed562fd9bbef00fbdbcb8b93e104ceb` |
| Decision | `2aa4fd0aee5f1a7d76eced07df8e51f3270d576c0287c40325ff56b0e7a7201f` |
| Aggregate/replay completion | `1687084aab679dff613e07fbffb3db7495be98f3179f8de14e8eff92192f15c2` |
| Root completion | `058c37579ddb5fef9e47d1df0e6da3cbf3ecdcf1420525ea1112847050ca80bb` |

## Interpretation

At this budget, the current MH-guided fixed-\(K\) algorithm explored many
topologies but did not provide robust held-out or native-field improvement.
The result does not identify poor partition-posterior convergence as the
cause. More narrowly, the combination of structural target, regularization,
local moves, and residence-weighted averaging did not act as a useful
stochastic local search in this atmospheric-like screen.

The failed oracle gates make causal attribution deliberately conservative:
the planted alternatives were not shown to be strongly and reproducibly
better than \(P_0\) under the inference and scoring setup. Consequently,
mobile failure cannot be attributed solely to failure to find a clearly
useful oracle partition. Conversely, the strong aligned degradation shows
that movement away from a good starting basis can itself be harmful at the
declared finite budget.

This experiment does **not** establish:

- convergence, irreducibility, or connectedness of the structural chain;
- a partition posterior or correct Bayesian averaging over partitions;
- inferential uncertainty or interval coverage;
- that the planted \(P_\star\) is the unique or posterior-optimal partition;
- failure or success on real atmospheric footprints or observations; or
- superiority of a fixed basis outside these scenarios.

## Follow-up

Stage 2 remains withheld under the frozen gate sequence. Before a new
real-footprint experiment, a separately designed experiment should establish
a benchmark with clear oracle separation and decide whether the practical
finite-run estimator should be a residence average, a training-only selected
state, or another explicitly non-Bayesian search output. Any new regularizer,
annealing/search schedule, topology objective, or best-state rule would define
a new algorithm and must be frozen before its held-out results are inspected.
