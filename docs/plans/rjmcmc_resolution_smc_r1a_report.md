# R1a Gamma–Beta resolution-SMC result

## What Was Tested

Three fixed synthetic targets were evaluated: a near-Gaussian two-cell Beta
allocation, the skewed G1 two-cell allocation, and a boundary-heavy four-cell
Dirichlet allocation. Each estimator/count/chart cell has 64 independent
replicates at 64, 256, 1,024 and 4,096 particles. The Gaussian closure is only
an intermediate SMC guide; every terminal weight uses the exact normalized
native Gaussian likelihood.

The successful matrix ran from clean detached source
`9548ca59a01cdc095bf372c91c4416a9fcfb7162` as Slurm array `18222470`.

## Terminology And Target

The target is the fixed-root allocation-marginal normalized Gaussian
likelihood. Accuracy and variance are scored on its non-negative linear scale.
Relative variance means between-replicate variance divided by the squared
quadrature oracle. The primary cost score multiplies relative variance by
median measured estimator wall time; a secondary mean-time score is retained
because first-call warm-up affected some cells.

## What Happened

All 12 task certificates passed. Independent post-run validation found 4,096
replicate records and 4,608 per-level records. Direct IID and no-resampling
SMC were exactly pathwise identical, terminal unresolved covariance was zero,
and conservation/update errors remained at floating-point roundoff:

- maximum mass error: `1.11e-16`;
- maximum conditional-mean update error: `8.88e-16`; and
- maximum unresolved-covariance update error: `1.11e-16`.

Bootstrap SMC reduced raw variance in the boundary-heavy target, but its
additional frontier and guide work erased that gain. The best boundary
relative-variance-times-median-cost ratio was 1.66 (SMC divided by IID), so the
planned twofold improvement was not reached. The minimum recorded bootstrap
ESS fraction was 0.0668.

Four of 64 estimator cells had a nominal linear-scale 95% interval that did
not contain the oracle. They were two path-identical direct/no-resampling
rows for one random seed domain and two bootstrap cells with opposite signed
errors. They are retained as finite-replicate evidence for wider R1 rather
than treated as a normalization failure. All 16 compatible-chart comparisons
agreed within three independent-replicate standard errors.

## Boundary-Heavy Summary

The table shows the 4,096-particle boundary-heavy comparison. The final column
is the estimator's relative-variance-times-median-seconds divided by the
corresponding direct-IID value; values below one favor SMC.

| Chart | Estimator | Mean Z | SE(Z) | Relative bias | Cost score / IID |
|---|---|---:|---:|---:|---:|
| column-first | bootstrap ESS 0.5 | 1.344771 | 0.00714 | -0.00681 | 3.15 |
| column-first | resample every refinement | 1.354110 | 0.00660 | 0.0000887 | 2.70 |
| column-first | direct IID | 1.352718 | 0.0108 | -0.000940 | 1.00 |
| column-first | matched no-resampling SMC | 1.352718 | 0.0108 | -0.000940 | 6.92 |
| row-first | bootstrap ESS 0.5 | 1.349360 | 0.00594 | -0.00342 | 2.30 |
| row-first | resample every refinement | 1.346456 | 0.00653 | -0.00556 | 2.79 |
| row-first | direct IID | 1.348140 | 0.0105 | -0.00432 | 1.00 |
| row-first | matched no-resampling SMC | 1.348140 | 0.0105 | -0.00432 | 6.92 |

## Interpretation

R1a validates the target, normalization and replay implementation while
rejecting the hoped-for cost-adjusted bootstrap advantage in this tiny
matrix. Raw variance reduction is real in several boundary-heavy cells, but
the implementation is much more involved per allocation than direct terminal
evaluation. The observed low intermediate ESS and ordering sensitivity
justify completing the wider bootstrap R1 matrix and the bounded
one-dimensional guided-proposal R2 test. They do not justify PARIS-scale R4.

## Outputs

The complete exact-SHA run root is:

`/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc/9548ca59a01cdc095bf372c91c4416a9fcfb7162`

Its `report/r1a_estimator_summary.csv` contains all linear-scale accuracy,
variance, timing and work metrics. `report/r1a_level_summary.csv` contains
per-level ESS, weight, ancestry and correction summaries.
`report/r1a_chart_summary.csv` contains compatible-chart comparisons, and
`r1a/` retains the original replicate and per-level JSONL shards.
