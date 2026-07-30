# Wider R1 Gamma–Beta resolution-SMC report

## Outcome

All 36 frozen tasks passed at source
`abdce3c30c65aebd88c3c4f27c588c71aaabe2c2`. Independent analysis
authenticated 17,024 replicate records and 55,680 per-level records against
their certificates and SHA-256 digests.

Bootstrap resolution-SMC did not establish a cost-adjusted advantage over
direct IID. None of the 380 bootstrap estimator cells beat IID on relative
variance times median measured wall time, and none reached the provisional
twofold target. The best boundary-heavy ratio was 2.18 for
`boundary_heavy_four_cell_generic_contrasts`, the column-first chart,
\(N=64\), and observation-energy ordering at ESS fraction 0.5.

By contrast, independently scrambled Sobol complete allocations beat direct
IID in 69 of 76 cells on the same primary score.

## Target and scoring

Here, \(Z\) is the fixed-root allocation-marginal normalized native Gaussian
likelihood and \(N\) is the particle or complete-allocation sample count.
Relative variance is the between-replicate variance of \(Z\) divided by the
squared oracle or reference value. The primary score multiplies that quantity
by median measured wall time. Log-likelihood errors are secondary reporting
coordinates.

The Gaussian closure is used only between refinements. Every terminal
particle is scored with the exact normalized native Gaussian density. The
two- and four-cell targets use independently converged Gauss–Jacobi values.
The two 16-cell targets use 16 independent 262,144-sample IID references and
are not labelled exact.

## Correctness and degeneration

Four of 420 exact-target cells differed by more than three combined Monte
Carlo standard errors. Two of 392 compatible-chart comparisons differed by
more than three replicate standard errors. None exceeded four standard
errors or repeated as a common-sign pattern across particle counts, so these
are retained as finite-replicate diagnostics rather than target failures.

The worst per-level ESS fraction was 0.02556 and the smallest
unique-ancestor count was three. These are scientific failures, not discarded
runs. Terminal unresolved covariance is exactly zero. Maximum mass and
covariance update errors were \(1.11\times10^{-16}\), and the maximum mean
update error was \(8.88\times10^{-16}\).

At \(N=4096\) for the row-first boundary-heavy row/column target:

| Estimator | Mean \(Z\) | SE | Relative RMSE | RelVar × median cost / IID |
|---|---:|---:|---:|---:|
| breadth, ESS 0.25 | 1.332425 | 0.0122 | 0.0528 | 12.2 |
| breadth, ESS 0.50 | 1.361680 | 0.00881 | 0.0367 | 6.65 |
| energy, ESS 0.25 | 1.371501 | 0.0130 | 0.0550 | 18.6 |
| energy, ESS 0.50 | 1.361918 | 0.00960 | 0.0399 | 10.5 |
| unfavourable, ESS 0.50 | 1.343840 | 0.0107 | 0.0445 | 12.6 |
| direct IID | 1.329766 | 0.00933 | 0.0423 | 1.00 |
| scrambled Sobol | 1.354127 | 0.000839 | 0.00345 | 0.0587 |

## Interpretation

The wider matrix confirms the R1a negative result. Prior-proposal SMC can
reduce raw variance, but frontier updates, repeated guide evaluations, and
resampling erase the gain at measured cost. Observation-energy ordering is
not uniformly favourable, while the deliberately unfavourable ordering
exposes severe ESS and ancestry loss.

R0 and R1a established path identity, conservation, child-swap equivariance,
provenance rejection, and checkpoint replay. Wider R1 reuses that validated
engine but does not emit a fresh checkpoint for every production cell. R2
repeats every restart boundary explicitly.

The canonical report, plots, raw JSONL records, summary tables, and checksums
are under:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc/abdce3c30c65aebd88c3c4f27c588c71aaabe2c2/report
```
