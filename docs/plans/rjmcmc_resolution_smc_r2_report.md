# Bounded R2 guided-proposal report

## Outcome

All three R2 tasks passed at frozen source
`abdce3c30c65aebd88c3c4f27c588c71aaabe2c2`. Every one of 12 saved
restart boundaries reproduced the uninterrupted scientific fingerprint and
matched its recorded checkpoint digest.

Four of 18 guided cells beat prior-proposal bootstrap SMC on relative
variance times median measured wall time. None reached a twofold improvement.
The best ratio was 0.598 for the row-first chart, \(N=256\), and the 32-bin
guide.

## Target and proposal

As in R1, \(Z\) denotes the allocation-marginal normalized native Gaussian
likelihood. Variance and cost ratios are relative to prior-proposal bootstrap
SMC under the same chart and particle count.

The proposal is continuous. Equal-prior-probability bins define a
piecewise-constant Gaussian guide factor multiplied by the exact Beta
density. Sampling uses exact truncated-Beta inversion, and terminal weights
include the exact evaluable prior/proposal correction. The Gauss–Jacobi
normalizer calculation is an audit, not the proposal target.

Guidance usually raised mean intermediate ESS, but coarser ladders could
still collapse. The worst prior ESS fraction was 0.153, whereas the worst
guided value was 0.0188. The maximum proposal-normalizer relative discrepancy
was 0.704 for the coarsest ladder and is retained as a guide-quality failure.

For the row-first chart at \(N=4096\):

| Proposal | Minimum ESS fraction | Max normalizer rel. error | Variance / prior | Median cost / prior | Var × cost / prior |
|---|---:|---:|---:|---:|---:|
| prior bootstrap | 0.186 | 0 | 1 | 1 | 1 |
| 8 bins | 0.0276 | 0.704 | 0.410 | 30.4 | 12.5 |
| 16 bins | 0.0853 | 0.143 | 0.251 | 33.4 | 8.37 |
| 32 bins | 0.673 | 0.000356 | 0.0747 | 39.5 | 2.95 |

## Interpretation

The 32-bin guide generally flattens incremental weights and raises ESS, but
the extra proposal construction and guide-evaluation work is expensive. The
bounded ladder does not provide evidence for promoting this implementation
to medium or PARIS scale. An analytic or amortized guide would be a redesign,
not a tuning continuation, and would need to repeat the exact tiny matrix.

The canonical report, plot, replicate and level diagnostics, checkpoint
certificates, and checksums are under:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc/abdce3c30c65aebd88c3c4f27c588c71aaabe2c2/report
```
