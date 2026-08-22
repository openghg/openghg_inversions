# Issue 637 HFC-134a diagnostic figures

These figures accompany the controlled `af0e8662` and `90a71c4` HFC-134a
Central Asia reproductions described in issue #637.

## February 2026 posterior baseline by chain

![February 2026 posterior baseline by chain](hfc134a_feb2026_baseline_by_chain.png)

Posterior mean baseline in concentration space, in ppt. Solid coloured lines
show individual `af0e8662` chains; dotted lines show `90a71c4` chains. The red
line is the archived anomalous output, the dashed black line is the archived
normal output, and crosses are observations. Failed chains occupy high-BC,
high-sigma states 70--100 ppt above observations.

## Full posterior mean time series

![Full posterior mean time series](hfc134a_full_timeseries_comparison.png)

Deterministic posterior mean concentration prediction, in ppt, for each
archived product and the four-chain pooled reproduction summaries. Pooling is
shown only to visualize the disagreement; these nonconverged traces are not
scientifically valid for pooled publication.

## Seeded likelihood control

![Seeded PEFO and additive-sigma baselines by chain](hfc134a_likelihood_seed637_baseline_by_chain.png)

Exact `90a71c4` source, identical serialized inputs, NumPyro, GCC 12.3,
four chains, 30,000 retained draws, and random seed 637. The PEFO arm retains
the production fractional-error likelihood and `Uniform(0, 0.1)` sigma. The
additive arm uses response-independent absolute sigma with a monthly,
site-specific `HalfNormal(5 ppt)` prior.

PEFO reproduced the failure: maximum BC R-hat 2.402, minimum bulk ESS 4.85,
and maximum between-chain baseline disagreement 106.26 ppt. The additive arm
had maximum R-hat 1.00021 across x/BC/sigma, minimum bulk ESS 39,598, and only
0.065 ppt maximum baseline disagreement. The black high observation in the
additive BCOB panel is a real isolated observation, not a modelled-baseline
excursion.

Tabular results: [diagnostics](hfc134a_likelihood_seed637_diagnostics.csv) and
[per-chain summary](hfc134a_likelihood_seed637_chain_summary.csv).

The third panel is a same-seed PEFO sensitivity which narrows the emissions
LogNormal arithmetic standard deviation from 2 to 1. It also fails: maximum BC
R-hat 1.917, minimum bulk ESS 5.51, and 130.31 ppt maximum between-chain
baseline disagreement. All four chains have substantially worse summed log
likelihood than the healthy chain in the original PEFO control. Narrowing the
emissions prior therefore does not remove the PEFO pathology for this seed.
