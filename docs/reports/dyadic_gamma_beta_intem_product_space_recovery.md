# InTEM/TAC-MHD positive product-space recovery

## Scope

This experiment scales the positive Gamma-Beta product-space model from two
grid cells to the committed EUROPE grid and real footprint-times-flux
sensitivities. Observations are synthetic emissions-only values, so the known
problematic boundary-condition fixtures are not opened.

The model uses:

- 47 committed TAC/MHD sensitivity rows on the 293 by 391 grid;
- 32 training and 15 deterministic held-out rows, split before candidate
  allocation or sensitivity-weighted refinement;
- six fixed InTEM outer groups with separate Gamma roots;
- hard-separated inner land and ocean groups;
- a sensitivity-weighted candidate forest with 100 inner terminal regions;
- 11 component roots, 106 maximum leaves, and 95 possible split indicators;
- eight semantic group-root variables and 95 permanent Beta coordinates;
- the controlled depth-kappa policy and 50% UK-total calibration;
- a truncated geometric prior on additional splits with continuation
  probability 0.5.

The planted truth changes one conservative high-level split of the main
inner-land component. The true partition has K=12. All other split fractions
remain at their prior expected mass fractions and all semantic root scalings
equal one.

## Matched 500/500 result

Each fit used one chain, 500 NUTS tuning draws, 500 retained draws,
`target_accept=0.95`, and matched observation and coordinate priors.

| Fit | Mean K | K range | Unique P | Truth-P retained-draw frequency | Inner-land field RMSE | Holdout RMSE | Holdout log predictive density | Structural acceptance | Divergences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent K/P | 12.87 | 12--15 | 12 | 0.480 | 0.0863 | 0.4793 | -37.40 | 0.088 | 0 |
| fixed true P | 12.00 | 12--12 | 1 | 1.000 | 0.0465 | 0.5514 | -37.48 | 0.000 | 0 |
| fixed root P | 11.00 | 11--11 | 1 | 0.000 | 0.4633 | 3.7907 | -50.66 | 0.000 | 0 |

The latent inversion meets the declared proof-of-concept goal. It matches the
fixed true-partition inversion under the declared holdout and field tolerances, while
decisively beating the underfit fixed partition. It traverses multiple K and P
states rather than remaining at the planted partition, and has no NUTS
divergences.

The latent full-grid field RMSE is 0.0790, below the fixed true value 0.0974.
This aggregate includes large outer areas whose true scaling is one;
the inner-land RMSE is the more discriminating spatial metric.

The latent structural kernel includes fixed-K swaps: it may merge one current
frontier branch and split another in one proposal. This lets the chain relocate
resolution without first accepting an intermediate K. The two fixed
comparators use normalized point-mass partition potentials and explicitly
disable swaps; retained masks are asserted to equal their declared P.

The 0.480 truth-P value is a retained-draw frequency from one chain, not a
well-estimated posterior probability. Replicated chains and a partition ESS are
needed before interpreting that number quantitatively.

Separately, a prior-only depth-two tree test runs the same custom structural
step over all five possible partitions. Ten thousand retained updates recover
every exact partition probability within 0.025 absolute error. This verifies
the local proposal and asymmetric Hastings correction independently of this
benchmark's favorable planted signal.

## Why the K prior matters

In an earlier split/merge-only sensitivity run with the same 100-region forest
and a uniform marginal prior over K=11 through
106, a 500/500 run averaged K=22.98 and visited 198 partitions. Held-out RMSE
remained reasonable at 0.531, but inner-land field RMSE rose to 0.177. The
fixed graph and sampler were working; the declared complexity prior was too
permissive for 32 training observations.

The geometric prior uses

```text
p(K) proportional to q**(K - K_min),  q = 0.5,
```

truncated to the candidate forest. Conditional on K, exact dynamic-programming
counts still make partitions uniform. This prior has an interpretable expected
number of extra splits and behaves consistently when the maximum candidate
forest is enlarged.

## Execution

```bash
HOME=/tmp MPLCONFIGDIR=/tmp .venv/bin/python \
  examples/basis/dyadic_gamma_beta_intem_product_space_recovery.py \
  --inner-regions 100 \
  --draws 500 \
  --tune 500 \
  --target-accept 0.95 \
  --k-continuation-probability 0.5
```

The declared run completed in about 25 seconds wall time on the local machine;
reported PyMC sampling times were approximately 5 seconds for latent K/P and
4 seconds for each fixed comparator. Forest construction, model compilation,
and posterior field reconstruction account for the remainder.

A separate 50/50 scale probe requested 250 inner terminal regions. Moment
constraints retained 244 maximum leaves, represented by 477 forest nodes and
233 split coordinates. The full latent/fixed/underfit comparison completed in
about 28 seconds, the latent chain visited six partitions over K=12--15, and
there were no divergences. Its holdout RMSE was 0.499 versus 0.532 for the
planted fixed partition and 3.800 for the underfit roots. Fifty retained draws
are not enough for inference; this only establishes that the static product
space remains executable at approximately the prototype's 250-region scale.

## Remaining limitations

- The observations are synthetic and omit baseline/boundary conditions by
  design.
- Only one observation-noise realization and one chain are reported.
- Candidate topology uses training sensitivities only. A regression multiplies
  every holdout design row by 100 and confirms the forest, partition prior, and
  planted truth are unchanged.
- The planted truth is a high-level tree split, so it is favorable to the
  candidate dictionary.
- Exact posterior partition probabilities are unavailable at this scale.
- The geometric K prior is reasonable but not calibrated from external domain
  knowledge; sensitivity to its continuation probability should be reported.
- Inactive Beta coordinates are updated by NUTS even when their splits are not
  active. The 100-region case is practical, but scaling should be measured
  before moving to substantially larger forests or time-varying partitions.
- Independent observation errors use a vectorized Normal likelihood. A full
  residual covariance remains supported through MvNormal but has not been
  performance-tested at this forest size.
- At a 32-inner-region sensitivity-weighted budget, ocean receives only its
  disconnected-component minimum and is separated but not further refined.
  The 100-region forest provides a larger dictionary, but posterior group-wise
  K and land/ocean split diagnostics should be added explicitly.
