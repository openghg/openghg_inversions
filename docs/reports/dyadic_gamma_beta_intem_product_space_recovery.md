# InTEM/TAC-MHD positive product-space recovery

## Scope

This experiment scales the positive Gamma-Beta product-space model from two
grid cells to the committed EUROPE grid and real footprint-times-flux
sensitivities. Observations are synthetic emissions-only values, so the known
problematic boundary-condition fixtures are not opened.

The model uses:

- 47 committed TAC/MHD sensitivity rows on the 293 by 391 grid;
- 32 training and 15 deterministic held-out rows;
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

| Fit | Mean K | K range | Unique P | Truth-P mass | Inner-land field RMSE | Holdout RMSE | Holdout log predictive density | Structural acceptance | Divergences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent K/P | 13.07 | 12--18 | 16 | 0.498 | 0.0735 | 0.4631 | -37.44 | 0.122 | 0 |
| fixed true P | 12.00 | 12--12 | 1 | 1.000 | 0.0421 | 0.4538 | -37.41 | 0.000 | 0 |
| fixed root P | 11.00 | 11--11 | 1 | 0.000 | 0.4620 | 3.8119 | -50.77 | 0.000 | 0 |

The latent inversion meets the declared proof-of-concept goal. It matches the
fixed true-partition inversion on held-out prediction and field recovery, while
decisively beating the underfit fixed partition. It traverses multiple K and P
states rather than remaining at the planted partition, and has no NUTS
divergences.

The latent full-grid field RMSE is 0.0802, slightly below the fixed true value
0.0878. This aggregate includes large outer areas whose true scaling is one;
the inner-land RMSE is the more discriminating spatial metric.

## Why the K prior matters

With the same 100-region forest and a uniform marginal prior over K=11 through
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

The declared run completed in about 30 seconds wall time on the local machine;
reported PyMC sampling times were approximately 5 seconds for latent K/P and
4 seconds for each fixed comparator. Forest construction, model compilation,
and posterior field reconstruction account for the remainder.

## Remaining limitations

- The observations are synthetic and omit baseline/boundary conditions by
  design.
- Only one observation-noise realization and one chain are reported.
- The planted truth is a high-level tree split, so it is favorable to the
  candidate dictionary.
- Exact posterior partition probabilities are unavailable at this scale.
- The geometric K prior is reasonable but not calibrated from external domain
  knowledge; sensitivity to its continuation probability should be reported.
- Inactive Beta coordinates are updated by NUTS even when their splits are not
  active. The 100-region case is practical, but scaling should be measured
  before moving to substantially larger forests or time-varying partitions.
- At a 32-inner-region sensitivity-weighted budget, ocean receives only its
  disconnected-component minimum and is separated but not further refined.
  The 100-region forest provides a larger dictionary, but posterior group-wise
  K and land/ocean split diagnostics should be added explicitly.

