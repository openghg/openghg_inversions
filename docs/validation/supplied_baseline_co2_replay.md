# Supplied-background CO2 implementation replay

The supplied-background CO2 implementation passed the archived translation
checks and reproduced the accepted controlled target within Monte Carlo error.
This validates implementation and accounting; it is not a new flux-recovery
experiment.

## Target and scientific accounting

The comparator is the accepted July 2021 WUR BASE/STILT known-background fit:
5,314 frozen site-hours at 25 sites, NAME inner transport, UOB BASE reference
flux, and 240 retained inner states. Its observation comparator is the paired
WUR inner enhancement. Raw WUR BASE flux truth on the corrected inner PARIS
mask is not rescored here. The archived posterior is the replay comparator.

The absolute mean is supplied background plus inner coherent affine plus
`H_inner @ x_inner`. The background is paired full concentration minus inner
enhancement. All outer retained columns, full-minus-inner affine contributions,
BC states and offsets are excluded. The inner lognormal prior is the actual
full prior's marginal, checked in latent Gaussian moments; it is not conditional
on outer states fixed at one. Historical positive flat-cell region IDs were
explicitly mapped to the persisted geometry's zero-based categories, with the
bijection and state order checked before selection.

The likelihood retains the full preparation's rank-512 aggregation factor,
diagonal remainder and reported-error array unchanged, plus five-hour OU
correlation with inferred site amplitudes and HalfNormal(0.75 ppm) priors.
Reported errors are zero in this fixture. Aggregation error is an inherited
Gaussian moment-closure approximation; any unresolved exterior modes remain.
Supplying the background does not remove all exterior uncertainty. The
historical enhancement preparation's different aggregation operator is not used.
OU amplitudes are refitted, so identical covariance construction does not imply
identical realized posterior covariance between runs.

## Validation and sampling

Production implementation revision: `5726d5c9eaa788f79978a61956e118ea9b55e767`.
The implementation is a separate follow-up to PR #994; its reviewed baseline,
mask and PARIS offset fixes remain intact.

The unchanged archived `--check-only` passed. The new API preflight used 75
actual rows and all 240 inner states at flux scales 0.8, 1.0 and 1.2. Absolute
and subtracted normalized log densities differed by at most 2.3e-13, and
residual gradients agreed under identical amplitude/error assumptions. Checks
also verified unchanged H columns, the explicit inner affine, latent prior
marginal, aggregation arrays and saved-input round trips.

Slurm job 19885003 completed with exit code zero in 1h45m19s, using about
3.86 GiB peak process RSS against a 12 GiB request. Four chains each retained
1,000 draws after 1,000 tuning iterations. Maximum parameter R-hat was
1.00703, minimum bulk ESS was 3,891.73, and both sampler blocks had zero
divergences and zero maximum-depth events. The declared convergence gate passed.
The native trace was saved before diagnostics.

Saved absolute means at draws 0, 499 and 999 of the first chain reconstruct as
background plus inner affine plus H times state, with maximum error
1.14e-13 ppm. These are deterministic model means, without sampled observation
noise or an OU realization.

## Comparison with the accepted fit

Flux functionals below are conditional retained-state posterior arithmetic
means on the archived inner PARIS support, including fixed affine functional
contributions. Units are annualized Tg CO2/year, a rate rather than July mass.
The archived draws were reconstructed independently and checked against their
stored functional draws before comparing the new fit. These quantities omit
unresolved native-grid flux uncertainty.

| Functional | Archived mean | Replay mean | Difference / summed MCSE |
| --- | ---: | ---: | ---: |
| GPP | -60,042.29 | -60,045.77 | 0.039 |
| TER | 43,639.41 | 43,673.81 | 0.391 |
| Fossil fuel | 4,209.77 | 4,216.04 | 0.923 |
| Ocean | -53.53 | -53.36 | 0.052 |
| Net biosphere | -16,402.88 | -16,371.96 | 0.569 |
| Total | -12,246.64 | -12,209.28 | 0.701 |

The final column is the absolute difference of posterior means divided by the
sum of the two estimated mean MCSEs. Runs used the same seed, and cross-run
dependence was not estimated; the summed MCSE is used as a conservative scale
without assuming independence. All six differences are below one summed MCSE
(maximum 0.923). This is descriptive implementation-replay evidence, not a
formal multiplicity-adjusted hypothesis test or a new physical uncertainty
interval. The accounting assertions and declared convergence gate are the
pass/fail checks; no additional statistical comparison gate is claimed.

In concentration space, deterministic posterior-mean inner-enhancement RMSE
against the paired WUR reference was 3.0500 ppm, compared with 3.0528 ppm in
the archive. Maximum site-hour difference between the two posterior-mean
predictions was 0.1355 ppm. Adding the same supplied background to prediction
and reference leaves residuals identical; that algebraic identity is not
independent evidence of physical model success.

## Scope and remaining limits

The change supports independent multisector, ordinary correlated CO2,
fixed-OU cached and scalar-sigma routes. Linked CO2/O2 supplied backgrounds and
regeneration of aggregation covariance remain outside this API. No default
scientific prior or error model changes. The fixed background is a diagnostic
control here, not a claim that production backgrounds are known exactly.

Validation also includes 65 final focused baseline tests, 94 existing
model/inference/builder tests, 66 preparation/handoff tests and 20 staged-output
tests, plus changed-path Ruff and whitespace checks. Independent review found
and verified fixes for scalar-cache input selection and PARIS baseline-sector
accounting. The full suite and documentation build were not run.
