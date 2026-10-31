# Retained biosphere prior: implementation replay

## What was tested

This is an implementation comparison against frozen retained-state experiments,
not a new calibration of physical biosphere or atmospheric boundary uncertainty.
The implementation is based on current remote devel `80782dd4`, rather than the
historical checkout `c913da1a`. The new branch is independent of draft PRs
[#788](https://github.com/openghg/openghg_inversions/pull/788) and
[#790](https://github.com/openghg/openghg_inversions/pull/790), whose current
prototype code and documentation were inspected before choosing the base.
Neither prototype is copied into the released implementation. #788's exact
coordinate transformations preserve one joint target; its Gaussian family
switch is a scientific prior change. #790's mean-preserving moment matching is
relevant, but its GPP/Ra/Rh decomposition is outside this work. Existing GPP/TER
inputs do not identify separate Ra/Rh fields.

The intervention changes one aggregate residual-gross arithmetic covariance
direction. The retained transport, affine reconstruction, reporting weights and
Gaussian aggregation-error moment approximation are fixed. No native prior or
transport reduction is regenerated. Positive lognormal is the primary family;
the Gaussian option is a signed-template diagnostic with protected lognormal
states and exact arithmetic cross-covariance.

## Terminology, reference and truth

“Residual gross” means the error of the best linear prior prediction of gross
exchange from net exchange and every protected coefficient. For lognormal
coefficients this does not describe a full nonlinear conditional distribution.
This term is defined here; adding it to the campaign glossary remains a
campaign documentation follow-up.

The reference flux is UOB BASE. Functional totals and uncertainties below are
annualized Tg CO2/year for the common inner PARIS support, using the archived
July reporting weights. Calibration uses only the supplied prior moments. No
truth or posterior fit is used to choose the uncertainty multiplier.

The accepted historical flux scores use raw WUR BASE inner flux truth for
July 2021 on the common PARIS domain, averaged over all 744 July hours. The
new sampling comparison uses the accepted posterior means and their Monte Carlo
standard errors (MCSE), rather than refitting or calibrating to that truth.
Concentration observations are the paired WUR BASE/STILT enhancements at 5,314
site-hours and 25 sites. They are not flux truth.

## Analytic replay

All ten file hashes in the full comparison's `staging.json` were verified.
The full prior arrays have SHA256
`47bfd498b66dc00bfbd70cccd3543ec0f1b22e63603a48d72402e6717a8d47f1`.
The labelled prior contains 264 states; 132 inner biosphere coefficients may
change and all other 132 coefficients are protected. There are 133 independent
controls, including net exchange. No redundant controls occur in this fixture.

At multiplier 0.5 the new implementation reproduces the saved covariance with
maximum absolute difference **0** in the recorded environment. Maximum absolute
change in control/state covariance is `1.73843e-13` in the controls' declared
units times coefficient units; the protected-state covariance rows are exactly
unchanged. The identity multiplier reproduces the original covariance exactly
and reuses the same correlated lognormal prior. An absolute residual-SD target
and the dimensionless multiplier produce matching covariance.

The table gives analytic prior standard deviations, all in annualized
Tg CO2/year; the values are fixture-specific reproduction targets.

| Functional | Existing SD | Requested SD |
| --- | ---: | ---: |
| Net biosphere | 8313.353249 | 8313.353249 |
| Residual gross | 8212.589028 | 4106.294514 |
| Marginal gross | 8313.353249 | 4304.286043 |
| GPP | 6318.215261 | 5222.413533 |
| TER | 5402.961980 | 4067.647623 |

Explained gross variance is `1665223.708772 (Tg CO2/year)^2`. Arithmetic and
latent lognormal covariance factorizations pass without repair. Mean GPP
`-63643.990027`, TER `47635.560928`, net `-16008.429099`, and gross
`111279.550955` remain unchanged, in the same flux units. Full functional
correlations are saved in the calibration artifact, including the existing
net/gross covariance. Twenty thousand independent prior draws provide a
secondary functional-SD check: the maximum relative SD difference was 0.56%,
with no negative coefficients. The analytic moments remain authoritative.

## Executable checks

The new focused tests exercise identity and rank-one covariance equations,
factor/absolute-target agreement, invalid lognormal moments and infeasible
marginal targets, redundant/zero controls, labelled permutations, unit changes,
shared lazy graph evaluation, Gaussian nonzero cross-covariance, exact
lognormal marginal selection, serialization and flux reconstruction.
Full and enhancement graph fixtures and the linked shared-state recipe verify
ordinary/cached predictions and normalized densities. Native linked prior
prediction checks the runner's calibration provenance and reconstructed output.
All 26 new focused checks passed locally before batch validation. Changed-path
Ruff, a targeted Mypy run across eleven changed source files, shell syntax and
`git diff --check` passed. A scoped Sphinx build of the new guide and both API
pages passed with warnings treated as errors, and its rendered mathematical
blocks and example download were checked. A first whole-site attempt used an
isolated tool environment without scientific imports and failed; it is not
reported as a full documentation build pass.

The generic calibration example is `examples/biosphere_prior/calibrate.py`.
`replay_calibration.py` adapts frozen NPZ data without importing historical
runners; `replay_sampling.py` uses current builders and preserves archive inputs.
All adapters require explicit input/output paths and reject existing output
directories. The frozen audit script, which writes alongside itself, was not
rerun or modified. Existing experiments and artifacts are unchanged.

## Bounded sampling comparison

The planned array crosses original/shrunk arithmetic covariance with lognormal
and diagnostic Gaussian biosphere families, using the enhancement preparation.
Each fit has two chains, 300 tuning plus 300 retained draws, seed 20260924,
state target acceptance 0.99 and amplitude target acceptance 0.9. The Gaussian
arm is additional diagnostic coverage; no physical positive-process claim is
made for negative coefficients.

Enhancement uses the exact selected marginal of the full prior, with 240 inner
states and its own paired rank-512-plus-diagonal aggregation covariance.
No BC, exterior flux or observation-offset states are fitted. The OU timescale
is fixed at five hours and independent site amplitudes have HalfNormal(0.75 ppm)
priors. Reference/observations/operators and error terms are common to all
arms. The accepted original comparator is the r2 retry; the accepted shrunk
comparator is r1. The first r1 original fit is excluded.

Each run saves environment versions, immutable input hashes, sampler settings,
functional posterior means/SDs/MCSEs, R-hat, ESS, divergences, maximum-depth
events and the fraction of negative biosphere coefficients. Comparison uses
mean differences divided by conservative summed MCSEs; identical chains are
not required. Bounded-chain convergence is reported rather than assumed.
Batch results will be added after the registered Slurm callbacks complete.

## Interpretation and limits

The historical full-model endpoints were subsequently found defective. Their
archived fits are implementation-replay evidence only. No new full-model
physical BC claim is made here. Any fresh full scientific comparison must use
repaired endpoints in both treatment and control, while preserving WUR
observations and surface-flux operators. The enhancement model omits BC and is
unaffected by that endpoint correction.

Better allocation between GPP and TER under a stronger gross prior is
regularization toward the reference, not newly established observational
identification. Preserved net mean/variance do not preserve every higher net
moment. Matching Gaussian and lognormal arithmetic moments does not make the
two joint distributions equivalent. The experimental factor 0.5 is never
installed as a production default.
