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

The following dimensionless arithmetic correlations come from the saved affine
functionals. They are covariance-derived correlations, not latent lognormal
correlations or a nonlinear conditional-distribution statement.

| Functional pair | Existing correlation | Requested correlation |
| --- | ---: | ---: |
| GPP / TER | +0.000000 | +0.595315 |
| net / gross | -0.155224 | -0.299802 |
| net / residual_gross | +0.000000 | -0.000000 |
| gross / residual_gross | +0.987879 | +0.954001 |

GPP is signed in this table. Net/gross covariance remains fixed; its
correlation changes because the marginal gross SD changes.

Existing arithmetic minimum eigenvalue: `0.0111633141`.
Requested arithmetic minimum eigenvalue: `0.00336437678`.
Existing latent lognormal minimum eigenvalue: `0.0109799459`.
Requested latent lognormal minimum eigenvalue: `0.00328670521`.
All four covariances are positive definite for this fixture.

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

Broader compatibility validation completed in Slurm job `19861121` against
implementation revision `a11381d1`: **347 tests passed on Python 3.12.11 and
347 passed on Python 3.13.7**. Both locked-dependency tox environments succeeded.
Coverage included the new calibration/model tests, existing correlated-state,
cached CO2 and linked fixed-OU recipes, model inference, affine outputs,
linked saved handoffs, prepared runners and centered boundary regressions.
The job completed in 22 minutes 38 seconds. This is relevant broader coverage,
not a claim that the entire repository suite was run.

## Bounded sampling comparison

All four arms in Slurm array `19861123` completed successfully against
implementation revision `a11381d1`; no model code changed during the array.
The completed array crosses original/shrunk arithmetic covariance with
lognormal and diagnostic Gaussian biosphere families. Each fit used two
chains, 300 tuning plus 300 retained draws, seed 20260924, state target
acceptance 0.99 and amplitude target acceptance 0.9.

All four arms had **zero divergences and zero maximum-tree-depth events** for
both NUTS steps. The following dimensionless diagnostics summarize the sampled
flux coefficients and site amplitudes. “Functional R-hat” is the maximum over
the six reported flux functionals. The negative percentage counts negative
biosphere coefficient entries over all selected coefficients and retained
chain/draws; it is not the percentage of draws containing any negative entry.

| Arm | Maximum coefficient/amplitude R-hat | Minimum bulk ESS | Maximum functional R-hat | Negative biosphere entries |
| --- | ---: | ---: | ---: | ---: |
| lognormal_original | 1.0353 | 488.1 | 1.0118 | 0.00% |
| lognormal_shrunk | 1.0236 | 505.3 | 1.0164 | 0.00% |
| gaussian_original | 1.0530 | 591.2 | 1.0042 | 15.27% |
| gaussian_shrunk | 1.0284 | 597.1 | 1.0118 | 14.97% |

These short chains do **not** pass the historical all-coefficient R-hat <= 1.01
convergence gate. They are bounded implementation-replay evidence, not a
replacement for the accepted four-chain scientific fits. The Gaussian
coefficients frequently become negative, confirming the signed-template
limitation; both lognormal arms stay strictly positive.

The lognormal replay means were compared with the accepted r2 original and
r1 shrunk archives. Across GPP, TER, fossil fuel, ocean, net biosphere and total
flux, the largest absolute mean difference was **1.342 times the conservative
sum of the new and archived MCSEs** (the shrunk ocean functional). Every
GPP/TER/net/fossil comparison was below 0.77 summed MCSEs. This assesses
agreement at the available Monte Carlo precision; it does not assume identical
chains or establish full convergence.

The table shows the positive-lognormal treatment effect on posterior means.
All flux values and MCSEs are annualized Tg CO2/year on the common inner PARIS
support. The ratio divides the shrunk-minus-original mean change by the sum of
the two mean MCSEs, not by posterior SD or a quadrature-combined MCSE.

| Functional | Original mean | Shrunk mean | Change | Summed MCSE | Change / summed MCSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPP | -60418.83 | -62672.10 | -2253.27 | 191.20 | -11.79 |
| TER | 43972.71 | 46166.95 | +2194.24 | 197.72 | +11.10 |
| NET_BIOSPHERE | -16446.12 | -16505.15 | -59.03 | 137.30 | -0.43 |
| FF | 4210.14 | 4190.94 | -19.20 | 16.49 | -1.16 |

The net change, -59.03, is below its summed MCSE of 137.30. Separate GPP and
TER allocation changes are about eleven summed MCSEs. The Gaussian diagnostic
also shifts GPP/TER substantially while its net change is -51.90 against a
summed MCSE of 129.81. This reproduces the direction of the regularization
finding without asserting new observational identification.

Enhancement consumes the exact selected marginal of the full prior, with 240
inner states and the paired enhancement product's own rank-512-plus-diagonal
aggregation covariance. No BC, exterior flux or observation-offset states were
fitted. The OU timescale is five hours and independent site amplitudes have
HalfNormal(0.75 ppm) priors. The observation product, transport, affine term,
reference and error terms are common to all arms. The Gaussian adapter checks
that the fixture's omitted protected states have zero biosphere covariance
before reproducing its selected conditional diagnostic.

Three saved draws per arm (both chains and the first/last retained draw) were
reopened and reconstructed as `fixed_prior_contribution + H @ flux_scaling`.
The maximum difference from the saved concentration contribution was **0 ppm**;
without baseline/offset terms it also equalled `modelled_concentration`
exactly. All four saved traces contain the calibration family, baseline and
changed covariances, functional support, full parent calibration and labelled
marginal selection. Original arms record multiplier 1 and shrunk arms 0.5.

The replay environment was Python 3.13.7, PyMC 6.3.2, PyTensor 3.3.2,
NumPy 2.2.6, SciPy 1.15.3 and OpenGHG
`0.17.0.post953.dev0+b9a9d1f2`. The uv lock SHA256 is
`6915527f896bd51d7abdc5b6de236a0c34a2b4ad0994eb4f2862565d77787939`.
All arms used the accepted original enhancement preparation SHA256
`bca5bef3e3f492c5f4ab302878b8c7df29b82c3e2db56b36d149b20b33a5f34f`.
The archived prior digest is recorded above.

The preserved local output root is
`OGI/.codex-worktrees/calibrated-biosphere-prior/build/biosphere-prior/replay-r1/`.
It contains `comparison.json`, `output_checks.json`, and one directory per
arm with `posterior.nc`, `report.json`, functional/state diagnostics and the
functional diagnostic CSV. Logs are in the adjacent `validation/` directory.
The table records report-file SHA256 identities for these saved outputs.

| Arm | Report SHA256 |
| --- | --- |
| lognormal_original | `ae01e4e400b1f8bbed7b250c2845ad95b6c2ceebed15f47d3a30b502e71ef629` |
| lognormal_shrunk | `8b6bbb98736a77c8024e81e40e2c9aed709f7c0e92388d447af59af1ec050e95` |
| gaussian_original | `4c9d1e45d10cddcd9e91d0408bfdf5424160857f8f5a98f6fba20fd1e8ef17d8` |
| gaussian_shrunk | `ff3aa33fed85851e1650fb3cbdd0dd03dff57c1a81b7e55e79536cfe0ca05834` |

Accepted score identities used by the comparison were
`09e028ef8152c34b62f4c4e7366b61579e525058fcea481d1278b2f16a600e23`
for r2 original and
`7ad4b999801064600df86f323676b47ee272ba02e03537d94f1971814e3e9b50`
for r1 shrunk. The failed r1 original fit was excluded. The comparator script
records these identities and all six functional comparisons without changing
the archived scores or recalibrating against truth.

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
