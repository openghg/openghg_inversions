# Lunt et al. (2016) trans-dimensional inversion background

## Purpose and status

This note defines the paper-first scientific target for the NumPy/Numba
trans-dimensional MCMC rewrite. It is based on a full review of:

- `Lunt_Estimation of trace gas fluxes with objectively.pdf`, the 17-page main
  article, Geoscientific Model Development 9, 3213-3229 (2016),
  doi:10.5194/gmd-9-3213-2016;
- `Lunt_Estimation of trace gas fluxes with objectively1.pdf`, the 5-page
  supplement.

The paper is normative for the scientific model. Printed acceptance equations
are treated as claims to verify against the normalized target, not as code to
copy literally. The legacy Fortran remains a behavioural comparison backend.
Current RHIME is a separate integration profile.

## Two implementation tracks

### Paper track - primary

The first reproduction target is the non-hierarchical pseudo-data example in
Sect. 4 of the main paper. It isolates spatial model selection from the later
hierarchical error model:

- one spatial emissions-scaling field;
- native-grid Voronoi nuclei;
- fixed lognormal emissions prior;
- fixed independent 5 ppb model-measurement error;
- variable number and placement of regions;
- posterior summaries reconstructed on the native grid.

The real-data hierarchy in Sect. 5 follows only after the pseudo-data target and
proposal kernels pass exact small-state tests.

### RHIME track - secondary adapter

Current RHIME provides prepared observations, fine-grid `fp_x_flux`, fixed
basis/boundary products, PyMC priors, and an enhancement-proportional independent
model-error model. These components are useful, but they do not define the
Lunt (2016) target. RHIME integration should adapt a declared paper or RHIME
profile to shared numerical kernels rather than blend the two likelihoods.

## Paper-defined state and forward model

The native inversion relationship is introduced as

\[
y = Hx + \epsilon
\]

in main-paper Eq. (2), PDF p. 1 / printed p. 3213. The partition state is

\[
m=(c,x),
\]

where `c` contains nucleus coordinates and `x` the emissions scaling associated
with each nucleus (Eq. 4, PDF p. 4 / printed p. 3216).

Every native CTM grid cell belongs wholly to its nearest nucleus. Nuclei are
restricted to native-grid cell centres, no two occupy the same cell, and the
native CTM resolution is the maximum posterior resolution (Sect. 2.1.1,
PDF p. 4).

Operationally, the paper's forward model is

\[
\hat y_t = b_t + \sum_{g=1}^{K} G_{tg} x_{z_g(c)},
\]

where `G` is the native-grid contribution matrix (footprint multiplied by prior
emissions), `z_g(c)` maps fine cell `g` to an active Voronoi region, and `b`
contains fixed outer-domain and boundary contributions when present. Equivalently,
the partition aggregates `G` into `H(c)` and evaluates `H(c) x`.

The full hierarchical target is main-paper Eq. (8), PDF p. 5 / printed p. 3217:

\[
p(m,\theta_x,\theta_y,k\mid y) \propto
p(y\mid m,\theta_y,k)
p(m\mid\theta_x,k)
p(\theta_x\mid k)
p(k)
p(\theta_y).
\]

`theta_x` is dimension-dependent: each active emissions region has its own
prior parameters. `theta_y` describes the data-space error model and is not
dimension-dependent.

## Priors

Main-paper Sect. 2.3.1 and Eqs. (13)-(19), PDF p. 6 / printed p. 3218, define:

1. A canonical unordered set of `k` distinct nuclei from `K` native cells:

   \[
   p(c\mid k) = \binom{K}{k}^{-1}.
   \]

2. A bounded uniform prior on the number of active regions.

3. A lognormal emissions-scaling prior. Prior inventory scaling is one
   everywhere and is independent of nucleus location.

4. Bounded uniform hyperpriors for emissions-prior log means/standard
   deviations and for the model-error standard deviations and correlation
   timescales.

The printed support in Eq. (17) is `k_min < k <= k_max` with normalizer
`k_max - k_min`, whereas the prose describes bounds and the Fortran permits a
state at `k_min`. The rewrite currently uses the conventional inclusive support
`k_min <= k <= k_max`, with `k_max - k_min + 1` values. Any paper-reproduction
configuration must record that choice explicitly.

## Likelihood and error model

The likelihood uses the quadratic form

\[
\Phi(m)=(y-\hat y)^T R^{-1}(y-\hat y),
\]

with

\[
R=\Sigma Q\Sigma.
\]

These are Eqs. (23)-(26), PDF p. 7 / printed p. 3219.

- `Sigma` contains grouped model-measurement standard deviations.
- `Q` is an exponential time-correlation matrix.
- For regularly spaced observations, `q=exp(-delta_t/tau)` and
  `Q[i,j]=q**abs(i-j)`.
- Eq. (29) gives the tridiagonal AR(1) precision.
- Eq. (30) gives
  `det(R) = prod(sigma_i**2) * (1-q**2)**(N-1)`.
- Sites are treated as spatially independent, producing block-diagonal `R`.

The cheap analytic precision assumes regular sampling. Gaps from downtime or
filtering require explicit irregular-time treatment, block splitting, or a fixed
correlation timescale (discussion, PDF p. 15 / printed p. 3227).

## Proposal schedule and kernels

Algorithm 1, PDF p. 9 / printed p. 3221, cycles deterministically through five
proposal types:

1. emissions coefficient update;
2. hyperparameter update;
3. birth;
4. death;
5. nucleus move.

Emissions and hyperparameter updates use additive centred Gaussian proposals.
Birth selects a vacant native cell uniformly and proposes the new scaling by an
additive Gaussian perturbation of the scaling in the pre-birth owning cell.
Death selects a nucleus uniformly and is the reverse transformation. The paper
sets the birth/death Jacobian to one. Proposal ratios are given in Eqs. (20)-(22),
PDF p. 7 / printed p. 3219.

The move proposal selects a nucleus and uses a Gaussian displacement centred on
its current location while carrying the coefficient with it. The paper calls
this symmetric, but does not define rounding, truncation, or rejection at the
finite-grid boundary. The rewrite will use a normalized discrete-Gaussian
categorical kernel and include its forward/reverse normalization explicitly.

Impossible birth/death attempts at dimension boundaries remain scheduled
self-transitions. This avoids silently changing the birth/death move-type ratio.

## Acceptance-equation audit

The normalized target and proposal kernel take precedence when the printed
formula is internally inconsistent:

- Eq. (18) includes the lognormal `1 / x` factor, but the fixed-dimensional
  emissions acceptance in Eq. (31) omits the expected `x_old / x_new` ratio for
  an additive proposal. The legacy `calc_pdf` routine includes it, as does the
  rewrite's normalized prior.
- Eq. (23) prints an incomplete multivariate Gaussian `2*pi` normalizer. It is
  constant because the observation dimension does not change, but the rewrite
  retains the full normalized likelihood.
- Eq. (33) prints `|R| / |R'|`; the Gaussian density implies the square-root
  determinant ratio. This must be resolved with a target-difference test before
  implementing error-hyperparameter updates.
- A linear-Gaussian birth proposal can generate a nonpositive scaling. The
  rewrite treats it as a zero-target rejected proposal.
- Discrete finite-grid moves are not automatically symmetric at boundaries.

Birth and death Eqs. (35)-(36) are recovered algebraically by the current
normalized target when the `p(c|k)` and location/count proposal factors are
retained explicitly and allowed to cancel.

## Reproduction profile A: Lunt2016-pseudo

Main-paper Sect. 4, PDF pp. 10-12 / printed pp. 3222-3224:

- EDGAR anthropogenic methane prior regridded to NAME resolution;
- inner domain `56 x 48` native cells;
- checkerboard truth with scalings `0.5` and `1.5`, prior scaling `1`;
- four sites: TTA, MHD, TAC, RGL;
- May-June 2014, 6-hour averages, 942 observations;
- independent Gaussian noise with standard deviation 5 ppb;
- non-hierarchical fixed 5 ppb model-measurement error;
- fixed emissions-prior uncertainty of 100% of initial scaling;
- initial `k=40`, bounded-uniform `k` described in prose as 5 to 500;
- posterior reconstructed and summarized on the native grid.

Paper validation targets:

- posterior `k = 29 +/- 7` versus 16 checkerboard truth regions;
- posterior-mean fine-grid prediction RMSE about 1.0 ppb against the
  **noise-free** pseudo-observations;
- expert 94-region grid RMSE about 2.0 ppb;
- correct 16-region layout RMSE about 0.6 ppb;
- random fixed layouts near the posterior mean `k` give about 6 ppb RMSE.

The archived data are not currently available locally. Until the server is
available, the implementation gate is a scaled-down synthetic case with the
same declared equations, not a claim of paper reproduction.

### Provisional local checkerboard benchmark

Before the archived inputs return, use two deliberately separate checks:

1. A fast structural test on an `8 x 8` grid with a `4 x 4` checkerboard of
   `2 x 2` blocks, alternating scalings `0.5` and `1.5`. It should verify the
   declared truth partition, prior-flux-weighted forward calculation, seeded
   independent noise, native-grid reconstruction, and noise-free RMSE formula.
2. A slow seeded recovery benchmark on the same grid using spatially smooth,
   positive synthetic footprint rows and the paper's arithmetic lognormal
   prior moments `(mean=1, sd=1)`. Use the local nucleus move, retain draws only
   after burn-in, and compare the posterior-mean grid and prediction with the
   all-ones prior baseline.

The recovery gate should assert robust improvements (lower noise-free
prediction RMSE, correct checkerboard contrast direction, and useful spatial
correlation), not an exact sampled partition or a golden posterior `k` from one
seed. Exact thresholds, iteration count, and proposal scales must be calibrated
across several fixed seeds before becoming CI assertions.

This is a mechanics/recovery benchmark, not a miniature scientific
reproduction. It replaces NAME/EDGAR sensitivities with synthetic smooth
kernels, uses 64 rather than 2688 native cells, and cannot be compared directly
with the paper's posterior `k` or ppb RMSE values.

## Reproduction profile B: Lunt2016-real

Main-paper Sect. 5.1, PDF pp. 13-14 / printed pp. 3225-3226, plus Supplement S1:

- March 2014 methane at MHD, TAC, RGL, and TTA;
- 1-minute observations averaged to 4-hour periods, 727 observations;
- dynamic inner domain `64 x 52` within the full `391 x 293` NAME domain;
- six fixed outer-domain emissions regions;
- four fixed-dimensional N/S/E/W boundary-curtain scaling parameters;
- one model-error scale per 7-day period, further divided by a 30% local-footprint
  threshold;
- site-block exponential temporal correlation with inferred `tau`;
- inferred emissions log-mean/log-standard-deviation hyperparameters;
- uniform `k` described as 5 to 800;
- 100,000 discarded burn-in iterations plus 500,000 iterations, saving every
  100th state.

Reported comparison targets include posterior `k=201 (145, 248)`, UK emissions
`2.28 (2.04, 2.52) Tg/yr`, Ireland emissions `0.49 (0.39, 0.60) Tg/yr`, and mean
correlation timescale `15 (7, 37) h`.

The supplement contains no additional RJMCMC equations or tuning details. Its
principal contribution is the fixed-block observation model:

- four MOZART boundary curtains, one scaling per N/S/E/W edge;
- six fixed far-field emissions regions outside the dynamic subdomain;
- only the inner subdomain participates in birth/death/move proposals.

## Current implementation alignment

Already implemented:

- canonical sorted nucleus subsets and `1 / comb(K, k)` prior;
- explicit normalized prior over `k`;
- normalized lognormal coefficient density;
- normalized independent Gaussian likelihood;
- coefficient, birth, death, globally symmetric move, and normalized local
  discrete-Gaussian move proposals;
- explicit proposal and target terms with pointwise birth/death balance tests;
- deterministic proposal schedule without the hyperparameter slot, with a
  backwards-compatible global move or paper-style local move selection;
- NumPy/Numba parity and fixed-seed replay;
- filtered RHIME `fp_x_flux` adapter;
- per-draw native-grid reconstruction, posterior mean/quantiles, retained-row
  selection, posterior-mean prediction, and comparison-vector RMSE;
- an independently enumerated fixed-`k`, fixed-coefficient location kernel that
  verifies proposal row normalization, self-transition mass, detailed balance,
  and stationarity on an irregular finite grid.

Next paper-first gaps:

1. a mixed-move finite transition oracle extending the completed location-only
   oracle to trans-dimensional birth/death accounting;
2. paper profile/configuration object and provenance metadata;
3. paper-like two-dimensional checkerboard benchmark;
4. sampler-side retained-draw/checkpoint output (postprocessing already accepts
   saved-row burn-in and thinning selections);
5. dimension-dependent emissions hyperparameters and their proposal cycle;
6. composite predictor with fixed outer emissions and boundary blocks;
7. grouped/site-block `sigma_y` and AR(1) `tau` likelihood;
8. real-data preparation and comparison once archived inputs are available.

Temporal partitions, multisector inference, and parallel tempering are not part
of the Lunt (2016) reproduction milestone. The main paper describes parallel
tempering only as possible future work.

## Data checklist for server recovery

Search for the following before attempting numerical reproduction:

- May-June 2014 6-hour NAME sensitivities for TTA/MHD/TAC/RGL;
- March 2014 4-hour NAME sensitivities and observations;
- EDGAR methane field on the `391 x 293` NAME grid;
- the `56 x 48` pseudo-data subdomain and checkerboard mask;
- the `64 x 52` real-data dynamic subdomain mask;
- six fixed outer-region definitions;
- four boundary-curtain sensitivity time series;
- site/time grouping indices for 7-day and local-influence error scales;
- exact prior/hyperprior bounds, proposal scales, seeds, and initial nuclei;
- any stored paper-era posterior traces or summary fields.

Every discovered artifact should be recorded with path, checksum, dimensions,
units, coordinate ordering, and its role in the forward model.
