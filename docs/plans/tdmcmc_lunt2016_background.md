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
prior parameters. In the paper these are directly the mean and standard
deviation of `log(x_i)`, so

\[
\log x_i \sim \mathcal{N}(\mu_i,\sigma_i^2).
\]

They are not arithmetic-space moments of `x_i`. `theta_y` describes the
data-space error model and is not dimension-dependent.

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

The paper does not specify how the new region's dimension-dependent
`(mu_i, sigma_i)` pair is generated during birth. The legacy code copies the
parent's arithmetic-moment parameters, but after pairs can vary independently
that rule proposes on an equality submanifold and supplies no valid reverse
density for a general state. The planned reference completion instead draws
the new log-space pair from its normalized bounded hyperprior; those proposal
densities then cancel the new region's normalized hyperprior factors. This is
an explicit implementation decision, not a recovered paper setting.

The move proposal selects a nucleus and uses a Gaussian displacement centred on
its current location while carrying the coefficient with it. The paper calls
this symmetric, but does not define rounding, truncation, or rejection at the
finite-grid boundary. The rewrite will use a normalized discrete-Gaussian
categorical kernel and include its forward/reverse normalization explicitly.

The literal deterministic birth-then-death schedule is not retained. A birth-only
step cannot be reversed within its own transition kernel, and likewise for a
death-only step. A seven-state fixed-coefficient counterexample confirmed that
a deterministic birth-then-death composition need not preserve the declared
target. The rewrite therefore uses a 50/50 birth/death mixture at each
structural slot; the equal move-type probabilities cancel in the acceptance
ratio. Two structural slots
per four-step cycle preserve the first rewrite's aggregate structural-attempt
frequency. When the hyperparameter step is added, the declared schedule profile
must be revisited. Unavailable boundary draws remain explicit self-transitions
rather than renormalizing the mixture.

## Legacy scheduler correctness finding

The checked legacy RJMCMC implementation is not a correct general sampler for
its stated posterior with respect to its executed structural schedule. This is
a posterior-invariance failure, not merely the absence of detailed balance.

All five inspected Fortran drivers actively select proposal types with
`modulo(it, n_moves)` while the alternative random-selection statements are
commented out:

- `acrg_hbtdmcmc_uncorr.f90`;
- `acrg_hbtdmcmc_corr.f90`;
- `acrg_hbtdmcmc_evencorr.f90`;
- `rjmcmc_time_uncorr.f90`;
- `rjmcmc_time_corr.f90`.

In each RJMCMC path, birth and death are separate successive one-way steps. A
birth-only kernel cannot reverse itself, nor can a death-only kernel. Systematic
scan is valid when every component kernel preserves the target, but these
structural components do not. A seven-state fixed-coefficient counterexample
gives target mean `k=2.0109`; one literal birth-then-death composition changes
it to `1.6883`, with total-variation error `0.3332`. The corrected
`0.5 * birth + 0.5 * death` mixture preserves the same target to floating-point
precision. One counterexample is sufficient to disprove the deterministic
scheduler as a generally valid posterior sampler.

Consequently, legacy RJMCMC outputs require revalidation. The most direct risk
is bias in posterior `k` and nucleus partitions, with coupled effects possible
for fluxes, intervals, predictions, and inferred hyperparameters. The finite
example does not determine the magnitude or direction of bias in any historical
production inversion, and good predictive fit would not establish posterior
correctness. Fixed-dimensional runs with `rjmcmc=0` are not affected by this
specific defect. Claims about individual paper results should also confirm the
exact code provenance used for those runs.

The rewrite intentionally corrects rather than reproduces this behavior. A
permanent finite regression test covers the legacy counterexample and the
stationarity of the mixed structural kernel. Remaining follow-up is to compare
legacy-emulated and corrected schedules on larger synthetic cases, including
saved-phase/thinning effects, and to validate the general continuous auxiliary
coefficient proposal separately.

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
- Algorithm 1's deterministic birth and death steps are not individually
  target-invariant. The rewrite uses reversible mixed structural steps while
  retaining the paper's proposal densities and the first rewrite's aggregate
  structural-attempt frequency.
- The dimension-changing equations omit a proposal for the new region's
  `(mu_i, sigma_i)` pair. Copying a parent pair is not a valid general
  dimension-matching rule once these pairs are inferred independently. The
  rewrite will draw the pair from its normalized hyperprior and test the full
  forward/reverse flux before exposing a hierarchical sampler.

Birth and death Eqs. (35)-(36) are recovered algebraically by the current
normalized target when the `p(c|k)` and location/count proposal factors are
retained explicitly and allowed to cancel.

## Reproduction profile A: Lunt2016-pseudo

Main-paper Sect. 4, PDF pp. 10-12 / printed pp. 3222-3224:

- EDGAR anthropogenic methane prior regridded to NAME resolution;
- inner domain `56 x 48` native cells, written consistently with the paper's
  longitude-first `391 x 293` full-domain convention;
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

The archived paper inputs are not currently available locally. Until the server
is available, the implementation gates are a scaled-down analytic case and a
test-data-backed NAME/EDGAR substitute with the same declared equations. Neither
is a claim of paper reproduction.

### Local checkerboard benchmark

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

The slow gate uses three seeded comparison runs with 40,000 transitions, a
15,000-row burn cutoff, thinning by 10, and a local-move scale of 1.4. Adaptive
chains share a seeded non-oracle 16-nucleus start with all coefficients at the
prior mean. Assertions cover broad improvement over the prior, fixed versus
variable `k` behavior, and oracle-versus-random fixed-layout ordering; they do
not require an exact partition, golden posterior `k`, or ranking between the
two adaptive methods.

This is a mechanics/recovery benchmark, not a miniature scientific
reproduction. It replaces NAME/EDGAR sensitivities with synthetic smooth
kernels, uses 64 rather than 2688 native cells, and cannot be compared directly
with the paper's posterior `k` or ppb RMSE values.

### Fair fixed-basis and adaptive comparison

The local comparison holds the observations, sensitivities, 5 ppb error,
lognormal coefficient prior, coefficient proposal scale, retained-row logic,
and coefficient proposal opportunities fixed. Every method receives 10,000
coefficient proposals. The movable fixed-`k` and trans-dimensional chains also
receive 10,000 location proposals. Both schedule 20,000 structural attempts,
but those attempts are boundary self-transitions for fixed `k` and can change
dimension only in the trans-dimensional chain. Both adaptive-geometry methods
start from the same seeded non-oracle 16-nucleus layout with all-one
coefficients.

The fixed-basis comparator is coefficient-only Metropolis-Hastings using the
same normalized target terms, not RHIME/PyMC. The oracle case is deliberately
given the true layout and is a lower-bound reference. Three random fixed layouts
are independent model replicates and are summarized separately rather than
pooled as posterior samples.

Results across three seeded runs are median `[range]`:

| Method | Prediction RMSE | Grid RMSE | Spatial correlation | High-low contrast |
| --- | ---: | ---: | ---: | ---: |
| All-ones prior | 15.293 | - | - | - |
| Oracle fixed truth layout | 2.045 `[1.980, 2.050]` | 0.058 `[0.054, 0.068]` | 0.994 `[0.993, 0.995]` | 1.025 `[1.013, 1.048]` |
| Movable fixed `k=16` | 4.213 `[2.798, 4.535]` | 0.298 `[0.216, 0.322]` | 0.831 `[0.804, 0.907]` | 0.868 `[0.842, 0.912]` |
| Random fixed `k=16` layouts | 10.155 `[9.848, 12.052]` | 0.482 `[0.478, 0.589]` | 0.415 `[0.272, 0.461]` | 0.302 `[0.259, 0.376]` |
| Trans-dimensional `k=8..28` | 3.523 `[2.477, 6.152]` | 0.425 `[0.155, 0.438]` | 0.679 `[0.602, 0.953]` | 0.786 `[0.539, 0.964]` |

The oracle behaves as expected, and arbitrary fixed layouts lose substantial
information. Both adaptive methods improve strongly on the prior and visit the
declared geometry space; the trans-dimensional chains visit multiple `k`
values. The short benchmark does not establish convergence or superiority of
one adaptive method over the other. In particular, its median prediction RMSE
favours the trans-dimensional runs while its median grid metrics are mixed.

Runtime is intentionally not compared. The reference fixed-layout helper still
rebuilds full Voronoi state, whereas a standard fixed-basis inversion would
pre-aggregate its design matrix. A fair production timing comparison requires
that optimized fixed-design path and matched convergence diagnostics.

### Test-data-backed NAME/EDGAR checkerboard

The repository contains enough raw emissions-side test data for a second
benchmark that is closer to the paper's forward model without using any
archived observations or boundary-condition products:

- `flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc` contains one
  EDGAR7/UKGHG methane flux field on the `293 lat x 391 lon` EUROPE grid;
- the TAC and MHD NAME files each contain 168 hourly footprints for 1--7
  January 2019 on that grid;
- consecutive six-hour means give 28 rows per site and 56 rows in total;
- `lat[157:205]`, `lon[244:300]` defines a `48 lat x 56 lon` NWEU crop spanning
  47.467--58.465 degrees north and -12.012--7.348 degrees east. It contains the
  TAC/MHD sites and the locations of the unavailable RGL/TTA paper sites;
- a regular `4 x 4` truth uses `12 lat x 14 lon` blocks with alternating
  scalings `0.5` and `1.5`.

The native contribution matrix is calculated positionally as

\[
G_{tg}=fp_{tg}\,F_g\,10^9,
\]

where the factor converts mole fraction to ppb. The raw flux and footprint
coordinate values differ by up to about `7.6e-6` degrees. Ordinary labelled
xarray multiplication silently intersects only part of the grid, so the loader
must first validate equal shapes and numerically close coordinates and then
multiply by position or explicitly override alignment.

The first pseudo-observation accounting contract was

\[
y_{\mathrm{full}}=b_{\mathrm{outer}}+G_{\mathrm{inner}}x_{\mathrm{truth}}
+\epsilon,\qquad
y_{\mathrm{inversion}}=y_{\mathrm{full}}-b_{\mathrm{outer}},
\]

with independent `epsilon ~ Normal(0, 5 ppb)`. This subtraction was useful for
validating the decomposition. The current benchmark instead fits `y_full`
directly and includes the seven-column outer-emissions design as an
always-active inferred predictor with unit prior means. No observed mole
fractions, boundary curtains, `bc_mod`, or boundary-condition file enters the
calculation.

The packaged EUROPE InTEM map is useful as an accounting layout but is not the
paper mask. It has labels zero through five for six fixed outer regions and a
`183 x 128` maximum-label inner rectangle. The `48 x 56` pseudo crop lies
inside that inner class, leaving a seventh fixed remainder between the crop and
the InTEM outer regions. Aggregating those seven fixed blocks and setting their
coefficients to one must equal `b_outer`; jointly inferring their coefficients
is now handled by the experimental composite predictor.

This substitute is deliberately limited. It uses two rather than four sites,
one January 2019 week rather than May--June 2014, 56 rather than 942 rows, and
EDGAR7 with UKGHG replacement rather than the paper's EDGAR version. The
all-ones versus truth noise-free prediction RMSE is 6.57 ppb, but the oracle
sixteen-block design is ill conditioned and several edge blocks are nearly
unseen. Assertions should therefore emphasize noise-free prediction RMSE,
including site-specific diagnostics. Unweighted grid RMSE, spatial correlation,
contrast, and sampled `k` are diagnostics only and must not be compared with
the paper's reported field or posterior-`k` results.

The original seeded 20,000-transition subtraction gate gave prediction RMSEs
of 6.57 ppb for the all-ones prior, 1.51 ppb for a fixed inversion given the
true sixteen rectangles, 1.73 ppb for a non-oracle sixteen-region
sensitivity-weighted quadtree basis, and 1.69 ppb for the trans-dimensional
inversion. The current joint-inner/outer implementation uses a five-slot
schedule and therefore gives each method 4,000 dynamic and 4,000 fixed
coefficient opportunities in 20,000 transitions. Its calibrated long-run
results must be regenerated before quoting replacement RMSEs. Sampled `k`
remains a mixing/posterior diagnostic governed by its declared prior, not a
field-recovery score. Movable fixed-`k` and random-layout controls remain useful
follow-up comparisons.

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
- continuous auxiliary-coefficient balance validation by varied pointwise
  checks and independent numerical quadrature including invalid proposal mass;
- a fixed outer schedule of coefficient, two reversible 50/50 birth/death
  mixture steps, and a configurable global or paper-style local nucleus move;
- NumPy/Numba parity and fixed-seed replay;
- filtered RHIME `fp_x_flux` adapter with explicit fixed design/offset inputs;
- an always-active fixed predictor whose coefficients have lognormal priors and
  a separate proposal slot when present;
- globally phased retained-draw collection and exact in-memory continuation;
- per-draw native-grid reconstruction, posterior mean/quantiles, retained-row
  selection, posterior-mean prediction, and comparison-vector RMSE;
- an independently enumerated fixed-`k`, fixed-coefficient location kernel that
  verifies proposal row normalization, self-transition mass, detailed balance,
  and stationarity on an irregular finite grid;
- an exact seven-state mixed-`k` birth/death subkernel that verifies nucleus-set
  combinatorics, move-count factors, mixture boundary self-mass, detailed
  balance, and stationarity at a special fixed coefficient/proposal density;
- total-prediction summaries that separate dynamic-inner, fixed-block, and
  fixed-offset contributions.

Next paper-first gaps:

1. opt-in per-region log-space emissions hyperparameters and their fully
   dimension-matched structural proposal cycle;
2. fixed boundary-curtain design inputs once reliable data are available;
3. grouped/site-block `sigma_y` and AR(1) `tau` likelihood;
4. real-data preparation and comparison once archived inputs are available.

The hierarchy implementation must require explicit bounded-uniform hyperprior
bounds, initialized active pairs, and proposal scales. None should be labelled
as a Lunt configuration until the archived settings return. The fixed-prior
pseudo-data/checkerboard path continues to use declared arithmetic lognormal
moments and must preserve its current seeded behavior.

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
