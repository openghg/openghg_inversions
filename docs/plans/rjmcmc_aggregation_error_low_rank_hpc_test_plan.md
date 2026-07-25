# Aggregation-error Gaussian hybrid HPC test plan

## Purpose and promotion boundary

This plan validates a normalized low-rank Gaussian approximation to the
conditional aggregation error induced by hiding native-cell allocations
inside active regions. It is the first bounded comparator before a learned
likelihood estimator (NLE).

The approximation has two distinct promotion levels:

1. **fixed partition:** relative likelihood and posterior accuracy are
   sufficient; and
2. **structural posterior:** absolute normalized evidence and cross-partition
   tower consistency are additionally required.

Passing the first level does not license posterior claims about \(P\) or
\(K\). Under one common native model, exact reduced likelihoods have identical
evidence and the posterior structural weights equal their declared prior
weights. Any material learned or approximate evidence preference is
approximation leakage unless a partition-indexed scientific prior or
discrepancy model is explicitly declared.

Run only experimental tests. Preserve every failed artifact and publish
completion markers last. Do not write experimental outputs to
`PARIS_inversions`.

Fill in before launch:

```text
branch: codex/rjmcmc-topology-conditioned-hmc
candidate revision: <git rev-parse HEAD>
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_aggregation_error/<revision>
```

## Model identity

For native masses \(X\), active region masses
\(A_j=\sum_{i\in G_j}X_i\), and one additive native concentration field
\(\alpha_i\),

\[
\left(X_i/A_j\right)_{i\in G_j}\mid A,P
\sim \operatorname{Dirichlet}\{(\alpha_i)_{i\in G_j}\}.
\]

The nominal renderer is the exact conditional mean. The conditional
aggregation residual has zero mean and covariance

\[
C_{\mathrm{agg},P}(A)=
\sum_j \frac{A_j^2}{\alpha_{G_j}+1}
H_j\{\operatorname{diag}(u_j)-u_ju_j^\mathsf T\}H_j^\mathsf T,
\qquad
u_{ji}=\alpha_i/\alpha_{G_j}.
\]

Here \(H\) is the response per unit physical native-cell mass stored by
`GammaBetaTreeProblem.sensitivity`. It is not an unconverted RHIME
`fp_x_flux` array. A bridge-level test must compare this convention with
`FullTilingProblem.design_column()` and `AdditiveAlphaPrior` before the PARIS
stage.

For diagonal measurement covariance \(D\), choose a fixed
Euclidean-orthonormal \(B\) in \(D^{-1/2}\)-whitened observation space,
without using the observed residual. With

\[
W=B^\mathsf TD^{-1/2}H
\]

and the corresponding projected aggregation covariance \(S_P(A)\), the
hybrid likelihood is

\[
\widetilde L_P(y\mid A)
=
\phi_n(r)
\frac{\phi_q(B^\mathsf Tr;0,I+S_P(A))}
     {\phi_q(B^\mathsf Tr;0,I)},
\qquad
r=D^{-1/2}\{y-\mu_P(A)\}.
\]

This is algebraically normalized and has exact retained conditional moments.
It is not generally the exact non-Gaussian Dirichlet-mixture likelihood.

The current production convention \(\kappa=2K\) changes the native
concentration as \(K\) changes. Use it only for within-\(K\) partition tests.
A cross-\(K\) representation-invariance test must freeze one cell-alpha field
and its total concentration across every \(P\) and \(K\).

## A0: source and algebra preflight

Run:

```bash
pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error.py \
  tests/experimental/rjmcmc/test_aggregation_error_low_rank.py

uv run ruff check \
  openghg_inversions/experimental/rjmcmc/aggregation_error.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error_low_rank.py \
  tests/experimental/rjmcmc/test_aggregation_error.py \
  tests/experimental/rjmcmc/test_aggregation_error_low_rank.py

uv run pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error.py \
  openghg_inversions/experimental/rjmcmc/aggregation_error_low_rank.py
```

Hard gates:

- two- and four-cell exact likelihoods remain normalized;
- row-first and column-first root charts agree;
- every exact projection agrees with the independent native-Gamma evidence
  reference;
- analytic conditional means and covariances agree with direct Dirichlet
  algebra and seeded Monte Carlo confidence intervals;
- the small-\(q\) likelihood equals the corresponding dense Gaussian density;
- \(S=0\), singleton regions, and a fine partition reduce to the original
  diagonal Gaussian exactly;
- label, cell-order, observation-order, and orthogonal summary-coordinate
  permutations preserve the appropriate results;
- singular \(S\) is supported, while non-orthonormal \(B\), non-PSD inputs,
  mismatched shapes, and non-finite values fail closed; and
- no result claims structural evidence invariance from normalization or
  moment matching alone.

Do not proceed on an algebra, normalization, or target-identity failure.

## A1: tiny scientific-shape gates

Use the exact two- and four-cell oracles in near-Gaussian, skewed,
boundary-heavy, heterogeneous-footprint, and equal-footprint regimes.
Predeclare pseudo-observations and quadrature orders.

For every regime compare:

- exact hidden-allocation likelihood;
- explicit hidden-share inference;
- dense Gaussian moment closure;
- low-rank closure at every rank from zero to full; and
- nominal filling as a deliberately incorrect sentinel.

Record conditional log-likelihood error, gradients in log-total/log-ratio
coordinates, posterior means/variances/intervals, log evidence, structural
posterior weights under a nonuniform structural prior, and common projected
posterior summaries.

The dense/full-rank implementation must agree with its direct Gaussian oracle
to a condition-scaled \(10^{-10}\) tolerance. Exact quadrature identities
retain their existing \(10^{-6}\)-level or tighter gates.

A material exact-versus-Gaussian difference is a scientific-shape failure,
not an implementation failure. Preserve it and withhold Gaussian structural
use.

## A2: moderate rank and operator matrix

Use native dimensions \(64\) and \(256\), observation dimensions \(32\) and
\(128\), and at least one PARIS-derived footprint subset. Include balanced,
heterogeneous, greedy/SLS, and random-recursive partitions.

Predeclare

```text
summary ranks: 0, 8, 16, 32, 64, 128, 256, full
```

truncated to valid dimension. Select \(B\) from \(H,D\), and prior-predictive
aggregation variation only. Split simulations by native-draw identity and
hold out entire partitions and operators.

Choose the smallest rank satisfying all applicable gates:

- 99th percentile absolute dense-closure log-likelihood error at most
  0.1 nat in posterior-relevant states;
- posterior means within 0.05 reference posterior SD;
- posterior SDs within 2%;
- central 95% interval endpoints within 0.05 reference posterior SD;
- no material SBC or predictive-coverage failure; and
- for structural use only, maximum pairwise log-evidence dispersion at most
  0.05 nat and total-variation distance of recovered structural weights from
  their declared prior at most 0.01.

If the dense closure passes but no truncated rank passes, increase rank or
change covariance representation. Do not invoke NLE for compression error.

## A3: frozen PARIS conditional screen

Use the checksum-verified May 2014 PARIS input with 1,382 observations and
23,424 native cells. Freeze, at each \(K=50,250\):

- largest-nominal, greedy/SLS, and two random-recursive development tilings;
- two untouched random-recursive held-out tilings;
- nominal, prior-draw, fixed-basis NUTS posterior, low-mass, and high-mass
  anchors; and
- all topology, operator, alpha-field, observation-order, error-model, and
  summary-basis hashes.

At every anchor:

1. compare analytic covariance actions and diagonals with streamed
   conditional Dirichlet simulations;
2. compare the rank grid with the dense Gaussian closure;
3. assess non-Gaussian shape along predeclared covariance, footprint, site,
   time, and tail-sensitive directions; and
4. report PIT, skewness, kurtosis, tail coverage, energy/classifier
   two-sample effects, conditional log scores, likelihood differences, and
   gradient differences.

Do not select summaries using the observed residual. Treat a changed \(H\),
\(D\), alpha field, observation order, or \(B\) as a new artifact requiring
revalidation.

Resource guidance:

- a dense \(1382^2\) float64 covariance is about 14.6 MiB;
- the current native design is about 247 MiB;
- low-rank factors are small relative to the design; and
- request 16 GiB per factor-build job and no more than four concurrent
  factor-build jobs.

Bounded sequential builds may run on a quiet private login node. Use Slurm for
matrices or retained chains, and keep aggregate login-node RSS below 200 GB.
Stop remote work if BP1 becomes unreachable.

## A4: fixed-partition posterior gate

Integrate the selected frozen approximation into deterministic \(K=50\) and
\(K=250\) bases first. Compare:

- selected low rank;
- the highest affordable rank or dense Gaussian reference; and
- the existing no-aggregation fixed-basis NUTS target as a scientific
  contrast, not as an identical model.

Require converged chains before interpreting approximation differences.
Compare total mass, fixed coefficients, predeclared regional sums, predictive
residuals, and gradients. Passing A4 licenses an experimental fixed-\(P\)
likelihood even if A5 fails.

## A5: structural evidence gate

Use a small frozen set of exact projections from one common native prior.
Estimate every normalized evidence with replicated annealed SMC or another
independently audited estimator whose Monte Carlo uncertainty is
substantially below 0.05 nat.

Test:

- within-\(K\) partition invariance under one fixed alpha field;
- cross-\(K\) invariance only under the explicitly \(K\)-independent alpha
  field;
- recovery of a nonuniform structural prior;
- common projected posterior summaries; and
- selected rank against the higher-rank reference.

Any evidence preference beyond tolerance is approximation leakage. If A4
passes and A5 fails, retain fixed-partition use and prohibit learned or
Gaussian posterior claims about \(P\) or \(K\).

## Learned successor

Only a Gaussian scientific-shape failure advances to learned density
estimation. The first successor is a normalized transported mixture in the
same fixed summary space:

1. standardize hidden-residual summaries by the analytic \(S_P(A)^{1/2}\);
2. fit a small pooled Gaussian mixture;
3. post-centre and whiten its exact mixture moments; and
4. convolve independent Gaussian measurement noise analytically.

Use native-draw, whole-partition, and whole-operator held-out splits. A flow or
permutation-invariant topology encoder is deferred until this mixture fails
held-out likelihood, tail, gradient, or calibration gates.

At finite training size, even a normalized consistent NLE may introduce
partition-dependent evidence error. Structural weights remain externally
fixed at their prior unless A5 passes.
