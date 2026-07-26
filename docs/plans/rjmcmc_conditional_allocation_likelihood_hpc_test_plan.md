# Conditional-allocation likelihood HPC test plan

## Status and decision

The normalized Gaussian closure remains useful for fixed-partition inference,
but the exact tiny-model screen showed that its shape is not generally exact.
The first transported Gaussian-mixture experiment is preserved at
`6ff3afe56416e701ac1fc4ae45676d08ea28229b`. Its bounded fitter produced
artifacts in three of eight development regimes and stopped in five regimes
because none of the fixed EM restarts converged to a valid artifact. The three
fitted artifacts did not pass all scientific-shape gates; projection-isolated
posterior-weighted tail errors reached approximately 0.59--1.90 nat and
structural-weight total-variation errors reached approximately 0.11--0.18.
The final certifier also found a separate catalogue assertion, but
repairing that assertion would not change the declared scientific hard stop.
The authoritative BP1 run is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_aggregation_transported_mixture/6ff3afe56416e701ac1fc4ae45676d08ea28229b/t2/11a43a24da003019e600c990da143573f527ce1b85ffeaedada80ec857edbd28
```

This plan tests a simpler successor before adding `sbi`: a frozen Monte Carlo
bank for the conditional within-region allocations. It estimates a
**normalized conditional-likelihood approximation for one fixed labelled
partition and summary projection**, not model evidence and not a posterior
over partitions.

Fill in before launch:

```text
branch: codex/rjmcmc-aggregation-conditional-likelihood
candidate revision: <git rev-parse HEAD>
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/<full SHA>
```

Run only focused experimental tests, focused Ruff, and focused Pyright.
Preserve failed artifacts, publish completion markers last, and write nothing
to `PARIS_inversions`.

## Probability contract

Let \(X\) be one native positive field, let \(P\) be a fixed labelled
partition, and retain the region masses

\[
A_j=\sum_{i\in G_j}X_i.
\]

One fixed native concentration field \(\alpha_i\) induces

\[
\left(X_i/A_j\right)_{i\in G_j}\mid A,P
\sim \operatorname{Dirichlet}\{(\alpha_i)_{i\in G_j}\}.
\]

The conditional mean renderer is

\[
g_P(A)_i=A_j\frac{\alpha_i}{\alpha_{G_j}}.
\]

For observation operator \(H\), diagonal measurement standard deviations
\(\sigma\), \(D=\operatorname{diag}(\sigma^2)\), and an
operator/prior-selected orthonormal summary basis \(B\), define

\[
r=D^{-1/2}\{y-Hg_P(A)-\mu_{\mathrm{fixed}}\},\qquad
z=B^\mathsf Tr,\qquad
r_\perp=r-Bz.
\]

For each frozen bank member \(s\), draw independent conditional shares within
every region and precompute the projected unit-mass residual factors
\(C_{sj}\). The projected aggregation residual is then linear in the retained
masses:

\[
e_s(A)=\sum_j A_j C_{sj}.
\]

The deterministic finite-bank likelihood is

\[
\widehat L_{P,S}(y\mid A)
=
\left\{\prod_i\sigma_i^{-1}\right\}
\phi_{n-q}(r_\perp)
\frac1S\sum_{s=1}^S\phi_q\{z;e_s(A),I\}.
\]

This is a normalized observation density for every finite frozen bank. It is
smooth in \(A\), and its gradient follows from the component
responsibilities. As \(S\) grows, the bank approaches the exact conditional
allocation integral only in the retained \(B\)-projected residual law. Unless
\(q=n\) or the hidden residual is known to lie in that subspace, the full
observation likelihood still contains the declared Gaussian-complement
approximation.

There are two distinct approximation errors:

1. finite-bank Monte Carlo error; and
2. when \(q<n\), the declared Gaussian complement ignores non-Gaussian hidden
   allocation residual outside the fixed summary subspace.

At \(q=n\), only finite-bank error remains. At a cell-wise fine partition,
the hidden allocation residual is exactly zero and the likelihood must reduce
to the original diagonal Gaussian independently of the bank.

The bank is part of the deterministic approximate target. Do not redraw it
during likelihood evaluation or between HMC steps. Fresh likelihood
randomness is not valid ordinary NUTS; an exact randomized estimator would
instead require pseudo-marginal MH with its randomness included in the state.

## Projection and structural interpretation

Suppose \(p_P(A)\) is the exact pushforward of one common native prior,
\(L_P\) is the exact conditional likelihood, and the externally randomized
partition \(P\) is independent of the native field and observations under the
declared joint model. Then

\[
\int L_P(y\mid A)\,p_P(A)\,dA=p(y)
\]

for every \(P\). Consequently the data do not update the representation:
\(p(P\mid y)=p(P)\). This experiment therefore treats the partition as an
external computational choice. It may:

- run one selected deterministic, greedy, or SLS basis;
- repeat the analysis over partitions drawn independently from the declared
  structural prior; and
- combine common scientific summaries with those external weights.

It must not use finite-bank evidence differences to select \(P\) or \(K\).
External structural weights must be normalized, declared before seeing the
observations, and independent of finite-bank likelihoods or evidence
estimates. Only partition-invariant native scientific quantities may be
combined across projections. Cross-projection evidence drift is a
surrogate-error diagnostic.

Initially fit one artifact per fixed topology. A universal
topology-conditioned network or permutation-invariant topology encoder is
out of scope.

## Artifact identity and resource model

Every bank artifact must record or bind:

- schema and algorithm versions;
- sample count and independent bank seed;
- exact PCG64 bit-generator identity;
- native alpha field and partition-label hashes;
- observation operator, error vector, summary basis, and observation-order
  hashes;
- region count, observation count, summary rank, and native-cell shape;
- projected unit-mass residual factors in canonical
  sample/summary/region order, with explicit axis labels in durable
  metadata; and
- a canonical artifact digest that replays after serialization.

The main stored array costs

\[
8SqK\ \text{bytes}.
\]

For \(K=250,q=8\), \(S=256,1024,4096\) require approximately 3.9, 15.6,
and 62.5 MiB respectively. Likelihood and gradient evaluation cost
\(O(SqK)\). Record build time, evaluation throughput, and peak RSS rather than
assuming the largest bank is preferable.

`storage_nbytes` measures owned NumPy arrays only. A decimal JSON expansion
and a gradient implementation that materializes a second \(S\times q\times K\)
array can each require several times that amount. Before C2, add or select a
binary persistence format with canonical JSON metadata, explicit little-endian
array dtypes/shapes, a whole-artifact digest, and atomic publication. The
value/gradient evaluator must contract against the stored bank without copying
the complete bank-sized tensor. Measure both serialization peak RSS and
gradient peak RSS.

The summary basis must be frozen from \(H,D\), the native prior, and
prior-predictive simulations only. It may not use the realized observation
residual.

## C0: focused source preflight

Run:

```bash
pixi run -e dev --frozen pytest -q -p no:cacheprovider \
  tests/experimental/rjmcmc/test_aggregation_error.py \
  tests/experimental/rjmcmc/test_aggregation_error_low_rank.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py

pixi run -e dev --frozen ruff check --no-cache \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py

pixi run -e dev --frozen pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mixture.py
```

Hard gates:

- seeded construction and serialization replay exactly;
- construction never advances the sampler or posterior RNG;
- empirical allocation means and covariances agree with the analytic
  Dirichlet moments;
- the likelihood is normalized in numerical tiny cases;
- analytic mass gradients agree with central finite differences;
- scalar and batched evaluation agree if both are exposed;
- rank zero and the fine-partition limit reduce to the diagonal Gaussian;
- label, cell, observation, and summary-coordinate permutations preserve the
  corresponding density;
- invalid, non-finite, non-orthonormal, or identity-mismatched inputs fail
  closed; and
- documentation calls the result a normalized conditional-likelihood
  approximation, not raw model evidence.

Do not proceed on a normalization, gradient, replay, or target-identity
failure.

## C1: tiny exact conditional-likelihood screen

Reuse the exact two- and four-cell common-native-model oracles. Include the
same predeclared regimes as the Gaussian and transported-mixture screens:

```text
near_gaussian
skewed
boundary_heavy
heterogeneous_footprint
equal_footprint
```

Reuse the checksum-pinned A1 `Regime` definitions, quadrature orders, state
grids, posterior summaries, and label catalogues rather than recreating
numerically similar cases. For the tiny screen use \(B=I_n\), so \(q=n\) and
the Gaussian-complement approximation is absent. Exact coordinate gradients
use the preserved central finite-difference protocol in log-total/logit
coordinates; transform the analytic mass gradient into those coordinates by
the explicit chain rule.

Keep executable blindness:

- development may use the predeclared near-Gaussian, skewed, and
  boundary-heavy operators with root/two-region development tilings;
- the heterogeneous operator and four-cell column tiling remain held out;
- predeclare a development/held-out retained-mass grid split;
- equal-footprint and fine-partition exact controls may not select bank size;
  and
- independent repeat-bank seeds may confirm the locked choice but may not
  retune it.

For each applicable family and regime:

1. freeze pseudo-observations, quadrature orders, and a full-rank summary
   transform;
2. build independent banks at
   \(S=64,256,1024,4096,16384\), truncated only by a declared resource gate;
3. compare exact and approximate conditional log likelihoods and gradients
   over a predeclared retained-mass grid;
4. compare posterior means, variances, central intervals, tails, and
   predictive coverage;
5. integrate the retained coordinates and compare evidence with the exact
   native model; and
6. repeat the selected bank size with at least four independent bank seeds.

The selected bank size must satisfy, on held-out states:

- median absolute conditional log-likelihood error at most 0.05 nat;
- 99th-percentile absolute error at most 0.2 nat;
- scaled gradient error at most 0.05;
- posterior means within 0.05 exact posterior SD;
- posterior SDs within 2%;
- central 95% interval endpoints within 0.05 exact posterior SD; and
- between-bank evidence range at most 0.05 nat.

These are approximation gates, not proof that structural weights may be
updated. Report error versus \(S\); a non-decreasing or unstable sequence is
a hard stop requiring variance reduction, a better summary factorization, or
a learned conditional density.

Publish aggregated JSON metrics only. Store any required statewise arrays in
small binary artifacts with explicit hashes; do not expand large grids or
banks into report JSON. Run a one-case timing smoke before the eight-case
development array. The boundary-heavy four-cell fine grid can contain
millions of states, so use direct oracle controls or streamed chunks rather
than retaining redundant dense copies.

## C2: moderate and PARIS feasibility

Use the frozen May 2014 PARIS identity:

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
sha256sum "${FROZEN_INPUT}"
```

Require the observed digest to match. Freeze:

- \(K=50\) and \(K=250\);
- largest-nominal, greedy/SLS, and two random-recursive development
  partitions;
- two untouched random-recursive held-out partitions;
- nominal, prior-draw, Gaussian fixed-basis NUTS posterior, low-mass, and
  high-mass anchors;
- summary ranks \(q=8,16,32\); and
- bank sizes \(S=128,512,2048\), with a larger value permitted only after
  the declared memory and time audit.

At every anchor and for at least two independent banks, record:

- log-likelihood and mass-gradient differences between banks;
- convergence against a much larger streamed reference bank on a bounded
  observation/summary subset;
- skewness, kurtosis, PIT, tail coverage, and energy/classifier two-sample
  diagnostics;
- construction time, artifact size, evaluation and gradient throughput, and
  peak RSS; and
- evidence leakage across externally weighted partitions as a diagnostic
  only.

Start on a quiet login node only for bounded one-case smoke tests and keep
aggregate RSS below 200 GB. Use Slurm for the retained matrix.

## C3: PyTensor and fixed-basis posterior integration

Proceed only after C1 passes and C2 identifies an affordable bank/rank pair.
Implement the likelihood in native PyTensor primitives:

- region masses to component means is one fixed tensor contraction;
- component log densities use a stable `logsumexp`;
- gradients must flow through both the conditional mean residual and the
  component means; and
- the fixed orthogonal complement and noise Jacobian remain explicit.

Compare compiled float64 PyTensor values and gradients with the independent
NumPy oracle at randomized interior states. Integrate the result into a
separate fixed-basis PyMC model using one scalar `Potential`. Persist the
scalar conditional joint likelihood as a deterministic diagnostic.

Do not fabricate a pointwise `log_likelihood.observed`: the reduced
conditional density is one dependent observation block. Validate ArviZ
outputs accordingly.

Run four-chain fixed-basis NUTS at \(K=50\) first, then \(K=250\). Compare:

- no-aggregation fixed-basis NUTS;
- Gaussian aggregation closure;
- at least two independent conditional-allocation banks; and
- the largest affordable reference bank.

Require ordinary R-hat, ESS, BFMI, divergence, and tree-depth gates before
interpreting scientific differences. Bank-to-bank posterior variation must
be smaller than posterior Monte Carlo uncertainty for the declared common
summaries.

## C4: optional learned density

Add `sbi` only if the finite conditional bank fails a declared accuracy,
memory, or throughput gate.

The first learned target should be the noise-free projected aggregation
residual conditional on retained scientific coordinates and every quantity
that changes its law. Keep measurement noise outside the learner where
possible. Start with a conditional MDN before MAF or NSF because an MDN is
normalized, auditable, and can be exported to native PyTensor primitives.

Use whole-native-draw, whole-partition, and whole-operator held-out splits.
Record architecture, preprocessing, float dtype, seeds, package versions,
training/validation identities, and a `state_dict`-style artifact rather than
relying on an opaque pickle.

`sbi`'s PyMC MCMC backend samples an `sbi`-owned posterior potential; it does
not insert an arbitrary Torch likelihood into the existing PyMC graph. Either:

1. keep the complete bounded posterior in Torch/`sbi`;
2. export a simple MDN to PyTensor; or
3. build and validate a custom Torch-autograd PyTensor operation.

A custom operation must include both residual and conditioning-coordinate
gradient paths. The current PyMC-to-NumPyro/JAX path cannot consume a generic
Torch operation without a separate JAX lowering.

## Completion and stop rules

- Every source change requires a clean pushed full SHA and a fresh detached
  worktree/run root.
- Stop at the first hard gate and preserve all partial artifacts.
- Do not tune against held-out partitions, operators, pseudo-observations, or
  bank seeds.
- Do not promote a conditional likelihood because its moments alone are
  correct.
- Do not allow an approximate evidence difference to update \(P\) or \(K\).
- Stop remote execution when BP1 becomes unreachable; do not reinterpret VPN
  loss as a scientific failure.
