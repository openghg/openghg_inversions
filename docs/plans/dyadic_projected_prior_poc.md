# Projection-Consistent Dyadic SLS Proof of Concept

## Purpose

This operational plan replaces the equal-region isotropic covariance used by
the first TAC/MHD stochastic local-search demo with a Gaussian construction
that is consistent with the fine-grid scaling-factor prior. The scientific
derivation and source notes are kept separately in
`docs/reports/rhime_bocquet_reduced_gaussian.md`.

The work remains isolated under `basis.experimental.dyadic`. It does not alter
the production RHIME likelihood, `run_hbmcmc.py`, or `fixedbasisMCMC`.

## Current result

The implementation and a full-week TAC/MHD run are complete on the
`codex/dyadic-projected-prior` branch:

- 333 aligned hourly TAC/MHD observation rows;
- native grid `(293, 391)` and 8x8-block dyadic search grid `(37, 49)`;
- 110,758 nonzero-prior-flux native cells;
- greedy exact-DFS initializer at `K=24`;
- 2,000 variable-count local-search proposals;
- best state at `K=28`;
- projected DFS improvement from `0.478183` to `0.479182`; and
- native-grid no-reduction DFS upper bound `3.563719`.

The best-state aggregation covariance has minimum eigenvalue
`3.18e-6`, and the independently constructed observation-space covariance
terms close to the innovation identity with Frobenius error `1.13e-10`.
Reproducible artifacts are in
`docs/plans/figures/dyadic_projected_prior/`.

The corrected DFS is additive over dyadic nodes. Consequently, an optimal
fixed-K frontier, or an optimal frontier under an additive per-region penalty,
can be found with tree dynamic programming. That deterministic solution should
be added as an oracle for this Gaussian case. SLS remains useful infrastructure
for later non-additive posterior or holdout objectives, but is not required to
optimize this exact Gaussian score.

## Projection convention

Let `D = diag(mu)` contain the prior flux, let `H` be the footprint operator,
and let `G = H D` be the flux-weighted sensitivity already used by RHIME. For a
partition membership matrix `A`, RHIME's summed regional columns are

\[
U = D A^T, \qquad H_P = H U = G A^T.
\]

The proof of concept retains this prolongation and defines the restriction as
the prior-precision-weighted left inverse

\[
\Gamma = (U^T B^{-1} U)^{-1} U^T B^{-1}.
\]

This convention has `Gamma U = I`. It also makes `U` the conditional-mean
prolongation in the Bocquet construction, so the projected coefficient and
fine-grid residual are independent under the Gaussian prior. This is not the
ordinary Euclidean pseudoinverse when prior flux varies within a region.

For independent fine-grid relative scaling errors with standard deviation
`tau`, `B = tau^2 D^2`. On nonzero-flux support, a region containing `n_k`
native cells then has

\[
B_{P,k} = \frac{\tau^2}{n_k}.
\]

The reduced signal and aggregation-error covariances are

\[
C_P = \sum_k B_{P,k} c_k c_k^T,
\qquad c_k = \sum_{i\in k} G_i,
\]

\[
C_{agg} = C_{full} - C_P,
\qquad C_{full} = \tau^2 G G^T.
\]

With observation-error covariance `R`, the effective reduced covariance is
`R_P = R + C_agg`, and the total innovation covariance is invariant:

\[
R_P + C_P = R + C_{full}.
\]

The DFS score is therefore additive over active regions:

\[
D(P) = \operatorname{tr}[(R+C_{full})^{-1} C_P].
\]

The native no-reduction DFS is the same expression with `C_P = C_full` and is
an actual upper bound for every valid partition, up to numerical tolerance.

## Work packets

1. **Gaussian projection model**
   - Precompute native full signal covariance and its innovation solve.
   - Sum native sensitivity and nonzero-flux support over every dyadic node.
   - Expose node DFS contributions, partition DFS, full-grid DFS, aggregation
     covariance, and effective observation covariance.
   - Test restriction/prolongation identities, residual independence,
     innovation invariance, and the no-reduction bound against dense formulas.

2. **Search integration**
   - Add a separate variable-count runner using exact node DFS contributions
     for initialization and search.
   - Preserve the existing isotropic runner as an explicitly labelled
     historical comparison.
   - Continue to show raw DFS, region count, and complexity penalty separately.

3. **TAC/MHD demonstration**
   - Retain the native prior-flux array in the fixture adapter.
   - Run the full-week example with the same filtering and row alignment as the
     existing demo.
   - Regenerate the trace, static comparison, GIF, and machine-readable
     manifest with the projected-prior convention and the true native-grid DFS
     reference.

4. **Review gate**
   - Run focused tests, Ruff, and Pyright.
   - Obtain a scientific-array/numerical review of formulas, support handling,
     and diagnostics before presenting the result as more than experimental.

All four packets are implemented. Independent scientific and code review found
and prompted fixes for cancellation-prone covariance subtraction, strict
integer configuration, invalid-frontier gathering, fixed-reference plotting,
and best-utility labelling. Focused branch verification is complete; commit
and push remain.

The follow-on optimizer, quadtree, covariance, and synthetic-inversion
diagnostics are tracked separately in `docs/plans/dyadic_basis_diagnostics.md`.

## Stop conditions and deferred work

- Zero-flux cells have zero prior variance and are excluded from coefficient
  support. A partition tile with no supported cells contributes no state-vector
  direction or DFS.
- This phase assumes independent, equal-variance relative scaling errors on
  native cells. Correlated priors, heterogeneous relative variances, signed
  flux partitioning, and non-Gaussian priors are deferred.
- The fixed diagonal `R` remains a demo approximation. The production model's
  inferred mismatch process is not reproduced here.
- The construction is exact for the declared Gaussian model. It is input to a
  stochastic optimizer, not yet a partition posterior or joint MCMC method.
