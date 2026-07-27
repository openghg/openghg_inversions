# Exact-mixture compression for Gamma--Dirichlet aggregation error

## Purpose and status

This document records the next bounded aggregation-error experiment after the
root-only eight- and sixteen-component EM Gaussian mixtures failed their
predeclared scientific or numerical gates.  The new experiment exploits the
known structure of the Gamma--Dirichlet marginal likelihood instead of fitting
an unconstrained density from scratch.

The intended sequence is:

1. construct the exact analytic noise-whitened residual covariance;
2. retain its leading eigendirections under an explicit discarded-variance
   rule;
3. construct a large, fixed scrambled-Sobol approximation to the exact
   continuous Gaussian mixture;
4. compress the residual locations by deterministic hard clustering;
5. replace every cluster by a moment-matched Gaussian component; and
6. test the normalized compressed likelihood against the existing exact
   quadrature oracle.

The first implementation is root-only.  This restriction is scientific, not
just convenient: for one retained root mass the residual covariance changes
only by a scalar square, so its eigenvectors are independent of the current
state.  Multiple retained regions require a separate fixed-projection or
Gaussian-plus-summary construction.

This remains an experimental fixed-partition likelihood.  It does not license
data-dependent weights for partitions or dimensions.

As of 2026-07-27, the root-only spectrum, finite-bank compression, normalized
likelihood, two-stage scientific driver, all-six certifier, focused tests, and
BP1 launch assets are implemented on
`codex/rjmcmc-exact-mixture-compression`.  The first complete local
development matrix passed with a common source size \(S=65{,}536\) and a
stable all-case compression suffix \(M=256,512,1024\).  This is local
development result was reproduced by the full-SHA BP1 G0/G1/G2 run at
`d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea`.  The next gate is the untouched
confirmation scrambles under
`docs/plans/rjmcmc_exact_mixture_confirmation_hpc_test_plan.md`.

## Related durable background

The main derivations and model qualifications live in the sibling
`inversions-knowledge` repository:

| Subject | Repository path |
|---|---|
| General marginalization and the Gamma--Dirichlet special case | `docs/derivations/non-gaussian-aggregation-error-by-marginalization.md`, especially “Gamma--Dirichlet and designed simplex allocations” and “Computation and validation guidance” |
| Gamma--Dirichlet prior, state-dependent covariance, and normalized learned residual likelihood | `docs/source-notes/aggregation-error-and-priors.md`, especially “Gamma--Dirichlet allocation model” and “Option 4B: learn the residual likelihood” |
| Estimator ladder, exact finite transport mixtures, and certification gates | `docs/research-questions/learning-non-gaussian-marginal-models.md` |

The current repository's implementation history and frozen scientific gates
are recorded in:

- `docs/plans/rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md`;
- `docs/plans/rjmcmc_conditional_residual_gmm_16_component_bp1_report.md`;
- `docs/plans/rjmcmc_bp1_handover.md`; and
- `docs/plans/tdmcmc_numpy_numba_rewrite.md`.

## Probability contract

Let \(G_k\) be the native cells represented by retained region \(k\).  Write

\[
\alpha_{k+}=\sum_{i\in G_k}\alpha_i,
\qquad
u_{ki}=\frac{\alpha_i}{\alpha_{k+}},
\qquad
w_k\sim\operatorname{Dirichlet}(\alpha_{G_k}).
\]

Conditional on the retained physical mass \(T_k>0\), native masses are

\[
x_{G_k}=T_k w_k,
\qquad
\widehat x_{G_k}=T_k u_k.
\]

For observation operator \(H\), fixed offset \(b\), and independent Gaussian
measurement error with

\[
D=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2),
\]

the mean prediction and unresolved aggregation perturbation are

\[
\bar y(T)=b+\sum_k T_k H_{G_k}u_k,
\]

\[
\delta(T,w)=\sum_k T_k H_{G_k}(w_k-u_k).
\]

The exact conditional likelihood is

\[
\boxed{
p(y\mid T)
=
\int
\mathcal N\!\left(y;\bar y(T)+\delta(T,w),D\right)
\prod_k\operatorname{Dir}(w_k;\alpha_{G_k})\,dw.
}
\]

This is an exact **continuous Gaussian location mixture**.  It is not normally
a finite GMM and it does not become Gaussian merely because the measurement
error is Gaussian.

The corresponding exact covariance is

\[
\operatorname{Cov}\{\delta(T,w)\mid T\}
=
\sum_k T_k^2 H_{G_k}
\frac{\operatorname{diag}(u_k)-u_ku_k^\mathsf T}
     {\alpha_{k+}+1}
H_{G_k}^\mathsf T.
\]

The existing Gaussian closure retains only this first- and second-moment
information.  The present experiment instead keeps an explicitly non-Gaussian
finite mixture in selected observation-space directions.

## Direct finite-mixture approximation

Draw or deterministically construct allocation vectors
\(w^{(s)}\), \(s=1,\ldots,S\), with non-negative weights
\(\omega_s\) that sum to one.  The finite approximation is

\[
\widehat p_S(y\mid T)
=
\sum_{s=1}^{S}\omega_s
\mathcal N\!\left(
y;\bar y(T)+\delta\{T,w^{(s)}\},D
\right).
\]

Every finite bank is therefore normalized by construction.  The current
scrambled-Sobol balanced-Beta construction uses equal weights
\(\omega_s=1/S\), nested powers of two, stable native-cell IDs, and canonical
Dirichlet-tree coordinates.  It is a deterministic randomized-quasi-Monte
Carlo approximation after its scramble is frozen; it is not an algebraically
exact quadrature rule.

The certified direct-bank screen passed eight of nine tiny oracle cases.  Its
sole failure, the boundary-heavy four-cell root, passed at \(S=16{,}384\) but
did not establish the required two-size suffix.  This makes a larger direct
bank a more promising source measure than either fitted EM GMM:

- the eight-component EM model showed stable boundary-heavy underfit;
- the sixteen-component EM model also failed boundary-heavy scientific gates
  and had numerical fit failures in the skewed two-cell case.

The new method uses the large bank offline and compresses it for repeated
likelihood evaluation.

## Exact residual spectrum and rank reduction

For one root, put \(\eta=\sum_i\alpha_i\), \(u_i=\alpha_i/\eta\), and define
the noise-whitened centred operator

\[
A
=
D^{-1/2}
\left(H-\bar h\mathbf 1^\mathsf T\right),
\qquad
\bar h=Hu.
\]

The unit-mass residual covariance is

\[
S_0
=
A\,
\frac{\operatorname{diag}(u)-uu^\mathsf T}{\eta+1}
A^\mathsf T.
\]

An equivalent numerically convenient factorization is

\[
S_0=CC^\mathsf T,
\qquad
C=A\operatorname{diag}\!\left(
\sqrt{\frac{u}{\eta+1}}
\right),
\]

because \(Au=0\).  Let

\[
S_0=U\Lambda U^\mathsf T,
\qquad
\lambda_1\geq\lambda_2\geq\cdots\geq0.
\]

For current root mass \(T\),

\[
S(T)=T^2S_0.
\]

Thus the eigenvectors are fixed for the entire root posterior.  Retaining the
first \(r\) directions minimizes expected squared discarded whitened residual
among all rank-\(r\) orthogonal projections, and

\[
\mathbb E\left[
\left\|(I-U_rU_r^\mathsf T)\delta\right\|^2
\right]
=
T^2\sum_{j>r}\lambda_j.
\]

If the discarded residual is simply omitted before convolution with unit
Gaussian noise, joint convexity of KL gives the useful distributional bound

\[
\operatorname{KL}(p_{\rm exact}\Vert p_{\rm projected})
\leq
\frac{T^2}{2}\sum_{j>r}\lambda_j,
\]

and Pinsker's inequality gives

\[
\operatorname{TV}(p_{\rm exact},p_{\rm projected})
\leq
\frac{|T|}{2}
\sqrt{\sum_{j>r}\lambda_j}.
\]

These are not pointwise or log-likelihood bounds.  Posterior-tail and evidence
checks remain necessary.

The implementation should normally do better than omission: model the leading
\(q\leq r\) directions by a non-Gaussian mixture, retain a Gaussian
moment-closure correction in directions \(q+1,\ldots,r\), and omit only the
certified small tail beyond \(r\).

## Moment-preserving mixture compression

Let

\[
z_s=U_q^\mathsf T\delta\{1,w^{(s)}\}
\]

be the unit-mass leading coordinates of direct-mixture component \(s\).
Partition these locations into \(M\) non-empty clusters
\(\mathcal C_1,\ldots,\mathcal C_M\).  For general source weights define

\[
\pi_m=\sum_{s\in\mathcal C_m}\omega_s,
\]

\[
\mu_m=\frac{1}{\pi_m}
\sum_{s\in\mathcal C_m}\omega_s z_s,
\]

\[
\Sigma_m=\frac{1}{\pi_m}
\sum_{s\in\mathcal C_m}\omega_s
(z_s-\mu_m)(z_s-\mu_m)^\mathsf T.
\]

Replace the discrete locations in cluster \(m\) by

\[
Z_m\sim\mathcal N(\mu_m,\Sigma_m).
\]

This preserves total probability, the finite bank's global mean, and its
global covariance exactly up to floating-point summation.  After convolution
with measurement noise, the retained-coordinate likelihood is

\[
\boxed{
q_M(z\mid T)
=
\sum_{m=1}^{M}\pi_m
\mathcal N\!\left(
z;T\mu_m,\ I_q+T^2\Sigma_m
\right).
}
\]

The grouping also gives a useful integrated error bound.  Joint convexity of
KL and the moment identities above imply

\[
\operatorname{KL}\!\left(
\widehat p_S\Vert q_M
\right)
\leq
\frac12\sum_{m=1}^{M}
\pi_m\log\det(I_q+\Sigma_m)
\]

for the unit-mass, unit-noise retained-coordinate mixture.  For small
within-cluster covariance,
\(\log\det(I+\Sigma_m)\approx\operatorname{tr}(\Sigma_m)\), so whitened
\(k\)-means distortion is a first-order proxy for this bound.  The actual
log-determinant bound should nevertheless be recorded and used alongside the
downstream scientific gates.

The compression is not exact beyond two moments.  Euclidean \(k\)-means in
whitened coordinates minimizes within-cluster squared displacement, not KL
or pointwise likelihood error.  Boundary and tail gates are therefore
decisive.

For the equal-weight Sobol bank, SciPy's tested
`scipy.cluster.vq.kmeans2` implementation is sufficient for the first
experiment.  Use several fixed \(k\)-means++ starts, fail on empty clusters,
select the smallest inertia, and sort the final components canonically before
serialization.  No new mixture-reduction dependency is justified initially.

## Normalized hybrid likelihood

Let \(r=D^{-1/2}\{y-\bar y(T)\}\) and
\(z=U_r^\mathsf T r\).  Directions outside \(U_r\) retain ordinary unit
measurement noise.  Directions \(q+1,\ldots,r\) use the Gaussian closure with
variance \(1+T^2\lambda_j\).  The complete root likelihood is

\[
\begin{aligned}
\log \widetilde p(y\mid T)
={}&-\sum_i\log\sigma_i\\
&+\log\phi_{n-r}\!\left(
(I-U_rU_r^\mathsf T)r;0,I
\right)\\
&+\sum_{j=q+1}^{r}
\log\phi\!\left(z_j;0,1+T^2\lambda_j\right)\\
&+\log\sum_{m=1}^{M}\pi_m
\phi_q\!\left(
z_{1:q};T\mu_m,I_q+T^2\Sigma_m
\right).
\end{aligned}
\]

This density is normalized.  It is equivalent to replacing selected
marginals of the full Gaussian closure while retaining a Gaussian complement,
as proposed in `inversions-knowledge/docs/source-notes/aggregation-error-and-priors.md`.

Special cases must be literal:

- \(q=0\): low-rank Gaussian closure only;
- \(q=r\): compressed mixture across the retained residual image;
- \(r=0\): ordinary diagonal Gaussian measurement likelihood;
- one source component per bank draw with zero within-component covariance:
  the direct finite bank;
- one native cell: no aggregation error.

## Why not rank mixture terms only by weight?

Dropping source components with total weight \(\epsilon\) gives a useful
probability-mass or total-variation control after renormalization.  It does
not uniformly control log likelihood: a very small component can dominate in
a tail where every retained component is much smaller.

The chosen method instead clusters nearby component locations, retains every
source component's weight through its cluster, and preserves the first two
moments.  Pure weight pruning may be added only as a separately gated
sensitivity experiment.

## Deferred Gamma--Beta tree integration

A consistent Gamma--Beta tree factorizes a Dirichlet allocation into
independent one-dimensional Beta splits.  This suggests a recursive
integration algorithm:

1. apply one-dimensional quadrature at each split;
2. combine child Gaussian-mixture messages;
3. moment-merge components after each convolution; and
4. add measurement noise at the root.

Without intermediate compression, this is an alternative construction of the
same exact continuous mixture, up to quadrature error.  Compression prevents
exponential component growth but can introduce dependence on the computational
tree.  Because the direct Sobol-bank route is simpler and already has strong
oracle evidence, tree message passing is deferred until the direct compressed
mixture is assessed.

## First implementation boundary

The first code increment provides:

- a root-only analytic residual spectrum;
- explicit explained-variance and discarded-tail diagnostics;
- a nested scrambled-Sobol direct bank in the retained eigencoordinates;
- deterministic SciPy clustering with multiple frozen starts;
- moment-preserving component refitting;
- a normalized NumPy root likelihood with Gaussian complement; and
- authenticated arrays and construction metadata sufficient for exact replay.

It does not yet provide:

- arbitrary multi-region conditioning;
- state-dependent eigenvectors;
- Gamma--Beta tree message passing;
- a PyTensor or JAX gradient implementation;
- structural inference over \(K\) or partitions; or
- a production RHIME interface.

## Validation requirements

### Numerical and algebraic tests

1. Orthonormality, eigenvalue order, and covariance reconstruction.
2. Exact agreement of analytic and simulated Dirichlet moments.
3. Compression preserves source-bank weight, mean, and covariance.
4. Literal normalization and limiting cases.
5. Dense direct-mixture parity when no compression or rank truncation occurs.
6. Region/native-cell permutation and stable-ID replay.
7. Serialization, hash, dtype, and malformed-artifact rejection.
8. Finite-difference gradient checks.

### Scientific tiny-oracle gates

Reuse the existing near-Gaussian, skewed, and boundary-heavy two- and
four-cell root cases and the frozen thresholds for:

- prior-weighted median absolute conditional log-likelihood error;
- posterior-weighted 99th-percentile error;
- scaled coordinate-gradient error;
- absolute log-evidence error;
- posterior mean and SD error; and
- interval-endpoint error.

Development may compare a predeclared bounded grid of bank size, retained
rank, and component count.  A configuration is selectable only through a
common all-case passing suffix or an equally strict rule frozen before BP1
submission.  Independent source scrambles and a protected scientific
confirmation remain required before any real-data promotion.

### Computational gates

Record separately:

- offline spectrum, bank-generation, and compression time;
- artifact bytes;
- repeated likelihood and finite-difference-gradient throughput;
- peak RSS; and
- cost relative to the direct bank and Gaussian closure.

The large bank is an offline construction cost.  Repeated inference cost must
scale with compressed component count and retained rank, not source-bank size.

## Known limitations

- Variance ranking controls mean-square whitened error, not pointwise
  likelihood or posterior-tail error.
- Hard clustering is not a KL-optimal Gaussian-mixture reduction.
- A full-covariance component costs \(O(q^3)\) if refactorized at every mass.
  The root scaling \(I+T^2\Sigma_m\) permits cached eigendecompositions in a
  later performance pass.
- Multiple regions have
  \(S(T)=\sum_kT_k^2S_k\); their eigenspaces generally change with \(T\).
  A fixed basis plus the normalized Gaussian-plus-summary correction is the
  likely extension.
- A computational Gamma--Beta integration tree can affect an approximation
  when intermediate mixtures are reduced, even though the unreduced
  Dirichlet law is tree invariant.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-07-27 | Stop increasing free-EM GMM component count. | Eight components showed stable boundary underfit; sixteen components added scientific and numerical failures. |
| 2026-07-27 | Compress a large direct Sobol mixture instead. | It preserves the exact mixture construction and the earlier direct bank already passed eight of nine oracle cases. |
| 2026-07-27 | Use exact analytic covariance eigenvectors, not PCA of simulator draws. | The covariance is known and SciPy supplies mature symmetric eigensolvers. |
| 2026-07-27 | Use SciPy \(k\)-means for the equal-weight first stage. | It avoids a new dependency; cluster moment refitting is small and directly testable. |
| 2026-07-27 | Keep the first experiment root-only. | The residual eigenvectors are state invariant only because root covariance scales as \(T^2S_0\). |
| 2026-07-27 | Defer recursive Gamma--Beta integration. | It is promising but introduces mixture-growth and computational-tree approximation questions absent from the simpler direct bank. |
| 2026-07-27 | Extend the pre-BP1 compression ladder through 512 and 1,024 components. | The first local development matrix selected \(S=65{,}536\); only the boundary-heavy four-cell case was non-monotone, passing at 64 and 256 components while 128 narrowly missed the 0.02 posterior-SD threshold at 0.02194. The larger points test for a stable suffix before the protocol is frozen; no confirmation seed was used. |
| 2026-07-27 | Cache component covariance eigendecompositions and evaluate components in one NumPy batch. | The BP1 scientific screen passed, but the original per-component Cholesky loop made \(M=256\) slower than the vectorized \(S=65{,}536\) source bank on several tiny cases. The cached batched evaluator retains the same density and gives a 16.7-fold speedup over a direct \(S=16{,}384\) bank in a bounded rank-three local benchmark. |
