# RHIME reduced Gaussian model under Bocquet-consistent aggregation

## Purpose and status

This document fixes the Gaussian background model for the experimental RHIME
dyadic proof of concept. It is a mathematical and implementation reference, not
an operational task tracker. In particular, it does not authorize changes to
the production `run_rhime` or legacy `fixedbasisMCMC` paths.

The central decision is:

> Preserve RHIME's existing prior-flux-weighted prolongation and summed
> `fp_x_flux` columns. Obtain an exact conditional Gaussian reduction by using
> the corresponding \(B^{-1}\)-weighted restriction, not the ordinary
> Euclidean pseudoinverse.

This choice gives an exact independent resolved/unresolved decomposition,
full observation-space aggregation covariance, a partition-invariant
innovation covariance, additive dyadic-node DFS scores, and a genuine
fine-grid DFS bound.

The supplied external note, *Reduced-dimension basis functions, prior
uncertainty, and aggregation error in RHIME-style inversions*, remains useful
background, especially for coefficient interpretation, prior calibration, and
the distinction between resolved maps and full fields. However, its Section 6
requires correction: the residual produced by an arbitrary projection is not
automatically independent of the projected coefficient. Consequently,
integrating a residual with covariance \(PBP^T\) while retaining an unrelated
prolongation is not generally the exact conditional marginal. This document
records the correction; it does not edit or claim to have edited the external
note.

## 1. RHIME convention and notation

Let there be \(M\) supported native grid cells, \(K\) active regions, and \(N\)
observations. The formulas below are for one source and one coefficient period;
additional source or time axes can be stacked only after their covariance and
membership semantics have been stated.

- \(\mu\in\mathbb R^M\) is the native prior mean flux in the same absolute-flux
  coordinate used by the forward model.
- \(\delta x\in\mathbb R^M\) is the native absolute flux anomaly around that
  mean, so the physical flux is \(x=\mu+\delta x\).
- \(D=\operatorname{diag}(\mu)\).
- \(H\in\mathbb R^{N\times M}\) maps native absolute flux anomalies to
  observation anomalies.
- \(y\in\mathbb R^N\) is the observation vector after the prior-mean and other
  declared baseline terms have been separated, and \(R\succ0\) is its base
  observation/model-error covariance before aggregation error.
- \(G=HD\) is the native RHIME relative-scaling design. In project data this is
  represented by aligned `fp_x_flux` columns, up to declared unit conversions.
- \(A\in\{0,1\}^{K\times M}\) is the region-membership matrix. Every supported
  cell belongs to exactly one active region. Thus
  \(AA^T=\operatorname{diag}(n_1,\ldots,n_K)\), where \(n_k\) is the number of
  supported native cells in region \(k\).
- \(\alpha\in\mathbb R^K\) is a zero-mean regional relative-scaling anomaly.
  A RHIME scale factor with prior mean one is \(1+\alpha\).

The native Gaussian prior considered by the proof of concept is

\[
\delta x\sim\mathcal N(0,B),
\qquad
B=s^2D^2,
\]

which means independent native relative errors with common standard deviation
\(s\). The formulas are defined on nonzero-flux support; zero flux is handled
separately in Section 8.

RHIME's existing prolongation from a regional scaling anomaly to an absolute
native anomaly is

\[
U=DA^T,
\qquad
\delta x_{\mathrm{resolved}}=U\alpha.
\]

This convention is not optional if the existing design columns are retained.
The reduced observation operator is

\[
H_P=HU=HDA^T=GA^T,
\]

so each reduced column is the **sum**, not the mean, of the native
`fp_x_flux` columns in its region. This is exactly what
`BasisOperator.sensitivity()` and the experimental dyadic multiscale design do.

The symbol \(P\) in a subscript denotes a partition. Residual projectors are
written \(E\) to avoid overloading that symbol.

## 2. What an exact Gaussian reduction requires

Start with any full-row-rank restriction \(\Gamma\in\mathbb R^{K\times M}\)
and define the coarse coefficient

\[
\alpha=\Gamma\delta x,
\qquad
B_P=\Gamma B\Gamma^T.
\]

The conditional Gaussian formulas give

\[
\mathbb E[\delta x\mid\alpha]
=\Lambda\alpha,
\qquad
\Lambda=B\Gamma^T(\Gamma B\Gamma^T)^{-1},
\]

and

\[
B_c
=\operatorname{Cov}(\delta x\mid\alpha)
=B-B\Gamma^T(\Gamma B\Gamma^T)^{-1}\Gamma B.
\]

Define the conditional residual

\[
\eta=\delta x-\Lambda\alpha.
\]

Then

\[
\Gamma\Lambda=I_K,
\qquad
\operatorname{Cov}(\alpha,\eta)=0,
\qquad
\eta\sim\mathcal N(0,B_c).
\]

Because \((\alpha,\eta)\) is jointly Gaussian, zero cross-covariance implies
independence. The exact observation model conditioned on the coarse coefficient
is therefore

\[
y\mid\alpha
\sim
\mathcal N(H\Lambda\alpha,\;R+HB_cH^T).
\]

These are the ordinary partitioned-Gaussian conditional mean and Schur
complement covariance. They are also Bocquet, Wu, and Chevallier's Bayesian
prolongation and refinement-error construction.

Two points are essential:

1. A restriction \(\Gamma\) determines its exact conditional prolongation
   \(\Lambda\). One cannot generally choose them independently.
2. An unconditional residual covariance is not enough. If the selected
   coefficient and residual are correlated, conditioning changes both the
   residual mean and its covariance.

## 3. Why the Euclidean shortcut is not exact

### 3.1 The Euclidean coefficient is valid

For the existing RHIME prolongation \(U=DA^T\), the ordinary Euclidean
pseudoinverse is

\[
\Gamma_E
=(U^TU)^{-1}U^T
=(AD^2A^T)^{-1}AD.
\]

It satisfies \(\Gamma_EU=I_K\), so
\(\alpha_E=\Gamma_E\delta x\) is a perfectly valid projected coefficient.
Its projected covariance under \(B=s^2D^2\) is

\[
B_E
=\Gamma_EB\Gamma_E^T
=s^2(AD^2A^T)^{-1}(AD^4A^T)(AD^2A^T)^{-1}.
\]

This is the \(D^2\)-weighted projected variance described in the supplied note.
For a region with constant \(\mu\), it reduces to \(s^2/n_k\), but it does not
do so for a general flux pattern.

### 3.2 Its residual is generally correlated

If the same Euclidean pair is used to define

\[
E_E=I-U\Gamma_E,
\qquad
r_E=E_E\delta x,
\]

then the coefficient-residual cross-covariance is

\[
K_E
=\operatorname{Cov}(\alpha_E,r_E)
=\Gamma_EBE_E^T
=\Gamma_EB-B_EU^T.
\]

For cell \(j\) in region \(k\), this is

\[
\operatorname{Cov}(\alpha_{E,k},r_{E,j})
=s^2\left[
\frac{\mu_j^3}{\sum_{i\in k}\mu_i^2}
-
\mu_j
\frac{\sum_{i\in k}\mu_i^4}
     {(\sum_{i\in k}\mu_i^2)^2}
\right],
\]

which is generally nonzero.

The exact conditional mean associated with \(\Gamma_E\) is instead

\[
\Lambda_E\alpha_E,
\qquad
\Lambda_E=B\Gamma_E^TB_E^{-1},
\]

and \(\Lambda_E\ne U\) in general. Equivalently,

\[
\mathbb E[r_E\mid\alpha_E]
=(\Lambda_E-U)\alpha_E
=K_E^TB_E^{-1}\alpha_E.
\]

Thus the model

\[
y\mid\alpha_E
\overset{\text{shortcut}}{\sim}
\mathcal N(HU\alpha_E,\;R+HE_EBE_E^TH^T)
\]

omits the conditional residual mean and uses the unconditional, rather than
conditional, residual covariance. It is an independent-residual approximation,
not the exact conditional marginal.

### 3.3 Two-cell counterexample

Take one region, \(s=1\), \(\mu=(1,2)^T\), and therefore

\[
U=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
B=\begin{bmatrix}1&0\\0&4\end{bmatrix},
\qquad
\Gamma_E=\begin{bmatrix}1/5&2/5\end{bmatrix}.
\]

Then

\[
B_E=17/25,
\qquad
\operatorname{Cov}(\alpha_E,r_E)
=\begin{bmatrix}-12/25&6/25\end{bmatrix}\ne0.
\]

The shortcut resolved-plus-residual covariance is

\[
UB_EU^T+E_EBE_E^T
=\frac1{25}
\begin{bmatrix}49&18\\18&76\end{bmatrix}
\ne B.
\]

The missing coefficient-residual cross terms are exactly what restore \(B\).
This counterexample is also a compact regression test for any implementation
that claims the Euclidean residual marginal is exact.

### 3.4 Correction to Section 6 of the supplied note

Section 6 of the supplied note writes
\(\delta x=U\alpha+x_\perp\), assigns
\(\operatorname{Cov}(x_\perp)=PBP^T\), and then integrates \(x_\perp\) while
retaining the mean operator \(HU\). That integration is exact only if

\[
\operatorname{Cov}(\alpha,x_\perp)
=\Gamma B(I-U\Gamma)^T=0,
\]

or, equivalently, if \(U\) is the Bayesian conditional prolongation induced by
\(\Gamma\). The Euclidean RHIME pair does not generally satisfy this condition.
The group-sum formulas later in the note can describe the unconditional
Euclidean projected-residual covariance, but they do not turn that covariance
into the exact conditional marginal while \(HU\) is retained.

## 4. Chosen model: preserve RHIME \(U\), change the restriction

To preserve \(U\), choose its \(B^{-1}\)-weighted left inverse:

\[
\boxed{
\Gamma_R
=(U^TB^{-1}U)^{-1}U^TB^{-1}
}.
\]

This is the restriction implied by the desired conditional prolongation. It
has the identities

\[
\Gamma_RU=I_K,
\qquad
B_P=\Gamma_RB\Gamma_R^T=(U^TB^{-1}U)^{-1},
\]

and

\[
B\Gamma_R^TB_P^{-1}=U.
\]

Therefore the Bayesian prolongation is exactly \(\Lambda=U\). With

\[
E_R=I-U\Gamma_R,
\qquad
\eta=E_R\delta x,
\]

the resolved coefficient and unresolved residual are independent:

\[
\operatorname{Cov}(\alpha_R,\eta)=0,
\qquad
B_c=E_RBE_R^T=B-UB_PU^T.
\]

For \(B=s^2D^2\) on nonzero-flux support,

\[
U^TB^{-1}U
=s^{-2}AA^T
=s^{-2}\operatorname{diag}(n_k),
\]

so

\[
\boxed{
B_P=s^2\operatorname{diag}(1/n_k)
}
\]

and

\[
\Gamma_R
=\operatorname{diag}(1/n_k)AD^{-1}.
\]

The selected coefficient is therefore the arithmetic mean of the native
relative anomalies \(\delta x_i/\mu_i\) within each region. The reconstruction
still applies that one relative anomaly to the native prior-flux pattern.

In the two-cell example above,
\(\Gamma_R=(1/2,1/4)\), \(B_P=1/2\), \(\Gamma_RU=1\), and the
coefficient-residual cross-covariance is zero.

## 5. Observation-space covariance and innovation invariance

Let \(g_i\) be native column \(i\) of \(G=HD\), and let

\[
h_k=\sum_{i\in k}g_i
\]

be reduced column \(k\) of \(H_P=HU\). The full native signal covariance is

\[
\boxed{
C_{\mathrm{full}}
=HBH^T
=s^2GG^T
=s^2\sum_i g_ig_i^T
}.
\]

The resolved regional signal covariance is

\[
\boxed{
C_P
=H_PB_PH_P^T
=s^2\sum_k\frac1{n_k}h_kh_k^T
}.
\]

The exact aggregation covariance is

\[
\boxed{
C_{\mathrm{agg}}(P)
=HB_cH^T
=C_{\mathrm{full}}-C_P
}.
\]

It is positive semidefinite under the stated model; this follows from
\(B_c=E_RBE_R^T\), not from numerical clipping. The effective observation
covariance for partition \(P\) is

\[
R_P=R+C_{\mathrm{agg}}(P).
\]

Most importantly,

\[
\boxed{
R_P+C_P
=R+C_{\mathrm{full}}
}
\]

is invariant to the partition. This is the RHIME form of Bocquet et al.'s
innovation-statistics identity. It is a useful numerical closure diagnostic
when \(C_P\) and \(C_{\mathrm{agg}}\) are constructed independently, but it is
not by itself a scientific oracle: defining aggregation error by subtraction
would make the identity tautological, and even independent code paths cannot
validate an incorrectly specified prior model.

No native \(M\times M\) covariance is needed. For \(N\) observations, all
quantities above can be formed as \(N\times N\) matrices from native and
region-summed observation columns.

## 6. DFS, additive node scores, and the fine-grid bound

Set the invariant innovation covariance

\[
S=R+C_{\mathrm{full}}.
\]

The exact reduced degrees of freedom for signal for partition \(P\), including
its aggregation covariance, is

\[
\boxed{
\operatorname{DFS}(P)
=\operatorname{tr}(S^{-1}C_P)
=\operatorname{tr}\left[
(R_P+C_P)^{-1}C_P
\right].
}
\]

Because \(C_P\) is a sum of rank-one regional terms, DFS is additive:

\[
\operatorname{DFS}(P)
=\sum_{k\in P}d_k,
\qquad
d_k
=\frac{s^2}{n_k}h_k^TS^{-1}h_k.
\]

For a dyadic dictionary, \(d_k\) can be precomputed for every candidate node.
A split gain is exactly the sum of the children's scores minus the parent's
score. This is not the historical sum-then-square quadratic proxy: the common
inverse covariance is the full invariant \(S^{-1}\), and the regional variance
is fixed by the weighted restriction.

The native full-grid DFS is

\[
\operatorname{DFS}_{\mathrm{full}}
=\operatorname{tr}(S^{-1}C_{\mathrm{full}})
=s^2\sum_i g_i^TS^{-1}g_i.
\]

Since \(C_{\mathrm{agg}}=C_{\mathrm{full}}-C_P\succeq0\),

\[
0\le\operatorname{DFS}(P)
\le\operatorname{DFS}_{\mathrm{full}}
\le\operatorname{rank}(C_{\mathrm{full}})
\le\min(N,M).
\]

The exact decomposition is

\[
\boxed{
\operatorname{DFS}_{\mathrm{full}}
=\operatorname{DFS}(P)
+\operatorname{tr}(S^{-1}C_{\mathrm{agg}}(P)).
}
\]

The second term is the information left in unresolved native contrasts. It is
nonnegative and vanishes for the supported native-cell partition. If the search
dictionary has factor-8 spatial leaves, the native-cell partition is not
reachable; the native full-grid value remains a bound, not an attainable search
state.

## 7. Four distinct modelling choices

These alternatives must not be blended silently.

| Choice | Restriction | Forward mean | Residual covariance | Status |
| --- | --- | --- | --- | --- |
| Preserve RHIME prolongation | \(\Gamma_R=(U^TB^{-1}U)^{-1}U^TB^{-1}\) | \(HU\alpha_R\) | \(H(B-UB_PU^T)H^T\) | **Chosen POC; exact Gaussian conditional** |
| Preserve Euclidean coefficient | \(\Gamma_E=(U^TU)^{-1}U^T\) | \(H\Lambda_E\alpha_E\), where \(\Lambda_E=B\Gamma_E^TB_E^{-1}\) | \(H[B-B\Gamma_E^TB_E^{-1}\Gamma_EB]H^T\) | Exact, but changes summed RHIME design columns |
| Independent-residual shortcut | Usually \(\Gamma_E\) | Retains \(HU\alpha_E\) | Adds \(HE_EBE_E^TH^T\) | Approximation; not an exact conditional marginal |
| Total-preserving calibration | Target-dependent regional scaling variance | Usually \(HU\beta\) | Must be redesigned for the target model | Different prior objective, not a substitute for conditional projection |

For independent relative native errors, a regional scale factor calibrated to
preserve the variance of a regional total would use

\[
\operatorname{Var}(\beta_k)
=s^2\frac{\sum_{i\in k}\mu_i^2}
          {(\sum_{i\in k}\mu_i)^2}.
\]

This is generally not \(s^2/n_k\). It answers a different question: matching
one aggregate functional rather than constructing the coefficient conditional
to the fine Gaussian state while preserving \(U\). If this larger or otherwise
different regional variance is combined with the conditional residual derived
for \(B_P=s^2/n_k\), resolved variance is counted again and the innovation
identity is broken. Total-preserving calibration may be a defensible separate
model, but it requires its own coherent joint covariance and must not be added
on top of the selected residual model.

## 8. Scope limits and caveats

### 8.1 Zero and near-zero prior flux

When \(\mu_i=0\), both \(D^{-1}\) and \(B^{-1}\) are undefined and the native
relative anomaly has no physical scale. The POC must operate on an explicit
support such as \(|\mu_i|>\epsilon_\mu\), require the corresponding
`fp_x_flux` column to be zero within a stated tolerance outside that support,
and prune zero-support regional coefficients. The support threshold is a model
choice and should be recorded with results; it must not change implicitly with
the partition.

### 8.2 Signed flux

Signed nonzero \(\mu_i\) is algebraically allowed: \(U\) retains the sign and
\(B=s^2D^2\) remains positive. Support should therefore use \(|\mu_i|\), not
\(\mu_i>0\). However, opposite-signed native columns can cancel in a region's
summed \(h_k\), and \(\Gamma_R\) averages ratios \(\delta x_i/\mu_i\). Regions
that cross source/sink classes or contain near-zero sign changes can therefore
have poor coefficient semantics even when the algebra is valid. Separate hard
classes or source components may be preferable.

### 8.3 Area, units, and coefficient meaning

The derivation assumes that \(\mu\), \(\delta x\), and \(H\) use mutually
consistent native coordinates. Flux density, cell total, and unit-converted
`fp_x_flux` are not interchangeable. Cell area and conversion factors must
appear exactly once, either in the absolute-flux coordinate/forward operator or
in an explicitly redefined prolongation. Dyadic observation columns must be
sum-coarsened. Dividing them by geometric area changes the coefficient meaning
and invalidates \(H_P=HU\) unless \(U\), \(B\), and \(\Gamma\) are changed with
it.

The count \(n_k\) is correct only for equal-variance independent native
relative errors in the declared native coordinate. It is not automatically a
geographic-area normalization.

### 8.4 Heterogeneous relative standard deviations

If native relative standard deviations are \(s_i\), then

\[
B=\operatorname{diag}(s_i^2\mu_i^2).
\]

The general weighted formula still applies. For disjoint regions and the same
\(U=DA^T\),

\[
(B_P)_{kk}
=\left(\sum_{i\in k}s_i^{-2}\right)^{-1},
\]

and \(\Gamma_R\) forms an inverse-variance-weighted mean of native relative
anomalies. Also,

\[
C_{\mathrm{full}}=\sum_i s_i^2g_ig_i^T.
\]

The common-\(s\) node-score formula must not be reused without these changes.

### 8.5 Correlated native priors

For a symmetric positive-definite correlated \(B\), the general formulas for
\(\Gamma_R\), \(B_P\), \(B_c\), and innovation invariance remain valid. But
\(B_P\) is generally dense, so
\(C_P=H_PB_PH_P^T\) no longer decomposes into independent node scores. The
current additive search objective is therefore specific to a covariance whose
weighted regional coefficients are independent. Bocquet et al. discuss a
whitened restriction for reducing the correlated case (their Equations
15--19); that changes tile interpretation and is a later design problem, not a
drop-in extension.

If \(B\) is singular rather than positive definite, all inverses above require
an explicitly declared supported subspace or generalized-inverse convention.

### 8.6 Factor-8 search coarsening

The current demonstration's factor 8 means an \(8\times8\) spatial block, up
to 64 native cells away from partial boundaries. It is a search-dictionary
coarsening, not a change to the underlying Gaussian fine state. Therefore:

- candidate columns are sums of native \(G\) columns;
- candidate support is the number of supported native cells, including partial
  boundary blocks;
- \(C_{\mathrm{full}}\) is still computed from native columns;
- no partition can recover contrasts within one coarsened leaf; and
- the native full-grid DFS is a bound, while the finest accessible factor-8
  partition may have a strictly lower DFS.

This factor-8 preprocessing should not be confused with the hierarchy-storage
factors quoted by Bocquet et al. for particular 2D-plus-time dictionaries.

## 9. Implementation boundary and phases

These phases describe dependency order and review boundaries. They deliberately
do not duplicate issue status, owners, or run logs.

### Phase A: experimental full-covariance Gaussian model

Keep the work under `openghg_inversions/basis/experimental/dyadic/` and use
full correlated observation covariance from the outset:

1. Build native \(G\), fixed support, \(C_{\mathrm{full}}\), and
   \(S=R+C_{\mathrm{full}}\).
2. Sum-coarsen only the candidate columns and support counts.
3. Precompute \(B_{P,k}=s^2/n_k\) and additive node scores
   \((s^2/n_k)h_k^TS^{-1}h_k\).
4. Recover \(C_P\) from region sums and construct
   \(C_{\mathrm{agg}}\) independently from cancellation-resistant centered
   native-column scatter within each region; compare it with
   \(C_{\mathrm{full}}-C_P\) only where subtraction is numerically safe.
5. Validate algebra and numerical positive-semidefiniteness without clipping
   materially negative eigenvalues.

The current experimental `rhime_gaussian.py` follows this architecture in
observation space. It should remain isolated until its focused tests, fixture
checks, and scientific conventions are reviewed. No Phase A change should
touch `openghg_inversions/rhime/runner.py`, the production RHIME model, or
`openghg_inversions/hbmcmc/hbmcmc.py`/`fixedbasisMCMC`.

### Phase B: experimental search and evidence

Use the additive score in fixed- and variable-count dyadic searches, while
reporting separately:

- partition DFS and native full-grid DFS;
- unresolved DFS and the exact decomposition residual;
- effective supported-region count;
- coarsening factor and inaccessible within-leaf information;
- support threshold, units, source/sign handling, and base \(R\); and
- sensitivity to plausible \(s\), support, and coarsening choices.

The historical isotropic-region objective in `objectives.py` remains a labelled
benchmark and must not be compared as though it used the same fine prior.

### Phase C: opt-in production boundary

Only after the experimental model is validated should production design begin.
The production interface should be opt-in and should make the following inputs
or products explicit: native prior covariance convention, support, reduced
prior covariance, aggregation covariance, observation ordering, and whether
the covariance is fixed or parameter dependent. Backward-compatible diagonal
likelihood behavior should remain the default until a separate API and
migration review approves otherwise.

### Phase D: PyMC correlated-likelihood implications

The current RHIME likelihood in `models/components.py` uses independent
`pm.Normal` observations with a vector `epsilon`. Exact aggregation error is a
full \(N\times N\) covariance. A production implementation would therefore
need an `MvNormal`-equivalent likelihood or a mathematically identical whitened
log-likelihood with

\[
\Sigma_y=\operatorname{diag}(\epsilon^2)+C_{\mathrm{agg}}(P).
\]

This boundary needs specific design work because:

- filtering and concatenation must keep covariance rows and columns aligned
  with observations;
- a latent, prediction-dependent `epsilon` makes \(\Sigma_y\) change during
  sampling, requiring repeated factorizations;
- dense Cholesky cost and memory may dominate for large \(N\);
- site/time block structure or fixed-covariance whitening may be exploitable
  only when justified by the error model; and
- existing model-error parameters must not absorb or double count the newly
  explicit aggregation covariance.

This is a later production concern, not a reason to diagonalize aggregation
error in the experimental correctness phase.

## 10. Minimum algebraic and regression checks

A focused implementation test set should include:

1. **Forward identity:** gathered candidate columns equal direct sums of native
   `fp_x_flux`, so \(H_P=HU\).
2. **Weighted left inverse:** \(\Gamma_RU=I\), including nonuniform signed
   nonzero \(\mu\).
3. **Conditional prolongation:**
   \(B\Gamma_R^T(\Gamma_RB\Gamma_R^T)^{-1}=U\).
4. **Independence:**
   \(\Gamma_RB(I-U\Gamma_R)^T=0\) to numerical tolerance.
5. **Prior decomposition:**
   \(B=UB_PU^T+B_c\) and \(B_c\succeq0\) on small dense examples.
6. **Observation decomposition:**
   \(C_{\mathrm{full}}=C_P+C_{\mathrm{agg}}\) and
   \(R_P+C_P=R+C_{\mathrm{full}}\) for every enumerated toy partition.
7. **DFS parity:** additive node DFS equals direct
   \(\operatorname{tr}[(R_P+C_P)^{-1}C_P]\).
8. **Bound and decomposition:** partition DFS never exceeds native full-grid
   DFS, and the resolved plus unresolved DFS terms equal the full value.
9. **Euclidean counterexample:** the two-cell case above has nonzero cross
   covariance and rejects the independent-residual shortcut.
10. **Support behavior:** zero-support nodes are pruned, partial coarsening
    blocks count only supported native cells, and excluded design columns are
    zero within a declared tolerance.
11. **Units and area:** an independently constructed native \(G\) matches
    project `fp_x_flux` after exactly one documented conversion.
12. **Numerics:** covariance symmetry is restored only for roundoff; materially
    negative modes or broken innovation invariance raise rather than being
    clipped away.

## 11. Repository reference points

- `openghg_inversions/basis/operators.py`:
  `BasisOperator.sensitivity()` sums `fp_x_flux` over one-hot region membership.
- `openghg_inversions/basis/basis_functions.py`:
  `BasisFunctions.sensitivity()` exposes that reduction.
- `openghg_inversions/inversion_data/preparation.py`:
  `_rhime_site_data_from_basis_functions()` builds the current RHIME `H` from
  `fp_x_flux`.
- `openghg_inversions/basis/experimental/dyadic/multiscale.py`:
  sum-preserving coarsening, candidate node columns, support-aware partial
  blocks, and direct-gather parity.
- `openghg_inversions/basis/experimental/dyadic/rhime_gaussian.py`:
  experimental native-signal covariance, \(s^2/n_k\) regional variances,
  invariant innovation covariance, additive node DFS, full-grid bound, and
  aggregation covariance.
- `openghg_inversions/basis/experimental/dyadic/objectives.py`:
  generic Gaussian DFS and the explicitly provisional isotropic-region and
  historical quadratic benchmark objectives.
- `tests/basis/experimental/test_dyadic_multiscale.py` and
  `tests/basis/experimental/test_dyadic_objectives.py`:
  existing sum/gather and generic DFS parity checks.
- `docs/plans/dyadic_partition_inference.md`, especially *Objectives and what
  they mean*, *Hackathon proof of concept*, *Validation requirements*, and
  *Risks and exactness boundaries*.
- `docs/plans/dyadic_sls_hackathon.md` and
  `docs/reports/dyadic_sls_poc_slides.md`:
  provenance for the factor-8 demonstration and the provisional covariance
  warning.
- `openghg_inversions/models/components.py`:
  current independent-normal likelihood, relevant to any future production
  correlated-covariance boundary.
- `openghg_inversions/rhime/runner.py` and
  `openghg_inversions/hbmcmc/hbmcmc.py`:
  production boundaries explicitly excluded from the experimental phase.

## 12. External references

1. Bocquet, M., Wu, L., and Chevallier, F. (2011), “Bayesian design of control
   space for optimal assimilation of observations. Part I: Consistent
   multiscale formalism,” *Quarterly Journal of the Royal Meteorological
   Society*, 137, 1340--1356, doi:10.1002/qj.837.

   - Section 2.3, printed pp. 1343--1344 / PDF pp. 4--5, Equations 2--14:
     projected prior, Bayesian prolongation, projector, refinement covariance,
     left-inverse and \(B^{-1}\)-symmetry identities, and the representation
     observation equation.
   - Section 2.5, printed p. 1344 / PDF p. 5, Equations 15--19: whitening for
     the correlated-prior case.
   - Section 3.2, printed p. 1345 / PDF p. 6, Equations 23--31: aggregation
     error, scale-dependent covariance, innovation invariance, and consistency
     of coarse and fine Gaussian analyses.
   - Section 4.1.2, printed p. 1346 / PDF p. 7, Equations 38--39: DFS under the
     invariant innovation covariance and tile-level contributions.

2. Bishop, C. M. (2006), *Pattern Recognition and Machine Learning*, Section
   2.3, “The Gaussian Distribution.”

   - Section 2.3.1, printed pp. 85--87 / supplied PDF pp. 105--107, especially
     Equations 2.65--2.82: partitioned Gaussian conditioning, conditional mean,
     and Schur-complement covariance.
   - Section 2.3.2, printed pp. 88--89 / supplied PDF pp. 108--109, Equations
     2.83--2.93: Gaussian marginalization and the distinction between marginal
     and conditional covariance.

3. *Reduced-dimension basis functions, prior uncertainty, and aggregation
   error in RHIME-style inversions* (supplied reference note, 2026-06-12).
   Sections 1--5 provide useful RHIME interpretation and prior-calibration
   background. Section 6 and the Euclidean-residual formulas that depend on its
   independence assumption require the correction stated in Sections 2--4 of
   this report.
