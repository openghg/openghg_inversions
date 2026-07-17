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

## 6. Representation objectives and lifted posterior approximations

### 6.1 DFS, additive node scores, and the fine-grid bound

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
nonnegative and vanishes for the supported native-grid partition. If the search
dictionary has factor-8 spatial leaves, the native-grid partition is not
reachable; the native full-grid value remains a bound, not an attainable search
state.

### 6.2 Exact projected posterior versus a lifted approximation

Bocquet et al.'s Equations 29--31 establish that an analysis performed in a
representation is the exact restriction of the native analysis. In the notation
of Section 2, if

\[
p(\delta x\mid y)=\mathcal N(m_a,P_a),
\qquad
\alpha=\Gamma\delta x,
\]

then

\[
p(\alpha\mid y)
=\mathcal N(\Gamma m_a,\Gamma P_a\Gamma^T).
\]

Equivalently, their Equation 31 is the pushforward identity

\[
p(\alpha\mid y)
=\int \delta(\alpha-\Gamma\delta x)\,
p(\delta x\mid y)\,d\delta x.
\]

This identity means that country totals, regional averages, or any other
declared linear summaries can be inferred directly and will agree with applying
the same restriction to a native-grid Gaussian inversion. It does not recover
the posterior update in directions discarded by \(\Gamma\).

The identity requires the paper's declared base-error model plus the exact
aggregation error in Equations 23--24. Additional scale-dependent model error
must be modelled explicitly; otherwise the innovation identity in Equation 27
and the exact pushforward result need not hold.

To compare representations of different dimensions as distributions in one
common space, each projected posterior must be lifted back to the native state
space. The natural Bocquet-compatible lift is

\[
q_\Gamma(\delta x\mid y)
=\int p(\delta x\mid\alpha)\,p(\alpha\mid y)\,d\alpha,
\]

where \(p(\delta x\mid\alpha)\) is the **prior** conditional distribution. Its
mean and covariance are

\[
m_\Gamma=\Lambda\,\mathbb E[\alpha\mid y],
\qquad
P_\Gamma=B_c+\Lambda\operatorname{Cov}(\alpha\mid y)\Lambda^T.
\]

This lifted approximation updates the selected summaries while leaving the
conditional unresolved contrasts at their prior distribution. It provides a
precise approximation-class interpretation of the candidate representations.
Within the class of native distributions that retain this prior conditional,
it is the forward-KL projection of the full posterior. Reverse KL does not in
general select the same approximation.

The KL chain rule gives

\[
\begin{aligned}
D_{\mathrm{KL}}\!\left[p(\delta x\mid y)\,\|\,p(\delta x)\right]
={}&D_{\mathrm{KL}}\!\left[p(\alpha\mid y)\,\|\,p(\alpha)\right]\\
&+\mathbb E_{p(\alpha\mid y)}
D_{\mathrm{KL}}\!\left[
p(\delta x\mid\alpha,y)\,\|\,p(\delta x\mid\alpha)
\right].
\end{aligned}
\]

The second term is also the forward KL divergence from the full posterior to
the prior-conditionally lifted approximation. Therefore

\[
\boxed{
D_{\mathrm{KL}}\!\left[p(\delta x\mid y)\,\|\,q_\Gamma(\delta x\mid y)\right]
=K_{\mathrm{full}}^{\mathrm{Bayes}}-K_\Gamma^{\mathrm{Bayes}}
}.
\]

Maximizing the projected Bayesian information gain is consequently equivalent
to choosing the member of this approximation class nearest to the full
posterior in forward KL divergence. Averaging over possible observations gives
the corresponding mutual-information identity

\[
\mathbb E_y D_{\mathrm{KL}}(p\|q_\Gamma)
=I(\delta x;y)-I(\alpha;y).
\]

Two other lifts do not define the same useful comparison. A deterministic
rank-\(K\) prolongation is singular in native space, so its KL divergence from a
full-rank posterior is normally infinite. Using the exact posterior conditional
\(p(\delta x\mid\alpha,y)\) instead of the prior conditional reconstructs the
full posterior for every \(\Gamma\), leaving every representation at distance
zero.

This lifted-posterior identity is a consequence of Bocquet et al.'s exact
coarse-analysis identities and the KL chain rule. The paper does not present it
as a separate hypothesis-class theorem.

### 6.3 Relation of the objectives to Bocquet et al.'s equations

The paper's criteria are different summaries of how much of the native update
is represented by \(\Pi_\omega=\Gamma_\omega^\dagger\Gamma_\omega\):

| Objective | Paper equations | Approximation interpretation |
| --- | --- | --- |
| Fisher | 33--36 | Prior-normalized sensitivity; Equation 36 is the weak-signal or inflated-\(R\) limit of DFS and is additive under Equation 50 |
| Aggregation-aware Fisher | 37 | Uses the partition-dependent \(R_\omega^{-1}\); nonlinear in \(\Pi_\omega\) and not a scalar-node DP objective |
| DFS | 38--39 | Trace of captured posterior uncertainty reduction; Equation 39 localizes it to a tile |
| Bayesian relative entropy | 40 and expected form 42 | Prior-to-posterior KL information; its representation-specific form gives the closest-lifted-posterior criterion above, but is not generally additive |
| Data-dependent mean criterion | 41 and representation forms 44--45 | Squared prior-weighted posterior-mean update captured by the representation |
| Expected data-dependent criterion | 43 | One-half of DFS under the declared prior-predictive distribution |

For the data-dependent criterion, set

\[
\delta m
=BH^TS^{-1}y,
\]

where \(y\) is already centered on the native prior prediction. Equation 41 is

\[
K_\sigma=\frac12\|\delta m\|_{B^{-1}}^2,
\]

and Equation 45 is, apart from the paper's normalization convention,

\[
J_\omega=\|\Pi_\omega\delta m\|_{B^{-1}}^2.
\]

Because \(\Pi_\omega\) is an orthogonal projector in the \(B^{-1}\) metric,
maximizing Equation 45 minimizes the squared \(B^{-1}\)-norm of the
posterior-mean update omitted by the representation. This is the mean-only
analogue of the full KL approximation result; it deliberately omits the
covariance terms in Equation 40.

Equation 38 can similarly be written in prior-whitened coordinates as

\[
J_\omega=\operatorname{tr}(\widetilde\Pi_\omega\mathcal A),
\qquad
\mathcal A
=B^{1/2}H^TS^{-1}HB^{1/2}.
\]

The difference from native DFS is

\[
\operatorname{DFS}_{\mathrm{full}}-J_\omega
=\operatorname{tr}[(I-\widetilde\Pi_\omega)\mathcal A],
\]

a trace measure of the uncertainty reduction omitted by the representation.
It is a useful approximation loss, but not itself a probability-distribution
distance.

If \(\lambda_i\) are eigenvalues of the prior-whitened Fisher matrix
\(B^{1/2}H^TR^{-1}HB^{1/2}\), the unrestricted criteria weight an observable
mode as

\[
\text{Fisher: }\lambda_i,
\qquad
\text{DFS: }\frac{\lambda_i}{1+\lambda_i},
\qquad
\text{expected Bayesian KL: }\frac12\log(1+\lambda_i).
\]

They favor similar directions but value already well-resolved directions
differently. Under a constrained aggregation dictionary they can therefore
rank partitions differently and should all be retained as explicit,
separately named experiments.

### 6.4 Using the realized observations to choose a representation

Equation 45 is intentionally data-dependent. Using it to choose \(P\) is not
automatically incoherent, but the inferential claim must match the role of the
selection:

- If \(P(y)\) is a predeclared **posterior-compression action**, then selecting
  it from \(y\) and reporting the exact posterior of the selected linear
  summary is a coherent Bayesian decision. Conditional on the observed data,
  \(P(y)\) is known and Equations 29--31 still give the exact posterior of that
  selected summary.
- The same result is not independent evidence that the selected geography is
  scientifically special, nor is its training score an out-of-sample
  performance estimate. Region boundaries and reported contrasts were chosen
  adaptively.
- If the selected partition is instead treated as a fixed generative model and
  its selection is omitted from the uncertainty calculation, posterior
  uncertainty can be understated and noise-specific refinements can be
  overinterpreted. This is especially relevant when the practical model is
  only approximately scale-consistent.
- If \(P\) is intended to be an uncertain scientific parameter, Equation 45 is
  a utility, not \(p(P\mid y)\). A prior and a genuinely
  partition-dependent likelihood are required for posterior inference over
  \(P\).

For every fixed candidate, Equation 43 gives the prior-predictive expectation
of the data-dependent score. Selection changes that expectation:

\[
\mathbb E\left[J_{\widehat P}(y)\right]
=\mathbb E\left[\max_P J_P(y)\right]
\geq \max_P\mathbb E[J_P(y)].
\]

The selected training score is therefore optimistically biased, with greater
potential optimism for a larger or more flexible partition dictionary. The
analysis increment used by Equation 45 contains not only source signal but also
realized observation error, prior-mean mismatch, transport error, and any
misspecification of \(H\), \(B\), or \(R\).

Bocquet et al. explicitly call attention to the possible "inversion crime" of
building an adaptive representation from the same observations used in the
inversion. For the experimental posterior-compression interpretation, the
minimum defensible protocol is:

1. Declare the representation dictionary, \(K\), covariance assumptions, and
   Equation 45 objective before inspecting results.
2. Label the selected partition as data-adaptive posterior compression, not a
   posterior draw or a discovered physical boundary.
3. Report Equation 45 on the fitting observations, but use held-out sites or
   times for predictive or compression assessment.
4. Compare with data-independent DFS from Equation 38 and Fisher from Equation
   36 so the effect of the realized innovation is visible.
5. Use cross-fitting or a design/inference split when repeated-sampling
   calibration or confirmatory claims matter. If tuning choices are compared,
   reserve an outer holdout for final evaluation.

A final descriptive inversion may refit with all observations after the
selection rule is frozen, but its intervals do not become selection-adjusted
merely because a holdout was used during development.

### 6.5 Dynamic programming and correlated geographic aggregation

Equation 45 is linear in \(\Pi_\omega\). Under the assumptions leading to
Equations 50--51, the projector is a sum of orthogonal tile projectors and

\[
J_\omega=\sum_{t\in\omega}\epsilon_t.
\]

Equation 64 makes the data-dependent matrix rank one. With
\(u=B^{1/2}H^TS^{-1}y\), a tile represented by \(v_t\) has score

\[
\epsilon_t=\frac{(v_t^Tu)^2}{v_t^Tv_t}.
\]

An exact fixed-\(K\) dynamic program is therefore available for a recursive
dyadic or quadtree dictionary:

\[
F(t,1)=\epsilon_t,
\qquad
F(t,k)=
\max_{k_1+\cdots+k_m=k}\sum_{j=1}^m F(c_j,k_j).
\]

A variable-count utility can be obtained by maximizing
\(F(\mathrm{root},K)-g(K)\). This is an optimization over a declared utility,
not Bayesian inference of \(K\) under the exact scale-consistent Gaussian
model. Equation 63 explains why the positive-semidefinite trace objectives are
otherwise monotone as the representation is refined.

The dynamic program is globally exact only within its declared recursive
dictionary. The current repository implementation uses one canonical binary
tree; it is not an optimizer over every general tiling considered by Bocquet et
al. A full fixed-count frontier can be computed once and combined afterward
with any penalty depending only on \(K\). Full Bayesian KL, held-out objectives,
and dense-covariance objectives are not made additive merely by using the same
tree.

Linearity in \(\Pi_\omega\) alone is not sufficient for this dynamic program.
The additive tile expansion also requires the orthogonality assumptions behind
Equation 50. Bocquet et al. accept a whitening redefinition in Equations 15--19
and 47--48 to recover this structure for correlated \(B\). They note that the
resulting coordinates are statistically independent linear combinations, not
literal geographic aggregates.

Retaining correlated \(B\) and geographic regions is mathematically possible,
but it creates global coupling. There are two exact conventions:

1. Preserve a piecewise-regional prolongation \(U\). Then

   \[
   \Gamma_U=(U^TB^{-1}U)^{-1}U^TB^{-1},
   \qquad
   B_P=(U^TB^{-1}U)^{-1}.
   \]

   The forward columns remain geographic region sums, but the selected
   coefficients are generalized least-squares regional amplitudes rather than
   literal arithmetic totals.
2. Preserve a literal area- or flux-weighted restriction \(\Gamma\). Then

   \[
   \Lambda=B\Gamma^T(\Gamma B\Gamma^T)^{-1}.
   \]

   The selected coefficients retain their literal aggregate meaning, but the
   Bayesian prolongation and \(H\Lambda\) are generally nonlocal rather than
   simple region sums.

For the second convention, the data-dependent score can be evaluated as

\[
J_\Gamma
=(\Gamma\delta m)^T(\Gamma B\Gamma^T)^{-1}(\Gamma\delta m).
\]

This displays both the literal projected analysis increment and the dense
reduced covariance that couples all selected regions.

In either convention, the reduced covariance is normally dense. A partition
evaluation requires construction and factorization of a \(K\times K\) matrix,
and changing one region can change the score assigned to every other region.
The scalar-node dynamic program is therefore unavailable. A single
factorization at \(K\) around 250 is modest; repeating the construction and
factorization for thousands of candidate partitions may dominate the search.
Sparse native precision, compactly supported covariance, separable operators,
cached pairwise tile interactions, and incremental Cholesky updates could make
this practical, but each requires a separate correctness and performance
study. Stochastic search remains the straightforward exact-objective fallback.

### 6.6 Region count, coefficient dimension, and information dimension

Three quantities must remain separate:

\[
K_{\mathrm{tree}}=|P|,
\qquad
K_{\mathrm{eff}}
=\#\{\text{active regions with supported native grid locations}\},
\qquad
\operatorname{DFS}(P).
\]

The paper uses \(N\) for tile count, \(N_{\mathrm{fg}}\) for native grid size,
and \(p\) for observation count. Repository \(K_{\mathrm{tree}}\) corresponds
to the paper's \(N\) only when every active region has support. Under the
current disjoint construction, \(K_{\mathrm{eff}}\) is the reduced coefficient
dimension and the rank of the supported prolongation. DFS is an information
dimension bounded by both the observation rank and \(K_{\mathrm{eff}}\); it is
not a region count.

The current DP constrains tree leaves, while diagnostics may report effective
supported regions. Unsupported leaves must be pruned or both counts must be
reported. Under whitening, \(K\) counts retained statistical coordinates, not
literal geographic regions.

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
weighted regional coefficients are independent. Section 6.5 records the two
exact ways to retain geographic regions, their different coefficient
semantics, and their global computational coupling. Bocquet et al.'s whitened
restriction in Equations 15--19 and 47--48 restores additive optimization by
changing the tile interpretation; it is a computationally useful alternative,
not a mathematical requirement for exact Gaussian projection.

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

Compare the data-independent DFS, Fisher, and data-dependent Equation 45
objectives. Use exact dynamic programming whenever Equation 50 gives additive
tile scores, and use the DP result as the fixed-count reference for stochastic
search. For Equation 45, record the observations used to select the partition
and reserve independent sites or times for predictive evaluation. Report
separately:

- partition DFS and native full-grid DFS;
- unresolved DFS and the exact decomposition residual;
- effective supported-region count;
- coarsening factor and inaccessible within-leaf information;
- support threshold, units, source/sign handling, and base \(R\); and
- sensitivity to plausible \(s\), support, and coarsening choices.

Use repeated semi-synthetic inversions to measure selection effects. Compare a
prespecified partition, data-independent DFS selection, same-data Equation 45
selection, and blocked holdout or cross-fitted Equation 45 selection. Evaluate
common prespecified native-grid or large-region functionals, not only the
adaptively selected regions. Record coverage, bias, held-out log predictive
density, partition stability, training-versus-holdout optimism, and sensitivity
to misspecified prior means, \(B\), \(R\), and transport.

The exact scale-consistent Gaussian likelihood is partition-invariant. Region
count and representation are therefore optimization or reporting decisions in
this model, not parameters learned through a nontrivial \(p(P,K\mid y)\).

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
   - Sections 4.1.1--4.1.3, printed pp. 1346--1347 / PDF pp. 7--8, Equations
     33--45: Fisher, DFS, Bayesian relative entropy, and data-dependent
     representation objectives.
   - Sections 4.2--4.3, printed pp. 1347--1348 / PDF pp. 8--9, Equations
     47--51: correlated-prior whitening and additive tile energies.
   - Sections 5.3 and 6.1, printed pp. 1350--1352 / PDF pp. 11--13, Equations
     62--64: region-count marginal utility, monotonicity, and the rank-one
     data-dependent objective.

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
