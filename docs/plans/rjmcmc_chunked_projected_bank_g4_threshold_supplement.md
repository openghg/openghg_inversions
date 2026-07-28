# BP1 supplement: observation-blind G4 source-bank thresholds

## Status and scope

This supplement freezes the scientific Gamma--Dirichlet calibration, the G4
validation catalogue, and every numerical pass rule before any G4 result is
constructed or inspected.  It supplements
[`rjmcmc_chunked_projected_bank_hpc_test_plan.md`](rjmcmc_chunked_projected_bank_hpc_test_plan.md).

G4 can certify finite-bank, nested-prefix, retained-rank, and independent
scramble stability.  It cannot prove absolute accuracy against the continuous
PARIS marginal because no exact 23,424-cell PARIS quadrature oracle exists.
The exact tiny-case quadrature suite remains the code-path oracle.

No realized `mf`, realized residual, protected catalogue, partition, \(K\), or
approximation result enters this supplement.

## Scientific native calibration

For native normalized area weights \(u_i=\texttt{nominal_weight}_i\), draw

\[
p\sim\operatorname{Dirichlet}(\eta u),\qquad
T\sim\operatorname{Gamma}(a,a),\qquad v=\operatorname{Var}(T)=a^{-1}.
\]

The simulated native scaling is \(T p_i/u_i\).  For a physical aggregate
with non-negative coefficients

\[
b_{Ri}=
\texttt{prior_flux}_i\,
\texttt{grid_cell_area}_i\,m_{Ri},
\]

define

\[
D_R =
\sum_i\frac{b_{Ri}^2}{u_i(\sum_jb_{Rj})^2}-1.
\]

Its exact coefficient of variation is

\[
\operatorname{CV}_R^2 =
v+(1+v)\frac{D_R}{\eta+1}.
\]

The broad aggregate is the complete modeled \(183\times128\) European inner
grid, latitude 36.469--79.057 degrees and longitude
\(-14.124\)--30.580 degrees.  It excludes the six `outer_design`
coefficients and `YaprioriBC`; it is not political EU membership.  Its mask is
identically one.  The local aggregate uses the frozen
`country_fraction.sel(country="GBR")`; that mask is not regenerated.

On frozen input SHA-256
`24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044`,

```text
D_modeled_domain = 8.874250601269965
D_GBR            = 117.60829531564202
domain total     = 58254.53159833662 mol s-1
GBR total        = 4105.507335053642 mol s-1
```

The supplied targets are

```text
modeled European-domain physical-total CV = 0.20
GBR physical-total CV                     = 0.50
```

The unique feasible two-moment solution is frozen as

```text
eta                         = 528.618161317525
root variance v             = 0.022861001527515423
root CV                     = 0.15119855001790006
root Gamma shape and rate a = 43.742615510366136
```

The executable driver must recompute the two \(D_R\) values and solved
parameters from exactly aligned prior/operator fields and fail closed on any
identity or numerical mismatch.  The observation-blind standardized
skewnesses, approximately 0.793 for the domain aggregate and 3.126 for GBR,
are recorded diagnostics rather than gates.  Matching two CVs does not make
the positive aggregate distributions Gaussian.

## Frozen bank catalogue

The source controls remain

```text
S_max:        65536
nested S:     16384, 32768, 65536
q ladder:     16, 32, 64, 128
development:  seed 731
confirmation: seeds 1877, 4099, 8317
```

The full analytic numerical spectrum remains \(r=1381\).  At each \(q\), the
finite equal-weight source mixture represents coordinates \(1{:}q\);
coordinates \(q+1{:}r\) use the analytic Gaussian moment complement.

Let \(z^{(a,N,q)}\) be the first \(N\) rows and \(q\) columns of the bank for
seed \(a\), and let \(\lambda_j\) be the analytic unit-root eigenvalues.
Population covariance always uses divisor \(N\).

## Moment gates

Set

\[
\tau_\lambda =
\max\{\text{spectrum eigenvalue tolerance},
1024\epsilon_{64}\max(1,\lambda_1)\},
\quad
A_q=\{j\le q:\lambda_j>\tau_\lambda\},
\]

\[
x_{sj}=z_{sj}/\sqrt{\lambda_j},\qquad
h_N=\sqrt{65536/N}.
\]

For every tested seed, \(N\), and \(q\), require

\[
\max_{j\in A_q}|\bar x_j|\le0.02h_N,
\]

\[
\frac{\|\widehat{\operatorname{Cov}}(x_{A_q})-I\|_F}
{\sqrt{|A_q|}}\le0.06h_N.
\]

Tiny directions are excluded only from relative division.  If any are
present, require

\[
\max|\bar z_j|\le0.02h_N\sqrt{\tau_\lambda},
\]

and maximum absolute covariance error, including tiny cross terms, no larger
than \(0.06h_N\tau_\lambda\).  Maximum diagonal relative error and maximum
empirical correlation are reported diagnostics.

## Marginal and joint tail stability

Every comparison below is applied both to adjacent nested prefixes
\((16384,32768)\), \((32768,65536)\) of one seed and to every pair among the
four seeds at \(N=65536\).  No failed comparison is averaged away.

For every retained standardized coordinate, require:

- two-sample empirical-CDF distance at most 0.020;
- absolute difference in each one-sided probability beyond \(+2\) and
  \(-2\) at most 0.005; and
- absolute difference beyond \(+3\) and \(-3\) at most 0.002.

The frozen one-based coordinate-pair catalogue is

```text
(1,2), (15,16), (16,17), (31,32), (32,33),
(63,64), (64,65), (127,128)
```

using only pairs present at the tested \(q\).  For every pair require:

- difference in
  \(P\{\max(|x_j|,|x_k|)\ge2\}\) at most 0.005; and
- difference in each of the four signed joint events
  \(P(sx_j\ge2,tx_k\ge2)\), \(s,t\in\{-1,+1\}\), at most 0.002.

For each \(d\in\{2,16,32,64,128\}\) present at \(q\), put
\(R_d=\sum_{j=1}^d x_j^2\).  Require:

- empirical-CDF distance for \(R_d\) at most 0.020;
- difference in probability above the \(\chi_d^2(0.99)\) marker at most
  0.005; and
- difference above the \(\chi_d^2(0.999)\) marker at most 0.002.

The chi-square values are fixed radial scale markers, not an assertion that
the source law is Gaussian.

## Exact observation-blind likelihood grid

G4 uses 256 equal-weight deterministic states indexed \(i=0,\ldots,255\).
No `mf` value is read.

Root masses are stratified exact Gamma quantiles:

\[
T_i =
F^{-1}_{\operatorname{Gamma}(a,a)}
\left(\frac{(73i\bmod256)+0.5}{256}\right),
\quad a=43.742615510366136.
\]

The held-out allocation catalogue uses the existing balanced-Dirichlet Sobol
coordinates, source seed 12947, \(S=256\), the complete numerical spectrum,
and the committed chunked traversal.  It is disjoint from every bank seed.

Measurement-noise coordinates use scrambled Sobol dimension 1382, 52 bits,
seed 12953, transformed with `scipy.special.ndtri` after clipping to
\([2^{-53},1-2^{-53}]\).

The six outer coefficients use a separate scrambled Sobol catalogue with
dimension six, 52 bits, and seed 12959.  They are transformed to the
arithmetic-mean-one, arithmetic-SD-one lognormal prior,

\[
\mu_{\log}=-\tfrac12\log2,\qquad
\sigma_{\log}=\sqrt{\log2}.
\]

For held-out unit-root whitened allocation residual \(r_i\), noise
\(\epsilon_i\), outer vector \(c_i\), mean design \(\bar H\), and noise scales
\(\sigma\), define

\[
b_i=\texttt{YaprioriBC}+\texttt{outer_design}\,c_i,
\]

\[
y_i=b_i+T_i\bar H+
\sigma\odot(T_i r_i+\epsilon_i).
\]

The bundle records and authenticates every mass, coefficient, offset,
observation, residual, operator, spectrum, and source-array hash.  Evaluating
\((y_i,b_i)\) and the translated pair \((y_i-b_i,0)\) must differ by at most

\[
4096\epsilon_{64}\max(1,|\ell_{\rm original}|,|\ell_{\rm translated}|).
\]

## Likelihood stability gates

All likelihoods are normalized in the original observation units and must be
finite.  Equal source weights must sum to one within
\(64\epsilon_{64}N\); every Gaussian-complement variance must be positive;
and nested row/coordinate identities must be exact.

For each seed and \(q\), compare \(N=16384\) and \(32768\) separately with
\(N=65536\) on all 256 states.  For absolute differences \(d_S\), require

```text
median(d_S) <= 0.05 nat
p99(d_S)    <= 0.20 nat
```

At \(N=65536\), compare each \(q=16,32,64\) with \(q=128\) under the same
limits.  The \(q=128\) self-comparison is exact.

For every \(q\), the pointwise range across all four seed likelihoods,
\(\max_a\ell_a-\min_a\ell_a\), must also have median at most 0.05 nat and
99th percentile at most 0.20 nat.

These are operational leakage tolerances.  A 0.05-nat change is about a 5%
likelihood-ratio change; 0.20 nat is about 22%.  They are not absolute error
bounds against the unavailable continuous PARIS oracle.

## Development suffix and confirmation

A development \(q\) passes only when:

1. its \(N=65536\) moment gates pass;
2. both nested-prefix moment, marginal-tail, joint-tail, and likelihood gates
   pass;
3. normalization, support, identity, finiteness, and offset-translation gates
   pass; and
4. its \(q\)-versus-128 likelihood gate passes.

The development lock is the smallest \(q\) beginning an all-larger passing
suffix of at least two consecutive ladder values.  Thus \(q=128\) alone
cannot lock: at minimum 64 and 128 must pass.  No common suffix is a terminal
G4 hard stop; the envelope is not extended after inspection.

Only after the development lock may seeds 1877, 4099, and 8317 run as one
homogeneous Slurm array.  Every \(q\) in the development-locked suffix must
pass every individual within-seed gate.  Every pairwise marginal and joint
tail comparison and every cross-seed likelihood-range gate must pass.  One
failure produces no source lock; seeds are not replaced, retried
selectively, or averaged.

The final source lock records the smallest \(q\) beginning the common
all-seed suffix.  It licenses G5 clustering only.  It does not license an RJ
likelihood, partition weights, a posterior, or production output.
