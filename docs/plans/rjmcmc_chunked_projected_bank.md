# Chunked projected Gamma--Dirichlet source bank

## Status and scope

This is a working background/design note for the next aggregation-error
experiment.  It belongs in `openghg_inversions` until the numerical
experiments have run; it is not yet a durable `inversions-knowledge`
derivation.

The implemented constructor is:

```python
build_chunked_projected_root_bank(...)
```

in
`openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture`.
It is experimental, single-root only, float64 throughout, and has a distinct
version-three source-artifact identity.  Existing version-one PCG64 and
version-two all-at-once Sobol artifacts are unchanged.

## Model and approximation boundary

For native Gamma shapes \(\alpha_i\), let

\[
\eta=\sum_i\alpha_i,\qquad u_i=\alpha_i/\eta,\qquad
w\sim\operatorname{Dirichlet}(\alpha).
\]

For one retained total mass \(T\), native design \(H\), diagonal measurement
scale \(D^{1/2}\), and fixed offset \(b\),

\[
y\mid T,w\sim
\mathcal N\{b+T H w,D\}.
\]

After centring at \(Hu\), whitening by \(D^{-1/2}\), and rotating into the
analytic residual eigenbasis \(U_r\), the exact likelihood is a continuous
Gaussian location mixture over

\[
z(w)=U_r^\mathsf T D^{-1/2}H(w-u).
\]

The proposed numerical approximation has three distinct layers:

1. a frozen scrambled-Sobol discretization of the Dirichlet law;
2. a non-Gaussian finite mixture in the leading \(q\) coordinates; and
3. an analytic Gaussian moment closure in coordinates \(q+1,\ldots,r\).

The full spectrum rank \(r\) and the source-bank rank \(q\) are deliberately
different.  The bank needs to store only \(z_{1:q}\).  The eigenvectors and
eigenvalues through \(r\) remain in `RootResidualSpectrum`, and
`CompressedRootMixture` adds the \(q+1:r\) covariance complement at
likelihood evaluation.  This is the same constant-in-parameter
moment-closure covariance discussed in `inversions-knowledge`, scaled by
\(T^2\); it is not another sampled residual.

For the first PARIS experiment, retain the complete numerical spectrum
\(r=1381\).  This isolates the Gaussian shape approximation from literal
projection omission.  A later reduced-\(r\) experiment would need its own
predeclared error criterion.

## Frozen Sobol construction

The existing version-two source catalogue is preserved:

- cells are ordered by stable scientific cell ID;
- the Dirichlet inverse uses a count-balanced binary tree in breadth-first
  node order;
- dimensions are contiguous blocks of at most 21,201;
- each block uses the existing version-two seed derivation;
- SciPy scrambled Sobol uses 52 bits and no optimizer;
- internal splits use `scipy.special.betaincinv`; and
- the right child is the represented complement
  `parent_mass - left_mass`.

The PARIS root has 23,423 split coordinates and therefore the frozen block
dimensions `[21201, 2222]`.

One persistent Sobol engine is created for each dimension block.  Calls to
`Sobol.random(C)` advance each engine sequentially over sample chunks.
Repeated `random_base2` calls are not used, and `fast_forward` is not part of
the contract.  In the pinned SciPy 1.15.2 environment, `fast_forward` with
the 52-bit engine has also been observed to fail because of an internal
integer-width mismatch.  Interrupted construction therefore restarts from
sample zero.

## Chunked projection

Let:

- \(S\): number of frozen Sobol samples;
- \(N\): native cells;
- \(q\): stored non-Gaussian coordinates;
- \(C\): allocation/sample chunk size; and
- \(P\): fixed projection microbatch size.

For each chunk, the constructor:

1. generates the next \(C\) Sobol rows in every dimension block;
2. evaluates each block's inverse-Beta coordinates in one SciPy array call;
3. materializes only that chunk's \(C\times N\) allocation shares;
4. projects fixed \(P\)-row microbatches into
   \(U_q^\mathsf T D^{-1/2}H(w-u)\);
5. writes those rows into the persistent \(S\times q\) bank; and
6. discards shares, inverse-Beta values, and live tree masses.

Holding \(P\) fixed while varying \(C\) fixes the BLAS matrix shape and hence
the projected floating-point result.  This separates memory/throughput
tuning from the scientific bank.  Both values are recorded in the v3
artifact.  The artifact SHA therefore records the engineering traversal even
when the projected scientific-array digest is unchanged.  The current
contract requires positive powers of two with

\[
P\le C\le S.
\]

The persistent bank is \(8Sq\) bytes.  The dominant explicit temporary arrays
scale as

\[
O\{8C(N+2\min(N-1,21201))\},
\]

for shares, one uniform block, and its inverse-Beta fractions, plus live
tree-mass and library workspaces.  This replaces the old
simultaneous \(8SN\) shares and largest \(8S\times21201\) uniform block.  At
\(S=65536,q=32\), projected locations occupy 16 MiB; at \(q=128\), 64 MiB.

The full process peak also includes the loaded NetCDF, physical-mass design,
spectrum construction, immutable-array copies, and artifact fingerprinting.
Resource tests must cover the complete constructor, not only its inner loop.

## Serialization

The Python object retains canonical JSON replay because this preserves the
existing artifact model and is manageable for an \(S\times q\) bank.  A
production HPC bundle should additionally store the projected locations as
one create-only little-endian float64 C-order `.npy` or uncompressed `.npz`
member with a small canonical JSON manifest.  The manifest should record:

- shape, dtype, order, and array SHA-256;
- \(S,q,r,C,P\), source seed, concentration, and SciPy version;
- frozen input and Git identities;
- alpha, design, noise, basis, catalogue, and Sobol block identities; and
- the complete spectrum eigenvalue and basis identities.

Do not emit one file per sample or component.  JSON-list serialization is an
identity/replay mechanism, not the preferred PARIS-scale interchange format.

## Scientific restrictions

The basis and all ranks must be observation-blind.  In particular, do not
choose \(q\), \(r\), concentration, component count, or a Sobol seed using
the realized PARIS residual, posterior fit, or apparent preference for a
partition.

The earlier \(\eta=100\) and \(\eta=500\) runs were useful engineering
scaling checks, but they represent different native priors.  A scientific
fixed-root or cross-partition experiment needs one common native
concentration chosen independently of \(K\).

An exact common-native marginal likelihood contains no information about a
purely computational partition.  Any finite preference among such
partitions is approximation leakage unless a partition-indexed scientific
model is explicitly declared.  This bank is therefore first a route to a
fixed-root marginal likelihood and an approximation audit, not authorization
for RJ acceptance or structural Bayes factors.

## Implementation decisions still gated by HPC evidence

The code fixes the mathematical and replay contracts, but the HPC protocol
must still:

- choose one projection microbatch \(P\) before resource tuning;
- select an allocation chunk \(C\) only by identical-output resource rules;
- freeze one authenticated full PARIS spectrum/basis artifact;
- validate the observation-blind \(q\) ladder on prior-predictive and
  independent-scramble cases;
- establish an acceptable source sample size \(S\); and
- test clustering separately after the source bank is locked.

These are described in
[`rjmcmc_chunked_projected_bank_hpc_test_plan.md`](rjmcmc_chunked_projected_bank_hpc_test_plan.md).
