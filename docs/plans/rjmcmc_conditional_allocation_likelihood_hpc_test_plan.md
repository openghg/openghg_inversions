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

The first reviewed implementation landed at
`3e30f9117bcba03920aafd338f7eea529c25b079`. C0 passed on BP1 with 69
focused aggregation-error tests, Ruff, and Pyright:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/3e30f9117bcba03920aafd338f7eea529c25b079/c0
```

The broader local aggregation/transported-mixture/fixed-basis-PyMC focused
group passed 109 tests. C0 validates implementation and target mechanics; it
does not replace the C1 scientific approximation screen.

The first C1 harness review found that raw quantiles over Gauss-quadrature
nodes would over-weight numerous negligible-probability tail nodes. The
preserved A1 protocol instead used normalized quadrature-prior weights and
exact-posterior weights. C1 must preserve those weighted diagnostics, add an
absolute approximate-versus-exact evidence gate, validate every retained
mass coordinate, and audit gradients at several predeclared states. A
development run may not start until those corrections pass focused review.

The corrected development-only harness now implements those requirements,
source-pins the bank-size ladder and seeds, removes protected held-out
numerical definitions, and publishes one atomic result per case. Its focused
local validation passed 32 tests plus Ruff, formatting, and Pyright, and its
independent final review found no development-launch blocker. A
smallest-bank \(S=64\) diagnostic pilot passed all declared per-bank gates in
four of nine cases (the three near-Gaussian cases and skewed two-cell root);
the complete ladder and confirmation seeds remain the authoritative
development test.

The first BP1 development collection at `0769829d0d7c4aef9ef2250e8f34ca8367854fd9`
completed all nine tasks but is an infrastructure/provenance hard stop:
compute nodes lacked `git` on `PATH`, so their otherwise complete JSON
artifacts recorded a null source revision. Preserve it without certifying it:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/0769829d0d7c4aef9ef2250e8f34ca8367854fd9/c1
```

The replacement interface accepts an explicit full source revision,
cross-checks it against Git when Git is available, and requires it when Git
is absent. The authoritative PCG64 C1 development screen completed at
`6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e`:

```text
branch: codex/rjmcmc-aggregation-conditional-likelihood
candidate revision: 6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e/c1
report: /group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e/c1/report/RESULTS.md
```

All nine Slurm tasks completed with the pinned source identity. Four passed:
the three near-Gaussian cases and the skewed four-cell row case. Five failed
the unchanged development gates:

- skewed four-cell root locked at \(S=1024\), but two confirmation scrambles
  failed the gradient gate;
- skewed two-cell root passed the development ladder at \(S=64\), but all
  three confirmation scrambles failed, including evidence, posterior
  summaries, or gradients;
- both boundary-heavy root cases failed to obtain a stable two-size passing
  suffix; and
- boundary-heavy four-cell row passed only at \(S=16384\), so it also lacked
  the required stable suffix.

This is a scientific development hard stop for the finite PCG64 bank, not an
implementation or provenance failure. It does not license structural
reweighting or C2/PARIS promotion. The next bounded experiment replaces only
the share-bank construction with one jointly scrambled Sobol net, retaining
the same cases, sizes, seeds, exact oracles, and scientific gates. If that
also fails, proceed to the predeclared normalized residual-image MDN/NLE
fallback rather than increasing the PCG64 ladder post hoc.

BP1 jobs must use Slurm account `chem007981`; the default account produced
`PartitionConfig` cancellations even in the `test` partition.

The bounded RQMC successor is implemented at
`e0b2166597b3baa360233eb3ff63ee325a30c263`. It replaces independent PCG64
draws with a stable-cell-ID, count-balanced Dirichlet tree driven by scrambled
Sobol coordinates. Each internal tree node uses an inverse-Beta split with
the exact summed concentrations of its children; the right child is formed
by subtraction from its represented parent mass. This avoids the severe
small-shape underflow that a normalized inverse-Gamma construction would
encounter for PARIS-scale per-cell concentrations.

The construction uses one joint Sobol net within each canonical node block.
Catalogues above SciPy's 21,201-dimension limit use independently scrambled
blocks, so their combined discrepancy must be assessed empirically. Artifact
schema v2 records the SciPy version, engine settings, canonical catalogue,
block dimensions, inverse transform, and seed derivation. Legacy PCG64
artifacts retain their exact v1 payload and numerical construction.

Local launch gates passed:

- 71 focused PCG64/core/RQMC-driver/certifier tests;
- exact analytic Dirichlet means and covariances for a four-cell depth-two
  root and a two-region product;
- near-zero cross-region allocation covariance;
- exact nested Sobol prefixes;
- forced multi-block replay, relabeling, and native-cell permutation checks;
- endpoint-heavy small-concentration checks over four scrambles;
- shell syntax and ShellCheck for both HPC harnesses;
- Ruff, formatting, Pyright, and `git diff --check`; and
- independent scientific and implementation review with no C1 blocker.

The frozen RQMC development protocol is
`dcb2ef2bebb0c7eefafbd49a225c864e1b8a7478c568c168ed1640dd91ea9f4b`.
It requires SciPy 1.15.2 and validates the complete matrix protocol, source
revision, and construction environment before scientific evaluation,
including for one-case Slurm invocations.

The first BP1 setup at `e973aab0773f432095c52335efd724513e08d16f`
stopped before preflight because the harness required an empty Git status
after creating its required untracked `.pixi` symlink. No Slurm task or
scientific evaluation ran. The corrected source-identity rule permits exactly
that authenticated link (or an environment where it is ignored) and rejects
every other tracked or untracked change; the link target is checked
separately at every execution boundary. Preserve the setup-only run root as
failed operational evidence and use a fresh full-SHA run root.

The next preflight-only attempt at
`ba9cead6fcf7002dc90e6c49ba104501dc3a44aa` passed source/environment
identity but stopped during pytest collection: the shared pixi environment's
editable import still pointed at the canonical checkout rather than the
detached candidate. No smoke evaluation or Slurm task ran. Both harnesses now
prepend the authenticated detached worktree to `PYTHONPATH`; preserve this
second preflight failure and again use a fresh full-SHA root.

At `f9c90ca4cb62cfa6b17c9078e8809f0f0f53a13d`, detached imports were
correct and 71 of 72 focused tests passed. The remaining legacy-PCG64 golden
test reproduced its direct construction bitwise but showed that NumPy 2.2.6
on BP1's x86 build emits the other already authenticated legacy stream
previously observed under NumPy 1.26. The corrected test retains exact
platform-local algorithm parity and requires the resulting artifact/factor
pair to be one of the two authenticated byte streams, without assuming that
NumPy's version string alone determines its Gamma/Dirichlet implementation.
No smoke evaluation or Slurm task ran; preserve this third preflight failure
and launch from a fresh full SHA.

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
- the heterogeneous operator and four-cell column tiling remain sealed in a
  separate later executable; the development executable stores only their
  opaque catalogue identity and digest;
- preserve the complete A1 quadrature grid and its normalized prior weights
  for evidence and posterior integration;
- use a separately declared deterministic pointwise validation view for
  conditional-likelihood errors, without treating either view as a separate
  evidence integral;
- equal-footprint and fine-partition exact controls may not select bank size;
  and
- independent repeat-bank seeds may confirm the locked choice but may not
  retune it.

The pointwise validation view is new in C1. A1 and T2 did not split their
retained-mass quadrature grids; T2 blindness instead used disjoint native
simulation draw-ID ranges. Do not claim that the C1 pointwise split was copied
from A1 or T2.

For each applicable family and regime:

1. freeze pseudo-observations, quadrature orders, and a full-rank summary
   transform;
2. with one development seed, build the prefix sequence
   \(S=64,256,1024,4096,16384\), truncated only by a declared resource gate;
3. compare exact and approximate conditional log likelihoods and gradients
   over the predeclared pointwise validation view, using normalized
   quadrature-prior and exact-posterior weights;
4. compare posterior means, variances, and central intervals for total mass
   and every active retained region mass/share coordinate;
5. integrate the retained coordinates and compare evidence with the exact
   native model; and
6. lock the smallest bank size for which it and every larger attempted size
   pass, requiring at least two consecutive passing sizes in a development
   ladder, then repeat only that locked size with at least three independent
   confirmation seeds.

The selected bank size must satisfy:

- prior-weighted median absolute conditional log-likelihood error on the
  pointwise validation view at most 0.05 nat;
- exact-posterior-weighted 99th-percentile absolute error on that view at
  most 0.2 nat;
- scaled gradient error at most 0.05;
- absolute approximate-versus-exact log-evidence error at most 0.05 nat;
- posterior means within 0.05 exact posterior SD;
- posterior SDs within 2%;
- central 95% interval endpoints within 0.05 exact posterior SD; and
- between-bank evidence range at most 0.05 nat.

Report unweighted full-grid median, p99, and maximum errors as tail
diagnostics, but do not substitute them for the weighted gates. Report the
preserved A1 cross-tiling exact-evidence range, approximate-evidence range,
and structural-weight total variation when enough tilings are present. These
remain approximation-leakage diagnostics and must not update structural
weights.

These are approximation gates, not proof that structural weights may be
updated. Report error versus \(S\); a non-decreasing or unstable sequence is
a hard stop requiring variance reduction, a better summary factorization, or
a learned conditional density.

Publish aggregated JSON metrics only. Store any required statewise arrays in
small binary artifacts with explicit hashes; do not expand large grids or
banks into report JSON. Run a one-case timing smoke before the nine-case
development array (three development operators by two-cell root, four-cell
root, and four-cell row). The boundary-heavy four-cell fine grid can contain
millions of states, so use direct oracle controls or streamed chunks rather
than retaining redundant dense copies.

Run the nine development cells as independent Slurm tasks with one atomic
per-case result apiece. Merge and certify them only after all expected case
identities are present. The current nine diagonal alpha/operator cases are a
deliberate C1 simplification; T2 used the fuller alpha-by-operator crossing.
The development CLI must use the source-pinned sample-count ladder and seeds;
command-line overrides are noncompliant and fail closed. Changing either
requires a new reviewed protocol revision and source SHA.

This development phase cannot complete full C1 by itself. Common-native
projection posterior reconstruction is deferred because the observation
bank does not retain aligned auxiliary projection factors. Predictive checks,
the full control frontier, and the separately sealed held-out
operator/partition confirmation remain promotion gates. The independent
cross-tiling evidence/tower and structural-TV merger is also pending; emitted
per-case evidence values are inputs, not a certificate.

### C1-RQMC BP1 launch

Use the committed array launcher:

```text
docs/plans/rjmcmc_conditional_allocation_assets/run_rqmc_c1_array.sbatch
```

From the canonical BP1 checkout:

```bash
set -euo pipefail
git fetch origin codex/rjmcmc-aggregation-conditional-likelihood
export RQMC_REVISION="$(git rev-parse origin/codex/rjmcmc-aggregation-conditional-likelihood)"
export RQMC_SOURCE="/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc-conditional-allocation-${RQMC_REVISION}"
export RQMC_RUN_ROOT="/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/${RQMC_REVISION}/c1-rqmc"
if [[ -e "${RQMC_SOURCE}" || -e "${RQMC_RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing source or run directory." >&2
  exit 2
fi
git worktree add --detach "${RQMC_SOURCE}" "${RQMC_REVISION}"
ln -s /group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi "${RQMC_SOURCE}/.pixi"
mkdir -p "${RQMC_RUN_ROOT}/cases" "${RQMC_RUN_ROOT}/logs" "${RQMC_RUN_ROOT}/preflight"
test "$(git -C "${RQMC_SOURCE}" rev-parse HEAD)" = "${RQMC_REVISION}"
source_status="$(git -C "${RQMC_SOURCE}" status --porcelain)"
test -z "${source_status}" || test "${source_status}" = "?? .pixi"
test "$(readlink -f "${RQMC_SOURCE}/.pixi")" = \
  "/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
```

The setup deliberately rejects an existing worktree or run root. Never delete
or replace one merely to rerun a failed gate; use the preserved failure and a
new source revision. Run the committed preflight harness, which preserves
source/environment identity, focused test/static-check output, and a bounded
smoke artifact:

```bash
cd "${RQMC_SOURCE}"
bash docs/plans/rjmcmc_conditional_allocation_assets/run_rqmc_c1_preflight.sh
```

Submit the nine immutable development cases with the required account:

```bash
sbatch \
  --account=chem007981 \
  --output="${RQMC_RUN_ROOT}/logs/%A_%a.out" \
  --export=ALL,RQMC_SOURCE,RQMC_RUN_ROOT,RQMC_REVISION \
  docs/plans/rjmcmc_conditional_allocation_assets/run_rqmc_c1_array.sbatch
```

After all tasks finish, require exactly nine nonempty canonical JSON files,
one for each frozen case ID. Require identical source revision, driver SHA,
A1 definition SHA, frozen protocol SHA, SciPy version, Sobol engine settings,
and seed ladder in every file. Preserve per-case construction catalogue and
block identities. Summarize each case's locked \(S\), development pass suffix,
confirmation-seed checks, between-scramble evidence range, and final
`scientific_pass`. Do not reinterpret a failed case or increase the ladder.

Run the committed certifier only after Slurm reports all nine tasks completed:

```bash
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_allocation_likelihood_rqmc_certify.py \
  --source-dir "${RQMC_SOURCE}" \
  --cases-dir "${RQMC_RUN_ROOT}/cases" \
  --preflight-dir "${RQMC_RUN_ROOT}/preflight" \
  --output-dir "${RQMC_RUN_ROOT}/report" \
  --expected-source-revision "${RQMC_REVISION}"
```

The certifier must atomically publish `summary.json`, `RESULTS.md`,
`sha256sums.txt`, and `COMPLETE.json`. `COMPLETE.json` certifies a complete,
internally consistent execution; its decision is either `pass` or
`hard_stop`, so its existence does not imply scientific passage.

Promotion rule:

- all nine cases passing permits work on the still-missing common-native
  projection/tower merger and held-out C1 gates;
- any case failing is an RQMC development hard stop and starts C4's
  normalized residual-image MDN/NLE prototype;
- neither outcome licenses data-dependent weights for \(P\) or \(K\).

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

For one fixed partition, construct an authenticated orthonormal basis \(Q_P\)
for the exact error-whitened aggregation-residual image. Do not train a
noise-free density in the full observation space: it is generally singular
there. With \(T=\sum_j A_j\) and retained mass shares \(w_j=A_j/T\), learn

\[
\xi=T^{-1}Q_P^\mathsf TD^{-1/2}H\{X-g_P(A)\}
\quad\text{conditional on }w.
\]

The alpha field, partition, observation operator, error model, residual-image
basis, and renderer are fixed artifact context. The realized observation is
not a training or selection input. Root partitions require no conditioning
input; two-region partitions require one logit-share input.

Start with a small full-covariance conditional MDN before MAF or NSF because
an MDN is normalized, auditable, and can be exported to native PyTensor
primitives. Its observation likelihood analytically convolves measurement
noise, so component covariance in residual coordinates is
\(I+T^2\Sigma_\ell(w)\), with the exact orthogonal Gaussian factor retained.
This preserves normalization and avoids asking the learner to reproduce
known noise.

Use whole conditional native-allocation draw IDs for train, validation,
simulator-test, and held-out splits. Predeclare a training-size ladder and
lock the smallest size passing all development cases before independent-seed
confirmation. Begin with eight mixture components and two 32-unit
float64 `tanh` layers. Permit one predeclared escalation to sixteen
components and two 64-unit layers only when training is stable but the smaller
model underfits; failure of both is a hard stop before a flow.

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
