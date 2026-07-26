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

Preflight passed completely at
`e36e4f7d796a34e97b3f6ab4c95f924105659ccf` (72 tests, Ruff, Pyright,
and smoke). Slurm array `18186074` then failed all nine tasks before
scientific evaluation because Git is not on the compute-node default
`PATH`; the source-identity guard consequently failed closed. The corrected
launcher loads BP1's `git/2.45.1-pqk5` module before any provenance query.
Preserve the complete failed array logs and use a fresh full-SHA run root.

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

### Certified C1-RQMC outcome

The source-pinned BP1 run at
`863353443488b7e08900a147f92039d444676d41`, Slurm array `18186093`,
completed all nine cases. The committed certifier published a complete
`hard_stop` decision:

- eight of nine cases passed;
- all near-Gaussian and skewed cases passed;
- boundary-heavy two-cell root and four-cell row passed;
- boundary-heavy four-cell root did not establish the required two-size
  passing suffix;
- its \(S=16{,}384\) evaluation passed, but \(S=1{,}024\) and \(4{,}096\)
  still failed only the posterior-SD relative-error gate, so the isolated
  final pass cannot select a locked bank size;
- all nine tasks completed in 4--11 seconds with peak task RSS below 171 MiB.

The certified report is
`/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/863353443488b7e08900a147f92039d444676d41/c1-rqmc/report/RESULTS.md`.
Its manifest SHA-256 is
`a65957426e5450e1079871acc6bad58f20798a7c3bb4ac4e84e5aa2a62b716b0`.
The result is scientifically much better than the PCG64 bank's four of nine
passes, but the predeclared rule prohibits extending the Sobol ladder after
seeing the failure. Proceed to C4's learned normalized conditional density.
This hard stop does not license structural evidence weights.

## C2: moderate and PARIS feasibility

This finite-bank stage is withheld because C1-RQMC hard-stopped. Do not run
this matrix by selectively promoting the eight passing tiny cases. The
corresponding learned-density rank and resource feasibility test is C4e.

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

This section remains the integration specification for a finite bank, but it
is withheld by the C1-RQMC hard stop. For an accepted learned artifact, follow
C4d instead. Proceed here only if a future predeclared finite-bank method
passes C1 and C2 identifies an affordable bank/rank pair.

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

## C4: learned residual-image density

The certified C1-RQMC result activates this bounded fallback. The learned
density is still an approximation to a conditional likelihood for one fixed
partition; it is not a learned posterior over partitions and must not supply
data-dependent weights for \(P\) or \(K\).

The initial design was checked against sibling `inversions-knowledge` revision
`e77d20cffe7ee0298d9106065c962d24198dabdc`, especially
`docs/research-questions/learning-non-gaussian-marginal-models.md`,
`docs/derivations/non-gaussian-aggregation-error-by-marginalization.md`, and
`docs/source-notes/sbi-vs-pymc.md`. The last file is a reviewed triaged-chat
note rather than an executable software certificate; its PyMC and `sbi` API
claims remain version-sensitive until focused tests pin them.

The scientific identity is stronger than equality of a few projected
moments. Let one proper native prior and observation model be fixed, let
\(p_P(A)\) be the exact pushforward prior for the retained masses under
partition \(P\), and let \(L_P(y\mid A)\) exactly integrate the discarded
within-region allocations. Then

\[
\int L_P(y\mid A)\,p_P(A)\,dA=p(y)
\]

for every representation \(P\). Exact marginalization is therefore
projection-invariant, and externally randomized \(P\) remains independent of
the data. Any variation in an approximate evidence across partitions or
values of \(K\) measures approximation leakage. It may be reported as a
failure diagnostic, but it may not be interpreted as evidence for a basis,
softmaxed into structural weights, or used to update the declared
structural prior.

### C4a: deterministic residual-image context and portable evaluator

For one fixed partition, construct and persist an authenticated orthonormal
basis \(Q_P\) for the exact error-whitened aggregation-residual image. Do not
fit a noise-free density in the full observation space: it is generally
singular there. With \(T=\sum_j A_j\) and retained mass shares
\(w_j=A_j/T\), define

\[
\xi=T^{-1}Q_P^\mathsf TD^{-1/2}H\{X-g_P(A)\}
\quad\text{conditional on }w.
\]

The alpha field, stable cell identities, labelled partition, observation
operator, diagonal error model, residual-image basis, rank tolerance, and
conditional-mean renderer are immutable artifact context. The realized
observation is not a basis, training, model-selection, or hyperparameter
input. Root partitions have a zero-dimensional conditioner; the four-cell
row case has one logit-share conditioner.

Implement the first evaluator in float64 NumPy with no Torch or `sbi`
dependency. A learned mixture supplies weights, residual-coordinate means,
and positive-definite covariances. After analytic convolution with known
measurement error, component covariance is
\(I+T^2\Sigma_\ell(w)\). Retain both the exact Gaussian factor orthogonal to
\(Q_P\) and the \(-\sum_i\log \sigma_i\) observation-noise Jacobian. Test
rank-zero, full-rank, root, row, and singleton-region cases; dense-Gaussian
parity; normalization; region and cell permutation invariance; serialization
round trips; context-hash rejection; and malformed covariance artifacts.

Publish no C4 HPC launch command until this implementation, its focused
tests, and an independent review have landed at a pushed full SHA.

### C4b: root-only zero-input GMM/MDN tiny baseline

First address the sole RQMC failure: the boundary-heavy four-cell root.
Because a root has no conditioning coordinate, fit a deterministic,
zero-input, eight-component full-covariance Gaussian mixture in residual
coordinates. This is the constant-network special case of an MDN and can be
trained by audited float64 NumPy EM; it deliberately introduces no Torch or
`sbi` dependency.

Reuse the frozen C1 exact quadrature cases, mass grid, posterior summaries,
gradient locations, and scientific thresholds. Split by whole native
allocation draw ID so no transforms of one simulator draw cross training,
validation, simulator-test, or untouched confirmation sets. The frozen
development protocol, subject to an implementation certificate before
launch, is:

- nested training sizes
  \(N=4{,}096,16{,}384,65{,}536,262{,}144\);
- 65,536 validation draws, 131,072 simulator-test draws, and a separately
  protected 131,072-draw density holdout;
- development seed 731 and independent confirmation seeds 1877, 4099, and
  8317, all under a new learned-density domain separator; and
- selection of the smallest common training size with a two-size passing
  suffix. A pass only at \(N=262{,}144\) is a hard stop rather than a locked
  size.

At each training size, run exactly three deterministic EM starts. Use stable
`logsumexp` responsibilities and Cholesky-based full-covariance calculations,
regularize every component covariance by \(10^{-8}I\) (equivalently a
\(10^{-4}\) standard-deviation floor), and fail rather than silently repair an
empty component. Cap each fit at 2,000 iterations; convergence requires an
objective change below \(10^{-7}\) nat per draw for 10 consecutive
iterations. At least two of the three starts must converge. Select among valid
starts by validation negative log likelihood, never by a protected scientific
gate. Validation-to-simulator-test generalization additionally requires

\[
\left|\operatorname{NLL}_{\rm test}
      -\operatorname{NLL}_{\rm validation}\right|
\le
\max\left(0.02q,\;5\,\operatorname{MCSE}_{\rm combined}\right),
\]

where \(q\) is residual-image rank and the combined MCSE is the standard error
of the difference between the two independent mean NLL estimates.

Also freeze and source-pin before launch:

- the whole-draw split hash and independent simulator seeds;
- deterministic initialization/restart identities;
- covariance floors and all convergence tolerances;
- the exact quadrature and density-test catalogues; and
- the independent-seed confirmation rule.

Do not invent or extend the ladder after observing a failure. The earlier
eight-component, full-covariance model is the only first-stage architecture.
One predeclared escalation to sixteen components is permitted only if EM is
numerically valid and the eight-component fit demonstrably underfits. If
either stage is unstable, non-normalized, or fails the unchanged value,
gradient, evidence, and posterior-summary gates, stop before adding a flow.

Store the fitted arrays, preprocessing, float dtype, split identities, seeds,
versions, training history, context digest, and whole-artifact digest in an
auditable state-dictionary-style payload rather than an opaque pickle.

The first reviewed implementation checkpoint is
`5167e36fedd2b0a93b2ba9bfd77534aefffcf485`. It supplies the portable
root-GMM trainer, authenticated fitted-bundle envelopes, the protected
catalogue commitment, a smoke profile, and focused tests. It is a development
checkpoint, not an HPC certificate. The authoritative candidate is the later
clean pushed full SHA containing the phase merger, protected certifier, and
HPC assets.

Freeze NumPy 2.2.6 and SciPy 1.15.2 for development, confirmation, and
protected certification. The committed Pixi lock contains these versions;
the executable also rejects a different runtime.

The protected object is a density holdout, not a catalogue of new operators.
It contains a concealed master seed and frozen metadata from which the
certifier derives 131,072 residual draws for each of the same six exact
contexts. The precommitted raw catalogue digest is
`83bec3945ebc90d5e25d0888b440fe56f761f9059cf01537fbb2227b81510b66`.
Only the seed-731 artifact at the common locked training size is promoted.
The protected stage evaluates that artifact without retraining. It applies
the frozen validation-versus-protected NLL rule above, reauthenticates the
unchanged likelihood, gradient, evidence, posterior-summary, normalization,
and four-bank evidence gates, and cannot alter the architecture, training
size, preprocessing, or seed. A protected failure is terminal for this model;
it may not trigger retuning. Even a protected pass keeps
`structural_inference_licensed=false`.

### C4b BP1 phase protocol

Use only immutable shard outputs and pure validating mergers:

The mergers regenerate the exact cases and simulator banks, authenticate the
portable artifact and training-prefix identities, and replay every downstream
likelihood, gradient, evidence, posterior, and generalization gate. They do
not rerun EM fitting. The deterministic fit transcript therefore remains
part of the trusted clean-source, immutable-shard provenance boundary rather
than an independently recomputed optimization certificate.

1. **G0 preflight.** Check out a clean detached full SHA, use the frozen Pixi
   environment, run the portable-evaluator, GMM, merger, and protected
   certifier focused tests, Ruff, Pyright, and the bounded smoke profile.
2. **G1 development shards.** Run 24 Slurm tasks: six cases by four training
   sizes. Each task uses development seed 731, constructs the source-pinned
   largest Sobol bank so prefixes retain the declared nested identity, fits
   exactly one requested prefix, and atomically publishes one canonical JSON
   artifact.
3. **G1 lock merger.** Require exactly the 24 declared files, reject symlinks
   and extras, reauthenticate every context, split, fitted envelope, runtime,
   and scientific gate, recompute the six-case pass pattern, and publish the
   smallest common two-size passing-suffix lock. Stop if no such lock exists;
   a pass only at 262,144 is a hard stop.
4. **G2 confirmation shards.** Bind every job to both the raw lock-file digest
   and its internal payload digest. Run 18 Slurm tasks: six cases by seeds
   1877, 4099, and 8317, all at the one locked size. Each task publishes one
   immutable JSON result.
5. **G2 development certifier.** Require exactly the 18 confirmation files,
   combine each with its nominated seed-731 development artifact, reapply
   every individual gate, and require the four-bank log-evidence range to be
   at most 0.05 nat. Publish
   `development_pass=true`,
   `eligible_for_protected_holdout=true`,
   `protected_holdout_pass=null`, and `scientific_pass=false`.
   Publish the certificate's raw SHA-256 in a separate immutable record and
   bind that digest into the G2 completion marker; G3 must authenticate
   against this G2 record rather than hashing an unpinned current file.
6. **G3 protected certification.** Only after G2 passes, transfer the sealed
   catalogue to an independent certification path. Authenticate the frozen
   runtime, live source, passing development certificate, and every nominated
   development shard before touching the catalogue. Then verify the
   catalogue's raw digest before JSON parsing or numerical access, open it
   once, evaluate all six promoted artifacts, and publish the final manifest
   and completion marker last. No development process may read the concealed
   master seed or derive the protected draws.

Failures and interrupted shards are evidence, not warnings. Preserve them and
rerun only the missing immutable shard under the same full SHA. Do not run a
monolithic six-case screen as the authoritative BP1 protocol.

### C4b BP1 operator instructions

Use these committed assets:

```text
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_preflight.sh
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_development_array.sbatch
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_merge_development.sh
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_confirmation_array.sbatch
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_certify_confirmation.sh
docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_protected_certify.sh
```

On BP1, fetch the branch, resolve one full candidate SHA, create a fresh
detached worktree, and attach the canonical frozen environment:

```bash
repository=/group/chem/acrg/brendan_for_codex/openghg_inversions
git -C "${repository}" fetch origin codex/rjmcmc-aggregation-conditional-likelihood
revision="$(git -C "${repository}" rev-parse origin/codex/rjmcmc-aggregation-conditional-likelihood)"
short_revision="${revision:0:12}"
source_root="/group/chem/acrg/brendan_for_codex/rjmcmc_gmm_worker_${short_revision}"
run_root="/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/${revision}"
git -C "${repository}" worktree add --detach "${source_root}" "${revision}"
ln -s /group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi "${source_root}/.pixi"
mkdir -p \
  "${run_root}/preflight" \
  "${run_root}/development" \
  "${run_root}/confirmation" \
  "${run_root}/lock" \
  "${run_root}/certificate" \
  "${run_root}/protected" \
  "${run_root}/markers/development" \
  "${run_root}/markers/confirmation" \
  "${run_root}/logs/development" \
  "${run_root}/logs/confirmation"
export GMM_SOURCE="${source_root}"
export GMM_RUN_ROOT="${run_root}"
export GMM_REVISION="${revision}"
```

Run G0 and stop unless the completion marker is present:

```bash
bash "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_preflight.sh"
```

Submit G1 and record the returned job ID:

```bash
development_job="$(
  sbatch --parsable --export=ALL \
    "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_development_array.sbatch"
)"
```

Monitor with `squeue`, `sacct`, and the run-root logs. Do not run the merger
until all 24 JSON artifacts and all 24 completion markers exist and every
array task completed successfully. Then:

```bash
bash "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_merge_development.sh"
```

The merger is a hard gate. If it publishes no common lock, preserve the 24
shards and stop. If it passes, submit G2:

```bash
confirmation_job="$(
  sbatch --parsable --export=ALL \
    "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_confirmation_array.sbatch"
)"
```

Require all 18 JSON artifacts, 18 completion markers, and successful Slurm
states before running:

```bash
bash "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_certify_confirmation.sh"
```

The certifier publishes
`certificate/development-certificate.raw.sha256` and binds it into the G2
completion marker. Inspect the canonical development certificate. G3 is
forbidden unless its decision is `pass` and
`eligible_for_protected_holdout` is true. Transfer the sealed catalogue
through an independent path only then. Do not display, source, or commit it.
Set its path and run:

```bash
export GMM_PROTECTED_CATALOGUE=/independent/secure/path/rjmcmc_c4b_protected_density_holdout_v1.json
bash "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_protected_certify.sh"
```

The protected script remains disabled when that variable is absent. The
Python certifier first authenticates the frozen runtime, source, development
certificate, and nominated shards; only an eligible candidate can touch the
catalogue. It then authenticates the precommitted catalogue digest before
parsing it and publishes either a pass or terminal scientific hard stop.
Archive the entire run root and a verified SHA-256 inventory; write nothing
to `PARIS_inversions`.

### C4c: conditional Torch MDN for the row case

Only if a conditional case needs a learner after the root baseline passes,
add an optional Torch training environment. Begin with eight mixture
components and two 32-unit float64 `tanh` layers; permit one predeclared
escalation to sixteen components and two 64-unit layers under the same
underfit-only rule. Use an additive-log-ratio/logit representation of
strictly positive shares and record its ordering and preprocessing in the
artifact.

Training remains outside the runtime likelihood. Export plain arrays into
the independently tested NumPy evaluator. Use whole-native-draw splits for
ordinary generalization and separate whole-partition and whole-operator
contexts as held-out transfer tests. A held-out partition or operator is a
new sealed context trained with the already frozen recipe; it must never
cause retrospective tuning of the development recipe.

`sbi` remains an optional training/comparison dependency, not a required
runtime dependency. Its PyMC MCMC backend samples an `sbi`-owned posterior
potential; it does not insert an arbitrary Torch likelihood into the existing
PyMC graph and therefore is not a PyMC likelihood bridge.

### C4d: PyTensor/PyMC export

Proceed only after the independent NumPy evaluator passes the tiny exact
oracles and confirmation seeds. Re-express the accepted fixed artifact in
native PyTensor primitives, then require float64 value and gradient parity at
randomized interior states, including both residual and conditioning paths.
Integrate it into a fixed-basis PyMC model as one dependent observation block
with a scalar joint-likelihood diagnostic. Do not fabricate a pointwise
`log_likelihood.observed`.

A custom Torch-autograd PyTensor operation is a last resort and requires the
same value/gradient and serialization gates. The PyMC-to-NumPyro/JAX path
cannot consume a generic Torch operation without a separately implemented
and tested JAX lowering.

### C4e: PARIS rank and resource gate

Before training on PARIS, measure the numerical rank and spectrum of each
authenticated residual-image context, together with build time, artifact
size, evaluation cost, and peak RSS. The dense full-covariance mixture in
C4b is a tiny-oracle baseline only. At the likely full rank
\(q=1382\), eight dense components require millions of covariance outputs
and repeated cubic factorizations; a conventional dense MDN is not an
acceptable production design.

Predeclare retained-rank error and resource budgets. If the image is not
small enough under those budgets, stop dense deployment and test a normalized
factor-analyzer mixture or a frozen truncated residual-image model with an
explicit complement. Truncation must be validated against streamed
simulation, and any cross-partition evidence drift remains leakage rather
than structural information. A universal topology encoder, a learned
partition posterior, and a flow are outside this phase unless these bounded
models fail for a documented scientific reason.

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
