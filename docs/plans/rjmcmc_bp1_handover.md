# RJMCMC BP1 handover

## Purpose

This is the operational entry point for continuing the experimental RJMCMC
work directly on BP1. The detailed mathematical and scientific contracts
remain in:

- [`rjmcmc_topology_conditioned_hmc_next_phase.md`](rjmcmc_topology_conditioned_hmc_next_phase.md);
- [`rjmcmc_topology_conditioned_hmc_hpc_test_plan.md`](rjmcmc_topology_conditioned_hmc_hpc_test_plan.md);
- [`rjmcmc_aggregation_error_low_rank_hpc_test_plan.md`](rjmcmc_aggregation_error_low_rank_hpc_test_plan.md);
- [`rjmcmc_full_tiling_compound_hmc.md`](rjmcmc_full_tiling_compound_hmc.md); and
- [`rjmcmc_partition_mixing_and_full_tiling_design.md`](rjmcmc_partition_mixing_and_full_tiling_design.md).

The current branch is `codex/rjmcmc-topology-conditioned-hmc`. Always use the
full current `origin` SHA at launch; do not infer a candidate from a short run
directory name.

## State at handover

The important source checkpoints are:

| Full revision | Meaning | Disposition |
|---|---|---|
| `7a1a1cc673a4b6a6ce0ed7b5123494ebd205b467` | topology-neutral total/contrast HMC metric | H2c hard stop at both \(K=50\) and \(K=250\) |
| `7f7b1509bf032d04c9839ec9fa4d7be69b03e1ab` | first topology-conditioned HMC implementation | failed D0 finite-binary64 structural reversibility audit; never resume |
| `e6199150e680d43e6e3c1388db45773c5337802a` | exact log-coordinate edge-flip and resolution-relocation involutions | focused local validation passed; BP1 validation incomplete |
| `16819f55cea5c6054b0113b751aa9833afa4fa9b` | normalized low-rank aggregation-error baseline | focused local validation passed; A1--A5 not run |
| `54045edf67c4703da5909b1fdd2a6081d0a61251` | cached fixed-partition aggregation factors and FullTiling bridge | focused local validation passed; benchmark on BP1 |

Later documentation commits are also part of the candidate. The authoritative
candidate is therefore:

```bash
git fetch origin
export CANDIDATE_REVISION="$(git rev-parse origin/codex/rjmcmc-topology-conditioned-hmc)"
git show --no-patch --format=fuller "${CANDIDATE_REVISION}"
```

The `7f7b150` D0 audit found 4 forward-valid moves among 10,004 at \(K=50\)
whose physical reverse fraction rounded to a binary64 endpoint. Skipping those
paths would bias the audit. The replacement in `e619915` does not draw or
preserve a Beta fraction: it permutes authoritative log-mass coordinate bit
patterns, has unit Jacobian, and scores the transformed target plus discrete
catalogue probabilities.

The repaired compound schedule identity is
`full_tiling_1_exact_log_mass_involution_1_topology_conditioned_pymc_hmc_v5`.
The durable checkpoint schema remains v3, but the changed runtime/schedule
identity makes earlier checkpoint and calibration artifacts fail closed.

The attempted BP1 validation of `e619915` was interrupted when VPN access
ended. It supplies neither a pass nor a failure. Preserve and label as
incomplete:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_h2d_worker_e619915
/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/e619915
/group/chem/acrg/brendan_for_codex/openghg_inversions/.agent-tracker/workers/rjmcmc-h2d-research-bp1-openghg-inversions-b8bb381cef3e/report.md
```

Preserve the failed `7f7b150` evidence:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/7f7b150
```

Do not publish a completion marker for an interrupted or hard-stopped stage.

## Scientific state

The fixed-basis controls established that continuous sampling was a major
bottleneck:

- local fixed-basis sampling gave root bulk ESS near 31 at both \(K\) values;
- diagonal NumPyro NUTS gave root bulk ESS 1,684 at \(K=50\) and 2,071 at
  \(K=250\); and
- dense NumPyro NUTS gave root bulk ESS 6,114 with zero divergences at
  \(K=50\).

Mobile topology chains nevertheless retained persistent likelihood
start-separation. H2/H2b/H2c then showed that one topology-neutral static HMC
metric did not generalize across tilings. H2d tests a deterministic
topology-conditioned Euclidean precision which changes between tilings but is
constant during each Hamiltonian trajectory. It is not RMHMC.

The aggregation-error track asks a different model question. If every
partition is only a representation of one common proper native model, exact
hidden-allocation marginalization gives the same evidence for every
partition. Data then cannot update \(P\) or \(K\); their posterior weights
equal their structural prior weights. Keep the two approximation uses
distinct: the Gaussian closure is an optional explicit approximation usable
inside RJ, whereas the NLE/GMM track is an alternative non-RJ marginal
likelihood. The latter's exact common-native-model limit is partition- and
\(K\)-invariant and licenses no structural learning; approximate evidence
drift is a leakage diagnostic, not a structural weight.

The current \(\kappa=2K\) allocation rule changes the native concentration
with \(K\). Cross-\(K\) representation-invariance tests must instead freeze one
native-cell alpha field and its total concentration. The summary basis must be
selected from frozen operator/prior information and must not use the observed
residual.

## Reproducible BP1 checkout

Work from a detached, clean, full-SHA worktree. Do not edit repository source
inside a run root.

```bash
cd /group/chem/acrg/brendan_for_codex/openghg_inversions
git fetch origin
export CANDIDATE_REVISION="$(git rev-parse origin/codex/rjmcmc-topology-conditioned-hmc)"
export CANDIDATE_WORKTREE="/group/chem/acrg/brendan_for_codex/rjmcmc_candidate_${CANDIDATE_REVISION}"
git worktree add --detach "${CANDIDATE_WORKTREE}" "${CANDIDATE_REVISION}"
cd "${CANDIDATE_WORKTREE}"
test "$(git rev-parse HEAD)" = "${CANDIDATE_REVISION}"
test -z "$(git status --porcelain)"
pixi install -e dev --frozen
```

Use `pixi run -e dev --frozen` for focused commands. Run only experimental
tests, focused Ruff, and focused Pyright; do not run the repository-wide tox
matrix.

Any source change or implementation requires a new reviewed and pushed commit,
a fresh detached worktree, and a new full-SHA run root. Launch, analysis, and
reporting scripts may be created beneath the immutable run root, but record
their content, command lines, environment, and hashes.

## Frozen PARIS identity

```bash
export FROZEN_INPUT=/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc
export FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
export FROZEN_INPUT_SHA=24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044
export OUTER_LABELS=intem_label_0,intem_label_1,intem_label_2,intem_label_3,intem_label_4,intem_label_5
export WEIGHT_POLICY=spherical-grid-cell-area-v1
sha256sum "${FROZEN_INPUT}"
```

Require the observed digest to equal `FROZEN_INPUT_SHA`. Verify every archived
comparison manifest before using it.

## Execution sequence

### Track H: topology-conditioned compound HMC

1. Create
   `/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/<full SHA>`.
2. Implement immutable run-root D0 launch/report scripts around the exact
   commands and gates in the H2d plan.
3. Run D0 only. The structural oracle must record at least 10,000 valid moves
   at each \(K\), include extreme log-mass contrasts, materialize every reverse
   path, and reproduce topology and log coordinates bitwise. Zero skipped
   reverses are permitted.
4. If and only if D0 passes, run D1.
5. If and only if D1 passes, run discarded D2 calibration and its untouched
   held-out topologies.
6. A held-out failure is an H2d hard stop. Do not add an unplanned grid.
7. Run D3 durability/smoke and D4 Slurm start-sensitivity chains only after D2
   passes.

If H2d fails, localize the failure among metric conditioning,
within-topology position curvature, and topology/continuous landing. A
source-HMC/structural-map/destination-HMC proposal is justified only after the
within-topology metric works. Position-dependent/Riemannian HMC is justified
only by strong within-topology curvature variation.

### Track A: aggregation error

This track is independent of an H2d calibration outcome:

1. Preserve the completed A0 implementation controls and the A1 Gaussian
   scientific-shape hard stop. Do not rerun them merely to erase a failure.
2. Preserve the transported-mixture T2 hard stop at `6ff3afe`; no held-out
   promotion was licensed.
3. Preserve the PCG64 conditional-allocation C1 development hard stop at
   `6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e`. Four of nine cases passed,
   but five failed the unchanged bank-convergence or confirmation gates.
   Continue its separate
   [HPC plan](rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md)
   with the bounded scrambled-Sobol balanced-Dirichlet successor implemented
   at `e0b2166597b3baa360233eb3ff63ee325a30c263`. Its certified BP1 screen at
   `863353443488b7e08900a147f92039d444676d41` passed eight of nine cases but
   hard-stopped on boundary-heavy four-cell root. Start the normalized
   residual-image density fallback; do not extend the RQMC ladder post hoc.
   The bounded order is:
   C4a deterministic residual-image context plus portable float64 NumPy
   evaluator; C4b root-only zero-input eight-component full-covariance
   GMM/MDN on the tiny exact cases without Torch or `sbi`; C4c a conditional
   Torch MDN for the row case only if needed; C4d native PyTensor/PyMC export
   only after tiny-oracle certification; and C4e a PARIS rank/resource probe
   before selecting a factor-analyzer or truncated-image production model.
   The dense full-covariance model is tiny-only and is not credible at
   \(q=1382\). The frozen root training ladder is
   \(4{,}096,16{,}384,65{,}536,262{,}144\) whole draws, with separate
   validation, simulator-test, and protected density-holdout sets. Lock the
   smallest size with a two-size passing suffix; passing only at the largest
   size is a hard stop. Exact seeds, split sizes, and confirmation rules are
   in the linked C4 plan and remain subject to an implementation certificate
   before launch.
4. Create and hash run-root analysis harnesses for the remaining
   moderate/PARIS conditional diagnostics. Reusable implementation belongs on
   the separate branch/worktree and must be reviewed, committed, and pushed
   before a new full-SHA run.
5. Benchmark the cached fixed-partition factor builder. Its storage is
   \(O(n_{\rm obs}K+Kq^2)\); public PSD validation is \(O(Kq^3)\). Start with
   \(q=32\) and \(q=64\).
6. Reuse the committed fixed-partition PyMC/NUTS Gaussian bridge only as a
   comparator. A conditional-allocation PyTensor target requires its own
   value/gradient parity gates and scalar joint-likelihood output.
7. Do not expose a diagonal `pm.Normal` component as if it were the corrected
   ArviZ `log_likelihood`; persist the scalar joint likelihood explicitly.
8. Keep partitions external to the approximate likelihood workflow. Combine
   common summaries using the declared structural prior weights; evidence
   differences are leakage diagnostics, not RJ or softmax weights.

Under the common proper native prior, exact conditional marginalization and
then integration over retained masses recovers the same \(p(y)\) for every
partition. Thus the data cannot update a representation-only \(P\) or \(K\).
The learned residual density is an approximation to that conditional
calculation, not a classifier of bases. Approximate evidence differences must
be treated as projection-invariance leakage and may never replace the
externally declared structural weights.

Keep this first integration downstream of `fixed_basis_nuts.py`, preferably in
`fixed_basis_aggregation_nuts.py`. Do not partially wire aggregation error
into the mobile HMC target. When creating that downstream module, move the
FullTiling borrowing bridge there or replace its `object.__new__` construction
with a private validated borrowed-array constructor.

The dense Gaussian closure failed exact scientific shape. The normalized
transported-mixture successor at `6ff3afe56416e701ac1fc4ae45676d08ea28229b`
then stopped at T2: three of eight development fits produced artifacts but
failed at least one scientific gate, and five exhausted their fixed EM
restarts. Preserve its authoritative run:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_aggregation_transported_mixture/6ff3afe56416e701ac1fc4ae45676d08ea28229b/t2/11a43a24da003019e600c990da143573f527ce1b85ffeaedada80ec857edbd28
```

The frozen conditional-allocation finite mixture and its RQMC successor are
documented in
[`rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md`](rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md).
It directly averages noise-convolved within-region Dirichlet allocations
conditional on the retained masses. The certified RQMC screen passed eight of
nine tiny cases and hard-stopped under its predeclared rule. The next bounded
stage therefore starts with a deterministic residual-image context and
portable NumPy likelihood evaluator, followed by a root-only zero-input
eight-component GMM/MDN. The first reviewed implementation checkpoint is
`5167e36fedd2b0a93b2ba9bfd77534aefffcf485`; it is not an HPC certificate.
The authoritative BP1 candidate is
`6ad8eee8c1d02289e1b130c4204b8bbe9c86135e`, containing the shard
merger, locked confirmation mode, protected certifier, and launch assets.
This is a candidate on
`codex/rjmcmc-aggregation-conditional-likelihood`. Its G0 preflight passed on
BP1 with 65 focused tests, Ruff, Pyright, and the bounded smoke screen. All
24 G1 development shards completed. The fresh run root is

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/6ad8eee8c1d02289e1b130c4204b8bbe9c86135e
```

The G1 merger hard-stopped before lock publication. Cross-node replay NLLs
differed by up to approximately \(1.1\times10^{-9}\). Applying the scoped
roundoff allowance then exposed, through the exact context hash, approximately
\(1.72\times10^{-16}\) basis variation in the same rank-3 residual subspace
from cross-node BLAS/LAPACK behavior. This is an implementation-portability
hard stop and supplies no scientific GMM result. G2 and G3 did not run, and
the protected catalogue was untouched.

The repaired implementation baseline is
`0803137ab96240342361f08afab38acdbaf3bb28`. It uses scale-aware replay
tolerances with fail-closed gate margins shared by development and protected
certification, plus context/basis schema v2 built from stable-cell-ordered
fixed-scalar means and modified Gram--Schmidt; SVD determines rank only, with
ambiguity gates. Resolve the pushed branch head after this plan-only status
update and rerun all of G0 and G1 under that fresh full-SHA run root. The old
24 shards are not reusable.

Do not add Torch or `sbi` to the runtime for this baseline. `sbi` remains an
optional training/comparison dependency and does not bridge an arbitrary
learned Torch likelihood into PyMC. This NLE/GMM baseline is a non-RJ
marginal-likelihood alternative: no outcome may update structural weights,
and partitions remain externally weighted by their declared prior.

The authoritative GMM protocol is restartable:

- 24 immutable development shards: six cases by four training sizes;
- one pure validating common-lock merger;
- 18 immutable confirmation shards: six cases by three frozen seeds, each
  bound to both raw and internal lock digests;
- one development certificate that retains `scientific_pass=false`; and
- one independent, one-shot protected density holdout after development
  passes.

The protected catalogue is not a set of new operators. It supplies a
concealed master seed and frozen metadata from which the certifier derives
131,072 residual draws for each of the same six exact contexts, and promotes
only the seed-731 artifact at the common locked size, without retraining or
retuning. Its
precommitted raw SHA-256 is
`83bec3945ebc90d5e25d0888b440fe56f761f9059cf01537fbb2227b81510b66`.
Keep the catalogue itself outside the repository and inaccessible to
development jobs. Follow stages G0--G3 in the detailed plan; stop at the first
hard gate.

The reviewed foundation at
`3e30f9117bcba03920aafd338f7eea529c25b079` passed C0 on BP1. Its
immutable run root is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/3e30f9117bcba03920aafd338f7eea529c25b079/c0
```

C1 exact scientific-shape and bank-convergence gates rejected the finite
PCG64 bank in five of nine development cases. The authoritative report is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_allocation_likelihood/6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e/c1/report/RESULTS.md
```

Do not promote C0 or the four passing cases as evidence that a finite PCG64
bank is generally accurate. For Slurm submissions use account `chem007981`;
the default account has produced `PartitionConfig` cancellations on this
cluster.

## `inversions-knowledge`

The local knowledge repository was clean at
`e77d20cffe7ee0298d9106065c962d24198dabdc` before publication. After the
private repository is pushed and cloned as a sibling of `openghg_inversions`,
record its remote URL and actual full SHA; do not assume this provisional SHA
remains current.

Read these sibling paths first:

```text
../inversions-knowledge/docs/research-trails/legacy-tdmcmc-to-experimental-rjmcmc.md
../inversions-knowledge/docs/research-questions/rjmcmc-hmc-nuts-and-transported-tuning.md
../inversions-knowledge/docs/research-questions/learning-non-gaussian-marginal-models.md
../inversions-knowledge/docs/research-questions/posterior-projection-conundrum.md
../inversions-knowledge/docs/source-notes/sbi-vs-pymc.md
../inversions-knowledge/docs/derivations/rjmcmc-dimension-matching-and-augmented-spaces.md
../inversions-knowledge/docs/derivations/non-gaussian-aggregation-error-by-marginalization.md
../inversions-knowledge/docs/derivations/posterior-projection-and-exact-marginalization.md
../inversions-knowledge/docs/workflows/validating-trans-dimensional-mcmc-kernels.md
```

## Resource, durability, and stop rules

- Check login-node load and available memory before bounded sequential work.
- Keep aggregate login-node RSS below 200 GB.
- Use Slurm for multi-chain, retained, or large matrix jobs.
- Record job IDs, source SHA, environment, run root, and the last successful
  gate.
- Preserve partial and failed artifacts. Write `complete.json` last and only
  after an independent checksum audit.
- Stop at the first hard gate; do not reinterpret a failure as a warning.
- Stop if BP1, the frozen input, or required filesystem paths become
  inaccessible.
- Write nothing from these experiments to `PARIS_inversions`.

## Required reports

Every stage report must name the worst diagnostic, not only its value, and
must distinguish passes, warnings, failures, and stages withheld by an earlier
gate. Include readable Markdown, machine-readable JSON/CSV, launch/job
inventory, exact restart/checkpoint evidence, source and environment
provenance, and a complete verified SHA-256 inventory.
