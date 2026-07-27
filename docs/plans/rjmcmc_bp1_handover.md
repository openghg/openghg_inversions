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

The original BP1 entry branch was
`codex/rjmcmc-topology-conditioned-hmc` at
`93175779440fe1c7da9351dfd28ca438169ef961`. Later experiments used dedicated
downstream branches and full revisions recorded below. Always use the exact
full revision attached to an immutable run root; do not infer a candidate from
a short run-directory name or from the current head of the original branch.

## 2026-07-26 to 2026-07-27 BP1 execution update

This update supersedes the pending-work language later in this document.

| Full revision | Branch | Result |
|---|---|---|
| `93175779440fe1c7da9351dfd28ca438169ef961` | `codex/rjmcmc-topology-conditioned-hmc` | requested clean BP1 starting revision |
| `5bb41399e45b78954488e286da3f40371dcb956e` | `codex/rjmcmc-topology-conditioned-hmc-d0-exact-mh` | exact-MH repair; H2d D0/D1 passed and D2 hard-stopped; Gaussian A0 passed and A1 failed scientific shape |
| `a004e526033432df4e893e63119cc9aa4928c95c` | `codex/rjmcmc-fixed-basis-aggregation-nuts` | aggregation-aware fixed-basis NUTS implementation; retained sampling withheld |
| `6ff3afe56416e701ac1fc4ae45676d08ea28229b` | `codex/rjmcmc-aggregation-transported-mixture` | normalized transported-mixture NumPy foundation; T2 development hard stop and held-out evaluation withheld |
| `e9e422fe3ab973898cffbd38df00b689efe212b8` | `codex/rjmcmc-mh-local-search-auth-parent-fix` | MH-guided local partition-search S0 passed at the factor-four budget |

All five revisions were clean and synchronized with
`https://github.com/openghg/openghg_inversions.git` when audited. The common
Pixi lock SHA-256 is
`4ed1244c33ffb7ef929bad73d8bd9944e49ed9b36b51fa05163b59b2a5b2f564`.

### H2d disposition

The H2d source result is under:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/5bb41399e45b78954488e286da3f40371dcb956e
```

D0 passed after jobs `18169133` and `18169140` each produced 10,000 valid,
bit-exact structural reverses with zero skipped reverse paths. The D0
certificate SHA-256 is
`d8ddd119d28ba86ee6a840dc357e5120fa20f823975d4b8c9af269ca1dd2e1ed`.
D1 passed, including portability jobs `18180788` and `18180790`; its
certificate SHA-256 is
`8df068e5e2750a47e83bca9500b31296beb74ba56aedff15c28ba03b0af63def`
and its artifact-inventory SHA-256 is
`b4e2cf737d96cfddd57704f35bfa2c8ab10372fd0193494cdbb766ca0e71a27c`.

D2 used:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_topology_conditioned_hmc/5bb41399e45b78954488e286da3f40371dcb956e/calibration/d2/af08d98c5fe38c745b4fbabaa410363aa5b25ce142e39fd52ce6df2d4787a152
```

Job `18181936` never started because the `default` account was denied. The
unchanged topology-freeze job completed as `18181938` and the 126-task
development array completed as `18181943`, both under `chem007981`. The
aggregate D2 hard-stop SHA-256 is
`833db35411276a7d57be1de6ac796999c4400ea43bd953ebf1b1b84c1fd2683a`;
the readable report SHA-256 is
`080dfa289fab6d02a9e1581d6045a0e3c2b5bd6b1cfc5328a4bcd514a374dda7`.

The formal stop at both dimensions was
`production-v3-zero-displacement-score-serialization-hard-stop`. Separately,
no candidate passed every scientific development gate at \(K=50\). At
\(K=250\), exactly one candidate, \(\epsilon=0.2,L=5\), passed the development
science gates, but it never reached held-out validation and is not a certified
calibration. D3 and D4 were withheld. This mobile evidence cannot separate
within-topology position curvature from topology/continuous landing and does
not license RMHMC or source-HMC/structural-map/destination-HMC.

### Aggregation-error disposition

The Gaussian run root is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_aggregation_error/5bb41399e45b78954488e286da3f40371dcb956e
```

A0 passed 56 focused tests with clean Ruff/Pyright; certificate SHA-256
`3830943d256a19530e3030a25e4d0691e3dbebec1fd83ae0b5750923efe87307`.
The final A1 job was `18181894`; certificate SHA-256
`a46c51951a752a34b3678cc1ea98cee097055c1a0ee1b8fa73ae7ce7349cbc34`.
The Gaussian implementation gates passed, but all eight scientific-shape
gates failed. Reported maxima include 5.9846 nat at a conditional anchor,
13.075 reference SD in posterior mean, 25.077 reference SD at an interval
endpoint, 4.064 nat in evidence, and 0.3892 total variation in structural
weights. Gaussian A2/A3, retained A4 sampling, and A5 were withheld.

The normalized transported-mixture foundation at `6ff3afe` is implementation
only: it contains no fitted scientific artifact and no PyMC integration.
Its source and test SHA-256 values are respectively
`ccec9bb5aa62b2e5369768205c51201cf6ffd303367905f4a2c1ea2dca996d75`
and
`459d80f47855b45ed69e74364313c6910a02e3cd7d81ed04cae563c813ec75b4`.

### T2 infrastructure attempts

Preserve all three non-scientific T2 attempts:

1. Inventory
   `da28b5049d68b426f3b5d3dcf20aaff2ac53b1df29d65655b632f64e90e80b97`
   stopped in the immutable binder before preflight. `BIND_FAILURE.json`
   SHA-256:
   `d438c54a754b9bafe5fb0d649a18ffd94f634eaeea90882f180c530092407756`.
2. Inventory
   `e3e237e7587dc80ae17fd7b2d0c26f01775581c3265fdf4e8c836bde2113d4e0`
   stopped in bound out-of-tree preflight because Pyright found 15 genuine
   harness type errors. The earlier hidden `.codex-run-staging` Pyright
   “pass” was the false negative: it analyzed zero files. The 15 errors were
   fixed without suppressions in v3. No scientific code ran in v2.
   `T2_PREFLIGHT_HARD_STOP.json` SHA-256:
   `2b4251ce40c29e5e49d9f355179a425d8ee0965a0a3450b7c6bf5377b00fd8e7`.
3. Inventory
   `539a99de8fc564a79d3dbdd984a52478b32c020fe829ed3b14711ff096b2e3bc`
   passed preflight, SHA-256
   `ddff6a7f337e330a2d6bf77d9f55153591e83ada7377a0f8d5feb6a038221401`,
   but every task in development array `18184746` failed before fitting
   because the bound absolute-path driver did not put the detached source root
   on `sys.path`. `ARRAY_INFRASTRUCTURE_FAILURE.json` SHA-256:
   `a186cda0f4563013bac62ee206cba638ed27e6c718c04312c56a8a880d6b26a3`.

The final v4 attempt used inventory
`11a43a24da003019e600c990da143573f527ce1b85ffeaedada80ec857edbd28`,
provenance SHA-256
`d84bd2390104390ee5a047178fca3405f869749c64ca1ea6f40c791e01549c13`,
and preflight SHA-256
`7764477963b084a372b8535887434d2282deec653805cbb4bdb77d75d024edaf`.
Compute-smoke job `18185082` passed on `bp1-compute050` in 24 seconds with
maximum RSS 271052 KiB; its evidence SHA-256 is
`a08120d06eb6da6e76dc1479fbbcea5e525d895a668b23d6af45c3162763c45d`.
Development job `18185086` then ran all eight cells. Every task exited 2
after 2 minutes 12 seconds to 10 minutes 49 seconds, with maximum observed
RSS 228972 KiB.

The formal development certificate, SHA-256
`99f6479d71bb47515479ad5652949fdf08d6314092aa98512213a927b61affaf`,
records an **infrastructure hard stop**: its canonical sorted `records` key
order contradicted the certifier's catalogue-order check. Its authenticated
pre-scan nevertheless found a complete terminal catalogue: five fitter hard
stops and three ready results. Independent read-only inspection of those
authenticated result files localizes the bounded scientific outcome. All
three fitted cells have `development_pass=false` because the
projection-isolated learned-mixture shape gate failed:
`near_gaussian/two_cell` at \(M=2\),
`near_gaussian/four_cell` at \(M=1\), and
`boundary_heavy/two_cell` at \(M=8\). The other five cells stopped
pre-observation because no fixed restart converged and remained valid; none
read a pseudo-observation or held-out datum.

Held-out evaluation was therefore withheld. The bounded T2 experiment is
finished; do not create a v5 or invent a T3. No A5 or structural \(P/K\)
claim is licensed. Structural weights remain externally fixed at their
declared prior until A5 absolute-evidence and tower gates pass.

The independently reviewed readable report is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_aggregation_transported_mixture/6ff3afe56416e701ac1fc4ae45676d08ea28229b/t2/11a43a24da003019e600c990da143573f527ce1b85ffeaedada80ec857edbd28/report/T2_RESULTS.md
```

Its SHA-256 is
`8ac9f1db197ec6880247dcce98f12ed5c61d6f07cf999249e10c06a24ac07ed9`;
the adjacent read-only report-provenance manifest has SHA-256
`946f843cda39d8147361ab19c81d40c6ab9637632aaab3f90582daae2417462f`.
No `complete.json` exists because T2 did not pass.

The sibling `inversions-knowledge` repository is not present on BP1. It has no
verified remote, pushed revision, or clean-status record and was not used by
the H2d, Gaussian A1, transported-mixture, or MH-guided local-search results.
The earlier
`e77d20cffe7ee0298d9106065c962d24198dabdc` remains provisional historical
context only.

### MH-guided local-search S0 disposition

The S0 experiment tested partition movement as posterior-informed stochastic
local search, not as converged partition-posterior inference. Its passing,
immutable factor-four run is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_synthetic/e9e422fe3ab973898cffbd38df00b689efe212b8/harness-2d9dc06812ab0802a3723c4cb7ef6e66612106d791a924b5558b3f49570f7106
```

It used a \(2\times4,\ K=4\) synthetic grid, deterministic balanced
`largest-nominal` \(P_0\), and aligned, one-edge, and one-relocation planted
truths. Each of four paired noise replicates per scenario used 8,000 excluded
conditioning cycles followed by 20,000 retained production cycles with no
thinning. The fixed and mobile arms forked the same conditioned state. Fixed
cycles had one root slice plus five allocation updates; mobile cycles added
two structural MH attempts.

All five fixed-topology local-versus-NUTS references passed after the
predeclared homogeneous factor-four rescue. The worst diagnostics were:
\(\hat R=1.002571\) for `root_total`, bulk ESS \(=2044.36\) and tail ESS
\(=1691.08\) for `leaf_mass[r0:1_c2:4]`, local MCSE/posterior SD \(=0.02943\)
for `top_half`, half-window difference/posterior SD \(=0.06074\) for
`top_right`, and local-versus-NUTS tolerance use \(=0.60038\) for
`bottom_right`. NUTS had zero divergences.

Median mobile/fixed held-out RMSE ratios were 0.96895 aligned, 0.14576 for the
one-edge mismatch, and 0.26806 for the one-relocation mismatch. Every one-move
replicate accepted exactly one correct structural move and reached
\(P_\star\); no run returned to \(P_0\). Aligned runs accepted no structural
move. This is strong finite-budget local-search evidence in favorable
one-move synthetic cases, not structural mixing evidence.

The final decision SHA-256 is
`2cef819c704f0d062cdb38dc09111fa08e230cf2d21ff4b9ba1dd059df1803ef`;
the root-completion SHA-256 is
`cdeda8440bfd71119f0509529620ebc5be48a06d37b3d18665357103185491f8`.
The final replay checked 75 external hashes with zero mismatches. All
factor-four Slurm tasks completed successfully. The repaired-primary jobs
were `18187568`--`18187576` and `18187661`. Factor-four jobs were `18187702`,
`18187704`--`18187708`, `18187803`, and `18187805`.

Preserve the earlier `ef27efd58c58afaa077d7b1b915a2d4498fbb751` run as
incomplete evidence. Its primary scientific jobs finished, but its
login-side conditional launcher failed before token creation because the
authorization parent directory was absent. The repaired revision changed
only that launcher contract, was independently reviewed, and used a fresh
detached worktree and SHA-addressed run root.

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_synthetic/ef27efd58c58afaa077d7b1b915a2d4498fbb751/harness-96bdfe118c6de4440efb26099ad83037ec227c3d8ad7cd706b9be9ad74d575f2
```

Its jobs were `18187306`--`18187314` and `18187483`. Do not promote or copy
its primary artifacts into the passing run.

The readable result is
[`../reports/rjmcmc_mh_guided_local_search_s0_results.md`](../reports/rjmcmc_mh_guided_local_search_s0_results.md).
S1 and S2 were not attempted. No real-data or partition-marginalization claim
is licensed. The next useful experiment is an atmospheric-like synthetic
screen starting from a reasonable deterministic basis and keeping planted
truths within a small witnessed local-move radius.

Agent-tracker was unavailable on BP1 because its configured database path was
not available. Coordination used native subagents plus direct Slurm
monitoring; this did not affect the immutable run provenance.

## State at handover

The important source checkpoints are:

| Full revision | Meaning | Disposition |
|---|---|---|
| `7a1a1cc673a4b6a6ce0ed7b5123494ebd205b467` | topology-neutral total/contrast HMC metric | H2c hard stop at both \(K=50\) and \(K=250\) |
| `7f7b1509bf032d04c9839ec9fa4d7be69b03e1ab` | first topology-conditioned HMC implementation | failed D0 finite-binary64 structural reversibility audit; never resume |
| `e6199150e680d43e6e3c1388db45773c5337802a` | exact log-coordinate edge-flip and resolution-relocation involutions | its original BP1 attempt remains incomplete; superseded by `5bb41399` |
| `16819f55cea5c6054b0113b751aa9833afa4fa9b` | normalized low-rank aggregation-error baseline | focused local validation passed; later A1 result is recorded above |
| `54045edf67c4703da5909b1fdd2a6081d0a61251` | cached fixed-partition aggregation factors and FullTiling bridge | focused local validation passed; benchmark on BP1 |
| `5bb41399e45b78954488e286da3f40371dcb956e` | bit-exact log-involution MH accounting | H2d stopped at D2; Gaussian closure stopped at A1 |
| `6ff3afe56416e701ac1fc4ae45676d08ea28229b` | normalized transported-mixture foundation | T2 development hard stop; held-out evaluation withheld |

The original branch checkout used for the initial handover was:

```bash
git fetch origin
export CANDIDATE_REVISION=93175779440fe1c7da9351dfd28ca438169ef961
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
equal their structural prior weights. A finite Gaussian closure or NLE may
violate that identity through approximation error, so structural use requires
absolute evidence and tower-property gates, not only a normalized conditional
density.

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
export CANDIDATE_REVISION=6ff3afe56416e701ac1fc4ae45676d08ea28229b
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

The bounded `5bb41399` H2d sequence finished at its D2 hard stop. Preserve its
D0, D1, and D2 artifacts. Do not run validation, D3, D4, or an unplanned
tuning grid from that candidate.

If H2d fails, localize the failure among metric conditioning,
within-topology position curvature, and topology/continuous landing. A
source-HMC/structural-map/destination-HMC proposal is justified only after the
within-topology metric works. Position-dependent/Riemannian HMC is justified
only by strong within-topology curvature variation.

### Track A: aggregation error

This track was independent of H2d. A0 passed, but the dense Gaussian closure
failed A1 scientific shape. Gaussian A2/A3, retained A4, and A5 were therefore
withheld. The bounded normalized transported-mixture successor at `6ff3afe`
also stopped at T2: five cells could not produce a valid fixed-restart fit,
and the three fitted development cells failed the projection-isolated
learned-mixture shape gate. The formal certificate additionally records the
catalogue-order infrastructure contradiction described above. Held-out
evaluation was withheld; there will be no v5 or unplanned T3. Permit
structural \(P/K\) claims only after A5 absolute-evidence and tower-property
gates.

Keep this first integration downstream of `fixed_basis_nuts.py`, preferably in
`fixed_basis_aggregation_nuts.py`. Do not partially wire aggregation error
into the mobile HMC target. When creating that downstream module, move the
FullTiling borrowing bridge there or replace its `object.__new__` construction
with a private validated borrowed-array constructor.

If the dense Gaussian closure fails scientific shape, try a normalized
transported mixture in the same fixed summary space before a conditional
flow/NLE. An NLE must estimate a normalized conditional aggregation-residual
density and must not be allowed to update structural weights until its
evidence error passes A5.

## `inversions-knowledge`

The local knowledge repository was clean at
`e77d20cffe7ee0298d9106065c962d24198dabdc` before publication. After the
private repository is pushed and cloned as a sibling of `openghg_inversions`,
record its remote URL and actual full SHA; do not assume this provisional SHA
remains current.

As of the BP1 results update, that sibling is absent and none of the reported
H2d, A1, T1, or MH-guided local-search evidence relied on it.

Read these sibling paths first:

```text
../inversions-knowledge/docs/research-trails/legacy-tdmcmc-to-experimental-rjmcmc.md
../inversions-knowledge/docs/research-questions/rjmcmc-hmc-nuts-and-transported-tuning.md
../inversions-knowledge/docs/research-questions/learning-non-gaussian-marginal-models.md
../inversions-knowledge/docs/research-questions/posterior-projection-conundrum.md
../inversions-knowledge/docs/derivations/rjmcmc-dimension-matching-and-augmented-spaces.md
../inversions-knowledge/docs/derivations/non-gaussian-aggregation-error-by-marginalization.md
../inversions-knowledge/docs/derivations/posterior-projection-and-exact-marginalization.md
../inversions-knowledge/docs/workflows/validating-trans-dimensional-mcmc-kernels.md
```

## Resource, durability, and stop rules

- Check login-node load and available memory before bounded sequential work.
- Keep aggregate login-node RSS below 200 GB.
- Use Slurm for multi-chain, retained, or large matrix jobs.
- On BP1 compute nodes load `git/2.45.1-pqk5` when Git provenance is needed
  and submit under the explicit `chem007981` account; do not inherit
  `default`.
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
