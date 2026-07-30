# BP1 test plan: chunked projected Gamma--Dirichlet bank

## Purpose

This protocol tests the memory-bounded source-bank constructor described in
[`rjmcmc_chunked_projected_bank.md`](rjmcmc_chunked_projected_bank.md), then
progresses—one hard gate at a time—to a compressed fixed-root marginal
likelihood.

It does not authorize RJ acceptance, inference on \(K\), partition weights,
or output to `PARIS_inversions`.

The completed calibrated BP1 result is recorded in
[`rjmcmc_chunked_projected_bank_g4_bp1_report.md`](rjmcmc_chunked_projected_bank_g4_bp1_report.md).
G0--G3 passed substantively, but G4 found decisive finite-source likelihood
instability at every retained rank.  No G4 development lock was published,
and the confirmation array, G5, and G6 were not run.

The next source-bank question is a separate exploratory IID-versus-QMC
attribution experiment, specified in
[`rjmcmc_chunked_projected_bank.md`](rjmcmc_chunked_projected_bank.md#required-iid-versus-qmc-attribution-experiment).
It does not reopen or weaken the completed G4 decision.  It asks whether the
same direct likelihood is more stable under independent Monte Carlo than
under the blocked scrambled-Sobol catalogue, and whether either method has
usable component-weight ESS at the tested scale.  Unlike the production-lock
stages below, this method-development comparison may iterate while retaining
and reporting every attempted configuration.

## Frozen input

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/
dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/
paris_may_2014_gamma_beta_native.nc

schema:
paris-may-2014-gamma-beta-native-v1

SHA-256:
24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044

shape:
1382 observations x 183 latitude x 128 longitude
```

Use branch `codex/rjmcmc-chunked-projected-bank`.  The actual candidate is
the fetched complete 40-character commit SHA, not a short SHA in this plan.
Create a clean detached worktree and a fresh run root:

```text
/group/chem/acrg/brendan_for_codex/
rjmcmc_chunked_projected_bank/<full-SHA>
```

Any scientific source change requires a new commit, push, worktree, and run
root.  Reporting-only repair may reuse immutable artifacts only under the
existing two-revision reporting rule.

## Frozen initial controls

These controls are not selected from the realized PARIS residual:

```text
source sample count S: 65536
full spectrum r: complete numerical rank (expected 1381)
q engineering envelope: 16, 32, 64, 128
component reference M: 256
source seeds: 731 (development), 1877, 4099, 8317 (confirmation)
dtype: float64 throughout
Sobol bits: 52
Sobol block rule: [21201, 2222] on the frozen root
```

The scientific calibration is now frozen in
[`rjmcmc_chunked_projected_bank_g4_threshold_supplement.md`](rjmcmc_chunked_projected_bank_g4_threshold_supplement.md).
It sets the modeled European-domain physical-total CV to 0.20 and the frozen
GBR physical-total CV to 0.50.  The resulting common native concentration is
\(\eta=528.618161317525\), with root variance
\(0.022861001527515423\).  These values are independent of partition and
\(K\); the driver must recompute them from authenticated prior fields and
fail closed if they do not replay.

## G0: preflight

On a quiet login node:

1. verify clean full-SHA provenance;
2. link only the canonical pinned `.pixi`;
3. authenticate the frozen input before reading it;
4. record Python, NumPy, SciPy, xarray, BLAS, and Pixi versions;
5. run only the focused aggregation-error tests;
6. run focused Ruff format/check and Pyright;
7. run a tiny v3 construction, JSON replay, and binary roundtrip; and
8. publish `G0_COMPLETE.txt` last.

Do not run repository-wide tox.

## G1: algorithm and replay lock

This stage uses synthetic and moderate operator sizes, not a full PARIS bank.
Exercise:

- \(S=(8,64,1024,65536)\) on the small synthetic root cases;
- \(S=(8,64,1024)\) on the forced-multiblock moderate case;
- \(q=0,1,q<r,q=r\);
- singleton, heterogeneous, and extremely small alpha values;
- one and forced-multiple Sobol dimension blocks;
- stable-cell permutations and incidental root-label representations;
- seeds 731, 1877, 4099, and 8317;
- several legal allocation chunks \(C\); and
- one fixed projection microbatch \(P\) across those \(C\).

Required gates:

1. existing v1/v2 golden JSON and SHA identities remain unchanged;
2. v3 repeated construction and JSON replay are exact;
3. Sobol coordinates/allocation shares match the v2 all-at-once catalogue
   exactly;
4. v3 projected arrays are bitwise identical across allocation chunks when
   \(P\) is fixed;
5. v3 versus v2 leading-\(q\) coordinates meet a frozen tight float64
   tolerance (record maximum absolute and ULP differences);
6. sample requests never exceed \(C\), and projection calls never exceed
   \(P\);
7. nested same-seed prefixes and stable-ID permutation invariance pass;
8. a direct \(q\)-rank bank and legacy full-\(r\) bank produce equivalent
   leading-\(q\) compression and likelihood; and
9. malformed metadata, ranks, chunks, identities, and binary arrays fail
   closed.

Predeclare a modest \(P\) ladder, for example 64, 128, and 256.  Select one
common \(P\) using only identical-output throughput on moderate synthetic
operators on an exclusive node, then lock it.  The selected \(P\) is not
retuned in G3.

Do not use `Sobol.fast_forward`.  Sequential engines restart from row zero
after interruption.

## G2: observation-blind spectrum lock

Construct the full PARIS root spectrum once from operator, errors, and the
fixed native alpha field.  The realized `mf` may be authenticated and carried
for a later likelihood screen but must not enter this computation.

Publish one create-only spectrum bundle containing:

- complete eigenvalues and \(1382\times r\) basis;
- mean design, noise scales, and native identities;
- explained-variance diagnostics;
- canonical eigenvector sign rule;
- little-endian array hashes; and
- source commit/input/environment identities.

The published bundle is authoritative.  Later seed/chunk tasks consume it
rather than independently rerunning `eigh`, because nearly degenerate
eigenspaces can rotate despite sign canonicalization.  A second-node
construction is a numerical audit, not an alternative source of truth.

## G3: frozen-PARIS resource and parity gate

No clustering or posterior is allowed in this stage.

### G3a: actual-input prefix parity

Using all 23,424 cells, \(S=256\), development seed 731, the locked \(P\),
and a modest \(q\):

- build the all-at-once v2 reference;
- build v3 at every candidate \(C\);
- authenticate catalogue and `[21201, 2222]` block identities;
- apply the G1 exact/tolerance parity gates; and
- verify no realized residual was accessed by the builder.

### G3b: full source resource matrix

Build \(S=65536,q_{\max}=128\), seed 731, with the predeclared \(C\) ladder
1024, 2048, 4096, and 8192.  Run the twelve homogeneous
\((C,\text{repeat})\) tasks as one Slurm array `0-11%1` after one excluded
warm-up; the concurrency cap preserves sequential timing while retaining one
auditable array identity.  Larger chunks reduce repeated tree-propagation
overhead but increase shares, uniform, and inverse-Beta fraction arrays; do
not add another candidate after inspecting the matrix.

Request one CPU, 16 GiB, and no more than 60 minutes per candidate.  Suggested
hard resource gates:

```text
sacct MaxRSS <= 12 GiB
no swap, OOM, or filesystem quota failure
one binary bank plus small metadata (no inode explosion)
complete constructor and fingerprint wall time <= 45 minutes
```

Each timing candidate requests an exclusive node.  The array remains
`0-11%1`, so candidates neither share nodes with unrelated jobs nor compete
with one another.  The recovery rationale and preserved non-exclusive
failure are recorded in
[`rjmcmc_chunked_projected_bank_g3_certifier_recovery.md`](rjmcmc_chunked_projected_bank_g3_certifier_recovery.md).

All candidates must produce the same projected-array digest because \(P\) is
fixed.  Select \(C\) by lowest median elapsed time among candidates passing
the resource gate; use smaller \(C\) as the deterministic tie-break.  A
resource miss does not justify lowering \(q_{\max}\).

Record constructor-stage and full-process RSS separately where possible.
Include input loading, spectrum consumption, immutable copies, JSON
fingerprinting, binary output, and checksum publication in full-process RSS.

## G4: source-bank and science lock

The separate observation-blind threshold supplement is
[`rjmcmc_chunked_projected_bank_g4_threshold_supplement.md`](rjmcmc_chunked_projected_bank_g4_threshold_supplement.md).
It must be committed and pushed in the same full-SHA source used for G0--G4
before inspecting any G4 result.  It gives exact formulas and numerical pass
values for:

- normalized mean error in analytic spectrum coordinates;
- relative covariance error and its treatment of tiny analytic eigenvalues;
- every one-dimensional and joint tail diagnostic;
- median and 99th-percentile source-likelihood differences across nested
  \(S\), \(q\), and independent scrambles;
- the exact prior-predictive mass/offset/operator grid;
- the minimum number of consecutive passing q values defining a common
  suffix; and
- the all-seed confirmation rule.

The inherited tiny-oracle likelihood reference values—0.05 nat median and
0.2 nat 99th-percentile absolute error—may inform that supplement, but they
are not automatically valid PARIS thresholds because no exact PARIS
quadrature oracle exists.  If the supplement cannot be justified
observation-blind, stop at the successful G3 engineering result.

Rebuild the selected development bank from zero and require identical binary
digest.  Publish a create-only little-endian float64 C-order location bank
and canonical manifest as specified in the background note.

For \(q=16,32,64,128\), using prefixes of the same \(q_{\max}\) bank:

- compare sample mean/covariance with analytic Dirichlet moments;
- inspect marginal and selected joint prior-predictive tails;
- compare normalized source-bank log likelihoods on a predeclared
  prior-predictive grid;
- compute between-prefix and nested-\(S\) stability diagnostics; and
- require all finite normalization and support gates.

The observed PARIS residual is not part of q selection.  If a common passing
suffix exists, choose the smallest \(q\) in that suffix; otherwise stop or
predeclare a larger observation-blind envelope.  Do not choose the
best-looking rank post hoc.

Run identical locked tests for seeds 1877, 4099, and 8317.  Every seed must
pass; do not average away a failure.

## G5: clustering gate

Only after G4 publishes a source lock:

1. consume the immutable source bank;
2. use the locked \(q\);
3. run a predeclared component ladder or fixed \(M=256\);
4. use three deterministic SciPy k-means++ restarts, fixed cluster seed 731,
   and a fixed iteration cap;
5. report clustering RSS separately; and
6. publish component weights, means, covariances, ordering, and identities.

Require:

- positive weights summing to one;
- exact cluster population mean/covariance closure;
- deterministic component ordering and replay;
- finite compression KL bound;
- prior-predictive source-versus-compressed likelihood gates; and
- confirmation on seeds 1877, 4099, and 8317 without retuning.

Moment and KL checks alone are insufficient: earlier tiny experiments showed
non-monotone pointwise accuracy as \(M\) changed.

## G6: fixed-root conditional PARIS screen

This stage is optional and runs only after all previous locks.  It may use the
realized PARIS observation to compare the locked finite source and compressed
fixed-root likelihoods over a predeclared total-mass grid and, if desired, a
fixed-root posterior.  It may not:

- tune \(q,r,M,S,\eta\), seeds, or gates;
- compare computational partitions or \(K\);
- turn approximation leakage into a structural weight;
- write to `PARIS_inversions`; or
- claim that the marginal likelihood is exact.

Any later partition/RJ experiment needs a separate tower/evidence protocol.

## Failure and recovery rules

Classify outcomes as:

- **incomplete/software failure**: missing artifact, identity mismatch,
  malformed output, invalid replay, job loss, or code exception; or
- **valid scientific/resource hard stop**: a complete authenticated artifact
  misses a predeclared numerical, stability, or resource threshold.

Preserve every failed run root.  Rerun only genuinely missing immutable array
tasks under the same source.  Do not change thresholds after seeing PARIS.
Certifiers publish decisions and completion/lock markers last.

## Work intentionally left to the HPC agent

The HPC agent may:

- add the frozen-input driver and create-only binary/manifest writer around
  the committed constructor;
- implement bounded RSS/timing instrumentation;
- draft and commit the observation-blind G4 threshold supplement before
  submitting any G4 task, or stop at G3 if it cannot be justified;
- submit and monitor the fixed matrices;
- debug software/launcher/checkpoint defects without weakening scientific
  gates;
- rerun missing immutable tasks;
- merge/checksum complete artifacts; and
- write a full BP1 report with job IDs and hard-gate decisions.

The HPC agent may not choose scientific concentration, \(q,r,M,S\), seeds, or
thresholds from the real-data results.  If the common concentration has not
been supplied, it must stop after G3 and report the exact remaining choice.

## Handover prompt

```text
Continue the chunked projected Gamma--Dirichlet source-bank experiment on
BP1 from branch codex/rjmcmc-chunked-projected-bank.

Read first:
- docs/plans/rjmcmc_chunked_projected_bank.md
- docs/plans/rjmcmc_chunked_projected_bank_hpc_test_plan.md
- docs/plans/rjmcmc_exact_mixture_paris_probe_bp1_report.md
- the relevant aggregation-error derivations in sibling inversions-knowledge

Fetch the branch, record the complete 40-character SHA, create a clean
detached worktree, link only the canonical pinned .pixi, and create:

/group/chem/acrg/brendan_for_codex/
rjmcmc_chunked_projected_bank/<full-SHA>

Follow G0--G6 in order and stop at the first hard gate. Do not run full tox.
Use only focused experimental tests, Ruff, and Pyright. Authenticate the
frozen NetCDF SHA before reading it. Publish artifacts create-only and
completion/lock markers last. Preserve failed evidence. Write nothing to
PARIS_inversions and do not open any protected catalogue.

The scientific bank must use the existing v2 Sobol coordinate catalogue,
float64 throughout, sequential persistent Sobol engines, a fixed projection
microbatch P, and only an allocation chunk C selected by identical-output
resource performance. Do not use Sobol.fast_forward. Freeze one authoritative
observation-blind full spectrum before seed tasks. Keep full r separate from
stored q; coordinates q+1:r remain the analytic Gaussian complement.

You may implement the real-input driver, binary manifest writer, bounded
resource instrumentation, the pre-G4 observation-blind threshold supplement,
and reporting needed by this plan. Commit and push such iteration before
using it scientifically; every source change gets a fresh full-SHA run root.
Do not choose eta, q, r, M, S, seeds, or gates from the observed PARIS
residual. If the common native concentration or G4 thresholds are not
scientifically frozen, stop after G3 and report that blocker.

Report exact SHAs, paths, job IDs, terminal states, checks, MaxRSS, timings,
artifact digests, stage decisions, and the first unreached gate.
```
