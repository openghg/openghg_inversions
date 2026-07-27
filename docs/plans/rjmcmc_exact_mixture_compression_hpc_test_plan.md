# BP1 test plan: exact aggregation mixture compression

## Purpose

This plan tests the root-only approximation specified in
`docs/plans/rjmcmc_exact_mixture_compression.md`.  It deliberately separates
two approximation errors:

1. replacing the exact continuous Gamma--Dirichlet Gaussian location mixture
   by a large equal-weight scrambled-Sobol mixture; and
2. replacing that finite mixture by moment-preserving Gaussian clusters.

The test uses the six frozen root cases and exact quadrature definitions from
the C1 conditional-allocation screen.  It is a tiny-oracle development test,
not a PARIS run, an RJ transition, or a production inversion.

## Candidate identity

The authoritative candidate is the complete pushed Git SHA recorded in the
run root.  Resolve it only after the implementation, driver, certifier, tests,
this plan, and all shell assets are committed:

```text
branch: codex/rjmcmc-exact-mixture-compression
candidate full SHA: git rev-parse origin/codex/rjmcmc-exact-mixture-compression
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_compression/<full-SHA>
```

Every rerun after a source change requires a new full-SHA worktree and new run
root.  Never reuse an artifact, marker, or lock from another SHA.

## Frozen scientific protocol

### Cases

The source and compression stages both cover exactly:

```text
near_gaussian__two_cell__root
near_gaussian__four_cell__root
skewed__two_cell__root
skewed__four_cell__root
boundary_heavy__two_cell__root
boundary_heavy__four_cell__root
```

The exact likelihood, state grids, checkerboard scoring view, prior weights,
posterior weights, gradient catalogue, and numerical thresholds are inherited
from the authenticated C1 definitions.  The realized residual is not used to
select the analytic eigenbasis.

### Source-bank ladder

The source stage uses one scramble seed, `731`, and the nested powers-of-two
ladder:

```text
65,536
262,144
1,048,576
```

For each case and size, the direct normalized finite mixture is compared with
exact quadrature.  A common source lock exists only if there is a suffix of at
least two sample sizes for which **all six cases** pass all inherited
scientific thresholds.  The selected size is the smallest size beginning
that common suffix.

This is a common computational size, not one common numerical source
artifact: every case has a different native operator and therefore a
different authenticated source-bank SHA.

### Compression ladder

Compression cannot run until the common source lock exists.  Every
compression shard must:

1. authenticate the lock;
2. rebuild its case-specific source bank at the common locked size;
3. reproduce the source SHA, input identities, spectrum identities, and
   scientific pass certificate; and
4. only then evaluate component counts
   `16, 32, 64, 128, 256, 512, 1,024`.

Compression uses:

- the complete numerical analytic residual rank;
- three deterministic SciPy `kmeans2` k-means++ starts;
- clustering seed `731`;
- at most 100 k-means iterations; and
- exact cluster weights, means, and population covariances.

A common compression decision is eligible only if there is a suffix of at
least two component counts for which **all six cases** pass every inherited
exact-quadrature scientific threshold.  Select the smallest component count
beginning that common suffix.

Incremental compressed-versus-source errors, the analytic projection bound,
the finite-bank compression KL bound, moment closure, storage, build time, and
evaluation throughput are recorded as diagnostics.  They do not replace the
exact scientific gates.

### What a pass means

A development pass establishes only that one root-only construction clears
the six frozen tiny-oracle cases under a common source size and a common
component count.  It does not establish:

- continuous-law exactness of the finite Sobol bank;
- arbitrary multi-region accuracy;
- PARIS-scale feasibility;
- PyMC/PyTensor gradients;
- partition- or \(K\)-invariant absolute evidence; or
- validity for RJ acceptance or structural weighting.

Independent source scrambles and a subsequent real-data conditional screen
are separate phases and may not be inferred from this run.

## Stage G0: preflight

Run on the quiet login node or as a short batch job.  The committed preflight
asset must verify:

1. the checkout is the exact full SHA and is clean apart from the canonical
   `.pixi` link;
2. the run root is fresh and outside `PARIS_inversions`;
3. the pinned Pixi environment is used without installation;
4. focused exact-mixture, driver, and certifier tests pass;
5. focused Ruff and Pyright checks pass;
6. the source-stage smoke case publishes canonical JSON;
7. the smoke replay without timings is byte-identical; and
8. a completion marker is published last.

Do not run the repository-wide tox matrix for this experimental screen.

## Stage G1: source development matrix

Submit six Slurm array tasks, one per case.  Each task evaluates the complete
three-size source ladder and publishes:

```text
source/<case-id>.json
markers/source/<case-id>.complete
logs/source/<case-id>.<job-id>.log
```

Suggested resources per task:

```text
account: chem007981
CPUs: 1
memory: 16 GiB
wall time: 24 hours
threads for BLAS/OpenMP/NumExpr: 1
```

The task must write the JSON atomically and the completion marker last.  A
missing artifact or marker is an incomplete run, not a scientific failure.

### G1 merger hard gate

Run the committed certifier only after exactly six artifacts and six valid
markers exist.  It must authenticate canonical JSON, the candidate revision,
driver and protocol digests, the exact case catalogue, every ladder entry,
and every per-case source certificate.

It publishes one of:

- an eligible common source lock using schema
  `rjmcmc-compressed-mixture-common-source-lock-v1`; or
- an ineligible source decision using schema
  `rjmcmc-compressed-mixture-common-source-decision-v1`.

If the decision is ineligible, stop.  Do not alter the sample ladder, gates,
seed, or suffix rule after examining results.

## Stage G2: compression development matrix

G2 is forbidden without the eligible G1 lock.  Submit six Slurm array tasks,
one per case, passing the exact lock path.  Each task evaluates the complete
five-count compression ladder and publishes:

```text
compression/<case-id>.json
markers/compression/<case-id>.complete
logs/compression/<case-id>.<job-id>.log
```

Use the same requested resources as G1.  Peak RSS and CPU utilization should
be recorded from Slurm accounting.

### G2 merger hard gate

After exactly six artifacts and six valid markers exist, the committed
certifier authenticates the source lock, confirms every case rebuilt the
locked source exactly, and computes the common all-case component-count
suffix.  It publishes:

```text
decision/common-compression-decision.json
```

An ineligible result is a scientific hard stop for this bounded component
ladder.  Do not add components, change clustering, relax thresholds, or
inspect a protected catalogue post hoc.

The 512- and 1,024-component points were added before the BP1 protocol was
frozen.  In the initial local six-case development run, all source ladders
passed and the common source size was 65,536.  Five cases passed the complete
16--256 compression ladder.  The boundary-heavy four-cell case passed at 64
and 256 components, but 128 components narrowly missed only the
posterior-SD-relative-error gate (0.02194 against a 0.02 threshold), despite a
monotonically decreasing integrated compression bound.  The two larger points
test for a stable high-component suffix.  No confirmation seed or protected
case was inspected when making this bounded extension.

## Deferred G3: confirmation

The development protocol records confirmation source seeds:

```text
1,877
4,099
8,317
```

G3 is intentionally deferred until G1 and G2 pass.  Before running G3, commit
a separate protocol that freezes the selected source size and component count
and applies all three source scrambles to all six cases without retuning.
Failure of any confirmation shard invalidates promotion.

## Diagnostics and interpretation

The report must distinguish:

- source-bank error relative to exact quadrature;
- compression error relative to the locked finite bank;
- compressed-likelihood error relative to exact quadrature;
- analytic source covariance versus empirical source covariance;
- exact finite-bank versus compressed global moments;
- projection KL/TV bounds;
- compression KL bound;
- offline construction time;
- likelihood evaluation throughput; and
- artifact memory/storage.

If G1 fails, clustering is not implicated.  If G1 passes and G2 fails, the
finite source is adequate on these cases but the selected compression family
or ladder is not.  If both pass, confirmation and PARIS-scale work become
justified, but no structural claim is yet allowed.

## Numerical and operational failures

The operator may debug a launcher, checkpoint, canonical-JSON, dependency, or
genuine numerical-software defect.  Preserve the failed run root and report.
Any source edit requires a new commit, push, full-SHA worktree, and run root,
starting again at G0.

Do not treat a scientific threshold failure as a software defect.  Do not
clamp probabilities, discard difficult cases, redraw failed clusters, relax
gates, or tune against the protected/confirmation seeds.

## Required final report

Record:

- candidate branch and full SHA;
- BP1 worktree and run root;
- Python, NumPy, and SciPy versions;
- all Slurm job IDs and terminal states;
- exact artifact and marker counts;
- source and compression common-pass tables;
- locked source size and component count, if eligible;
- all scientific metrics by case and ladder point;
- compression bounds and moment errors;
- elapsed time, throughput, MaxRSS, and CPU efficiency;
- SHA-256 inventory of the complete run;
- whether confirmation was authorized;
- the exact hard gate reached; and
- explicit confirmation that nothing was written to `PARIS_inversions`.
