# BP1 test plan: exact-mixture independent-scramble confirmation

## Purpose

This protocol is the confirmation stage authorized by the successful
development run at:

```text
development revision: d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea
development run root:
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_compression/d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea
```

It does not rerun or retune the development ladders.  It freezes:

```text
source sample count: 65,536
compressed component count: 256
cluster restart seed: 731
independent source scrambles: 1,877, 4,099, 8,317
```

All six exact tiny root cases are run at all three source scrambles.  All 18
artifacts must pass.  There is no protected catalogue and no structural
inference in this protocol.

## Authenticated development inputs

The confirmation implementation pins:

```text
source decision raw SHA-256:
61fa3ff7bd2aee439b532a8c70633df05cc3cc452c6a44aff377ac4ad613fa9a

source lock internal SHA-256:
80d631763473f7a9262ef0314b4efcab73e44a1d85697bc9aee243a4e3f41ead

compression decision raw SHA-256:
932efada6a7ad1894696c40fe78c515f2da661cfa780ddb7c659304c370bdf5e
```

Every confirmation shard authenticates those canonical files, the
development revision, protocol and driver identities, selected sizes, exact
case input identity, analytic spectrum identity, and absence of protected or
production output.

The optimized confirmation candidate may have a later Git SHA because it
uses a cached batched evaluator.  It consumes the immutable development
decisions as authenticated inputs and may not alter their contents.

## Confirmation matrix

Cases:

```text
near_gaussian__two_cell__root
near_gaussian__four_cell__root
skewed__two_cell__root
skewed__four_cell__root
boundary_heavy__two_cell__root
boundary_heavy__four_cell__root
```

For each case and source seed:

1. rebuild the exact C1 oracle state;
2. verify its case and analytic-spectrum identities against the development
   source lock;
3. construct exactly 65,536 scrambled-Sobol allocations;
4. score the direct finite mixture against exact quadrature;
5. compress to exactly 256 components with three fixed k-means++ starts using
   cluster seed 731;
6. score the compressed likelihood against exact quadrature and against its
   source bank;
7. verify moment closure and a finite non-negative compression KL bound; and
8. publish canonical JSON and then its completion marker.

The realized observation is unavailable to basis selection.  Source sizes,
component counts, source seeds, cluster controls, cases, thresholds, and
confirmation rule cannot be overridden.

## Gate C0: confirmation preflight

Use a clean detached full-SHA worktree with the canonical frozen `.pixi`
environment.  The committed preflight must:

- verify the candidate SHA and clean status;
- authenticate both `d23e9d9` development decisions;
- run the focused exact-mixture, driver, certifier, and confirmation tests;
- run focused Ruff formatting/checks and Pyright;
- record Python, NumPy, SciPy, and Pixi versions; and
- publish `PREFLIGHT_COMPLETE.txt` last.

Any source change requires a new commit, push, full-SHA worktree, run root,
and C0 rerun.

## Gate C1: 18-shard Slurm matrix

Submit one array with 18 tasks:

```text
array index = case index * 3 + source-seed index
account: chem007981
CPUs per task: 1
memory: 16 GiB
wall time: 4 hours
BLAS/OpenMP/NumExpr threads: 1
```

Each task publishes exactly:

```text
confirmation/<case-id>__seed<seed>.json
markers/confirmation/<case-id>__seed<seed>.complete
logs/confirmation/<case-id>__seed<seed>.<job-id>.log
```

A scientific failure still publishes a valid artifact and marker.  Missing
artifacts, malformed JSON, nonzero exits, missing markers, or identity
failures are execution failures.

## Gate C2: all-artifact certifier

C2 is forbidden until exactly 18 artifacts and 18 exact markers exist.  The
committed certifier authenticates:

- the exact case-by-seed Cartesian product;
- confirmation candidate, driver, and protocol identities;
- all pinned development identities;
- fixed \(S\), \(M\), and cluster seed;
- no retuning, observed-residual basis selection, protected access, production
  output, or structural authority; and
- every scientific and moment-closure check.

It publishes one canonical decision.  `eligible=true` requires all 18
artifacts to pass.  No partial-case, majority, pooled, or average criterion is
allowed.

## Interpretation

An eligible C2 decision establishes that the root-only approximation is
stable across the three independent source scrambles on the six exact tiny
cases.  It justifies a PARIS rank/resource probe and fixed-partition
conditional likelihood experiment.

It does not establish:

- a multi-region approximation;
- a partition- or \(K\)-invariant production evidence calculation;
- a PyTensor gradient;
- an RJ acceptance ratio;
- a PARIS posterior; or
- authority to update structural prior weights with data.

If any shard fails, preserve the complete matrix and stop.  Do not change
seeds, sizes, gates, cluster restarts, or cases after inspecting confirmation.

## Required report

Record:

- confirmation full SHA, worktree, and run root;
- development decision paths and all three pinned digests;
- C0 checks and runtime versions;
- Slurm job ID and every task state;
- elapsed time, MaxRSS, construction time, storage, and evaluation throughput;
- per-case/per-seed source and compressed scientific metrics;
- maximum moment-closure errors and compression KL bounds;
- C2 decision and failures, if any;
- a verified SHA-256 inventory; and
- explicit confirmation that nothing was written to `PARIS_inversions`.
