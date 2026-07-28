# G3 certifier recovery and isolated resource rerun

## Preserved failed execution

The create-only run rooted at

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/
92cd2335d02119dd8114bb6808d1848433e03fc2
```

is sealed as failed certification evidence.  G0, G1, authoritative G2, the
different-node G2 audit, G3a, the excluded warm-up, and all twelve elements of
Slurm array `18201289` completed.  The twelve `projected_locations.npy` files
are byte-identical with SHA-256
`aec20f2d3fd1c93c6ba52c2fbc4a84986121debafbad48e3b7e07943911a33a7`.

Certifier job `18206117` then exited `3:0`.  Its preserved
`g3_decision.json` has SHA-256
`dc523dcda5f6301c7eda97498138d40da68412982856081b34cf2bc4f4dcdc49`.
It records every internal and binary-identity check as passing, but every
Slurm state as `null` and every `MaxRSS` as zero.  No `G3_COMPLETE.txt` was
published and no G4 stage ran.

This was a reporting implementation failure, not a resource or scientific
failure.  The certifier queried `sacct` using `JobIDRaw` while indexing
candidates by logical `array_job_task` identity.  On BP1, for example, task
`18201289_0` has logical `JobID=18201289_0` but physical
`JobIDRaw=18201296`; consequently none of the accounting rows joined.
The repair queries `JobID`, retaining `.batch` aggregation because that row
contains `MaxRSS`.

The failed decision, logs, candidate manifests, arrays, resource records, and
markers must not be changed, deleted, linked, copied, or relabelled.

## Predeclared recovery

The reporting repair is committed and pushed before execution.  Its exact
40-character origin SHA gets a new detached worktree and a fresh SHA-keyed
run root.  No artifact from the failed root is reused.  G0 through G3 are run
again in their original order, with the same input, calibration, ladders,
seeds, thresholds, resource limits, and create-only rules.

Two timing-selected engineering stages now request exclusive nodes:

- G1, which selects the fixed projection microbatch \(P\); and
- every element of the sequential G3 resource array, which selects allocation
  chunk \(C\).

The first execution requested only isolated cores.  Its candidates therefore
shared nodes with unrelated CPU- and memory-heavy jobs, and the nominal
\(C=8192\) median was only about 0.7% below the \(C=2048\) median.  Exclusive
nodes remove that external-workload confound.  The G3 array remains
`--array=0-11%1` to prevent candidates from competing with one another.
This isolation correction was frozen before any recovery result and changes
no catalogue coordinate, model quantity, scientific threshold, or projected
bank identity.

Only a passing repaired G3 certificate may open G4.  Any earlier gate failure
is preserved and stops the new run.  The protected catalogue remains sealed,
and nothing is written to `PARIS_inversions`.
