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

## First recovery preflight failure

The first recovery implementation was pushed at
`4a02a742079aefeff087cec45ae26fcb86ffcd83` and executed from its own
detached worktree into the fresh create-only run root

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/
4a02a742079aefeff087cec45ae26fcb86ffcd83
```

G0 passed all 91 focused tests and then stopped at the Ruff format gate
because the newly added exclusive-launcher regression required mechanical
formatting.  The preserved `g0/preflight.log` has SHA-256
`8aa182638dbbc5ef996b2983196759445574a692794c01f33cc50888357fda39`.
No `G0_COMPLETE.txt` marker was published and no Slurm job was submitted.
The run root and its failed preflight log are preserved without modification
or artifact reuse.  The formatting-only repair and this provenance record
require another pushed full SHA, detached worktree, and fresh SHA-keyed run
root.

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

## Human substantive adjudication

The project owner subsequently clarified that the G0--G4 gate names are
documentation and reproducibility scaffolding, not a requirement to repeat
scientifically decisive work after an automation-only failure.  A complete
stage may be adjudicated as passed when its underlying evidence is sufficient
and the automatic check failed only for a diagnosed technical reason.

The repaired certifier was therefore applied read-only to the complete
`92cd2335d02119dd8114bb6808d1848433e03fc2` matrix.  It passed all four
chunks and all twelve tasks, selecting \(C=8192\) and \(P=256\).  The new
separate recertification report has SHA-256
`d4825a511f06c93e6f6dcbe3beee1c2b890f9633b8e2fec0b60a5794741cff3e`.
The original run root remains unchanged.

The exclusive-node recovery array `18207955` was stopped once the owner
clarified that whole-node isolation had not been requested.  Tasks 0--7
completed; tasks 8--11 were deliberately cancelled before starting.  The
three complete \(C=1024\) repeats, three complete \(C=2048\) repeats, and
the completed \(C=4096\) candidates all reproduced projected-array SHA-256
`7f309f7560bd9695d5d3093e1542b7e5d42c0b8634abdd2c5a95da7dc61d86a0`
and binary-file SHA-256
`aec20f2d3fd1c93c6ba52c2fbc4a84986121debafbad48e3b7e07943911a33a7`.

G3 is consequently a substantive pass.  For the current execution, the
smallest fully replicated same-SHA chunk, \(C=1024\), and its already locked
\(P=64\) are used as engineering controls for G4.  This choice does not alter
the native model, Sobol catalogue, or projected scientific bank.  The
human-adjudication artifact authenticates both the complete prior matrix and
the current same-SHA three-repeat reference, records both Git revisions, and
publishes the current `G3_COMPLETE.txt` marker last.
