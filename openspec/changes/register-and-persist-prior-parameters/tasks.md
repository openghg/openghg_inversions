# Tasks

## 1. Shared prior construction

- [x] 1.1 Register labelled prior arrays and capture inline scalars and symbolic associations at construction; verify ordinary/reparameterized lognormal density, scalar-only and length-one-array checks.
- [x] 1.2 Preserve canonical full-state parameters and select active states in the graph; verify reversed labels, invalid labels and all-fixed low-level construction without executing unused priors.
- [x] 1.3 Materialize related required model/prior inputs and auxiliary coordinates together; verify shared lazy execution counts, lazy scalar capture and unchanged borrowed objects.

## 2. Persistence and delivery

- [x] 2.1 Save construction-owned prior records and resolve required array references through existing trace persistence; verify real sample/save/reload, missing references, disabled lazy priors and full/prepared/saved standard parity.
- [x] 2.2 Update maintained documentation and the 872 bugfix fragment; pass focused and broader relevant pytest coverage, changed-path Ruff, OpenSpec validation and subagent correctness/simplicity reviews before pushing #873.

Validation completed: 66 standard handoff/prior checks and 344 RHIME/legacy
runner checks passed; targeted stage-mock rerun passed (55 checks). Changed-path
Ruff, strict OpenSpec validation and diff checks passed. Spec, correctness and
simplicity subagents found no remaining blockers. Earlier broader model coverage
had one dense-covariance PyTensor/Numba float32/float64 failure, reproduced on the
unchanged branch; this dependency failure remains outside the prior fix.
