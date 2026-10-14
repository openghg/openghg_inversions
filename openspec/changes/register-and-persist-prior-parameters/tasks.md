# Tasks

## 1. Shared prior construction

- [x] 1.1 Register labelled prior arrays and capture inline scalars and symbolic associations at construction; verify ordinary/reparameterized lognormal density, scalar-only and length-one-array checks.
- [x] 1.2 Preserve canonical full-state parameters and select active states in the graph; verify reversed labels, invalid labels and all-fixed inputs without executing unused priors.
- [ ] 1.3 Materialize related required model/prior inputs and auxiliary coordinates together; verify shared lazy execution counts, lazy scalar capture and unchanged borrowed objects.

## 2. Persistence and delivery

- [x] 2.1 Save construction-owned prior records and resolve required array references through existing trace persistence; verify real sample/save/reload, missing references, disabled lazy priors and full/prepared/saved standard parity.
- [ ] 2.2 Update maintained documentation and the 872 bugfix fragment; pass focused and broader relevant pytest coverage, changed-path Ruff, OpenSpec validation and subagent correctness/simplicity reviews before pushing #873.
