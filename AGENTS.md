# Repository Guidance

## RHIME architecture and scientific model development

New RHIME work must follow the simplicity and locality rules in
`docs/development/rhime_model_development.rst`. In particular, keep model
runners procedural, keep concrete model construction readable in scientific
order, use ordinary callable components, forward resolved values explicitly,
and accept small duplication when it keeps a model recipe understandable.

The active delivery roadmap is
`docs/plans/run_rhime_readability_and_modifiability.md`. Near-term 6 km nested-
domain, CO2-family, and verification-games feature landing is recorded in
`docs/plans/rhime_model_family_expansion.md`. These documents supersede the
semantic-compiler plans as production architecture.

## Numerical array ownership and execution

Treat xarray objects as borrowed and potentially Dask-backed. Do not mutate
inputs in place or hide copying, computation, persistence, densification, or
rechunking behind properties.

Indexed dimension coordinates are normally eager and may be validated directly;
auxiliary coordinates may remain lazy. Preserve shared Dask graphs and
materialize related arrays together at a named PyMC, serialization, or
eager-kernel boundary. Use `openghg_inversions.array_ops.to_dense` when dense
chunk payloads are required; for sparse Dask inputs, this preserves outer Dask
laziness.

See `docs/plans/numerical_data_ownership_and_execution_boundaries.md` for the
full rationale, terminology, and review checklist.

Consolidate validation at the boundary which owns an input, then trust locally
constructed intermediates. Prefer ordinary xarray transpose/alignment patterns
and Pint unit conversion to repeated custom runtime assertions; see
`docs/development/validation_and_xarray.rst`.

## Testing

Keep the worktree-local `uv` environment lean. A plain `uv sync` installs the
runtime dependencies plus pytest and Ruff; Jupyter, tox, Pyright, and Mypy are
intentionally excluded from the default dependency group.

Run relevant tests and lint only the changed Python paths while iterating:

```bash
uv run pytest tests/test_array_ops.py
uv run ruff check path/to/changed_file.py tests/path/to/changed_test.py
git diff --check
```

Use focused test paths while iterating and run the relevant broader pytest
coverage before handing off a change. We still support Python 3.10, so avoid
syntax, typing, and dependency features that require newer Python versions.

Do not run tox locally in a Codex-managed worktree. Submit compatibility,
full-suite, and type-check environments to Slurm with
`sbatch scripts/slurm_tox.sh`; pass tox arguments after the script name when a
subset is sufficient (for example, `sbatch scripts/slurm_tox.sh -e type`). The
runner builds tox environments on node-local storage and removes them when the
job exits.

## Release notes

Use Towncrier fragments for user-visible changes. Agents must add a concise
`newsfragments/<issue>.<type>.md` file rather than editing `CHANGELOG.md`
directly; use `+` in place of an issue number when there is no tracked issue.
Choose one of `feature`, `bugfix`, `doc`, `removal`, or `misc` for `<type>`.
The existing `CHANGELOG.md` remains the published, human-readable changelog
for users and developers. During release preparation, a maintainer runs
`towncrier build --version <version> --yes` to prepend the collected notes, commits
the generated changelog, and then creates the GitHub release. The PyPI release
workflow verifies that no unassembled fragments remain.

For cluster test runs, prefer a writable node-local PyTensor cache such as
`base_compiledir=${TMPDIR:-/tmp}/pytensor-${USER}`. Add it as a
comma-separated `PYTENSOR_FLAGS` entry without discarding existing flags or
defining `base_compiledir` twice.
Before asking the user to archive a chat, offer to run
`scripts/clean_local_envs.sh`. Run it only after the user agrees because it
deletes the current worktree's `.venv` and `.tox` directories.
