# Repository Guidance

## Work Tracking

GitHub Issues track work that needs project discussion. Linear may also be used
for implementation tasks, especially personal or subdivided work.

- For Linear issue IDs such as `OPE-17`, use the Linear MCP integration.

## RHIME architecture and scientific model development

New RHIME work must follow the simplicity and locality rules in
`docs/development/rhime_model_development.rst`. In particular, keep model
runners procedural, keep concrete model construction readable in scientific
order, use ordinary callable components, forward resolved values explicitly,
and accept small duplication when it keeps a model recipe understandable.

Use `docs/development/specifications.rst` to find the behavioral contracts,
their delivery status, and the distinction between development principles and
API requirements. Plans preserve dated proposals, research, and delivery
history; they do not define current production architecture or feature support.
See `docs/plans/README.md` for dispositions and owning issues.

## Specification changes

Only change specifications under `openspec/` after conferring with a developer
and agreeing on the proposed changes. Keep specifications focused on behavioral
contracts and tolerant of external changes, including changes to dependencies
and surrounding systems. Do not use them to record every implementation choice
or incidental detail.

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

See `docs/development/numerical_array_ownership.rst` for the maintained
explanation of borrowing, sparse storage, and execution boundaries.

Consolidate validation at the boundary which owns an input, then trust locally
constructed intermediates. Prefer ordinary xarray transpose/alignment patterns
and Pint unit conversion to repeated custom runtime assertions; see
`docs/development/validation_and_xarray.rst`.

## Testing

Keep the worktree-local `uv` environment lean. A plain `uv sync` installs the
runtime dependencies plus pytest and Ruff; Jupyter, tox, Pyright, and Mypy are
intentionally excluded from the default dependency group.

Bounded test runs, linting and type checking may run directly on the login node.
Use focused paths while iterating:

```bash
uv run pytest tests/test_array_ops.py
uv run ruff check path/to/changed_file.py tests/path/to/changed_test.py
uv run --extra dev mypy path/to/changed_file.py
git diff --check
```

The type-check command explicitly opts into the developer tools; ordinary test
and lint runs can keep the default environment lean. If the required tools are
already installed, invoke them directly from that environment.

Use focused test paths while iterating and run the relevant broader pytest
coverage before handing off a change. We support Python 3.12 and 3.13.

Use tox for final CI-equivalent validation, rather than routine iteration.
On the cluster, tox invocations that create or recreate environments must run
on Slurm; this restriction does not apply to direct bounded checks in an
existing environment. Submit the required CI environments with
`sbatch scripts/slurm_tox.sh`, passing tox arguments after the script name
(for example, `sbatch scripts/slurm_tox.sh -e py312-openghgCur,py313-openghgCur`).
The runner builds tox environments on node-local storage and removes them when
the job exits. It runs tox environments and pytest serially by default, with
one numerical thread per process. Explicit tox `-p` and `PYTEST_XDIST_WORKERS`
overrides enable concurrency; budget their combined memory within the Slurm
allocation. Each tox environment gets its own temporary directory.
Real-data fixtures use a fresh store per pytest invocation, shared by that run's
local workers, with separate ground and satellite stores.

The runner prefers uv-managed Python installations, which include development
headers. Set `UV_PYTHON_PREFERENCE` explicitly to override this choice. Scientific
tox environments report their resolved interpreter and check its headers and
configured C++ compiler before pytest. The runner preserves scientific
`PYTENSOR_FLAGS` settings while replacing `base_compiledir` with a node-local
cache; tox forwards those flags and the selected numerical thread limits.

## Release notes

Use Towncrier fragments for user-visible changes. Agents must add a concise
`newsfragments/<issue>.<type>.md` file rather than editing `CHANGELOG.md`
directly. When there is no tracked issue, run
`uv run --extra dev towncrier create +.bugfix.md --no-edit` (substitute the
fragment type) and replace the placeholder text in the uniquely named
`+<random>.<type>.md` file. Never commit a bare `+.<type>.md` filename, which
conflicts when independent branches add a fragment of the same type.
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
