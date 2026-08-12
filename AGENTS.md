# Repository Guidance

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

## Testing

The default local tox workflow tests against the current OpenGHG release and
runs Ruff:

```bash
tox -p --parallel-no-spinner
```

While iterating, use a focused tox environment or test path rather than the
default full set:

```bash
tox -e py310-openghgCur -- tests/test_array_ops.py
tox -e lint
```

Run the default full tox set exactly once when the branch is ready to push as a
draft PR. If that run fails, make and check fixes with focused commands first,
then rerun the full set once to verify the fixes. GitHub Actions runs current,
previous, and `devel` independently.

For final review or release-sensitive dependency changes, run the explicit
full compatibility matrix:

```bash
tox -p --parallel-no-spinner -e py310-openghgCur,py310-openghgPrev,py310-openghgDev,lint
```

This adds Python 3.10 test environments for the previous OpenGHG minor
release and OpenGHG `devel`. The previous-release environment defaults to
`openghg==0.18.0`; set `OPENGHG_PREV_SPEC` to test another deterministic
package spec. The tox test environments use `pytest-xdist` with
`--dist=worksteal`, matching the main CI pytest invocation. The `type` tox
environment is available for focused type-checking, but it is not part of
either tox set.

Install tox with the uv-backed runner:

```bash
uv tool install tox --with tox-uv
```

If a machine has limited cores, set `PYTEST_XDIST_WORKERS=1` or run tox
serially. Automated agents should run tox without allocating a PTY and use
`--parallel-no-spinner` for parallel runs.

The tox test environments do not require a C++ compiler. PyTensor can use its
Python implementations when no compiler is configured, so automated agents do
not need to load a compiler module before running tests.

For cluster test runs, prefer a writable node-local PyTensor cache such as
`base_compiledir=${TMPDIR:-/tmp}/pytensor-${USER}`. Add it as a
comma-separated `PYTENSOR_FLAGS` entry without discarding existing flags or
defining `base_compiledir` twice.
