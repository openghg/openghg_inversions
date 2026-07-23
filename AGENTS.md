# Repository Guidance

## Testing

The default local tox workflow is intentionally fast: it tests against the
current OpenGHG release and runs Ruff. Run it in parallel with:

```bash
tox -p --parallel-no-spinner
```

This fast default is the required local check before pushing a draft PR.
GitHub Actions runs current, previous, and `devel` independently.

For final review or release-sensitive dependency changes, run the explicit
full compatibility matrix:

```bash
tox -p --parallel-no-spinner -e py310-openghgCur,py310-openghgPrev,py310-openghgDev,lint
```

This adds Python 3.10 test environments for the previous OpenGHG minor
release and OpenGHG `devel`. The previous-release environment defaults to
`openghg==0.18.0`; set `OPENGHG_PREV_SPEC` to test another deterministic
package spec. The tox test environments use `pytest-xdist` with
`--dist=loadscope`, matching the main CI pytest invocation. The `type` tox
environment is available for focused type-checking, but it is not part of
either tox set.

Install tox with the uv-backed runner:

```bash
uv tool install tox --with tox-uv
```

While iterating, target one environment or one test path before running the full set:

```bash
tox -e py310-openghgCur -- tests/test_array_ops.py
```

If a machine has limited cores, set `PYTEST_XDIST_WORKERS=1` or run tox
serially. Automated agents should run tox without allocating a PTY and use
`--parallel-no-spinner` for parallel runs.
