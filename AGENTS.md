# Repository Guidance

## Testing

The local tox workflow mirrors the GitHub Actions test matrix closely enough to use as the pre-PR check. Before pushing a PR, run the full tox set:

```bash
tox -p
```

This runs the Python 3.10 test environments against the current OpenGHG release, the previous OpenGHG minor release, and OpenGHG `devel`, plus the Ruff lint checks. The tox test environments use `pytest-xdist` with `--dist=loadscope`, matching the main CI pytest invocation. The `type` tox environment is available for focused type-checking, but it is not part of the default pre-PR tox set.

Install tox with the uv-backed runner:

```bash
uv tool install tox --with tox-uv
```

While iterating, target one environment or one test path before running the full set:

```bash
tox -e py310-openghgCur -- tests/test_array_ops.py
```

If a machine has limited cores, set `PYTEST_XDIST_WORKERS=1` or run tox without `-p`, but still run the full tox set before asking for review.
