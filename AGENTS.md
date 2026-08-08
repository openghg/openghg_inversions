# Repository Guidance

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

The tox test environments fail before pytest when the effective
`pytensor.config.cxx` is empty. On Rocky Linux and Blue Pebble, the launcher
first tries the `gcc/12.3.0-sknc` module in the same shell that starts pytest.
The default `PYTENSOR_COMPILER_BOOTSTRAP=auto` mode targets Rocky Linux or
recognized Blue Pebble hostnames. Use `PYTENSOR_COMPILER_MODULE` and
`PYTENSOR_MODULE_INIT` (default `/etc/profile.d/modules.sh`) for different
module setups, `PYTENSOR_COMPILER_BOOTSTRAP=always` to enable module loading on
other hosts, or `PYTENSOR_FLAGS='cxx=/path/to/g++'` to select a compiler
directly. Setting `PYTENSOR_COMPILER_BOOTSTRAP=never` disables module loading
but keeps the fail-fast compiler check.

For cluster test runs, prefer a writable node-local PyTensor cache such as
`base_compiledir=${TMPDIR:-/tmp}/pytensor-${USER}`. Add it as a
comma-separated `PYTENSOR_FLAGS` entry without discarding existing flags or
defining `base_compiledir` twice.
