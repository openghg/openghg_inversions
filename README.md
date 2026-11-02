<img src="https://github.com/openghg/logo/raw/main/OpenGHG_Logo_Landscape.png" width="100">

# OpenGHG Inversions

OpenGHG Inversions is a Python package that is being developed as part of the [OpenGHG project](https://openghg.org) with the aim of merging the data-processing and simulation modelling capabilities of OpenGHG with the atmospheric Bayesian inverse models developed by the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol, UK.

Current regional inversion work uses RHIME: the standard, multisector, and
nested-domain recipes provide complete runners, while the advanced CO₂ family
provides prepared-input model-building and replay interfaces.
[Choose a RHIME model recipe](docs/usage/model_recipes.rst) from the supported
workflows. Historical HBMCMC execution and runtime request translation have
been removed; see the [migration guide](docs/usage/legacy_and_migration.rst).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10650595.svg)](https://doi.org/10.5281/zenodo.10650595)

## Installation

### Using Pixi (recommended for development)

OpenGHG Inversions reads and writes NetCDF/HDF5 data through OpenGHG,
`xarray`, `h5netcdf`, `h5py`, and `netcdf4`. If these packages
are installed from unrelated PyPI wheels, their bundled HDF5 libraries can
be incompatible. The Pixi environment in this repository installs the
compiled HDF5/NetCDF stack from conda-forge and installs
`openghg_inversions` in editable mode.

Install [Pixi](https://pixi.prefix.dev/latest/installation/), then run:

```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pixi install -e dev
pixi run -e dev python -c "import openghg_inversions, h5py, h5netcdf, netCDF4"
```

Useful development commands:

```bash
pixi run -e dev test
pixi run -e dev lint
pixi run -e dev typecheck
pixi run -e dev tox
pixi run -e dev docs-preview
```

The `tox` Pixi task runs the fast default tox set (current OpenGHG plus Ruff)
in parallel without an interactive spinner.

The `docs-preview` task incrementally builds the Sphinx documentation with
`tox -e docs`, serves it at `http://127.0.0.1:8765/`, and opens it in Safari on
macOS. Keep the command running while reading the docs and press Ctrl-C to stop
the server. Each invocation builds once before starting the static server, so
stop and rerun it after changing a source file. To use another port, avoid
opening Safari, or discard Sphinx's cached doctrees before building, run, for
example:

```bash
pixi run -e dev docs-preview --port 8766 --no-open
pixi run -e dev docs-preview --fresh
```

Preview output and cached doctrees remain in the ignored `docs/_build`
directory so later builds only rebuild changed pages. Regenerate the checked-in
API reference pages separately with `pixi run -e dev tox -e docs-api` after
changing the package layout. `uv run python scripts/preview_docs.py clean`
removes all preview output without rebuilding it.

The default uv group is intentionally limited to pytest and Ruff. To opt into
the larger development group for documentation work, use:

```bash
uv run --group uv_dev python scripts/preview_docs.py
uv run --group uv_dev python scripts/preview_docs.py --port 8766 --no-open
uv run --group uv_dev python scripts/preview_docs.py --fresh
uv run python scripts/preview_docs.py clean
```

To run the optional real country-file HDF5 smoke check on a machine that
can access the ACRG country files, set the country directory and run the
Pixi task:

```bash
OPENGHG_COUNTRY_FILE_SMOKE_DIR=/group/chem/acrg/LPDM/countries pixi run -e dev country-file-smoke
```

The smoke check opens `country_EUROPE_EEZ_PARIS_gapfilled.nc` and
`country_EUROPE.nc` with xarray's default backend, `h5netcdf`, and
`netcdf4`, then exercises `openghg_inversions._country_file.load_country_dataset`.
It prints the `xarray`, `h5netcdf`, `h5py`, and `netCDF4` versions and the
per-engine result. Without `OPENGHG_COUNTRY_FILE_SMOKE_DIR`, the real-file
tests are skipped so uv/pip CI does not need access to cluster data.

To test against a local OpenGHG checkout without replacing the Pixi-managed
HDF5/NetCDF dependencies, install only the local package code:

```bash
pixi run -e dev python -m pip install --no-deps -e ~/Documents/openghg
```

Avoid running plain `pip install -U h5py h5netcdf netcdf4` inside the Pixi
environment, as that can reintroduce incompatible wheels.

### Using pip

```bash
pip install openghg-inversions
```

### Using uv (faster alternative)

```bash
uv pip install openghg-inversions
```

Or with uv's project management:

```bash
# Add to your project
uv add openghg-inversions

# Or install in a virtual environment
uv venv
uv pip install openghg-inversions
```

### Development Installation

If you want to contribute or modify the package:

**With Pixi (recommended when working with NetCDF/HDF5 data):**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pixi install -e dev
```

**With uv:**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
uv sync
```

This creates the lean local environment used for focused tests and linting.
Jupyter, tox, Pyright, and Mypy are available only when explicitly requested
with `uv sync --group uv_dev`.

**With pip:**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pip install -e ".[dev]"
```

## Installation and Setup
As OpenGHG Inversions is dependent on OpenGHG, please ensure that when running locally you are using Python 3.12 or 3.13 on Linux or MacOS. Please see the [OpenGHG project](https://github.com/openghg/openghg/) for further installation instructions of OpenGHG and setting up an object store.

### Setup a virtual environment

Check that you have Python 3.12 or 3.13:
```bash
python --version
```
(Note for Bristol ACRG group: If you are on Blue Pebble, the default anaconda module `lang/python/anaconda` is Python 3.9. Use `module avail` to list other options; `lang/python/miniconda/3.12.2.inc-perl-5.30.0` will work.)

Make a virtual environment
```bash
python -m venv openghg_inv
```

Next activate the environment
```bash
source openghg_inv/bin/activate
```

### Installation using `pip`

First you'll need to clone the repository

```bash
git clone https://github.com/openghg/openghg_inversions.git
```

Next make sure `pip` and related install tools are up to date and then install OpenGHG Inversions using the editable install flag (`-e`)

```bash
pip install --upgrade pip setuptools wheel
pip install -e openghg_inversions
```

Optionally, install the developer requirements (there is more information about this in the "Contributing" section below):
``` bash
pip install -r requirements-dev.txt
```

### Verify that PyMC is using fast linear algebra libraries
At this point, run

``` bash
python -c "import pymc"
```
This should run without printing any messages.
If you receive a message about `pymc` or `pytensor` using the `numpy` C-API, then your inversions might run slowly because the fast linear algebra libraries used by `numpy` haven't been found.

Solutions to this are:
1. Use the Pixi development environment above, which installs `numpy` and the NetCDF/HDF5 stack from conda-forge.
2. Try `python -m pip install numpy` after upgrading `pip, setuptools, wheel`.
3. Create a `conda` env, install `numpy` using `conda`, then use `pip` to upgrade `pip, setuptools, wheel` and install `openghg_inversions`.


## Using OpenGHG Inversions

### Getting Started

For current workflows, start with the
[RHIME model recipe guide](docs/usage/model_recipes.rst). For a broader
overview, see the [getting-started guide](docs/usage/getting_started.rst).

### Modern RHIME entry points

New RHIME runs can be launched without calling an internal source file path:

```python
from openghg_inversions.rhime import run_standard, run_multisector, run_nested

result = run_standard(
    species="ch4",
    sites=["TAC"],
    averaging_period=["1h"],
    domain="EUROPE",
    start_date="2019-01-01",
    end_date="2019-01-02",
    output_path="outputs",
    output_name="example",
    flux_sources=["total-ukghg-edgar7"],
    mismatch_model="pollution_event",
)
```

Nested outer/inner grids use the explicit `run_nested` entry point.
It masks the outer response over the inner extent, retains separate native
bases, and samples separate `x_outer` and `x_inner` state blocks. See the
[nested-domain model family](docs/usage/nested_domain_model_family.rst).
Nested runs support `output_format="none"` for an in-memory result and
`output_format="paris"` for separate outer- and inner-grid flux products plus
one shared concentration product; single-grid output formats are rejected.

For SLURM batch scripts and installed environments, use the console entry point:

```bash
rhime run 2019-01-01 2019-01-02 -c rhime.ini --output-path outputs
rhime run --model multisector 2019-01-01 2019-01-02 -c rhime_multisector.ini
rhime run --model nested 2019-01-01 2019-02-01 -c rhime_nested.ini
```

The new RHIME config template is available at
`openghg_inversions/config/templates/rhime_template.ini`. New configs should use
`flux_sources`; the legacy `emissions_name` spelling is unsupported.
See the [RHIME terminology and quickstart](docs/usage/rhime.rst) page for the
canonical config vocabulary.

RHIME terminology:

- `species`: primary gas or tracer name used for object-store lookup and output naming.
- `source`: OpenGHG metadata key used to retrieve flux data.
- `flux_sources`: RHIME field containing requested OpenGHG flux `source` values.
- `sector_sources`: optional one-to-one mapping from RHIME sector names to unique OpenGHG flux `source` values.
- `sector_priors`: optional complete mapping from RHIME sector names to flux-scaling priors; omit it to use a shared `x_prior`.
- `sector`: model component optimized separately, currently backed by one unique flux `source`.
- `tracer`: additional species used to constrain the primary species through linked forward models.
- `emissions_name`: legacy compatibility spelling only; use `flux_sources` in new RHIME configs.

### Migrating old HBMCMC workflows

The direct `fixedbasisMCMC` and `inferpymc` implementation has been removed in
0.8. The 0.7.x release line is the last line containing it and the
`--legacy-fixedbasis` option.

The complete `openghg_inversions.hbmcmc` namespace, old dictionary/cache
adapters and runtime spelling translation are removed. Use canonical settings
with `run_standard` or `rhime run`; preserve a matching 0.7 environment for
historical execution. Modern `output_format="legacy"` and `InversionOutput`
exports remain independently supported.

See the [HBMCMC-to-RHIME migration guide](docs/usage/legacy_and_migration.rst)
for parameter mappings, return-type changes, batch-script updates, and removed
interfaces.

### Results

Standard and multisector runners return an `InversionOutput` with the annotated
trace, observations and retained basis. Nested execution returns two native
`InversionOutput` views in a `NestedOutput`. Ordinary and cached CO₂ outputs use
`Co2Output`, which borrows the preparation and trace and retains calculated
products. See the
[RHIME guide](docs/usage/rhime.rst) for output formats and persistence.



## Contributing

### Code quality tools

To contribute to `openghg_inversions`, you should also install the developer packages:
```bash
pip install -r requirements-dev.txt
```
This will install the packages `pytest`, `pytest-xdist`, `ruff`, `tox`, and `tox-uv`.

We use `ruff` to lint our code. To check for lint issues, run:
``` bash
ruff check openghg_inversions
```
in your `openghg_inversions` repository (with your virtual env activated).

To fix issues that Ruff can safely update, run:
``` bash
ruff check --fix openghg_inversions
```

You can run the tests using:
``` bash
pytest
```
in the `openghg_inversions` repository. (Make sure your virtual env is activated.)

### Using `tox` to check code

Alternatively, use `tox` to run tests and check the code format.
`tox` creates isolated environments to run the tests, which means it can test against different
versions of OpenGHG.
It does this automatically, so you don't need to manage pip or conda virtual environments to do this.

To install `tox` globally in a "safe" way, use:

```bash
uv tool install tox --with tox-uv
```
or, within a virtual environment, install `tox` and `tox-uv`.

The default checks the locked OpenGHG release, borrowed-reference typing, and Ruff:

```bash
tox -p --parallel-no-spinner
```

This is the required check before pushing a draft pull request. GitHub Actions
tests the committed `uv.lock` against the latest OpenGHG release and tests the
OpenGHG devel branch separately. It fails if the lockfile lags a new release.

The normal CI suite always uses PyTensor's Python linker to avoid repeated JIT
compilation. Tests marked `compiled_backend` also form a small Numba compatibility
suite covering likelihood values and gradients, predictive draws, cached sampling
and spawned chains. They remain part of the normal Python run, with the same
assertions. A separate serial pytest process reruns only that marked subset with
Numba for changes to production code, backend-sensitive tests, dependencies or CI
configuration, and for pushes. The
[backend selector](.github/scripts/select-test-linker.cjs) owns those path rules;
incomplete or failed change detection enables the additional compiled checks.

For explicit marked Numba checks from Actions, use:

```bash
gh workflow run workflow.yaml --ref <branch> -f compiled_backend=true
```

To select the same checks locally, start a fresh process:

```bash
PYTENSOR_FLAGS=linker=numba DASK_NUM_WORKERS=2 uv run pytest \
  -m 'compiled_backend and not slow and not create_frozen'
```

Local pytest, tox and production defaults are unchanged. To opt into Python
linking for a focused local check with no existing PyTensor flags, start a new process with
`PYTENSOR_FLAGS=linker=py DASK_NUM_WORKERS=2 uv run pytest <test-path>`.
If `PYTENSOR_FLAGS` is already set, preserve its other entries and add or replace
only `linker`. The marked suite checks our integration with PyTensor's Numba
lowering, including intentional Python fallbacks for custom Ops and the cached
amplitude sampler's private CVM graph. For a broader check, explicitly run the
whole suite with `PYTENSOR_FLAGS=linker=numba` and omit `-m compiled_backend`.
The marked subset does not establish full float32 predictive accuracy; the known
upstream float32 multivariate-normal issue remains tracked separately in
[#770](https://github.com/openghg/openghg_inversions/issues/770).
The suite can still invoke other compiled backends such as NumPyro/JAX explicitly.

On a Slurm cluster, submit tox from the repository root instead of creating its
environments on a shared worktree filesystem:

```bash
sbatch scripts/slurm_tox.sh
sbatch scripts/slurm_tox.sh -e type
```

The Slurm runner creates `TOX_WORK_DIR` on node-local storage and removes it on
exit. It continues to use the shared uv cache for downloaded artifacts; files
still have to be installed into each isolated tox environment, but those
node-local copies are temporary. Tox environments and pytest run serially by
default, with two Dask workers and one numerical thread per process. The runner
preserves an explicit `DASK_NUM_WORKERS` setting through tox. Budget Dask workers
together with pytest workers and parallel tox environments when overriding
these defaults; a worker limit is not a total job memory limit.

Direct pytest and GitHub Actions bypass the Slurm runner. To bound Dask during
a direct real-data test run, set `DASK_NUM_WORKERS=2 uv run pytest <test-path>`.
GitHub's scientific jobs use two pytest workers, two Dask workers per process
and set OpenBLAS, OpenMP, MKL, NumExpr and BLIS thread limits to one. These
per-process limits are not a total job memory bound or a limit on JAX/XLA pools.

An optional serial CI probe records CPU model, affinity/quota, effective loaded
thread pools and PyTensor cache settings from the actual pytest process.
Cache-lock acquisition time is an upper bound on waiting, not a measurement of
compilation time or proof of contention. Installed CLI and sampling subprocesses
are outside these reports. Diagnostics do not import PyTensor before test
collection or change scientific settings. To run it in the locked Python
3.12 and 3.13 GitHub environments:

```bash
gh workflow run workflow.yaml --ref <branch> -f runtime_comparison=true
```

This profiles the full suite serially with two Dask workers and one numerical
thread and selects the default compiled backend, recording elapsed/CPU time and
peak RSS. Four focused runs then compare
one/two numerical threads in reverse order, also with xdist disabled and two Dask
workers per process.
Every run has fresh PyTensor/Numba caches. The existing float64 scientific probes
retain their backend, precision and assertions. Logs, JUnit, process diagnostics,
wall time and aggregate child CPU time are uploaded as `test-runtime-*` artifacts.
This small sample does not establish the best settings for the complete suite or contention
between pytest workers. To run locally, choose a fresh output directory:

```bash
uv run --group test python scripts/compare_test_runtime.py /tmp/ogi-runtime-comparison
```

Use Slurm for cluster comparisons.

Scientific tox environments check the resolved Python's development headers
and PyTensor's configured C++ compiler before pytest. The Slurm runner prefers
uv-managed Python installations, which include those headers. If PyTensor's
compiler was explicitly disabled with `cxx=`, the compilation check is skipped.

On a cluster compute node, a writable node-local PyTensor compilation cache
also avoids shared-filesystem contention. Preserve any existing
comma-separated `PYTENSOR_FLAGS` entries when adding it:

```bash
PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=${TMPDIR:-/tmp}/pytensor-${USER}" \
  tox -e py312-openghgCur
```

If `PYTENSOR_FLAGS` already defines `base_compiledir`, update that entry
instead of adding the same key twice.

For final review or release-sensitive dependency changes, also test against
OpenGHG devel:

```bash
tox -p --parallel-no-spinner -e py312-openghgCur,py312-openghgDev,borrowed-types,lint
```

To specify individual jobs, you can use, e.g.:

```bash
tox -e py312-openghgDev
```

to run the tests against the devel branch.

Use `tox -l` to list all options.

To pass arguments to pytest, Ruff, mypy, etc, you can use, e.g.

```bash
tox -- "tests/test_hbmcmc_boundary.py"
```

which will pass the test path as a positional argument to the commands invoked by tox.

### Using branches

Published PyPI packages and tagged releases are the supported installation
targets. The `devel` branch is unsupported integration for the next monthly
release; users should not run scientific work from it.

Contributors create feature branches from `devel` and merge only release-ready
changes through reviewed pull requests. Active feature PRs are checked weekly
for drift from `devel`; same-repository PRs can opt into clean automatic updates
with the `auto-sync-devel` label. Current-line hotfixes start from `main`, are
released as patch versions, and are then forwarded to `devel`.

See the [release and branch maintenance guide](docs/development/releasing.rst)
for the automated monthly release, hotfix, synchronization, and stale-PR
workflows.

## Citation and contributors

If you use this software, please cite the version-specific Zenodo DOI for the
release you used.

The recommended prose description is:

> We use RHIME, the Regional Hierarchical Inversion Modelling Environment,
> implemented in the `openghg_inversions` Python package.

The formal software citation lists the principal creators of the citable
software artifact. Additional code, testing, documentation, scientific, and
project contributions are recorded in the Zenodo metadata and GitHub history.

## References
Ganesan et al. (2014),_ACP_;

Western et al. (2021), _Enviro. Sci. Tech Lett._
