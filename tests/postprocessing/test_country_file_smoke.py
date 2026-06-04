"""Optional smoke tests for real country-file HDF5 backend compatibility."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import xarray as xr

from openghg_inversions._country_file import (
    COUNTRY_FILE_ENGINE_ORDER,
    COUNTRY_FILE_SELECTED_ENGINE_ATTR,
    load_country_dataset,
)

COUNTRY_FILE_SMOKE_DIR_ENV = "OPENGHG_COUNTRY_FILE_SMOKE_DIR"
COUNTRY_FILE_NAMES = (
    "country_EUROPE_EEZ_PARIS_gapfilled.nc",
    "country_EUROPE.nc",
)
ENGINES = (None, "h5netcdf", "netcdf4")

pytestmark = pytest.mark.country_file_smoke


def _engine_label(engine: str | None) -> str:
    return "default" if engine is None else engine


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _is_pixi_environment() -> bool:
    return bool(os.environ.get("PIXI_ENVIRONMENT_NAME") or os.environ.get("PIXI_PROJECT_ROOT"))


def _require_backend(engine: str | None) -> None:
    if engine == "h5netcdf":
        pytest.importorskip("h5netcdf")
    elif engine == "netcdf4":
        pytest.importorskip("netCDF4")


def _country_file_path(country_file_smoke_dir: Path, file_name: str) -> Path:
    path = country_file_smoke_dir / file_name
    if not path.exists():
        pytest.skip(f"Country-file smoke input {path} is not available")
    return path


def _summarise_dataset(dataset: xr.Dataset) -> str:
    data_vars = ", ".join(str(var) for var in dataset.data_vars) or "<no data variables>"
    sizes = ", ".join(f"{dim}={size}" for dim, size in dataset.sizes.items())
    return f"sizes=({sizes}); data_vars=({data_vars})"


@pytest.fixture(scope="module", autouse=True)
def print_country_file_dependency_versions() -> None:
    versions = {
        "xarray": _version("xarray"),
        "h5netcdf": _version("h5netcdf"),
        "h5py": _version("h5py"),
        "netCDF4": _version("netCDF4"),
    }
    version_report = ", ".join(
        f"{package}={package_version}" for package, package_version in versions.items()
    )
    print(f"country-file smoke dependency versions: {version_report}")


@pytest.fixture(scope="module")
def country_file_smoke_dir() -> Path:
    smoke_dir = os.environ.get(COUNTRY_FILE_SMOKE_DIR_ENV)
    if smoke_dir is None:
        pytest.skip(f"Set {COUNTRY_FILE_SMOKE_DIR_ENV} to run real country-file smoke tests")

    return Path(smoke_dir)


@pytest.mark.parametrize("file_name", COUNTRY_FILE_NAMES)
@pytest.mark.parametrize("engine", ENGINES, ids=_engine_label)
def test_xarray_open_country_file_engine_matrix(
    country_file_smoke_dir: Path,
    file_name: str,
    engine: str | None,
) -> None:
    """Open real country files with each xarray backend used during HDF5 debugging."""
    _require_backend(engine)

    path = _country_file_path(country_file_smoke_dir, file_name)
    engine_label = _engine_label(engine)
    kwargs = {} if engine is None else {"engine": engine}

    try:
        with xr.open_dataset(path, **kwargs) as dataset:
            loaded = dataset.load()
    except Exception as error:
        print(f"{file_name} engine={engine_label}: failed {type(error).__name__}: {error}")
        if not _is_pixi_environment():
            pytest.xfail(f"{file_name} engine={engine_label} failed outside Pixi: {error}")
        raise

    print(f"{file_name} engine={engine_label}: ok {_summarise_dataset(loaded)}")
    assert "country" in loaded.variables or "region" in loaded.variables


@pytest.mark.parametrize("file_name", COUNTRY_FILE_NAMES)
def test_load_country_dataset_real_files(
    country_file_smoke_dir: Path,
    file_name: str,
) -> None:
    """Exercise the shared country-file loader against real cluster inputs."""
    path = _country_file_path(country_file_smoke_dir, file_name)

    try:
        dataset = load_country_dataset(path)
    except Exception as error:
        print(f"{file_name} load_country_dataset: failed {type(error).__name__}: {error}")
        if not _is_pixi_environment():
            pytest.xfail(f"{file_name} load_country_dataset failed outside Pixi: {error}")
        raise

    selected_engine = dataset.attrs[COUNTRY_FILE_SELECTED_ENGINE_ATTR]
    print(
        f"{file_name} load_country_dataset: ok selected_engine={selected_engine}; {_summarise_dataset(dataset)}"
    )
    assert selected_engine in COUNTRY_FILE_ENGINE_ORDER
    assert "country" in dataset.variables or "region" in dataset.variables
