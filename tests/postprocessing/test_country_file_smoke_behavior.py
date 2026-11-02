"""An explicitly requested compatibility smoke must report failures."""

import sys

import pytest

import test_country_file_smoke as smoke


@pytest.fixture
def enabled_smoke_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(smoke.COUNTRY_FILE_SMOKE_DIR_ENV, str(tmp_path))
    return smoke.country_file_smoke_dir.__wrapped__()


def test_country_file_smoke_skips_without_opt_in(monkeypatch):
    monkeypatch.delenv(smoke.COUNTRY_FILE_SMOKE_DIR_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=smoke.COUNTRY_FILE_SMOKE_DIR_ENV):
        smoke.country_file_smoke_dir.__wrapped__()


def test_enabled_country_file_smoke_requires_input_file(enabled_smoke_dir):
    with pytest.raises(pytest.fail.Exception, match="not a file"):
        smoke.test_xarray_open_country_file_engine_matrix(
            enabled_smoke_dir, smoke.COUNTRY_FILE_NAMES[0], None,
        )


@pytest.mark.parametrize("engine,module_name", [("h5netcdf", "h5netcdf"), ("netcdf4", "netCDF4")])
def test_enabled_country_file_smoke_requires_backend(enabled_smoke_dir, monkeypatch, engine, module_name):
    name = smoke.COUNTRY_FILE_NAMES[0]
    (enabled_smoke_dir / name).touch()
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(ModuleNotFoundError, match=module_name):
        smoke.test_xarray_open_country_file_engine_matrix(enabled_smoke_dir, name, engine)


@pytest.mark.parametrize("entry_point", ["xarray", "application"])
def test_enabled_country_file_smoke_propagates_read_failure(enabled_smoke_dir, monkeypatch, entry_point):
    name = smoke.COUNTRY_FILE_NAMES[0]
    (enabled_smoke_dir / name).touch()
    monkeypatch.delenv("PIXI_ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)

    def fail(*args, **kwargs):
        raise OSError("HDF5 read failed")

    monkeypatch.setattr(smoke.xr, "open_dataset", fail)
    monkeypatch.setattr(smoke, "load_country_dataset", fail)
    with pytest.raises(OSError, match="HDF5 read failed"):
        if entry_point == "xarray":
            smoke.test_xarray_open_country_file_engine_matrix(enabled_smoke_dir, name, None)
        else:
            smoke.test_load_country_dataset_real_files(enabled_smoke_dir, name)
