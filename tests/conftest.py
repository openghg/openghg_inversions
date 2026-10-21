from collections import namedtuple
from pathlib import Path
import shutil
from collections.abc import Callable, Iterator
from unittest.mock import patch

from filelock import FileLock
import numpy as np
import pytest
from openghg.standardise import (
    standardise_surface,
    standardise_bc,
    standardise_flux,
    standardise_footprint,
    standardise_column,
)
import xarray as xr
import zarr

from openghg_inversions.basis.basis_functions import BASIS_ARTIFACT_SOURCE_ATTR, BasisFunctions
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from tests.helpers import make_trace

_raw_data_path = Path(__file__).resolve().parent / "data"
_TEST_STORE_DIR_NAME = "openghg_inversions_testing_store"
_SATELLITE_TEST_STORE_DIR_NAME = "openghg_inversions_satellite_testing_store"


@pytest.fixture(scope="session")
def user_data_path(tmp_path_factory, request):
    """Share read-only input stores between local workers in this pytest run."""
    run_path = tmp_path_factory.getbasetemp()
    if hasattr(request.config, "workerinput"):
        run_path = run_path.parent
    data_path = run_path / "openghg_inversions_test_data"
    data_path.mkdir(exist_ok=True)
    return data_path


@pytest.fixture(scope="session")
def writable_data_path(tmp_path_factory):
    """Per-run root for test artifacts that are written during pytest."""
    return tmp_path_factory.mktemp("openghg_inversions_test_data")


@pytest.fixture(scope="session")
def raw_data_path():
    return _raw_data_path


@pytest.fixture(scope="session")
def merged_data_dir(writable_data_path, merged_data_file_name):
    merged_data_dir = writable_data_path / "openghg_inversions_testing_merged_data_dir"
    merged_data_dir.mkdir(exist_ok=True)

    if not (merged_data_dir / (merged_data_file_name + ".zarr.zip")).exists():
        with xr.open_dataset(_raw_data_path / (merged_data_file_name + ".nc")) as ds:
            with zarr.ZipStore(merged_data_dir / (merged_data_file_name + ".zarr.zip"), mode="w") as store:
                ds.to_zarr(store)

            ds.to_zarr(merged_data_dir / (merged_data_file_name + "no_zip" + ".zarr"))

    return merged_data_dir


@pytest.fixture(scope="session")
def merged_data_file_name():
    return "merged_data_test_tac_combined_scenario_v14"


@pytest.fixture
def europe_country_file(raw_data_path):
    """Provides path to the EUROPE countryfile"""
    return raw_data_path / "country_EUROPE.nc"


@pytest.fixture
def tiny_paris_country_file(tmp_path: Path) -> Path:
    """Keep all PARIS country labels on a small grid with unequal cell areas."""
    countries = xr.Dataset(
        {
            "country": (
                ("lat", "lon"),
                [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 10],
                ],
            ),
            "name": (
                "ncountries",
                [
                    "OCEAN", "SWE", "SVN", "SVK", "PRT", "POL", "NOR", "NLD",
                    "LUX", "ITA", "IRL", "HUN", "HRV", "GBR", "FRA", "FIN",
                    "ESP", "DNK", "DEU", "CZE", "CHE", "BEL", "AUT",
                ],
            ),
        },
        coords={"lat": [48.0, 49.0, 50.0, 51.0], "lon": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]},
    )
    path = tmp_path / "country_TINY_PARIS.nc"
    countries.to_netcdf(path)
    return path


@pytest.fixture
def fake_multisector_basis_functions() -> Callable[..., BasisFunctions]:
    """Provide a builder for a one-cell basis artifact with sector prior fluxes."""

    def _build(*, artifact_source: str = "generated") -> BasisFunctions:
        """Build a one-cell basis artifact with source-specific prior flux."""
        basis = xr.DataArray(
            [[1]],
            dims=("lat", "lon"),
            coords={"lat": [0.0], "lon": [0.0]},
            name="basis",
        )
        flux = xr.DataArray(
            [[[1.0]], [[3.0]]],
            dims=("source", "lat", "lon"),
            coords={
                "source": ["ff-inventory", "ocean-inventory"],
                "lat": [0.0],
                "lon": [0.0],
            },
            name="flux",
        )
        flux.attrs["units"] = "mol/m2/s"
        return BasisFunctions.from_flat_basis(
            basis_flat=basis,
            flux=flux,
            operator_kwargs={"state_dim": "region"},
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: artifact_source},
        )

    return _build


@pytest.fixture
def fake_multisector_basis_functions_matching_country_grid() -> Callable[..., BasisFunctions]:
    """Provide a builder for a multisector basis artifact on a country grid."""

    def _build(country_file: Path, *, coord_offset: float = 0.0) -> BasisFunctions:
        """Build a one-region multisector basis artifact on a test country grid."""
        with xr.open_dataset(country_file) as country_grid:
            lat = country_grid.lat.values + coord_offset
            lon = country_grid.lon.values + coord_offset
            basis = xr.DataArray(
                np.ones((country_grid.sizes["lat"], country_grid.sizes["lon"]), dtype=int),
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                name="basis",
            )
        flux = xr.concat(
            [
                xr.ones_like(basis, dtype=float).expand_dims(source=["ff-inventory"]),
                (3.0 * xr.ones_like(basis, dtype=float)).expand_dims(source=["ocean-inventory"]),
            ],
            dim="source",
        ).rename("flux")
        flux.attrs["units"] = "mol/m2/s"
        return BasisFunctions.from_flat_basis(
            basis_flat=basis,
            flux=flux,
            operator_kwargs={"state_dim": "region"},
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "country-grid-multisector-test"},
        )

    return _build


@pytest.fixture
def fake_source_specific_multisector_basis_functions() -> Callable[..., BasisFunctions]:
    """Provide a builder for source-specific multisector basis artifacts."""

    def _build(*, artifact_source: str = "generated") -> BasisFunctions:
        """Build a one-cell source-specific basis artifact for multisector tests."""
        basis = xr.DataArray(
            [[1]],
            dims=("lat", "lon"),
            coords={"lat": [0.0], "lon": [0.0]},
            name="basis",
        )
        flux = xr.DataArray(
            [[[1.0]], [[3.0]]],
            dims=("source", "lat", "lon"),
            coords={
                "source": ["ff-inventory", "ocean-inventory"],
                "lat": [0.0],
                "lon": [0.0],
            },
            name="flux",
        )
        flux.attrs["units"] = "mol/m2/s"
        return BasisFunctions.from_multi_source_flat_basis(
            basis_flat={"ff-inventory": basis, "ocean-inventory": basis},
            flux=flux,
            operator_kwargs={"state_dim": "region"},
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: artifact_source},
        )

    return _build


@pytest.fixture
def multisector_postprocessing_inv_out(
    fake_multisector_basis_functions: Callable[..., BasisFunctions],
) -> Callable[[BasisFunctions | None], InversionOutput]:
    """Provide a builder for small multisector postprocessing outputs."""

    def _build(basis_functions: BasisFunctions | None = None) -> InversionOutput:
        """Build a small multisector output for flux reconstruction tests."""
        inv_inputs = xr.Dataset(
            {
                "H": (("region", "nmeasure"), [[1.0]]),
                "mf": ("nmeasure", [10.0]),
                "mf_error": ("nmeasure", [1.0]),
                "mf_repeatability": ("nmeasure", [0.5]),
                "mf_variability": ("nmeasure", [0.25]),
                "site_indicator": ("nmeasure", [0]),
            },
            coords={
                "region": [0],
                "nmeasure": [0],
                "site": ("nmeasure", ["TAC"]),
                "time": (
                    "nmeasure",
                    np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]"),
                ),
            },
        )
        inv_inputs["mf"].attrs["units"] = "ppm"

        return InversionOutput(
            trace=make_trace(
                posterior=xr.Dataset(
                    {
                        "x_ff": (("chain", "draw", "region"), np.array([[[0.0], [2.0]]])),
                        "x_ocean": (
                            ("chain", "draw", "region"),
                            np.array([[[2.0 / 3.0], [0.0]]]),
                        ),
                    },
                    coords={"chain": [0], "draw": [0, 1], "region": [0]},
                ),
                prior=xr.Dataset(
                    {
                        "x_ff": (("chain", "draw", "region"), np.ones((1, 2, 1))),
                        "x_ocean": (("chain", "draw", "region"), np.ones((1, 2, 1))),
                    },
                    coords={"chain": [0], "draw": [0, 1], "region": [0]},
                ),
            ),
            inv_inputs=inv_inputs.set_index(nmeasure=["site", "time"]),
            basis_functions=basis_functions or fake_multisector_basis_functions(),
            run_metadata={
                "start_date": "2019-01-01",
                "end_date": "2019-01-02",
                "sites": ["TAC"],
                "split_by_sectors": True,
            },
            model_metadata={
                "species": "ch4",
                "domain": "EUROPE",
                "sectors": [
                    {
                        "name": "FF",
                        "flux_source": "ff-inventory",
                        "variable_suffix": "ff",
                    },
                    {
                        "name": "Ocean",
                        "flux_source": "ocean-inventory",
                        "variable_suffix": "ocean",
                    },
                ],
            },
        )

    return _build


@pytest.fixture
def eastasia_country_file(raw_data_path):
    """Provides path to the EASTASIA countryfile"""
    return raw_data_path / "country_EASTASIA.nc"


@pytest.fixture
def country_ds(raw_data_path):
    """Provides EUROPE countryfile dataset"""
    ds = xr.load_dataset(raw_data_path / "country_EUROPE.nc")
    yield ds


@pytest.fixture
def country_ds_eastasia(raw_data_path):
    """Provides EUROPE countryfile dataset"""
    ds = xr.load_dataset(raw_data_path / "country_EASTASIA.nc")
    yield ds


@pytest.fixture(scope="session")
def session_config_mocker(user_data_path) -> Iterator[dict]:
    mock_config = {
        "object_store": {
            "inversions_tests": {
                "path": str(user_data_path / _TEST_STORE_DIR_NAME),
                "permissions": "r",
            },
            "inversions_satellite_tests": {
                "path": str(user_data_path / _SATELLITE_TEST_STORE_DIR_NAME),
                "permissions": "r",
            },
        },
        "user_id": "test-id-123",
        "config_version": "2",
    }

    with patch("openghg.objectstore._local_store.read_local_config", return_value=mock_config):
        yield mock_config


# TEST DATA
TestData = namedtuple("TestData", ["func", "metadata", "path", "data_type"])
test_data_list = []
satellite_test_data_list = []

## Obs data
tac_obs_metadata = {
    "source_format": "openghg",
    "network": "decc",
    "site": "tac",
    "inlet": "185m",
    "instrument": "picarro",
}
tac_obs_data_path = _raw_data_path / "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc"
test_data_list.append(TestData(standardise_surface, tac_obs_metadata, tac_obs_data_path, "surface"))

mhd_obs_metadata = {
    "source_format": "openghg",
    "network": "agage",
    "site": "mhd",
    "inlet": "10m",
    "instrument": "gcmd",
    "calibration_scale": "WMO-x2004a",
}
mhd_obs_data_path = _raw_data_path / "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(TestData(standardise_surface, mhd_obs_metadata, mhd_obs_data_path, "surface"))

## Satellite Column data
satellite_gosat_obs_metadata = {
    "source_format": "openghg",
    "satellite": "gosat",
    "network": "gosat",
    "domain": "southamerica",
    "instrument": "tanso-fts",
    "species": "ch4",
}
satellite_gosat_obs_data_path = (
    _raw_data_path / "satellite" / "column" / "gosat-fts_gosat_20160101_ch4-column.nc"
)
satellite_test_data_list.append(
    TestData(standardise_column, satellite_gosat_obs_metadata, satellite_gosat_obs_data_path, "column")
)

## BC data
# Both BC fixtures contain a single January field with monthly source coverage.
bc_metadata = {
    "species": "ch4",
    "bc_input": "cams",
    "domain": "europe",
    "store": "inversions_tests",
    "period": "monthly",
}
bc_data_path = _raw_data_path / "bc_ch4_europe_cams_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_bc, bc_metadata, bc_data_path, "boundary_conditions"))

satellite_bc_metadata = {
    "species": "ch4",
    "bc_input": "cams",
    "domain": "southamerica",
    "store": "inversions_satellite_tests",
    "period": "monthly",
}
satellite_bc_data_path = _raw_data_path / "satellite" / "bc" / "ch4_SOUTHAMERICA_201601_CAMS-inversion.nc"
satellite_test_data_list.append(
    TestData(standardise_bc, satellite_bc_metadata, satellite_bc_data_path, "boundary_conditions")
)

## Footprint data
tac_footprints_metadata = {
    "site": "tac",
    "domain": "europe",
    "model": "name",
    "inlet": "185m",
    # Bound temporary index arrays when the ACRG parser transposes the source.
    "chunks": {"time": 24},
    # "metmodel": "ukv",
}
tac_footprints_data_path = _raw_data_path / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(
    TestData(standardise_footprint, tac_footprints_metadata, tac_footprints_data_path, "footprints")
)

mhd_footprints_metadata = {
    "site": "mhd",
    "domain": "europe",
    "model": "name",
    "inlet": "10m",
    "source_format": "paris",
    # "metmodel": "ukv",
}
mhd_footprints_data_path = _raw_data_path / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(
    TestData(standardise_footprint, mhd_footprints_metadata, mhd_footprints_data_path, "footprints")
)

footprints_satellite_metadata = {
    "satellite": "GOSAT",
    "domain": "southamerica",
    "model": "NAME",
    "inlet": "column",
    "source_format": "acrg_org",
    "obs_region": "brazil",
    "species": "ch4",
}
footprints_satellite_data = (
    _raw_data_path / "satellite" / "footprints" / "GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc"
)
satellite_test_data_list.append(
    TestData(standardise_footprint, footprints_satellite_metadata, footprints_satellite_data, "footprints")
)

## Flux data
flux_metadata = {"species": "ch4", "source": "total-ukghg-edgar7", "domain": "europe"}
flux_data_path = _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_flux, flux_metadata, flux_data_path, "flux"))

flux_dim_shuffle_metadata = {"species": "ch4", "source": "total-ukghg-edgar7-shuffled", "domain": "europe"}
flux_dim_shuffled_data_path = (
    _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data_dim_shuffled.nc"
)
test_data_list.append(
    TestData(standardise_flux, flux_dim_shuffle_metadata, flux_dim_shuffled_data_path, "flux")
)

flux_satellite_metadata = {"species": "ch4", "source": "SWAMPS", "domain": "southamerica"}
flux_satellite_datapath = (
    _raw_data_path / "satellite" / "flux" / "ch4_SOUTHAMERICA_2016_SWAMPS-v32-5_Saunois-Annual-Mean.nc"
)
satellite_test_data_list.append(
    TestData(standardise_flux, flux_satellite_metadata, flux_satellite_datapath, "flux")
)


def _populate_test_store(store_name: str, store_config: dict, manifest: list[TestData]) -> None:
    """Populate one run-local input store, retrying incomplete setup under a native lock."""
    store_path = Path(store_config["path"])
    ready_path = store_path.with_suffix(".ready")
    with FileLock(store_path.with_suffix(".lock"), timeout=120):
        if ready_path.exists() and store_path.is_dir():
            return

        if store_path.exists():
            shutil.rmtree(store_path)
        ready_path.unlink(missing_ok=True)
        store_config["permissions"] = "rw"
        try:
            for test_data in manifest:
                metadata = {**test_data.metadata, "store": store_name}
                test_data.func(filepath=test_data.path, **metadata)
        finally:
            store_config["permissions"] = "r"

        ready_path.touch()


@pytest.fixture(scope="session")
def openghg_test_store(session_config_mocker) -> None:
    """Populate ground inputs once per pytest run, shared read-only by local workers."""
    _populate_test_store(
        "inversions_tests", session_config_mocker["object_store"]["inversions_tests"], test_data_list
    )


@pytest.fixture(scope="session")
def openghg_satellite_test_store(session_config_mocker) -> None:
    """Populate satellite inputs independently of the ground store."""
    _populate_test_store(
        "inversions_satellite_tests",
        session_config_mocker["object_store"]["inversions_satellite_tests"],
        satellite_test_data_list,
    )


@pytest.fixture(scope="session")
def default_bc_basis_directory(writable_data_path, raw_data_path) -> Path:
    """Create a worker-local default BC basis directory for real-data tests."""
    bc_basis_function_path = writable_data_path / "bc_basis_functions"
    domain_dir = bc_basis_function_path / "EUROPE"
    domain_dir.mkdir(parents=True, exist_ok=True)

    # copy basis file into default location if there isn't a file with the same name there
    if not (domain_dir / "NESW_EUROPE_2019.nc").exists():
        shutil.copy(
            (raw_data_path / "bc_basis_NESW_EUROPE_2019.nc"),
            (domain_dir / "NESW_EUROPE_2019.nc"),
        )

    return bc_basis_function_path


@pytest.fixture(scope="module")
def tac_ch4_data_args(openghg_test_store):
    data_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        # "met_model": "ukv",
        "averaging_period": ["1h"],
    }
    return data_args


@pytest.fixture(scope="module")
def mhd_and_tac_ch4_data_args(openghg_test_store):
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        # "met_model": "ukv",
        "averaging_period": ["1h", "1h"],
    }
    return data_args
