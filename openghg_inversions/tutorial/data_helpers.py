from collections import namedtuple
from importlib.resources import files

from openghg.standardise import (
    standardise_surface,
    standardise_bc,
    standardise_flux,
    standardise_footprint,
    standardise_column,
)


_raw_data_path = files("openghg_inversions.data")


# TEST DATA
TestData = namedtuple("TestData", ["func", "metadata", "path", "data_type"])
test_data_list = []

## Obs data

### TAC surface obs
tac_obs_metadata = {
    "source_format": "openghg",
    "network": "decc",
    "site": "tac",
    "inlet": "185m",
    "instrument": "picarro",
}
tac_obs_data_path = _raw_data_path / "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc"
test_data_list.append(TestData(standardise_surface, tac_obs_metadata, tac_obs_data_path, "surface"))

### MHD surface obs
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

### Satellite Column data
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
test_data_list.append(
    TestData(standardise_column, satellite_gosat_obs_metadata, satellite_gosat_obs_data_path, "column")
)

## BC data
bc_metadata = {"species": "ch4", "bc_input": "cams", "domain": "europe", "store": "inversions_tests"}
bc_data_path = _raw_data_path / "bc_ch4_europe_cams_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_bc, bc_metadata, bc_data_path, "boundary_conditions"))

satellite_bc_metadata = {
    "species": "ch4",
    "bc_input": "cams",
    "domain": "southamerica",
    "store": "inversions_tests",
}
satellite_bc_data_path = _raw_data_path / "satellite" / "bc" / "ch4_SOUTHAMERICA_201601_CAMS-inversion.nc"
test_data_list.append(
    TestData(standardise_bc, satellite_bc_metadata, satellite_bc_data_path, "boundary_conditions")
)

## Footprint data

### TAC footprints
tac_footprints_metadata = {
    "site": "tac",
    "domain": "europe",
    "model": "name",
    "inlet": "185m",
    # "metmodel": "ukv",
}
tac_footprints_data_path = _raw_data_path / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(
    TestData(standardise_footprint, tac_footprints_metadata, tac_footprints_data_path, "footprints")
)

### MHD footprints
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

### Satellite footprints
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
test_data_list.append(
    TestData(standardise_footprint, footprints_satellite_metadata, footprints_satellite_data, "footprints")
)

## Flux data

### CH4 EDGAR flux
flux_metadata = {"species": "ch4", "source": "total-ukghg-edgar7", "domain": "europe"}
flux_data_path = _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_flux, flux_metadata, flux_data_path, "flux"))

### Shuffled CH4 EDGAR flux
flux_dim_shuffle_metadata = {"species": "ch4", "source": "total-ukghg-edgar7-shuffled", "domain": "europe"}
flux_dim_shuffled_data_path = (
    _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data_dim_shuffled.nc"
)
test_data_list.append(
    TestData(standardise_flux, flux_dim_shuffle_metadata, flux_dim_shuffled_data_path, "flux")
)

### South America flux (for satellite tests)
flux_satellite_metadata = {"species": "ch4", "source": "SWAMPS", "domain": "southamerica"}
flux_satellite_datapath = (
    _raw_data_path / "satellite" / "flux" / "ch4_SOUTHAMERICA_2016_SWAMPS-v32-5_Saunois-Annual-Mean.nc"
)
test_data_list.append(TestData(standardise_flux, flux_satellite_metadata, flux_satellite_datapath, "flux"))


def add_test_data(store: str = "inversions_test_store") -> None:
    """Create object store for running tests."""
    for test_data in test_data_list:
        standardise_fn = test_data.func
        file_path = test_data.path
        metadata = test_data.metadata
        metadata["store"] = store
        standardise_fn(filepath=file_path, **metadata)


# COUNTRY FILE PATHS
country_europe_path = _raw_data_path / "country_EUROPE.nc"
country_eastasia_path = _raw_data_path / "country_EASTASIA.nc"
country_southamerica_path = _raw_data_path / "satellite" / "country" / "country_SOUTHAMERICA.nc"
