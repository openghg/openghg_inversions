import pytest
import xarray as xr

from openghg_inversions.postprocessing.countries import Countries, CountryRegions, paris_regions_dict
from openghg_inversions.postprocessing._country_codes import CountryInfoList

def test_country_regions_missing_check():
    paris_regions_countries = CountryInfoList(
        [
            "AUT",
            "BEL",
            "CHE",
            "CZE",
            "DEU",
            "DNK",
            "ESP",
            "FRA",
            "GBR",
            "HRV",
            "HUN",
            "IRL",
            "ITALY",
            "LUX",
            "NLD",
            "POLAND",
            "PRT",
            "SVK",
            "SVN",
        ]
    )

    paris_regions = CountryRegions(paris_regions_dict['europe'])

    # check 1: "ITALY" vs "ITA" and "POLAND" vs "POL" doesn't affect check
    missing = paris_regions.region_countries_missing_from(paris_regions_countries)

    assert "CW_EU" not in missing

    # check 2: omitting countries required by CW_EU definition flags missing countries
    missing = paris_regions.region_countries_missing_from(paris_regions_countries[3:])

    assert "CW_EU" in missing


def test_country_regions_align(country_ds):
    """Check that aligning country regions defined with alpha3 codes results in definitions with input names
    for EUROPE domain."""
    paris_regions = CountryRegions(paris_regions_dict['europe'])
    countries_list = CountryInfoList(country_ds.name.values)

    assert list(paris_regions.align(countries_list).to_dict()["BELUX"]) == ["BELGIUM", "LUXEMBOURG"]

@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_with_regions(country_code, country_ds, europe_country_file):
    """Check that country regions combine with countries correctly in EUROPE domain."""
    countries = Countries.from_file(
        domain="EUROPE", country_regions=paris_regions_dict['europe'], country_code=country_code,
        country_file=europe_country_file
    )

    assert len(countries.country_selections) == len(country_ds.name) + len(paris_regions_dict['europe'])

@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_with_regions_EASTASIA(country_code, country_ds_eastasia, eastasia_country_file):
    """Check that country regions combine with countries correctly in EASTASIA domain."""
    countries = Countries.from_file(
        domain="EASTASIA", country_regions=paris_regions_dict.get('eastasia'), country_code=country_code,
        country_file=eastasia_country_file
    )

    assert len(countries.country_selections) == len(country_ds_eastasia.name) + len(paris_regions_dict.get('eastasia', []))


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_skips_regions_with_missing_country_codes(country_code, eastasia_country_file):
    """Check that regions with missing country names/codes are skipped instead of raising."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    countries = Countries.from_file(
        domain="EASTASIA",
        country_regions=country_regions,
        country_code=country_code,
        country_file=eastasia_country_file,
        drop_missing_regions=True,
    )

    assert "EASTERN_ASIA" not in countries.country_selections


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_warns_when_regions_dropped(country_code, eastasia_country_file):
    """Check that dropping regions due to missing country names/codes emits a warning."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    with pytest.warns(UserWarning, match="Dropping country regions with unmatched countries"):
        countries = Countries.from_file(
            domain="EASTASIA",
            country_regions=country_regions,
            country_code=country_code,
            country_file=eastasia_country_file,
            drop_missing_regions=True,
        )

    assert "EASTERN_ASIA" not in countries.country_selections


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_raises_on_missing_regions_by_default(country_code, eastasia_country_file):
    """Check that missing countries in region definitions raise by default."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    with pytest.raises(ValueError, match="Could not find the following countries needed for regions"):
        Countries.from_file(
            domain="EASTASIA",
            country_regions=country_regions,
            country_code=country_code,
            country_file=eastasia_country_file,
        )
