import re

import pytest

from openghg_inversions.postprocessing._country_codes import (
    extend_abbrs,
    get_country_code,
    CountryInfoList,
    remove_common_words,
)


# Test string formatting
@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        ("land", "land"),
        ("of the", " "),  # whitespace is not removed
    ],
)
def test_remove_common_words(input_string, expected):
    assert remove_common_words(input_string) == expected


@pytest.mark.parametrize(
    ("input_string", "expected_to_match"),
    [
        ("St. Vincent", "Saint Vincent"),
        ("N. Wales", "North Wales"),
        ("F. Scott Fitzgerald", "Francis Scott Fitzgerald"),
        ("S.S. Great Britain", "Silver Surfer Great Britain"),
        ("R.M.S.", "Root Mean Square"),
    ],
)
def test_extend_abbrs(input_string, expected_to_match):
    pat = extend_abbrs(input_string)
    assert re.search(pat, expected_to_match, flags=re.IGNORECASE)


# Test `get_country_code`, which is the country code inference code behind CountryInfo and CountryInfoList
@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        # special value, left unchanged
        ("OCEAN", "OCEAN"),
        ("Sea", "Sea"),
        ("land", "land"),
        # unrecognised values, left unchanged
        ("N. Cyprus", "N. Cyprus"),
        ("Pluto", "Pluto"),
        ("Texas", "Texas"),
        # some cases that should work
        ("N. Korea", "PRK"),
        ("Democratic People's Republic of Korea", "PRK"),
        ("S. Korea", "KOR"),
        ("Republic of Korea", "KOR"),
        ("Korea (Rep. of)", "KOR"),
        ("The Republic of the Congo", "COG"),
        ("Democratic Republic of Congo", "COD"),
        ("DRC", "COD"),
        ("Faeroe Islands", "FRO"),
    ],
)
def test_get_country_code(input_string, expected):
    assert get_country_code(input_string) == expected


# Test CountryInfoList
@pytest.mark.parametrize(
    ("country_code", "expected"),
    [
        ("alpha2", ["US", "GB", "Ocean"]),
        ("alpha3", ["USA", "GBR", "Ocean"]),
        (None, ["United States", "UK", "Ocean"]),
    ],
)
def test_country_info_list_modes(country_code, expected):
    """Test that retrieved values act like list in given format.

    Note that "Ocean" is ignored by the country code lookup, while "United States"
    and "UK" return the expected results.

    When country_code is None, the input values are returned.
    """
    country_list = CountryInfoList(["United States", "UK", "Ocean"], country_code=country_code)

    # NOTE: comparison with ordinary list succeeds
    assert country_list == expected

    # convert to list to get list of strings according to country_code
    assert list(country_list) == expected


def test_country_info_list_extend():
    c_list1 = CountryInfoList(["United States", "UK", "Ocean"])
    c_list2 = CountryInfoList(["United States", "UK", "Ocean"])

    c_list1.extend(["UAE", "Greenland"])
    c_list2.extend(CountryInfoList(["UAE", "Greenland"]))

    # check that extending by ordinary list or CountryInfoList have same effect
    assert c_list1 == c_list2


def test_country_info_list_select():
    """Check that selecting by another list returns the input names of the original list."""
    c_list = CountryInfoList(["United States", "UK", "Ocean"])

    c_list_selected = c_list.select_by_country_info(["USA", "Great Britain"])
    assert c_list_selected == ["US", "GB"]

    # convert to list with country_code = None gives list of input names
    assert list(c_list_selected) == ["United States", "UK"]
