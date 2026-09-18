"""Ownership and compatibility tests for legacy fixed-basis preparation."""

from pathlib import Path

import pytest

import openghg_inversions.hbmcmc.hbmcmc as hbmcmc
import openghg_inversions.inversion_data as inversion_data
import openghg_inversions.inversion_data.preparation as shared_preparation
from openghg_inversions.hbmcmc.preparation import (
    FixedBasisPreparedData,
    prepare_fixedbasis_inversion_data,
)


@pytest.mark.parametrize(
    ("module", "name", "canonical"),
    [
        (inversion_data, "FixedBasisPreparedData", FixedBasisPreparedData),
        (inversion_data, "prepare_fixedbasis_inversion_data", prepare_fixedbasis_inversion_data),
        (shared_preparation, "FixedBasisPreparedData", FixedBasisPreparedData),
        (
            shared_preparation,
            "prepare_fixedbasis_inversion_data",
            prepare_fixedbasis_inversion_data,
        ),
    ],
)
def test_former_fixedbasis_imports_warn_and_resolve_to_hbmcmc(module, name, canonical) -> None:
    """Former import locations remain aliases during the compatibility period."""
    with pytest.warns(FutureWarning, match="has moved to openghg_inversions.hbmcmc.preparation"):
        alias = getattr(module, name)

    assert alias is canonical


def test_fixedbasis_preparation_contract_is_owned_by_hbmcmc() -> None:
    """The runner and contract resolve to the compatibility-owned module."""
    assert FixedBasisPreparedData.__module__ == "openghg_inversions.hbmcmc.preparation"
    assert prepare_fixedbasis_inversion_data.__module__ == "openghg_inversions.hbmcmc.preparation"
    assert hbmcmc.FixedBasisPreparedData is FixedBasisPreparedData
    assert hbmcmc.prepare_fixedbasis_inversion_data is prepare_fixedbasis_inversion_data


@pytest.mark.parametrize(
    "relative_path",
    ["rhime/standard.py", "rhime/multisector.py", "rhime/preparation.py"],
)
def test_modern_rhime_modules_do_not_depend_on_fixedbasis_preparation(relative_path: str) -> None:
    """Modern recipes and preparation do not import the legacy contract."""
    package_root = Path(__file__).parents[1] / "openghg_inversions"
    source = (package_root / relative_path).read_text()

    assert "FixedBasisPreparedData" not in source
    assert "prepare_fixedbasis_inversion_data" not in source
