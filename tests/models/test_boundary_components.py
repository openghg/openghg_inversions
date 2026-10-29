"""Shared boundary helpers preserve imports and warn about known constant overlap."""

import importlib
import warnings

from dask import array as da, delayed
import numpy as np
import pytest

from openghg_inversions.models.additive_boundary import warn_boundary_offset_confounding


def test_prototype_concentration_record_name_remains_an_alias():
    from openghg_inversions.models import AffineTerm, ConcentrationTerm
    from openghg_inversions.models import components

    assert AffineTerm is ConcentrationTerm is components.AffineTerm


@pytest.mark.parametrize("channel", [None, "co2", "o2"])
@pytest.mark.parametrize("mean_response", [np.ones(3), np.array([1.0, 1.0 + 5e-13])])
def test_unit_boundary_mean_warns_for_unrestricted_offsets(channel, mean_response):
    with pytest.warns(UserWarning, match="likelihood identifies only their combined constant") as caught:
        warn_boundary_offset_confounding(mean_response, drop_first=False, channel=channel)
    assert len(caught) == 1
    if channel is not None:
        assert str(caught[0].message).startswith(f"{channel}: ")


@pytest.mark.parametrize(
    "mean_response,drop_first",
    [
        (np.ones(3), True),
        (np.full(3, 0.8), False),
        (np.array([1.0, 0.8]), False),
        (np.array([1.0, 1.0 + 2e-12]), False),
    ],
)
def test_warning_is_limited_to_unrestricted_unit_mean_response(mean_response, drop_first):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_boundary_offset_confounding(mean_response, drop_first=drop_first)
    assert not caught


def test_warning_rejects_lazy_payload_without_computing_it():
    calls = []

    @delayed
    def response():
        calls.append("computed")
        return np.ones(3)

    lazy_response = da.from_delayed(response(), shape=(3,), dtype=float)
    with pytest.raises(TypeError, match="eager NumPy array"):
        warn_boundary_offset_confounding(lazy_response, drop_first=False)
    assert calls == []


@pytest.mark.parametrize(
    "legacy_name,shared_name,names",
    [
        (
            "openghg_inversions.rhime.co2.centered_boundary",
            "openghg_inversions.models.boundary",
            (
                "OPERAND_NAMES", "CenteredBoundaryMap", "CenteredBoundaryResult",
                "add_centered_boundary", "bind_boundary_operands", "require_centered_precision",
                "restore_absolute_predictives",
            ),
        ),
        (
            "openghg_inversions.rhime.co2.co2_boundary",
            "openghg_inversions.models.additive_boundary",
            ("AdditiveBoundaryResult", "CenteredBoundary", "add_centered_boundary", "prepare_centered_boundary"),
        ),
    ],
)
def test_existing_boundary_imports_are_aliases_of_shared_helpers(legacy_name, shared_name, names):
    legacy = importlib.import_module(legacy_name)
    shared = importlib.import_module(shared_name)
    assert set(legacy.__all__) == set(names)
    for name in names:
        assert getattr(legacy, name) is getattr(shared, name)
