"""Retain the rectangle actually masked, independently of basis flux labels."""

from dataclasses import replace

import pytest

from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime import nested as nested_module
from openghg_inversions.rhime.params import RhimeConfig
from tests.test_nested_core import _nested
from tests.test_nested_preparation_contracts import _data
from tests.test_staged_workflow import _params


def _metadata(prepared):
    return nested_module._nested_domain_artifact_metadata(
        prepared, outer_domain="EUROPE", view="outer",
        outer_grid=nested_module._grid_identity(prepared.outer),
        inner_grid=nested_module._grid_identity(prepared.inner),
    )["outer_support"]


def test_external_mask_bounds_remain_unknown_or_explicitly_declared():
    supplied = _nested()
    assert _metadata(supplied)["inner_extent"] is None
    assert _metadata(supplied)["mask_extent_source"] == "unknown"
    extent = (49.9, 50.4, -1.8, -1.2)
    declared = replace(supplied, outer_overlap_mask_extent=extent).validated()
    assert declared.outer_overlap_mask_extent == extent
    assert _metadata(declared)["inner_extent"] == {
        "latitude": [49.9, 50.4], "longitude": [-1.8, -1.2],
    }
    assert _metadata(declared)["mask_extent_source"] == "caller_supplied"


def test_preparation_records_footprint_rectangle_even_when_basis_flux_labels_differ(monkeypatch):
    monkeypatch.setattr(nested_module, "get_domain_info", lambda: {})
    native = _nested()
    outer = _data([50.0, 51.0], [-2.0, -1.0])
    inner = _data([49.9, 50.4], [-1.8, -1.2])
    inner = replace(inner, site_data={
        "A": inner.site_data["A"],
        "B": inner.site_data["B"].assign_coords(lat=inner.site_data["B"].lat + 1e-7),
    })
    # Observation/error policy and basis fitting are independent of this retained
    # fact. Exercise real shared-grid masking and the final combine boundary.
    outer = RhimeMergedData(outer.site_data, outer.flux_data, outer.site_options)
    inner = RhimeMergedData(inner.site_data, inner.flux_data, inner.site_options)
    monkeypatch.setattr(nested_module, "prepare_observation_errors", lambda data, **kwargs: data)
    monkeypatch.setattr(nested_module, "filter_observations", lambda data, options, **kwargs: (data, options))
    prepared_sites = {}

    def prepare_domain(data, **kwargs):
        prepared_sites[kwargs["domain"]] = data.site_data
        return native.inner if kwargs["domain"] == "EUROPE-6km" else native.outer

    monkeypatch.setattr(nested_module, "_prepare_one_domain", prepare_domain)
    config = RhimeConfig.from_params(_params(), multisector=False)
    prepared = nested_module.prepare_nested_rhime_inputs(
        outer, inner,
        **config.data.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
        inner_domain="EUROPE-6km", inner_basis_algorithm=None, inner_nbasis=2,
        outer_basis_inputs={}, inner_basis_inputs={},
    )
    for sites in prepared_sites.values():
        assert sites["A"].get_index("lat").equals(sites["B"].get_index("lat"))
    assert prepared.inner.basis_functions.flux.lat.min().item() == 50.0
    assert prepared.outer_overlap_mask_extent == (49.9, 50.4, -1.8, -1.2)
    assert prepared.validated().outer_overlap_mask_extent == prepared.outer_overlap_mask_extent
    support = _metadata(prepared)
    assert support["mask_extent_source"] == "inner_footprint_grid"
    assert support["inner_extent"] == {"latitude": [49.9, 50.4], "longitude": [-1.8, -1.2]}


@pytest.mark.parametrize("extent", [
    (0.0, 1.0, 2.0), (0.0, 1.0, 2.0, 3.0, 4.0),
    (1.0, 0.0, 2.0, 3.0), (0.0, 1.0, 3.0, 2.0),
    (float("nan"), 1.0, 2.0, 3.0), (0.0, 1.0, 2.0, float("inf")),
])
def test_external_mask_bounds_reject_malformed_rectangles(extent):
    native = _nested()
    with pytest.raises(ValueError, match="four finite, ordered rectangle bounds"):
        nested_module.combine_nested_rhime_inputs(
            native.outer, native.inner, outer_overlap_masked=True, outer_overlap_mask_extent=extent,
        )
