"""Explicit BC algorithm selection preserves directional and file-backed science."""

from copy import deepcopy
from importlib import import_module
import json

import dask.array as da
from dask import compute
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import BasisFunctions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis import support
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.preparation import resolve_bc_basis
from openghg_inversions.rhime.requests import load_params
from tests.test_bc_basis_resources import _contributions, _basis
from tests.test_staged_workflow import _params, _prepared


def test_omitted_and_explicit_nesw_preserve_directional_labels_and_lazy_inputs(monkeypatch):
    monkeypatch.setattr(support, "load_bc_basis", lambda *a, **k: pytest.fail("NESW selected a BC file"))
    site = _contributions()
    before = site.copy(deep=True)
    results = []
    with Callback(pretask=lambda *args: pytest.fail("NESW computed borrowed arrays")):
        for selection in ({}, {"bc_basis_algorithm": "NESW"}):
            request = _params(use_bc=True, **selection)
            original = deepcopy(request)
            config = RhimeConfig.from_params(load_params(overrides=request), multisector=False)
            assert config.data.bc_basis_algorithm == "NESW"
            assert isinstance(config.data.bc_basis_algorithm, str)
            basis = resolve_bc_basis(config.data)
            assert basis is None
            results.append(bc_sensitivity({"TAC": site}, bc_basis=basis)["TAC"].H_bc)
            assert request == original
    assert "H_bc" not in site
    assert all(isinstance(result.data, da.Array) for result in results)
    assert results[0].bc_region.values.tolist() == ["n", "e", "s", "w"]
    actual = compute(*results)
    xr.testing.assert_identical(actual[0], actual[1])
    np.testing.assert_array_equal(actual[0], [[1, 1], [2, 2], [3, 3], [4, 4]])
    xr.testing.assert_identical(site, before)


@pytest.mark.parametrize("algorithm", [None, "unselected-algorithm"])
def test_files_override_inactive_algorithm_without_reinterpreting_time_alignment(tmp_path, algorithm):
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    late, early = tmp_path / "a.nc", tmp_path / "z.nc"
    _basis(times[1:], [3]).to_netcdf(late)
    _basis(times[:1], [2]).assign_coords(lat=[50.0 + 1e-6]).to_netcdf(early)
    config = RhimeConfig.from_params(_params(
        use_bc=True, bc_basis_algorithm=algorithm, bc_basis_paths=[late, early],
    ), multisector=False)
    saved = config.data.to_params()
    basis = resolve_bc_basis(config.data)
    snapshot = basis.copy(deep=True)
    result = bc_sensitivity({"TAC": _contributions()}, bc_basis=basis)["TAC"].H_bc
    np.testing.assert_array_equal(result.compute(), [[20], [30]])
    xr.testing.assert_identical(basis, snapshot)
    assert config.data.to_params() == saved


@pytest.mark.parametrize("algorithm", [None, "unknown", 123])
def test_applicable_algorithm_is_validated_before_resource_work(algorithm):
    with pytest.raises(ValueError, match="bc_basis_algorithm.*NESW"):
        RhimeConfig.from_params(_params(use_bc=True, bc_basis_algorithm=algorithm), multisector=False)


@pytest.mark.parametrize("algorithm", [None, "unknown", "NESW"])
def test_disabled_bc_does_not_load_selected_resources_or_activate_algorithm(monkeypatch, tmp_path, algorithm):
    monkeypatch.setattr(support, "load_bc_basis", lambda *a, **k: pytest.fail("disabled BC loaded files"))
    config = RhimeConfig.from_params(_params(
        use_bc=False, bc_basis_algorithm=algorithm, bc_basis_paths=[tmp_path / "absent.nc"],
    ), multisector=False)
    assert resolve_bc_basis(config.data) is None


@pytest.mark.parametrize("multisector", [False, True])
@pytest.mark.parametrize("selection", [{}, {"bc_basis_algorithm": "NESW"}, {"bc_basis_algorithm": None}])
def test_bc_selection_survives_preparation_manifest_roundtrip(tmp_path, multisector, selection):
    stage = import_module("openghg_inversions.rhime." + ("multisector_stages" if multisector else "_standard_stages"))
    sources = ["first", "second"] if multisector else ["synthetic-total"]
    config = RhimeConfig.from_params(_params(flux_sources=sources, **selection), multisector=multisector)
    prepared = _prepared()
    if multisector:
        source = xr.IndexVariable("source", sources)
        inputs = prepared.inv_inputs.assign(H=xr.concat([prepared.inv_inputs.H, prepared.inv_inputs.H * 2], dim=source))
        basis = BasisFunctions.from_flat_basis(
            prepared.basis_functions.flat_basis(),
            flux=xr.concat([prepared.basis_functions.flux, prepared.basis_functions.flux * 3], dim=source),
        )
        prepared = PreparedInputs(inputs, basis, prepared.site_metadata)
    manifest = stage.save_preparation(prepared, data_config=config.data, output_dir=tmp_path / "saved")
    record = json.loads(manifest.read_text())
    assert record["data"]["bc_basis_algorithm"] == config.data.bc_basis_algorithm
    loaded, data, _ = stage.load_preparation(manifest)
    assert data.bc_basis_algorithm == config.data.bc_basis_algorithm
    assert data.bc_basis_paths is None
    xr.testing.assert_identical(loaded.inv_inputs, prepared.inv_inputs)


def test_nested_inner_preparation_keeps_bc_disabled_with_outer_file_selection(monkeypatch, tmp_path):
    from openghg_inversions.rhime import nested

    monkeypatch.setattr(nested, "get_domain_info", lambda: {})
    monkeypatch.setattr(support, "load_bc_basis", lambda *a, **k: pytest.fail("inner BC selection opened files"))
    outer = RhimeConfig.from_params(_params(
        use_bc=True, bc_basis_paths=[tmp_path / "outer.nc"], bc_basis_algorithm="inactive",
    ), multisector=False).data
    inner = nested.inner_data_config(outer, inner_domain="6km")
    assert not inner.use_bc
    assert inner.bc_basis_paths is None
    assert resolve_bc_basis(inner) is None
    assert outer.use_bc
    assert outer.bc_basis_paths == (tmp_path / "outer.nc",)
