"""Focused tests for modern nested-domain RHIME composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import openghg_inversions.rhime.nested as nested_module
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.cli import main
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.inversion_data.preparation import _SiteOptions
from openghg_inversions.models import RhimeModelSpec, SectorSpec, build_nested_rhime_model_from_spec
from openghg_inversions.rhime.nested import (
    align_inner_merged_to_outer_observations,
    build_nested_rhime_model_result,
    combine_nested_rhime_inputs,
    mask_outer_merged_for_inner_domain,
)
from openghg_inversions.rhime.runner import materialize_pymc_inputs
from openghg_inversions.rhime.params import RhimeRunnerSetup
from openghg_inversions.rhime import params as rhime_params
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeOutputSpec, RhimeRunSpec


def _basis(lat: list[float], lon: list[float], labels: np.ndarray) -> BasisFunctions:
    basis = xr.DataArray(
        labels,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="basis",
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    flux.attrs["units"] = "mol m-2 s-1"
    return BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": "region"},
    )


def _prepared(
    *,
    times: list[str],
    sensitivity: np.ndarray,
    basis: BasisFunctions,
) -> RhimePreparedInputs:
    nmeasure = pd.MultiIndex.from_arrays(
        [["TAC"] * len(times), pd.to_datetime(times)],
        names=("site", "time"),
    )
    inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), sensitivity),
            "mf": ("nmeasure", np.arange(len(times), dtype=float) + 10.0),
            "mf_error": ("nmeasure", np.ones(len(times))),
            "min_error": ("nmeasure", np.zeros(len(times))),
            "site_indicator": ("nmeasure", np.zeros(len(times), dtype=int)),
        },
        coords={
            "region": np.arange(sensitivity.shape[0]),
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
    )
    inputs["mf"].attrs["units"] = "ppm"
    return RhimePreparedInputs(
        inputs,
        basis,
        xr.Dataset(
            {"averaging_period": ("site", np.asarray(["1h"], dtype=object))},
            coords={"site": ["TAC"]},
        ),
    )


def _run_spec() -> RhimeRunSpec:
    model = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="total",
                flux_source="inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="total",
            ),
        ),
        use_bc=False,
        no_model_error=True,
    )
    return RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=("TAC",),
        averaging_period=("1h",),
        model=model,
        output=RhimeOutputSpec(
            output_format="none",
            output_name="nested-test",
            save_inversion_output=False,
        ),
    )


def test_combine_nested_inputs_retains_native_bases_and_aligns_nearest_time() -> None:
    outer_basis = _basis([50.0], [-2.0, -1.0], np.array([[1, 2]]))
    inner_basis = _basis([51.0, 51.5], [-1.5], np.array([[1], [2]]))
    outer = _prepared(
        times=["2019-01-01T00:00", "2019-01-01T01:00"],
        sensitivity=np.array([[1.0, 2.0], [3.0, 4.0]]),
        basis=outer_basis,
    )
    inner = _prepared(
        times=["2019-01-01T00:10", "2019-01-01T01:10"],
        sensitivity=np.array([[5.0, 6.0], [7.0, 8.0]]),
        basis=inner_basis,
    )

    nested = combine_nested_rhime_inputs(outer, inner, time_tolerance="15min")

    assert nested.outer.basis_functions is not nested.inner.basis_functions
    assert nested.combined.basis_functions.operator is nested.outer.basis_functions.operator
    assert nested.combined.inv_inputs["H_inner"].dims == ("inner_region", "nmeasure")
    np.testing.assert_allclose(nested.combined.inv_inputs["H_inner"], [[5.0, 6.0], [7.0, 8.0]])
    assert nested.combined.inv_inputs["H"].dims == ("region", "nmeasure")
    # Revalidation must preserve the explicitly selected nearest-time policy.
    nested.validated()


def test_combine_nested_inputs_rejects_unmatched_time_instead_of_zero_filling() -> None:
    basis = _basis([50.0], [-2.0], np.array([[1]]))
    outer = _prepared(
        times=["2019-01-01T00:00"],
        sensitivity=np.array([[1.0]]),
        basis=basis,
    )
    inner = _prepared(
        times=["2019-01-01T02:00"],
        sensitivity=np.array([[2.0]]),
        basis=basis,
    )

    try:
        combine_nested_rhime_inputs(outer, inner, time_tolerance="15min")
    except ValueError as exc:
        assert "cannot be aligned" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("Expected an unmatched nested observation to be rejected.")


@dataclass(frozen=True)
class _FluxData:
    data: xr.Dataset
    metadata: dict[str, str]


def _site_options() -> _SiteOptions:
    return _SiteOptions.from_inputs(
        sites=["TAC"],
        averaging_period=["1h"],
        inlet=["100m"],
        fp_height=["100m"],
        instrument=[None],
        platform=["surface"],
        obs_data_level=[None],
        met_model=[None],
        max_level=[None],
    )


def test_mask_outer_merged_zeroes_overlap_lazily_without_mutating_inputs() -> None:
    time = pd.date_range("2019-01-01", periods=1)
    lat = [0.0, 1.0, 2.0]
    lon = [0.0, 1.0, 2.0]
    values = da.ones((1, 3, 3), chunks=(1, 3, 3))
    outer_site = xr.Dataset(
        {
            "fp": (("time", "lat", "lon"), values),
            "fp_x_flux": (("time", "lat", "lon"), values),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    outer_flux = xr.Dataset(
        {"flux": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    inner_site = xr.Dataset(
        {"fp": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
        coords={"time": time, "lat": [1.0], "lon": [1.0]},
    )
    outer = RhimeMergedData(
        fp_all={"TAC": outer_site, ".flux": {"inventory": _FluxData(outer_flux, {"source": "x"})}},
        site_options=_site_options(),
    )
    inner = RhimeMergedData(
        fp_all={"TAC": inner_site},
        site_options=_site_options(),
    )

    masked = mask_outer_merged_for_inner_domain(outer, inner)

    assert isinstance(masked.fp_all["TAC"]["fp"].data, da.Array)
    assert isinstance(masked.fp_all[".flux"]["inventory"].data["flux"].data, da.Array)
    assert float(masked.fp_all["TAC"]["fp"].isel(time=0, lat=1, lon=1).compute()) == 0.0
    assert float(masked.fp_all["TAC"]["fp"].isel(time=0, lat=0, lon=0).compute()) == 1.0
    assert float(masked.fp_all[".flux"]["inventory"].data["flux"].isel(time=0, lat=1, lon=1).compute()) == 0.0
    assert float(outer.fp_all["TAC"]["fp"].isel(time=0, lat=1, lon=1).compute()) == 1.0


def test_inner_merged_alignment_mirrors_filtered_outer_times_with_tolerance() -> None:
    outer_times = pd.to_datetime(["2019-01-01T00:00", "2019-01-01T02:00"])
    inner_times = pd.to_datetime(
        ["2019-01-01T00:10", "2019-01-01T01:10", "2019-01-01T02:10"]
    )
    outer = RhimeMergedData(
        fp_all={"TAC": xr.Dataset({"mf": ("time", [1.0, 2.0])}, coords={"time": outer_times})},
        site_options=_site_options(),
    )
    inner = RhimeMergedData(
        fp_all={
            "TAC": xr.Dataset(
                {"fp": (("time", "lat", "lon"), np.arange(3.0).reshape(3, 1, 1))},
                coords={"time": inner_times, "lat": [1.0], "lon": [1.0]},
            )
        },
        site_options=_site_options(),
    )

    aligned = align_inner_merged_to_outer_observations(
        outer,
        inner,
        time_tolerance="15min",
    )

    np.testing.assert_array_equal(aligned.fp_all["TAC"]["time"], outer_times)
    np.testing.assert_allclose(aligned.fp_all["TAC"]["fp"].values[:, 0, 0], [0.0, 2.0])
    np.testing.assert_array_equal(inner.fp_all["TAC"]["time"], inner_times)


def test_nested_model_uses_two_labelled_state_blocks_and_shared_likelihood() -> None:
    outer_basis = _basis([50.0], [-2.0, -1.0], np.array([[1, 2]]))
    inner_basis = _basis([51.0, 51.5], [-1.5], np.array([[1], [2]]))
    outer = _prepared(
        times=["2019-01-01T00:00", "2019-01-01T01:00"],
        sensitivity=np.array([[1.0, 2.0], [3.0, 4.0]]),
        basis=outer_basis,
    )
    inner = _prepared(
        times=["2019-01-01T00:00", "2019-01-01T01:00"],
        sensitivity=np.array([[5.0, 6.0], [7.0, 8.0]]),
        basis=inner_basis,
    )
    prepared = combine_nested_rhime_inputs(outer, inner)
    run_spec = _run_spec()

    model = build_nested_rhime_model_from_spec(prepared.combined.inv_inputs, run_spec.model)
    assert {"hx_outer", "hx_inner", "x_outer", "x_inner", "mu_outer", "mu_inner", "mu", "y"} <= set(
        model.named_vars
    )

    result = build_nested_rhime_model_result(
        prepared=prepared,
        model_inputs=prepared.combined.inv_inputs,
        run_spec=run_spec,
    )
    assert result.variable_roles["flux_scale:outer"] == "x_outer"
    assert result.variable_roles["flux_scale:inner"] == "x_inner"
    assert result.variable_roles["concentration"] == "y"
    assert result.supported_output_formats == ("none",)


def test_nested_materialization_computes_both_sensitivities_at_named_boundary() -> None:
    basis = _basis([50.0], [-2.0], np.array([[1]]))
    outer = _prepared(
        times=["2019-01-01T00:00"],
        sensitivity=np.array([[1.0]]),
        basis=basis,
    )
    inner = _prepared(
        times=["2019-01-01T00:00"],
        sensitivity=np.array([[2.0]]),
        basis=basis,
    )
    prepared = combine_nested_rhime_inputs(outer, inner)
    lazy_outer = da.from_array(prepared.combined.inv_inputs["H"].values, chunks=(1, 1))
    lazy_inner = da.from_array(prepared.combined.inv_inputs["H_inner"].values, chunks=(1, 1))
    lazy_dataset = prepared.combined.inv_inputs.copy(deep=False)
    lazy_dataset["H"] = xr.Variable(("region", "nmeasure"), lazy_outer)
    lazy_dataset["H_inner"] = xr.Variable(("inner_region", "nmeasure"), lazy_inner)
    lazy_prepared = RhimePreparedInputs(
        lazy_dataset,
        prepared.combined.basis_functions,
        prepared.combined.site_metadata,
    )

    materialized = materialize_pymc_inputs(
        lazy_prepared,
        aggregation_error_mode="none",
        additional_variables=("H_inner",),
    )

    assert not isinstance(materialized["H"].data, da.Array)
    assert not isinstance(materialized["H_inner"].data, da.Array)
    assert lazy_prepared.inv_inputs["H"].data is lazy_outer
    assert lazy_prepared.inv_inputs["H_inner"].data is lazy_inner


def test_cli_run_rhime_nested_passes_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "nested.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_name = "nested"\n', encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run_rhime_nested(*, config_file: str, **kwargs: object) -> None:
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs

    monkeypatch.setattr("openghg_inversions.rhime.run_rhime_nested", fake_run_rhime_nested)

    main(["run-rhime-nested", "-c", str(config_file)])

    assert seen == {"config_file": str(config_file), "kwargs": {}}


def test_nested_preparation_uses_native_inner_domain_and_safe_basis_default(monkeypatch) -> None:
    basis = _basis([50.0], [-2.0], np.array([[1]]))
    prepared = _prepared(
        times=["2019-01-01T00:00"],
        sensitivity=np.array([[1.0]]),
        basis=basis,
    )
    merged = RhimeMergedData(fp_all={"TAC": xr.Dataset()}, site_options=_site_options())
    setup = RhimeRunnerSetup(
        run_spec=_run_spec(),
        sampler=RhimeSampler(),
        data_args={
            "species": "ch4",
            "sites": ["TAC"],
            "averaging_period": ["1h"],
            "domain": "EUROPE",
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "output_name": "nested-test",
            "flux_sources": ["inventory"],
            "use_bc": True,
            "basis_algorithm": "weighted",
            "nbasis": 100,
        },
    )
    retrieval_args: list[dict[str, object]] = []
    preparation_args: list[dict[str, object]] = []

    def fake_retrieve(data_args, *, multisector):
        assert multisector is False
        retrieval_args.append(dict(data_args))
        return merged

    def fake_prepare(domain_merged, data_args):
        assert domain_merged.sites == ("TAC",)
        preparation_args.append(dict(data_args))
        return prepared

    monkeypatch.setattr(nested_module, "retrieve_or_reload_rhime_data", fake_retrieve)
    monkeypatch.setattr(nested_module, "filter_rhime_observations", lambda value, args: value)
    monkeypatch.setattr(
        nested_module,
        "align_inner_merged_to_outer_observations",
        lambda outer, inner, time_tolerance: inner,
    )
    monkeypatch.setattr(nested_module, "mask_outer_merged_for_inner_domain", lambda outer, inner: outer)
    monkeypatch.setattr(nested_module, "_prepare_one_domain", fake_prepare)

    nested_module.prepare_nested_rhime_inputs(
        setup,
        inner_domain="6km",
        inner_footprint_store="inner-fp",
        inner_emissions_store="inner-flux",
        inner_nbasis=40,
    )

    assert len(retrieval_args) == 2
    inner_args = retrieval_args[1]
    assert inner_args["domain"] == "EUROPE-6km"
    assert inner_args["footprint_store"] == "inner-fp"
    assert inner_args["emissions_store"] == "inner-flux"
    assert inner_args["use_bc"] is False
    assert inner_args["basis_algorithm"] == "quadtree"
    assert inner_args["nbasis"] == 40
    assert inner_args["fp_basis_case"] is None
    assert inner_args["basis_output_path"] is None
    assert preparation_args[0]["basis_algorithm"] == "weighted"
    assert preparation_args[1]["basis_algorithm"] == "quadtree"


def test_nested_automatic_basis_budget_uses_bounded_sensitivity_share() -> None:
    time = pd.date_range("2019-01-01", periods=2)
    lat = [0.0, 1.0]
    lon = [0.0, 1.0]
    outer_values = da.ones((2, 2, 2), chunks=(2, 2, 1))
    inner_values = 3.0 * da.ones((2, 2, 2), chunks=(2, 2, 1))
    outer = RhimeMergedData(
        fp_all={
            "TAC": xr.Dataset(
                {"fp_x_flux": (("lat", "lon", "time"), outer_values)},
                coords={"lat": lat, "lon": lon, "time": time},
            )
        },
        site_options=_site_options(),
    )
    inner = RhimeMergedData(
        fp_all={
            "TAC": xr.Dataset(
                {"fp_x_flux": (("lat", "lon", "time"), inner_values)},
                coords={"lat": lat, "lon": lon, "time": time},
            )
        },
        site_options=_site_options(),
    )

    outer_nbasis, inner_nbasis = nested_module._allocate_nested_nbasis(
        outer,
        inner,
        total_nbasis=100,
    )

    assert (outer_nbasis, inner_nbasis) == (40, 60)
    assert isinstance(outer.fp_all["TAC"]["fp_x_flux"].data, da.Array)
    assert isinstance(inner.fp_all["TAC"]["fp_x_flux"].data, da.Array)


def test_nested_preparation_routes_automatic_basis_budget(monkeypatch) -> None:
    basis = _basis([50.0], [-2.0], np.array([[1]]))
    prepared = _prepared(
        times=["2019-01-01T00:00"],
        sensitivity=np.array([[1.0]]),
        basis=basis,
    )
    setup = RhimeRunnerSetup(
        run_spec=_run_spec(),
        sampler=RhimeSampler(),
        data_args={
            "species": "ch4",
            "sites": ["TAC"],
            "averaging_period": ["1h"],
            "domain": "EUROPE",
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "output_name": "nested-test",
            "flux_sources": ["inventory"],
            "nbasis": 100,
        },
    )
    preparation_args: list[dict[str, object]] = []

    def fake_retrieve(data_args, *, multisector):
        assert multisector is False
        value = 9.0 if data_args["domain"] == "EUROPE-6km" else 1.0
        dataset = xr.Dataset(
            {"fp_x_flux": (("time", "lat", "lon"), [[[value]]])},
            coords={"time": pd.date_range("2019-01-01", periods=1), "lat": [50.0], "lon": [-2.0]},
        )
        return RhimeMergedData(fp_all={"TAC": dataset}, site_options=_site_options())

    def fake_prepare(domain_merged, data_args):
        preparation_args.append(dict(data_args))
        return prepared

    monkeypatch.setattr(nested_module, "retrieve_or_reload_rhime_data", fake_retrieve)
    monkeypatch.setattr(nested_module, "filter_rhime_observations", lambda value, args: value)
    monkeypatch.setattr(
        nested_module,
        "align_inner_merged_to_outer_observations",
        lambda outer, inner, time_tolerance: inner,
    )
    monkeypatch.setattr(nested_module, "mask_outer_merged_for_inner_domain", lambda outer, inner: outer)
    monkeypatch.setattr(nested_module, "_prepare_one_domain", fake_prepare)

    nested_module.prepare_nested_rhime_inputs(setup, inner_domain="6km")

    assert preparation_args[0]["nbasis"] == 40
    assert preparation_args[1]["nbasis"] == 60


def test_legacy_outer_region_definition_name_normalizes_to_modern_path() -> None:
    with pytest.warns(UserWarning, match="outer_region_definition_file.*deprecated"):
        normalized = rhime_params.normalise_rhime_params(
            {"outer_region_definition_file": "/data/EUHROB.nc"}
        )

    assert normalized == {"outer_regions_path": "/data/EUHROB.nc"}


def test_outer_regions_path_routes_through_modern_rhime_setup() -> None:
    setup = nested_module.resolve_rhime_options(
        params={
            "species": "ch4",
            "sites": ["TAC"],
            "averaging_period": ["1h"],
            "domain": "EUROPE",
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "output_name": "nested",
            "output_format": "none",
            "flux_sources": ["inventory"],
            "outer_regions_path": "/data/EUHROB.nc",
        },
        multisector=False,
    )

    assert setup.data_args["outer_regions_path"] == "/data/EUHROB.nc"
