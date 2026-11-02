"""Coherent supplied backgrounds retain marginal priors and selected operators."""

from dataclasses import replace

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_supplied_baseline
from openghg_inversions.rhime.co2.co2_preparation import co2_supplied_baseline_input_names
from test_co2_preparation import _canonical_inputs, _reduction
from openghg_inversions.rhime.co2 import prepare_co2_inputs


def _case(multisource=False):
    canonical = _canonical_inputs()
    old_basis = canonical.basis_functions
    basis = BasisFunctions.from_flat_basis(
        old_basis.operator.basis_flat, old_basis.flux,
        operator_kwargs={"state_dim": "region"}, region_labels="range1",
    )
    canonical = replace(canonical, basis_functions=basis)
    prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=None)
    if multisource:
        flat = basis.operator.basis_flat
        basis = BasisFunctions.from_multi_source_flat_basis(
            {"bio": flat, "fossil": flat},
            {"bio": basis.flux, "fossil": basis.flux},
            operator_kwargs={"state_dim": "region"},
        )
        idx = basis.operator.basis_matrix.indexes["region"]
        inputs = prepared.inv_inputs.drop_dims(["region", "region_cov"])
        inputs = inputs.assign_coords(xr.Coordinates.from_pandas_multiindex(idx, "region"))
        inputs = inputs.assign_coords(region_cov=("region_cov", idx.to_numpy()))
        inputs["H"] = xr.DataArray(da.from_array(np.arange(12.).reshape(3, 4) + .125), dims=("nmeasure", "region"))
        inputs.H.attrs["units"] = "ppm"
        inputs["alpha_prior_mean"] = xr.DataArray([.8, 1.5, 1.2, 2.], dims="region", attrs={"units": "1"})
        inputs["alpha_prior_covariance"] = xr.DataArray(
            np.eye(4) + .2 * np.ones((4, 4)), dims=("region", "region_cov"), attrs={"units": "1"},
        )
        prepared = Co2PreparedInputs(PreparedInputs(inputs, basis, canonical.site_metadata), "dense")
    inputs = prepared.inv_inputs
    observations = inputs.mf
    mask = xr.ones_like(inputs.alpha_prior_mean, dtype=bool)
    mask.data = np.array([False, True, True, False] if multisource else [True, False])
    support = xr.full_like(basis.flux, True, dtype=bool)
    kwargs = dict(
        inner_states=mask, inner_support=support,
        inner_fixed_contribution=xr.full_like(observations, .75).assign_attrs(units="ppm"),
        baseline_site_metadata=prepared.site_metadata, species="co2", provenance="synthetic absolute exterior",
        constituents=("boundary", "exterior_flux"),
    )
    baseline = xr.full_like(observations, .4).assign_attrs(units="mmol/mol", species="co2")
    return prepared, baseline, kwargs


@pytest.mark.parametrize("multisource", [False, True])
def test_selects_marginal_prior_and_coherent_columns(multisource):
    prepared, baseline, kwargs = _case(multisource)
    before = prepared.inv_inputs.copy(deep=True)
    result = prepare_co2_supplied_baseline(prepared, baseline, **kwargs)
    positions = np.flatnonzero(kwargs["inner_states"])
    np.testing.assert_allclose(result.inv_inputs.H, prepared.inv_inputs.H.isel(region=positions))
    np.testing.assert_allclose(result.inv_inputs.alpha_prior_mean, prepared.inv_inputs.alpha_prior_mean.isel(region=positions))
    np.testing.assert_allclose(result.inv_inputs.alpha_prior_covariance,
        prepared.inv_inputs.alpha_prior_covariance.isel(region=positions, region_cov=positions))
    np.testing.assert_allclose(result.inv_inputs.fixed_prior_contribution, .75)
    np.testing.assert_allclose(result.inv_inputs.supplied_baseline, 400.)
    assert "inner_fixed_contribution" not in result.inv_inputs
    assert result.inv_inputs.aggregation_error_covariance.data is prepared.inv_inputs.aggregation_error_covariance.data
    assert "unresolved exterior" in result.inv_inputs.supplied_baseline.attrs["aggregation_error_warning"]
    assert co2_supplied_baseline_input_names(result.inv_inputs) == ("supplied_baseline",)
    assert result.inv_inputs.indexes["region"].equals(result.basis_functions.operator.basis_matrix.indexes["region"])
    xr.testing.assert_identical(prepared.inv_inputs, before)
    if multisource:
        assert hasattr(result.inv_inputs.H.data, "__dask_graph__")
        assert list(result.inv_inputs.original_region_in_source.values) == [1, 0]


@pytest.mark.parametrize("multisource", [False, True])
def test_roundtrip_retains_native_support_and_original_labels(multisource, tmp_path):
    prepared, baseline, kwargs = _case(multisource)
    result = prepare_co2_supplied_baseline(prepared, baseline, **kwargs)
    path = tmp_path / "inner-co2.nc"
    result.save(path)
    restored = Co2PreparedInputs.load(path)
    xr.testing.assert_identical(restored.inv_inputs, result.inv_inputs)
    xr.testing.assert_identical(restored.basis_functions.flux, result.basis_functions.flux)
    assert restored.basis_functions.operator.basis_matrix.indexes["region"].equals(result.inv_inputs.indexes["region"])


@pytest.mark.parametrize("bad", ["policy", "metadata", "unlabelled", "lazy_selection", "empty", "species"])
def test_rejects_ambiguous_preparation_inputs(bad):
    prepared, baseline, kwargs = _case()
    if bad == "policy":
        kwargs["aggregation_error_policy"] = "recomputed"
    elif bad == "metadata":
        kwargs["baseline_site_metadata"] = prepared.site_metadata.assign(averaging_period=("site", ["8h", "8h"]))
    elif bad == "unlabelled":
        kwargs["inner_states"] = xr.DataArray([True, False], dims="region")
    elif bad == "lazy_selection":
        kwargs["inner_states"] = kwargs["inner_states"].chunk()
    elif bad == "empty":
        kwargs["inner_states"] = xr.zeros_like(kwargs["inner_states"])
    else:
        kwargs["species"] = "ch4"
    with pytest.raises(ValueError):
        prepare_co2_supplied_baseline(prepared, baseline, **kwargs)


def test_removes_all_boundary_operands_and_rejects_boundary_sampling():
    prepared, baseline, kwargs = _case()
    inputs = prepared.inv_inputs.assign(H_bc=(("bc_region", "nmeasure"), np.ones((1, 3))))
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    result = prepare_co2_supplied_baseline(prepared, baseline, **kwargs)
    assert "H_bc" not in result.inv_inputs
    assert "bc_region" not in result.inv_inputs.dims
    with pytest.raises(ValueError, match="sampled boundary"):
        co2_supplied_baseline_input_names(result.inv_inputs, use_bc=True)


@pytest.mark.parametrize("gathered", [False, True])
def test_independent_multisector_native_response_and_absolute_mean(gathered):
    from openghg_inversions.rhime.baseline import prepare_supplied_baseline
    from openghg_inversions.rhime.multisector import build_multisector_model, materialize_multisector_inputs
    from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec
    from test_prepared_inputs_serialisation import _multisource_prepared_inputs

    original = _multisource_prepared_inputs()
    inputs = original.inv_inputs.assign_attrs(original.inv_inputs.attrs)
    inputs.mf.attrs["units"] = "ppm"
    if gathered:
        inputs = inputs.drop_dims(("source", "region"))
        inputs = inputs.assign_coords(xr.Coordinates.from_pandas_multiindex(
            original.basis_functions.operator.basis_matrix.indexes["region"], "region"))
        inputs["H"] = (("region", "nmeasure"), np.ones((4, 1)))
    original = replace(original, inv_inputs=inputs)
    observations = original.inv_inputs.mf
    flux = original.basis_functions.flux
    native = xr.DataArray(
        da.from_array([[[[1.], [30.]]], [[[2.], [40.]]]]),
        dims=("source", "lat", "lon", "nmeasure"),
        coords={**flux.coords, **observations.coords}, attrs={"units": "ppm"},
    )
    support = xr.DataArray([[True, False]], dims=("lat", "lon"), coords={d: flux[d] for d in ("lat", "lon")})
    result = prepare_supplied_baseline(
        original, xr.full_like(observations, 8.).assign_attrs(units="ppm", species="co2"),
        native_response=native, inner_support=support,
        inner_fixed_contribution=xr.full_like(observations, .75).assign_attrs(units="ppm"),
        baseline_site_metadata=original.site_metadata, species="co2", provenance="independent background",
        constituents=("boundary", "exterior_flux"),
    )
    assert hasattr(result.inv_inputs.H.data, "__dask_graph__")
    np.testing.assert_allclose(to_dense(result.inv_inputs.H).values.reshape(-1), [1., 0., 0., 2.])
    spec = RhimeModelSpec(
        species="co2", domain="EUROPE", use_bc=False, likelihood=FixedErrorSettings(),
        sectors=tuple(SectorSpec(s, s, {"pdf": "normal", "mu": 1., "sigma": .2}, s) for s in ("B", "A")),
    )
    materialized, priors = materialize_multisector_inputs(result, spec)
    built = build_multisector_model(materialized, spec, priors=priors)
    model = built.model
    np.testing.assert_allclose(model.compile_fn(model["mu_baseline"], point_fn=False)(), 8.)
    residual = materialized.drop_vars(["supplied_baseline", "inner_fixed_contribution"])
    residual["mf"] = residual.mf - 8.75
    reference = build_multisector_model(residual, spec, priors=priors).model
    point = model.initial_point()
    np.testing.assert_allclose(model.compile_logp()(point), reference.compile_logp()(reference.initial_point()))
    np.testing.assert_allclose(model.compile_dlogp()(point), reference.compile_dlogp()(reference.initial_point()))
    with pytest.raises(ValueError, match="add_offset"):
        materialize_multisector_inputs(result, replace(spec, add_offset=True))


def test_coherent_h_may_retain_only_nonzero_states_of_larger_basis():
    prepared, baseline, kwargs = _case(multisource=True)
    inputs = prepared.inv_inputs.isel(region=[1, 2, 3], region_cov=[1, 2, 3])
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    kwargs["inner_states"] = xr.ones_like(inputs.alpha_prior_mean, dtype=bool)
    kwargs["inner_states"].data = np.array([True, True, False])
    result = prepare_co2_supplied_baseline(prepared, baseline, **kwargs)
    np.testing.assert_allclose(result.inv_inputs.H, inputs.H.isel(region=[0, 1]))
    np.testing.assert_array_equal(result.inv_inputs.original_region_in_source, [1, 0])


def test_inherited_low_rank_payload_stays_borrowed():
    prepared, baseline, kwargs = _case()
    inputs = prepared.inv_inputs.drop_vars("aggregation_error_covariance").drop_dims("nmeasure_cov")
    inputs["low_rank_factor"] = xr.DataArray(
        da.ones((3, 1)), dims=("nmeasure", "agg_rank"), attrs={"units": "ppm"},
    )
    inputs["diagonal_residual_variance"] = xr.DataArray(
        da.ones(3), dims="nmeasure", attrs={"units": "ppm**2"},
    )
    prepared = Co2PreparedInputs(replace(prepared.rhime_inputs, inv_inputs=inputs), "low_rank")
    result = prepare_co2_supplied_baseline(prepared, baseline, **kwargs)
    for name in ("low_rank_factor", "diagonal_residual_variance"):
        assert result.inv_inputs[name].data is prepared.inv_inputs[name].data
