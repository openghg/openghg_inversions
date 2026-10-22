"""Focused tests for the experimental Ramsden et al. two-gas model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.experimental.ramsden2022 import (
    RamsdenChannelSpec,
    RamsdenModelSpec,
    RamsdenPreparedInputs,
    RamsdenSectorSpec,
    build_ramsden_model,
    run_ramsden_from_prepared_inputs,
)
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.models import get_coord_registry
from openghg_inversions.rhime import RhimeSampler


X_PRIOR = {"pdf": "normal", "mu": [2.0, 3.0], "sigma": 0.5}
SIGMA_PRIOR = {"pdf": "halfnormal", "sigma": 0.2}


def _channel_inputs(
    sensitivities: dict[str, list[list[float]]],
    *,
    observation_count: int,
    state: tuple[str, ...] = ("west", "east"),
    with_bc: bool = False,
    units: str = "ppm",
) -> xr.Dataset:
    """Create a tiny canonical channel with labelled sources and observations."""
    sources = tuple(sensitivities)
    h = np.stack([sensitivities[source] for source in sources], axis=-1)
    time = np.datetime64("2020-01-01") + np.arange(observation_count).astype("timedelta64[D]")
    coords = {
        "region": np.asarray(state),
        "nmeasure": np.arange(observation_count),
        "source": np.asarray(sources),
        "time": ("nmeasure", time),
        "site": ("nmeasure", np.where(np.arange(observation_count) % 2, "AAA", "BBB")),
    }
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "H": (("nmeasure", "region", "source"), h),
        "mf": (("nmeasure",), np.zeros(observation_count)),
        "mf_error": (("nmeasure",), np.full(observation_count, 0.1)),
    }
    if with_bc:
        data_vars["H_bc"] = (
            ("nmeasure", "bc_region"),
            np.ones((observation_count, 1)),
        )
    result = xr.Dataset(data_vars, coords=coords)
    for name in ("mf", "mf_error", "min_error", "H", "H_bc"):
        if name in result:
            result[name].attrs["units"] = units
    return result


def _prepared_inputs(
    *,
    tracer_state: tuple[str, ...] = ("west", "east"),
    primary_bc: bool = False,
    tracer_bc: bool = False,
) -> RamsdenPreparedInputs:
    """Create unequal-axis methane and ethane inputs for two sectors."""
    primary = _channel_inputs(
        {
            "ff_ch4": [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            "bio_ch4": [[0.5, 0.0], [0.0, 1.0], [0.5, 0.5]],
        },
        observation_count=3,
        with_bc=primary_bc,
        units="ppm",
    )
    tracer = _channel_inputs(
        {"ff_c2h6": [[4.0, 0.0], [0.0, 5.0]]},
        observation_count=2,
        state=tracer_state,
        with_bc=tracer_bc,
        units="ppb",
    )
    return RamsdenPreparedInputs(
        primary=primary,
        tracer=tracer,
        tracer_design_reference_ratios={"ff_c2h6": None},
    )


def _model_spec(
    *,
    fixed_ratio: float | None = 0.25,
    ratio_prior: dict[str, object] | None = None,
    ratio_resolution: str = "spatial",
    reference_ratio: float | None = None,
    primary_bc: bool = False,
    tracer_bc: bool = False,
) -> RamsdenModelSpec:
    """Create the two-sector paper-shaped model specification."""
    return RamsdenModelSpec(
        primary=RamsdenChannelSpec(
            "ch4",
            "ppm",
            sigma_prior=SIGMA_PRIOR,
            use_bc=primary_bc,
            bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        ),
        tracer=RamsdenChannelSpec(
            "c2h6",
            "ppb",
            sigma_prior=SIGMA_PRIOR,
            use_bc=tracer_bc,
            bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        ),
        sectors=(
            RamsdenSectorSpec(
                "fossil",
                "ff_ch4",
                X_PRIOR,
                tracer_flux_source="ff_c2h6",
                fixed_ratio=fixed_ratio,
                ratio_prior=ratio_prior,
                ratio_resolution=ratio_resolution,  # type: ignore[arg-type]
                reference_ratio=reference_ratio,
            ),
            RamsdenSectorSpec("biogenic", "bio_ch4", X_PRIOR),
        ),
    )


def _initial_values(model: pm.Model, *variables: str) -> dict[str, np.ndarray]:
    """Evaluate requested variables together at one PyMC initial point."""
    function = model.compile_fn(
        model.replace_rvs_by_values([model[variable] for variable in variables]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return {
        variable: np.asarray(value)
        for variable, value in zip(variables, function(model.initial_point()), strict=True)
    }


def test_two_sector_equations_share_x_and_only_fossil_emits_tracer() -> None:
    """Both gases share fossil x while the biogenic state affects methane only."""
    prepared = _prepared_inputs()
    model = build_ramsden_model(prepared, _model_spec())
    values = _initial_values(model, "x_fossil", "mu_ch4_fossil", "mu_ch4_biogenic", "mu_ch4", "mu_c2h6")

    np.testing.assert_allclose(values["x_fossil"], [2.0, 3.0])
    np.testing.assert_allclose(values["mu_ch4_fossil"], [2.0, 6.0, 5.0])
    np.testing.assert_allclose(values["mu_ch4_biogenic"], [1.0, 3.0, 2.5])
    np.testing.assert_allclose(values["mu_ch4"], [3.0, 9.0, 7.5])
    np.testing.assert_allclose(values["mu_c2h6"], [2.0, 3.75])
    assert "mu_c2h6_biogenic" not in model.named_vars
    assert "emission_ratio_biogenic" not in model.named_vars


def test_differently_retained_channel_sensitivities_use_distinct_backend_axes() -> None:
    """Shared states allow channel-specific exact-zero sensitivity columns."""
    prepared = _prepared_inputs()
    primary = prepared.primary.copy(deep=True).assign_coords(
        basis_group=("region", ["outer", "inner"])
    )
    tracer = prepared.tracer.copy(deep=True).assign_coords(
        basis_group=("region", ["outer", "inner"])
    )
    primary["H"].loc[{"source": "ff_ch4", "region": "west"}] = 0.0
    tracer["H"].loc[{"source": "ff_c2h6", "region": "east"}] = 0.0
    prepared = RamsdenPreparedInputs(
        primary=primary,
        tracer=tracer,
        tracer_design_reference_ratios=prepared.tracer_design_reference_ratios,
    )

    model = build_ramsden_model(prepared, _model_spec())

    assert model.named_vars_to_dims["hx_fossil"] == (
        "nmeasure_ch4",
        "region_retained_fossil_ch4",
    )
    assert model.named_vars_to_dims["hx_c2h6_fossil"] == (
        "nmeasure_c2h6",
        "region_retained_fossil_c2h6",
    )
    registry = get_coord_registry(model)
    assert registry is not None
    np.testing.assert_array_equal(
        registry.auxiliary_coords["basis_group_retained_fossil_ch4"], ["inner"]
    )
    np.testing.assert_array_equal(
        registry.auxiliary_coords["basis_group_retained_fossil_c2h6"], ["outer"]
    )
    values = _initial_values(model, "mu_ch4_fossil", "mu_c2h6_fossil")
    np.testing.assert_allclose(values["mu_ch4_fossil"], [0.0, 6.0, 3.0])
    np.testing.assert_allclose(values["mu_c2h6_fossil"], [2.0, 0.0])


def test_direct_and_reference_ratio_multiplier_contracts_are_equivalent() -> None:
    """A pre-scaled reference inventory and multiplier match a direct ratio."""
    direct = build_ramsden_model(_prepared_inputs(), _model_spec(fixed_ratio=0.2))
    compatible_inputs = _prepared_inputs()
    compatible_inputs = RamsdenPreparedInputs(
        primary=compatible_inputs.primary,
        tracer=compatible_inputs.tracer.assign(H=compatible_inputs.tracer.H * 0.1),
        tracer_design_reference_ratios={"ff_c2h6": 0.1},
    )
    compatible = build_ramsden_model(
        compatible_inputs,
        _model_spec(fixed_ratio=2.0, reference_ratio=0.1),
    )
    direct_values = _initial_values(direct, "mu_c2h6")
    compatible_values = _initial_values(compatible, "mu_c2h6", "emission_ratio_fossil", "ratio_multiplier_fossil")

    np.testing.assert_allclose(
        direct_values["mu_c2h6"],
        compatible_values["mu_c2h6"],
    )
    np.testing.assert_allclose(compatible_values["emission_ratio_fossil"], 0.2)
    np.testing.assert_allclose(compatible_values["ratio_multiplier_fossil"], 2.0)


@pytest.mark.parametrize(
    ("resolution", "expected_dims", "expected_shape"),
    [
        ("scalar", (), ()),
        ("spatial", ("region",), (2,)),
    ],
)
def test_ratio_resolution_controls_labelled_parameter_shape(
    resolution: str,
    expected_dims: tuple[str, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Scalar ratios broadcast while spatial ratios retain the state dimension."""
    model = build_ramsden_model(
        _prepared_inputs(),
        _model_spec(fixed_ratio=0.25, ratio_resolution=resolution),
    )

    assert model.named_vars_to_dims.get("emission_ratio_fossil", ()) == expected_dims
    values = _initial_values(model, "emission_ratio_fossil")
    assert values["emission_ratio_fossil"].shape == expected_shape


def test_unequal_observation_axes_have_namespaced_error_models() -> None:
    """Independent observation grids receive distinct likelihood and sigma names."""
    model = build_ramsden_model(_prepared_inputs(), _model_spec())

    assert model.named_vars_to_dims["y_ch4"] == ("nmeasure_ch4",)
    assert model.named_vars_to_dims["y_c2h6"] == ("nmeasure_c2h6",)
    assert model.named_vars_to_dims["sigma_ch4"] == ("nsigma_site_ch4", "nsigma_period_ch4")
    assert model.named_vars_to_dims["sigma_c2h6"] == ("nsigma_site_c2h6", "nsigma_period_c2h6")
    values = _initial_values(model, "epsilon_ch4", "epsilon_c2h6")
    assert values["epsilon_ch4"].shape == (3,)
    assert values["epsilon_c2h6"].shape == (2,)
    for base in ("Y", "error", "min_error", "sigma_site_index", "sigma_period_index"):
        assert f"{base}_ch4" in model.named_vars
        assert f"{base}_c2h6" in model.named_vars


def test_absolute_error_equation_and_minimum_floor() -> None:
    """Epsilon follows the paper's absolute error equation plus RHIME's floor."""
    prepared = _prepared_inputs()
    model = build_ramsden_model(prepared, _model_spec())
    values = _initial_values(model, "sigma_ch4", "epsilon_ch4")
    sigma = values["sigma_ch4"]
    expected = np.sqrt(0.1**2 + sigma[[0, 1, 0], 0] ** 2)
    np.testing.assert_allclose(values["epsilon_ch4"], expected)

    floored_primary = prepared.primary.copy()
    floored_primary["min_error"] = xr.full_like(prepared.primary.mf, 0.5).assign_attrs(units="ppm")
    floored = RamsdenPreparedInputs(
        primary=floored_primary,
        tracer=prepared.tracer,
        tracer_design_reference_ratios=prepared.tracer_design_reference_ratios,
    )
    floored_model = build_ramsden_model(floored, _model_spec())
    floored_values = _initial_values(floored_model, "epsilon_ch4")
    np.testing.assert_allclose(floored_values["epsilon_ch4"], 0.5)


def test_state_coordinate_mismatch_is_rejected() -> None:
    """Shared states must have identical primary and tracer coordinate labels."""
    with pytest.raises(ValueError, match="state coordinates must match exactly"):
        build_ramsden_model(
            _prepared_inputs(tracer_state=("east", "west")),
            _model_spec(),
        )


def test_equal_positional_labels_with_different_basis_maps_are_rejected() -> None:
    """Matching region numbers cannot hide different channel basis geometry."""
    prepared = _prepared_inputs(tracer_state=("west", "east"))
    flux = xr.DataArray(
        np.ones((2, 1)),
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": [0.0]},
        attrs={"units": "mol/m2/s"},
    )
    primary_basis = BasisFunctions.from_flat_basis(
        xr.DataArray([[0], [1]], dims=("lat", "lon"), coords=flux.coords),
        flux,
    )
    tracer_basis = BasisFunctions.from_flat_basis(
        xr.DataArray([[1], [0]], dims=("lat", "lon"), coords=flux.coords),
        flux,
    )
    prepared = RamsdenPreparedInputs(
        primary=prepared.primary.assign_coords(region=[0, 1]),
        tracer=prepared.tracer.assign_coords(region=[0, 1]),
        tracer_design_reference_ratios=prepared.tracer_design_reference_ratios,
        primary_basis_functions=primary_basis,
        tracer_basis_functions=tracer_basis,
    )

    with pytest.raises(ValueError, match="spatial basis maps must match exactly"):
        build_ramsden_model(prepared, _model_spec())


def test_units_and_tracer_ratio_provenance_are_validated() -> None:
    """Unit-scale and pre-scaled tracer declarations reject silent factor errors."""
    prepared = _prepared_inputs()
    numeric_units = RamsdenPreparedInputs(
        primary=prepared.primary,
        tracer=prepared.tracer.assign(mf=prepared.tracer.mf.assign_attrs(units="1e-9")),
        tracer_design_reference_ratios=prepared.tracer_design_reference_ratios,
    )
    build_ramsden_model(numeric_units, _model_spec())

    wrong_units = RamsdenPreparedInputs(
        primary=prepared.primary,
        tracer=prepared.tracer.assign(mf=prepared.tracer.mf.assign_attrs(units="ppt")),
        tracer_design_reference_ratios=prepared.tracer_design_reference_ratios,
    )
    with pytest.raises(ValueError, match="do not match declared observation_units"):
        build_ramsden_model(wrong_units, _model_spec())

    wrong_ratio = RamsdenPreparedInputs(
        primary=prepared.primary,
        tracer=prepared.tracer,
        tracer_design_reference_ratios={"ff_c2h6": 0.1},
    )
    with pytest.raises(ValueError, match="does not match model reference_ratio"):
        build_ramsden_model(wrong_ratio, _model_spec())


def test_duplicate_sources_and_negative_ratio_support_are_rejected() -> None:
    """Non-identifiable source reuse and physically negative ratios fail early."""
    base = _model_spec()
    duplicate_sector = RamsdenSectorSpec("duplicate", "ff_ch4", X_PRIOR)
    with pytest.raises(ValueError, match="unique primary flux sources"):
        build_ramsden_model(
            _prepared_inputs(),
            RamsdenModelSpec(base.primary, base.tracer, (*base.sectors, duplicate_sector)),
        )

    with pytest.raises(ValueError, match="non-negative support"):
        build_ramsden_model(
            _prepared_inputs(),
            _model_spec(
                fixed_ratio=None,
                ratio_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
            ),
        )


@pytest.mark.parametrize(
    ("primary_bc", "tracer_bc", "expected"),
    [
        (False, False, set()),
        (True, False, {"bc_ch4", "mu_bc_ch4"}),
        (False, True, {"bc_c2h6", "mu_bc_c2h6"}),
        (True, True, {"bc_ch4", "mu_bc_ch4", "bc_c2h6", "mu_bc_c2h6"}),
    ],
)
def test_boundary_components_are_optional_per_channel(
    primary_bc: bool,
    tracer_bc: bool,
    expected: set[str],
) -> None:
    """Each gas independently opts into its own labelled boundary component."""
    model = build_ramsden_model(
        _prepared_inputs(primary_bc=primary_bc, tracer_bc=tracer_bc),
        _model_spec(primary_bc=primary_bc, tracer_bc=tracer_bc),
    )

    present = {name for name in model.named_vars if name.startswith(("bc_", "mu_bc_"))}
    assert present == expected


@pytest.mark.slow
def test_tiny_model_samples_through_rhime_sampler() -> None:
    """The isolated historical model runs through the modern RHIME sampler."""
    sampler = RhimeSampler(
        draws=2,
        tune=2,
        chains=1,
        progressbar=False,
        sample_kwargs={"random_seed": 7, "compute_convergence_checks": False, "cores": 1},
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )

    result = run_ramsden_from_prepared_inputs(
        prepared_inputs=_prepared_inputs(),
        model_spec=_model_spec(
            fixed_ratio=None,
            ratio_prior={"pdf": "uniform", "lower": 0.1, "upper": 0.4},
        ),
        sampler=sampler,
    )

    assert result.sampler is not sampler
    assert result.sampler == sampler
    posterior = result.idata["posterior"]
    log_likelihood = result.idata["log_likelihood"]
    assert posterior.sizes["chain"] == 1
    assert posterior.sizes["draw"] == 2
    assert posterior.sizes["region"] == 2
    assert {"x_fossil", "x_biogenic", "emission_ratio_fossil"} <= set(posterior)
    assert {"y_ch4", "y_c2h6"} <= set(log_likelihood)


def test_default_custom_sampler_uses_namespaced_predictive_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied sampler's standard y default is adapted before sampling."""
    sampler = RhimeSampler(draws=2, tune=2, chains=1)
    captured: dict[str, object] = {}

    def fake_sample(configured_sampler: RhimeSampler, model: pm.Model) -> xr.DataTree:
        captured["predictive"] = configured_sampler.sample_posterior_predictive
        captured["model"] = model
        return xr.DataTree()

    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    result = run_ramsden_from_prepared_inputs(
        prepared_inputs=_prepared_inputs(),
        model_spec=_model_spec(),
        sampler=sampler,
    )

    assert captured["predictive"] == ("y_ch4", "y_c2h6")
    assert result.sampler is not sampler


def _assembled_channel(sources: tuple[str, ...], sites: dict[str, list[float]], units: str) -> xr.Dataset:
    """Assemble retained observations through the actual raw preparation API."""
    datasets = {}
    for site, values in sites.items():
        h = np.ones((2, len(values), len(sources)))
        h[:, -1, :] = np.nan
        datasets[site] = xr.Dataset(
            {
                "H": (("region", "time", "source"), h, {"units": units}),
                "mf": ("time", values, {"units": units}),
                "mf_mod": ("time", np.zeros(len(values)), {"units": units}),
                "mf_error": ("time", np.full(len(values), 0.1), {"units": units}),
                "mf_repeatability": ("time", np.full(len(values), 0.1)),
                "mf_variability": ("time", np.zeros(len(values))),
            },
            coords={
                "region": ["west", "east"],
                "source": list(sources),
                "time": np.array(["2020-01-02", "2020-01-01", "2020-01-03"], dtype="datetime64[ns]"),
            },
        )
    return make_inv_inputs(datasets)


@pytest.mark.parametrize("calculated", [False, True])
def test_raw_preparation_builds_distinct_channels_with_likelihood_owned_floors(calculated: bool) -> None:
    """Real assembly drops invalid H rows before each gas chooses its floor."""
    primary = _assembled_channel(
        ("ff_ch4", "bio_ch4"), {"BBB": [6., 2., 1000.], "AAA": [12., 4., 2000.]}, "ppm",
    )
    tracer = _assembled_channel(("ff_c2h6",), {"CCC": [9., 3., 999.]}, "ppb")
    assert "min_error" not in primary and "min_error" not in tracer
    assert "site_indicator" not in primary and "site_names" not in primary
    assert "site_indicator" not in tracer and "site_names" not in tracer
    original_primary = primary.copy(deep=True)
    original_tracer = tracer.copy(deep=True)
    spec = _model_spec()
    if calculated:
        spec = replace(
            spec,
            primary=replace(spec.primary, min_error="residual", min_error_options={"by_site": True}),
            tracer=replace(spec.tracer, min_error="percentile"),
        )
    model = build_ramsden_model(RamsdenPreparedInputs(primary, tracer, {"ff_c2h6": None}), spec)

    values = _initial_values(model, "min_error_ch4", "min_error_c2h6", "epsilon_ch4", "epsilon_c2h6")
    expected_primary = [2., 2., 4., 4.] if calculated else np.zeros(4)
    expected_tracer = [2.7, 2.7] if calculated else np.zeros(2)
    np.testing.assert_allclose(values["min_error_ch4"], expected_primary)
    np.testing.assert_allclose(values["min_error_c2h6"], expected_tracer)
    if calculated:
        np.testing.assert_allclose(values["epsilon_ch4"], expected_primary)
        np.testing.assert_allclose(values["epsilon_c2h6"], expected_tracer)
    registry = get_coord_registry(model)
    assert registry is not None
    np.testing.assert_array_equal(registry.original_coords["nsigma_site_ch4"], ["BBB", "AAA"])
    np.testing.assert_array_equal(registry.original_coords["nsigma_site_c2h6"], ["CCC"])
    assert model.named_vars_to_dims["y_ch4"] == ("nmeasure_ch4",)
    assert model.named_vars_to_dims["y_c2h6"] == ("nmeasure_c2h6",)
    xr.testing.assert_identical(primary, original_primary)
    xr.testing.assert_identical(tracer, original_tracer)


def test_explicit_channel_floor_overrides_previously_supplied_dataset_floor() -> None:
    """The likelihood's explicit floor choice wins over a historical final floor."""
    prepared = _prepared_inputs()
    prepared = replace(prepared, primary=prepared.primary.assign(min_error=xr.full_like(prepared.primary.mf, 9.)))
    spec = _model_spec()
    model = build_ramsden_model(prepared, replace(spec, primary=replace(spec.primary, min_error={"BBB": .5, "AAA": .8})))
    np.testing.assert_allclose(_initial_values(model, "min_error_ch4")["min_error_ch4"], [.5, .8, .5])


@pytest.mark.parametrize(("method", "operand"), [("residual", "mf_mod"), ("percentile", "mf")])
def test_calculated_floor_requires_eager_operands_without_computing_them(method: str, operand: str) -> None:
    """Named estimators require explicitly materialized concentration operands."""
    import dask.array as da
    from dask import delayed

    @delayed
    def unused():
        raise AssertionError("The builder must not execute the lazy floor dependency")

    prepared = _prepared_inputs()
    prepared = replace(
        prepared,
        primary=prepared.primary.assign({operand: xr.DataArray(
            da.from_delayed(unused(), shape=(3,), dtype=float),
            dims="nmeasure", coords=prepared.primary.mf.coords, attrs={"units": "ppm"},
        )}),
    )
    spec = _model_spec()
    with pytest.raises(ValueError, match="minimum-error estimation requires eager observations"):
        build_ramsden_model(prepared, replace(spec, primary=replace(spec.primary, min_error=method)))


@pytest.mark.parametrize("options", [{"unknown": True}, {"by_site": "yes"}])
def test_channel_floor_options_are_validated(options: dict[str, object]) -> None:
    """Malformed likelihood options fail even when using the default floor."""
    spec = _model_spec()
    with pytest.raises(ValueError, match="min_error_options"):
        build_ramsden_model(_prepared_inputs(), replace(spec, primary=replace(spec.primary, min_error_options=options)))


@pytest.mark.parametrize("floor", [None, 0.0, 0.5])
def test_lazy_observations_accept_default_and_scalar_floors(floor: float | None) -> None:
    """Direct floor choices keep lazy observations valid at the PyMC boundary."""
    import dask.array as da

    prepared = _prepared_inputs()
    lazy_primary = prepared.primary.copy()
    lazy_primary["mf"] = prepared.primary.mf.copy(data=da.from_array(np.zeros(3), chunks=2))
    lazy_prepared = replace(prepared, primary=lazy_primary)
    spec = _model_spec()
    model = build_ramsden_model(lazy_prepared, replace(spec, primary=replace(spec.primary, min_error=floor)))
    values = _initial_values(model, "Y_ch4", "min_error_ch4")
    np.testing.assert_array_equal(values["Y_ch4"], np.zeros(3))
    np.testing.assert_array_equal(values["min_error_ch4"], np.full(3, floor or 0.0))
    assert isinstance(lazy_prepared.primary.mf.data, da.Array)
