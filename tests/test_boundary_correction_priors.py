"""Independent equations, physical coefficients, and ordinary/cached parity."""

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
from scipy.linalg import helmert
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import restore_absolute_predictives
from openghg_inversions.models.boundary_corrections import (
    add_boundary_correction,
    calibrate_pattern_width,
    correction_design,
    functional_sd,
    materialize_boundary_correction,
    prepare_boundary_correction,
)
from openghg_inversions.models.coords import (
    registered_model,
    get_coord_registry,
    restore_inferencedata_coords,
)
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_model import build_co2_model
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
)
from openghg_inversions.array_ops import select_gathered_data_array
from test_co2_cached_sigma import _inputs
from test_co2_o2_fixed_ou import _prepared, _kwargs


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def operands(obs, c=400.0):
    row = str(obs.dims[0])
    n = obs.size
    m = xr.DataArray(
        np.resize([[0.6, 0.3], [0.2, 0.9], [0.5, 0.3], [0.8, 0.4]], (n, 2)),
        dims=(row, "bc_region"),
        coords={row: obs[row], "bc_region": ["N", "S"]},
        attrs={"units": "1"},
    )
    a = m.copy(data=np.arange(2 * n).reshape(n, 2) * 0.01 - 0.025).assign_attrs(units="ppm")
    return xr.Dataset(
        {"H_bc_anomaly": a, "H_bc_mass": m, "bc_reference": xr.DataArray(c, attrs={"units": "ppm"})},
        attrs={"boundary_representation": "centered-v1"},
    )


@pytest.mark.parametrize("method", ["level", "A", "curtain_contrasts", "independent_curtains"])
def test_independent_operator_equations_and_exact_covariance(method):
    obs = _inputs().mf
    raw = operands(obs)
    args = {"level_sd": 5.0} if method != "independent_curtains" else {}
    if method != "level":
        args["pattern_sd"] = 0.7
    prepared = materialize_boundary_correction(
        prepare_boundary_correction(raw, obs, method=method, **args), obs
    )
    m, a = raw.H_bc_mass.values, raw.H_bc_anomaly.values
    g = m.sum(axis=1)
    pattern = {
        "A": a,
        "curtain_contrasts": m @ helmert(2, full=False).T,
        "independent_curtains": m,
        "level": np.empty((4, 0)),
    }[method]
    expected = np.column_stack(
        (g[:, None] if method != "independent_curtains" else np.empty((4, 0)), pattern)
    )
    np.testing.assert_allclose(correction_design(prepared), expected, rtol=0, atol=1e-15)
    widths = np.r_[np.full(prepared.sizes["bc_level_term"], 5), np.full(pattern.shape[1], 0.7)]
    factor = expected * widths
    # Symmetric deterministic points have exactly identity covariance, without Monte Carlo error.
    k = factor.shape[1]
    epsilon = np.r_[np.eye(k), -np.eye(k)] * np.sqrt(k)
    draws = epsilon @ factor.T
    np.testing.assert_allclose(draws.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(draws.T @ draws / (2 * k), factor @ factor.T, atol=1e-14)
    labelled = xr.DataArray(factor, dims=("nmeasure", "coefficient"), coords={"nmeasure": obs.nmeasure})
    mean = xr.ones_like(obs) / obs.size
    assert float(functional_sd(labelled, mean, observation_dim="nmeasure")) == pytest.approx(
        np.linalg.norm(factor.mean(axis=0))
    )
    difference = xr.zeros_like(obs).copy(data=[-1.0, 1.0, 0.0, 0.0])
    assert float(functional_sd(labelled, difference, observation_dim="nmeasure")) == pytest.approx(
        np.linalg.norm(factor[1] - factor[0])
    )
    with registered_model() as model:
        result = add_boundary_correction(prepared)
    assert "bc" not in model
    point = model.initial_point()
    for name in point:
        point[name] = np.full_like(point[name], -1.3)  # A permits negative 1+v.
    physical = raw.H_bc_anomaly.sum("bc_region") + 400 * raw.H_bc_mass.sum("bc_region")
    absolute, centered = model.compile_fn(
        model.replace_rvs_by_values([result.output, result.likelihood_contribution]), inputs=model.value_vars
    )(point)
    np.testing.assert_allclose(absolute, physical - 1.3 * expected.sum(axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        absolute - obs, centered - (obs - prepared.bc_reference_shift), rtol=0, atol=1e-12
    )


def test_iid_curtains_helmert_equivalence_requires_specific_widths():
    obs = _inputs().mf
    raw = operands(obs)
    iid = prepare_boundary_correction(raw, obs, method="independent_curtains", pattern_sd=1.2)
    contrast = prepare_boundary_correction(
        raw, obs, method="curtain_contrasts", level_sd=1.2 / np.sqrt(2), pattern_sd=1.2
    )
    fi = correction_design(iid) * 1.2
    fc = correction_design(contrast) * np.array([1.2 / np.sqrt(2), 1.2])
    np.testing.assert_allclose(fi @ fi.T, fc @ fc.T, atol=1e-15)
    assert not np.allclose(fi @ fi.T, correction_design(contrast) @ correction_design(contrast).T)


@pytest.mark.parametrize("period_level", [False, True])
def test_period_support_scalar_reference_and_roundtrip(tmp_path, period_level):
    obs = _inputs().mf.assign_coords(
        time=("nmeasure", pd.to_datetime(["2021-07-01", "2021-07-09", "2021-07-17", "2021-07-25"]))
    )
    obs = obs.set_index(nmeasure=["site", "time"])
    raw = operands(obs, 209300.0)
    kwargs = dict(method="A", level_sd=5.0, pattern_sd=0.8, reference="whole_window_transport_mean")
    monthly = prepare_boundary_correction(raw, obs, **kwargs)
    periods = prepare_boundary_correction(
        raw, obs, **kwargs, frequency="8D", anchor_time="2021-07-01", level_by_period=period_level
    )
    assert float(monthly.bc_reference) == float(periods.bc_reference)
    assert periods.sizes["bc_level_term"] == (4 if period_level else 1)
    assert periods.sizes["bc_pattern_term"] == 8  # All K pattern coefficients per period.
    np.testing.assert_array_equal(periods.bc_period_index, range(4))
    np.testing.assert_allclose(
        (periods.bc_pattern_operator**2).sum("bc_pattern_term"),
        (monthly.bc_pattern_operator**2).sum("bc_pattern_term"),
        atol=1e-18,
    )
    np.testing.assert_allclose(
        periods.bc_prescribed_centered + periods.bc_reference_shift,
        raw.H_bc_anomaly.sum("bc_region") + raw.bc_reference * raw.H_bc_mass.sum("bc_region"),
        atol=3e-11,
        rtol=0,
    )
    expected_c = float(raw.bc_reference) + float(raw.H_bc_anomaly.sum() / raw.H_bc_mass.sum())
    assert float(periods.bc_reference) == expected_c
    assert periods.bc_prescribed_centered.dtype == np.float64
    path = tmp_path / "correction.nc"
    encode_multiindexes_for_storage(periods).to_netcdf(path)
    with xr.open_dataset(path) as opened:
        replay = restore_serialisation_multiindexes(opened.load(), strict=True)
    xr.testing.assert_identical(
        materialize_boundary_correction(replay, obs), materialize_boundary_correction(periods, obs)
    )
    reordered = periods.isel(nmeasure=[1, 0, 2, 3])
    with pytest.raises(ValueError, match="align"):
        materialize_boundary_correction(reordered, obs)


def test_reference_change_preserves_physical_bc_but_changes_A_covariance():
    obs = _inputs().mf
    raw = operands(obs, 209300.0)
    p0 = prepare_boundary_correction(raw, obs, method="A", level_sd=5, pattern_sd=0.8)
    p1 = prepare_boundary_correction(raw, obs, method="A", level_sd=5, pattern_sd=0.8, reference=209301.0)
    np.testing.assert_allclose(
        p0.bc_prescribed_centered + p0.bc_reference_shift,
        p1.bc_prescribed_centered + p1.bc_reference_shift,
        rtol=0,
        atol=3e-11,
    )
    assert not np.allclose(p0.bc_pattern_operator, p1.bc_pattern_operator)


def test_preparation_preserves_laziness_and_shared_graph():
    obs = _inputs().mf
    raw = operands(obs)
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return raw.H_bc_anomaly.values

    lazy = raw.assign(
        H_bc_anomaly=raw.H_bc_anomaly.copy(data=da.from_delayed(payload(), shape=(4, 2), dtype=float))
    )
    prepared = prepare_boundary_correction(
        lazy, obs, method="A", level_sd=5.0, pattern_sd=0.8, reference="whole_window_transport_mean"
    )
    assert calls == []
    materialize_boundary_correction(prepared, obs)
    assert calls == [1]


@pytest.mark.parametrize("method", ["level", "A", "curtain_contrasts"])
def test_single_ordinary_cached_density_and_gradient_at_nonzero_coefficients(method):
    inv = _inputs()
    raw = operands(inv.mf)
    kw = dict(method=method, level_sd=0.2)
    if method != "level":
        kw["pattern_sd"] = 0.3
    prepared = prepare_boundary_correction(raw, inv.mf, **kw)
    kwargs = dict(
        retained_prior=CorrelatedLognormalPrior(
            inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf + prepared.bc_reference_shift,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, "dense"),
        boundary_correction=prepared,
    )
    ou = {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
    cached = build_co2_cached_sigma_model(inv.H, **kwargs, **ou)
    stock = build_co2_model(
        inv.H, **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=ou
    )
    _density_parity(cached, stock)


def _density_parity(cached, stock):
    model = cached.model
    point = model.initial_point()
    for name in point:
        if name.endswith(("bc_level", "bc_pattern")):
            point[name] = np.full_like(point[name], 0.17)
    amplitude = np.exp(point["ou_site_amplitude_log__"])
    cached.shared_cache.update(cached.target.refresh(amplitude))
    quadratic = model.compile_fn(
        model.replace_rvs_by_values([cached.cached_state_logp]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)[0]
    np.testing.assert_allclose(
        model.compile_logp(vars=model.observed_RVs)(point), quadratic, atol=1e-8, rtol=1e-10
    )
    np.testing.assert_allclose(
        model.compile_logp()(point), stock.compile_logp()(point), atol=1e-8, rtol=1e-10
    )
    np.testing.assert_allclose(
        model.compile_dlogp()(point), stock.compile_dlogp()(point), atol=1e-8, rtol=1e-10
    )


@pytest.mark.parametrize("method", ["level", "A", "curtain_contrasts"])
def test_linked_ordinary_cached_channel_independence_and_o2_precision(method):
    kwargs = _kwargs(_prepared(), interleaved=True)
    corrections = {}
    for channel, c, width in (("co2", 400.0, 0.2), ("o2", 209300.0, 0.003)):
        row = str(kwargs[f"{channel}_sensitivity"].dims[0])
        obs = select_gathered_data_array(
            kwargs["observations"],
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row)
        raw = operands(obs, c)
        kw = dict(method=method, level_sd=width)
        if method != "level":
            kw["pattern_sd"] = 0.3
        corrections[channel] = prepare_boundary_correction(raw, obs, **kw)
    kwargs["boundary_correction"] = corrections
    ou = {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
    cached = build_co2_o2_cached_sigma_model(**kwargs, **ou)
    stock = build_co2_o2_model(**kwargs, **ou)
    _density_parity(cached, stock)
    assert "co2_bc" not in cached.model and "o2_bc" not in cached.model
    point = cached.model.initial_point()
    point["o2_bc_level"] = np.array([0.000123])
    expressions = cached.model.replace_rvs_by_values(
        [cached.model["o2_mu_bc_level"], cached.model["co2_mu_bc_level"]]
    )
    oxygen, carbon = cached.model.compile_fn(
        expressions, inputs=cached.model.value_vars, on_unused_input="ignore"
    )(point)
    rows = kwargs["observations"].species.values == "o2"
    np.testing.assert_allclose(
        oxygen[rows], corrections["o2"].bc_mass_response * 0.000123, rtol=0, atol=1e-18
    )
    np.testing.assert_array_equal(oxygen[~rows], 0.0)
    np.testing.assert_array_equal(carbon, 0.0)


def test_calibration_is_explicit_and_predictive_reference_is_restored_once():
    obs = _inputs().mf
    raw = operands(obs)
    p = prepare_boundary_correction(raw, obs, method="A", level_sd=5.0, pattern_sd=0.8)
    width = calibrate_pattern_width(p.bc_pattern_operator, 1.0, coefficient_dim="bc_pattern_term")
    assert np.median(np.linalg.norm(p.bc_pattern_operator.values * width, axis=1)) == pytest.approx(1.0)
    assert float(p.bc_pattern_sd) == 0.8
    trace = xr.DataTree.from_dict(
        {
            "constant_data": xr.Dataset({"boundary_reference_shift": p.bc_reference_shift}),
            "prior_predictive": xr.Dataset({"y": obs}),
        }
    )
    restore_absolute_predictives(trace)
    expected = obs + p.bc_reference_shift
    xr.testing.assert_allclose(trace.prior_predictive.y, expected)
    restore_absolute_predictives(trace)
    xr.testing.assert_allclose(trace.prior_predictive.y, expected)


@pytest.mark.parametrize(
    "options",
    [
        dict(level_sd=0),
        dict(pattern_sd=None),
        dict(reference="rolling"),
        dict(frequency="8D"),
        dict(level_by_period=True),
    ],
)
def test_invalid_scientific_options_fail_at_preparation(options):
    obs = _inputs().mf
    kw = dict(method="A", level_sd=5, pattern_sd=0.8)
    kw.update(options)
    with pytest.raises(ValueError):
        prepare_boundary_correction(operands(obs), obs, **kw)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("method", ["A", "curtain_contrasts", "level", "independent_curtains"])
def test_public_co2_build_trace_replay_and_components(tmp_path, cached, method):
    from openghg_inversions.rhime.co2.inference import build_co2_inference, annotate_co2_inference
    from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations
    from openghg_inversions.serialization import load_trace, save_trace
    from test_co2_affine_output import _prepared as affine_prepared

    prepared, _ = affine_prepared()
    widths = {} if method == "independent_curtains" else {"level_sd": 5.0}
    if method != "level":
        widths["pattern_sd"] = 0.8
    correction = prepare_boundary_correction(
        operands(prepared.inv_inputs.mf), prepared.inv_inputs.mf, method=method, **widths
    )
    options = {"boundary_correction": correction, "no_model_error": True}
    if cached:
        options.pop("no_model_error")
        options.update(
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs={
                "tau_hours": 5.0,
                "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
            },
        )
    built, cache = build_co2_inference(prepared, options)
    with built.model:
        trace = pm.sample_prior_predictive(draws=4, random_seed=237)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    trace = annotate_co2_inference(trace, built, prepared, options, cache)
    trace["posterior"] = trace.prior.to_dataset().isel(draw=slice(0, 2))
    save_trace(trace, tmp_path / "trace.nc")
    replay = load_trace(tmp_path / "trace.nc")
    expected_units = "1" if method == "A" else "ppm"
    if method != "level":
        assert replay.prior.bc_pattern.attrs["units"] == expected_units
        assert replay.prior.bc_pattern.attrs["support"] == "real"
    if method != "independent_curtains":
        assert replay.prior.bc_level.attrs["units"] == "ppm"
    components = reconstruct_co2_concentrations(replay, prepared)
    if method != "level":
        assert components.boundary_pattern_coefficient_prior.attrs["units"] == expected_units
    assert components.boundary_level_contribution_prior.attrs["units"] == "ppm"
    np.testing.assert_allclose(
        components.boundary_prior,
        components.boundary_prescribed_prior
        + components.boundary_level_contribution_prior
        + components.boundary_pattern_contribution_prior,
    )
    np.testing.assert_allclose(
        components.modelled_prior, components.pollution_prior + components.boundary_prior
    )
    # Restored constants and labelled physical coefficients alone reconstruct each draw.
    physical = replay.constant_data.bc_prescribed_centered + replay.constant_data.bc_reference_shift
    for kind in ("level", "pattern"):
        if f"bc_{kind}" in replay.prior:
            physical = physical + xr.dot(
                replay.constant_data[f"bc_{kind}_operator"], replay.prior[f"bc_{kind}"], dim=f"bc_{kind}_term"
            )
    xr.testing.assert_allclose(physical.transpose(*replay.prior.mu_bc.dims), replay.prior.mu_bc)


def test_correction_unit_conversion_and_borrowed_inputs():
    obs = _inputs().mf
    raw = operands(obs)
    original = raw.copy(deep=True)
    correction = prepare_boundary_correction(raw, obs, method="A", level_sd=5, pattern_sd=0.8)
    saved = correction.copy(deep=True)
    converted_obs = obs.assign_attrs(units="mol/mol")
    converted = materialize_boundary_correction(correction, converted_obs)
    np.testing.assert_allclose(converted.bc_prescribed_centered, correction.bc_prescribed_centered * 1e-6)
    np.testing.assert_allclose(converted.bc_pattern_operator, correction.bc_pattern_operator * 1e-6)
    assert float(converted.bc_level_sd) == pytest.approx(5e-6)
    assert float(converted.bc_pattern_sd) == 0.8
    xr.testing.assert_identical(raw, original)
    xr.testing.assert_identical(correction, saved)


def test_artifact_metadata_survives_default_xarray_attribute_policy():
    obs = _inputs().mf
    with xr.set_options(keep_attrs=False):
        p = prepare_boundary_correction(operands(obs), obs, method="A", level_sd=5.0, pattern_sd=0.8)
        eager = materialize_boundary_correction(p, obs)
        assert eager.attrs == p.attrs
        assert eager.bc_pattern_sd.attrs["units"] == "1"
        assert eager.bc_pattern_term.attrs["method"] == "A"
        materialize_boundary_correction(eager, obs)


@pytest.mark.parametrize("cached", [False, True])
def test_pymc_boundary_materializes_shared_flux_and_correction_graph_once(cached):
    inv = _inputs()
    raw = operands(inv.mf)
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return inv.H.values

    shared = da.from_delayed(payload(), shape=(4, 2), dtype=float)
    flux = inv.H.copy(data=shared)
    raw = raw.assign(H_bc_anomaly=raw.H_bc_anomaly.copy(data=shared))
    correction = prepare_boundary_correction(
        raw, inv.mf, method="A", level_sd=5.0, pattern_sd=0.8, reference="whole_window_transport_mean"
    )
    kwargs = dict(
        retained_prior=CorrelatedLognormalPrior(
            inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, "dense"),
        boundary_correction=correction,
    )
    if cached:
        build_co2_cached_sigma_model(
            flux, **kwargs, tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75}
        )
    else:
        build_co2_model(flux, **kwargs)
    assert calls == [1]


@pytest.mark.parametrize("period_level", [False, True])
def test_period_coefficients_ordinary_cached_density_and_global_offset_warning(period_level):
    inv = _inputs()
    raw = operands(inv.mf)
    raw["H_bc_mass"] = (raw.H_bc_mass / raw.H_bc_mass.sum("bc_region")).assign_attrs(units="1")
    correction = prepare_boundary_correction(
        raw,
        inv.mf,
        method="A",
        level_sd=0.2,
        pattern_sd=0.3,
        frequency="2h",
        anchor_time="2021-01-01",
        level_by_period=period_level,
    )
    kwargs = dict(
        retained_prior=CorrelatedLognormalPrior(
            inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, "dense"),
        boundary_correction=correction,
        offset_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
        offset_per_site=False,
    )
    ou = {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
    with pytest.warns(UserWarning, match="combined constant"):
        cached = build_co2_cached_sigma_model(inv.H, **kwargs, **ou)
    stock_kwargs = {name: value for name, value in kwargs.items() if name != "offset_per_site"}
    with pytest.warns(UserWarning, match="combined constant"):
        stock = build_co2_model(
            inv.H,
            **stock_kwargs,
            offset_args={"per_site": False},
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs=ou,
        )
    _density_parity(cached, stock)
    assert cached.model.named_vars_to_dims["bc_level"] == ("bc_level_term",)
    assert cached.model["bc_level"].eval().size == (3 if period_level else 1)


def test_linked_prepared_runner_trace_and_basic_products():
    from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2, annotate_co2_o2_trace
    from openghg_inversions.rhime.co2.co2_o2_outputs import (
        Co2O2Output,
        Co2O2OutputOptions,
        calculate_co2_o2_products,
    )

    prepared = _prepared()
    corrections = {}
    for channel, reference in (("co2", 400.0), ("o2", 209300.0)):
        row = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        obs = select_gathered_data_array(
            prepared.observations,
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row)
        corrections[channel] = prepare_boundary_correction(
            operands(obs, reference), obs, method="A", level_sd=0.01, pattern_sd=0.1
        )
    built = build_rhime_co2_o2(
        prepared_inputs=prepared,
        boundary_correction=corrections,
        independent_error_sd=_kwargs(prepared)["independent_error_sd"],
    )
    with built.model:
        trace = pm.sample_prior_predictive(draws=4, random_seed=237)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    trace = annotate_co2_o2_trace(trace, built=built)
    trace["posterior"] = trace.prior.to_dataset()
    output = Co2O2Output(prepared, trace, "2021-01-01", "2021-02-01", "EUROPE")
    calculate_co2_o2_products(result=output, output=Co2O2OutputOptions())
    for channel in ("co2", "o2"):
        components = output.outputs[f"{channel}_concentration_components"]
        assert components.boundary_pattern_coefficient_posterior.attrs["units"] == "1"
        np.testing.assert_allclose(
            components.boundary_posterior,
            components.boundary_prescribed_posterior
            + components.boundary_level_posterior
            + components.boundary_pattern_posterior,
        )


@pytest.mark.parametrize("method", ["level", "A", "curtain_contrasts", "independent_curtains"])
def test_one_curtain_retains_A_coefficient_but_has_no_curtain_contrast(method):
    obs = _inputs().mf
    raw = operands(obs).isel(bc_region=[0])
    kwargs = {} if method == "independent_curtains" else {"level_sd": 5.0}
    if method != "level":
        kwargs["pattern_sd"] = 0.2
    p = materialize_boundary_correction(prepare_boundary_correction(raw, obs, method=method, **kwargs), obs)
    assert p.sizes["bc_pattern_term"] == (1 if method in {"A", "independent_curtains"} else 0)
    with registered_model() as model:
        add_boundary_correction(p)
    assert ("bc_pattern" in model) == (method in {"A", "independent_curtains"})
    assert np.isfinite(model.compile_logp()(model.initial_point()))
