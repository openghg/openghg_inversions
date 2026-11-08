"""Additive anomaly structures and anomaly scaling: equations, covariance and parity."""

from dataclasses import replace

import dask
import dask.array as da
import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr
from dask import delayed
from scipy.linalg import helmert

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import get_coord_registry, registered_model, restore_inferencedata_coords
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import co2_runner, inference
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_model import build_co2_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations
from test_co2_bound_frontends import _draws, _single_with_operands
from test_co2_cached_sigma import _inputs
from test_co2_o2_fixed_ou import _kwargs, _prepared
from boundary_test_utils import centered_transport, raw_transport
from openghg_inversions.models.boundary import BoundCenteredTransport

METHODS = ["level", "contrasts", "independent", "scaling"]
OU = {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _normal(sigma):
    return {"pdf": "normal", "mu": 0.0, "sigma": sigma}


def _operands(obs, c=400.0, normalised=True):
    row = str(obs.dims[0])
    n = obs.size
    mass = xr.DataArray(
        np.resize([[0.6, 0.3], [0.2, 0.9], [0.5, 0.3], [0.8, 0.4]], (n, 2)),
        dims=(row, "bc_region"), coords={row: obs[row], "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    if normalised:
        mass = (mass / mass.sum("bc_region")).assign_attrs(units="1")
    anomaly = mass.copy(data=np.arange(2 * n).reshape(n, 2) * 0.01 - 0.025).assign_attrs(units="ppm")
    return centered_transport(anomaly, mass, xr.DataArray(c, attrs={"units": "ppm"}))


def _choice(method, level_sd=5.0, scale=0.7, **periods):
    if method == "level":
        return AdditiveChoice(_normal(level_sd), 0.0, anomalies="period_contrasts", **periods)
    if method == "contrasts":
        return AdditiveChoice(_normal(level_sd), scale, anomalies="period_contrasts", **periods)
    if method == "independent":
        return AdditiveChoice(None, scale, anomalies="independent", **periods)
    return AnomalyScalingChoice(_normal(level_sd), scale, **periods)


def _eager(bound):
    return replace(bound, data=bound.data.compute())


def _designs(bound):
    with registered_model() as model:
        term = bound.add()
    return model, term, [block.design.transpose("nmeasure", ...).values for block in term.blocks]


@pytest.mark.parametrize("method", METHODS)
def test_affine_designs_and_absolute_equation_respond_through_mass(method):
    obs = _inputs().mf
    transport = _operands(obs, normalised=False)
    m, a = transport.H_bc_mass.values, transport.H_bc_anomaly.values
    level = m.sum(axis=1)[:, None]
    assert not np.allclose(level, 1.0)  # The level responds through M1, not 1.
    expected = {
        "level": [level],
        "contrasts": [level, m @ helmert(2, full=False).T],
        "independent": [m],
        "scaling": [level, a],
    }[method]
    model, term, designs = _designs(_eager(_choice(method).bind(obs, transport)))
    for design, reference in zip(designs, expected, strict=True):
        np.testing.assert_allclose(design, reference, rtol=0, atol=1e-15)
    assert "bc" not in model
    point = {name: np.full_like(value, -1.3) for name, value in model.initial_point().items()}
    absolute, centered = model.compile_fn(
        model.replace_rvs_by_values([term.output, term.likelihood_contribution]), inputs=model.value_vars,
    )(point)
    physical = a.sum(axis=1) + 400 * m.sum(axis=1)
    np.testing.assert_allclose(absolute, physical - 1.3 * sum(d.sum(axis=1) for d in expected), atol=1e-12)
    np.testing.assert_allclose(absolute - centered, term.observation_shift.values, rtol=0, atol=1e-12)


def test_independent_curtains_match_contrasts_only_for_specific_widths():
    obs = _inputs().mf
    transport = _operands(obs, normalised=False)
    s, k = 1.2, 2
    _, _, (iid,) = _designs(_eager(_choice("independent", scale=s).bind(obs, transport)))
    _, _, (level, contrasts) = _designs(_eager(_choice("contrasts", level_sd=s / np.sqrt(k), scale=s).bind(obs, transport)))
    independent = s**2 * iid @ iid.T
    np.testing.assert_allclose(independent, s**2 / k * level @ level.T + s**2 * contrasts @ contrasts.T, atol=1e-15)
    assert not np.allclose(independent, s**2 * level @ level.T + s**2 * contrasts @ contrasts.T)


def test_independent_curtains_across_periods_need_one_mean_per_period():
    obs = _inputs().mf
    transport = _operands(obs, normalised=False)
    s, k = 1.2, 2
    periods = {"frequency": "2h", "anchor_time": "2021-01-01"}
    bound = _eager(_choice("independent", scale=s, **periods).bind(obs, transport))
    assert bound.data.indexes["bc_region"].get_level_values("bc_period").nunique() > 1
    _, _, (iid,) = _designs(bound)
    independent = s**2 * iid @ iid.T

    def contrasts_plus_mean(by_period):
        choice = replace(_choice("contrasts", level_sd=s / np.sqrt(k), scale=s, **periods), mean_by_period=by_period)
        _, _, (level, contrasts) = _designs(_eager(choice.bind(obs, transport)))
        return s**2 / k * level @ level.T + s**2 * contrasts @ contrasts.T

    np.testing.assert_allclose(independent, contrasts_plus_mean(True), atol=1e-14)
    assert not np.allclose(independent, contrasts_plus_mean(False))


def test_independent_anomalies_are_sampled_on_the_states_without_a_basis():
    obs = _inputs().mf
    transport = _operands(obs, normalised=False)
    bound = _eager(_choice("independent", scale=0.3, frequency="2h", anchor_time="2021-01-01").bind(obs, transport))
    model, term, (design,) = _designs(bound)
    assert "bc_anomaly_basis" not in model and "bc_anomaly_latent" not in model
    assert term.sampled_rvs == (model["bc_anomaly"],)
    np.testing.assert_array_equal(design, bound.data.H_bc_mass.transpose("nmeasure", "bc_region").values)
    assert bound.anomaly_factor().dims == ("nmeasure", "bc_region")


def test_temporal_contrast_covariance_is_not_full_weighted_centering():
    obs = _inputs().mf
    transport = _operands(obs)
    periods = {"frequency": "2h", "anchor_time": "2021-01-01"}
    bound = _eager(_choice("contrasts", level_sd=0.2, scale=0.3, **periods).bind(obs, transport))
    model, _, (level, contrasts) = _designs(bound)
    assert contrasts.shape == (4, 3 * (2 - 1))
    rows = np.array([0, 1, 0, 2])  # Receptor period of each observation row.
    q = helmert(2, full=False).T
    native = transport.H_bc_mass.values @ q
    expected = 0.2**2 * level @ level.T
    for i in range(4):
        for j in range(4):
            if rows[i] == rows[j]:
                expected[i, j] += 0.3**2 * float(native[i] @ native[j])
    covariance = 0.2**2 * level @ level.T + 0.3**2 * contrasts @ contrasts.T
    np.testing.assert_allclose(covariance, expected, atol=1e-16)
    z = np.array([0.12, -0.03, 0.04])
    point = model.initial_point() | {"bc_anomaly_latent": z}
    anomaly = model.compile_fn(model.replace_rvs_by_values([model["bc_anomaly"]]), inputs=model.value_vars,
                               on_unused_input="ignore")(point)[0]
    np.testing.assert_allclose(anomaly, (q @ z[None, :]).reshape(-1), atol=1e-16)  # Curtain-major states.

    labels = bound.data.indexes["bc_region"]
    weights = xr.DataArray(
        np.ones((2, 3)), dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": labels.levels[0], "bc_period": labels.levels[1]}, attrs={"units": "1"},
    )
    weighted = AdditiveChoice(_normal(0.2), 0.3, **periods).bind(obs, raw_transport(transport, weights=weights))
    weighted_model, _, (_, weighted_contrasts) = _designs(_eager(weighted))
    assert weighted_contrasts.shape == (4, 2 * 3 - 1)
    states = model["bc_anomaly_basis"].eval()
    weighted_states = weighted_model["bc_anomaly_basis"].eval()
    assert not np.allclose(states @ states.T, weighted_states @ weighted_states.T)


@pytest.mark.parametrize("weight_scale", [1.0, 1e307])
def test_weighted_factor_and_graph_share_the_centered_design(weight_scale):
    """Lazy factors and sampled anomalies obey the same nonuniform centering."""
    obs = _inputs().mf
    periods = {"frequency": "2h", "anchor_time": "2021-01-01"}
    transport = _operands(obs, normalised=False)
    states = BoundCenteredTransport.bind(transport, obs, **periods).indexes["bc_region"]
    raw_weights = np.array([[1., 2., 3.], [4., 5., 6.]])
    weights = xr.DataArray(
        weight_scale * raw_weights, dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": states.levels[0], "bc_period": states.levels[1]}, attrs={"units": "1"},
    )
    transport = raw_transport(transport.assign(H_bc_mass=transport.H_bc_mass.chunk()), weights=weights)
    bound = AdditiveChoice(_normal(0.2), 0.3, **periods).bind(obs, transport)
    factor = bound.anomaly_factor()
    assert dask.is_dask_collection(factor)
    model, _, (_, design) = _designs(_eager(bound))
    normalized = raw_weights.ravel() / raw_weights.sum()
    basis = model["bc_anomaly_basis"].eval()
    np.testing.assert_allclose(normalized @ basis, 0.0, atol=1e-16)
    np.testing.assert_allclose(factor.compute().values, 0.3 * design, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("values", "lazy", "message"),
    [
        ([[1., 2., 3.], [4., 0., 6.]], False, "finite and strictly positive"),
        ([[1., 2., np.inf], [4., 5., 6.]], False, "finite and strictly positive"),
        ([[1., 2., 3.], [4., 5., 6.]], True, "Materialize bc_centering_weights"),
    ],
)
def test_weighted_factor_rejects_invalid_or_lazy_weights(values, lazy, message):
    """The lazy factor validates weights exactly as model construction does."""
    obs = _inputs().mf
    periods = {"frequency": "2h", "anchor_time": "2021-01-01"}
    transport = _operands(obs, normalised=False)
    states = BoundCenteredTransport.bind(transport, obs, **periods).indexes["bc_region"]
    weights = xr.DataArray(
        np.array(values), dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": states.levels[0], "bc_period": states.levels[1]}, attrs={"units": "1"},
    )
    if lazy:
        weights = weights.chunk()
    bound = AdditiveChoice(_normal(0.2), 0.3, **periods).bind(obs, raw_transport(transport, weights=weights))
    with pytest.raises(ValueError, match=message):
        bound.anomaly_factor()


def test_reference_reexpresses_prescribed_boundary_but_changes_scaling_covariance():
    obs = _inputs().mf
    transport = _operands(obs, 209300.0, normalised=False)
    prescribed = (transport.H_bc_anomaly + transport.bc_reference * transport.H_bc_mass).sum("bc_region")
    bound = {}
    for reference in ("stored", 209301.0, "whole_window_transport_mean"):
        data = _eager(AnomalyScalingChoice(_normal(5.0), 0.8, reference=reference).bind(obs, transport)).data
        np.testing.assert_allclose(
            (data.H_bc_anomaly + data.bc_reference * data.H_bc_mass).sum("bc_region"), prescribed, rtol=0, atol=3e-11,
        )
        assert data.H_bc_anomaly.dtype == np.float64
        bound[reference] = data
    assert not np.allclose(bound["stored"].H_bc_anomaly, bound[209301.0].H_bc_anomaly)
    assert float(bound["whole_window_transport_mean"].bc_reference) == pytest.approx(
        209300.0 + float(transport.H_bc_anomaly.sum() / transport.H_bc_mass.sum()), rel=0, abs=1e-9,
    )
    assert float(bound["whole_window_transport_mean"].H_bc_anomaly.sum()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", ["level", "scaling"])
def test_period_mean_shifts_respond_through_each_period_mass(method):
    obs = _inputs().mf
    transport = _operands(obs, normalised=False)
    choice = _choice(method, mean_by_period=True, frequency="2h", anchor_time="2021-01-01")
    model, _, designs = _designs(_eager(choice.bind(obs, transport)))
    np.testing.assert_allclose(designs[0], transport.H_bc_mass.values.sum(axis=1)[:, None] * np.eye(3)[[0, 1, 0, 2]])
    assert model["bc_mean_shift"].eval().shape == (3,)
    assert model.named_vars_to_dims["bc_mean_shift"] == ("bc_mean_period",)


def test_binding_is_lazy_and_materializes_the_whole_window_reference_once():
    obs = _inputs().mf
    transport = _operands(obs)
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return transport.H_bc_anomaly.values

    lazy = raw_transport(transport.assign(
        H_bc_anomaly=transport.H_bc_anomaly.copy(data=da.from_delayed(payload(), shape=(4, 2), dtype=float)),
    ))
    bound = AnomalyScalingChoice(_normal(5.0), 0.8, reference="whole_window_transport_mean").bind(obs, lazy)
    assert calls == []
    (eager,) = dask.compute(bound)
    assert calls == [1]
    xr.testing.assert_allclose(eager.data, _eager(bound).data)


@pytest.mark.parametrize("method", METHODS)
def test_one_curtain_keeps_scalings_and_independent_states_but_no_contrast(method):
    obs = _inputs().mf
    transport = _operands(obs)
    transport = centered_transport(transport.H_bc_anomaly.isel(bc_region=[0]), xr.ones_like(transport.H_bc_mass.isel(bc_region=[0])),
                                 transport.bc_reference)
    with registered_model() as model:
        _eager(_choice(method).bind(obs, transport)).add()
    assert ("bc_scaling" in model) == (method == "scaling")
    assert ("bc_anomaly" in model) == (method == "independent")
    assert "bc_anomaly_latent" not in model
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize(
    "options",
    [
        dict(mean_prior=None, anomaly_scale=0.3),
        dict(mean_prior=None, anomaly_scale=0.0, anomalies="independent"),
        dict(mean_prior=None, anomaly_scale=0.3, anomalies="period_contrasts", mean_by_period=True,
             frequency="monthly"),
        dict(anomalies="curtains"),
        dict(mean_by_period=True),
        dict(frequency="monthly", anchor_time="2021-01-01"),
        dict(anomaly_scale=-1.0),
    ],
)
def test_invalid_additive_choices_fail_when_chosen(options):
    with pytest.raises(ValueError):
        AdditiveChoice(**{"mean_prior": _normal(1.0), "anomaly_scale": 0.3, **options})


@pytest.mark.parametrize("options", [dict(scale_sd=0.0), dict(reference="rolling"), dict(reference=np.nan),
                                     dict(mean_by_period=True),
                                     dict(mean_prior=None, mean_by_period=True, frequency="monthly")])
def test_invalid_scaling_choices_fail_when_chosen(options):
    with pytest.raises(ValueError):
        AnomalyScalingChoice(**{"mean_prior": _normal(1.0), "scale_sd": 0.3, **options})


def _density_parity(cached, stock):
    model = cached.model
    point = model.initial_point()
    for name in point:
        if name.endswith(("bc_mean_shift", "bc_anomaly_latent", "bc_scaling")):
            point[name] = np.full_like(point[name], 0.17)
    amplitude = np.exp(point["ou_site_amplitude_log__"])
    cached.shared_cache.update(cached.target.refresh(amplitude))
    quadratic = model.compile_fn(
        model.replace_rvs_by_values([cached.cached_state_logp]), inputs=model.value_vars, on_unused_input="ignore",
    )(point)[0]
    np.testing.assert_allclose(model.compile_logp(vars=model.observed_RVs)(point), quadratic, atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(model.compile_logp()(point), stock.compile_logp()(point), atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(model.compile_dlogp()(point), stock.compile_dlogp()(point), atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_single_ordinary_cached_density_and_gradient(method):
    inv = _inputs()
    transport = _operands(inv.mf)
    bound = _choice(method, level_sd=0.2, scale=0.3).bind(inv.mf, transport)
    kwargs = dict(
        retained_prior=CorrelatedLognormalPrior(inv.alpha_prior_mean, inv.alpha_prior_covariance,
                                                covariance_dim="region_cov"),
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf + 400.0,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, "dense"),
        boundary=bound,
    )
    cached = build_co2_cached_sigma_model(inv.H, **kwargs, **OU)
    stock = build_co2_model(inv.H, **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=OU)
    _density_parity(cached, stock)


@pytest.mark.parametrize("method", METHODS)
def test_linked_ordinary_cached_channels_are_independent_at_oxygen_precision(method):
    kwargs = _kwargs(_prepared(), interleaved=True)
    boundary = {}
    for channel, c, width in (("co2", 400.0, 0.2), ("o2", 209300.0, 0.003)):
        row = str(kwargs[f"{channel}_sensitivity"].dims[0])
        obs = select_gathered_data_array(
            kwargs["observations"], key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row)
        # Prepared operands use the raw ``nmeasure`` axis, labelled by native rows.
        transport = _operands(obs.rename({row: "nmeasure"}), c)
        boundary[channel] = _choice(method, level_sd=width, scale=0.3).bind(obs, transport)
    kwargs["boundary"] = boundary
    cached = build_co2_o2_cached_sigma_model(**kwargs, **OU)
    stock = build_co2_o2_model(**kwargs, **OU)
    _density_parity(cached, stock)
    model = cached.model
    assert "co2_bc" not in model and "o2_bc" not in model
    point = model.initial_point()
    independent = method == "independent"
    point["o2_bc_anomaly" if independent else "o2_bc_mean_shift"] = (
        np.full(2, 0.000123) if independent else np.array(0.000123)
    )
    signal = "mu_bc_anomaly" if independent else "mu_bc_mean_shift"
    oxygen, carbon = model.compile_fn(
        model.replace_rvs_by_values([model[f"o2_{signal}"], model[f"co2_{signal}"]]),
        inputs=model.value_vars, on_unused_input="ignore",
    )(point)
    rows = kwargs["observations"].species.values == "o2"
    np.testing.assert_allclose(oxygen[rows], np.full(np.count_nonzero(rows), 0.000123), rtol=0, atol=1e-18)
    np.testing.assert_array_equal(oxygen[~rows], 0.0)
    np.testing.assert_array_equal(carbon, 0.0)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_public_co2_trace_products_reconstruct_each_structure(cached, method):
    prepared = _single_with_operands(additive=False)
    choice = _choice(method, level_sd=0.5, scale=0.2)
    options = (dict(likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=OU) if cached
               else dict(no_model_error=True))
    built, _ = inference.build_co2_inference(prepared, {"boundary": choice, **options})
    metadata = built.output_contract.metadata
    assert metadata["boundary_formulation"] == ("anomaly_scaling" if method == "scaling" else "additive")
    trace = co2_runner.annotate_co2_trace(_draws(built), built, concentration_units="ppm")
    components = reconstruct_co2_concentrations(trace, prepared)
    for group in ("prior", "posterior"):
        total = components[f"boundary_reference_{group}"]
        if method != "independent":
            assert trace[group].bc_mean_shift.attrs["units"] == "ppm"
            total = total + components[f"boundary_mean_contribution_{group}"]
        if method in ("contrasts", "independent"):
            latent = "bc_anomaly" if method == "independent" else "bc_anomaly_latent"
            assert trace[group][latent].attrs["units"] == "ppm"
            total = total + components[f"boundary_anomaly_contribution_{group}"]
        if method == "scaling":
            # Exported components alone decompose the scaling contribution.
            assert components[f"boundary_anomaly_{group}"].attrs["units"] == "1"
            assert "bc_region" in components[f"boundary_anomaly_{group}"].dims
            total = total + components[f"boundary_anomaly_contribution_{group}"]
        xr.testing.assert_allclose(components[f"boundary_{group}"], total.transpose(*components[f"boundary_{group}"].dims))
        xr.testing.assert_allclose(components[f"modelled_{group}"],
                                   components[f"pollution_{group}"] + components[f"boundary_{group}"])



def test_missing_scaling_draws_fail_boundary_reconstruction():
    prepared = _single_with_operands(additive=False)
    built, _ = inference.build_co2_inference(
        prepared, {"boundary": _choice("scaling", level_sd=0.5, scale=0.2), "no_model_error": True},
    )
    trace = co2_runner.annotate_co2_trace(_draws(built), built, concentration_units="ppm")
    trace["posterior"] = trace["posterior"].to_dataset().drop_vars("mu_bc_scaling")
    with pytest.raises(ValueError, match="boundary_anomaly_concentration"):
        reconstruct_co2_concentrations(trace, prepared)

def test_restored_trace_coordinates_label_period_contrast_states():
    prepared = _single_with_operands(additive=False)
    choice = _choice("contrasts", frequency="2h", anchor_time="2021-01-01")
    built, _ = inference.build_co2_inference(prepared, {"boundary": choice, "no_model_error": True})
    with built.model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=3)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    anomaly = trace.prior.bc_anomaly
    assert list(anomaly.indexes["bc_region"].names) == ["bc_curtain", "bc_period"]
    np.testing.assert_allclose(
        anomaly.groupby("bc_period").sum().values, 0.0, atol=1e-12,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mean", [True, False])
def test_anomaly_factor_is_the_scaled_anomaly_block(method, mean):
    if not mean and method == "level":
        pytest.skip("A boundary needs a mean shift or sampled anomalies.")
    obs = _inputs().mf
    choice = _choice(method, scale=0.3, frequency="2h", anchor_time="2021-01-01")
    if not mean:
        choice = replace(choice, mean_prior=None)
    bound = _eager(choice.bind(obs, _operands(obs, normalised=False)))
    factor = bound.anomaly_factor()
    _, _, designs = _designs(bound)
    assert len(designs) == (method != "level") + (mean and method != "independent")
    if method == "level":
        assert factor is None
        return
    assert factor.attrs["units"] == "ppm"
    np.testing.assert_allclose(factor.transpose("nmeasure", ...).values, 0.3 * designs[-1], rtol=0, atol=1e-15)
