"""Compact linked recording preserves each derived scientific output at fixed draws."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import load_trace, save_trace

from test_co2_o2_additive_boundary import _additive_inputs
from test_co2_o2_fixed_ou import _kwargs, _prepared
from test_centered_cached_linked_runner import _prepared_boundaries
from boundary_test_utils import bound_linked_model_arguments


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _model(cached, baseline, offsets, scales=None):
    if baseline == "additive":
        kwargs = _additive_inputs()
        for channel, scale in zip(("co2", "o2"), scales):
            kwargs["bc_anomaly_scale"][channel] = scale
    elif baseline == "centered":
        prepared, _ = _prepared_boundaries(mixed=False)
        kwargs = _kwargs(prepared)
        kwargs["boundary_operands"] = prepared.boundary_operands
        kwargs["bc_prior"] = {
            channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.01} for channel in ("co2", "o2")
        }
    else:
        kwargs = _kwargs(_prepared(), interleaved=True, native_multiindex=True)
    if offsets:
        kwargs["offset_prior"] = {
            channel: {"pdf": "normal", "mu": 0.1, "sigma": 0.3} for channel in offsets
        }
        kwargs["offset_args"] = {channel: {"per_site": True} for channel in offsets}
    ou = {"tau_hours": 6.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
    built = (build_co2_o2_cached_sigma_model if cached else build_co2_o2_model)(
        **bound_linked_model_arguments({**kwargs, **ou})
    )
    return built.model if cached else built


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "baseline,offsets,scales",
    [
        ("none", (), None), ("centered", ("o2",), None),
        ("additive", ("co2", "o2"), (0.04, 0.04)),
        ("additive", ("co2", "o2"), (0.0, 0.04)),
        ("additive", ("co2", "o2"), (0.04, 0.0)),
        ("additive", ("co2", "o2"), (0.0, 0.0)),
    ],
)
def test_compact_recording_restores_all_deterministics_at_identical_states(
    monkeypatch, tmp_path, cached, baseline, offsets, scales
):
    """Compare compact reconstruction exactly with ordinary recording at the same states."""
    model = _model(cached, baseline, offsets, scales)
    # Ordinary NDArray recording is the exact per-state oracle. Prior predictive
    # evaluation can use a differently optimized/vectorized concentration graph.
    # One chain isolates graph evaluation; this oracle does not test chain ordering.
    point = model.initial_point()
    backend = pm.backends.NDArray(model=model, test_point=point)
    backend.setup(draws=4, chain=0)
    rng = np.random.default_rng(183)
    for _ in range(4):
        backend.record({name: np.asarray(value + .01 * rng.normal(size=value.shape))
                        for name, value in point.items()}, in_warmup=False)
    reference = pm.to_inference_data(
        prior={variable.name: backend.get_values(variable.name) for variable in model.free_RVs + model.deterministics},
        model=model,
    )
    gradient = model.compile_dlogp()
    original_gradient = gradient(point)
    recorded = {}

    def sampled(**kwargs):
        names = kwargs["var_names"]
        recorded["names"] = names
        assert {rv.name for rv in model.free_RVs} <= set(names)
        for variable in model.deterministics:
            if "observation" in model.named_vars_to_dims[variable.name]:
                assert variable.name not in names
        return xr.DataTree.from_dict(
            {
                "posterior": reference.prior.dataset[names],
                "constant_data": reference.constant_data.dataset,
                "observed_data": reference.observed_data.dataset,
            }
        )

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    trace = RhimeSampler(
        draws=4, tune=2, chains=1, burn=1,
        sample_kwargs={"idata_kwargs": {"log_likelihood": False}},
        sample_prior_predictive=False, sample_posterior_predictive=False,
    ).sample(model, recompute_deterministics=True)
    assert recorded["names"]
    np.testing.assert_array_equal(gradient(point), original_gradient)
    expected = restore_inferencedata_coords(reference, get_coord_registry(model)).prior.dataset.isel(draw=slice(1, None))
    expected = expected.assign_coords(draw=trace.posterior.draw)
    assert set(trace.posterior.data_vars) == set(expected.data_vars)
    for name in expected.data_vars:
        xr.testing.assert_equal(trace.posterior[name], expected[name])
        assert trace.posterior[name].dtype == expected[name].dtype
    if scales is not None:
        for channel, scale in zip(("co2", "o2"), scales):
            assert (f"{channel}_bc_anomaly_latent" in trace.posterior) == (scale > 0)
            assert f"{channel}_bc_mean_shift" in trace.posterior
            if scale == 0:
                assert f"{channel}_bc_anomaly" not in trace.posterior
                assert f"{channel}_mu_bc_anomaly" not in trace.posterior
    if offsets == ("o2",):
        np.testing.assert_array_equal(trace.posterior.o2_offset, trace.posterior.offset_concentration)
    if baseline == "none":
        assert isinstance(trace.posterior.indexes["observation"], pd.MultiIndex)
        assert trace.posterior.indexes["observation"].equals(expected.indexes["observation"])
    path = tmp_path / "linked-compact.nc"
    save_trace(trace, path)
    restored = load_trace(path)
    original, reopened = trace.posterior.to_dataset(), restored.posterior.to_dataset()
    if baseline == "none":
        # Recent xarray retains second-resolution indexes until NetCDF normalizes them to ns.
        baseline_path = tmp_path / "linked-full-prior.nc"
        save_trace(reference, baseline_path)
        baseline_index = load_trace(baseline_path).prior.indexes["observation"]
        restored_index = reopened.indexes["observation"]
        assert isinstance(restored_index, pd.MultiIndex)
        pd.testing.assert_index_equal(restored_index, baseline_index, exact=True)
        assert restored_index.names == original.indexes["observation"].names
        np.testing.assert_array_equal(restored_index, original.indexes["observation"])
        original, reopened = (dataset.reset_index("observation") for dataset in (original, reopened))
    xr.testing.assert_identical(reopened, original)
    for name in original.data_vars:
        assert reopened[name].dtype == original[name].dtype


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("scales", [(0.04, 0.04), (0.0, 0.04), (0.04, 0.0), (0.0, 0.0)])
def test_compact_sampler_preserves_normalized_likelihood_and_absolute_products(tmp_path, cached, scales):
    """Exercise recording, return, prediction, and serialization on actual small chains."""
    from scipy.stats import multivariate_normal

    from openghg_inversions.rhime.co2.boundary_output import restore_absolute_predictives
    from openghg_inversions.rhime.co2.co2_cached_sigma_runner import _sampler_for_cached_graph
    from test_co2_o2_fixed_ou import _dense

    kwargs = _additive_inputs()
    for channel, scale in zip(("co2", "o2"), scales):
        kwargs["bc_anomaly_scale"][channel] = scale
    kwargs["offset_prior"] = {"o2": {"pdf": "normal", "mu": 0.0, "sigma": 0.1}}
    kwargs["offset_args"] = {"o2": {"per_site": True}}
    ou = {"tau_hours": 6.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}}
    built = (build_co2_o2_cached_sigma_model if cached else build_co2_o2_model)(
        **bound_linked_model_arguments({**kwargs, **ou})
    )
    model = built.model if cached else built
    sampler_options = dict(
        draws=3, tune=2, chains=1, burn=1,
        sample_kwargs={
            "cores": 1, "random_seed": 93, "progressbar": False,
            "compute_convergence_checks": False,
        },
        sample_posterior_predictive=True,
    )
    sampler = RhimeSampler(**sampler_options)
    if cached:
        sampler = _sampler_for_cached_graph(
            sampler, cached_model=built, sigma_target_accept=0.8, state_target_accept=0.8,
        )
    trace = sampler.sample(model, recompute_deterministics=True)
    original_observed = kwargs["observations"] - trace.constant_data.boundary_reference_shift
    xr.testing.assert_allclose(trace.observed_data.y, original_observed)
    reference_built = (build_co2_o2_cached_sigma_model if cached else build_co2_o2_model)(
        **bound_linked_model_arguments({**kwargs, **ou})
    )
    reference_model = reference_built.model if cached else reference_built
    reference_sampler = RhimeSampler(**sampler_options)
    if cached:
        reference_sampler = _sampler_for_cached_graph(
            reference_sampler, cached_model=reference_built,
            sigma_target_accept=0.8, state_target_accept=0.8,
        )
    reference_sampler.sample_kwargs["trace"] = pm.backends.NDArray(model=reference_model)
    reference = reference_sampler.sample(reference_model)
    for group in ("posterior", "prior", "log_likelihood", "posterior_predictive", "prior_predictive", "observed_data"):
        xr.testing.assert_allclose(trace[group].to_dataset(), reference[group].to_dataset(), rtol=1e-12, atol=1e-9)
    assert set(trace.posterior.data_vars) == {
        variable.name for variable in model.free_RVs + model.deterministics
    }
    assert set(trace.prior.data_vars) == set(trace.posterior.data_vars)
    tau = {"co2:A": 6.0, "o2:A": 6.0, "o2:B": 6.0}
    for draw in range(2):
        mean = trace.posterior.modelled_concentration.values[0, draw]
        amplitude = trace.posterior.ou_site_amplitude.values[0, draw]
        expected = multivariate_normal.logpdf(
            kwargs["observations"].values, mean=mean, cov=_dense(kwargs, amplitude, tau),
        )
        np.testing.assert_allclose(trace.log_likelihood.y.values[0, draw], expected, atol=1e-7)
    assert trace.log_likelihood.y.dims == ("chain", "draw")
    restore_absolute_predictives(trace)
    restore_absolute_predictives(reference)
    xr.testing.assert_allclose(trace.observed_data.y, kwargs["observations"])
    xr.testing.assert_identical(trace.observed_data.y, reference.observed_data.y)
    assert trace.posterior_predictive.y.shape == (1, 2, 5)
    assert trace.prior_predictive.y.shape == (1, 2, 5)
    assert float(trace.posterior_predictive.y.isel(observation=slice(2, None)).min()) > 199900
    for group in ("posterior", "prior"):
        values = trace[group]
        for channel, scale in zip(("co2", "o2"), scales):
            assert (f"{channel}_bc_anomaly_latent" in values) == (scale > 0)
            assert f"{channel}_bc_mean_shift" in values
            if scale == 0:
                assert f"{channel}_bc_anomaly" not in values
                assert f"{channel}_mu_bc_anomaly" not in values
        np.testing.assert_array_equal(values.o2_offset, values.offset_concentration)
        np.testing.assert_allclose(
            values.modelled_concentration,
            values.co2_o2_flux_contribution + trace.constant_data.fixed_prior_contribution
            + values.baseline_concentration,
            atol=1e-10,
        )
    path = tmp_path / "linked-sampled.nc"
    save_trace(trace, path)
    restored = load_trace(path)
    reference_path = tmp_path / "linked-sampled-ordinary.nc"
    save_trace(reference, reference_path)
    restored_reference = load_trace(reference_path)
    for group in ("posterior", "prior", "log_likelihood", "posterior_predictive", "prior_predictive", "observed_data"):
        xr.testing.assert_identical(restored[group].dataset, trace[group].dataset)
        xr.testing.assert_allclose(restored[group].to_dataset(), restored_reference[group].to_dataset(), rtol=1e-12, atol=1e-9)


def test_restoration_borrows_free_states_and_preserves_input_metadata():
    from openghg_inversions.inference.sampling import _restore_deterministics

    with pm.Model() as model:
        state = pm.Normal("state")
        pm.Deterministic("signal", state * 2.0)
    values = np.array([[0.1, 0.3], [-0.2, 0.4]], dtype="float64")
    posterior = xr.Dataset(
        {"state": (("chain", "draw"), values, {"units": "ppm"})},
        coords={"chain": ["first", "second"], "draw": [13, 14]},
        attrs={"burn": 2, "description": "borrowed posterior"},
    )
    original = posterior.copy(deep=True)
    result = _restore_deterministics(posterior, model)
    xr.testing.assert_identical(posterior, original)
    assert np.shares_memory(result.state.data, values)
    assert result.signal.dtype == values.dtype
    xr.testing.assert_allclose(result.signal, 2 * posterior.state)
    assert result.attrs == posterior.attrs


@pytest.mark.parametrize("observed", [10.0, [10.0, 12.0]], ids=["scalar", "vector"])
@pytest.mark.parametrize("shared", [False, True], ids=["constant", "shared-data"])
def test_restoration_conditions_observed_dependent_deterministics(observed, shared):
    from openghg_inversions.inference.sampling import _restore_deterministics

    values = np.asarray(observed, dtype="float64")
    dims = "observation" if values.ndim else None
    with pm.Model(coords={"observation": ["first", "second"]}) as model:
        state = pm.Normal("state")
        data = pm.Data("data", values, dims=dims) if shared else values
        observed_rv = pm.Normal("y", state, 1.0, observed=data, dims=dims)
        residual = pm.Deterministic("residual", observed_rv - state, dims=dims)
        pm.Deterministic("squared_residual", residual**2, dims=dims)
    posterior = xr.Dataset({"state": (("chain", "draw"), [[1.0, 2.0]])})
    expected = np.array([9.0, 8.0]) if values.ndim == 0 else np.array([[9.0, 11.0], [8.0, 10.0]])

    for _ in range(2):
        restored = _restore_deterministics(posterior, model)
        np.testing.assert_array_equal(restored.residual.values[0], expected)
        np.testing.assert_array_equal(restored.squared_residual.values[0], expected**2)
    if shared:
        pm.set_data({"data": values + 1.0}, model=model)
        updated = _restore_deterministics(posterior, model)
        np.testing.assert_array_equal(updated.residual.values[0], expected + 1.0)


def test_restoration_matches_recording_with_an_explicit_observed_transform():
    from openghg_inversions.inference.sampling import _restore_deterministics

    with pm.Model(coords={"observation": ["first", "second"]}) as model:
        state = pm.Normal("state")
        observed = pm.HalfNormal(
            "y", observed=[2.0, 3.0], transform=pm.distributions.transforms.log,
            dims="observation",
        )
        pm.Deterministic("residual", observed - state, dims="observation")
    backend = pm.backends.NDArray(model=model)
    backend.setup(draws=2, chain=0)
    for value in (1.0, 2.0):
        backend.record({"state": np.asarray(value)}, in_warmup=False)
    posterior = xr.Dataset({"state": (("chain", "draw"), [[1.0, 2.0]])})

    restored = _restore_deterministics(posterior, model)
    np.testing.assert_array_equal(restored.residual.values[0], backend.get_values("residual"))


@pytest.mark.parametrize("likelihood", [False, True, ["y"]])
def test_compact_sampler_preserves_warmup_and_explicit_likelihood_selection(monkeypatch, likelihood):
    with pm.Model(coords={"observation": ["co2", "o2"]}) as model:
        state = pm.Normal("state")
        signal = pm.Deterministic("signal", state * np.array([1.0, 2.0]), dims="observation")
        pm.Normal("y", signal, 0.5, observed=np.array([1.0, 2.0]), dims="observation")
    posterior = xr.Dataset(
        {"state": (("chain", "draw"), [[0.2, 0.4]])},
        coords={"chain": [0], "draw": [0, 1]},
    )
    warmup = xr.Dataset(
        {"state": (("chain", "draw"), [[0.1, -0.3, 0.7]])},
        coords={"chain": [0], "draw": [0, 1, 2]},
    )
    calls = []

    def sampled(**kwargs):
        assert kwargs["var_names"] == ["state"]
        assert kwargs["idata_kwargs"]["log_likelihood"] is False
        return xr.DataTree.from_dict({"posterior": posterior, "warmup_posterior": warmup})

    def likelihood_fn(trace, **kwargs):
        calls.append(kwargs)
        assert "signal" not in trace.posterior
        return trace

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.compute_log_likelihood", likelihood_fn)
    options = {"idata_kwargs": {"log_likelihood": likelihood, "save_warmup": True}}
    sampler = RhimeSampler(
        draws=2, tune=3, burn=1, chains=1, sample_kwargs=options,
        sample_prior_predictive=False, sample_posterior_predictive=False,
    )
    trace = sampler.sample(model, recompute_deterministics=True)
    assert len(calls) == (0 if likelihood is False else 1)
    if calls:
        assert calls[0]["var_names"] == (None if likelihood is True else ["y"])
    assert sampler.sample_kwargs["idata_kwargs"]["log_likelihood"] == likelihood
    for name, source in (("posterior", posterior.isel(draw=slice(1, None)).assign_coords(draw=[0])), ("warmup_posterior", warmup)):
        np.testing.assert_array_equal(trace[name].state, source.state)
        np.testing.assert_allclose(trace[name].signal, source.state.data[..., None] * [1.0, 2.0])
        assert trace[name].signal.dims == ("chain", "draw", "observation")


@pytest.mark.parametrize(
    "backend,idata_options,sample_options,draws,tune,burn",
    [
        pytest.param("nutpie", {}, {}, 2, 0, 0, id="external-backend"),
        pytest.param("pymc", {"include_transformed": True}, {}, 2, 0, 0, id="transformed"),
        pytest.param("pymc", {}, {"trace": object()}, 2, 0, 0, id="explicit-backend"),
        pytest.param("pymc", {}, {"callback": lambda **_: None}, 2, 0, 0, id="callback"),
        pytest.param("pymc", {}, {}, 2, 0, 2, id="empty-posterior"),
    ],
)
def test_compact_request_preserves_alternative_recording_options(
    monkeypatch, backend, idata_options, sample_options, draws, tune, burn,
):
    with pm.Model() as model:
        state = pm.Normal("state")
        pm.Deterministic("signal", 2 * state)
    posterior = xr.Dataset(
        {"state": (("chain", "draw"), [[0.2, 0.4]]), "signal": (("chain", "draw"), [[0.4, 0.8]])},
        coords={"chain": [0], "draw": [0, 1]},
    )
    warmup = posterior.isel(draw=slice(0, 1))

    def sampled(**kwargs):
        assert "var_names" not in kwargs
        assert kwargs["idata_kwargs"] == {"log_likelihood": True, **idata_options}
        for name, value in sample_options.items():
            assert kwargs[name] is value
        groups = {"posterior": posterior}
        if sample_options.get("discard_tuned_samples") is False:
            groups["warmup_posterior"] = warmup
        return xr.DataTree.from_dict(groups)

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    trace = RhimeSampler(
        draws=draws, tune=tune, burn=burn, chains=1, nuts_sampler=backend,
        sample_kwargs={"idata_kwargs": idata_options, **sample_options},
        sample_prior_predictive=False, sample_posterior_predictive=False,
    ).sample(model, recompute_deterministics=True)
    expected = posterior.isel(draw=slice(burn, None))
    if burn:
        expected = expected.assign_coords(draw=np.arange(expected.sizes["draw"]))
    xr.testing.assert_equal(trace.posterior.to_dataset(), expected)
    if "warmup_posterior" in trace.children:
        assert trace.warmup_posterior.sizes["draw"] == 1
        assert set(trace.warmup_posterior.data_vars) == {"state", "signal"}


@pytest.mark.parametrize("compact", [False, True])
def test_requested_log_prior_matches_retained_nuts_states(compact):
    from scipy.stats import norm

    with pm.Model() as model:
        state = pm.Normal("state", mu=0.3, sigma=1.7)
        signal = pm.Deterministic("signal", 2 * state)
        pm.Normal("y", signal, 0.5, observed=1.0)
    trace = RhimeSampler(
        draws=6, tune=10, burn=4, chains=2, progressbar=False,
        sample_prior_predictive=5,
        sample_kwargs={
            "random_seed": 831, "cores": 1, "compute_convergence_checks": False,
            "discard_tuned_samples": False,
            "idata_kwargs": {"log_prior": True, "save_warmup": True},
        },
    ).sample(model, recompute_deterministics=compact)

    for name in ("posterior", "sample_stats", "log_likelihood", "log_prior", "posterior_predictive"):
        np.testing.assert_array_equal(trace[name].draw, [0, 1])
        np.testing.assert_array_equal(trace[name].chain, [0, 1])
        assert trace[name].attrs["burn"] == 4
    np.testing.assert_allclose(trace.log_prior.state, norm.logpdf(trace.posterior.state, 0.3, 1.7))
    np.testing.assert_allclose(trace.log_likelihood.y, norm.logpdf(1.0, 2 * trace.posterior.state, 0.5))
    np.testing.assert_allclose(trace.posterior.signal, 2 * trace.posterior.state)
    for name in ("warmup_posterior", "warmup_sample_stats"):
        np.testing.assert_array_equal(trace[name].draw, np.arange(10))
    for name in ("prior", "prior_predictive"):
        np.testing.assert_array_equal(trace[name].draw, np.arange(5))


@pytest.mark.parametrize("burn", [0, 2, 4])
@pytest.mark.parametrize("independent_draws", [3, 5])
def test_burn_and_relabel_share_one_group_boundary(burn, independent_draws):
    from openghg_inversions.inference.sampling import _burn_retained_draws

    posterior = xr.Dataset(
        {"state": (("chain", "draw"), [[0.2, 0.4, 0.6]])},
        coords={"chain": [7], "draw": [10, 11, 12]},
    )
    independent = xr.Dataset(
        {"independent": (("chain", "draw"), [np.arange(independent_draws)])},
        coords={"chain": [9], "draw": np.arange(independent_draws) + 100},
        attrs={"source": "independent"},
    )
    sampled_names = ("posterior", "sample_stats", "log_likelihood", "log_prior")
    untouched_names = (
        "warmup_posterior", "warmup_sample_stats", "warmup_log_likelihood", "warmup_log_prior",
        "prior", "prior_predictive", "posterior_predictive", "predictions",
    )
    raw = xr.DataTree.from_dict({
        **dict.fromkeys(sampled_names, posterior),
        **dict.fromkeys(untouched_names, independent),
        "constant_data": xr.Dataset({"constant": 1}),
    })
    original = raw.copy(deep=True)
    trace = _burn_retained_draws(raw.copy(deep=False), burn=burn)
    expected = posterior.isel(draw=slice(burn, None))
    expected = expected.assign_coords(draw=np.arange(expected.sizes["draw"]))
    expected.attrs["burn"] = burn
    for name in sampled_names:
        xr.testing.assert_identical(trace[name].to_dataset(), expected)
    for name in (*untouched_names, "constant_data"):
        xr.testing.assert_identical(trace[name], original[name])
    xr.testing.assert_identical(raw, original)


@pytest.mark.parametrize("compact", [False, True])
def test_burn_preserves_warmup_when_burn_exceeds_tuning(monkeypatch, compact):
    with pm.Model() as model:
        state = pm.Normal("state")
        pm.Deterministic("signal", 2 * state)
    posterior = xr.Dataset({"state": (("chain", "draw"), [[.2, .4, .6]])},
                           coords={"chain": [0], "draw": [0, 1, 2]})
    warmup = posterior.isel(draw=slice(0, 1))
    stats = xr.Dataset({"diverging": (("chain", "draw"), [[False, False, True]])},
                       coords=posterior.coords)

    def sampled(**kwargs):
        assert ("var_names" in kwargs) == compact
        groups = {"posterior": posterior, "sample_stats": stats,
                  "warmup_posterior": warmup, "warmup_sample_stats": stats.isel(draw=[0])}
        if not compact:
            groups["posterior"] = posterior.assign(signal=2 * posterior.state)
            groups["warmup_posterior"] = warmup.assign(signal=2 * warmup.state)
        return xr.DataTree.from_dict(groups)

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    trace = RhimeSampler(draws=3, tune=1, burn=2, chains=1, nuts_sampler="pymc",
                         sample_kwargs={"discard_tuned_samples": False, "idata_kwargs": {"log_likelihood": False}},
                         sample_prior_predictive=False, sample_posterior_predictive=False).sample(
        model, recompute_deterministics=compact,
    )
    np.testing.assert_array_equal(trace.posterior.state, [[.6]])
    np.testing.assert_array_equal(trace.sample_stats.diverging, [[True]])
    xr.testing.assert_identical(trace.warmup_posterior.state, warmup.state)
    xr.testing.assert_identical(trace.warmup_sample_stats.diverging, stats.diverging.isel(draw=[0]))


@pytest.mark.parametrize("compact", [False, True])
def test_warmup_likelihood_keeps_its_own_labels_and_length(monkeypatch, compact):
    with pm.Model() as model:
        state = pm.Normal("state")
        pm.Deterministic("signal", state * 2)
    posterior = xr.Dataset(
        {"state": (("chain", "draw"), [[0.2, 0.4, 0.6]])},
        coords={"chain": [7], "draw": [10, 11, 12]},
    )
    warmup = xr.Dataset(
        {"state": (("chain", "draw"), [[0.1, 0.3]])},
        coords={"chain": [7], "draw": [-2, -1]},
        attrs={"phase": "tuning"},
    )
    warmup_density = warmup.rename(state="y")
    warmup_stats = warmup.rename(state="energy")

    def sampled(**kwargs):
        groups = {"posterior": posterior, "warmup_posterior": warmup,
                  "warmup_sample_stats": warmup_stats, "warmup_log_likelihood": warmup_density}
        if not compact:
            for name in ("posterior", "warmup_posterior"):
                groups[name] = groups[name].assign(signal=2 * groups[name].state)
        return xr.DataTree.from_dict(groups)

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    trace = RhimeSampler(
        draws=3, tune=2, burn=2, chains=1,
        sample_kwargs={"idata_kwargs": {"log_likelihood": False}},
        sample_prior_predictive=False, sample_posterior_predictive=False,
    ).sample(model, recompute_deterministics=compact)
    xr.testing.assert_identical(trace.warmup_posterior.state, warmup.state)
    assert trace.warmup_posterior.attrs == warmup.attrs
    xr.testing.assert_identical(trace.warmup_sample_stats.to_dataset(), warmup_stats)
    xr.testing.assert_identical(trace.warmup_log_likelihood.to_dataset(), warmup_density)
    np.testing.assert_array_equal(trace.posterior.chain, [7])
    np.testing.assert_array_equal(trace.posterior.draw, [0])


@pytest.mark.parametrize("likelihood", [False, True, ["second"]])
def test_compact_likelihood_selection_and_unequal_prediction_draws(monkeypatch, likelihood):
    from scipy.stats import norm

    with pm.Model() as model:
        state = pm.Normal("state")
        signal = pm.Deterministic("signal", state * 2)
        pm.Normal("first", signal, 0.5, observed=1.0)
        pm.Normal("second", signal, 0.7, observed=-1.0)
    posterior = xr.Dataset({"state": (("chain", "draw"), [[0.2, 0.4, 0.6]])})

    def sampled(**kwargs):
        assert kwargs["var_names"] == ["state"]
        assert kwargs["idata_kwargs"]["log_likelihood"] is False
        return xr.DataTree.from_dict({"posterior": posterior})

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    trace = RhimeSampler(
        draws=3, burn=1, tune=1, chains=1, sample_prior_predictive=5,
        sample_posterior_predictive=["first", "second"],
        sample_kwargs={"random_seed": 83, "idata_kwargs": {"log_likelihood": likelihood}},
    ).sample(model, recompute_deterministics=True)
    assert trace.posterior.sizes["draw"] == trace.posterior_predictive.sizes["draw"] == 2
    assert trace.prior.sizes["draw"] == trace.prior_predictive.sizes["draw"] == 5
    np.testing.assert_allclose(trace.posterior.signal, [[0.8, 1.2]])
    if likelihood is False:
        assert "log_likelihood" not in trace.children
    else:
        names = {"first", "second"} if likelihood is True else {"second"}
        assert set(trace.log_likelihood.data_vars) == names
        for name in names:
            observed, sd = (1.0, 0.5) if name == "first" else (-1.0, 0.7)
            np.testing.assert_allclose(trace.log_likelihood[name], norm.logpdf(observed, [[0.8, 1.2]], sd))


def test_compact_request_keeps_explicit_posterior_selection_rejection(monkeypatch):
    def sampled(**kwargs):
        pytest.fail("Explicit posterior selection must reject before sampling.")

    monkeypatch.setattr("openghg_inversions.inference.sampling.pm.sample", sampled)
    sampler = RhimeSampler(sample_prior_predictive=False, sample_posterior_predictive=False)
    sampler.sample_kwargs = {"var_names": ["state"]}
    with pytest.raises(ValueError, match="var_names"):
        sampler.sample(pm.Model(), recompute_deterministics=True)
