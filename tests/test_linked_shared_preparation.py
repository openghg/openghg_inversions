"""Frozen linked preparation and its backward-compatible scientific codec."""

from dataclasses import replace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytensor
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
from openghg_inversions.rhime.co2.prior_configuration import BoundaryCorrectionConfig
from openghg_inversions.rhime.co2.shared_preparation import prepare_linked_shared_inputs
from test_boundary_correction_priors import operands
from test_co2_o2_fixed_ou import _prepared


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.fixture
def native():
    base = _prepared()
    raw = {}
    for channel in ("co2", "o2"):
        obs = select_gathered_data_array(
            base.observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation="nmeasure")
        raw[channel] = operands(obs, c=400. if channel == "co2" else 209300.)
    return replace(base, boundary_operands=raw)


def options(base, mode="multiplicative"):
    mean = base.retained_prior.mean
    outer = mean.copy(data=[False, False, True, True, True])
    sampled = mean.copy(data=[False, False, True, False, False])
    calibration = None
    if mode == "multiplicative_shrunk":
        outer = mean.copy(data=[True, True, False, True, True])
        sampled = mean.copy(data=[True, True, False, False, False])
        weights = xr.DataArray(
            [[-1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]], dims=("process", "state"),
            coords={"process": ["GPP", "TER"], **dict(mean.coords)}, attrs={"units": "kg/year"},
        )
        reference = xr.DataArray(
            [-1., 1.], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"},
        )
        calibration = calibrate_biosphere_prior(
            base.retained_prior, biosphere=sampled, weights=weights, reference=reference, residual_sd_multiplier=.5,
        )
    return dict(
        boundary={channel: BoundaryCorrectionConfig("A", level_sd=(1., "ppm"), pattern_sd=(.3, "1"))
                  for channel in ("co2", "o2")},
        outer=outer, sampled_outer=sampled, mode=mode,
        delta_sd={"co2": xr.DataArray(2000., attrs={"units": "ppb"}),
                  "o2": xr.DataArray(.5, attrs={"units": "ppm"})},
        calibrated_prior=calibration,
    )


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_shared_codec_preserves_resolved_inputs(native, mode, extension, tmp_path, monkeypatch):
    kwargs = options(native, mode)
    prepared = prepare_linked_shared_inputs(native, **kwargs)
    assert not native.shared_baseline
    assert prepared.retained_prior is native.retained_prior
    assert prepared.aggregation_error is native.aggregation_error
    assert prepared.shared_baseline["co2"].delta_sd.item() == 2.
    assert prepared.shared_baseline["o2"].reference.item() == 209300.
    assert prepared.boundary_operands is native.boundary_operands
    tree = prepared.to_datatree()
    assert tree.attrs["schema_version"] == 4
    monkeypatch.setattr(
        "openghg_inversions.biosphere_prior.calibrate_biosphere_prior",
        lambda *a, **kw: pytest.fail("loading recalibrated frozen inputs"),
    )
    path = tmp_path / f"prepared.{extension}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_equal(restored.boundary_operands[channel], native.boundary_operands[channel])
        xr.testing.assert_identical(restored.shared_baseline[channel], prepared.shared_baseline[channel])
        assert restored.shared_baseline[channel].attrs["mode"] == mode
    xr.testing.assert_identical(restored.retained_prior.mean, native.retained_prior.mean)
    xr.testing.assert_equal(restored.aggregation_error.covariance, native.aggregation_error.covariance)
    for channel in ("co2", "o2"):
        np.testing.assert_array_equal(restored.shared_baseline[channel].original_outer, kwargs["outer"])
        np.testing.assert_array_equal(restored.shared_baseline[channel].original_sampled_outer, kwargs["sampled_outer"])


def test_preparation_and_tree_borrow_lazy_transport(native):
    lazy = replace(
        native, co2_sensitivity=native.co2_sensitivity.chunk(), o2_sensitivity=native.o2_sensitivity.chunk(),
        boundary_operands={channel: value.chunk({"nmeasure": 1}) for channel, value in native.boundary_operands.items()},
    )
    with Callback(pretask=lambda *a: pytest.fail("preparation materialized borrowed transport")):
        prepared = prepare_linked_shared_inputs(lazy, **options(lazy))
        tree = prepared.to_datatree()
    assert prepared.co2_sensitivity.data is lazy.co2_sensitivity.data
    assert isinstance(tree["shared_baseline_o2"].flux_sensitivity.data, da.Array)
    assert isinstance(tree["shared_baseline_o2"].bc_prescribed_centered.data, da.Array)


@pytest.mark.parametrize("frequency", [None, "2h"])
def test_lazy_receptor_metadata_does_not_execute_transport(native, frequency):
    from dask import delayed

    def forbidden():
        pytest.fail("boundary preparation computed scientific payloads")

    observations = native.observations.copy(data=da.from_delayed(
        delayed(forbidden)(), shape=native.observations.shape, dtype=float,
    ))
    observations = observations.assign_coords({
        name: native.observations[name].copy(data=da.from_array(native.observations[name].values, chunks=2))
        for name in ("site", "time")
    })
    sensitivities = {
        f"{channel}_sensitivity": getattr(native, f"{channel}_sensitivity").copy(data=da.from_delayed(
            delayed(forbidden)(), shape=getattr(native, f"{channel}_sensitivity").shape, dtype=float,
        )) for channel in ("co2", "o2")
    }
    lazy = replace(native, observations=observations,
                   **sensitivities)
    kwargs = options(lazy)
    kwargs["boundary"] = {
        channel: BoundaryCorrectionConfig(
            "A", level_sd=(1., "ppm"), pattern_sd=(.3, "1"), frequency=frequency,
            anchor_time="2021-01-01" if frequency else None,
        ) for channel in ("co2", "o2")
    }
    result = prepare_linked_shared_inputs(lazy, **kwargs)
    assert result.observations.data is observations.data
    for channel, frozen in result.shared_baseline.items():
        assert isinstance(frozen.flux_sensitivity.data, da.Array)
        assert not isinstance(frozen.site.data, da.Array)


def test_subsets_preserve_whole_support_means_and_refuse_repreparation(native):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    for channel, frozen in prepared.shared_baseline.items():
        subset = frozen.isel({frozen.attrs["native_observation_dim"]: [0]})
        xr.testing.assert_identical(subset.bc_pattern_mean, frozen.bc_pattern_mean)
        xr.testing.assert_identical(subset.outer_column_mean, frozen.outer_column_mean)
        xr.testing.assert_identical(subset.reference, frozen.reference)
    with pytest.raises(ValueError, match="already frozen"):
        prepare_linked_shared_inputs(prepared, **options(native))


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_artifacts_have_no_shared_defaults(native, version):
    if version == 1:
        native = replace(native, boundary_operands={})
    if version == 3:
        native = replace(
            native,
            boundary_correction_sensitivity={c: v.H_bc_mass for c, v in native.boundary_operands.items()},
            bc_centering_weights={c: xr.ones_like(v.bc_region, dtype=float).assign_attrs(units="1")
                                  for c, v in native.boundary_operands.items()},
        )
    tree = native.to_datatree()
    assert tree.attrs["schema_version"] == version
    assert not Co2O2PreparedInputs.from_datatree(tree).shared_baseline


@pytest.mark.parametrize("fault", ["channel", "prior", "rows", "units", "mode", "selected_prior", "selector"])
def test_reject_mismatched_frozen_identity(native, fault):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    shared = {channel: value.copy(deep=True) for channel, value in prepared.shared_baseline.items()}
    if fault == "channel":
        shared.pop("o2")
    elif fault == "prior":
        shared["o2"]["original_prior_mean"] = shared["o2"].original_prior_mean * 2
    elif fault == "rows":
        shared["o2"] = shared["o2"].isel(o2_measure=[0])
    elif fault == "units":
        shared["o2"].attrs["units"] = "ppb"
    elif fault == "mode":
        shared["o2"].attrs["mode"] = "fixed"
    elif fault == "selected_prior":
        shared["o2"]["retained_mean"] = shared["o2"].retained_mean * 2
    else:
        shared["o2"]["original_outer"] = ~shared["o2"].original_outer
    with pytest.raises(ValueError):
        replace(prepared, shared_baseline=shared).to_datatree()


@pytest.mark.parametrize("fault", ["old_version", "missing_channel", "invalid_width"])
def test_external_shared_artifact_validation(native, fault):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    tree = prepared.to_datatree()
    if fault == "old_version":
        tree.attrs["schema_version"] = 3
    elif fault == "missing_channel":
        del tree["shared_baseline_o2"]
    else:
        tree["shared_baseline_o2"]["delta_sd"] = xr.DataArray(-1., attrs={"units": "ppm"})
    with pytest.raises(ValueError):
        Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("fault", ["sensitivity", "fixed", "boundary"])
def test_external_codec_binds_frozen_operators_to_original_inputs(native, fault):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    if fault == "sensitivity":
        prepared = replace(prepared, co2_sensitivity=2 * prepared.co2_sensitivity)
    elif fault == "fixed":
        prepared = replace(prepared, fixed_prior_contribution=prepared.fixed_prior_contribution + 1.)
    else:
        changed = prepared.boundary_operands["o2"].copy()
        changed["H_bc_anomaly"] = (changed.H_bc_anomaly + 1.).assign_attrs(changed.H_bc_anomaly.attrs)
        prepared = replace(prepared, boundary_operands={**prepared.boundary_operands, "o2": changed})
    with pytest.raises(ValueError, match="Shared baseline"):
        Co2O2PreparedInputs.from_datatree(prepared.to_datatree())


def test_to_datatree_defers_lazy_observation_values_and_unit_coordinates(native):
    from dask import delayed

    prepared = prepare_linked_shared_inputs(native, **options(native))

    @delayed
    def borrowed():
        pytest.fail("to_datatree executed the shared observation/unit graph")

    graph = borrowed()
    obs = native.observations.copy(data=da.from_delayed(graph, shape=native.observations.shape, dtype=float))
    obs = obs.assign_coords(observation_units=("observation", da.from_delayed(graph, shape=obs.shape, dtype=str)))
    # Canonical joint arrays share their observation-unit coordinate owner.
    units = obs.observation_units
    prepared = replace(
        prepared, observations=obs,
        fixed_prior_contribution=prepared.fixed_prior_contribution.assign_coords(observation_units=units),
        aggregation_error=replace(prepared.aggregation_error,
                                  covariance=prepared.aggregation_error.covariance.assign_coords(observation_units=units)),
    )
    with Callback(pretask=lambda *a: pytest.fail("to_datatree computed borrowed data")):
        tree = prepared.to_datatree()
    assert isinstance(tree["joint"].observation_units.data, da.Array)


def test_subset_validation_reuses_frozen_whole_window_reference(native):
    from openghg_inversions.rhime.co2.shared_preparation import _native_channel, _validate_operator_identity

    kwargs = options(native)
    kwargs["boundary"] = {channel: replace(config, reference="whole_window_transport_mean")
                          for channel, config in kwargs["boundary"].items()}
    prepared = prepare_linked_shared_inputs(native, **kwargs)
    frozen = prepared.shared_baseline["o2"].isel(o2_measure=[0])
    obs = _native_channel(native.observations, "o2", "o2_measure").isel(o2_measure=[0]).assign_attrs(units="ppm")
    fixed = _native_channel(native.fixed_prior_contribution, "o2", "o2_measure").isel(o2_measure=[0])
    _validate_operator_identity(
        frozen, native.o2_sensitivity.isel(o2_measure=[0]), fixed,
        native.boundary_operands["o2"].isel(nmeasure=[0]), obs, native.retained_prior,
    )


def test_surface_example_frozen_codec_accepts_dask_reduction_roundoff(tmp_path):
    from pathlib import Path
    import runpy

    workflow = runpy.run_path(str(Path(__file__).parents[1] / "examples/shared_baseline/linked_workflow.py"))
    prepared = workflow["prepare_control"]()
    path = tmp_path / "surface-prepared.nc"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored.shared_baseline[channel], prepared.shared_baseline[channel].compute())


@pytest.mark.parametrize("fault", ["missing_raw", "delta_vector", "delta_units", "additive"])
def test_explicit_preparation_rejects_missing_or_unsupported_inputs(native, fault):
    kwargs = options(native)
    if fault == "missing_raw":
        native = replace(native, boundary_operands={"co2": native.boundary_operands["co2"]})
    elif fault == "delta_vector":
        kwargs["delta_sd"]["o2"] = xr.DataArray([.5], dims="width", attrs={"units": "ppm"})
    elif fault == "delta_units":
        kwargs["delta_sd"]["o2"] = xr.DataArray(.5)
    else:
        kwargs["mode"] = "additive"
    with pytest.raises(ValueError, match="oxidative allocation" if fault == "additive" else "co2|scalar"):
        prepare_linked_shared_inputs(native, **kwargs)
