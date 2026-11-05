"""Frozen linked preparation and its current-only scientific codec."""

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
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.shared_baseline import SharedBaselineChoice
from openghg_inversions.rhime.co2.shared_preparation import prepare_linked_shared_inputs
from test_shared_baseline import operands as bound_operands
from test_co2_o2_fixed_ou import _prepared


def operands(obs, c=400.):
    raw = bound_operands(obs, c=c)
    return xr.Dataset(dict(H_bc_anomaly=raw.anomaly, H_bc_mass=raw.mass, bc_reference=raw.reference),
                      attrs={"boundary_representation": "centered-v1"})


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
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
        choices={channel: SharedBaselineChoice(AnomalyScalingChoice(None, .3), 2. if channel == "co2" else .5, mode)
                 for channel in ("co2", "o2")},
        outer=outer, sampled_outer=sampled,
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
    assert prepared.shared_baseline["co2"].data.delta_sd.item() == 2.
    assert prepared.shared_baseline["o2"].data.reference.item() == 209300.
    assert prepared.boundary_operands is native.boundary_operands
    tree = prepared.to_datatree()
    assert tree.attrs["schema_version"] == 6
    monkeypatch.setattr(SharedBaselineChoice, "bind", lambda *a, **kw: pytest.fail("load rebound frozen support"))
    monkeypatch.setattr(
        "openghg_inversions.biosphere_prior.calibrate_biosphere_prior",
        lambda *a, **kw: pytest.fail("loading recalibrated frozen inputs"),
    )
    path = tmp_path / f"prepared.{extension}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_equal(restored.boundary_operands[channel], native.boundary_operands[channel])
        xr.testing.assert_identical(restored.shared_baseline[channel].data, prepared.shared_baseline[channel].data)
        assert restored.shared_baseline[channel].data.attrs["mode"] == mode
    xr.testing.assert_identical(restored.retained_prior.mean, native.retained_prior.mean)
    xr.testing.assert_equal(restored.aggregation_error.covariance, native.aggregation_error.covariance)
    for channel in ("co2", "o2"):
        np.testing.assert_array_equal(restored.shared_baseline[channel].data.original_outer, kwargs["outer"])
        np.testing.assert_array_equal(restored.shared_baseline[channel].data.original_sampled_outer, kwargs["sampled_outer"])


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
    kwargs["choices"] = {
        channel: replace(choice, boundary=replace(choice.boundary, frequency=frequency,
                          anchor_time="2021-01-01" if frequency else None))
        for channel, choice in kwargs["choices"].items()
    }
    result = prepare_linked_shared_inputs(lazy, **kwargs)
    assert result.observations.data is observations.data
    for channel, bound in result.shared_baseline.items():
        frozen = bound.data
        assert isinstance(frozen.flux_sensitivity.data, da.Array)
        assert not isinstance(frozen.site.data, da.Array)


def test_subsets_preserve_whole_support_means_and_refuse_repreparation(native):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    for channel, bound in prepared.shared_baseline.items():
        frozen = bound.data
        subset = frozen.isel({frozen.attrs["native_observation_dim"]: [0]})
        xr.testing.assert_identical(subset.bc_pattern_mean, frozen.bc_pattern_mean)
        xr.testing.assert_identical(subset.outer_column_mean, frozen.outer_column_mean)
        xr.testing.assert_identical(subset.reference, frozen.reference)
    with pytest.raises(ValueError, match="already frozen"):
        prepare_linked_shared_inputs(prepared, **options(native))


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_old_artifacts_have_no_shared_defaults(native, version):
    tree = native.to_datatree()
    tree.attrs["schema_version"] = version
    with pytest.raises(ValueError, match="prepare again"):
        Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("fault", ["channel", "prior", "rows", "units", "mode", "selected_prior", "selector"])
def test_reject_mismatched_frozen_identity(native, fault):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    shared = {channel: replace(value, data=value.data.copy(deep=True)) for channel, value in prepared.shared_baseline.items()}
    if fault == "channel":
        shared.pop("o2")
    elif fault == "prior":
        shared["o2"].data["original_prior_mean"] = shared["o2"].data.original_prior_mean * 2
    elif fault == "rows":
        shared["o2"] = replace(shared["o2"], data=shared["o2"].data.isel(o2_measure=[0]))
    elif fault == "units":
        shared["o2"].data.attrs["units"] = "ppb"
    elif fault == "mode":
        shared["o2"].data.attrs["mode"] = "fixed"
    elif fault == "selected_prior":
        shared["o2"].data["retained_mean"] = shared["o2"].data.retained_mean * 2
    else:
        shared["o2"].data["original_outer"] = ~shared["o2"].data.original_outer
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
    kwargs["choices"] = {channel: replace(choice, boundary=replace(choice.boundary, reference="whole_window_transport_mean"))
                         for channel, choice in kwargs["choices"].items()}
    prepared = prepare_linked_shared_inputs(native, **kwargs)
    frozen = prepared.shared_baseline["o2"].data.isel(o2_measure=[0])
    obs = _native_channel(native.observations, "o2", "o2_measure").isel(o2_measure=[0]).assign_attrs(units="ppm")
    fixed = _native_channel(native.fixed_prior_contribution, "o2", "o2_measure").isel(o2_measure=[0])
    _validate_operator_identity(
        frozen, native.o2_sensitivity.isel(o2_measure=[0]), fixed,
        native.boundary_operands["o2"].isel(nmeasure=[0]), obs, native.retained_prior,
    )


@pytest.mark.parametrize("fault", ["missing_raw", "additive"])
def test_explicit_preparation_rejects_missing_or_unsupported_inputs(native, fault):
    kwargs = options(native)
    if fault == "missing_raw":
        native = replace(native, boundary_operands={"co2": native.boundary_operands["co2"]})
    else:
        kwargs["choices"] = {channel: replace(choice, mode="additive") for channel, choice in kwargs["choices"].items()}
    with pytest.raises(ValueError, match="oxidative allocation" if fault == "additive" else "co2"):
        prepare_linked_shared_inputs(native, **kwargs)


@pytest.fixture
def gathered_native(native):
    from openghg_inversions.rhime.co2.co2_o2_preparation import prepare_co2_o2_inputs
    from test_rhime_co2_o2 import _inputs

    inputs = _inputs(gathered_state=True)
    inputs["o2_units"] = "ppm"
    for channel in ("co2", "o2"):
        row = str(inputs[f"{channel}_observations"].dims[0])
        inputs[f"{channel}_observations"] = select_gathered_data_array(
            native.observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row)
    return replace(prepare_co2_o2_inputs(**inputs), boundary_operands=native.boundary_operands)


@pytest.mark.parametrize("extension", ["nc", "zarr"])
@pytest.mark.parametrize("empty_outer", [False, True])
def test_gathered_shared_preparation_survives_repeated_save_load(gathered_native, tmp_path, extension, empty_outer):
    import pandas as pd

    base = gathered_native
    kwargs = options(base)
    if empty_outer:
        kwargs.update(outer=base.retained_prior.mean.copy(data=np.zeros(5, dtype=bool)),
                      sampled_outer=base.retained_prior.mean.copy(data=np.zeros(5, dtype=bool)))
    original = prepare_linked_shared_inputs(base, **kwargs)
    actual = original
    for generation in range(3):
        path = tmp_path / f"shared-{generation}.{extension}"
        actual.save(path)
        actual = Co2O2PreparedInputs.load(path)
        xr.testing.assert_identical(actual.observations, original.observations)
        xr.testing.assert_identical(actual.retained_prior.mean, original.retained_prior.mean)
        xr.testing.assert_identical(actual.retained_prior.arithmetic_covariance,
                                    original.retained_prior.arithmetic_covariance)
        for channel in ("co2", "o2"):
            expected, restored = original.shared_baseline[channel].data, actual.shared_baseline[channel].data
            xr.testing.assert_identical(restored, expected)
            xr.testing.assert_identical(getattr(actual, f"{channel}_sensitivity"), getattr(original, f"{channel}_sensitivity"))
            index = restored.indexes["physical_outer_state"]
            assert isinstance(index, pd.MultiIndex)
            assert index.names == expected.indexes["physical_outer_state"].names
            if empty_outer:
                for left, right in zip(index.levels, expected.indexes["physical_outer_state"].levels, strict=True):
                    np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, True, "6", None])
def test_old_or_malformed_version_rejected_before_payload(native, version):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    tree = prepared.to_datatree()
    tree.attrs["schema_version"] = version
    with Callback(pretask=lambda *a: pytest.fail("unsupported format executed payload")):
        with pytest.raises(ValueError, match="prepare again"):
            Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("fault", ["partial", "old_shared"])
def test_shared_schema_preflight_before_payload(native, fault):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    tree = prepared.to_datatree()
    if fault == "partial":
        del tree["shared_baseline_o2"]
    else:
        tree["shared_baseline_o2"].attrs["shared_snapshot_schema"] = "shared-atmospheric-baseline-v1"
    with Callback(pretask=lambda *a: pytest.fail("invalid shared format executed payload")):
        with pytest.raises(ValueError, match="prepare again"):
            Co2O2PreparedInputs.from_datatree(tree)


def test_preparation_metadata_and_widths_share_one_small_boundary(native):
    from dask import delayed
    calls = []

    @delayed
    def metadata():
        calls.append("metadata")
        return 1.

    token = da.from_delayed(metadata(), shape=(), dtype=float)
    observations = native.observations.assign_coords({
        name: native.observations[name].copy(data=da.where(token == 1, native.observations[name].data,
                                                         native.observations[name].data))
        for name in ("site", "time", "observation_units")
    })
    kwargs = options(native)
    prepared = prepare_linked_shared_inputs(replace(native, observations=observations), **kwargs)
    assert calls == ["metadata"]
    assert prepared.observations is observations


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_ordinary_codec_also_uses_only_version_six(native, tmp_path, extension):
    tree = native.to_datatree()
    assert tree.attrs["schema_version"] == 6
    path = tmp_path / f"ordinary.{extension}"
    native.save(path)
    restored = Co2O2PreparedInputs.load(path)
    assert not restored.shared_baseline
    xr.testing.assert_identical(restored.observations, native.observations)


@pytest.mark.parametrize("fault", ["mass", "reference", "ratio_values", "ratio_provenance", "native_units", "calibration"])
def test_live_shared_boundary_rejects_changed_scientific_inputs(native, monkeypatch, fault):
    from openghg_inversions.rhime.co2 import shared_workflow
    prepared = prepare_linked_shared_inputs(native, **options(native, "multiplicative_shrunk"))
    prepared = replace(prepared, independent_error_sd=xr.full_like(prepared.observations, .1))
    if fault in {"mass", "reference"}:
        raw = prepared.boundary_operands["o2"].copy()
        name = "H_bc_mass" if fault == "mass" else "bc_reference"
        raw[name] = (raw[name] * 2).assign_attrs(raw[name].attrs)
        prepared = replace(prepared, boundary_operands={**prepared.boundary_operands, "o2": raw})
    elif fault == "ratio_values":
        ratio = prepared.o2_co2_flux_ratio
        prepared = replace(prepared, o2_co2_flux_ratio=ratio.copy(data=ratio.data * 2))
    elif fault in {"ratio_provenance", "native_units"}:
        sensitivity = prepared.o2_sensitivity.assign_attrs(
            **({"oxidation_ratio_provenance": "{}"} if fault == "ratio_provenance" else {"units": "ppb"}))
        prepared = replace(prepared, o2_sensitivity=sensitivity)
    else:
        shared = {channel: replace(value, data=value.data.copy(deep=True)) for channel, value in prepared.shared_baseline.items()}
        for bound in shared.values():
            bound.data["calibration_baseline_mean"] = bound.data.calibration_baseline_mean * 2
        prepared = replace(prepared, shared_baseline=shared)
    monkeypatch.setattr(shared_workflow, "build_shared_co2_o2_model", lambda *a, **kw: pytest.fail("fit invalid inputs"))
    with pytest.raises(ValueError):
        shared_workflow.build_linked_shared_workflow(prepared)


@pytest.mark.parametrize("extension", ["nc", "zarr"])
@pytest.mark.parametrize("fault", ["old_version", "partial", "old_correction"])
def test_file_schema_preflight_precedes_payload_loading(native, tmp_path, monkeypatch, extension, fault):
    from openghg_inversions.serialization import save_datatree
    prepared = prepare_linked_shared_inputs(native, **options(native))
    tree = prepared.to_datatree()
    if fault == "old_version":
        tree.attrs["schema_version"] = 4
    elif fault == "partial":
        del tree["shared_baseline_o2"]
    else:
        tree["shared_baseline_o2"].attrs["shared_choice_json"] = '{"mode": "legacy"}'
    path = tmp_path / f"unsupported.{extension}"
    save_datatree(tree, path)
    monkeypatch.setattr(xr.DataTree, "load", lambda *a, **kw: pytest.fail("invalid file loaded numerical payloads"))
    with pytest.raises(ValueError, match="prepare again"):
        Co2O2PreparedInputs.load(path)


def test_preparation_preserves_fixed_contribution_unit_validation(native):
    fixed = native.fixed_prior_contribution.assign_coords(
        observation_units=xr.full_like(native.fixed_prior_contribution.observation_units, "ppb"))
    with pytest.raises(ValueError, match="fixed contribution units"):
        prepare_linked_shared_inputs(replace(native, fixed_prior_contribution=fixed), **options(native))


def test_raw_time_snapshot_identity_never_rebinds_on_load(native, monkeypatch, tmp_path):
    from test_rhime_co2_o2 import _inputs
    from openghg_inversions.rhime.co2.co2_o2_preparation import prepare_co2_o2_inputs

    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    for channel in ("co2", "o2"):
        obs = select_gathered_data_array(native.observations, key=channel, key_dim="species",
                                        ragged_dim="channel_observation", stack_dim="observation")
        row = f"{channel}_measure"
        for name, array in list(inputs.items()):
            if isinstance(array, xr.DataArray):
                coords = {dim: obs.time.values for dim in (row, f"{row}_cov") if dim in array.dims}
                if coords:
                    inputs[name] = array.assign_coords(coords)
        inputs[f"{channel}_observations"] = inputs[f"{channel}_observations"].assign_coords(
            site=(row, obs.site.values), time=(row, obs.time.values))
    native = prepare_co2_o2_inputs(**inputs)
    raw = {}
    for channel in ("co2", "o2"):
        obs = select_gathered_data_array(native.observations, key=channel, key_dim="species",
                                        ragged_dim="channel_observation", stack_dim="observation").rename(observation="nmeasure")
        value = operands(obs, c=400. if channel == "co2" else 209300.)
        raw[channel] = value.rename(nmeasure="time")
    native = replace(native, boundary_operands=raw)
    prepared = prepare_linked_shared_inputs(native, **options(native))
    path = tmp_path / "time.nc"
    prepared.save(path)
    monkeypatch.setattr(SharedBaselineChoice, "bind", lambda *a, **kw: pytest.fail("snapshot load rebound support"))
    monkeypatch.setattr("openghg_inversions.models.boundary.CenteredOperands.bind",
                        lambda *a, **kw: pytest.fail("snapshot load rebuilt calendar"))
    restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored.shared_baseline[channel].data, prepared.shared_baseline[channel].data)


@pytest.mark.parametrize("channel", ["co2", "o2"])
def test_snapshot_load_rejects_missing_raw_pattern_operand(native, channel):
    prepared = prepare_linked_shared_inputs(native, **options(native))
    tree = prepared.to_datatree()
    del tree[f"shared_baseline_{channel}"]["bc_pattern_raw"]
    with pytest.raises(ValueError, match="missing frozen scientific operands; prepare again"):
        Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("extension", ["nc", "zarr"])
@pytest.mark.parametrize("boundary", [None, AdditiveChoice(None, .3, anomalies="period_contrasts")])
def test_snapshot_choice_codec_retains_prescribed_and_additive_choices(native, tmp_path, extension, boundary):
    kwargs = options(native)
    kwargs["choices"] = {channel: replace(choice, boundary=boundary) for channel, choice in kwargs["choices"].items()}
    prepared = prepare_linked_shared_inputs(native, **kwargs)
    path = tmp_path / f"choice.{extension}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        assert restored.shared_baseline[channel].choice == prepared.shared_baseline[channel].choice
        xr.testing.assert_identical(restored.shared_baseline[channel].data, prepared.shared_baseline[channel].data)


@pytest.mark.parametrize("boundary", [None, AdditiveChoice(None, .3, anomalies="period_contrasts"),
                                      AnomalyScalingChoice(None, .3)])
def test_snapshot_rejects_changed_stored_reference_with_equal_physical_boundary(native, boundary):
    kwargs = options(native)
    kwargs["choices"] = {channel: replace(choice, boundary=boundary) for channel, choice in kwargs["choices"].items()}
    prepared = prepare_linked_shared_inputs(native, **kwargs)
    tree = prepared.to_datatree()
    frozen = tree["shared_baseline_co2"]
    mass = frozen.native_H_bc_mass.sum("native_bc_region")
    frozen["reference"] = (frozen.reference + 1.).assign_attrs(frozen.reference.attrs)
    frozen["bc_prescribed_centered"] = (frozen.bc_prescribed_centered - mass).assign_attrs(frozen.bc_prescribed_centered.attrs)
    frozen["bc_reference_shift"] = (frozen.bc_reference_shift + mass).assign_attrs(frozen.bc_reference_shift.attrs)
    # A compensating anomaly/shift change preserves the physical BC but changes
    # a reference that these three choices explicitly promise to keep stored.
    np.testing.assert_allclose(frozen.bc_prescribed_centered + frozen.bc_reference_shift, frozen.physical_bc_prior)
    with pytest.raises(ValueError, match="stored reference values must match"):
        Co2O2PreparedInputs.from_datatree(tree)
