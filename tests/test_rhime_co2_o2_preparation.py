"""Focused preparation contracts for the CO2/O2 recipe."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import warnings

from dask import array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_preparation import _stack
from openghg_inversions.serialization import decode_cf_multiindexes, encode_cf_multiindexes, save_datatree


def _inputs(
    *,
    co2_o2_ocean_loading: float = 0.0,
    o2_co2_ocean_loading: float = 0.0,
    ratio_state: list[str] | None = None,
    ratio_source: list[str] | None = None,
    ratio_values: list[float] | None = None,
    ratio_direction: str = "O2 flux per CO2 flux",
    ratio_sign: str = "signed; positive CO2 flux has negative O2 loading",
    ratio_available: bool = True,
    unavailable_reason: str = "",
) -> dict[str, object]:
    state = ["gpp:1", "ter:1", "ff:1", "co2-ocean:1", "o2-ocean:1"]
    mean = xr.DataArray(
        np.ones(5),
        dims="state",
        coords={
            "state": state,
            "source": ("state", ["GPP", "TER", "FF", "ocean", "ocean"]),
            "tracer_scope": ("state", ["shared", "shared", "shared", "co2", "o2"]),
        },
    )
    co2_labels = ["c1", "c2"]
    o2_labels = ["o1", "o2", "o3"]
    co2 = xr.DataArray(
        da.from_array([2.0, 3.0], chunks=1),
        dims="co2_measure",
        coords={
            "co2_measure": co2_labels,
            "time": ("co2_measure", np.array(["2021-01-01", "2021-01-03"], dtype="datetime64[D]")),
        },
    )
    o2 = xr.DataArray(
        da.from_array([-4.0, -5.0, -6.0], chunks=1),
        dims="o2_measure",
        coords={
            "o2_measure": o2_labels,
            "time": (
                "o2_measure",
                np.array(["2021-01-02", "2021-01-04", "2021-01-05"], dtype="datetime64[D]"),
            ),
        },
    )
    return {
        "co2_observations": co2,
        "o2_observations": o2,
        "co2_prior_forward_mean": co2 - 0.25,
        "o2_prior_forward_mean": o2 + 0.5,
        "co2_sensitivity": xr.DataArray(
            da.from_array(
                [[1, 2, 3, 4, co2_o2_ocean_loading], [0.5, 1, 1.5, 2, 0]],
                chunks=(1, 5),
            ),
            dims=("co2_measure", "state"),
            coords={"co2_measure": co2_labels, "state": state},
        ),
        "o2_sensitivity": xr.DataArray(
            da.from_array(
                [
                    [-1, -2, -3, o2_co2_ocean_loading, 5],
                    [-0.5, -1, -1.5, 0, 2.5],
                    [-0.2, -0.4, -0.6, 0, 1],
                ],
                chunks=(1, 5),
            ),
            dims=("o2_measure", "state"),
            coords={"o2_measure": o2_labels, "state": state},
        ),
        "o2_co2_flux_ratio": (
            xr.DataArray(
                da.from_array(
                    [-1.1, -1.0, -1.4] if ratio_values is None else ratio_values,
                    chunks=1,
                ),
                dims="state",
                coords={
                    "state": state[:3] if ratio_state is None else ratio_state,
                    "source": (
                        "state",
                        ["GPP", "TER", "FF"] if ratio_source is None else ratio_source,
                    ),
                },
                attrs={
                    "direction": ratio_direction,
                    "sign_convention": ratio_sign,
                    "provenance": "Verification Games source-resolved O2:CO2 ratios",
                },
            )
            if ratio_available
            else None
        ),
        "o2_co2_flux_ratio_unavailable_reason": unavailable_reason,
        "co2_aggregation_covariance": xr.DataArray(
            da.from_array(np.eye(2), chunks=(1, 2)),
            dims=("co2_measure", "co2_measure_cov"),
            coords={"co2_measure": co2_labels, "co2_measure_cov": co2_labels},
        ),
        "co2_o2_aggregation_covariance": xr.DataArray(
            da.from_array(np.zeros((2, 3)), chunks=(1, 3)),
            dims=("co2_measure", "o2_measure"),
            coords={"co2_measure": co2_labels, "o2_measure": o2_labels},
        ),
        "o2_aggregation_covariance": xr.DataArray(
            da.from_array(np.eye(3), chunks=(1, 3)),
            dims=("o2_measure", "o2_measure_cov"),
            coords={"o2_measure": o2_labels, "o2_measure_cov": o2_labels},
        ),
        "retained_prior": CorrelatedLognormalPrior(mean, np.eye(5) * 0.01),
        "co2_units": "ppm",
        "o2_units": "per meg",
    }


def test_preparation_preserves_lazy_channels_with_staggered_unequal_times(tmp_path) -> None:
    inputs = _inputs()
    ratio = inputs["o2_co2_flux_ratio"]
    assert isinstance(ratio, xr.DataArray)
    prepared = prepare_co2_o2_inputs(**inputs)

    assert isinstance(prepared.observations.data, da.Array)
    assert isinstance(prepared.fixed_prior_contribution.data, da.Array)
    assert isinstance(prepared.co2_sensitivity.data, da.Array)
    assert isinstance(prepared.o2_sensitivity.data, da.Array)
    assert isinstance(prepared.o2_co2_flux_ratio.data, da.Array)
    assert prepared.o2_co2_flux_ratio.data is ratio.data
    assert not isinstance(prepared.aggregation_error.covariance.data, da.Array)
    assert prepared.observations["species"].values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert prepared.observations["channel_observation"].values.tolist() == [
        "c1",
        "c2",
        "o1",
        "o2",
        "o3",
    ]
    covariance = prepared.aggregation_error.covariance
    assert covariance is not None
    assert covariance.indexes["observation"].equals(prepared.observations.indexes["observation"])
    np.testing.assert_array_equal(
        covariance.indexes["observation_cov"].values,
        prepared.observations.indexes["observation"].values,
    )
    artifact = xr.Dataset(
        {
            "observations": prepared.observations,
            "fixed_prior_contribution": prepared.fixed_prior_contribution,
            "aggregation_error_covariance": covariance,
        }
    )
    path = tmp_path / "prepared_observations.nc"
    encode_cf_multiindexes(artifact, ("observation", "observation_cov")).to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as stored:
        restored = decode_cf_multiindexes(stored.load(), ("observation", "observation_cov"))
    assert restored.indexes["observation"].equals(prepared.observations.indexes["observation"])
    np.testing.assert_array_equal(
        restored.indexes["observation_cov"].values,
        prepared.observations.indexes["observation"].values,
    )
    np.testing.assert_array_equal(
        prepared.observations["time"],
        np.array(
            ["2021-01-01", "2021-01-03", "2021-01-02", "2021-01-04", "2021-01-05"],
            dtype="datetime64[D]",
        ),
    )
    np.testing.assert_allclose(
        prepared.fixed_prior_contribution,
        [-8.25, -2.25, -2.5, -4.0, -5.3],
    )
    assert prepared.fixed_prior_contribution.attrs["mathematical_name"] == ("H m - H_alpha Pi m")
    ratio_provenance = prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"]
    assert '"state": ["gpp:1", "ter:1", "ff:1"]' in ratio_provenance
    assert '"value": [-1.1, -1.0, -1.4]' in ratio_provenance


def test_canonicalizes_sensitivity_state_metadata_without_mutating_inputs() -> None:
    inputs = _inputs()
    originals: dict[str, xr.DataArray] = {}
    for name in ("co2_sensitivity", "o2_sensitivity"):
        sensitivity = inputs[name]
        assert isinstance(sensitivity, xr.DataArray)
        sensitivity = sensitivity.assign_coords(
            source=("state", ["stale"] * 5),
            tracer_scope=("state", ["wrong"] * 5),
        )
        inputs[name] = sensitivity
        originals[name] = sensitivity.copy(deep=True)

    prepared = prepare_co2_o2_inputs(**inputs)
    prior = inputs["retained_prior"]
    assert isinstance(prior, CorrelatedLognormalPrior)
    for name, prepared_sensitivity in (
        ("co2_sensitivity", prepared.co2_sensitivity),
        ("o2_sensitivity", prepared.o2_sensitivity),
    ):
        np.testing.assert_array_equal(prepared_sensitivity["source"], prior.mean["source"])
        np.testing.assert_array_equal(prepared_sensitivity["tracer_scope"], prior.mean["tracer_scope"])
        xr.testing.assert_identical(inputs[name], originals[name])


@pytest.mark.parametrize("gpp_sources", [("GPP", "GPP"), ("gpp", "gpp"), ("GPP", "gpp")])
def test_source_spelling_is_consistent_across_retained_states(gpp_sources) -> None:
    inputs = _inputs()
    state = ["gpp:1", "gpp:2", "ter:1", "ff:1", "co2-ocean:1", "o2-ocean:1"]
    sources = [*gpp_sources, "TER", "FF", "ocean", "ocean"]
    prior = inputs["retained_prior"]
    mean = prior.mean.isel(state=[0, 0, 1, 2, 3, 4]).assign_coords(state=state, source=("state", sources))
    inputs["retained_prior"] = CorrelatedLognormalPrior(mean, np.eye(6) * 0.01)
    for name in ("co2_sensitivity", "o2_sensitivity"):
        inputs[name] = inputs[name].isel(state=[0, 0, 1, 2, 3, 4]).assign_coords(state=state)
    inputs["o2_co2_flux_ratio"] = (
        inputs["o2_co2_flux_ratio"]
        .isel(state=[0, 0, 1, 2])
        .assign_coords(state=state[:4], source=("state", sources[:4]))
    )

    if gpp_sources[0] != gpp_sources[1]:
        with pytest.raises(ValueError, match="source labels.*consistent spelling"):
            prepare_co2_o2_inputs(**inputs)
    else:
        prepared = prepare_co2_o2_inputs(**inputs)
        xr.testing.assert_identical(prepared.retained_prior.mean, mean)


def _with_state_index(array: xr.DataArray, index: pd.MultiIndex) -> xr.DataArray:
    """Return an array with one replacement state index for boundary tests."""
    result = array.reset_index("state")
    removable = {
        "state",
        "source",
        "tracer_scope",
        "region_in_source",
        *array.indexes["state"].names,
    }
    result = result.drop_vars([name for name in removable if name in result.coords])
    return result.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "state"))


def test_rejects_sensitivity_with_stale_gathered_state_level_names() -> None:
    inputs = _inputs()
    state_index = pd.MultiIndex.from_arrays(
        [
            ["GPP", "TER", "FF", "ocean", "ocean"],
            ["shared", "shared", "shared", "co2", "o2"],
            [1, 1, 1, 1, 1],
        ],
        names=("source", "tracer_scope", "region_in_source"),
    )
    prior = inputs["retained_prior"]
    assert isinstance(prior, CorrelatedLognormalPrior)
    inputs["retained_prior"] = CorrelatedLognormalPrior(
        _with_state_index(prior.mean, state_index),
        np.eye(5) * 0.01,
    )
    for name in ("co2_sensitivity", "o2_sensitivity"):
        value = inputs[name]
        assert isinstance(value, xr.DataArray)
        inputs[name] = _with_state_index(value, state_index)
    ratio = inputs["o2_co2_flux_ratio"]
    assert isinstance(ratio, xr.DataArray)
    inputs["o2_co2_flux_ratio"] = _with_state_index(ratio, state_index[:3])

    sensitivity = inputs["co2_sensitivity"]
    assert isinstance(sensitivity, xr.DataArray)
    inputs["co2_sensitivity"] = _with_state_index(
        sensitivity,
        state_index.set_names(("bad_source", "bad_scope", "bad_region")),
    )

    with pytest.raises(ValueError, match="state labels and index level names"):
        prepare_co2_o2_inputs(**inputs)


def test_native_datetime_labels_roundtrip_with_datetime_auxiliary(tmp_path) -> None:
    co2_index = pd.DatetimeIndex(["2021-01-01", "2021-01-03"], name="co2_measure")
    o2_index = pd.DatetimeIndex(
        ["2021-01-02", "2021-01-04", "2021-01-05"],
        name="o2_measure",
    )
    co2 = xr.DataArray(
        [2.0, 3.0],
        dims="co2_measure",
        coords={
            "co2_measure": co2_index,
            "time": ("co2_measure", co2_index.values),
        },
    )
    o2 = xr.DataArray(
        [-4.0, -5.0, -6.0],
        dims="o2_measure",
        coords={"o2_measure": o2_index, "time": ("o2_measure", o2_index.values)},
    )
    originals = (co2.copy(deep=True), o2.copy(deep=True))

    stacked = _stack(co2, o2, co2_units="ppm", o2_units="per meg", name="observations")
    expected = pd.MultiIndex.from_tuples(
        [("co2", label) for label in co2_index] + [("o2", label) for label in o2_index],
        names=("species", "channel_observation"),
    )
    assert stacked.indexes["observation"].equals(expected)
    assert np.issubdtype(stacked["time"].dtype, np.datetime64)
    xr.testing.assert_identical(co2, originals[0])
    xr.testing.assert_identical(o2, originals[1])

    path = tmp_path / "mixed_observation_labels.nc"
    encoded = encode_cf_multiindexes(stacked.to_dataset(name="observations"), "observation")
    encoded.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as stored:
        restored = decode_cf_multiindexes(stored.load(), "observation")
    assert restored.indexes["observation"].equals(expected)
    assert np.issubdtype(restored["time"].dtype, np.datetime64)


def test_native_labels_preserve_integer_string_collision() -> None:
    co2 = xr.DataArray([2.0], dims="co2_measure", coords={"co2_measure": [1]})
    o2 = xr.DataArray([-4.0], dims="o2_measure", coords={"o2_measure": ["1"]})

    stacked = _stack(co2, o2, co2_units="ppm", o2_units="per meg", name="observations")

    labels = stacked.indexes["observation"].get_level_values("channel_observation")
    assert labels.tolist() == [1, "1"]
    assert [type(label) for label in labels] == [int, str]


def test_multiindex_observation_labels_roundtrip_without_future_warnings(tmp_path) -> None:
    co2_index = pd.MultiIndex.from_tuples(
        [
            ("TAC", pd.Timestamp("2021-01-01")),
            ("MHD", pd.Timestamp("2021-01-03")),
        ],
        names=("site", "time"),
    )
    o2_index = pd.MultiIndex.from_tuples(
        [
            ("TAC", pd.Timestamp("2021-01-02")),
            ("MHD", pd.Timestamp("2021-01-04")),
            ("TAC", pd.Timestamp("2021-01-05")),
        ],
        names=("site", "time"),
    )
    co2 = xr.DataArray(
        [2.0, 3.0],
        dims="co2_measure",
        coords=xr.Coordinates.from_pandas_multiindex(co2_index, "co2_measure"),
    )
    o2 = xr.DataArray(
        [-4.0, -5.0, -6.0],
        dims="o2_measure",
        coords=xr.Coordinates.from_pandas_multiindex(o2_index, "o2_measure"),
    )
    originals = (co2.copy(deep=True), o2.copy(deep=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        stacked = _stack(co2, o2, co2_units="ppm", o2_units="per meg", name="observations")

    expected = pd.MultiIndex.from_tuples(
        [("co2", *label) for label in co2_index] + [("o2", *label) for label in o2_index],
        names=("species", "site", "time"),
    )
    assert stacked.indexes["observation"].equals(expected)
    assert stacked["site"].values.tolist() == ["TAC", "MHD", "TAC", "MHD", "TAC"]
    assert np.issubdtype(stacked["time"].dtype, np.datetime64)
    xr.testing.assert_identical(co2, originals[0])
    xr.testing.assert_identical(o2, originals[1])

    path = tmp_path / "multiindex_observation_labels.nc"
    encoded = encode_cf_multiindexes(stacked.to_dataset(name="observations"), "observation")
    encoded.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as stored:
        restored = decode_cf_multiindexes(stored.load(), "observation")
    assert restored.indexes["observation"].equals(expected)


def test_tracks_unavailable_ratio_values_without_claiming_source_resolved_provenance() -> None:
    prepared = prepare_co2_o2_inputs(
        **_inputs(
            ratio_available=False,
            unavailable_reason="Only spatially resolved native O2 flux treatment is documented.",
        )
    )

    provenance = prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"]
    assert prepared.o2_co2_flux_ratio is None
    assert prepared.o2_co2_flux_ratio_unavailable_reason.startswith("Only spatially")
    assert '"status": "unavailable"' in provenance
    assert '"unavailable_reason"' in provenance
    assert '"value"' not in provenance


@pytest.mark.parametrize(
    ("ratio_available", "reason"),
    [(False, ""), (True, "Ratio values and an unavailable reason were both supplied.")],
)
def test_requires_exactly_one_ratio_values_or_unavailable_reason(
    ratio_available: bool,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        prepare_co2_o2_inputs(**_inputs(ratio_available=ratio_available, unavailable_reason=reason))


def test_rejects_co2_loading_on_o2_ocean_state() -> None:
    with pytest.raises(ValueError, match="CO2 sensitivity.*O2-specific ocean"):
        prepare_co2_o2_inputs(**_inputs(co2_o2_ocean_loading=0.1))


def test_rejects_o2_loading_on_co2_ocean_state() -> None:
    with pytest.raises(ValueError, match="O2 sensitivity.*CO2-specific ocean"):
        prepare_co2_o2_inputs(**_inputs(o2_co2_ocean_loading=0.1))


def test_rejects_ratio_provenance_with_nonshared_state_labels() -> None:
    with pytest.raises(ValueError, match="state labels.*retained shared states"):
        prepare_co2_o2_inputs(**_inputs(ratio_state=["gpp:1", "ter:1", "co2-ocean:1"]))


def test_rejects_ratio_provenance_with_mismatched_sources() -> None:
    with pytest.raises(ValueError, match="sources.*retained shared states"):
        prepare_co2_o2_inputs(**_inputs(ratio_source=["GPP", "TER", "ocean"]))


@pytest.mark.parametrize("ratio_values", [[-1.1, 0.0, -1.4], [-1.1, np.nan, -1.4]])
def test_rejects_unsigned_or_nonfinite_available_ratios(ratio_values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite negative"):
        prepare_co2_o2_inputs(**_inputs(ratio_values=ratio_values))


@pytest.mark.parametrize(
    ("direction", "sign", "message"),
    [
        ("CO2 flux per O2 flux", "signed; positive CO2 flux has negative O2 loading", "direction"),
        ("O2 flux per CO2 flux", "unsigned", "sign_convention"),
    ],
)
def test_rejects_ambiguous_ratio_direction_or_sign(
    direction: str,
    sign: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_co2_o2_inputs(**_inputs(ratio_direction=direction, ratio_sign=sign))


def _durable_inputs(*, ratio_available: bool = True) -> dict[str, object]:
    """Build unequal native MultiIndexes and a correlated, nonunit state prior."""
    inputs = _inputs(
        ratio_available=ratio_available,
        unavailable_reason="Ratios embedded in native paired flux." if not ratio_available else "",
    )
    inputs["o2_units"] = "ppm"
    indexes = {
        "co2": pd.MultiIndex.from_tuples(
            [("TAC", pd.Timestamp("2021-01-03")), ("MHD", pd.Timestamp("2021-01-01"))],
            names=("site", "time"),
        ),
        "o2": pd.MultiIndex.from_tuples(
            [
                ("MHD", pd.Timestamp("2021-01-04")),
                ("TAC", pd.Timestamp("2021-01-02")),
                ("TAC", pd.Timestamp("2021-01-05")),
            ],
            names=("site", "time"),
        ),
    }

    def native_index(array: xr.DataArray, dim: str, index: pd.MultiIndex) -> xr.DataArray:
        result = array.drop_vars([name for name, coord in array.coords.items() if coord.dims == (dim,)])
        return result.assign_coords(xr.Coordinates.from_pandas_multiindex(index, dim))

    for channel, index in indexes.items():
        dim = f"{channel}_measure"
        for suffix in ("observations", "prior_forward_mean", "sensitivity", "aggregation_covariance"):
            name = f"{channel}_{suffix}"
            inputs[name] = native_index(inputs[name], dim, index)
        name = f"{channel}_aggregation_covariance"
        inputs[name] = native_index(inputs[name], f"{dim}_cov", index.set_names(("site_cov", "time_cov")))
    cross = xr.full_like(inputs["co2_o2_aggregation_covariance"], 0.08)
    cross = native_index(cross, "co2_measure", indexes["co2"])
    inputs["co2_o2_aggregation_covariance"] = native_index(
        cross, "o2_measure", indexes["o2"].set_names(("site_cross", "time_cross"))
    )
    state_index = pd.MultiIndex.from_arrays(
        [
            ["GPP", "TER", "FF", "ocean", "ocean"],
            ["shared", "shared", "shared", "co2", "o2"],
            [2, 1, 3, 1, 2],
        ],
        names=("source", "tracer_scope", "region_in_source"),
    )
    mean = _with_state_index(inputs["retained_prior"].mean, state_index).copy(
        data=np.array([0.8, 1.2, 0.9, 1.1, 0.7])
    )
    inputs["retained_prior"] = CorrelatedLognormalPrior(mean, np.eye(5) * 0.01 + 0.002)
    for name in ("co2_sensitivity", "o2_sensitivity"):
        inputs[name] = _with_state_index(inputs[name], state_index)
    if ratio_available:
        inputs["o2_co2_flux_ratio"] = _with_state_index(inputs["o2_co2_flux_ratio"], state_index[:3])
    inputs["boundary_sensitivity"] = {
        channel: xr.DataArray(
            np.arange(1, len(index) + 1, dtype=float)[:, None],
            dims=(f"{channel}_measure", "boundary_state"),
            coords={
                **xr.Coordinates.from_pandas_multiindex(index, f"{channel}_measure"),
                "boundary_state": [f"{channel}:north"],
            },
            attrs={"units": "ppm"},
        )
        for channel, index in indexes.items()
    }
    inputs["provenance"] = {"producer": "public linked fixture", "input_revision": "fixture-v1"}
    return inputs


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("ratio_available", [True, False])
@pytest.mark.parametrize("saved_error", [True, False])
def test_complete_linked_prepared_round_trip(
    tmp_path: Path, suffix: str, ratio_available: bool, saved_error: bool
) -> None:
    """Persist every linked scientific field without changing labels or names."""
    prepared = prepare_co2_o2_inputs(**_durable_inputs(ratio_available=ratio_available))
    prepared = replace(
        prepared,
        aggregation_error=replace(
            prepared.aggregation_error,
            covariance=prepared.aggregation_error.covariance.rename("joint_covariance"),
        ),
    )
    if saved_error:
        error = xr.full_like(prepared.observations, 0.4).rename("reported_sd")
        prepared = replace(prepared, independent_error_sd=error)
    originals = {
        name: getattr(prepared, name).compute().copy(deep=True)
        for name in ("observations", "fixed_prior_contribution", "co2_sensitivity", "o2_sensitivity")
    }
    path = tmp_path / f"linked{suffix}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)

    for name, expected in originals.items():
        xr.testing.assert_identical(getattr(restored, name), expected)
        xr.testing.assert_identical(getattr(prepared, name), expected)
    xr.testing.assert_identical(restored.aggregation_error.covariance, prepared.aggregation_error.covariance)
    xr.testing.assert_identical(restored.retained_prior.mean, prepared.retained_prior.mean)
    xr.testing.assert_identical(
        restored.retained_prior.arithmetic_covariance, prepared.retained_prior.arithmetic_covariance
    )
    assert restored.provenance == prepared.provenance
    assert restored.o2_co2_flux_ratio_unavailable_reason == prepared.o2_co2_flux_ratio_unavailable_reason
    if ratio_available:
        xr.testing.assert_identical(restored.o2_co2_flux_ratio, prepared.o2_co2_flux_ratio)
        assert restored.o2_co2_flux_ratio.sizes["state"] == 3
    else:
        assert restored.o2_co2_flux_ratio is None
    if saved_error:
        xr.testing.assert_identical(restored.independent_error_sd, prepared.independent_error_sd)
    else:
        assert restored.independent_error_sd is None
    for channel, boundary in prepared.boundary_sensitivity.items():
        xr.testing.assert_identical(restored.boundary_sensitivity[channel], boundary)
    np.testing.assert_array_equal(restored.aggregation_error.covariance.values[:2, 2:], np.full((2, 3), 0.08))
    assert isinstance(prepared.co2_sensitivity.data, da.Array)
    assert isinstance(restored.observations.indexes["observation"], pd.MultiIndex)
    assert restored.observations.indexes["observation"].names == ["species", "site", "time"]
    assert (
        restored.retained_prior.mean.indexes["state"].names
        == prepared.retained_prior.mean.indexes["state"].names
    )


def test_linked_save_computes_shared_payloads_together(tmp_path: Path) -> None:
    """Materialize shared observation, sensitivity, ratio, and error graphs once."""
    prepared = prepare_co2_o2_inputs(**_inputs())
    executions = []

    @delayed
    def shared_payload() -> np.ndarray:
        executions.append("payload")
        return np.array([2.0, 3.0, -4.0, -5.0, -6.0])

    data = da.from_delayed(shared_payload(), shape=(5,), dtype=float)
    scale = data[0] / 2.0
    prepared = replace(
        prepared,
        observations=prepared.observations.copy(data=data),
        fixed_prior_contribution=prepared.fixed_prior_contribution.copy(data=data + 1.0),
        co2_sensitivity=prepared.co2_sensitivity.copy(data=prepared.co2_sensitivity.data * scale),
        o2_sensitivity=prepared.o2_sensitivity.copy(data=prepared.o2_sensitivity.data * scale),
        o2_co2_flux_ratio=prepared.o2_co2_flux_ratio.copy(data=prepared.o2_co2_flux_ratio.data * scale),
        independent_error_sd=xr.full_like(prepared.observations, 0.4).copy(data=da.ones(5) * scale),
    )
    tree = prepared.to_datatree()
    assert executions == []
    assert tree.attrs["schema_version"] == 1
    prepared.save(tmp_path / "shared.nc")
    assert executions == ["payload"]
    assert isinstance(prepared.observations.data, da.Array)
    restored = Co2O2PreparedInputs.load(tmp_path / "shared.nc")
    assert restored.observations.observation_units.values.tolist() == [
        "ppm",
        "ppm",
        "per meg",
        "per meg",
        "per meg",
    ]
    np.testing.assert_array_equal(restored.independent_error_sd.values, np.ones(5))


@pytest.mark.parametrize("corruption", ["version", "covariance_units", "error", "ratio", "index_metadata"])
def test_linked_loader_rejects_corrupt_scientific_artifacts(corruption: str) -> None:
    """Reject malformed schema metadata and inconsistent scientific values."""
    prepared = prepare_co2_o2_inputs(**_durable_inputs())
    prepared = replace(prepared, independent_error_sd=xr.full_like(prepared.observations, 0.4))
    tree = prepared.to_datatree().copy(deep=True)
    if corruption == "version":
        tree.attrs["schema_version"] = True
    elif corruption == "covariance_units":
        units = tree["joint"]["observation_units_cov"]
        tree["joint"]["observation_units_cov"] = units.copy(data=np.full(units.size, "per meg"))
    elif corruption == "error":
        error = tree["independent_error"]["independent_error_sd"]
        tree["independent_error"]["independent_error_sd"] = error.copy(data=np.full(error.size, -0.4))
    elif corruption == "ratio":
        ratio = tree["flux_ratio"]["o2_co2_flux_ratio"]
        tree["flux_ratio"]["o2_co2_flux_ratio"] = ratio.copy(data=np.array([-1.2, -1.0, -1.4]))
    else:
        tree["joint"].attrs["multiindex_dims_json"] = "[]"

    with pytest.raises(ValueError):
        Co2O2PreparedInputs.from_datatree(tree)


def test_linked_preparation_accepts_labelled_independent_error() -> None:
    """Retain valid borrowed errors and reject invalid eager values or row units."""
    inputs = _inputs()
    prepared = prepare_co2_o2_inputs(**inputs)
    error = xr.full_like(prepared.observations, 0.4).rename("reported_sd")
    with_error = prepare_co2_o2_inputs(**inputs, independent_error_sd=error)
    assert with_error.independent_error_sd is error
    with pytest.raises(ValueError, match="positive"):
        prepare_co2_o2_inputs(**inputs, independent_error_sd=error.copy(data=np.zeros(error.size)))
    with pytest.raises(ValueError, match="units"):
        prepare_co2_o2_inputs(
            **inputs, independent_error_sd=error.assign_coords(observation_units=("observation", ["ppm"] * 5))
        )


@pytest.mark.parametrize("field", ["observations", "fixed_prior_contribution"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1.0 + 1.0j, "invalid"])
@pytest.mark.parametrize("boundary", ["save", "from_datatree"])
def test_linked_artifact_rejects_invalid_joint_values(
    tmp_path: Path, field: str, invalid: object, boundary: str
) -> None:
    """Reject non-finite and non-real joint payloads at each artifact boundary."""
    prepared = prepare_co2_o2_inputs(**_inputs())
    original = getattr(prepared, field)
    corrupted = original.copy(data=np.full(original.size, invalid))
    if boundary == "save":
        prepared = replace(prepared, **{field: corrupted})
        with pytest.raises(ValueError, match=f"{field} must contain only finite real numeric values"):
            prepared.save(tmp_path / "invalid.nc")
        assert not (tmp_path / "invalid.nc").exists()
    else:
        tree = prepared.to_datatree()
        variable = "observed_concentration" if field == "observations" else field
        stored = tree["joint"][variable]
        tree["joint"][variable] = stored.copy(data=corrupted.data)
        with pytest.raises(ValueError, match=f"{field} must contain only finite real numeric values"):
            Co2O2PreparedInputs.from_datatree(tree)


@pytest.mark.parametrize("field", ["observed_concentration", "fixed_prior_contribution"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_linked_load_rejects_nonfinite_saved_joint_values(tmp_path: Path, field: str, invalid: float) -> None:
    """Reject non-finite concentrations and intercepts read from an external file."""
    tree = prepare_co2_o2_inputs(**_inputs()).to_datatree()
    original = tree["joint"][field]
    tree["joint"][field] = original.copy(data=np.full(original.size, invalid))
    path = tmp_path / "corrupted.nc"
    save_datatree(tree, path)
    with pytest.raises(ValueError, match="finite real numeric"):
        Co2O2PreparedInputs.load(path)


def test_linked_serialization_rejects_mutated_boundary_channels(tmp_path: Path) -> None:
    """Reject unsupported keys added to a borrowed boundary mapping before I/O."""
    prepared = prepare_co2_o2_inputs(**_durable_inputs())
    prepared.boundary_sensitivity["n2o"] = prepared.boundary_sensitivity["co2"]
    with pytest.raises(ValueError, match="keyed only by co2 and o2"):
        prepared.to_datatree()
    with pytest.raises(ValueError, match="keyed only by co2 and o2"):
        prepared.save(tmp_path / "unsupported.nc")
    assert not (tmp_path / "unsupported.nc").exists()


def test_linked_loader_rejects_unexpected_boundary_nodes() -> None:
    """Reject extra serialized boundary channels instead of discarding their data."""
    tree = prepare_co2_o2_inputs(**_durable_inputs()).to_datatree()
    tree["n2o_boundary"] = tree["co2_boundary"].copy()
    with pytest.raises(ValueError, match="Unsupported serialized boundary nodes.*n2o_boundary"):
        Co2O2PreparedInputs.from_datatree(tree)


def test_linked_preparation_preserves_shared_lazy_error_payload_and_units(tmp_path: Path) -> None:
    """Defer lazy error data and auxiliary units, then serialize their shared graph once."""
    inputs = _inputs()
    reference = prepare_co2_o2_inputs(**inputs)
    executions = []

    @delayed
    def shared_payload() -> np.ndarray:
        executions.append("payload")
        return np.array([2.0, 3.0, -4.0, -5.0, -6.0])

    @delayed
    def error_units() -> np.ndarray:
        executions.append("units")
        return np.array(["ppm", "ppm", "per meg", "per meg", "per meg"])

    data = da.from_delayed(shared_payload(), shape=(5,), dtype=float)
    for channel, rows in (("co2", slice(0, 2)), ("o2", slice(2, 5))):
        inputs[f"{channel}_observations"] = inputs[f"{channel}_observations"].copy(data=data[rows])
        inputs[f"{channel}_prior_forward_mean"] = inputs[f"{channel}_prior_forward_mean"].copy(
            data=data[rows] + 0.5
        )
    error = reference.observations.copy(data=abs(data) * 0.1 + 0.4).assign_coords(
        observation_units=("observation", da.from_delayed(error_units(), shape=(5,), dtype="U7"))
    )
    prepared = prepare_co2_o2_inputs(**inputs, independent_error_sd=error)
    assert executions == []
    assert prepared.independent_error_sd is error
    assert isinstance(prepared.independent_error_sd.data, da.Array)
    assert isinstance(prepared.independent_error_sd.observation_units.data, da.Array)
    prepared.to_datatree()
    assert executions == []
    prepared.save(tmp_path / "lazy-error.nc")
    assert sorted(executions) == ["payload", "units"]


@pytest.mark.parametrize("invalid", [0.0, -0.4, np.nan, np.inf, 1.0 + 1.0j])
def test_linked_save_validates_deferred_lazy_error(tmp_path: Path, invalid: float | complex) -> None:
    """Permit lazy errors through preparation and reject invalid values before saving."""
    inputs = _inputs()
    reference = prepare_co2_o2_inputs(**inputs)
    executions = []

    @delayed
    def error_payload() -> np.ndarray:
        executions.append("error")
        return np.full(reference.observations.size, invalid)

    error = reference.observations.copy(
        data=da.from_delayed(error_payload(), shape=(5,), dtype=np.asarray(invalid).dtype)
    )
    prepared = prepare_co2_o2_inputs(**inputs, independent_error_sd=error)
    assert executions == []
    with pytest.raises(ValueError, match="finite positive real"):
        prepared.save(tmp_path / "invalid-error.nc")
    assert executions == ["error"]
    assert not (tmp_path / "invalid-error.nc").exists()
