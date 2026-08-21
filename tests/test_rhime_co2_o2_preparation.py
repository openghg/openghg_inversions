"""Focused preparation contracts for the CO2/O2 recipe."""

from __future__ import annotations

import warnings

from dask import array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions._serialization_codecs import (
    _TAGGED_JSON_VALUE_ENCODING,
    _decode_tagged_json_value,
)
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_preparation import _stack


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
        "co2_operator": xr.DataArray(
            da.from_array(
                [[1, 2, 3, 4, co2_o2_ocean_loading], [0.5, 1, 1.5, 2, 0]],
                chunks=(1, 5),
            ),
            dims=("co2_measure", "state"),
            coords={"co2_measure": co2_labels, "state": state},
        ),
        "o2_operator": xr.DataArray(
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


def _decoded_observation_labels(array: xr.DataArray) -> list[object]:
    encoded = array["within_tracer_observation"]
    assert array["within_tracer_observation_encoding"].item() == _TAGGED_JSON_VALUE_ENCODING
    return [_decode_tagged_json_value(value) for value in encoded.values]


def test_preparation_preserves_lazy_channels_with_staggered_unequal_times() -> None:
    inputs = _inputs()
    ratio = inputs["o2_co2_flux_ratio"]
    assert isinstance(ratio, xr.DataArray)
    prepared = prepare_co2_o2_inputs(**inputs)

    assert isinstance(prepared.observations.data, da.Array)
    assert isinstance(prepared.fixed_prior_contribution.data, da.Array)
    assert isinstance(prepared.co2_operator.data, da.Array)
    assert isinstance(prepared.o2_operator.data, da.Array)
    assert isinstance(prepared.o2_co2_flux_ratio.data, da.Array)
    assert prepared.o2_co2_flux_ratio.data is ratio.data
    assert not isinstance(prepared.aggregation_error.covariance.data, da.Array)
    assert prepared.observations["tracer"].values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert _decoded_observation_labels(prepared.observations) == [
        "c1",
        "c2",
        "o1",
        "o2",
        "o3",
    ]
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
    assert prepared.fixed_prior_contribution.attrs["mathematical_name"] == (
        "H m - H_alpha Pi m"
    )
    ratio_provenance = prepared.o2_operator.attrs["oxidation_ratio_provenance"]
    assert '"state": ["gpp:1", "ter:1", "ff:1"]' in ratio_provenance
    assert '"value": [-1.1, -1.0, -1.4]' in ratio_provenance


def test_canonicalizes_operator_state_metadata_without_mutating_inputs() -> None:
    inputs = _inputs()
    originals: dict[str, xr.DataArray] = {}
    for name in ("co2_operator", "o2_operator"):
        operator = inputs[name]
        assert isinstance(operator, xr.DataArray)
        operator = operator.assign_coords(
            source=("state", ["stale"] * 5),
            tracer_scope=("state", ["wrong"] * 5),
        )
        inputs[name] = operator
        originals[name] = operator.copy(deep=True)

    prepared = prepare_co2_o2_inputs(**inputs)
    prior = inputs["retained_prior"]
    assert isinstance(prior, CorrelatedLognormalPrior)
    for name, prepared_operator in (
        ("co2_operator", prepared.co2_operator),
        ("o2_operator", prepared.o2_operator),
    ):
        np.testing.assert_array_equal(prepared_operator["source"], prior.mean["source"])
        np.testing.assert_array_equal(
            prepared_operator["tracer_scope"], prior.mean["tracer_scope"]
        )
        xr.testing.assert_identical(inputs[name], originals[name])


def test_native_label_encoding_roundtrips_mixed_types_and_datetime_auxiliary(tmp_path) -> None:
    co2_index = pd.Index([1, "1"], dtype=object, name="co2_measure")
    o2_index = pd.DatetimeIndex(
        ["2021-01-02", "2021-01-04", "2021-01-05"],
        name="o2_measure",
    )
    co2 = xr.DataArray(
        [2.0, 3.0],
        dims="co2_measure",
        coords={
            "co2_measure": co2_index,
            "time": ("co2_measure", np.array(["2021-01-01", "2021-01-03"], dtype="datetime64[D]")),
        },
    )
    o2 = xr.DataArray(
        [-4.0, -5.0, -6.0],
        dims="o2_measure",
        coords={"o2_measure": o2_index, "time": ("o2_measure", o2_index.values)},
    )
    originals = (co2.copy(deep=True), o2.copy(deep=True))

    stacked = _stack(co2, o2, co2_units="ppm", o2_units="per meg", name="observations")
    expected = [*co2_index, *o2_index]
    decoded = _decoded_observation_labels(stacked)
    assert decoded == expected
    assert [type(value) for value in decoded] == [type(value) for value in expected]
    assert np.issubdtype(stacked["time"].dtype, np.datetime64)
    xr.testing.assert_identical(co2, originals[0])
    xr.testing.assert_identical(o2, originals[1])

    path = tmp_path / "mixed_observation_labels.nc"
    stacked.to_dataset(name="observations").to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as restored:
        assert _decoded_observation_labels(restored["observations"]) == expected


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

    expected = [*co2_index, *o2_index]
    assert _decoded_observation_labels(stacked) == expected
    assert stacked["site"].values.tolist() == ["TAC", "MHD", "TAC", "MHD", "TAC"]
    assert np.issubdtype(stacked["time"].dtype, np.datetime64)
    xr.testing.assert_identical(co2, originals[0])
    xr.testing.assert_identical(o2, originals[1])

    path = tmp_path / "multiindex_observation_labels.nc"
    stacked.to_dataset(name="observations").to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as restored:
        decoded = _decoded_observation_labels(restored["observations"])
        assert decoded == expected
        assert all(isinstance(label, tuple) for label in decoded)


def test_tracks_unavailable_ratio_values_without_claiming_source_resolved_provenance() -> None:
    prepared = prepare_co2_o2_inputs(
        **_inputs(
            ratio_available=False,
            unavailable_reason="Only spatially resolved native O2 flux treatment is documented.",
        )
    )

    provenance = prepared.o2_operator.attrs["oxidation_ratio_provenance"]
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
        prepare_co2_o2_inputs(
            **_inputs(ratio_available=ratio_available, unavailable_reason=reason)
        )


def test_rejects_co2_loading_on_o2_ocean_state() -> None:
    with pytest.raises(ValueError, match="CO2 operator.*O2-specific ocean"):
        prepare_co2_o2_inputs(**_inputs(co2_o2_ocean_loading=0.1))


def test_rejects_o2_loading_on_co2_ocean_state() -> None:
    with pytest.raises(ValueError, match="O2 operator.*CO2-specific ocean"):
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
