"""Focused tests for the recipe-specific CO2 prepared-input boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions._labelled_matrices import renamed_column_coordinates
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.rhime.co2 import (
    co2_cached_sigma_runner,
    co2_runner,
    prepare_co2_scalar_sigma_eigenbasis,
    run_rhime_co2,
    run_rhime_co2_cached_sigma,
)


def _canonical_inputs() -> RhimePreparedInputs:
    observations = pd.MultiIndex.from_arrays(
        [
            ["MHD", "TAC", "MHD"],
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        ],
        names=("site", "time"),
    )
    observation_coords = xr.Coordinates.from_pandas_multiindex(observations, "nmeasure")
    basis = xr.DataArray(
        [[1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [51.0], "lon": [-2.0, -1.0]},
        name="basis",
    )
    flux = xr.DataArray(
        [[1.0, 1.0]],
        dims=basis.dims,
        coords=basis.coords,
        attrs={"units": "mol m-2 s-1"},
        name="flux",
    )
    inv_inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), np.zeros((2, 3))),
            "mf": ("nmeasure", [401.0, 402.0, 403.0]),
            "mf_error": ("nmeasure", [0.1, 0.2, 0.3]),
            "state_is_active": ("region", [True, False]),
            "state_fixed_value": ("region", [1.0, 0.9]),
        },
        coords={"region": [1, 2], **observation_coords},
    )
    inv_inputs["mf"].attrs["units"] = "ppm"
    inv_inputs["mf_error"].attrs["units"] = "ppm"
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=BasisFunctions.from_flat_basis(
            basis_flat=basis,
            flux=flux,
            operator_kwargs={"state_dim": "region"},
        ),
        site_metadata=xr.Dataset(
            {"averaging_period": ("site", np.asarray(["1h", "2h"], dtype=object))},
            coords={"site": ["MHD", "TAC"]},
        ),
    )


def _reduction(canonical: RhimePreparedInputs) -> CoherentGaussianReduction:
    observation_index = canonical.inv_inputs.indexes["nmeasure"]
    observation_coords = xr.Coordinates.from_pandas_multiindex(
        observation_index,
        "observation",
    )
    state = xr.DataArray([1, 2], dims="state", name="state")
    observations = xr.DataArray(
        [401.0, 402.0, 403.0],
        dims="observation",
        coords=observation_coords,
        attrs={"units": "ppm"},
    )
    covariance_values = np.asarray([[0.5, 0.12, 0.04], [0.12, 0.4, 0.08], [0.04, 0.08, 0.3]])
    unresolved = xr.DataArray(
        covariance_values,
        dims=("observation", "observation_cov"),
        coords={
            **{str(name): coord for name, coord in observation_coords.items()},
            **renamed_column_coordinates(
                observations,
                row_dim="observation",
                column_dim="observation_cov",
            ),
        },
        attrs={"units": "(ppm)^2"},
    )
    return CoherentGaussianReduction(
        retained_mean=xr.DataArray(
            [1.1, 0.9],
            dims="state",
            coords={"state": state},
            attrs={"units": "1"},
        ),
        retained_covariance=xr.DataArray(
            [[0.2, 0.03], [0.03, 0.15]],
            dims=("state", "state_cov"),
            coords={"state": state, "state_cov": state.values},
            attrs={"units": "1"},
        ),
        effective_observation_operator=xr.DataArray(
            [[0.7, 0.1], [0.3, 0.4], [0.2, 0.6]],
            dims=("observation", "state"),
            coords={**observation_coords, "state": state},
            attrs={"units": "ppm"},
        ),
        native_observation_mean=observations.rename("native_observation_mean"),
        observation_intercept=(observations - xr.DataArray([0.8, 0.7, 0.9], dims="observation")).rename(
            "observation_intercept"
        ),
        unresolved_observation_covariance=unresolved,
        projection_strategy="unit-test-projection",
    )


def test_prepare_co2_inputs_maps_reduction_and_preserves_canonical_metadata() -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)

    prepared = prepare_co2_inputs(
        canonical,
        reduction,
        aggregation_error_rank=None,
        provenance={"source": "test"},
    )

    assert prepared.aggregation_error_mode == "dense"
    assert prepared.provenance == {
        "projection_strategy": "unit-test-projection",
        "source": "test",
    }
    np.testing.assert_allclose(prepared.inv_inputs["H"], reduction.effective_observation_operator)
    np.testing.assert_allclose(prepared.inv_inputs["alpha_prior_mean"], reduction.retained_mean)
    np.testing.assert_allclose(
        prepared.inv_inputs["alpha_prior_covariance"],
        reduction.retained_covariance,
    )
    np.testing.assert_allclose(
        prepared.inv_inputs["fixed_prior_contribution"],
        reduction.observation_intercept,
    )
    np.testing.assert_allclose(
        prepared.inv_inputs["aggregation_error_covariance"],
        reduction.unresolved_observation_covariance,
    )
    xr.testing.assert_identical(
        prepared.inv_inputs["state_is_active"],
        canonical.inv_inputs["state_is_active"],
    )
    assert prepared.inv_inputs.indexes["nmeasure"].equals(canonical.inv_inputs.indexes["nmeasure"])
    assert "aggregation_error_covariance" not in canonical.inv_inputs


def test_prepare_co2_inputs_constructs_truncated_low_rank_representation() -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)

    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=1)

    assert prepared.aggregation_error_mode == "low_rank"
    assert prepared.inv_inputs["low_rank_factor"].shape == (3, 1)
    assert "aggregation_error_covariance" not in prepared.inv_inputs
    reconstructed_diagonal = (prepared.inv_inputs["low_rank_factor"] ** 2).sum(
        "agg_rank"
    ) + prepared.inv_inputs["diagonal_residual_variance"]
    np.testing.assert_allclose(
        reconstructed_diagonal,
        np.diag(reduction.unresolved_observation_covariance),
    )
    assert prepared.provenance["aggregation_error"]["requested_rank"] == 1


def test_prepare_co2_inputs_caps_default_low_rank_at_observation_count() -> None:
    canonical = _canonical_inputs()

    prepared = prepare_co2_inputs(canonical, _reduction(canonical))

    assert prepared.inv_inputs["low_rank_factor"].shape == (3, 3)
    assert prepared.provenance["aggregation_error"]["requested_rank"] == 3


@pytest.mark.parametrize(("existing_rank", "requested_rank"), [(1, 2), (2, 1)])
def test_prepare_co2_inputs_replaces_stale_aggregation_rank_coordinate(
    existing_rank: int,
    requested_rank: int,
) -> None:
    canonical = _canonical_inputs()
    canonical.inv_inputs["low_rank_factor"] = xr.DataArray(
        np.ones((3, existing_rank)),
        dims=("nmeasure", "agg_rank"),
        coords={"agg_rank": np.arange(10, 10 + existing_rank)},
        attrs={"units": "ppm"},
    )
    canonical.inv_inputs["diagonal_residual_variance"] = xr.DataArray(
        np.ones(3),
        dims="nmeasure",
        attrs={"units": "(ppm)^2"},
    )

    prepared = prepare_co2_inputs(
        canonical,
        _reduction(canonical),
        aggregation_error_rank=requested_rank,
    )

    factor = prepared.inv_inputs["low_rank_factor"]
    assert factor.shape == (3, requested_rank)
    np.testing.assert_array_equal(factor["agg_rank"], np.arange(requested_rank))
    assert np.isfinite(factor).all()


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("representation", ["dense", "low_rank"])
def test_co2_prepared_inputs_round_trip(
    tmp_path: Path,
    suffix: str,
    representation: str,
) -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    prepared = prepare_co2_inputs(
        canonical,
        reduction,
        aggregation_error_rank=1 if representation == "low_rank" else None,
        provenance={"source": "round-trip"},
    )
    path = tmp_path / f"co2-prepared{suffix}"

    prepared.save(path)
    restored = Co2PreparedInputs.load(path)

    assert restored.aggregation_error_mode == representation
    assert restored.provenance == prepared.provenance
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)


def test_prepare_co2_inputs_rejects_conflicting_projection_strategy() -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)

    with pytest.raises(ValueError, match="projection_strategy conflicts"):
        prepare_co2_inputs(
            canonical,
            reduction,
            provenance={"projection_strategy": "different-strategy"},
        )


def test_prepare_co2_inputs_rejects_wrong_covariance_units() -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    reduction.unresolved_observation_covariance.attrs["units"] = "(ppb)^2"

    with pytest.raises(ValueError, match="same numeric scale"):
        prepare_co2_inputs(canonical, reduction)


def test_co2_prepared_inputs_load_rejects_wrong_low_rank_units() -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=1)
    tree = prepared.to_datatree()
    node = cast(xr.DataTree, tree["rhime_inputs/inv_inputs"])
    dataset = node.to_dataset()
    dataset["low_rank_factor"].attrs["units"] = "ppb"
    node.ds = dataset

    with pytest.raises(ValueError, match="same numeric scale"):
        Co2PreparedInputs.from_datatree(tree)


def test_reloaded_truncated_low_rank_artifact_drives_cached_runner_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=1)
    path = tmp_path / "co2-low-rank.nc"
    prepared.save(path)
    restored = Co2PreparedInputs.load(path)
    received: dict[str, Any] = {}

    class ModelBoundaryReached(Exception):
        pass

    def build_model(_sensitivity: xr.DataArray, **kwargs: Any) -> None:
        received.update(kwargs)
        raise ModelBoundaryReached

    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "build_co2_cached_sigma_model",
        build_model,
    )

    with pytest.raises(ModelBoundaryReached):
        run_rhime_co2_cached_sigma(
            prepared_inputs=restored,
            tau_hours=24.0,
            site_amplitude_prior_scale=0.75,
        )

    selected = received["aggregation_error"]
    assert selected.mode == "low_rank"
    assert selected.factor is not None
    assert selected.factor.shape == (3, 1)


def test_standard_runner_uses_real_co2_prepared_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = _canonical_inputs()
    prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=1)
    received: dict[str, Any] = {}

    class ModelBoundaryReached(Exception):
        pass

    def build_model(_sensitivity: xr.DataArray, **kwargs: Any) -> None:
        received.update(kwargs)
        raise ModelBoundaryReached

    monkeypatch.setattr(co2_runner, "build_co2_model", build_model)

    with pytest.raises(ModelBoundaryReached):
        run_rhime_co2(
            prepared_inputs=prepared,
            fixed_model_mismatch=0.0,
            no_model_error=True,
        )

    assert received["aggregation_error"].mode == "low_rank"


def test_scalar_sigma_preparation_uses_artifact_aggregation_mode() -> None:
    canonical = _canonical_inputs()
    prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=1)

    eigenbasis = prepare_co2_scalar_sigma_eigenbasis(prepared)

    assert eigenbasis.aggregation_error_mode == "low_rank"


def test_standard_runner_materializes_aggregation_payload_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_inputs()
    prepared = prepare_co2_inputs(
        canonical,
        _reduction(canonical),
        aggregation_error_rank=1,
    )
    factor = prepared.inv_inputs["low_rank_factor"]
    diagonal = prepared.inv_inputs["diagonal_residual_variance"]
    factor_values = np.asarray(factor.values)
    diagonal_values = np.asarray(diagonal.values)
    executions = 0

    @delayed
    def aggregation_payload() -> tuple[np.ndarray, np.ndarray]:
        nonlocal executions
        executions += 1
        return factor_values, diagonal_values

    payload = aggregation_payload()
    factor.data = da.from_delayed(payload[0], shape=factor.shape, dtype=float)
    diagonal.data = da.from_delayed(payload[1], shape=diagonal.shape, dtype=float)

    class ModelBoundaryReached(Exception):
        pass

    def build_model(_sensitivity: xr.DataArray, **_kwargs: Any) -> None:
        raise ModelBoundaryReached

    monkeypatch.setattr(co2_runner, "build_co2_model", build_model)

    with pytest.raises(ModelBoundaryReached):
        run_rhime_co2(
            prepared_inputs=prepared,
            fixed_model_mismatch=0.0,
            no_model_error=True,
        )

    assert executions == 1
