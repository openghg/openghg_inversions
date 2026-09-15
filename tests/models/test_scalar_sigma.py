"""Tests for the labelled global scalar-sigma eigen likelihood."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.scalar_sigma import (
    SCALAR_SIGMA_CACHE_SCHEMA,
    SCALAR_SIGMA_MODE_DIM,
    ScalarSigmaEigenbasis,
    add_scalar_sigma_eigen_likelihood,
    load_scalar_sigma_eigenbasis,
    prepare_scalar_sigma_eigenbasis,
    save_scalar_sigma_eigenbasis,
    scalar_sigma_base_covariance,
)
from openghg_inversions.observation_error import AggregationError


LABELS = np.array(["MHD-0", "MHD-1", "TAC-0"])


def _observations() -> tuple[xr.DataArray, xr.DataArray]:
    observations = xr.DataArray(
        [1.1, 0.8, 1.4],
        dims="nmeasure",
        coords={"nmeasure": LABELS},
        name="mf",
        attrs={"units": "ppm"},
    )
    error = xr.DataArray(
        [0.1, 0.2, 0.15],
        dims="nmeasure",
        coords={"nmeasure": LABELS},
        name="mf_error",
        attrs={"units": "ppm"},
    )
    return observations, error


def _dense_aggregation() -> AggregationError:
    covariance = np.array([[0.4, 0.08, 0.02], [0.08, 0.3, 0.04], [0.02, 0.04, 0.5]])
    return AggregationError(
        mode="dense",
        marginal_variance=np.diag(covariance),
        covariance=xr.DataArray(
            covariance,
            dims=("nmeasure", "nmeasure_cov"),
            coords={"nmeasure": LABELS, "nmeasure_cov": LABELS},
        ),
    )


def _eigenbasis(covariance: np.ndarray) -> ScalarSigmaEigenbasis:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    mode = np.arange(covariance.shape[0])
    return ScalarSigmaEigenbasis(
        eigenvectors=xr.DataArray(
            eigenvectors,
            dims=("nmeasure", SCALAR_SIGMA_MODE_DIM),
            coords={"nmeasure": LABELS, SCALAR_SIGMA_MODE_DIM: mode},
        ),
        eigenvalues=xr.DataArray(
            eigenvalues,
            dims=SCALAR_SIGMA_MODE_DIM,
            coords={SCALAR_SIGMA_MODE_DIM: mode},
        ),
        concentration_units="ppm",
    )


def test_eigen_logp_and_gradient_match_dense_covariance() -> None:
    """The eigen target and gradient equal the materialized Gaussian."""
    covariance = np.array([[1.8, 0.4, 0.1], [0.4, 1.2, 0.2], [0.1, 0.2, 0.7]])
    basis = _eigenbasis(covariance)
    residual_value = np.array([0.3, -0.8, 0.5])
    sigma_value = 0.25
    residual = pt.vector("residual", dtype="float64")
    sigma = pt.scalar("sigma", dtype="float64")
    logp = basis.logp(residual, pt.zeros_like(residual), sigma)
    evaluate = pytensor.function([residual, sigma], [logp, pt.grad(logp, sigma)])

    actual, gradient = evaluate(residual_value, sigma_value)
    complete = covariance + np.eye(3) * sigma_value**2
    expected = multivariate_normal.logpdf(residual_value, mean=np.zeros(3), cov=complete)
    inverse = np.linalg.inv(complete)
    expected_gradient = sigma_value * (
        residual_value @ inverse @ inverse @ residual_value - np.trace(inverse)
    )

    assert float(actual) == pytest.approx(expected, abs=2.0e-12)
    assert float(gradient) == pytest.approx(expected_gradient, abs=2.0e-11)


def test_positive_sigma_rescues_a_zero_base_mode() -> None:
    """A positive inferred sigma makes a semidefinite base covariance valid."""
    basis = _eigenbasis(np.diag([0.0, 2.0, 3.0]))
    residual = pt.vector("residual", dtype="float64")
    sigma = pt.scalar("sigma", dtype="float64")
    evaluate = pytensor.function(
        [residual, sigma],
        basis.logp(residual, pt.zeros_like(residual), sigma),
    )

    assert np.isfinite(evaluate(np.array([0.2, -0.3, 0.1]), 0.4))
    assert float(evaluate(np.array([0.2, -0.3, 0.1]), 0.0)) == -np.inf


@pytest.mark.parametrize("mode", ["none", "diagonal", "low_rank", "dense"])
def test_base_covariance_materializes_aligned_payloads_once(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """All selected lazy covariance payloads share one eager boundary."""
    observations, error = _observations()
    error = error.copy(data=da.from_array(error.values, chunks=2))
    diagonal = np.array([0.3, 0.4, 0.5])
    factor = np.array([[0.4], [0.2], [0.1]])
    dense = factor @ factor.T + np.diag(diagonal)
    if mode == "dense":
        aggregation = AggregationError(
            mode="dense",
            marginal_variance=np.diag(dense),
            covariance=xr.DataArray(
                da.from_array(dense, chunks=(2, 2)),
                dims=("nmeasure", "nmeasure_cov"),
                coords={"nmeasure": LABELS, "nmeasure_cov": LABELS},
            ),
        )
        expected_aggregation = dense
    elif mode == "low_rank":
        aggregation = AggregationError(
            mode="low_rank",
            marginal_variance=np.sum(factor**2, axis=1) + diagonal,
            factor=xr.DataArray(
                da.from_array(factor, chunks=(2, 1)),
                dims=("nmeasure", "rank"),
                coords={"nmeasure": LABELS},
            ),
            diagonal_variance=xr.DataArray(
                da.from_array(diagonal, chunks=2),
                dims="nmeasure",
                coords={"nmeasure": LABELS},
            ),
        )
        expected_aggregation = dense
    elif mode == "diagonal":
        aggregation = AggregationError(
            mode="diagonal",
            marginal_variance=diagonal,
            diagonal_variance=xr.DataArray(
                da.from_array(diagonal, chunks=2),
                dims="nmeasure",
                coords={"nmeasure": LABELS},
            ),
        )
        expected_aggregation = np.diag(diagonal)
    else:
        aggregation = AggregationError(mode="none", marginal_variance=np.zeros(3))
        expected_aggregation = np.zeros((3, 3))

    import openghg_inversions.models.scalar_sigma as scalar_sigma

    real_compute = scalar_sigma.dask_compute
    calls = 0

    def recording_compute(*arrays: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_compute(*arrays)

    monkeypatch.setattr(scalar_sigma, "dask_compute", recording_compute)
    actual = scalar_sigma_base_covariance(observations, error, aggregation)

    assert calls == 1
    np.testing.assert_allclose(
        actual,
        expected_aggregation + np.diag(np.square(error.values)),
    )


def test_preparation_rejects_misaligned_error_and_covariance() -> None:
    """Positional equality cannot substitute for exact observation labels."""
    observations, error = _observations()
    with pytest.raises(ValueError, match="exact ordered observation coordinate"):
        scalar_sigma_base_covariance(
            observations,
            error.sel(nmeasure=LABELS[::-1]),
            _dense_aggregation(),
        )

    aggregation = _dense_aggregation()
    assert aggregation.covariance is not None
    aggregation = AggregationError(
        mode="dense",
        marginal_variance=aggregation.marginal_variance,
        covariance=aggregation.covariance.assign_coords(nmeasure_cov=LABELS[::-1]),
    )
    with pytest.raises(ValueError, match="same values in the same order"):
        scalar_sigma_base_covariance(observations, error, aggregation)


def test_preparation_uses_one_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation reuses one decomposition for PSD validation and caching."""
    observations, error = _observations()
    real_eigh = np.linalg.eigh
    calls = 0

    def recording_eigh(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return real_eigh(array)

    monkeypatch.setattr(np.linalg, "eigh", recording_eigh)
    prepare_scalar_sigma_eigenbasis(
        observations=observations,
        observation_error=error,
        aggregation_error=_dense_aggregation(),
    )

    assert calls == 1


@pytest.mark.parametrize(
    ("units", "concentration_scale"),
    [("ppm", 1.0), ("mol/mol", 1.0e-6)],
)
def test_covariance_validation_is_invariant_to_concentration_units(
    units: str,
    concentration_scale: float,
) -> None:
    """Indefinite and asymmetric covariance fail at ppm and mol/mol scales."""
    observations, error = _observations()
    observations = observations.copy(data=observations.values * concentration_scale)
    observations.attrs["units"] = units
    error = error.copy(data=np.zeros(error.shape))
    error.attrs["units"] = units
    covariance_scale = concentration_scale**2

    indefinite = np.diag([1.0, 0.5, -1.0e-6]) * covariance_scale
    with pytest.raises(ValueError, match="positive semidefinite"):
        prepare_scalar_sigma_eigenbasis(
            observations=observations,
            observation_error=error,
            aggregation_error=AggregationError(
                mode="dense",
                marginal_variance=np.diag(indefinite),
                covariance=xr.DataArray(
                    indefinite,
                    dims=("nmeasure", "nmeasure_cov"),
                    coords={"nmeasure": LABELS, "nmeasure_cov": LABELS},
                ),
            ),
        )

    asymmetric = np.eye(3) * covariance_scale
    asymmetric[0, 1] = 1.0e-6 * covariance_scale
    with pytest.raises(ValueError, match="symmetric"):
        scalar_sigma_base_covariance(
            observations,
            error,
            AggregationError(
                mode="dense",
                marginal_variance=np.diag(asymmetric),
                covariance=xr.DataArray(
                    asymmetric,
                    dims=("nmeasure", "nmeasure_cov"),
                    coords={"nmeasure": LABELS, "nmeasure_cov": LABELS},
                ),
            ),
        )


def test_cache_round_trip_checks_schema_labels_and_units(tmp_path: Path) -> None:
    """The small xarray loader is the sole cache trust boundary."""
    observations, error = _observations()
    basis = prepare_scalar_sigma_eigenbasis(
        observations=observations,
        observation_error=error,
        aggregation_error=_dense_aggregation(),
    )
    path = save_scalar_sigma_eigenbasis(tmp_path / "scalar-sigma.nc", basis)

    loaded = load_scalar_sigma_eigenbasis(
        path,
        observations=observations,
        observation_error=error,
    )
    np.testing.assert_allclose(loaded.eigenvectors, basis.eigenvectors)
    np.testing.assert_allclose(loaded.eigenvalues, basis.eigenvalues)

    with pytest.raises(ValueError, match="exact ordered observation coordinate"):
        load_scalar_sigma_eigenbasis(
            path,
            observations=observations.sel(nmeasure=LABELS[::-1]),
            observation_error=error.sel(nmeasure=LABELS[::-1]),
        )
    with pytest.raises(ValueError, match="exact ordered observation coordinate"):
        load_scalar_sigma_eigenbasis(
            path,
            observations=observations,
            observation_error=error.sel(nmeasure=LABELS[::-1]),
        )
    with pytest.raises(ValueError, match="matching units"):
        load_scalar_sigma_eigenbasis(
            path,
            observations=observations.assign_attrs(units="ppb"),
            observation_error=error.assign_attrs(units="ppb"),
        )
    with pytest.raises(ValueError, match="matching units"):
        load_scalar_sigma_eigenbasis(
            path,
            observations=observations,
            observation_error=error.assign_attrs(units="ppb"),
        )

    malformed = xr.load_dataset(path)
    malformed.attrs["schema"] = "unknown"
    malformed.to_netcdf(tmp_path / "bad-schema.nc")
    with pytest.raises(ValueError, match="unsupported schema"):
        load_scalar_sigma_eigenbasis(
            tmp_path / "bad-schema.nc",
            observations=observations,
            observation_error=error,
        )

    malformed = xr.load_dataset(path)
    malformed["eigenvectors"] = malformed["eigenvectors"] * 2.0
    malformed.to_netcdf(tmp_path / "non-orthogonal.nc")
    with pytest.raises(ValueError, match="orthonormal"):
        load_scalar_sigma_eigenbasis(
            tmp_path / "non-orthogonal.nc",
            observations=observations,
            observation_error=error,
        )


def test_cache_round_trip_preserves_multiindex_labels(tmp_path: Path) -> None:
    """CF encoding preserves site/time observation identity in NetCDF."""
    index = pd.MultiIndex.from_arrays(
        [
            ["MHD", "MHD", "TAC"],
            pd.date_range("2021-01-01", periods=3, freq="h"),
        ],
        names=("site", "time"),
    )
    observations = xr.DataArray(
        [1.1, 0.8, 1.4],
        dims="nmeasure",
        coords={"nmeasure": index},
        attrs={"units": "ppm"},
    )
    error = xr.DataArray(
        [0.1, 0.2, 0.15],
        dims="nmeasure",
        coords={"nmeasure": index},
        attrs={"units": "ppm"},
    )
    aggregation = AggregationError(
        mode="none",
        marginal_variance=np.zeros(3),
    )
    basis = prepare_scalar_sigma_eigenbasis(
        observations=observations,
        observation_error=error,
        aggregation_error=aggregation,
    )
    path = save_scalar_sigma_eigenbasis(tmp_path / "multiindex.nc", basis)

    loaded = load_scalar_sigma_eigenbasis(
        path,
        observations=observations,
        observation_error=error,
    )

    assert loaded.eigenvectors.indexes["nmeasure"].equals(observations.indexes["nmeasure"])


def test_thin_likelihood_matches_dense_covariance_and_uses_shared_prior_policy() -> None:
    """Graph construction consumes only a trusted basis and resolved inputs."""
    observations, error = _observations()
    aggregation = _dense_aggregation()
    basis = prepare_scalar_sigma_eigenbasis(
        observations=observations,
        observation_error=error,
        aggregation_error=aggregation,
    )
    mean_value = np.array([1.0, 1.0, 1.0])
    with registered_model(coords={"nmeasure": LABELS}) as model:
        mean = pm.Data("mean", mean_value, dims="nmeasure")
        add_scalar_sigma_eigen_likelihood(
            observations=observations,
            observation_error=error,
            aggregation_error=aggregation,
            mean=mean,
            eigenbasis=basis,
            sigma_prior={"pdf": "halfnormal", "sigma": 0.75},
        )

    point = model.initial_point()
    point["sigma_global_log__"] = np.asarray(np.log(0.5))
    actual = float(model.compile_logp(vars=[model["y"]])(point))
    expected = multivariate_normal.logpdf(
        observations.values,
        mean=mean_value,
        cov=scalar_sigma_base_covariance(observations, error, aggregation) + np.eye(3) * 0.25,
    )
    assert actual == pytest.approx(expected, abs=2.0e-6)
    assert {"sigma_global", "epsilon", "y"}.issubset(model.named_vars)

    with registered_model(coords={"nmeasure": LABELS}):
        mean = pm.Data("mean", mean_value, dims="nmeasure")
        with pytest.raises(ValueError, match="positive prior"):
            add_scalar_sigma_eigen_likelihood(
                observations=observations,
                observation_error=error,
                aggregation_error=aggregation,
                mean=mean,
                eigenbasis=basis,
                sigma_prior={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            )


def test_save_requires_netcdf_suffix(tmp_path: Path) -> None:
    """The cache writer does not guess a storage format from arbitrary paths."""
    basis = _eigenbasis(np.eye(3))
    with pytest.raises(ValueError, match="'.nc' suffix"):
        save_scalar_sigma_eigenbasis(tmp_path / "scalar-sigma.npz", basis)


def test_cache_dataset_uses_the_published_schema() -> None:
    """The serializable cache exposes its normal xarray schema directly."""
    dataset = _eigenbasis(np.eye(3)).to_dataset()
    assert dataset.attrs["schema"] == SCALAR_SIGMA_CACHE_SCHEMA
    assert set(dataset.data_vars) == {"eigenvectors", "eigenvalues"}
