"""Tests for the filtered RHIME-to-TD-MCMC input adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.tdmcmc.rhime_adapter import problem_from_rhime_inputs


def _shuffled_input_dataset() -> xr.Dataset:
    """Build a tiny dataset whose sensitivity dimensions require transposition."""
    canonical_sensitivity = np.arange(1.0, 13.0).reshape(2, 2, 3)
    sensitivity = xr.DataArray(
        canonical_sensitivity.transpose(2, 0, 1),
        dims=("lon", "nmeasure", "lat"),
        coords={
            "lon": [10.0, 20.0, 30.0],
            "nmeasure": ["second", "first"],
            "lat": [50.0, 51.0],
        },
    )
    observations = xr.DataArray(
        [100.0, 200.0],
        dims="nmeasure",
        coords={"nmeasure": ["first", "second"]},
    )
    observation_sd = xr.DataArray(
        [1.0, 2.0],
        dims="nmeasure",
        coords={"nmeasure": ["first", "second"]},
    )
    return xr.Dataset(
        {
            "fp_x_flux": sensitivity,
            "mf": observations,
            "mf_error": observation_sd,
            "ignored_bc": xr.DataArray([4.0, 5.0], dims="bc_region"),
        }
    )


def _adapter_kwargs() -> dict[str, object]:
    """Return the common prior and active-count arguments for adapter tests."""
    return {
        "k_min": 1,
        "k_max": 3,
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.5,
    }


def test_adapter_transposes_flattens_and_aligns_exactly() -> None:
    """Shuffled inputs should become longitude-fast numerical core arrays."""
    dataset = _shuffled_input_dataset()

    problem = problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]

    np.testing.assert_array_equal(
        problem.sensitivities,
        np.array(
            [
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        problem.grid_coordinates,
        np.array(
            [
                [50.0, 10.0],
                [50.0, 20.0],
                [50.0, 30.0],
                [51.0, 10.0],
                [51.0, 20.0],
                [51.0, 30.0],
            ]
        ),
    )
    np.testing.assert_array_equal(problem.observations, [100.0, 200.0])
    np.testing.assert_array_equal(problem.observation_sd, [1.0, 2.0])
    np.testing.assert_allclose(np.exp(problem.log_k_prior), np.full(3, 1.0 / 3.0))
    assert problem.k_min == 1
    assert problem.k_max == 3
    assert problem.coefficient_prior_mean == 1.0
    assert problem.coefficient_prior_sd == 0.5


def test_adapter_preserves_custom_names_and_k_prior() -> None:
    """Custom variable names and a declared normalized K prior should pass through."""
    dataset = _shuffled_input_dataset().rename(
        {"fp_x_flux": "fine_G", "mf": "observed", "mf_error": "fixed_error"}
    )
    log_k_prior = np.log(np.array([0.25, 0.75]))

    problem = problem_from_rhime_inputs(
        dataset,
        k_min=1,
        k_max=2,
        coefficient_prior_mean=1.2,
        coefficient_prior_sd=0.3,
        log_k_prior=log_k_prior,
        sensitivity_name="fine_G",
        observation_name="observed",
        observation_sd_name="fixed_error",
    )

    np.testing.assert_array_equal(problem.log_k_prior, log_k_prior)
    assert problem.coefficient_prior_mean == 1.2
    assert problem.coefficient_prior_sd == 0.3


@pytest.mark.parametrize("name", ["fp_x_flux", "mf", "mf_error"])
def test_adapter_rejects_missing_required_variables(name: str) -> None:
    """A missing selected input variable should produce a targeted error."""
    dataset = _shuffled_input_dataset().drop_vars(name)

    with pytest.raises(ValueError, match=name):
        problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sensitivity", "message"),
    [
        (
            xr.DataArray(
                np.ones((2, 2)),
                dims=("nmeasure", "lat"),
                coords={"nmeasure": [0, 1], "lat": [50.0, 51.0]},
            ),
            "missing=.*lon",
        ),
        (
            xr.DataArray(
                np.ones((1, 2, 2, 3)),
                dims=("source", "nmeasure", "lat", "lon"),
                coords={
                    "source": ["total"],
                    "nmeasure": [0, 1],
                    "lat": [50.0, 51.0],
                    "lon": [10.0, 20.0, 30.0],
                },
            ),
            "extra=.*source",
        ),
    ],
    ids=("missing-grid-dimension", "multi-sector-extra-dimension"),
)
def test_adapter_rejects_malformed_sensitivity_dims(
    sensitivity: xr.DataArray,
    message: str,
) -> None:
    """Missing grid axes and multi-sector sensitivity should fail clearly."""
    dataset = xr.Dataset(
        {
            "fp_x_flux": sensitivity,
            "mf": xr.DataArray([1.0, 2.0], dims="nmeasure"),
            "mf_error": xr.DataArray([0.1, 0.2], dims="nmeasure"),
        }
    )

    with pytest.raises(ValueError, match=message):
        problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("mf", xr.DataArray([[1.0], [2.0]], dims=("nmeasure", "site"))),
        ("mf_error", xr.DataArray([[0.1], [0.2]], dims=("nmeasure", "site"))),
    ],
    ids=("observation-extra-dimension", "error-extra-dimension"),
)
def test_adapter_rejects_malformed_observation_dims(name: str, replacement: xr.DataArray) -> None:
    """Observations and errors must be one-dimensional on nmeasure."""
    dataset = _shuffled_input_dataset()
    dataset[name] = replacement.assign_coords(nmeasure=dataset["nmeasure"])

    with pytest.raises(ValueError, match=f"{name!r}.*exactly dimensions"):
        problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize("dimension", ["lat", "lon"])
def test_adapter_rejects_missing_grid_coordinates(dimension: str) -> None:
    """A positional grid axis without scientific coordinates should fail."""
    dataset = _shuffled_input_dataset().drop_indexes(dimension).drop_vars(dimension)

    with pytest.raises(ValueError, match=f"{dimension!r} dimension coordinate"):
        problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dimension", "values", "message"),
    [
        ("lat", [50.0, np.nan], "finite"),
        ("lat", [50.0, 50.0], "unique"),
        ("lon", ["west", "centre", "east"], "numeric"),
    ],
    ids=("nonfinite", "duplicate", "nonnumeric"),
)
def test_adapter_rejects_malformed_grid_coordinates(
    dimension: str,
    values: list[object],
    message: str,
) -> None:
    """Grid coordinates must be numeric, finite, and unique."""
    dataset = _shuffled_input_dataset().assign_coords({dimension: values})

    with pytest.raises(ValueError, match=message):
        problem_from_rhime_inputs(dataset, **_adapter_kwargs())  # type: ignore[arg-type]


def test_adapter_rejects_non_dataset_input() -> None:
    """The public seam should reject non-xarray containers."""
    with pytest.raises(TypeError, match="xarray.Dataset"):
        problem_from_rhime_inputs(object(), **_adapter_kwargs())  # type: ignore[arg-type]
