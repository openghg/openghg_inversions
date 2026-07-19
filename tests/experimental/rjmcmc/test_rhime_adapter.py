"""Tests for the filtered RHIME-to-TD-MCMC input adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.rhime_adapter import (
    _align_nmeasure_exact,
    problem_from_rhime_inputs,
)


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
    assert problem.fixed_block is None
    assert problem.fixed_offset is not None
    np.testing.assert_array_equal(problem.fixed_offset, np.zeros(2))


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


def test_adapter_transposes_fixed_design_and_broadcasts_scalar_priors() -> None:
    """An explicit fixed design should become measurement-major core arrays."""
    dataset = _shuffled_input_dataset()
    fixed_design = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dataset["outer_design"] = xr.DataArray(
        fixed_design,
        dims=("outer_region", "nmeasure"),
        coords={
            "outer_region": ["north", "south", "west"],
            "nmeasure": dataset.coords["nmeasure"],
        },
    )

    problem = problem_from_rhime_inputs(
        dataset,
        **_adapter_kwargs(),  # type: ignore[arg-type]
        fixed_design_name="outer_design",
        fixed_coefficient_prior_mean=1.5,
        fixed_coefficient_prior_sd=0.25,
    )

    assert problem.fixed_block is not None
    np.testing.assert_array_equal(problem.fixed_block.design, fixed_design.T)
    np.testing.assert_array_equal(problem.fixed_block.coefficient_prior_mean, [1.5, 1.5, 1.5])
    np.testing.assert_array_equal(problem.fixed_block.coefficient_prior_sd, [0.25, 0.25, 0.25])


def test_adapter_accepts_per_column_fixed_priors() -> None:
    """Fixed prior vectors should preserve their declared column order."""
    dataset = _shuffled_input_dataset()
    dataset["outer_design"] = xr.DataArray(
        np.arange(1.0, 5.0).reshape(2, 2),
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": dataset.coords["nmeasure"], "outer_region": ["a", "b"]},
    )

    problem = problem_from_rhime_inputs(
        dataset,
        **_adapter_kwargs(),  # type: ignore[arg-type]
        fixed_design_name="outer_design",
        fixed_coefficient_prior_mean=[1.0, 2.0],
        fixed_coefficient_prior_sd=np.array([0.2, 0.4]),
    )

    assert problem.fixed_block is not None
    np.testing.assert_array_equal(problem.fixed_block.coefficient_prior_mean, [1.0, 2.0])
    np.testing.assert_array_equal(problem.fixed_block.coefficient_prior_sd, [0.2, 0.4])


def test_adapter_accepts_offset_without_a_fixed_design() -> None:
    """A selected offset should not require or create fixed coefficients."""
    dataset = _shuffled_input_dataset()
    dataset["baseline"] = xr.DataArray(
        [0.25, -0.5],
        dims="nmeasure",
        coords={"nmeasure": dataset.coords["nmeasure"]},
    )

    problem = problem_from_rhime_inputs(
        dataset,
        **_adapter_kwargs(),  # type: ignore[arg-type]
        fixed_offset_name="baseline",
    )

    assert problem.fixed_block is None
    assert problem.fixed_offset is not None
    np.testing.assert_array_equal(problem.fixed_offset, [0.25, -0.5])


def test_nmeasure_alignment_rejects_coordinate_order_mismatch() -> None:
    """Optional arrays must preserve the filtered measurement index and order."""
    sensitivity = xr.DataArray(
        np.ones((2, 1, 1)),
        dims=("nmeasure", "lat", "lon"),
        coords={"nmeasure": ["first", "second"], "lat": [50.0], "lon": [0.0]},
    )
    fixed_design = xr.DataArray(
        np.ones((2, 1)),
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": ["second", "first"], "outer_region": ["outer"]},
    )

    with pytest.raises(ValueError, match="align exactly.*nmeasure"):
        _align_nmeasure_exact(sensitivity, fixed_design, "outer_design")


@pytest.mark.parametrize(
    "design",
    [
        xr.DataArray(np.ones(2), dims="nmeasure"),
        xr.DataArray(np.ones((2, 1, 1)), dims=("nmeasure", "sector", "region")),
        xr.DataArray(np.ones((2, 1)), dims=("time", "outer_region")),
    ],
    ids=("one-dimensional", "three-dimensional", "missing-nmeasure"),
)
def test_adapter_rejects_malformed_fixed_design_dimensions(design: xr.DataArray) -> None:
    """A fixed design must contain exactly one measurement and one column axis."""
    dataset = _shuffled_input_dataset()
    dataset["outer_design"] = design

    with pytest.raises(ValueError, match="exactly two dimensions including 'nmeasure'"):
        problem_from_rhime_inputs(
            dataset,
            **_adapter_kwargs(),  # type: ignore[arg-type]
            fixed_design_name="outer_design",
            fixed_coefficient_prior_mean=1.0,
            fixed_coefficient_prior_sd=0.2,
        )


@pytest.mark.parametrize(
    ("mean", "standard_deviation", "message"),
    [
        (None, 0.2, "requires both"),
        (1.0, None, "requires both"),
        ([1.0, 2.0, 3.0], 0.2, "one value per"),
        (1.0, [0.2, np.nan], "finite"),
    ],
    ids=("missing-mean", "missing-sd", "wrong-length", "nonfinite-sd"),
)
def test_adapter_rejects_malformed_fixed_prior_moments(
    mean: object,
    standard_deviation: object,
    message: str,
) -> None:
    """An explicit fixed design requires valid scalar or per-column priors."""
    dataset = _shuffled_input_dataset()
    dataset["outer_design"] = xr.DataArray(
        np.ones((2, 2)),
        dims=("nmeasure", "outer_region"),
    )

    with pytest.raises(ValueError, match=message):
        problem_from_rhime_inputs(
            dataset,
            **_adapter_kwargs(),  # type: ignore[arg-type]
            fixed_design_name="outer_design",
            fixed_coefficient_prior_mean=mean,  # type: ignore[arg-type]
            fixed_coefficient_prior_sd=standard_deviation,  # type: ignore[arg-type]
        )


def test_adapter_rejects_fixed_priors_without_a_design_name() -> None:
    """Fixed priors must not cause an implicit search for a design variable."""
    with pytest.raises(ValueError, match="require an explicit fixed_design_name"):
        problem_from_rhime_inputs(
            _shuffled_input_dataset(),
            **_adapter_kwargs(),  # type: ignore[arg-type]
            fixed_coefficient_prior_mean=1.0,
            fixed_coefficient_prior_sd=0.2,
        )


@pytest.mark.parametrize("name", ["outer_design", "baseline"])
def test_adapter_rejects_nonfinite_optional_prediction_inputs(name: str) -> None:
    """Selected fixed-design and offset variables must contain finite values."""
    dataset = _shuffled_input_dataset()
    if name == "outer_design":
        dataset[name] = xr.DataArray(
            [[1.0], [np.nan]],
            dims=("nmeasure", "outer_region"),
        )
        optional_kwargs: dict[str, object] = {
            "fixed_design_name": name,
            "fixed_coefficient_prior_mean": 1.0,
            "fixed_coefficient_prior_sd": 0.2,
        }
    else:
        dataset[name] = xr.DataArray([0.0, np.inf], dims="nmeasure")
        optional_kwargs = {"fixed_offset_name": name}

    with pytest.raises(ValueError, match=f"{name!r}.*finite"):
        problem_from_rhime_inputs(
            dataset,
            **_adapter_kwargs(),  # type: ignore[arg-type]
            **optional_kwargs,  # type: ignore[arg-type]
        )


def test_adapter_rejects_malformed_fixed_offset_dimensions() -> None:
    """A selected fixed offset must be one-dimensional along nmeasure."""
    dataset = _shuffled_input_dataset()
    dataset["baseline"] = xr.DataArray(
        np.ones((2, 1)),
        dims=("nmeasure", "component"),
    )

    with pytest.raises(ValueError, match="'baseline'.*exactly dimensions"):
        problem_from_rhime_inputs(
            dataset,
            **_adapter_kwargs(),  # type: ignore[arg-type]
            fixed_offset_name="baseline",
        )


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
