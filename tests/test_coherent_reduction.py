"""Tests for exact labelled native-to-retained Gaussian reduction."""

from dataclasses import replace

from dask import delayed
from dask import array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions._labelled_matrices import with_square_matrix_diagnostics
from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    project_native_covariance,
)
from openghg_inversions.basis.operators import BucketBasisOperator
from openghg_inversions.coherent_reduction import (
    _reduce_native_covariance_products as reduce_native_gaussian,
)
from openghg_inversions.coherent_reduction import (
    reduce_native_gaussian as reduce_native_gaussian_from_native,
)
from openghg_inversions.native_covariance import SeparableExponentialCovariance


def _raw_projected_problem() -> tuple[
    NativeCovarianceProducts,
    xr.DataArray,
    xr.DataArray,
    np.ndarray,
    SeparableExponentialCovariance,
    xr.DataArray,
]:
    """Return one small separable-covariance fixture and its dense ``B``."""
    latitude = xr.DataArray(
        [-1.0, 1.0],
        dims="lat",
        coords={"lat": [-1.0, 1.0]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.5],
        dims="lon",
        coords={"lon": [10.0, 11.5]},
        attrs={"units": "degrees_east"},
    )
    covariance = SeparableExponentialCovariance(
        latitude,
        longitude,
        sigma=1.2,
        correlation_length=1.1,
    )
    basis = BucketBasisOperator(
        xr.DataArray(
            [[1, 1], [2, 2]],
            dims=("lat", "lon"),
            coords={"lat": latitude, "lon": longitude},
        ),
        state_dim="state",
    )
    sensitivity = xr.DataArray(
        [
            [[0.3, -0.2], [0.5, 0.7]],
            [[-0.4, 0.6], [0.2, 0.1]],
            [[0.8, 0.2], [-0.3, 0.4]],
        ],
        dims=("observation", "lat", "lon"),
        coords={
            "observation": ["MHD-1", "TAC-1", "RGL-1"],
            "lat": latitude,
            "lon": longitude,
        },
        attrs={"units": "ppt"},
    )
    native_mean = xr.DataArray(
        [[0.8, 1.1], [1.3, 0.9]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        attrs={"units": "1"},
        name="native_mean",
    )
    prolongation = to_dense(
        basis.native_prolongation(sensitivity, native_dims=covariance.native_dims)
    ).compute()
    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_sensitivity=sensitivity,
        observation_dim="observation",
    )
    latitude_factor = np.exp(
        -np.abs(latitude.values[:, None] - latitude.values[None, :]) / 1.1
    )
    longitude_factor = np.exp(
        -np.abs(longitude.values[:, None] - longitude.values[None, :]) / 1.1
    )
    dense_covariance = 1.2**2 * np.kron(latitude_factor, longitude_factor)
    return products, native_mean, sensitivity, dense_covariance, covariance, prolongation


def _projected_problem() -> tuple[
    NativeCovarianceProducts,
    xr.DataArray,
    xr.DataArray,
    np.ndarray,
]:
    """Return the product-level view used by focused internal validation tests."""
    products, native_mean, sensitivity, dense_covariance, _, _ = _raw_projected_problem()
    return products, native_mean, sensitivity, dense_covariance


def test_public_reduction_constructs_and_reduces_products_atomically() -> None:
    """The public operation accepts one native model rather than mixable products."""
    products, native_mean, sensitivity, _, covariance, prolongation = _raw_projected_problem()

    result = reduce_native_gaussian_from_native(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_mean=native_mean,
        native_sensitivity=sensitivity,
        observation_dim="observation",
    )

    expected = reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=sensitivity,
    )
    xr.testing.assert_allclose(result.retained_mean, expected.retained_mean)
    xr.testing.assert_allclose(
        result.unresolved_observation_covariance,
        expected.unresolved_observation_covariance,
    )


def test_exact_reduction_matches_dense_nonzero_mean_oracle() -> None:
    """The retained prior, centred forward model, and residual match dense algebra."""
    products, native_mean, sensitivity, dense_covariance = _projected_problem()
    result = reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=sensitivity,
    )

    restriction = products.restriction.values.reshape(2, 4)
    h_matrix = sensitivity.values.reshape(3, 4)
    mean_values = native_mean.values.reshape(4)
    retained_covariance = restriction @ dense_covariance @ restriction.T
    effective_operator = (
        h_matrix
        @ dense_covariance
        @ restriction.T
        @ np.linalg.solve(retained_covariance, np.eye(2))
    )
    unresolved = (
        h_matrix @ dense_covariance @ h_matrix.T
        - effective_operator @ retained_covariance @ effective_operator.T
    )

    np.testing.assert_allclose(result.retained_mean, restriction @ mean_values)
    np.testing.assert_allclose(result.retained_covariance, retained_covariance)
    np.testing.assert_allclose(result.effective_observation_operator, effective_operator)
    np.testing.assert_allclose(result.native_observation_mean, h_matrix @ mean_values)
    np.testing.assert_allclose(
        result.observation_intercept,
        h_matrix @ mean_values - effective_operator @ restriction @ mean_values,
    )
    np.testing.assert_allclose(result.unresolved_observation_covariance, unresolved, atol=1e-12)
    assert result.retained_mean.dims == ("state",)
    assert result.retained_covariance.dims == ("state", "state_cov")
    assert result.unresolved_observation_covariance.dims == (
        "observation",
        "observation_cov",
    )
    assert result.retained_mean.attrs["units"] == "1"
    assert result.native_observation_mean.attrs["units"] == "ppt"
    assert result.unresolved_observation_covariance.attrs["units"] == "(ppt)^2"
    assert result.projection_strategy == "preserve_bucket_prolongation"
    assert result == result
    assert result != reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=sensitivity,
    )


def test_centered_form_and_prior_predictive_covariance_are_coherent() -> None:
    """The affine and centred means agree and unresolved variance closes the ledger."""
    products, native_mean, sensitivity, _ = _projected_problem()
    result = reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=sensitivity,
    )
    alpha = xr.DataArray(
        [0.6, 1.4],
        dims="state",
        coords={"state": result.retained_mean.state},
    )

    affine = result.observation_intercept + xr.dot(
        result.effective_observation_operator,
        alpha,
        dim="state",
    )
    centered = result.native_observation_mean + xr.dot(
        result.effective_observation_operator,
        alpha - result.retained_mean,
        dim="state",
    )
    xr.testing.assert_allclose(affine, centered)

    h_alpha = result.effective_observation_operator.values
    c_alpha = result.retained_covariance.values
    recovered_native_covariance = (
        result.unresolved_observation_covariance.values
        + h_alpha @ c_alpha @ h_alpha.T
    )
    np.testing.assert_allclose(
        recovered_native_covariance,
        products.native_observation_covariance.values,
        atol=1e-12,
    )


def test_diagonal_native_observation_view_is_rejected() -> None:
    """A diagonal ``H B H.T`` view cannot determine the dense Schur complement."""
    products, native_mean, sensitivity, _ = _projected_problem()
    diagonal_products = replace(
        products,
        native_observation_covariance=products.native_observation_covariance.isel(
            observation_cov=xr.DataArray([0, 1, 2], dims="observation")
        ),
        observation_covariance_view="diagonal",
    )

    with pytest.raises(ValueError, match="dense|diagonal|unresolved"):
        reduce_native_gaussian(
            products=diagonal_products,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


@pytest.mark.parametrize("reordered_input", ["native_mean", "native_sensitivity"])
def test_reordered_input_labels_are_rejected(reordered_input: str) -> None:
    """The reducer never silently reorders native or observation axes."""
    products, native_mean, sensitivity, _ = _projected_problem()
    if reordered_input == "native_mean":
        native_mean = native_mean.isel(lat=[1, 0])
    else:
        sensitivity = sensitivity.isel(observation=[1, 0, 2])

    with pytest.raises(ValueError, match="labels|same order|match exactly"):
        reduce_native_gaussian(
            products=products,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_invalid_lazy_structure_fails_without_executing_payload() -> None:
    """Dimension validation precedes the named joint materialization boundary."""
    _, native_mean, sensitivity, _, covariance, prolongation = _raw_projected_problem()
    executions = 0

    @delayed
    def produce_mean() -> np.ndarray:
        """Return mean values and record forbidden graph execution."""
        nonlocal executions
        executions += 1
        return native_mean.values

    lazy_mean = xr.DataArray(
        da.from_delayed(produce_mean(), shape=native_mean.shape, dtype=native_mean.dtype),
        dims=("wrong_lat", "lon"),
        coords={"wrong_lat": ("wrong_lat", native_mean.lat.values), "lon": native_mean.lon},
        attrs=native_mean.attrs,
    )

    with pytest.raises(ValueError, match="dimensions"):
        reduce_native_gaussian_from_native(
            covariance=covariance,
            basis_prolongation=prolongation,
            state_dim="state",
            native_mean=lazy_mean,
            native_sensitivity=sensitivity,
            observation_dim="observation",
        )
    assert executions == 0


def test_invalid_basis_grid_fails_without_executing_lazy_inputs() -> None:
    """Basis/native-grid validation precedes joint payload materialization."""
    _, native_mean, sensitivity, _, covariance, prolongation = _raw_projected_problem()
    executions = 0

    @delayed
    def produce_inputs() -> tuple[np.ndarray, np.ndarray]:
        """Return both native inputs and record forbidden graph execution."""
        nonlocal executions
        executions += 1
        return native_mean.values, sensitivity.values

    payload = produce_inputs()
    lazy_mean = native_mean.copy(
        data=da.from_delayed(payload[0], shape=native_mean.shape, dtype=native_mean.dtype)
    )
    lazy_sensitivity = sensitivity.copy(
        data=da.from_delayed(payload[1], shape=sensitivity.shape, dtype=sensitivity.dtype)
    )
    reordered_prolongation = prolongation.isel(lat=[1, 0])

    with pytest.raises(ValueError, match="exactly match"):
        reduce_native_gaussian_from_native(
            covariance=covariance,
            basis_prolongation=reordered_prolongation,
            state_dim="state",
            native_mean=lazy_mean,
            native_sensitivity=lazy_sensitivity,
            observation_dim="observation",
        )
    assert executions == 0


def test_related_lazy_inputs_materialize_one_shared_graph_once() -> None:
    """The named eager boundary jointly computes related mean and sensitivity arrays."""
    _, native_mean, sensitivity, _, covariance, prolongation = _raw_projected_problem()
    executions = 0

    @delayed
    def produce_inputs() -> tuple[np.ndarray, np.ndarray]:
        """Return both native inputs from one recorded upstream task."""
        nonlocal executions
        executions += 1
        return native_mean.values, sensitivity.values

    payload = produce_inputs()
    lazy_mean = native_mean.copy(
        data=da.from_delayed(payload[0], shape=native_mean.shape, dtype=native_mean.dtype)
    )
    lazy_sensitivity = sensitivity.copy(
        data=da.from_delayed(payload[1], shape=sensitivity.shape, dtype=sensitivity.dtype)
    )

    result = reduce_native_gaussian_from_native(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_mean=lazy_mean,
        native_sensitivity=lazy_sensitivity,
        observation_dim="observation",
    )

    assert executions == 1
    np.testing.assert_allclose(
        result.native_observation_mean,
        xr.dot(sensitivity, native_mean, dim=["lat", "lon"]),
    )


@pytest.mark.parametrize(
    "target",
    ["effective-state", "state-column", "cross-column", "observation-column", "duplicate-state"],
)
def test_product_axis_mismatches_are_rejected(target: str) -> None:
    """Every redundant product axis is bound to the canonical typed labels."""
    products, native_mean, sensitivity, _ = _projected_problem()
    if target == "effective-state":
        products = replace(
            products,
            effective_observation_operator=products.effective_observation_operator.assign_coords(
                state=["wrong-1", "wrong-2"]
            ),
        )
    elif target == "state-column":
        products = replace(
            products,
            state_covariance=products.state_covariance.assign_coords(state_cov=[1, 0]),
        )
    elif target == "cross-column":
        products = replace(
            products,
            observation_state_cross_covariance=(
                products.observation_state_cross_covariance.assign_coords(state_cov=[1, 0])
            ),
        )
    elif target == "observation-column":
        products = replace(
            products,
            native_observation_covariance=products.native_observation_covariance.assign_coords(
                observation_cov=["RGL-1", "TAC-1", "MHD-1"]
            ),
        )
    else:
        products = replace(
            products,
            effective_observation_operator=products.effective_observation_operator.assign_coords(
                state=[0, 0]
            ),
        )

    with pytest.raises(ValueError, match="labels|unique|same order|match exactly"):
        reduce_native_gaussian(
            products=products,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


@pytest.mark.parametrize("missing_units", ["mean", "sensitivity"])
def test_missing_linked_units_are_rejected(missing_units: str) -> None:
    """The reducer does not manufacture units absent from its linked inputs."""
    products, native_mean, sensitivity, _ = _projected_problem()
    if missing_units == "mean":
        native_mean = native_mean.copy()
        native_mean.attrs = {}
    else:
        sensitivity = sensitivity.copy()
        sensitivity.attrs = {}

    with pytest.raises(ValueError, match="units|dimensionless"):
        reduce_native_gaussian(
            products=products,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_incoherent_effective_operator_is_rejected() -> None:
    """The reducer solves from cross-covariance and checks OPE-17's direct view."""
    products, native_mean, sensitivity, _ = _projected_problem()
    corrupted_operator = products.effective_observation_operator.copy(
        data=products.effective_observation_operator.data + 0.1
    )
    incoherent = replace(
        products,
        effective_observation_operator=corrupted_operator,
    )

    with pytest.raises(ValueError, match="stored H_alpha|Incoherent"):
        reduce_native_gaussian(
            products=incoherent,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_incoherent_restriction_is_rejected() -> None:
    """The product-level kernel checks ``Pi U_* = I`` before computing ``Pi m``."""
    products, native_mean, sensitivity, _ = _projected_problem()
    incoherent = replace(
        products,
        restriction=products.restriction + 0.1,
    )

    with pytest.raises(ValueError, match="Pi U|retained-state identity|Incoherent"):
        reduce_native_gaussian(
            products=incoherent,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_public_boundary_recomputes_products_when_sensitivity_changes_off_subspace() -> None:
    """An H change orthogonal to ``U_*`` cannot retain covariance from the old H."""
    _, native_mean, sensitivity, dense_covariance, covariance, prolongation = (
        _raw_projected_problem()
    )
    original = reduce_native_gaussian_from_native(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_mean=native_mean,
        native_sensitivity=sensitivity,
        observation_dim="observation",
    )
    perturbation = xr.zeros_like(sensitivity)
    perturbation.values[0] = np.array([[1.0, -1.0], [0.0, 0.0]])
    changed_sensitivity = sensitivity + perturbation
    changed_sensitivity.attrs = sensitivity.attrs

    changed = reduce_native_gaussian_from_native(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_mean=native_mean,
        native_sensitivity=changed_sensitivity,
        observation_dim="observation",
    )

    xr.testing.assert_allclose(
        changed.effective_observation_operator,
        original.effective_observation_operator,
    )
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(changed.native_observation_mean, original.native_observation_mean)
    changed_h = changed_sensitivity.values.reshape(3, 4)
    recovered = (
        changed.unresolved_observation_covariance.values
        + changed.effective_observation_operator.values
        @ changed.retained_covariance.values
        @ changed.effective_observation_operator.values.T
    )
    np.testing.assert_allclose(recovered, changed_h @ dense_covariance @ changed_h.T)


def test_reduction_owns_retained_covariance_snapshot() -> None:
    """Mutating the input product after preparation cannot change the atomic result."""
    products, native_mean, sensitivity, _ = _projected_problem()
    result = reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=sensitivity,
    )
    expected = result.retained_covariance.copy(deep=True)

    products.state_covariance.values[0, 0] += 10.0

    xr.testing.assert_identical(result.retained_covariance, expected)


def test_redundant_retained_covariance_is_rejected_without_pseudoinverse() -> None:
    """A singular ``C_alpha`` fails before any generalized inverse can be selected."""
    products, native_mean, sensitivity, _ = _projected_problem()
    singular = replace(
        products,
        state_covariance=products.state_covariance.copy(
            data=np.zeros_like(products.state_covariance.values)
        ),
    )

    with pytest.raises(ValueError, match="positive definite|full rank|pseudoinverse"):
        reduce_native_gaussian(
            products=singular,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_materially_negative_unresolved_covariance_is_rejected() -> None:
    """Incoherent positive-semidefinite inputs are not clipped into a valid residual."""
    products, native_mean, sensitivity, _ = _projected_problem()
    explained = (
        products.effective_observation_operator.values
        @ products.state_covariance.values
        @ products.effective_observation_operator.values.T
    )
    too_small_hbh = products.native_observation_covariance.copy(data=0.5 * explained)
    incoherent = replace(products, native_observation_covariance=too_small_hbh)

    with pytest.raises(ValueError, match="positive semidefinite|Incoherent"):
        reduce_native_gaussian(
            products=incoherent,
            native_mean=native_mean,
            native_sensitivity=sensitivity,
        )


def test_nonlexical_multiindex_labels_survive_reduction() -> None:
    """Gathered state and observation identities retain typed order and levels."""
    products, native_mean, sensitivity, _ = _projected_problem()
    state_index = pd.MultiIndex.from_tuples(
        [("z-source", 7), ("a-source", 2)],
        names=("source", "region"),
    )
    state_column_index = state_index.set_names(("source_cov", "region_cov"))
    observation_index = pd.MultiIndex.from_tuples(
        [
            ("TAC", pd.Timestamp("2021-01-03")),
            ("MHD", pd.Timestamp("2021-01-01")),
            ("RGL", pd.Timestamp("2021-01-02")),
        ],
        names=("site", "time"),
    )
    observation_column_index = observation_index.set_names(("site_cov", "time_cov"))

    def relabel(
        array: xr.DataArray,
        indexes: tuple[tuple[str, pd.MultiIndex], ...],
    ) -> xr.DataArray:
        """Rebuild one fixture array with explicit typed MultiIndex axes."""
        coords = {
            str(name): coordinate
            for name, coordinate in array.coords.items()
            if all(dim not in {item[0] for item in indexes} for dim in coordinate.dims)
        }
        result = xr.DataArray(
            array.values,
            dims=array.dims,
            coords=coords,
            attrs=array.attrs,
            name=array.name,
        )
        for dim, index in indexes:
            result = result.assign_coords(xr.Coordinates.from_pandas_multiindex(index, dim))
        return result

    labelled_products = replace(
        products,
        restriction=relabel(products.restriction, (("state", state_index),)),
        prolongation=relabel(products.prolongation, (("state", state_index),)),
        state_covariance=relabel(
            products.state_covariance,
            (("state", state_index), ("state_cov", state_column_index)),
        ),
        effective_observation_operator=relabel(
            products.effective_observation_operator,
            (("observation", observation_index), ("state", state_index)),
        ),
        observation_state_cross_covariance=relabel(
            products.observation_state_cross_covariance,
            (("observation", observation_index), ("state_cov", state_column_index)),
        ),
        native_observation_covariance=relabel(
            products.native_observation_covariance,
            (
                ("observation", observation_index),
                ("observation_cov", observation_column_index),
            ),
        ),
    )
    labelled_sensitivity = relabel(
        sensitivity,
        (("observation", observation_index),),
    )

    result = reduce_native_gaussian(
        products=labelled_products,
        native_mean=native_mean,
        native_sensitivity=labelled_sensitivity,
    )

    assert result.retained_mean.get_index("state").equals(state_index)
    assert result.effective_observation_operator.get_index("observation").equals(
        observation_index
    )
    assert result.retained_mean.get_index("state").tolist() == state_index.tolist()
    assert result.native_observation_mean.get_index("observation").tolist() == (
        observation_index.tolist()
    )


def _dense_products(
    *,
    covariance: np.ndarray,
    sensitivity: np.ndarray,
    restriction: np.ndarray,
    state_labels: list[str],
) -> NativeCovarianceProducts:
    """Build exact labelled product blocks for one dense test restriction."""
    state_dim = "state"
    state_column_dim = "state_cov"
    observation_dim = "observation"
    observation_column_dim = "observation_cov"
    state_covariance = restriction @ covariance @ restriction.T
    b_pi_t = covariance @ restriction.T
    prolongation = b_pi_t @ np.linalg.solve(state_covariance, np.eye(len(state_labels)))
    cross_covariance = sensitivity @ b_pi_t
    effective_operator = cross_covariance @ np.linalg.solve(
        state_covariance,
        np.eye(len(state_labels)),
    )
    observations = ["obs-3", "obs-1", "obs-2"]
    native = ["cell-d", "cell-a", "cell-c", "cell-b"]
    return NativeCovarianceProducts(
        restriction=xr.DataArray(
            restriction,
            dims=(state_dim, "native"),
            coords={state_dim: state_labels, "native": native},
            attrs={"units": "1"},
        ),
        prolongation=xr.DataArray(
            prolongation,
            dims=("native", state_dim),
            coords={"native": native, state_dim: state_labels},
            attrs={"units": "1"},
        ),
        state_covariance=xr.DataArray(
            state_covariance,
            dims=(state_dim, state_column_dim),
            coords={state_dim: state_labels, state_column_dim: state_labels},
            attrs={"units": "1"},
        ),
        effective_observation_operator=xr.DataArray(
            effective_operator,
            dims=(observation_dim, state_dim),
            coords={observation_dim: observations, state_dim: state_labels},
            attrs={"units": "ppt"},
        ),
        observation_state_cross_covariance=xr.DataArray(
            cross_covariance,
            dims=(observation_dim, state_column_dim),
            coords={observation_dim: observations, state_column_dim: state_labels},
            attrs={"units": "ppt"},
        ),
        native_observation_covariance=xr.DataArray(
            sensitivity @ covariance @ sensitivity.T,
            dims=(observation_dim, observation_column_dim),
            coords={observation_dim: observations, observation_column_dim: observations},
            attrs={"units": "(ppt)^2"},
        ),
        strategy="test_restriction",
        observation_covariance_view="dense",
    )


def test_full_retention_accepts_roundoff_around_zero_unresolved_covariance() -> None:
    """A vanishing Schur complement uses its parent-product numerical scale."""
    covariance = np.array(
        [
            [1.7, 0.2, 0.1, 0.0],
            [0.2, 1.3, 0.3, 0.1],
            [0.1, 0.3, 1.5, 0.4],
            [0.0, 0.1, 0.4, 1.2],
        ]
    )
    sensitivity_values = np.array(
        [[0.31, -0.27, 0.83, 0.14], [0.22, 0.71, -0.19, 0.53], [-0.41, 0.29, 0.67, 0.91]]
    )
    products = _dense_products(
        covariance=covariance,
        sensitivity=sensitivity_values,
        restriction=np.eye(4),
        state_labels=["d", "a", "c", "b"],
    )
    native_mean = xr.DataArray(
        [0.8, 1.0, 1.2, 0.9],
        dims="native",
        coords={"native": ["cell-d", "cell-a", "cell-c", "cell-b"]},
        attrs={"units": "1"},
    )
    native_sensitivity = xr.DataArray(
        sensitivity_values,
        dims=("observation", "native"),
        coords={
            "observation": ["obs-3", "obs-1", "obs-2"],
            "native": native_mean.native,
        },
        attrs={"units": "ppt"},
    )

    result = reduce_native_gaussian(
        products=products,
        native_mean=native_mean,
        native_sensitivity=native_sensitivity,
    )

    np.testing.assert_allclose(result.unresolved_observation_covariance, 0.0, atol=1e-12)
    assert (
        result.unresolved_observation_covariance.attrs["positive_semidefinite_tolerance"]
        > 0.0
    )


@pytest.mark.parametrize(
    ("smallest_variance", "should_pass"),
    [(1e-6, True), (1e-10, False)],
    ids=["conditioned-valid", "solve-tolerance-failure"],
)
def test_coherent_solve_has_an_explicit_conditioning_limit(
    smallest_variance: float,
    should_pass: bool,
) -> None:
    """Near-conditioned solves pass or fail according to the declared tolerance."""
    rotation, _ = np.linalg.qr(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 1.0, 2.0, 3.0],
                [3.0, 4.0, 1.0, 2.0],
                [2.0, 3.0, 4.0, 1.0],
            ]
        )
    )
    covariance = rotation @ np.diag([1.0, 0.4, 0.02, smallest_variance]) @ rotation.T
    sensitivity_values = np.array(
        [[0.31, -0.27, 0.83, 0.14], [0.22, 0.71, -0.19, 0.53], [-0.41, 0.29, 0.67, 0.91]]
    )
    products = _dense_products(
        covariance=covariance,
        sensitivity=sensitivity_values,
        restriction=np.eye(4),
        state_labels=["d", "a", "c", "b"],
    )
    native_mean = xr.DataArray(
        [0.8, 1.0, 1.2, 0.9],
        dims="native",
        coords={"native": ["cell-d", "cell-a", "cell-c", "cell-b"]},
        attrs={"units": "1"},
    )
    native_sensitivity = xr.DataArray(
        sensitivity_values,
        dims=("observation", "native"),
        coords={
            "observation": ["obs-3", "obs-1", "obs-2"],
            "native": native_mean.native,
        },
        attrs={"units": "ppt"},
    )

    if should_pass:
        result = reduce_native_gaussian(
            products=products,
            native_mean=native_mean,
            native_sensitivity=native_sensitivity,
        )
        np.testing.assert_allclose(result.unresolved_observation_covariance, 0.0, atol=1e-10)
    else:
        with pytest.raises(ValueError, match="ill-conditioned|solve tolerance"):
            reduce_native_gaussian(
                products=products,
                native_mean=native_mean,
                native_sensitivity=native_sensitivity,
            )


def test_positive_semidefinite_diagnostic_modes() -> None:
    """The shared diagnostic accepts singular/roundoff PSD and rejects material negatives."""
    def matrix(diagonal: list[float]) -> xr.DataArray:
        """Return one labelled diagonal matrix."""
        return xr.DataArray(
            np.diag(diagonal),
            dims=("row", "column"),
            coords={"row": ["a", "b"], "column": ["a", "b"]},
        )

    singular = with_square_matrix_diagnostics(
        matrix([1.0, 0.0]),
        mathematical_name="singular PSD",
        require_positive_semidefinite=True,
    )
    assert singular.attrs["minimum_eigenvalue"] == 0.0
    within_roundoff = with_square_matrix_diagnostics(
        matrix([1.0, -1e-12]),
        mathematical_name="roundoff PSD",
        require_positive_semidefinite=True,
        positive_semidefinite_tolerance_scale=1.0,
    )
    assert within_roundoff.attrs["minimum_eigenvalue"] == -1e-12
    with pytest.raises(ValueError, match="positive semidefinite"):
        with_square_matrix_diagnostics(
            matrix([1.0, -1e-4]),
            mathematical_name="indefinite",
            require_positive_semidefinite=True,
            positive_semidefinite_tolerance_scale=1.0,
        )
    with pytest.raises(ValueError, match="either"):
        with_square_matrix_diagnostics(
            matrix([1.0, 0.5]),
            mathematical_name="conflicting modes",
            require_positive_definite=True,
            require_positive_semidefinite=True,
        )
    large = with_square_matrix_diagnostics(
        xr.DataArray(np.eye(513), dims=("large_row", "large_column")),
        mathematical_name="large PSD",
        require_positive_semidefinite=True,
    )
    assert large.attrs["psd_diagnostic"] == "skipped_full_eigendecomposition_above_512"


def test_two_basis_choices_recover_projected_posteriors_and_equal_evidence() -> None:
    """Exact reduction commutes with Gaussian updating for two different bases."""
    covariance = np.array(
        [
            [1.4, 0.3, 0.1, 0.2],
            [0.3, 1.1, 0.2, 0.0],
            [0.1, 0.2, 1.3, 0.4],
            [0.2, 0.0, 0.4, 1.2],
        ]
    )
    sensitivity_values = np.array(
        [[0.4, -0.2, 0.8, 0.1], [0.3, 0.7, -0.1, 0.5], [-0.4, 0.2, 0.6, 0.9]]
    )
    mean_values = np.array([0.8, 1.0, 1.2, 0.9])
    observation_error = np.diag([0.2, 0.3, 0.25])
    observations = np.array([0.7, 1.4, 0.2])
    native_labels = ["cell-d", "cell-a", "cell-c", "cell-b"]
    observation_labels = ["obs-3", "obs-1", "obs-2"]
    native_mean = xr.DataArray(
        mean_values,
        dims="native",
        coords={"native": native_labels},
        attrs={"units": "1"},
    )
    native_sensitivity = xr.DataArray(
        sensitivity_values,
        dims=("observation", "native"),
        coords={"observation": observation_labels, "native": native_labels},
        attrs={"units": "ppt"},
    )
    predictive_covariance = (
        sensitivity_values @ covariance @ sensitivity_values.T + observation_error
    )
    innovation = observations - sensitivity_values @ mean_values
    native_gain = covariance @ sensitivity_values.T @ np.linalg.solve(
        predictive_covariance,
        np.eye(3),
    )
    native_posterior_mean = mean_values + native_gain @ innovation
    native_posterior_covariance = covariance - native_gain @ sensitivity_values @ covariance
    evidence_sign, evidence_logdet = np.linalg.slogdet(predictive_covariance)
    assert evidence_sign > 0
    native_log_evidence = -0.5 * (
        3 * np.log(2.0 * np.pi)
        + evidence_logdet
        + innovation @ np.linalg.solve(predictive_covariance, innovation)
    )

    restrictions = (
        (np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]) / 2.0, ["z", "a"]),
        (
            np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            ["outer", "inner-b", "inner-a"],
        ),
    )
    reduced_log_evidence: list[float] = []
    for restriction, state_labels in restrictions:
        products = _dense_products(
            covariance=covariance,
            sensitivity=sensitivity_values,
            restriction=restriction,
            state_labels=state_labels,
        )
        result = reduce_native_gaussian(
            products=products,
            native_mean=native_mean,
            native_sensitivity=native_sensitivity,
        )
        h_alpha = result.effective_observation_operator.values
        c_alpha = result.retained_covariance.values
        conditional_covariance = observation_error + result.unresolved_observation_covariance.values
        reduced_predictive_covariance = (
            h_alpha @ c_alpha @ h_alpha.T + conditional_covariance
        )
        reduced_gain = c_alpha @ h_alpha.T @ np.linalg.solve(
            reduced_predictive_covariance,
            np.eye(3),
        )
        reduced_posterior_mean = result.retained_mean.values + reduced_gain @ innovation
        reduced_posterior_covariance = c_alpha - reduced_gain @ h_alpha @ c_alpha

        np.testing.assert_allclose(reduced_posterior_mean, restriction @ native_posterior_mean)
        np.testing.assert_allclose(
            reduced_posterior_covariance,
            restriction @ native_posterior_covariance @ restriction.T,
        )
        sign, logdet = np.linalg.slogdet(reduced_predictive_covariance)
        assert sign > 0
        reduced_log_evidence.append(
            -0.5
            * (
                3 * np.log(2.0 * np.pi)
                + logdet
                + innovation @ np.linalg.solve(reduced_predictive_covariance, innovation)
            )
        )

    np.testing.assert_allclose(reduced_log_evidence, native_log_evidence)
