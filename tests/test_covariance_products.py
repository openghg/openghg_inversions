"""Test labelled covariance projection, eager batching, and numerical contracts."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed
from dask import array as da

from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    PreserveBucketProlongation,
    RetainedProjection,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.operators import BucketBasisOperator
from openghg_inversions.native_covariance import SeparableExponentialCovariance


def _problem() -> tuple[
    SeparableExponentialCovariance,
    BucketBasisOperator,
    xr.DataArray,
    np.ndarray,
]:
    """Return a labelled small-grid problem and its dense native covariance."""
    latitude = xr.DataArray(
        [-1.0, 0.5, 2.0],
        dims="lat",
        coords={"lat": [-1.0, 0.5, 2.0]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.0],
        dims="lon",
        coords={"lon": [10.0, 11.0]},
        attrs={"units": "degrees_east"},
    )
    covariance = SeparableExponentialCovariance(
        latitude=latitude,
        longitude=longitude,
        sigma=1.4,
        correlation_length=0.9,
    )
    basis_flat = xr.DataArray(
        [[1, 1], [1, 2], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        name="basis_flat",
    )
    basis_operator = BucketBasisOperator(basis_flat, state_dim="state")
    native_sensitivity = xr.DataArray(
        np.array(
            [
                [[0.2, -0.1], [0.4, 0.7], [0.3, -0.2]],
                [[-0.3, 0.8], [0.1, -0.4], [0.9, 0.5]],
                [[0.6, 0.2], [-0.7, 0.3], [0.2, 0.1]],
            ]
        ),
        dims=("observation", "lat", "lon"),
        coords={
            "observation": ["MHD-1", "TAC-1", "RGL-1"],
            "lat": latitude,
            "lon": longitude,
        },
        name="native_sensitivity",
    )
    k_lat = np.exp(-np.abs(latitude.values[:, np.newaxis] - latitude.values[np.newaxis, :]) / 0.9)
    k_lon = np.exp(-np.abs(longitude.values[:, np.newaxis] - longitude.values[np.newaxis, :]) / 0.9)
    dense_b = 1.4**2 * np.kron(k_lat, k_lon)
    return covariance, basis_operator, native_sensitivity, dense_b


def _dense_operators(
    basis_operator: BucketBasisOperator,
    dense_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense-oracle prolongation, restriction, and retained covariance."""
    matrix = basis_operator.basis_matrix.transpose("lat", "lon", "state").compute()
    u = np.asarray(matrix.data.todense()).reshape(6, 2)
    precision_u = np.linalg.solve(dense_b, u)
    c_alpha = np.linalg.inv(u.T @ precision_u)
    restriction = c_alpha @ precision_u.T
    return u, restriction, c_alpha


def _project(
    covariance: SeparableExponentialCovariance,
    basis_operator: BucketBasisOperator,
    native_sensitivity: xr.DataArray,
    **options,
) -> NativeCovarianceProducts:
    """Project the shared problem with its canonical observation dimension."""
    basis_prolongation = to_dense(
        basis_operator.native_prolongation(
            native_sensitivity,
            native_dims=covariance.native_dims,
        )
    ).compute()
    return project_native_covariance(
        covariance=covariance,
        basis_prolongation=basis_prolongation,
        state_dim=basis_operator.meta.state_dim,
        native_sensitivity=native_sensitivity,
        observation_dim="observation",
        **options,
    )


def test_preserve_bucket_strategy_derives_compatible_restriction() -> None:
    """The derived restriction satisfies Pi_U U_bucket = identity with stable labels."""
    covariance, basis_operator, _, dense_b = _problem()
    u, expected_pi, _ = _dense_operators(basis_operator, dense_b)

    projection = PreserveBucketProlongation().projection(
        covariance,
        to_dense(basis_operator.basis_matrix).compute(),
        native_dims=covariance.native_dims,
        state_dim="state",
    )

    assert projection.strategy == "preserve_bucket_prolongation"
    assert projection.restriction.dims == ("state", "lat", "lon")
    assert projection.restriction.state.values.tolist() == [0, 1]
    np.testing.assert_allclose(
        projection.restriction.values.reshape(2, 6), expected_pi, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        projection.restriction.values.reshape(2, 6) @ u,
        np.eye(2),
        rtol=1e-11,
        atol=1e-11,
    )


def test_array_product_handoffs_use_identity_equality() -> None:
    """Array-bearing handoffs do not invoke ambiguous xarray value equality."""
    covariance, basis_operator, h, _ = _problem()
    products = _project(covariance, basis_operator, h)

    assert products == products
    assert products != _project(covariance, basis_operator, h)
    projection = RetainedProjection(products.restriction, products.strategy)
    assert projection == projection
    assert projection != RetainedProjection(products.restriction, products.strategy)


def test_products_match_dense_oracle_and_coherent_forward_model() -> None:
    """C_alpha, H_alpha, H B Pi.T, and H B H.T match dense coherent oracles."""
    covariance, basis_operator, h, dense_b = _problem()
    u, expected_pi, expected_c_alpha = _dense_operators(basis_operator, dense_b)
    h_matrix = h.values.reshape(3, 6)

    products = _project(covariance, basis_operator, h)

    expected_h_alpha = h_matrix @ u
    expected_hb_pi_t = h_matrix @ dense_b @ expected_pi.T
    expected_hbht = h_matrix @ dense_b @ h_matrix.T
    np.testing.assert_allclose(products.state_covariance, expected_c_alpha, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        products.effective_observation_operator, expected_h_alpha, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        expected_hb_pi_t,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(products.native_observation_covariance, expected_hbht, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        expected_h_alpha @ expected_c_alpha,
        rtol=1e-11,
        atol=1e-11,
    )
    assert products.state_covariance.dims == ("state", "state_cov")
    assert products.effective_observation_operator.dims == ("observation", "state")
    assert products.observation_state_cross_covariance.dims == ("observation", "state_cov")
    assert products.native_observation_covariance.dims == ("observation", "observation_cov")
    assert products.native_observation_covariance.observation.values.tolist() == [
        "MHD-1",
        "TAC-1",
        "RGL-1",
    ]
    assert products.native_observation_covariance.observation_cov.values.tolist() == [
        "MHD-1",
        "TAC-1",
        "RGL-1",
    ]


def test_covariance_natural_prolongation_is_the_bucket_prolongation() -> None:
    """The compatible Pi makes B Pi.T C_alpha^-1 exactly equal U_bucket."""
    covariance, basis_operator, h, dense_b = _problem()
    u, _, _ = _dense_operators(basis_operator, dense_b)
    products = _project(covariance, basis_operator, h)
    pi = products.restriction.values.reshape(2, 6)

    u_star = dense_b @ pi.T @ np.linalg.inv(products.state_covariance.values)

    np.testing.assert_allclose(u_star, u, rtol=1e-11, atol=1e-11)


def test_dense_and_diagonal_products_are_invariant_to_rhs_batching() -> None:
    """RHS batching preserves dense products and the diagonal view."""
    covariance, basis_operator, h, _ = _problem()

    full_batch = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="dense",
        observation_batch_size=64,
    )
    single_observation_batches = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="dense",
        observation_batch_size=1,
    )
    diagonal = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="diagonal",
        observation_batch_size=2,
    )

    for batched, full in (
        (single_observation_batches.restriction, full_batch.restriction),
        (single_observation_batches.prolongation, full_batch.prolongation),
        (single_observation_batches.state_covariance, full_batch.state_covariance),
        (
            single_observation_batches.effective_observation_operator,
            full_batch.effective_observation_operator,
        ),
        (
            single_observation_batches.observation_state_cross_covariance,
            full_batch.observation_state_cross_covariance,
        ),
        (
            single_observation_batches.native_observation_covariance,
            full_batch.native_observation_covariance,
        ),
    ):
        xr.testing.assert_allclose(batched, full)
    assert full_batch.observation_covariance_view == "dense"
    assert diagonal.observation_covariance_view == "diagonal"
    assert diagonal.native_observation_covariance.dims == ("observation",)
    expected_diagonal = xr.DataArray(
        np.diag(full_batch.native_observation_covariance.values),
        dims="observation",
        coords={"observation": h.observation},
        name="native_observation_covariance",
    )
    np.testing.assert_allclose(diagonal.native_observation_covariance, expected_diagonal)


def test_single_source_native_prolongation_preserves_native_auxiliary_coordinates() -> None:
    """Canonical single-source U retains native-axis and scalar context coordinates."""
    covariance, basis_operator, h, _ = _problem()
    native_layout = h.assign_coords(
        cell_area=(("lat", "lon"), np.arange(6).reshape(3, 2)),
        latitude_band=("lat", ["south", "middle", "north"]),
        grid_mapping="latitude_longitude",
    )

    prolongation = basis_operator.native_prolongation(
        native_layout,
        native_dims=covariance.native_dims,
    )

    assert prolongation.dims == ("lat", "lon", "state")
    xr.testing.assert_identical(prolongation.coords["cell_area"], native_layout.coords["cell_area"])
    xr.testing.assert_identical(
        prolongation.coords["latitude_band"],
        native_layout.coords["latitude_band"],
    )
    assert prolongation.coords["grid_mapping"].item() == "latitude_longitude"


@pytest.mark.parametrize("invalid_batch_size", [0, -1])
def test_observation_batch_size_must_be_positive(invalid_batch_size: int) -> None:
    """The batching operation requires a positive block size."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(ValueError, match="positive"):
        _project(
            covariance,
            basis_operator,
            h,
            observation_batch_size=invalid_batch_size,
        )


def test_fractional_batch_size_is_not_silently_truncated() -> None:
    """Static typing does not justify changing a supplied numerical value."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(TypeError):
        _project(covariance, basis_operator, h, observation_batch_size=1.9)
@pytest.mark.parametrize("collision", ["observation-dimension", "auxiliary-coordinate"])
def test_state_column_name_avoids_observation_namespace_collisions(collision: str) -> None:
    """The generated state column avoids observation dimensions and coordinates."""
    covariance, basis_operator, h, _ = _problem()
    observation_dim = "observation"
    if collision == "observation-dimension":
        h = h.rename(observation="state_cov")
        observation_dim = "state_cov"
    else:
        h = h.assign_coords(state_cov=("observation", ["MHD", "TAC", "RGL"]))
    prolongation = to_dense(
        basis_operator.native_prolongation(h, native_dims=covariance.native_dims)
    ).compute()

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_sensitivity=h,
        observation_dim=observation_dim,
    )

    state_column_dim = products.observation_state_cross_covariance.dims[1]
    assert state_column_dim != observation_dim
    assert state_column_dim not in h.coords
    assert products.observation_state_cross_covariance.shape == (3, 2)


@pytest.mark.parametrize(
    "target_values",
    [[1e20, -1.0, 1.0], [1e-20, -1e-20, 1e-20]],
    ids=["unrelated-huge-positive", "small-scale-no-unit-floor"],
)
def test_diagonal_observation_covariance_uses_per_observation_negative_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    target_values: list[float],
) -> None:
    """Negative tolerance is local, scale-invariant, and has no absolute unit floor."""
    covariance, basis_operator, h, _ = _problem()
    original_apply = type(covariance).apply

    def mixed_scale_apply(self, rhs):
        """Return the parameterized observation variances."""
        rhs_dim = next(dim for dim in rhs.dims if dim not in covariance.native_dims)
        if not str(rhs_dim).startswith("observation"):
            return original_apply(self, rhs)
        squared_norm = (rhs * rhs).sum(dim=list(covariance.native_dims))
        target = xr.DataArray(
            target_values,
            dims=rhs_dim,
            coords={rhs_dim: rhs.coords[rhs_dim]},
        )
        return rhs * (target / squared_norm)

    monkeypatch.setattr(type(covariance), "apply", mixed_scale_apply)

    with pytest.raises(ValueError, match="negative|nonnegative|diagonal"):
        _project(
            covariance,
            basis_operator,
            h,
            observation_covariance="diagonal",
        )


def test_diagonal_observation_covariance_allows_tiny_local_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny negative diagonal within its local numerical scale is tolerated."""
    covariance, basis_operator, h, _ = _problem()
    original_apply = type(covariance).apply

    def roundoff_apply(self, rhs):
        """Create one tiny negative through cancellation of order-one terms."""
        rhs_dim = next(dim for dim in rhs.dims if dim not in covariance.native_dims)
        if not str(rhs_dim).startswith("observation"):
            return original_apply(self, rhs)
        result = xr.zeros_like(rhs)
        rhs_values = np.asarray(rhs.values)
        result_values = np.asarray(result.values)
        result_values[..., 0] = rhs_values[..., 0]
        result_values[..., 2] = rhs_values[..., 2]
        middle_rhs = rhs_values[..., 1].reshape(-1)
        middle_result = result_values[..., 1].reshape(-1)
        middle_result[0] = 1.0 / middle_rhs[0]
        middle_result[1] = (-1.0 - 1e-14) / middle_rhs[1]
        return result

    monkeypatch.setattr(type(covariance), "apply", roundoff_apply)

    products = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="diagonal",
    )

    assert products.native_observation_covariance.values[1] == 0.0
    assert products.native_observation_covariance.attrs["diagonal_nonnegative"] is True


def test_lazy_sensitivity_is_materialized_at_explicit_eager_boundary() -> None:
    """The product boundary accepts lazy H and returns eager products."""
    covariance, basis_operator, h, _ = _problem()
    lazy_h = h.chunk({"observation": 1})
    basis_prolongation = to_dense(
        basis_operator.native_prolongation(h, native_dims=covariance.native_dims)
    ).compute()

    expected = _project(covariance, basis_operator, h)
    actual = project_native_covariance(
        covariance=covariance,
        basis_prolongation=basis_prolongation,
        state_dim="state",
        native_sensitivity=lazy_h,
        observation_dim="observation",
    )

    xr.testing.assert_allclose(actual.native_observation_covariance, expected.native_observation_covariance)


def test_lazy_custom_restriction_is_materialized_in_explicit_rhs_blocks() -> None:
    """Custom Dask Pi follows the explicit retained-state RHS block path."""
    covariance, basis_operator, h, _ = _problem()

    class LazyRestrictionStrategy:
        """Return a valid numerical projection with a deliberately lazy restriction."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Chunk the restriction to exercise the explicit eager boundary."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(
                valid.restriction.chunk({state_dim: 1}),
                "lazy_restriction",
            )

    expected = _project(covariance, basis_operator, h)
    actual = _project(
        covariance,
        basis_operator,
        h,
        strategy=LazyRestrictionStrategy(),
        observation_batch_size=1,
    )

    xr.testing.assert_allclose(actual.state_covariance, expected.state_covariance)
    xr.testing.assert_allclose(
        actual.observation_state_cross_covariance,
        expected.observation_state_cross_covariance,
    )


def test_single_chunk_lazy_restriction_is_materialized_once() -> None:
    """A full-state Dask Pi producer executes once despite state RHS blocking."""
    covariance, basis_operator, h, _ = _problem()
    executions = 0

    class SingleChunkRestrictionStrategy:
        """Wrap a valid restriction in one delayed full-state chunk."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a delayed Pi whose producer records each execution."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            values = np.asarray(valid.restriction.values)

            @delayed
            def produce_restriction() -> np.ndarray:
                """Return the full restriction and record graph execution."""
                nonlocal executions
                executions += 1
                return values

            lazy_values = da.from_delayed(
                produce_restriction(),
                shape=values.shape,
                dtype=values.dtype,
            )
            restriction = xr.DataArray(
                lazy_values,
                dims=valid.restriction.dims,
                coords=valid.restriction.coords,
                attrs=valid.restriction.attrs,
            )
            return RetainedProjection(restriction, "single_chunk")

    _project(
        covariance,
        basis_operator,
        h,
        strategy=SingleChunkRestrictionStrategy(),
        observation_batch_size=1,
    )

    assert executions == 1


def test_product_units_follow_dimensionless_scaling_contract() -> None:
    """State arrays are dimensionless while H products retain or square H units."""
    covariance, basis_operator, h, _ = _problem()
    products = _project(covariance, basis_operator, h.assign_attrs(units="ppt"))

    for array in (
        products.restriction,
        products.prolongation,
        products.state_covariance,
    ):
        assert array.attrs["units"] == "1"
    assert products.effective_observation_operator.attrs["units"] == "ppt"
    assert products.observation_state_cross_covariance.attrs["units"] == "ppt"
    assert products.native_observation_covariance.attrs["units"] == "(ppt)^2"


@pytest.mark.parametrize("multiindex", [False, True], ids=["index", "multiindex"])
def test_square_products_preserve_typed_row_and_column_indexes(multiindex: bool) -> None:
    """Square product axes carry symmetric typed Index or MultiIndex labels."""
    covariance, basis_operator, h, _ = _problem()
    if multiindex:
        index = pd.MultiIndex.from_arrays(
            [["MHD", "TAC", "RGL"], pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])],
            names=("site", "time"),
        )
        h = h.drop_indexes("observation").drop_vars("observation")
        h = h.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "observation"))
        h.coords["site"].attrs["semantic_role"] = "station"
        h.coords["time"].attrs["semantic_role"] = "sample_time"
    else:
        index = pd.Index(["MHD", 7, ("RGL", 1)], dtype=object, name="observation")
        h = h.assign_coords(observation=index)

    matrix = _project(covariance, basis_operator, h).native_observation_covariance
    row_index = matrix.indexes["observation"]
    column_index = matrix.indexes["observation_cov"]

    assert isinstance(column_index, type(row_index))
    assert list(column_index) == list(row_index)
    assert matrix.sel(observation=row_index[0], observation_cov=column_index[0]).ndim == 0
    if multiindex:
        assert matrix.coords["site"].attrs["semantic_role"] == "station"
        assert matrix.coords["site_cov"].attrs["semantic_role"] == "station"
        assert matrix.coords["time"].attrs["semantic_role"] == "sample_time"
        assert matrix.coords["time_cov"].attrs["semantic_role"] == "sample_time"


def test_retained_state_multiindex_survives_the_eager_boundary() -> None:
    """Payload materialization preserves retained-state index structure."""
    covariance, basis_operator, h, _ = _problem()
    prolongation = basis_operator.native_prolongation(h, native_dims=covariance.native_dims)
    state_index = pd.MultiIndex.from_tuples(
        [("z-source", 7), ("a-source", 2)],
        names=("source", "region"),
    )
    state_coordinates = {
        str(name): coordinate
        for name, coordinate in prolongation.coords.items()
        if "state" not in coordinate.dims
    }
    prolongation = xr.DataArray(
        prolongation.data,
        dims=prolongation.dims,
        coords=state_coordinates,
        attrs=prolongation.attrs,
    ).assign_coords(xr.Coordinates.from_pandas_multiindex(state_index, "state"))

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_sensitivity=h,
        observation_dim="observation",
    )

    assert products.restriction.indexes["state"].equals(state_index)
    assert products.prolongation.indexes["state"].equals(state_index)
    assert products.state_covariance.indexes["state"].equals(state_index)
    assert list(products.state_covariance.indexes["state_cov"]) == list(state_index)


def test_product_kernel_avoids_throwaway_covariance_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covariance applies only to one Pi block and two observation RHS blocks."""
    covariance, basis_operator, h, _ = _problem()
    original_apply = type(covariance).apply
    calls = 0

    def counted_apply(self, rhs):
        """Count each real covariance application before delegating."""
        nonlocal calls
        calls += 1
        return original_apply(self, rhs)

    monkeypatch.setattr(type(covariance), "apply", counted_apply)

    _project(covariance, basis_operator, h, observation_batch_size=2)

    assert calls == 3


def test_projection_strategy_protocol_accepts_structural_implementations() -> None:
    """A compatible strategy is accepted without inheriting from the public protocol."""
    covariance, basis_operator, h, _ = _problem()

    class DelegatingStrategy:
        """Delegate projection construction while exposing a distinct strategy name."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a coherent projection using the established bucket strategy."""
            projection = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(projection.restriction, "delegating_strategy")

    strategy: RetainedProjectionStrategy = DelegatingStrategy()

    expected = _project(covariance, basis_operator, h)
    products = _project(covariance, basis_operator, h, strategy=strategy)

    assert products.strategy == "delegating_strategy"
    xr.testing.assert_allclose(products.prolongation, expected.prolongation)
    xr.testing.assert_allclose(products.state_covariance, expected.state_covariance)


@pytest.mark.parametrize("coordinate_change", ["mismatch", "reordered", "empty-intersection"])
def test_custom_projection_requires_exact_native_coordinates(
    coordinate_change: str,
) -> None:
    """Custom Pi native labels must match before product alignment or contraction."""
    covariance, basis_operator, h, _ = _problem()

    class RelabelledStrategy:
        """Relabel Pi to exercise strict native-coordinate checks."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return numeric Pi with deliberately invalid latitude labels."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            latitude = valid.restriction.coords["lat"].values
            if coordinate_change == "reordered":
                invalid_latitude = latitude[::-1]
            elif coordinate_change == "empty-intersection":
                invalid_latitude = latitude + 1000.0
            else:
                invalid_latitude = latitude + 0.25
            restriction = valid.restriction.assign_coords(lat=invalid_latitude)
            return RetainedProjection(restriction, "relabeled")

    with pytest.raises(ValueError, match=r"join='exact'|align"):
        _project(covariance, basis_operator, h, strategy=RelabelledStrategy())


def test_projection_strategy_must_return_full_rank_restriction() -> None:
    """A zero authoritative Pi is rejected because derived C_alpha is singular."""
    covariance, basis_operator, h, _ = _problem()

    class SingularStrategy:
        """Return a zero restriction with the expected dimensions."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately singular retained projection."""
            return RetainedProjection(
                xr.zeros_like(basis_prolongation.transpose(state_dim, *native_dims)),
                "singular",
            )

    with pytest.raises(ValueError, match="empty retained state|positive definite|full rank"):
        _project(covariance, basis_operator, h, strategy=SingularStrategy())


def test_projection_strategy_restriction_must_be_real() -> None:
    """A dynamic strategy cannot silently discard an imaginary component."""
    covariance, basis_operator, h, _ = _problem()

    class ComplexStrategy:
        """Return a complex restriction at the dynamic strategy seam."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Add an imaginary component to a valid restriction."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(valid.restriction.astype(complex) + 1j, "complex")

    with pytest.raises(ValueError, match="finite real"):
        _project(covariance, basis_operator, h, strategy=ComplexStrategy())


def test_redundant_bucket_states_are_rejected() -> None:
    """Linearly dependent retained coordinates fail instead of being pseudoinverted."""
    covariance, basis_operator, _, _ = _problem()
    prolongation = to_dense(basis_operator.basis_matrix).compute()
    redundant = xr.concat(
        [
            prolongation.sel(state=0),
            prolongation.sel(state=0),
        ],
        dim=xr.IndexVariable("state", ["west", "west-copy"]),
    ).transpose("lat", "lon", "state")

    with pytest.raises(ValueError, match="positive definite|redundant|rank"):
        PreserveBucketProlongation().projection(
            covariance,
            redundant,
            native_dims=covariance.native_dims,
            state_dim="state",
        )


def test_products_require_exact_native_sensitivity_alignment() -> None:
    """Product construction does not silently intersect a different H grid."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(ValueError, match=r"join='exact'|align|lon"):
        _project(covariance, basis_operator, h.assign_coords(lon=h.lon[::-1]))


def test_class_blocked_products_match_dense_oracle() -> None:
    """The compatible restriction and all products remain coherent with class masks."""
    covariance, basis_operator, h, dense_unblocked = _problem()
    labels = xr.DataArray(
        [["land", "sea"], ["land", "sea"], ["sea", "sea"]],
        dims=("lat", "lon"),
        coords={"lat": covariance.latitude, "lon": covariance.longitude},
    )
    covariance = SeparableExponentialCovariance(
        covariance.latitude,
        covariance.longitude,
        sigma=covariance.sigma,
        correlation_length=covariance.correlation_length,
        class_labels=labels,
    )
    flat_labels = labels.values.reshape(-1)
    dense_b = dense_unblocked * (flat_labels[:, None] == flat_labels[None, :])
    u, expected_pi, expected_c = _dense_operators(basis_operator, dense_b)
    h_matrix = h.values.reshape(3, 6)

    products = _project(covariance, basis_operator, h)

    np.testing.assert_allclose(products.restriction.values.reshape(2, 6), expected_pi, atol=2e-10)
    np.testing.assert_allclose(products.state_covariance, expected_c, atol=2e-10)
    np.testing.assert_allclose(products.effective_observation_operator, h_matrix @ u, atol=2e-10)
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        h_matrix @ dense_b @ expected_pi.T,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        products.native_observation_covariance,
        h_matrix @ dense_b @ h_matrix.T,
        atol=2e-10,
    )
