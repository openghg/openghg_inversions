"""Test labelled covariance projection, eager batching, and numerical contracts."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
    assert projection.prolongation.dims == ("lat", "lon", "state")
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


def test_dense_batching_preallocates_instead_of_concatenating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense observation batching avoids the former xr.concat block-assembly path."""
    covariance, basis_operator, h, _ = _problem()

    def reject_concat(*args: object, **kwargs: object) -> None:
        """Fail if the former all-block concatenation path is used."""
        raise AssertionError(f"unexpected xr.concat call: {args!r}, {kwargs!r}")

    monkeypatch.setattr(xr, "concat", reject_concat)

    products = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="dense",
        observation_batch_size=1,
    )

    assert products.native_observation_covariance.shape == (3, 3)


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


@pytest.mark.parametrize(
    "invalid_batch_size",
    [True, "2", 1.9],
    ids=["boolean", "string", "non-integral-float"],
)
def test_observation_batch_size_requires_integral_non_boolean(invalid_batch_size: object) -> None:
    """The RHS block size rejects values that merely coerce to an integer."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises((TypeError, ValueError), match="integer|integral|Boolean"):
        _project(
            covariance,
            basis_operator,
            h,
            observation_batch_size=invalid_batch_size,
        )


def test_observation_batch_size_accepts_numpy_integral() -> None:
    """NumPy integer scalars satisfy the public integral block-size contract."""
    covariance, basis_operator, h, _ = _problem()

    products = _project(
        covariance,
        basis_operator,
        h,
        observation_batch_size=np.int64(2),
    )

    assert products.native_observation_covariance.shape == (3, 3)


def test_lazy_sensitivity_is_rejected_at_explicit_materialization_boundary() -> None:
    """The eager kernel rejects Dask H instead of silently computing its full graph."""
    covariance, basis_operator, h, _ = _problem()
    lazy_h = h.chunk({"observation": 1})
    basis_prolongation = to_dense(
        basis_operator.native_prolongation(h, native_dims=covariance.native_dims)
    ).compute()

    with pytest.raises(ValueError, match="upstream materialization|lazy|Dask"):
        project_native_covariance(
            covariance=covariance,
            basis_prolongation=basis_prolongation,
            state_dim="state",
            native_sensitivity=lazy_h,
            observation_dim="observation",
        )


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
                valid.prolongation,
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
    else:
        index = pd.Index(["MHD", 7, ("RGL", 1)], dtype=object, name="observation")
        h = h.assign_coords(observation=index)

    matrix = _project(covariance, basis_operator, h).native_observation_covariance
    row_index = matrix.indexes["observation"]
    column_index = matrix.indexes["observation_cov"]

    assert isinstance(column_index, type(row_index))
    assert list(column_index) == list(row_index)
    assert matrix.sel(observation=row_index[0], observation_cov=column_index[0]).ndim == 0


def test_product_kernel_avoids_throwaway_covariance_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covariance application occurs only for the two real observation RHS blocks."""
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

    assert calls == 2


def test_each_dense_matrix_diagnostic_uses_one_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State and observation covariance diagnostics each compute eigenvalues once."""
    covariance, basis_operator, h, _ = _problem()
    original_eigvalsh = np.linalg.eigvalsh
    calls = 0

    def counted_eigvalsh(matrix):
        """Count symmetric eigensolver calls before delegating."""
        nonlocal calls
        calls += 1
        return original_eigvalsh(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", counted_eigvalsh)

    products = _project(covariance, basis_operator, h)

    assert calls == 2
    assert "minimum_eigenvalue" in products.state_covariance.attrs
    assert "minimum_eigenvalue" in products.native_observation_covariance.attrs


def test_projection_strategy_must_return_covariance_natural_prolongation() -> None:
    """Custom strategies cannot return a Pi/U pair that violates B Pi.T = U C_alpha."""
    covariance, basis_operator, h, _ = _problem()

    class IncoherentStrategy:
        """Perturb an otherwise compatible prolongation to violate the invariant."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately incoherent restriction/prolongation pair."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(valid.restriction, 2.0 * valid.prolongation, "incoherent")

    for sigma in (covariance.sigma, 1e-8):
        scaled = SeparableExponentialCovariance(
            covariance.latitude,
            covariance.longitude,
            sigma=sigma,
            correlation_length=covariance.correlation_length,
        )
        with pytest.raises(ValueError, match="incoherent|B Pi"):
            _project(scaled, basis_operator, h, strategy=IncoherentStrategy())


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
            return RetainedProjection(
                projection.restriction,
                projection.prolongation,
                "delegating_strategy",
            )

    strategy: RetainedProjectionStrategy = DelegatingStrategy()

    products = _project(covariance, basis_operator, h, strategy=strategy)

    assert products.strategy == "delegating_strategy"


def test_projection_strategy_names_must_be_nonempty() -> None:
    """Empty built-in and custom strategy identifiers are rejected."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(ValueError, match="non-empty"):
        PreserveBucketProlongation(name="")

    class EmptyNameStrategy:
        """Delegate a valid projection but remove its stable identifier."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a coherent projection with an invalid empty name."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(valid.restriction, valid.prolongation, "")

    with pytest.raises(ValueError, match="non-empty"):
        _project(covariance, basis_operator, h, strategy=EmptyNameStrategy())


@pytest.mark.parametrize("role", ["restriction", "prolongation"])
def test_projection_strategy_rejects_complex_arrays(role: str) -> None:
    """Projection products use real covariance algebra and reject complex strategy arrays."""
    covariance, basis_operator, h, _ = _problem()

    class ComplexStrategy:
        """Add an imaginary component to one otherwise valid projection array."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately complex restriction or prolongation."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            restriction = valid.restriction
            prolongation = valid.prolongation
            if role == "restriction":
                restriction = restriction.astype(complex) + 1j
            else:
                prolongation = prolongation.astype(complex) + 1j
            return RetainedProjection(restriction, prolongation, "complex")

    with pytest.raises(ValueError, match="finite real"):
        _project(covariance, basis_operator, h, strategy=ComplexStrategy())


@pytest.mark.parametrize("role", ["restriction", "prolongation"])
@pytest.mark.parametrize("coordinate_change", ["mismatch", "reordered", "empty-intersection"])
def test_custom_projection_requires_exact_native_coordinates(
    role: str,
    coordinate_change: str,
) -> None:
    """Custom Pi and U native labels must match before product alignment or contraction."""
    covariance, basis_operator, h, _ = _problem()

    class RelabelledStrategy:
        """Relabel one projection array to exercise strict native-coordinate checks."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a coherent numeric pair with deliberately invalid latitude labels."""
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
            restriction = valid.restriction
            prolongation = valid.prolongation
            if role == "restriction":
                restriction = restriction.assign_coords(lat=invalid_latitude)
            else:
                prolongation = prolongation.assign_coords(lat=invalid_latitude)
            return RetainedProjection(restriction, prolongation, "relabeled")

    with pytest.raises(ValueError, match=r"native coordinate 'lat'.*exactly match"):
        _project(covariance, basis_operator, h, strategy=RelabelledStrategy())


def test_projection_strategy_must_return_full_rank_restriction() -> None:
    """A formally compatible zero Pi/U pair is rejected because C_alpha is singular."""
    covariance, basis_operator, h, _ = _problem()

    class SingularStrategy:
        """Return a zero restriction and prolongation with the expected dimensions."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately singular retained projection."""
            return RetainedProjection(
                xr.zeros_like(basis_prolongation.transpose(state_dim, *native_dims)),
                xr.zeros_like(basis_prolongation),
                "singular",
            )

    with pytest.raises(ValueError, match="empty retained state|positive definite|full rank"):
        _project(covariance, basis_operator, h, strategy=SingularStrategy())


def test_redundant_bucket_states_are_rejected() -> None:
    """Linearly dependent retained coordinates fail instead of being pseudoinverted."""
    covariance, basis_operator, _, _ = _problem()
    prolongation = basis_operator.basis_matrix.compute()
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


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        pytest.param(
            lambda h: h.assign_coords(lon=h.lon[::-1]),
            "coordinate|align|longitude|lon",
            id="misordered-native-coordinate",
        ),
        pytest.param(
            lambda h: h.where(h.observation != "TAC-1"),
            "finite|NaN",
            id="non-finite-sensitivity",
        ),
        pytest.param(
            lambda h: h.astype(complex) + 1j,
            "finite real",
            id="complex-sensitivity",
        ),
    ],
)
def test_products_reject_invalid_native_sensitivity(transform, message: str) -> None:
    """Product construction rejects misaligned or non-finite native sensitivity."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(ValueError, match=message):
        _project(covariance, basis_operator, transform(h))


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
