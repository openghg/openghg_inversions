"""Test the supplied-Pi-first retained projection strategy and its Pi builder."""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from openghg_inversions.basis.covariance_products import (
    PreserveBucketProlongation,
    SuppliedRestriction,
    _apply_restriction_blocks,
    _prepare_projection,
    absolute_weighted_regional_mean_restriction,
    project_native_covariance,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance

SOURCES = ("GPP", "FF")
SOURCE_DIM = "native_source"
NATIVE_DIMS = (SOURCE_DIM, "lat", "lon")
LATITUDE = np.array([50.0, 51.0, 52.5])
LONGITUDE = np.array([-2.0, -1.0, 0.5, 1.5])
# Source-specific regions; label 0 is an active region and -1 marks an inactive
# cell with no retained weight.
REGIONS = {
    "GPP": np.array([[0, 0, 2, 2], [0, 3, 3, 2], [3, 3, -1, 2]]),
    "FF": np.array([[4, 4, 4, 7], [4, 4, 7, 7], [-1, 7, 7, 7]]),
}
CORRELATION_LENGTHS = {"GPP": 1.3, "FF": 0.8}
SIGMAS = {"GPP": 0.7, "FF": 0.4}


def _state_index() -> pd.MultiIndex:
    """Return retained labels: source order, then active regions numbered from zero."""
    return pd.MultiIndex.from_tuples(
        [(source, position) for source in SOURCES for position in range(np.unique(REGIONS[source][REGIONS[source] >= 0]).size)],
        names=("source", "region_in_source"),
    )


def _labelled(values, dims: tuple[str, ...], *, observation: bool = False) -> xr.DataArray:
    """Attach the shared native and observation coordinates."""
    coords: dict[str, object] = {}
    if SOURCE_DIM in dims:
        coords[SOURCE_DIM] = list(SOURCES)
    if "lat" in dims:
        coords["lat"] = LATITUDE
    if "lon" in dims:
        coords["lon"] = LONGITUDE
    if observation:
        coords["observation"] = [f"site-{index}" for index in range(values.shape[0])]
    return xr.DataArray(values, dims=dims, coords=coords)


def _regions() -> dict[str, xr.DataArray]:
    """Return labelled integer region maps for each source."""
    return {source: _labelled(labels, ("lat", "lon")) for source, labels in REGIONS.items()}


def _flux_weights() -> xr.DataArray:
    """Return signed ``q = prior flux * cell area``; GPP is mostly negative."""
    rng = np.random.default_rng(31)
    area = np.cos(np.deg2rad(LATITUDE))[:, None] * np.ones((1, LONGITUDE.size))
    flux = np.stack([rng.normal(-0.3, 1.0, size=area.shape), rng.normal(2.0, 1.0, size=area.shape)])
    return _labelled(flux * area, NATIVE_DIMS).assign_attrs(units="kg s-1")


def _oracle_restriction(weights: np.ndarray) -> np.ndarray:
    """Directly evaluate the absolute-weighted regional mean as a dense matrix."""
    rows = []
    for source_index, source in enumerate(SOURCES):
        for region in np.unique(REGIONS[source]):
            if region < 0:
                continue
            row = np.zeros(weights.shape)
            mask = REGIONS[source] == region
            row[source_index] = np.abs(weights[source_index]) * mask
            rows.append(row.reshape(-1) / row.sum())
    return np.array(rows)


def _restriction() -> xr.DataArray:
    """Build the production-like source-blocked absolute-flux-weighted ``Pi``."""
    return absolute_weighted_regional_mean_restriction(_regions(), _flux_weights())


def _covariance(class_blocked: bool) -> tuple[IndependentSourceCovariance, np.ndarray]:
    """Return the labelled source-blocked covariance and its dense oracle."""
    latitude = xr.DataArray(LATITUDE, dims="lat", coords={"lat": LATITUDE})
    longitude = xr.DataArray(LONGITUDE, dims="lon", coords={"lon": LONGITUDE})
    class_values = np.array(
        [["land", "land", "sea", "sea"], ["land", "land", "land", "sea"], ["sea", "land", "sea", "sea"]]
    )
    class_labels = xr.DataArray(class_values, dims=("lat", "lon"), coords={"lat": LATITUDE, "lon": LONGITUDE})
    components = {}
    blocks = []
    for source in SOURCES:
        length = CORRELATION_LENGTHS[source]
        components[source] = SeparableExponentialCovariance(
            latitude,
            longitude,
            sigma=SIGMAS[source],
            correlation_length=length,
            class_labels=class_labels if class_blocked else None,
        )
        k_lat = np.exp(-np.abs(LATITUDE[:, None] - LATITUDE[None, :]) / length)
        k_lon = np.exp(-np.abs(LONGITUDE[:, None] - LONGITUDE[None, :]) / length)
        block = SIGMAS[source] ** 2 * np.kron(k_lat, k_lon)
        if class_blocked:
            flat = class_values.reshape(-1)
            block = block * (flat[:, None] == flat[None, :])
        blocks.append(block)
    cells = blocks[0].shape[0]
    dense = np.zeros((len(SOURCES) * cells, len(SOURCES) * cells))
    for index, block in enumerate(blocks):
        dense[index * cells : (index + 1) * cells, index * cells : (index + 1) * cells] = block
    return IndependentSourceCovariance(components, source_dim=SOURCE_DIM), dense


def _sensitivity() -> xr.DataArray:
    """Return a labelled native sensitivity with four observations."""
    rng = np.random.default_rng(1053)
    values = rng.normal(size=(4, len(SOURCES), LATITUDE.size, LONGITUDE.size))
    return _labelled(values, ("observation", *NATIVE_DIMS), observation=True).assign_attrs(units="ppm")


def _flat(array: xr.DataArray, leading: str) -> np.ndarray:
    """Flatten labelled ``(leading, *native)`` data to a dense matrix."""
    data = array.transpose(leading, *NATIVE_DIMS).data
    data = data.todense() if isinstance(data, sparse.SparseArray) else np.asarray(data)
    return data.reshape(data.shape[0], -1)


def _project(strategy, *, class_blocked: bool = False, **options):
    """Project the shared sensitivity with one strategy."""
    covariance, _ = _covariance(class_blocked)
    return project_native_covariance(
        covariance=covariance,
        strategy=strategy,
        state_dim="state",
        native_sensitivity=_sensitivity(),
        observation_dim="observation",
        **options,
    )


def _replace_values(restriction: xr.DataArray, edit) -> xr.DataArray:
    """Return a copy of ``Pi`` whose dense values are changed by ``edit``."""
    dense = restriction.data.todense()
    edit(dense)
    return restriction.copy(data=sparse.COO.from_numpy(dense))


def test_builder_matches_direct_absolute_weighted_regional_mean() -> None:
    """Rows are exact source-blocked regional means with inactive cells excluded."""
    weights = _flux_weights()
    restriction = _restriction()
    pi = _flat(restriction, "state")

    assert isinstance(restriction.data, sparse.COO)
    assert restriction.dims == ("state", *NATIVE_DIMS)
    assert restriction.indexes["state"].equals(_state_index())
    assert restriction.attrs["units"] == "1"
    np.testing.assert_allclose(pi, _oracle_restriction(weights.values), rtol=1e-15, atol=0.0)
    assert np.all(restriction.data.data > 0.0)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(pi @ np.ones(pi.shape[1]), np.ones(pi.shape[0]), atol=1e-15)
    inactive = np.concatenate([REGIONS[source].reshape(-1) < 0 for source in SOURCES])
    assert not pi[:, inactive].any()
    for row, (source, region) in enumerate(restriction.indexes["state"]):
        support = restriction.isel(state=row).data.todense() > 0.0
        other_sources = [index for index, label in enumerate(SOURCES) if label != source]
        assert not support[other_sources].any()
        active_labels = np.unique(REGIONS[source][REGIONS[source] >= 0])
        assert np.array_equal(support[SOURCES.index(source)], REGIONS[source] == active_labels[region])


def test_builder_state_labels_match_multisource_bucket_operator() -> None:
    """With every cell active, including label 0, retained labels match the operator's numbering."""
    regions = {source: labels.where(labels >= 0, 9) for source, labels in _regions().items()}
    operator = MultiSourceBucketBasisOperator(regions)

    restriction = absolute_weighted_regional_mean_restriction(regions, _flux_weights())

    assert restriction.indexes["state"].equals(operator.basis_matrix.indexes["state"])
    assert restriction.indexes["state"].names == operator.basis_matrix.indexes["state"].names


def test_builder_materializes_lazy_inputs() -> None:
    """Dask-backed region maps and weights give the same restriction."""
    regions = {source: labels.chunk({"lat": 1}) for source, labels in _regions().items()}

    lazy = absolute_weighted_regional_mean_restriction(regions, _flux_weights().chunk({"lon": 2}))

    xr.testing.assert_identical(lazy, _restriction())


@pytest.mark.parametrize(
    "problem", ["float-labels", "source-order", "unindexed-source", "grid", "zero-region", "non-finite"]
)
def test_builder_rejects_invalid_inputs(problem: str) -> None:
    """Labels must be integers on the exact grid, in source order, with usable weights."""
    regions = _regions()
    weights = _flux_weights()
    if problem == "float-labels":
        regions["FF"] = regions["FF"].astype(float)
        match = "integer region labels"
    elif problem == "source-order":
        regions = {"FF": regions["FF"], "GPP": regions["GPP"]}
        match = "labels must equal the regions keys"
    elif problem == "unindexed-source":
        weights = weights.drop_vars(SOURCE_DIM)
        match = "indexed native source coordinate"
    elif problem == "grid":
        regions["GPP"] = regions["GPP"].isel(lon=slice(None, None, -1))
        match = r"join='exact'|align"
    elif problem == "zero-region":
        weights = weights.where((weights[SOURCE_DIM] != "FF") | (regions["FF"] != 4), 0.0)
        match = "Region 4 of source 'FF' has zero total"
    else:
        weights = weights.where((weights.lat != LATITUDE[0]) | (weights.lon != LONGITUDE[0]))
        match = "finite real"

    with pytest.raises(ValueError, match=match):
        absolute_weighted_regional_mean_restriction(regions, weights)


@pytest.mark.parametrize("class_blocked", [False, True], ids=["separable", "class-blocked"])
@pytest.mark.parametrize("observation_covariance", ["dense", "diagonal"])
def test_supplied_restriction_products_match_dense_oracle(
    class_blocked: bool,
    observation_covariance: str,
) -> None:
    """Products follow the supplied-Pi-first equations used by verification-games."""
    _, dense_b = _covariance(class_blocked)
    restriction = _restriction()

    products = _project(
        SuppliedRestriction(restriction, require_positivity_preserving=True),
        class_blocked=class_blocked,
        observation_covariance=observation_covariance,
        observation_batch_size=3,
    )

    pi = _flat(restriction, "state")
    h = _flat(_sensitivity(), "observation")
    b_pi_t = dense_b @ pi.T
    c_alpha = pi @ b_pi_t
    u_star = np.linalg.solve(c_alpha, b_pi_t.T).T
    actual_u_star = _flat(products.prolongation.transpose("state", *NATIVE_DIMS), "state").T

    np.testing.assert_allclose(products.state_covariance, c_alpha, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(actual_u_star, u_star, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(pi @ actual_u_star, np.eye(pi.shape[0]), atol=1e-11)
    np.testing.assert_allclose(b_pi_t, actual_u_star @ c_alpha, atol=1e-12)
    np.testing.assert_allclose(products.effective_observation_operator, h @ u_star, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(products.observation_state_cross_covariance, h @ b_pi_t, atol=1e-12)
    expected_hbht = h @ dense_b @ h.T
    if observation_covariance == "diagonal":
        expected_hbht = np.diag(expected_hbht)
    np.testing.assert_allclose(products.native_observation_covariance, expected_hbht, atol=1e-12)
    assert products.strategy == "supplied_restriction"
    assert products.state_covariance.indexes["state"].equals(_state_index())
    assert products.effective_observation_operator.indexes["state"].equals(_state_index())


def test_supplied_restriction_keeps_labels_provenance_and_sparse_storage() -> None:
    """The products carry the supplied sparse Pi, its labels, and admissibility record."""
    restriction = _restriction()
    strategy = SuppliedRestriction(restriction, require_positivity_preserving=True)

    result = _project(strategy).restriction

    # Borrowed, not copied: the strategy reads the caller's array at projection time.
    assert strategy.restriction is restriction
    assert isinstance(result.data, sparse.COO)
    assert result.dims == ("state", *NATIVE_DIMS)
    assert result.indexes["state"].equals(_state_index())
    assert result.attrs["policy"] == "absolute-weighted regional mean"
    assert result.attrs["definition"] == restriction.attrs["definition"]
    assert result.attrs["restriction_source"] == "supplied"
    assert result.attrs["positivity_preserving_required"] is True
    assert result.attrs["positivity_preserving"] is True
    assert result.attrs["positive_row_sums"] is True
    assert result.attrs["nonnegative"] is True
    assert result.attrs["row_normalized"] is True
    assert result.attrs["units"] == "1"
    assert "restriction_source" not in restriction.attrs


def test_signed_restriction_is_refused_only_when_positivity_is_required() -> None:
    """Only a nonnegative Pi guarantees positive retained scalings."""

    def make_signed(values: np.ndarray) -> None:
        """Flip one weight of the second retained row."""
        values[1, 0, 0, 2] = -1e-18

    restriction = _replace_values(_restriction(), make_signed)

    with pytest.raises(ValueError, match=r"row \('GPP', 1\) has weight -1\.000e-18"):
        _project(SuppliedRestriction(restriction, require_positivity_preserving=True))
    products = _project(SuppliedRestriction(restriction))

    assert products.restriction.attrs["positivity_preserving"] is False
    assert products.restriction.attrs["positivity_preserving_required"] is False
    assert products.restriction.attrs["nonnegative"] is False


def test_zero_row_is_refused_when_positivity_is_required() -> None:
    """A positivity-preserving restriction needs positive weight in every row."""

    def zero_row(values: np.ndarray) -> None:
        """Remove every weight from one retained row."""
        values[1] = 0.0

    strategy = SuppliedRestriction(
        _replace_values(_restriction(), zero_row),
        require_positivity_preserving=True,
    )

    with pytest.raises(ValueError, match=r"positive weight in every restriction row; row \('GPP', 1\)"):
        strategy.projection(_covariance(False)[0], native_dims=NATIVE_DIMS, state_dim="state")


def test_positivity_preserving_restriction_permits_a_signed_conditional_lift() -> None:
    """The covariance-natural lift may be signed while Pi preserves positivity."""
    products = _project(SuppliedRestriction(_restriction(), require_positivity_preserving=True))

    assert float(products.prolongation.min()) < 0.0


@pytest.mark.parametrize("change", ["duplicate-state", "missing-dim", "reordered-native", "non-finite", "fill-value"])
def test_supplied_restriction_rejects_invalid_labels_and_values(change: str) -> None:
    """Supplied Pi must use unique labels, exact dimensions and grid, and finite real values."""
    restriction = _restriction()
    if change == "duplicate-state":
        restriction = restriction.isel(state=[0, 0, 1])
        match = "must be unique"
    elif change == "missing-dim":
        restriction = restriction.isel({SOURCE_DIM: 0}, drop=True)
        match = "Dimensions"
    elif change == "reordered-native":
        restriction = restriction.isel(lon=slice(None, None, -1))
        match = r"join='exact'|align"
    elif change == "non-finite":
        restriction = _replace_values(restriction, lambda values: values.__setitem__((0, 0, 0, 0), np.nan))
        match = "finite real"
    else:
        restriction = restriction.copy(data=sparse.full_like(restriction.data, 0.5))
        match = "zero fill value"

    with pytest.raises(ValueError, match=match):
        _project(SuppliedRestriction(restriction))


def test_redundant_supplied_restriction_is_rejected() -> None:
    """Duplicate retained rows fail the full-row-rank C_alpha factorization."""

    def duplicate_row(values: np.ndarray) -> None:
        """Copy the first retained row into the second."""
        values[1] = values[0]

    with pytest.raises(ValueError, match="positive definite|ill-conditioned"):
        _project(SuppliedRestriction(_replace_values(_restriction(), duplicate_row)))


def test_lazy_dense_supplied_restriction_matches_eager_sparse() -> None:
    """Storage and execution backend do not change the supplied-Pi products."""
    restriction = _restriction()
    lazy_dense = restriction.copy(data=restriction.data.todense()).chunk({"state": 2})

    expected = _project(SuppliedRestriction(restriction), class_blocked=True)
    actual = _project(SuppliedRestriction(lazy_dense), class_blocked=True)

    assert isinstance(actual.restriction.data, np.ndarray)
    xr.testing.assert_allclose(actual.state_covariance, expected.state_covariance)
    xr.testing.assert_allclose(actual.prolongation, expected.prolongation)


def test_bucket_strategy_accepts_sparse_and_lazy_prolongation() -> None:
    """The bucket strategy materializes and densifies the U it owns."""
    state_index = _state_index()
    values = np.zeros((len(SOURCES), LATITUDE.size, LONGITUDE.size, len(state_index)))
    for column, (source, region) in enumerate(state_index):
        active_labels = np.unique(REGIONS[source][REGIONS[source] >= 0])
        values[SOURCES.index(source), ..., column] = REGIONS[source] == active_labels[region]
    prolongation = _labelled(values, (*NATIVE_DIMS, "state")).assign_coords(
        xr.Coordinates.from_pandas_multiindex(state_index, "state")
    )

    expected = _project(PreserveBucketProlongation(prolongation))
    sparse_u = prolongation.copy(data=sparse.COO.from_numpy(values))
    for candidate in (sparse_u, sparse_u.chunk({"lat": 1})):
        actual = _project(PreserveBucketProlongation(candidate))
        xr.testing.assert_allclose(actual.state_covariance, expected.state_covariance)
        xr.testing.assert_allclose(actual.prolongation, expected.prolongation)


def test_supplied_restriction_moments_and_monte_carlo_close() -> None:
    """Pi m and C_alpha are the declared moments of alpha = Pi x."""
    covariance, dense_b = _covariance(class_blocked=True)
    restriction = _restriction()
    rng = np.random.default_rng(40)
    mean_values = 1.0 + 0.1 * rng.normal(size=(len(SOURCES), LATITUDE.size, LONGITUDE.size))

    reduction = reduce_native_gaussian(
        covariance=covariance,
        strategy=SuppliedRestriction(restriction, require_positivity_preserving=True),
        state_dim="state",
        native_mean=_labelled(mean_values, NATIVE_DIMS),
        native_sensitivity=_sensitivity(),
        observation_dim="observation",
    )

    pi = _flat(restriction, "state")
    m = mean_values.reshape(-1)
    np.testing.assert_allclose(reduction.retained_mean, pi @ m, atol=1e-14)
    np.testing.assert_allclose(reduction.retained_covariance, pi @ dense_b @ pi.T, atol=1e-13)
    assert reduction.projection_strategy == "supplied_restriction"

    samples = rng.multivariate_normal(m, dense_b, size=200_000, method="cholesky")
    alpha = samples @ pi.T
    np.testing.assert_allclose(alpha.mean(axis=0), reduction.retained_mean, atol=5e-3)
    np.testing.assert_allclose(np.cov(alpha, rowvar=False), reduction.retained_covariance, atol=5e-3)


def test_restriction_rhs_blocks_are_densified_one_block_at_a_time() -> None:
    """A sparse Pi reaches the covariance action only as bounded dense RHS blocks."""
    covariance, _ = _covariance(class_blocked=False)
    seen: list[tuple[type, int]] = []

    class RecordingCovariance:
        """Delegate to the source covariance while recording RHS payloads."""

        native_dims = covariance.native_dims

        def apply(self, rhs: xr.DataArray) -> xr.DataArray:
            """Record the RHS backend and column count before delegating."""
            seen.append((type(rhs.data), rhs.sizes["state_cov"]))
            return covariance.apply(rhs)

    _apply_restriction_blocks(
        RecordingCovariance(),  # type: ignore[arg-type]
        _restriction(),
        native_dims=NATIVE_DIMS,
        state_dim="state",
        column_dim="state_cov",
        rhs_block_size=2,
    )

    assert seen == [(np.ndarray, 2), (np.ndarray, 2), (np.ndarray, 1)]


def test_production_shape_restriction_stays_sparse_with_bounded_memory() -> None:
    """The 361 x 96,768 production restriction is built and validated without densifying.

    A dense float64 restriction at this shape needs about 279 MB. The sparse
    path stores at most one weight per native source cell, so building,
    validating, and preparing it stay well below that bound.
    """
    source_rows = {"GPP": 115, "TER": 115, "FF": 74, "ocean": 57}
    n_lat, n_lon = 168, 144
    assert len(source_rows) * n_lat * n_lon == 96_768
    rng = np.random.default_rng(361)
    coords = {"lat": np.linspace(-80.0, 80.0, n_lat), "lon": np.linspace(-180.0, 179.0, n_lon)}

    tracemalloc.start()
    regions = {}
    for source, count in source_rows.items():
        # Region labels 0..count-1, with about 45% of cells inactive (label -1).
        labels = rng.integers(0, count, size=(n_lat, n_lon))
        labels.reshape(-1)[:count] = np.arange(count)
        labels.reshape(-1)[count:][rng.random(n_lat * n_lon - count) < 0.45] = -1
        regions[source] = xr.DataArray(labels, dims=("lat", "lon"), coords=coords)
    weights = xr.DataArray(
        rng.normal(size=(len(source_rows), n_lat, n_lon)),
        dims=NATIVE_DIMS,
        coords={SOURCE_DIM: list(source_rows), **coords},
    )
    restriction = absolute_weighted_regional_mean_restriction(regions, weights)
    projection = SuppliedRestriction(restriction, require_positivity_preserving=True).projection(
        None,  # type: ignore[arg-type]
        native_dims=NATIVE_DIMS,
        state_dim="state",
    )
    prepared = _prepare_projection(
        projection,
        native_dims=NATIVE_DIMS,
        state_dim="state",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    shape = (361, *weights.shape)
    dense_bytes = np.prod(shape) * np.dtype(np.float64).itemsize
    assert dense_bytes > 270e6
    assert prepared.restriction.shape == shape
    assert isinstance(prepared.restriction.data, sparse.COO)
    assert 40_000 < prepared.restriction.data.nnz < 96_768
    assert prepared.restriction.attrs["row_normalized"] is True
    assert prepared.restriction.attrs["positivity_preserving"] is True
    assert peak < 0.1 * dense_bytes
