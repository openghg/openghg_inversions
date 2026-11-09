"""Test the supplied-Pi-first retained projection strategy."""

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
    project_native_covariance,
)
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance

SOURCES = ("GPP", "FF")
SOURCE_DIM = "native_source"
NATIVE_DIMS = (SOURCE_DIM, "lat", "lon")
LATITUDE = np.array([50.0, 51.0, 52.5])
LONGITUDE = np.array([-2.0, -1.0, 0.5, 1.5])
# Source-specific regions: GPP has three regions and FF two.
REGIONS = {
    "GPP": np.array([[1, 1, 2, 2], [1, 3, 3, 2], [3, 3, 3, 2]]),
    "FF": np.array([[1, 1, 1, 2], [1, 1, 2, 2], [2, 2, 2, 2]]),
}
CORRELATION_LENGTHS = {"GPP": 1.3, "FF": 0.8}
SIGMAS = {"GPP": 0.7, "FF": 0.4}


def _state_index() -> pd.MultiIndex:
    """Return retained (source, region) labels in source-block order."""
    return pd.MultiIndex.from_tuples(
        [(source, int(region)) for source in SOURCES for region in np.unique(REGIONS[source])],
        names=("source", "region"),
    )


def _labelled(values, dims: tuple[str, ...], *, state: bool = False, observation: bool = False):
    """Attach the shared native, state, and observation coordinates."""
    coords: dict[str, object] = {}
    if SOURCE_DIM in dims:
        coords[SOURCE_DIM] = list(SOURCES)
    if "lat" in dims:
        coords["lat"] = LATITUDE
    if "lon" in dims:
        coords["lon"] = LONGITUDE
    if observation:
        coords["observation"] = [f"site-{index}" for index in range(values.shape[0])]
    array = xr.DataArray(values, dims=dims, coords=coords)
    if state:
        array = array.assign_coords(xr.Coordinates.from_pandas_multiindex(_state_index(), "state"))
    return array


def _covariance(class_blocked: bool) -> tuple[IndependentSourceCovariance, np.ndarray]:
    """Return the labelled source-blocked covariance and its dense oracle."""
    latitude = xr.DataArray(LATITUDE, dims="lat", coords={"lat": LATITUDE})
    longitude = xr.DataArray(LONGITUDE, dims="lon", coords={"lon": LONGITUDE})
    class_values = np.array(
        [["land", "land", "sea", "sea"], ["land", "land", "land", "sea"], ["sea", "land", "sea", "sea"]]
    )
    class_labels = xr.DataArray(
        class_values,
        dims=("lat", "lon"),
        coords={"lat": LATITUDE, "lon": LONGITUDE},
    )
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


def _absolute_flux_weighted_restriction() -> xr.DataArray:
    """Build a production-like sparse source-blocked absolute-flux-weighted mean.

    ``q_si = prior_flux_si * area_i`` is signed; each row averages native scalings
    within one source and region with weights ``|q_si| / sum_{j in r} |q_sj|``.
    """
    rng = np.random.default_rng(31)
    area = np.cos(np.deg2rad(LATITUDE))[:, None] * np.ones((1, LONGITUDE.size))
    state_index = _state_index()
    shape = (len(state_index), len(SOURCES), LATITUDE.size, LONGITUDE.size)
    values = np.zeros(shape)
    for row, (source, region) in enumerate(state_index):
        source_index = SOURCES.index(source)
        flux = rng.normal(0.0, 1.0, size=area.shape) + (0.3 if source == "GPP" else 2.0)
        weights = np.abs(flux * area) * (REGIONS[source] == region)
        values[row, source_index] = weights / weights.sum()
    return _labelled(sparse.COO.from_numpy(values), ("state", *NATIVE_DIMS), state=True).assign_attrs(
        policy="absolute-flux-weighted regional mean"
    )


def _bucket_prolongation() -> xr.DataArray:
    """Return the source-blocked bucket indicator ``U`` carrying retained labels."""
    state_index = _state_index()
    values = np.zeros((len(SOURCES), LATITUDE.size, LONGITUDE.size, len(state_index)))
    for column, (source, region) in enumerate(state_index):
        values[SOURCES.index(source), ..., column] = REGIONS[source] == region
    return _labelled(values, (*NATIVE_DIMS, "state"), state=True)


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


def test_absolute_flux_weighted_fixture_is_a_source_blocked_mean() -> None:
    """The fixture rows are nonnegative, single-source, and sum to one."""
    restriction = _absolute_flux_weighted_restriction()
    pi = _flat(restriction, "state")

    assert np.all(pi >= 0.0)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(pi @ np.ones(pi.shape[1]), np.ones(pi.shape[0]), atol=1e-14)
    for row, (source, region) in enumerate(restriction.indexes["state"]):
        support = restriction.isel(state=row).data.todense() > 0.0
        other_sources = [index for index, label in enumerate(SOURCES) if label != source]
        assert not support[other_sources].any()
        assert np.array_equal(support[SOURCES.index(source)], REGIONS[source] == region)


@pytest.mark.parametrize("class_blocked", [False, True], ids=["separable", "class-blocked"])
@pytest.mark.parametrize("observation_covariance", ["dense", "diagonal"])
def test_supplied_restriction_products_match_dense_oracle(
    class_blocked: bool,
    observation_covariance: str,
) -> None:
    """Products follow the supplied-Pi-first equations used by verification-games."""
    covariance, dense_b = _covariance(class_blocked)
    restriction = _absolute_flux_weighted_restriction()
    sensitivity = _sensitivity()

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=_bucket_prolongation(),
        state_dim="state",
        native_sensitivity=sensitivity,
        observation_dim="observation",
        observation_covariance=observation_covariance,
        observation_batch_size=3,
        strategy=SuppliedRestriction(restriction, retained_distribution="lognormal"),
    )

    pi = _flat(restriction, "state")
    h = _flat(sensitivity, "observation")
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
    covariance, _ = _covariance(class_blocked=False)
    restriction = _absolute_flux_weighted_restriction()

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=_bucket_prolongation(),
        state_dim="state",
        native_sensitivity=_sensitivity(),
        observation_dim="observation",
        strategy=SuppliedRestriction(restriction, retained_distribution="lognormal"),
    )

    result = products.restriction
    assert isinstance(result.data, sparse.COO)
    assert result.dims == ("state", *NATIVE_DIMS)
    assert result.indexes["state"].equals(_state_index())
    assert result.attrs["policy"] == "absolute-flux-weighted regional mean"
    assert result.attrs["restriction_source"] == "supplied"
    assert result.attrs["retained_distribution"] == "lognormal"
    assert result.attrs["lognormal_admissible"] is True
    assert result.attrs["nonnegative"] is True
    assert result.attrs["row_normalized"] is True
    assert result.attrs["units"] == "1"
    assert "restriction_source" not in restriction.attrs


def _signed_restriction() -> xr.DataArray:
    """Return a signed variant of the fixture, as from a signed policy."""
    restriction = _absolute_flux_weighted_restriction()
    dense = restriction.data.todense()
    dense[0, 0, 0, 0] = -dense[0, 0, 0, 0]
    return restriction.copy(data=sparse.COO.from_numpy(dense))


def test_signed_restriction_is_refused_for_lognormal_and_allowed_for_gaussian() -> None:
    """Only a nonnegative Pi guarantees positive retained scalings."""
    covariance, _ = _covariance(class_blocked=False)
    restriction = _signed_restriction()
    options = {
        "covariance": covariance,
        "basis_prolongation": _bucket_prolongation(),
        "state_dim": "state",
        "native_sensitivity": _sensitivity(),
        "observation_dim": "observation",
    }

    with pytest.raises(ValueError, match="LogNormal retained state requires a nonnegative"):
        project_native_covariance(
            **options, strategy=SuppliedRestriction(restriction, retained_distribution="lognormal")
        )
    products = project_native_covariance(**options, strategy=SuppliedRestriction(restriction))

    assert products.restriction.attrs["lognormal_admissible"] is False
    assert products.restriction.attrs["nonnegative"] is False


def test_zero_row_is_refused_for_lognormal() -> None:
    """A LogNormal retained scaling needs positive weight in every row."""
    covariance, _ = _covariance(class_blocked=False)
    restriction = _absolute_flux_weighted_restriction()
    dense = restriction.data.todense()
    dense[1] = 0.0
    strategy = SuppliedRestriction(
        restriction.copy(data=sparse.COO.from_numpy(dense)),
        retained_distribution="lognormal",
    )

    with pytest.raises(ValueError, match="positive weight in every"):
        strategy.projection(
            covariance,
            _bucket_prolongation(),
            native_dims=NATIVE_DIMS,
            state_dim="state",
        )


def test_admissible_restriction_permits_a_signed_conditional_lift() -> None:
    """The covariance-natural lift may be signed while Pi remains admissible."""
    covariance, _ = _covariance(class_blocked=False)

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=_bucket_prolongation(),
        state_dim="state",
        native_sensitivity=_sensitivity(),
        observation_dim="observation",
        strategy=SuppliedRestriction(
            _absolute_flux_weighted_restriction(),
            retained_distribution="lognormal",
        ),
    )

    assert float(products.prolongation.min()) < 0.0


@pytest.mark.parametrize(
    "change",
    ["reordered-state", "renamed-state", "missing-dim", "reordered-native", "non-finite", "fill-value"],
)
def test_supplied_restriction_rejects_invalid_labels_and_values(change: str) -> None:
    """Supplied Pi must use exact labels, dimensions, and finite real values."""
    covariance, _ = _covariance(class_blocked=False)
    restriction = _absolute_flux_weighted_restriction()
    if change == "reordered-state":
        restriction = restriction.isel(state=slice(None, None, -1))
        match = "labels must match"
    elif change == "renamed-state":
        index = _state_index().set_levels(["Gpp", "Ff"], level="source")
        restriction = restriction.drop_vars(["state", "source", "region"]).assign_coords(
            xr.Coordinates.from_pandas_multiindex(index, "state")
        )
        match = "labels must match"
    elif change == "missing-dim":
        restriction = restriction.isel({SOURCE_DIM: 0}, drop=True)
        match = "must have dimensions"
    elif change == "reordered-native":
        restriction = restriction.isel(lon=slice(None, None, -1))
        match = r"join='exact'|align"
    elif change == "non-finite":
        dense = restriction.data.todense()
        dense[0, 0, 0, 0] = np.nan
        restriction = restriction.copy(data=sparse.COO.from_numpy(dense))
        match = "finite real"
    else:
        restriction = restriction.copy(data=sparse.COO.from_numpy(restriction.data.todense(), fill_value=0.0))
        restriction = restriction.copy(data=sparse.full_like(restriction.data, 0.5))
        match = "zero fill value"

    with pytest.raises(ValueError, match=match):
        SuppliedRestriction(restriction).projection(
            covariance,
            _bucket_prolongation(),
            native_dims=NATIVE_DIMS,
            state_dim="state",
        )


def test_redundant_supplied_restriction_is_rejected() -> None:
    """Duplicate retained rows fail the full-row-rank C_alpha factorization."""
    covariance, _ = _covariance(class_blocked=False)
    restriction = _absolute_flux_weighted_restriction()
    dense = restriction.data.todense()
    dense[1] = dense[0]

    with pytest.raises(ValueError, match="positive definite|ill-conditioned"):
        project_native_covariance(
            covariance=covariance,
            basis_prolongation=_bucket_prolongation(),
            state_dim="state",
            native_sensitivity=_sensitivity(),
            observation_dim="observation",
            strategy=SuppliedRestriction(restriction.copy(data=sparse.COO.from_numpy(dense))),
        )


def test_lazy_dense_supplied_restriction_matches_eager_sparse() -> None:
    """Storage and execution backend do not change the supplied-Pi products."""
    covariance, _ = _covariance(class_blocked=True)
    restriction = _absolute_flux_weighted_restriction()
    lazy_dense = restriction.copy(data=restriction.data.todense()).chunk({"state": 2})
    options = {
        "covariance": covariance,
        "basis_prolongation": _bucket_prolongation(),
        "state_dim": "state",
        "native_sensitivity": _sensitivity(),
        "observation_dim": "observation",
    }

    expected = project_native_covariance(**options, strategy=SuppliedRestriction(restriction))
    actual = project_native_covariance(**options, strategy=SuppliedRestriction(lazy_dense))

    assert isinstance(actual.restriction.data, np.ndarray)
    xr.testing.assert_allclose(actual.state_covariance, expected.state_covariance)
    xr.testing.assert_allclose(actual.prolongation, expected.prolongation)


def test_sparse_basis_prolongation_matches_dense_for_bucket_strategy() -> None:
    """A sparse U stays sparse to the strategy, which densifies it when required."""
    covariance, _ = _covariance(class_blocked=False)
    prolongation = _bucket_prolongation()
    options = {
        "covariance": covariance,
        "state_dim": "state",
        "native_sensitivity": _sensitivity(),
        "observation_dim": "observation",
        "strategy": PreserveBucketProlongation(),
    }

    expected = project_native_covariance(basis_prolongation=prolongation, **options)
    actual = project_native_covariance(
        basis_prolongation=prolongation.copy(data=sparse.COO.from_numpy(prolongation.values)),
        **options,
    )

    xr.testing.assert_allclose(actual.state_covariance, expected.state_covariance)
    xr.testing.assert_allclose(actual.prolongation, expected.prolongation)


def test_supplied_restriction_moments_and_monte_carlo_close() -> None:
    """Pi m and C_alpha are the declared moments of alpha = Pi x."""
    covariance, dense_b = _covariance(class_blocked=True)
    restriction = _absolute_flux_weighted_restriction()
    rng = np.random.default_rng(40)
    mean_values = 1.0 + 0.1 * rng.normal(size=(len(SOURCES), LATITUDE.size, LONGITUDE.size))
    native_mean = _labelled(mean_values, NATIVE_DIMS)

    reduction = reduce_native_gaussian(
        covariance=covariance,
        basis_prolongation=_bucket_prolongation(),
        state_dim="state",
        native_mean=native_mean,
        native_sensitivity=_sensitivity(),
        observation_dim="observation",
        strategy=SuppliedRestriction(restriction, retained_distribution="lognormal"),
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
    restriction = _absolute_flux_weighted_restriction()
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
        restriction,
        native_dims=NATIVE_DIMS,
        state_dim="state",
        column_dim="state_cov",
        rhs_block_size=2,
    )

    assert seen == [(np.ndarray, 2), (np.ndarray, 2), (np.ndarray, 1)]


def test_production_shape_restriction_stays_sparse_with_bounded_memory() -> None:
    """The 361 x 96,768 production restriction is validated without densifying.

    A dense float64 restriction at this shape needs about 279 MB. The sparse
    path stores one weight per native source cell, so validation and
    canonicalization stay well below that bound.
    """
    source_rows = {"GPP": 115, "TER": 115, "FF": 74, "ocean": 57}
    n_lat, n_lon = 168, 144
    cells = n_lat * n_lon
    assert len(source_rows) * cells == 96_768
    rng = np.random.default_rng(361)
    state_tuples = [(source, region) for source, count in source_rows.items() for region in range(count)]
    state_index = pd.MultiIndex.from_tuples(state_tuples, names=("source", "region"))

    tracemalloc.start()
    rows, sources, flat_cells, weights = [], [], [], []
    first_row = 0
    for source_index, count in enumerate(source_rows.values()):
        region = rng.integers(0, count, size=cells)
        region[:count] = np.arange(count)
        q = np.abs(rng.normal(size=cells))
        totals = np.bincount(region, weights=q, minlength=count)
        rows.append(first_row + region)
        sources.append(np.full(cells, source_index))
        flat_cells.append(np.arange(cells))
        weights.append(q / totals[region])
        first_row += count
    flat = np.concatenate(flat_cells)
    coords = np.stack([np.concatenate(rows), np.concatenate(sources), flat // n_lon, flat % n_lon])
    shape = (len(state_index), len(source_rows), n_lat, n_lon)
    pi_data = sparse.COO(coords, np.concatenate(weights), shape=shape)
    u_data = sparse.COO(coords[[1, 2, 3, 0]], np.ones(coords.shape[1]), shape=(*shape[1:], shape[0]))
    native_coords = {
        SOURCE_DIM: list(source_rows),
        "lat": np.linspace(-80.0, 80.0, n_lat),
        "lon": np.linspace(-180.0, 179.0, n_lon),
    }
    state_coords = xr.Coordinates.from_pandas_multiindex(state_index, "state")
    restriction = xr.DataArray(pi_data, dims=("state", *NATIVE_DIMS), coords=native_coords).assign_coords(
        state_coords
    )
    prolongation = xr.DataArray(u_data, dims=(*NATIVE_DIMS, "state"), coords=native_coords).assign_coords(
        state_coords
    )

    projection = SuppliedRestriction(restriction, retained_distribution="lognormal").projection(
        None,  # type: ignore[arg-type]
        prolongation,
        native_dims=NATIVE_DIMS,
        state_dim="state",
    )
    prepared = _prepare_projection(
        projection,
        native_reference=prolongation.isel(state=0, drop=True),
        native_dims=NATIVE_DIMS,
        state_dim="state",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    dense_bytes = np.prod(shape) * np.dtype(np.float64).itemsize
    assert dense_bytes > 270e6
    assert isinstance(prepared.restriction.data, sparse.COO)
    assert prepared.restriction.data.nnz == 96_768
    assert prepared.restriction.attrs["row_normalized"] is True
    assert prepared.restriction.attrs["lognormal_admissible"] is True
    assert peak < 0.1 * dense_bytes
