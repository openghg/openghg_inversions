"""Tests for the shared-kernel correlated (GPP/TER) native source covariance."""

from __future__ import annotations

import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.basis.covariance_products import (
    PreserveBucketProlongation,
    SuppliedRestriction,
    absolute_weighted_regional_mean_restriction,
)
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.correlated_source_covariance import (
    CorrelatedSourceCovariance,
    gpp_ter_source_covariance,
)
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.observation_error import prepare_low_rank_aggregation_error
from openghg_inversions.source_covariance import IndependentSourceCovariance

SOURCES = ("GPP", "TER", "FF")
SOURCE_DIM = "native_source"
NATIVE_DIMS = (SOURCE_DIM, "lat", "lon")
LATITUDE = np.array([50.0, 51.0, 52.5])
LONGITUDE = np.array([-2.0, -1.0, 0.5, 1.5])
CLASSES = np.array([["land", "land", "sea", "sea"], ["land", "land", "land", "sea"], ["sea", "land", "sea", "sea"]])
# GPP and TER share the kernel and differ only in sigma; FF has its own kernel.
SIGMAS = {"GPP": 0.7, "TER": 0.9, "FF": 0.4}
LENGTHS = {"GPP": (1.3, 2.1), "TER": (1.3, 2.1), "FF": (0.8, 0.8)}
REGIONS = {
    "GPP": np.array([[0, 0, 2, 2], [0, 3, 3, 2], [3, 3, -1, 2]]),
    "TER": np.array([[0, 0, 2, 2], [0, 3, 3, 2], [3, 3, -1, 2]]),
    "FF": np.array([[4, 4, 4, 7], [4, 4, 7, 7], [-1, 7, 7, 7]]),
}
N_OBS = 7
RHO = 0.7


def _coordinate(values: np.ndarray, dim: str) -> xr.DataArray:
    """Return an indexed degree coordinate."""
    return xr.DataArray(values, dims=dim, coords={dim: values})


def _labelled(values, dims: tuple[str, ...]) -> xr.DataArray:
    """Attach native, source and observation coordinates."""
    coords: dict[str, object] = {}
    if SOURCE_DIM in dims:
        coords[SOURCE_DIM] = list(SOURCES)
    if "lat" in dims:
        coords["lat"] = LATITUDE
    if "lon" in dims:
        coords["lon"] = LONGITUDE
    if "observation" in dims:
        coords["observation"] = [f"obs-{index}" for index in range(N_OBS)]
    return xr.DataArray(values, dims=dims, coords=coords)


def _components(
    *,
    ter_length: tuple[float, float] | None = None,
    ter_labels: xr.DataArray | None = None,
    order: tuple[str, ...] = SOURCES,
    blocked: bool = True,
) -> dict[str, SeparableExponentialCovariance]:
    """Return per-source actions on the shared grid, class-blocked by default."""
    labels = (
        xr.DataArray(CLASSES, dims=("lat", "lon"), coords={"lat": LATITUDE, "lon": LONGITUDE}) if blocked else None
    )
    lengths = dict(LENGTHS)
    if ter_length is not None:
        lengths["TER"] = ter_length
    return {
        source: SeparableExponentialCovariance(
            _coordinate(LATITUDE, "lat"),
            _coordinate(LONGITUDE, "lon"),
            sigma=SIGMAS[source],
            latitude_correlation_length=lengths[source][0],
            longitude_correlation_length=lengths[source][1],
            class_labels=ter_labels if source == "TER" and ter_labels is not None else labels,
        )
        for source in order
    }


def _unit_kernel(source: str, blocked: bool = True) -> np.ndarray:
    """Return the dense (class-blocked) unit kernel of one source."""
    lat_length, lon_length = LENGTHS[source]
    k_lat = np.exp(-np.abs(LATITUDE[:, None] - LATITUDE[None, :]) / lat_length)
    k_lon = np.exp(-np.abs(LONGITUDE[:, None] - LONGITUDE[None, :]) / lon_length)
    flat = CLASSES.reshape(-1)
    mask = flat[:, None] == flat[None, :] if blocked else True
    return np.kron(k_lat, k_lon) * mask


def _dense_b(rho: float, order: tuple[str, ...] = SOURCES, blocked: bool = True) -> np.ndarray:
    """Build B = S (R kron K) S for GPP/TER plus an independent FF block, in ``order``."""
    cells = LATITUDE.size * LONGITUDE.size
    dense = np.zeros((len(order) * cells, len(order) * cells))
    for row, first in enumerate(order):
        for column, second in enumerate(order):
            if first == second:
                block = SIGMAS[first] ** 2 * _unit_kernel(first, blocked)
            elif {first, second} == {"GPP", "TER"}:
                block = rho * SIGMAS[first] * SIGMAS[second] * _unit_kernel("GPP", blocked)
            else:
                continue
            dense[row * cells : (row + 1) * cells, column * cells : (column + 1) * cells] = block
    return dense


def _covariance(rho: float) -> CorrelatedSourceCovariance:
    """Return the correlated action through the GPP/TER option."""
    return gpp_ter_source_covariance(
        _components(),
        gpp_source="GPP",
        ter_source="TER",
        gpp_ter_correlation=rho,
        source_dim=SOURCE_DIM,
    )


def _sensitivity() -> xr.DataArray:
    """Return a native sensitivity whose GPP and TER responses nearly coincide."""
    rng = np.random.default_rng(188)
    common = rng.normal(size=(N_OBS, LATITUDE.size, LONGITUDE.size))
    values = np.stack(
        [
            -common + 0.2 * rng.normal(size=common.shape),
            common + 0.2 * rng.normal(size=common.shape),
            rng.normal(size=common.shape),
        ],
        axis=1,
    )
    return _labelled(values, ("observation", *NATIVE_DIMS)).assign_attrs(units="ppm")


def _weights() -> xr.DataArray:
    """Return signed flux-times-area weights; GPP is negative uptake."""
    rng = np.random.default_rng(1049)
    shape = (LATITUDE.size, LONGITUDE.size)
    flux = np.stack([-rng.uniform(0.5, 2.0, shape), rng.uniform(0.5, 2.0, shape), rng.uniform(0.1, 3.0, shape)])
    return _labelled(flux, NATIVE_DIMS)


def _restriction() -> xr.DataArray:
    """Return the source-blocked absolute-flux-weighted ``Pi``."""
    regions = {source: _labelled(REGIONS[source], ("lat", "lon")) for source in SOURCES}
    return absolute_weighted_regional_mean_restriction(regions, _weights())


def _bucket_prolongation() -> xr.DataArray:
    """Return a dense source-blocked bucket prolongation ``U`` (native x state)."""
    restriction = _restriction()
    dense = (restriction.data.todense() > 0).astype(float)
    return restriction.copy(data=dense).transpose(*NATIVE_DIMS, "state")


def _flat(array: xr.DataArray, leading: str) -> np.ndarray:
    """Flatten labelled ``(leading, *native)`` data to a dense matrix."""
    data = array.transpose(leading, *NATIVE_DIMS).data
    data = data.todense() if isinstance(data, sparse.SparseArray) else np.asarray(data)
    return data.reshape(data.shape[0], -1)


def _reduce(covariance, strategy=None, sensitivity=None, restriction=None):
    """Run the OGI coherent reduction for one covariance action."""
    sensitivity = _sensitivity() if sensitivity is None else sensitivity
    if strategy is None:
        strategy = SuppliedRestriction(_restriction() if restriction is None else restriction)
    return reduce_native_gaussian(
        covariance=covariance,
        strategy=strategy,
        state_dim="state",
        native_mean=xr.ones_like(sensitivity.isel(observation=0, drop=True)),
        native_sensitivity=sensitivity,
        observation_dim="observation",
        observation_batch_size=3,
    )


def _reference(rho: float) -> dict[str, np.ndarray]:
    """Return the dense joint reduction for the supplied ``Pi``."""
    b = _dense_b(rho)
    pi = _flat(_restriction(), "state")
    h = _flat(_sensitivity(), "observation")
    c_alpha = pi @ b @ pi.T
    h_alpha = np.linalg.solve(c_alpha, pi @ b @ h.T).T
    a = h @ b @ h.T - h_alpha @ c_alpha @ h_alpha.T
    return {"C": c_alpha, "H": h_alpha, "A": 0.5 * (a + a.T)}


def _state_blocks(state_index, first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean retained-state masks of two sources."""
    sources = np.asarray(state_index.get_level_values("source"))
    return sources == first, sources == second


def test_zero_correlation_is_bit_identical_to_independent_sources() -> None:
    """The default rho = 0 reproduces the independent path bit for bit."""
    components = _components()
    independent = IndependentSourceCovariance(components, source_dim=SOURCE_DIM)
    correlated = gpp_ter_source_covariance(components, gpp_source="GPP", ter_source="TER", source_dim=SOURCE_DIM)
    rhs = _sensitivity().rename(observation="rhs")

    xr.testing.assert_identical(correlated.apply(rhs), independent.apply(rhs))
    xr.testing.assert_identical(correlated.solve(rhs), independent.solve(rhs))

    for strategy in (SuppliedRestriction(_restriction()), PreserveBucketProlongation(_bucket_prolongation())):
        expected = _reduce(independent, strategy)
        actual = _reduce(correlated, strategy)
        for field in (
            "retained_mean",
            "retained_covariance",
            "effective_observation_operator",
            "observation_intercept",
            "unresolved_observation_covariance",
        ):
            xr.testing.assert_identical(getattr(actual, field), getattr(expected, field))
        expected_lrpd = prepare_low_rank_aggregation_error(
            expected.unresolved_observation_covariance, rank=3, output_dim="observation", covariance_dim="observation_cov"
        )
        actual_lrpd = prepare_low_rank_aggregation_error(
            actual.unresolved_observation_covariance, rank=3, output_dim="observation", covariance_dim="observation_cov"
        )
        np.testing.assert_array_equal(actual_lrpd.aggregation_error.factor, expected_lrpd.aggregation_error.factor)


@pytest.mark.parametrize("rho", [RHO, -0.6])
def test_apply_and_solve_match_dense_shared_kernel_reference(rho: float) -> None:
    """Apply and solve equal B = S (R kron K) S and its inverse in float64."""
    covariance = _covariance(rho)
    rhs = _sensitivity().rename(observation="rhs")
    flat_rhs = _flat(rhs, "rhs").T
    dense = _dense_b(rho)

    np.testing.assert_allclose(_flat(covariance.apply(rhs), "rhs").T, dense @ flat_rhs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        _flat(covariance.solve(rhs), "rhs").T, np.linalg.solve(dense, flat_rhs), rtol=1e-9, atol=1e-9
    )
    assert covariance.correlation_matrix.sel({SOURCE_DIM: "GPP", f"{SOURCE_DIM}_cov": "TER"}).item() == rho


def test_joint_reduction_keeps_cross_blocks_that_a_per_source_reduction_drops() -> None:
    """C_alpha and A carry the GPP/TER cross blocks of the dense joint reduction.

    A per-source reduction (each source reduced alone, A summed) has a zero
    GPP/TER block in C_alpha and the wrong A, so it fails these checks.
    """
    reduction = _reduce(_covariance(RHO))
    reference = _reference(RHO)
    c_alpha = reduction.retained_covariance
    gpp, ter = _state_blocks(c_alpha.indexes["state"], "GPP", "TER")

    np.testing.assert_allclose(c_alpha, reference["C"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(reduction.effective_observation_operator, reference["H"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(reduction.unresolved_observation_covariance, reference["A"], rtol=1e-9, atol=1e-11)
    cross = np.asarray(c_alpha.values)[np.ix_(gpp, ter)]
    assert np.abs(cross).max() > 0.1 * np.abs(np.asarray(c_alpha.values)[np.ix_(gpp, gpp)]).max()

    # Per-source reduction, as in the production gate: reduce each source
    # alone and sum A. Its C_alpha has no GPP/TER block, and its A misses the
    # cross term, so it does not match the joint dense reference.
    components = _components()
    separate_a = np.zeros((N_OBS, N_OBS))
    restriction, sensitivity = _restriction(), _sensitivity()
    for source in SOURCES:
        rows = np.asarray(restriction.indexes["state"].get_level_values("source")) == source
        single = IndependentSourceCovariance({source: components[source]}, source_dim=SOURCE_DIM)
        part = _reduce(
            single,
            sensitivity=sensitivity.sel({SOURCE_DIM: [source]}),
            restriction=restriction.isel(state=rows).sel({SOURCE_DIM: [source]}),
        )
        separate_a += np.asarray(part.unresolved_observation_covariance.values)
    np.testing.assert_allclose(separate_a, _reference(0.0)["A"], rtol=1e-9, atol=1e-11)
    gap = np.abs(separate_a - reference["A"]).max()
    assert gap > 1e3 * 1e-11 and gap > 0.05 * np.abs(reference["A"]).max()


def test_low_rank_aggregation_error_keeps_the_cross_source_term() -> None:
    """The LRPD factorization of the joint A reproduces the dense reference."""
    reduction = _reduce(_covariance(RHO))
    reference = _reference(RHO)["A"]
    unresolved = reduction.unresolved_observation_covariance

    full = prepare_low_rank_aggregation_error(
        unresolved, rank=N_OBS, output_dim="observation", covariance_dim="observation_cov"
    ).aggregation_error
    factor = np.asarray(full.factor.values)
    np.testing.assert_allclose(
        factor @ factor.T + np.diag(np.asarray(full.diagonal_variance.values)), reference, rtol=1e-9, atol=1e-10
    )

    rank = 3
    truncated = prepare_low_rank_aggregation_error(
        unresolved, rank=rank, output_dim="observation", covariance_dim="observation_cov"
    ).aggregation_error
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    top = eigenvectors[:, -rank:] * np.sqrt(eigenvalues[-rank:])
    expected = top @ top.T
    expected += np.diag(np.maximum(np.diag(reference) - np.diag(expected), 0.0))
    factor = np.asarray(truncated.factor.values)
    actual = factor @ factor.T + np.diag(np.asarray(truncated.diagonal_variance.values))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    # The joint low-rank A differs materially from the per-source one.
    assert np.abs(actual - _reference(0.0)["A"]).max() > 0.05 * np.abs(reference).max()


def test_positive_rho_cancels_signed_net_biosphere_variance() -> None:
    """rho > 0 on scaling anomalies lowers Var(GPP + TER) of signed fluxes."""
    weights = _weights()
    functional = np.concatenate(
        [_flat(weights.sel({SOURCE_DIM: [source]}).expand_dims(f=[0]), "f")[0] for source in ("GPP", "TER")]
    )
    functional = np.concatenate([functional, np.zeros(LATITUDE.size * LONGITUDE.size)])

    labelled = _labelled(functional.reshape(len(SOURCES), LATITUDE.size, LONGITUDE.size), NATIVE_DIMS)

    def net_variance(rho: float) -> float:
        """Return Var(w_G x_G + w_T x_T) = w' B w through the covariance action."""
        return float((labelled * _covariance(rho).apply(labelled)).sum())

    assert net_variance(RHO) < net_variance(0.0) < net_variance(-0.6)


@pytest.mark.parametrize(
    ("components", "correlations", "error", "match"),
    [
        (lambda: _components(), {("GPP", "TER"): 1.2}, ValueError, r"\|rho\| <= 1"),
        (lambda: _components(), {("GPP", "TER"): float("nan")}, ValueError, "finite"),
        (lambda: _components(ter_length=(1.3, 3.0)), {("GPP", "TER"): 0.5}, ValueError, "share one unit kernel"),
        (lambda: _components(), {("GPP", "missing"): 0.5}, ValueError, "unconfigured source"),
        (lambda: _components(), {("GPP", "TER"): 0.5, ("TER", "GPP"): 0.5}, ValueError, "more than once"),
        (
            lambda: {source: _components(ter_length=LENGTHS["GPP"])["GPP"] for source in ("A", "B", "C")},
            {("A", "B"): 0.9, ("B", "C"): 0.9, ("A", "C"): -0.9},
            ValueError,
            "not positive semidefinite",
        ),
    ],
)
def test_invalid_correlations_are_refused(components, correlations, error, match) -> None:
    """Out-of-range, non-PSD and unshared-kernel correlations fail clearly."""
    with pytest.raises(error, match=match):
        CorrelatedSourceCovariance(components(), correlations, source_dim=SOURCE_DIM)


def test_kernel_identity_ignores_cosmetic_class_label_differences() -> None:
    """Equal class partitions share a kernel whatever the label values or metadata."""
    codes = np.where(CLASSES == "land", 7.0, 2.0)  # float codes, reversed order, named, with attrs
    relabelled = xr.DataArray(
        codes, dims=("lon", "lat")[::-1], coords={"lat": LATITUDE, "lon": LONGITUDE}, name="pft", attrs={"a": 1}
    ).transpose("lon", "lat")
    covariance = CorrelatedSourceCovariance(
        _components(ter_labels=relabelled), {("GPP", "TER"): RHO}, source_dim=SOURCE_DIM
    )
    rhs = _sensitivity().rename(observation="rhs")
    xr.testing.assert_allclose(covariance.apply(rhs), _covariance(RHO).apply(rhs), rtol=1e-15, atol=0)

    merged = xr.DataArray(
        np.where(CLASSES == "land", "land", "land"), dims=("lat", "lon"), coords={"lat": LATITUDE, "lon": LONGITUDE}
    )
    with pytest.raises(ValueError, match="share one unit kernel"):
        CorrelatedSourceCovariance(_components(ter_labels=merged), {("GPP", "TER"): RHO}, source_dim=SOURCE_DIM)


def test_differing_kernels_are_allowed_when_uncorrelated() -> None:
    """The shared-kernel check applies only to nonzero correlations."""
    covariance = gpp_ter_source_covariance(
        _components(ter_length=(1.3, 3.0)), gpp_source="GPP", ter_source="TER", source_dim=SOURCE_DIM
    )
    assert covariance.correlations == {("GPP", "TER"): 0.0}


def test_correlated_sources_must_expose_a_comparable_kernel() -> None:
    """An action without sigma/to_dataset cannot be proven to share a kernel."""

    class Opaque:
        """Minimal invertible action without a serializable configuration."""

        def __init__(self, inner: SeparableExponentialCovariance) -> None:
            """Wrap one separable action."""
            self.inner = inner
            self.native_dims = inner.native_dims

        def apply(self, rhs: xr.DataArray) -> xr.DataArray:
            """Delegate apply."""
            return self.inner.apply(rhs)

        def solve(self, rhs: xr.DataArray) -> xr.DataArray:
            """Delegate solve."""
            return self.inner.solve(rhs)

    components = _components()
    opaque = {"GPP": Opaque(components["GPP"]), "TER": Opaque(components["TER"])}
    with pytest.raises(TypeError, match="to_dataset"):
        CorrelatedSourceCovariance(opaque, {("GPP", "TER"): 0.5})


@pytest.mark.parametrize("rho", [1.0, -1.0])
def test_unit_correlation_is_psd_but_has_no_solve(rho: float) -> None:
    """|rho| = 1 is a valid singular covariance: apply works, solve is refused."""
    covariance = _covariance(rho)
    rhs = _sensitivity().rename(observation="rhs")
    np.testing.assert_allclose(_flat(covariance.apply(rhs), "rhs").T, _dense_b(rho) @ _flat(rhs, "rhs").T, atol=1e-13)
    with pytest.raises(np.linalg.LinAlgError, match=r"R is not positive definite .*\|rho\| = 1"):
        covariance.solve(rhs)


def _ordered_rhs(order: tuple[str, ...]) -> xr.DataArray:
    """Return an identity right-hand side over the native state in ``order``, rhs axis in the middle."""
    size = len(order) * LATITUDE.size * LONGITUDE.size
    eye = np.eye(size).reshape(size, len(order), LATITUDE.size, LONGITUDE.size)
    coords = {"rhs": np.arange(size), SOURCE_DIM: list(order), "lat": LATITUDE, "lon": LONGITUDE}
    return xr.DataArray(eye, dims=("rhs", *NATIVE_DIMS), coords=coords).transpose("lat", "rhs", SOURCE_DIM, "lon")


def _as_matrix(result: xr.DataArray) -> np.ndarray:
    """Return the dense matrix whose columns are the actions on identity columns."""
    values = result.transpose("rhs", *NATIVE_DIMS).values
    return values.reshape(values.shape[0], -1).T


@pytest.mark.parametrize(
    ("order", "blocked"),
    [(("TER", "FF", "GPP"), True), (("GPP", "TER", "FF"), False)],
)
def test_source_order_and_unblocked_kernel_match_dense_reference(order: tuple[str, ...], blocked: bool) -> None:
    """TER before GPP and the unblocked (Cholesky) solve path match S (R kron K) S."""
    covariance = gpp_ter_source_covariance(
        _components(order=order, blocked=blocked),
        gpp_source="GPP",
        ter_source="TER",
        gpp_ter_correlation=RHO,
        source_dim=SOURCE_DIM,
    )
    dense = _dense_b(RHO, order, blocked)
    rhs = _ordered_rhs(order)
    np.testing.assert_allclose(_as_matrix(covariance.apply(rhs)), dense, atol=1e-13)
    np.testing.assert_allclose(_as_matrix(covariance.solve(rhs)) @ dense, np.eye(dense.shape[0]), atol=1e-8)


def test_bucket_prolongation_reduction_matches_dense_reference_with_rho() -> None:
    """With rho != 0 the solve-based bucket strategy gives the dense joint C_alpha and A."""
    reduction = _reduce(_covariance(RHO), PreserveBucketProlongation(_bucket_prolongation()))
    b = _dense_b(RHO)
    u = _flat(_bucket_prolongation().transpose("state", *NATIVE_DIMS), "state").T
    h = _flat(_sensitivity(), "observation")
    c_alpha = np.linalg.inv(u.T @ np.linalg.solve(b, u))
    h_alpha = h @ u
    a = h @ b @ h.T - h_alpha @ c_alpha @ h_alpha.T

    np.testing.assert_allclose(reduction.retained_covariance, c_alpha, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(reduction.unresolved_observation_covariance, a, rtol=1e-7, atol=1e-8)
    gpp, ter = _state_blocks(reduction.retained_covariance.indexes["state"], "GPP", "TER")
    assert np.abs(np.asarray(reduction.retained_covariance.values)[np.ix_(gpp, ter)]).max() > 1e-2


def _three_source_components() -> dict[str, SeparableExponentialCovariance]:
    """Return three sources sharing the GPP kernel with different sigmas."""
    shared = _components()["GPP"]
    return {
        name: SeparableExponentialCovariance(
            _coordinate(LATITUDE, "lat"),
            _coordinate(LONGITUDE, "lon"),
            sigma=sigma,
            latitude_correlation_length=LENGTHS["GPP"][0],
            longitude_correlation_length=LENGTHS["GPP"][1],
            class_labels=shared.class_labels,
        )
        for name, sigma in (("A", 0.5), ("B", 1.1), ("C", 0.8))
    }


def test_three_source_group_matches_dense_reference_and_names_a_singular_r() -> None:
    """A three-source group applies S (R kron K) S; a singular R without |rho| = 1 is named."""
    names = ("A", "B", "C")
    correlation = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    pairs = {(names[i], names[j]): correlation[i, j] for i in range(3) for j in range(i + 1, 3)}
    covariance = CorrelatedSourceCovariance(_three_source_components(), pairs, source_dim=SOURCE_DIM)
    sigma = np.repeat([0.5, 1.1, 0.8], LATITUDE.size * LONGITUDE.size)
    dense = sigma[:, None] * np.kron(correlation, _unit_kernel("GPP")) * sigma[None, :]
    rhs = _ordered_rhs(names)
    np.testing.assert_allclose(_as_matrix(covariance.apply(rhs)), dense, atol=1e-13)
    np.testing.assert_allclose(_as_matrix(covariance.solve(rhs)) @ dense, np.eye(dense.shape[0]), atol=1e-8)

    singular = {("A", "B"): 0.5, ("A", "C"): 0.5, ("B", "C"): -0.5}  # det R = 0, no |rho| = 1
    covariance = CorrelatedSourceCovariance(_three_source_components(), singular, source_dim=SOURCE_DIM)
    with pytest.raises(np.linalg.LinAlgError, match="R is not positive definite .*near-.*zero eigenvalue"):
        covariance.solve(rhs)


class _ConfiguredAction:
    """Generic action exposing sigma and to_dataset, but no canonical kernel comparison."""

    def __init__(self, inner: SeparableExponentialCovariance) -> None:
        """Wrap one separable action."""
        self.inner = inner
        self.native_dims = inner.native_dims
        self.sigma = inner.sigma
        self.latitude = inner.latitude
        self.longitude = inner.longitude

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Delegate apply."""
        return self.inner.apply(rhs)

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Delegate solve."""
        return self.inner.solve(rhs)

    def to_dataset(self) -> xr.Dataset:
        """Delegate the configuration."""
        return self.inner.to_dataset()


def test_generic_actions_are_compared_by_configuration_without_sigma() -> None:
    """The to_dataset fallback accepts equal kernels with different sigma and refuses others."""
    components = {source: _ConfiguredAction(action) for source, action in _components().items()}
    covariance = CorrelatedSourceCovariance(components, {("GPP", "TER"): RHO}, source_dim=SOURCE_DIM)
    rhs = _sensitivity().rename(observation="rhs")
    np.testing.assert_allclose(_flat(covariance.apply(rhs), "rhs").T, _dense_b(RHO) @ _flat(rhs, "rhs").T, atol=1e-13)
    with pytest.raises(ValueError, match="share one unit kernel"):
        CorrelatedSourceCovariance(components, {("GPP", "FF"): 0.3}, source_dim=SOURCE_DIM)


def test_uncorrelated_sources_on_different_grids_are_refused() -> None:
    """Every source must use the same grid values, correlated or not."""
    components = _components()
    shifted = LONGITUDE + 0.25
    components["FF"] = SeparableExponentialCovariance(
        _coordinate(LATITUDE, "lat"), _coordinate(shifted, "lon"), sigma=0.4, correlation_length=0.8
    )
    with pytest.raises(ValueError, match="grid differs"):
        gpp_ter_source_covariance(components, gpp_source="GPP", ter_source="TER", source_dim=SOURCE_DIM)


def test_serialization_round_trip_keeps_correlations_and_kernels(tmp_path) -> None:
    """to_dataset/from_dataset restore R, pair correlations and every component kernel."""
    covariance = _covariance(RHO)
    path = tmp_path / "correlated.nc"
    covariance.to_dataset().to_netcdf(path)
    with xr.open_dataset(path) as saved:
        restored = CorrelatedSourceCovariance.from_dataset(saved.load())

    assert restored.source_labels == covariance.source_labels
    assert dict(restored.correlations) == {("GPP", "TER"): RHO}
    xr.testing.assert_identical(restored.correlation_matrix, covariance.correlation_matrix)
    rhs = _sensitivity().rename(observation="rhs")
    xr.testing.assert_identical(restored.apply(rhs), covariance.apply(rhs))
    for source in SOURCES:
        xr.testing.assert_identical(
            restored.source_covariances[source].to_dataset(), covariance.source_covariances[source].to_dataset()
        )
    zero = gpp_ter_source_covariance(_components(), gpp_source="GPP", ter_source="TER", source_dim=SOURCE_DIM)
    assert dict(CorrelatedSourceCovariance.from_dataset(zero.to_dataset()).correlations) == {("GPP", "TER"): 0.0}
