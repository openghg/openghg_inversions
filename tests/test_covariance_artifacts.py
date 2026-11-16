"""Test versioned persistence of covariance products and coherent reductions."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    PreserveBucketProlongation,
    SuppliedRestriction,
    absolute_weighted_regional_mean_restriction,
    covariance_natural_prolongation,
    project_native_covariance,
)
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.covariance_artifacts import (
    ATTRS_ATTR,
    AXES_ATTR,
    GATHERED_DIM,
    CoherentReductionArtifact,
    NativeCovarianceProductsArtifact,
    coherent_from_datatree,
    coherent_to_datatree,
    content_id,
    load_coherent_reduction,
    load_products,
    products_from_datatree,
    products_to_datatree,
    save_coherent_reduction,
    save_products,
    source_id,
    with_prolongation,
)
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance

LATITUDE = np.array([50.0, 51.0, 52.5])
LONGITUDE = np.array([-2.0, -1.0, 0.5, 1.5])
# Non-lexical source order and ragged regions per source.
SOURCES = ("GPP", "FF")
REGIONS = {
    "GPP": np.array([[0, 0, 2, 2], [0, 3, 3, 2], [3, 3, -1, 2]]),
    "FF": np.array([[4, 4, 7, 7], [4, 4, 7, 7], [-1, 7, 7, 7]]),
}
PRODUCT_NAMES = (
    "restriction",
    "prolongation",
    "state_covariance",
    "effective_observation_operator",
    "observation_state_cross_covariance",
    "native_observation_covariance",
)
REDUCTION_NAMES = (
    "restriction",
    "retained_mean",
    "retained_covariance",
    "effective_observation_operator",
    "native_observation_mean",
    "observation_intercept",
    "unresolved_observation_covariance",
)


def _grid() -> tuple[xr.DataArray, xr.DataArray]:
    """Return labelled latitude and longitude coordinates with units."""
    latitude = xr.DataArray(LATITUDE, dims="lat", coords={"lat": LATITUDE}, attrs={"units": "degrees_north"})
    longitude = xr.DataArray(LONGITUDE, dims="lon", coords={"lon": LONGITUDE}, attrs={"units": "degrees_east"})
    return latitude, longitude


def _observations(count: int, *, datetime_index: bool) -> xr.Coordinates:
    """Return observation labels: plain strings or a (site, time) MultiIndex with an aux coordinate."""
    if not datetime_index:
        return xr.Coordinates({"observation": [f"site-{index}" for index in range(count)]})
    times = pd.to_datetime(["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T00:00", "2020-01-01T00:00:00.000000001"], format="ISO8601")
    index = pd.MultiIndex.from_arrays([["TAC", "TAC", "MHD", "RGL"][:count], times[:count]], names=("site", "time"))
    coordinates = xr.Coordinates.from_pandas_multiindex(index, "observation")
    return coordinates.assign(release_lat=("observation", np.linspace(50.0, 53.0, count)))


def _single_source_problem(
    *, class_blocked: bool = False, datetime_index: bool = False
) -> tuple[SeparableExponentialCovariance, PreserveBucketProlongation, xr.DataArray, xr.DataArray]:
    """Return covariance, bucket strategy, sensitivity ``H`` and mean ``m`` on one source."""
    latitude, longitude = _grid()
    class_labels = xr.DataArray(
        [["land", "land", "sea", "sea"], ["land", "land", "land", "sea"], ["sea", "land", "sea", "sea"]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    covariance = SeparableExponentialCovariance(
        latitude, longitude, sigma=0.8, correlation_length=1.1, class_labels=class_labels if class_blocked else None
    )
    buckets = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [2, 2, 1, 1]])
    prolongation = xr.DataArray(
        (buckets[..., None] == np.arange(3)).astype(float),
        dims=("lat", "lon", "state"),
        coords={"lat": latitude, "lon": longitude, "state": ["north", "east", "south"]},
    )
    rng = np.random.default_rng(40)
    observation_coords = _observations(4, datetime_index=datetime_index)
    sensitivity = xr.DataArray(
        rng.normal(size=(4, LATITUDE.size, LONGITUDE.size)),
        dims=("observation", "lat", "lon"),
        coords={**observation_coords, "lat": latitude, "lon": longitude},
        attrs={"units": "ppm"},
    )
    mean = xr.DataArray(
        rng.uniform(0.5, 1.5, size=(LATITUDE.size, LONGITUDE.size)),
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    return covariance, PreserveBucketProlongation(prolongation), sensitivity, mean


def _multisource_problem(
    *, class_blocked: bool = False,
) -> tuple[IndependentSourceCovariance, SuppliedRestriction, xr.DataArray, xr.DataArray]:
    """Return a gathered multisource problem with a sparse supplied ``Pi``."""
    latitude, longitude = _grid()
    native = {"native_source": list(SOURCES), "lat": latitude, "lon": longitude}
    class_labels = xr.DataArray(
        [["a", "a", "b", "b"], ["a", "a", "a", "b"], ["b", "a", "b", "b"]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
    )
    covariance = IndependentSourceCovariance(
        {
            "GPP": SeparableExponentialCovariance(latitude, longitude, sigma=0.7, correlation_length=1.3),
            "FF": SeparableExponentialCovariance(
                latitude,
                longitude,
                sigma=0.4,
                correlation_length=0.8,
                class_labels=class_labels if class_blocked else None,
            ),
        }
    )
    rng = np.random.default_rng(1053)
    weights = xr.DataArray(rng.normal(1.0, 1.0, size=(2, 3, 4)), dims=("native_source", "lat", "lon"), coords=native)
    regions = {
        source: xr.DataArray(labels, dims=("lat", "lon"), coords={"lat": latitude, "lon": longitude})
        for source, labels in REGIONS.items()
    }
    restriction = absolute_weighted_regional_mean_restriction(regions, weights)
    sensitivity = xr.DataArray(
        rng.normal(size=(4, 2, 3, 4)),
        dims=("observation", "native_source", "lat", "lon"),
        coords={**_observations(4, datetime_index=False), **native},
        attrs={"units": "ppm"},
    )
    mean = xr.DataArray(rng.uniform(0.5, 1.5, size=(2, 3, 4)), dims=("native_source", "lat", "lon"), coords=native)
    return covariance, SuppliedRestriction(restriction, require_positivity_preserving=True), sensitivity, mean


PROBLEMS: dict[str, Callable[[], tuple]] = {
    "single-source": _single_source_problem,
    "class-blocked": lambda: _single_source_problem(class_blocked=True),
    "datetime-multiindex": lambda: _single_source_problem(datetime_index=True),
    "gathered-multisource": _multisource_problem,
    "class-blocked-multisource": lambda: _multisource_problem(class_blocked=True),
}


def _products_artifact(
    problem: str, view: str = "dense"
) -> tuple[NativeCovarianceProductsArtifact, xr.DataArray, xr.DataArray]:
    """Project one problem and wrap the products with binding metadata."""
    covariance, strategy, sensitivity, mean = PROBLEMS[problem]()
    products = project_native_covariance(
        covariance=covariance,
        strategy=strategy,
        state_dim="state",
        native_sensitivity=sensitivity,
        observation_dim="observation",
        observation_covariance=view,
    )
    artifact = NativeCovarianceProductsArtifact(
        products=products,
        covariance=covariance,
        inputs_id="sha256:prepared-inputs",
        provenance={"basis": {"name": problem, "regions": 3}},
    )
    return artifact, sensitivity, mean


def _coherent_artifact(problem: str) -> CoherentReductionArtifact:
    """Reduce one problem and wrap the reduction with binding metadata."""
    covariance, strategy, sensitivity, mean = PROBLEMS[problem]()
    reduction = reduce_native_gaussian(
        covariance=covariance,
        strategy=strategy,
        state_dim="state",
        native_mean=mean,
        native_sensitivity=sensitivity,
        observation_dim="observation",
    )
    return CoherentReductionArtifact(
        reduction=reduction,
        covariance=covariance,
        inputs_id="sha256:prepared-inputs",
        provenance={"basis": {"name": problem, "regions": 3}},
    )


def _assert_same_array(actual: xr.DataArray, expected: xr.DataArray) -> None:
    """Require equal values, labels with their types and order, and attributes."""
    assert type(actual.data) is type(expected.data)
    assert actual.dtype == np.float64
    xr.testing.assert_equal(actual, expected)
    assert actual.name == expected.name
    assert actual.attrs == expected.attrs
    # Shared axes are stored once, so a coordinate carries the union of the
    # attributes the products gave it.
    for name, coordinate in expected.coords.items():
        assert coordinate.attrs.items() <= actual.coords[name].attrs.items()
    for dim in actual.dims:
        actual_index, expected_index = actual.get_index(dim), expected.get_index(dim)
        assert actual_index.equals(expected_index)
        assert list(actual_index.names) == list(expected_index.names)
        if isinstance(expected_index, pd.MultiIndex):
            for level, expected_level in enumerate(expected_index.levels):
                assert actual_index.levels[level].dtype.kind == expected_level.dtype.kind


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("view", ["dense", "diagonal"])
def test_products_round_trip_through_netcdf(problem: str, view: str, tmp_path: Path) -> None:
    """Products keep values, typed labels, units, provenance and identities through NetCDF."""
    artifact, _, _ = _products_artifact(problem, view)
    path = tmp_path / "products.nc"
    save_products(artifact, path)
    loaded = load_products(path)

    for name in PRODUCT_NAMES:
        _assert_same_array(getattr(loaded.products, name), getattr(artifact.products, name))
    assert loaded.products.strategy == artifact.products.strategy
    assert loaded.products.observation_covariance_view == view
    assert loaded.inputs_id == artifact.inputs_id
    assert dict(loaded.provenance) == {"basis": {"name": problem, "regions": 3}}
    xr.testing.assert_identical(loaded.covariance.to_dataset(), artifact.covariance.to_dataset())
    assert source_id(loaded) == source_id(artifact)
    assert content_id(loaded) == content_id(artifact)


def test_ragged_nonlexical_states_and_source_order_survive(tmp_path: Path) -> None:
    """Configured source order and ragged retained states are not sorted or padded."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    save_products(artifact, tmp_path / "products.nc")
    loaded = load_products(tmp_path / "products.nc")

    state = loaded.products.state_covariance.get_index("state")
    assert list(state) == [("GPP", 0), ("GPP", 1), ("GPP", 2), ("FF", 0), ("FF", 1)]
    assert list(loaded.products.restriction.get_index("native_source")) == ["GPP", "FF"]
    assert list(loaded.covariance.source_labels) == ["GPP", "FF"]
    assert list(loaded.products.state_covariance.get_index("state_cov").names) == ["source_cov", "region_in_source_cov"]


def test_datetime_multiindex_keeps_nanoseconds_and_auxiliary_coordinates(tmp_path: Path) -> None:
    """Datetime MultiIndex levels and auxiliary coordinates survive on row and column axes."""
    artifact, _, _ = _products_artifact("datetime-multiindex")
    save_products(artifact, tmp_path / "products.nc")
    loaded = load_products(tmp_path / "products.nc").products.native_observation_covariance

    times = loaded.get_index("observation").get_level_values("time")
    assert times[-1] == pd.Timestamp("2020-01-01T00:00:00.000000001")
    np.testing.assert_array_equal(loaded["release_lat"].values, loaded["release_lat_cov"].values)
    assert loaded.get_index("observation_cov").names == ["site_cov", "time_cov"]


def test_supplied_sparse_restriction_round_trips_without_densifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sparse supplied Pi is gathered on disk and loads as COO without a dense copy."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    restriction = artifact.products.restriction
    assert isinstance(restriction.data, sparse.COO)

    def refuse_densify(*args, **kwargs):
        raise AssertionError("Pi was densified")

    monkeypatch.setattr(sparse.COO, "todense", refuse_densify)
    path = tmp_path / "products.nc"
    save_products(artifact, path, include_prolongation=False)
    loaded = load_products(path)
    monkeypatch.undo()

    assert isinstance(loaded.products.restriction.data, sparse.COO)
    np.testing.assert_array_equal(loaded.products.restriction.data.coords, restriction.data.coords)
    np.testing.assert_array_equal(loaded.products.restriction.data.data, restriction.data.data)
    with xr.open_datatree(path) as tree:
        group = tree["products/restriction"]
        assert group["restriction"].dims == (GATHERED_DIM,)
        assert group.sizes[GATHERED_DIM] == restriction.data.nnz
        assert group[GATHERED_DIM].attrs["compress"] == "state native_source lat lon"


def test_zarr_round_trip_keeps_sparse_restriction_and_identity(tmp_path: Path) -> None:
    """The same layout and identity checks hold for Zarr storage."""
    artifact = _coherent_artifact("gathered-multisource")
    save_coherent_reduction(artifact, tmp_path / "coherent.zarr")
    loaded = load_coherent_reduction(tmp_path / "coherent.zarr")

    assert isinstance(loaded.reduction.restriction.data, sparse.COO)
    assert content_id(loaded) == content_id(artifact)
    assert source_id(loaded) == source_id(artifact)


def test_string_labels_hash_alike_across_numpy_string_dtypes() -> None:
    """Fixed-width, object and variable-width StringDType labels give one digest."""
    from openghg_inversions.covariance_artifacts import _normalized_values

    labels = ["MHD", "TAC", "Mace Head"]
    expected = _normalized_values(np.array(labels), "labels")
    assert _normalized_values(np.array(labels, dtype=object), "labels") == expected
    assert _normalized_values(np.array(labels, dtype=np.dtypes.StringDType()), "labels") == expected


def test_restriction_provenance_round_trips_for_both_strategies(tmp_path: Path) -> None:
    """Derived and supplied restrictions both record their source and keep typed attributes."""
    derived, _, _ = _products_artifact("single-source")
    supplied, _, _ = _products_artifact("gathered-multisource")
    for name, artifact in (("derived", derived), ("supplied", supplied)):
        save_products(artifact, tmp_path / f"{name}.nc")
        attrs = load_products(tmp_path / f"{name}.nc").products.restriction.attrs
        assert attrs["restriction_source"] == name
        assert attrs == artifact.products.restriction.attrs
    assert attrs["positivity_preserving_required"] is True


def test_prolongation_is_optional_and_can_be_rebuilt(tmp_path: Path) -> None:
    """Omitting U_* keeps source_id, changes content_id, and the rebuilt U_* matches."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    path = tmp_path / "products.nc"
    save_products(artifact, path, include_prolongation=False)
    loaded = load_products(path)

    assert loaded.products.prolongation is None
    with xr.open_datatree(path) as tree:
        assert "prolongation" not in tree["products"].children
    assert source_id(loaded) == source_id(artifact)
    assert content_id(loaded, include_prolongation=False) == content_id(artifact, include_prolongation=False)
    assert content_id(artifact, include_prolongation=False) != content_id(artifact)

    rebuilt = with_prolongation(loaded)
    assert rebuilt.products.prolongation is not None
    xr.testing.assert_allclose(rebuilt.products.prolongation, artifact.products.prolongation, rtol=1e-12, atol=1e-14)
    assert rebuilt.products.prolongation.attrs == artifact.products.prolongation.attrs
    assert source_id(rebuilt) == source_id(artifact)
    assert with_prolongation(artifact) is artifact

    with pytest.raises(ValueError, match="needs products with U_"):
        save_products(loaded, tmp_path / "again.nc")


def test_prolongation_alone_may_be_stored_as_float32(tmp_path: Path) -> None:
    """U_* may use float32 storage; every other product stays float64."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    path = tmp_path / "products.nc"
    save_products(artifact, path, prolongation_dtype="float32")
    loaded = load_products(path)

    prolongation = loaded.products.prolongation
    assert prolongation is not None and prolongation.dtype == np.float32
    xr.testing.assert_allclose(prolongation.astype(np.float64), artifact.products.prolongation, rtol=1e-6)
    for name in PRODUCT_NAMES[2:]:
        assert getattr(loaded.products, name).dtype == np.float64
    assert loaded.products.restriction.data.dtype == np.float64
    assert source_id(loaded) == source_id(artifact)
    assert content_id(loaded, prolongation_dtype="float32") == content_id(artifact, prolongation_dtype="float32")
    assert content_id(artifact, prolongation_dtype="float32") != content_id(artifact)

    with pytest.raises(ValueError, match="prolongation_dtype must be"):
        save_products(artifact, tmp_path / "half.nc", prolongation_dtype="float16")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["state_covariance", "restriction", "native_observation_covariance"])
def test_only_prolongation_may_be_single_precision(name: str) -> None:
    """Loading refuses float32 storage of factorized, cancellation-sensitive or sparse products."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    tree = _edit(products_to_datatree(artifact), f"/products/{name}", lambda ds: ds.astype(np.float32))
    with pytest.raises(ValueError, match=f"{name} must be stored as float64; got float32"):
        products_from_datatree(tree)


def test_rebuild_requires_matching_retained_labels() -> None:
    """The positional U_* solve refuses a restriction and C_alpha with different state order."""
    artifact, _, _ = _products_artifact("single-source")
    products = artifact.products
    with pytest.raises(ValueError, match="identical retained-state labels"):
        covariance_natural_prolongation(
            artifact.covariance, products.restriction.isel(state=[1, 0, 2]), products.state_covariance
        )


def test_dense_and_diagonal_views_share_one_source_identity() -> None:
    """Views of the same inputs share source_id and differ in content_id."""
    dense, _, _ = _products_artifact("datetime-multiindex", "dense")
    diagonal, _, _ = _products_artifact("datetime-multiindex", "diagonal")
    assert source_id(dense) == source_id(diagonal)
    assert content_id(dense) != content_id(diagonal)

    other_inputs = NativeCovarianceProductsArtifact(dense.products, dense.covariance, inputs_id="sha256:other")
    assert source_id(other_inputs) != source_id(dense)


@pytest.mark.parametrize("problem", ["single-source", "datetime-multiindex", "gathered-multisource"])
def test_coherent_reduction_round_trips_bound_to_one_root_identity(problem: str, tmp_path: Path) -> None:
    """The coherent artifact round-trips and shares source_id with its dense products."""
    artifact = _coherent_artifact(problem)
    path = tmp_path / "coherent.nc"
    save_coherent_reduction(artifact, path)
    loaded = load_coherent_reduction(path)

    for name in REDUCTION_NAMES:
        _assert_same_array(getattr(loaded.reduction, name), getattr(artifact.reduction, name))
    assert loaded.reduction.projection_strategy == artifact.reduction.projection_strategy
    assert content_id(loaded) == content_id(artifact)
    products, _, _ = _products_artifact(problem)
    assert source_id(loaded) == source_id(products)

    with xr.open_datatree(path) as tree:
        assert tree.attrs["content_id"] == content_id(artifact)
        assert set(tree["reduction"].children) == set(REDUCTION_NAMES)
        assert tree.attrs["native_family"] == "gaussian"


def test_shared_axes_are_stored_once_with_a_declared_mirror_convention() -> None:
    """Product groups carry no coordinates; the axes node declares column-axis mirrors."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    tree = products_to_datatree(artifact)
    axes = tree["products"]
    registry = __import__("json").loads(axes.attrs[AXES_ATTR])

    assert registry["native"] == {"dims": ["native_source", "lat", "lon"]}
    assert registry["state_column"] == {
        "dim": "state_cov",
        "mirrors": "state",
        "names": {"state": "state_cov", "source": "source_cov", "region_in_source": "region_in_source_cov"},
    }
    for name in PRODUCT_NAMES:
        group = axes[name].to_dataset(inherit=False)
        assert set(group.coords) <= {GATHERED_DIM}
        assert group[name].dtype == np.float64
        assert ATTRS_ATTR in group[name].attrs


def test_reserved_product_names_cannot_label_an_axis() -> None:
    """An axis coordinate named like a product fails clearly at save."""
    covariance, strategy, sensitivity, _ = _single_source_problem()
    products = project_native_covariance(
        covariance=covariance,
        strategy=strategy,
        state_dim="state",
        native_sensitivity=sensitivity.assign_coords(prolongation=("observation", np.arange(4))),
        observation_dim="observation",
    )
    with pytest.raises(ValueError, match="'prolongation' is reserved"):
        products_to_datatree(NativeCovarianceProductsArtifact(products, covariance, "inputs"))


def _edit(tree: xr.DataTree, path: str, edit: Callable[[xr.Dataset], xr.Dataset]) -> xr.DataTree:
    """Return a copy of ``tree`` with one node's own dataset edited."""
    result = tree.copy(deep=True)
    result[path].dataset = edit(result[path].to_dataset(inherit=False))
    return result


def _with_attrs(**attrs: object) -> Callable[[xr.Dataset], xr.Dataset]:
    """Return an edit that updates dataset attributes."""
    return lambda dataset: dataset.assign_attrs(attrs)


MALFORMED: dict[str, tuple[str, Callable[[xr.Dataset], xr.Dataset], str]] = {
    "newer-version": ("/", _with_attrs(schema_version=2), "Unsupported .* schema_version 2"),
    "wrong-schema": ("/", _with_attrs(schema="other"), "Expected schema"),
    "extra-root-attr": ("/", _with_attrs(note="x"), "Unexpected root attribute 'note'"),
    "bad-view": ("/", _with_attrs(observation_covariance_view="banded"), "Unsupported observation_covariance_view"),
    "bad-provenance": ("/", _with_attrs(provenance="[1]"), "provenance must be a JSON object"),
    "bad-axes-json": ("/products", _with_attrs(**{AXES_ATTR: "{"}), "must be a JSON object"),
    "float32": (
        "/products/state_covariance",
        lambda ds: ds.astype(np.float32),
        "must be stored as float64",
    ),
    "wrong-units": (
        "/products/effective_observation_operator",
        lambda ds: ds.assign(effective_observation_operator=ds["effective_observation_operator"].assign_attrs(units="ppb")),
        "linear in H must share one units",
    ),
    "extra-variable": (
        "/products/state_covariance",
        lambda ds: ds.assign(extra=ds["state_covariance"]),
        "Unexpected state_covariance variable 'extra'",
    ),
    "mirror-label": (
        "/products",
        lambda ds: ds.assign_coords(region_in_source_cov=("state_cov", ds["region_in_source_cov"].values[::-1])),
        "not an exact copy",
    ),
    "gathered-index": (
        "/products/restriction",
        lambda ds: ds.assign_coords({GATHERED_DIM: ds[GATHERED_DIM].copy(data=ds[GATHERED_DIM].values + 10**6)}),
        "within the restriction shape",
    ),
    "tampered-value": (
        "/products/effective_observation_operator",
        lambda ds: ds * 1.0000001,
        "content_id does not match",
    ),
    "covariance-binding": (
        "/covariance",
        lambda ds: ds.assign_coords(lat=ds["lat"] + 1.0),
        "Embedded covariance labels on 'lat'",
    ),
    "unknown-covariance": ("/covariance", _with_attrs(schema="other"), "Unsupported covariance configuration schema"),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_products_fail_clearly(case: str) -> None:
    """Malformed schema, labels, values and identities raise focused errors."""
    artifact, _, _ = _products_artifact("gathered-multisource")
    tree = products_to_datatree(artifact)
    products_from_datatree(tree)  # the unedited tree is valid
    path, edit, message = MALFORMED[case]
    edited = _edit(tree, path, edit)
    if case == "tampered-value":
        # Preserve units and attributes; change only the numbers.
        edited = _edit(
            tree,
            path,
            lambda ds: ds.assign(
                effective_observation_operator=ds["effective_observation_operator"].copy(
                    data=ds["effective_observation_operator"].values * 1.0000001
                )
            ),
        )
    with pytest.raises(ValueError, match=message):
        products_from_datatree(edited)


def test_missing_required_product_and_non_positive_c_alpha_fail() -> None:
    """A missing required group or an indefinite C_alpha fails before identity checks."""
    artifact, _, _ = _products_artifact("single-source")
    tree = products_to_datatree(artifact)
    del tree["products"]["state_covariance"]
    with pytest.raises(ValueError, match="Missing product group 'state_covariance'"):
        products_from_datatree(tree)

    tree = _edit(
        products_to_datatree(artifact),
        "/products/state_covariance",
        lambda ds: ds.assign(state_covariance=ds["state_covariance"].copy(data=-ds["state_covariance"].values)),
    )
    with pytest.raises(ValueError, match="positive definite"):
        products_from_datatree(tree)


def test_coherent_tree_rejects_tampering_and_wrong_family() -> None:
    """The coherent artifact verifies its root identity and approximation record."""
    tree = coherent_to_datatree(_coherent_artifact("single-source"))
    coherent_from_datatree(tree)
    with pytest.raises(ValueError, match="native_family"):
        coherent_from_datatree(_edit(tree, "/", _with_attrs(native_family="lognormal")))
    tampered = _edit(
        tree,
        "/reduction/observation_intercept",
        lambda ds: ds.assign(observation_intercept=ds["observation_intercept"].copy(data=ds["observation_intercept"].values + 1e-9)),
    )
    with pytest.raises(ValueError, match="content_id does not match"):
        coherent_from_datatree(tampered)


def test_identity_is_computed_from_content_not_cached() -> None:
    """Editing a product changes content_id; the artifact never holds a stale identity."""
    artifact, _, _ = _products_artifact("single-source")
    before = content_id(artifact)
    edited = NativeCovarianceProductsArtifact(
        products=NativeCovarianceProducts(
            **{name: getattr(artifact.products, name) for name in PRODUCT_NAMES if name != "state_covariance"},
            state_covariance=artifact.products.state_covariance.copy(data=artifact.products.state_covariance.values * 2.0),
            strategy=artifact.products.strategy,
            observation_covariance_view="dense",
        ),
        covariance=artifact.covariance,
        inputs_id=artifact.inputs_id,
    )
    assert content_id(edited) != before
    assert source_id(edited) == source_id(artifact)
    assert not hasattr(artifact, "content_id")


def test_save_does_not_mutate_borrowed_products(tmp_path: Path) -> None:
    """Encoding borrows products: their data, coordinates and attributes are unchanged."""
    artifact, _, _ = _products_artifact("datetime-multiindex")
    snapshots = {name: getattr(artifact.products, name).copy(deep=True) for name in PRODUCT_NAMES}
    save_products(artifact, tmp_path / "products.nc")
    for name, snapshot in snapshots.items():
        xr.testing.assert_identical(getattr(artifact.products, name), snapshot)
