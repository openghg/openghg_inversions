import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.algorithms._weighted import (
    bucket_split_landsea_basis,
    load_landsea_indices,
)


def _assert_labels_do_not_cross_classes(labels: np.ndarray, classes: np.ndarray) -> None:
    for label in np.unique(labels):
        if label == 0:
            continue
        label_classes = np.unique(classes[labels == label])
        assert len(label_classes) == 1, f"label {label} crosses classes {label_classes}"


def test_weighted_landsea_basis_labels_do_not_cross_landsea_classes(tmp_path):
    """Regression test for #318: weighted land/sea labels stay class-local."""
    landsea = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ]
    )
    xr.Dataset({"country": (("lat", "lon"), landsea)}).to_netcdf(tmp_path / "country-land-sea_TEST.nc")

    grid = np.array(
        [
            [8.0, 7.0, 5.0, 5.0],
            [6.0, 4.0, 4.0, 3.0],
            [3.0, 3.0, 3.0, 6.0],
            [1.0, 1.0, 7.0, 8.0],
        ]
    )

    labels = bucket_split_landsea_basis(
        grid,
        bucket=12.0,
        domain="TEST",
        country_directory=str(tmp_path),
    )

    assert labels.shape == grid.shape
    assert labels.min() > 0
    _assert_labels_do_not_cross_classes(labels, landsea)


def test_weighted_basis_requires_a_matching_domain_landsea_file() -> None:
    with pytest.raises(FileNotFoundError, match="basis_algorithm='quadtree'"):
        load_landsea_indices("NESTED-DOMAIN-WITHOUT-A-PACKAGED-MASK")


def test_weighted_basis_rejects_landsea_shape_mismatch(tmp_path) -> None:
    xr.Dataset({"country": (("lat", "lon"), np.ones((2, 2), dtype=int))}).to_netcdf(
        tmp_path / "country-land-sea_NESTED.nc"
    )

    with pytest.raises(ValueError, match=r"Land-sea mask shape \(2, 2\).*grid shape \(3, 3\)"):
        bucket_split_landsea_basis(
            np.ones((3, 3)),
            bucket=1.0,
            domain="NESTED",
            country_directory=str(tmp_path),
        )
