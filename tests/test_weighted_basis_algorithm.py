import numpy as np
import xarray as xr

from openghg_inversions.basis.algorithms._weighted import bucket_split_landsea_basis, load_landsea_indices


def _assert_labels_do_not_cross_classes(labels: np.ndarray, classes: np.ndarray) -> None:
    for label in np.unique(labels):
        if label == 0:
            continue
        label_classes = np.unique(classes[labels == label])
        assert len(label_classes) == 1, f"label {label} crosses classes {label_classes}"


def test_centralasia_landsea_grid_is_packaged():
    assert load_landsea_indices("CENTRALASIA").shape == (310, 330)


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
