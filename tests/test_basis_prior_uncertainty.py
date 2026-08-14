"""Tests for basis-aware prior uncertainty projection and calibration."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import (
    calibrate_basis_prior_stdev,
    project_basis_prior_stdev,
)
from openghg_inversions.basis.basis_functions import BasisFunctions


def _grid(values: list[list[float | int]], *, name: str) -> xr.DataArray:
    """Return a small labelled grid."""
    array = np.asarray(values)
    return xr.DataArray(
        array,
        dims=("lat", "lon"),
        coords={"lat": np.arange(array.shape[0]), "lon": np.arange(array.shape[1])},
        name=name,
    )


def _basis_functions(
    basis: xr.DataArray,
    flux: xr.DataArray,
    *,
    state_dim: str = "region",
) -> BasisFunctions:
    """Build retained basis functions for one grid."""
    return BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": state_dim},
    )


def test_project_equal_cells_recovers_one_over_sqrt_n() -> None:
    """Equal cell weights produce the expected one-over-root-n state width."""
    basis = _grid([[1, 1, 2, 2]], name="basis").astype(int)
    flux = _grid([[1.0, 1.0, 1.0, 1.0]], name="flux")
    area = xr.ones_like(flux).rename("area")

    projected = project_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=area,
        grid_cell_prior_stdev=0.8,
    )

    assert projected.name == "x_prior_stdev"
    assert projected.dims == ("region",)
    np.testing.assert_allclose(projected, [0.8 / np.sqrt(2), 0.8 / np.sqrt(2)])


def test_project_unequal_weights_uses_flux_area_totals() -> None:
    """Unequal flux-area weights follow the weighted Gaussian moment formula."""
    basis = _grid([[1, 1]], name="basis").astype(int)
    flux = _grid([[1.0, 2.0]], name="flux")
    area = _grid([[1.0, 3.0]], name="area")

    projected = project_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=area,
        grid_cell_prior_stdev=0.6,
    )

    expected = 0.6 * np.sqrt(1.0**2 + 6.0**2) / 7.0
    np.testing.assert_allclose(projected, [expected])


def test_project_scales_large_finite_weights_before_squaring() -> None:
    """Finite large weights do not overflow the relative-moment calculation."""
    basis = _grid([[1, 1]], name="basis").astype(int)
    flux = _grid([[1.0e200, 1.0e200]], name="flux")

    projected = project_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=xr.ones_like(flux),
        grid_cell_prior_stdev=1.0,
    ).compute()

    assert np.isfinite(projected).all()
    np.testing.assert_allclose(projected, [1.0 / np.sqrt(2)])


def test_project_rejects_negative_grid_cell_stdev() -> None:
    basis = _grid([[1]], name="basis").astype(int)
    flux = _grid([[1.0]], name="flux")

    with pytest.raises(ValueError, match="non-negative"):
        project_basis_prior_stdev(
            _basis_functions(basis, flux),
            area_grid=xr.ones_like(flux),
            grid_cell_prior_stdev=-0.1,
        )


def test_project_ragged_multisource_preserves_operator_order() -> None:
    """Ragged source states are selected by label and gathered in operator order."""
    basis_b = _grid([[1, 1]], name="basis").astype(int)
    basis_a = _grid([[1, 2]], name="basis").astype(int)
    flux = xr.concat(
        [
            _grid([[1.0, 1.0]], name="flux").expand_dims(source=["A"]),
            _grid([[1.0, 1.0]], name="flux").expand_dims(source=["B"]),
        ],
        dim="source",
    )
    basis_functions = BasisFunctions.from_multi_source_flat_basis(
        {"B": basis_b, "A": basis_a},
        flux,
        operator_kwargs={"state_dim": "state"},
    )
    source_stdev = xr.DataArray(
        [2.0, 1.0],
        dims="source",
        coords={"source": ["A", "B"]},
    )

    projected = project_basis_prior_stdev(
        basis_functions,
        area_grid=xr.ones_like(basis_a, dtype=float),
        grid_cell_prior_stdev=source_stdev,
    )

    assert projected.dims == ("state",)
    assert projected.indexes["state"].tolist() == [("B", 0), ("A", 0), ("A", 1)]
    np.testing.assert_allclose(projected, [1.0 / np.sqrt(2), 2.0, 2.0])


def test_calibrate_target_relative_stdev_returns_achieved_diagnostics() -> None:
    """Unit projection linearity calibrates an aggregate target exactly."""
    basis = _grid([[1, 1, 2, 2]], name="basis").astype(int)
    flux = _grid([[1.0, 1.0, 1.0, 1.0]], name="flux")
    area = xr.ones_like(flux).rename("area")
    target = xr.ones_like(flux).rename("target")

    calibrated = calibrate_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=area,
        target_matrix=target,
        target_relative_stdev=0.25,
    )

    np.testing.assert_allclose(calibrated["grid_cell_prior_stdev"], 0.5)
    np.testing.assert_allclose(calibrated["x_prior_stdev"], [0.5 / np.sqrt(2)] * 2)
    np.testing.assert_allclose(calibrated["state_total"], [2.0, 2.0])
    np.testing.assert_allclose(calibrated["target_state_total"], [2.0, 2.0])
    np.testing.assert_allclose(calibrated["target_total"], 4.0)
    np.testing.assert_allclose(calibrated["achieved_target_stdev"], 1.0)
    np.testing.assert_allclose(calibrated["achieved_target_relative_stdev"], 0.25)
    assert calibrated["target_status"].compute().item() == "ok"
    assert calibrated["calibration_status"].compute().item() == "ok"


def test_calibrate_mean_total_with_dask_backed_targets() -> None:
    """Mean-total calibration supports normal dask-backed multi-target arrays."""
    basis = _grid([[1, 1, 2, 2]], name="basis").astype(int).chunk({"lon": 2})
    flux = _grid([[1.0, 1.0, 1.0, 1.0]], name="flux").chunk({"lon": 2})
    target = xr.DataArray(
        [[[1.0, 1.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]]],
        dims=("target", "lat", "lon"),
        coords={"target": ["small", "all"], "lat": [0], "lon": np.arange(4)},
    ).chunk({"target": 1, "lon": 2})

    calibrated = calibrate_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=xr.ones_like(flux),
        target_matrix=target,
        target_relative_stdev=0.5,
        target_statistic="mean-total",
    ).compute()

    achieved_ratio = float(calibrated["achieved_target_stdev"].mean()) / float(
        np.abs(calibrated["target_total"]).mean()
    )
    np.testing.assert_allclose(achieved_ratio, 0.5)
    assert calibrated["target_statistic"].item() == "mean-total"


def test_calibrate_excludes_inactive_states_from_target_uncertainty() -> None:
    """A labelled activity mask makes calibration match the sampled state prior."""
    basis = _grid([[1, 1, 2, 2]], name="basis").astype(int)
    flux = _grid([[1.0, 1.0, 1.0, 1.0]], name="flux")
    basis_functions = _basis_functions(basis, flux)
    state_coord = basis_functions.operator.basis_matrix.coords["region"]
    active = xr.DataArray(
        [False, True],
        dims="region",
        coords={"region": state_coord.values[::-1]},
    )

    calibrated = calibrate_basis_prior_stdev(
        basis_functions,
        area_grid=xr.ones_like(flux),
        target_matrix=xr.ones_like(flux),
        target_relative_stdev=0.25,
        state_is_active=active,
    ).compute()

    np.testing.assert_array_equal(calibrated["state_is_active"], [True, False])
    np.testing.assert_allclose(calibrated["x_prior_stdev"].isel(region=1), 0.0)
    np.testing.assert_allclose(calibrated["achieved_target_relative_stdev"], 0.25)


def test_calibrate_rejects_target_dimension_aliasing_state() -> None:
    basis = _grid([[1, 1]], name="basis").astype(int)
    flux = _grid([[1.0, 1.0]], name="flux")
    target = xr.ones_like(flux).expand_dims(region=["target"])

    with pytest.raises(ValueError, match="aliases the basis state dimension"):
        calibrate_basis_prior_stdev(
            _basis_functions(basis, flux),
            area_grid=xr.ones_like(flux),
            target_matrix=target,
            target_relative_stdev=0.5,
        )


def test_calibrate_rejects_zero_target_width() -> None:
    basis = _grid([[1]], name="basis").astype(int)
    flux = _grid([[1.0]], name="flux")

    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_basis_prior_stdev(
            _basis_functions(basis, flux),
            area_grid=xr.ones_like(flux),
            target_matrix=xr.ones_like(flux),
            target_relative_stdev=0.0,
        )


def test_calibrate_rejects_active_state_with_zero_projected_width() -> None:
    basis = _grid([[1]], name="basis").astype(int)
    flux = _grid([[0.0]], name="flux")
    active = xr.DataArray([True], dims="region", coords={"region": [0]})

    with pytest.raises(ValueError, match="zero or non-finite"):
        calibrate_basis_prior_stdev(
            _basis_functions(basis, flux),
            area_grid=xr.ones_like(flux),
            target_matrix=xr.ones_like(flux),
            target_relative_stdev=0.5,
            state_is_active=active,
        )


def test_calibrate_shared_basis_selects_source_targets_by_label() -> None:
    """Shared-basis calibration retains flux order while aligning target labels."""
    basis = _grid([[1, 1]], name="basis").astype(int)
    flux = xr.concat(
        [
            _grid([[1.0, 1.0]], name="flux").expand_dims(source=["B"]),
            _grid([[2.0, 2.0]], name="flux").expand_dims(source=["A"]),
        ],
        dim="source",
    )
    requested = xr.DataArray(
        [0.25, 0.5],
        dims="source",
        coords={"source": ["A", "B"]},
    )

    calibrated = calibrate_basis_prior_stdev(
        _basis_functions(basis, flux),
        area_grid=xr.ones_like(basis, dtype=float),
        target_matrix=xr.ones_like(basis, dtype=float),
        target_relative_stdev=requested,
    ).compute()

    assert calibrated["x_prior_stdev"].dims == ("source", "region")
    assert calibrated["source"].values.tolist() == ["B", "A"]
    np.testing.assert_allclose(calibrated["x_prior_stdev"].sel(source="A"), [0.25])
    np.testing.assert_allclose(calibrated["x_prior_stdev"].sel(source="B"), [0.5])
    np.testing.assert_allclose(
        calibrated["achieved_target_relative_stdev"],
        [0.5, 0.25],
    )


def test_calibrate_ragged_multisource_keeps_gathered_state_coordinate() -> None:
    """Ragged calibration separates source diagnostics from the state MultiIndex."""
    basis_b = _grid([[1, 1]], name="basis").astype(int)
    basis_a = _grid([[1, 2]], name="basis").astype(int)
    flux = xr.concat(
        [
            _grid([[1.0, 1.0]], name="flux").expand_dims(source=["A"]),
            _grid([[1.0, 1.0]], name="flux").expand_dims(source=["B"]),
        ],
        dim="source",
    )
    basis_functions = BasisFunctions.from_multi_source_flat_basis(
        {"B": basis_b, "A": basis_a},
        flux,
        operator_kwargs={"state_dim": "state"},
    )
    state_index = basis_functions.operator.basis_matrix.coords["state"].to_index()
    state_is_active = xr.DataArray(
        [True, False, True],
        dims="state",
        coords={"state": state_index},
    ).isel(state=[2, 0, 1])

    calibrated = calibrate_basis_prior_stdev(
        basis_functions,
        area_grid=xr.ones_like(basis_a, dtype=float),
        target_matrix=xr.ones_like(basis_a, dtype=float),
        target_relative_stdev=xr.DataArray(
            [0.25, 0.5],
            dims="source",
            coords={"source": ["A", "B"]},
        ),
        state_is_active=state_is_active,
    ).compute()

    assert calibrated["x_prior_stdev"].indexes["state"].tolist() == [
        ("B", 0),
        ("A", 0),
        ("A", 1),
    ]
    assert calibrated["calibration_source"].values.tolist() == ["B", "A"]
    np.testing.assert_array_equal(calibrated["state_is_active"], [True, False, True])
    np.testing.assert_allclose(calibrated["x_prior_stdev"].isel(state=1), 0.0)
    np.testing.assert_allclose(
        calibrated["state_prior_stdev_numerator"].isel(state=1),
        0.0,
    )
    np.testing.assert_allclose(
        calibrated["achieved_target_relative_stdev"],
        [0.5, 0.25],
    )

    with pytest.raises(ValueError, match="aliases the basis state dimension"):
        calibrate_basis_prior_stdev(
            basis_functions,
            area_grid=xr.ones_like(basis_a, dtype=float),
            target_matrix=xr.ones_like(basis_a, dtype=float).expand_dims(state=["target"]),
            target_relative_stdev=0.5,
        )


def test_calibrate_distinguishes_zero_from_signed_cancellation() -> None:
    """Degenerate target statuses do not silently produce fallback widths."""
    basis = _grid([[1, 1]], name="basis").astype(int)
    area = _grid([[1.0, 1.0]], name="area")
    target = _grid([[1.0, 1.0]], name="target")

    zero_flux = _grid([[0.0, 0.0]], name="flux")
    zero = calibrate_basis_prior_stdev(
        _basis_functions(basis, zero_flux),
        area_grid=area,
        target_matrix=target,
        target_relative_stdev=0.5,
    )
    assert zero["target_status"].compute().item() == "zero"
    assert zero["calibration_status"].compute().item() == "zero"
    assert np.isnan(zero["grid_cell_prior_stdev"])

    cancelling_flux = _grid([[1.0, -1.0]], name="flux")
    cancellation = calibrate_basis_prior_stdev(
        _basis_functions(basis, cancelling_flux),
        area_grid=area,
        target_matrix=target,
        target_relative_stdev=0.5,
    )
    assert cancellation["target_status"].compute().item() == "cancellation"
    assert cancellation["calibration_status"].compute().item() == "cancellation"
    assert np.isnan(cancellation["grid_cell_prior_stdev"])
    assert not cancellation["state_is_active"].any().compute().item()
    assert (cancellation["x_prior_stdev"] == 0).all().compute().item()
    assert (cancellation["state_prior_stdev_numerator"] == 0).all().compute().item()
