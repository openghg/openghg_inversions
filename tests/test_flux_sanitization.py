import numpy as np
import pytest
import xarray as xr

from openghg_inversions.flux_sanitization import (
    NONFINITE_CHECKED_ATTR,
    NONFINITE_CHECKED_COMPUTED,
    NONFINITE_CHECKED_NOT_COUNTED,
    NONFINITE_COUNT_ATTR,
    NONFINITE_FILL_VALUE_ATTR,
    NONFINITE_FRACTION_ATTR,
    NONFINITE_POLICY_ATTR,
    NONFINITE_POLICY_ZERO_FILL,
    NONFINITE_TOTAL_ATTR,
    NonFiniteFluxWarning,
    sanitize_flux_nonfinite,
)


def _nonfinite_flux() -> xr.DataArray:
    """Build a tiny flux field containing NaN and infinite values."""
    return xr.DataArray(
        np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0], "lon": [1.0, 2.0]},
        name="flux",
        attrs={"units": "mol/m2/s"},
    )


def test_sanitize_flux_nonfinite_lazy_preserves_shape_dtype_and_attrs() -> None:
    """Lazy sanitation replaces non-finite values and records the policy without counts."""
    flux = _nonfinite_flux()

    sanitized = sanitize_flux_nonfinite(flux, context="unit test")

    assert sanitized.name == "flux"
    assert sanitized.dtype == np.dtype("float32")
    xr.testing.assert_equal(sanitized.lat, flux.lat)
    xr.testing.assert_equal(sanitized.lon, flux.lon)
    np.testing.assert_allclose(sanitized.values, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert sanitized.attrs["units"] == "mol/m2/s"
    assert sanitized.attrs[NONFINITE_POLICY_ATTR] == NONFINITE_POLICY_ZERO_FILL
    assert sanitized.attrs[NONFINITE_FILL_VALUE_ATTR] == 0.0
    assert sanitized.attrs[NONFINITE_CHECKED_ATTR] == NONFINITE_CHECKED_NOT_COUNTED
    assert NONFINITE_COUNT_ATTR not in sanitized.attrs
    assert "history" in sanitized.attrs


def test_sanitize_flux_nonfinite_is_idempotent_when_attrs_are_trusted() -> None:
    """A second trusted call returns the same object without rebuilding the graph."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="unit test")

    again = sanitize_flux_nonfinite(sanitized, context="second pass")

    assert again is sanitized


def test_sanitize_flux_nonfinite_count_audits_lazy_metadata() -> None:
    """Count mode computes metadata even when lazy zero-fill attrs are present."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="unit test")

    audited = sanitize_flux_nonfinite(sanitized, context="audit test", check="count")

    assert audited is not sanitized
    assert audited.attrs[NONFINITE_CHECKED_ATTR] == NONFINITE_CHECKED_COMPUTED
    assert audited.attrs[NONFINITE_COUNT_ATTR] == 0
    assert audited.attrs[NONFINITE_TOTAL_ATTR] == 4
    assert audited.attrs[NONFINITE_FRACTION_ATTR] == 0.0


def test_sanitize_flux_nonfinite_count_records_exact_metadata() -> None:
    """Count mode computes exact non-finite count metadata and warns when needed."""
    flux = _nonfinite_flux()

    with pytest.warns(NonFiniteFluxWarning, match="contains 3 non-finite values"):
        sanitized = sanitize_flux_nonfinite(flux, context="audit test", check="count", warn=True)

    assert sanitized.attrs[NONFINITE_CHECKED_ATTR] == NONFINITE_CHECKED_COMPUTED
    assert sanitized.attrs[NONFINITE_COUNT_ATTR] == 3
    assert sanitized.attrs[NONFINITE_TOTAL_ATTR] == 4
    assert sanitized.attrs[NONFINITE_FRACTION_ATTR] == 0.75
