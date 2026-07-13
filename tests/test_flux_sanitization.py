import numpy as np
from pathlib import Path
import pytest
import xarray as xr

from openghg_inversions.flux_sanitization import (
    NONFINITE_CHECKED_COMPUTED,
    NONFINITE_CHECKED_NOT_COUNTED,
    NONFINITE_METADATA_ATTR,
    NONFINITE_POLICY_ZERO_FILL,
    FluxNonFiniteMetadata,
    NonFiniteFluxWarning,
    copy_flux_nonfinite_attrs,
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


def _metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


def test_sanitize_flux_nonfinite_lazy_preserves_shape_dtype_and_attrs() -> None:
    """Lazy sanitation replaces non-finite values and records the policy without counts."""
    flux = _nonfinite_flux()

    sanitized = sanitize_flux_nonfinite(flux, context="unit test")
    metadata = _metadata(sanitized)

    assert sanitized.name == "flux"
    assert sanitized.dtype == np.dtype("float32")
    xr.testing.assert_equal(sanitized.lat, flux.lat)
    xr.testing.assert_equal(sanitized.lon, flux.lon)
    np.testing.assert_allclose(sanitized.values, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert sanitized.attrs["units"] == "mol/m2/s"
    assert isinstance(sanitized.attrs[NONFINITE_METADATA_ATTR], str)
    assert metadata.schema_version == 1
    assert metadata.policy == NONFINITE_POLICY_ZERO_FILL
    assert metadata.fill_value == 0.0
    assert metadata.checked == NONFINITE_CHECKED_NOT_COUNTED
    assert metadata.context == "unit test"
    assert metadata.count is None
    assert "history" in sanitized.attrs


def test_sanitize_flux_nonfinite_is_idempotent_when_attrs_are_trusted() -> None:
    """A second trusted call returns the same object without rebuilding the graph."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="unit test")

    again = sanitize_flux_nonfinite(sanitized, context="second pass")

    assert again is sanitized


def test_sanitize_flux_nonfinite_count_does_not_invent_counts_after_lazy_fill() -> None:
    """Count mode preserves lazy metadata when original non-finite values are gone."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="unit test")

    with pytest.warns(NonFiniteFluxWarning, match="original non-finite count cannot be recovered"):
        audited = sanitize_flux_nonfinite(sanitized, context="audit test", check="count")
    metadata = _metadata(audited)

    assert audited is sanitized
    assert metadata.checked == NONFINITE_CHECKED_NOT_COUNTED
    assert metadata.count is None
    assert metadata.total is None
    assert metadata.fraction is None


def test_sanitize_flux_nonfinite_count_records_exact_metadata() -> None:
    """Count mode computes exact non-finite count metadata and warns when needed."""
    flux = _nonfinite_flux()

    with pytest.warns(NonFiniteFluxWarning, match="contains 3 non-finite values"):
        sanitized = sanitize_flux_nonfinite(flux, context="audit test", check="count", warn=True)
    metadata = _metadata(sanitized)

    assert metadata.checked == NONFINITE_CHECKED_COMPUTED
    assert metadata.count == 3
    assert metadata.total == 4
    assert metadata.fraction == 0.75


def test_sanitize_flux_nonfinite_count_metadata_is_idempotent() -> None:
    """Repeated count mode trusts an exact audit without rescanning the data."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="audit test", check="count")

    again = sanitize_flux_nonfinite(sanitized, context="second audit", check="count")

    assert again is sanitized
    assert _metadata(again).count == 3


def test_sanitize_flux_nonfinite_json_metadata_survives_netcdf_roundtrip(tmp_path: Path) -> None:
    """The canonical metadata attr is a string that survives NetCDF serialization."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="roundtrip test", check="count")
    path = tmp_path / "flux.nc"

    sanitized.to_dataset().to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as loaded:
        loaded_metadata = _metadata(loaded["flux"])

    assert isinstance(sanitized.attrs[NONFINITE_METADATA_ATTR], str)
    assert loaded_metadata.policy == NONFINITE_POLICY_ZERO_FILL
    assert loaded_metadata.checked == NONFINITE_CHECKED_COMPUTED
    assert loaded_metadata.count == 3
    assert loaded_metadata.total == 4


def test_copy_flux_nonfinite_attrs_copies_json_metadata() -> None:
    """Canonical JSON metadata is copied while preserving existing target attrs."""
    target = xr.Dataset(attrs={"title": "output"})
    flux = xr.DataArray(
        [1.0],
        attrs={NONFINITE_METADATA_ATTR: FluxNonFiniteMetadata(context="copy test").to_json()},
    )

    result = copy_flux_nonfinite_attrs(target, flux)
    metadata = _metadata(result)

    assert result.attrs["title"] == "output"
    assert metadata.policy == NONFINITE_POLICY_ZERO_FILL
    assert metadata.context == "copy test"
