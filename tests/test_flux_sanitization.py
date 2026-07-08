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
    attrs_declare_zero_filled_nonfinite,
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


def test_sanitize_flux_nonfinite_count_audits_lazy_metadata() -> None:
    """Count mode computes metadata even when lazy zero-fill attrs are present."""
    sanitized = sanitize_flux_nonfinite(_nonfinite_flux(), context="unit test")

    audited = sanitize_flux_nonfinite(sanitized, context="audit test", check="count")
    metadata = _metadata(audited)

    assert audited is not sanitized
    assert metadata.checked == NONFINITE_CHECKED_COMPUTED
    assert metadata.count == 0
    assert metadata.total == 4
    assert metadata.fraction == 0.0


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


def test_copy_flux_nonfinite_attrs_normalises_legacy_field_attrs() -> None:
    """Legacy per-field metadata is read but copied as the canonical JSON attr."""
    target = xr.Dataset(attrs={"title": "output"})
    legacy_flux = xr.DataArray(
        [1.0],
        attrs={
            "openghg_inversions:non_finite_policy": "zero_fill",
            "openghg_inversions:non_finite_fill_value": 0.0,
            "openghg_inversions:non_finite_checked": "computed",
            "openghg_inversions:non_finite_count": 2,
            "openghg_inversions:non_finite_total": 10,
            "openghg_inversions:non_finite_fraction": 0.2,
        },
    )

    result = copy_flux_nonfinite_attrs(target, legacy_flux)
    metadata = _metadata(result)

    assert result.attrs["title"] == "output"
    assert metadata.policy == NONFINITE_POLICY_ZERO_FILL
    assert metadata.checked == NONFINITE_CHECKED_COMPUTED
    assert metadata.count == 2
    assert metadata.total == 10
    assert metadata.fraction == 0.2
    assert "openghg_inversions:non_finite_policy" not in result.attrs


def test_attrs_declare_zero_filled_nonfinite_accepts_json_metadata() -> None:
    """Trusted attrs include the canonical JSON metadata attr."""
    metadata = FluxNonFiniteMetadata(context="unit test")

    assert attrs_declare_zero_filled_nonfinite({NONFINITE_METADATA_ATTR: metadata.to_json()})
