"""Unit conversion helpers for merged inversion observations."""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import xarray as xr
from openghg.util import (
    assign_units,  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
    cf_ureg,  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
)

_MOLE_FRACTION_UNITS = (
    (1.0, "mol/mol"),
    (1e-6, "ppm"),
    (1e-9, "ppb"),
    (1e-12, "ppt"),
    (1e-15, "ppq"),
)
_OBSERVATION_VARIABLES = {
    "mf",
    "mf_error",
    "mf_mod",
    "mf_prior_factor",
    "mf_prior_upper_level_factor",
    "mf_repeatability",
    "mf_variability",
}


def _mole_fraction_scale(raw_units: Any, *, context: str) -> float:
    """Return the multiplicative scale of observation units against mol/mol."""
    if isinstance(raw_units, Real) and not isinstance(raw_units, bool):
        quantity = cf_ureg.Quantity(float(raw_units), "mol/mol")
    else:
        try:
            raw_text = str(raw_units).strip()
            if raw_text.lower() in {"ppm", "ppb", "ppt", "ppq"}:
                raw_text = raw_text.lower()
            parsed = cf_ureg.parse_expression(raw_text)
            quantity = cf_ureg.Quantity(float(parsed), "mol/mol") if isinstance(parsed, Real) else parsed
        except Exception as exc:
            raise ValueError(f"Could not parse observation units {raw_units!r} for {context}.") from exc

    try:
        return float(quantity.to("mol/mol").magnitude)
    except Exception as exc:
        raise ValueError(
            f"Observation units {raw_units!r} for {context} are not compatible with a molar mixing ratio."
        ) from exc


def _canonical_mole_fraction_unit(raw_units: Any, *, context: str) -> tuple[str, float]:
    """Return a Pint-compatible unit name and legacy numeric unit scale."""
    scale = _mole_fraction_scale(raw_units, context=context)
    for expected_scale, unit_name in _MOLE_FRACTION_UNITS:
        if np.isclose(scale, expected_scale, rtol=1e-12, atol=0.0):
            return unit_name, expected_scale

    # Pint unit attributes cannot contain an arbitrary scale factor. Convert
    # uncommon mole-fraction scales to the unscaled mol/mol representation.
    return "mol/mol", 1.0


def _variables_to_align(dataset: xr.Dataset) -> list[str]:
    """Return concentration-valued time variables aligned by OpenGHG."""
    return [
        str(name)
        for name, variable in dataset.data_vars.items()
        if "time" in variable.dims
        and (
            str(name) in _OBSERVATION_VARIABLES
            or str(name).startswith("mf_mod")
            or str(name).startswith("bc_")
            or str(name).startswith("fp_x_flux")
        )
    ]


def _pint_compatible_dataset(
    dataset: xr.Dataset,
    variable_names: Sequence[str],
    *,
    fallback_units: Any,
) -> xr.Dataset:
    """Copy selected variables with Pint-compatible OpenGHG unit attributes."""
    compatible = dataset[list(variable_names)].copy()
    for name in variable_names:
        raw_units = dataset[name].attrs.get("units", fallback_units)
        source_scale = _mole_fraction_scale(raw_units, context=f"variable {name!r}")
        source_unit, canonical_scale = _canonical_mole_fraction_unit(
            raw_units,
            context=f"variable {name!r}",
        )

        if not np.isclose(source_scale, canonical_scale, rtol=1e-12, atol=0.0):
            compatible[name] = compatible[name] * (source_scale / canonical_scale)
        compatible[name].attrs = dataset[name].attrs.copy()
        compatible[name].attrs["units"] = source_unit
    return compatible


def align_observation_units(
    fp_all: Mapping[str, Any],
    sites: Sequence[str],
    *,
    require_units: bool = False,
) -> dict[str, Any]:
    """Convert retained site datasets to the first site's observation units.

    OpenGHG's Pint registry validates each unit and ``assign_units`` performs
    lazy xarray conversion. A defined schema covers observations, errors,
    column-prior factors, modelled concentrations, boundary terms, and
    footprint-times-flux fields. Missing per-variable units inherit the site's
    ``mf`` units. Unrelated footprint, meteorology, coordinates, and metadata
    are preserved.

    Args:
        fp_all: Merged site datasets and dot-prefixed metadata.
        sites: Retained site names in canonical observation order.
        require_units: Require unit metadata from every retained dataset.
            Legacy data may instead provide one numeric ``.units`` fallback
            when all site ``mf`` attributes are missing.

    Returns:
        A shallow copy of ``fp_all`` with copied, unit-aligned site datasets
        and a numeric legacy ``.units`` scale.

    Raises:
        ValueError: If a retained site is not an xarray Dataset, lacks ``mf``
            or required unit metadata, or contains non-mole-fraction units in
            a concentration-valued variable.
    """
    aligned = dict(fp_all)
    if not sites:
        return aligned

    raw_mf_units: dict[str, Any] = {}
    for site in sites:
        dataset = fp_all.get(site)
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(f"Retained site {site!r} does not contain an xarray Dataset.")
        if "mf" not in dataset:
            raise ValueError(f"Retained site {site!r} is missing required observation variable `mf`.")
        raw_units = dataset["mf"].attrs.get("units")
        if raw_units is not None:
            raw_mf_units[site] = raw_units

    missing_units = [site for site in sites if site not in raw_mf_units]
    if missing_units:
        if raw_mf_units:
            raise ValueError(f"Retained observation sites are missing `mf` unit metadata: {missing_units!r}.")
        legacy_units = fp_all.get(".units")
        if legacy_units is None:
            if require_units:
                raise ValueError(
                    f"Retained observation sites are missing `mf` unit metadata: {missing_units!r}."
                )
            return aligned
        raw_mf_units = dict.fromkeys(sites, legacy_units)

    target_site = sites[0]
    target_unit, target_scale = _canonical_mole_fraction_unit(
        raw_mf_units[target_site],
        context=f"site {target_site!r} variable 'mf'",
    )

    for site in sites:
        dataset = fp_all[site]
        variable_names = _variables_to_align(dataset)
        compatible = _pint_compatible_dataset(
            dataset,
            variable_names,
            fallback_units=raw_mf_units[site],
        )
        try:
            converted = assign_units(
                compatible,
                target_units={name: target_unit for name in variable_names},
            )
        except Exception as exc:
            raise ValueError(
                f"Could not convert observation-valued variables for site {site!r} to {target_unit!r}."
            ) from exc

        site_aligned = dataset.copy()
        for name in variable_names:
            site_aligned[name] = converted[name]
        aligned[site] = site_aligned

    aligned[".units"] = target_scale
    return aligned
