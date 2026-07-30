"""Align observation units across retained inversion sites.

Fresh retrieval delegates conversion to OpenGHG ``ModelScenario`` objects.
This module handles reloaded/direct site datasets with the xarray Pint
accessor, using the first retained site as the target without mutating inputs.
Legacy numeric ``.units`` metadata is only a fallback when every site lacks
``mf`` unit attributes.
"""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import xarray as xr
from openghg.util import (
    cf_ureg,  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
)

_LEGACY_SCALE_UNITS = {
    1.0: "mol/mol",
    1e-6: "ppm",
    1e-9: "ppb",
    1e-12: "ppt",
    1e-15: "ppq",
}
_LEGACY_SCALED_UNIT_NAMES = {
    "1e-6": "ppm",
    "1e-06": "ppm",
    "1e-9": "ppb",
    "1e-09": "ppb",
    "1e-12": "ppt",
    "1e-15": "ppq",
}
_OBSERVATION_UNIT_MARKERS = ("mf", "bc_", "fp_x_flux")
_UNITLESS_OBSERVATION_METADATA = {"mf_number_of_observations"}
_LEGACY_UNIT_INHERITANCE = {
    "mf",
    "mf_error",
    "mf_prior_factor",
    "mf_prior_upper_level_factor",
    "mf_repeatability",
    "mf_variability",
}
_LEGACY_UNIT_INHERITANCE_PREFIXES = ("mf_mod", "bc_", "fp_x_flux")


def _pint_unit_name(raw_units: Any, *, context: str) -> str:
    """Return a Pint-compatible unit name.

    Args:
        raw_units: Unit string or supported standard numeric legacy scale.
        context: Description included in validation errors.

    Returns:
        A stripped unit string, with standard legacy scales normalized.

    Raises:
        ValueError: If the value is empty or is an unsupported numeric scale.
    """
    if isinstance(raw_units, Real | np.floating) and not isinstance(raw_units, bool):
        for scale, unit_name in _LEGACY_SCALE_UNITS.items():
            if np.isclose(float(raw_units), scale, rtol=1e-6, atol=0.0):
                return unit_name
        raise ValueError(f"Unsupported numeric observation-unit scale {raw_units!r} for {context}.")
    if isinstance(raw_units, str) and raw_units.strip():
        unit_name = raw_units.strip()
        scaled_unit = " ".join(unit_name.lower().split()).removesuffix(" mol/mol")
        return _LEGACY_SCALED_UNIT_NAMES.get(scaled_unit, unit_name)
    raise ValueError(f"Invalid observation units {raw_units!r} for {context}.")


def mole_fraction_unit_scale(raw_units: Any, *, context: str) -> float:
    """Return a mole-fraction unit's multiplicative scale against mol/mol.

    Args:
        raw_units: Pint-compatible unit string or standard numeric legacy scale.
        context: Description included in validation errors.

    Returns:
        The dimensionless scale relative to ``mol/mol``.

    Raises:
        ValueError: If units are invalid, unsupported, or dimensionally
            incompatible with a molar mixing ratio.
    """
    unit_name = _pint_unit_name(raw_units, context=context)
    try:
        return float(cf_ureg.Quantity(1.0, unit_name).to("mol/mol").magnitude)
    except Exception as exc:
        raise ValueError(
            f"Observation units {raw_units!r} for {context} are not compatible with a molar mixing ratio."
        ) from exc


def _observation_variables(dataset: xr.Dataset) -> list[str]:
    """Return unit-bearing time variables selected for reload-time alignment.

    Args:
        dataset: Per-site merged scenario dataset.

    Returns:
        Variables matching OpenGHG's ``mf``/``bc_``/``fp_x_flux`` convention,
        excluding known unitless observation metadata.
    """
    return [
        str(name)
        for name, variable in dataset.data_vars.items()
        if "time" in variable.dims
        and variable.attrs.get("units") is not None
        and any(marker in str(name) for marker in _OBSERVATION_UNIT_MARKERS)
        and str(name) not in _UNITLESS_OBSERVATION_METADATA
    ]


def align_observation_units(
    fp_all: Mapping[str, Any],
    sites: Sequence[str],
    *,
    require_units: bool = False,
) -> dict[str, Any]:
    """Align reloaded site datasets to the first retained site's units.

    Freshly retrieved scenarios already share units through
    ``ModelScenario.footprints_data_merge(output_units=...)``. This coordinator
    covers reloaded or directly supplied ``fp_all`` mappings, where each site is
    a separate dataset and no ``ModelScenario`` object remains.

    Numeric ``fp_all[".units"]`` is accepted only as a fallback for legacy
    artifacts where every retained site lacks ``mf`` unit metadata.

    Args:
        fp_all: Site datasets plus dot-prefixed merged-data metadata.
        sites: Retained sites in authoritative order.
        require_units: Raise when neither site attributes nor legacy metadata
            supply units.

    Returns:
        A copied mapping with aligned retained datasets and numeric ``.units``.

    Raises:
        ValueError: If datasets or unit metadata are missing, partial, invalid,
            or dimensionally incompatible.
    """
    aligned = dict(fp_all)
    if not sites:
        return aligned

    datasets: list[xr.Dataset] = []
    for site in sites:
        dataset = fp_all.get(site)
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(f"Retained site {site!r} does not contain an xarray Dataset.")
        if "mf" not in dataset:
            raise ValueError(f"Retained site {site!r} is missing required observation variable `mf`.")
        datasets.append(dataset)

    raw_mf_units = [dataset["mf"].attrs.get("units") for dataset in datasets]
    missing_units = [site for site, units in zip(sites, raw_mf_units, strict=True) if units is None]
    if missing_units:
        if len(missing_units) != len(sites):
            raise ValueError(f"Retained observation sites are missing `mf` unit metadata: {missing_units!r}.")
        legacy_units = fp_all.get(".units")
        if legacy_units is None:
            if require_units:
                raise ValueError(
                    f"Retained observation sites are missing `mf` unit metadata: {missing_units!r}."
                )
            return aligned
        fallback_unit = _pint_unit_name(legacy_units, context="legacy `.units` metadata")
        raw_mf_units = [fallback_unit] * len(sites)

    target_unit = _pint_unit_name(
        raw_mf_units[0],
        context=f"site {sites[0]!r} variable 'mf'",
    )

    for site, dataset, raw_units in zip(sites, datasets, raw_mf_units, strict=True):
        source_unit = _pint_unit_name(raw_units, context=f"site {site!r} variable 'mf'")
        compatible = dataset.copy()
        for name, variable in compatible.data_vars.items():
            if "time" in variable.dims and (
                str(name) in _LEGACY_UNIT_INHERITANCE
                or str(name).startswith(_LEGACY_UNIT_INHERITANCE_PREFIXES)
            ):
                compatible[name].attrs.setdefault("units", source_unit)

        variable_names = _observation_variables(compatible)
        for name in variable_names:
            compatible[name].attrs["units"] = _pint_unit_name(
                compatible[name].attrs["units"],
                context=f"site {site!r} variable {name!r}",
            )
        if all(compatible[name].attrs["units"] == target_unit for name in variable_names):
            aligned[site] = compatible
            continue

        try:
            converted = (
                compatible[variable_names]
                .pint.quantify()
                .pint.to({name: target_unit for name in variable_names})
                .pint.dequantify()
            )
        except Exception as exc:
            raise ValueError(
                f"Could not convert observation-valued variables for site {site!r} to {target_unit!r}."
            ) from exc
        site_aligned = compatible.copy()
        for name in variable_names:
            site_aligned[name] = converted[name]
        aligned[site] = site_aligned

    aligned[".units"] = mole_fraction_unit_scale(
        target_unit,
        context=f"site {sites[0]!r} variable 'mf'",
    )
    return aligned
