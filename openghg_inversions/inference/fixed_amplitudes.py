"""Supplied OU site amplitudes and their recorded source for fixed-θ sampling.

Fixed-θ state sampling holds the mismatch hyperparameters at supplied values
and samples only the state. Its posterior is conditional on those values: it
does not propagate their uncertainty. The values may come from configuration,
from a labelled file, or from a summary of a previous run's posterior draws.
The source is recorded alongside the run so that a fixed-θ result can always
be traced to where its θ came from.

Amplitudes are concentrations. Configured values are in the observations'
concentration units. A file must declare its units, either as the variable's
``units`` attribute or as :attr:`FixedSiteAmplitudeFile.units` when the
variable has none; :func:`in_concentration_units` converts compatible
mole-fraction units to the fit's units and rejects anything else.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from openghg_inversions.inversion_data._units import mole_fraction_unit_scale


OU_SITE_AMPLITUDE = "ou_site_amplitude"
OU_SITE_DIM = "ou_site"
SAMPLE_DIMS = ("chain", "draw")
Statistic = Literal["median", "mean"]
_STATISTICS = ("median", "mean")


@dataclass(frozen=True)
class FixedSiteAmplitudeFile:
    """Request to read fixed OU site amplitudes from a labelled NetCDF file.

    The file is read when the model is built, not when configuration is parsed.

    Attributes:
        path: NetCDF file. A RHIME posterior written with groups is read from
            its ``posterior`` group by default; a flat file from its root.
        variable: Amplitude variable, labelled by ``ou_site``.
        statistic: Summary over ``chain`` and ``draw`` when the variable has
            posterior sample dimensions; ignored for a plain site vector.
        group: Optional group to read. ``None`` selects ``posterior`` when the
            file has that group, and the root otherwise.
        units: Concentration units of a variable that has no ``units``
            attribute. A unitless variable without this is rejected rather than
            assumed to be in the fit's units. When the variable declares units,
            this must describe the same scale.
    """

    path: Path
    variable: str = OU_SITE_AMPLITUDE
    statistic: Statistic = "median"
    group: str | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.statistic not in _STATISTICS:
            raise ValueError(f"statistic must be one of {_STATISTICS!r}, not {self.statistic!r}.")


@dataclass(frozen=True)
class FixedSiteAmplitudes:
    """Resolved fixed OU site amplitudes with a JSON-serialisable source record.

    Attributes:
        values: Non-negative amplitude for every site, as one scalar or a
            site-label mapping.
        source: Description of where the values came from. Always contains
            ``kind`` (``"supplied"`` or ``"file"``).
        units: Concentration units of ``values``. ``None`` means the
            observations' concentration units, as for configured values.
    """

    values: float | Mapping[str, float]
    source: Mapping[str, Any] = field(default_factory=lambda: {"kind": "supplied"})
    units: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_variable(request: FixedSiteAmplitudeFile) -> tuple[xr.DataArray, str | None]:
    tree = xr.open_datatree(request.path)
    try:
        group = request.group
        if group is None and "posterior" in tree.children:
            group = "posterior"
        node = tree if group is None else tree[group]
        if request.variable not in node.data_vars:
            where = "the file root" if group is None else f"group {group!r}"
            raise ValueError(f"{request.path} has no variable {request.variable!r} in {where}.")
        return node[request.variable].load(), group
    finally:
        tree.close()


def load_fixed_site_amplitudes(request: FixedSiteAmplitudeFile) -> FixedSiteAmplitudes:
    """Read and summarise fixed OU site amplitudes from a labelled file.

    Args:
        request: File, variable, group and summary statistic to use.

    Returns:
        Site-label mapping and a source record with the resolved path, its
        SHA-256, the group and variable read, the statistic, and the number of
        posterior samples summarised (``0`` for a plain site vector).

    Raises:
        ValueError: If the variable is missing, is not labelled only by
            ``ou_site`` after summarising ``chain``/``draw``, contains
            negative or non-finite values, or has no declared units.
    """
    path = request.path.expanduser().resolve()
    request = FixedSiteAmplitudeFile(path, request.variable, request.statistic, request.group, request.units)
    values, group = _read_variable(request)
    units = _declared_units(values, request)
    sample_dims = [dim for dim in SAMPLE_DIMS if dim in values.dims]
    n_samples = int(np.prod([values.sizes[dim] for dim in sample_dims])) if sample_dims else 0
    if sample_dims:
        values = getattr(values, request.statistic)(sample_dims)
    if values.dims != (OU_SITE_DIM,) or OU_SITE_DIM not in values.coords:
        raise ValueError(
            f"{request.variable!r} must be labelled by {OU_SITE_DIM!r} (after summarising "
            f"{SAMPLE_DIMS}); got dimensions {values.dims}."
        )
    array = np.asarray(values.values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{request.variable!r} in {path} must be finite and non-negative.")
    labels = [str(label) for label in values[OU_SITE_DIM].values]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{request.variable!r} in {path} has duplicate {OU_SITE_DIM} labels.")
    source: dict[str, Any] = {
        "kind": "file",
        "path": str(path),
        "sha256": _sha256(path),
        "group": group,
        "variable": request.variable,
        "statistic": request.statistic if sample_dims else None,
        "n_samples": n_samples,
        "units": units,
    }
    return FixedSiteAmplitudes(dict(zip(labels, array.tolist(), strict=True)), source, units)


def _declared_units(values: xr.DataArray, request: FixedSiteAmplitudeFile) -> str:
    attribute = values.attrs.get("units")
    attribute = None if attribute is None or str(attribute).strip() == "" else str(attribute)
    if attribute is None and request.units is None:
        raise ValueError(
            f"{request.variable!r} in {request.path} has no units attribute. Add one or set "
            "units in the file request; unitless amplitudes are not assumed to match the fit."
        )
    if attribute is not None and request.units is not None:
        context = f"fixed OU amplitudes in {request.path}"
        if not np.isclose(
            mole_fraction_unit_scale(attribute, context=context),
            mole_fraction_unit_scale(request.units, context=context),
            rtol=1e-12, atol=0.0,
        ):
            raise ValueError(
                f"{request.variable!r} in {request.path} declares units {attribute!r}, "
                f"but the request says {request.units!r}."
            )
    return attribute if attribute is not None else str(request.units)


def in_concentration_units(amplitudes: FixedSiteAmplitudes, units: str) -> FixedSiteAmplitudes:
    """Express supplied amplitudes in the fit's concentration units.

    Values without units are already in the fit's units. Values with
    mole-fraction units are rescaled; the source then records the declared
    units, the fit units and the factor applied.

    Raises:
        ValueError: If either unit cannot be read as a mole fraction.
    """
    if amplitudes.units is None:
        return amplitudes
    context = "fixed OU site amplitudes"
    factor = mole_fraction_unit_scale(amplitudes.units, context=context) / mole_fraction_unit_scale(
        units, context=f"{context} (fit units)"
    )
    if np.isclose(factor, 1.0, rtol=1e-12, atol=0.0):
        factor = 1.0
    values: float | Mapping[str, float]
    if isinstance(amplitudes.values, Mapping):
        values = {site: float(value) * factor for site, value in amplitudes.values.items()}
    else:
        values = float(amplitudes.values) * factor
    source = {**amplitudes.source, "units": amplitudes.units, "fit_units": units, "unit_factor": factor}
    return FixedSiteAmplitudes(values, source, units)


def fixed_amplitude_input_paths(value: object) -> tuple[Path, ...]:
    """Return the files a fixed-amplitude request or a recorded source reads.

    Accepts a :class:`FixedSiteAmplitudeFile`, resolved
    :class:`FixedSiteAmplitudes`, or a recorded ``ou_site_amplitude_source``
    mapping. Supplied values read no file.
    """
    if isinstance(value, FixedSiteAmplitudeFile):
        return (value.path.expanduser().resolve(),)
    if isinstance(value, FixedSiteAmplitudes):
        value = value.source
    if isinstance(value, Mapping) and value.get("kind") == "file" and isinstance(value.get("path"), str):
        return (Path(value["path"]).expanduser().resolve(),)
    return ()


def resolve_fixed_site_amplitudes(
    value: float | Mapping[str, float] | FixedSiteAmplitudeFile | FixedSiteAmplitudes,
) -> FixedSiteAmplitudes:
    """Return supplied amplitudes with a source record, reading a file if requested."""
    if isinstance(value, FixedSiteAmplitudes):
        return value
    if isinstance(value, FixedSiteAmplitudeFile):
        return load_fixed_site_amplitudes(value)
    return FixedSiteAmplitudes(value, {"kind": "supplied"})


__all__ = [
    "FixedSiteAmplitudeFile",
    "FixedSiteAmplitudes",
    "fixed_amplitude_input_paths",
    "in_concentration_units",
    "load_fixed_site_amplitudes",
    "resolve_fixed_site_amplitudes",
]
