"""Prepare and reconstruct observation-aligned sigma component data.

``SigmaComponentData`` holds the one-dimensional, observation-aligned,
non-negative integer indexes created by ``prepare_sigma_component_data``.
Those indexes can be registered by any inversion backend and later passed to
``reconstruct_sigma_aligned`` with an ArviZ trace. These helpers neither create
PyMC objects nor mutate the supplied ``InferenceData``.
"""

from __future__ import annotations

from dataclasses import dataclass

from arviz import InferenceData
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.inversion_inputs import DatetimeLike, make_sigma_freq


def _validate_integer_index(index: xr.DataArray, *, label: str) -> None:
    """Validate that an index is one-dimensional, finite, and non-negative."""
    if index.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional; got dimensions {index.dims!r}.")
    if index.size == 0:
        raise ValueError(f"{label} must contain at least one observation.")

    values = np.asarray(index.values)
    is_real_number = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)
    if np.issubdtype(values.dtype, np.bool_) or not is_real_number:
        raise ValueError(f"{label} must contain integer values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{label} must contain integer values.")
    if np.any(values < 0):
        raise ValueError(f"{label} must contain only non-negative values.")


def _as_index_data_array(
    index: xr.DataArray | np.ndarray,
    *,
    output_dim: str,
    name: str,
    output_coord: xr.DataArray | None = None,
) -> xr.DataArray:
    """Normalize and validate an observation-aligned integer index."""
    if isinstance(index, xr.DataArray):
        if index.ndim != 1 or output_dim not in index.dims:
            raise ValueError(
                f"{name} must be one-dimensional and aligned to {output_dim!r}; "
                f"got dimensions {index.dims!r}."
            )
        result = index.transpose(output_dim)
    else:
        values = np.asarray(index)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional; got shape {values.shape!r}.")
        coords = {output_dim: output_coord} if output_coord is not None else None
        result = xr.DataArray(values, dims=(output_dim,), coords=coords)

    _validate_integer_index(result, label=name)
    return result.astype(int).rename(name)


def _extract_time_coord(data: xr.DataArray, *, output_dim: str) -> xr.DataArray | None:
    """Return an observation-aligned time coordinate when one is available."""
    if "time" in data.coords:
        coord = data.coords["time"]
        if coord.dims == (output_dim,):
            return coord

    if output_dim in data.indexes:
        index = data.indexes[output_dim]
        if isinstance(index, pd.MultiIndex) and "time" in index.names:
            return xr.DataArray(
                index.get_level_values("time").to_numpy(),
                dims=(output_dim,),
                coords={output_dim: data.coords[output_dim]},
                name="time",
            )
    return None


def _validate_alignment(site_index: xr.DataArray, freq_index: xr.DataArray, *, output_dim: str) -> None:
    """Validate positional and labelled alignment between sigma indexes."""
    if site_index.sizes[output_dim] != freq_index.sizes[output_dim]:
        raise ValueError(
            "Sigma site and frequency indexes must have the same observation length; "
            f"got {site_index.sizes[output_dim]} and {freq_index.sizes[output_dim]}."
        )

    if output_dim in site_index.indexes and output_dim in freq_index.indexes:
        if not site_index.indexes[output_dim].equals(freq_index.indexes[output_dim]):
            raise ValueError(
                f"Sigma site and frequency indexes must have matching {output_dim!r} coordinates."
            )


@dataclass(frozen=True)
class SigmaComponentData:
    """Prepared observation-alignment data for a sigma model component.

    Args:
        site_index: One-dimensional, finite, non-negative, integer-valued site
            index with dimensions exactly ``(output_dim,)``. For a shared
            sigma component this is an all-zero array.
        freq_index: One-dimensional, finite, non-negative, integer-valued
            sigma-period index with dimensions exactly ``(output_dim,)``.
        site_index_name: Model-data name used for ``site_index``.
        freq_index_name: Model-data name used for ``freq_index``.
        output_dim: Observation dimension shared by both indexes.

    Raises:
        ValueError: If either index is not a finite, non-negative integer
            vector or the indexes are not observation-aligned.
    """

    site_index: xr.DataArray
    freq_index: xr.DataArray
    site_index_name: str
    freq_index_name: str
    output_dim: str = "nmeasure"

    def __post_init__(self) -> None:
        """Validate the prepared-data invariant for direct construction."""
        if not self.site_index_name or not self.freq_index_name:
            raise ValueError("Sigma model-data names must be non-empty.")
        if self.site_index.dims != (self.output_dim,):
            raise ValueError(
                f"Sigma site index must have dimensions {(self.output_dim,)!r}; got {self.site_index.dims!r}."
            )
        if self.freq_index.dims != (self.output_dim,):
            raise ValueError(
                f"Sigma frequency index must have dimensions {(self.output_dim,)!r}; "
                f"got {self.freq_index.dims!r}."
            )
        _validate_integer_index(self.site_index, label=self.site_index_name)
        _validate_integer_index(self.freq_index, label=self.freq_index_name)
        _validate_alignment(self.site_index, self.freq_index, output_dim=self.output_dim)

    @property
    def nsigma_site(self) -> int:
        """Number of site positions required by the latent sigma array."""
        return int(self.site_index.max().item()) + 1

    @property
    def nsigma_time(self) -> int:
        """Number of period positions required by the latent sigma array."""
        return int(self.freq_index.max().item()) + 1


def prepare_sigma_component_data(
    site_indicator: xr.DataArray | np.ndarray,
    sigma_freq_index: xr.DataArray | np.ndarray | None = None,
    sigma_freq: str | None = None,
    *,
    per_site: bool = True,
    output_dim: str = "nmeasure",
    var_name: str = "sigma",
    anchor_time: DatetimeLike | None = None,
) -> SigmaComponentData:
    """Prepare observation indexes used to align a latent sigma array.

    Explicit frequency codes and an explicit data-array name are preserved.
    When an explicit index is not supplied, periods are derived with
    :func:`make_sigma_freq`, including its all-zero default and compact
    handling of gaps between used periods.

    Args:
        site_indicator: Observation-aligned non-negative site codes.
        sigma_freq_index: Optional explicit observation-aligned period codes.
        sigma_freq: Optional frequency used to derive period codes. ``None``
            creates a single all-observation period.
        per_site: Whether the effective sigma index varies by site.
        output_dim: Observation dimension name.
        var_name: Sigma variable name used to derive non-standard model-data
            names.
        anchor_time: Optional time used to anchor fixed-duration frequency
            bins. This preserves period boundaries when early observations are
            absent or filtered.

    Returns:
        Validated sigma component data in a frozen dataclass.

    Raises:
        ValueError: If indexes are invalid or misaligned, or a requested
            frequency cannot be derived because no time coordinate is present.
    """
    output_dim = str(output_dim)
    site_name = "site_indicator" if per_site else f"{var_name}_site_indicator"
    freq_name = "sigma_freq_index" if var_name == "sigma" else f"{var_name}_freq_indicator"
    if isinstance(sigma_freq_index, xr.DataArray) and sigma_freq_index.name is not None:
        freq_name = str(sigma_freq_index.name)

    site_index = _as_index_data_array(
        site_indicator,
        output_dim=output_dim,
        name="site_indicator",
    )
    if not per_site:
        site_index = xr.zeros_like(site_index)
    site_index = site_index.rename(site_name)

    if sigma_freq_index is None:
        time_coord = _extract_time_coord(site_index, output_dim=output_dim)
        if sigma_freq is not None and time_coord is None:
            raise ValueError(
                f"Cannot derive {freq_name!r}: no time coordinate aligned to {output_dim!r} was found."
            )
        source = site_index if time_coord is None else time_coord
        freq_index = make_sigma_freq(
            source,
            freq=sigma_freq,
            anchor_time=anchor_time,
        ).rename(freq_name)
    else:
        output_coord = site_index.coords.get(output_dim)
        freq_index = _as_index_data_array(
            sigma_freq_index,
            output_dim=output_dim,
            name=freq_name,
            output_coord=output_coord,
        )

    _validate_alignment(site_index, freq_index, output_dim=output_dim)
    return SigmaComponentData(
        site_index=site_index,
        freq_index=freq_index,
        site_index_name=site_name,
        freq_index_name=freq_name,
        output_dim=output_dim,
    )


def _component_data_from_dataset(
    model_data: xr.Dataset,
    *,
    var_name: str,
    output_dim: str,
) -> SigmaComponentData:
    """Extract prepared sigma indexes from an xarray model-data dataset."""
    freq_name = "sigma_freq_index" if var_name == "sigma" else f"{var_name}_freq_indicator"
    shared_site_name = f"{var_name}_site_indicator"
    site_name = shared_site_name if shared_site_name in model_data else "site_indicator"

    missing = [name for name in (site_name, freq_name) if name not in model_data]
    if missing:
        raise ValueError(f"Model data are missing required sigma index variable(s): {missing!r}.")

    site_index = _as_index_data_array(
        model_data[site_name],
        output_dim=output_dim,
        name=site_name,
    )
    freq_index = _as_index_data_array(
        model_data[freq_name],
        output_dim=output_dim,
        name=freq_name,
    )
    return SigmaComponentData(
        site_index=site_index,
        freq_index=freq_index,
        site_index_name=site_name,
        freq_index_name=freq_name,
        output_dim=output_dim,
    )


def reconstruct_sigma_aligned(
    idata: InferenceData,
    model_data: SigmaComponentData | xr.Dataset | None = None,
    *,
    group: str = "posterior",
    var_name: str = "sigma",
    output_dim: str = "nmeasure",
    output_name: str = "sigma_aligned",
) -> xr.DataArray:
    """Reconstruct observation-aligned sigma values from an inference trace.

    Args:
        idata: ArviZ inference data containing the latent sigma posterior.
        model_data: Prepared sigma data or an xarray model-data dataset. Dataset
            inputs must contain ``sigma_freq_index`` (or
            ``<var_name>_freq_indicator``) and either
            ``<var_name>_site_indicator``, which takes precedence, or
            ``site_indicator``. Each index must be one-dimensional on
            ``output_dim``. When omitted, ``idata.constant_data`` is used.
        group: InferenceData group containing the latent sigma variable.
        var_name: Latent sigma variable name.
        output_dim: Observation dimension name used by the model data.
        output_name: Name assigned to the reconstructed array.

    Returns:
        A data array in which ``nsigma_site`` and ``nsigma_time`` are replaced
        by ``output_dim``. Other trace dimensions, typically ``chain`` and
        ``draw``, and observation coordinates are preserved.

    Raises:
        ValueError: If the requested trace group, sigma variable, dimensions,
            or model-data indexes are missing or incompatible.
    """
    trace_group = getattr(idata, group, None)
    if not isinstance(trace_group, xr.Dataset):
        raise ValueError(f"InferenceData does not contain an xarray dataset group {group!r}.")
    if var_name not in trace_group:
        raise ValueError(f"InferenceData group {group!r} does not contain variable {var_name!r}.")

    if model_data is None:
        model_data = getattr(idata, "constant_data", None)
        if not isinstance(model_data, xr.Dataset):
            raise ValueError(
                "Sigma model data were not supplied and InferenceData has no constant_data group."
            )

    prepared = (
        model_data
        if isinstance(model_data, SigmaComponentData)
        else _component_data_from_dataset(model_data, var_name=var_name, output_dim=output_dim)
    )
    sigma = trace_group[var_name]
    required_dims = ("nsigma_site", "nsigma_time")
    missing_dims = [dim for dim in required_dims if dim not in sigma.dims]
    if missing_dims:
        raise ValueError(f"Trace variable {var_name!r} is missing sigma dimension(s): {missing_dims!r}.")
    if prepared.nsigma_site > sigma.sizes["nsigma_site"]:
        raise ValueError(
            f"{prepared.site_index_name!r} contains code {prepared.nsigma_site - 1}, "
            f"but {var_name!r} has only {sigma.sizes['nsigma_site']} site positions."
        )
    if prepared.nsigma_time > sigma.sizes["nsigma_time"]:
        raise ValueError(
            f"{prepared.freq_index_name!r} contains code {prepared.nsigma_time - 1}, "
            f"but {var_name!r} has only {sigma.sizes['nsigma_time']} period positions."
        )

    aligned = sigma.isel(
        nsigma_site=prepared.site_index,
        nsigma_time=prepared.freq_index,
    )
    return aligned.rename(output_name)
