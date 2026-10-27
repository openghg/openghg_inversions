"""Provider-owned structural checks at the surface input boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from openghg.store import BoundaryConditions, Flux, Footprints, ObsSurface
from openghg.types import ValidationError
import xarray as xr


def _coordinate_view(dataset: xr.Dataset) -> xr.Dataset:
    # Custom inputs historically allow integer grid labels. Cast only the view
    # passed to the provider schema, keeping the scientific inputs untouched.
    return dataset.assign_coords(
        {
            name: value.astype(float)
            for name, value in dataset.coords.items()
            if name in {"lat", "lon", "height"} and np.issubdtype(value.dtype, np.integer)
        }
    )


def _footprint_schema_options(
    dataset: xr.Dataset, *, mode: str, particle_locations: bool
) -> dict[str, object]:
    resolved = mode == "time_resolved"
    return {
        "particle_locations": particle_locations,
        "time_resolved": resolved,
        "source_format": "PARIS" if resolved and "fp_HiTRes" not in dataset else None,
    }


def validate_surface_inputs(
    observations: xr.Dataset,
    site_footprint_data: xr.Dataset,
    footprint_data: Mapping[str, xr.Dataset],
    flux_data: Mapping[str, xr.Dataset],
    *,
    transport: Mapping[str, Literal["integrated", "time_resolved"]],
    species: str,
    boundary_data: xr.Dataset | None,
    observation_variable: str = "mf",
) -> None:
    """Check raw-input variable, dimension and dtype metadata without execution.

    Only the selected primary variable is renamed in the temporary provider
    view. Units, alignment and scientific coverage remain with their owning steps.
    """
    _validate_surface_transport(
        observations, site_footprint_data, footprint_data, transport=transport,
        species=species, particle_locations=boundary_data is not None,
        observation_variable=observation_variable,
    )
    _validate_surface_forcing(flux_data, boundary_data=boundary_data)


def _validate_surface_transport(
    observations: xr.Dataset,
    site_footprint_data: xr.Dataset,
    footprint_data: Mapping[str, xr.Dataset],
    *,
    transport: Mapping[str, Literal["integrated", "time_resolved"]],
    species: str,
    particle_locations: bool,
    observation_variable: str,
) -> None:
    """Validate raw observation/transport metadata before role conversion and alignment."""
    context = "observations"
    try:
        obs_view = (
            observations.rename({observation_variable: species})
            if observation_variable != species and observation_variable in observations
            else observations
        )
        ObsSurface.validate_data(obs_view, species=species)
        context = "site_footprint_data"
        site_mode = "integrated" if "fp" in site_footprint_data else "time_resolved"
        Footprints.validate_data(
            _coordinate_view(site_footprint_data),
            **_footprint_schema_options(
                site_footprint_data, mode=site_mode, particle_locations=particle_locations
            ),
        )
        for source, dataset in footprint_data.items():
            context = f"footprint_data[{source!r}]"
            Footprints.validate_data(
                _coordinate_view(dataset),
                **_footprint_schema_options(dataset, mode=transport[source], particle_locations=False),
            )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


def _validate_surface_forcing(
    flux_data: Mapping[str, xr.Dataset], *, boundary_data: xr.Dataset | None
) -> None:
    """Validate retrieved forcing metadata before native scientific calculations."""
    context = "flux_data"
    try:
        for source, dataset in flux_data.items():
            context = f"flux_data[{source!r}]"
            Flux.validate_data(_coordinate_view(dataset))
        if boundary_data is not None:
            context = "boundary_data"
            BoundaryConditions.validate_data(_coordinate_view(boundary_data))
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc
