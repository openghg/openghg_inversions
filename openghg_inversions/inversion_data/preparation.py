"""Prepare inversion data for modern RHIME and legacy fixedbasis runners.

``prepare_rhime_inputs`` returns backend-neutral observations, sensitivities,
basis metadata, and site metadata; component-specific model arrays are
intentionally absent. ``prepare_fixedbasis_inversion_data`` is a compatibility
adapter that retains the input variables required by ``fixedbasisMCMC``,
including its sigma-period index.

Preparation can read OpenGHG object stores or local merged-data artifacts,
write merged-data and basis artifacts, emit warnings and progress messages,
and record timing information. Neither public entry point constructs a PyMC
model.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import basis_functions_wrapper, make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.filters import filtering
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data._site_options import (
    expand_site_option,
    is_column_observation,
)
from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import normalise_min_error_options
from openghg_inversions.sigma import SigmaAlignment

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float
SiteStringOption = Sequence[str | None] | str | None
SiteInletOption = Sequence[str | slice | None] | str | None
SiteIntegerOption = Sequence[int | None] | int | None


@dataclass
class FixedBasisPreparedData:
    """Data prepared for the legacy fixedbasis runner.

    Args:
        fp_all: Raw merged-data container returned by data gathering or reload.
        fp_data: Forward-model data after basis functions and filters.
        inv_inputs: Canonical inversion inputs consumed by model builders.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_objects: Basis objects returned by ``basis_functions_wrapper``.
        basis_artifact_source: Source of the flux basis artifact.
        basis_artifact_path: Path to the flux basis artifact, when loaded or saved.
        is_column: Whether any retained observation is a column observation.
    """

    fp_all: dict
    fp_data: dict | None = None
    inv_inputs: xr.Dataset | None = None
    sites: list[str] = field(default_factory=list)
    averaging_period: list[str | None] = field(default_factory=list)
    basis_objects: dict[str, BasisFunctions] = field(default_factory=dict)
    basis_artifact_source: str = "generated"
    basis_artifact_path: str | None = None
    is_column: bool = False


@dataclass(frozen=True)
class RhimePreparedInputs:
    """Modern RHIME preparation contract.

    Args:
        inv_inputs: Canonical inversion inputs consumed by RHIME model
            builders.
        basis_functions: Retained flux basis object used to derive
            output-boundary basis and flux arrays.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_artifact_source: Description of whether the basis was generated
            or loaded from an artifact.
        basis_artifact_path: Path to the basis artifact, when loaded or saved.
        site_lats: Release latitudes aligned to ``sites``, when available.
        site_lons: Release longitudes aligned to ``sites``, when available.
    """

    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    basis_artifact_source: str
    basis_artifact_path: str | None = None
    site_lats: tuple[float, ...] | None = None
    site_lons: tuple[float, ...] | None = None


def _normalise_site_strings(
    value: Sequence[str | None] | str | None,
    *,
    length: int,
    name: str,
) -> list[str | None]:
    """Normalize and validate one optional-string value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str)]
    if invalid:
        raise ValueError(f"`{name}` entries must be strings or None. Invalid value(s): {invalid!r}.")
    return normalized


def _normalise_site_integers(
    value: Sequence[int | None] | int | None,
    *,
    length: int,
    name: str,
) -> list[int | None]:
    """Normalize and validate one optional integer value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))

    invalid = [
        item
        for item in normalized
        if item is not None and (not isinstance(item, Integral) or isinstance(item, bool))
    ]
    if invalid:
        raise ValueError(f"`{name}` entries must be integers or None. Invalid value(s): {invalid!r}.")
    return [None if item is None else int(item) for item in normalized]


def _normalise_site_inlets(
    value: Sequence[str | slice | None] | str | None,
    *,
    length: int,
) -> list[str | slice | None]:
    """Normalize inlet selectors, including legacy per-site slice selectors."""
    normalized = list(expand_site_option(value, nsites=length, name="inlet"))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str | slice)]
    if invalid:
        raise ValueError(f"`inlet` entries must be strings, slices, or None. Invalid value(s): {invalid!r}.")
    return normalized


@dataclass(frozen=True)
class _SiteOptions:
    """All runner inputs whose positions are aligned to ``sites``.

    Every field has the same length and ordering. Selection always creates a
    new complete record so no option can drift independently from its site.
    """

    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    inlet: tuple[str | slice | None, ...]
    fp_height: tuple[str | None, ...]
    instrument: tuple[str | None, ...]
    platform: tuple[str | None, ...]
    obs_data_level: tuple[str | None, ...]
    met_model: tuple[str | None, ...]
    max_level: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Freeze supplied sequences and enforce the common-length invariant."""
        field_names = (
            "sites",
            "averaging_period",
            "inlet",
            "fp_height",
            "instrument",
            "platform",
            "obs_data_level",
            "met_model",
            "max_level",
        )
        for name in field_names:
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if not self.sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError(f"Site names must be unique: {self.sites!r}.")

        expected_length = len(self.sites)
        misaligned = {
            name: len(getattr(self, name))
            for name in field_names[1:]
            if len(getattr(self, name)) != expected_length
        }
        if misaligned:
            raise ValueError(
                "Every site-aligned option must have the same length as `sites`; "
                f"expected {expected_length}, got {misaligned!r}."
            )

    @classmethod
    def from_inputs(
        cls,
        *,
        sites: Sequence[str],
        averaging_period: Sequence[str | None] | str | None,
        inlet: Sequence[str | slice | None] | str | None,
        fp_height: Sequence[str | None] | str | None,
        instrument: Sequence[str | None] | str | None,
        platform: Sequence[str | None] | str | None,
        obs_data_level: Sequence[str | None] | str | None,
        met_model: Sequence[str | None] | str | None,
        max_level: Sequence[int | None] | int | None,
    ) -> _SiteOptions:
        """Normalize all site options and validate their common length.

        Site names are uppercased. Scalar option values are broadcast, while
        sequences must match the number of sites. Inlets also support legacy
        ``slice`` selectors; maximum levels reject booleans.

        Raises:
            ValueError: If no sites are supplied, site names are duplicated,
                an option has the wrong length, or an entry has an invalid
                type.
        """
        normalized_sites = [site.upper() for site in sites]
        if not normalized_sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(normalized_sites)) != len(normalized_sites):
            raise ValueError(f"Site names must be unique: {normalized_sites!r}.")
        nsites = len(normalized_sites)
        return cls(
            sites=tuple(normalized_sites),
            averaging_period=tuple(
                _normalise_site_strings(averaging_period, length=nsites, name="averaging_period")
            ),
            inlet=tuple(_normalise_site_inlets(inlet, length=nsites)),
            fp_height=tuple(_normalise_site_strings(fp_height, length=nsites, name="fp_height")),
            instrument=tuple(_normalise_site_strings(instrument, length=nsites, name="instrument")),
            platform=tuple(_normalise_site_strings(platform, length=nsites, name="platform")),
            obs_data_level=tuple(
                _normalise_site_strings(obs_data_level, length=nsites, name="obs_data_level")
            ),
            met_model=tuple(_normalise_site_strings(met_model, length=nsites, name="met_model")),
            max_level=tuple(_normalise_site_integers(max_level, length=nsites, name="max_level")),
        )

    def select_indices(self, indices: Sequence[int]) -> _SiteOptions:
        """Return a new complete option record restricted to ``indices``."""

        def select(values: Sequence[Any]) -> tuple[Any, ...]:
            return tuple(values[index] for index in indices)

        return _SiteOptions(
            sites=select(self.sites),
            averaging_period=select(self.averaging_period),
            inlet=select(self.inlet),
            fp_height=select(self.fp_height),
            instrument=select(self.instrument),
            platform=select(self.platform),
            obs_data_level=select(self.obs_data_level),
            met_model=select(self.met_model),
            max_level=select(self.max_level),
        )

    @property
    def is_column(self) -> bool:
        """Whether any retained site uses a supported column-data selector."""
        return any(
            is_column_observation(inlet, platform)
            for inlet, platform in zip(self.inlet, self.platform, strict=True)
        )

    def retain_sites(self, retained_sites: Sequence[str], *, context: str) -> _SiteOptions:
        """Return options for retained sites in their supplied order.

        Raises:
            ValueError: If requested or retained names are duplicated, or a
                retained name was not in the original request.
        """
        normalized_retained = [site.upper() for site in retained_sites]
        index_by_site = {site: index for index, site in enumerate(self.sites)}
        if len(index_by_site) != len(self.sites):
            raise ValueError(f"{context} cannot align duplicate requested site names: {self.sites!r}.")

        missing_sites = [site for site in normalized_retained if site not in index_by_site]
        if missing_sites:
            raise ValueError(f"{context} returned site(s) that were not requested: {missing_sites!r}.")
        if len(set(normalized_retained)) != len(normalized_retained):
            raise ValueError(f"{context} returned duplicate site names: {normalized_retained!r}.")

        return self.select_indices([index_by_site[site] for site in normalized_retained])


@dataclass
class _MergedInversionData:
    """Merged data and complete site-aligned metadata shared by preparation paths."""

    fp_all: dict
    site_options: _SiteOptions

    @property
    def sites(self) -> tuple[str, ...]:
        """Retained site names."""
        return self.site_options.sites

    @property
    def averaging_period(self) -> tuple[str | None, ...]:
        """Retained averaging periods aligned to :attr:`sites`."""
        return self.site_options.averaging_period


def _first_scalar_data_value(ds: xr.Dataset, names: tuple[str, ...]) -> float:
    """Return the first scalar value found in a dataset variable or coordinate."""
    for name in names:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values).reshape(-1)
        if values.size:
            return float(values[0])
    return np.nan


def _site_release_coordinates(
    fp_data: dict[str, xr.Dataset],
    sites: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return release lat/lon values aligned to retained sites."""
    lats: list[float] = []
    lons: list[float] = []
    for site in sites:
        site_data = fp_data.get(site)
        if site_data is None:
            lats.append(np.nan)
            lons.append(np.nan)
            continue
        lats.append(_first_scalar_data_value(site_data, ("release_lat", "sitelats")))
        lons.append(_first_scalar_data_value(site_data, ("release_lon", "sitelons")))
    return tuple(lats), tuple(lons)


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    site_options: _SiteOptions,
) -> _SiteOptions:
    """Align site-level options when loaded merged data lacks requested sites."""
    sites_merged = [site for site in fp_all if not site.startswith(".")]
    if all(site in sites_merged for site in site_options.sites):
        return site_options

    keep_indices = [index for index, site in enumerate(site_options.sites) if site in sites_merged]
    dropped_sites = [site for site in site_options.sites if site not in sites_merged]
    if not keep_indices:
        raise ValueError(
            "Loaded merged data does not include any requested sites. "
            f"Requested sites: {site_options.sites}. Available merged-data sites: {sites_merged}."
        )

    print(f"\nDropping {dropped_sites} sites as they are not included in the merged data object.\n")
    return site_options.select_indices(keep_indices)


def _select_fp_all_sites(fp_all: dict, sites: Sequence[str]) -> dict:
    """Keep requested sites and prune site-keyed calibration scales."""
    site_names = set(sites)
    selected = {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}

    scales = selected.get(".scales")
    if isinstance(scales, Mapping):
        selected[".scales"] = {site: scales[site] for site in sites if site in scales}

    return selected


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: Sequence[str],
    start_date: str,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    calculate_min_error: Literal["percentile", "residual"] | None,
    min_error_per_site: bool,
) -> xr.Dataset:
    """Create backend-neutral inversion inputs with min-error compatibility.

    Args:
        fp_data: Filtered per-site observations and sensitivity data.
        sites: Retained sites in observation order.
        start_date: Anchor for fixed-duration boundary-condition periods.
        bc_freq: Optional boundary-condition period frequency.
        min_error: Minimum-error value or calculation method.
        calculate_min_error: Deprecated minimum-error calculation argument.
        min_error_per_site: Whether calculated minimum error varies by site.

    Returns:
        Canonical observation-aligned inputs without component-specific model
        data.

    Warns:
        FutureWarning: If ``calculate_min_error`` is supplied.
    """
    if calculate_min_error is not None:
        warnings.warn(
            "`calculate_min_error` is deprecated. Please use `min_error` to pass the calculation method instead.",
            FutureWarning,
            stacklevel=3,
        )
        min_error = calculate_min_error

    if min_error is None:
        min_error = 0.0
    elif isinstance(min_error, int) and not isinstance(min_error, bool):
        min_error = float(min_error)
    elif isinstance(min_error, dict):
        missing_sites = [site for site in sites if site not in min_error]
        if missing_sites:
            raise ValueError(
                "`min_error` dictionaries must include a value for every retained site. "
                f"Missing site(s): {missing_sites!r}."
            )

    return make_inv_inputs(
        fp_data,
        sites=list(sites),
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_per_site=min_error_per_site,
        start_date=start_date,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values")
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values")


def _prepare_merged_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> _MergedInversionData:
    """Gather or reload merged data and align site metadata.

    ``flux_sources`` contains modern OpenGHG flux ``source`` values. This
    helper passes them to lower-level data loading through the legacy
    ``emissions_name`` argument. Retrieval may access OpenGHG object stores,
    print progress, and optionally save merged data. Reload reads a local
    artifact. Both paths retain one complete :class:`_SiteOptions` record.

    Returns:
        Merged per-site data and aligned retained-site options.

    Raises:
        ValueError: If site options are invalid, no requested sites are loaded,
            or retrieval returns misaligned metadata.
        RuntimeError: If neither retrieval nor reload produces merged data.
    """
    if use_tracer:
        raise ValueError("Tracer inversions are not supported by this preparation path.")
    site_options = _SiteOptions.from_inputs(
        sites=sites,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
    )
    rerun_merge = True
    fp_all: dict | None = None
    if reload_merged_data and merged_data_dir is not None:
        try:
            fp_all = load_merged_data(merged_data_dir, species, start_date, output_name, merged_data_name)
        except ValueError as exc:
            print(f"{exc}, re-running data merge.")
        else:
            print("Successfully read in merged data.\n")
            fp_all[".split_by_sectors"] = split_by_sectors
            rerun_merge = False
            site_options = _drop_sites_missing_from_loaded_data(
                fp_all=fp_all,
                site_options=site_options,
            )
            fp_all = _select_fp_all_sites(fp_all, site_options.sites)
    elif reload_merged_data:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    if rerun_merge:
        (
            fp_all,
            retained_sites,
            retained_inlet,
            retained_fp_height,
            retained_instrument,
            retained_averaging_period,
        ) = data_processing_surface_notracer(
            species=species,
            sites=list(site_options.sites),
            domain=domain,
            averaging_period=list(site_options.averaging_period),
            start_date=start_date,
            end_date=end_date,
            obs_data_level=list(site_options.obs_data_level),
            platform=list(site_options.platform),
            met_model=list(site_options.met_model),
            fp_model=fp_model,
            fp_height=list(site_options.fp_height),
            fp_species=fp_species,
            emissions_name=flux_sources,
            inlet=list(site_options.inlet),
            instrument=list(site_options.instrument),
            max_level=list(site_options.max_level),
            calibration_scale=calibration_scale,
            use_bc=use_bc,
            bc_input=bc_input,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            split_by_sectors=split_by_sectors,
            averagingerror=averaging_error,
            save_merged_data=save_merged_data,
            merged_data_name=merged_data_name,
            merged_data_dir=merged_data_dir,
            output_name=output_name,
            flux_non_finite_check=flux_non_finite_check,
        )
        site_options = site_options.retain_sites(retained_sites, context="Data gathering")
        retained_count = len(site_options.sites)
        returned_metadata = {
            "averaging_period": retained_averaging_period,
            "inlet": retained_inlet,
            "fp_height": retained_fp_height,
            "instrument": retained_instrument,
        }
        misaligned_lengths = {
            name: len(values) for name, values in returned_metadata.items() if len(values) != retained_count
        }
        if misaligned_lengths:
            raise ValueError(
                "Data gathering returned metadata with lengths that do not match retained sites; "
                f"expected {retained_count}, got {misaligned_lengths!r}."
            )

    if fp_all is None:
        raise RuntimeError("Data preparation did not create or load merged data.")
    if not site_options.sites:
        raise ValueError("No sites remain after data gathering.")
    fp_all = _select_fp_all_sites(fp_all, site_options.sites)

    flux_entries = fp_all.get(".flux")
    if isinstance(flux_entries, Mapping):
        for source, flux_data in flux_entries.items():
            data = getattr(flux_data, "data", None)
            if isinstance(data, xr.Dataset) and "flux" in data:
                data["flux"] = sanitize_flux_nonfinite(
                    data["flux"],
                    context="merged inversion data preparation",
                    source=str(source),
                    check=flux_non_finite_check,
                    warn=flux_non_finite_check == "count",
                )

    return _MergedInversionData(
        fp_all=fp_all,
        site_options=site_options,
    )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    site_options: _SiteOptions,
    filters: Any,
) -> tuple[dict, _SiteOptions]:
    """Apply filters and keep site-aligned metadata in sync."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in site_options.sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in site_options.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        keep_indices = [index for index, site in enumerate(site_options.sites) if site not in dropped_sites]
        if not keep_indices:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        site_options = site_options.select_indices(keep_indices)
        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data, site_options


def _set_domain_attrs(fp_data: dict, sites: Sequence[str], domain: str) -> None:
    """Attach the legacy domain attribute expected by downstream code."""
    for site in sites:
        fp_data[site].attrs["Domain"] = domain


def _bc_basis_directory_arg(bc_basis_directory: str | Path | None) -> str | None:
    """Normalize BC basis directory arguments for legacy helpers."""
    return str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory


def _validate_multisector_sensitivity_sources(
    sensitivity: xr.DataArray,
    *,
    site: str,
    flux_sources: list[str],
) -> xr.DataArray:
    """Validate and order one site's source-resolved sensitivity."""
    if "source" not in sensitivity.coords:
        raise ValueError(
            f"Site {site!r} sensitivity is missing the 'source' coordinate required for "
            f"flux source(s) {flux_sources!r}."
        )

    source_labels = [str(source) for source in sensitivity.coords["source"].values]
    available_sources = list(dict.fromkeys(source_labels))
    duplicate_sources = (
        [source for source in available_sources if source_labels.count(source) > 1]
        if "source" in sensitivity.dims
        else []
    )
    missing_sources = [source for source in flux_sources if source not in available_sources]
    extra_sources = [source for source in available_sources if source not in flux_sources]
    if duplicate_sources or missing_sources or extra_sources:
        raise ValueError(
            f"Site {site!r} sensitivity source layout does not match requested flux sources; "
            f"missing source(s): {missing_sources!r}; extra source(s): {extra_sources!r}; "
            f"duplicate source(s): {duplicate_sources!r}."
        )
    if "source" in sensitivity.dims:
        return sensitivity.sel(source=flux_sources)
    return sensitivity


def _rhime_site_data_from_basis_functions(
    *,
    merged: _MergedInversionData,
    basis_functions: BasisFunctions,
    domain: str,
    split_by_sectors: bool,
    flux_sources: list[str],
    use_bc: bool,
    bc_basis_case: str,
    bc_basis_directory: str | None,
) -> dict:
    """Apply retained basis functions to one prepared merged-data stage."""
    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_x_flux_name = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"

    for site in merged.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            continue
        fp_x_flux = fp_data[site][fp_x_flux_name]
        timing_start = timer_start()
        sensitivity = basis_functions.sensitivity(fp_x_flux)
        state_dims = [dim for dim in sensitivity.dims if dim not in fp_x_flux.dims]
        if "region" in sensitivity.dims:
            state_dim = "region"
        elif len(state_dims) == 1:
            state_dim = cast(str, state_dims[0])
        else:
            raise ValueError(
                "Could not identify the RHIME sensitivity state dimension from "
                f"sensitivity dims {sensitivity.dims!r} and fp_x_flux dims {fp_x_flux.dims!r}."
            )
        if split_by_sectors:
            sensitivity = _validate_multisector_sensitivity_sources(
                sensitivity,
                site=site,
                flux_sources=flux_sources,
            )
        if "source" in sensitivity.coords and "source" not in sensitivity.dims:
            fp_data[site] = fp_data[site].drop_vars(fp_x_flux_name)
            orphan_dims = [
                dim
                for dim in fp_x_flux.dims
                if dim in fp_data[site].dims
                and all(dim not in variable.dims for variable in fp_data[site].data_vars.values())
            ]
            if orphan_dims:
                fp_data[site] = fp_data[site].drop_dims(orphan_dims)
        fp_data[site]["H"] = sensitivity
        log_timing(
            "rhime.prepare_inputs.footprint_sensitivity",
            timer_seconds(timing_start),
            site=site,
            nmeasure=fp_data[site].sizes.get("time"),
            state_size=sensitivity.sizes.get(state_dim),
            sources=sensitivity.sizes.get("source"),
        )

    if use_bc:
        with timed("rhime.prepare_inputs.bc_sensitivity", sites=len(merged.sites)):
            fp_data = bc_sensitivity(
                fp_data,
                domain=domain,
                basis_case=bc_basis_case,
                bc_basis_directory=bc_basis_directory,
            )

    return fp_data


def _filter_merged_inversion_data(
    *,
    merged: _MergedInversionData,
    filters: Any,
) -> _MergedInversionData:
    """Filter merged RHIME data as a separate pre-basis preparation stage.

    Args:
        merged: Merged site data and site-aligned metadata from data gathering
            or reload.
        filters: Filter configuration accepted by
            :func:`openghg_inversions.filters.filtering`.

    Returns:
        Merged data containing filtered site datasets, with empty sites and
        all of their aligned options removed. If no filters are configured and
        all sites contain data, the original merged data are returned.

    Raises:
        ValueError: If every requested site is removed by filtering.
    """
    if filters is None and all(merged.fp_all[site].sizes.get("time", 0) > 0 for site in merged.sites):
        return merged

    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_data, site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    fp_all = _select_fp_all_sites({**merged.fp_all, **fp_data}, site_options.sites)
    return _MergedInversionData(fp_all=fp_all, site_options=site_options)


def prepare_fixedbasis_inversion_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    calculate_min_error: Literal["percentile", "residual"] | None = None,
    min_error_options: Mapping[str, Any] | None = None,
    return_basis_objects: bool = False,
    merged_data_only: bool = False,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> FixedBasisPreparedData:
    """Prepare data for legacy ``fixedbasisMCMC`` and its output adapters.

    This adapter preserves the fixed-basis inversion-input contract. Unless
    ``merged_data_only`` is true, the returned ``inv_inputs`` includes
    ``sigma_freq_index(nmeasure)`` derived from ``sigma_freq`` and anchored to
    ``start_date``. Modern RHIME preparation intentionally omits this
    component-specific variable.

    Args:
        sites: Requested observation site names. At least one unique site is
            required.
        averaging_period: Observation averaging period, either scalar or
            aligned to ``sites``.
        inlet: Inlet selector, either scalar or aligned to ``sites``. Entries
            may be strings, legacy ``slice`` selectors, or ``None``.
        fp_height: Footprint inlet height, either scalar or aligned to
            ``sites``.
        instrument: Observation instrument, either scalar or aligned to
            ``sites``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        obs_data_level: Observation data level, either scalar or aligned to
            ``sites``.
        met_model: Footprint meteorological model, either scalar or aligned to
            ``sites``.
        max_level: Maximum column level, either scalar or aligned to ``sites``.
            Entries must be integers or ``None``.
        min_error: Numeric minimum error or ``"residual"``/``"percentile"``
            calculation method.
        calculate_min_error: Deprecated calculation-method spelling.
        min_error_options: Calculated minimum-error options. The only supported
            key is boolean ``by_site``.

    Returns:
        Prepared legacy data, including forward-model inputs and optional basis
        objects. When ``merged_data_only`` is true, only merged data and
        retained site metadata are populated.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, or if minimum-error options are invalid.

    Warns:
        FutureWarning: If deprecated ``calculate_min_error`` is supplied.
    """
    min_error_options = normalise_min_error_options(min_error_options)
    merged = _prepare_merged_data(
        species=species,
        sites=sites,
        domain=domain,
        averaging_period=averaging_period,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        flux_sources=flux_sources,
        split_by_sectors=split_by_sectors,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        met_model=met_model,
        fp_model=fp_model,
        fp_height=fp_height,
        fp_species=fp_species,
        inlet=inlet,
        instrument=instrument,
        max_level=max_level,
        calibration_scale=calibration_scale,
        obs_data_level=obs_data_level,
        platform=platform,
        use_tracer=use_tracer,
        use_bc=use_bc,
        bc_input=bc_input,
        averaging_error=averaging_error,
        reload_merged_data=reload_merged_data,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        flux_non_finite_check=flux_non_finite_check,
    )

    if merged_data_only:
        return FixedBasisPreparedData(
            fp_all=merged.fp_all,
            sites=list(merged.sites),
            averaging_period=list(merged.averaging_period),
            is_column=merged.site_options.is_column,
        )

    bc_basis_directory_arg = _bc_basis_directory_arg(bc_basis_directory)

    basis_result = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=bc_basis_directory_arg,
        country_directory=country_directory,
        fp_all=merged.fp_all,
        use_bc=use_bc,
        species=species,
        domain=domain,
        start_date=start_date,
        fix_outer_regions=fix_basis_outer_regions,
        emissions_name=flux_sources,
        outputname=output_name,
        output_path=basis_output_path,
        return_basis_objects=True,
    )
    fp_data, fixedbasis_basis_objects = cast(tuple[dict, dict[str, BasisFunctions]], basis_result)
    emissions_basis = fixedbasis_basis_objects["emissions"]
    basis_source = emissions_basis.basis_artifact_source or "generated"
    basis_path = getattr(emissions_basis, "basis_artifact_path", None)
    basis_objects = fixedbasis_basis_objects if return_basis_objects else {}

    fp_data, prepared_site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    prepared_sites = list(prepared_site_options.sites)
    prepared_averaging_period = list(prepared_site_options.averaging_period)
    _set_domain_attrs(fp_data, prepared_sites, domain)

    inv_inputs = _make_inv_inputs(
        fp_data=fp_data,
        sites=prepared_sites,
        start_date=start_date,
        bc_freq=bc_freq,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_per_site=min_error_options["by_site"],
    )
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=sigma_freq,
        anchor_time=start_date,
    )
    inv_inputs["sigma_freq_index"] = sigma_alignment.period_index.rename("sigma_freq_index")
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)

    return FixedBasisPreparedData(
        fp_all=_select_fp_all_sites(merged.fp_all, prepared_sites),
        fp_data=fp_data,
        inv_inputs=inv_inputs,
        basis_objects=basis_objects,
        basis_artifact_source=basis_source,
        basis_artifact_path=basis_path,
        sites=prepared_sites,
        averaging_period=prepared_averaging_period,
        is_column=prepared_site_options.is_column,
    )


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: Mapping[str, Any] | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimePreparedInputs:
    """Prepare modern RHIME inputs without exposing legacy fixedbasis containers.

    Observation filters are applied once to merged data before basis loading or
    generation. The same filtered site datasets and aligned metadata are then
    used for sensitivity construction.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        sites: Requested observation site names.
        domain: Model domain name.
        averaging_period: Observation averaging period, either scalar or
            site-aligned.
        start_date: Inclusive inversion start date.
        end_date: Exclusive inversion end date.
        output_name: Base output name used for data and basis artifacts.
        flux_sources: OpenGHG flux ``source`` values requested for the run.
        split_by_sectors: Whether to keep sector-resolved sensitivity inputs
            with a ``source`` provenance coordinate. Semantic sector names are
            applied later by the model specification.
        inlet: Inlet selector, either scalar or aligned to ``sites``. Entries
            may be strings, legacy ``slice`` selectors, or ``None``.
        fp_height: Footprint inlet height, either scalar or aligned to
            ``sites``.
        instrument: Observation instrument, either scalar or aligned to
            ``sites``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        obs_data_level: Observation data level, either scalar or aligned to
            ``sites``.
        met_model: Footprint meteorological model, either scalar or aligned to
            ``sites``.
        max_level: Maximum column level, either scalar or aligned to ``sites``.
            Entries must be integers or ``None``.
        min_error: Numeric minimum error or ``"residual"``/``"percentile"``
            calculation method.
        min_error_options: Calculated minimum-error options. The only supported
            key is boolean ``by_site``.
        use_tracer: Unsupported placeholder for tracer inversions, where an
            additional species constrains the primary species through linked
            forward models.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, or if minimum-error options are invalid.
    """
    min_error_options = normalise_min_error_options(min_error_options)
    with timed("rhime.prepare_inputs.merged_data", sites=len(sites), split_by_sectors=split_by_sectors):
        merged = _prepare_merged_data(
            species=species,
            sites=sites,
            domain=domain,
            averaging_period=averaging_period,
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
            flux_sources=flux_sources,
            split_by_sectors=split_by_sectors,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
            inlet=inlet,
            instrument=instrument,
            max_level=max_level,
            calibration_scale=calibration_scale,
            obs_data_level=obs_data_level,
            platform=platform,
            use_tracer=use_tracer,
            use_bc=use_bc,
            bc_input=bc_input,
            averaging_error=averaging_error,
            reload_merged_data=reload_merged_data,
            save_merged_data=save_merged_data,
            merged_data_dir=merged_data_dir,
            merged_data_name=merged_data_name,
            flux_non_finite_check=flux_non_finite_check,
        )

    with timed("rhime.prepare_inputs.obs_filtering", sites=len(merged.sites), filters=filters is not None):
        filtered_merged = _filter_merged_inversion_data(merged=merged, filters=filters)

    with timed(
        "rhime.prepare_inputs.basis_build",
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
    ):
        basis_functions = make_basis_functions(
            basis_algorithm=basis_algorithm,
            nbasis=nbasis,
            fp_basis_case=fp_basis_case,
            basis_directory=basis_directory,
            country_directory=country_directory,
            fp_all=filtered_merged.fp_all,
            species=species,
            domain=domain,
            start_date=start_date,
            fix_outer_regions=fix_basis_outer_regions,
            emissions_name=flux_sources,
            outputname=output_name,
            output_path=basis_output_path,
        )

    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(filtered_merged.sites)):
        fp_data = _rhime_site_data_from_basis_functions(
            merged=filtered_merged,
            basis_functions=basis_functions,
            domain=domain,
            split_by_sectors=split_by_sectors,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=_bc_basis_directory_arg(bc_basis_directory),
        )
    basis_source = basis_functions.basis_artifact_source or "generated"
    basis_path = getattr(basis_functions, "basis_artifact_path", None)
    _set_domain_attrs(fp_data, filtered_merged.sites, domain)

    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(filtered_merged.sites)):
        inv_inputs = _make_inv_inputs(
            fp_data=fp_data,
            sites=filtered_merged.sites,
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            calculate_min_error=None,
            min_error_per_site=min_error_options["by_site"],
        )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    site_lats, site_lons = _site_release_coordinates(fp_data, filtered_merged.sites)
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(filtered_merged.sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )

    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        basis_artifact_source=basis_source,
        basis_artifact_path=basis_path,
        sites=tuple(filtered_merged.sites),
        averaging_period=tuple(filtered_merged.averaging_period),
        site_lats=site_lats,
        site_lons=site_lons,
    )
