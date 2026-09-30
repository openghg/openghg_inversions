"""Retrieve or reload RHIME observations, transport, flux and boundary data.

Acquisition owns store/cache access and the complete site-aligned selection
record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import xarray as xr

from openghg_inversions._timing import timed
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data._site_options import (
    expand_site_boolean_option,
    expand_site_option,
    is_column_observation,
)
from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data

SiteStringOption = Sequence[str | None] | str | None
SiteInletOption = Sequence[str | slice | None] | str | None
SiteIntegerOption = Sequence[int | None] | int | None
SiteBooleanOption = Sequence[bool | None] | bool | None


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


def _normalise_site_booleans(
    value: SiteBooleanOption,
    *,
    length: int,
    name: str,
) -> list[bool | None]:
    """Normalize one optional boolean selector per requested site."""
    return list(expand_site_boolean_option(value, nsites=length, name=name))


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
    time_resolved: tuple[bool | None, ...]

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
            "time_resolved",
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
        time_resolved: SiteBooleanOption = None,
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
            time_resolved=tuple(_normalise_site_booleans(time_resolved, length=nsites, name="time_resolved")),
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
            time_resolved=select(self.time_resolved),
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
class RhimeMergedData:
    """Merged RHIME data and complete site-aligned metadata between stages.

    Args:
        fp_all: Merged per-site datasets plus shared flux, boundary-condition,
            and calibration entries.
        site_options: Complete site-aligned acquisition options retained after
            retrieval or filtering.

    Notes:
        This is a supported orchestration handoff. Its datasets remain
        backend-neutral and may be Dask-backed; later stages must treat them as
        borrowed.
    """

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

    @property
    def platform(self) -> tuple[str | None, ...]:
        """Retained observation platforms aligned to :attr:`sites`."""
        return self.site_options.platform


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


def _validate_loaded_time_resolved_selector(
    fp_all: Mapping[str, Any],
    site_options: _SiteOptions,
) -> None:
    """Reject cached sites whose explicit time-resolution selector differs."""
    mismatched_sites: list[str] = []
    for site, selector in zip(site_options.sites, site_options.time_resolved, strict=True):
        if selector is None or site not in fp_all:
            continue
        site_data = fp_all[site]
        cached_selector = (
            site_data.attrs.get("openghg_inversions_time_resolved")
            if isinstance(site_data, xr.Dataset)
            else None
        )
        if cached_selector != str(selector).lower():
            mismatched_sites.append(site)
    if mismatched_sites:
        raise ValueError(
            "Loaded merged data does not match the requested `time_resolved` selector for "
            f"site(s): {mismatched_sites!r}."
        )


def _validate_loaded_sector_layout(fp_all: Mapping[str, Any], *, split_by_sectors: bool) -> None:
    """Reject a cached merged-data artifact with a different sector layout.

    The serialized ``.split_by_sectors`` marker records whether the cache
    contains source-resolved sensitivities.  Missing provenance is treated as
    the legacy combined layout, so it cannot be relabelled as sector-resolved.

    Args:
        fp_all: Loaded merged-data artifact and its serialized metadata.
        split_by_sectors: Whether the current run requires source-resolved
            sensitivities.

    Raises:
        ValueError: If the cached sector layout cannot satisfy this run.
    """
    stored_split_by_sectors = bool(fp_all.get(".split_by_sectors", False))
    if stored_split_by_sectors != split_by_sectors:
        raise ValueError(
            "Loaded merged data has an incompatible `split_by_sectors` layout: "
            f"artifact split_by_sectors={stored_split_by_sectors!r}, "
            f"requested split_by_sectors={split_by_sectors!r}."
        )


def _select_fp_all_sites(fp_all: dict, sites: Sequence[str]) -> dict:
    """Keep requested sites and shared entries."""
    site_names = set(sites)
    return {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}


def _retrieve_or_reload_merged_data(
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
    emissions_domain: str | None = None,
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    time_resolved: SiteBooleanOption = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimeMergedData:
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
            or retrieval returns invalid retained-site names.
    """
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
        time_resolved=time_resolved,
    )
    fp_all: dict | None = None
    if reload_merged_data and merged_data_dir is not None:
        try:
            fp_all = load_merged_data(merged_data_dir, species, start_date, output_name, merged_data_name)
        except ValueError as exc:
            print(f"{exc}, re-running data merge.")
    elif reload_merged_data:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    if fp_all is not None:
        _validate_loaded_time_resolved_selector(fp_all, site_options)
        _validate_loaded_sector_layout(fp_all, split_by_sectors=split_by_sectors)
        print("Successfully read in merged data.\n")
        fp_all[".split_by_sectors"] = split_by_sectors
        site_options = _drop_sites_missing_from_loaded_data(fp_all=fp_all, site_options=site_options)
    else:
        # Requested options remain authoritative; legacy metadata is redundant.
        fp_all, retained_sites, *_ = data_processing_surface_notracer(
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
            time_resolved=list(site_options.time_resolved),
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
            emissions_domain=emissions_domain,
            split_by_sectors=split_by_sectors,
            averagingerror=averaging_error,
            save_merged_data=save_merged_data,
            merged_data_name=merged_data_name,
            merged_data_dir=merged_data_dir,
            output_name=output_name,
            flux_non_finite_check=flux_non_finite_check,
        )
        site_options = site_options.retain_sites(retained_sites, context="Data gathering")

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

    return RhimeMergedData(
        fp_all=fp_all,
        site_options=site_options,
    )



def retrieve_or_reload_rhime_data(
    data_args: Mapping[str, Any],
    *,
    multisector: bool,
    merged_data: RhimeMergedData | None = None,
) -> RhimeMergedData:
    """Retrieve, reload, or accept externally supplied merged RHIME data.

    Passing ``merged_data`` is the explicit no-I/O path.  The object remains
    borrowed and is returned unchanged after a sector-layout compatibility check.
    Otherwise this stage may read OpenGHG stores or a local merged artifact,
    optionally write merged data, sanitize flux arrays, print progress, and
    emit warnings.  ``data_args`` is never mutated.

    Raises:
        ValueError: If ``data_args`` requests unsupported ``use_tracer=True``
            or the supplied merged data has an incompatible sector layout.
    """
    if data_args.get("use_tracer", False):
        raise ValueError("`use_tracer=True` is not supported; tracer inversions are not implemented.")
    if merged_data is not None:
        stored_multisector = bool(merged_data.fp_all.get(".split_by_sectors", False))
        if stored_multisector != multisector:
            raise ValueError(
                "External RHIME merged data has an incompatible sector layout: "
                f"artifact split_by_sectors={stored_multisector!r}, "
                f"runner multisector={multisector!r}."
            )
        return merged_data

    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(data_args["sites"]),
        split_by_sectors=multisector,
    ):
        return _retrieve_or_reload_merged_data(
            species=data_args["species"],
            sites=data_args["sites"],
            domain=data_args["domain"],
            averaging_period=data_args["averaging_period"],
            start_date=data_args["start_date"],
            end_date=data_args["end_date"],
            output_name=data_args["output_name"],
            flux_sources=data_args["flux_sources"],
            split_by_sectors=multisector,
            bc_store=data_args["bc_store"],
            obs_store=data_args["obs_store"],
            footprint_store=data_args["footprint_store"],
            emissions_store=data_args["emissions_store"],
            emissions_domain=data_args["emissions_domain"],
            met_model=data_args["met_model"],
            fp_model=data_args["fp_model"],
            fp_height=data_args["fp_height"],
            fp_species=data_args["fp_species"],
            time_resolved=data_args["time_resolved"],
            inlet=data_args["inlet"],
            instrument=data_args["instrument"],
            max_level=data_args["max_level"],
            calibration_scale=data_args["calibration_scale"],
            obs_data_level=data_args["obs_data_level"],
            platform=data_args["platform"],
            use_bc=data_args["use_bc"],
            bc_input=data_args["bc_input"],
            averaging_error=data_args["averaging_error"],
            reload_merged_data=data_args["reload_merged_data"],
            save_merged_data=data_args["save_merged_data"],
            merged_data_dir=data_args["merged_data_dir"],
            merged_data_name=data_args["merged_data_name"],
            flux_non_finite_check=data_args["flux_non_finite_check"],
        )
