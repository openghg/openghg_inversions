"""Retrieve or reload RHIME observations, transport, flux and boundary data.

Acquisition owns store/cache access and consumes the complete site-aligned
selection record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import xarray as xr

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data import _site_options
from openghg_inversions.inversion_data.get_data import (
    _retrieve_inversion_data_from_options,
)
from openghg_inversions.inversion_data.serialise import OutputFormat, load_merged_data


@dataclass
class RhimeMergedData:
    """Borrowed merged datasets, selectors and selected retrieval provenance.

    ``site_data`` contains merged observations and transport for each retained
    site; ``flux_data`` maps source labels to datasets. Construction never
    computes or copies numerical arrays. ``save`` writes the acquisition
    boundary, before configured filters, basis functions or sensitivities.
    ``boundary_data`` holds optional boundary conditions. ``split_by_sectors``
    identifies the source layout; ``provenance`` contains selected input and
    OpenGHG identifiers; ``acquisition`` records retrieval facts and processing
    stage. Records may carry acquired or subsequently filtered data, but saving
    requires an explicit ``acquisition["stage"] == "acquired"`` declaration.
    Modern loading restores selectors and never accesses OpenGHG stores.
    """

    site_data: dict[str, xr.Dataset]
    flux_data: dict[str, xr.Dataset]
    site_options: _site_options.SiteOptions
    boundary_data: xr.Dataset | None = None
    split_by_sectors: bool = False
    provenance: dict = field(default_factory=dict)
    acquisition: dict = field(default_factory=dict)
    _artifact: xr.DataTree | None = field(default=None, init=False, repr=False)
    _zip_store: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if set(self.site_data) != set(self.site_options.sites):
            raise ValueError("Site datasets must match retained SiteOptions labels.")
        datasets = [*self.site_data.values(), *self.flux_data.values()]
        if self.boundary_data is not None:
            datasets.append(self.boundary_data)
        if any(not isinstance(data, xr.Dataset) for data in datasets):
            raise TypeError("RhimeMergedData accepts xarray datasets, not OpenGHG wrappers.")
        if any(not isinstance(source, str) for source in self.flux_data):
            raise TypeError("Flux source labels must be strings.")
        from ._merged_artifact import selected_provenance

        inputs = {
            **{f"observations:{site}": selected_provenance(None) for site in self.site_data},
            **{f"footprints:{site}": selected_provenance(None) for site in self.site_data},
            **{f"flux:{source}": selected_provenance(data) for source, data in self.flux_data.items()},
            **(
                {"boundary": selected_provenance(self.boundary_data)}
                if self.boundary_data is not None
                else {}
            ),
        }
        if not self.provenance:
            self.provenance = {"openghg": {"version": "unknown", "commit": "unknown"}, "inputs": inputs}
        if set(self.provenance) != {"openghg", "inputs"}:
            raise ValueError("Provenance contains unsupported metadata.")
        software = self.provenance["openghg"]
        if set(software) != {"version", "commit"} or any(
            not isinstance(value, str) for value in software.values()
        ):
            raise ValueError("OpenGHG provenance requires version and commit strings (or 'unknown').")
        if set(self.provenance["inputs"]) != set(inputs):
            raise ValueError("Provenance must identify each retained input.")
        for identity in self.provenance["inputs"].values():
            if set(identity) != {"store", "uuid", "dataversion"} or any(
                not isinstance(value, str | int)
                and not (isinstance(value, list) and all(isinstance(item, str | int) for item in value))
                for value in identity.values()
            ):
                raise ValueError("Input provenance supports only store, uuid and dataversion.")
        allowed_facts = {
            "stage",
            "species",
            "domain",
            "start_date",
            "end_date",
            "emissions_domain",
            "fp_model",
            "fp_species",
            "calibration_scale",
            "use_bc",
            "bc_input",
            "averaging_error",
            "flux_non_finite_check",
            "legacy_import",
        }
        if set(self.acquisition) - allowed_facts or any(
            value is not None and not isinstance(value, str | bool) for value in self.acquisition.values()
        ):
            raise ValueError("Unsupported acquisition metadata.")

    @property
    def sites(self) -> tuple[str, ...]:
        """Retained site names."""
        return self.site_options.sites

    @property
    def averaging_period(self) -> tuple[str | None, ...]:
        """Retained averaging periods aligned to sites."""
        return self.site_options.averaging_period

    @property
    def platform(self) -> tuple[str | None, ...]:
        """Retained observation platforms aligned to sites."""
        return self.site_options.platform

    def validate_for_preparation(
        self,
        *,
        species: str,
        domain: str,
        start_date: str,
        end_date: str,
        split_by_sectors: bool,
    ) -> None:
        """Bind known acquisition facts to a downstream scientific request.

        Species and domain comparisons ignore case; date spellings may differ
        when they identify the same instant; timezone-naive dates mean UTC.
        Changing a known window requires
        fresh acquisition: this operation neither slices nor relabels data.
        Missing, ``None`` and ``"unknown"`` historical facts remain unknown.
        Recorded selectors and unrelated model choices are not compared.
        Validation reads metadata only and never computes borrowed arrays.

        Raises:
            ValueError: If a known fact or the source layout conflicts with the
                requested preparation.
        """
        requested = {
            "species": species,
            "domain": domain,
            "start_date": start_date,
            "end_date": end_date,
        }
        for name, value in requested.items():
            recorded = self.acquisition.get(name)
            if recorded is None or recorded == "unknown":
                continue
            if not isinstance(recorded, str):
                raise ValueError(f"Merged-data acquisition {name} must be a string or unknown.")
            if name in {"start_date", "end_date"}:
                compatible = pd.to_datetime(recorded, utc=True) == pd.to_datetime(value, utc=True)
            else:
                compatible = recorded.casefold() == value.casefold()
            if not compatible:
                raise ValueError(
                    f"Merged data has incompatible {name}: acquired {recorded!r}, requested {value!r}. "
                    "Acquire data for the requested species, domain and window."
                )
        if self.split_by_sectors != split_by_sectors:
            raise ValueError("Merged data has an incompatible split_by_sectors layout.")

    def with_site_data(
        self,
        site_data: Mapping[str, xr.Dataset],
        *,
        stage: str,
        context: str = "Merged-data selection",
    ) -> RhimeMergedData:
        """Replace retained site datasets with their selectors and provenance.

        Mapping order determines retained-site order. All sites must already
        belong to this record. ``stage`` explicitly declares the resulting
        processing stage (``"acquired"``, ``"filtered"`` or ``"unknown"``).
        The result owns new mappings and descriptive metadata; datasets and
        numerical arrays remain borrowed, including any backing open files.
        Close the original loaded record only after using all borrowed data.

        Raises:
            ValueError: If retained sites or the stage are invalid, or a
                filtered record is relabelled as acquired.
        """
        if stage not in {"acquired", "filtered", "unknown"}:
            raise ValueError("stage must be unknown, acquired or filtered.")
        if self.acquisition.get("stage") == "filtered" and stage == "acquired":
            raise ValueError("Filtered merged data cannot be relabelled as acquired.")
        site_options = self.site_options.retain_sites(tuple(site_data), context=context)
        provenance = deepcopy(self.provenance)
        provenance["inputs"] = {
            key: value
            for key, value in provenance["inputs"].items()
            if not key.startswith(("observations:", "footprints:"))
            or key.split(":", 1)[1] in site_data
        }
        return RhimeMergedData(
            site_data=dict(site_data),
            flux_data=dict(self.flux_data),
            boundary_data=self.boundary_data,
            site_options=site_options,
            split_by_sectors=self.split_by_sectors,
            provenance=provenance,
            acquisition={**self.acquisition, "stage": stage},
        )

    @classmethod
    def from_legacy_fp_all(
        cls, fp_all: dict, site_options: _site_options.SiteOptions, *, acquisition: dict | None = None
    ) -> RhimeMergedData:
        """Adapt the remaining legacy scientific producers (#821; remove in 0.9)."""
        from ._merged_artifact import selected_provenance

        stored_stage = fp_all.get(".artifact_stage")
        acquisition = dict(acquisition or {})
        requested_stage = acquisition.get("stage", stored_stage or "unknown")
        if stored_stage is not None and requested_stage != stored_stage:
            raise ValueError("Requested acquisition stage conflicts with the stored artifact stage.")
        acquisition["stage"] = requested_stage
        fluxes = fp_all.get(".flux", {})
        boundary = fp_all.get(".bc")
        provenance = fp_all.get(
            ".provenance",
            {
                "openghg": {"version": "unknown", "commit": "unknown"},
                "inputs": {
                    **{f"flux:{source}": selected_provenance(value) for source, value in fluxes.items()},
                    **({"boundary": selected_provenance(boundary)} if boundary is not None else {}),
                    **{f"observations:{site}": selected_provenance(None) for site in site_options.sites},
                    **{f"footprints:{site}": selected_provenance(None) for site in site_options.sites},
                },
            },
        )
        retained_inputs = {
            f"{kind}:{site}" for kind in ("observations", "footprints") for site in site_options.sites
        }
        retained_inputs.update(f"flux:{source}" for source in fluxes)
        if boundary is not None:
            retained_inputs.add("boundary")
        provenance = {
            **provenance,
            "inputs": {key: value for key, value in provenance["inputs"].items() if key in retained_inputs},
        }
        return cls(
            site_data={site: fp_all[site] for site in site_options.sites},
            flux_data={
                source: value if isinstance(value, xr.Dataset) else value.data
                for source, value in fluxes.items()
            },
            boundary_data=boundary if isinstance(boundary, xr.Dataset) else getattr(boundary, "data", None),
            site_options=site_options,
            split_by_sectors=bool(fp_all.get(".split_by_sectors", False)),
            provenance=provenance,
            acquisition=acquisition or {},
        )

    def to_legacy_fp_all(self) -> dict:
        """Adapt to remaining wrapper-based scientific consumers (#821; remove in 0.9).

        Only this explicit adapter reconstructs OpenGHG wrappers. Flux and boundary
        dataset containers are shallow copies; site datasets remain borrowed.
        Consumers must copy site containers before assigning variables or attrs.
        Numerical arrays remain shared.
        """
        from openghg.dataobjects import BoundaryConditionsData, FluxData

        result = dict(self.site_data)
        result[".flux"] = {
            source: FluxData(data=data.copy(deep=False), metadata={"data_type": "flux"})
            for source, data in self.flux_data.items()
        }
        if self.boundary_data is not None:
            result[".bc"] = BoundaryConditionsData(data=self.boundary_data.copy(deep=False), metadata={})
        result[".split_by_sectors"] = self.split_by_sectors
        if self.acquisition.get("stage") in {"acquired", "filtered"}:
            result[".artifact_stage"] = self.acquisition["stage"]
        return result

    @classmethod
    def load(
        cls,
        merged_data_dir: str | Path,
        *,
        species: str | None = None,
        start_date: str | None = None,
        output_name: str | None = None,
        merged_data_name: str | None = None,
        output_format: OutputFormat | None = None,
    ) -> RhimeMergedData:
        """Open one versioned artifact lazily, restoring all acquisition selectors.

        Missing, corrupt, legacy or unsupported artifacts raise without format
        fallback or reacquisition. Call ``close`` after consuming loaded arrays.
        Names without a suffix use ``output_format`` or ``zarr.zip``.
        ``species`` and ``start_date`` only help locate the file; use
        :meth:`validate_for_preparation` to check a scientific reuse request.
        """
        from ._merged_artifact import artifact_path, load_artifact

        path, output_format = artifact_path(
            merged_data_dir, species, start_date, output_name, merged_data_name, output_format
        )
        return load_artifact(cls, path, output_format)

    @classmethod
    def load_legacy(
        cls,
        merged_data_dir: str | Path,
        *,
        site_options: _site_options.SiteOptions,
        species: str | None = None,
        start_date: str | None = None,
        output_name: str | None = None,
        merged_data_name: str | None = None,
        output_format: OutputFormat | None = None,
        split_by_sectors: bool = False,
        acquisition_stage: str = "unknown",
    ) -> RhimeMergedData:
        """Import an old cache using its missing selectors (deprecated in 0.8).

        Removed in 0.9. Supply the complete selectors used for acquisition;
        legacy files cannot recover them. Calling ``save`` on the returned
        record migrates it to the modern format only when ``acquisition_stage`` is
        explicitly ``"acquired"``. The default ``"unknown"`` preserves uncertainty
        in old files. A stored filtered-stage marker cannot be overridden.
        """
        if acquisition_stage not in {"unknown", "acquired", "filtered"}:
            raise ValueError("acquisition_stage must be unknown, acquired or filtered.")
        if output_format is not None and output_format not in {"netcdf", "zarr", "zarr.zip"}:
            raise ValueError(f"Unsupported merged-data format {output_format!r}.")
        warnings.warn(
            "RhimeMergedData.load_legacy is deprecated in 0.8 and will be removed in 0.9; save the result to migrate.",
            DeprecationWarning,
            stacklevel=2,
        )
        fp_all = load_merged_data(
            merged_data_dir, species, start_date, output_name, merged_data_name, output_format=output_format
        )
        stored_stage = fp_all.get(".artifact_stage")
        if stored_stage is not None and acquisition_stage not in {"unknown", stored_stage}:
            raise ValueError("Requested acquisition_stage conflicts with the stored artifact stage.")
        acquisition_stage = stored_stage or acquisition_stage
        _validate_loaded_time_resolved_selector(fp_all, site_options)
        _validate_loaded_sector_layout(fp_all, split_by_sectors=split_by_sectors)
        site_options = _drop_sites_missing_from_loaded_data(fp_all=fp_all, site_options=site_options)
        return cls.from_legacy_fp_all(
            fp_all, site_options, acquisition={"stage": acquisition_stage, "legacy_import": True}
        )

    @classmethod
    def from_options(
        cls,
        *,
        species: str,
        site_options: _site_options.SiteOptions,
        domain: str,
        start_date: str,
        end_date: str,
        output_name: str,
        flux_sources: Sequence[str] | None,
        split_by_sectors: bool = False,
        bc_store: str = "user",
        obs_store: str = "user",
        footprint_store: str = "user",
        emissions_store: str = "user",
        emissions_domain: str | None = None,
        fp_model: str | None = None,
        fp_species: str | None = None,
        calibration_scale: str | None = None,
        use_bc: bool = True,
        bc_input: str | None = None,
        averaging_error: bool = True,
        save_merged_data: bool = False,
        merged_data_dir: str | None = None,
        merged_data_name: str | None = None,
        flux_non_finite_check: FluxNonFiniteCheck = "lazy",
    ) -> RhimeMergedData:
        """Acquire fresh merged data using complete aligned selectors.

        Reads OpenGHG stores and retains all selector fields together when sites
        are unavailable. Returned datasets may be lazy and remain borrowed by
        later preparation. ``flux_sources`` names OpenGHG sources; ``averaging_error``
        controls inclusion of observation variability. Optional saving uses the
        versioned dataset-only codec and is disabled by default. Retrieval never
        attempts cache loading; use :meth:`load` for an explicit artifact request.
        Store, selector, merge and optional serialization errors propagate.
        """
        if save_merged_data and merged_data_dir is None:
            raise ValueError("Saving merged data requires merged_data_dir.")
        # Requested options remain authoritative; legacy metadata is redundant.
        fp_all, retained_sites, *_ = _retrieve_inversion_data_from_options(
            species=species,
            site_options=site_options,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            fp_model=fp_model,
            fp_species=fp_species,
            emissions_name=flux_sources,
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
            save_merged_data=False,
            merged_data_name=merged_data_name,
            merged_data_dir=merged_data_dir,
            output_name=output_name,
            flux_non_finite_check=flux_non_finite_check,
        )
        site_options = site_options.retain_sites(retained_sites, context="Data gathering")
        fp_all = _select_fp_all_sites(fp_all, site_options.sites)
        result = cls.from_legacy_fp_all(
            fp_all,
            site_options,
            acquisition={
                "stage": "acquired",
                "species": species,
                "domain": domain,
                "start_date": start_date,
                "end_date": end_date,
                "emissions_domain": emissions_domain,
                "fp_model": fp_model,
                "fp_species": fp_species,
                "calibration_scale": calibration_scale,
                "use_bc": use_bc,
                "bc_input": bc_input,
                "averaging_error": averaging_error,
                "flux_non_finite_check": flux_non_finite_check,
            },
        )
        if save_merged_data and merged_data_dir is not None:
            result.save(
                merged_data_dir,
                species=species,
                start_date=start_date,
                output_name=output_name,
                merged_data_name=merged_data_name,
            )
        return result

    def save(
        self,
        merged_data_dir: str | Path,
        *,
        species: str | None = None,
        start_date: str | None = None,
        output_name: str | None = None,
        merged_data_name: str | None = None,
        output_format: OutputFormat = "zarr.zip",
    ) -> None:
        """Serialize explicitly acquired datasets and selectors, executing lazy arrays.

        ``acquisition["stage"]`` must be ``"acquired"``. Unknown-phase legacy
        imports and filtered records are rejected; use a new acquisition or
        explicitly identify a known pre-filter legacy input when importing.
        Names ending in ``.nc``, ``.zarr`` or ``.zarr.zip`` select the format;
        otherwise ``output_format`` applies. Naming requires ``merged_data_name``
        or all of ``species``, ``start_date`` and ``output_name``.
        """
        from ._merged_artifact import artifact_path, save_artifact

        path, output_format = artifact_path(
            merged_data_dir, species, start_date, output_name, merged_data_name, output_format
        )
        save_artifact(self, path, output_format)

    def close(self) -> None:
        """Release files backing a loaded artifact after its arrays are consumed."""
        if self._artifact is not None:
            self._artifact.close()
        if self._zip_store is not None:
            self._zip_store.close()


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    site_options: _site_options.SiteOptions,
) -> _site_options.SiteOptions:
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
    site_options: _site_options.SiteOptions,
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
    """Reject supplied or cached merged data with a different sector layout.

    The serialized ``.split_by_sectors`` marker records whether the cache
    contains source-resolved sensitivities.  Missing provenance is treated as
    the legacy combined layout, so it cannot be relabelled as sector-resolved.

    Args:
        fp_all: Supplied or loaded merged data and its layout metadata.
        split_by_sectors: Whether the current run requires source-resolved
            sensitivities.

    Raises:
        ValueError: If the sector layout cannot satisfy this run.
    """
    stored_split_by_sectors = bool(fp_all.get(".split_by_sectors", False))
    if stored_split_by_sectors != split_by_sectors:
        raise ValueError(
            "Merged data has an incompatible `split_by_sectors` layout: "
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
    averaging_period: _site_options.SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: Sequence[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    emissions_domain: str | None = None,
    met_model: _site_options.SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: _site_options.SiteStringOption = None,
    fp_species: str | None = None,
    time_resolved: _site_options.SiteBooleanOption = None,
    inlet: _site_options.SiteInletOption = None,
    instrument: _site_options.SiteStringOption = None,
    max_level: _site_options.SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: _site_options.SiteStringOption = None,
    platform: _site_options.SiteStringOption = None,
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
    artifact. Both paths retain one complete :class:`_site_options.SiteOptions` record. Explicit
    reload errors propagate without attempting fresh acquisition.

    Returns:
        Merged per-site data and aligned retained-site options.

    Raises:
        ValueError: If site options are invalid, no requested sites are loaded,
            or retrieval returns invalid retained-site names.
    """
    site_options = _site_options.SiteOptions.from_inputs(
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
    if reload_merged_data:
        if merged_data_dir is None:
            raise ValueError("Explicit merged-data reload requires `merged_data_dir`.")
        merged = RhimeMergedData.load(
            merged_data_dir,
            species=species,
            start_date=start_date,
            output_name=output_name,
            merged_data_name=merged_data_name,
        )
        try:
            merged.validate_for_preparation(
                species=species,
                domain=domain,
                start_date=start_date,
                end_date=end_date,
                split_by_sectors=split_by_sectors,
            )
        except ValueError:
            merged.close()
            raise
        return merged
    return RhimeMergedData.from_options(
        species=species,
        site_options=site_options,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        flux_sources=flux_sources,
        split_by_sectors=split_by_sectors,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        emissions_domain=emissions_domain,
        fp_model=fp_model,
        fp_species=fp_species,
        calibration_scale=calibration_scale,
        use_bc=use_bc,
        bc_input=bc_input,
        averaging_error=averaging_error,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        flux_non_finite_check=flux_non_finite_check,
    )
