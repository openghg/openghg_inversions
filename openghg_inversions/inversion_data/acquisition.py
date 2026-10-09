"""Retrieve RHIME observations, transport, flux and boundary data.

Acquisition owns store access and consumes the complete site-aligned
selection record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from openghg.dataobjects import FluxData
from openghg.retrieve import get_bc
from openghg.types import SearchError

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data import _site_options
from openghg_inversions.inversion_data._provenance import (
    InputProvenance, MergedDataProvenance, selected_provenance,
)
from openghg_inversions.inversion_data._site_options import (
    SiteOptions, is_column_observation, is_column_platform, is_satellite_platform,
)
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.getters import get_flux_data, get_footprint_data, get_obs_data
from openghg_inversions.inversion_data.scenario import merged_scenario_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquisitionFacts:
    """Known retrieval choices; absent values make no compatibility claim.

    Metadata never copy or compute the borrowed scientific arrays.
    """

    species: str | None = None
    domain: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    emissions_domain: str | None = None
    fp_model: str | None = None
    fp_species: str | None = None
    calibration_scale: str | None = None
    use_bc: bool | None = None
    bc_input: str | None = None
    averaging_error: bool | None = None
    flux_non_finite_check: FluxNonFiniteCheck | None = None


@dataclass
class RhimeMergedData:
    """Borrowed merged datasets, selectors and selected retrieval provenance.

    Construction never computes or copies numerical arrays. Records carry
    acquired or subsequently filtered data in memory. Callers own the lifetime
    of any files backing the datasets they supply.

    Attributes:
        site_data: Merged observations and transport datasets keyed by retained
            site name. Keys must match ``site_options.sites``. Datasets and
            their potentially lazy arrays are borrowed.
        flux_data: Flux datasets keyed by OpenGHG source label. Dataset
            containers and numerical arrays are borrowed.
        site_options: Complete retained acquisition selectors, including site
            order and aligned averaging periods, inlets and other selectors.
        boundary_data: Borrowed boundary-condition dataset, or ``None`` when
            boundary conditions were not acquired.
        split_by_sectors: Whether the acquired flux layout retains individual
            sources for sector-resolved preparation.
        provenance: A :class:`~openghg_inversions.inversion_data.MergedDataProvenance`
            record of OpenGHG software identity and selected input store, UUID
            and data-version identities. Missing input records default to
            unknown identities; unexpected input labels are rejected. These
            descriptions do not establish scientific compatibility.
        acquisition: Known retrieval choices, checked by
            :meth:`validate_for_preparation` when a caller supplies this record
            to a runner. Missing facts remain unknown.
    """

    site_data: dict[str, xr.Dataset]
    flux_data: dict[str, xr.Dataset]
    site_options: _site_options.SiteOptions
    boundary_data: xr.Dataset | None = None
    split_by_sectors: bool = False
    provenance: MergedDataProvenance = field(default_factory=MergedDataProvenance)
    acquisition: AcquisitionFacts = field(default_factory=AcquisitionFacts)

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
        if (
            set(self.provenance.observations) - set(self.site_data)
            or set(self.provenance.footprints) - set(self.site_data)
            or set(self.provenance.flux) - set(self.flux_data)
            or (self.provenance.boundary is not None and self.boundary_data is None)
        ):
            raise ValueError("Provenance identifies inputs absent from the merged data.")
        self.provenance = replace(
            self.provenance,
            observations={site: self.provenance.observations.get(site, InputProvenance()) for site in self.site_data},
            footprints={site: self.provenance.footprints.get(site, InputProvenance()) for site in self.site_data},
            flux={source: self.provenance.flux.get(source, InputProvenance()) for source in self.flux_data},
            boundary=(
                self.provenance.boundary if self.provenance.boundary is not None else InputProvenance()
            ) if self.boundary_data is not None else None,
        )

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
        """Reject supplied data with known facts conflicting with preparation.

        Species and domain comparisons ignore case; date spellings may differ
        when they identify the same instant; timezone-naive dates mean UTC.
        Changing a known window requires
        fresh acquisition: this operation neither slices nor relabels data.
        Missing, ``None`` and ``"unknown"`` historical facts remain unknown.
        Recorded selectors and unrelated model choices are not compared.
        Validation reads metadata only and never computes borrowed arrays.

        Raises:
            ValueError: If a known fact or the source layout conflicts with the
                requested preparation, or a recorded date is invalid.
        """
        requested = {
            "species": species,
            "domain": domain,
            "start_date": start_date,
            "end_date": end_date,
        }
        for name, value in requested.items():
            recorded = getattr(self.acquisition, name)
            if recorded is None or recorded == "unknown":
                continue
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
        context: str = "Merged-data selection",
    ) -> RhimeMergedData:
        """Replace retained site datasets with their selectors and provenance.

        Mapping order determines retained-site order. All sites must already
        belong to this record.
        The result owns new mappings and descriptive metadata; datasets and
        numerical arrays remain borrowed, including any backing open files.

        Raises:
            ValueError: If retained sites are invalid.
        """
        site_options = self.site_options.retain_sites(tuple(site_data), context=context)
        return RhimeMergedData(
            site_data=dict(site_data),
            flux_data=dict(self.flux_data),
            boundary_data=self.boundary_data,
            site_options=site_options,
            split_by_sectors=self.split_by_sectors,
            provenance=self.provenance.retain_sites(site_data),
            acquisition=self.acquisition,
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
        flux_non_finite_check: FluxNonFiniteCheck = "lazy",
    ) -> RhimeMergedData:
        """Acquire fresh merged data using complete aligned selectors.

        Reads OpenGHG stores and retains all selector fields together when sites
        are unavailable. Returned datasets may be lazy and remain borrowed by
        later preparation. ``flux_sources`` names OpenGHG sources; ``averaging_error``
        controls inclusion of observation variability. Store, selector and
        merge errors propagate.
        """
        return _retrieve_inversion_data_from_options(
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
            flux_non_finite_check=flux_non_finite_check,
        )


logger = logging.getLogger(__name__)

def interpolate_flux_to_footprint_grid(
    flux_dict: dict[str, FluxData],
    footprint_data: Any,
) -> dict[str, FluxData]:
    """Interpolate flux density onto a footprint grid without mutating inputs."""
    target = footprint_data.data["fp"]
    interpolated: dict[str, FluxData] = {}
    for source, flux_data in flux_dict.items():
        flux = flux_data.data["flux"].interp(
            lat=target["lat"],
            lon=target["lon"],
            method="nearest",
        )
        dataset = flux.to_dataset(name="flux")
        dataset.attrs = dict(flux_data.data.attrs)
        interpolated[source] = FluxData(
            data=dataset,
            metadata=dict(flux_data.metadata),
        )
    return interpolated


def add_obs_error(sites: Sequence[str], site_data: dict, add_averaging_error: bool = True) -> None:
    """Create `mf_error` variable.

    The `mf_error` variable contains either `mf_repeatability`, `mf_variability`
    or the square root of the sum of the squares of both, if `add_averaging_error` is True.

    This function modifies `site_data` in place, adding `mf_error` and making sure that both
    `mf_repeatability` and `mf_variability` are present.

    Note: OpenGHG resampling pools supplied variability with the spread between input means,
    weighted by observation counts when available. Supplied variability may itself serve as
    instrument uncertainty. If neither variability nor counts is present, resampling instead
    calculates variability from the input concentrations, giving zero for a window with one
    finite observation. Pooling a single input instead retains its supplied variability, subject
    to numerical precision. When counts are present but variability is absent, the weighted
    resampling path does not create variability.

    See :doc:`/development/observation_uncertainty` for missing-value behavior and policies.

    Args:
        sites: list of site names to process
        site_data: dictionary of `ModelScenario` objects, keyed by site names
        add_averaging_error: if True, combine repeatability and variability to make `mf_error`
            variable. Otherwise, `mf_error` will equal `mf_repeatability` if it is present, otherwise
            it will equal `mf_variability`.

    Returns:
        None, modifies `site_data` in place.
    """
    # TODO: do we want to fill missing values in repeatability or variability?
    for site in sites:
        ds = site_data[site]
        mf_long_name = ds.mf.attrs.get("long_name", "")
        mf_units = ds.mf.attrs.get("units", None)

        variability_missing = False
        if "mf_variability" not in ds:
            ds["mf_variability"] = xr.zeros_like(ds.mf)
            variability_missing = True
        ds["mf_variability"].attrs["long_name"] = mf_long_name + "_variability"
        ds["mf_variability"].attrs["units"] = mf_units

        if "mf_repeatability" not in ds:
            if variability_missing:
                raise ValueError(f"Obs data for site {site} is missing both repeatability and variability.")

            ds["mf_repeatability"] = xr.zeros_like(ds.mf_variability)

            ds["mf_error"] = ds["mf_variability"]

            if add_averaging_error:
                logger.info(
                    "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site %s", site
                )

        elif add_averaging_error:
            # Fill with zeros so that if one of repeatability and variability is not NaN, then mf_error will not be NaN.
            ds["mf_error"] = np.sqrt(
                ds["mf_repeatability"].fillna(0) ** 2 + ds["mf_variability"].fillna(0) ** 2
            )
        else:
            ds["mf_error"] = ds["mf_repeatability"]

        ds["mf_repeatability"].attrs["long_name"] = mf_long_name + "_repeatability"
        ds["mf_repeatability"].attrs["units"] = mf_units
        ds["mf_error"].attrs["long_name"] = mf_long_name + "_error"
        ds["mf_error"].attrs["units"] = mf_units

        # warnings/info for debugging
        err0 = (ds["mf_error"] == 0) | (
            ds["mf_error"].isnull()
        )  # might have NaN if add_averaging_error is False

        if err0.any():
            percent0 = 100 * err0.mean()
            logger.warning(
                (
                    "`mf_error` is zero/nan for %.2f percent of times at site %s;"
                    "filling with max(median(mf_error), std(mf))."
                ),
                percent0,
                site,
            )

            mf_err_da = ds["mf_error"].as_numpy()  # load into memory to avoid Dask issues
            fill_value = np.nanmax(
                [
                    mf_err_da.where(mf_err_da != 0).dropna(dim="time").median(),
                    ds["mf"].std(dim="time"),
                ]
            )
            ds["mf_error"] = mf_err_da.where(mf_err_da != 0, fill_value)
            info_msg = (
                "If `averaging_period` matches the frequency of the obs data, then `mf_variability` "
                "will be zero. Try setting `averaging_period = None`."
            )
            logger.info(info_msg)


def _retrieve_inversion_data_from_options(
    *,
    site_options: SiteOptions,
    species: str,
    domain: str,
    start_date: str,
    end_date: str,
    calibration_scale: str | None = None,
    fp_model: str | None = None,
    fp_species: str | None = None,
    emissions_name: Sequence[str] | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    bc_store: str | None = None,
    obs_store: str | list[str] | None = None,
    footprint_store: str | list[str] | None = None,
    emissions_store: str | None = None,
    emissions_domain: str | None = None,
    split_by_sectors: bool = False,
    averagingerror: bool = True,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimeMergedData:
    """Acquire the modern dataset record from already resolved site selectors."""
    merged_sites: dict[str, xr.Dataset] = {}
    observation_provenance = {}
    footprint_provenance = {}
    boundary_provenance = None

    # Get flux data
    if emissions_name is None:
        raise ValueError("`emissions_name` must be specified")

    flux_dict = get_flux_data(
        sources=emissions_name,
        species=species,
        domain=emissions_domain or domain,
        start_date=start_date,
        end_date=end_date,
        store=emissions_store,
        flux_non_finite_check=flux_non_finite_check,
    )
    retained_flux_dict = flux_dict
    flux_provenance = {source: selected_provenance(value, emissions_store) for source, value in flux_dict.items()}

    # Get BC data
    if use_bc is True:
        try:
            bc_data = get_bc(
                species=species,
                domain=domain,
                bc_input=bc_input,
                start_date=start_date,
                end_date=end_date,
                store=bc_store,
            )
        except SearchError as e:
            raise SearchError("Could not find matching boundary conditions.") from e
        else:
            # This public getter rejects multiple UUIDs and selects latest.
            boundary_provenance = selected_provenance(
                bc_data, bc_store, requested_version="latest"
            )
    else:
        bc_data = None

    # get obs and footprints, and make scenarios for each site
    check_scales = set()
    site_indices_to_keep = []
    output_units: str | None = None

    keep_variables = [
        f"{species}",
        f"{species}_variability",
        f"{species}_repeatability",
        f"{species}_number_of_observations",
        "inlet",  # needed if multiple inlets combined
        "inlet_height",  # sometimes needed if inlet='multiple' (may be outdated soon)
    ]
    warnings.warn(f"Dropping all variables besides {keep_variables}", stacklevel=2)
    for i, site in enumerate(site_options.sites):
        # Get observations data
        site_platform = site_options.platform[i]
        if isinstance(site_platform, str) and site_platform.lower() == "flask":
            avg_period = None
        else:
            avg_period = site_options.averaging_period[i]

        site_data = get_obs_data(
            site=site,
            species=species,
            inlet=site_options.inlet[i],
            start_date=start_date,
            domain=domain,
            platform=site_platform,
            end_date=end_date,
            data_level=site_options.obs_data_level[i],
            average=avg_period,
            instrument=site_options.instrument[i],
            calibration_scale=calibration_scale,
            max_level=site_options.max_level[i],
            stores=obs_store,
            keep_variables=keep_variables,
        )

        if site_data is None:
            print(f"No obs. found, continuing model run without {site}.\n")
            continue

        # Get footprints data
        footprint_data = get_footprint_data(
            site=site,
            domain=domain,
            platform=site_platform,
            fp_height=site_options.fp_height[i],
            start_date=start_date,
            end_date=end_date,
            model=fp_model,
            met_model=site_options.met_model[i],
            fp_species=fp_species,
            averaging_period=site_options.averaging_period[i],
            time_resolved=site_options.time_resolved[i],
            obs_data=site_data,
            stores=footprint_store,
        )
        if footprint_data is None:
            print(
                f"\nNo footprint data found for {site} with inlet/height {site_options.fp_height[i]}, model {fp_model}, and domain {domain}.",
                f"Check these values.\nContinuing model run without {site}.\n",
            )
            continue  # skip this site

        scenario_platform = (
            "site-column"
            if is_column_observation(site_options.inlet[i], site_platform) and not is_column_platform(site_platform)
            else site_platform
        )
        scenario_flux_dict = (
            interpolate_flux_to_footprint_grid(flux_dict, footprint_data)
            if emissions_domain is not None and emissions_domain.lower() != domain.lower()
            else flux_dict
        )
        if scenario_flux_dict is not flux_dict:
            retained_flux_dict = scenario_flux_dict

        try:
            scenario_combined = merged_scenario_data(
                site_data,
                footprint_data,
                scenario_flux_dict,
                bc_data,
                platform=scenario_platform,
                max_level=site_options.max_level[i],
                split_by_sectors=split_by_sectors,
                output_units=output_units,
            )
        except (TypeError, ValueError) as exc:
            if output_units is None:
                raise
            raise ValueError(
                f"Could not merge site {site!r} using target observation units {output_units!r}."
            ) from exc
        if output_units is None:
            scenario_units = scenario_combined["mf"].attrs.get("units")
            if not isinstance(scenario_units, str) or not scenario_units:
                raise ValueError(f"No observation units detected for the first retained site {site!r}.")
            mole_fraction_unit_scale(
                scenario_units,
                context=f"site {site!r} variable 'mf'",
            )
            output_units = scenario_units
        merged_sites[site] = scenario_combined
        # Observation retrieval can combine UUIDs and discard their selected versions;
        # never infer those versions from the remaining catalog metadata.
        observation_provenance[site] = selected_provenance(site_data, obs_store)
        footprint_provenance[site] = selected_provenance(
            footprint_data, footprint_store, requested_version="latest"
        )

        if not is_satellite_platform(site_platform):
            check_scales.add(scenario_combined.scale)

        site_indices_to_keep.append(i)
    if len(site_indices_to_keep) == 0:
        raise SearchError("No site data found. Exiting process.")

    retained = site_options.select_indices(site_indices_to_keep)
    for site, selector in zip(retained.sites, retained.time_resolved, strict=True):
        merged_sites[site].attrs["openghg_inversions_time_resolved"] = str(selector).lower()

    # if "satellite" not in footprint_data.metadata:
    # check for consistency of calibration scales
    if len(check_scales) > 1:
        msg = f"Not all sites using the same calibration scale: {len(check_scales)} scales found."
        logger.warning(msg)

    # create `mf_error`
    add_obs_error(retained.sites, merged_sites, add_averaging_error=averagingerror)

    return RhimeMergedData(
        site_data=merged_sites,
        flux_data={source: value.data for source, value in retained_flux_dict.items()},
        boundary_data=bc_data.data if bc_data is not None else None,
        site_options=retained,
        split_by_sectors=split_by_sectors,
        provenance=MergedDataProvenance.from_retrieval(
            observations=observation_provenance,
            footprints=footprint_provenance,
            flux=flux_provenance,
            boundary=boundary_provenance,
        ),
        acquisition=AcquisitionFacts(
            species=species,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            emissions_domain=emissions_domain,
            fp_model=fp_model,
            fp_species=fp_species,
            calibration_scale=calibration_scale,
            use_bc=use_bc,
            bc_input=bc_input,
            averaging_error=averagingerror,
            flux_non_finite_check=flux_non_finite_check,
        ),
    )
