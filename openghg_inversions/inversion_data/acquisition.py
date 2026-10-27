"""Retrieve RHIME observations, transport, flux and boundary data.

Acquisition owns store access and consumes the complete site-aligned
selection record. It returns a borrowed ``MergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import pandas as pd
import xarray as xr
from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData, ObsData
from openghg.retrieve import get_bc
from openghg.types import SearchError

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data import _site_options
from openghg_inversions.inversion_data._provenance import (
    InputProvenance, MergedDataProvenance, selected_provenance,
)
from openghg_inversions.inversion_data._site_options import (
    is_column_observation, is_column_platform, is_satellite_platform,
)
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.getters import get_flux_data, get_footprint_data, get_obs_data
from openghg_inversions.inversion_data.scenario import merged_scenario_data
from openghg_inversions.inversion_data.surface import (
    TransportMode, _align_footprint, _dataset, _native_releases, _observation_roles, _surface_operations, prepare_surface_site,
)
from openghg_inversions.inversion_data._surface_support import required_flux_support

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
    flux_domain: str | None = None
    fp_model: str | None = None
    fp_species: str | None = None
    calibration_scale: str | None = None
    use_bc: bool | None = None
    bc_input: str | None = None
    flux_non_finite_check: FluxNonFiniteCheck | None = None


@dataclass
class MergedData:
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
            raise TypeError("MergedData accepts xarray datasets, not OpenGHG wrappers.")
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

        Args:
            species: Requested species, compared without case when recorded.
            domain: Requested transport domain, compared without case when recorded.
            start_date: Requested start as a date/time string. Naive values mean UTC.
            end_date: Requested end as a date/time string. Equivalent UTC instants
                agree even when their string spellings differ.
            split_by_sectors: Whether preparation expects separate flux sources.

        Returns:
            None. Compatible data and their borrowed arrays remain unchanged.

        Raises:
            ValueError: If a known fact or the source layout conflicts with the
                requested preparation, or a compared recorded/requested date is invalid.
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
    ) -> MergedData:
        """Replace retained site datasets with their selectors and provenance.

        Mapping order determines retained-site order. All sites must already
        belong to this record.
        The result owns new mappings and descriptive metadata; datasets and
        numerical arrays remain borrowed, including any backing open files.

        Raises:
            ValueError: If retained sites are invalid.
        """
        site_options = self.site_options.retain_sites(tuple(site_data), context=context)
        return MergedData(
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
        flux_store: str | None = "user",
        flux_domain: str | None = None,
        fp_model: str | None = None,
        fp_species: str | None = None,
        calibration_scale: str | None = None,
        use_bc: bool = True,
        bc_input: str | None = None,
        flux_non_finite_check: FluxNonFiniteCheck = "lazy",
        surface_transport: Mapping[str, TransportMode] | None = None,
                    ) -> MergedData:
        """Acquire fresh merged data using complete aligned selectors.

        Reads OpenGHG stores and retains all selector fields together when sites
        are unavailable. Returned datasets may be lazy and remain borrowed by
        later preparation. ``flux_sources`` names OpenGHG sources. Store,
        selector and merge errors propagate. ``flux_store`` defaults to ``"user"``;
        ``flux_domain`` selects the source flux dataset domain, defaulting to
        the footprint ``domain``. It does not imply a nested inner domain.
        ``surface_transport`` explicitly selects independent per-source surface
        CO2/O2 convolution, bypassing ModelScenario. Ordinary boundaries use
        OpenGHG baseline sensitivities with climatology disabled.
        Resolved transport requires time-resolved selectors and complete flux history
        and residual months. Mixed native source axes are rejected until native
        reporting can represent them; supplied-input preparation supports them.
        """
        if surface_transport is not None:
            acquired = acquire_surface_data(
                species=species, site_options=site_options, domain=domain,
                start_date=start_date, end_date=end_date, flux_sources=flux_sources,
                transport=surface_transport, split_by_sectors=split_by_sectors,
                bc_store=bc_store, obs_store=obs_store, footprint_store=footprint_store,
                flux_store=flux_store, flux_domain=flux_domain, fp_model=fp_model,
                fp_species=fp_species, calibration_scale=calibration_scale,
                use_bc=use_bc, bc_input=bc_input, flux_non_finite_check=flux_non_finite_check,
            )
            return cls(
                site_data=acquired.site_data, flux_data=acquired.flux_data,
                boundary_data=acquired.boundary_data, site_options=acquired.site_options,
                split_by_sectors=acquired.split_by_sectors, provenance=acquired.provenance,
                acquisition=acquired.acquisition,
            )
        merged_sites: dict[str, xr.Dataset] = {}
        observation_provenance = {}
        footprint_provenance = {}
        boundary_provenance = None

        # Get flux data
        if flux_sources is None:
            raise ValueError("`flux_sources` must be specified")

        flux_dict = get_flux_data(
            sources=list(flux_sources),
            species=species,
            domain=flux_domain or domain,
            start_date=start_date,
            end_date=end_date,
            store=flux_store,
            flux_non_finite_check=flux_non_finite_check,
        )
        retained_flux_dict = flux_dict
        flux_provenance = {source: selected_provenance(value, flux_store) for source, value in flux_dict.items()}

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

        for i, site_data, footprint_data in _get_site_inputs(
            species=species, site_options=site_options, domain=domain,
            start_date=start_date, end_date=end_date, obs_store=obs_store,
            footprint_store=footprint_store, fp_model=fp_model, fp_species=fp_species,
            calibration_scale=calibration_scale,
        ):
            site = site_options.sites[i]
            site_platform = site_options.platform[i]
            scenario_platform = (
                "site-column"
                if is_column_observation(site_options.inlet[i], site_platform) and not is_column_platform(site_platform)
                else site_platform
            )
            scenario_flux_dict = (
                interpolate_flux_to_footprint_grid(flux_dict, footprint_data)
                if flux_domain is not None and flux_domain.lower() != domain.lower()
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

        return cls(
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
                flux_domain=flux_domain,
                fp_model=fp_model,
                fp_species=fp_species,
                calibration_scale=calibration_scale,
                use_bc=use_bc,
                bc_input=bc_input,
                flux_non_finite_check=flux_non_finite_check,
            ),
        )



def _with_selected_provenance[Data: ObsData | FootprintData | FluxData | BoundaryConditionsData](
    value: Data, identity: InputProvenance
) -> Data:
    """Own selected wrapper metadata while borrowing its dataset and payloads."""
    result = copy(value)
    result.metadata = {**value.metadata, **asdict(identity)}
    return result


def _get_site_inputs(
    *, species: str, site_options: _site_options.SiteOptions, domain: str,
    start_date: str, end_date: str, obs_store: str, footprint_store: str,
    fp_model: str | None, fp_species: str | None, calibration_scale: str | None,
) -> Iterator[tuple[int, ObsData, FootprintData]]:
    """Retrieve retained observation/footprint pairs with their original selector indices."""
    keep_variables = [
        f"{species}",
        f"{species}_variability",
        f"{species}_repeatability",
        f"{species}_number_of_observations",
        f"{species}_error",
        "mf", "mf_error", "mf_repeatability", "mf_variability", "mf_number_of_observations",
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

        # Own selected identity metadata before any later provider operation.
        site_data = _with_selected_provenance(site_data, selected_provenance(site_data, obs_store))
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

        footprint_data = _with_selected_provenance(
            footprint_data, selected_provenance(footprint_data, footprint_store, requested_version="latest")
        )
        yield i, site_data, footprint_data



def acquire_surface_data(
    *,
    species: str,
    site_options: _site_options.SiteOptions,
    domain: str,
    start_date: str,
    end_date: str,
    flux_sources: Sequence[str] | None,
    transport: Mapping[str, TransportMode],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    flux_store: str | None = "user",
    flux_domain: str | None = None,
    fp_model: str | None = None,
    fp_species: str | None = None,
    calibration_scale: str | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> MergedData:
    """Acquire surface inputs using retained native releases to plan flux support.

    Shared selectors follow :meth:`MergedData.from_options`. ``transport`` must
    specify exactly one integrated or time-resolved mode per flux source, for
    continuous surface CO2/O2 mole fractions. Boundaries use ordinary contributions.

    Retrieve observation/footprint pairs first, then the union of backward
    histories and complete residual months. Missing sites are dropped together
    with their selectors. The returned :class:`MergedData` retains full retrieved
    native flux support and borrows datasets and lazy arrays; selected identities
    own wrapper metadata. Older inventories and climatological years are never substituted.

    Raises:
        ValueError: Unsupported selectors, incompatible native source axes,
            inadequate temporal support, or invalid surface/boundary inputs.
        SearchError: No retained sites or requested flux/boundary data are available.
    """
    if flux_sources is None:
        raise ValueError("`flux_sources` must be specified")
    if species.lower() not in {"co2", "o2"}:
        raise ValueError("Surface acquisition currently supports only CO2 and O2.")
    if flux_domain is not None and flux_domain.lower() != domain.lower():
        raise ValueError(
            "Fresh surface acquisition requires matching flux and footprint domains "
            "to retain native reporting axes; use prepare_surface_site for supplied inputs."
        )
    if not transport or set(transport) != set(flux_sources):
        raise ValueError("surface_transport must specify exactly one mode per flux source.")
    if any(mode not in {"integrated", "time_resolved"} for mode in transport.values()):
        raise ValueError("Surface transport modes must be 'integrated' or 'time_resolved'.")
    for inlet, platform in zip(site_options.inlet, site_options.platform, strict=True):
        if is_column_observation(inlet, platform) or (
            platform is not None and str(platform).lower() not in
            {"surface", "insitu", "in-situ", "continuous", "tower"}
        ):
            raise ValueError("Surface acquisition supports only continuous surface platforms.")
    if "time_resolved" in transport.values() and not all(site_options.time_resolved):
        raise ValueError("Time-resolved transport requires time_resolved=True site selectors.")

    retained_inputs = list(_get_site_inputs(
        species=species, site_options=site_options, domain=domain,
        start_date=start_date, end_date=end_date, obs_store=obs_store,
        footprint_store=footprint_store, fp_model=fp_model, fp_species=fp_species,
        calibration_scale=calibration_scale,
    ))
    if not retained_inputs:
        raise SearchError("No site data found. Exiting process.")
    align, _, _ = _surface_operations()
    support: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, observations, footprint in retained_inputs:
        observation_view = _dataset(observations)
        roles = _observation_roles(observation_view, species=species.lower(), observation_variable=None)
        observation_view = observation_view.rename({name: role for name, role in roles.items() if name != role})
        aligned_obs, _ = _align_footprint(observation_view, _dataset(footprint), align)
        native = _native_releases(_dataset(footprint), aligned_obs)
        for source, mode in transport.items():
            first, last = required_flux_support(native, mode=mode)
            if source in support:
                first = min(first, support[source][0])
                last = max(last, support[source][1])
            support[source] = first, last

    # Native reporting currently requires common source axes. Query their union
    # so mixed modes do not create artificial differences in retained time axes.
    flux_first = min(first for first, _ in support.values())
    flux_last = max(last for _, last in support.values())
    flux_dict: dict[str, FluxData] = {}
    flux_provenance: dict[str, InputProvenance] = {}
    for source in flux_sources:
        first, last = flux_first, flux_last
        # OpenGHG's date slicing subtracts one second from the exclusive end.
        query_end = last.ceil("s")
        retrieved = get_flux_data(
            sources=[source], species=species, domain=domain,
            start_date=str(first), end_date=str(query_end), store=flux_store,
            flux_non_finite_check=flux_non_finite_check,
            allow_older_flux=False, include_previous_interval=True,
        )[source]
        times = retrieved.data.indexes.get("time")
        # Some catalogs omit the source period. One bounded retry includes the
        # preceding label when the returned cadence reveals a shifted interval.
        if isinstance(times, pd.DatetimeIndex) and len(times) > 1 and times[0] > first:
            cadence = times[1] - times[0]
            if cadence > pd.Timedelta(0):
                retrieved = get_flux_data(
                    sources=[source], species=species, domain=domain,
                    start_date=str(first - cadence), end_date=str(query_end), store=flux_store,
                    flux_non_finite_check=flux_non_finite_check,
                    allow_older_flux=False, include_previous_interval=True,
                )[source]
        identity = selected_provenance(retrieved, flux_store)
        flux_provenance[source] = identity
        flux_dict[source] = _with_selected_provenance(retrieved, identity)
    if len(flux_dict) > 1:
        try:
            xr.align(*(value.data.flux for value in flux_dict.values()), join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                "Surface native reporting cannot yet represent mixed native source axes; "
                "use prepare_surface_site for source-wise supplied-input responses."
            ) from exc

    boundary_provenance = None
    bc_data = None
    if use_bc:
        try:
            retrieved_bc = get_bc(
                species=species, domain=domain, bc_input=bc_input,
                start_date=start_date, end_date=end_date, store=bc_store,
            )
        except SearchError as exc:
            raise SearchError("Could not find matching boundary conditions.") from exc
        boundary_provenance = selected_provenance(retrieved_bc, bc_store, requested_version="latest")
        bc_data = _with_selected_provenance(retrieved_bc, boundary_provenance)

    merged_sites: dict[str, xr.Dataset] = {}
    observation_provenance: dict[str, InputProvenance] = {}
    footprint_provenance: dict[str, InputProvenance] = {}
    output_units: str | None = None
    check_scales = set()
    for i, observations, footprint in retained_inputs:
        site = site_options.sites[i]
        try:
            prepared = prepare_surface_site(
                observations, footprint, flux_dict, species=species, transport=transport,
                boundary_data=bc_data, output_units=output_units,
            )
        except (TypeError, ValueError) as exc:
            if output_units is None:
                raise
            raise ValueError(
                f"Could not merge site {site!r} using target observation units {output_units!r}."
            ) from exc
        if output_units is None:
            scenario_units = prepared.mf.attrs.get("units")
            if not isinstance(scenario_units, str) or not scenario_units:
                raise ValueError(f"No observation units detected for the first retained site {site!r}.")
            mole_fraction_unit_scale(scenario_units, context=f"site {site!r} variable 'mf'")
            output_units = scenario_units
        prepared.attrs["openghg_inversions_time_resolved"] = str(site_options.time_resolved[i]).lower()
        merged_sites[site] = prepared
        observation_provenance[site] = selected_provenance(observations)
        footprint_provenance[site] = selected_provenance(footprint)
        check_scales.add(prepared.scale)
    if len(check_scales) > 1:
        logger.warning("Not all sites using the same calibration scale: %s scales found.", len(check_scales))
    retained = site_options.select_indices([i for i, _, _ in retained_inputs])
    return MergedData(
        site_data=merged_sites,
        flux_data={source: value.data for source, value in flux_dict.items()},
        boundary_data=bc_data.data if bc_data is not None else None,
        site_options=retained, split_by_sectors=split_by_sectors,
        provenance=MergedDataProvenance.from_retrieval(
            observations=observation_provenance, footprints=footprint_provenance,
            flux=flux_provenance, boundary=boundary_provenance,
        ),
        acquisition=AcquisitionFacts(
            species=species, domain=domain, start_date=start_date, end_date=end_date,
            flux_domain=flux_domain, fp_model=fp_model, fp_species=fp_species,
            calibration_scale=calibration_scale, use_bc=use_bc, bc_input=bc_input,
            flux_non_finite_check=flux_non_finite_check,
        ),
    )


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
