"""Retrieve RHIME observations, transport, flux and boundary data.

Acquisition owns store access and consumes the complete site-aligned
selection record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

import pandas as pd
import xarray as xr

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data import _site_options
from openghg_inversions.inversion_data._provenance import (
    InputProvenance, MergedDataProvenance, selected_provenance,
)


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
    def from_legacy_fp_all(
        cls, fp_all: dict, site_options: _site_options.SiteOptions, *,
        acquisition: AcquisitionFacts | None = None,
    ) -> RhimeMergedData:
        """Adapt the remaining legacy scientific producers (#821; remove in 0.9)."""
        fluxes = fp_all.get(".flux", {})
        boundary = fp_all.get(".bc")
        provenance = MergedDataProvenance(
            observations={site: InputProvenance() for site in site_options.sites},
            footprints={site: InputProvenance() for site in site_options.sites},
            flux={source: selected_provenance(value) for source, value in fluxes.items()},
            boundary=selected_provenance(boundary) if boundary is not None else None,
        )
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
            acquisition=acquisition or AcquisitionFacts(),
        )

    def to_legacy_fp_all(self) -> dict:
        """Adapt to remaining wrapper-based scientific consumers (#821; remove in 0.9).

        Only this explicit adapter reconstructs OpenGHG wrappers. Flux and boundary
        dataset containers are shallow copies; site datasets remain borrowed.
        Consumers must copy site containers before assigning variables or attrs.
        Numerical arrays remain shared. Wrapper metadata carries selected source
        identities, not arbitrary OpenGHG catalogue fields.
        """
        from openghg.dataobjects import BoundaryConditionsData, FluxData

        result = dict(self.site_data)
        result[".flux"] = {
            source: FluxData(
                data=data.copy(deep=False),
                metadata={"data_type": "flux", **asdict(self.provenance.flux[source])},
            )
            for source, data in self.flux_data.items()
        }
        if self.boundary_data is not None:
            result[".bc"] = BoundaryConditionsData(
                data=self.boundary_data.copy(deep=False),
                metadata=asdict(self.provenance.boundary),
            )
        result[".split_by_sectors"] = self.split_by_sectors
        return result

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
        from .get_data import _retrieve_inversion_data_from_options

        result = _retrieve_inversion_data_from_options(
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
        return result
