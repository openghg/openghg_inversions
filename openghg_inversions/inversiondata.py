from dataclasses import dataclass, field
from typing import cast, Optional, TypeVar, Union

from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, FootprintData, FluxData, ObsData
from openghg.retrieve import get_bc, get_footprint, get_flux, get_obs_surface
from openghg.types import SearchError


@dataclass
class SiteData:
    species: str
    site: str
    inlet: Optional[str] = None
    averaging_period: Optional[str] = None
    instrument: Optional[str] = None
    fp_height: Optional[str] = None

    def get_obs_data(
        self, start_date: Optional[str], end_date: Optional[str], stores: Optional[list[str]]
    ) -> ObsData:
        args = {
            "species": self.species,
            "site": self.site,
            "inlet": self.inlet,
            "average": self.averaging_period,
            "instrument": self.instrument,
            "start_date": start_date,
            "end_date": end_date,
        }

        if stores is None:
            obs_data = get_obs_surface(**args)
            if obs_data is None:
                # TODO: when is None returned, but a SearchError not raised?
                raise SearchError(f"No data found for {str(self)} from {start_date} to {end_date}.")
            return obs_data

        errors = []
        for store in stores:
            args["store"] = store

            try:
                obs_data = get_obs_surface(**args)
            except SearchError as e:
                errors.append(e)
            else:
                if obs_data is not None:
                    return obs_data

        # at this point, no data was found in any store, so re-raise errors
        raise SearchError(*errors)

    def get_footprint_data(
        self,
        domain: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model: Optional[str] = None,
        stores: Optional[list[str]] = None,
    ) -> FootprintData:
        args = {
            "species": self.species,
            "site": self.site,
            "inlet": self.fp_height,
            "start_date": start_date,
            "end_date": end_date,
            "domain": domain,
            "model": model,
        }

        if stores is None:
            footprint_data = get_footprint(**args)
            if footprint_data is None:
                # TODO: when is None returned, but a SearchError not raised?
                raise SearchError(f"No data found for {str(self)} from {start_date} to {end_date}.")
            return footprint_data

        errors = []
        for store in stores:
            args["store"] = store

            try:
                footprint_data = get_footprint(**args)
            except SearchError as e:
                errors.append(e)
            else:
                if footprint_data is not None:
                    return footprint_data

        # at this point, no data was found in any store, so re-raise errors
        raise SearchError(*errors)


ID = TypeVar("ID", bound="InversionData")


@dataclass
class InversionData:
    species: str  # TODO is this redundant since this info is part of SiteData?
    domain: str
    start_date: str
    end_date: str
    model: str
    sites: list[SiteData]
    fluxes: dict[str, FluxData]
    bc: Optional[BoundaryConditionsData] = None
    scenarios: list[ModelScenario] = field(default_factory=list)
    units: Optional[float] = None  # TODO: use a property to get this from sites?

    @classmethod
    def from_config(
        cls: type[ID],
        species: str,
        sites: list[str],
        domain: str,
        averaging_periods: list[str],
        start_date: str,
        end_date: str,
        sources: list[str],
        fp_heights: list[str],
        fp_model: str = "NAME",
        inlets: Optional[list[Union[str, None]]] = None,
        instruments: Optional[list[Union[str, None]]] = None,
        bc_input: Optional[str] = None,
        obs_stores: Optional[list[str]] = None,
        footprint_stores: Optional[list[str]] = None,
        flux_stores: Optional[list[str]] = None,
        bc_stores: Optional[list[str]] = None,
        make_scenarios: bool = True,
    ) -> ID:

        site_datas = []
        if inlets is None:
            inlets = cast(list[Union[str, None]], [None] * len(sites))
        if instruments is None:
            instruments = cast(list[Union[str, None]], [None] * len(sites))

        for site, inlet, instrument, averaging_period, fp_height in zip(
                    sites, inlets, instruments, averaging_periods, fp_heights
            ):
            site_datas.append(
                SiteData(
                    species=species,
                    site=site,
                    inlet=inlet,
                    averaging_period=averaging_period,
                    instrument=instrument,
                    fp_height=fp_height,

                )
            )

        flux_dict = {}
        for source in sources:
            try:
                # TODO need to use flux_store(s) here...
                flux_data_result = get_flux(
                    species=species, domain=domain, source=source, start_date=start_date, end_date=end_date
                )
            except SearchError as e:
                raise SearchError(f"Flux file with source '{source}' could not be retrieved.") from e
            else:
                flux_dict[source] = flux_data_result

        inv_data = cls(
            species=species, domain=domain, model=fp_model, start_date=start_date, end_date=end_date, sites=site_datas, fluxes=flux_dict
        )

        if make_scenarios:
            inv_data.make_scenarios(obs_stores=obs_stores, footprint_stores=footprint_stores)

        return inv_data

    def make_scenarios(
        self, obs_stores: Optional[list[str]] = None, footprint_stores: Optional[list[str]] = None
    ) -> None:
        for site in self.sites:
            obs_data = site.get_obs_data(
                start_date=self.start_date, end_date=self.end_date, stores=obs_stores
            )
            footprint_data = site.get_footprint_data(
                domain=self.domain,
                start_date=self.start_date,
                end_date=self.end_date,
                model=self.model,
                stores=footprint_stores,
            )

            # TODO: need to rescale BC with units from obs?
            scenario = ModelScenario(obs=obs_data, footprint=footprint_data, bc=self.bc, flux=self.fluxes)

            # TODO: align, etc?
            self.scenarios.append(scenario)
