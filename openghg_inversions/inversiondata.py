from dataclasses import dataclass
from typing import Optional

from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, FootprintData, FluxData, ObsData
from openghg.retrieve import get_bc, get_footprint, get_flux, get_obs_surface
from openghg.types import SearchError


@dataclass
class SiteData:
    species: str
    site: str
    inlet: Optional[str]
    averaging_period: Optional[str]
    instrument: Optional[str]
    fp_height: Optional[str]

    def get_obs_data(
        self, start_date: Optional[str], end_date: Optional[str], stores: Optional[list[Optional[str]]]
    ) -> ObsData:
        if stores is None:
            stores = [None]

        args = {
            "species": self.species,
            "site": self.site,
            "inlet": self.inlet,
            "average": self.averaging_period,
            "instrument": self.instrument,
            "start_date": start_date,
            "end_date": end_date,
        }

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
        model: Optional[str],
        met_model: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        stores: Optional[list[Optional[str]]],
    ) -> FootprintData:
        if stores is None:
            stores = [None]

        args = {
            "species": self.species,
            "site": self.site,
            "inlet": self.fp_height,
            "start_date": start_date,
            "end_date": end_date,
            "domain": domain,
            "model": model,
            "met_model": met_model,
        }

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


@dataclass
class InversionData:
    species: str
    domain: str
    sites: list[SiteData]
    units: float
    bc: BoundaryConditionsData
    fluxes: dict[str, FluxData]
    scenarios: list[ModelScenario]


    @classmethod
    def from_config(cls):
        pass
