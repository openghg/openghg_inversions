"""Module with code related to country maps."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import cast, Literal, TypeVar
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import Self

import xarray as xr
from openghg_inversions import convert, utils

from openghg_inversions.array_ops import align_sparse_lat_lon, get_xr_dummies, sparse_xr_dot
from openghg_inversions.utils import get_country_file_path
from ._country_codes import CountryInfoList
from .inversion_output import InversionOutput

# type for xr.Dataset *or* xr.DataArray
DataSetOrArray = TypeVar("DataSetOrArray", xr.DataArray, xr.Dataset)


def get_area_grid(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Return array containing area of each grid cell centered on given coordinates.

    This creates an xr.DataArray with coordinate dimensions ("lat", "lon") containing
    the area of each grid cell centered on the coordinates.

    Args:
        lat: latitude values
        lon: longitude values

    Returns:
        xr.DataArray with grid cell areas.
    """
    ag_vals = utils.areagrid(lat.values, lon.values)
    return xr.DataArray(ag_vals, coords=[lat, lon], dims=["lat", "lon"], name="area_grid")


paris_regions_dict = {
    "europe":{
        "BELUX": ["BEL", "LUX"],
        "BENELUX": ["BEL", "LUX", "NLD"],
        "CW_EU": [
            "AUT",
            "BEL",
            "CHE",
            "CZE",
            "DEU",
            "ESP",
            "FRA",
            "GBR",
            "HRV",
            "HUN",
            "IRL",
            "ITA",
            "LUX",
            "NLD",
            "POL",
            "PRT",
            "SVK",
            "SVN",
        ],
        "EU_GRP2": ["AUT", "BEL", "CHE", "DEU", "DNK", "FRA", "GBR", "IRL", "ITA", "LUX", "NLD"],
        "NW_EU": ["BEL", "DEU", "DNK", "FRA", "GBR", "IRL", "LUX", "NLD"],
        "NW_EU2": ["BEL", "DEU", "FRA", "GBR", "IRL", "LUX", "NLD"],
        "NW_EU_CONTINENT": ["BEL", "DEU", "FRA", "LUX", "NLD"],
    },
    "eastasia":{
        "EASTERN_ASIA":["EChi1", "S.Kor", "N.Kor", "Japan"]
    }
    }


class CountryRegions:
    """Regions defined by combining several countries."""

    def __init__(self, country_regions: Mapping[str, Sequence[str]] | None = None) -> None:
        """Create CountryRegions object from dictionary defining regions.

        Args:
            country_regions: mapping from region name to sequence of countries
            that comprise the region.

        Raises:
            ValueError: if the `country_regions` input isn't in the right
            format.

        """
        country_regions = country_regions or {}  # default to empty list if None is passed

        # validate input
        if not isinstance(country_regions, dict) or any(
            not isinstance(v, Sequence) for v in country_regions.values()
        ):
            raise ValueError(
                "Country regions are not in the correct format; they must be a dictionary mapping"
                "region names to the list of countries forming that region."
            )

        self._regions = country_regions

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Load country regions from JSON file.

        Args:
            filepath: path to JSON file containing country region definitions.

        Returns:
            new CountryRegions object representing the regions on the given
            file.

        """
        with open(filepath, encoding="utf8") as f:
            country_regions = json.load(f)

        return cls(country_regions)

    def __bool__(self) -> bool:
        """Return True if region definitions are not empty."""
        return bool(self._regions)

    def to_dict(self, country_code: Literal["alpha2", "alpha3"] | None = None) -> dict[str, CountryInfoList]:
        """Return dict mapping region name to CountryInfoList of countries comprising the region.

        This method is used to apply country region definitions to be applied in cases where
        the country names from the country file aren't in the same format as the country names/codes
        in the country region definitions.

        Args:
            country_code: country code to set for countries comprising a region
            in the output.

        Returns:
            dict mapping country regions to CountryInfoLists of the countries
            comprising each region.

        """
        return {
            region: CountryInfoList(region_countries, country_code=country_code)
            for region, region_countries in self._regions.items()
        }

    def region_countries_missing_from(
        self, country_list: CountryInfoList | Iterable[str]
    ) -> dict[str, CountryInfoList]:
        """Report countries from region definitions that are missing from the given country list.

        Args:
            country_list: List of countries to compare with. This is usually the
            list of country names from a country file.

        Returns:
            dict mapping regions to a list of missing countries.

        """
        country_list = CountryInfoList(country_list)

        missing = defaultdict(CountryInfoList)

        for region, region_countries in self.to_dict().items():
            for rc in region_countries:
                if rc not in country_list:
                    missing[region].append(rc)

        return missing

    def align(self, country_list: CountryInfoList) -> Self:
        """Return CountryRegions with values aligned to a given list of countries.

        This is used to make sure that the region definitions have the same "input names"
        as the given country list, which is necessary if country codes are not being used.
        """
        aligned_country_regions = {
            region: country_list.select_by_country_info(region_countries)
            for region, region_countries in self._regions.items()
        }
        return type(self)(aligned_country_regions)

    def all_region_countries_present_in(self, country_list: CountryInfoList) -> bool:
        """Return True if all countries needed to define regions are in the given country list."""
        return not self.region_countries_missing_from(country_list)

    @property
    def region_names(self) -> list[str]:
        """List of region names stored."""
        return list(self._regions.keys())


class Countries:
    """Class to load country files and create country traces.

    A list of specifying a subset of the countries in the country file can be provided.
    Multiple Country objects can be merged together to use multiple country files.
    """

    def __init__(
        self,
        countries: xr.Dataset,
        country_selections: list[str] | None = None,
        country_code: Literal["alpha2", "alpha3"] | None = None,
        country_regions: dict[str, list[str]] | str | Path | None = None,
        domain: str | None = None
    ) -> None:
        """Create Countries object given country map Dataset and optional list of countries to select.

        Args:
            countries: country map Dataset with `country` and `name` data variables.
            country_selections: optional list of country names to select.
            country_code: if not None, convert country names to specified codes. These names or codes
              will be used in the `country` coordinate.
            country_regions: dict mapping country region names (e.g. "BENELUX") to a
              list of (country codes) of the countries comprising that regions (e.g.
              `["BEL", "NLD", "LUX"]`). Alternatively, a path (or string representing a path)
              to a JSON file with a similar specification can be passed.

        """
        self.country_code = country_code
        self.country_labels = CountryInfoList(countries.name.values, country_code=country_code)
        self.domain = domain

        # get country regions
        if isinstance(country_regions, str | Path):
            self.country_regions = CountryRegions.from_file(country_regions)
        else:
            self.country_regions = CountryRegions(country_regions)

        self.country_regions = self.country_regions.align(self.country_labels)

        # check that country regions are specified in correct country code
        missing_countries = self.country_regions.region_countries_missing_from(self.country_labels)
        if missing_countries:
            msg = "\n".join(
                f"{region}: {list(countries)}" for region, countries in missing_countries.items() if countries
            )
            raise ValueError(f"Could not find the following countries needed for regions:\n{msg}")

        # create matrix with dimensions: lat, lon, country
        self.matrix = get_xr_dummies(countries.country, cat_dim="country", categories=self.country_labels)

        # add regions to matrix
        if self.country_regions:
            try:
                region_vectors = [
                    self.matrix.sel(country=region_countries).sum("country").expand_dims(country=[region])
                    for region, region_countries in self.country_regions.to_dict(
                        country_code=self.country_code
                    ).items()
                ]
            except KeyError:
                raise ValueError(
                    "Country region definitions not consistent with country file names. Try setting `country_code`."
                )
            else:
                region_matrix = xr.concat(
                    region_vectors,
                    dim="country",
                )
                self.matrix = xr.concat([self.matrix, region_matrix], dim="country")

        self.area_grid = get_area_grid(countries.lat, countries.lon)

        # restrict matrix to selected countries
        if country_selections is not None:
            # check that selected countries are in the input country file
            selections_check = []

            for selection in country_selections:
                if selection not in self.country_labels:
                    selections_check.append(selection)
            if selections_check:
                raise ValueError(
                    "Selected country/countries are not in `name` variable of "
                    f"`countries` Dataset: {selections_check}"
                )

            # add regions to selection
            self.country_selections = CountryInfoList(country_selections, country_code=self.country_code)
            self.country_selections.extend(self.country_regions.region_names)

            # only keep selected countries in country matrix
            filt = self.matrix.country.isin(country_selections)
            self.matrix = self.matrix.where(filt, drop=True)
        else:
            self.country_selections = self.country_labels + self.country_regions.region_names

    @classmethod
    def from_file(
        cls,
        country_file: str | Path | None = None,
        domain: str | None = None,
        country_selections: list[str] | None = None,
        country_code: Literal["alpha2", "alpha3"] | None = None,
        country_regions: dict[str, list[str]] | str | Path | None = None,
    ) -> Self:
        """Create Countries object given country map Dataset and optional list of countries to select.

        Args:
            country_file: path to country file
            domain: used to select country file from `openghg_inversions/countries` if `country_file` is None.
            country_selections: optional list of country names to select.
            country_code: if not None, convert country names to specified codes. These names or codes
              will be used in the `country` coordinate.
            country_regions: dict mapping country region names (e.g. "BENELUX") to a
              list of (country codes) of the countries comprising that regions (e.g.
              `["BEL", "NLD", "LUX"]`). Alternatively, a path (or string representing a path)
              to a JSON file with a similar specification can be passed.

        """
        country_file_path = get_country_file_path(country_file=country_file, domain=domain)
        return cls(
            xr.open_dataset(country_file_path),
            country_code=country_code,
            country_selections=country_selections,
            country_regions=country_regions,
        )

    def get_x_to_country_mat(
        self,
        inv_out: InversionOutput,
        sparse: bool = False,
    ) -> xr.DataArray:
        """Construct a sparse matrix mapping from x sensitivities to country totals.

        Args:
            inv_out: InversionOutput object, used to get basis functions and flux.
            sparse: if True, values of returned DataArray are `sparse.COO` array.

        Returns:
            xr.DataArray with coordinate dimensions ("country", "basis_region")
        """
        # multiply flux and basis and align to country lat/lon
        basis = align_sparse_lat_lon(inv_out.basis, inv_out.flux)
        flux_x_basis = align_sparse_lat_lon(inv_out.flux * basis, self.area_grid)

        # compute matrix/tensor product: country_mat.T @ (area_grid * flux * basis_mat)
        # transpose doesn't need to be taken explicitly because alignment is done by dimension name
        result = self.matrix @ (self.area_grid * flux_x_basis)

        if sparse:
            return result

        return result.as_numpy()

    @staticmethod
    def _get_country_trace(
        species: str,
        x_trace: DataSetOrArray,
        x_to_country: xr.DataArray,
    ) -> DataSetOrArray:
        """Calculate trace(s) for total country emissions.

        The totals are in grams/year.

        Args:
            species: name of species, e.g. "co2", "ch4", "sf6", etc.
            x_trace: xr.DataArray or xr.Dataset with coordinate dimensions ("draw", "nx").
                Note: "nx" be replaced with another name, as long as the same coordinate name
                was used in `get_x_to_country_matrix`.
            x_to_country: xr.DataArray with result from `get_x_to_country_mat`

        Returns:
            xr.DataArray with coordinate dimensions ("country", "draw")

        TODO: there is a "country unit" conversion in the old code, but it seems to always produce
              1.0, based on how it is used in hbmcmc
        """
        raw_trace = sparse_xr_dot(x_to_country, x_trace)
        molar_mass = convert.molar_mass(species)
        return raw_trace * 365 * 24 * 3600 * molar_mass  # type: ignore

    def get_country_trace(
        self,
        inv_out: InversionOutput,
    ) -> xr.Dataset:
        """Calculate trace(s) for total country emissions.

        Args:
            species: name of species, e.g. "co2", "ch4", "sf6", etc.
            inv_out: InversionOutput

        Returns:
            xr.Dataset with coordinate dimensions ("country", "draw")

        TODO: there is a "country unit" conversion in the old code, but it seems to always product
              1.0, based on how it is used in hbmcmc
        """
        x_to_country_mat = self.get_x_to_country_mat(inv_out)
        x_trace = inv_out.get_trace_dataset(var_names="x")

        species = inv_out.species

        country_traces = Countries._get_country_trace(species, x_trace, x_to_country_mat)

        rename_dict = {dv: "country_" + str(dv).split("_")[1] for dv in country_traces.data_vars}
        country_traces = country_traces.rename_vars(rename_dict)

        for dv in country_traces.data_vars:
            suffix = str(dv).removeprefix("country_")
            country_traces[dv].attrs["units"] = "g/yr"
            country_traces[dv].attrs["long_name"] = f"{suffix}_country_flux_total"

        return country_traces

    def merge(self, other: Countries | list[Countries]) -> None:
        """Merge in another Countries object (in-place).

        Args:
            other: (list of) Countries object(s) to merge into this Country object.

        Returns:
            None (updates in-place)
        """
        if not isinstance(other, list):
            other = [other]

        other = cast(list[Countries], other)

        all_country_selections = self.country_selections.copy()
        for countries in other:
            all_country_selections.extend(countries.country_selections)

        if len(set(all_country_selections)) < len(all_country_selections):
            raise ValueError("Duplicate countries selected. Make sure `country_selections` are disjoint.")

        self.country_selections = all_country_selections
        self.matrix = xr.concat([self.matrix] + [countries.matrix for countries in other], dim="country")
