"""
Module with code related to country maps.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, cast, Literal, Optional, TypeVar, Union

import xarray as xr
from openghg_inversions import convert, utils
from xarray.core.common import DataWithCoords

from openghg_inversions.array_ops import align_sparse_lat_lon, get_xr_dummies, sparse_xr_dot
from .inversion_output import InversionOutput

# type for xr.Dataset *or* xr.DataArray
DataSetOrArray = TypeVar("DataSetOrArray", bound=DataWithCoords)


def get_area_grid(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Return xr.DataArray with coordinate dimensions ("lat", "lon") containing
    the area of each grid cell centered on the coordinates.

    Args:
        lat: latitude values
        lon: longitude values

    Returns:
        xr.DataArray with grid cell areas.
    """
    ag_vals = utils.areagrid(lat.values, lon.values)
    return xr.DataArray(ag_vals, coords=[lat, lon], dims=["lat", "lon"], name="area_grid")


@functools.lru_cache
def get_iso3166_codes() -> dict[str, Any]:
    """Load dictionary mapping alpha-2 country codes to other country information."""
    postprocessing_path = Path(__file__).parent
    with open(postprocessing_path / "iso3166.json", "r", encoding="utf8") as f:
        iso3166 = json.load(f)
    return iso3166


def get_country_code(
    x: str, iso3166: Optional[dict[str, dict[str, Any]]] = None, code: Literal["alpha2", "alpha3"] = "alpha3"
) -> str:
    """Get alpha-2 or alpha-3 (default) country code given the name of a country."""
    if iso3166 is None:
        iso3166 = get_iso3166_codes()

    # first try to match long names, ignoring "The " at the beginning of a name
    for v in iso3166.values():  # type: ignore
        if x.lower().lstrip("the ") == v["iso_long_name"].lower().lstrip("the "):
            return v[code]

    # next try to match unofficial names
    for v in iso3166.values():  # type: ignore
        if any(x.lower() == name.lower() for name in v["unofficial_names"]):
            return v[code]

    # next try to match substrings...
    for v in iso3166.values():
        names = [v["iso_long_name"].lower()] + [name.lower() for name in v["unofficial_names"]]
        if any(x.lower() in name for name in names):
            return v[code]

    # if no matches are found, return x
    return x


class Countries:
    """Class to load country files (and list of countries to use from that file), and provide methods
    to create country traces bases on these country files.

    Multiple Country objects can be merged together to use multiple country files.
    """

    def __init__(
        self,
        countries: xr.Dataset,
        country_selections: Optional[list[str]] = None,
        country_code: Literal["alpha2", "alpha3"] | None = None,
    ) -> None:
        """Create Countries object given country map Dataset and optional list of countries to select.

        Args:
            countries: country map Dataset with `country` and `name` data variables.
            country_selections: optional list of country names to select.
        """
        if country_code is None:
            country_labels = countries.name.values
        else:
            # apply `get_country_code` to each element of `country` coordinate
            country_labels = list(
                map(functools.partial(get_country_code, code=country_code), map(str, countries.name.values))
            )

        self.matrix = get_xr_dummies(countries.country, cat_dim="country", categories=country_labels)
        self.area_grid = get_area_grid(countries.lat, countries.lon)

        if country_selections is not None:
            # check that selected countries are in the `name` variable of `countries` Dataset
            selections_check = []
            all_countries = list(map(lambda x: str(x).lower(), countries.name.values))
            for selection in country_selections:
                if selection.lower() not in all_countries:
                    selections_check.append(selection)
            if selections_check:
                raise ValueError(
                    "Selected country/countries are not in `name` variable of "
                    f"`countries` Dataset: {selections_check}"
                )

            # only keep selected countries in country matrix
            filt = self.matrix.country.isin(country_selections)
            self.matrix = self.matrix.where(filt, drop=True)
            self.country_selections = country_selections
        else:
            self.country_selections = list(self.matrix.country.values)

    def get_x_to_country_mat(
        self,
        inv_out: InversionOutput,
        sparse: bool = False,
    ) -> xr.DataArray:
        """Construct a sparse matrix mapping from x sensitivities to country totals.

        Args:
            countries: xr.Dataset from country file. Must have variables: "country" with coordinate
                dimensions ("lat", "lon"), and "name" (with no coordinate dimensions).
            hbmcmc_outs: xr.Dataset from `hbmcmc_postprocessouts`.
            flux: flux used in inversion. If a constant flux was used, then `aprioriflux` from
                `hbmcmc_outs` can be used.
            area_grid: areas of each grid cell in inversion domain.
            basis_functions: xr.DataArray with coordinate dimensions ("lat", "lon") whose values assign
                grid cells to basis function boxes.
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
        # result = sparse_xr_dot(self.matrix, self.area_grid * flux_x_basis)
        if sparse:
            return result

        # hack since `.to_numpy()` doesn't work correctly with sparse arrays
        return xr.apply_ufunc(lambda x: x.todense(), result)

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

    @staticmethod
    def _country_region_traces(country_traces: xr.Dataset, country_regions: dict[str, list[str]] | Path) -> xr.Dataset:
        if isinstance(country_regions, Path):
            with open(country_regions, "r", encoding="utf8") as f:
                _country_regions = json.load(f)
            if not isinstance(_country_regions, dict) or any(not isinstance(v, list) for v in _country_regions.values()):
                raise ValueError(f"Country regions from file {country_regions} is not in the correct format."
                                 " It must be a dictionary mapping regions to the list of countries forming that region.")
            country_regions = _country_regions

        region_traces = []

        for region, countries in country_regions.items():
            try:
                region_ds = (
                    country_traces.sel(country=countries).sum("country").expand_dims({"country": [region]})
                )
            except KeyError as e:
                print(f"Country region {region} was not added due to key error {e}.")
            else:
                region_traces.append(region_ds)

        if not region_traces:
            return xr.Dataset()

        return xr.concat(region_traces, dim="country")

    def get_country_trace(self, inv_out: InversionOutput, country_regions: dict[str, list[str]] | Path | None = None,
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
        x_trace = inv_out.get_trace_dataset(unstack_nmeasure=False, var_names="x")

        species = inv_out.species

        country_traces = Countries._get_country_trace(species, x_trace, x_to_country_mat)

        rename_dict = {dv: "country_" + str(dv).split("_")[1] for dv in country_traces.data_vars}
        country_traces = country_traces.rename_vars(rename_dict)

        if country_regions is not None:
            region_traces = self._country_region_traces(country_traces, country_regions)
            country_traces = xr.merge([country_traces, region_traces])

        for dv in country_traces.data_vars:
            suffix = str(dv).removeprefix("country_")
            country_traces[dv].attrs["units"] = "g/yr"
            country_traces[dv].attrs["long_name"] = f"{suffix}_country_flux_total"

        return country_traces

    def merge(self, other: Union[Countries, list[Countries]]) -> None:
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
