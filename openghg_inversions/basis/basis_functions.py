"""BasisFunctions object to encapsulate representation of basis.

Example usage:

>> def apply_basis_functions(ds: xr.Dataset, bf: BasisFunctions) -> xr.Dataset:
>>     if "fp_x_flux" not in ds:
>>         return ds
>>     return bf.sensitivities(ds.fp_x_flux).rename("H").to_dataset()

"""

from pathlib import Path
from typing import Self, cast
import warnings

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import force_align, concat_gather_data_arrays, get_xr_dummies
from openghg_inversions.config.paths import Paths
from openghg_inversions.utils import read_netcdfs

openghginv_path = Paths.openghginv


class BasisFunctions:
    """Basis functions for flux sensitivities.

    This class provides methods for using aggregation or "bucket" basis functions.
    This means that the basis functions can be represented as a 2D lat/lon array, with
    integer labels for each basis region.

    We do not allow time varying basis functions with this class.
    TODO: revisit this?

    Though basis functions are usually applied to the combined footprint x flux array,
    to use the basis functions in post-processing, we also need the flux used in the inversion.

    If the flux varies with time, then we need to align this with other time coordinates.
    Currently, this is not supported.
    TODO: add support for time varying flux?

    TODO: construct from matrix with "regions" already by default, and add "from_flat" as class method?
    Or make the base class do that, and make a subclass for bucket basis functions?
    """

    def __init__(
        self,
        basis_flat: xr.DataArray,
        flux: xr.DataArray,
        region_dim: str = "region",
        chunks: dict | None = None,
    ):
        self.basis_flat = basis_flat.isel(time=0, drop=True) if "time" in basis_flat.dims else basis_flat
        self.flux = flux.reindex_like(self.basis_flat, method="nearest")  # align lat/lon

        self.region_dim = region_dim
        self.basis_matrix = get_xr_dummies(basis_flat, cat_dim=self.region_dim)

        if chunks is not None:
            self.basis_matrix = self.basis_matrix.chunk(**chunks)
        else:
            self.basis_matrix = self.basis_matrix.chunk()

        self.labels = np.unique(basis_flat)
        self.labels_shuffled = np.unique(basis_flat)
        np.random.shuffle(self.labels_shuffled)  # TODO make method so this can be re-shuffled

        self.interpolation_matrix = self.basis_matrix * self.flux

        # Currently no support for flux times
        if self.interpolation_matrix.sizes.get("time", 0) > 1:
            warnings.warn("Dropping time from interpolation matrix.")
            self.interpolation_matrix = self.interpolation_matrix.isel(time=0, drop=True)
        elif "time" in self.interpolation_matrix.dims:
            self.interpolation_matrix = self.interpolation_matrix.squeeze("time", drop=True)

    # TODO: make generic over DataArray and Dataset
    # TODO: add alignment option
    def interpolate(self, data: xr.DataArray, flux: bool = False) -> xr.DataArray:
        """Map from regions to lat/lon."""
        if self.region_dim not in data.dims:
            raise ValueError(
                f"Region dim {self.region_dim} missing (data dims {data.dims}); cannot interpolate."
            )
        if flux:
            return xr.dot(self.interpolation_matrix, data, dim=self.region_dim)
        return xr.dot(self.basis_matrix, data, dim=self.region_dim)

    def sensitivity(self, fp_x_flux: xr.DataArray) -> xr.DataArray:
        """Create sensitivity ("H") matrix from footprint x flux array."""
        # TODO: check this still works if we have a stacked source/region dimension in the basis function
        interp = self.basis_matrix.reindex_like(fp_x_flux, method="nearest")  # should align lat/lon
        interp = interp.transpose("lat", "lon", ...)
        return (
            xr.dot(fp_x_flux, interp, dim=["lat", "lon"]).as_numpy().transpose(self.region_dim, "time", ...)
        )

    def save(self, path: str | Path) -> None:
        to_save = xr.Dataset({"basis": self.basis_flat, "flux": self.flux})
        to_save.to_netcdf(path, mode="w")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        ds = xr.open_dataset(path)
        return cls(basis_flat=ds.basis, flux=ds.flux)

    def save_acrg(
        self,
        basis_algorithm: str,
        output_dir: str,
        domain: str,
        species: str,
        output_name: str | None = None,
    ) -> None:
        """Save basis functions to netCDF.

        Args:
          basis_algorithm (str):
            name of basis algorithm (e.g. "quadtree" or "weighted")
          output_dir (str):
            root directory to save basis functions
          domain (str):
            domain of inversion; basis is saved in a "domain" directory inside `output_dir`
          species (str):
            species of inversion
          output_name (str,optional):
            File output name
            Default None

        Returns:
            None. Saves basis dataset to netCDF.
        """
        basis_out_path = Path(output_dir, domain.upper())

        if not basis_out_path.exists():
            basis_out_path.mkdir(parents=True)

        start_date = str(self.basis_flat.time.min().values)[:7]  # year and month

        if output_name is None:
            output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}.nc"
        else:
            output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}.nc"

        self.save(basis_out_path / output_name)

    @classmethod
    def load_acrg(
        cls,
        domain: str,
        basis_case: str,
        basis_directory: str | None = None,
        flux: xr.DataArray | None = None,
    ) -> Self:
        """Read in basis function(s) from file given basis case and domain, and return as an
        xarray Dataset.

        The basis function files should be stored as on paths of the form:
            <basis_directory>/<domain>/<basis_case>_<domain>*.nc

        For instance: domain = EUROPE, basis_directory = /group/chem/acrg/LPDM/basis_functions,
        and basis_case = sub_transd would find files such as:

            /group/chem/acrg/LPDM/basis_functions/EUROPE/sub_transd_EUROPE_2014.nc

        Basis functions created by algorithms in OpenGHG inversions will be stored using
        this path format.

        Args:
            domain: domain name. The basis files should be sub-categorised by the domain.
            basis_case: basis case to read in. Examples of basis cases are "voronoi", "sub-transd",
                "sub-country_mask", "INTEM".
            basis_directory: basis_directory can be specified if files are not in the default
                directory (i.e. `openghg_inversions/basis_functions`). Must point to a directory that
                contains subfolders organized by domain.
            flux: needs to be provided if not found in stored files.

        Returns:
            BasisFunction object corresponding to combined matching basis functions
        """
        if basis_directory is None:
            basis_path = openghginv_path / "basis_functions"
            if not basis_path.exists():
                basis_path.mkdir()
                raise ValueError(
                    f"Default basis directory {basis_path} was empty. "
                    "Add basis files or specify `basis_path`."
                )
        else:
            basis_path = Path(basis_directory)

        file_path = (basis_path / domain).glob(f"{basis_case}_{domain}*.nc")
        files = sorted(list(file_path))

        if len(files) == 0:
            raise FileNotFoundError(
                f"Can't find basis function files for domain '{domain}'and basis_case '{basis_case}' "
            )

        basis_ds = read_netcdfs(files)

        if "flux" not in basis_ds and flux is None:
            raise ValueError("Flux not stored with basis functions; please provide flux.")

        flux = cast(xr.DataArray, basis_ds.get("flux") or flux)

        return cls(basis_ds.basis, flux=flux)

    # @classmethod
    # def from_algorithm(cls) -> Self:
    #     """Compute basis from algorithm."""
    #     pass

    def plot(self, shuffle=False, **kwargs) -> None:
        """Plot basis.

        Shuffle labels to make regions easier to see.
        """
        if not shuffle:
            self.basis_flat.plot(**kwargs)
        else:
            bf_shuf = self.basis_flat.copy()
            bf_shuf.values = self.labels_shuffled[self.basis_flat.values.astype(int) - 1]
            bf_shuf.plot(**kwargs)


class MultiSectorBasisFunctions(BasisFunctions):
    def __init__(
        self,
        basis_flat: dict[str, xr.DataArray],
        flux: dict[str, xr.DataArray],
        region_dim: str = "region",
        chunks: dict | None = None,
    ):
        assert all(k in flux for k in basis_flat), "basis flat and flux must have same keys"

        self.basis_flat = {
            k: v.isel(time=0, drop=True) if "time" in v.dims else v for k, v in basis_flat.items()
        }

        flux_aligned = {
            k: v.reindex_like(self.basis_flat[k], method="nearest") for k, v in flux.items()
        }  # align lat/lon
        flux_to_concat = [v.expand_dims({"source": [k]}) for k, v in flux_aligned.items()]
        self.flux = xr.concat(flux_to_concat, dim="source")

        self.region_dim = region_dim
        self.sectoral_basis_matrices = {
            k: get_xr_dummies(v, cat_dim="sector_region") for k, v in self.basis_flat.items()
        }
        self.basis_matrix = concat_gather_data_arrays(
            self.sectoral_basis_matrices,
            key_dim="source",
            ragged_dim="sector_region",
            stack_dim=self.region_dim,
        )

        if chunks is not None:
            self.basis_matrix = self.basis_matrix.chunk(**chunks)
        else:
            self.basis_matrix = self.basis_matrix.chunk()

        # TODO: decide what to do about labels and plotting
        # self.labels = np.unique(basis_flat)
        # self.labels_shuffled = np.unique(basis_flat)
        # np.random.shuffle(self.labels_shuffled)  # TODO make method so this can be re-shuffled

        # need to explicitly broadcast flux source to match "source" level
        # of region multi-index
        self.interpolation_matrix = self.basis_matrix * self._align_source(self.flux)

        # Currently no support for flux times
        if self.interpolation_matrix.sizes.get("time", 0) > 1:
            warnings.warn("Dropping time from interpolation matrix.")
            self.interpolation_matrix = self.interpolation_matrix.isel(time=0, drop=True)
        elif "time" in self.interpolation_matrix.dims:
            self.interpolation_matrix = self.interpolation_matrix.squeeze("time", drop=True)

    def _align_source(self, other: xr.DataArray) -> xr.DataArray:
        if "source" not in other.dims:
            # nothing to align
            return other
        region_index = self.basis_matrix[self.region_dim]
        other_on_region = other.sel(source=region_index.source)

        # Force region coordinate to be identical
        other_on_region = other_on_region.assign_coords({self.region_dim: region_index})

        # Drop source coordinate to prevent alignment conflict
        other_on_region = other_on_region.drop_vars("source")

        return other_on_region

    def sensitivity(self, fp_x_flux: xr.DataArray) -> xr.DataArray:
        """Create sensitivity ("H") matrix from footprint x flux array."""
        # TODO: check this still works if we have a stacked source/region dimension in the basis function
        interp = force_align(self.basis_matrix, fp_x_flux, dims=["lat", "lon"])
        interp = interp.transpose("lat", "lon", ...)
        return xr.dot(self._align_source(fp_x_flux), interp, dim=["lat", "lon"]).as_numpy()

    def plot(self, shuffle=False, **kwargs) -> None:
        raise NotImplementedError()
