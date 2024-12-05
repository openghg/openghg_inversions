from functools import partial
import logging
from typing import Any, Callable, ChainMap, Iterable, cast, Literal, Mapping, overload
from typing_extensions import Self

import numpy as np
import pandas as pd
import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData
from openghg.retrieve import get_obs_surface, get_footprint, get_flux, get_bc
from openghg.types import SearchError
from openghg.util import split_function_inputs

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.basis.algorithms import quadtree_algorithm, weighted_algorithm
from openghg_inversions.models.config.config_parser import ModelGraph, Node


logger = logging.getLogger("openghg_inversions.data_functions")
logger.setLevel(logging.INFO)  # Have to set level for logger as well as handler


def fp_x_flux(fp: FootprintData, flux: FluxData | dict[str, FluxData]) -> xr.DataArray:
    """Calculate array with footprints * flux.

    TODO: make this a function that accepts a model scenario and source?
    """
    ms = ModelScenario(footprint=fp, flux=flux)
    ms.scenario = fp.data  # HACK

    if ms.species.lower() == "co2":
        result = ms._calc_modelled_obs_HiTRes(output_TS=False, output_fpXflux=True)
    else:
        result = ms._calc_modelled_obs_integrated(output_TS=False, output_fpXflux=True)

    return result


def fp_x_bc(fp: FootprintData, bc: BoundaryConditionsData, units: float | None = None) -> xr.Dataset:
    pass


def nesw_bc(fp: FootprintData | xr.Dataset, bc: BoundaryConditionsData | xr.Dataset) -> xr.DataArray:
    """This only works for inert footprints, and only applies the NESW boundary conditions."""
    directions = ["n", "e", "s", "w"]

    if isinstance(fp, FootprintData):
        fp = fp.data

    if isinstance(bc, BoundaryConditionsData):
        bc = bc.data

    fp_bc = fp[[f"particle_locations_{x}" for x in directions]].rename(
        {f"particle_locations_{x}": x for x in directions}
    )
    bc_aligned = (
        bc.rename({f"vmr_{x}": x for x in directions})
        .reindex_like(fp_bc, method="ffill")
        .chunk(time=fp_bc.chunks["time"])
    )

    result = (fp_bc * bc_aligned).sum(["lat", "lon", "height"]).to_dataarray(dim="region", name="h_bc")

    return result


@overload
def align_obs_and_other(
    obs: xr.DataArray,
    other: xr.DataArray,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray]: ...


@overload
def align_obs_and_other(
    obs: xr.Dataset,
    other: xr.DataArray,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.Dataset, xr.DataArray]: ...


@overload
def align_obs_and_other(
    obs: xr.DataArray,
    other: xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray, xr.Dataset]: ...


@overload
def align_obs_and_other(
    obs: xr.Dataset,
    other: xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]: ...


def align_obs_and_other(
    obs: xr.DataArray | xr.Dataset,
    other: xr.DataArray | xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]:
    """
    Slice and resample obs and footprint data to align along time

    This slices the date to the smallest time frame
    spanned by both the footprint and obs, using the sliced start date
    The time dimension is resampled based on the resample_to input using the mean.
    The resample_to options are:
     - "coarsest" - resample to the coarsest resolution between obs and footprints
     - "obs" - resample to observation data frequency
     - "footprint" - resample to footprint data frequency
     - a valid resample period e.g. "2H"

    Args:
        resample_to: Resample option to use: either data based or using a valid pandas resample period.
        platform: Observation platform used to decide whether to resample

    Returns:
        tuple: Two xarray DataArrays with aligned time dimensions
    """
    resample_keyword_choices = ("obs", "other", "coarsest")

    # Check whether resample has been requested by specifying a specific period rather than a keyword
    if resample_to in resample_keyword_choices:
        force_resample = False
    else:
        force_resample = True

    if platform is not None:
        platform = platform.lower()
        # Do not apply resampling for "satellite" (but have re-included "flask" for now)
        if platform == "satellite":
            return obs, other

    # Whether sampling period is present or we need to try to infer this
    infer_sampling_period = False
    # Get the period of measurements in time
    obs_attributes = obs.attrs
    if "averaged_period" in obs_attributes:
        obs_data_period_s = float(obs_attributes["averaged_period"])
    elif "sampling_period" in obs_attributes:
        sampling_period = obs_attributes["sampling_period"]
        if sampling_period == "NOT_SET":
            infer_sampling_period = True
        elif sampling_period == "multiple":
            # If we have a varying sampling_period, make sure we always resample to footprint
            obs_data_period_s = 1.0
        else:
            obs_data_period_s = float(sampling_period)
    elif "sampling_period_estimate" in obs_attributes:
        estimate = obs_attributes["sampling_period_estimate"]
        logger.warning(f"Using estimated sampling period of {estimate}s for observational data")
        obs_data_period_s = float(estimate)
    else:
        infer_sampling_period = True

    if infer_sampling_period:
        # Attempt to derive sampling period from frequency of data
        obs_data_period_s = np.nanmedian((obs.time.data[1:] - obs.time.data[0:-1]) / 1e9).astype("float32")

        obs_data_period_s_min = np.diff(obs.time.data).min() / 1e9
        obs_data_period_s_max = np.diff(obs.time.data).max() / 1e9

        max_diff = (obs_data_period_s_max - obs_data_period_s_min).astype(float)

        # Check if the periods differ by more than 1 second
        if max_diff > 1.0:
            raise ValueError("Sample period can be not be derived from observations")

        estimate = f"{obs_data_period_s:.1f}"
        logger.warning(f"Sampling period was estimated (inferred) from data frequency: {estimate}s")
        obs.attrs["sampling_period_estimate"] = estimate

    # TODO: Check regularity of the data - will need this to decide is resampling
    # is appropriate or need to do checks on a per time point basis

    obs_data_period_ns = obs_data_period_s * 1e9
    obs_data_timeperiod = pd.Timedelta(obs_data_period_ns, unit="ns")

    # Derive the footprints period from the frequency of the data
    other_data_period_ns = np.nanmedian((other.time.data[1:] - other.time.data[0:-1]).astype("int64"))
    other_data_timeperiod = pd.Timedelta(other_data_period_ns, unit="ns")

    # If resample_to is set to "coarsest", check whether "obs" or "footprint" have lower resolution
    if resample_to == "coarsest":
        if obs_data_timeperiod >= other_data_timeperiod:
            resample_to = "obs"
        elif obs_data_timeperiod < other_data_timeperiod:
            resample_to = "footprint"

    # Here we want timezone naive pd.Timestamps
    # Add sampling period to end date to make sure resample includes these values when matching
    obs_startdate = pd.Timestamp(obs.time[0].values)
    obs_enddate = pd.Timestamp(obs.time[-1].values) + obs_data_timeperiod
    footprint_startdate = pd.Timestamp(other.time[0].values)
    footprint_enddate = pd.Timestamp(other.time[-1].values) + other_data_timeperiod

    start_date = max(obs_startdate, footprint_startdate)
    end_date = min(obs_enddate, footprint_enddate)

    # Ensure lower range is covered for obs
    start_obs_slice = start_date - pd.Timedelta("1ns")
    # Ensure extra buffer is added for footprint based on fp timeperiod.
    # This is to ensure footprint can be forward-filled to obs (in later steps)
    start_other_slice = start_date - (other_data_timeperiod - pd.Timedelta("1ns"))
    # Subtract very small time increment (1 nanosecond) to make this an exclusive selection
    end_slice = end_date - pd.Timedelta("1ns")

    obs = obs.sel(time=slice(start_obs_slice, end_slice))
    other = other.sel(time=slice(start_other_slice, end_slice))

    if obs.time.size == 0 or other.time.size == 0:
        raise ValueError("Obs data and Footprint data don't overlap")
    # Only non satellite datasets with different periods need to be resampled
    timeperiod_diff_s = np.abs(obs_data_timeperiod - other_data_timeperiod).total_seconds()
    tolerance = 1e-9  # seconds

    if timeperiod_diff_s >= tolerance or force_resample:
        offset = pd.Timedelta(hours=start_date.hour + start_date.minute / 60.0 + start_date.second / 3600.0)
        offset = cast(pd.Timedelta, offset)

        if resample_to == "obs":
            resample_period = str(round(obs_data_timeperiod / np.timedelta64(1, "h"), 5)) + "H"
            other = other.resample(indexer={"time": resample_period}, offset=offset).mean()

        elif resample_to == "footprint":
            resample_period = str(round(other_data_timeperiod / np.timedelta64(1, "h"), 5)) + "H"
            obs = obs.resample(indexer={"time": resample_period}, offset=offset).mean()

        else:
            resample_period = resample_to
            other = other.resample(indexer={"time": resample_period}, offset=offset).mean()
            obs = obs.resample(indexer={"time": resample_period}, offset=offset).mean()

    return obs, other


class MultiObs:
    def __init__(
        self,
        species,
        start_date,
        end_date,
        sites,
        inlets,
        store=None,
        obs_store=None,
        averaging_period=None,
        **kwargs,
    ) -> None:
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self._sites = sites
        self._inlets = inlets
        self.store = obs_store or store

        self.average = averaging_period or [None] * len(self._sites)

        valid_kwargs, _ = split_function_inputs(kwargs, get_obs_surface)
        self.kwargs = valid_kwargs

        self.obs = []
        self.sites = []
        self.inlets = []

        for site, inlet, avg in zip(self._sites, self._inlets, self.average):
            try:
                obs = get_obs_surface(
                    species=self.species,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    site=site,
                    inlet=inlet,
                    average=avg,
                    store=self.store,
                    **self.kwargs,
                )
            except Exception as e:
                print(f"Couldn't get obs for site {site} and inlet {inlet} from store {self.store}: {e}")
            else:
                self.obs.append(obs)
                self.sites.append(site)
                self.inlets.append(inlet)

        if not self.obs:
            raise SearchError(
                f"No obs. found for {self.species} at sites {self._sites} in store {self.store}"
            )

        self._combined_ds = xr.concat(
            [x.data.expand_dims(site=[x.metadata["site"]]) for x in self.obs], dim="site"
        )

    def combined_ds(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def data(self) -> xr.Dataset:
        return self._combined_ds

    @classmethod
    def from_node(cls, node: Node, comp_data_args: dict[str, ChainMap], **kwargs) -> Self:
        if not node.type.endswith("likelihood"):
            raise ValueError(f"{repr(node)} is not a likelihood: it has type {node.type}.")

        get_kwargs = comp_data_args[node.name]
        get_kwargs.update(kwargs)

        return cls(**get_kwargs)


class MultiFootprint:
    def __init__(
        self,
        domain,
        start_date,
        end_date,
        fp_heights,
        sites,
        fp_species="inert",
        store=None,
        footprint_store=None,
        model=None,
        met_model=None,
        **kwargs,
    ) -> None:
        self.domain = domain
        self.species = fp_species
        self.start_date = start_date
        self.end_date = end_date
        self._sites = sites
        self._inlets = fp_heights
        self.store = footprint_store or store
        self.model = model
        self.met_model = met_model

        valid_kwargs, _ = split_function_inputs(kwargs, get_obs_surface)
        self.kwargs = valid_kwargs

        # need more robust way to do this...
        if "species" in self.kwargs:
            del self.kwargs["species"]

        self.footprints = []
        self.sites = []
        self.inlets = []
        for site, inlet in zip(self._sites, self._inlets):
            try:
                fp = get_footprint(
                    domain=self.domain,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    species=self.species,
                    site=site,
                    inlet=inlet,
                    model=self.model,
                    met_model=self.met_model,
                    store=self.store,
                    **self.kwargs,
                )
            except Exception as e:
                print(
                    f"Couldn't get footprint for site {site} and inlet {inlet} from store {self.store}: {e}"
                )
            else:
                self.footprints.append(fp)
                self.sites.append(site)
                self.inlets.append(inlet)

        self._combined_ds = xr.concat(
            [x.data.expand_dims(site=[x.metadata["site"]]) for x in self.footprints], dim="site"
        )

    def combined_ds(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def data(self) -> xr.Dataset:
        return self._combined_ds

    @classmethod
    def from_node(cls, node: Node, comp_data_args: dict[str, ChainMap], **kwargs) -> Self:
        if not node.type.endswith("likelihood"):
            raise ValueError(f"{repr(node)} is not a likelihood: it has type {node.type}.")

        get_kwargs = comp_data_args[node.name]
        get_kwargs.update(kwargs)

        return cls(**get_kwargs)


class ComponentData:
    component_name: str
    _component_registry: dict = {}

    def __init__(self, node: Node, *args, **kwargs) -> None:
        if node.type != self.component_name:
            raise ValueError(
                f"{repr(node)} must have type '{self.component_name}'; received node of type '{node.type}'"
            )

        self.node = node

    @classmethod
    def __init_subclass__(cls):
        """Register ModelComponents by name, for lookup by model config."""
        ComponentData._component_registry[cls.component_name] = cls


class Flux(ComponentData):

    component_name = "flux"

    def __init__(
        self, node: Node, comp_data_args: Mapping, basis: xr.DataArray | None = None, **kwargs
    ) -> None:
        super().__init__(node)

        have, missing = split_function_inputs(comp_data_args, get_flux)  # type: ignore
        if "emissions_store" in missing:
            have["store"] = missing["emissions_store"]

        self.flux_data = get_flux(**have, **kwargs)

        self.flux = self.flux_data.data.flux

        self._basis = basis
        self.input_dim = f"{node.short_name}_region"

        self._h_matrix = None

    @property
    def basis(self) -> xr.DataArray:
        if self._basis is None:
            raise AttributeError("Basis has not been provided, or has not been computed yet.")
        return self._basis

    def compute_basis(self, mean_fp: xr.DataArray, **kwargs) -> None:
        """Compute basis and store.

        This does not compute basis functions with a time coordinate, which
        is fine for the current setup, but will need to be changed when
        temporal covariance is added.
        """
        # TODO: add fixed outer regions
        try:
            basis_info = self.node.basis.copy()
        except AttributeError:
            basis_info = self.node.data_args.get("basis", {}).copy()

        basis_info.update(kwargs)

        # TODO: choose a coordinate name for region!!!
        try:
            basis_algorithm = basis_info.pop("algorithm")
        except KeyError as e:
            raise ValueError(
                f"{repr(self.node)} doesn't specify a basis algorithm. Pass `algorithm` as a kwarg."
            ) from e

        if basis_algorithm == "quadtree":
            basis_func = quadtree_algorithm
        elif basis_algorithm == "weighted":
            basis_func = weighted_algorithm
        else:
            raise ValueError(f"basis algorithm {basis_algorithm} not found.")

        func = partial(basis_func, **basis_info)

        fp_x_flux = mean_fp * self.flux.mean("time")

        self.flat_basis = xr.apply_ufunc(func, fp_x_flux.as_numpy()).rename("basis")
        self._basis = get_xr_dummies(self.flat_basis, cat_dim=self.input_dim)

    def compute_h_matrix(self, footprint: xr.Dataset) -> None:
        flux = self.flux.reindex_like(footprint, method="ffill").chunk(time=footprint.chunks["time"])  # type: ignore
        fp_x_flux = footprint.fp * flux
        self._h_matrix = fp_x_flux @ self.basis

    @property
    def h_matrix(self) -> xr.DataArray:
        if self._h_matrix is None:
            raise AttributeError(f"h_matrix for {self.node.name} has not been computed yet.")
        self._h_matrix.attrs["origin"] = self.node.name
        self._h_matrix.attrs["param"] = "h_matrix"
        return self._h_matrix


class BoundaryConditions(ComponentData):
    """Data getting functions for BoundaryConditions model component.

    NOTE: currently this only allows NESW bc boundary conditions
    """

    component_name = "bc"

    def __init__(self, node: Node, comp_data_args: Mapping, **kwargs) -> None:
        super().__init__(node)

        have, missing = split_function_inputs(comp_data_args, get_bc)  # type: ignore
        if "bc_store" in missing:
            have["store"] = missing["bc_store"]

        self.bc_data = get_bc(**have, **kwargs)

        self.bc = self.bc_data.data

        self.input_dim = f"{node.short_name}_region"

        self._h_matrix = None

    def compute_h_matrix(self, footprint: xr.Dataset) -> None:
        self._h_matrix = nesw_bc(footprint, self.bc).rename(region=self.input_dim)

    @property
    def h_matrix(self) -> xr.DataArray:
        if self._h_matrix is None:
            raise AttributeError(f"h_matrix for {self.node.name} has not been computed yet.")
        self._h_matrix.attrs["origin"] = self.node.name
        self._h_matrix.attrs["param"] = "h_matrix"
        return self._h_matrix


class Tracer(ComponentData):

    component_name = "tracer"

    def __init__(self, node: Node, flux_data: Flux) -> None:
        super().__init__(node)

        self.input_dim = flux_data.input_dim
        self._h_matrix = flux_data.h_matrix

    @property
    def h_matrix(self) -> xr.DataArray:
        self._h_matrix.attrs["origin"] = self.node.name
        self._h_matrix.attrs["param"] = "h_matrix"
        return self._h_matrix


class MultisectorFlux(ComponentData):

    component_name = "multisector_flux"

    def __init__(self, node: Node) -> None:
        super().__init__(node)

    def merge_data(self, comp_data: dict[str, ComponentData]) -> None:
        to_merge = []

        for child in self.node.children:
            if child.skip:
                continue

            child_data = comp_data[child.name]
            to_merge.append(child_data.h_matrix.rename(f"{child_data.node.name}_h".replace(".", "_")))  # type: ignore

        self.h_matrix = xr.merge(to_merge)


class Baseline(ComponentData):

    component_name = "baseline"

    def __init__(self, node: Node) -> None:
        super().__init__(node)

    def merge_data(self, comp_data: dict[str, ComponentData]) -> None:
        to_merge = []

        for child in self.node.children:
            if child.skip:
                continue

            child_data = comp_data[child.name]
            if isinstance(child_data, BoundaryConditions):
                to_merge.append(child_data.h_matrix.rename(f"{child_data.node.name}_h".replace(".", "_")))
            # else:
            #     # TODO: need to add Offset ComponentData and heck here
            #     to_merge.append(child_data.h_matrix)

        self.h_matrix = xr.merge(to_merge)


class ForwardModel(ComponentData):

    component_name = "forward_model"

    def __init__(self, node: Node, comp_data_args: Mapping) -> None:
        super().__init__(node)

        self._multi_footprint = MultiFootprint(**comp_data_args)
        self.footprints = self._multi_footprint.data
        self.mean_fp = self.footprints.fp.mean(["site", "time"])

    def merge_data(self, comp_data: dict[str, ComponentData]) -> None:
        to_merge = []

        for child in self.node.children:
            if child.skip:
                continue

            child_data = comp_data[child.name]
            if isinstance(child_data, Flux | Tracer):
                to_merge.append(child_data.h_matrix.rename(f"{child_data.node.name}_h".replace(".", "_")))
            else:
                to_merge.append(child_data.h_matrix)  # type: ignore

        self.h_matrix = xr.merge(to_merge)


# TODO: likelihoods, Sigma, need to align Likelihood and Forward
# TODO: functions to add data to pm.Data correctly
def site_indicator_func(sites: Iterable[str]) -> Callable:
    site_dict = {site: i for i, site in enumerate(sites)}
    return lambda arr: [site_dict.get(x, -1) for x in arr]


class Sigma(ComponentData):

    component_name = "sigma"

    def __init__(self, node: Node) -> None:
        super().__init__(node)

    def get_data(self, parent: ComponentData) -> None:
        """Get data from parent.

        Args:
            parent: a ComponentData object that has obs info; typically a likelihood.
                This could also be a `MultiObs` object.

        Returns:
            None, stores data
        """

        si_func = site_indicator_func(parent.sites)  # type: ignore

        # need to stack and unstack here?
        self.site_indicator = (xr.apply_ufunc(si_func, parent.data.stack(nmeasure=["site", "time"]).site)   # type: ignore
                               .unstack("nmeasure")
                               .rename("site_indicator"))
        self.site_indicator.attrs["origin"] = self.node.name
        self.site_indicator.attrs["param"] = "site_indicator"

        self.to_merge = [self.site_indicator]


class LikelihoodComponentData(ComponentData):
    """Wrapper for MultiObs. Should be subclasses for each likelihood ModelComponent"""

    component_name = "likelihood"

    def __init_subclass__(cls) -> None:
        """Check that the subclass component names end with 'likelihood'."""
        super().__init_subclass__()
        assert cls.component_name.endswith("likelihood")

    def __init__(self, node: Node, comp_data_args: Mapping) -> None:
        super().__init__(node)

        self._multi_obs = MultiObs(**comp_data_args)
        self.obs = self._multi_obs.data

        self.y_obs = self.obs.mf.rename("y_obs")
        # self.y_obs = self.obs.mf.stack(nmeasure=["site", "time"]).rename("y_obs")
        self.y_obs.attrs["param"] = "y_obs"

        self._to_merge = [self.y_obs]

    @property
    def to_merge(self) -> list[xr.DataArray]:
        for x in self._to_merge:
            x.attrs["origin"] = self.node.name

        return self._to_merge

    def __getattr__(self, name: str, /) -> Any:
        """Pass through attribute requires to underlying MultiObs object."""
        return getattr(self._multi_obs, name)

    def merge_data(self, comp_data: dict[str, ComponentData]) -> None:
        children_to_merge = []

        for child in self.node.children:
            if child.skip:
                continue

            child_data = comp_data[child.name]

            if hasattr(child_data, "to_merge"):
                children_to_merge.extend(child_data.to_merge)  # type: ignore ...we just checked this exists

        self.merged_data = xr.merge(self.to_merge + children_to_merge)

        if "averaged_period" in self.obs.attrs:
            self.merged_data.attrs["averaged_period"] = self.obs.attrs["averaged_period"]


class GaussianLikelihood(LikelihoodComponentData):

    component_name = "gaussian_likelihood"

    def __init__(self, node: Node, comp_data_args: Mapping, ffill_error: bool = True) -> None:
        super().__init__(node, comp_data_args)
        repeatability = (
            self.data.mf_repeatability if "mf_repeatability" in self.data else xr.zeros_like(self.y_obs)
        )
        variability = self.data.mf_variability if "mf_variability" in self.data else xr.zeros_like(self.y_obs)

        if ffill_error:
            repeatability = repeatability.ffill("time")
            variability = variability.ffill("time")

        self.error = np.sqrt(repeatability**2 + variability**2).rename("error")
        self.error.attrs["param"] = "error"

        self.to_merge.append(self.error)


class RHIMELikelihood(GaussianLikelihood):

    component_name = "rhime_likelihood"

    def __init__(
        self,
        node: Node,
        comp_data_args: Mapping,
        ffill_error: bool = True,
        min_error: np.ndarray | xr.DataArray | float = 0.0,
    ) -> None:
        super().__init__(node, comp_data_args, ffill_error)

        # TODO: add options to calculate min error...
        if isinstance(min_error, float) or (isinstance(min_error, np.ndarray) and min_error.ndim == 0):
            self.min_error = min_error * xr.ones_like(self.y_obs).rename("min_error")
        elif isinstance(min_error, np.ndarray):
            self.min_error = xr.DataArray(min_error, coords=self.y_obs.coords).rename("min_error")
        else:
            self.min_error = min_error.rename("min_error")  # type: ignore

        self.min_error.attrs["param"] = "min_error"

        self.to_merge.append(self.min_error)
