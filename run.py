from datetime import datetime
from getpass import getuser
from pathlib import Path
from typing import Iterable, cast

import arviz as az
from arviz.data.base import dict_to_dataset
import pandas as pd
import pymc as pm
from pymc.backends.arviz import coords_and_dims_for_inferencedata, find_constants, find_observations
import xarray as xr

from openghg_inversions.data_functions import ComponentData
from openghg_inversions.models.config.data_parser import make_data_dict
from openghg_inversions.models.config.config_parser import ModelGraph

try:
    import tomllib as toml
except ModuleNotFoundError:
    import pip._vendor.tomli as toml


def get_sampling_args(conf: dict | None = None) -> dict:
    """Extract pymc sampling args from config, or provide defaults."""
    default_sampling_args = {"draws": 10000, "tune": 10000, "chains": 4, "nuts_sampler": "numpyro"}

    if conf is None:
        return default_sampling_args

    sampling_args = conf.get("sampling", conf)

    # fill any missing values with defaults
    sampling_args.update(default_sampling_args)

    return sampling_args


def default_output_name() -> str:
    return f"inversion_{getuser()}_{datetime.now()}"


def default_idata(model: pm.Model) -> az.InferenceData:
    """Code from pymc.backends.arviz to make idata from model when no sampling has been done.

    TODO: ask PyMC to add this feature?
    """
    coords, dims = coords_and_dims_for_inferencedata(model)
    id_dict = {}

    obs_data = find_observations(model)

    if obs_data:
        id_dict["observed_data"] = dict_to_dataset(obs_data, library=pm, coords=coords, dims=dims)

    constant_data = find_constants(model)

    if constant_data:
        constants_ds = dict_to_dataset(constant_data, library=pm, coords=coords, dims=dims)
        # provisional handling of scalars in constant
        # data to prevent promotion to rank 1
        # in the future this will be handled by arviz
        scalars = [var_name for var_name, value in constant_data.items() if np.ndim(value) == 0]
        for s in scalars:
            s_dim_0_name = f"{s}_dim_0"
            constants_ds = constants_ds.squeeze(s_dim_0_name, drop=True)

        id_dict["constant_data"] = constants_ds

    return az.InferenceData(**id_dict)


def component_data_to_datatree(
    comp_data: dict[str, ComponentData], keywords: str | Iterable[str] = ("flat_basis", "flux")
) -> xr.DataTree:
    dataset_dict = {}

    if isinstance(keywords, str):
        keywords = [keywords]
    else:
        keywords = list(keywords)

    for k, v in comp_data.items():
        to_merge = []
        for kwd in keywords:
            if hasattr(v, kwd):
                to_merge.append(getattr(v, kwd))
        if to_merge:
            dataset_dict[k] = xr.merge(to_merge)

    return xr.DataTree.from_dict(dataset_dict)


def load_config(conf_path: str | Path) -> dict:
    conf_path = Path(conf_path)
    with open(conf_path, "rb") as f:
        conf = toml.load(f)
    return conf


def setup_model_graph(conf: dict, verbose: bool = False) -> ModelGraph:
    # parse config to make model graph
    mg = ModelGraph.from_config(conf)

    # get data for model
    data_dict, comp_data = make_data_dict(mg, conf)
    mg.add_data(data_dict)
    mg.component_data.update(comp_data)

    # build model
    mg.build_model(verbose=verbose)

    return mg


def sample(
    mg: ModelGraph,
    conf: dict | None = None,
    prior_predictive: bool = True,
    posterior: bool = True,
    posterior_predictive: bool = True,
) -> az.InferenceData:
    if posterior_predictive is True and posterior is False:
        raise ValueError("Cannot sample posterior predictive without sampling posterior.")

    conf = get_sampling_args(conf)

    if posterior:
        with mg.model:
            idata = pm.sample(**conf)

            if posterior_predictive:
                idata.extend(pm.sample_posterior_predictive(idata))

            if prior_predictive:
                idata.extend(pm.sample_prior_predictive(samples=conf["draws"]))
    elif prior_predictive:
        with mg.model:
            idata = cast(az.InferenceData, pm.sample_prior_predictive(samples=conf["draws"]))
    else:
        idata = default_idata(mg.model)

    # convert "nmeasure" coords back to site and time
    output_coords = {node.component().output_dim: node.component().output_coord for node in mg.nodes if node.type.endswith("likelihood")}  # type: ignore
    idata.assign_coords(output_coords, inplace=True)

    return idata


def main(
    conf_path: str | Path,
    verbose: bool = False,
    output_path: str | Path | None = None,
    output_name: str | None = None,
    sample_prior_predictive: bool = True,
    sample_posterior: bool = True,
    sample_posterior_predictive: bool = True,
    save_trace: bool = True,
    save_summary: bool = True,
    save_basis_functions: bool = True,
) -> None:
    conf = load_config(conf_path)

    mg = setup_model_graph(conf, verbose)

    idata = sample(
        mg,
        conf,
        posterior=sample_posterior,
        posterior_predictive=sample_posterior_predictive,
        prior_predictive=sample_prior_predictive,
    )

    # output
    output_path = Path(output_path) if output_path is not None else Path.cwd()
    output_name = output_name or default_output_name()

    # NOTE: saving the trace also saves the data used by model components, e.g. H matrices, observations, etc.
    if save_trace:
        idata.to_netcdf(str(output_path / (output_name + "_trace.nc")))

    if save_summary and sample_posterior:
        summary = az.summary(idata)
        summary.to_csv(output_path / (output_name + "_summary.csv"))

    if save_basis_functions:
        basis_functions = component_data_to_datatree(mg.component_data)
        basis_functions.to_netcdf(output_path / (output_name + "_basis_functions.nc"))
