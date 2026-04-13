from types import MappingProxyType

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import (
    InferPyMCModelSetup,
    _canonicalise_inferpymc_dataset,
    build_inferpymc_model,
    inferpymc,
)
from openghg_inversions.models.coords import restore_inferencedata_coords


@pytest.fixture(scope="module")
def inferpymc_inputs_dataset(mhd_and_tac_fp_data) -> xr.Dataset:
    return make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        min_error_per_site=False,
        start_date="2019-01-01",
    )


@pytest.fixture(scope="module")
def inferpymc_args(inferpymc_inputs_dataset: xr.Dataset) -> MappingProxyType:
    args = {
        "inv_inputs": inferpymc_inputs_dataset,
        "nit": 1,
        "burn": 0,
        "tune": 0,
        "nchain": 1,
        "xprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "bcprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
        "sigma_per_site": True,
        "offsetprior": {"pdf": "normal", "mu": 0, "sigma": 1},
        "add_offset": False,
        "use_bc": True,
        "reparameterise_log_normal": False,
        "pollution_events_from_obs": True,
        "no_model_error": False,
        "offset_args": {"drop_first": False, "offset_freq": "D"},
        "power": 1.99,
        "verbose": False,
        "sampler_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
    }
    return MappingProxyType(args)


def test_build_inferpymc_model_returns_setup_for_pymc(inferpymc_args: dict) -> None:
    setup = build_inferpymc_model(
        inv_inputs=inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
        nuts_sampler="pymc",
    )

    assert isinstance(setup, InferPyMCModelSetup)
    assert isinstance(setup.model, pm.Model)
    assert isinstance(setup.step1, pm.NUTS)
    assert isinstance(setup.step2, pm.Slice)
    assert setup.sample_kwargs["step"] == [setup.step1, setup.step2]


def test_build_inferpymc_model_returns_no_steps_for_numpyro(inferpymc_args: dict) -> None:
    setup = build_inferpymc_model(
        inv_inputs=inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
        nuts_sampler="numpyro",
    )

    assert isinstance(setup, InferPyMCModelSetup)
    assert isinstance(setup.model, pm.Model)
    assert isinstance(setup.step1, pm.NUTS)
    assert isinstance(setup.step2, pm.Slice)
    assert setup.sample_kwargs["step"] is None


@pytest.fixture(scope="module")
def inferpymc_with_bc_result(inferpymc_args: dict):
    return inferpymc(**inferpymc_args)


def test_inferpymc_runs_on_inversion_inputs(
    inferpymc_inputs_dataset: xr.Dataset, inferpymc_with_bc_result
) -> None:
    result = inferpymc_with_bc_result

    expected_keys = {
        "xouts",
        "sigouts",
        "Ytrace",
        "OFFSETtrace",
        "convergence",
        "step1",
        "step2",
        "model",
        "trace",
        "bcouts",
        "YBCtrace",
    }
    assert expected_keys.issubset(result.keys())

    assert result["xouts"].sizes["nx"] == inferpymc_inputs_dataset.sizes["region"]
    assert result["sigouts"].sizes["nsigma_time"] == len(
        np.unique(inferpymc_inputs_dataset["sigma_freq_index"].values)
    )
    assert result["sigouts"].sizes["nsigma_site"] == len(
        np.unique(inferpymc_inputs_dataset["site_indicator"].values)
    )

    y = inferpymc_inputs_dataset["mf"].values
    assert y.size in result["Ytrace"].shape
    assert result["OFFSETtrace"].shape == result["Ytrace"].shape
    assert result["YBCtrace"].shape == result["Ytrace"].shape

    assert np.isfinite(result["Ytrace"]).all()
    assert np.isfinite(result["OFFSETtrace"]).all()
    assert np.isfinite(result["YBCtrace"]).all()


def test_inferpymc_model_contains_expected_variables(
    inferpymc_inputs_dataset: xr.Dataset, inferpymc_with_bc_result
) -> None:
    model = inferpymc_with_bc_result["model"]

    expected_named_vars = {
        "x",
        "bc",
        "sigma",
        "hx",
        "hbc",
        "Y",
        "error",
        "min_error",
        "mu",
        "mu_bc",
        "epsilon",
        "y",
    }
    assert expected_named_vars.issubset(model.named_vars)

    assert len(model.coords["nmeasure"]) == inferpymc_inputs_dataset.sizes["nmeasure"]
    assert len(model.coords["nx"]) == inferpymc_inputs_dataset.sizes["region"]
    assert len(model.coords["nbc"]) == inferpymc_inputs_dataset.sizes["bc_region"]
    assert len(model.coords["nsigma_site"]) == len(
        np.unique(inferpymc_inputs_dataset["site_indicator"].values)
    )
    assert len(model.coords["nsigma_time"]) == len(
        np.unique(inferpymc_inputs_dataset["sigma_freq_index"].values)
    )


def test_build_inferpymc_model_accepts_dataset(
    inferpymc_args: dict, inferpymc_inputs_dataset: xr.Dataset
) -> None:
    setup = build_inferpymc_model(
        inv_inputs=inferpymc_inputs_dataset,
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
        nuts_sampler="pymc",
    )

    assert isinstance(setup, InferPyMCModelSetup)
    assert isinstance(setup.model, pm.Model)
    assert len(setup.model.coords["nmeasure"]) == inferpymc_inputs_dataset.sizes["nmeasure"]
    assert len(setup.model.coords["nx"]) == inferpymc_inputs_dataset.sizes["region"]


def test_canonicalise_inferpymc_dataset_preserves_dataset_observation_coords(
    inferpymc_inputs_dataset: xr.Dataset,
) -> None:
    canonical = _canonicalise_inferpymc_dataset(inferpymc_inputs_dataset, use_bc=True)

    assert {"H", "H_bc", "mf", "mf_error", "site_indicator", "sigma_freq_index", "min_error"}.issubset(
        canonical.data_vars
    )
    assert canonical["H"].dims == ("nmeasure", "nx")
    assert canonical["H_bc"].dims == ("nmeasure", "nbc")
    assert "time" in canonical.coords
    assert canonical.indexes["nmeasure"].equals(inferpymc_inputs_dataset.indexes["nmeasure"])


def test_restore_inferencedata_coords_helper_restores_multiindex() -> None:
    multi_index = pd.MultiIndex.from_arrays(
        [["MHD", "MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-01"])],
        names=["site", "time"],
    )
    posterior = xr.Dataset(
        data_vars={"x": (("chain", "draw", "nmeasure"), np.zeros((1, 1, 3)))},
        coords={"chain": [0], "draw": [0], "nmeasure": np.arange(3)},
    )
    idata = az.InferenceData(posterior=posterior)

    restored = restore_inferencedata_coords(idata, {"nmeasure": multi_index})

    assert "nmeasure" in restored.posterior.indexes
    assert restored.posterior.indexes["nmeasure"].equals(multi_index)


def test_inferpymc_runs_without_boundary_conditions(inferpymc_args: dict) -> None:
    args = dict(inferpymc_args)
    args["use_bc"] = False

    result = inferpymc(**args)

    assert "bcouts" not in result
    assert "YBCtrace" not in result

    model = result["model"]
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars
