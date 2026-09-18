import arviz as az
import numpy as np
import pytest
import xarray as xr

import openghg_inversions.hbmcmc.hbmcmc as hbmcmc_module
from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC


def _deterministic_inferpymc_results(
    inv_inputs: xr.Dataset,
    *,
    use_bc: bool,
    reparameterise_log_normal: bool,
) -> dict[str, object]:
    """Build realistic fixedbasis sampler results without posterior sampling.

    Args:
        inv_inputs: Fully prepared fixedbasis model inputs.
        use_bc: Whether boundary-condition variables should be included.
        reparameterise_log_normal: Whether to include the sampler's latent
            lognormal variable alongside the public scaling factor.

    Returns:
        A fresh legacy ``inferpymc`` result mapping with deterministic prior,
        posterior, and predictive traces derived from ``inv_inputs``.
    """
    draw_count = 8
    nregion = inv_inputs.sizes["region"]
    draw_offsets = np.linspace(-0.1, 0.1, draw_count)
    x_posterior = 1.0 + draw_offsets[:, None] + 0.01 * np.arange(nregion)[None, :]
    x_prior = 1.0 + 2 * draw_offsets[:, None] + 0.01 * np.arange(nregion)[None, :]
    h_matrix = inv_inputs["H"].transpose("region", "nmeasure").values
    mu_posterior = x_posterior @ h_matrix
    mu_prior = x_prior @ h_matrix

    site_index = np.asarray(inv_inputs["site_indicator"].values, dtype=int)
    sigma_index = np.asarray(inv_inputs["sigma_freq_index"].values, dtype=int)
    nsite = int(site_index.max(initial=0)) + 1
    nsigma_time = int(sigma_index.max(initial=0)) + 1
    sigma_posterior = np.broadcast_to(
        1.0 + draw_offsets[:, None, None],
        (draw_count, nsite, nsigma_time),
    ).copy()
    sigma_prior = np.broadcast_to(
        1.2 + draw_offsets[:, None, None],
        (draw_count, nsite, nsigma_time),
    ).copy()
    observation_error = np.asarray(inv_inputs["mf_error"].values)
    epsilon_posterior = np.sqrt(
        observation_error[None, :] ** 2 + sigma_posterior[:, site_index, sigma_index] ** 2
    )
    epsilon_prior = np.sqrt(observation_error[None, :] ** 2 + sigma_prior[:, site_index, sigma_index] ** 2)

    posterior = {
        "x": x_posterior[None, ...],
        "sigma": sigma_posterior[None, ...],
        "mu": mu_posterior[None, ...],
        "epsilon": epsilon_posterior[None, ...],
    }
    prior = {
        "x": x_prior[None, ...],
        "sigma": sigma_prior[None, ...],
        "mu": mu_prior[None, ...],
        "epsilon": epsilon_prior[None, ...],
    }
    dims = {
        "x": ["nx"],
        "sigma": ["nsigma_site", "nsigma_time"],
        "mu": ["nmeasure"],
        "epsilon": ["nmeasure"],
        "y": ["nmeasure"],
    }
    if reparameterise_log_normal:
        posterior["x_latent"] = np.log(x_posterior)[None, ...]
        prior["x_latent"] = np.log(x_prior)[None, ...]
        dims["x_latent"] = ["nx"]
    coords = {
        "nx": inv_inputs["region"].values,
        "nsigma_site": np.arange(nsite),
        "nsigma_time": np.arange(nsigma_time),
        "nmeasure": np.arange(inv_inputs.sizes["nmeasure"]),
    }

    if use_bc:
        h_bc = inv_inputs["H_bc"].transpose("bc_region", "nmeasure").values
        nbc = h_bc.shape[0]
        bc_posterior = 1.0 + 0.5 * draw_offsets[:, None] + 0.01 * np.arange(nbc)[None, :]
        bc_prior = 1.0 + draw_offsets[:, None] + 0.01 * np.arange(nbc)[None, :]
        mu_bc_posterior = bc_posterior @ h_bc
        mu_bc_prior = bc_prior @ h_bc
        posterior.update({"bc": bc_posterior[None, ...], "mu_bc": mu_bc_posterior[None, ...]})
        prior.update({"bc": bc_prior[None, ...], "mu_bc": mu_bc_prior[None, ...]})
        dims.update({"bc": ["nbc"], "mu_bc": ["nmeasure"]})
        coords["nbc"] = np.arange(nbc)
    else:
        mu_bc_posterior = np.zeros_like(mu_posterior)
        mu_bc_prior = np.zeros_like(mu_prior)

    y_posterior = mu_posterior + mu_bc_posterior
    y_prior = mu_prior + mu_bc_prior
    trace = az.from_dict(
        posterior=posterior,
        prior=prior,
        posterior_predictive={"y": y_posterior[None, ...]},
        prior_predictive={"y": y_prior[None, ...]},
        coords=coords,
        dims=dims,
    )
    trace_groups: dict[str, xr.Dataset] = {}
    nmeasure_index = inv_inputs.indexes["nmeasure"]
    for group in trace.groups():
        dataset = trace[group]
        if "nmeasure" in dataset.dims:
            level_names = list(nmeasure_index.names)
            dataset = dataset.assign_coords(
                {name: ("nmeasure", nmeasure_index.get_level_values(name)) for name in level_names}
            ).set_index(nmeasure=level_names)
        if "nbc" in dataset.dims:
            bc_index = inv_inputs.indexes["bc_region"]
            level_names = list(bc_index.names)
            dataset = dataset.assign_coords(
                {name: ("nbc", bc_index.get_level_values(name)) for name in level_names}
            ).set_index(nbc=level_names)
        trace_groups[group] = dataset
    trace = az.InferenceData(**trace_groups)
    return {
        "trace": trace,
        "model": object(),
        "xouts": x_posterior,
    }


@pytest.fixture
def deterministic_sampler(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace only ``inferpymc`` and expose the fully prepared call arguments."""
    calls: list[dict[str, object]] = []

    def fake_inferpymc(**kwargs: object) -> dict[str, object]:
        """Return deterministic results for the prepared fixedbasis inputs."""
        calls.append(dict(kwargs))
        inv_inputs = kwargs["inv_inputs"]
        assert isinstance(inv_inputs, xr.Dataset)
        return _deterministic_inferpymc_results(
            inv_inputs,
            use_bc=bool(kwargs["use_bc"]),
            reparameterise_log_normal=bool(kwargs.get("reparameterise_log_normal", False)),
        )

    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    return calls


@pytest.fixture
def mcmc_args(
    tmp_path,
    tac_ch4_data_args,
    merged_data_dir,
    merged_data_file_name,
    default_bc_basis_directory,
    europe_country_file,
):
    mcmc_args = tac_ch4_data_args.copy()
    mcmc_args.update(
        {
            "outputname": "test_run",
            "outputpath": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "xprior": {"pdf": "lognormal", "stdev": 1.0},
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "merged_data_name": merged_data_file_name,
            "bc_basis_directory": default_bc_basis_directory,
            "country_file": europe_country_file,
        }
    )
    return mcmc_args


def test_full_inversion(mcmc_args):
    """Run one real fixed-basis integration with current priors and NumPyro."""
    mcmc_args["reload_merged_data"] = False
    mcmc_args["nuts_sampler"] = "numpyro"
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yerror_repeatability" in out
    assert "Yerror_variability" in out

    # sanity check for modelled values to make sure baseline has correct order of magnitude
    assert np.mean(np.abs(out.Yobs.values - out.Yapriori.values)) < 0.5 * np.mean(out.Yobs.values)


def test_full_inversion_paris_outputs(mcmc_args, deterministic_sampler):
    """Test full inversion including loading data with PARIS output format."""
    mcmc_args["reload_merged_data"] = False
    mcmc_args["output_format"] = "paris"
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yapost" in out


def test_full_inversion_flux_dim_shuffled(mcmc_args, deterministic_sampler):
    mcmc_args["emissions_name"] = ["total-ukghg-edgar7-shuffled"]
    mcmc_args["reload_merged_data"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_lognormal_infer(mcmc_args, deterministic_sampler):
    mcmc_args["xprior"] = {"pdf": "lognormal", "stdev": 2.0}
    out = fixedbasisMCMC(**mcmc_args)

    expected_sigma = str(np.sqrt(np.log(5)))

    # look for a few decimal places of expected sigma in output attributes
    assert expected_sigma[:4] in out.attrs["Emissions Prior"]
    assert deterministic_sampler[0]["xprior"] == {
        "pdf": "lognormal",
        "mu": 0.5 * np.log(0.2),
        "sigma": np.sqrt(np.log(5)),
    }


def test_inversion_if_merged_data_does_not_exist(mcmc_args, deterministic_sampler):
    """Test that inversion runs if reload_merged_data is True, but
    no merged data exists under the default merged data name.
    """
    mcmc_args["merged_data_name"] = None
    fixedbasisMCMC(**mcmc_args)
