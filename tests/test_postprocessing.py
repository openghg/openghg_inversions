import pytest
import xarray as xr
from pathlib import Path
import arviz as az

from openghg_inversions.hbmcmc.hbmcmc import _resolve_output_format, fixedbasisMCMC
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.legacy_outputs import make_legacy_hbmcmc_output
from openghg_inversions.postprocessing.make_outputs import basic_output, make_country_outputs
from openghg_inversions.postprocessing.make_paris_outputs import (
    make_paris_flux_outputs_from_rhime,
    make_paris_outputs,
)


@pytest.fixture
def mcmc_args(tmp_path, tac_ch4_data_args, merged_data_dir, merged_data_file_name):
    mcmc_args = tac_ch4_data_args.copy()
    mcmc_args.update(
        {
            "outputname": "test_run",
            "outputpath": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "nit": 100,
            "burn": 0,
            "tune": 0,
            "nchain": 2,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "merged_data_name": merged_data_file_name,
            "nuts_sampler": "numpyro",
        }
    )
    return mcmc_args


@pytest.fixture(scope="module")
def inv_out(raw_data_path):
    return InversionOutput.load(raw_data_path / "inversion_output.nc")


@pytest.fixture(scope="module")
def inv_out_eastasia(raw_data_path):
    return InversionOutput.load(raw_data_path / "inversion_output_EASTASIA.nc")


def test_rhime_flux_reprocessing(europe_country_file, raw_data_path):
    """Check that we can re-run PARIS flux outputs on standard RHIME outputs."""
    rhime_outs = xr.open_dataset(raw_data_path / "standard_rhime_outs.nc")
    paris_outs = make_paris_flux_outputs_from_rhime(
        rhime_outs, species="ch4", domain="europe", country_file=europe_country_file
    )

    assert "flux_total_prior" in paris_outs
    assert "flux_total_posterior" in paris_outs


def test_rhime_flux_reprocessing_eastasia(eastasia_country_file, raw_data_path):
    """Check that we can re-run PARIS flux outputs on standard RHIME outputs from EASTASIA."""
    rhime_outs = xr.open_dataset(raw_data_path / "standard_rhime_outs_EASTASIA.nc")
    paris_outs = make_paris_flux_outputs_from_rhime(
        rhime_outs, species="hfc23", domain="eastasia", country_file=eastasia_country_file
    )

    assert "flux_total_prior" in paris_outs
    assert "flux_total_posterior" in paris_outs


def test_basic_outputs(inv_out, europe_country_file):
    """Test creation of basic output for EUROPE domain.

    The default stats calculated are "mean" and "quantile".
    Check that these are all present.
    """
    outs = basic_output(inv_out, country_file=europe_country_file)

    conc_vars = ["y_posterior_predictive", "y_prior_predictive"]
    for x in ["flux", "scaling", "country", "mu_bc"]:
        for y in ["prior", "posterior"]:
            conc_vars.append(x + "_" + y)

    stats = ["mean", "quantile"]

    for cv in conc_vars:
        for stat in stats:
            assert cv + "_" + stat in outs


def test_basic_outputs_eastasia(inv_out_eastasia, eastasia_country_file):
    """Test creation of basic output for EASTASIA domain.

    The default stats calculated are "mean" and "quantile".
    Check that these are all present.
    """
    outs = basic_output(inv_out_eastasia, country_file=eastasia_country_file)

    conc_vars = ["y_posterior_predictive", "y_prior_predictive"]
    for x in ["flux", "scaling", "country", "mu_bc"]:
        for y in ["prior", "posterior"]:
            conc_vars.append(x + "_" + y)

    stats = ["mean", "quantile"]

    for cv in conc_vars:
        for stat in stats:
            assert cv + "_" + stat in outs


@pytest.mark.parametrize("offset", [False, True])
def test_make_paris_outputs(inv_out, europe_country_file, tmpdir, offset):
    """Check that we can create and save PARIS outputs for EUROPE domain"""

    if offset:
        # fake an offset trace
        inv_out.trace.posterior["offset"] = xr.ones_like(inv_out.trace.posterior["mu_bc"])
        inv_out.trace.prior["offset"] = xr.ones_like(inv_out.trace.prior["mu_bc"])
        inv_out.trace_ds["offset_posterior"] = xr.ones_like(inv_out.trace_ds.mu_bc_posterior)
        inv_out.trace_ds["offset_prior"] = xr.ones_like(inv_out.trace_ds.mu_bc_prior)

    print(inv_out.trace.posterior)

    flux_outs, conc_outs = make_paris_outputs(
        inv_out, country_file=europe_country_file, obs_avg_period="1h", domain="europe"
    )

    if offset:
        assert "Yapriori_bias" in conc_outs

    # check we can write to netCDF
    flux_outs.to_netcdf(tmpdir / "flux.nc")
    conc_outs.to_netcdf(tmpdir / "conc.nc")


def test_save_inversion_output(mcmc_args, tmpdir):
    """Check that we can save and reload inversion outputs"""
    mcmc_args["save_inversion_output"] = str(tmpdir / "inv_out.nc")
    mcmc_args["output_format"] = "inv_out"
    inv_out = fixedbasisMCMC(**mcmc_args)

    inv_out_reloaded = InversionOutput.load(tmpdir / "inv_out.nc")

    assert inv_out == inv_out_reloaded


def test_country_outputs_lognormal_reparam_conflict(mcmc_args, europe_country_file):
    """Check country outputs ignore reparameterized latent-only traces."""
    mcmc_args["output_format"] = "inv_out"
    mcmc_args["reparameterise_log_normal"] = True
    mcmc_args["xprior"] = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}

    inv_out = fixedbasisMCMC(**mcmc_args)
    trace_ds = inv_out.get_trace_dataset(var_names="x")
    assert "x_prior" in trace_ds
    assert "x_posterior" in trace_ds
    assert "x_latent_prior" not in trace_ds
    assert "x_latent_posterior" not in trace_ds

    country_outs = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")
    assert "country_prior_mean" in country_outs
    assert "country_posterior_mean" in country_outs


def test_hbmcmc_postprocessing_saves_legacy_output(mcmc_args, tmpdir):
    mcmc_args["output_format"] = "hbmcmc_postprocessing"
    mcmc_args["outputpath"] = str(tmpdir)

    outputs = fixedbasisMCMC(**mcmc_args)
    output_file = define_output_filename(
        outputpath=str(tmpdir),
        species=mcmc_args["species"],
        domain=mcmc_args["domain"],
        outputname=mcmc_args["outputname"],
        start_date=mcmc_args["start_date"],
        ext=".nc",
    )

    assert Path(output_file).exists()
    reloaded = xr.open_dataset(output_file)
    assert reloaded.sizes["nmeasure"] == outputs.sizes["nmeasure"]


def test_resolve_output_format_canonicalizes_paris_compatibility():
    with pytest.warns(UserWarning, match="Use `output_format = 'paris'` instead"):
        resolved = _resolve_output_format("hbmcmc", paris_postprocessing=True, is_column=False)

    assert resolved == "paris"


def test_paris_postprocessing_compatibility_matches_paris_output_format(mcmc_args):
    explicit_args = mcmc_args.copy()
    explicit_args["output_format"] = "paris"

    compat_args = mcmc_args.copy()
    compat_args["output_format"] = "hbmcmc"
    compat_args["paris_postprocessing"] = True

    explicit = fixedbasisMCMC(**explicit_args)
    with pytest.warns(UserWarning, match="Use `output_format = 'paris'` instead"):
        compat = fixedbasisMCMC(**compat_args)

    assert set(explicit.data_vars) == set(compat.data_vars)
    assert explicit.sizes == compat.sizes
    assert explicit["Yobs"].dims == compat["Yobs"].dims
    assert explicit["Yapost"].dims == compat["Yapost"].dims


def test_hbmcmc_postprocessing_preserves_expected_vars_attrs_and_coords(mcmc_args, tmpdir):
    mcmc_args["output_format"] = "hbmcmc_postprocessing"
    mcmc_args["outputpath"] = str(tmpdir)

    outputs = fixedbasisMCMC(**mcmc_args)

    expected_vars = [
        "Yobs",
        "Yerror",
        "Yerror_repeatability",
        "Yerror_variability",
        "Yapriori",
        "Ymod68",
        "country68",
        "fluxapriori",
        "basisfunctions",
    ]
    for var_name in expected_vars:
        assert var_name in outputs
        assert "longname" in outputs[var_name].attrs

    assert outputs["Yobs"].dims == ("nmeasure",)
    assert outputs["Ymod68"].dims == ("nmeasure", "nUI")
    assert outputs["country68"].dims == ("countrynames", "nUI")
    assert "UInum" in outputs.coords
    assert "countrynames" in outputs.coords


def test_inv_out_and_trace_outputs_preserve_downstream_dims_and_custom_paths(mcmc_args, tmpdir):
    trace_path = Path(tmpdir) / "custom_trace.nc"
    inv_out_path = Path(tmpdir) / "custom_inv_out.nc"
    mcmc_args["output_format"] = "inv_out"
    mcmc_args["save_trace"] = str(trace_path)
    mcmc_args["save_inversion_output"] = str(inv_out_path)

    inv_out = fixedbasisMCMC(**mcmc_args)

    assert trace_path.exists()
    assert inv_out_path.exists()
    assert inv_out.obs.dims == ("nmeasure",)
    assert inv_out.obs_err.dims == ("nmeasure",)
    assert inv_out.trace_ds["x_posterior"].dims == ("draw", "nx")
    assert "site" in inv_out.obs.coords
    assert "time" in inv_out.obs.coords
    assert "time" not in inv_out.flux.dims
    if "flux_time" in inv_out.flux.coords:
        assert "flux_time" in inv_out.flux.dims


def test_inversion_output_construction_and_explicit_predictive_sampling(mcmc_args):
    mcmc_args["output_format"] = "inv_out"
    base_inv_out = fixedbasisMCMC(**mcmc_args)

    posterior_only_trace = az.InferenceData(
        **{
            group: getattr(base_inv_out.trace, group)
            for group in ("posterior", "sample_stats", "observed_data", "constant_data")
            if group in base_inv_out.trace
        }
    )
    inv_out = InversionOutput(
        obs=base_inv_out.obs.reset_index("nmeasure", drop=True),
        obs_err=base_inv_out.obs_err.reset_index("nmeasure", drop=True),
        obs_prior_factor=(
            base_inv_out.obs_prior_factor.reset_index("nmeasure", drop=True)
            if base_inv_out.obs_prior_factor is not None
            else None
        ),
        obs_prior_upper_level_factor=(
            base_inv_out.obs_prior_upper_level_factor.reset_index("nmeasure", drop=True)
            if base_inv_out.obs_prior_upper_level_factor is not None
            else None
        ),
        obs_repeatability=base_inv_out.obs_repeatability.reset_index("nmeasure", drop=True),
        obs_variability=base_inv_out.obs_variability.reset_index("nmeasure", drop=True),
        flux=base_inv_out.flux.squeeze(drop=True),
        basis=base_inv_out.basis,
        trace=posterior_only_trace,
        site_indicators=base_inv_out.site_indicators.reset_index("nmeasure", drop=True),
        times=base_inv_out.times.reset_index("nmeasure", drop=True),
        start_date=base_inv_out.start_date,
        end_date=base_inv_out.end_date,
        species=base_inv_out.species,
        domain=base_inv_out.domain,
        site_names=base_inv_out.site_names,
    )

    assert "prior" not in inv_out.trace
    assert "posterior_predictive" not in inv_out.trace
    assert {
        "y_obs",
        "y_obs_error",
        "y_obs_repeatability",
        "y_obs_variability",
    }.issubset(inv_out.obs_inputs.data_vars)

    with pytest.warns(UserWarning, match="no longer samples predictive distributions"):
        inv_out.sample_predictive_distributions()

    assert "prior" not in inv_out.trace
    assert "prior_predictive" not in inv_out.trace
    assert "posterior_predictive" not in inv_out.trace


def test_hbmcmc_postprocessing_output_matches_legacy_core_fields(raw_data_path, europe_country_file):
    """Regression test for deterministic prior concentration terms in legacy-compatible output."""
    with xr.open_dataset(raw_data_path / "standard_rhime_outs.nc") as legacy:
        inv_out = InversionOutput.load(raw_data_path / "inversion_output.nc")
        compat = make_legacy_hbmcmc_output(
            inv_out=inv_out,
            mcmc_results={
                "xouts": legacy["xtrace"],
                "sigouts": legacy["sigtrace"],
                "bcouts": legacy["bctrace"],
            },
            sigma_freq_index=legacy["sigmafreqindex"].values,
            Hx=legacy["xsensitivity"].values.T,
            Hbc=legacy["bcsensitivity"].values.T,
            country_file=europe_country_file,
            use_bc=True,
        )

        assert (compat["Yapriori"].values == legacy["Yapriori"].values).all()
        assert (compat["YaprioriBC"].values == legacy["YaprioriBC"].values).all()
        assert (compat["Yobs"].values == legacy["Yobs"].values).all()

        xr.testing.assert_allclose(compat["fluxapriori"], legacy["fluxapriori"], rtol=1e-6, atol=1e-9)

        assert compat["Yapriori"].dims == legacy["Yapriori"].dims == ("nmeasure",)
        assert compat["YaprioriBC"].dims == legacy["YaprioriBC"].dims == ("nmeasure",)
        assert compat["Yobs"].dims == legacy["Yobs"].dims == ("nmeasure",)
        assert compat["Ymod68"].dims == legacy["Ymod68"].dims == ("nmeasure", "nUI")
        assert compat["country68"].dims == legacy["country68"].dims == ("countrynames", "nUI")
        assert compat["countrymean"].dims == legacy["countrymean"].dims == ("countrynames",)
        assert compat.sizes["nmeasure"] == legacy.sizes["nmeasure"]
        assert compat["Yobs"].sizes["nmeasure"] == compat["Yapriori"].sizes["nmeasure"]
        assert compat["Yapriori"].attrs["units"] == legacy["Yapriori"].attrs["units"]
        assert compat["YaprioriBC"].attrs["units"] == legacy["YaprioriBC"].attrs["units"]
        assert compat["Yobs"].attrs["units"] == legacy["Yobs"].attrs["units"]
        assert "UInum" in compat.coords
        assert "countrynames" in compat.coords

        for dv in compat.data_vars:
            assert "longname" in compat[dv].attrs
