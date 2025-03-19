import pytest
import xarray as xr

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import basic_output
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


@pytest.fixture
def inv_out(raw_data_path):
    return InversionOutput.load(raw_data_path / "inversion_output.nc")

@pytest.fixture
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


def test_make_paris_outputs(inv_out, europe_country_file, tmpdir):
    """Check that we can create and save PARIS outputs for EUROPE domain"""
    flux_outs, conc_outs = make_paris_outputs(inv_out, country_file=europe_country_file, obs_avg_period="1h", domain="europe")

    flux_outs.to_netcdf(tmpdir / "flux.nc")
    conc_outs.to_netcdf(tmpdir / "conc.nc")

def test_make_paris_outputs_eastasia(inv_out_eastasia, eastasia_country_file, tmpdir):
    """Check that we can create and save PARIS outputs for EASTASIA domain"""
    flux_outs, conc_outs = make_paris_outputs(inv_out_eastasia, country_file=eastasia_country_file, obs_avg_period="1h", domain="eastasia")

    flux_outs.to_netcdf(tmpdir / "flux.nc")
    conc_outs.to_netcdf(tmpdir / "conc.nc")

def test_save_inversion_output(mcmc_args, tmpdir):
    """Check that we can save and reload inversion outputs"""
    mcmc_args["save_inversion_output"] = str(tmpdir / "inv_out.nc")
    mcmc_args["output_format"] = "inv_out"
    inv_out = fixedbasisMCMC(**mcmc_args)

    inv_out_reloaded = InversionOutput.load(tmpdir / "inv_out.nc")

    assert inv_out == inv_out_reloaded
