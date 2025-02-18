import pytest
import xarray as xr

from openghg_inversions.postprocessing.make_paris_outputs import make_paris_flux_outputs_from_rhime


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


def test_rhime_flux_reprocessing(europe_country_file, raw_data_path):
    rhime_outs = xr.open_dataset(raw_data_path / "standard_rhime_outs.nc")
    paris_outs = make_paris_flux_outputs_from_rhime(rhime_outs, species="ch4", domain="europe", country_file=europe_country_file)

    assert "flux_total_prior" in paris_outs
    assert "flux_total_posterior" in paris_outs
