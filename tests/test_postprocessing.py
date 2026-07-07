import pytest
import arviz as az
import numpy as np
import xarray as xr

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC
from openghg_inversions.postprocessing.inversion_output import InversionOutput, make_inv_out_for_fixed_basis_mcmc
from openghg_inversions.postprocessing.make_outputs import basic_output, make_flux_outputs
from openghg_inversions.postprocessing.make_paris_outputs import (
    _country_flux_uncertainty_diagnostics,
    _fine_coord_over_outer_extent,
    make_paris_flux_outputs_from_rhime,
    make_paris_outputs,
    nested_inner_domain_label,
    paris_flux_output,
)


class _FluxWrapper:
    def __init__(self, data: xr.DataArray):
        self.data = data


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


def test_country_flux_uncertainty_diagnostics_adds_summary_attr(capsys):
    time = np.array([0.0, 31.0])
    country = np.array(["GBR", "DEU"])
    percentile = np.array([0.159, 0.841])
    posterior = xr.DataArray(
        [[10.0, 20.0], [12.0, 18.0]],
        dims=["time", "country"],
        coords={"time": time, "country": country},
    )
    prior = xr.DataArray(
        [[11.0, 21.0], [11.0, 21.0]],
        dims=["time", "country"],
        coords={"time": time, "country": country},
    )
    posterior_percentile = xr.concat([posterior * 0.5, posterior * 1.5], dim="percentile").assign_coords(
        percentile=percentile
    )
    prior_percentile = xr.concat([prior * 0.7, prior * 1.3], dim="percentile").assign_coords(
        percentile=percentile
    )
    ds = xr.Dataset(
        {
            "country_flux_total_posterior": posterior,
            "country_flux_total_prior": prior,
            "percentile_country_flux_total_posterior": posterior_percentile,
            "percentile_country_flux_total_prior": prior_percentile,
        }
    )

    result = _country_flux_uncertainty_diagnostics(ds, label="outer")

    captured = capsys.readouterr()
    assert "DIAGNOSTIC country_flux_uncertainty" in captured.out
    assert "outer_country_flux_uncertainty_diagnostic" in result.attrs
    assert "percentile_order_violations=0" in result.attrs["outer_country_flux_uncertainty_diagnostic"]


def test_fine_coord_over_outer_extent_uses_inner_resolution():
    outer = xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["lat"], name="lat")
    inner = xr.DataArray(np.array([0.25, 0.75, 1.25]), dims=["lat"], name="lat")

    result = _fine_coord_over_outer_extent(outer, inner)

    np.testing.assert_allclose(result.values, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))


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

    flux_outs, inner_flux_outs, conc_outs = make_paris_outputs(
        inv_out, country_file=europe_country_file, obs_avg_period="1h", domain="europe"
    )

    assert inner_flux_outs is None

    if offset:
        assert "Yapriori_bias" in conc_outs

    assert conc_outs.attrs["prior_factor_adjustment_applied"] in {"true", "false"}

    if "Yobs_prior_factor" in conc_outs and "Yobs_prior_upper_level_factor" in conc_outs:
        for name in ["Yobs", "Yapost", "Yapriori", "qYapost", "qYapriori"]:
            assert f"{name}_modeled" in conc_outs

        factor = conc_outs["Yobs_prior_factor"] + conc_outs["Yobs_prior_upper_level_factor"]
        xr.testing.assert_allclose(conc_outs["Yapriori"] - conc_outs["Yapriori_modeled"], factor)

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


def test_nested_make_inv_out_uses_flux_grid_mask(tmp_path) -> None:
    """Ensure PARIS nested mask is evaluated on flux grid when basis/flux grids differ."""
    country_file = tmp_path / "country_toy.nc"
    xr.Dataset(
        {
            "country": (
                ["lat", "lon"],
                np.ones((2, 2), dtype=np.int16),
            ),
            "name": (["name"], np.array(["OCEAN", "France"])),
        },
        coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    ).to_netcdf(country_file, engine="h5netcdf")

    outer_basis = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=int),
        dims=["lat", "lon"],
        coords={"lat": [10.0, 11.0], "lon": [10.0, 11.0]},
    )
    inner_basis = xr.DataArray(
        np.ones((3, 3), dtype=int),
        dims=["lat", "lon"],
        coords={"lat": [10.0, 10.5, 11.0], "lon": [10.0, 10.5, 11.0]},
    )

    outer_flux = xr.DataArray(
        10.0 * np.ones((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": [np.datetime64("2022-01-01")], "lat": [0.0, 0.5], "lon": [0.0, 0.5]},
    )
    inner_flux = xr.DataArray(
        np.array([[[20.0, 25.0, 30.0], [30.0, 35.0, 40.0], [40.0, 45.0, 50.0]]]),
        dims=["time", "lat", "lon"],
        coords={"time": [np.datetime64("2022-01-01")], "lat": [0.0, 0.25, 0.5], "lon": [0.0, 0.25, 0.5]},
    )

    inner_fp = xr.DataArray(
        np.array([[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]]),
        dims=["time", "lat", "lon"],
        coords={"time": [np.datetime64("2022-01-01")], "lat": [0.0, 0.25, 0.5], "lon": [0.0, 0.25, 0.5]},
    )

    mf = xr.DataArray(
        [1800.0],
        dims=["time"],
        coords={"time": [np.datetime64("2022-01-01")]},
        attrs={"units": "ppb", "long_name": "observed_mole_fraction"},
    )
    site_ds_inner = xr.Dataset(
        {
            "fp": inner_fp,
            "mf": mf,
            "mf_error": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
            "mf_variability": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
            "mf_repeatability": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
        }
    )
    site_ds_standard = xr.Dataset(
        {
            "mf": mf,
            "mf_error": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
            "mf_variability": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
            "mf_repeatability": xr.DataArray([1.0], dims=["time"], coords={"time": mf.time}),
        }
    )

    site_tree = xr.DataTree.from_dict({"inner": site_ds_inner, "standard": site_ds_standard})

    trace = az.InferenceData(
        posterior=xr.Dataset(
            {
                "x": xr.DataArray(np.ones((2, 4)), dims=["draw", "nx"]),
                "x_inner": xr.DataArray(np.ones((2, 1)), dims=["draw", "nx_inner"]),
            }
        ),
        prior=xr.Dataset(
            {
                "x": xr.DataArray(np.ones((2, 4)), dims=["draw", "nx"]),
                "x_inner": xr.DataArray(np.ones((2, 1)), dims=["draw", "nx_inner"]),
            }
        ),
    )

    fp_data = {
        "SITE1": site_tree,
        ".basis": outer_basis,
        ".basis_inner": inner_basis,
        ".flux": {"SITE1": _FluxWrapper(outer_flux)},
        ".inner_flux": {"SITE1": _FluxWrapper(inner_flux)},
    }

    inv_out = make_inv_out_for_fixed_basis_mcmc(
        fp_data=fp_data,
        Y=np.array([1800.0]),
        Ytime=np.array([np.datetime64("2022-01-01")]),
        error=np.array([1.0]),
        obs_repeatability=np.array([1.0]),
        obs_variability=np.array([1.0]),
        site_indicator=np.array([0]),
        site_names=np.array(["SITE1"]),
        mcmc_results={
            "xouts_inner": np.ones((2, 1)),
            "xouts": np.ones((2, 4)),
            "trace": trace,
            "model": None,
        },
        start_date="2022-01-01",
        end_date="2022-01-01",
        species="ch4",
        domain="toy",
    )

    assert inv_out.flux.sizes["lat"] == 2
    assert inv_out.flux.sizes["lon"] == 2
    assert bool(inv_out.flux.notnull().all())

    assert inv_out.flux.isel(flux_time=0).sel(lat=0.5, lon=0.5).item() == 0.0
    assert inv_out.flux.isel(flux_time=0).sel(lat=0.0, lon=0.0).item() == 0.0

    flux_outs, inner_flux_outs = paris_flux_output(inv_out, country_file=country_file, inner_domain="6km")

    assert inner_flux_outs is not None
    assert nested_inner_domain_label("EUROPE", "6km") == "europe-6km"
    assert inner_flux_outs.attrs["domain"] == "toy-6km"
    assert inner_flux_outs.attrs["inner_domain"] == "toy-6km"
    assert inner_flux_outs.attrs["spatial_resolution"] == "toy-6km"
    assert flux_outs.sizes["country"] == 2
    assert inner_flux_outs.sizes["country"] == 2
    assert flux_outs.sizes["latitude"] == 2
    assert flux_outs.sizes["longitude"] == 2
    assert inner_flux_outs.sizes["latitude"] == 3
    assert inner_flux_outs.sizes["longitude"] == 3
    assert bool(np.isfinite(flux_outs["flux_total_posterior"]).all())
    assert bool(np.isfinite(inner_flux_outs["flux_total_posterior"]).all())
    assert bool(np.isfinite(flux_outs["country_flux_total_posterior"]).all())
    assert bool(np.isfinite(inner_flux_outs["country_flux_total_posterior"]).all())
    assert flux_outs["flux_total_posterior"].isel(time=0).sel(latitude=0.0, longitude=0.0).item() == 20.0
    assert flux_outs["flux_total_posterior"].isel(time=0).sel(latitude=0.5, longitude=0.5).item() == 50.0
    xr.testing.assert_allclose(
        inner_flux_outs["flux_total_posterior_inversion_grid"],
        inner_flux_outs["flux_total_posterior"],
    )


def test_flux_output_drops_singleton_basis_time_before_monthly_flux_alignment() -> None:
    flux = xr.DataArray(
        np.array([[[5.0]], [[7.0]]]),
        dims=["time", "lat", "lon"],
        coords={
            "time": [np.datetime64("2022-02-01"), np.datetime64("2022-03-01")],
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    basis = xr.DataArray(
        np.ones((1, 1, 1, 1), dtype=float),
        dims=["time", "nx", "lat", "lon"],
        coords={
            "time": [np.datetime64("2022-01-01")],
            "nx": [1],
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    trace = az.InferenceData(
        posterior=xr.Dataset({"x": xr.DataArray([[2.0]], dims=["draw", "nx"], coords={"nx": [1]})}),
        prior=xr.Dataset({"x": xr.DataArray([[1.0]], dims=["draw", "nx"], coords={"nx": [1]})}),
    )
    inv_out = InversionOutput(
        obs=xr.DataArray(
            [1800.0],
            dims=["nmeasure"],
            attrs={"units": "ppb", "long_name": "observed_mole_fraction"},
        ),
        obs_err=xr.DataArray([1.0], dims=["nmeasure"]),
        obs_repeatability=xr.DataArray([1.0], dims=["nmeasure"]),
        obs_variability=xr.DataArray([1.0], dims=["nmeasure"]),
        flux=flux,
        basis=basis,
        trace=trace,
        site_indicators=xr.DataArray([0], dims=["nmeasure"]),
        times=xr.DataArray([np.datetime64("2022-02-01")], dims=["nmeasure"]),
        start_date="2022-02-01",
        end_date="2022-04-01",
        species="ch4",
        domain="EUROPE",
    )

    flux_outs = make_flux_outputs(
        inv_out,
        stats=["mean"],
        report_flux_on_inversion_grid=False,
        include_scale_factors=False,
    )

    assert flux_outs.sizes["flux_time"] == 2
    assert bool(np.isfinite(flux_outs["flux_posterior_mean"]).all())
    np.testing.assert_allclose(flux_outs["flux_posterior_mean"].values[:, 0, 0], [10.0, 14.0])
