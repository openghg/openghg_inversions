import inspect
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import openghg_inversions.hbmcmc.hbmcmc as hbmcmc_module
from openghg_inversions.basis.basis_functions import BASIS_ARTIFACT_SOURCE_ATTR, BasisFunctions
from openghg_inversions.hbmcmc.hbmcmc import _resolve_output_format, fixedbasisMCMC
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    basic_output,
    make_country_outputs,
    make_flux_outputs,
    observation_inputs_for_outputs,
)

from openghg_inversions.postprocessing.make_paris_outputs import (
    _flux_interval_midpoints,
    make_paris_outputs,
    paris_flux_output,
)


def _minimal_fixedbasis_inv_inputs() -> xr.Dataset:
    """Build the smallest fixedbasis inv_inputs dataset used by contract tests."""
    ds = xr.Dataset(
        data_vars={
            "H": (("region", "nmeasure"), np.array([[0.25]], dtype="float64")),
            "mf": (("nmeasure",), np.array([1900.0], dtype="float64")),
            "mf_error": (("nmeasure",), np.array([2.0], dtype="float64")),
            "mf_repeatability": (("nmeasure",), np.array([1.0], dtype="float64")),
            "mf_variability": (("nmeasure",), np.array([1.5], dtype="float64")),
            "site_indicator": (("nmeasure",), np.array([0])),
            "sigma_freq_index": (("nmeasure",), np.array([0])),
            "min_error": (("nmeasure",), np.array([0.0], dtype="float64")),
        },
        coords={
            "region": np.array([0]),
            "nmeasure": np.array([0]),
            "site": (("nmeasure",), np.array(["TAC"])),
            "time": (("nmeasure",), np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")),
        },
    )
    return ds.set_index(nmeasure=["site", "time"])


def _minimal_fixedbasis_fp_data() -> dict:
    """Build fixedbasis fp_data with the legacy side-channel keys still required downstream."""
    lat = np.array([52.0])
    lon = np.array([1.0])
    return {
        ".basis": xr.DataArray(
            np.array([[0]]),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name="basis",
        ),
        ".flux": {
            "total": xr.Dataset(
                {"flux": (("lat", "lon"), np.array([[1.0]], dtype="float32"))},
                coords={"lat": lat, "lon": lon},
            )
        },
        "TAC": xr.Dataset(
            {
                "mf": ("time", np.array([1900.0], dtype="float64"), {"units": "1e-09 mol/mol"}),
                "mf_error": ("time", np.array([2.0], dtype="float64"), {"units": "1e-09 mol/mol"}),
                "mf_repeatability": ("time", np.array([1.0], dtype="float64"), {"units": "1e-09 mol/mol"}),
                "mf_variability": ("time", np.array([1.5], dtype="float64"), {"units": "1e-09 mol/mol"}),
            },
            coords={"time": np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")},
        ),
    }


def _minimal_fixedbasis_basis_functions() -> BasisFunctions:
    """Build retained basis functions matching the minimal fixedbasis fixture."""
    fp_data = _minimal_fixedbasis_fp_data()
    basis = fp_data[".basis"] + 1
    flux = fp_data[".flux"]["total"].flux
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: "test"},
    )


def _minimal_fixedbasis_prepared_data(**overrides):
    """Build a prepared-data object using the fixedbasis preparation contract exported by hbmcmc."""
    defaults = {
        "fp_all": {"TAC": object(), ".flux": {"total": object()}},
        "fp_data": _minimal_fixedbasis_fp_data(),
        "inv_inputs": _minimal_fixedbasis_inv_inputs(),
        "sites": ["TAC"],
        "averaging_period": ["1H"],
        "basis_objects": {"emissions": _minimal_fixedbasis_basis_functions()},
    }
    defaults.update(overrides)
    return hbmcmc_module.FixedBasisPreparedData(**defaults)


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
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "merged_data_name": merged_data_file_name,
            "bc_basis_directory": default_bc_basis_directory,
            "country_file": europe_country_file,
            "nuts_sampler": "numpyro",
        }
    )
    return mcmc_args


@pytest.fixture
def slow_mcmc_args(mcmc_args):
    """Restore the higher-draw postprocessing sampler settings for slow tests."""
    mcmc_args = mcmc_args.copy()
    mcmc_args.update({"nit": 100, "nchain": 2})
    return mcmc_args


@pytest.fixture
def inv_out(mcmc_args):
    """Return a modern fixedbasis inversion output for postprocessing tests."""
    mcmc_args["output_format"] = "inv_out"
    result = fixedbasisMCMC(**mcmc_args)
    assert isinstance(result, InversionOutput)
    return result


def test_fixedbasisMCMC_return_basis_objects_preserves_positional_output_format():
    """New retained-basis option should not shift existing positional API."""
    params = list(inspect.signature(fixedbasisMCMC).parameters)

    assert params.index("return_basis_objects") > params.index("power")
    assert params.index("return_basis_objects") > params.index("output_format")
    assert params.index("return_basis_objects") == params.index("kwargs") - 1


def test_fixedbasisMCMC_can_return_basis_objects_in_mcmc_args(mcmc_args):
    """Retained basis objects are opt-in debug output, not inferpymc inputs."""
    mcmc_args["output_format"] = "mcmc_args"
    mcmc_args["return_basis_objects"] = True

    result = fixedbasisMCMC(**mcmc_args)

    assert isinstance(result, dict)
    assert isinstance(result["basis_objects"]["emissions"], BasisFunctions)
    assert "basis_objects" not in result["inv_inputs"]


def test_fixedbasisMCMC_uses_fixedbasis_preparation_contract_for_mcmc_args(monkeypatch, tmp_path):
    """The fixedbasis runner consumes the fixedbasis-specific preparation boundary."""
    prepared = _minimal_fixedbasis_prepared_data(fp_data={})
    captured_kwargs = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        captured_kwargs.update(kwargs)
        return prepared

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )

    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="contract",
        output_format="mcmc_args",
        return_basis_objects=True,
        use_bc=False,
    )

    assert captured_kwargs["output_name"] == "contract"
    assert captured_kwargs["split_by_sectors"] is False
    assert captured_kwargs["return_basis_objects"] is True
    assert captured_kwargs["merged_data_only"] is False
    assert isinstance(result, dict)
    assert result["inv_inputs"] is prepared.inv_inputs
    assert result["basis_objects"] is prepared.basis_objects
    assert "basis_objects" not in result["inv_inputs"]


def test_fixedbasisMCMC_merged_data_returns_prepared_fp_all(monkeypatch, tmp_path):
    """The merged-data output remains a preparation-only compatibility path."""
    fp_all = {"TAC": object(), ".flux": {"total": object()}}
    prepared = _minimal_fixedbasis_prepared_data(fp_all=fp_all, fp_data=None, inv_inputs=None)
    captured_kwargs = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        captured_kwargs.update(kwargs)
        return prepared

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )

    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="merged",
        output_format="merged_data",
        reload_merged_data=True,
        use_bc=False,
    )

    assert result is fp_all
    assert captured_kwargs["merged_data_only"] is True
    assert captured_kwargs["reload_merged_data"] is False


def test_fixedbasisMCMC_hbmcmc_output_uses_legacy_postprocess_path(monkeypatch, tmp_path):
    """The default hbmcmc output path still consumes legacy postprocessing args."""
    prepared = _minimal_fixedbasis_prepared_data()
    captured_inferpymc_args = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        return prepared

    def fake_inferpymc(**kwargs):
        captured_inferpymc_args.update(kwargs)
        return {
            "trace": object(),
            "model": object(),
            "xouts": np.array([[1.0], [1.1]], dtype="float64"),
        }

    def fake_inferpymc_postprocessouts(xouts):
        return xr.Dataset({"xtrace_mean": (("nx",), xouts.mean(axis=0))})

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc_postprocessouts", fake_inferpymc_postprocessouts)

    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="legacy",
        output_format="hbmcmc",
        use_bc=False,
    )

    assert captured_inferpymc_args["inv_inputs"] is prepared.inv_inputs
    assert isinstance(result, xr.Dataset)
    assert result["xtrace_mean"].dims == ("nx",)
    assert result["xtrace_mean"].values.tolist() == [1.05]


def test_fixedbasisMCMC_inv_out_returns_modern_output_without_legacy_adapter(monkeypatch, tmp_path):
    """The fixedbasis inv_out path returns modern InversionOutput without legacy adapters."""
    prepared = _minimal_fixedbasis_prepared_data()

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        return prepared

    def fake_inferpymc(**kwargs):
        return {
            "trace": az.from_dict(
                posterior={"x": np.ones((1, 1, 1))},
                coords={"nx": [0]},
                dims={"x": ["nx"]},
            ),
            "model": object(),
            "xouts": np.array([[1.0]], dtype="float64"),
        }

    def fail_inferpymc_postprocessouts(**kwargs):
        raise AssertionError("output_format='inv_out' must not call inferpymc_postprocessouts")

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="inv-out",
        output_format="inv_out",
        use_bc=False,
    )

    assert isinstance(result, InversionOutput)
    assert result.inv_inputs is prepared.inv_inputs
    assert result.basis_functions is prepared.basis_objects["emissions"]


def test_fixedbasisMCMC_paris_postprocessing_receives_modern_output(monkeypatch, tmp_path):
    """Fixedbasis PARIS postprocessing uses modern InversionOutput internally."""
    prepared = _minimal_fixedbasis_prepared_data()
    captured = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        captured["prepare_kwargs"] = kwargs
        return prepared

    def fake_inferpymc(**kwargs):
        return {
            "trace": az.from_dict(
                posterior={
                    "x": np.ones((1, 1, 1)),
                    "y": np.ones((1, 1, 1)),
                    "epsilon": np.ones((1, 1, 1)),
                },
                prior={
                    "x": np.ones((1, 1, 1)),
                    "y": np.ones((1, 1, 1)),
                    "epsilon": np.ones((1, 1, 1)),
                },
                coords={"nx": [0], "nmeasure": [0]},
                dims={"x": ["nx"], "y": ["nmeasure"], "epsilon": ["nmeasure"]},
            ),
            "model": object(),
            "xouts": np.array([[1.0]], dtype="float64"),
        }

    def fake_make_paris_outputs(inv_out, **kwargs):
        captured["inv_out"] = inv_out
        captured["paris_kwargs"] = kwargs
        return (
            xr.Dataset({"flux_total_posterior": ("time", np.array([1.0]))}, coords={"time": [0.0]}),
            xr.Dataset({"Yobs": ("time", np.array([1900.0]))}, coords={"time": [0.0]}),
        )

    monkeypatch.setattr(
        hbmcmc_module, "prepare_fixedbasis_inversion_data", fake_prepare_fixedbasis_inversion_data
    )
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.make_paris_outputs.make_paris_outputs",
        fake_make_paris_outputs,
    )
    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="paris-modern",
        output_format="paris",
        use_bc=False,
    )

    assert captured["prepare_kwargs"]["return_basis_objects"] is True
    assert isinstance(captured["inv_out"], InversionOutput)
    assert captured["inv_out"].basis_functions is prepared.basis_objects["emissions"]
    assert isinstance(result, xr.Dataset)
    assert "Yobs" in result


def test_fixedbasisMCMC_basic_postprocessing_receives_modern_output(monkeypatch, tmp_path):
    """Fixedbasis basic postprocessing uses modern InversionOutput internally."""
    prepared = _minimal_fixedbasis_prepared_data()
    captured = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        captured["prepare_kwargs"] = kwargs
        return prepared

    def fake_inferpymc(**kwargs):
        return {
            "trace": az.from_dict(
                posterior={
                    "x": np.ones((1, 1, 1)),
                    "y": np.ones((1, 1, 1)),
                    "epsilon": np.ones((1, 1, 1)),
                },
                prior={
                    "x": np.ones((1, 1, 1)),
                    "y": np.ones((1, 1, 1)),
                    "epsilon": np.ones((1, 1, 1)),
                },
                coords={"nx": [0], "nmeasure": [0]},
                dims={"x": ["nx"], "y": ["nmeasure"], "epsilon": ["nmeasure"]},
            ),
            "model": object(),
            "xouts": np.array([[1.0]], dtype="float64"),
        }

    def fake_basic_output(inv_out, country_file=None):
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        return xr.Dataset({"ok": ((), 1)})

    monkeypatch.setattr(
        hbmcmc_module, "prepare_fixedbasis_inversion_data", fake_prepare_fixedbasis_inversion_data
    )
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    monkeypatch.setattr("openghg_inversions.postprocessing.make_outputs.basic_output", fake_basic_output)

    result = fixedbasisMCMC(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        outputpath=str(tmp_path),
        outputname="basic-modern",
        output_format="basic",
        use_bc=False,
    )

    assert captured["prepare_kwargs"]["return_basis_objects"] is True
    assert isinstance(captured["inv_out"], InversionOutput)
    assert isinstance(result, xr.Dataset)
    assert result["ok"].item() == 1


@pytest.mark.parametrize("missing_key", [".basis", ".flux"])
def test_fixedbasisMCMC_requires_legacy_fixedbasis_fp_data(monkeypatch, tmp_path, missing_key):
    """Postprocessed fixedbasis outputs require the legacy fp_data side-channel keys."""
    fp_data = _minimal_fixedbasis_fp_data()
    del fp_data[missing_key]
    prepared = _minimal_fixedbasis_prepared_data(fp_data=fp_data)

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        return prepared

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )

    with pytest.raises(RuntimeError, match="legacy fixed-basis data"):
        fixedbasisMCMC(
            species="ch4",
            sites=["TAC"],
            domain="EUROPE",
            averaging_period=["1H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            outputpath=str(tmp_path),
            outputname="missing",
            output_format="hbmcmc",
            use_bc=False,
        )


@pytest.mark.parametrize(
    "flux_times, flux_period, inv_start, inv_end, expected",
    [
        # Monthly inversion, monthly flux: overlap = full month, midpoint = mid-month
        (
            [pd.Timestamp("2019-02-01")],
            pd.DateOffset(months=1),
            pd.Timestamp("2019-02-01"),
            pd.Timestamp("2019-03-01"),
            [pd.Timestamp("2019-02-01") + (pd.Timestamp("2019-03-01") - pd.Timestamp("2019-02-01")) / 2],
        ),
        # 3-monthly inversion, yearly flux: yearly interval clipped to Jan-Apr,
        # so midpoint is mid-Feb, not mid-year (Jul)
        (
            [pd.Timestamp("2019-01-01")],
            pd.DateOffset(years=1),
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-04-01"),
            [pd.Timestamp("2019-01-01") + (pd.Timestamp("2019-04-01") - pd.Timestamp("2019-01-01")) / 2],
        ),
        # 3-monthly inversion, yearly flux, flux starts before inversion: the flux
        # time (Jan) differs from the inversion start (Feb), as in the original bug.
        # The overlap is clipped to Feb-May, so the midpoint is still mid-March,
        # not mid-year (Jul) and not mid-January.
        (
            [pd.Timestamp("2019-01-01")],
            pd.DateOffset(years=1),
            pd.Timestamp("2019-02-01"),
            pd.Timestamp("2019-05-01"),
            [pd.Timestamp("2019-02-01") + (pd.Timestamp("2019-05-01") - pd.Timestamp("2019-02-01")) / 2],
        ),
        # 3-monthly inversion, monthly flux: three flux steps each fully within
        # the inversion period, so each midpoint is the middle of its own month
        (
            [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01"), pd.Timestamp("2019-03-01")],
            pd.DateOffset(months=1),
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-04-01"),
            [
                pd.Timestamp("2019-01-01") + (pd.Timestamp("2019-02-01") - pd.Timestamp("2019-01-01")) / 2,
                pd.Timestamp("2019-02-01") + (pd.Timestamp("2019-03-01") - pd.Timestamp("2019-02-01")) / 2,
                pd.Timestamp("2019-03-01") + (pd.Timestamp("2019-04-01") - pd.Timestamp("2019-03-01")) / 2,
            ],
        ),
        # 2-yearly inversion, yearly flux: two flux steps, each fully within
        # inversion period, so midpoints are mid-2019 and mid-2020
        (
            [pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")],
            pd.DateOffset(years=1),
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2021-01-01"),
            [
                pd.Timestamp("2019-01-01") + (pd.Timestamp("2020-01-01") - pd.Timestamp("2019-01-01")) / 2,
                pd.Timestamp("2020-01-01") + (pd.Timestamp("2021-01-01") - pd.Timestamp("2020-01-01")) / 2,
            ],
        ),
    ],
)
def test_flux_interval_midpoints(flux_times, flux_period, inv_start, inv_end, expected):
    """Check midpoint timestamps are computed from the flux/inversion period overlap."""
    midpoints, valid_indices = _flux_interval_midpoints(flux_times, flux_period, inv_start, inv_end)
    assert midpoints == expected
    # Also verify that valid_indices are correct (0-indexed positions in flux_times
    # of the flux periods that overlap the inversion period)
    assert len(valid_indices) == len(expected)
    assert valid_indices == list(range(len(expected)))  # For these test cases, all flux times have overlap


def test_flux_interval_midpoints_with_non_overlapping_times():
    """Test that non-overlapping flux times are correctly filtered out.

    This test verifies the fix for the bug where all 13 flux times (2012-2024)
    were being written to output even when the inversion period was only 2023-2024.
    """
    # Flux times spanning 2012-2024 (yearly intervals)
    flux_times = [pd.Timestamp(f"{year}-01-01") for year in range(2012, 2025)]  # 13 times
    flux_period = pd.DateOffset(years=1)
    inv_start = pd.Timestamp("2023-01-01")
    inv_end = pd.Timestamp("2024-01-01")

    midpoints, valid_indices = _flux_interval_midpoints(flux_times, flux_period, inv_start, inv_end)

    # Only the 2023 flux interval (index 11) overlaps with the inversion period
    assert len(midpoints) == 1
    assert len(valid_indices) == 1
    assert valid_indices[0] == 11  # 2023 is at index 11 (year 2012 = 0, ..., 2023 = 11)

    # The midpoint should be the midpoint of 2023-01-01 to 2024-01-01
    expected_midpoint = (
        pd.Timestamp("2023-01-01") + (pd.Timestamp("2024-01-01") - pd.Timestamp("2023-01-01")) / 2
    )
    assert midpoints[0] == expected_midpoint


def test_paris_flux_output_timestamp(inv_out, europe_country_file):
    """Check that the flux output time coordinate is the midpoint of the inversion period.

    The flux file has a yearly period but the inversion is shorter; the output
    timestamp should be the midpoint of the overlap between the flux interval
    and the inversion period (i.e. the midpoint of the inversion period itself),
    not 6 months into the flux's own year.
    """
    flux_outs = paris_flux_output(inv_out, country_file=europe_country_file, flux_frequency="yearly")

    # time is stored as days since Unix epoch; convert back for comparison
    actual = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(flux_outs.time.values[0]))
    expected = inv_out.period_midpoint

    assert actual == expected


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


def test_fixedbasis_flux_and_country_outputs_use_modern_basis_functions(inv_out, europe_country_file):
    """Fixedbasis postprocessing reconstructs products from retained basis functions."""
    flux_outs = make_flux_outputs(
        inv_out,
        include_scale_factors=False,
        report_flux_on_inversion_grid=False,
    )
    country_outs = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")

    assert "flux_posterior_mean" in flux_outs
    assert "country_posterior_mean" in country_outs


@pytest.mark.parametrize("offset", [False, True])
def test_make_paris_outputs(inv_out, europe_country_file, tmpdir, offset):
    """Check that we can create and save PARIS outputs for EUROPE domain"""

    if offset:
        # fake an offset trace
        inv_out.trace.posterior["offset"] = xr.ones_like(inv_out.trace.posterior["mu_bc"])
        inv_out.trace.prior["offset"] = xr.ones_like(inv_out.trace.prior["mu_bc"])

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

    assert isinstance(inv_out, InversionOutput)
    inv_out_reloaded = InversionOutput.load(tmpdir / "inv_out.nc")

    assert inv_out_reloaded.species == inv_out.species
    assert inv_out_reloaded.domain == inv_out.domain
    assert isinstance(inv_out_reloaded.basis_functions, BasisFunctions)
    xr.testing.assert_identical(inv_out_reloaded.inv_inputs, inv_out.inv_inputs)


def test_country_outputs_lognormal_reparam_conflict(mcmc_args, europe_country_file):
    """Check country outputs ignore reparameterized latent-only traces."""
    mcmc_args["output_format"] = "inv_out"
    mcmc_args["reparameterise_log_normal"] = True
    mcmc_args["xprior"] = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}

    inv_out = fixedbasisMCMC(**mcmc_args)
    assert isinstance(inv_out, InversionOutput)
    trace_ds = inv_out.trace_dataset(var_roles="flux_scale")
    assert "x_prior" in trace_ds
    assert "x_posterior" in trace_ds
    assert "x_latent_prior" not in trace_ds
    assert "x_latent_posterior" not in trace_ds

    country_outs = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")
    assert "country_prior_mean" in country_outs
    assert "country_posterior_mean" in country_outs


def test_hbmcmc_postprocessing_saves_legacy_output(mcmc_args, tmpdir):
    """Legacy postprocessing output can still be saved and reloaded."""
    mcmc_args["output_format"] = "hbmcmc_postprocessing"
    mcmc_args["outputpath"] = str(tmpdir)

    outputs = fixedbasisMCMC(**mcmc_args)
    assert isinstance(outputs, xr.Dataset)
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
    """The old PARIS compatibility switch resolves to the canonical output format."""
    with pytest.warns(UserWarning, match="Use `output_format = 'paris'` instead"):
        resolved = _resolve_output_format("hbmcmc", paris_postprocessing=True, is_column=False)

    assert resolved == "paris"


def test_paris_postprocessing_compatibility_matches_paris_output_format(mcmc_args):
    """Compatibility PARIS output matches the explicit canonical format."""
    explicit_args = mcmc_args.copy()
    explicit_args["output_format"] = "paris"

    compat_args = mcmc_args.copy()
    compat_args["output_format"] = "hbmcmc"
    compat_args["paris_postprocessing"] = True

    explicit = fixedbasisMCMC(**explicit_args)
    with pytest.warns(UserWarning, match="Use `output_format = 'paris'` instead"):
        compat = fixedbasisMCMC(**compat_args)

    assert isinstance(explicit, xr.Dataset)
    assert isinstance(compat, xr.Dataset)
    assert set(explicit.data_vars) == set(compat.data_vars)
    assert explicit.sizes == compat.sizes
    assert explicit["Yobs"].dims == compat["Yobs"].dims
    assert explicit["Yapost"].dims == compat["Yapost"].dims


def test_hbmcmc_postprocessing_preserves_expected_vars_attrs_and_coords(mcmc_args, tmpdir):
    """Legacy-style postprocessing keeps its core vars, attrs, and coords."""
    mcmc_args["output_format"] = "hbmcmc_postprocessing"
    mcmc_args["outputpath"] = str(tmpdir)

    outputs = fixedbasisMCMC(**mcmc_args)
    assert isinstance(outputs, xr.Dataset)

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
    """Saved trace and inversion output files preserve downstream-facing dims."""
    trace_path = Path(tmpdir) / "custom_trace.nc"
    inv_out_path = Path(tmpdir) / "custom_inv_out.nc"
    mcmc_args["output_format"] = "inv_out"
    mcmc_args["save_trace"] = str(trace_path)
    mcmc_args["save_inversion_output"] = str(inv_out_path)

    inv_out = fixedbasisMCMC(**mcmc_args)

    assert trace_path.exists()
    assert inv_out_path.exists()
    assert isinstance(inv_out, InversionOutput)
    assert isinstance(inv_out.basis_functions, BasisFunctions)
    obs_inputs = observation_inputs_for_outputs(inv_out)
    assert obs_inputs["y_obs"].dims == ("nmeasure",)
    assert obs_inputs["y_obs_error"].dims == ("nmeasure",)
    assert inv_out.trace_dataset(var_roles="flux_scale")["x_posterior"].dims == ("draw", "region")
    assert "site" in obs_inputs.coords
    assert "time" in obs_inputs.coords
    assert "time" not in inv_out.flux.dims
    if "flux_time" in inv_out.flux.coords:
        assert "flux_time" in inv_out.flux.dims
