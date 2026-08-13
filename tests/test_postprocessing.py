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
from openghg_inversions.postprocessing import legacy_outputs
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    basic_output,
    make_country_outputs,
    make_flux_outputs,
    observation_inputs_for_outputs,
)
from openghg_inversions.postprocessing.make_paris_outputs import (
    DEFAULT_PARIS_TEMPLATE_VERSION,
    _assign_flux_time_bounds,
    _flux_interval_midpoints,
    infer_flux_frequency,
    make_paris_outputs,
    paris_concentration_outputs,
    paris_flux_output,
    paris_template_files,
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


def _deterministic_inferpymc_results(
    inv_inputs: xr.Dataset,
    *,
    use_bc: bool,
    reparameterise_log_normal: bool,
) -> dict[str, object]:
    """Build fresh realistic traces from prepared inputs without sampling.

    Args:
        inv_inputs: Fully prepared fixedbasis model inputs.
        use_bc: Whether boundary-condition variables should be included.
        reparameterise_log_normal: Whether to include the sampler's latent
            lognormal variable alongside the public scaling factor.

    Returns:
        A legacy ``inferpymc`` result mapping containing deterministic prior,
        posterior, and predictive traces suitable for real postprocessing and
        serialization.
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
        """Return independent deterministic results for each fixedbasis call."""
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
def inv_out(mcmc_args, deterministic_sampler):
    """Return a modern fixedbasis inversion output for postprocessing tests."""
    mcmc_args["output_format"] = "inv_out"
    result = fixedbasisMCMC(**mcmc_args)
    assert isinstance(result, InversionOutput)
    return result


def test_fixedbasisMCMC_return_basis_objects_preserves_positional_output_format():
    """New options should not shift the existing positional output API."""
    params = list(inspect.signature(fixedbasisMCMC).parameters)

    assert params.index("time_resolved") > params.index("power")
    assert params.index("time_resolved") > params.index("output_format")
    assert params.index("return_basis_objects") > params.index("power")
    assert params.index("return_basis_objects") > params.index("output_format")
    assert params.index("return_basis_objects") == params.index("time_resolved") + 1
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
        flux_non_finite_check="count",
        use_bc=False,
    )

    assert captured_kwargs["output_name"] == "contract"
    assert captured_kwargs["split_by_sectors"] is False
    assert captured_kwargs["return_basis_objects"] is True
    assert captured_kwargs["merged_data_only"] is False
    assert captured_kwargs["flux_non_finite_check"] == "count"
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


def test_fixedbasisMCMC_hbmcmc_output_uses_modern_legacy_formatter(monkeypatch, tmp_path):
    """Deprecated hbmcmc output aliases route through the modern legacy formatter."""
    prepared = _minimal_fixedbasis_prepared_data()
    captured_inferpymc_args = {}
    captured_formatter_args = {}

    def fake_prepare_fixedbasis_inversion_data(**kwargs):
        return prepared

    def fake_inferpymc(**kwargs):
        captured_inferpymc_args.update(kwargs)
        return {
            "trace": az.from_dict(
                posterior={"x": np.ones((1, 2, 1)), "sigma": np.ones((1, 2, 1, 1))},
                coords={"nx": [0], "nsigma_site": [0], "nsigma_time": [0]},
                dims={"x": ["nx"], "sigma": ["nsigma_site", "nsigma_time"]},
            ),
            "model": object(),
            "xouts": np.array([[1.0], [1.1]], dtype="float64"),
        }

    def fail_inferpymc_postprocessouts(**kwargs):
        raise AssertionError("output_format='legacy' must not call inferpymc_postprocessouts")

    def fake_make_legacy_hbmcmc_output(inv_out, country_file=None, use_bc=False):
        captured_formatter_args["inv_out"] = inv_out
        captured_formatter_args["country_file"] = country_file
        captured_formatter_args["use_bc"] = use_bc
        return xr.Dataset({"xtrace_mean": (("nx",), np.array([1.05]))})

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fake_prepare_fixedbasis_inversion_data,
    )
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", fake_inferpymc)
    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr(legacy_outputs, "make_legacy_hbmcmc_output", fake_make_legacy_hbmcmc_output)

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
    assert isinstance(captured_formatter_args["inv_out"], InversionOutput)
    assert captured_formatter_args["country_file"] is None
    assert captured_formatter_args["use_bc"] is False
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


def test_fixedbasisMCMC_satellite_paris_postprocessing_receives_modern_output(
    monkeypatch, tmp_path
):
    """Satellite PARIS postprocessing uses modern column-compatible output internally."""
    prepared = _minimal_fixedbasis_prepared_data(is_column=True)
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
        concentration = xr.Dataset(
            {
                "Yobs": ("time", np.array([1900.0])),
                "time_bnds": (("time", "nbnds"), np.array([[0.0, 1.0]])),
            },
            coords={"time": [0.5], "nbnds": [0, 1]},
        )
        concentration.time.attrs = {
            "bounds": "time_bnds",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
        concentration.time_bnds.attrs = {
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
        return (
            xr.Dataset({"flux_total_posterior": ("time", np.array([1.0]))}, coords={"time": [0.0]}),
            concentration,
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
        inlet="column",
        platform="satellite",
        time_resolved=True,
        use_bc=False,
    )

    assert captured["prepare_kwargs"]["return_basis_objects"] is True
    assert captured["prepare_kwargs"]["time_resolved"] is True
    assert captured["prepare_kwargs"]["platform"] == "satellite"
    assert isinstance(captured["inv_out"], InversionOutput)
    assert captured["inv_out"].basis_functions is prepared.basis_objects["emissions"]
    assert isinstance(result, xr.Dataset)
    assert "Yobs" in result
    concentration_path = next(tmp_path.glob("*_conc_*.nc"))
    with xr.open_dataset(concentration_path, decode_cf=False) as saved_concentration:
        assert saved_concentration.time_bnds.attrs["units"] == "days since 1970-01-01 00:00:00"
        assert saved_concentration.time_bnds.attrs["calendar"] == "proleptic_gregorian"


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


def test_paris_flux_interval_keeps_january_annual_prior_for_june_inversion():
    """PARIS retains and clips a January annual interval for June observations."""
    midpoints, valid_indices = _flux_interval_midpoints(
        [pd.Timestamp("2019-01-01")],
        pd.DateOffset(years=1),
        pd.Timestamp("2019-06-01"),
        pd.Timestamp("2019-07-01"),
    )

    assert midpoints == [pd.Timestamp("2019-06-16")]
    assert valid_indices == [0]


@pytest.mark.parametrize(
    ("time_period", "expected"),
    [
        ("annual", "yearly"),
        ("ANNUAL", "yearly"),
        ("1 Year", "yearly"),
        ("1 YEAR", "yearly"),
        ("monthly", "monthly"),
        ("MONTHLY", "monthly"),
        ("1 Month", "monthly"),
    ],
)
def test_infer_flux_frequency_normalizes_authoritative_period_spellings(time_period, expected):
    """Authoritative period spellings are normalized case-insensitively."""
    flux = xr.DataArray([1.0], dims="flux_time", attrs={"time_period": time_period})

    assert infer_flux_frequency(flux) == expected


def test_infer_flux_frequency_recognizes_calendar_annual_period_without_attrs():
    """January starts spanning a leap year identify an annual calendar period."""
    flux = xr.DataArray(
        np.ones(3),
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])},
    )

    assert infer_flux_frequency(flux) == "yearly"


def test_infer_flux_frequency_recognizes_unequal_calendar_month_period_without_attrs():
    """Unequal month-start gaps identify a monthly calendar period."""
    flux = xr.DataArray(
        np.ones(3),
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])},
    )

    assert infer_flux_frequency(flux) == "monthly"


def test_infer_flux_frequency_defaults_metadata_free_singleton_to_yearly():
    """A metadata-free singleton retains the historical yearly default."""
    flux = xr.DataArray(
        [1.0],
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2019-01-01"])},
    )

    assert infer_flux_frequency(flux) == "yearly"


@pytest.mark.parametrize("time_period", ["", "NaT", np.nan])
def test_infer_flux_frequency_treats_missing_period_attrs_as_absent(time_period):
    """Missing-valued period attributes fall back to calendar timestamps."""
    flux = xr.DataArray(
        np.ones(3),
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])},
        attrs={"time_period": time_period},
    )

    assert infer_flux_frequency(flux) == "yearly"


def test_infer_flux_frequency_preserves_positive_fixed_period():
    """A positive fixed period remains available to PARIS interval logic."""
    flux = xr.DataArray(
        [1.0],
        dims="flux_time",
        attrs={"time_period": "36h"},
    )

    assert infer_flux_frequency(flux) == "36h"


@pytest.mark.parametrize("time_period", ["2 years", "3 months", "0 days", "-1 day"])
def test_infer_flux_frequency_rejects_unsupported_or_nonpositive_period(time_period):
    """Unsupported calendar multiples and non-positive fixed periods are rejected."""
    flux = xr.DataArray(
        [1.0],
        dims="flux_time",
        attrs={"time_period": time_period},
    )

    with pytest.raises(ValueError, match="Flux period"):
        infer_flux_frequency(flux)


def test_assign_flux_time_bounds_reports_non_overlapping_period_bounds():
    """No overlapping period reports flux and inversion bounds with the frequency."""
    flux = xr.Dataset(coords={"time": pd.to_datetime(["2019-01-01"])})

    with pytest.raises(ValueError) as exc_info:
        _assign_flux_time_bounds(
            flux,
            flux_frequency="yearly",
            inv_start=pd.Timestamp("2021-01-01"),
            inv_end=pd.Timestamp("2021-02-01"),
        )

    message = str(exc_info.value).lower()
    assert "flux" in message
    assert "inversion" in message
    assert "frequency" in message
    assert "yearly" in message
    assert "2019-01-01" in message
    assert "2020-01-01" in message
    assert "2021-01-01" in message
    assert "2021-02-01" in message


def test_assign_flux_time_bounds_reports_empty_flux_times():
    """An empty flux coordinate raises the same contextual overlap error."""
    flux = xr.Dataset(coords={"time": pd.DatetimeIndex([])})

    with pytest.raises(ValueError) as exc_info:
        _assign_flux_time_bounds(
            flux,
            flux_frequency="yearly",
            inv_start=pd.Timestamp("2021-01-01"),
            inv_end=pd.Timestamp("2021-02-01"),
        )

    message = str(exc_info.value)
    assert "yearly" in message
    assert "no flux timestamps" in message
    assert "2021-01-01" in message
    assert "2021-02-01" in message


@pytest.mark.parametrize("flux_frequency", ["NaT", "0 days", "-1 day"])
def test_assign_flux_time_bounds_rejects_nonpositive_explicit_period(flux_frequency):
    """Explicit PARIS periods must be finite and positive."""
    flux = xr.Dataset(coords={"time": pd.to_datetime(["2021-01-01"])})

    with pytest.raises(ValueError, match="positive fixed duration"):
        _assign_flux_time_bounds(
            flux,
            flux_frequency=flux_frequency,
            inv_start=pd.Timestamp("2021-01-01"),
            inv_end=pd.Timestamp("2021-02-01"),
        )


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


def test_paris_flux_output_uses_january_annual_period_for_june_run(inv_out, europe_country_file):
    """Public PARIS output clips a retained January annual period to a June run."""
    inv_out.run_metadata["start_date"] = "2019-06-01"
    inv_out.run_metadata["end_date"] = "2019-07-01"

    assert infer_flux_frequency(inv_out.flux) == "yearly"
    flux_outs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        flux_frequency=infer_flux_frequency(inv_out.flux),
    )

    assert flux_outs.sizes["time"] == 1
    actual = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(flux_outs.time.values[0]))
    assert actual == pd.Timestamp("2019-06-16")


def test_latest_paris_flux_output_reports_clipped_annual_midpoint_and_bounds(inv_out, europe_country_file):
    """Latest PARIS flux reports the exact midpoint and bounds of a clipped annual prior."""
    inv_out.run_metadata["start_date"] = "2019-06-01"
    inv_out.run_metadata["end_date"] = "2019-07-01"

    flux_outs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        inversion_grid=False,
        flux_frequency=infer_flux_frequency(inv_out.flux),
        template_version="latest",
    )

    epoch = pd.Timestamp("1970-01-01")
    actual = epoch + pd.Timedelta(days=float(flux_outs.time.values[0]))
    bounds = epoch + pd.to_timedelta(flux_outs.time_bnds.values[0], unit="D")

    assert flux_outs.sizes["time"] == 1
    assert actual == pd.Timestamp("2019-06-16")
    assert list(bounds) == [pd.Timestamp("2019-06-01"), pd.Timestamp("2019-07-01")]


def test_legacy_paris_concentration_shifts_hourly_observations_to_midpoints(inv_out):
    """Legacy PARIS concentration shifts hourly observation starts by 30 minutes."""
    observation_starts = pd.DatetimeIndex(observation_inputs_for_outputs(inv_out).time.values).unique()
    expected = (observation_starts + pd.Timedelta(minutes=30) - pd.Timestamp("1970-01-01")) / pd.Timedelta(
        days=1
    )

    result = paris_concentration_outputs(inv_out, obs_avg_period="1h")

    assert result.time.dims == ("time",)
    np.testing.assert_array_equal(result.time.values, expected.to_numpy())


def test_latest_paris_concentration_reports_hourly_midpoints_and_bounds(inv_out):
    """Latest PARIS concentration reports hourly midpoints and exact start/end bounds."""
    epoch = pd.Timestamp("1970-01-01")
    starts = pd.DatetimeIndex(observation_inputs_for_outputs(inv_out).time.values)
    expected_starts = (starts - epoch) / pd.Timedelta(days=1)
    expected_ends = (starts + pd.Timedelta(hours=1) - epoch) / pd.Timedelta(days=1)
    expected_midpoints = (starts + pd.Timedelta(minutes=30) - epoch) / pd.Timedelta(days=1)

    result = paris_concentration_outputs(inv_out, obs_avg_period="1h", template_version="latest")

    assert result.time.dims == ("index",)
    assert result.time_bnds.dims == ("index", "nbnds")
    np.testing.assert_array_equal(result.time.values, expected_midpoints.to_numpy())
    np.testing.assert_array_equal(
        result.time_bnds.values,
        np.column_stack([expected_starts.to_numpy(), expected_ends.to_numpy()]),
    )


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


def test_paris_template_registry_requires_explicit_latest():
    """PARIS output keeps the legacy templates by default for the next release."""
    legacy = paris_template_files(DEFAULT_PARIS_TEMPLATE_VERSION)
    latest = paris_template_files("latest")

    assert DEFAULT_PARIS_TEMPLATE_VERSION == "legacy"
    assert legacy.concentration_version == "v03"
    assert legacy.flux_version == "legacy"
    assert latest.concentration_version == "v04"
    assert latest.flux_version == "v03"
    assert legacy.concentration.exists()
    assert legacy.flux.exists()
    assert latest.concentration.exists()
    assert latest.flux.exists()


def test_save_inversion_output(mcmc_args, tmpdir, deterministic_sampler):
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


def test_country_outputs_lognormal_reparam_conflict(
    mcmc_args,
    europe_country_file,
    deterministic_sampler,
):
    """Check country outputs ignore reparameterized latent-only traces."""
    mcmc_args["output_format"] = "inv_out"
    mcmc_args["reparameterise_log_normal"] = True
    mcmc_args["xprior"] = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}

    inv_out = fixedbasisMCMC(**mcmc_args)
    assert isinstance(inv_out, InversionOutput)
    assert deterministic_sampler[0]["reparameterise_log_normal"] is True
    assert deterministic_sampler[0]["xprior"] == mcmc_args["xprior"]
    assert "x_latent" in inv_out.trace.posterior
    trace_ds = inv_out.trace_dataset(var_roles="flux_scale")
    assert "x_prior" in trace_ds
    assert "x_posterior" in trace_ds
    assert "x_latent_prior" not in trace_ds
    assert "x_latent_posterior" not in trace_ds

    country_outs = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")
    assert "country_prior_mean" in country_outs
    assert "country_posterior_mean" in country_outs


def test_hbmcmc_postprocessing_saves_legacy_output(mcmc_args, tmpdir, deterministic_sampler):
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


def test_resolve_output_format_rejects_column_legacy_output():
    """Legacy HBMCMC formatting remains unsupported for column observations."""
    with pytest.raises(ValueError, match="column observations"):
        _resolve_output_format("hbmcmc", paris_postprocessing=False, is_column=True)


def test_fixedbasis_mcmc_detects_scalar_column_inlet(mcmc_args):
    """A scalar column inlet rejects legacy output before data preparation."""
    mcmc_args["inlet"] = "column"
    mcmc_args["output_format"] = "hbmcmc"

    with pytest.raises(ValueError, match="column observations"):
        fixedbasisMCMC(**mcmc_args)


def test_fixedbasis_mcmc_detects_platform_only_satellite(
    mcmc_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar satellite platform rejects legacy output before retrieval."""

    def fail_preparation(**kwargs: object) -> None:
        """Fail if an unambiguous satellite request reaches retrieval."""
        raise AssertionError("Satellite legacy output should fail before preparation.")

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        fail_preparation,
    )
    mcmc_args["inlet"] = None
    mcmc_args["platform"] = "satellite"
    mcmc_args["output_format"] = "legacy"

    with pytest.raises(ValueError, match="column observations"):
        fixedbasisMCMC(**mcmc_args)


def test_fixedbasis_mcmc_uses_retained_column_status_after_mixed_site_drop(
    mcmc_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped column site does not block legacy output for retained surface data."""

    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        lambda **kwargs: _minimal_fixedbasis_prepared_data(is_column=False),
    )

    def stop_at_sampling(**kwargs: object) -> None:
        """Prove output validation passed without running a sampler."""
        raise RuntimeError("sampling reached")

    monkeypatch.setattr(hbmcmc_module.mcmc, "inferpymc", stop_at_sampling)
    mcmc_args["sites"] = ["TAC", "GOSAT-BRAZIL"]
    mcmc_args["averaging_period"] = ["1H", "1H"]
    mcmc_args["inlet"] = ["100m", "column"]
    mcmc_args["platform"] = ["surface", "satellite"]
    mcmc_args["output_format"] = "legacy"

    with pytest.raises(RuntimeError, match="sampling reached"):
        fixedbasisMCMC(**mcmc_args)


def test_fixedbasis_mcmc_rejects_retained_column_from_mixed_request(
    mcmc_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-preparation validation rejects a retained mixed column site."""
    monkeypatch.setattr(
        hbmcmc_module,
        "prepare_fixedbasis_inversion_data",
        lambda **kwargs: _minimal_fixedbasis_prepared_data(is_column=True),
    )
    mcmc_args["sites"] = ["TAC", "GOSAT-BRAZIL"]
    mcmc_args["averaging_period"] = ["1H", "1H"]
    mcmc_args["inlet"] = ["100m", None]
    mcmc_args["platform"] = ["surface", "satellite"]
    mcmc_args["output_format"] = "legacy"

    with pytest.raises(ValueError, match="column observations"):
        fixedbasisMCMC(**mcmc_args)


def test_paris_postprocessing_compatibility_matches_paris_output_format(
    mcmc_args,
    deterministic_sampler,
):
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


def test_hbmcmc_postprocessing_preserves_expected_vars_attrs_and_coords(
    mcmc_args,
    tmpdir,
    deterministic_sampler,
):
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
    for interval_name in ("Ymod68", "Ymod95", "country68", "country95"):
        assert np.isfinite(outputs[interval_name].values).sum() == outputs[interval_name].size
    assert "UInum" in outputs.coords
    assert "countrynames" in outputs.coords


def test_inv_out_and_trace_outputs_preserve_downstream_dims_and_custom_paths(
    mcmc_args,
    tmpdir,
    deterministic_sampler,
):
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
