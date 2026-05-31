from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, cast

import numpy as np
import arviz as az
import pymc as pm
import pytest
import xarray as xr

import openghg_inversions.models as models
import openghg_inversions.models.rhime as rhime_models_module
import openghg_inversions.hbmcmc.inversion_pymc as legacy_mcmc
import openghg_inversions.rhime as rhime_public
import openghg_inversions.rhime.params as rhime_params
import openghg_inversions.rhime.outputs as rhime_outputs
import openghg_inversions.rhime.sampling as rhime_sampling
import openghg_inversions.rhime.specs as rhime_specs
import openghg_inversions.inversion_data.preparation as prep_module
import openghg_inversions.rhime.runner as rhime_module
from openghg_inversions.basis.basis_functions import BASIS_ARTIFACT_SOURCE_ATTR, BasisFunctions
from openghg_inversions.cli import main
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.models import (
    build_rhime_model,
    build_rhime_model_from_spec,
    build_rhime_multisector_model,
    build_rhime_multisector_model_from_spec,
    safe_pymc_name,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput, LegacyInversionOutput
from openghg_inversions.rhime import (
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeResult,
    RhimeRunSpec,
    RhimeSampler,
    SectorSpec,
    params_from_config,
    resolve_flux_sources,
    run_rhime,
    run_rhime_multisector,
)


@pytest.fixture(scope="module")
def rhime_inv_inputs(mhd_and_tac_fp_data) -> xr.Dataset:
    return make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        sigma_freq="3h",
        min_error=0.0,
        start_date="2019-01-01",
    )


@pytest.fixture
def multisector_inv_inputs(rhime_inv_inputs: xr.Dataset) -> xr.Dataset:
    ds = rhime_inv_inputs.copy()
    ds["H"] = xr.concat(
        [
            rhime_inv_inputs["H"].expand_dims(source=["total-ukghg-edgar7"]),
            (2.0 * rhime_inv_inputs["H"]).expand_dims(source=["sector-2"]),
        ],
        dim="source",
    )
    return ds


@pytest.fixture
def builder_args() -> dict:
    return {
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
        "sigma_per_site": True,
        "offset_prior": {"pdf": "normal", "mu": 0, "sigma": 1},
        "add_offset": False,
        "use_bc": True,
        "pollution_events_from_obs": True,
        "no_model_error": False,
        "power": 1.99,
    }


def _fake_basis_functions(*, artifact_source: str = "generated") -> BasisFunctions:
    """Build a minimal one-cell basis artifact for RHIME tests."""
    basis = xr.DataArray(
        [[1]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
        name="basis",
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: artifact_source},
    )


def _site_dataset(values: list[float] | None = None) -> xr.Dataset:
    """Build a minimal site dataset with footprint-times-flux values."""
    values = values if values is not None else [2.0, 3.0]
    time = np.array(
        [f"2019-01-01T{hour:02d}:00:00" for hour in range(len(values))],
        dtype="datetime64[ns]",
    )
    fp_x_flux = xr.DataArray(
        np.array(values, dtype=float).reshape(len(values), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        name="fp_x_flux",
    )
    return xr.Dataset(
        {
            "fp_x_flux": fp_x_flux,
            "mf": ("time", np.linspace(10.0, 10.0 + len(values) - 1, len(values))),
            "mf_error": ("time", np.ones(len(values))),
            "mf_repeatability": ("time", np.full(len(values), 0.5)),
            "mf_variability": ("time", np.full(len(values), 0.25)),
        },
        coords={"time": time},
    )


def _minimal_inv_inputs() -> xr.Dataset:
    """Build a minimal single-measurement RHIME inversion-input dataset."""
    return xr.Dataset(
        {"H": (("region", "nmeasure"), [[1.0]])},
        coords={"region": [0], "nmeasure": [0]},
    )


def _minimal_output_inv_inputs() -> xr.Dataset:
    """Build minimal inversion inputs for output adapter tests."""
    inv_inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), [[1.0]]),
            "mf": ("nmeasure", [10.0]),
            "mf_error": ("nmeasure", [1.0]),
            "mf_repeatability": ("nmeasure", [0.5]),
            "mf_variability": ("nmeasure", [0.25]),
            "site_indicator": ("nmeasure", [0]),
            "time": ("nmeasure", np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")),
        },
        coords={"region": [0], "nmeasure": [0]},
    )
    inv_inputs["mf"].attrs["units"] = "ppm"
    return inv_inputs


def _minimal_output_specs(output_format: rhime_specs.OutputFormat = "inv_out") -> tuple[
    RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec
]:
    """Build minimal RHIME specs for output helper tests."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
    )
    output_spec = RhimeOutputSpec(
        output_format=output_format,
        output_name="test",
        save_inversion_output=False,
    )
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
    )
    return model_spec, output_spec, run_spec


def _minimal_output_idata() -> az.InferenceData:
    """Build a minimal posterior trace with one region."""
    return az.from_dict(
        posterior={"x": np.ones((1, 1, 1))},
        coords={"region": [0]},
        dims={"x": ["region"]},
    )


class _SpyBasisFunctions:
    """BasisFunctions test double that records direct sensitivity calls."""

    basis_artifact_source = "datatree"

    def __init__(self, sensitivity: xr.DataArray) -> None:
        self.sensitivity_calls: list[xr.DataArray] = []
        self._sensitivity = sensitivity

    @property
    def flux(self) -> xr.DataArray:
        raise AssertionError("RHIME preparation should not materialise flux from BasisFunctions.")

    @property
    def operator(self) -> object:
        raise AssertionError("RHIME preparation should not materialise a basis matrix.")

    def flat_basis(self) -> xr.DataArray:
        raise AssertionError("RHIME preparation should not request a flat basis.")

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        self.sensitivity_calls.append(fp_x_flux)
        return self._sensitivity


class _DynamicSpyBasisFunctions(_SpyBasisFunctions):
    """BasisFunctions test double that derives sensitivity shape from input time."""

    def __init__(self) -> None:
        super().__init__(xr.DataArray())

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        self.sensitivity_calls.append(fp_x_flux)
        return xr.DataArray(
            np.ones((1, fp_x_flux.sizes["time"])),
            dims=("region", "time"),
            coords={"region": [0], "time": fp_x_flux.time},
            name="H",
        )


def test_build_rhime_model_contains_expected_variables(
    rhime_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    model = build_rhime_model(rhime_inv_inputs, **builder_args)

    assert isinstance(model, pm.Model)
    expected = {"x", "mu", "bc", "mu_bc", "sigma", "epsilon", "y"}
    assert expected.issubset(model.named_vars)


def test_build_rhime_multisector_model_contains_expected_variables(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    sectors = ["total-ukghg-edgar7", "sector-2"]
    model = build_rhime_multisector_model(multisector_inv_inputs, sectors=sectors, **builder_args)

    expected = {
        "x_total_ukghg_edgar7",
        "mu_total_ukghg_edgar7",
        "x_sector_2",
        "mu_sector_2",
        "mu",
        "bc",
        "mu_bc",
        "sigma",
        "epsilon",
        "y",
    }
    assert expected.issubset(model.named_vars)
    region_coord = model.coords["region"]
    assert region_coord is not None
    assert len(region_coord) == multisector_inv_inputs.sizes["region"]


def test_build_rhime_multisector_model_uses_sector_names_for_variables(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    """Sector labels can differ from OpenGHG source values used for data selection."""
    model = build_rhime_multisector_model(
        multisector_inv_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        sector_variable_suffixes={"FF": "ff", "ocean": "ocean"},
        sector_priors={"FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}},
        **builder_args,
    )

    expected = {
        "x_ff",
        "mu_ff",
        "x_ocean",
        "mu_ocean",
        "mu",
        "bc",
        "mu_bc",
        "sigma",
        "epsilon",
        "y",
    }
    assert expected.issubset(model.named_vars)


def test_build_rhime_multisector_model_requires_multiple_sectors(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    with pytest.raises(ValueError, match="at least two sectors"):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["total-ukghg-edgar7"],
            **builder_args,
        )


def test_models_exports_rhime_builders() -> None:
    assert models.build_rhime_model is build_rhime_model
    assert models.build_rhime_model_from_spec is build_rhime_model_from_spec
    assert models.build_rhime_multisector_model is build_rhime_multisector_model
    assert models.build_rhime_multisector_model_from_spec is build_rhime_multisector_model_from_spec
    assert models.safe_pymc_name is safe_pymc_name
    assert isinstance(models.DEFAULT_X_PRIOR, dict)
    assert isinstance(models.DEFAULT_BC_PRIOR, dict)
    assert isinstance(models.DEFAULT_SIGMA_PRIOR, dict)
    assert isinstance(models.DEFAULT_OFFSET_PRIOR, dict)


def test_resolve_flux_sources_prefers_new_name() -> None:
    assert resolve_flux_sources(flux_sources=["new"], emissions_name=["legacy"]) == ["new"]
    assert resolve_flux_sources(emissions_name=["legacy"]) == ["legacy"]


def test_direct_sector_spec_records_source_backing() -> None:
    """Direct Python specs distinguish model sectors from OpenGHG sources."""
    sector = SectorSpec(
        name="FF",
        flux_source="ff-inventory",
        x_prior={"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
        variable_suffix="ff",
    )

    assert sector.name == "FF"
    assert sector.flux_source == "ff-inventory"


def test_public_rhime_dataclasses_keep_existing_positional_order() -> None:
    """New default fields do not intercept existing positional construction."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
    )
    output_spec = RhimeOutputSpec(output_format="none")
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
        True,
    )
    output_metadata = {"path": "trace.nc"}
    result = RhimeResult(
        run_spec,
        model_spec,
        output_spec,
        _minimal_inv_inputs(),
        cast(Any, object()),
        output_metadata,
    )

    assert run_spec.split_by_sectors is True
    assert not hasattr(run_spec, "sampler")
    assert not hasattr(run_spec, "sampling")
    assert result.output_metadata == output_metadata
    assert result.sampler == RhimeSampler()


def test_unreleased_sampling_compatibility_shims_are_absent() -> None:
    """Unreleased same-branch sampling compatibility names are not public API."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
    )
    output_spec = RhimeOutputSpec(output_format="none")
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
    )

    assert not hasattr(rhime_public, "RhimeSamplingSpec")
    assert not hasattr(rhime_sampling, "RhimeSamplingSpec")
    assert not hasattr(RhimeResult, "sampling_spec")
    with pytest.raises(TypeError, match="nit"):
        RhimeSampler(nit=7)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="sampling_spec"):
        RhimeResult(
            run_spec,
            model_spec,
            output_spec,
            _minimal_inv_inputs(),
            cast(Any, object()),
            sampling_spec=RhimeSampler(),  # type: ignore[call-arg]
        )


def test_rhime_runner_setup_builds_specs_before_preparation(tmp_path: Path) -> None:
    """Raw runner params normalize into model, output, and sampling specs."""
    params = {
        "species": "ch4",
        "sites": "TAC",
        "averaging_period": "1h",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["ff-source", "gpp-source", "ter-source", "ocean-source"],
        "sector_sources": {
            "FF": "ff-source",
            "GPP": "gpp-source",
            "TER": "ter-source",
            "ocean": "ocean-source",
        },
        "output_path": str(tmp_path),
        "output_name": "test",
        "output_format": "none",
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        "sector_priors": {
            "GPP": {"pdf": "normal", "mu": 0.7, "sigma": 0.2},
            "TER": {"pdf": "normal", "mu": 1.3, "sigma": 0.3},
        },
        "draws": "7",
        "burn": "1",
        "tune": "2",
        "chains": "3",
        "sample_kwargs": {"random_seed": 42},
        "posterior_predictive_kwargs": {"random_seed": 43},
    }

    setup = rhime_params.make_rhime_runner_setup(
        params=params,
        multisector=True,
        data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
    )

    assert setup.data_args["flux_sources"] == ["ff-source", "gpp-source", "ter-source", "ocean-source"]
    assert setup.data_args["split_by_sectors"] is True
    assert "sector_sources" not in setup.data_args
    assert setup.run_spec.sites == ("TAC",)
    assert setup.run_spec.averaging_period == ("1h",)
    assert setup.run_spec.output.output_format == "none"
    assert setup.sampler == RhimeSampler(
        draws=7,
        burn=1,
        tune=2,
        chains=3,
        nuts_sampler="pymc",
        progressbar=False,
        sample_kwargs={"random_seed": 42},
        posterior_predictive_kwargs={"random_seed": 43},
    )
    assert [sector.name for sector in setup.run_spec.model.sectors] == ["FF", "GPP", "TER", "ocean"]
    assert [sector.flux_source for sector in setup.run_spec.model.sectors] == [
        "ff-source",
        "gpp-source",
        "ter-source",
        "ocean-source",
    ]
    assert setup.run_spec.model.sectors[0].x_prior == {"pdf": "normal", "mu": 1.0, "sigma": 0.5}
    assert setup.run_spec.model.sectors[1].x_prior == {"pdf": "normal", "mu": 0.7, "sigma": 0.2}


def test_legacy_sampling_names_are_not_rhime_config_aliases() -> None:
    """RHIME accepts modern sampler names only; legacy HBMCMC names are unsupported."""
    params = rhime_params.normalise_rhime_params(
        {
            "nit": "7",
            "nchain": "3",
            "verbose": True,
            "sampler_kwargs": {"random_seed": 42},
        }
    )

    with pytest.raises(ValueError, match="nit"):
        rhime_params.validate_supported_params(
            params,
            data_params=set(inspect.signature(prepare_rhime_inputs).parameters),
        )


def test_build_rhime_model_from_spec_forwards_single_sector_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard spec wrapper forwards its sector prior as ``x_prior``."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}

    def fake_build_rhime_model(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(rhime_models_module, "build_rhime_model", fake_build_rhime_model)
    inv_inputs = _minimal_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        sigma_per_site=False,
    )

    model = build_rhime_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["x_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.2}
    assert seen["kwargs"]["bc_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.1}
    assert seen["kwargs"]["sigma_per_site"] is False


def test_build_rhime_model_from_spec_requires_one_sector() -> None:
    """The standard spec wrapper rejects multi-sector specs."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec("FF", "ff-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "ff"),
            SectorSpec("ocean", "ocean-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.3}, "ocean"),
        ),
    )

    with pytest.raises(ValueError, match="exactly one sector"):
        build_rhime_model_from_spec(_minimal_inv_inputs(), model_spec)


def test_build_rhime_multisector_model_from_spec_preserves_sector_source_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec wrapper keeps sector labels separate from OpenGHG source values."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}

    def fake_build_rhime_multisector_model(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        rhime_models_module,
        "build_rhime_multisector_model",
        fake_build_rhime_multisector_model,
    )
    inv_inputs = _minimal_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
            SectorSpec(
                name="ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                variable_suffix="ocean",
            ),
        ),
    )

    model = build_rhime_multisector_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["sectors"] == ["FF", "ocean"]
    assert seen["kwargs"]["sector_sources"] == {"FF": "ff-inventory", "ocean": "ocean-inventory"}
    assert seen["kwargs"]["sector_variable_suffixes"] == {"FF": "ff", "ocean": "ocean"}
    assert seen["kwargs"]["sector_priors"] == {
        "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
    }


def test_rhime_sampler_runs_pymc_sampling_and_predictive_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RhimeSampler owns PyMC sampling, burn slicing, and predictive groups."""

    class FakePosterior:
        sizes = {"draw": 6}

    class FakeInferenceData:
        posterior = FakePosterior()

        def __init__(self) -> None:
            self.isel_kwargs: dict[str, Any] | None = None
            self.extensions: list[Any] = []

        def isel(self, **kwargs: Any) -> "FakeInferenceData":
            self.isel_kwargs = kwargs
            return self

        def extend(self, other: Any) -> None:
            self.extensions.append(other)

    fake_idata = FakeInferenceData()
    seen: dict[str, Any] = {}

    def fake_sample(**kwargs: Any) -> Any:
        seen["sample_kwargs"] = kwargs
        return fake_idata

    def fake_prior_predictive(draws: int, model: pm.Model) -> str:
        seen["prior_predictive"] = {"draws": draws, "model": model}
        return "prior"

    def fake_posterior_predictive(trace: Any, **kwargs: Any) -> str:
        seen["posterior_predictive"] = {"trace": trace, **kwargs}
        return "posterior"

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", fake_sample)
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_prior_predictive",
        fake_prior_predictive,
    )
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_posterior_predictive",
        fake_posterior_predictive,
    )
    model = pm.Model()
    sampler = RhimeSampler(
        draws=7,
        burn=1,
        tune=2,
        chains=3,
        nuts_sampler="numpyro",
        progressbar=True,
        sample_kwargs={"target_accept": 0.9},
        posterior_predictive_kwargs={"random_seed": 42},
    )

    idata = sampler.sample(model)

    assert idata is fake_idata
    assert seen["sample_kwargs"]["draws"] == 7
    assert seen["sample_kwargs"]["tune"] == 2
    assert seen["sample_kwargs"]["chains"] == 3
    assert seen["sample_kwargs"]["nuts_sampler"] == "numpyro"
    assert seen["sample_kwargs"]["progressbar"] is True
    assert seen["sample_kwargs"]["cores"] == 3
    assert seen["sample_kwargs"]["target_accept"] == 0.9
    assert seen["sample_kwargs"]["return_inferencedata"] is True
    assert seen["sample_kwargs"]["idata_kwargs"] == {"log_likelihood": True}
    assert fake_idata.isel_kwargs == {"draw": slice(1, None)}
    assert seen["prior_predictive"] == {"draws": 6, "model": model}
    assert seen["posterior_predictive"] == {
        "trace": fake_idata,
        "model": model,
        "var_names": ["y"],
        "random_seed": 42,
    }
    assert fake_idata.extensions == ["prior", "posterior"]


def test_params_from_config_maps_legacy_emissions_name(tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
emissions_name = ["legacy-source"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"
""",
        encoding="utf-8",
    )

    params = params_from_config(config_file)
    assert params["flux_sources"] == ["legacy-source"]


@pytest.mark.parametrize("prior_name", ["x_prior", "bc_prior", "sigma_prior", "offset_prior"])
def test_params_from_config_rejects_malformed_prior_options(tmp_path: Path, prior_name: str) -> None:
    """Malformed structured prior config values fail during RHIME normalization."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        f"""
[RHIME.PDF]
{prior_name} = {{"pdf": "normal"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=prior_name):
        params_from_config(config_file)


def test_params_from_config_rejects_malformed_sector_priors(tmp_path: Path) -> None:
    """Malformed sector prior config values name the bad option."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[RHIME.PDF]
sector_priors = {"FF": {"pdf": "normal"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sector_priors"):
        params_from_config(config_file)


def test_run_rhime_rejects_string_prior_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid prior types fail before RHIME data preparation is called."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="x_prior"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            x_prior="bad",
        )


def test_run_rhime_rejects_malformed_min_error_options_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid min-error option mappings fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="min_error_options"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            min_error_options="bad",
        )


def test_run_rhime_rejects_malformed_power_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid likelihood power values fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="power"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            power="bad",
        )


def test_run_rhime_multisector_rejects_non_mapping_sector_prior_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid sector prior values fail before RHIME data preparation is called."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="sector_priors"):
        run_rhime_multisector(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["FF", "GPP"],
            sector_priors={"FF": "bad"},
        )


def test_run_rhime_multisector_rejects_non_mapping_sector_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid sector-source mappings fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="sector_sources"):
        run_rhime_multisector(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["ff-source", "gpp-source"],
            sector_sources="bad",
        )


def test_run_rhime_multisector_rejects_duplicate_sanitized_sector_names() -> None:
    """Duplicate PyMC suffixes fail during setup, before RHIME data preparation."""
    with pytest.raises(ValueError, match="duplicate sanitized name"):
        rhime_module._make_rhime_runner_setup(
            params={
                "species": "ch4",
                "sites": ["TAC"],
                "averaging_period": ["1h"],
                "domain": "EUROPE",
                "start_date": "2019-01-01",
                "end_date": "2019-01-02",
                "output_name": "test",
                "output_format": "none",
                "flux_sources": ["ff-source", "gpp-source"],
                "sector_sources": {"Sector 2": "ff-source", "sector-2": "gpp-source"},
            },
            multisector=True,
        )


def test_new_rhime_docs_use_flux_sources_for_examples() -> None:
    """New RHIME docs keep legacy names out of config/API examples."""
    rhime_doc = Path("docs/usage/rhime.rst").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    template = Path("openghg_inversions/config/templates/rhime_template.ini").read_text(encoding="utf-8")

    assert "flux_sources" in rhime_doc
    assert "sector_sources" in rhime_doc
    assert "flux_sources" in readme
    assert "flux_sources" in template
    assert "sector_sources" in template
    assert "emissions_name =" not in rhime_doc
    assert "emissions_name =" not in readme
    assert "emissions_name =" not in template
    assert "draws =" in template
    assert "chains =" in template
    assert "nit =" not in template
    assert "nchain =" not in template
    assert "sample_kwargs =" in template
    assert "sampler_kwargs =" not in template
    assert "Legacy compatibility spelling" in rhime_doc


def test_cleanup_plan_records_issue_400_decisions() -> None:
    """The cleanup plan records current PR decisions without example completions."""
    plan_doc = Path("docs/plans/clean_up_inversions_refactor.md").read_text(encoding="utf-8")

    assert "PR #434 / Issue #400" in plan_doc
    assert "#383 / PR #412 merged" not in plan_doc
    assert "one public concrete `RhimeModelSpec`" in plan_doc
    assert "Do not split the public API" in plan_doc
    assert "SemanticModel" in plan_doc
    assert "RhimeSampler" in plan_doc
    assert "openghg_inversions.rhime.runner" in plan_doc
    assert "openghg_inversions.rhime.model_specs" not in plan_doc
    assert "openghg_inversions.models.rhime" in plan_doc
    assert "prior-predictive-only" in plan_doc
    assert "Deferred Issue #431 data-preparation spec" in plan_doc
    assert "should not introduce `RhimeDataSpec`" in plan_doc
    assert "Deferred Issue #383 / Issue #429 output boundary" in plan_doc


def _rhime_preparation_args(data_args: dict, flux_sources: list[str]) -> dict:
    args = data_args.copy()
    args.pop("emissions_name", None)
    args.update(
        {
            "output_name": "prep_test",
            "flux_sources": flux_sources,
            "basis_algorithm": "quadtree",
            "nbasis": 4,
            "use_bc": True,
        }
    )
    return args


def test_rhime_prepared_inputs_contract_exposes_only_modern_fields() -> None:
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        sites=("TAC",),
        averaging_period=("1H",),
        basis_artifact_source="generated",
    )

    for legacy_attr in (
        "fp_all",
        "fp_data",
        "basis",
        "flux",
        "basis_objects",
        "return_basis_objects",
    ):
        assert not hasattr(prepared, legacy_attr)


def test_prepare_rhime_inputs_single_sector_reloads_merged_data(
    tac_ch4_data_args, merged_data_dir, merged_data_file_name
) -> None:
    args = _rhime_preparation_args(tac_ch4_data_args, tac_ch4_data_args["emissions_name"])
    args.update(
        {
            "reload_merged_data": True,
            "merged_data_dir": str(merged_data_dir),
            "merged_data_name": merged_data_file_name,
        }
    )

    prepared = prepare_rhime_inputs(**args)

    assert isinstance(prepared, RhimePreparedInputs)
    assert prepared.basis_artifact_source == "generated"
    assert isinstance(prepared.basis_functions, BasisFunctions)
    assert prepared.sites == ("TAC",)
    assert "source" not in prepared.inv_inputs["H"].dims
    for legacy_attr in ("fp_all", "fp_data", "basis", "flux", "basis_objects"):
        assert not hasattr(prepared, legacy_attr)


def test_prepare_rhime_inputs_multisector_keeps_source_dimension(tac_ch4_data_args) -> None:
    flux_sources = ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"]
    args = _rhime_preparation_args(tac_ch4_data_args, flux_sources)
    args["split_by_sectors"] = True

    prepared = prepare_rhime_inputs(**args)

    assert prepared.basis_artifact_source == "generated"
    assert isinstance(prepared.basis_functions, BasisFunctions)
    assert "source" in prepared.inv_inputs["H"].dims
    assert set(prepared.inv_inputs["H"].coords["source"].values) == set(flux_sources)


def test_prepare_rhime_inputs_uses_basis_sensitivity_without_legacy_side_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_data = _site_dataset([2.0])
    sensitivity = xr.DataArray(
        [[8.0]],
        dims=("state", "time"),
        coords={"state": [0], "time": site_data.time},
        name="H",
    )
    expected_sensitivity = sensitivity.rename({"state": "region"})
    basis_functions = _SpyBasisFunctions(sensitivity)
    captured_fp_data_keys: set[str] = set()

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> _SpyBasisFunctions:
        assert "return_basis_objects" not in kwargs
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        assert fp_all["TAC"] is site_data
        return basis_functions

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        nonlocal captured_fp_data_keys
        captured_fp_data_keys = set(fp_data)
        assert sites == ["TAC"]
        assert set(fp_data) == {"TAC"}
        xr.testing.assert_identical(fp_data["TAC"]["H"], expected_sensitivity)
        return _minimal_inv_inputs()

    def forbidden_basis_functions_wrapper(*args: object, **kwargs: object) -> None:
        raise AssertionError("RHIME preparation should use make_basis_functions and direct sensitivity.")

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)
    monkeypatch.setattr(
        prep_module,
        "basis_functions_wrapper",
        forbidden_basis_functions_wrapper,
        raising=False,
    )

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="direct_sensitivity",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert prepared.basis_functions is basis_functions
    assert prepared.basis_artifact_source == "datatree"
    assert captured_fp_data_keys == {"TAC"}
    assert len(basis_functions.sensitivity_calls) == 1
    xr.testing.assert_identical(basis_functions.sensitivity_calls[0], site_data["fp_x_flux"])


def test_prepare_rhime_inputs_matches_direct_sensitivity_inv_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_data = _site_dataset([2.0, 3.0])
    basis_functions = _fake_basis_functions()

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        return basis_functions

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="equivalence",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    expected_site_data = site_data.copy()
    expected_site_data["H"] = basis_functions.sensitivity(expected_site_data["fp_x_flux"])
    expected_inv_inputs = make_inv_inputs(
        {"TAC": expected_site_data},
        sites=["TAC"],
        bc_freq=None,
        sigma_freq=None,
        min_error=0.0,
        start_date="2019-01-01",
    )

    xr.testing.assert_identical(prepared.inv_inputs["H"], expected_inv_inputs["H"])
    xr.testing.assert_identical(prepared.inv_inputs["mf"], expected_inv_inputs["mf"])


def test_prepare_rhime_inputs_prunes_reloaded_merged_data_to_requested_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_fp_all_keys: set[str] = set()

    def fake_load_merged_data(*args: object, **kwargs: object) -> dict:
        return {
            "TAC": _site_dataset([2.0]),
            "MHD": _site_dataset([3.0]),
            ".flux": object(),
            ".species": "CH4",
        }

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        nonlocal captured_fp_all_keys
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_fp_all_keys = set(fp_all)
        return _fake_basis_functions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC"]
        return _minimal_inv_inputs()

    monkeypatch.setattr(prep_module, "load_merged_data", fake_load_merged_data)
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="reload_prune",
        flux_sources=["total-ukghg-edgar7"],
        reload_merged_data=True,
        merged_data_dir=str(tmp_path),
        use_bc=False,
    )

    assert "TAC" in captured_fp_all_keys
    assert "MHD" not in captured_fp_all_keys
    assert {key for key in captured_fp_all_keys if key.startswith(".")} == {
        ".flux",
        ".species",
        ".split_by_sectors",
    }


@pytest.mark.parametrize(
    ("averaging_period", "expected"),
    [
        ("1H", ["1H", "1H"]),
        (None, [None, None]),
    ],
)
def test_prepare_rhime_inputs_normalises_averaging_period_to_site_count(
    monkeypatch: pytest.MonkeyPatch,
    averaging_period: str | None,
    expected: list[str | None],
) -> None:
    captured_averaging_period: list[str | None] | None = None
    site_data = {"TAC": _site_dataset([2.0]), "MHD": _site_dataset([3.0])}

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str | None]]:
        nonlocal captured_averaging_period
        averaging_period_arg = kwargs["averaging_period"]
        assert isinstance(averaging_period_arg, list)
        captured_averaging_period = averaging_period_arg
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            captured_averaging_period,
        )

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        return _DynamicSpyBasisFunctions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC", "MHD"]
        return xr.Dataset(
            {"H": (("region", "nmeasure"), [[1.0, 1.0]])},
            coords={"region": [0], "nmeasure": [0, 1]},
        )

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC", "MHD"],
        domain="EUROPE",
        averaging_period=averaging_period,
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="normalise_avg",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert captured_averaging_period == expected
    assert prepared.averaging_period == tuple(expected)


def test_prepare_rhime_inputs_rejects_misaligned_averaging_period_list() -> None:
    with pytest.raises(ValueError, match="List averaging_period does not have specified length"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=["1H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="bad_avg",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


@pytest.mark.parametrize("averaging_period", [1, ["1H", 2]])
def test_prepare_rhime_inputs_rejects_non_string_averaging_period_values(
    averaging_period: object,
) -> None:
    with pytest.raises(ValueError, match="averaging_period"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=cast(Any, averaging_period),
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="bad_avg_type",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


def test_run_rhime_leaves_scalar_averaging_period_for_shared_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_averaging_period: object = None
    original_signature = inspect.signature(prepare_rhime_inputs)

    def fake_prepare_rhime_inputs(**kwargs: object) -> None:
        nonlocal captured_averaging_period
        captured_averaging_period = kwargs["averaging_period"]
        raise RuntimeError("stop after data argument capture")

    setattr(fake_prepare_rhime_inputs, "__signature__", original_signature)
    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fake_prepare_rhime_inputs)

    with pytest.raises(RuntimeError, match="stop after data argument capture"):
        run_rhime(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period="1H",
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="avg_scalar",
            flux_sources=["total-ukghg-edgar7"],
            output_format="none",
        )

    assert captured_averaging_period == "1H"


def test_prepare_rhime_inputs_treats_min_error_none_as_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_min_error: object = None
    site_data = _site_dataset([2.0])

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        return _DynamicSpyBasisFunctions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        nonlocal captured_min_error
        captured_min_error = kwargs["min_error"]
        return _minimal_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="min_error_none",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
        min_error=None,
    )

    assert captured_min_error == 0.0


def test_prepare_rhime_inputs_aligns_averaging_period_after_empty_site_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_data = {"TAC": _site_dataset([2.0]), "MHD": _site_dataset([])}

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        return _fake_basis_functions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC"]
        return _minimal_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC", "MHD"],
        domain="EUROPE",
        averaging_period=["1H", "2H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filter_drop",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert prepared.sites == ("TAC",)
    assert prepared.averaging_period == ("1H",)


def test_prepare_rhime_inputs_rejects_all_sites_dropped_after_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_data = {"TAC": _site_dataset([]), "MHD": _site_dataset([])}

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        return _fake_basis_functions()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)

    with pytest.raises(ValueError, match="No sites remain"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=["1H", "2H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="filter_drop",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


def test_params_from_config_rejects_unsupported_deprecated_option(tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
flux_sources = ["total-ukghg-edgar7"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"

[RHIME.DATA]
calculate_min_error = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calculate_min_error"):
        params_from_config(config_file)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("reparameterise_log_normal", "true"),
        ("mcmc_type", '"hmc"'),
    ],
)
def test_params_from_config_rejects_unsupported_legacy_runner_options(
    tmp_path: Path, name: str, value: str
) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        f"""
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
flux_sources = ["total-ukghg-edgar7"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"

[RHIME.MCMC]
{name} = {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=name):
        params_from_config(config_file)


def test_run_rhime_rejects_unknown_parameter_before_data_preparation(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "definitely_not_a_rhime_parameter": True,
    }

    with pytest.raises(ValueError, match="Unsupported RHIME parameter"):
        run_rhime(**args)


def test_run_rhime_rejects_unsupported_output_format(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "output_format": "legacy",
    }

    with pytest.raises(ValueError, match="Unsupported RHIME output_format"):
        run_rhime(**args)


def test_run_rhime_can_validate_output_format_without_output_path() -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_name": "test",
        "output_format": "legacy",
    }

    with pytest.raises(ValueError, match="Unsupported RHIME output_format"):
        run_rhime(**args)


def test_required_parameter_validation_allows_missing_output_path_for_in_memory_runs() -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
    }

    rhime_params.validate_required_params(args)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("species", ""),
        ("sites", []),
        ("domain", "  "),
    ],
)
def test_required_parameter_validation_rejects_empty_values(name: str, value) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
    }
    args[name] = value

    with pytest.raises(ValueError, match=name):
        rhime_params.validate_required_params(args)


def test_output_path_validation_allows_output_none_without_path() -> None:
    rhime_specs.validate_output_path_settings(
        output_format="none",
        output_path=None,
        save_trace=False,
        save_inversion_output=True,
        multisector=False,
    )


def test_output_path_validation_rejects_default_standard_save_without_path() -> None:
    with pytest.raises(ValueError, match="output_path"):
        rhime_specs.validate_output_path_settings(
            output_format="inv_out",
            output_path=None,
            save_trace=False,
            save_inversion_output=True,
            multisector=False,
        )


def test_make_output_spec_validates_trace_save_path() -> None:
    with pytest.raises(ValueError, match="save_trace"):
        rhime_specs.make_output_spec(
            output_format="inv_out",
            output_path=None,
            output_name="test",
            save_trace=True,
            save_inversion_output=False,
            country_file=None,
            paris_postprocessing_kwargs=None,
            multisector=False,
        )


def test_make_standard_output_bundle_returns_outputs_without_mutating_result() -> None:
    model_spec, output_spec, run_spec = _minimal_output_specs()
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        sites=("TAC",),
        averaging_period=("1h",),
        basis_artifact_source="generated",
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert bundle.outputs == {"inversion_output": bundle.inv_out}
    assert bundle.output_metadata == {"inversion_output_contract": "modern"}


def test_modern_inversion_output_save_load_roundtrip(tmp_path: Path) -> None:
    """Modern RHIME InversionOutput preserves retained inputs, basis, and metadata."""
    model_spec, output_spec, run_spec = _minimal_output_specs()
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(artifact_source="unit-test"),
        sites=("TAC",),
        averaging_period=("1h",),
        basis_artifact_source="unit-test",
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    output_file = tmp_path / "modern_inv_out.nc"
    bundle.inv_out.save(output_file)
    reloaded = InversionOutput.load(output_file)

    assert reloaded.species == "ch4"
    assert reloaded.domain == "EUROPE"
    assert reloaded.start_date == "2019-01-01"
    assert reloaded.run_metadata["basis_artifact_source"] == "unit-test"
    assert reloaded.output_metadata["output_format"] == "inv_out"
    xr.testing.assert_identical(reloaded.inv_inputs, prepared.inv_inputs)
    xr.testing.assert_identical(reloaded.basis_functions.flux, prepared.basis_functions.flux)
    xr.testing.assert_equal(
        reloaded.basis_functions.operator.basis_matrix,
        prepared.basis_functions.operator.basis_matrix,
    )


def test_standard_basic_output_uses_legacy_adapter_without_inferpymc(monkeypatch) -> None:
    """RHIME basic postprocessing adapts modern output without using inferpymc legacy postprocess."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="basic")
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        sites=("TAC",),
        averaging_period=("1h",),
        basis_artifact_source="generated",
    )
    captured: dict[str, Any] = {}

    def fail_inferpymc_postprocessouts(**kwargs: Any) -> None:
        raise AssertionError("run_rhime output helpers must not call inferpymc_postprocessouts")

    def fake_basic_output(inv_out: LegacyInversionOutput, country_file: str | None = None) -> xr.Dataset:
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        return xr.Dataset({"ok": ((), 1)})

    monkeypatch.setattr(legacy_mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr("openghg_inversions.postprocessing.make_outputs.basic_output", fake_basic_output)

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file="countries.json",
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert isinstance(captured["inv_out"], LegacyInversionOutput)
    assert captured["country_file"] == "countries.json"
    assert bundle.output_metadata["inversion_output_contract"] == "modern"
    assert bundle.output_metadata["postprocessing_input_contract"] == "legacy_adapter"
    assert "basic" in bundle.outputs


def test_standard_paris_output_uses_legacy_adapter_without_inferpymc(monkeypatch) -> None:
    """RHIME PARIS postprocessing adapts modern output without using inferpymc legacy postprocess."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="paris")
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        sites=("TAC",),
        averaging_period=("1h",),
        basis_artifact_source="generated",
    )
    captured: dict[str, Any] = {}

    def fail_inferpymc_postprocessouts(**kwargs: Any) -> None:
        raise AssertionError("run_rhime output helpers must not call inferpymc_postprocessouts")

    def fake_make_paris_outputs(
        inv_out: LegacyInversionOutput,
        country_file: str | None = None,
        domain: str | None = None,
        obs_avg_period: str = "4h",
        **kwargs: Any,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        captured["domain"] = domain
        captured["obs_avg_period"] = obs_avg_period
        return xr.Dataset({"flux": ((), 1)}), xr.Dataset({"conc": ((), 1)})

    monkeypatch.setattr(legacy_mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.make_paris_outputs.make_paris_outputs",
        fake_make_paris_outputs,
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file="countries.json",
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert isinstance(captured["inv_out"], LegacyInversionOutput)
    assert captured["country_file"] == "countries.json"
    assert captured["domain"] == "EUROPE"
    assert captured["obs_avg_period"] == "1h"
    assert bundle.output_metadata["inversion_output_contract"] == "modern"
    assert bundle.output_metadata["postprocessing_input_contract"] == "legacy_adapter"
    assert "paris_flux" in bundle.outputs
    assert "paris_concentration" in bundle.outputs


def test_save_inferencedata_prefers_h5netcdf(tmp_path: Path) -> None:
    class FakeInferenceData:
        def __init__(self) -> None:
            self.calls = []

        def to_netcdf(self, path, **kwargs):
            self.calls.append((path, kwargs))

    idata = FakeInferenceData()
    path = tmp_path / "trace.nc"

    rhime_outputs._save_inferencedata(idata, path)  # type: ignore[reportArgumentType]

    assert idata.calls == [(str(path), {"engine": "h5netcdf", "compress": True})]


def test_save_inferencedata_falls_back_after_h5netcdf_failure(tmp_path: Path) -> None:
    class FakeInferenceData:
        def __init__(self) -> None:
            self.calls = []

        def to_netcdf(self, path, **kwargs):
            self.calls.append((path, kwargs))
            if kwargs.get("engine") == "h5netcdf":
                raise ValueError("h5netcdf unavailable")

    idata = FakeInferenceData()
    path = tmp_path / "trace.nc"

    rhime_outputs._save_inferencedata(idata, path)  # type: ignore[reportArgumentType]

    assert idata.calls == [
        (str(path), {"engine": "h5netcdf", "compress": True}),
        (str(path), {"compress": True}),
    ]


def test_supported_parameter_validation_accepts_sigma_per_site(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "sigma_per_site": False,
    }

    rhime_params.validate_supported_params(
        args,
        data_params=set(inspect.signature(prepare_rhime_inputs).parameters),
    )


def test_run_rhime_rejects_multiple_flux_sources(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["a", "b"],
            "output_path": str(tmp_path),
            "output_name": "test",
        }
    )
    args.pop("emissions_name")

    with pytest.raises(ValueError, match="exactly one flux source"):
        run_rhime(**args)


def test_run_rhime_multisector_rejects_single_flux_source(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["total-ukghg-edgar7"],
            "output_path": str(tmp_path),
            "output_name": "test",
        }
    )
    args.pop("emissions_name")

    with pytest.raises(ValueError, match="at least two flux sources"):
        run_rhime_multisector(**args)


def test_run_rhime_api_smoke(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": args.pop("emissions_name"),
            "output_name": "rhime_test",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "draws": 1,
            "burn": 0,
            "tune": 0,
            "chains": 1,
            "reload_merged_data": False,
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sample_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )

    result = run_rhime(**args)

    assert isinstance(result, RhimeResult)
    assert isinstance(result.basis_functions, BasisFunctions)
    assert not hasattr(result, "basis_objects")
    posterior = cast(Any, result.idata).posterior
    assert "x" in posterior
    assert "mu" in posterior
    assert result.run_spec.split_by_sectors is False
    assert "inversion_output" in result.outputs
    assert isinstance(result.inv_out, InversionOutput)
    assert result.output_metadata["inversion_output_contract"] == "modern"
    inv_input_long_names = [
        result.inv_inputs.mf.attrs.get("long_name", ""),
        result.inv_inputs.mf_error.attrs.get("long_name", ""),
        result.inv_inputs.mf_repeatability.attrs.get("long_name", ""),
        result.inv_inputs.mf_variability.attrs.get("long_name", ""),
    ]
    assert all("number_of_observations" not in long_name for long_name in inv_input_long_names)
    output_file = tmp_path / "rhime_test2019-01-01_inversion_output.nc"
    assert output_file.exists()
    reloaded = InversionOutput.load(output_file)
    assert reloaded.species == "ch4"
    assert reloaded.domain == args["domain"]
    assert isinstance(reloaded.basis_functions, BasisFunctions)
    xr.testing.assert_identical(reloaded.inv_inputs, result.inv_inputs)
    obs_long_names = [
        reloaded.inv_inputs.mf.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_error.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_repeatability.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_variability.attrs.get("long_name", ""),
    ]
    assert all("number_of_observations" not in long_name for long_name in obs_long_names)


def test_run_rhime_multisector_api_smoke(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"],
            "sector_sources": {
                "FF": "total-ukghg-edgar7",
                "ocean": "total-ukghg-edgar7-shuffled",
            },
            "output_name": "rhime_multisector_test",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "draws": 1,
            "burn": 0,
            "tune": 0,
            "chains": 1,
            "reload_merged_data": False,
            "output_format": "none",
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sample_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )
    args.pop("emissions_name")

    result = run_rhime_multisector(**args)

    assert isinstance(result, RhimeResult)
    assert isinstance(result.basis_functions, BasisFunctions)
    assert not hasattr(result, "basis_objects")
    assert result.run_spec.split_by_sectors is True
    assert [sector.name for sector in result.model_spec.sectors] == ["FF", "ocean"]
    posterior = cast(Any, result.idata).posterior
    assert "x_ff" in posterior
    assert "x_ocean" in posterior
    assert "sector_flux_diagnostics" in result.outputs
    assert list(result.outputs["sector_flux_diagnostics"].coords["sector"].values) == ["FF", "ocean"]


def test_cli_run_rhime_passes_config_and_overrides(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_name = "test"\n', encoding="utf-8")
    seen = {}

    def fake_run_rhime(*, config_file, **kwargs):
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs

    monkeypatch.setattr("openghg_inversions.rhime.run_rhime", fake_run_rhime)

    main(
        [
            "run-rhime",
            "2019-01-01",
            "2019-01-02",
            "-c",
            str(config_file),
            "--output-path",
            str(tmp_path),
            "--kwargs",
            '{"draws": 1}',
        ]
    )

    assert seen["config_file"] == str(config_file)
    assert seen["kwargs"]["start_date"] == "2019-01-01"
    assert seen["kwargs"]["end_date"] == "2019-01-02"
    assert seen["kwargs"]["output_path"] == str(tmp_path)
    assert seen["kwargs"]["draws"] == 1


def test_cli_run_rhime_multisector_passes_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_name = "test"\n', encoding="utf-8")
    seen = {}

    def fake_run_rhime_multisector(*, config_file, **kwargs):
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs

    monkeypatch.setattr("openghg_inversions.rhime.run_rhime_multisector", fake_run_rhime_multisector)

    main(["run-rhime-multisector", "-c", str(config_file)])

    assert seen["config_file"] == str(config_file)
    assert seen["kwargs"] == {}


def test_safe_pymc_name_sanitizes_source_names() -> None:
    assert safe_pymc_name("total-ukghg-edgar7") == "total_ukghg_edgar7"
    assert safe_pymc_name("Sector 2") == "sector_2"
