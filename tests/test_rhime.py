from __future__ import annotations

from pathlib import Path

import pymc as pm
import pytest
import xarray as xr

import openghg_inversions.models as models
import openghg_inversions.rhime as rhime_module
from openghg_inversions.cli import main
from openghg_inversions.inversion_data import PreparedInversionData, prepare_inversion_data
from openghg_inversions.inversion_data.preparation import (
    _drop_sites_missing_from_loaded_data,
    _make_inv_inputs,
)
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.models import (
    build_rhime_model,
    build_rhime_multisector_model,
    safe_pymc_name,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.rhime import (
    RhimeResult,
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
    assert len(model.coords["region"]) == multisector_inv_inputs.sizes["region"]


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
    assert models.build_rhime_multisector_model is build_rhime_multisector_model
    assert models.safe_pymc_name is safe_pymc_name
    assert isinstance(models.DEFAULT_X_PRIOR, dict)
    assert isinstance(models.DEFAULT_BC_PRIOR, dict)
    assert isinstance(models.DEFAULT_SIGMA_PRIOR, dict)
    assert isinstance(models.DEFAULT_OFFSET_PRIOR, dict)


def test_resolve_flux_sources_prefers_new_name() -> None:
    assert resolve_flux_sources(flux_sources=["new"], emissions_name=["legacy"]) == ["new"]
    assert resolve_flux_sources(emissions_name=["legacy"]) == ["legacy"]


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


def _shared_preparation_args(data_args: dict, flux_sources: list[str]) -> dict:
    args = data_args.copy()
    args.pop("emissions_name", None)
    args.update(
        {
            "output_name": "prep_test",
            "flux_sources": flux_sources,
            "basis_algorithm": "quadtree",
            "nbasis": 4,
            "use_bc": True,
            "return_basis_objects": True,
        }
    )
    return args


def test_prepare_inversion_data_single_sector_reloads_merged_data(
    tac_ch4_data_args, merged_data_dir, merged_data_file_name
) -> None:
    args = _shared_preparation_args(tac_ch4_data_args, tac_ch4_data_args["emissions_name"])
    args.update(
        {
            "reload_merged_data": True,
            "merged_data_dir": str(merged_data_dir),
            "merged_data_name": merged_data_file_name,
        }
    )

    prepared = prepare_inversion_data(**args)

    assert isinstance(prepared, PreparedInversionData)
    assert prepared.inv_inputs is not None
    assert prepared.basis is not None
    assert prepared.flux is not None
    assert prepared.sites == ["TAC"]
    assert "source" not in prepared.inv_inputs["H"].dims


def test_prepare_inversion_data_multisector_keeps_source_dimension(tac_ch4_data_args) -> None:
    flux_sources = ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"]
    args = _shared_preparation_args(tac_ch4_data_args, flux_sources)
    args["split_by_sectors"] = True

    prepared = prepare_inversion_data(**args)

    assert prepared.inv_inputs is not None
    assert prepared.flux is not None
    assert "source" in prepared.inv_inputs["H"].dims
    assert set(prepared.inv_inputs["H"].coords["source"].values) == set(flux_sources)
    assert set(prepared.flux.coords["source"].values) == set(flux_sources)


def test_loaded_merged_data_alignment_checks_site_membership() -> None:
    sites, inlet, fp_height, instrument, max_level, averaging_period = _drop_sites_missing_from_loaded_data(
        fp_all={"TAC": object(), "MHD": object(), ".flux": object()},
        sites=["TAC", "RGL"],
        inlet=["185m", "90m"],
        fp_height=["185m", "90m"],
        instrument=["inst-1", "inst-2"],
        max_level=17,
        averaging_period=["1H", "2H"],
    )

    assert sites == ["TAC"]
    assert inlet == ["185m"]
    assert fp_height == ["185m"]
    assert instrument == ["inst-1"]
    assert max_level == 17
    assert averaging_period == ["1H"]


def test_loaded_merged_data_alignment_rejects_no_matching_sites() -> None:
    with pytest.raises(ValueError, match="does not include any requested sites"):
        _drop_sites_missing_from_loaded_data(
            fp_all={"TAC": object(), "MHD": object(), ".flux": object()},
            sites=["RGL", "BSD"],
            inlet=["90m", "248m"],
            fp_height=["90m", "248m"],
            instrument=["inst-1", "inst-2"],
            max_level=17,
            averaging_period=["1H", "2H"],
        )


def test_prepare_inversion_data_prunes_reloaded_merged_data_to_requested_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_fp_all_keys: set[str] = set()

    def fake_load_merged_data(*args: object, **kwargs: object) -> dict:
        return {
            "TAC": object(),
            "MHD": object(),
            ".flux": object(),
            ".species": "CH4",
        }

    def fake_basis_functions_wrapper(**kwargs: object) -> dict:
        nonlocal captured_fp_all_keys
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_fp_all_keys = set(fp_all)
        return {"TAC": xr.Dataset(coords={"time": [0]})}

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC"]
        return xr.Dataset({"H": (("region", "nmeasure"), [[1.0]])})

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.load_merged_data",
        fake_load_merged_data,
    )
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.basis_functions_wrapper",
        fake_basis_functions_wrapper,
    )
    monkeypatch.setattr("openghg_inversions.inversion_data.preparation.make_inv_inputs", fake_make_inv_inputs)

    prepare_inversion_data(
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
def test_prepare_inversion_data_normalises_averaging_period_to_site_count(
    monkeypatch: pytest.MonkeyPatch,
    averaging_period: str | None,
    expected: list[str | None],
) -> None:
    captured_averaging_period: list[str | None] | None = None

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str | None]]:
        nonlocal captured_averaging_period
        captured_averaging_period = kwargs["averaging_period"]
        assert isinstance(captured_averaging_period, list)
        return (
            {".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            captured_averaging_period,
        )

    def fake_basis_functions_wrapper(**kwargs: object) -> dict:
        return {
            "TAC": xr.Dataset(coords={"time": [0]}),
            "MHD": xr.Dataset(coords={"time": [0]}),
        }

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC", "MHD"]
        return xr.Dataset({"H": (("region", "nmeasure"), [[1.0, 1.0]])})

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.basis_functions_wrapper",
        fake_basis_functions_wrapper,
    )
    monkeypatch.setattr("openghg_inversions.inversion_data.preparation.make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_inversion_data(
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
    assert prepared.averaging_period == expected


def test_prepare_inversion_data_rejects_misaligned_averaging_period_list() -> None:
    with pytest.raises(ValueError, match="List averaging_period does not have specified length"):
        prepare_inversion_data(
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
def test_prepare_inversion_data_rejects_non_string_averaging_period_values(
    averaging_period: object,
) -> None:
    with pytest.raises(ValueError, match="averaging_period"):
        prepare_inversion_data(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=averaging_period,
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
    original_signature = rhime_module.inspect.signature(rhime_module._prepare_data)

    def fake_prepare_data(**kwargs: object) -> None:
        nonlocal captured_averaging_period
        captured_averaging_period = kwargs["averaging_period"]
        raise RuntimeError("stop after data argument capture")

    fake_prepare_data.__signature__ = original_signature
    monkeypatch.setattr(rhime_module, "_prepare_data", fake_prepare_data)

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


def test_prepare_inversion_data_treats_min_error_none_as_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_min_error: object = None

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_basis_functions_wrapper(**kwargs: object) -> dict:
        return {"TAC": xr.Dataset(coords={"time": [0]})}

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        nonlocal captured_min_error
        captured_min_error = kwargs["min_error"]
        return xr.Dataset({"H": (("region", "nmeasure"), [[1.0]])})

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.basis_functions_wrapper",
        fake_basis_functions_wrapper,
    )
    monkeypatch.setattr("openghg_inversions.inversion_data.preparation.make_inv_inputs", fake_make_inv_inputs)

    prepare_inversion_data(
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


def test_prepare_inversion_data_rejects_min_error_dict_missing_retained_site() -> None:
    with pytest.raises(ValueError, match="MHD"):
        _make_inv_inputs(
            fp_data={},
            sites=["TAC", "MHD"],
            start_date="2019-01-01",
            bc_freq=None,
            sigma_freq=None,
            min_error={"TAC": 1.0},
            calculate_min_error=None,
            min_error_options=None,
        )


def test_calculate_min_error_warning_is_user_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_min_error: object = None

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        nonlocal captured_min_error
        captured_min_error = kwargs["min_error"]
        return xr.Dataset({"H": (("region", "nmeasure"), [[1.0]])})

    monkeypatch.setattr("openghg_inversions.inversion_data.preparation.make_inv_inputs", fake_make_inv_inputs)

    with pytest.warns(FutureWarning, match="calculate_min_error"):
        _make_inv_inputs(
            fp_data={"TAC": xr.Dataset(coords={"time": [0]})},
            sites=["TAC"],
            start_date="2019-01-01",
            bc_freq=None,
            sigma_freq=None,
            min_error=0.0,
            calculate_min_error="residual",
            min_error_options=None,
        )

    assert captured_min_error == "residual"


def test_prepare_inversion_data_aligns_averaging_period_after_empty_site_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_basis_functions_wrapper(**kwargs: object) -> dict:
        return {
            "TAC": xr.Dataset(coords={"time": [0]}),
            "MHD": xr.Dataset(coords={"time": []}),
        }

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        assert sites == ["TAC"]
        return xr.Dataset({"H": (("region", "nmeasure"), [[1.0]])})

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.basis_functions_wrapper",
        fake_basis_functions_wrapper,
    )
    monkeypatch.setattr("openghg_inversions.inversion_data.preparation.make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_inversion_data(
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

    assert prepared.sites == ["TAC"]
    assert prepared.averaging_period == ["1H"]


def test_prepare_inversion_data_rejects_all_sites_dropped_by_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_basis_functions_wrapper(**kwargs: object) -> dict:
        return {
            "TAC": xr.Dataset(coords={"time": []}),
            "MHD": xr.Dataset(coords={"time": []}),
        }

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.preparation.basis_functions_wrapper",
        fake_basis_functions_wrapper,
    )

    with pytest.raises(ValueError, match="No sites remain after filtering"):
        prepare_inversion_data(
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

    rhime_module._validate_required_params(args)


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
        rhime_module._validate_required_params(args)


def test_output_path_validation_allows_output_none_without_path() -> None:
    rhime_module._validate_output_path_settings(
        output_format="none",
        output_path=None,
        save_trace=False,
        save_inversion_output=True,
        multisector=False,
    )


def test_output_path_validation_rejects_default_standard_save_without_path() -> None:
    with pytest.raises(ValueError, match="output_path"):
        rhime_module._validate_output_path_settings(
            output_format="inv_out",
            output_path=None,
            save_trace=False,
            save_inversion_output=True,
            multisector=False,
        )


def test_save_inferencedata_prefers_h5netcdf(tmp_path: Path) -> None:
    class FakeInferenceData:
        def __init__(self) -> None:
            self.calls = []

        def to_netcdf(self, path, **kwargs):
            self.calls.append((path, kwargs))

    idata = FakeInferenceData()
    path = tmp_path / "trace.nc"

    rhime_module._save_inferencedata(idata, path)

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

    rhime_module._save_inferencedata(idata, path)

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

    rhime_module._validate_supported_params(args)


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
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": False,
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sampler_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )

    result = run_rhime(**args)

    assert isinstance(result, RhimeResult)
    assert "x" in result.idata.posterior
    assert "mu" in result.idata.posterior
    assert result.run_spec.split_by_sectors is False
    assert "inversion_output" in result.outputs
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
    obs_long_names = [
        reloaded.obs.attrs.get("long_name", ""),
        reloaded.obs_err.attrs.get("long_name", ""),
        reloaded.obs_repeatability.attrs.get("long_name", ""),
        reloaded.obs_variability.attrs.get("long_name", ""),
    ]
    assert all("number_of_observations" not in long_name for long_name in obs_long_names)


def test_run_rhime_multisector_api_smoke(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"],
            "output_name": "rhime_multisector_test",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": False,
            "output_format": "none",
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sampler_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )
    args.pop("emissions_name")

    result = run_rhime_multisector(**args)

    assert isinstance(result, RhimeResult)
    assert result.run_spec.split_by_sectors is True
    assert "x_total_ukghg_edgar7" in result.idata.posterior
    assert "x_total_ukghg_edgar7_shuffled" in result.idata.posterior
    assert "sector_flux_diagnostics" in result.outputs


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
            '{"nit": 1}',
        ]
    )

    assert seen["config_file"] == str(config_file)
    assert seen["kwargs"]["start_date"] == "2019-01-01"
    assert seen["kwargs"]["end_date"] == "2019-01-02"
    assert seen["kwargs"]["output_path"] == str(tmp_path)
    assert seen["kwargs"]["nit"] == 1


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
