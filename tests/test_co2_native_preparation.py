"""Configured CO2 acquisition-to-prepared stage and boundary representation selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from examples.surface_centered_pilot import acquire_supplied, surface_inputs
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.models.boundary import AdditiveChoice, MultiplicativeChoice, boundary_preparation
from openghg_inversions.rhime.co2 import Co2PreparedInputs, handoffs
from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2PreparationConfig
from openghg_inversions.rhime.co2.co2_preparation import check_prepared_boundary
from openghg_inversions.rhime.co2.configuration import Co2PreparationConfig
from openghg_inversions.rhime.co2.execution import load_co2_params, run_co2

WINDOW = {"start_date": "2020-01-16", "end_date": "2020-01-17", "domain": "TOY"}
ADDITIVE = {"type": "additive", "anomaly_scale": 0.2, "mean_shift_prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.5}}


@pytest.fixture
def merged() -> MergedData:
    """Offline surface acquisition holding centered boundary operands."""
    obs, fp, fluxes, bc = surface_inputs()
    return MergedData(
        site_data={"TOY": acquire_supplied(obs, fp, fluxes, bc)}, flux_data=fluxes, boundary_data=bc,
        site_options=SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"),
    )


@pytest.fixture
def partition(tmp_path: Path, merged: MergedData) -> Path:
    """One flux basis region over the native grid."""
    path = tmp_path / "partition.nc"
    flux = merged.flux_data["fossil"].flux.isel(time=0, drop=True)
    xr.ones_like(flux, dtype=int).rename("basis").to_dataset().to_netcdf(path)
    return path


def request(partition: Path, **prepared: object) -> dict:
    """A complete configured CO2 request with a build-route preparation."""
    return {
        "format_version": 1, "recipe": "co2",
        "prepared_inputs": {
            **WINDOW,
            "acquisition": {
                "sites": ["TOY"], "averaging_period": "3h", "flux_sources": ["biosphere", "fossil"],
                "surface_transport": {"biosphere": "integrated", "fossil": "integrated"},
                "boundary_reference": 400.0, "boundary_reference_units": "ppm", "boundary_mass_tolerance": 1e-6,
            },
            "observations": {"filters": ["daytime9to5"]},
            "basis": {"partition_path": str(partition)},
            "native_prior": {"sigma": 0.2, "correlation_length": 1.0},
            "aggregation_error_rank": "exact",
            **prepared,
        },
        "likelihood": {"kind": "additive_sigma", "no_model_error": True, "fixed_model_mismatch": 0.0},
    }


def prepare(tmp_path: Path, config: dict, merged: MergedData, name: str) -> Co2PreparedInputs:
    """Run the configured prepare stage and reopen its artifact."""
    arguments = Co2PreparationConfig.from_params(load_co2_params(params=config), require_source=True)
    manifest = handoffs.prepare(
        **arguments.preparation_arguments(), merged_data=merged, output_dir=tmp_path / name,
    )
    record = json.loads(manifest.read_text())
    assert "affine_reconstruction" in record
    return handoffs.load_preparation(manifest).prepared


@pytest.mark.parametrize("choice,requested,expected", [
    (None, None, ("ordinary", False)),
    (None, "centered", ("centered", False)),
    (MultiplicativeChoice({"pdf": "normal", "mu": 1.0, "sigma": 0.1}), None, ("ordinary", False)),
    (MultiplicativeChoice({"pdf": "normal", "mu": 1.0, "sigma": 0.1}), "centered", ("centered", False)),
    (AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.1), None, ("centered", True)),
    (AdditiveChoice(None, 0.1, anomalies="independent"), "centered", ("centered", False)),
])
def test_choices_declare_their_preparation(choice, requested, expected) -> None:
    needs = boundary_preparation(choice, requested)
    assert (needs.representation, needs.centering_weights) == expected


def test_centered_only_choices_reject_ordinary_preparation() -> None:
    with pytest.raises(ValueError, match="additive boundary requires boundary_representation='centered'"):
        AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.1).preparation("ordinary")
    with pytest.raises(ValueError, match="must be one of"):
        boundary_preparation(None, "absolute")


def test_configuration_resolves_representation_from_boundary_type(partition: Path) -> None:
    config = request(partition)
    resolved = Co2PreparationConfig.from_params(config, require_source=True)
    assert resolved.native is not None and resolved.boundary is None
    assert resolved.native.acquisition["site_options"].sites == ("TOY",)
    assert resolved.native.aggregation_error_rank is None
    config["model"] = {"boundary": ADDITIVE}
    with pytest.raises(ValueError, match="requires prepared_inputs.bc_centering_weights_path"):
        Co2PreparationConfig.from_params(config)
    config["prepared_inputs"]["boundary_representation"] = "ordinary"
    with pytest.raises(ValueError, match="requires boundary_representation='centered'"):
        Co2PreparationConfig.from_params(config)


@pytest.mark.parametrize("change,message", [
    ({"path": "prepared.nc"}, "remove the build options"),
    ({"reconstruction_path": "affine.nc"}, "produces its own affine companion"),
    ({"native_prior": {"mean": 1.0}}, "native_prior.sigma"),
    ({"bc_centering_weights_path": "weights.nc"}, "requires centered boundaries"),
    ({"basis": {"algorithm": "voronoi"}}, "basis.algorithm"),
    ({"acquisition": {"sites": ["TOY"], "averaging_period": "3h"}}, "flux_sources"),
])
def test_build_route_rejects_inconsistent_options(partition: Path, change: dict, message: str) -> None:
    config = request(partition)
    config["prepared_inputs"].update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        Co2PreparationConfig.from_params(config)


def test_build_route_requires_window_and_linked_route_rejects_it(partition: Path) -> None:
    config = request(partition)
    del config["prepared_inputs"]["end_date"]
    with pytest.raises(ValueError, match="end_date for a new preparation"):
        Co2PreparationConfig.from_params(config)
    with pytest.raises(ValueError, match="configured linked preparation is not yet available"):
        Co2O2PreparationConfig.from_params({"prepared_inputs": request(partition)["prepared_inputs"]})


def test_one_merged_dataset_feeds_both_representations(tmp_path: Path, merged: MergedData, partition: Path) -> None:
    ordinary = prepare(tmp_path, request(partition), merged, "ordinary")
    centered = prepare(tmp_path, request(partition, boundary_representation="centered"), merged, "centered")
    absolute, operands = ordinary.inv_inputs, centered.inv_inputs
    assert "boundary_representation" not in absolute.attrs and "H_bc_anomaly" not in absolute
    assert operands.attrs["boundary_representation"] == "centered-v1" and "H_bc" not in operands
    collapsed = operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass
    np.testing.assert_allclose(absolute.H_bc.transpose(*collapsed.dims), collapsed, rtol=0, atol=1e-12)
    for name in ("mf", "H", "alpha_prior_mean", "alpha_prior_covariance", "aggregation_error_covariance"):
        xr.testing.assert_allclose(absolute[name], operands[name])
    # Neither preparation changed the borrowed merged data.
    assert merged.site_data["TOY"].attrs["boundary_representation"] == "centered-v1"


def test_additive_choice_builds_centered_weighted_preparation(
    tmp_path: Path, merged: MergedData, partition: Path,
) -> None:
    weights = xr.DataArray(
        np.full((4, 1), 0.25), dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": list("nesw"), "bc_period": pd.to_datetime(["2020-01-16"])},
        attrs={"units": "1"}, name="bc_centering_weights",
    )
    path = tmp_path / "weights.nc"
    weights.to_dataset().to_netcdf(path)
    config = request(partition, bc_centering_weights_path=str(path))
    config["model"] = {"boundary": {**ADDITIVE, "frequency": "1D", "anchor_time": "2020-01-16"}}
    prepared = prepare(tmp_path, config, merged, "additive")
    assert prepared.inv_inputs.attrs["boundary_representation"] == "centered-v1"
    xr.testing.assert_allclose(prepared.inv_inputs.bc_centering_weights, weights)


def test_absolute_merged_data_cannot_produce_centered(tmp_path: Path, merged: MergedData, partition: Path) -> None:
    from openghg_inversions.inversion_data import select_boundary_representation

    absolute = MergedData(
        site_data=select_boundary_representation(merged.site_data, "ordinary"),
        flux_data=merged.flux_data, boundary_data=merged.boundary_data, site_options=merged.site_options,
    )
    with pytest.raises(ValueError, match="cannot produce centered operands"):
        prepare(tmp_path, request(partition, boundary_representation="centered"), absolute, "bad")


def test_supplied_merged_data_must_match_the_requested_window(
    tmp_path: Path, merged: MergedData, partition: Path,
) -> None:
    from dataclasses import replace

    from openghg_inversions.inversion_data import AcquisitionFacts

    other = replace(merged, acquisition=AcquisitionFacts(species="co2", start_date="2020-02-01"))
    with pytest.raises(ValueError, match="incompatible start_date"):
        prepare(tmp_path, request(partition), other, "window")


def test_saved_preparations_are_checked_against_the_boundary_choice(
    tmp_path: Path, merged: MergedData, partition: Path,
) -> None:
    ordinary = prepare(tmp_path, request(partition), merged, "ordinary").inv_inputs
    centered = prepare(tmp_path, request(partition, boundary_representation="centered"), merged, "centered").inv_inputs
    additive = AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.1, anomalies="independent")
    multiplicative = MultiplicativeChoice({"pdf": "normal", "mu": 1.0, "sigma": 0.1})
    check_prepared_boundary(centered, multiplicative)
    check_prepared_boundary(ordinary, multiplicative)
    check_prepared_boundary(centered, additive)
    with pytest.raises(ValueError, match="have ordinary boundary sensitivity, but centered is required"):
        check_prepared_boundary(ordinary, additive)
    with pytest.raises(ValueError, match="have centered boundary sensitivity, but ordinary is required"):
        check_prepared_boundary(centered, multiplicative, "ordinary")
    with pytest.raises(ValueError, match="requires prepared bc_centering_weights"):
        check_prepared_boundary(centered, AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.1))
    with pytest.raises(ValueError, match="no boundary sensitivity"):
        check_prepared_boundary(ordinary.drop_vars("H_bc"), multiplicative)


def test_staged_import_and_sampling_reject_an_inconsistent_preparation(
    tmp_path: Path, merged: MergedData, partition: Path,
) -> None:
    from openghg_inversions.rhime.co2 import Co2ModelConfig

    arguments = Co2PreparationConfig.from_params(request(partition), require_source=True)
    manifest = handoffs.prepare(**arguments.preparation_arguments(), merged_data=merged, output_dir=tmp_path / "p")
    imported = {
        **request(partition), "prepared_inputs": {"path": str(tmp_path / "p" / "prepared-inputs.nc"), **WINDOW},
        "model": {"boundary": dict(ADDITIVE)},
    }
    with pytest.raises(ValueError, match="centered is required"):
        handoffs.prepare(
            **Co2PreparationConfig.from_params(imported, require_source=True).preparation_arguments(),
            output_dir=tmp_path / "imported",
        )
    model = Co2ModelConfig.from_params(imported)
    with pytest.raises(ValueError, match="centered is required"):
        handoffs.sample(manifest, model=model, output_dir=tmp_path / "s")


def test_full_runner_builds_from_configuration(tmp_path: Path, merged: MergedData, partition: Path) -> None:
    config = request(partition, boundary_representation="centered")
    config["model"] = {"boundary": {"prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.01}}}
    config["sampling"] = {"draws": 20, "tune": 20, "chains": 1, "progressbar": False, "random_seed": 3}
    config["outputs"] = {"output_format": "none"}
    result = run_co2(params=config, merged_data=merged)
    assert result.idata["posterior"].sizes["draw"] == 20
    assert result.prepared.inv_inputs.attrs["boundary_representation"] == "centered-v1"
    with pytest.raises(ValueError, match="merged_data requires prepared_inputs.acquisition"):
        run_co2(params={**config, "prepared_inputs": {"path": "prepared.nc", **WINDOW}}, merged_data=merged)


@pytest.mark.parametrize("template", ["co2.toml", "co2_fixed_ou.toml"])
def test_template_build_block_is_a_valid_preparation(template: str) -> None:
    import tomllib

    from openghg_inversions.rhime.co2 import co2_config_templates

    lines = co2_config_templates()[template].read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# To build a new preparation"))
    block = []
    for line in lines[start + 3:]:
        if not line.startswith("#"):
            break
        block.append(line.removeprefix("#").removeprefix(" "))
    window = "\n".join(f'{name} = "{value}"' for name, value in WINDOW.items())
    config = tomllib.loads(f"[prepared_inputs]\n{window}\n" + "\n".join(block))
    resolved = Co2PreparationConfig.from_params(config, require_source=True)
    assert resolved.native is not None and resolved.boundary_representation == "centered"
