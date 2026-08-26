"""RHIME output construction and persistence helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions._timing import timed
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import (
    OutputFilenameConvention,
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeRunSpec,
)
from openghg_inversions.serialization import reset_serialisation_multiindexes
from openghg_inversions.utils import ncdf_encoding, write_netcdf_preserving_bounds_attrs


@dataclass
class RhimeResult:
    """Complete result of a standard or multisector RHIME recipe."""

    run_spec: RhimeRunSpec
    model_spec: RhimeModelSpec
    output_spec: RhimeOutputSpec
    inv_inputs: xr.Dataset
    idata: az.InferenceData
    output_metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    basis_functions: BasisFunctions | None = None
    model: pm.Model | None = None
    inv_out: InversionOutput | None = None
    sampler: RhimeSampler = field(default_factory=RhimeSampler)
    model_build_result: RhimeModelBuildResult | None = None


def _structured_metadata(value: Any) -> Any:
    """Convert array-backed spec values to lossless JSON-compatible metadata.

    Args:
        value: Nested metadata value, possibly backed by NumPy or xarray.

    Returns:
        Scalars and recursively structured dictionaries/lists. DataArrays keep
        explicit dimensions, dimension coordinates, and values.
    """
    if isinstance(value, xr.DataArray):
        materialized = value.compute()
        return {
            "dims": [str(dim) for dim in materialized.dims],
            "coords": {
                str(dim): _structured_metadata(materialized.coords[dim].to_numpy())
                for dim in materialized.dims
                if dim in materialized.coords
            },
            "values": _structured_metadata(materialized.to_numpy()),
        }
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _structured_metadata(value.item())
        return [_structured_metadata(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _structured_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_structured_metadata(item) for item in value]
    return value


def _resolve_output_path(
    save_setting: str | Path | bool, output_path: str | None, filename: str
) -> Path | None:
    """Resolve an optional output path from a bool/path save setting."""
    if not save_setting:
        return None
    if isinstance(save_setting, str | Path):
        return Path(save_setting)
    if output_path is None:
        raise ValueError("An output path is required when saving RHIME artifacts.")
    return Path(output_path) / filename


def _define_output_filename(
    output_path: str | Path,
    species: str,
    domain: str,
    output_name: str,
    start_date: str,
    *,
    filename_convention: OutputFilenameConvention = "rhime",
    ext: str = ".nc",
) -> Path:
    """Create a derived NetCDF filename using the selected convention.

    Args:
        output_path: Destination directory.
        species: Primary gas or tracer name.
        domain: Model domain name.
        output_name: User-selected product name.
        start_date: Inversion start date.
        filename_convention: RHIME or legacy filename ordering.
        ext: Filename extension including its leading dot.

    Returns:
        Complete derived-output path.
    """
    if filename_convention == "legacy":
        filename = f"{species.upper()}_{domain}_{output_name}_{start_date}{ext}"
    else:
        filename = f"{output_name}_{species}_{domain}_{start_date}{ext}"
    return Path(output_path) / filename


def _define_derived_output_filename(
    output_spec: RhimeOutputSpec,
    *,
    species: str,
    domain: str,
    output_name: str,
    start_date: str,
    ext: str = ".nc",
) -> Path:
    """Create a derived-output filename using the requested convention.

    Args:
        output_spec: Output paths and filename convention.
        species: Primary gas or tracer name.
        domain: Model domain name.
        output_name: User-selected product name.
        start_date: Inversion start date.
        ext: Filename extension including its leading dot.

    Returns:
        Complete derived-output path.

    Raises:
        ValueError: If no output directory was supplied.
    """
    if output_spec.output_path is None:
        raise ValueError("An output path is required when saving RHIME outputs.")
    return _define_output_filename(
        output_spec.output_path,
        species,
        domain,
        output_name,
        start_date,
        filename_convention=output_spec.output_filename_convention,
        ext=ext,
    )


def _save_inferencedata(idata: az.InferenceData, path: str | Path) -> None:
    """Save inference data while preserving metadata and serializable coords.

    Root and group attributes are preserved while group MultiIndexes are reset
    on a serialization copy. The h5netcdf, ArviZ-default, and netcdf4 backends
    are attempted in that order.

    Args:
        idata: Inference data to serialize.
        path: Destination NetCDF path.

    Raises:
        RuntimeError: If every supported NetCDF backend fails.
    """
    if isinstance(idata, az.InferenceData):
        idata = cast(Any, az.InferenceData)(
            attrs=dict(idata.attrs),
            **{group: reset_serialisation_multiindexes(idata[group]) for group in idata.groups()},
        )

    failures = []
    for engine in ("h5netcdf", None, "netcdf4"):
        try:
            if engine is None:
                idata.to_netcdf(str(path), compress=True)
            else:
                idata.to_netcdf(str(path), engine=engine, compress=True)
        except Exception as exc:
            engine_name = "arviz-default" if engine is None else engine
            failures.append(f"{engine_name}: {exc}")
        else:
            return

    joined_failures = "\n".join(failures)
    raise RuntimeError(
        f"Could not save RHIME trace to {path}. Tried h5netcdf, ArviZ default, and netcdf4:\n{joined_failures}"
    )


def _save_requested_trace(result: RhimeResult) -> None:
    """Save the sampled trace when requested by the resolved output spec."""
    trace_path = _resolve_output_path(
        result.output_spec.save_trace,
        result.output_spec.output_path,
        f"{result.output_spec.output_name}{result.run_spec.start_date}_trace.nc",
    )
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with timed("rhime.output.trace_save", path=trace_path):
        _save_inferencedata(result.idata, trace_path)
    result.output_metadata["trace_path"] = str(trace_path)


def _make_inversion_output(
    *,
    result: RhimeResult,
    prepared: RhimePreparedInputs,
) -> InversionOutput:
    """Create a modern InversionOutput without fixedbasis legacy adapters.

    Args:
        result: Sampled recipe result and model-owned output contract.
        prepared: Retained canonical inputs and basis functions.

    Returns:
        Complete modern inversion-output artifact.
    """
    model_build_result = cast(RhimeModelBuildResult, result.model_build_result)
    model_metadata = cast(dict[str, Any], _structured_metadata(asdict(result.model_spec)))
    model_metadata["variable_roles"] = dict(model_build_result.variable_roles)
    builder_metadata = dict(model_build_result.metadata)
    for key in ("model_builder", "likelihood_builder", "likelihood_kwargs"):
        if key in result.output_metadata:
            builder_metadata[key] = result.output_metadata[key]
    if builder_metadata:
        model_metadata["builder"] = _structured_metadata(builder_metadata)
    return InversionOutput(
        inv_inputs=prepared.inv_inputs,
        basis_functions=prepared.basis_functions,
        trace=result.idata,
        run_metadata={
            "start_date": result.run_spec.start_date,
            "end_date": result.run_spec.end_date,
            "sites": list(result.run_spec.sites),
            "averaging_period": list(result.run_spec.averaging_period),
            "split_by_sectors": result.run_spec.split_by_sectors,
            "basis_artifact_source": prepared.basis_artifact_source,
            "basis_artifact_path": prepared.basis_artifact_path,
        },
        model_metadata=model_metadata,
        output_metadata={
            "output_format": result.output_spec.output_format,
            "output_path": result.output_spec.output_path,
            "output_name": result.output_spec.output_name,
            "save_trace": result.output_spec.save_trace,
            "save_inversion_output": result.output_spec.save_inversion_output,
            "sampler": _sampler_metadata(result.sampler),
        },
        provenance={
            "contract": "modern_rhime_inversion_output",
            "compatibility_issue": "401",
            "basis_representation": "operator-backed",
        },
    )


def _sampler_metadata(sampler: RhimeSampler | None) -> dict[str, int | str] | None:
    """Return serialisable sampler metadata for compatibility output attrs."""
    if sampler is None:
        return None
    return {
        "draws": sampler.draws,
        "burn": sampler.burn,
        "tune": sampler.tune,
        "chains": sampler.chains,
        "nuts_sampler": sampler.nuts_sampler,
    }


def make_standard_rhime_outputs(
    *,
    result: RhimeResult,
    prepared: RhimePreparedInputs,
) -> None:
    """Create and attach the requested standard RHIME outputs.

    Args:
        result: Sampled standard result receiving requested products.
        prepared: Retained canonical inputs and basis functions.
    """
    output_spec = result.output_spec
    run_spec = result.run_spec
    model_spec = result.model_spec
    if output_spec.output_format == "none":
        return

    outputs: dict[str, Any] = {}
    output_metadata: dict[str, Any] = {}
    with timed("rhime.output.inversion_output_create", output_format=output_spec.output_format):
        inv_out = _make_inversion_output(
            result=result,
            prepared=prepared,
        )
    outputs["inversion_output"] = inv_out
    output_metadata["inversion_output_contract"] = "modern"

    inv_out_path = _resolve_output_path(
        output_spec.save_inversion_output,
        output_spec.output_path,
        f"{output_spec.output_name}{run_spec.start_date}_inversion_output.nc",
    )
    if output_spec.output_format == "basic":
        from openghg_inversions.postprocessing.make_outputs import basic_output

        output_metadata["postprocessing_input_contract"] = "modern_inversion_output"
        with timed("rhime.output.basic_postprocess"):
            outputs["basic"] = basic_output(inv_out, country_file=output_spec.country_file)
    elif output_spec.output_format == "paris":
        from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

        output_metadata["postprocessing_input_contract"] = "modern_inversion_output"
        obs_avg_period = prepared.averaging_period[0] or "0h"
        kwargs = output_spec.paris_postprocessing_kwargs or {}
        with timed("rhime.output.paris_postprocess"):
            flux_outs, conc_outs = make_paris_outputs(
                inv_out,
                country_file=output_spec.country_file,
                domain=model_spec.domain,
                obs_avg_period=obs_avg_period,
                **kwargs,
            )
        outputs["paris_flux"] = flux_outs
        outputs["paris_concentration"] = conc_outs

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            conc_file = _define_derived_output_filename(
                output_spec,
                species=model_spec.species,
                domain=model_spec.domain,
                output_name=output_spec.output_name + "_conc",
                start_date=run_spec.start_date,
                ext=".nc",
            )
            flux_file = _define_derived_output_filename(
                output_spec,
                species=model_spec.species,
                domain=model_spec.domain,
                output_name=output_spec.output_name + "_flux",
                start_date=run_spec.start_date,
                ext=".nc",
            )
            with timed("rhime.output.paris_concentration_netcdf_write", path=conc_file):
                write_netcdf_preserving_bounds_attrs(conc_outs, conc_file, unlimited_dims=["time"])
            with timed("rhime.output.paris_flux_netcdf_write", path=flux_file):
                write_netcdf_preserving_bounds_attrs(flux_outs, flux_file, unlimited_dims=["time"])
            output_metadata["paris_concentration_path"] = str(conc_file)
            output_metadata["paris_flux_path"] = str(flux_file)
    elif output_spec.output_format == "legacy":
        from openghg_inversions.postprocessing.legacy_outputs import make_legacy_hbmcmc_output

        output_metadata["postprocessing_input_contract"] = "modern_inversion_output"
        with timed("rhime.output.legacy_postprocess"):
            legacy_out = make_legacy_hbmcmc_output(
                inv_out,
                country_file=output_spec.country_file,
                use_bc=model_spec.use_bc,
            )
        outputs["legacy"] = legacy_out

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            legacy_file = _define_derived_output_filename(
                output_spec,
                species=model_spec.species,
                domain=model_spec.domain,
                output_name=output_spec.output_name,
                start_date=run_spec.start_date,
                ext=".nc",
            )
            with timed("rhime.output.legacy_netcdf_write", path=legacy_file):
                legacy_out.to_netcdf(legacy_file, mode="w", encoding=ncdf_encoding(legacy_out))
            output_metadata["legacy_output_path"] = str(legacy_file)

    if inv_out_path is not None:
        inv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with timed("rhime.output.inversion_output_save", path=inv_out_path):
            inv_out.save(inv_out_path)
        output_metadata["inversion_output_path"] = str(inv_out_path)
    _save_requested_trace(result)

    result.inv_out = inv_out
    result.outputs.update(outputs)
    result.output_metadata.update(output_metadata)


def _make_multisector_flux_diagnostics(
    inv_out: InversionOutput,
) -> xr.Dataset:
    """Create sector-aware flux diagnostics with the shared postprocessing layer."""
    from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs

    return make_sector_flux_outputs(
        inv_out,
        stats=["mean"],
        include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )


def make_multisector_rhime_outputs(
    *,
    result: RhimeResult,
    prepared: RhimePreparedInputs,
) -> None:
    """Create and attach the requested multisector RHIME outputs.

    Args:
        result: Sampled multisector result receiving requested products.
        prepared: Retained source-resolved inputs and basis functions.
    """
    output_spec = result.output_spec
    if output_spec.output_format == "none":
        return

    paris_kwargs = dict(output_spec.paris_postprocessing_kwargs or {})
    if output_spec.output_format == "paris":
        template_version = paris_kwargs.pop("template_version", "latest")
        if template_version != "latest":
            raise ValueError(
                "Multi-sector PARIS output supports only template_version='latest'."
            )
        supported = {
            "country_selections",
            "flux_frequency",
            "inversion_grid",
            "report_mode",
            "time_point",
        }
        if unexpected := ", ".join(sorted(paris_kwargs.keys() - supported)):
            raise ValueError(
                f"Unsupported multi-sector latest PARIS postprocessing kwargs: {unexpected}."
            )

    run_spec = result.run_spec
    model_spec = result.model_spec
    with timed("rhime.output.inversion_output_create", output_format=output_spec.output_format):
        inv_out = _make_inversion_output(
            result=result,
            prepared=prepared,
        )
    with timed("rhime.output.multisector_diagnostics"):
        diagnostics = _make_multisector_flux_diagnostics(inv_out)
    outputs: dict[str, Any] = {
        "inversion_output": inv_out,
        "sector_flux_diagnostics": diagnostics,
    }
    output_metadata: dict[str, Any] = {"inversion_output_contract": "modern"}
    inv_out_path = _resolve_output_path(
        output_spec.save_inversion_output,
        output_spec.output_path,
        f"{output_spec.output_name}{run_spec.start_date}_inversion_output.nc",
    )

    if output_spec.output_format == "paris":
        from openghg_inversions.postprocessing.make_paris_outputs import (
            infer_flux_frequency,
            paris_concentration_outputs,
            paris_flux_output,
        )

        flux_frequency = paris_kwargs.pop("flux_frequency", None)
        if flux_frequency is None:
            flux_frequency = infer_flux_frequency(inv_out.flux)
        time_point = paris_kwargs.pop("time_point", "midpoint")
        report_mode = paris_kwargs.pop("report_mode", False)
        inversion_grid = paris_kwargs.pop("inversion_grid", True)
        country_selection_kwargs = {}
        if "country_selections" in paris_kwargs:
            country_selection_kwargs["country_selections"] = paris_kwargs.pop("country_selections")

        obs_avg_period = prepared.averaging_period[0] or "0h"
        conc_outs = paris_concentration_outputs(
            inv_out,
            report_mode=report_mode,
            obs_avg_period=obs_avg_period,
            template_version="latest",
        )
        flux_outs = paris_flux_output(
            inv_out,
            country_file=output_spec.country_file,
            time_point=time_point,
            report_mode=report_mode,
            inversion_grid=inversion_grid,
            flux_frequency=flux_frequency,
            template_version="latest",
            **country_selection_kwargs,
        )
        outputs.update(
            {
                "paris_flux": flux_outs,
                "paris_concentration": conc_outs,
            }
        )
        output_metadata["paris_note"] = (
            "Multi-sector latest PARIS sector-aware flux and total concentration outputs were generated."
        )

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            conc_file = _define_derived_output_filename(
                output_spec,
                species=model_spec.species,
                domain=model_spec.domain,
                output_name=output_spec.output_name + "_conc",
                start_date=run_spec.start_date,
            )
            flux_file = _define_derived_output_filename(
                output_spec,
                species=model_spec.species,
                domain=model_spec.domain,
                output_name=output_spec.output_name + "_flux",
                start_date=run_spec.start_date,
            )
            write_netcdf_preserving_bounds_attrs(conc_outs, conc_file, unlimited_dims=["index"])
            write_netcdf_preserving_bounds_attrs(flux_outs, flux_file, unlimited_dims=["time"])
            output_metadata.update(
                {
                    "paris_concentration_path": str(conc_file),
                    "paris_flux_path": str(flux_file),
                }
            )

    if output_spec.output_path is not None:
        Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
        diagnostics_path = (
            Path(output_spec.output_path)
            / f"{output_spec.output_name}{run_spec.start_date}_sector_flux_diagnostics.nc"
        )
        with timed("rhime.output.multisector_diagnostics_netcdf_write", path=diagnostics_path):
            diagnostics.to_netcdf(diagnostics_path, mode="w", encoding=ncdf_encoding(diagnostics))
        output_metadata["sector_flux_diagnostics_path"] = str(diagnostics_path)

    if inv_out_path is not None:
        inv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with timed("rhime.output.inversion_output_save", path=inv_out_path):
            inv_out.save(inv_out_path)
        output_metadata["inversion_output_path"] = str(inv_out_path)
    _save_requested_trace(result)

    result.inv_out = inv_out
    result.outputs.update(outputs)
    result.output_metadata.update(output_metadata)
