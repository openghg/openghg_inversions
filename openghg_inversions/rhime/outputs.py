"""RHIME output construction and persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import xarray as xr

from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions._timing import timed
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.inference.sampling import describe_recorded_sampler
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.output_views import make_inversion_output
from openghg_inversions.postprocessing.paris_options import NestedParisOptions, ParisOptions
from openghg_inversions.rhime.builders import RhimeModelBuildResult, callable_metadata, RhimeLikelihoodBuilder
from openghg_inversions.rhime.specs import (
    OutputFilenameConvention,
    RhimeModelSpec,
    RhimeOutputSpec,
)
from openghg_inversions.serialization import save_trace
from openghg_inversions.models.priors import prior_records, trace_prior_records
from openghg_inversions.utils import ncdf_encoding, write_netcdf_preserving_bounds_attrs

from ._stage_artifacts import _validate_filename_component


def validate_output_resources(output_spec: RhimeOutputSpec) -> None:
    """Check country reporting resources without creating products or paths.

    Any explicitly selected ``country_file`` must be an existing file, even
    when the requested output format does not use it. ``basic`` and ``paris``
    outputs require an explicit country file. This check reads filesystem
    metadata only; it does not open or validate the file contents.

    Args:
        output_spec: Resolved output choices. The specification is not mutated.

    Raises:
        FileNotFoundError: If the selected country file is missing or is a directory.
        ValueError: If country reporting is requested without a country file.
    """
    if output_spec.country_file is not None and not Path(output_spec.country_file).is_file():
        raise FileNotFoundError(output_spec.country_file)
    if output_spec.output_format in ("basic", "paris") and output_spec.country_file is None:
        raise ValueError("country_file must be supplied for country reporting outputs.")


def validate_standard_output_resources(output_spec: RhimeOutputSpec) -> None:
    """Check reporting resources and file-product destinations before standard execution.

    Args:
        output_spec: Resolved output choices. PARIS and legacy products require
            ``output_path`` because standard runners return the scientific view,
            not the derived products. Use the product functions directly for
            in-memory reporting.

    Raises:
        FileNotFoundError: If the selected country file is missing or is a directory.
        ValueError: If a required country file or product destination is absent.
    """
    validate_output_resources(output_spec)
    if output_spec.output_format in ("paris", "legacy") and output_spec.output_path is None:
        raise ValueError("`output_path` is required for standard PARIS or legacy products.")
    if output_spec.output_format == "paris":
        ParisOptions.from_params(output_spec.paris_postprocessing_kwargs)


def validate_multisector_output_resources(output_spec: RhimeOutputSpec) -> None:
    """Validate multisector reporting resources and file-product destinations.

    Args:
        output_spec: Current output policy, independent of sampled scientific meaning.

    Raises:
        ValueError: If the format is single-sector-only or PARIS has no destination.
        FileNotFoundError: If a selected country file is unavailable.
    """
    if output_spec.output_format in ("basic", "legacy"):
        raise ValueError(f"RHIME output_format {output_spec.output_format!r} supports only single-sector runs.")
    validate_output_resources(output_spec)
    if output_spec.output_format == "paris" and output_spec.output_path is None:
        raise ValueError("`output_path` is required for multisector PARIS products.")
    if output_spec.output_format == "paris":
        options = ParisOptions.from_params(
            output_spec.paris_postprocessing_kwargs, default_template_version="latest",
        )
        if options.template_version != "latest":
            raise ValueError("Multi-sector PARIS output supports only template_version='latest'.")


def validate_nested_output_resources(output_spec: RhimeOutputSpec) -> None:
    """Check nested reporting resources and destinations before doing work."""
    if output_spec.output_format not in ("none", "paris"):
        raise ValueError("`run_nested` currently requires output_format='none' or 'paris'.")
    validate_output_resources(output_spec)
    if output_spec.output_format == "paris" and output_spec.output_path is None:
        raise ValueError("`output_path` is required for nested PARIS products.")
    if output_spec.output_format == "paris":
        NestedParisOptions.from_params(output_spec.paris_postprocessing_kwargs)


def annotate_likelihood_trace(
    idata: xr.DataTree,
    *,
    builder_identity: dict[str, str],
    likelihood_kwargs: Mapping[str, Any] | None,
) -> None:
    """Persist custom-likelihood provenance in place.

    Array-valued options are converted to JSON-compatible values; labelled
    arrays cross an explicit eager serialization boundary.

    Args:
        idata: Sampled trace to annotate in place.
        builder_identity: Importable module and qualified-name provenance for
            the likelihood builder.
        likelihood_kwargs: Resolved builder options to preserve as structured
            JSON metadata.

    Returns:
        None. The input trace is annotated in place.
    """
    idata.attrs["rhime_likelihood_builder"] = json.dumps(builder_identity, sort_keys=True)
    idata.attrs["rhime_likelihood_kwargs"] = json.dumps(
        _structured_metadata(dict(likelihood_kwargs or {})), sort_keys=True
    )


def _structured_metadata(value: Any) -> Any:
    """Convert array-backed spec values to JSON-compatible metadata.

    Args:
        value: Nested metadata value, possibly backed by NumPy or xarray.

    Returns:
        Scalars and recursively structured dictionaries/lists. DataArrays keep
        explicit dimensions, dimension coordinates, and values. Paths become
        strings, Python and NumPy dates/times become ISO strings, and timedeltas
        become strings.
    """
    return _json_value(value, materialize_dataarrays=True)


def _resolve_output_path(
    save_setting: str | Path | bool, output_path: str | Path | None, filename: str
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


def postprocess_standard_outputs(inv_out: InversionOutput, output_spec: RhimeOutputSpec) -> None:
    """Compute and write requested PARIS or legacy products from a scientific result.

    Args:
        inv_out: Borrowed scientific result, independent of output policy.
        output_spec: Resolved product and destination choices. The caller must
            first call :func:`validate_standard_output_resources`. ``none``, ``inv_out`` and ``basic``
            produce no derived files; use ``basic_output(inv_out)`` for in-memory products.

    Raises:
        ValueError: If the format is incompatible or filename values are invalid.
        OSError: If a selected product cannot be written.
    """
    if output_spec.output_format == "none":
        return
    supported = inv_out.model_metadata.get("supported_output_formats")
    if supported is None:
        raise ValueError("Scientific output lacks supported-output capabilities; use product functions directly for views without a model contract.")
    if output_spec.output_format not in supported:
        raise ValueError(
            f"Model does not declare output_format={output_spec.output_format!r} compatible. "
            f"Declared formats: {supported!r}."
        )
    for name, value in (("species", inv_out.model_metadata["species"]),
                        ("domain", inv_out.model_metadata["domain"]),
                        ("output_name", output_spec.output_name),
                        ("start_date", inv_out.run_metadata["start_date"])):
        _validate_filename_component(name, value)
    if output_spec.output_format == "paris":
        from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

        obs_avg_period = inv_out.run_metadata["averaging_period"][0] or "0h"
        options = ParisOptions.from_params(output_spec.paris_postprocessing_kwargs)
        with timed("rhime.output.paris_postprocess"):
            flux_outs, conc_outs = make_paris_outputs(
                inv_out,
                country_file=output_spec.country_file,
                domain=inv_out.model_metadata["domain"],
                obs_avg_period=obs_avg_period,
                template_version=options.template_version,
                time_point=options.time_point,
                flux_frequency=options.flux_frequency,
                report_mode=options.report_mode,
                inversion_grid=options.inversion_grid,
                country_selections=options.country_selections,
            )

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            conc_file = _define_derived_output_filename(
                output_spec,
                species=inv_out.model_metadata["species"],
                domain=inv_out.model_metadata["domain"],
                output_name=output_spec.output_name + "_conc",
                start_date=inv_out.run_metadata["start_date"],
                ext=".nc",
            )
            flux_file = _define_derived_output_filename(
                output_spec,
                species=inv_out.model_metadata["species"],
                domain=inv_out.model_metadata["domain"],
                output_name=output_spec.output_name + "_flux",
                start_date=inv_out.run_metadata["start_date"],
                ext=".nc",
            )
            with timed("rhime.output.paris_concentration_netcdf_write", path=conc_file):
                write_netcdf_preserving_bounds_attrs(
                    conc_outs, conc_file, unlimited_dims=["index" if "index" in conc_outs.dims else "time"]
                )
            with timed("rhime.output.paris_flux_netcdf_write", path=flux_file):
                write_netcdf_preserving_bounds_attrs(flux_outs, flux_file, unlimited_dims=["time"])
    elif output_spec.output_format == "legacy":
        from openghg_inversions.postprocessing.legacy_outputs import make_legacy_hbmcmc_output

        with timed("rhime.output.legacy_postprocess"):
            legacy_out = make_legacy_hbmcmc_output(
                inv_out,
                country_file=output_spec.country_file,
                use_bc=inv_out.model_metadata["use_bc"],
            )

        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            legacy_file = _define_derived_output_filename(
                output_spec,
                species=inv_out.model_metadata["species"],
                domain=inv_out.model_metadata["domain"],
                output_name=output_spec.output_name,
                start_date=inv_out.run_metadata["start_date"],
                ext=".nc",
            )
            with timed("rhime.output.legacy_netcdf_write", path=legacy_file):
                legacy_out.to_netcdf(legacy_file, mode="w", encoding=ncdf_encoding(legacy_out))


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


def postprocess_multisector_outputs(
    inv_out: InversionOutput, output_spec: RhimeOutputSpec,
) -> dict[str, Any]:
    """Compute sector diagnostics and requested PARIS products, writing derived files.

    Args:
        inv_out: Scientific samples and original preparation/run meaning.
        output_spec: Current reporting, naming and destination choices.

    Returns:
        Requested in-memory products. ``none`` computes and writes no products.
        Scientific result and trace exports remain a separate runner operation.

    Raises:
        ValueError: If products or reporting options are unsupported.
        FileNotFoundError: If a reporting resource is unavailable.
    """
    supported = inv_out.model_metadata.get("supported_output_formats")
    if supported is None or output_spec.output_format not in supported:
        raise ValueError(
            f"Model does not declare output_format={output_spec.output_format!r} compatible. "
            f"Declared formats: {supported!r}."
        )
    validate_multisector_output_resources(output_spec)
    if output_spec.output_format == "none":
        return {}

    for name, value in (("species", inv_out.model_metadata["species"]),
                        ("domain", inv_out.model_metadata["domain"]),
                        ("output_name", output_spec.output_name),
                        ("start_date", inv_out.run_metadata["start_date"])):
        _validate_filename_component(name, value)
    with timed("rhime.output.multisector_diagnostics"):
        diagnostics = _make_multisector_flux_diagnostics(inv_out)
    outputs: dict[str, Any] = {
        "inversion_output": inv_out,
        "sector_flux_diagnostics": diagnostics,
    }
    if output_spec.output_format == "paris":
        from openghg_inversions.postprocessing.make_paris_outputs import (
            infer_flux_frequency,
            paris_concentration_outputs,
            paris_flux_output,
        )

        options = ParisOptions.from_params(
            output_spec.paris_postprocessing_kwargs, default_template_version="latest",
        )
        flux_frequency = options.flux_frequency
        if flux_frequency is None:
            flux_frequency = infer_flux_frequency(inv_out.flux)

        obs_avg_period = inv_out.run_metadata["averaging_period"][0] or "0h"
        conc_outs = paris_concentration_outputs(
            inv_out,
            report_mode=options.report_mode,
            obs_avg_period=obs_avg_period,
            template_version="latest",
        )
        flux_outs = paris_flux_output(
            inv_out,
            country_file=output_spec.country_file,
            time_point=options.time_point,
            report_mode=options.report_mode,
            inversion_grid=options.inversion_grid,
            flux_frequency=flux_frequency,
            template_version="latest",
            country_selections=options.country_selections,
        )
        outputs.update(
            {
                "paris_flux": flux_outs,
                "paris_concentration": conc_outs,
            }
        )
        if output_spec.output_path is not None:
            Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
            conc_file = _define_derived_output_filename(
                output_spec,
                species=inv_out.model_metadata["species"],
                domain=inv_out.model_metadata["domain"],
                output_name=output_spec.output_name + "_conc",
                start_date=inv_out.run_metadata["start_date"],
            )
            flux_file = _define_derived_output_filename(
                output_spec,
                species=inv_out.model_metadata["species"],
                domain=inv_out.model_metadata["domain"],
                output_name=output_spec.output_name + "_flux",
                start_date=inv_out.run_metadata["start_date"],
            )
            write_netcdf_preserving_bounds_attrs(conc_outs, conc_file, unlimited_dims=["index"])
            write_netcdf_preserving_bounds_attrs(flux_outs, flux_file, unlimited_dims=["time"])

    if output_spec.output_path is not None:
        Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
        diagnostics_path = (
            Path(output_spec.output_path)
            / f"{output_spec.output_name}{inv_out.run_metadata['start_date']}_sector_flux_diagnostics.nc"
        )
        with timed("rhime.output.multisector_diagnostics_netcdf_write", path=diagnostics_path):
            diagnostics.to_netcdf(diagnostics_path, mode="w", encoding=ncdf_encoding(diagnostics))

    return outputs


def annotate_sample_trace(
    built: RhimeModelBuildResult,
    posterior: xr.DataTree,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Attach construction-owned prior and optional likelihood metadata after sampling.

    Args:
        built: Sampled graph and its construction-owned metadata.
        posterior: Trace annotated in place; numerical payloads are not copied.
        likelihood_builder: Custom likelihood whose identity is recorded.
        likelihood_kwargs: Serializable options for that likelihood.
    """
    posterior.attrs["rhime_priors"] = json.dumps(prior_records(built.model))
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            posterior, builder_identity=callable_metadata(likelihood_builder),
            likelihood_kwargs=likelihood_kwargs,
        )


def make_model_output(
    *,
    prepared: PreparedInputs,
    model_spec: RhimeModelSpec,
    posterior: xr.DataTree,
    output_contract: OutputContract,
    start_date: str,
    end_date: str,
    split_by_sectors: bool,
    actual_inference: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> InversionOutput:
    """Bind sampled states to their borrowed scientific context.

    Args:
        prepared: Retained scientific inputs and basis.
        model_spec: Model choices used for sampling.
        posterior: Sampler-produced or externally validated trace with prior metadata.
        output_contract: Constructed model roles and supported products.
        start_date: Requested inversion start, independently of retained observations.
        end_date: Requested inversion end.
        split_by_sectors: Whether preparation retained separate source contributions.
        actual_inference: Description of the inference that ran.
        provenance: Recorded package provenance for replay; omission collects the
            installed package provenance for an in-memory run.

    Returns:
        A scientific view borrowing the prepared arrays and trace, without writing files.
    """
    priors = trace_prior_records(posterior)
    model_metadata = model_spec.describe(priors=priors)
    model_metadata["constructed_priors"] = priors
    if model_metadata["likelihood"] is not None:
        model_metadata["likelihood"] = model_metadata["likelihood"]["options"]
    for name in ("state_activity", "bc_state_activity"):
        policy = getattr(model_spec, name)
        model_metadata[name] = None if policy is None else asdict(policy)
    for sector, record in zip(model_spec.sectors, model_metadata["sectors"], strict=True):
        role = "flux_scale" if len(model_spec.sectors) == 1 else f"flux_scale:{sector.name}"
        variable = output_contract.variable_roles.get(role)
        if variable is not None and variable in priors:
            record["x_prior"] = priors[variable]
        record["state_activity"] = None if sector.state_activity is None else asdict(sector.state_activity)
    metadata = dict(output_contract.metadata)
    for attr, key in (("rhime_likelihood_builder", "likelihood_builder"),
                      ("rhime_likelihood_kwargs", "likelihood_kwargs")):
        if attr in posterior.attrs:
            metadata[key] = json.loads(posterior.attrs[attr])
    contract = replace(output_contract, metadata=metadata) if metadata != output_contract.metadata else output_contract
    return make_inversion_output(
        prepared=prepared,
        trace=posterior,
        contract=contract,
        run_metadata={
            "start_date": start_date,
            "end_date": end_date,
            "sites": list(prepared.sites),
            "averaging_period": list(prepared.averaging_period),
            "split_by_sectors": split_by_sectors,
        },
        model_metadata=_structured_metadata(model_metadata),
        output_metadata={"sampler": describe_recorded_sampler(actual_inference)},
        provenance=installed_ogi_provenance() if provenance is None else dict(provenance),
    )


def save_inversion_output(
    inv_out: InversionOutput, output_spec: RhimeOutputSpec, *, suffix: str = "",
) -> dict[str, Path]:
    """Export the scientific result and trace without computing derived products.

    Args:
        inv_out: Scientific result to serialize without changing its metadata.
        output_spec: Resolved export choices. ``inv_out`` defaults to exporting the
            scientific result; ``none`` writes nothing. Explicit paths retain
            their destinations and existing files follow normal overwrite behavior.
        suffix: Optional suffix appended to each resolved artifact stem, including
            explicit destinations. For example, ``"_inner"`` selects the inner view.

    Returns:
        Written artifact paths keyed by ``inversion_output{suffix}_path`` and
        ``trace{suffix}_path``. Disabled exports are absent.

    Raises:
        ValueError: If filenames or destinations are invalid.
        OSError: If an artifact cannot be written.
    """
    if output_spec.output_format == "none":
        return {}
    if not output_spec.save_inversion_output and not output_spec.save_trace:
        return {}
    if suffix:
        _validate_filename_component("suffix", suffix)
    for name, value in (("output_name", output_spec.output_name),
                        ("start_date", inv_out.run_metadata["start_date"])):
        _validate_filename_component(name, value)
    stem = f"{output_spec.output_name}{inv_out.run_metadata['start_date']}"
    inversion_path = _resolve_output_path(
        output_spec.save_inversion_output,
        output_spec.output_path,
        f"{stem}_inversion_output.nc",
    )
    trace_path = _resolve_output_path(output_spec.save_trace, output_spec.output_path, f"{stem}_trace.nc")
    paths = {}
    if inversion_path is not None:
        inversion_path = inversion_path.with_name(f"{inversion_path.stem}{suffix}{inversion_path.suffix}")
        inversion_path.parent.mkdir(parents=True, exist_ok=True)
        with timed("rhime.output.inversion_output_save", path=inversion_path):
            inv_out.save(inversion_path)
        paths[f"inversion_output{suffix}_path"] = inversion_path
    if trace_path is not None:
        trace_path = trace_path.with_name(f"{trace_path.stem}{suffix}{trace_path.suffix}")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with timed("rhime.output.trace_save", path=trace_path):
            save_trace(inv_out.trace, trace_path)
        paths[f"trace{suffix}_path"] = trace_path
    return paths
