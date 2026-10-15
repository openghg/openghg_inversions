"""RHIME output construction and persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import pymc as pm
import xarray as xr

from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions._timing import timed
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.output_views import make_inversion_output
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import (
    OutputFilenameConvention,
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeRunSpec,
)
from openghg_inversions.serialization import save_trace
from openghg_inversions.models.priors import prior_records, trace_prior_records
from openghg_inversions.utils import ncdf_encoding, write_netcdf_preserving_bounds_attrs


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


def validate_output_filenames(run_spec: RhimeRunSpec) -> None:
    """Check metadata interpolated into product filenames before writing."""
    for name, value in (("species", run_spec.model.species), ("domain", run_spec.model.domain),
                        ("output_name", run_spec.output.output_name), ("start_date", run_spec.start_date)):
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"{name} must be a non-empty filename component without directories.")


@dataclass
class RhimeResult:
    """Complete result of a concrete RHIME recipe, including coherent CO2.

    Args:
        run_spec: Top-level dates, sites, model, and output settings for the run.
        model_spec: Scientific options used to construct the model.
        output_spec: Output products, paths, and naming settings.
        inv_inputs: Prepared inversion inputs supplied to the model.
        idata: Posterior samples and diagnostics returned by the sampler.
        output_metadata: Output paths, timing, and provenance accumulated during
            the run. A new empty dictionary is used by default.
        outputs: In-memory postprocessing products keyed by product name. A new
            empty dictionary is used by default.
        basis_functions: Basis operator and flux retained for postprocessing.
        model: Concrete sampled PyMC model, when retained.
        inv_out: Modern inversion output, when postprocessing created one.
        sampler: Live sampling configuration, when supplied. Defaults to ``None``.
            Results constructed from saved samples need no sampler. Descriptive sampler
            metadata belongs in ``output_metadata["sampler"]`` for replay.
            A supplied live sampler inserts or overwrites that entry in the supplied
            dictionary at construction; later sampler changes do not refresh it.
        model_build_result: Model, variable-role manifest, and builder metadata,
            when available.
        output_contract: Durable scientific roles and reconstruction mapping.
            Defaults to the live build result contract when one is supplied.
    """

    run_spec: RhimeRunSpec
    model_spec: RhimeModelSpec
    output_spec: RhimeOutputSpec
    inv_inputs: xr.Dataset
    idata: xr.DataTree
    output_metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    basis_functions: BasisFunctions | None = None
    model: pm.Model | None = None
    inv_out: InversionOutput | None = None
    sampler: RhimeSampler | None = None
    model_build_result: RhimeModelBuildResult | None = None
    output_contract: OutputContract | None = None

    def __post_init__(self) -> None:
        """Retain output information independently of the optional live model."""
        if self.sampler is not None:
            self.output_metadata["sampler"] = self.sampler.describe()
        if self.output_contract is None and self.model_build_result is not None:
            self.output_contract = self.model_build_result.output_contract


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


def _save_requested_trace(result: RhimeResult) -> dict[str, str]:
    """Save the sampled trace when requested; return its ``trace_path`` metadata."""
    trace_path = _resolve_output_path(
        result.output_spec.save_trace,
        result.output_spec.output_path,
        f"{result.output_spec.output_name}{result.run_spec.start_date}_trace.nc",
    )
    if trace_path is None:
        return {}
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with timed("rhime.output.trace_save", path=trace_path):
        save_trace(result.idata, trace_path)
    return {"trace_path": str(trace_path)}


def _make_inversion_output(
    *,
    result: RhimeResult,
    prepared: RhimePreparedInputs,
    variable_roles: Mapping[str, str] | None = None,
    state_dimension_mapping: Mapping[str, str] | None = None,
) -> InversionOutput:
    """Create a modern InversionOutput without fixedbasis legacy adapters.

    Args:
        result: Sampled recipe result and model-owned output contract.
        prepared: Retained canonical inputs and basis functions.
        variable_roles: Optional override for the semantic role-to-variable
            mapping. Defaults to ``result.output_contract.variable_roles``.
            Nested RHIME passes a per-domain override here: its builder
            declares tagged roles (``"flux_scale:outer"``, ``"flux_scale:inner"``,
            etc.) so one shared trace can be viewed as two ordinary,
            single-grid ``InversionOutput`` contracts.
        state_dimension_mapping: Optional explicit mapping from the selected
            trace state dimension to the retained basis operator state
            dimension. Nested views use this to normalize selected trace
            variables without modifying the shared sampled trace.

    Returns:
        Complete modern inversion-output artifact.
    """
    contract = result.output_contract
    if contract is None:
        raise ValueError("RHIME result is missing its scientific output contract.")
    builder_metadata = dict(contract.metadata)
    for key in ("model_builder", "likelihood_builder", "likelihood_kwargs"):
        if key in result.output_metadata:
            builder_metadata[key] = result.output_metadata[key]
    contract = replace(
        contract,
        variable_roles=contract.variable_roles if variable_roles is None else variable_roles,
        state_dimension_mapping=(
            contract.state_dimension_mapping if state_dimension_mapping is None else state_dimension_mapping
        ),
        metadata=_structured_metadata(builder_metadata),
    )
    constructed_priors = prior_records(result.model) if result.model is not None else trace_prior_records(result.idata)
    model_metadata = result.model_spec.to_record(priors=constructed_priors)
    model_metadata["constructed_priors"] = constructed_priors
    if model_metadata["likelihood"] is not None:
        model_metadata["likelihood"] = model_metadata["likelihood"]["options"]
    # Keep existing in-memory activity provenance; saved phases reject these
    # explicit policies before writing a sample record.
    for name in ("state_activity", "bc_state_activity"):
        policy = getattr(result.model_spec, name)
        model_metadata[name] = None if policy is None else asdict(policy)
    for sector, record in zip(result.model_spec.sectors, model_metadata["sectors"], strict=True):
        role = "flux_scale" if len(result.model_spec.sectors) == 1 else f"flux_scale:{sector.name}"
        variable = contract.variable_roles.get(role)
        if variable is not None and variable in constructed_priors:
            record["x_prior"] = constructed_priors[variable]
        record["state_activity"] = None if sector.state_activity is None else asdict(sector.state_activity)
    return make_inversion_output(
        prepared=prepared,
        trace=result.idata,
        contract=contract,
        run_metadata={
            "start_date": result.run_spec.start_date,
            "end_date": result.run_spec.end_date,
            "sites": list(result.run_spec.sites),
            "averaging_period": list(result.run_spec.averaging_period),
            "split_by_sectors": result.run_spec.split_by_sectors,
        },
        model_metadata=_structured_metadata(model_metadata),
        output_metadata={
            "output_format": result.output_spec.output_format,
            "output_path": result.output_spec.output_path,
            "output_name": result.output_spec.output_name,
            "save_trace": result.output_spec.save_trace,
            "save_inversion_output": result.output_spec.save_inversion_output,
            "sampler": result.output_metadata.get("sampler"),
        },
        provenance={
            "contract": "modern_rhime_inversion_output",
            "compatibility_issue": "401",
            "basis_representation": "operator-backed",
        },
    )


def make_standard_rhime_outputs(
    *,
    result: RhimeResult,
    prepared: RhimePreparedInputs,
) -> RhimeResult:
    """Create the requested standard RHIME outputs.

    Args:
        result: Sampled standard result. It is not modified.
        prepared: Retained canonical inputs and basis functions.

    Returns:
        A new result holding ``result``'s values plus the inversion output,
        in-memory products and written-file metadata.

    Raises:
        ValueError: If country reporting is requested without a country file.
        FileNotFoundError: If a selected country file is missing or is a directory.
    """
    output_spec = result.output_spec
    validate_output_resources(output_spec)
    run_spec = result.run_spec
    model_spec = result.model_spec
    if output_spec.output_format == "none":
        return result
    validate_output_filenames(run_spec)

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
                write_netcdf_preserving_bounds_attrs(
                    conc_outs, conc_file, unlimited_dims=["index" if "index" in conc_outs.dims else "time"]
                )
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
    output_metadata.update(_save_requested_trace(result))
    return replace(
        result,
        inv_out=inv_out,
        outputs={**result.outputs, **outputs},
        output_metadata={**result.output_metadata, **output_metadata},
    )


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

    Raises:
        ValueError: If country reporting is requested without a country file.
        FileNotFoundError: If a selected country file is missing or is a directory.
    """
    output_spec = result.output_spec
    validate_output_resources(output_spec)
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
    validate_output_filenames(run_spec)
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
    output_metadata.update(_save_requested_trace(result))

    result.inv_out = inv_out
    result.outputs.update(outputs)
    result.output_metadata.update(output_metadata)
