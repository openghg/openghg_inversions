"""Contains functions for running all steps of the MCMC inversion using PyMC.

This module handles getting data, filtering, applying basis functions, sampling,
and processing the outputs.

Notes
-----
If not using on an HPC in the terminal you should do::

    export OPENBLAS_NUM_THREADS=XX

and/or::

    export OMP_NUM_THREADS=XX

where XX is the number of chains you are running.

If running in Spyder do this before launching Spyder, else you will use every
available thread. Apart from being annoying it will also slow down your run
due to unnecessary forking.

RHIME with OpenGHG expects ALL data to already be included in the
object stores and for the paths to object stores to already be set in
the users OpenGHG config file (default location: ~/.openghg/openghg.conf).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal, cast
import warnings

import arviz as az
import numpy as np
import xarray as xr

import openghg_inversions.hbmcmc.inversion_pymc as mcmc
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.models.priors import lognormal_mu_sigma
from openghg_inversions.utils import ncdf_encoding
from openghg_inversions.inversion_data import FixedBasisPreparedData, prepare_fixedbasis_inversion_data
from openghg_inversions.postprocessing.inversion_output import (
    InversionOutput,
    _reset_serialisation_multiindexes,
)


def update_log_normal_prior(prior):
    """Convert `mean` and `stdev` to parameters for PyMC lognormal."""
    if prior["pdf"].lower() == "lognormal" and "stdev" in prior:
        stdev = float(prior["stdev"])
        mean = float(prior.get("mean", 1.0))

        mu, sigma = lognormal_mu_sigma(mean, stdev)
        prior["mu"] = mu
        prior["sigma"] = sigma

        del prior["stdev"]
        if "mean" in prior:
            del prior["mean"]
    return prior


def _extract_post_process_args(inv_inputs: xr.Dataset) -> dict[str, object]:
    """Extract legacy-shaped postprocessing arrays from inversion inputs.

    NOTE: Transitional compatibility bridge. This extracts legacy-shaped arrays
    from already-prepared inversion inputs; it is not a core data-preparation step.

    Args:
        inv_inputs: Dataset produced by ``make_inv_inputs``.

    Returns:
        Dictionary of legacy-shaped arrays still expected by postprocessing
        functions.
    """
    y = inv_inputs.mf.values
    obs_prior_factor = inv_inputs.mf_prior_factor.values if "mf_prior_factor" in inv_inputs else None
    obs_prior_upper_level_factor = (
        inv_inputs.mf_prior_upper_level_factor.values if "mf_prior_upper_level_factor" in inv_inputs else None
    )
    return {
        "Ytime": inv_inputs.time.values,
        "obs_repeatability": inv_inputs.mf_repeatability.values,
        "obs_variability": inv_inputs.mf_variability.values,
        "obs_prior_factor": obs_prior_factor if obs_prior_factor is not None else np.zeros_like(y),
        "obs_prior_upper_level_factor": (
            obs_prior_upper_level_factor if obs_prior_upper_level_factor is not None else np.zeros_like(y)
        ),
        "Hx": inv_inputs.H.values,
        "Y": inv_inputs.mf.values,
        "error": inv_inputs.mf_error.values,
        "siteindicator": inv_inputs.site_indicator.values,
        "sigma_freq_index": inv_inputs.sigma_freq_index.values,
        "min_error": inv_inputs.min_error.values,
    }


def _require_fixedbasis_inv_inputs(prepared: FixedBasisPreparedData) -> xr.Dataset:
    """Return prepared model inputs or raise for an invalid fixedbasis contract."""
    if prepared.inv_inputs is None:
        raise RuntimeError("Fixed-basis data preparation did not produce model inputs.")
    return prepared.inv_inputs


def _require_fixedbasis_legacy_data(prepared: FixedBasisPreparedData) -> dict:
    """Return legacy fixedbasis forward data required by postprocessing."""
    if prepared.fp_data is None:
        raise RuntimeError("Fixed-basis data preparation did not produce forward data.")
    missing_legacy_keys = [key for key in (".basis", ".flux") if key not in prepared.fp_data]
    if missing_legacy_keys:
        raise RuntimeError(
            "Fixed-basis data preparation did not produce legacy fixed-basis data. "
            f"Missing key(s): {missing_legacy_keys!r}."
        )
    return prepared.fp_data


def _require_fixedbasis_emissions_basis(prepared: FixedBasisPreparedData) -> BasisFunctions:
    """Return retained emissions basis functions or raise for an invalid fixedbasis contract."""
    try:
        basis_functions = prepared.basis_objects["emissions"]
    except KeyError as exc:
        raise RuntimeError("Fixed-basis data preparation did not retain emissions BasisFunctions.") from exc

    if not isinstance(basis_functions, BasisFunctions):
        raise RuntimeError(
            f"Fixed-basis data preparation returned invalid emissions basis object {type(basis_functions)!r}."
        )
    return basis_functions


def _canonicalize_fixedbasis_trace(trace: object, basis_functions: BasisFunctions) -> object:
    """Rename legacy fixedbasis trace dims back to modern model dims."""
    if not isinstance(trace, az.InferenceData):
        return trace

    rename_map = {
        "nx": basis_functions.operator.meta.state_dim,
        "nbc": "bc_region",
    }
    renamed_groups: dict[str, xr.Dataset] = {}

    for group in trace.groups():
        ds = trace[group]
        applicable = {
            old: new
            for old, new in rename_map.items()
            if old != new
            and (old in ds.dims or old in ds.coords)
            and new not in ds.dims
            and new not in ds.coords
        }
        renamed_groups[group] = ds.rename(applicable) if applicable else ds.copy()

    return cast(Any, az.InferenceData)(**renamed_groups)


def _inv_inputs_from_rerun_arrays(
    *,
    Hx: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    siteindicator: np.ndarray,
    sigma_freq_index: np.ndarray,
    Ytime: np.ndarray,
    Hbc: np.ndarray | None = None,
    min_error: float | np.ndarray = 0.0,
) -> xr.Dataset:
    """Build a minimal inferpymc-compatible dataset from saved output arrays.

    NOTE: this is a temporary fix for `rerun_output` while `inferpymc` is being refactored.
    This should be removed once `rerun_output` is updated.

    Args:
        Hx: Emissions sensitivity array with shape ``(region, nmeasure)``.
        Y: Observation vector.
        error: Observation error vector.
        siteindicator: Site indicator for each observation.
        sigma_freq_index: Sigma-period indicator for each observation.
        Ytime: Observation timestamps.
        Hbc: Optional BC sensitivity array with shape ``(bc_region, nmeasure)``.
        min_error: Minimum error values to attach to the dataset.

    Returns:
        Minimal dataset compatible with the dataset-first ``inferpymc`` path.
    """
    nmeasure = len(Y)
    coords: dict[str, object] = {
        "nmeasure": np.arange(nmeasure),
        "time": ("nmeasure", Ytime),
    }
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "H": (("region", "nmeasure"), Hx),
        "mf": (("nmeasure",), Y),
        "mf_error": (("nmeasure",), error),
        "site_indicator": (("nmeasure",), siteindicator.astype(int)),
        "sigma_freq_index": (("nmeasure",), sigma_freq_index.astype(int)),
        "min_error": (
            ("nmeasure",),
            np.broadcast_to(np.asarray(min_error), (nmeasure,)),
        ),
    }
    coords["region"] = np.arange(Hx.shape[0])

    if Hbc is not None:
        data_vars["H_bc"] = (("bc_region", "nmeasure"), Hbc)
        coords["bc_region"] = np.arange(Hbc.shape[0])

    return xr.Dataset(data_vars=data_vars, coords=coords)


# ------------------------------------------------------------
# Output format handling
# ------------------------------------------------------------


@dataclass
class _OutputContext:
    """Resolved output settings and runtime objects for final output handling."""

    output_format: str
    outputpath: str
    outputname: str
    species: str
    domain: str
    start_date: str
    averaging_period: list[str | None]
    use_bc: bool
    country_file: str | None
    paris_postprocessing_kwargs: dict | None
    legacy_postprocess_args: dict
    mcmc_args: dict
    mcmc_results: dict
    inversion_output_args: dict
    inv_out: InversionOutput | None = None
    paths: dict[str, Path] = field(default_factory=dict)


def _resolve_output_format(
    output_format: str,
    *,
    paris_postprocessing: bool,
    is_column: bool,
) -> str:
    """Resolve deprecated aliases and validate the canonical output format."""
    if paris_postprocessing is True:
        output_format = "paris"
        warnings.warn(
            "The `paris_postprocessing` argument will be deprecated. Use `output_format = 'paris'` instead."
        )

    resolved_output_format = output_format.lower()
    if resolved_output_format in {"hbmcmc", "hbmcmc_postprocessing"}:
        warnings.warn(
            f"output_format={resolved_output_format!r} is deprecated; use output_format='legacy' instead.",
            UserWarning,
            stacklevel=2,
        )
        resolved_output_format = "legacy"

    if is_column and resolved_output_format == "legacy":
        raise ValueError(
            "Legacy HBMCMC output formatting is not supported for column observations; "
            "use output_format='inv_out' or a modern RHIME output format."
        )

    return resolved_output_format


def _resolve_trace_path(
    save_trace: str | Path | bool, outputpath: str, outputname: str, start_date: str
) -> Path | None:
    """Resolve the output path for a saved trace file."""
    if not save_trace:
        return None
    if isinstance(save_trace, str | Path):
        return Path(save_trace)
    return Path(outputpath) / (outputname + f"{start_date}_trace.nc")


def _resolve_inversion_output_path(
    save_inversion_output: str | Path | bool, outputpath: str, outputname: str, start_date: str
) -> Path | None:
    """Resolve the output path for a saved inversion-output file."""
    if not save_inversion_output:
        return None
    if isinstance(save_inversion_output, str | Path):
        return Path(save_inversion_output)
    return Path(outputpath) / (outputname + f"{start_date}_inversion_output.nc")


def _build_output_context(
    *,
    output_format: str,
    outputpath: str,
    outputname: str,
    species: str,
    domain: str,
    start_date: str,
    averaging_period: list[str | None],
    use_bc: bool,
    country_file: str | None,
    paris_postprocessing_kwargs: dict | None,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    legacy_postprocess_args: dict,
    mcmc_args: dict,
    mcmc_results: dict,
    inversion_output_args: dict,
) -> _OutputContext:
    """Build the output context used by later output-handling stages.

    Args:
        output_format: Canonical output mode string after normalization.
        outputpath: Directory for saved outputs.
        outputname: Base output name for saved files.
        species: Species identifier for output naming.
        domain: Domain identifier for output naming.
        start_date: Inversion start date used in output naming.
        averaging_period: Observation averaging period list for PARIS outputs.
        use_bc: Whether boundary-condition outputs are enabled.
        country_file: Optional country definition file passed to postprocessing.
        paris_postprocessing_kwargs: Optional keyword arguments for PARIS output creation.
        save_trace: Trace save setting passed to ``fixedbasisMCMC``.
        save_inversion_output: InversionOutput save setting passed to ``fixedbasisMCMC``.
        legacy_postprocess_args: Legacy postprocessing argument dictionary built during inversion setup.
        mcmc_args: Arguments passed to ``mcmc.inferpymc``.
        mcmc_results: Raw sampler results from ``mcmc.inferpymc``.
        inversion_output_args: Arguments needed to build modern ``InversionOutput``.

    Returns:
        _OutputContext: Context object for final output handling.
    """
    paths: dict[str, Path] = {}

    trace_path = _resolve_trace_path(save_trace, outputpath, outputname, start_date)
    if trace_path is not None:
        paths["trace"] = trace_path

    inversion_output_path = _resolve_inversion_output_path(
        save_inversion_output, outputpath, outputname, start_date
    )
    if inversion_output_path is not None:
        paths["inversion_output"] = inversion_output_path

    return _OutputContext(
        output_format=output_format,
        outputpath=outputpath,
        outputname=outputname,
        species=species,
        domain=domain,
        start_date=start_date,
        averaging_period=averaging_period,
        use_bc=use_bc,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
        legacy_postprocess_args=legacy_postprocess_args,
        mcmc_args=mcmc_args,
        mcmc_results=mcmc_results,
        inversion_output_args=inversion_output_args,
        paths=paths,
    )


def _build_inversion_output_args(
    *,
    prepared: FixedBasisPreparedData,
    legacy_postprocess_args: dict,
    mcmc_args: dict,
    mcmc_results: dict,
    sites: list[str],
    averaging_period: list[str | None],
    start_date: str,
    end_date: str,
    species: str,
    domain: str,
    output_format: str,
    outputpath: str,
    outputname: str,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
) -> dict:
    """Build explicit arguments for fixedbasis modern ``InversionOutput``.

    Args:
        prepared: Prepared fixedbasis data including retained basis functions.
        legacy_postprocess_args: Legacy postprocessing values computed alongside inversion inputs.
        mcmc_args: Arguments passed to ``mcmc.inferpymc``.
        mcmc_results: Raw inversion outputs from ``mcmc.inferpymc``.
        sites: Site names used in the inversion.
        averaging_period: Observation averaging periods used in the inversion.
        start_date: Inversion start date.
        end_date: Inversion end date.
        species: Species name for output metadata.
        domain: Domain name for output metadata.
        output_format: Requested fixedbasis output format.
        outputpath: Directory for saved outputs.
        outputname: Base output name for saved files.
        save_trace: Trace save setting passed to ``fixedbasisMCMC``.
        save_inversion_output: InversionOutput save setting passed to ``fixedbasisMCMC``.

    Returns:
        dict: Explicit keyword arguments for ``InversionOutput``.
    """
    basis_functions = _require_fixedbasis_emissions_basis(prepared)
    return {
        "trace": _canonicalize_fixedbasis_trace(mcmc_results["trace"], basis_functions),
        "inv_inputs": _require_fixedbasis_inv_inputs(prepared),
        "basis_functions": basis_functions,
        "run_metadata": {
            "start_date": start_date,
            "end_date": end_date,
            "sites": list(sites),
            "averaging_period": list(averaging_period),
            "split_by_sectors": False,
            "basis_artifact_source": prepared.basis_artifact_source,
            "basis_artifact_path": prepared.basis_artifact_path,
        },
        "model_metadata": {"species": species, "domain": domain},
        "output_metadata": {
            "output_format": output_format,
            "output_path": outputpath,
            "output_name": outputname,
            "save_trace": save_trace,
            "save_inversion_output": save_inversion_output,
            "legacy_hbmcmc_attrs": _legacy_hbmcmc_attrs_from_mcmc_args(mcmc_args),
        },
        "provenance": {
            "contract": "modern_fixedbasis_inversion_output",
            "compatibility_issue": "416",
            "basis_representation": "operator-backed",
            "legacy_postprocessing_fields": sorted(str(key) for key in legacy_postprocess_args),
        },
    }


def _format_legacy_prior_attr(prior: object) -> str | None:
    """Format a prior dictionary using the historical HBMCMC attribute shape."""
    if not isinstance(prior, dict):
        return None
    return ",".join(f"{key},{value}" for key, value in prior.items())


def _legacy_hbmcmc_attrs_from_mcmc_args(mcmc_args: dict) -> dict[str, str]:
    """Return legacy output attrs that are still needed by old fixedbasis workflows."""
    attrs = {
        "Burn in": str(int(mcmc_args["burn"])),
        "Tuning steps": str(int(mcmc_args["tune"])),
        "Number of chains": str(int(mcmc_args["nchain"])),
        "Error for each site": str(mcmc_args["sigma_per_site"]),
    }

    prior_attrs = {
        "Emissions Prior": _format_legacy_prior_attr(mcmc_args.get("xprior")),
        "Model error Prior": _format_legacy_prior_attr(mcmc_args.get("sigprior")),
    }
    if mcmc_args.get("use_bc"):
        prior_attrs["BCs Prior"] = _format_legacy_prior_attr(mcmc_args.get("bcprior"))
    if mcmc_args.get("add_offset"):
        prior_attrs["Offset Prior"] = _format_legacy_prior_attr(mcmc_args.get("offsetprior"))

    attrs.update({name: value for name, value in prior_attrs.items() if value is not None})
    return attrs


def _get_inversion_output(context: _OutputContext) -> InversionOutput:
    """Build and cache the modern InversionOutput object for output handling."""
    if context.inv_out is None:
        context.inv_out = InversionOutput(**context.inversion_output_args)
    return context.inv_out


def _handle_core_output_artifacts(context: _OutputContext) -> None:
    """Write core inversion artifacts needed before final output dispatch."""
    trace_path = context.paths.get("trace")
    if trace_path is not None:
        trace = context.mcmc_results["trace"]
        if isinstance(trace, az.InferenceData):
            trace = cast(Any, az.InferenceData)(
                **{group: _reset_serialisation_multiindexes(trace[group]) for group in trace.groups()}
            )
        trace.to_netcdf(str(trace_path), engine="netcdf4", compress=True)

    inversion_output_path = context.paths.get("inversion_output")
    if inversion_output_path is not None:
        _get_inversion_output(context).save(inversion_output_path)


def _finalize_output(context: _OutputContext) -> xr.Dataset | dict | InversionOutput:
    """Dispatch the final output path for a completed inversion run."""
    if context.output_format == "mcmc_results":
        return context.mcmc_results

    if context.output_format == "inv_out":
        return _get_inversion_output(context)

    start_post = time.time()

    if context.output_format == "basic":
        from ..postprocessing.make_outputs import basic_output

        outputs = basic_output(_get_inversion_output(context), country_file=context.country_file)
        end_post = time.time()
        print(f"Post processing Complete. Time taken = {end_post - start_post:.2f} seconds")
        return outputs

    if context.output_format == "legacy":
        from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename

        from ..postprocessing.legacy_outputs import make_legacy_hbmcmc_output

        outputs = make_legacy_hbmcmc_output(
            inv_out=_get_inversion_output(context),
            country_file=context.country_file,
            use_bc=context.use_bc,
        )
        output_filename = define_output_filename(
            context.outputpath,
            context.species,
            context.domain,
            context.outputname,
            context.start_date,
            ext=".nc",
        )
        Path(context.outputpath).mkdir(parents=True, exist_ok=True)
        outputs.to_netcdf(output_filename, encoding=ncdf_encoding(outputs), mode="w")
        end_post = time.time()
        print(f"Post processing Complete. Time taken = {end_post - start_post:.2f} seconds")
        return outputs

    if context.output_format == "paris":
        from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename

        from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

        obs_avg_period = context.averaging_period[0] or "0h"
        if not context.averaging_period[0]:
            logging.info("Default obs averaging period %s used in PARIS post-processing.", obs_avg_period)
        paris_postprocessing_kwargs = context.paris_postprocessing_kwargs or {}
        flux_outs, conc_outs = make_paris_outputs(
            _get_inversion_output(context),
            country_file=context.country_file,
            domain=context.domain,
            obs_avg_period=obs_avg_period,
            **paris_postprocessing_kwargs,
        )

        conc_output_filename = define_output_filename(
            context.outputpath,
            context.species,
            context.domain,
            context.outputname + "_conc",
            context.start_date,
            ext=".nc",
        )
        flux_output_filename = define_output_filename(
            context.outputpath,
            context.species,
            context.domain,
            context.outputname + "_flux",
            context.start_date,
            ext=".nc",
        )
        Path(context.outputpath).mkdir(parents=True, exist_ok=True)

        conc_outs.to_netcdf(
            conc_output_filename, unlimited_dims=["time"], mode="w", encoding=ncdf_encoding(conc_outs)
        )
        flux_outs.to_netcdf(
            flux_output_filename, unlimited_dims=["time"], mode="w", encoding=ncdf_encoding(flux_outs)
        )

        logging.info("PARIS concentration outputs saved to %s", conc_output_filename)
        logging.info("PARIS flux outputs saved to %s", flux_output_filename)

        end_post = time.time()
        print(f"Post processing Complete. Time taken = {end_post - start_post:.2f} seconds")
        return xr.merge([conc_outs, flux_outs.rename(time="flux_time")])

    raise ValueError(f"Unsupported fixedbasisMCMC output_format {context.output_format!r}.")


# ------------------------------------------------------------
# Main MCMC script
# ------------------------------------------------------------


def fixedbasisMCMC(
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None],
    start_date: str,
    end_date: str,
    outputpath: str,
    outputname: str,
    bc_store: str = "user",  # Do we want to set defaults for the object stores?
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: list | None = None,
    fp_model: str | None = None,  # Changed to none. When "NAME" specified FPs are not found
    fp_height: list[str] | None = None,
    fp_species: str | None = None,
    emissions_name: list[str] | None = None,
    inlet: list[str] | None = None,
    instrument: list[str] | None = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: list | None = None,
    platform: list[str | None] | str | None = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | None = None,
    country_directory: str | None = None,
    country_file: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    xprior: dict = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 1.0, "lower": 0.0},
    bcprior: dict = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
    sigprior: dict = {"pdf": "uniform", "lower": 0.1, "upper": 3},
    offsetprior: dict = {"pdf": "normal", "mu": 0, "sigma": 1},
    offset_args: dict | None = None,
    nit: int = int(2.5e5),
    burn: int = 50000,
    tune: int = int(1.25e5),
    nchain: int = 2,
    filters: None | list | dict[str, list[str] | None] = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    country_unit_prefix: str | None = None,
    add_offset: bool = False,
    verbose: bool = False,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    save_trace: str | Path | bool = False,
    save_inversion_output: str | Path | bool = False,
    min_error: Literal["percentile", "residual"] | dict[str, float] | None | float = 0.0,
    calculate_min_error: Literal["percentile", "residual"] | None = None,
    min_error_options: dict | None = None,
    output_format: Literal[
        "hbmcmc",
        "hbmcmc_postprocessing",
        "legacy",
        "paris",
        "basic",
        "merged_data",
        "inv_out",
        "mcmc_args",
        "mcmc_results",
    ] = "legacy",
    paris_postprocessing: bool = False,
    paris_postprocessing_kwargs: dict | None = None,
    power: dict | float = 1.99,
    return_basis_objects: bool = False,
    **kwargs,
) -> xr.Dataset | dict | InversionOutput:
    """Script to run hierarchical Bayesian MCMC (RHIME) for inference of emissions.

    Uses PyMC to solve the inverse problem. This is now a legacy compatibility
    entry point; new workflows should call ``run_rhime`` directly.

    Args:
        species: Atmospheric trace gas species of interest (e.g. 'co2').
        sites: List of measurement site names.
        domain: Model domain. (NB. Does not necessarily correspond to the inversion domain)
        averaging_period: Averaging period of observations (must match number of sites).
        start_date: Start time of inversion: "YYYY-mm-dd".
        end_date: End time of inversion: "YYYY-mm-dd".
        outputname: Unique identifier for output/run name.
        outputpath: Path to where output should be saved.
        bc_store: Name of object store containing boundary conditions files.
        obs_store: Name of object store containing measurements files.
        footprint_store: Name of object store containing footprints files.
        emissions_store: Name of object store containing emissions/flux files.
        met_model: Meteorological model used in the LPDM (e.g. 'ukv').
        fp_model: LPDM used for generating footprints (e.g. 'NAME').
        fp_height: Inlet height modelled for sites in LPDM (must match number of sites).
        fp_species: Species name associated with footprints in the object store.
        emissions_name: List of keyword "source" args used for retrieving emissions files
            from 'emissions_store'.
        inlet: Specific inlet height for the site (must match number of sites).
        instrument: Specific instrument for the site (must match number of sites).
        calibration_scale: Calibration scale to use for measurements data.
        obs_data_level: Data quality level for measurements data. (must match number of sites)
        use_tracer: Option to use inverse model that uses tracers of species
            (e.g. d13C, CO, C2H4).
        use_bc: When True, use and infer boundary conditions.
        fp_basis_case: Name of basis function to use for emission.
        basis_directory: Directory containing the basis function.
        bc_basis_case: Name of basis case type for boundary conditions (NOTE, I don't
            think that currently you can do anything apart from scaling NSEW
            boundary conditions if you want to scale these monthly.)
        bc_basis_directory: Directory containing the boundary condition basis functions
            (e.g. files starting with "NESW").
        country_directory: Directory containing land-sea and InTEM outer region files for deriving
            basis functions. If None, will use default files.
        country_file: Path to the country definition file.
        bc_input: Variable for calling BC data from 'bc_store' - equivalent of
            'emissions_name' for fluxes.
        basis_algorithm: Select basis function algorithm for creating basis
            function file for emissions on the fly. Current fixedbasis and
            ``run_hbmcmc.py`` configs support ``"quadtree"`` or
            ``"weighted"``; ``"weighted"`` distinguishes between land-sea
            regions. The lower-level Python basis API also supports
            ``"region_constrained"`` when a caller supplies an already loaded
            ``region_classes`` field.
        nbasis: Number of basis functions that you want if using quadtree derived
            basis function. This will optimise to closest value that fits with
            quadtree splitting algorithm, i.e. nbasis % 4 = 1.
        xprior: Dictionary containing information about the prior PDF for emissions.
            The entry "pdf" is the name of the analytical PDF used, see
            https://docs.pymc.io/api/distributions/continuous.html for PDFs
            built into pymc3, although they may have to be coded into the script.
            The other entries in the dictionary should correspond to the shape
            parameters describing that PDF as the online documentation,
            e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sd":1}.
            Note that the standard deviation should be used rather than the
            precision. Currently all variables are considered iid.
        bcprior: Same as xprior but for boundary conditions.
        sigprior: Same as xprior but for model error.
        offsetprior: Same as xprior but for bias offset. Only used is addoffset=True.
        offset_args: Dictionary of args to pass to `make_offset`. For instance
            `{"drop_first": False}` will put an offset on all site (rather than using 0
            offset for the first site). If "offset_freq" is passed, then the
            offset will be applied at the specified frequency (e.g. monthly).
        nit: Number of iterations for MCMC.
        burn: Number of iterations to burn/discard in MCMC.
        tune: Number of iterations to use to tune step size.
        nchain: Number of independent chains to run (there is no way at all of
            knowing whether your distribution has converged by running only
            one chain).
        filters: List of filters to apply to all sites, or dictionary with sites as keys
            and a list of filters for each site, e.g. filters = {"MHD": ["pblh_inlet_diff",
            "pblh_min"], "JFJ": None}.
        fix_basis_outer_regions: When set to True uses InTEM regions to derive basis functions for inner region.
            Default False.
        averaging_error: Adds the variability in the averaging period to the measurement
            error if set to True.
        bc_freq: The perdiod over which the baseline is estimated. Set to "monthly"
            to estimate per calendar month; set to a number of days,
            as e.g. "30D" for 30 days; or set to None to estimate to have one
            scaling for the whole inversion period.
        sigma_freq: As bc_freq, but for model sigma.
        sigma_per_site: Whether a model sigma value will be calculated for each site
            independantly (True) or all sites together (False).
            Default: True.
        country_unit_prefix: A prefix for scaling the country emissions. Current options are:
            'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
            To add additional options add to convert.prefix.
            Default is none and no scaling will be applied (output in g).
        add_offset: Add an offset (intercept) to all sites but the first in the site list.
            Default False.
        verbose: When True, prints progress bar of mcmc.inferpymc.
        reload_merged_data: If True, reads fp_all object from a pickle file, instead of rerunning get_data.
        save_merged_data: If True, saves the merged data object (fp_all) as a pickle file.
        merged_data_dir: Path to a directory of merged data objects. For saving to or reading from.
        merged_data_name: Name of files in which are the merged data objects. For saving to or reading from.
        basis_output_path: If set, save the basis functions to this path. Used for testing.
        save_trace: If True, save arviz `InferenceData` trace to `outputpath`. Alternatively,
            a file path (including file name and extension) can be passed, and the trace will be
            saved there.
        merged_data_only: If True, save merged data, and do nothing else.
        min_error: If float, the value represents the minimun error. Otherwise, compute min model error
            using the "residual" method or the "percentile" method. (See `openghg_inversions.model_error.py` for
            details.) Combines the functionality of the previous min_error and calculate_min_error parameters.
            None only an option to accomodate old ini files.
        calculate_min_error: Is deprecated and will be removed in a future update.
        min_error_options: Dictionary of additional arguments to pass the the function used to calculate min. model
            error (as specified by `min_error`).
        output_format: Select what is returned/saved by inversion.
            - "legacy": (default) return old HBMCMC-compatible output formatting, computed from modern
              `InversionOutput`, and save result as netCDF. Deprecated aliases: "hbmcmc" and
              "hbmcmc_postprocessing".
            - "merged_data": return `fp_all` dictionary, no further processing and inversion *not* run
            - "inv_out": return modern `InversionOutput` object
            - "basic": return basic output created by new `postprocessing` submodule
            - "paris": return flux and concentration datasets with PARIS formatting; these are also saved
              as netCDF files in the directory `outputpath`
            - "mcmc_args": return the arguments passed to `fixedbasisMCMC`, but do not run the inversion
            - "mcmc_results": return the results of `fixedbasisMCMC` with no further processing
        paris_postprocessing_kwargs: Dict of kwargs to pass to `make_paris_outputs`.
        power: Power to raise pollution event size to if using pollution events from obs. Default is 1.99.
        return_basis_objects: If True, include retained basis objects in ``output_format="mcmc_args"``
            debug output. Fixedbasis output modes that construct modern inversion output retain them
            internally regardless of this setting. They are not passed to ``inferpymc``.

    Returns:
        xr.Dataset | dict: Results from the inversion in a Dataset if skip_post_processing==False,
            in a dictionary if True.
    """
    # Check if any observations are column based.
    if inlet is not None:
        is_column = any(i == "column" for i in inlet)
    else:
        is_column = False

    output_format = cast(
        Literal[
            "legacy",
            "paris",
            "basic",
            "merged_data",
            "inv_out",
            "mcmc_args",
            "mcmc_results",
        ],
        _resolve_output_format(
            output_format,
            paris_postprocessing=paris_postprocessing,
            is_column=is_column,
        ),
    )
    needs_modern_inv_out = output_format not in {"merged_data", "mcmc_args"} or bool(save_inversion_output)

    start_data = time.time()
    prepared = prepare_fixedbasis_inversion_data(
        species=species,
        sites=sites,
        domain=domain,
        averaging_period=averaging_period,
        start_date=start_date,
        end_date=end_date,
        output_name=outputname,
        flux_sources=emissions_name,
        split_by_sectors=False,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        met_model=met_model,
        fp_model=fp_model,
        fp_height=fp_height,
        fp_species=fp_species,
        inlet=inlet,
        instrument=instrument,
        max_level=max_level,
        calibration_scale=calibration_scale,
        obs_data_level=obs_data_level,
        platform=platform,
        use_tracer=use_tracer,
        use_bc=use_bc,
        fp_basis_case=fp_basis_case,
        basis_directory=basis_directory,
        bc_basis_case=bc_basis_case,
        bc_basis_directory=bc_basis_directory,
        country_directory=country_directory,
        bc_input=bc_input,
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        filters=filters,
        fix_basis_outer_regions=fix_basis_outer_regions,
        averaging_error=averaging_error,
        bc_freq=bc_freq,
        sigma_freq=sigma_freq,
        reload_merged_data=False if output_format == "merged_data" else reload_merged_data,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        basis_output_path=basis_output_path,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_options=min_error_options,
        return_basis_objects=return_basis_objects or needs_modern_inv_out,
        merged_data_only=output_format == "merged_data",
    )

    if output_format == "merged_data":
        return prepared.fp_all  # type: ignore

    inv_inputs = _require_fixedbasis_inv_inputs(prepared)
    sites = prepared.sites
    averaging_period = prepared.averaging_period

    # TODO keep this config separate from mcmc_args in the future
    mcmc_config = {
        "inv_inputs": inv_inputs,
        "xprior": update_log_normal_prior(xprior),
        "sigprior": sigprior,
        "nit": nit,
        "burn": burn,
        "tune": tune,
        "nchain": nchain,
        "sigma_per_site": sigma_per_site,
        "offsetprior": offsetprior,
        "add_offset": add_offset,
        "offset_args": offset_args,
        "power": power,
        "use_bc": use_bc,
        "verbose": verbose,
    }
    if use_bc:
        mcmc_config["bcprior"] = update_log_normal_prior(bcprior)

    mcmc_args = mcmc_config.copy()
    # add any additional kwargs to mcmc_args (these aren't needed for post processing)
    mcmc_args.update(kwargs)
    return_mcmc_args = mcmc_args.copy()
    if return_basis_objects:
        return_mcmc_args["basis_objects"] = prepared.basis_objects

    end_data = time.time()

    print(f"Data extraction and preparation complete. Time taken = {end_data - start_data:.2f} seconds")

    # for debugging
    if output_format == "mcmc_args":
        return return_mcmc_args

    fp_data = _require_fixedbasis_legacy_data(prepared)
    legacy_postprocess_args = _extract_post_process_args(inv_inputs)

    legacy_postprocess_args.update(
        {
            "domain": domain,
            "species": species,
            "sites": sites,
            "start_date": start_date,
            "end_date": end_date,
            "outputname": outputname,
            "outputpath": outputpath,
            "country_unit_prefix": country_unit_prefix,
            "fp_data": fp_data,
            "emissions_name": emissions_name,
            "country_file": country_file,
        }
    )

    if use_bc and "H_bc" in inv_inputs:
        legacy_postprocess_args["Hbc"] = inv_inputs.H_bc.values

    # cast float64 to float32
    for k in list(legacy_postprocess_args.keys()):  # use list to get keys before modifying dict
        v = legacy_postprocess_args[k]
        if isinstance(v, np.ndarray) and v.dtype == "float64":
            legacy_postprocess_args[k] = v.astype("float32")

    start_inversion = time.time()

    # Run PyMC inversion
    mcmc_results = mcmc.inferpymc(**mcmc_args)  # type: ignore

    end_inversion = time.time()

    print(f"MCMC Inversion complete. Time taken = {end_inversion - start_inversion:.2f} seconds")

    inversion_output_args = _build_inversion_output_args(
        prepared=prepared,
        legacy_postprocess_args=legacy_postprocess_args,
        mcmc_args=mcmc_args,
        mcmc_results=mcmc_results,
        sites=sites,
        averaging_period=averaging_period,
        start_date=start_date,
        end_date=end_date,
        species=species,
        domain=domain,
        output_format=output_format,
        outputpath=outputpath,
        outputname=outputname,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
    )

    output_context = _build_output_context(
        output_format=output_format,
        outputpath=outputpath,
        outputname=outputname,
        species=species,
        domain=domain,
        start_date=start_date,
        averaging_period=averaging_period,
        use_bc=use_bc,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        legacy_postprocess_args=legacy_postprocess_args,
        mcmc_args=mcmc_args,
        mcmc_results=mcmc_results,
        inversion_output_args=inversion_output_args,
    )
    _handle_core_output_artifacts(output_context)

    return _finalize_output(output_context)


def rerun_output(input_file: str, outputname: str, outputpath: str, verbose: bool = False) -> None:
    """Rerun the MCMC code using inputs from a previous output.

    This allows reproducibility of results without the need to transfer all raw input files.

    Args:
        input_file: Full path to previously written ncdf file.
        outputname: Unique identifier new for output/run name.
        outputpath: Path to where output should be saved.
        verbose: When True, prints progress bar of mcmc.inferpymc.

    Note:
        At the moment fluxapriori in the output is the mean apriori flux
        over the inversion period and so will not be identical to the
        original a priori flux, if it varies over the inversion period.

    TODO: replace this legacy-output replay path with an explicit modern rerun input.
    """

    def isFloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    ds_in = xr.load_dataset(input_file)

    # Read inputs from ncdf output
    start_date = ds_in.attrs["Start date"]
    end_date = ds_in.attrs["End date"]
    Hx = ds_in.xsensitivity.values.T
    Hbc = ds_in.bcsensitivity.values.T
    Y = ds_in.Yobs.values
    Ytime = ds_in.Ytime.values
    error = ds_in.Yerror.values
    siteindicator = ds_in.siteindicator.values
    sigma_freq_index = ds_in.sigmafreqindex.values
    xprior_string = ds_in.attrs["Emissions Prior"].split(",")
    xprior = {k: float(v) if isFloat(v) else v for k, v in zip(xprior_string[::2], xprior_string[1::2])}
    bcprior_string = ds_in.attrs["BCs Prior"].split(",")
    bcprior = {k: float(v) if isFloat(v) else v for k, v in zip(bcprior_string[::2], bcprior_string[1::2])}
    sigprior_string = ds_in.attrs["Model error Prior"].split(",")
    sigprior = {k: float(v) if isFloat(v) else v for k, v in zip(sigprior_string[::2], sigprior_string[1::2])}
    if "Offset Prior" in ds_in.attrs:
        offsetprior_string = ds_in.attrs["Offset Prior"].split(",")
        offsetprior = {
            k: float(v) if isFloat(v) else v
            for k, v in zip(offsetprior_string[::2], offsetprior_string[1::2])
        }
        add_offset = True
    else:
        add_offset = False
        offsetprior = None
    nit = len(ds_in.steps)
    burn = int(ds_in.attrs["Burn in"])
    tune = int(ds_in.attrs["Tuning steps"])
    nchain = int(ds_in.attrs["Number of chains"])
    sigma_per_site = ds_in.attrs["Error for each site"] == "True"
    sites = ds_in.sitenames.values

    file_list = input_file.split("/")[-1].split("_")
    species = file_list[0]
    domain = file_list[1]
    if ds_in.countrymean.attrs["units"] != "g":
        country_unit_prefix = ds_in.countrymean.attrs["units"][0]
    else:
        country_unit_prefix = None

    inv_inputs = _inv_inputs_from_rerun_arrays(
        Hx=Hx,
        Hbc=Hbc,
        Y=Y,
        error=error,
        siteindicator=siteindicator,
        sigma_freq_index=sigma_freq_index,
        Ytime=Ytime,
    )

    mcmc_results = mcmc.inferpymc(
        inv_inputs=inv_inputs,
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        nit=nit,
        burn=burn,
        tune=tune,
        nchain=nchain,
        sigma_per_site=sigma_per_site,
        offsetprior=offsetprior,
        add_offset=add_offset,
        verbose=verbose,
    )

    xouts = mcmc_results["xouts"]
    sigouts = mcmc_results["sigouts"]
    convergence = mcmc_results["convergence"]
    step1 = mcmc_results["step1"]
    step2 = mcmc_results["step2"]
    Ytrace = mcmc_results["Ytrace"]
    OFFSETtrace = mcmc_results["OFFSETtrace"]
    bcouts = mcmc_results.get("bcouts")
    YBCtrace = mcmc_results.get("YBCtrace")

    mcmc.inferpymc_postprocessouts(
        xouts=xouts,
        bcouts=bcouts,
        sigouts=sigouts,
        convergence=convergence,
        Hx=Hx,
        Hbc=Hbc,
        Y=Y,
        error=error,
        Ytrace=Ytrace,
        OFFSETtrace=OFFSETtrace,
        YBCtrace=YBCtrace,
        step1=step1,
        step2=step2,
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        offsetprior=offsetprior,
        Ytime=Ytime,
        siteindicator=siteindicator,
        sigma_freq_index=sigma_freq_index,
        domain=domain,
        species=species,
        sites=sites,
        start_date=start_date,
        end_date=end_date,
        outputname=outputname,
        outputpath=outputpath,
        country_unit_prefix=country_unit_prefix,
        burn=burn,
        tune=tune,
        nchain=nchain,
        sigma_per_site=sigma_per_site,
        emissions_name=None,
        add_offset=add_offset,
        rerun_file=ds_in,
    )
