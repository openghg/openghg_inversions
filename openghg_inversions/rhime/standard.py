"""Readable scientific recipe for a standard single-sector RHIME inversion.

The public recipe deliberately spells out its complete execution order. Small
amounts of forwarding shared with the multisector recipe are retained so a
scientist can read or copy this module without reconstructing a pipeline.
"""

from __future__ import annotations

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.basis import prepare_basis
from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.rhime.preparation import acquire_merged_data, resolve_basis_inputs

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from dask import is_dask_collection
from dask.base import unpack_collections

import pandas as pd
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.models.state_activity import prepare_linear_sensitivity
from openghg_inversions.observation_error import (
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.model_error import MinErrorConfig

from ._model_building import (
    ForwardModelTerms,
    add_rhime_likelihood,
    builtin_model_build_result,
)
from .builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
)

from .materialization import (
    _materialize_pymc_inputs_and_priors,
)
from .outputs import (
    make_rhime_output,
    postprocess_standard_outputs,
    save_inversion_output,
    annotate_sample_trace,
    validate_standard_output_resources,
)
from .params import RhimeConfig
from .requests import load_params
from .preparation import (
    assemble_rhime_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
)
from .sampling import sample_rhime_model
from .specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    AdditiveSigmaSettings,
    PollutionEventSettings,
    RhimeModelSpec,
)


_STANDARD_FLUX_INPUT_NAMES = ("H",)
_OBSERVATION_INPUT_NAMES = ("mf", "mf_error")
_BASELINE_INPUT_NAMES = ("H_bc",)


def _require_component_inputs(
    model_inputs: xr.Dataset,
    names: tuple[str, ...],
    *,
    owner: str,
) -> None:
    """Fail before materialization with the selected component named."""
    missing = [name for name in names if name not in model_inputs]
    if missing:
        raise ValueError(f"{owner} requires prepared input(s) {missing!r}.")


def standard_model_input_names(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[str, ...]:
    """Declare arrays required by selected standard-model components.

    Args:
        prepared: Backend-neutral prepared inputs.
        model_spec: Resolved standard-model component options.

    Returns:
        Prepared variable names selected for coordinated materialization.

    Raises:
        ValueError: If a selected component's required input is absent or its
            aggregation-error representation is ambiguous.
    """
    return _standard_input_names(prepared.inv_inputs, model_spec)


def _standard_input_names(model_inputs: xr.Dataset, model_spec: RhimeModelSpec) -> tuple[str, ...]:
    """Own selected standard component names and single-sector layout checks."""
    if len(model_spec.sectors) != 1:
        raise ValueError("Standard model specs must include exactly one sector.")
    _require_component_inputs(
        model_inputs,
        _STANDARD_FLUX_INPUT_NAMES,
        owner="Standard flux component",
    )
    sensitivity = model_inputs["H"]
    source = sensitivity.coords.get("source")
    gathered = any(isinstance(index, pd.MultiIndex) and "source" in index.names
                   for index in sensitivity.indexes.values())
    if "source" in sensitivity.dims or (source is not None and gathered):
        raise ValueError("Standard models require a single-sector prepared H layout.")
    _require_component_inputs(
        model_inputs,
        _OBSERVATION_INPUT_NAMES,
        owner="Observation-error component",
    )
    names = [
        *_STANDARD_FLUX_INPUT_NAMES,
        *_OBSERVATION_INPUT_NAMES,
    ]
    likelihood_inputs = (
        () if model_spec.likelihood is None else model_spec.likelihood.required_prepared_inputs
    )
    if likelihood_inputs:
        _require_component_inputs(
            model_inputs,
            likelihood_inputs,
            owner="Selected built-in likelihood",
        )
        names.extend(likelihood_inputs)
    if model_spec.use_bc:
        _require_component_inputs(
            model_inputs,
            _BASELINE_INPUT_NAMES,
            owner="Standard baseline component selected by `use_bc=True`",
        )
        names.extend(_BASELINE_INPUT_NAMES)
    aggregation_names = aggregation_error_input_names(
        model_inputs,
        model_spec.aggregation_error_mode,
    )
    _require_component_inputs(
        model_inputs,
        aggregation_names,
        owner=(
            "Aggregation-error component selected by "
            f"`aggregation_error_mode={model_spec.aggregation_error_mode!r}`"
        ),
    )
    names.extend(aggregation_names)
    return tuple(names)


def _standard_prior_parameters(model_spec: RhimeModelSpec) -> dict[str, PriorArgs]:
    """Select configured priors for enabled components before state pruning."""
    priors = {"x": model_spec.sectors[0].x_prior}
    if model_spec.use_bc and model_spec.bc_prior is not None:
        priors["bc"] = model_spec.bc_prior
    if model_spec.add_offset and model_spec.offset_prior is not None:
        priors["offset_latent"] = model_spec.offset_prior
    likelihood = model_spec.likelihood
    if isinstance(likelihood, (PollutionEventSettings, AdditiveSigmaSettings)) and likelihood.sigma_prior is not None:
        priors["sigma"] = likelihood.sigma_prior
    if isinstance(likelihood, PollutionEventSettings) and isinstance(likelihood.power, Mapping):
        priors["power"] = likelihood.power
    return priors


def materialize_standard_inputs(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[xr.Dataset, dict[str, PriorArgs]]:
    """Materialize selected standard inputs and enabled priors in one Dask operation.

    The caller owns this explicit PyMC boundary. Preparation and requested model
    settings stay borrowed; only selected arrays and prior parameters are computed.
    Construction subsequently decides which state priors survive pruning.

    Args:
        prepared: Borrowed canonical inputs and retained basis.
        model_spec: Scientific choices selecting the required inputs and priors.

    Returns:
        A shallow copy of the prepared Dataset with selected inputs and their
        coordinates materialized, plus prior parameter mappings keyed by model
        variable name. Unselected preparation variables remain borrowed and may
        be lazy; construction decides which selected priors enter the graph.

    Raises:
        ValueError: If the model has other than one sector or a selected input
            is missing.
    """
    return _materialize_pymc_inputs_and_priors(
        prepared, variable_names=standard_model_input_names(prepared, model_spec),
        prior_parameters=_standard_prior_parameters(model_spec),
    )


def build_standard_model(
    model_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    priors: Mapping[str, PriorArgs],
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sigma_alignment: SigmaAlignment | None = None,
) -> RhimeModelBuildResult:
    """Construct the standard scientific graph from already materialized values.

    Materialize related inputs and enabled priors together before this call,
    for example with :func:`materialize_standard_inputs`. The model settings
    select components; ``priors`` supplies their numerical parameters without
    replacing those settings. Unrelated lazy input extensions are not consumed.

    Args:
        model_inputs: Canonical arrays with selected scientific inputs eager.
        model_spec: Resolved scientific choices for exactly one sector.
        priors: Eager selected prior mappings keyed by ``x``, ``bc``,
            ``offset_latent``, ``sigma`` and ``power``. ``x`` is required;
            omitted unconfigured optional priors use their built-in defaults.
        likelihood_builder: Optional ordinary custom likelihood callable.
        likelihood_kwargs: Options expanded only into the custom likelihood.
        sigma_alignment: Optional precomputed mismatch alignment.

    Returns:
        Concrete PyMC model and its one validated scientific output contract.

    Raises:
        ValueError: If selected inputs or prior parameters are lazy, a selected
            prior mapping or required input is missing, the model has other
            than one sector, or likelihood choices are inconsistent.
        TypeError: If the custom likelihood returns an invalid value.
    """
    names = _standard_input_names(model_inputs, model_spec)
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    missing_priors = _standard_prior_parameters(model_spec).keys() - priors.keys()
    if missing_priors:
        raise ValueError(f"Missing materialized prior parameter mappings: {sorted(missing_priors)!r}.")
    likelihood_settings = model_spec.likelihood
    lazy_priors, _ = unpack_collections(priors)
    if any(is_dask_collection(model_inputs[name]) for name in names) or lazy_priors:
        raise ValueError("Standard model inputs and prior parameters must be materialized before building.")

    timing_start = timer_start()
    flux_sensitivity = model_inputs["H"]
    observations = model_inputs["mf"]
    observation_error = model_inputs["mf_error"]
    minimum_error = model_inputs.get("min_error")
    aggregation_error = resolve_aggregation_error(model_inputs, model_spec.aggregation_error_mode)

    use_bc = model_spec.use_bc
    add_offset = model_spec.add_offset
    offset_args = model_spec.offset_args
    sector = model_spec.sectors[0]
    state_activity = sector.state_activity if sector.state_activity is not None else model_spec.state_activity
    bc_state_activity = model_spec.bc_state_activity

    x_prior = dict(priors["x"])
    bc_prior = dict(priors.get("bc", DEFAULT_BC_PRIOR))
    offset_prior = dict(priors.get("offset_latent", DEFAULT_OFFSET_PRIOR))

    prepared_flux = prepare_linear_sensitivity(flux_sensitivity)
    prepared_boundary = (
        prepare_linear_sensitivity(model_inputs["H_bc"])
        if use_bc
        else None
    )

    with registered_model() as model:
        flux_component = add_linear_component(
            prepared_flux,
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=state_activity,
        )
        pollution_mean = flux_component.output

        boundary_mean = None
        if prepared_boundary is not None:
            boundary_mean = add_linear_component(
                prepared_boundary,
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
            ).output

        offset = None
        if add_offset:
            offset = add_offset_component(
                observations,
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )

        baseline_mean = boundary_mean
        if offset is not None:
            baseline_mean = offset if baseline_mean is None else baseline_mean + offset
        modelled_mean = pollution_mean if baseline_mean is None else pollution_mean + baseline_mean

        forward = ForwardModelTerms(
            total=modelled_mean,
            pollution=pollution_mean,
            baseline=baseline_mean,
        )
        add_rhime_likelihood(
            settings=likelihood_settings,
            likelihood_builder=likelihood_builder,
            likelihood_kwargs=likelihood_kwargs,
            forward=forward,
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            minimum_error=minimum_error,
            output_dim="nmeasure",
            sigma_alignment=sigma_alignment,
            prior_parameters=priors,
        )
    result = builtin_model_build_result(
        model, model_spec=model_spec, multisector=False,
        input_names=tuple(str(name) for name in model_inputs.data_vars),
    )
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=False)
    return result


def prepare_standard(
    merged_data: RhimeMergedData, *,
    domain: str, start_date: str, flux_sources: tuple[str, ...],
    averaging_error: bool, filters: Any, basis_algorithm: str | None, nbasis: int,
    fix_basis_outer_regions: bool, use_bc: bool, bc_freq: str | None,
    min_error: MinErrorConfig, min_error_options: Mapping[str, bool],
    basis_inputs: Mapping[str, Any], bc_basis: xr.Dataset | None = None,
) -> RhimePreparedInputs:
    """Prepare observations and sensitivities from supplied scientific resources.

    Args:
        domain: Domain label for prepared inputs and basis metadata.
        start_date: Requested start anchoring boundary-condition periods.
        flux_sources: Ordered source identities; the first weights basis design.
        averaging_error: Include variability when deriving observation errors.
        filters: Observation-filter choices interpreted by the filtering owner.
        basis_algorithm: Selected basis-generation algorithm.
        nbasis: Requested generated basis size.
        fix_basis_outer_regions: Keep the supplied outer partition fixed.
        use_bc: Construct boundary-condition sensitivities.
        bc_freq: Baked boundary-condition scaling period.
        min_error: Minimum-error policy interpreted during assembly.
        min_error_options: Options for the minimum-error calculation.
        merged_data: Borrowed acquired observations, transport and current flux.
        basis_inputs: Selected operator path or loaded partition and support maps.
        bc_basis: Loaded boundary-condition basis, when selected.

    Returns:
        Retained backend-neutral inputs and their current native-flux binding.
    """
    preparation_start = timer_start()
    # Preparation works on per-site datasets; ``merged_data`` stays acquired data.
    with_errors = prepare_observation_errors(merged_data.site_data, averaging_error=averaging_error)
    filtered, site_options = filter_observations(with_errors, merged_data.site_options, filters=filters)

    runtime_flux = resolve_runtime_flux(merged_data.flux_data, split_by_sectors=merged_data.split_by_sectors)
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=basis_algorithm):
        basis_functions = prepare_basis(
            site_data=filtered,
            runtime_flux=runtime_flux,
            weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
            domain=domain, start_date=start_date, basis_algorithm=basis_algorithm, nbasis=nbasis,
            fix_outer_regions=fix_basis_outer_regions,
            **basis_inputs,
        )

    sensitivities = build_sensitivities(
        filtered,
        basis_functions,
        flux_sources=flux_sources, use_bc=use_bc,
        bc_basis=bc_basis,
        multisector=False,
    )

    prepared = assemble_rhime_inputs(
        sensitivities,
        basis_functions,
        site_options=site_options,
        domain=domain, start_date=start_date, bc_freq=bc_freq, min_error=min_error,
        min_error_options=min_error_options, use_bc=use_bc,
    )
    log_timing(
        "rhime.prepare_inputs",
        timer_seconds(preparation_start),
        multisector=False,
        nmeasure=prepared.inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared.sites),
        regions=prepared.inv_inputs.sizes.get("region"),
        sources=prepared.inv_inputs.sizes.get("source"),
        basis_source=prepared.basis_artifact_source,
    )
    return prepared


def run_standard(
    *,
    config_file: str | Path | None = None,
    config: RhimeConfig | None = None,
    merged_data: RhimeMergedData | None = None,
    prepared: RhimePreparedInputs | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> InversionOutput:
    """Run a standard single-sector RHIME inversion.

    The visible process is resolve → acquire or accept supplied data → observation
    errors → filter → basis → sensitivities → assemble → materialize → build → sample → scientific output →
    requested outputs.

    Selected basis/support files and reporting country files are checked before
    acquisition, including when merged data is supplied. Basis partitions and
    support maps are loaded at that boundary. Saved bases are rebound to current
    flux; ``basis_output_path`` is an exact retained-artifact destination.

    Without supplied data, this runner acquires fresh data. Supplying ``prepared``
    skips acquisition and preparation and joins the same materialize/build/sample
    sequence. It cannot be combined with ``merged_data``.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        config: Complete request from ``RhimeConfig.from_params``.
            It is used without resolving again and cannot
            be combined with ``config_file`` or raw run parameters in ``kwargs``.
        merged_data: Optional externally supplied merged scientific data.
            Passing this borrowed in-memory handoff bypasses OpenGHG acquisition
            and resumes at observation-error preparation. This may derive error arrays
            and compute zero-error fallback diagnostics.
            The acquisition owner checks recorded species, domain, time window,
            and sector layout without mutating it.
        prepared: Optional borrowed canonical scientific inputs, including
            alternative or cached preparation. Bypasses acquisition and preparation.
        likelihood_builder: Optional Python-only callable invoked with a
            completed forward-model mean and explicit error-model inputs in
            the active PyMC model. It must return the canonical observed
            variable ``y`` and create the canonical error scale ``epsilon``.
            The callable is never read from configuration or stored in
            run/model specifications.
        likelihood_kwargs: Options specific to the custom likelihood. Common
            scientific arrays are passed explicitly by the recipe.
        **kwargs: RHIME run parameters using snake-case names, such as
            ``output_path``, ``output_name``, ``flux_sources``, and
            ``x_prior``. ``species`` names the primary gas or tracer used for
            object-store lookup and output naming. ``flux_sources`` contains
            OpenGHG flux ``source`` values. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Scientific inversion output borrowing the prepared arrays, basis and
        sampled trace, including when ``output_format="none"``. Product functions
        consume this object directly; requested files are written before return.
        Build a model directly with :func:`build_standard_model` for graph checks.

    Raises:
        TypeError: If a likelihood builder is not callable or returns the
            wrong result type.
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, the flux-source count is invalid, or likelihood
            variables or requested-output compatibility are invalid.

    """
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    if prepared is not None and merged_data is not None:
        raise ValueError("Pass either prepared or merged_data, not both.")
    setup_start = timer_start()
    if config is not None:
        if config_file is not None or kwargs:
            raise ValueError("Pass either resolved `config` or `config_file`/run parameters, not both.")
        if config.data.split_by_sectors:
            raise ValueError("`run_standard` received configuration for a different sector layout.")
    else:
        params = load_params(config_file=config_file, overrides=kwargs)
        config = RhimeConfig.from_params(params, multisector=False)
    if likelihood_builder is not None and config.model.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
    sampler = resolve_sampler(config.sampler)
    log_timing("rhime.runner_setup", timer_seconds(setup_start), multisector=False)
    if likelihood_builder is None and config.model.likelihood is None:
        raise ValueError("A standard RHIME run requires a built-in or custom likelihood.")

    validate_standard_output_resources(config.output)
    # 1. Resolve resources, acquire or validate data, then prepare scientific inputs.
    if prepared is None:
        basis_inputs = resolve_basis_inputs(config.data)
        bc_basis = (
            load_bc_basis(config.data.bc_basis_paths)
            if config.data.use_bc and config.data.bc_basis_paths is not None else None
        )

        if merged_data is None:
            merged_data = acquire_merged_data(config.data)
        else:
            merged_data.validate_for_preparation(
                **config.data.select("species", "domain", "start_date", "end_date", "split_by_sectors"),
            )

        prepared = prepare_standard(
            merged_data, basis_inputs=basis_inputs, bc_basis=bc_basis,
            **config.data.select(
                "domain", "start_date", "flux_sources", "averaging_error", "filters",
                "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                "min_error", "min_error_options",
            ),
        )
    else:
        prepared = prepared.validated()

    if config.data.basis_output_path is not None:
        Path(config.data.basis_output_path).parent.mkdir(parents=True, exist_ok=True)
        prepared.basis_functions.save(config.data.basis_output_path)

    # 2. Materialize selected inputs and priors together, then build and sample.
    model_inputs, priors = materialize_standard_inputs(prepared, config.model)
    built = build_standard_model(
        model_inputs, config.model, priors=priors,
        likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )
    built.output_contract.validate_requested_output(config.output.output_format)

    posterior = sample_rhime_model(built, sampler)
    annotate_sample_trace(
        built, posterior, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )

    # 3. Bind the scientific view, write requested products, then export artifacts.
    output_start = timer_start()
    with (timed("rhime.output.inversion_output_create", output_format=config.output.output_format)
          if config.output.output_format != "none" else nullcontext()):
        inv_out = make_rhime_output(
            prepared=prepared, model_spec=config.model, posterior=posterior,
            output_contract=built.output_contract, start_date=config.data.start_date,
            end_date=config.data.end_date, actual_inference=sampler.describe(),
            provenance=installed_ogi_provenance(),
            split_by_sectors=config.data.split_by_sectors,
        )
    postprocess_standard_outputs(inv_out, config.output)
    save_inversion_output(inv_out, config.output)
    log_timing(
        "rhime.output_total", timer_seconds(output_start),
        multisector=False, output_format=config.output.output_format,
    )
    return inv_out
