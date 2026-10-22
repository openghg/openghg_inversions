"""Source-resolved RHIME recipe using one stacked flux state.

Independent sector priors and activity policies populate one full flux state
``x``. Sectors identify sampled state components; acquisition sources identify
inventories and responses. Their current one-to-one ``sector_sources`` mapping
binds each sector prior to its sensitivity columns. The saved state retains its
``source`` coordinate, so select a sector's state using its mapped source with
``x.sel(source=...)``. The gathered sensitivity ``hx`` maps the full state to
pollution ``mu``; output roles retain local basis labels for native flux products.
"""

from __future__ import annotations

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.basis import prepare_basis

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dask import is_dask_collection
from dask.base import unpack_collections

import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.models._flux import (
    _prepared_sources,
    _select_sector_design,
)
from openghg_inversions.observation_error import (
    aggregation_error_input_names,
)
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.model_error import MinErrorConfig
from .specs import (
    AdditiveSigmaSettings,
    PollutionEventSettings,
    RhimeModelSpec,
)

from .standard import _build_independent_rhime_model
from .builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
)

from .materialization import _materialize_pymc_inputs_and_priors
from .outputs import (
    annotate_sample_trace,
    make_model_output,
    postprocess_multisector_outputs,
    save_inversion_output,
    validate_multisector_output_resources,
)
from .requests import load_params
from .params import RhimeConfig
from .preparation import (
    acquire_merged_data,
    resolve_basis_inputs,
    assemble_rhime_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
)
from .sampling import sample_rhime_model


_MULTISECTOR_FLUX_INPUT_NAMES = ("H",)
_OBSERVATION_INPUT_NAMES = ("mf", "mf_error")
_BASELINE_INPUT_NAMES = ("H_bc",)


def multisector_model_input_names(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[str, ...]:
    """Select scientific arrays required by the configured multisector components.

    Args:
        prepared: Borrowed canonical source-resolved inputs.
        model_spec: Resolved sector, baseline and likelihood choices.

    Returns:
        Variable names for the explicit PyMC materialization boundary.

    Raises:
        ValueError: If selected components require missing inputs or ambiguous errors.
    """
    return _multisector_input_names(prepared.inv_inputs, model_spec)


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


def _multisector_input_names(
    model_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
) -> tuple[str, ...]:
    """Declare arrays required by selected multisector-model components.

    Args:
        model_inputs: Canonical scientific input Dataset.
        model_spec: Resolved multisector component options.

    Returns:
        Prepared variable names selected for coordinated materialization.

    Raises:
        ValueError: If selected inputs are absent, sector identities or source
            layouts conflict, or the aggregation-error representation is ambiguous.
    """
    _require_component_inputs(
        model_inputs,
        _MULTISECTOR_FLUX_INPUT_NAMES,
        owner="Multisector flux component",
    )
    sectors = model_spec.sectors
    if len(sectors) < 2:
        raise ValueError("Multi-sector RHIME requires at least two sectors.")

    sector_names = [sector.name for sector in sectors]
    duplicate_names = list(dict.fromkeys(name for name in sector_names if sector_names.count(name) > 1))
    if duplicate_names:
        raise ValueError(
            f"Multi-sector RHIME requires unique sector names; duplicate sector {duplicate_names[0]!r}."
        )
    if any(not name.strip() for name in sector_names):
        raise ValueError("Multi-sector RHIME requires non-empty sector names.")

    source_sectors: dict[str, list[str]] = {}
    for sector in sectors:
        source_sectors.setdefault(sector.flux_source, []).append(sector.name)
    duplicate_sources = {source: names for source, names in source_sectors.items() if len(names) > 1}
    if duplicate_sources:
        details = ", ".join(
            f"source {source!r} is mapped by sectors {names!r}" for source, names in duplicate_sources.items()
        )
        raise ValueError(
            "Multi-sector RHIME requires a distinct source for each current sector; " + details + "."
        )

    suffixes = [sector.variable_suffix for sector in sectors]
    if any(not suffix.strip() for suffix in suffixes):
        raise ValueError("Multi-sector RHIME requires non-empty PyMC variable suffixes.")
    duplicate_suffixes = list(dict.fromkeys(suffix for suffix in suffixes if suffixes.count(suffix) > 1))
    if duplicate_suffixes:
        raise ValueError(
            "Multi-sector RHIME requires unique PyMC variable suffixes; "
            f"duplicate suffix {duplicate_suffixes[0]!r}."
        )

    available_sources = _prepared_sources(model_inputs["H"])
    missing_sources = [
        (sector.name, sector.flux_source) for sector in sectors if sector.flux_source not in available_sources
    ]
    if missing_sources:
        details = ", ".join(f"sector {name!r} -> source {source!r}" for name, source in missing_sources)
        raise ValueError(
            f"Source data required by {details} is not present in prepared H.source; "
            f"available source(s): {available_sources!r}."
        )
    declared_sources = [sector.flux_source for sector in sectors]
    if set(declared_sources) != set(available_sources):
        raise ValueError(
            "Multisector model sources must cover the retained source layout exactly; "
            f"declared {declared_sources!r}, prepared {available_sources!r}."
        )
    for sector in sectors:
        _select_sector_design(
            model_inputs["H"], sector=sector.name, source=sector.flux_source,
            variable_suffix=sector.variable_suffix, namespace_state_dim=False,
        )

    _require_component_inputs(
        model_inputs,
        _OBSERVATION_INPUT_NAMES,
        owner="Observation-error component",
    )
    names = [
        *_MULTISECTOR_FLUX_INPUT_NAMES,
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
            owner="Multisector baseline component selected by `use_bc=True`",
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


def _validate_multisector_basis_layout(
    basis_functions: BasisFunctions,
    model_spec: RhimeModelSpec,
    inv_inputs: xr.Dataset,
) -> None:
    """Require each retained source basis to match its prepared sector design.

    Args:
        basis_functions: Retained source-specific basis operators.
        model_spec: Sector declarations selecting those sources.
        inv_inputs: Prepared source-resolved sensitivity arrays.

    Raises:
        ValueError: If a source basis is missing or its state coordinate does
            not match the corresponding prepared sensitivity design.
    """
    available_sources = _prepared_sources(inv_inputs["H"])
    declared_sources = [sector.flux_source for sector in model_spec.sectors]
    if set(declared_sources) != set(available_sources) or len(declared_sources) != len(available_sources):
        raise ValueError(
            "Multisector model sources must cover the retained source layout exactly; "
            f"declared {declared_sources!r}, prepared {available_sources!r}."
        )
    design = inv_inputs["H"]
    for sector in model_spec.sectors:
        try:
            source_basis = basis_functions.for_source(sector.flux_source)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Sector {sector.name!r} requires source {sector.flux_source!r}, "
                "but the retained basis has no matching source-specific basis."
            ) from exc
        state_dim = source_basis.operator.meta.state_dim
        basis_index = source_basis.operator.basis_matrix.get_index(state_dim)
        sector_design = _select_sector_design(
            design,
            sector=sector.name,
            source=sector.flux_source,
            variable_suffix=sector.variable_suffix,
        )
        prepared_state_dims = [str(dim) for dim in sector_design.dims if dim != "nmeasure"]
        if len(prepared_state_dims) != 1:
            raise ValueError(
                f"Sector {sector.name!r} -> source {sector.flux_source!r} must have exactly "
                f"one prepared state dimension; found {prepared_state_dims!r}."
            )
        prepared_index = sector_design.get_index(prepared_state_dims[0])
        if not basis_index.equals(prepared_index):
            raise ValueError(
                "Retained source-specific basis state coordinates do not match prepared H; "
                f"sector {sector.name!r} -> source {sector.flux_source!r}: "
                f"basis has {len(basis_index)} regions, prepared H has {len(prepared_index)} "
                "with different coordinates."
            )


def _multisector_prior_parameters(model_spec: RhimeModelSpec) -> dict[str, PriorArgs]:
    """Select enabled priors before state pruning, retaining sector identities."""
    priors = {f"x_{sector.variable_suffix}": sector.x_prior for sector in model_spec.sectors}
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


def materialize_multisector_inputs(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[xr.Dataset, dict[str, PriorArgs]]:
    """Materialize selected multisector arrays and enabled priors together.

    Args:
        prepared: Borrowed source-labelled inputs and retained native flux.
        model_spec: Choices selecting scientific components and priors.

    Returns:
        Shallow input Dataset with selected values eager and eager prior mappings.
        Unselected arrays and the original preparation remain borrowed and lazy.

    Raises:
        ValueError: If a required component input or source layout is absent.
    """
    _validate_multisector_basis_layout(
        prepared.basis_functions,
        model_spec,
        prepared.inv_inputs,
    )
    return _materialize_pymc_inputs_and_priors(
        prepared,
        variable_names=multisector_model_input_names(prepared, model_spec),
        prior_parameters=_multisector_prior_parameters(model_spec),
    )


def build_multisector_model(
    model_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    priors: Mapping[str, PriorArgs],
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sigma_alignment: SigmaAlignment | None = None,
) -> RhimeModelBuildResult:
    """Construct the multisector graph from already materialized values.

    Source-prior blocks populate one labelled full state. The shared standard
    recipe applies one gathered sensitivity, then adds boundary conditions,
    offsets and the observation likelihood.

    Args:
        model_inputs: Canonical Dataset with selected scientific inputs eager.
        model_spec: Resolved sector, baseline and likelihood choices.
        priors: Eager prior mappings keyed by scientific model variable name.
        likelihood_builder: Optional ordinary custom likelihood callable.
        likelihood_kwargs: Options expanded only into that callable.
        sigma_alignment: Optional precomputed mismatch alignment.

    Returns:
        Concrete PyMC model and its validated scientific output contract.

    Raises:
        ValueError: If selected inputs or prior parameters are missing or lazy,
            sector identities or source layouts conflict, or likelihood choices
            are inconsistent.
        TypeError: If the custom likelihood returns an invalid value.
    """
    names = _multisector_input_names(model_inputs, model_spec)
    if likelihood_builder is not None and model_spec.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with built-in likelihood settings.")
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    missing_priors = _multisector_prior_parameters(model_spec).keys() - priors.keys()
    if missing_priors:
        raise ValueError(f"Missing materialized prior parameter mappings: {sorted(missing_priors)!r}.")
    lazy_priors, _ = unpack_collections(priors)
    if any(is_dask_collection(model_inputs[name]) for name in names) or lazy_priors:
        raise ValueError("Multisector model inputs and prior parameters must be materialized before building.")

    return _build_independent_rhime_model(
        model_inputs, model_spec, priors=priors,
        likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
        sigma_alignment=sigma_alignment, multisector=True,
    )


def prepare_multisector(
    merged_data: RhimeMergedData,
    *,
    domain: str,
    start_date: str,
    flux_sources: tuple[str, ...],
    averaging_error: bool,
    filters: Any,
    basis_algorithm: str | None,
    nbasis: int,
    fix_basis_outer_regions: bool,
    use_bc: bool,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    min_error_options: Mapping[str, bool],
    basis_inputs: Mapping[str, Any],
    bc_basis: xr.Dataset | None = None,
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
    merged_data.validate_scientific_inputs(
        split_by_sectors=True, use_bc=use_bc, flux_sources=flux_sources,
        require_footprints=basis_inputs.get("basis_path") is None and basis_inputs.get("basis_partition") is None,
    )
    # Preparation works on per-site datasets; ``merged_data`` stays acquired data.
    with_errors = prepare_observation_errors(
        merged_data.site_data,
        averaging_error=averaging_error,
    )
    filtered, site_options = filter_observations(
        with_errors,
        merged_data.site_options,
        filters=filters,
    )

    runtime_flux = resolve_runtime_flux(
        merged_data.flux_data,
        split_by_sectors=merged_data.split_by_sectors,
    )
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=basis_algorithm):
        basis_functions = prepare_basis(
            site_data=filtered,
            runtime_flux=runtime_flux,
            weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
            domain=domain,
            start_date=start_date,
            basis_algorithm=basis_algorithm,
            nbasis=nbasis,
            fix_outer_regions=fix_basis_outer_regions,
            **basis_inputs,
        )

    sensitivities = build_sensitivities(
        filtered,
        basis_functions,
        flux_sources=flux_sources,
        use_bc=use_bc,
        bc_basis=bc_basis,
        multisector=True,
    )

    prepared = assemble_rhime_inputs(
        sensitivities,
        basis_functions,
        site_options=site_options,
        domain=domain,
        start_date=start_date,
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_options=min_error_options,
        use_bc=use_bc,
    )
    log_timing(
        "rhime.prepare_inputs",
        timer_seconds(preparation_start),
        multisector=True,
        nmeasure=prepared.inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared.sites),
        regions=prepared.inv_inputs.sizes.get("region"),
        sources=prepared.inv_inputs.sizes.get("source"),
        basis_source=prepared.basis_artifact_source,
    )
    return prepared


def run_multisector(
    *,
    config_file: str | Path | None = None,
    config: RhimeConfig | None = None,
    merged_data: RhimeMergedData | None = None,
    prepared: RhimePreparedInputs | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> InversionOutput:
    """Run a source-resolved multisector inversion.

    The visible process is resolve → acquire or accept supplied data → observation
    errors → filter → basis → sensitivities → assemble → materialize → build → sample → scientific output →
    requested outputs.
    This module keeps source layout validation and sector-aware outputs beside
    that process instead of hiding them behind standard/multisector branching.

    Selected basis/support files and reporting country files are checked before
    acquisition, including when merged data is supplied. Basis partitions and
    support maps are loaded at that boundary. Saved bases are rebound to current
    flux; ``basis_output_path`` is an exact retained-artifact destination.

    Without supplied data, this runner acquires fresh data. Supplying ``prepared``
    joins the same materialize/build/sample sequence without acquisition or preparation.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        config: Complete request from ``RhimeConfig.from_params``.
            It is used without resolving again and cannot
            be combined with ``config_file`` or raw run parameters in ``kwargs``.
        merged_data: Optional externally supplied source-resolved merged
            scientific data. Passing this in-memory handoff bypasses OpenGHG
            acquisition and resumes at observation-error preparation after validating
            recorded species, domain, time window, and sector layout.
        prepared: Optional borrowed canonical inputs and retained basis. Cannot
            be combined with ``merged_data``.
        likelihood_builder: Optional Python-only callable invoked with a
            completed forward-model mean and explicit error-model inputs in
            the active PyMC model. It must return the canonical observed
            variable ``y`` and create the canonical error scale ``epsilon``.
            The callable is never read from configuration or stored in
            run/model specifications.
        likelihood_kwargs: Options specific to the custom likelihood. Common
            scientific arrays are passed explicitly by the recipe.
        **kwargs: RHIME run parameters using snake-case names. Multi-sector
            runs require at least two ``flux_sources`` and may include a
            complete ``sector_priors`` mapping keyed by sector name. When model
            sector labels differ from OpenGHG source values, pass
            ``sector_sources`` as a one-to-one mapping from sector name to one
            unique value in ``flux_sources``. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Scientific inversion output borrowing preparation and trace, even when
        output is disabled. Use scientific product functions for in-memory diagnostics.

    Raises:
        TypeError: If a likelihood builder is not callable or returns the
            wrong result type.
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, fewer than two flux sources are provided, or
            likelihood variables or requested-output compatibility are invalid.

    """
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    if prepared is not None and merged_data is not None:
        raise ValueError("Pass either prepared or merged_data, not both.")
    setup_start = timer_start()
    if config is not None:
        if config_file is not None or kwargs:
            raise ValueError("Pass either resolved `config` or `config_file`/run parameters, not both.")
        if not config.data.split_by_sectors:
            raise ValueError("`run_multisector` received configuration for a different sector layout.")
    else:
        params = load_params(config_file=config_file, overrides=kwargs)
        config = RhimeConfig.from_params(params, multisector=True)
    if likelihood_builder is not None and config.model.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
    sampler = resolve_sampler(config.sampler)
    log_timing("rhime.runner_setup", timer_seconds(setup_start), multisector=True)
    if likelihood_builder is None and config.model.likelihood is None:
        raise ValueError("A multisector RHIME run requires a built-in or custom likelihood.")

    validate_multisector_output_resources(config.output)
    # 1. Resolve resources, acquire or validate supplied data, then prepare.
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
                **config.data.select(
                    "species",
                    "domain",
                    "start_date",
                    "end_date",
                    "split_by_sectors",
                ),
            )
        prepared = prepare_multisector(
            merged_data,
            basis_inputs=basis_inputs,
            bc_basis=bc_basis,
            **config.data.select(
                "domain",
                "start_date",
                "flux_sources",
                "averaging_error",
                "filters",
                "basis_algorithm",
                "nbasis",
                "fix_basis_outer_regions",
                "use_bc",
                "bc_freq",
                "min_error",
                "min_error_options",
            ),
        )
    else:
        prepared = prepared.validated()

    if config.data.basis_output_path is not None:
        Path(config.data.basis_output_path).parent.mkdir(parents=True, exist_ok=True)
        prepared.basis_functions.save(config.data.basis_output_path)

    # 2. Materialize selected inputs and enabled priors, then build the graph.
    model_inputs, priors = materialize_multisector_inputs(prepared, config.model)
    model_build_result = build_multisector_model(
        model_inputs,
        config.model,
        priors=priors,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    model_build_result.output_contract.validate_requested_output(config.output.output_format)
    # 3. Inference receives the completed graph and its variable roles.
    idata = sample_rhime_model(
        model_build_result,
        sampler,
    )
    annotate_sample_trace(
        model_build_result,
        idata,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    # 4. Bind scientific meaning, compute products, then export result and trace.
    inv_out = make_model_output(
        prepared=prepared,
        model_spec=config.model,
        posterior=idata,
        output_contract=model_build_result.output_contract,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        actual_inference=sampler.describe(),
        split_by_sectors=config.data.split_by_sectors,
    )
    postprocess_multisector_outputs(inv_out, config.output)
    save_inversion_output(inv_out, config.output)
    return inv_out


# Established public import retained through the 0.9 compatibility cycle.
run_rhime_multisector = run_multisector
