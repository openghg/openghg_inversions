"""Reduce per-source native priors jointly into linked CO2/O2 prepared inputs (#1051).

Each native flux source has its own dimensionless scaling field and Gaussian
prior, configured as for the CO2 route
(:class:`~openghg_inversions.rhime.co2.configuration.Co2MultiSourceNativePrior`).
A source is declared ``shared`` (one field scales both its CO2 and its O2
flux, for example GPP, TER and fossil fuel) or specific to one tracer (the
CO2 and O2 ocean fields, which are independent sources with their own priors).

The CO2 and O2 native responses are stacked on one joint observation axis,
CO2 rows first. A tracer-specific source has zero response in the other
tracer's rows. One coherent reduction then gives the retained prior
``C_alpha``, both channels' effective sensitivities and the joint unresolved
covariance ``A`` with its CO2/O2 cross block, which
:func:`~openghg_inversions.rhime.co2.co2_o2_preparation.prepare_co2_o2_inputs`
forms into the linked artifact.

The O2 responses must already carry their signed O2 fluxes per source: for a
shared source, the response to the source's O2 flux, which embeds any
spatially varying O2:CO2 ratio. No ratio is applied here. Acquisition and
configuration of the two channels belong to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.basis_functions import FluxWeightedBasis, basis_functions_from_flat_basis
from openghg_inversions.basis.covariance_products import (
    PreserveBucketProlongation,
    RetainedProjectionStrategy,
    SuppliedRestriction,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import MergedData, select_boundary_representation
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models.boundary import Preparation
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
    build_sensitivities,
    filter_observations,
    resolve_basis_inputs,
    resolve_bc_basis,
)

from .co2_native_preparation import (
    NATIVE_SOURCE_DIM,
    MultiSourceNativeResources,
    _flux_weighted_restriction,
    _load_centering_weights,
    _multisource_record,
    _native_source_sensitivity,
    _retained_marginal,
    _source_class_labels,
    _source_covariance,
    _source_flux,
    _source_sigmas,
    _sectoral_response_only,
)
from .co2_o2_preparation import Co2O2PreparedInputs, _stack, prepare_co2_o2_inputs

if TYPE_CHECKING:
    from .co2_o2_configuration import Co2O2NativePreparationConfig
    from .configuration import Co2MultiSourceNativePrior

TRACER_SCOPES = ("shared", "co2", "o2")
"""Tracer scopes of a native source: one field for both tracers, or one tracer only."""

_JOINT_OBSERVATION = "joint_observation"


@dataclass(frozen=True)
class LinkedNativeChannel:
    """One tracer's observations and native per-source responses.

    Attributes:
        observations: Observations on one indexed native observation
            dimension with unique labels, distinct from the other channel's.
        native_sensitivity: Native response ``H`` on ``(observation,
            native_source, lat, lon)``, in ``units`` per unit native scaling,
            with exactly the sources whose tracer scope is ``"shared"`` or
            this channel. O2 responses are signed and embed their O2 fluxes.
        units: Observation units of this channel.
        independent_error_sd: Optional positive independent-error standard
            deviations on the observation dimension, in ``units``.
    """

    observations: xr.DataArray
    native_sensitivity: xr.DataArray
    units: str
    independent_error_sd: xr.DataArray | None = None


def prepare_multisource_native_co2_o2_inputs(
    *,
    co2: LinkedNativeChannel,
    o2: LinkedNativeChannel,
    basis: FluxWeightedBasis,
    prior: Co2MultiSourceNativePrior,
    source_states: Mapping[str, tuple[str, str]],
    resources: MultiSourceNativeResources | None = None,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    boundary_operands: Mapping[str, xr.Dataset] | None = None,
    bc_centering_weights: Mapping[str, xr.DataArray] | None = None,
    provenance: Mapping[str, Any] | None = None,
    observation_batch_size: int = 64,
) -> Co2O2PreparedInputs:
    """Reduce per-source native priors over both tracers into linked prepared inputs.

    Each native source ``s`` has a constant mean ``m_s`` and covariance
    ``sigma_s**2 K_s``, built exactly as on the CO2 route, so sources are
    independent except a configured GPP/TER correlation. In particular the
    CO2 and O2 ocean sources are independent and take only their own
    configured priors.

    Args:
        co2: CO2 observations and native responses.
        o2: O2 observations and native responses, on their own observation
            dimension; times and lengths may differ from CO2's.
        basis: Basis over every native source, with a
            :class:`~openghg_inversions.basis.operators.MultiSourceBucketBasisOperator`
            (one partition per source). Its flux, on ``source``, is each
            source's prior flux in its own tracer (the O2 ocean source's is
            the O2 ocean flux). It weights the absolute-flux-weighted
            restriction and any amplitude calibration.
        prior: Per-source native priors keyed by the basis sources.
        source_states: For every basis source, its retained ``(source,
            tracer_scope)`` labels: a sector name (``"gpp"``, ``"ter"``,
            ``"ff"`` or ``"ocean"``, any case) and one of
            :data:`TRACER_SCOPES`. The CO2 and O2 ocean sources are
            ``("ocean", "co2")`` and ``("ocean", "o2")``.
        resources: Class labels, retained regions and calibration targets
            read from configured files.
        boundary_sensitivity: Optional absolute boundary sensitivities, as in
            :func:`~openghg_inversions.rhime.co2.co2_o2_preparation.prepare_co2_o2_inputs`.
        boundary_operands: Optional raw centered boundary operands, likewise.
        bc_centering_weights: Optional centering weights, likewise.
        provenance: Extra JSON-safe provenance recorded with the artifact.
        observation_batch_size: Covariance right-hand sides per block.

    Returns:
        Linked prepared inputs whose retained states are labelled by
        ``(source, tracer_scope, region)``, with each source's configured
        marginal and the prior's latent-correlation floor. No flux ratio is
        recorded separately, since the O2 responses embed it.

    Raises:
        ValueError: If sources, scopes, responses, grids or regions do not
            match, or a reduction or linked-preparation contract fails.
    """
    resources = resources or MultiSourceNativeResources()
    operator = basis.operator
    if not isinstance(operator, MultiSourceBucketBasisOperator):
        raise ValueError("Linked native preparation requires a source-specific (multisource) bucket basis.")
    sources = list(operator.source_labels)
    for name, labels in (("prior sources", prior.sources), ("source_states", source_states)):
        if set(labels) != set(sources):
            raise ValueError(f"Linked native {name} {sorted(labels)!r} differ from basis sources {sorted(sources)!r}.")
    states = _source_states(source_states, sources)
    co2_dim, o2_dim = (_observation_dim(channel, name) for channel, name in ((co2, "CO2"), (o2, "O2")))
    if co2_dim == o2_dim:
        raise ValueError("CO2 and O2 observations require distinct dimension names.")
    if (co2.independent_error_sd is None) != (o2.independent_error_sd is None):
        raise ValueError("Supply independent_error_sd for both channels or neither.")

    sensitivity = xr.concat(
        [
            _joint_rows(channel, name, states, sources, offset)
            for channel, name, offset in ((co2, "co2", 0), (o2, "o2", co2.observations.size))
        ],
        dim=_JOINT_OBSERVATION,
        join="exact",
    )
    latitude, longitude = sensitivity["lat"], sensitivity["lon"]
    grid = xr.Dataset(coords={"lat": latitude, "lon": longitude})
    flux = _source_flux(basis, operator, grid)
    native_mean = xr.DataArray(
        np.stack([np.full((latitude.size, longitude.size), prior.sources[source].mean) for source in sources]),
        dims=(NATIVE_SOURCE_DIM, "lat", "lon"),
        coords={NATIVE_SOURCE_DIM: sources, "lat": latitude, "lon": longitude},
        attrs={"units": "1"},
    )
    classes = {source: _source_class_labels(resources.class_labels, source, grid) for source in sources}
    sigmas, calibration = _source_sigmas(prior, sources, flux, classes, resources.calibration_targets, grid)
    covariance = _source_covariance(prior, sources, sigmas, classes, latitude, longitude)

    state_dim = operator.meta.state_dim
    strategy: RetainedProjectionStrategy
    if prior.restriction == "bucket":
        strategy = PreserveBucketProlongation(
            operator._native_prolongation(native_mean, native_dims=(NATIVE_SOURCE_DIM, "lat", "lon"))
        )
    else:
        restriction = _flux_weighted_restriction(operator, flux, resources.retained_regions, grid)
        strategy = SuppliedRestriction(
            restriction, require_positivity_preserving=True, name="absolute_flux_weighted_regional_mean",
        )
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=strategy, state_dim=state_dim, native_mean=native_mean,
        native_sensitivity=sensitivity, observation_dim=_JOINT_OBSERVATION,
        observation_batch_size=observation_batch_size,
    )

    # Relabel the retained (native source, region) states by their linked (source, tracer_scope, region).
    retained = reduction.retained_mean.indexes[state_dim]
    if not isinstance(retained, pd.MultiIndex):  # pragma: no cover - operator invariant
        raise TypeError("Multisource retained states require a (source, region) MultiIndex.")
    linked_index = pd.MultiIndex.from_tuples(
        [(*states[str(source)], region) for source, region in retained],
        names=("source", "tracer_scope", operator.region_in_source_dim),
    )
    if not linked_index.is_unique:
        raise ValueError("source_states must give every native source a distinct (source, tracer_scope) pair.")
    state_coords = xr.Coordinates.from_pandas_multiindex(linked_index, state_dim)
    mean = xr.DataArray(
        reduction.retained_mean.values, dims=(state_dim,), coords=state_coords, attrs={"units": "1"},
        name="retained_mean",
    )
    retained_prior = CorrelatedLognormalPrior(
        mean, np.asarray(reduction.retained_covariance.transpose(state_dim, ...).values, dtype=np.float64)
    )
    native_marginal = _retained_marginal(prior, reduction.retained_mean)
    marginal = None if native_marginal is None else mean.copy(data=np.asarray(native_marginal.values).astype(str))

    nco2 = co2.observations.size
    rows = {"co2": slice(None, nco2), "o2": slice(nco2, None)}
    channels = {"co2": co2, "o2": o2}

    def channel_rows(array: xr.DataArray, channel: str) -> xr.DataArray:
        """Return one channel's rows of a joint-observation array on its native labels."""
        observations = channels[channel].observations
        dim = str(observations.dims[0])
        selected = array.isel({_JOINT_OBSERVATION: rows[channel]}).drop_vars(_JOINT_OBSERVATION)
        selected = selected.rename({_JOINT_OBSERVATION: dim}).assign_coords(observations[dim].coords)
        if state_dim in selected.dims:
            levels = [name for name in selected.coords if state_dim in selected[name].dims]
            selected = selected.drop_vars(levels).assign_coords(state_coords).transpose(dim, state_dim)
        return selected

    unresolved = np.asarray(reduction.unresolved_observation_covariance.values, dtype=np.float64)

    def block(row: str, column: str) -> xr.DataArray:
        """Label one block of the joint unresolved covariance with its channels' native axes."""
        row_obs, column_obs = channels[row].observations, channels[column].observations
        row_dim, column_dim = str(row_obs.dims[0]), str(column_obs.dims[0])
        column_name = f"{column_dim}_cov" if row == column else column_dim
        return xr.DataArray(
            unresolved[rows[row], rows[column]], dims=(row_dim, column_name),
            coords={row_dim: row_obs[row_dim], column_name: column_obs.indexes[column_dim].values},
        )

    independent_error = None
    if co2.independent_error_sd is not None and o2.independent_error_sd is not None:
        independent_error = _stack(
            co2.independent_error_sd, o2.independent_error_sd, co2_units=co2.units, o2_units=o2.units,
            name="independent_error_sd",
        )
    record = _multisource_record(prior, sources, sigmas, calibration, classes, covariance, reduction, state_dim)
    record["source_states"] = {source: list(states[source]) for source in sources}
    record["joint_observation_order"] = "CO2 rows then O2 rows; tracer-specific sources respond in their tracer only"
    metadata = dict(provenance or {})
    metadata["native_preparation"] = {**dict(metadata.get("native_preparation", {})), **record}
    return prepare_co2_o2_inputs(
        co2_observations=co2.observations,
        o2_observations=o2.observations,
        co2_prior_forward_mean=channel_rows(reduction.native_observation_mean, "co2"),
        o2_prior_forward_mean=channel_rows(reduction.native_observation_mean, "o2"),
        co2_sensitivity=channel_rows(reduction.effective_observation_operator, "co2"),
        o2_sensitivity=channel_rows(reduction.effective_observation_operator, "o2"),
        o2_co2_flux_ratio=None,
        o2_co2_flux_ratio_unavailable_reason=(
            "Native O2 responses embed each shared source's O2 flux, including any spatially varying "
            "O2:CO2 ratio, before reduction."
        ),
        co2_aggregation_covariance=block("co2", "co2"),
        co2_o2_aggregation_covariance=block("co2", "o2"),
        o2_aggregation_covariance=block("o2", "o2"),
        retained_prior=retained_prior,
        co2_units=co2.units,
        o2_units=o2.units,
        provenance=metadata,
        boundary_sensitivity=boundary_sensitivity,
        boundary_operands=boundary_operands,
        bc_centering_weights=bc_centering_weights,
        independent_error_sd=independent_error,
        prior_marginal=marginal,
        latent_correlation_floor=None if marginal is None else prior.latent_correlation_floor,
    )


_CENTERED_OPERANDS = ("H_bc_anomaly", "H_bc_mass", "bc_reference")


def prepare_native_co2_o2(
    native: Co2O2NativePreparationConfig,
    *,
    domain: str,
    start_date: str,
    end_date: str,
    boundaries: Mapping[str, Preparation],
    merged_data: Mapping[str, MergedData] | None = None,
) -> Co2O2PreparedInputs:
    """Acquire both tracers and prepare one linked artifact from native responses.

    For each tracer in turn, as on the configured CO2 route: acquire (or
    accept) its merged surface data, select its boundary representation,
    filter observations, derive observation errors, project its per-source
    responses on its flux sources' layers of the configured partition, and
    gather each retained row's native per-source response. The flux sources
    are then renamed to their native sources and
    :func:`prepare_multisource_native_co2_o2_inputs` reduces both tracers
    jointly.

    Args:
        native: Configured acquisition and native choices.
        domain: Transport domain.
        start_date: Requested window start.
        end_date: Requested window end.
        boundaries: Boundary preparation each channel's configured model needs.
        merged_data: Optional borrowed merged data per channel, skipping acquisition.

    Returns:
        Linked prepared inputs with each channel's observations, independent
        errors and boundary transport.

    Raises:
        ValueError: If supplied data conflict with the request, a channel's
            observations are not in a ppm-scale mole fraction (per-meg O2 needs
            OPE-86), or a preparation contract fails.
    """
    from .co2_native_preparation import MultiSourceNativeResources

    basis_inputs = resolve_basis_inputs(native.basis)
    bc_basis = resolve_bc_basis(native.basis)
    partition = basis_inputs.get("basis_partition")
    sources = list(native.native_prior.sources)
    if not isinstance(partition, xr.DataArray) or "source" not in partition.dims:
        raise ValueError("Linked preparation needs a basis partition with a source dimension keyed by native source.")
    missing = sorted(set(sources) - {str(source) for source in partition.indexes["source"]})
    if missing:
        raise ValueError(f"The basis partition has no layer for native source(s) {missing!r}.")
    layers = {source: partition.sel(source=source, drop=True) for source in sources}
    weights = None
    if native.bc_centering_weights_path is not None:
        weights = _load_centering_weights(native.bc_centering_weights_path)
    resources = MultiSourceNativeResources.load(native.native_prior)

    channels: dict[str, LinkedNativeChannel] = {}
    fluxes: dict[str, xr.DataArray] = {}
    boundary_operands: dict[str, xr.Dataset] = {}
    boundary_sensitivity: dict[str, xr.DataArray] = {}
    for channel in ("co2", "o2"):
        renamed = native.flux_sources(channel)
        prepared = _prepare_channel(
            native, channel, domain=domain, start_date=start_date, end_date=end_date,
            boundary=boundaries[channel], layers={flux: layers[source] for flux, source in renamed.items()},
            basis_inputs=basis_inputs, bc_basis=bc_basis,
            merged_data=None if merged_data is None else merged_data[channel],
        )
        channels[channel], flux, operands, absolute = prepared
        fluxes.update({str(name): flux.sel(source=name, drop=True) for name in flux.indexes["source"]})
        if operands is not None:
            boundary_operands[channel] = operands
        if absolute is not None:
            boundary_sensitivity[channel] = absolute

    # Each native source's prior flux in the tracer that weights it.
    weighting = []
    for source in sources:
        state = native.source_states[source]
        tracer = state.tracer_scope if state.tracer_scope != "shared" else (state.flux_tracer or "co2")
        flux_source = getattr(state, f"{tracer}_flux_source")
        weighting.append(fluxes[f"{tracer}:{flux_source}"].expand_dims(source=[source]))
    basis = basis_functions_from_flat_basis(flux=xr.concat(weighting, dim="source", join="exact"), basis_flat=layers)
    representation = {channel: boundaries[channel].representation for channel in ("co2", "o2")}
    return prepare_multisource_native_co2_o2_inputs(
        co2=channels["co2"], o2=channels["o2"], basis=basis, prior=native.native_prior,
        source_states={source: (state.source, state.tracer_scope) for source, state in native.source_states.items()},
        resources=resources,
        boundary_sensitivity=boundary_sensitivity or None,
        boundary_operands=boundary_operands or None,
        bc_centering_weights=(
            {channel: weights for channel in boundary_operands} if weights is not None else None
        ),
        provenance={"native_preparation": {
            "boundary_representation": representation,
            "flux_sources": {channel: native.flux_sources(channel) for channel in ("co2", "o2")},
            "source_state_options": {source: state.to_params() for source, state in native.source_states.items()},
        }},
    )


def _prepare_channel(
    native: Co2O2NativePreparationConfig,
    channel: str,
    *,
    domain: str,
    start_date: str,
    end_date: str,
    boundary: Preparation,
    layers: Mapping[str, xr.DataArray],
    basis_inputs: Mapping[str, Any],
    bc_basis: Any,
    merged_data: MergedData | None,
) -> tuple[LinkedNativeChannel, xr.DataArray, xr.Dataset | None, xr.DataArray | None]:
    """Acquire and prepare one tracer: its channel inputs, window-mean flux and boundary transport.

    The flux is keyed ``"<channel>:<flux source>"`` so both tracers' fluxes
    can be held together before they are renamed to native sources.
    """
    acquisition = dict(native.acquisition[channel])
    flux_sources = tuple(acquisition["flux_sources"])
    use_bc = bool(acquisition.get("use_bc", True))
    if merged_data is None:
        merged_data = MergedData.from_options(
            species=channel, domain=domain, start_date=start_date, end_date=end_date, **acquisition,
        )
    else:
        merged_data.validate_for_preparation(
            species=channel, domain=domain, start_date=start_date, end_date=end_date, split_by_sectors=False,
        )
    merged_data.validate_scientific_inputs(
        split_by_sectors=False, use_bc=use_bc, flux_sources=flux_sources, require_footprints=False,
    )
    site_data = select_boundary_representation(merged_data.site_data, boundary.representation)
    filtered, site_options = filter_observations(
        site_data, merged_data.site_options, filters=native.filters.get(channel),
    )
    filtered = prepare_observation_errors(filtered, averaging_error=native.averaging_error.get(channel, True))
    sectoral = resolve_runtime_flux(merged_data.flux_data, split_by_sectors=True, explicit_surface=True)
    basis_functions = prepare_basis(
        site_data=filtered, runtime_flux=sectoral, weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=native.basis.basis_algorithm,
        nbasis=native.basis.nbasis, fix_outer_regions=False,
        **{**basis_inputs, "basis_partition": dict(layers)},
    )
    sensitivities = build_sensitivities(
        _sectoral_response_only(filtered), basis_functions, flux_sources=flux_sources, use_bc=use_bc,
        bc_basis=bc_basis, multisector=True,
    )
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options, domain=domain, start_date=start_date, use_bc=use_bc,
    )
    operator = basis_functions.operator
    inputs = canonical.inv_inputs
    units = str(inputs["mf"].attrs.get("units", ""))
    try:
        scale = mole_fraction_unit_scale(units, context=f"{channel} observations")
    except Exception as exc:
        raise ValueError(
            f"Linked {channel} observations must be a mole fraction; per-meg delta(O2/N2) needs OPE-86."
        ) from exc
    if not np.isclose(scale, 1e-6, rtol=1e-12, atol=0.0):
        raise ValueError(f"Linked {channel} observations must be in ppm; got {units!r}.")
    # Same numerical scale, relabelled so both channels and their transport carry one unit label.
    dim = f"{channel}_measure"
    observations = inputs["mf"].rename({"nmeasure": dim}).assign_attrs(units="ppm")
    error = inputs["mf_error"].rename({"nmeasure": dim}).assign_attrs(units="ppm")
    response = _native_source_sensitivity(filtered, canonical, list(operator.source_labels))
    response = response.rename(observation=dim)
    keep = [name for name in observations.coords if dim in observations[name].dims]
    observations = observations.reset_coords([name for name in observations.coords if name not in keep], drop=True)
    error = error.reset_coords([name for name in error.coords if name not in keep], drop=True)
    renamed = native.flux_sources(channel)
    response = response.assign_coords({NATIVE_SOURCE_DIM: [renamed[str(name)] for name in response[NATIVE_SOURCE_DIM].values]})

    operands = absolute = None
    if use_bc and inputs.attrs.get("boundary_representation") == "centered-v1":
        operands = xr.Dataset(
            {name: inputs[name].assign_attrs(units="ppm") if name != "H_bc_mass" else inputs[name]
             for name in _CENTERED_OPERANDS},
            attrs={"boundary_representation": "centered-v1"},
        )
    elif use_bc and "H_bc" in inputs:
        absolute = inputs["H_bc"].transpose("nmeasure", ...).rename({"nmeasure": dim}).assign_attrs(units="ppm")
    flux = sectoral.mean("time") if "time" in sectoral.dims else sectoral
    flux = flux.assign_coords(source=[f"{channel}:{name}" for name in flux.indexes["source"]])
    return (
        LinkedNativeChannel(observations, response, units="ppm", independent_error_sd=error),
        flux, operands, absolute,
    )


def _source_states(source_states: Mapping[str, tuple[str, str]], sources: list[str]) -> dict[str, tuple[str, str]]:
    """Check each native source's retained (source, tracer_scope) labels."""
    states: dict[str, tuple[str, str]] = {}
    for source in sources:
        value = source_states[source]
        if (
            not isinstance(value, tuple | list) or len(value) != 2
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise ValueError(f"source_states[{source!r}] must be a (source, tracer_scope) pair of names.")
        sector, scope = value
        if scope not in TRACER_SCOPES:
            raise ValueError(f"source_states[{source!r}] tracer_scope must be one of {TRACER_SCOPES!r}; got {scope!r}.")
        states[source] = (sector, scope)
    return states


def _observation_dim(channel: LinkedNativeChannel, name: str) -> str:
    """Return a channel's one indexed observation dimension."""
    observations = channel.observations
    if observations.ndim != 1:
        raise ValueError(f"{name} observations must be one-dimensional.")
    dim = str(observations.dims[0])
    if dim not in observations.indexes or not observations.indexes[dim].is_unique:
        raise ValueError(f"{name} observations require unique labels on {dim!r}.")
    if not channel.units.strip():
        raise ValueError(f"{name} units must be non-empty.")
    return dim


def _joint_rows(
    channel: LinkedNativeChannel,
    name: str,
    states: Mapping[str, tuple[str, str]],
    sources: list[str],
    offset: int,
) -> xr.DataArray:
    """Place one channel's native responses on the joint rows, zero for the other tracer's sources."""
    dim = str(channel.observations.dims[0])
    response = channel.native_sensitivity
    if set(response.dims) != {dim, NATIVE_SOURCE_DIM, "lat", "lon"}:
        raise ValueError(f"{name.upper()} native_sensitivity must use dimensions ({dim}, {NATIVE_SOURCE_DIM}, lat, lon).")
    if not response.indexes[dim].equals(channel.observations.indexes[dim]):
        raise ValueError(f"{name.upper()} native_sensitivity rows must match its observations.")
    given = {str(source) for source in response.indexes[NATIVE_SOURCE_DIM]}
    expected = {source for source in sources if states[source][1] in ("shared", name)}
    if given != expected:
        raise ValueError(
            f"{name.upper()} native_sensitivity must hold exactly the shared and {name}-only sources "
            f"{sorted(expected)!r}; got {sorted(given)!r}."
        )
    response = response.reindex({NATIVE_SOURCE_DIM: sources}, fill_value=0.0)
    rows = np.arange(offset, offset + channel.observations.size)
    levels = [coord for coord in response.coords if dim in response[coord].dims]
    return (
        response.drop_vars(levels)
        .rename({dim: _JOINT_OBSERVATION})
        .assign_coords({_JOINT_OBSERVATION: rows})
        .transpose(_JOINT_OBSERVATION, NATIVE_SOURCE_DIM, "lat", "lon")
        .drop_attrs(deep=False)
    )


__all__ = ["TRACER_SCOPES", "LinkedNativeChannel", "prepare_multisource_native_co2_o2_inputs", "prepare_native_co2_o2"]
