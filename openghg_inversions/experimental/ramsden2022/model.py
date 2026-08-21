"""Modern PyMC implementation of the Ramsden et al. (2022) two-gas model.

The implementation is deliberately prepared-input first. Each gas is supplied
as a canonical RHIME-style :class:`xarray.Dataset`, normally obtained from a
``RhimePreparedInputs.inv_inputs`` object. Data retrieval and postprocessing
remain outside this historical comparison module.

For primary-gas sector ``s`` with state ``x_s`` the forward models are

``mu_primary = sum_s(H_primary,s @ x_s)``

and, for sectors that emit the tracer,

``mu_tracer = sum_s(H_tracer,s @ (R_s * x_s))``.

The two gases may have different sites, timestamps, and observation counts.
Their conditional likelihoods are independent and use the paper's absolute
model-error definition, ``sqrt(measurement_error**2 + sigma**2)``.

Use :func:`build_ramsden_model` to construct the graph without sampling and
:func:`run_ramsden_from_prepared_inputs` to sample it. Coupled primary/tracer
sensitivities must have exactly matching labelled state coordinates. A ratio may
be applied directly to a ratio-free tracer sensitivity, or interpreted as a
multiplier when the sensitivity already includes an explicit reference ratio. Boundary states
are optional and independent by gas. The caller is responsible for consistent
units; this module performs no retrieval, conversion, or postprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models._flux import (
    _namespace_sector_state_coords,
    _select_sector_design,
    safe_pymc_name,
)
from openghg_inversions.models.components import (
    add_linear_component,
    add_model_data,
    apply_linear_sensitivity,
)
from openghg_inversions.models.coords import add_coords, registered_model
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    prepare_linear_sensitivity,
)
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.sigma import SigmaAlignment

RatioResolution = Literal["scalar", "spatial"]


@dataclass(frozen=True)
class RamsdenPreparedInputs:
    """Canonical primary-gas and tracer inputs for one joint inversion.

    Each dataset requires ``H``, ``mf``, ``mf_error``, ``min_error``, and
    ``site_indicator`` on ``nmeasure``; ``H_bc`` is required when that
    channel's boundary component is enabled. Observation axes may differ.
    Coupled sector designs must have exactly matching labelled state
    coordinates and, for positional numeric labels, retained basis objects.

    Args:
        primary: Canonical RHIME-style dataset for the primary gas.
        tracer: Canonical RHIME-style dataset for the tracer gas. Its ``H``
            matrix must use the same labelled state layout as ``primary.H``.
        tracer_design_reference_ratios: Reference ratio already included in
            each tracer ``H`` source, keyed by tracer source label. Use
            ``None`` for a ratio-free design. These declarations are checked
            against each sector specification to prevent applying a ratio
            twice or omitting it.
        primary_basis_functions: Retained basis used to construct ``primary.H``.
            Required together with ``tracer_basis_functions`` when state
            labels are positional numbers.
        tracer_basis_functions: Retained basis used to construct ``tracer.H``.
            Its source-specific spatial map must equal the primary map for
            every coupled sector.
    """

    primary: xr.Dataset
    tracer: xr.Dataset
    tracer_design_reference_ratios: Mapping[str, float | None]
    primary_basis_functions: BasisFunctions | None = None
    tracer_basis_functions: BasisFunctions | None = None


@dataclass(frozen=True)
class RamsdenChannelSpec:
    """Likelihood and boundary settings for one observed gas.

    Args:
        species: Gas name used to namespace model variables.
        observation_units: Units shared by observations, measurement errors,
            minimum errors, forward-model outputs, and the model-error prior.
            Supported dataset unit attributes are checked by mol/mol scale;
            values are not converted.
        sigma_prior: PyMC prior mapping accepted by
            :func:`~openghg_inversions.models.priors.parse_prior` for absolute
            model error. The prior is expanded by site and period, not
            multiplied by the modelled enhancement.
        sigma_per_site: Whether model error is independent by observation site.
        sigma_frequency: Optional pandas-compatible frequency for model-error
            periods.
        sigma_frequency_anchor: Optional anchor for fixed-duration periods.
        use_bc: Whether this channel has an independent boundary component.
        bc_prior: Prior mapping for the state that scales this channel's
            ``H_bc`` design.
    """

    species: str
    observation_units: str
    sigma_prior: dict[str, Any]
    sigma_per_site: bool = True
    sigma_frequency: str | None = None
    sigma_frequency_anchor: DatetimeLike | None = None
    use_bc: bool = False
    bc_prior: dict[str, Any] | None = None


@dataclass(frozen=True)
class RamsdenSectorSpec:
    """Shared primary-gas state and optional tracer coupling for one sector.

    Args:
        name: Semantic sector name.
        primary_flux_source: Source-provenance label selected from
            ``primary.H``.
        x_prior: Prior mapping for the primary-gas scaling state. It must
            broadcast to the selected state dimension.
        tracer_flux_source: Source-provenance label selected from ``tracer.H``.
            ``None`` means this sector contributes no tracer, as for the
            non-fossil methane sector in Ramsden et al. (2022).
        ratio_prior: Prior for a sampled direct emission ratio or historical
            ratio multiplier. Exactly one of ``ratio_prior`` and
            ``fixed_ratio`` is required for a tracer-emitting sector. The
            distribution must have non-negative support.
        fixed_ratio: Fixed direct emission ratio or historical multiplier.
        ratio_resolution: ``"spatial"`` creates one sampled/fixed ratio value
            per state element, so its prior must broadcast to that state;
            ``"scalar"`` shares one value across the sector.
        reference_ratio: ``None`` selects the paper's direct-ratio contract:
            ``tracer.H`` must be ratio-free and the ratio parameter is applied
            directly. A positive value selects historical compatibility:
            ``tracer.H`` must already include this reference ratio, the sampled
            parameter is a dimensionless multiplier, and
            ``emission_ratio = reference_ratio * ratio_multiplier`` is exposed
            for interpretation. Direct and reference ratios are dimensionless
            molar ratios (moles tracer per mole primary).
    """

    name: str
    primary_flux_source: str
    x_prior: dict[str, Any]
    tracer_flux_source: str | None = None
    ratio_prior: dict[str, Any] | None = None
    fixed_ratio: float | None = None
    ratio_resolution: RatioResolution = "spatial"
    reference_ratio: float | None = None


@dataclass(frozen=True)
class RamsdenModelSpec:
    """Complete specification for the historical two-gas model.

    Args:
        primary: Primary-gas likelihood settings.
        tracer: Tracer-gas likelihood settings.
        sectors: Ordered primary sectors. At least one sector must also emit
            the tracer. Sanitized channel names must be distinct, and sanitized
            sector names must be non-empty and unique.
    """

    primary: RamsdenChannelSpec
    tracer: RamsdenChannelSpec
    sectors: tuple[RamsdenSectorSpec, ...]


@dataclass(frozen=True)
class RamsdenResult:
    """Result returned by :func:`run_ramsden_from_prepared_inputs`.

    Args:
        prepared_inputs: Labelled inputs consumed by the model.
        model_spec: Historical model specification.
        model: Built PyMC model.
        idata: Joint inference data preserving shared-state/ratio covariance
            and the two namespaced likelihood and predictive variables.
        sampler: Modern RHIME sampler used for the run.
    """

    prepared_inputs: RamsdenPreparedInputs
    model_spec: RamsdenModelSpec
    model: pm.Model
    idata: az.InferenceData
    sampler: RhimeSampler


@dataclass(frozen=True)
class _SectorDesigns:
    """Validated primary and optional tracer designs for one sector."""

    primary: xr.DataArray
    tracer: xr.DataArray | None
    state_dim: str


def _channel_suffix(spec: RamsdenChannelSpec) -> str:
    """Return a stable non-empty namespace suffix for one gas."""
    suffix = safe_pymc_name(spec.species)
    if not suffix:
        raise ValueError("Ramsden channel species names must contain at least one letter or digit.")
    return suffix


def _validate_channel_spec(spec: RamsdenChannelSpec, *, label: str) -> None:
    """Validate one gas-channel specification."""
    if not spec.species.strip():
        raise ValueError(f"Ramsden {label} species must be non-empty.")
    if not spec.observation_units.strip():
        raise ValueError(f"Ramsden {label} observation_units must be non-empty.")
    if _mole_fraction_unit_scale(spec.observation_units) is None:
        raise ValueError(
            f"Ramsden {label} observation_units {spec.observation_units!r} are not a supported "
            "mole-fraction unit."
        )
    if spec.use_bc and spec.bc_prior is None:
        raise ValueError(f"Ramsden {label} bc_prior must be explicit when use_bc=True.")


def _mole_fraction_unit_scale(units: object) -> float | None:
    """Return the mol/mol scale represented by a supported unit declaration."""
    if isinstance(units, (float, int)) and not isinstance(units, bool):
        scale = float(units)
        return scale if math.isfinite(scale) and scale > 0 else None
    if not isinstance(units, str):
        return None

    normalized = units.strip().lower().replace("μ", "u").replace("µ", "u")
    try:
        numeric_scale = float(normalized)
    except ValueError:
        pass
    else:
        return numeric_scale if math.isfinite(numeric_scale) and numeric_scale > 0 else None

    aliases = {
        "mol/mol": 1.0,
        "mol mol-1": 1.0,
        "mol mol^-1": 1.0,
        "ppm": 1e-6,
        "umol/mol": 1e-6,
        "ppb": 1e-9,
        "nmol/mol": 1e-9,
        "ppt": 1e-12,
        "pmol/mol": 1e-12,
    }
    if normalized in aliases:
        return aliases[normalized]

    pieces = normalized.split(maxsplit=1)
    if len(pieces) == 2 and pieces[1] in {"mol/mol", "mol mol-1", "mol mol^-1"}:
        try:
            scale = float(pieces[0])
        except ValueError:
            return None
        return scale if math.isfinite(scale) and scale > 0 else None
    return None


def _validate_sector_spec(sector: RamsdenSectorSpec) -> None:
    """Validate one shared-sector and ratio declaration."""
    if not sector.name.strip():
        raise ValueError("Ramsden sector names must be non-empty.")
    if not sector.primary_flux_source.strip():
        raise ValueError(f"Ramsden sector {sector.name!r} requires a primary_flux_source.")
    if sector.ratio_resolution not in ("scalar", "spatial"):
        raise ValueError(f"Ramsden sector {sector.name!r} ratio_resolution must be 'scalar' or 'spatial'.")

    has_prior = sector.ratio_prior is not None
    has_fixed = sector.fixed_ratio is not None
    if sector.tracer_flux_source is None:
        if has_prior or has_fixed or sector.reference_ratio is not None:
            raise ValueError(f"Non-tracer Ramsden sector {sector.name!r} must not define ratio settings.")
        return

    if not sector.tracer_flux_source.strip():
        raise ValueError(f"Ramsden sector {sector.name!r} has an empty tracer_flux_source.")
    if has_prior == has_fixed:
        raise ValueError(
            f"Tracer-emitting Ramsden sector {sector.name!r} requires exactly one of "
            "`ratio_prior` and `fixed_ratio`."
        )
    if sector.fixed_ratio is not None and (
        not math.isfinite(float(sector.fixed_ratio)) or float(sector.fixed_ratio) < 0
    ):
        raise ValueError(f"Ramsden sector {sector.name!r} fixed_ratio must be finite and non-negative.")
    if sector.ratio_prior is not None:
        _validate_ratio_prior(sector)
    if sector.reference_ratio is not None and (
        not math.isfinite(float(sector.reference_ratio)) or float(sector.reference_ratio) <= 0
    ):
        raise ValueError(f"Ramsden sector {sector.name!r} reference_ratio must be finite and positive.")


def _validate_ratio_prior(sector: RamsdenSectorSpec) -> None:
    """Require a sampled emission ratio to have non-negative support."""
    assert sector.ratio_prior is not None
    pdf = str(sector.ratio_prior.get("pdf", "")).lower()
    positive_support = {
        "chisquared",
        "exponential",
        "gamma",
        "halfcauchy",
        "halfnormal",
        "halfstudentt",
        "inversegamma",
        "lognormal",
        "pareto",
        "weibull",
    }
    if pdf in positive_support:
        return
    if pdf in {"truncatednormal", "uniform"}:
        lower = sector.ratio_prior.get("lower")
        if lower is None:
            lower_value = math.nan
        else:
            try:
                lower_value = float(lower)
            except (TypeError, ValueError):
                lower_value = math.nan
        if math.isfinite(lower_value) and lower_value >= 0:
            return
    raise ValueError(
        f"Ramsden sector {sector.name!r} ratio_prior must use a distribution with non-negative support."
    )


def _validate_model_spec(model_spec: RamsdenModelSpec) -> None:
    """Validate complete model metadata before mutating a PyMC model."""
    _validate_channel_spec(model_spec.primary, label="primary")
    _validate_channel_spec(model_spec.tracer, label="tracer")
    primary_suffix = _channel_suffix(model_spec.primary)
    tracer_suffix = _channel_suffix(model_spec.tracer)
    if primary_suffix == tracer_suffix:
        raise ValueError("Ramsden primary and tracer species must produce distinct model names.")
    if not model_spec.sectors:
        raise ValueError("Ramsden models require at least one primary-gas sector.")
    for sector in model_spec.sectors:
        _validate_sector_spec(sector)
    if not any(sector.tracer_flux_source is not None for sector in model_spec.sectors):
        raise ValueError("Ramsden models require at least one tracer-emitting sector.")

    suffixes = [safe_pymc_name(sector.name) for sector in model_spec.sectors]
    if any(not suffix for suffix in suffixes):
        raise ValueError("Ramsden sector names must produce non-empty model variable names.")
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("Ramsden sector names must produce unique model variable names.")

    primary_sources = [sector.primary_flux_source for sector in model_spec.sectors]
    if len(set(primary_sources)) != len(primary_sources):
        raise ValueError("Ramsden sectors must select unique primary flux sources.")
    tracer_sources = [
        sector.tracer_flux_source for sector in model_spec.sectors if sector.tracer_flux_source is not None
    ]
    if len(set(tracer_sources)) != len(tracer_sources):
        raise ValueError("Ramsden tracer-emitting sectors must select unique tracer flux sources.")


def _validate_channel_data(
    data: xr.Dataset,
    spec: RamsdenChannelSpec,
    *,
    label: str,
) -> None:
    """Validate canonical variables needed by one gas channel."""
    required = {"H", "mf", "mf_error", "min_error", "site_indicator"}
    missing = sorted(required - set(data.data_vars))
    if missing:
        raise ValueError(f"Ramsden {label} inputs are missing required variable(s): {missing!r}.")
    for name in required:
        if "nmeasure" not in data[name].dims:
            raise ValueError(f"Ramsden {label} input {name!r} must include the 'nmeasure' dimension.")
    if data.sizes.get("nmeasure", 0) == 0:
        raise ValueError(f"Ramsden {label} inputs require at least one observation.")
    if spec.use_bc and "H_bc" not in data:
        raise ValueError(f"Ramsden {label} inputs require 'H_bc' when use_bc=True.")
    if "H_bc" in data and "nmeasure" not in data["H_bc"].dims:
        raise ValueError(f"Ramsden {label} input 'H_bc' must include the 'nmeasure' dimension.")

    expected_scale = _mole_fraction_unit_scale(spec.observation_units)
    assert expected_scale is not None
    for name in ("mf", "mf_error", "min_error", "H", "H_bc"):
        if name not in data:
            continue
        units = data[name].attrs.get("units")
        if units is None:
            if name == "mf":
                raise ValueError(
                    f"Ramsden {label} input 'mf' requires a units attribute matching "
                    f"{spec.observation_units!r}."
                )
            continue
        actual_scale = _mole_fraction_unit_scale(units)
        if actual_scale is None or not math.isclose(actual_scale, expected_scale):
            raise ValueError(
                f"Ramsden {label} input {name!r} units {units!r} do not match declared "
                f"observation_units {spec.observation_units!r}."
            )


def _rename_observation_axis(data: xr.DataArray, suffix: str) -> xr.DataArray:
    """Give one channel a unique observation dimension and auxiliary coords."""
    output_dim = f"nmeasure_{suffix}"
    result = data.rename({"nmeasure": output_dim})
    if isinstance(result.indexes.get(output_dim), pd.MultiIndex):
        result = result.reset_index(output_dim)

    coord_renames = {
        str(name): f"{suffix}_{name}"
        for name, coord in result.coords.items()
        if name != output_dim and output_dim in coord.dims
    }
    if coord_renames:
        result = result.rename(coord_renames)
    return result.assign_coords({output_dim: np.arange(result.sizes[output_dim])})


def _state_dim(design: xr.DataArray, *, observation_dim: str, label: str) -> str:
    """Return the sole state dimension for a selected sector design."""
    dims = [str(dim) for dim in design.dims if dim != observation_dim]
    if len(dims) != 1:
        raise ValueError(
            f"Ramsden {label} sector designs require exactly one state dimension; found {dims!r}."
        )
    return dims[0]


def _state_index(design: xr.DataArray, state_dim: str) -> pd.Index:
    """Return a comparable labelled index for one state dimension."""
    index = design.indexes.get(state_dim)
    if index is not None:
        return index
    return pd.Index(np.asarray(design.coords[state_dim].values), name=state_dim)


def _basis_map_for_source(
    basis_functions: BasisFunctions,
    *,
    source: str,
    label: str,
) -> xr.DataArray:
    """Select the spatial basis map used by one sensitivity source."""
    basis = basis_functions.flat_basis()
    if isinstance(basis, dict):
        try:
            return basis[source]
        except KeyError as exc:
            raise ValueError(f"Ramsden {label} retained basis has no map for source {source!r}.") from exc
    return basis


def _validate_shared_basis(
    prepared_inputs: RamsdenPreparedInputs,
    sector: RamsdenSectorSpec,
    *,
    state_index: pd.Index,
) -> None:
    """Verify that a coupled sector uses one spatial basis in both channels."""
    primary_basis = prepared_inputs.primary_basis_functions
    tracer_basis = prepared_inputs.tracer_basis_functions
    if (primary_basis is None) != (tracer_basis is None):
        raise ValueError(
            "Ramsden prepared inputs must provide both primary and tracer basis functions, or neither."
        )

    if primary_basis is None or tracer_basis is None:
        if pd.api.types.is_numeric_dtype(state_index.dtype):
            raise ValueError(
                f"Ramsden sector {sector.name!r} has positional numeric state labels; provide both "
                "retained basis functions so spatial coupling can be verified."
            )
        return

    assert sector.tracer_flux_source is not None
    primary_map = _basis_map_for_source(
        primary_basis,
        source=sector.primary_flux_source,
        label="primary",
    )
    tracer_map = _basis_map_for_source(
        tracer_basis,
        source=sector.tracer_flux_source,
        label="tracer",
    )
    if not primary_map.equals(tracer_map):
        raise ValueError(
            f"Ramsden sector {sector.name!r} primary and tracer spatial basis maps must match exactly."
        )


def _validate_ratio_provenance(
    prepared_inputs: RamsdenPreparedInputs,
    sector: RamsdenSectorSpec,
) -> None:
    """Check tracer-design ratio provenance against one sector declaration."""
    assert sector.tracer_flux_source is not None
    try:
        actual = prepared_inputs.tracer_design_reference_ratios[sector.tracer_flux_source]
    except KeyError as exc:
        raise ValueError(
            f"Ramsden tracer source {sector.tracer_flux_source!r} requires an explicit "
            "tracer_design_reference_ratios entry."
        ) from exc

    expected = sector.reference_ratio
    if actual is None and expected is None:
        return
    if actual is None or expected is None or not math.isclose(float(actual), float(expected)):
        raise ValueError(
            f"Ramsden sector {sector.name!r} tracer design reference ratio {actual!r} does not "
            f"match model reference_ratio {expected!r}."
        )


def _select_sector_designs(
    prepared_inputs: RamsdenPreparedInputs,
    sector: RamsdenSectorSpec,
) -> _SectorDesigns:
    """Select and exactly align primary/tracer designs for one shared state."""
    variable_suffix = safe_pymc_name(sector.name)
    primary = _select_sector_design(
        prepared_inputs.primary["H"],
        sector=sector.name,
        source=sector.primary_flux_source,
        variable_suffix=variable_suffix,
    )
    primary_state_dim = _state_dim(primary, observation_dim="nmeasure", label="primary")

    if sector.tracer_flux_source is None:
        return _SectorDesigns(primary=primary, tracer=None, state_dim=primary_state_dim)

    tracer = _select_sector_design(
        prepared_inputs.tracer["H"],
        sector=sector.name,
        source=sector.tracer_flux_source,
        variable_suffix=variable_suffix,
    )
    tracer_state_dim = _state_dim(tracer, observation_dim="nmeasure", label="tracer")
    primary_index = _state_index(primary, primary_state_dim)
    tracer_index = _state_index(tracer, tracer_state_dim)
    if not primary_index.equals(tracer_index):
        raise ValueError(
            f"Ramsden sector {sector.name!r} primary and tracer state coordinates must match exactly."
        )
    if tracer_state_dim != primary_state_dim:
        tracer = tracer.rename({tracer_state_dim: primary_state_dim})
    _validate_shared_basis(prepared_inputs, sector, state_index=primary_index)
    _validate_ratio_provenance(prepared_inputs, sector)

    return _SectorDesigns(primary=primary, tracer=tracer, state_dim=primary_state_dim)


def _fixed_parameter(
    name: str,
    value: float,
    *,
    state: TensorVariable,
    state_dim: str,
    resolution: RatioResolution,
) -> TensorVariable:
    """Expose a fixed scalar or state-aligned parameter as a deterministic."""
    scalar = pt.as_tensor_variable(pm.floatX(value))
    if resolution == "spatial":
        return pm.Deterministic(name, pt.ones_like(state) * scalar, dims=state_dim)
    return pm.Deterministic(name, scalar)


def _ratio_tensors(
    sector: RamsdenSectorSpec,
    *,
    state: TensorVariable,
    state_dim: str,
    variable_suffix: str,
) -> tuple[TensorVariable, TensorVariable]:
    """Create the forward ratio factor and direct molar-ratio diagnostic."""
    compatibility_mode = sector.reference_ratio is not None
    parameter_name = (
        f"ratio_multiplier_{variable_suffix}" if compatibility_mode else f"emission_ratio_{variable_suffix}"
    )
    prior_dims = state_dim if sector.ratio_resolution == "spatial" else None
    if sector.ratio_prior is not None:
        kwargs = {"dims": prior_dims} if prior_dims is not None else {}
        parameter = parse_prior(parameter_name, sector.ratio_prior, **kwargs)
    else:
        assert sector.fixed_ratio is not None
        parameter = _fixed_parameter(
            parameter_name,
            float(sector.fixed_ratio),
            state=state,
            state_dim=state_dim,
            resolution=sector.ratio_resolution,
        )

    if not compatibility_mode:
        return parameter, parameter

    assert sector.reference_ratio is not None
    emission_ratio = pm.Deterministic(
        f"emission_ratio_{variable_suffix}",
        pm.floatX(sector.reference_ratio) * parameter,
        dims=prior_dims,
    )
    return parameter, emission_ratio


def _sum_terms(terms: list[TensorVariable], *, name: str, dim: str) -> TensorVariable:
    """Register the ordered sum of non-empty observation-space terms."""
    if not terms:
        raise ValueError(f"Cannot construct {name!r} without at least one forward term.")
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return pm.Deterministic(name, total, dims=dim)


def _add_boundary_component(
    prepared: PreparedLinearSensitivity | None,
    spec: RamsdenChannelSpec,
    *,
    suffix: str,
) -> TensorVariable | None:
    """Add an independent optional boundary state for one channel."""
    if prepared is None:
        return None

    output_dim = f"nmeasure_{suffix}"
    assert spec.bc_prior is not None
    return add_linear_component(
        prepared,
        data_name=f"hbc_{suffix}",
        prior_args=dict(spec.bc_prior),
        var_name=f"bc_{suffix}",
        output_name=f"mu_bc_{suffix}",
        output_dim=output_dim,
        compute_deterministic=True,
    ).output


def _prepare_boundary_sensitivity(
    data: xr.Dataset,
    spec: RamsdenChannelSpec,
    *,
    suffix: str,
) -> PreparedLinearSensitivity | None:
    """Prepare one optional namespaced channel boundary sensitivity."""
    if not spec.use_bc:
        return None
    output_dim = f"nmeasure_{suffix}"
    sensitivity = _rename_observation_axis(data["H_bc"], suffix)
    state_renames = {
        str(dim): f"{dim}_{suffix}_bc"
        for dim in sensitivity.dims
        if str(dim) != output_dim
    }
    if state_renames:
        sensitivity = sensitivity.rename(state_renames)
    state_dim = next(str(dim) for dim in sensitivity.dims if str(dim) != output_dim)
    if state_dim not in sensitivity.coords:
        sensitivity = sensitivity.assign_coords(
            {state_dim: np.arange(sensitivity.sizes[state_dim])}
        )
    return prepare_linear_sensitivity(sensitivity, output_dim=output_dim)


def _namespace_retained_sensitivity(
    prepared: PreparedLinearSensitivity,
    *,
    suffix: str,
) -> PreparedLinearSensitivity:
    """Namespace only a structurally retained backend state axis."""
    retained_dim = next(
        str(dim) for dim in prepared.sensitivity.dims if dim != prepared.output_dim
    )
    if retained_dim == prepared.state_dim:
        return prepared
    return PreparedLinearSensitivity(
        sensitivity=_namespace_sector_state_coords(
            prepared.sensitivity,
            variable_suffix=suffix,
            observation_dim=prepared.output_dim,
            namespace_state_dim=True,
        ),
        removed=prepared.removed,
        output_dim=prepared.output_dim,
    )


def _add_absolute_error_likelihood(
    data: xr.Dataset,
    spec: RamsdenChannelSpec,
    *,
    suffix: str,
    mu: TensorVariable,
    mu_bc: TensorVariable | None,
) -> None:
    """Add one namespaced Gaussian likelihood with absolute model error."""
    output_dim = f"nmeasure_{suffix}"
    observations = add_model_data(_rename_observation_axis(data["mf"], suffix), f"Y_{suffix}")
    measurement_error = add_model_data(
        _rename_observation_axis(data["mf_error"], suffix),
        f"error_{suffix}",
    )
    min_error = add_model_data(
        _rename_observation_axis(data["min_error"], suffix),
        f"min_error_{suffix}",
    )

    alignment = SigmaAlignment.from_frequency(
        data["site_indicator"],
        frequency=spec.sigma_frequency,
        per_site=spec.sigma_per_site,
        anchor_time=spec.sigma_frequency_anchor,
    )
    site_index = add_model_data(
        _rename_observation_axis(alignment.site_index, suffix),
        f"sigma_site_index_{suffix}",
    )
    period_index = add_model_data(
        _rename_observation_axis(alignment.period_index, suffix),
        f"sigma_period_index_{suffix}",
    )
    site_dim = f"nsigma_site_{suffix}"
    period_dim = f"nsigma_period_{suffix}"
    add_coords(
        {
            site_dim: np.arange(alignment.nsite),
            period_dim: np.arange(alignment.nperiod),
        }
    )
    sigma = parse_prior(
        f"sigma_{suffix}",
        dict(spec.sigma_prior),
        dims=(site_dim, period_dim),
    )
    sigma_aligned = sigma[site_index, period_index]
    epsilon = pm.Deterministic(
        f"epsilon_{suffix}",
        pt.maximum(  # type: ignore[operator]
            pt.sqrt(measurement_error**2 + sigma_aligned**2), min_error
        ),
        dims=output_dim,
    )
    total_mu = mu if mu_bc is None else mu + mu_bc
    pm.Normal(
        f"y_{suffix}",
        mu=total_mu,
        sigma=epsilon,
        observed=observations,
        dims=output_dim,
    )


def build_ramsden_model(
    prepared_inputs: RamsdenPreparedInputs,
    model_spec: RamsdenModelSpec,
) -> pm.Model:
    """Build the historical Ramsden methane/ethane model with modern PyMC.

    Args:
        prepared_inputs: Two canonical gas datasets. Each requires ``H``,
            ``mf``, ``mf_error``, ``min_error``, and ``site_indicator`` on
            ``nmeasure``; ``H_bc`` is required for enabled boundary states.
            Observation coordinates may differ, but coupled sector state
            coordinates must match exactly.
        model_spec: Shared-state, ratio, likelihood, unit, and boundary
            metadata. Direct ratios require ratio-free tracer sensitivities;
            reference-ratio mode requires tracer sensitivities that already include
            the declared reference ratio.

    Returns:
        Built PyMC model ready for
        :class:`~openghg_inversions.rhime.sampling.RhimeSampler`.

    Raises:
        ValueError: If model metadata, required input variables, source labels,
            or shared state coordinates are invalid.

    Notes:
        This function builds a PyMC graph only. It performs no data retrieval,
        sampling, unit conversion, or postprocessing. Observation, error,
        sigma-prior, and forward-model values must already use each channel's
        declared ``observation_units``.
    """
    _validate_model_spec(model_spec)
    _validate_channel_data(prepared_inputs.primary, model_spec.primary, label="primary")
    _validate_channel_data(prepared_inputs.tracer, model_spec.tracer, label="tracer")
    sector_sensitivities = [
        (sector, _select_sector_designs(prepared_inputs, sector)) for sector in model_spec.sectors
    ]

    primary_suffix = _channel_suffix(model_spec.primary)
    tracer_suffix = _channel_suffix(model_spec.tracer)
    primary_dim = f"nmeasure_{primary_suffix}"
    tracer_dim = f"nmeasure_{tracer_suffix}"
    prepared_terms = []
    for sector, sensitivities in sector_sensitivities:
        variable_suffix = safe_pymc_name(sector.name)
        primary_sensitivity = _namespace_retained_sensitivity(
            prepare_linear_sensitivity(
                _rename_observation_axis(sensitivities.primary, primary_suffix),
                output_dim=primary_dim,
            ),
            suffix=f"{variable_suffix}_{primary_suffix}",
        )
        tracer_sensitivity = (
            None
            if sensitivities.tracer is None
            else _namespace_retained_sensitivity(
                prepare_linear_sensitivity(
                    _rename_observation_axis(sensitivities.tracer, tracer_suffix),
                    output_dim=tracer_dim,
                ),
                suffix=f"{variable_suffix}_{tracer_suffix}",
            )
        )
        prepared_terms.append(
            (sector, sensitivities, primary_sensitivity, tracer_sensitivity)
        )
    primary_boundary = _prepare_boundary_sensitivity(
        prepared_inputs.primary,
        model_spec.primary,
        suffix=primary_suffix,
    )
    tracer_boundary = _prepare_boundary_sensitivity(
        prepared_inputs.tracer,
        model_spec.tracer,
        suffix=tracer_suffix,
    )

    with registered_model() as model:
        primary_terms: list[TensorVariable] = []
        tracer_terms: list[TensorVariable] = []

        for sector, sensitivities, primary_sensitivity, tracer_sensitivity in prepared_terms:
            variable_suffix = safe_pymc_name(sector.name)
            add_coords(sensitivities.primary.coords, model_dims=(sensitivities.state_dim,))
            state = parse_prior(
                f"x_{variable_suffix}",
                sector.x_prior,
                dims=sensitivities.state_dim,
            )
            primary_terms.append(
                apply_linear_sensitivity(
                    primary_sensitivity,
                    state,
                    data_name=f"hx_{variable_suffix}",
                    output_name=f"mu_{primary_suffix}_{variable_suffix}",
                )
            )

            if tracer_sensitivity is None:
                continue
            ratio_factor, _ = _ratio_tensors(
                sector,
                state=state,
                state_dim=sensitivities.state_dim,
                variable_suffix=variable_suffix,
            )
            tracer_terms.append(
                apply_linear_sensitivity(
                    tracer_sensitivity,
                    state * ratio_factor,
                    data_name=f"hx_{tracer_suffix}_{variable_suffix}",
                    output_name=f"mu_{tracer_suffix}_{variable_suffix}",
                )
            )

        mu_primary = _sum_terms(primary_terms, name=f"mu_{primary_suffix}", dim=primary_dim)
        mu_tracer = _sum_terms(tracer_terms, name=f"mu_{tracer_suffix}", dim=tracer_dim)
        primary_bc = _add_boundary_component(
            primary_boundary,
            model_spec.primary,
            suffix=primary_suffix,
        )
        tracer_bc = _add_boundary_component(
            tracer_boundary,
            model_spec.tracer,
            suffix=tracer_suffix,
        )
        _add_absolute_error_likelihood(
            prepared_inputs.primary,
            model_spec.primary,
            suffix=primary_suffix,
            mu=mu_primary,
            mu_bc=primary_bc,
        )
        _add_absolute_error_likelihood(
            prepared_inputs.tracer,
            model_spec.tracer,
            suffix=tracer_suffix,
            mu=mu_tracer,
            mu_bc=tracer_bc,
        )

    return model


def run_ramsden_from_prepared_inputs(
    *,
    prepared_inputs: RamsdenPreparedInputs,
    model_spec: RamsdenModelSpec,
    sampler: RhimeSampler | None = None,
) -> RamsdenResult:
    """Build and sample a historical two-gas model from canonical inputs.

    Args:
        prepared_inputs: Canonical primary and tracer datasets.
        model_spec: Historical model specification.
        sampler: Modern RHIME sampling configuration controlling seeds,
            chains, draws, and predictive output. When omitted, the standard
            sampler is used with both namespaced gas observations included in
            posterior predictive sampling.

    Returns:
        Joint result containing the built model and labelled inference data.

    Raises:
        ValueError: If model metadata or prepared inputs are invalid.

    Notes:
        This function samples the model and may run multiple chains. It does
        not retrieve data, convert units, or write postprocessed products.
        Sampling exceptions raised by
        :class:`~openghg_inversions.rhime.sampling.RhimeSampler` are
        propagated.
    """
    model = build_ramsden_model(prepared_inputs, model_spec)
    primary_suffix = _channel_suffix(model_spec.primary)
    tracer_suffix = _channel_suffix(model_spec.tracer)
    predictive_names = (f"y_{primary_suffix}", f"y_{tracer_suffix}")
    if sampler is None:
        sampler = RhimeSampler(sample_posterior_predictive=predictive_names)
    elif sampler.sample_posterior_predictive == ("y",):
        sampler = RhimeSampler(
            draws=sampler.draws,
            burn=sampler.burn,
            tune=sampler.tune,
            chains=sampler.chains,
            nuts_sampler=sampler.nuts_sampler,
            progressbar=sampler.progressbar,
            sample_kwargs=sampler.sample_kwargs,
            sample_prior_predictive=sampler.sample_prior_predictive,
            sample_posterior_predictive=predictive_names,
            posterior_predictive_kwargs=sampler.posterior_predictive_kwargs,
        )
    elif isinstance(sampler.sample_posterior_predictive, tuple):
        missing = set(sampler.sample_posterior_predictive) - set(model.named_vars)
        if missing:
            raise ValueError(
                f"Ramsden sampler posterior-predictive variables are absent from the model: "
                f"{sorted(missing)!r}."
            )
    idata = sampler.sample(model)
    return RamsdenResult(
        prepared_inputs=prepared_inputs,
        model_spec=model_spec,
        model=model,
        idata=idata,
        sampler=sampler,
    )
