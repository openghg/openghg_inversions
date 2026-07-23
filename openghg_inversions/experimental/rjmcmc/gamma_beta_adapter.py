"""Adapt filtered RHIME arrays to the experimental Gamma--Beta tree model.

This module is the explicit real-data seam for the fixed-direction dyadic
baseline.  RHIME ``fp_x_flux`` is a response to a unit scaling factor, whereas
the Gamma--Beta state is expressed in physical-mass coordinates.  Callers
must therefore provide a strictly positive native-grid ``nominal_weight``.
For the weight field used by the model, the adapter computes
``sensitivity_per_mass = fp_x_flux / nominal_weight``.  It never guesses a
flux field, floors zero weights, or discovers boundary-condition variables.

Spatial arrays are transposed to ``(lat, lon)`` and flattened in C order.
Weights are normalized to sum to one by default, making the root-total prior
mean one and preserving the all-one scaling prediction.  If normalization is
disabled, the root prior mean is the supplied weight sum.  Optional fixed
offset and always-active design fields are consumed only through explicitly
named data variables.

The main entry points are :func:`gamma_beta_problem_from_rhime_inputs` and
:func:`initialize_gamma_beta_state`.  The former returns both the immutable
numerical problem and the weight-normalization metadata needed to audit the
coordinate conversion.  The latter constructs a deterministic starting
frontier without consuming random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
)
from openghg_inversions.experimental.rjmcmc.rhime_adapter import (
    _align_nmeasure_exact,
    _fixed_design_dimension,
    _fixed_prior_moments,
    _grid_axis_values,
    _numeric_values,
    _require_exact_dims,
    _require_variable,
)

FloatArray = NDArray[np.float64]
_SENSITIVITY_DIMS = ("nmeasure", "lat", "lon")


@dataclass(frozen=True, slots=True)
class GammaBetaRHIMEAdapterResult:
    """Immutable Gamma--Beta problem plus weight-conversion metadata.

    Attributes:
        problem: Numerical fixed-tree problem in physical-mass coordinates.
        weight_normalization_factor: Sum of the caller-supplied nominal
            weights.  Dividing the supplied field by this value gives the
            model weight field when ``weights_normalized`` is true.
        weights_normalized: Whether the adapter normalized the supplied
            weights to sum to one.
        spatial_shape: Canonical ``(lat, lon)`` native-grid shape.
        latitudes: Read-only latitude coordinate in native row order.
        longitudes: Read-only longitude coordinate in native column order.
    """

    problem: GammaBetaTreeProblem
    weight_normalization_factor: float
    weights_normalized: bool
    spatial_shape: tuple[int, int]
    latitudes: FloatArray
    longitudes: FloatArray

    @property
    def nominal_weight(self) -> FloatArray:
        """Return the read-only C-order weight vector used by the model.

        Returns:
            Strictly positive vector with one entry per native grid cell.
        """
        return self.problem.prior.nominal_cell_mass


def _aligned_nominal_weight(
    sensitivity: xr.DataArray,
    nominal_weight: ArrayLike,
) -> FloatArray:
    """Validate and return nominal weights in canonical spatial order.

    Args:
        sensitivity: Canonical ``(nmeasure, lat, lon)`` sensitivity array.
        nominal_weight: Positive ndarray-like field with shape ``(lat, lon)``,
            or a data array containing exactly those dimensions in any order.

    Returns:
        Owned, strictly positive ``float64`` array with shape ``(lat, lon)``.

    Raises:
        ValueError: If dimensions, coordinate alignment, shape, finiteness, or
            positivity are invalid.
    """
    expected_shape = (sensitivity.sizes["lat"], sensitivity.sizes["lon"])
    if isinstance(nominal_weight, xr.DataArray):
        _require_exact_dims(nominal_weight, "nominal_weight", ("lat", "lon"))
        candidate = nominal_weight.transpose("lat", "lon")
        reference = sensitivity.isel(nmeasure=0, drop=True)
        try:
            _, candidate = xr.align(reference, candidate, join="exact", copy=False)
        except ValueError as error:
            raise ValueError(
                "nominal_weight must align exactly with fp_x_flux along 'lat' and 'lon'."
            ) from error
        values = _numeric_values(candidate, "nominal_weight")
    else:
        try:
            values = np.asarray(nominal_weight, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("nominal_weight must contain numeric values.") from error
    if values.shape != expected_shape:
        raise ValueError(f"nominal_weight must have shape {expected_shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("nominal_weight must contain only finite values.")
    if np.any(values <= 0.0):
        raise ValueError("nominal_weight must be strictly positive in every cell.")
    return np.array(values, dtype=np.float64, copy=True, order="C")


def _fixed_components(
    sensitivity: xr.DataArray,
    inv_inputs: xr.Dataset,
    *,
    fixed_design_name: str | None,
    fixed_offset_name: str | None,
    fixed_coefficient_prior_mean: ArrayLike | None,
    fixed_coefficient_prior_sd: ArrayLike | None,
) -> tuple[FixedDesignBlock | None, FloatArray | None]:
    """Build explicitly selected fixed model components.

    Args:
        sensitivity: Canonical sensitivity used as the measurement reference.
        inv_inputs: Filtered RHIME input dataset.
        fixed_design_name: Explicit optional fixed-design variable name.
        fixed_offset_name: Explicit optional fixed-offset variable name.
        fixed_coefficient_prior_mean: Scalar or per-column arithmetic means.
        fixed_coefficient_prior_sd: Scalar or per-column arithmetic standard
            deviations.

    Returns:
        Optional immutable fixed design block and optional offset vector.

    Raises:
        ValueError: If selection, dimensions, alignment, values, or prior
            moments are malformed.
    """
    fixed_block = None
    if fixed_design_name is None:
        if fixed_coefficient_prior_mean is not None or fixed_coefficient_prior_sd is not None:
            raise ValueError("fixed coefficient prior moments require an explicit fixed_design_name.")
    else:
        if fixed_coefficient_prior_mean is None or fixed_coefficient_prior_sd is None:
            raise ValueError(
                "fixed_design_name requires both fixed_coefficient_prior_mean and fixed_coefficient_prior_sd."
            )
        fixed_design = _require_variable(inv_inputs, fixed_design_name)
        coefficient_dimension = _fixed_design_dimension(fixed_design, fixed_design_name)
        fixed_design = _align_nmeasure_exact(sensitivity, fixed_design, fixed_design_name)
        fixed_design = fixed_design.transpose("nmeasure", coefficient_dimension)
        design_values = _numeric_values(fixed_design, fixed_design_name)
        if not np.all(np.isfinite(design_values)):
            raise ValueError(f"{fixed_design_name!r} must contain only finite values.")
        n_fixed = fixed_design.sizes[coefficient_dimension]
        fixed_block = FixedDesignBlock(
            design=design_values,
            coefficient_prior_mean=_fixed_prior_moments(
                fixed_coefficient_prior_mean,
                name="fixed_coefficient_prior_mean",
                n_columns=n_fixed,
            ),
            coefficient_prior_sd=_fixed_prior_moments(
                fixed_coefficient_prior_sd,
                name="fixed_coefficient_prior_sd",
                n_columns=n_fixed,
            ),
        )

    fixed_offset = None
    if fixed_offset_name is not None:
        fixed_offset_array = _require_variable(inv_inputs, fixed_offset_name)
        _require_exact_dims(fixed_offset_array, fixed_offset_name, ("nmeasure",))
        fixed_offset_array = _align_nmeasure_exact(
            sensitivity,
            fixed_offset_array,
            fixed_offset_name,
        )
        fixed_offset = _numeric_values(fixed_offset_array, fixed_offset_name)
        if not np.all(np.isfinite(fixed_offset)):
            raise ValueError(f"{fixed_offset_name!r} must contain only finite values.")
    return fixed_block, fixed_offset


def gamma_beta_problem_from_rhime_inputs(
    inv_inputs: xr.Dataset,
    *,
    nominal_weight: ArrayLike,
    k_min: int,
    k_max: int,
    concentration: float,
    root_variance: float = 0.25,
    probabilities_by_k: ArrayLike | None = None,
    normalize_weights: bool = True,
    likelihood_power: float = 1.0,
    sensitivity_name: str = "fp_x_flux",
    observation_name: str = "mf",
    observation_sd_name: str = "mf_error",
    fixed_design_name: str | None = None,
    fixed_offset_name: str | None = None,
    fixed_coefficient_prior_mean: ArrayLike | None = None,
    fixed_coefficient_prior_sd: ArrayLike | None = None,
) -> GammaBetaRHIMEAdapterResult:
    """Build a Gamma--Beta tree problem from filtered native RHIME inputs.

    ``fp_x_flux`` is interpreted as the response to unit scaling.  The
    Gamma--Beta model instead acts on physical mass, so the returned
    sensitivity is exactly ``fp_x_flux / model_nominal_weight``.  The caller
    must supply the weight field explicitly and every weight must be strictly
    positive.  No zero flooring or boundary-condition discovery occurs.

    Args:
        inv_inputs: Filtered prepared RHIME dataset.
        nominal_weight: Positive native-grid mass/flux weights aligned to
            ``lat`` and ``lon``.
        k_min: Smallest active frontier size with prior support.
        k_max: Largest active frontier size with prior support.
        concentration: Positive common Beta split concentration.
        root_variance: Positive Gamma root-total prior variance in the model's
            weight units.
        probabilities_by_k: Optional non-negative K-indexed prior masses.  If
            omitted, the marginal prior is uniform from ``k_min`` through
            ``k_max``.  Supplied masses must assign mass only inside that same
            range and are normalized by :class:`TreePartitionPrior`.
        normalize_weights: Normalize supplied weights to sum to one.  With the
            default, the root prior mean is one; otherwise it is the original
            weight sum.
        likelihood_power: Non-negative Gaussian likelihood multiplier.
        sensitivity_name: Unit-scaling sensitivity data-variable name.
        observation_name: Observation data-variable name.
        observation_sd_name: Fixed observation-error data-variable name.
        fixed_design_name: Optional explicitly selected always-active design.
        fixed_offset_name: Optional explicitly selected additive offset.
        fixed_coefficient_prior_mean: Positive arithmetic lognormal-prior
            means for selected fixed columns.
        fixed_coefficient_prior_sd: Positive arithmetic lognormal-prior
            standard deviations for selected fixed columns.

    Returns:
        Immutable problem and auditable weight-normalization metadata.

    Raises:
        TypeError: If ``inv_inputs`` is not a dataset or
            ``normalize_weights`` is not Boolean.
        ValueError: If required variables, dimensions, coordinates, K support,
            weights, fixed components, or numerical values are malformed.
    """
    if not isinstance(inv_inputs, xr.Dataset):
        raise TypeError("inv_inputs must be an xarray.Dataset.")
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise TypeError("normalize_weights must be Boolean.")

    sensitivity = _require_variable(inv_inputs, sensitivity_name)
    observations = _require_variable(inv_inputs, observation_name)
    observation_sd = _require_variable(inv_inputs, observation_sd_name)
    _require_exact_dims(sensitivity, sensitivity_name, _SENSITIVITY_DIMS)
    _require_exact_dims(observations, observation_name, ("nmeasure",))
    _require_exact_dims(observation_sd, observation_sd_name, ("nmeasure",))
    sensitivity = sensitivity.transpose(*_SENSITIVITY_DIMS)
    try:
        sensitivity, observations, observation_sd = xr.align(
            sensitivity,
            observations,
            observation_sd,
            join="exact",
            copy=False,
        )
    except ValueError as error:
        raise ValueError(
            f"{observation_name!r} and {observation_sd_name!r} must align "
            f"exactly with {sensitivity_name!r} along 'nmeasure'."
        ) from error

    supplied_weight = _aligned_nominal_weight(sensitivity, nominal_weight)
    normalization_factor = float(supplied_weight.sum())
    if not np.isfinite(normalization_factor) or normalization_factor <= 0.0:
        raise ValueError("nominal_weight must have a finite, strictly positive sum.")
    if normalize_weights:
        model_weight = supplied_weight / normalization_factor
    else:
        model_weight = supplied_weight

    sensitivity_values = _numeric_values(sensitivity, sensitivity_name)
    if not np.all(np.isfinite(sensitivity_values)):
        raise ValueError(f"{sensitivity_name!r} must contain only finite values.")
    per_mass_sensitivity = sensitivity_values / model_weight[np.newaxis, :, :]
    per_mass_sensitivity = per_mass_sensitivity.reshape(
        sensitivity.sizes["nmeasure"],
        -1,
        order="C",
    )

    spatial_shape = (int(model_weight.shape[0]), int(model_weight.shape[1]))
    tree = CanonicalDyadicTree.from_shape(spatial_shape)
    root_mean = 1.0 if normalize_weights else normalization_factor
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        model_weight,
        concentration=concentration,
        root_mean=root_mean,
        root_variance=root_variance,
    )
    uniform_partition_prior = TreePartitionPrior.uniform_k(
        tree,
        minimum_k=k_min,
        maximum_k=k_max,
    )
    if probabilities_by_k is None:
        partition_prior = uniform_partition_prior
    else:
        try:
            masses = np.asarray(probabilities_by_k, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("probabilities_by_k must contain numeric values.") from error
        if masses.ndim != 1:
            raise ValueError("probabilities_by_k must be a one-dimensional K-indexed vector.")
        positive_indices = np.flatnonzero(masses > 0.0)
        if np.any((positive_indices < k_min) | (positive_indices > k_max)):
            raise ValueError(
                "probabilities_by_k must assign zero mass outside the declared k_min through k_max."
            )
        padded = np.zeros(max(masses.size, k_max + 1), dtype=np.float64)
        padded[: masses.size] = masses
        partition_prior = TreePartitionPrior.from_marginal_probabilities(tree, padded)

    fixed_block, fixed_offset = _fixed_components(
        sensitivity,
        inv_inputs,
        fixed_design_name=fixed_design_name,
        fixed_offset_name=fixed_offset_name,
        fixed_coefficient_prior_mean=fixed_coefficient_prior_mean,
        fixed_coefficient_prior_sd=fixed_coefficient_prior_sd,
    )
    problem = GammaBetaTreeProblem(
        observations=_numeric_values(observations, observation_name),
        observation_sd=_numeric_values(observation_sd, observation_sd_name),
        sensitivity=per_mass_sensitivity,
        prior=prior,
        partition_prior=partition_prior,
        likelihood_power=likelihood_power,
        fixed_offset=fixed_offset,
        fixed_block=fixed_block,
    )
    latitudes = np.array(_grid_axis_values(sensitivity, "lat"), copy=True)
    longitudes = np.array(_grid_axis_values(sensitivity, "lon"), copy=True)
    latitudes.setflags(write=False)
    longitudes.setflags(write=False)
    return GammaBetaRHIMEAdapterResult(
        problem=problem,
        weight_normalization_factor=normalization_factor,
        weights_normalized=bool(normalize_weights),
        spatial_shape=spatial_shape,
        latitudes=latitudes,
        longitudes=longitudes,
    )


def initialize_gamma_beta_state(
    problem: GammaBetaTreeProblem,
    *,
    k: int,
) -> GammaBetaTreeState:
    """Construct a deterministic prior-mean Gamma--Beta starting state.

    Starting from the root, this initializer repeatedly splits the eligible
    active leaf with largest nominal mass.  Equal masses are resolved by the
    stable node ID.  Active fractions use their Beta means, the root total
    uses its Gamma mean, and always-active coefficients use their arithmetic
    lognormal-prior means.

    Args:
        problem: Immutable Gamma--Beta tree problem.
        k: Requested active frontier size with positive partition-prior mass.

    Returns:
        Fully built immutable starting state.

    Raises:
        TypeError: If ``problem`` or ``k`` has the wrong type.
        ValueError: If ``k`` is outside tree or partition-prior support.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer.")
    selected_k = int(k)
    if not 1 <= selected_k <= len(problem.tree.leaf_ids):
        raise ValueError(f"k must lie between 1 and {len(problem.tree.leaf_ids)}.")
    probabilities = problem.partition_prior.p_k
    if selected_k >= probabilities.size or probabilities[selected_k] <= 0.0:
        raise ValueError(f"k={selected_k} has zero partition-prior probability.")

    frontier = DyadicFrontier.root(problem.tree)
    while len(frontier) < selected_k:
        eligible = problem.tree.splittable_nodes(frontier)
        if not eligible:
            raise ValueError(f"The tree cannot construct a frontier with k={selected_k}.")
        selected_node = min(
            eligible,
            key=lambda node_id: (-float(problem.node_nominal_mass[node_id]), node_id),
        )
        frontier = frontier.split(problem.tree, selected_node)

    split_nodes = frontier.active_split_nodes(problem.tree)
    fractions = np.array(
        [
            alpha / (alpha + beta)
            for alpha, beta in (problem.prior.beta_parameters(node_id) for node_id in split_nodes)
        ],
        dtype=np.float64,
    )
    root_total = problem.prior.root_shape / problem.prior.root_rate
    fixed_coefficients: ArrayLike | None = None
    if problem.fixed_block is not None:
        fixed_coefficients = problem.fixed_block.coefficient_prior_mean
    return build_gamma_beta_tree_state(
        problem,
        frontier=frontier,
        root_total=root_total,
        active_fractions=fractions,
        fixed_coefficients=fixed_coefficients,
    )


__all__ = [
    "GammaBetaRHIMEAdapterResult",
    "gamma_beta_problem_from_rhime_inputs",
    "initialize_gamma_beta_state",
]
