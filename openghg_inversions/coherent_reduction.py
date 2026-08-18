"""Exact labelled Gaussian reduction from native covariance products.

The public operation in this module is a named eager numerical boundary. It
accepts one labelled native mean, covariance, sensitivity, and retained basis;
constructs their linked product blocks with
:func:`openghg_inversions.basis.project_native_covariance`; and reduces them
before the native sensitivity can be substituted. It materializes the related
mean and sensitivity together and returns one backend-neutral scientific
handoff.

For ``x ~ N(m, B)`` and ``alpha = Pi x``, the reduced conditional observation
model is

``y | alpha ~ N(H m + H_alpha (alpha - Pi m), R + A)``,

where ``H_alpha = H B Pi.T C_alpha^-1`` and
``A = H B H.T - H_alpha C_alpha H_alpha.T``. The implementation uses a
Cholesky solve and never forms an explicit inverse.

This first exact contract is intentionally local. Approximation, persistence,
likelihood adapters, and arbitrary reporting-functional reconstruction are
separate downstream operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from dask.base import compute
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import xarray as xr
from xarray.namedarray.pycompat import is_chunked_array

from openghg_inversions._labelled_matrices import with_square_matrix_diagnostics
from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.covariance_products import (
    _preflight_native_covariance_projection,
    NativeCovarianceProducts,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

MAX_COHERENT_SOLVE_RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class CoherentGaussianReduction:
    """One exact, labelled native-to-retained Gaussian reduction.

    The fields are eager construction-time snapshots. Freezing prevents field
    reassignment but does not make the contained xarray objects immutable; the
    object deliberately uses identity equality rather than generated xarray
    value comparisons.

    Attributes:
        retained_mean: Dimensionless ``Pi m`` on the retained-state axis.
        retained_covariance: Positive-definite ``C_alpha`` on distinct
            retained-state row and column axes.
        effective_observation_operator: ``H_alpha`` mapping retained-state
            perturbations to observations.
        native_observation_mean: ``H m`` on the observation axis.
        observation_intercept: Affine intercept ``H m - H_alpha Pi m``.
        unresolved_observation_covariance: Positive-semidefinite ``A`` on
            distinct observation row and column axes.
        projection_strategy: Scientific strategy that selected ``Pi``.
    """

    retained_mean: xr.DataArray
    retained_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    native_observation_mean: xr.DataArray
    observation_intercept: xr.DataArray
    unresolved_observation_covariance: xr.DataArray
    projection_strategy: str


def reduce_native_gaussian(
    *,
    covariance: InvertibleNativeCovarianceAction,
    basis_prolongation: xr.DataArray,
    state_dim: str,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_batch_size: int = 64,
    strategy: RetainedProjectionStrategy | None = None,
) -> CoherentGaussianReduction:
    """Atomically prepare the exact centred Gaussian model for one ``B/H/Pi`` set.

    ``native_mean`` and ``native_sensitivity`` are borrowed and may be
    Dask-backed. They are converted to dense form and materialized together at
    this named boundary so a shared Dask graph is executed only once. The
    operation then constructs the dense OPE-17 product view locally and reduces
    it before the source sensitivity can be substituted or mixed with products
    from another native model.

    Args:
        covariance: Invertible labelled native covariance action ``B``.
        basis_prolongation: Canonical eager basis prolongation used by the
            retained projection strategy.
        state_dim: Retained-state dimension shared by the prolongation and
            restriction.
        native_mean: Native scaling mean ``m`` with exactly the product's
            ordered native dimensions and declared dimensionless units.
        native_sensitivity: Native sensitivity ``H`` with the observation
            dimension followed by the ordered native dimensions.
        observation_dim: Observation dimension in ``native_sensitivity``.
        observation_batch_size: Positive covariance right-hand-side batch
            size forwarded explicitly to the product projection.
        strategy: Optional authoritative retained restriction strategy.

    Returns:
        One eager labelled result containing the retained prior, centred
        effective forward model, and unresolved observation covariance.

    Raises:
        TypeError: If public inputs are not the documented xarray/product
            types.
        ValueError: If dimensions, coordinates, units, values, covariance
            definiteness, or coherent product identities disagree.
    """
    if not isinstance(native_mean, xr.DataArray):
        raise TypeError("native_mean must be an xarray.DataArray")
    if not isinstance(native_sensitivity, xr.DataArray):
        raise TypeError("native_sensitivity must be an xarray.DataArray")
    if not isinstance(basis_prolongation, xr.DataArray):
        raise TypeError("basis_prolongation must be an xarray.DataArray")
    if not isinstance(state_dim, str) or not state_dim:
        raise ValueError("state_dim must be a non-empty string")
    if not isinstance(observation_dim, str) or not observation_dim:
        raise ValueError("observation_dim must be a non-empty string")
    if not all(isinstance(dim, str) for dim in covariance.native_dims):
        raise ValueError("covariance native dimensions must have string names")
    native_dims = tuple(str(dim) for dim in covariance.native_dims)
    _require_dimensions(native_mean, "native_mean", native_dims)
    _require_dimensions(
        native_sensitivity,
        "native_sensitivity",
        (observation_dim, *native_dims),
    )
    _require_axes_equal(native_sensitivity, native_mean, native_dims, context="native_mean")
    if native_mean.attrs.get("units") != "1":
        raise ValueError("native_mean must be dimensionless with units '1'")
    _preflight_native_covariance_projection(
        covariance=covariance,
        basis_prolongation=basis_prolongation,
        state_dim=state_dim,
        native_sensitivity=native_sensitivity,
        observation_dim=observation_dim,
        observation_covariance="dense",
        observation_batch_size=observation_batch_size,
    )
    mean, sensitivity = compute(to_dense(native_mean), to_dense(native_sensitivity))
    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=basis_prolongation,
        state_dim=state_dim,
        native_sensitivity=sensitivity,
        observation_dim=observation_dim,
        observation_covariance="dense",
        observation_batch_size=observation_batch_size,
        strategy=strategy,
    )
    return _reduce_native_covariance_products(
        products=products,
        native_mean=mean,
        native_sensitivity=sensitivity,
    )


def _reduce_native_covariance_products(
    *,
    products: NativeCovarianceProducts,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
) -> CoherentGaussianReduction:
    """Reduce one locally bound product set after the public atomic boundary."""
    if not isinstance(products, NativeCovarianceProducts):
        raise TypeError("products must be a NativeCovarianceProducts instance")
    if products.observation_covariance_view != "dense":
        raise ValueError(
            "Coherent Gaussian reduction requires dense H B H.T products; "
            "a diagonal view cannot determine unresolved covariance"
        )

    restriction = _require_array(products.restriction, "products.restriction")
    if restriction.ndim < 2:
        raise ValueError("products.restriction must contain a state axis and native dimensions")
    if not all(isinstance(dim, str) for dim in restriction.dims):
        raise ValueError("products.restriction dimensions must have string names")
    state_dim = str(restriction.dims[0])
    native_dims = tuple(str(dim) for dim in restriction.dims[1:])
    if len(set(restriction.dims)) != len(restriction.dims):
        raise ValueError("products.restriction dimensions must be distinct")

    effective_input = _require_array(
        products.effective_observation_operator,
        "products.effective_observation_operator",
    )
    if effective_input.ndim != 2 or effective_input.dims[1] != state_dim:
        raise ValueError(
            "products.effective_observation_operator must have observation and retained-state dimensions"
        )
    if not isinstance(effective_input.dims[0], str):
        raise ValueError("The observation dimension must have a string name")
    observation_dim = str(effective_input.dims[0])
    if observation_dim in restriction.dims:
        raise ValueError("Observation and native/retained dimension names must be distinct")
    restriction = _validated_numeric_array(
        restriction,
        name="products.restriction",
        expected_dims=(state_dim, *native_dims),
    )
    effective_input = _validated_numeric_array(
        effective_input,
        name="products.effective_observation_operator",
        expected_dims=(observation_dim, state_dim),
    )
    if not isinstance(products.strategy, str) or not products.strategy:
        raise ValueError("products.strategy must be a non-empty string")

    _require_dimensions(native_mean, "native_mean", native_dims)
    _require_dimensions(
        native_sensitivity,
        "native_sensitivity",
        (observation_dim, *native_dims),
    )
    _require_axes_equal(restriction, native_mean, native_dims, context="native_mean")
    _require_axes_equal(
        restriction,
        native_sensitivity,
        native_dims,
        context="native_sensitivity",
    )
    _require_axis_equal(
        effective_input,
        observation_dim,
        native_sensitivity,
        observation_dim,
        context="native_sensitivity observations",
    )
    mean, sensitivity = compute(to_dense(native_mean), to_dense(native_sensitivity))
    mean = _validated_numeric_array(
        mean,
        name="native_mean",
        expected_dims=native_dims,
    )
    sensitivity = _validated_numeric_array(
        sensitivity,
        name="native_sensitivity",
        expected_dims=(observation_dim, *native_dims),
    )

    prolongation = _validated_numeric_array(
        _require_array(products.prolongation, "products.prolongation"),
        name="products.prolongation",
        expected_dims=(*native_dims, state_dim),
    )
    _require_axes_equal(restriction, prolongation, native_dims, context="products.prolongation")
    _require_axis_equal(
        restriction,
        state_dim,
        prolongation,
        state_dim,
        context="products.prolongation states",
    )
    _require_axis_equal(
        restriction,
        state_dim,
        effective_input,
        state_dim,
        context="products.effective_observation_operator states",
    )
    state_covariance = _require_array(products.state_covariance, "products.state_covariance")
    if state_covariance.ndim != 2 or state_covariance.dims[0] != state_dim:
        raise ValueError("products.state_covariance must begin with the retained-state dimension")
    if not isinstance(state_covariance.dims[1], str):
        raise ValueError("The retained covariance column dimension must have a string name")
    state_column_dim = str(state_covariance.dims[1])
    if state_column_dim == state_dim:
        raise ValueError("products.state_covariance requires distinct row and column dimensions")
    state_covariance = _validated_numeric_array(
        state_covariance,
        name="products.state_covariance",
        expected_dims=(state_dim, state_column_dim),
    )
    _require_axis_equal(
        state_covariance,
        state_dim,
        state_covariance,
        state_column_dim,
        context="products.state_covariance columns",
    )
    _require_axis_equal(
        restriction,
        state_dim,
        state_covariance,
        state_dim,
        context="products.state_covariance states",
    )
    state_covariance = with_square_matrix_diagnostics(
        state_covariance,
        mathematical_name="C_alpha",
        require_positive_definite=True,
    )
    condition_number = float(state_covariance.attrs["condition_number"])
    solve_relative_tolerance = float(
        max(
            1e-10,
            64.0 * np.finfo(np.float64).eps * condition_number,
        )
    )
    if solve_relative_tolerance > MAX_COHERENT_SOLVE_RELATIVE_TOLERANCE:
        raise ValueError(
            "C_alpha is too ill-conditioned for the coherent solve tolerance; "
            "choose a non-redundant retained restriction"
        )
    pi_u = (
        np.asarray(restriction.values).reshape(restriction.sizes[state_dim], -1)
        @ np.asarray(prolongation.values).reshape(-1, prolongation.sizes[state_dim])
    )
    _require_close(
        pi_u,
        np.eye(restriction.sizes[state_dim]),
        context="Pi U_* and the retained-state identity",
        relative_tolerance=solve_relative_tolerance,
    )

    cross_covariance = _validated_numeric_array(
        _require_array(
            products.observation_state_cross_covariance,
            "products.observation_state_cross_covariance",
        ),
        name="products.observation_state_cross_covariance",
        expected_dims=(observation_dim, state_column_dim),
    )
    _require_axis_equal(
        effective_input,
        observation_dim,
        cross_covariance,
        observation_dim,
        context="products.observation_state_cross_covariance observations",
    )
    _require_axis_equal(
        state_covariance,
        state_column_dim,
        cross_covariance,
        state_column_dim,
        context="products.observation_state_cross_covariance states",
    )

    native_observation_covariance = _require_array(
        products.native_observation_covariance,
        "products.native_observation_covariance",
    )
    if native_observation_covariance.ndim != 2:
        raise ValueError("products.native_observation_covariance must be a dense square matrix")
    if not isinstance(native_observation_covariance.dims[1], str):
        raise ValueError("The observation covariance column dimension must have a string name")
    observation_column_dim = str(native_observation_covariance.dims[1])
    native_observation_covariance = _validated_numeric_array(
        native_observation_covariance,
        name="products.native_observation_covariance",
        expected_dims=(observation_dim, observation_column_dim),
    )
    _require_axis_equal(
        native_observation_covariance,
        observation_dim,
        native_observation_covariance,
        observation_column_dim,
        context="products.native_observation_covariance columns",
    )
    _require_axis_equal(
        effective_input,
        observation_dim,
        native_observation_covariance,
        observation_dim,
        context="products.native_observation_covariance observations",
    )
    native_observation_covariance = with_square_matrix_diagnostics(
        native_observation_covariance,
        mathematical_name="H B H.T",
    )

    _validate_units(
        mean=mean,
        sensitivity=sensitivity,
        restriction=restriction,
        prolongation=prolongation,
        state_covariance=state_covariance,
        effective_operator=effective_input,
        cross_covariance=cross_covariance,
        native_observation_covariance=native_observation_covariance,
    )

    covariance_values = np.asarray(state_covariance.values, dtype=np.float64)
    cross_values = np.asarray(cross_covariance.values, dtype=np.float64)
    try:
        factor = cho_factor(covariance_values, lower=True, check_finite=True)
        solved_values = cho_solve(factor, cross_values.T, check_finite=True).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("C_alpha must be positive definite; no pseudoinverse is used") from exc
    _require_solve_residual(
        covariance_values,
        solved_values.T,
        cross_values.T,
        relative_tolerance=solve_relative_tolerance,
    )

    effective_operator = effective_input.copy(data=solved_values).rename(
        "effective_observation_operator"
    )
    effective_operator = effective_operator.assign_attrs(
        {
            **effective_input.attrs,
            "mathematical_name": "H_alpha",
            "definition": "H B Pi.T C_alpha^-1",
        }
    )
    _require_close(
        solved_values,
        np.asarray(effective_input.values),
        context="stored H_alpha and the solve from C_alpha/H B Pi.T",
        relative_tolerance=solve_relative_tolerance,
    )
    direct_effective = xr.dot(sensitivity, prolongation, dim=list(native_dims)).transpose(
        observation_dim,
        state_dim,
    )
    _require_close(
        solved_values,
        np.asarray(direct_effective.values),
        context="solved H_alpha and H U_*",
        relative_tolerance=solve_relative_tolerance,
    )

    retained_mean = xr.dot(restriction, mean, dim=list(native_dims)).transpose(state_dim)
    retained_mean = retained_mean.rename("retained_mean").assign_attrs(
        mathematical_name="Pi m",
        units="1",
    )
    native_observation_mean = xr.dot(sensitivity, mean, dim=list(native_dims)).transpose(
        observation_dim
    )
    sensitivity_units = sensitivity.attrs.get("units")
    linear_units = {"units": sensitivity_units} if sensitivity_units is not None else {}
    native_observation_mean = native_observation_mean.rename("native_observation_mean").assign_attrs(
        mathematical_name="H m",
        **linear_units,
    )
    retained_contribution = xr.dot(effective_operator, retained_mean, dim=state_dim)
    observation_intercept = (native_observation_mean - retained_contribution).rename(
        "observation_intercept"
    )
    observation_intercept = observation_intercept.assign_attrs(
        mathematical_name="H m - H_alpha Pi m",
        **linear_units,
    )

    hbh_values = np.asarray(native_observation_covariance.values, dtype=np.float64)
    explained_values = solved_values @ cross_values.T
    unresolved_values = hbh_values - explained_values
    subtraction_scale = max(
        float(np.max(np.abs(hbh_values))) if hbh_values.size else 0.0,
        float(np.max(np.abs(explained_values))) if explained_values.size else 0.0,
    )
    asymmetry = (
        float(np.max(np.abs(unresolved_values - unresolved_values.T)))
        if unresolved_values.size
        else 0.0
    )
    if asymmetry > solve_relative_tolerance * subtraction_scale:
        raise ValueError("Incoherent covariance products: unresolved covariance is not symmetric")
    unresolved_values = (unresolved_values + unresolved_values.T) * 0.5
    unresolved_covariance = native_observation_covariance.copy(data=unresolved_values).rename(
        "unresolved_observation_covariance"
    )
    unresolved_covariance = unresolved_covariance.assign_attrs(
        mathematical_name="A = H B_perp H.T",
        definition="H B H.T - H_alpha C_alpha H_alpha.T",
    )
    unresolved_covariance = with_square_matrix_diagnostics(
        unresolved_covariance,
        mathematical_name="A = H B_perp H.T",
        require_positive_semidefinite=True,
        positive_semidefinite_tolerance_scale=subtraction_scale,
        positive_semidefinite_relative_tolerance=solve_relative_tolerance,
    )
    reconstructed_hbh = unresolved_values + solved_values @ covariance_values @ solved_values.T
    _require_close(
        reconstructed_hbh,
        np.asarray(native_observation_covariance.values),
        context="A + H_alpha C_alpha H_alpha.T and H B H.T",
        relative_tolerance=solve_relative_tolerance,
    )

    return CoherentGaussianReduction(
        retained_mean=retained_mean,
        retained_covariance=state_covariance.copy(deep=True).rename("retained_covariance"),
        effective_observation_operator=effective_operator,
        native_observation_mean=native_observation_mean,
        observation_intercept=observation_intercept,
        unresolved_observation_covariance=unresolved_covariance,
        projection_strategy=products.strategy,
    )


def _require_array(value: object, name: str) -> xr.DataArray:
    """Return one required DataArray or fail at the product boundary."""
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    return value


def _validated_numeric_array(
    array: xr.DataArray,
    *,
    name: str,
    expected_dims: tuple[str, ...],
) -> xr.DataArray:
    """Validate one eager finite-real array with an exact dimension order."""
    if array.dims != expected_dims:
        raise ValueError(f"{name} must have dimensions {expected_dims!r}; got {array.dims!r}")
    if is_chunked_array(array.data):
        raise ValueError(f"{name} must be eager at the coherent-reduction boundary")
    values = np.asarray(array.values)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_dimensions(
    array: xr.DataArray,
    name: str,
    expected_dims: tuple[str, ...],
) -> None:
    """Validate dimensions before any eager payload materialization."""
    if array.dims != expected_dims:
        raise ValueError(f"{name} must have dimensions {expected_dims!r}; got {array.dims!r}")


def _require_axes_equal(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    dims: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Require exact ordered indexes on several same-named dimensions."""
    for dim in dims:
        _require_axis_equal(reference, dim, candidate, dim, context=context)


def _require_axis_equal(
    left: xr.DataArray,
    left_dim: str,
    right: xr.DataArray,
    right_dim: str,
    *,
    context: str,
) -> None:
    """Require one typed labelled axis to match another exactly and in order."""
    if left_dim not in left.indexes or right_dim not in right.indexes:
        raise ValueError(f"{context} requires labelled indexes")
    left_index = left.get_index(left_dim)
    right_index = right.get_index(right_dim)
    if not left_index.is_unique or not right_index.is_unique:
        raise ValueError(f"{context} indexes must contain unique labels")
    if not left_index.equals(right_index):
        raise ValueError(f"{context} labels must match exactly in the same order")


def _validate_units(
    *,
    mean: xr.DataArray,
    sensitivity: xr.DataArray,
    restriction: xr.DataArray,
    prolongation: xr.DataArray,
    state_covariance: xr.DataArray,
    effective_operator: xr.DataArray,
    cross_covariance: xr.DataArray,
    native_observation_covariance: xr.DataArray,
) -> None:
    """Reject contradictory declared units without requiring optional metadata."""
    for name, array in (
        ("native_mean", mean),
        ("restriction", restriction),
        ("prolongation", prolongation),
        ("state_covariance", state_covariance),
    ):
        units = array.attrs.get("units")
        if units != "1":
            raise ValueError(f"{name} must be dimensionless with units '1'; got {units!r}")
    sensitivity_units = sensitivity.attrs.get("units")
    if sensitivity_units is None:
        declared = {
            name: array.attrs.get("units")
            for name, array in (
                ("effective_observation_operator", effective_operator),
                ("observation_state_cross_covariance", cross_covariance),
                ("native_observation_covariance", native_observation_covariance),
            )
            if array.attrs.get("units") is not None
        }
        if declared:
            raise ValueError(
                "native_sensitivity units are missing while linked covariance products declare units"
            )
        return
    for name, array in (
        ("effective_observation_operator", effective_operator),
        ("observation_state_cross_covariance", cross_covariance),
    ):
        units = array.attrs.get("units")
        if units != sensitivity_units:
            raise ValueError(f"{name} units {units!r} do not match native sensitivity units")
    covariance_units = native_observation_covariance.attrs.get("units")
    expected_covariance_units = f"({sensitivity_units})^2"
    if covariance_units != expected_covariance_units:
        raise ValueError(
            "native_observation_covariance units must be the square of native sensitivity units"
        )


def _require_solve_residual(
    matrix: np.ndarray,
    solution: np.ndarray,
    right_hand_side: np.ndarray,
    *,
    relative_tolerance: float,
) -> None:
    """Require a scale-aware Cholesky solve residual."""
    residual = matrix @ solution - right_hand_side
    scale = max(
        float(np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(solution, ord=np.inf)),
        float(np.linalg.norm(right_hand_side, ord=np.inf)),
    )
    tolerance = relative_tolerance * scale if scale > 0.0 else 0.0
    if float(np.linalg.norm(residual, ord=np.inf)) > tolerance:
        raise ValueError("C_alpha solve residual exceeds the scale-aware tolerance")


def _require_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    context: str,
    relative_tolerance: float = 1e-10,
) -> None:
    """Require one coherent numerical identity with a scale-aware tolerance."""
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    scale = max(
        float(np.max(np.abs(actual_values))) if actual_values.size else 0.0,
        float(np.max(np.abs(expected_values))) if expected_values.size else 0.0,
    )
    tolerance = relative_tolerance * scale if scale > 0.0 else 0.0
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=relative_tolerance,
        atol=tolerance,
    ):
        raise ValueError(f"Incoherent covariance products: {context} disagree")
