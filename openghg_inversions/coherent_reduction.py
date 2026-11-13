"""Exact labelled Gaussian reduction from one native model.

For ``x ~ N(m, B)`` and ``alpha = Pi x``, the reduced conditional
observation model is

``y | alpha ~ N(H m + H_alpha (alpha - Pi m), R + A)``,

where ``H_alpha = H B Pi.T C_alpha^-1`` and
``A = H B H.T - H_alpha C_alpha H_alpha.T``. The public function constructs
all linked covariance products from the same ``B/H/Pi`` inputs before applying
these equations.

The same reduction splits a native state as
``x = m + U_* (alpha - Pi m) + x_perp``, with ``U_* = B Pi.T C_alpha^-1`` and
``x_perp`` independent of ``alpha`` with covariance
``B_perp = B - U_* C_alpha U_*.T``. For a labelled native linear functional
``Q``, such as a country total, :func:`project_native_functionals` returns the
products needed to condition ``Q x_perp`` on the data as well as on
``alpha``. Without them a functional reports only ``Q m + D (alpha - Pi m)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from dask.base import compute
import numpy as np
from sparse import COO, SparseArray  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions._labelled_matrices import (
    coordinates_on_dims,
    matrix_column_dim,
    square_matrix,
    to_column_axis,
)
from openghg_inversions.array_ops import (
    materialize_together,
    require_unique_index,
    same_index,
    sparse_xr_dot,
    to_dense,
)
from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction


@dataclass(frozen=True, slots=True, eq=False)
class NativeFunctionalProducts:
    """Reduction-side products for labelled native linear functionals ``Q``.

    With ``x = m + U_* (alpha - Pi m) + x_perp``, a functional splits as
    ``Q x = Q m + D (alpha - Pi m) + Q x_perp`` with ``D = Q U_*``. Given
    ``alpha``, any other model parameters ``beta`` and the data ``y``,
    ``Q x_perp`` is Gaussian with

    - mean ``Q B_perp H.T (A + R)^-1 r``, where
      ``r = y - y_0 - H_alpha (alpha - Pi m) - G beta`` is the draw's residual;
    - covariance ``Q B_perp Q.T - Q B_perp H.T (A + R)^-1 H B_perp Q.T``.

    Under the prior, ``Q x_perp`` is independent of ``alpha`` with covariance
    ``Q B_perp Q.T``. For a Gaussian retained state and the exact ``A``, the
    resulting functional mean and covariance equal the native-grid Gaussian
    posterior for any retained partition. With a LogNormal retained state the
    conditioning on each draw is still exact under the reduced model, but the
    reduced model itself is the moment-matched approximation described under
    "Assumptions and limitations" in the coherent-reduction guide.

    Attributes:
        functional_mean: ``Q m`` on the functional axis, in the functional's
            units.
        retained_response: ``D = Q U_*`` on ``(functional_dim, state_dim)``.
            A state marginalised into ``A`` has a zero column.
        unresolved_observation_covariance: ``Q B_perp H.T`` on
            ``(functional_dim, observation_dim)``.
        unresolved_covariance: Positive-semidefinite ``Q B_perp Q.T`` on
            distinct functional row and column axes.
        retained_covariance: ``C_alpha`` of the full-restriction reduction the
            products were computed for. It identifies that reduction and is
            needed to marginalise states later.
        effective_observation_operator: ``H_alpha`` of that reduction, with
            the sensitivity units the residuals must use.
        marginalised: Boolean state mask of states already moved into the
            unresolved part by :func:`marginalise_functional_states`, or
            ``None`` for full-restriction products.
    """

    functional_mean: xr.DataArray
    retained_response: xr.DataArray
    unresolved_observation_covariance: xr.DataArray
    unresolved_covariance: xr.DataArray
    retained_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    marginalised: xr.DataArray | None = None

    @property
    def functional_dim(self) -> str:
        """Return the functional row dimension."""
        return str(self.functional_mean.dims[0])

    @property
    def state_dim(self) -> str:
        """Return the retained-state dimension of ``D``."""
        return str(self.retained_response.dims[1])

    @property
    def observation_dim(self) -> str:
        """Return the observation dimension of ``Q B_perp H.T``."""
        return str(self.unresolved_observation_covariance.dims[1])


@dataclass(frozen=True, slots=True, eq=False)
class CoherentGaussianReduction:
    """One exact, labelled native-to-retained Gaussian reduction.

    Attributes:
        retained_mean: Dimensionless ``Pi m`` on the retained-state axis.
        retained_covariance: Positive-definite ``C_alpha`` on distinct
            retained-state row and column axes.
        effective_observation_operator: ``H_alpha`` mapping retained-state
            perturbations to observations.
        native_observation_mean: ``H m`` on the observation axis.
        observation_intercept: Affine intercept ``H m - H_alpha Pi m``.
        unresolved_observation_covariance: ``A`` on distinct observation row
            and column axes. It is a positive-semidefinite covariance up to
            floating-point roundoff.
        projection_strategy: Scientific strategy that selected ``Pi``.
        functional_products: Products for the requested native functionals,
            or ``None`` when none were requested.
    """

    retained_mean: xr.DataArray
    retained_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    native_observation_mean: xr.DataArray
    observation_intercept: xr.DataArray
    unresolved_observation_covariance: xr.DataArray
    projection_strategy: str
    functional_products: NativeFunctionalProducts | None = None


def reduce_native_gaussian(
    *,
    covariance: InvertibleNativeCovarianceAction,
    strategy: RetainedProjectionStrategy,
    state_dim: str,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_batch_size: int = 64,
    functionals: xr.DataArray | None = None,
    functional_dim: str = "functional",
) -> CoherentGaussianReduction:
    """Prepare the exact centred Gaussian model for one ``B/H/Pi/m`` set.

    This is a named eager numerical boundary. ``native_mean`` and
    ``native_sensitivity`` may be Dask-backed; their payloads are materialized
    together so a shared graph is executed once. The strategy then selects and
    materializes ``Pi`` from its own inputs. Inputs are borrowed and are not
    mutated.

    The inputs must use the covariance action's native dimensions and carry
    exactly matching indexed coordinates on shared dimensions. Unit conversion
    belongs to the upstream OpenGHG/pint-xarray preparation boundary: ``m`` and
    ``Pi`` describe dimensionless scaling, while ``H`` is already expressed
    in the desired observation units.

    Args:
        covariance: Invertible labelled native covariance action ``B``.
        strategy: Authoritative retained restriction strategy, owning the
            inputs that define ``Pi``.
        state_dim: Retained-state dimension of ``Pi``.
        native_mean: Dimensionless native scaling mean ``m``.
        native_sensitivity: Native sensitivity ``H`` in canonical observation
            units.
        observation_dim: Observation dimension in ``native_sensitivity``.
        observation_batch_size: Covariance right-hand-side batch size.
        functionals: Optional native linear functionals ``Q`` on
            ``(functional_dim, *native_dims)``, in the reporting units per
            unit scaling. See :func:`project_native_functionals`.
        functional_dim: Functional row dimension of ``functionals``.

    Returns:
        The retained prior, centred effective forward model, and unresolved
        observation covariance, plus the functional products when
        ``functionals`` is given.

    Raises:
        ValueError: If xarray cannot transpose or exactly align the labelled
            inputs, or if covariance-product construction fails.
    """
    native_dims = tuple(covariance.native_dims)
    mean = native_mean.transpose(*native_dims)
    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)

    # The covariance action checks the grids of H and Pi when it is applied,
    # but m meets only xr.dot, whose default inner join could silently discard
    # native cells. This index-only check executes no lazy payload.
    mean, sensitivity = xr.align(mean, sensitivity, join="exact", copy=False)
    mean, sensitivity = materialize_together(mean, sensitivity, purpose="native mean m and sensitivity H")

    products = project_native_covariance(
        covariance=covariance,
        strategy=strategy,
        state_dim=state_dim,
        native_sensitivity=sensitivity,
        observation_dim=observation_dim,
        observation_covariance="dense",
        observation_batch_size=observation_batch_size,
    )
    reduction = _reduce_native_covariance_products(
        products=products,
        native_mean=mean,
        native_sensitivity=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
        observation_dim=observation_dim,
    )
    if functionals is None:
        return reduction
    functional_products = project_native_functionals(
        covariance=covariance,
        products=products,
        functionals=functionals,
        functional_dim=functional_dim,
        native_mean=mean,
        native_sensitivity=sensitivity,
        observation_dim=observation_dim,
        batch_size=observation_batch_size,
    )
    return replace(reduction, functional_products=functional_products)


def project_native_functionals(
    *,
    covariance: InvertibleNativeCovarianceAction,
    products: NativeCovarianceProducts,
    functionals: xr.DataArray,
    functional_dim: str,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    batch_size: int = 64,
) -> NativeFunctionalProducts:
    """Compute the coherent-reduction products of native linear functionals.

    This is a named eager numerical boundary. ``B`` is applied to bounded
    blocks of functional rows, ``B Q.T``, and each block is contracted with
    ``H`` and ``Q`` at once, so neither ``B`` nor the full ``B Q.T`` is held.
    A sparse or Dask-backed ``Q`` is computed once and stays sparse; only one
    row block at a time is densified. Inputs are borrowed.

    A functional that should be reported jointly, such as one country total
    per source, needs one row per term so that the joint covariance across
    rows is available; sums of rows are then reconstructed per draw.

    Args:
        covariance: The native covariance action ``B`` used for ``products``.
        products: Dense-view products from :func:`project_native_covariance`
            for the same ``B``, ``H`` and ``Pi``.
        functionals: Native functionals ``Q`` on
            ``(functional_dim, *native_dims)`` with a unique functional index.
            Optional ``units`` describe one functional per unit scaling.
        functional_dim: Functional row dimension.
        native_mean: Dimensionless native mean ``m``, exactly aligned with
            the covariance grid.
        native_sensitivity: The native sensitivity ``H`` used for
            ``products``.
        observation_dim: Observation dimension of ``H``.
        batch_size: Positive number of functional rows per covariance block.

    Returns:
        ``Q m``, ``D = Q U_*``, ``Q B_perp H.T`` and ``Q B_perp Q.T``.

    Raises:
        ValueError: If labels or dimensions do not match, the products are
            not dense-view, or ``Q`` holds non-finite values.
    """
    native_dims = tuple(covariance.native_dims)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if products.observation_covariance_view != "dense":
        raise ValueError("Functional products require dense-view native covariance products.")
    if set(functionals.dims) != {functional_dim, *native_dims}:
        raise ValueError(
            f"Functionals must have dimensions ({functional_dim!r}, *{native_dims!r}); "
            f"got {functionals.dims!r}."
        )
    require_unique_index(functionals, functional_dim, name="Functionals")
    state_dim = str(products.restriction.dims[0])
    functional = _materialize_functionals(functionals.transpose(functional_dim, *native_dims))
    mean = native_mean.transpose(*native_dims)
    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)
    functional, mean, sensitivity = xr.align(functional, mean, sensitivity, join="exact", copy=False)
    mean, sensitivity = materialize_together(mean, sensitivity, purpose="functional mean m and sensitivity H")

    functional_mean = to_dense(sparse_xr_dot(functional, mean, dim=list(native_dims)))
    functional_mean = functional_mean.transpose(functional_dim)
    response = to_dense(sparse_xr_dot(functional, products.prolongation, dim=list(native_dims)))
    response = response.transpose(functional_dim, state_dim)

    # Q B Q.T and H B Q.T, one block of B Q.T at a time.
    occupied = {str(name) for array in (functional, sensitivity) for name in (*array.dims, *array.coords)}
    column_dim = matrix_column_dim(functional_dim, occupied)
    count = functional.sizes[functional_dim]
    q_b_q = np.empty((count, count))
    h_b_q = np.empty((sensitivity.sizes[observation_dim], count))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        rhs = to_column_axis(
            to_dense(functional.isel({functional_dim: slice(start, stop)})),
            row_dim=functional_dim,
            column_dim=column_dim,
            leading_dims=native_dims,
        )
        b_q = covariance.apply(rhs)
        q_b_q[:, start:stop] = to_dense(sparse_xr_dot(functional, b_q, dim=list(native_dims))).transpose(
            functional_dim, column_dim
        ).values
        h_b_q[:, start:stop] = xr.dot(sensitivity, b_q, dim=list(native_dims)).transpose(
            observation_dim, column_dim
        ).values

    # Remove the retained part, B_perp = B - U_* C_alpha U_*.T:
    # Q B_perp H.T = Q B H.T - D C_alpha H_alpha.T and
    # Q B_perp Q.T = Q B Q.T - D C_alpha D.T. D, C_alpha and H_alpha all
    # follow the state order of Pi, so they are combined positionally.
    d_values = np.asarray(response.values, dtype=np.float64)
    d_c = d_values @ np.asarray(products.state_covariance.values, dtype=np.float64)
    effective = products.effective_observation_operator.transpose(observation_dim, state_dim)
    if not same_index(effective.indexes[state_dim], response.indexes[state_dim]):
        raise ValueError("Functional response and effective operator state labels differ.")
    cross_values = h_b_q.T - d_c @ np.asarray(effective.values, dtype=np.float64).T
    covariance_values = q_b_q - d_c @ d_values.T
    covariance_values = (covariance_values + covariance_values.T) * 0.5

    q_units = functional.attrs.get("units")
    h_units = sensitivity.attrs.get("units")
    q_attrs = {"units": q_units} if q_units is not None else {}
    cross_attrs = {"units": f"({q_units}) ({h_units})"} if q_units is not None and h_units is not None else {}
    square_attrs = {"units": f"({q_units})^2"} if q_units is not None else {}
    return NativeFunctionalProducts(
        functional_mean=functional_mean.rename("functional_mean").assign_attrs(
            mathematical_name="Q m", **q_attrs
        ),
        retained_response=response.rename("retained_response").assign_attrs(
            mathematical_name="D", definition="Q U_*", **q_attrs
        ),
        unresolved_observation_covariance=xr.DataArray(
            cross_values,
            dims=(functional_dim, observation_dim),
            coords={
                **coordinates_on_dims(functional, {functional_dim}),
                **coordinates_on_dims(sensitivity, {observation_dim}),
            },
            name="unresolved_observation_covariance",
            attrs={
                "mathematical_name": "Q B_perp H.T",
                "definition": "Q B H.T - D C_alpha H_alpha.T",
                **cross_attrs,
            },
        ),
        unresolved_covariance=square_matrix(
            covariance_values,
            functional,
            row_dim=functional_dim,
            column_dim=column_dim,
            name="unresolved_covariance",
            attrs={
                "mathematical_name": "Q B_perp Q.T",
                "definition": "Q B Q.T - D C_alpha D.T",
                **square_attrs,
            },
        ),
        retained_covariance=products.state_covariance,
        effective_observation_operator=effective,
    )


def marginalise_functional_states(
    products: NativeFunctionalProducts,
    marginalised: xr.DataArray,
) -> NativeFunctionalProducts:
    """Move marginalised retained states of functional products into ``x_perp``.

    Integrating states ``M`` out of a full-restriction reduction gives the
    coherent reduction with the kept rows ``K`` of ``Pi``. With
    ``G = C_MK C_KK^-1`` and ``S = C_MM - G C_KM``, the products become
    ``D'_K = D_K + D_M G``, ``D'_M = 0``,
    ``Q B'_perp H.T = Q B_perp H.T + D_M S H_M.T`` and
    ``Q B'_perp Q.T = Q B_perp Q.T + D_M S D_M.T``, where ``H_M`` are the
    marginalised columns of ``H_alpha``. Marginalised states' functional
    contribution then goes through the unresolved path, consistently with an
    ``A`` that absorbed ``H_M S H_M.T``. A reduction that marginalises states
    into ``A`` must apply this to its functional products as well. Inputs are
    borrowed.

    Args:
        products: Full-restriction functional products, carrying that
            reduction's ``C_alpha`` and ``H_alpha``.
        marginalised: Boolean mask on the retained-state axis; ``True``
            marks a state to integrate out.

    Returns:
        Products for the kept-state reduction on the full state axis, so a
        marginalised state held at its prior mean contributes nothing through
        ``D``, with ``marginalised`` recorded. Unchanged products when no
        state is marginalised.

    Raises:
        ValueError: If the products were already marginalised, state labels
            differ, or every state is marginalised.
    """
    if products.marginalised is not None:
        raise ValueError(
            "Functional products are already marginalised; start from full-restriction products."
        )
    state_dim = products.state_dim
    state_index = products.retained_response.indexes[state_dim]
    if marginalised.dims != (state_dim,) or not same_index(marginalised.indexes[state_dim], state_index):
        raise ValueError("The marginalised mask must use the functional products' state labels.")
    covariance = products.retained_covariance
    effective = products.effective_observation_operator
    mask = np.asarray(marginalised.values, dtype=bool)
    if not mask.any():
        return products
    if mask.all():
        raise ValueError("At least one retained state must be kept.")
    kept = ~mask

    c = np.asarray(covariance.values, dtype=np.float64)
    gain = np.linalg.solve(c[np.ix_(kept, kept)], c[np.ix_(kept, mask)]).T
    schur = c[np.ix_(mask, mask)] - gain @ c[np.ix_(kept, mask)]
    d = np.asarray(products.retained_response.values, dtype=np.float64)
    d_m = d[:, mask]
    h_m = np.asarray(effective.values, dtype=np.float64)[:, mask]
    response = np.zeros_like(d)
    response[:, kept] = d[:, kept] + d_m @ gain
    unresolved = products.unresolved_covariance.values + d_m @ schur @ d_m.T
    return replace(
        products,
        retained_response=products.retained_response.copy(data=response),
        unresolved_observation_covariance=products.unresolved_observation_covariance.copy(
            data=products.unresolved_observation_covariance.values + d_m @ schur @ h_m.T
        ),
        unresolved_covariance=products.unresolved_covariance.copy(data=(unresolved + unresolved.T) * 0.5),
        marginalised=marginalised.astype(bool),
    )


def _materialize_functionals(functionals: xr.DataArray) -> xr.DataArray:
    """Compute ``Q`` once, keeping sparse storage sparse, and require finite real values."""
    (data,) = compute(functionals.data)
    if isinstance(data, SparseArray):
        data = COO(data)
        if data.fill_value != 0:
            raise ValueError("Sparse functionals must use a zero fill value.")
        stored = data.data
    else:
        data = np.asarray(data)
        stored = data
    if np.iscomplexobj(stored) or not np.all(np.isfinite(stored)):
        raise ValueError("Functionals must contain finite real values.")
    return functionals.copy(data=data)


def _reduce_native_covariance_products(
    *,
    products: NativeCovarianceProducts,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
    observation_dim: str,
) -> CoherentGaussianReduction:
    """Apply the reduction equations to products constructed immediately above."""
    restriction = products.restriction
    state_covariance = products.state_covariance
    effective_operator = products.effective_observation_operator
    native_observation_covariance = products.native_observation_covariance

    retained_mean = xr.dot(restriction, native_mean, dim=list(native_dims)).transpose(state_dim)
    retained_mean = retained_mean.rename("retained_mean").assign_attrs(
        mathematical_name="Pi m",
        units="1",
    )
    native_observation_mean = xr.dot(
        native_sensitivity,
        native_mean,
        dim=list(native_dims),
    ).transpose(observation_dim)
    sensitivity_units = native_sensitivity.attrs.get("units")
    linear_units = {"units": sensitivity_units} if sensitivity_units is not None else {}
    native_observation_mean = native_observation_mean.rename("native_observation_mean").assign_attrs(
        mathematical_name="H m",
        **linear_units,
    )
    observation_intercept = (
        native_observation_mean
        - xr.dot(effective_operator, retained_mean, dim=state_dim)
    ).rename("observation_intercept")
    observation_intercept = observation_intercept.assign_attrs(
        mathematical_name="H m - H_alpha Pi m",
        **linear_units,
    )

    # A = H B H.T - H_alpha C_alpha H_alpha.T, formed positionally in NumPy.
    # H_alpha's state columns and both C_alpha axes follow Pi's state index,
    # so their orders already agree.
    unresolved_values = (
        np.asarray(native_observation_covariance.values, dtype=np.float64)
        - np.asarray(effective_operator.values, dtype=np.float64)
        @ np.asarray(state_covariance.values, dtype=np.float64)
        @ np.asarray(effective_operator.values, dtype=np.float64).T
    )
    # The two algebraically symmetric terms can differ at roundoff after the
    # retained solve. Symmetrization defines the numerical covariance returned
    # by this eager kernel; scientific PSD identities remain covered by tests.
    unresolved_values = (unresolved_values + unresolved_values.T) * 0.5
    unresolved_covariance = native_observation_covariance.copy(
        data=unresolved_values
    ).rename("unresolved_observation_covariance")
    unresolved_covariance = unresolved_covariance.assign_attrs(
        mathematical_name="A = H B_perp H.T",
        definition="H B H.T - H_alpha C_alpha H_alpha.T",
    )

    return CoherentGaussianReduction(
        retained_mean=retained_mean,
        retained_covariance=state_covariance.rename("retained_covariance"),
        effective_observation_operator=effective_operator,
        native_observation_mean=native_observation_mean,
        observation_intercept=observation_intercept,
        unresolved_observation_covariance=unresolved_covariance,
        projection_strategy=products.strategy,
    )
