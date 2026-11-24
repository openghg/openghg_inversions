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

:func:`marginalise_low_information_states` optionally applies a
``StateActivity.min_information`` threshold to one reduction. Retained states
whose prior-predicted information falls below the threshold are integrated out
exactly at fit time: their effective-operator columns become zero, so model
construction treats them as zero-sensitivity states, and their unresolved
variance is added to ``A``. :func:`marginalisation_terms` computes those
changes from ``C_alpha``, ``H_alpha`` and the mask alone, so a prepared
artifact can store the full-restriction reduction and derive the fitted
operator and ``A`` after loading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any

from dask.base import compute
import numpy as np
from scipy.linalg import cho_factor, cho_solve
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

if TYPE_CHECKING:
    from openghg_inversions.models.state_activity import StateActivity


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
        reduction_identity: :func:`reduction_identity` of that reduction. It
            depends on the native observation operator ``H``, the covariance
            ``B``, the restriction ``Pi`` and the native mean, which ``Q m``
            and ``D`` alone do not identify.
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
    reduction_identity: str
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
        native_identity: :func:`native_reduction_identity` of the native
            inputs ``H``, ``B``, ``Pi`` and ``m`` this reduction was computed
            from. The reduced arrays alone cannot identify them.
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
    native_identity: str
    functional_products: NativeFunctionalProducts | None = None

    @property
    def identity(self) -> str:
        """Return the :func:`reduction_identity` of this reduction."""
        return reduction_identity(
            native_identity=self.native_identity,
            retained_covariance=self.retained_covariance,
            effective_observation_operator=self.effective_observation_operator,
            unresolved_observation_covariance=self.unresolved_observation_covariance,
        )


def reduction_identity(
    *,
    native_identity: str,
    retained_covariance: xr.DataArray,
    effective_observation_operator: xr.DataArray,
    unresolved_observation_covariance: xr.DataArray,
) -> str:
    """Return the identity of one coherent reduction.

    Different native problems can share ``C_alpha``, ``H_alpha`` and ``A``:
    if one observation sees either of two exchangeable cells, swapping the
    cells leaves all three unchanged but reverses the covariance between a
    cell's functional and the observation. The identity therefore combines
    the :func:`native_reduction_identity` of the native inputs with a hash
    of the three reduced arrays. Anyone holding the native identity and the
    stored arrays, such as a prepared artifact, can recompute it, so a
    stored ``A`` or ``H_alpha`` that was changed after reduction no longer
    matches.

    Arrays are hashed as float64 values in their stored axis order together
    with their index labels (not dimension names), so the identity survives
    dimension renaming. It is an exact digest, not a tolerance-based
    comparison.

    Args:
        native_identity: :func:`native_reduction_identity` of the native
            inputs, as recorded by the reduction.
        retained_covariance: ``C_alpha``.
        effective_observation_operator: ``H_alpha``.
        unresolved_observation_covariance: The exact dense ``A``.

    Returns:
        ``"sha256:<hex digest>"``.

    Raises:
        ValueError: If ``native_identity`` is not a native identity digest.
    """
    if not native_identity.startswith("sha256:"):
        raise ValueError("native_identity must be a native_reduction_identity digest.")
    digest = sha256()
    digest.update(b"native\x1e" + native_identity.encode())
    for name, array in (
        ("C_alpha", retained_covariance),
        ("H_alpha", effective_observation_operator),
        ("A", unresolved_observation_covariance),
    ):
        _update_digest(digest, name, array)
    return f"sha256:{digest.hexdigest()}"


def native_reduction_identity(
    *,
    covariance: InvertibleNativeCovarianceAction,
    products: NativeCovarianceProducts,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
) -> str:
    """Return the identity of the native inputs of one coherent reduction.

    It hashes the native sensitivity ``H``, the restriction ``Pi``, the
    native mean ``m`` and the prolongation ``U_* = B Pi.T C_alpha^-1``, each
    with its index labels. ``B`` enters through ``U_*`` and, when the
    covariance action can describe itself with ``to_dataset()`` (as
    :class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
    does), through that description. Sparse payloads are hashed by their
    coordinates and values without densifying.

    Args:
        covariance: The native covariance action ``B``.
        products: Native covariance products for ``B``, ``H`` and ``Pi``.
        native_mean: Native mean ``m`` on the covariance's native dims.
        native_sensitivity: Native ``H``.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    native_dims = tuple(covariance.native_dims)
    digest = sha256()
    digest.update(f"{type(covariance).__module__}.{type(covariance).__qualname__}".encode())
    describe = getattr(covariance, "to_dataset", None)
    if callable(describe):
        description = describe()
        digest.update(json.dumps(description.attrs, sort_keys=True, default=str).encode())
        for name in sorted(map(str, description.variables)):
            _update_digest(digest, f"B.{name}", description[name])
    observation_dim = next(str(dim) for dim in native_sensitivity.dims if dim not in native_dims)
    for name, array in (
        ("H", native_sensitivity.transpose(observation_dim, *native_dims)),
        ("Pi", products.restriction),
        ("m", native_mean.transpose(*native_dims)),
        ("U_star", products.prolongation),
    ):
        _update_digest(digest, name, array)
    return f"sha256:{digest.hexdigest()}"


def _update_digest(digest, name: str, array: xr.DataArray) -> None:
    """Add one labelled array to ``digest``: its name, index labels, shape and values."""
    digest.update(name.encode() + b"\x1e")
    for dim in array.dims:
        labels = array.indexes[dim].tolist() if dim in array.indexes else list(range(array.sizes[dim]))
        digest.update("\x1f".join(map(str, labels)).encode())
        digest.update(b"\x1e")
    data = array.data
    if isinstance(data, SparseArray):
        sparse = data if isinstance(data, COO) else data.tocoo()
        digest.update(np.asarray(sparse.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(sparse.coords, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(sparse.data, dtype=np.float64).tobytes())
        digest.update(np.asarray(float(sparse.fill_value), dtype=np.float64).tobytes())
        return
    values = np.asarray(array.values)
    if values.dtype.kind in "biuf":
        values = values.astype(np.float64)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update("\x1f".join(map(str, values.ravel().tolist())).encode())


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
        native_identity=native_reduction_identity(
            covariance=covariance, products=products, native_mean=mean, native_sensitivity=sensitivity
        ),
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
        native_identity=reduction.native_identity,
    )
    return replace(reduction, functional_products=functional_products)


def _unresolved_observation_values(products: NativeCovarianceProducts) -> np.ndarray:
    """Return the exact symmetrized ``A = H B H.T - H_alpha C_alpha H_alpha.T`` values.

    The reduction kernel and the functional products both use this, so their
    identities are computed from bit-identical ``A``.
    """
    # Formed positionally in NumPy: H_alpha's state columns and both C_alpha
    # axes follow Pi's state index, so their orders already agree.
    effective = np.asarray(products.effective_observation_operator.values, dtype=np.float64)
    values = (
        np.asarray(products.native_observation_covariance.values, dtype=np.float64)
        - effective @ np.asarray(products.state_covariance.values, dtype=np.float64) @ effective.T
    )
    # The two algebraically symmetric terms can differ at roundoff after the
    # retained solve. Symmetrization defines the numerical covariance returned
    # by this eager kernel; scientific PSD identities remain covered by tests.
    return (values + values.T) * 0.5


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
    native_identity: str | None = None,
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
        native_identity: :func:`native_reduction_identity` of these inputs
            when the caller has already computed it, as
            :func:`reduce_native_gaussian` does; otherwise it is computed
            here.

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
        reduction_identity=reduction_identity(
            native_identity=native_identity
            or native_reduction_identity(
                covariance=covariance, products=products, native_mean=mean, native_sensitivity=sensitivity
            ),
            retained_covariance=products.state_covariance.rename("retained_covariance"),
            effective_observation_operator=products.effective_observation_operator,
            unresolved_observation_covariance=products.native_observation_covariance.copy(
                data=_unresolved_observation_values(products)
            ),
        ),
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
    native_identity: str,
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

    unresolved_values = _unresolved_observation_values(products)
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
        native_identity=native_identity,
    )


@dataclass(frozen=True, slots=True, eq=False)
class InformationSelection:
    """A coherent reduction after applying a prior-information threshold.

    Attributes:
        reduction: The full-restriction reduction the threshold was applied
            to, unchanged. Store this one; derive the fitted view with
            :attr:`marginalised_reduction` or :func:`marginalisation_terms`.
        information: Prior-predicted information
            ``I_j = sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2`` on the retained
            state axis, where ``s_j`` is the retained prior standard deviation.
        threshold: Threshold applied to each state; NaN where none applies.
        marginalised: Boolean mask of states moved into ``A``.
        unresolved_variance_increase: Change in ``diag A`` on the observation
            axis, in squared observation units.
    """

    reduction: CoherentGaussianReduction
    information: xr.DataArray
    threshold: xr.DataArray
    marginalised: xr.DataArray
    unresolved_variance_increase: xr.DataArray

    @property
    def marginalised_reduction(self) -> CoherentGaussianReduction:
        """Return the fitted view of :attr:`reduction` with the marginalised states integrated out."""
        return marginalise_retained_states(self.reduction, self.marginalised)

    def to_dataset(self) -> xr.Dataset:
        """Return the per-state and per-observation diagnostics for saving."""
        return xr.Dataset(
            {
                "state_information": self.information,
                "state_information_threshold": self.threshold,
                "state_information_marginalised": self.marginalised,
                "unresolved_variance_increase": self.unresolved_variance_increase,
            }
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the thresholds and marginalised states."""
        state_dim = str(self.information.dims[0])
        labels = self.information.indexes[state_dim]
        mask = np.asarray(self.marginalised.values, dtype=bool)
        information = np.asarray(self.information.values, dtype=float)
        threshold = np.asarray(self.threshold.values, dtype=float)
        increase = np.asarray(self.unresolved_variance_increase.values, dtype=float)
        return {
            "criterion": "I_j = sum_o (H_alpha[o, j] * s_j)**2 / sigma_o**2 < min_information_j",
            "n_states": int(mask.size),
            "n_marginalised": int(mask.sum()),
            "marginalised_states": [
                {
                    "state": _json_label(labels[i]),
                    "information": float(information[i]),
                    "threshold": float(threshold[i]),
                }
                for i in np.flatnonzero(mask)
            ],
            "min_retained_information": (
                float(information[~mask].min()) if (~mask).any() else None
            ),
            "max_unresolved_variance_increase": float(increase.max()) if increase.size else 0.0,
            "total_unresolved_variance_increase": float(increase.sum()),
        }


def _json_label(label: Any) -> Any:
    """Return a JSON-safe state label, keeping numbers numeric."""
    if isinstance(label, tuple):
        return [_json_label(item) for item in label]
    if isinstance(label, np.generic):
        label = label.item()
    return json.loads(json.dumps(label, default=str))


def retained_state_information(
    reduction: CoherentGaussianReduction,
    observation_variance: xr.DataArray,
) -> xr.DataArray:
    """Return each retained state's prior-predicted information.

    For retained prior standard deviation ``s_j = sqrt(C_alpha[j, j])``,
    effective operator ``H_alpha`` and prior error variance ``sigma_o**2``,
    ``I_j = sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2``. It is the expected
    squared signal-to-noise ratio of one prior standard deviation of state
    ``j`` and uses only the prior and the operator, never observed values.

    Args:
        reduction: Coherent reduction supplying ``H_alpha`` and ``C_alpha``.
        observation_variance: Positive finite prior observation-error or
            model-mismatch variance on the reduction's observation dimension
            and exact labels, in squared observation units. Do not include
            ``A`` itself.

    Returns:
        Dimensionless information on the retained-state axis, carrying the
        reduction's state coordinates.

    Raises:
        ValueError: If the variance is not one-dimensional on the observation
            axis with matching labels, or is non-finite or non-positive.
    """
    operator = reduction.effective_observation_operator
    observation_dim = str(operator.dims[0])
    if observation_variance.dims != (observation_dim,):
        raise ValueError(
            f"observation_variance must have dimensions ({observation_dim!r},); "
            f"found {observation_variance.dims!r}."
        )
    variance, _ = xr.align(observation_variance, operator, join="exact", copy=False)
    variance_values = np.asarray(to_dense(variance).values, dtype=np.float64)
    if not np.isfinite(variance_values).all() or (variance_values <= 0).any():
        raise ValueError("observation_variance must be finite and positive.")
    operator_values = np.asarray(operator.values, dtype=np.float64)
    prior_variance = np.diag(np.asarray(reduction.retained_covariance.values, dtype=np.float64))
    information = prior_variance * np.sum(operator_values**2 / variance_values[:, None], axis=0)
    return reduction.retained_mean.copy(data=information).rename("state_information").assign_attrs(
        units="1",
        long_name="prior-predicted state information",
        mathematical_name="sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2",
    )


@dataclass(frozen=True, slots=True, eq=False)
class MarginalisationTerms:
    """Fit-time changes from integrating marginalised retained states into ``A``.

    Attributes:
        effective_observation_operator: ``H'`` with ``H'_K = H_K + H_D G`` and
            zero marginalised columns, labelled like ``H_alpha``.
        unresolved_factor: ``F = H_D L`` on ``(observation_dim,
            marginalised_dim)``, where ``S = L L.T`` is the Schur complement
            ``C_DD - G C_KD``. The fitted unresolved covariance is
            ``A + F F.T``, so a low-rank ``A`` gains ``F`` as extra factor
            columns exactly.
    """

    effective_observation_operator: xr.DataArray
    unresolved_factor: xr.DataArray


def marginalisation_terms(
    *,
    retained_covariance: xr.DataArray,
    effective_observation_operator: xr.DataArray,
    marginalised: xr.DataArray,
    marginalised_dim: str = "marginalised_state",
) -> MarginalisationTerms:
    """Return the exact fit-time operator and ``A`` increment of marginalising retained states.

    Split the retained states into kept ``K`` and marginalised ``D``. With
    ``G = C_DK C_KK^-1`` and ``S = C_DD - G C_KD``, the exact conditional
    model given ``alpha_K`` is

    ``y | alpha_K ~ N(H m + H'_K (alpha_K - Pi_K m), R + A + H_D S H_D.T)``,

    where ``H'_K = H_K + H_D G``. This is the coherent reduction with
    restriction ``Pi_K``. Only ``C_alpha``, ``H_alpha`` and the mask are
    needed, so the terms can be derived from a stored full-restriction
    reduction.

    Args:
        retained_covariance: Full-restriction ``C_alpha`` on state row and
            column axes in the state order of ``effective_observation_operator``.
        effective_observation_operator: Full-restriction ``H_alpha`` on
            ``(observation_dim, state_dim)``.
        marginalised: Boolean mask on ``state_dim`` with exactly the
            operator's state labels.
        marginalised_dim: Column dimension of ``unresolved_factor``, labelled
            by the marginalised states.

    Returns:
        The fitted operator and the factor ``F`` of ``A' - A``. With no state
        marginalised the operator is returned unchanged and ``F`` has no
        columns.

    Raises:
        ValueError: If the mask is not a boolean state-aligned array, or the
            arrays have the wrong dimensions.
    """
    if effective_observation_operator.ndim != 2:
        raise ValueError("effective_observation_operator must be observation-by-state.")
    observation_dim, state_dim = (str(dim) for dim in effective_observation_operator.dims)
    if marginalised.dims != (state_dim,):
        raise ValueError(f"marginalised must have dimensions ({state_dim!r},).")
    mask, _ = xr.align(marginalised, effective_observation_operator[state_dim], join="exact", copy=False)
    drop = np.asarray(mask.values)
    if drop.dtype != np.dtype(bool):
        raise ValueError("marginalised must contain only boolean values.")
    count = effective_observation_operator.sizes[state_dim]
    if retained_covariance.shape != (count, count):
        raise ValueError("retained_covariance must be square on the operator's states.")
    keep = ~drop
    operator = np.asarray(effective_observation_operator.values, dtype=np.float64)
    covariance = np.asarray(retained_covariance.values, dtype=np.float64)
    operator_d = operator[:, drop]
    c_dd = covariance[np.ix_(drop, drop)]
    if keep.any() and drop.any():
        c_kd = covariance[np.ix_(keep, drop)]
        gain = cho_solve(cho_factor(covariance[np.ix_(keep, keep)]), c_kd).T
        schur = c_dd - gain @ c_kd
    else:
        gain = np.zeros((int(drop.sum()), int(keep.sum())))
        schur = c_dd
    schur = (schur + schur.T) * 0.5
    # S is a Schur complement of a positive-definite C_alpha, so it is PSD
    # up to roundoff; an eigen square root tolerates rank deficiency.
    eigenvalues, vectors = np.linalg.eigh(schur) if drop.any() else (np.zeros(0), np.zeros((0, 0)))
    root = vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

    updated = np.zeros_like(operator)
    updated[:, keep] = operator[:, keep] + operator_d @ gain
    labels = effective_observation_operator[state_dim].values[drop]
    units = effective_observation_operator.attrs.get("units")
    factor = xr.DataArray(
        operator_d @ root,
        dims=(observation_dim, marginalised_dim),
        coords={
            **coordinates_on_dims(effective_observation_operator, {observation_dim}),
            marginalised_dim: np.arange(int(drop.sum())),
            f"{marginalised_dim}_label": (marginalised_dim, [str(label) for label in labels]),
        },
        name="marginalisation_unresolved_factor",
        attrs={"units": units, "mathematical_name": "H_D S^(1/2)"} if units is not None else {
            "mathematical_name": "H_D S^(1/2)"
        },
    )
    return MarginalisationTerms(
        effective_observation_operator=effective_observation_operator.copy(data=updated),
        unresolved_factor=factor,
    )


def marginalise_retained_states(
    reduction: CoherentGaussianReduction,
    marginalised: xr.DataArray,
) -> CoherentGaussianReduction:
    """Return the fitted view of a reduction with selected retained states integrated out.

    Applies :func:`marginalisation_terms`: ``H'_K = H_K + H_D G`` with zero
    marginalised columns, ``A' = A + H_D S H_D.T`` and the recomputed
    intercept. The full retained-state axis, prior moments, native identity
    and any full-restriction functional products are kept. The result is a
    fit-time view: store the full-restriction reduction and the mask, not
    this, because its ``A`` and ``H_alpha`` no longer reproduce the products'
    :func:`reduction_identity`.

    Args:
        reduction: Full-restriction coherent reduction.
        marginalised: Boolean mask on the retained-state axis with exactly the
            reduction's state labels.

    Returns:
        ``reduction`` itself when no state is selected, otherwise its fitted
        view.

    Raises:
        ValueError: If the mask is not a boolean state-aligned array.
    """
    terms = marginalisation_terms(
        retained_covariance=reduction.retained_covariance,
        effective_observation_operator=reduction.effective_observation_operator,
        marginalised=marginalised,
    )
    if not terms.unresolved_factor.sizes[str(terms.unresolved_factor.dims[1])]:
        return reduction
    state_dim = str(reduction.retained_mean.dims[0])
    factor = np.asarray(terms.unresolved_factor.values, dtype=np.float64)
    unresolved = np.asarray(reduction.unresolved_observation_covariance.values, dtype=np.float64)
    unresolved = unresolved + factor @ factor.T
    unresolved = (unresolved + unresolved.T) * 0.5
    effective = terms.effective_observation_operator
    # H m - H' Pi m, written as a change of the stored intercept so that a
    # prepared artifact, which keeps the intercept but not H m, derives the
    # same value.
    intercept = (
        reduction.observation_intercept
        + xr.dot(reduction.effective_observation_operator - effective, reduction.retained_mean, dim=state_dim)
    ).rename("observation_intercept")
    intercept = intercept.assign_attrs(reduction.observation_intercept.attrs)
    return replace(
        reduction,
        effective_observation_operator=effective,
        observation_intercept=intercept,
        unresolved_observation_covariance=reduction.unresolved_observation_covariance.copy(data=unresolved),
    )


def marginalise_low_information_states(
    reduction: CoherentGaussianReduction,
    *,
    observation_variance: xr.DataArray,
    policy: StateActivity | None,
) -> InformationSelection:
    """Apply ``policy.min_information`` to one coherent reduction.

    States with :func:`retained_state_information` below their threshold are
    selected for marginalisation. The selection keeps the full-restriction
    reduction unchanged; :attr:`InformationSelection.marginalised_reduction`
    or :func:`marginalisation_terms` give the fitted view. Thresholds may
    apply to every state or only to labelled groups, see
    :class:`~openghg_inversions.models.StateActivity`. Only
    ``min_information`` and ``group_coord`` are read from the policy; its
    explicit activity mask, fixed values and fixed groups remain model-time
    choices.

    Marginalised states receive no posterior update. Retained-state
    reconstructions hold them at their fixed values, which should equal the
    retained prior mean, and uncertainty computed only from retained states
    omits their contribution.

    Args:
        reduction: Coherent reduction whose states are screened.
        observation_variance: Prior observation-error or model-mismatch
            variance per observation; see :func:`retained_state_information`.
        policy: Activity policy. ``None`` or ``min_information=None`` keeps
            every state.

    Returns:
        The full-restriction reduction, the marginalised-state mask and the
        diagnostics.

    Raises:
        ValueError: If the variance or grouped thresholds are invalid.
    """
    from openghg_inversions.models.state_activity import information_thresholds

    information = retained_state_information(reduction, observation_variance)
    if policy is None or policy.min_information is None:
        threshold = xr.full_like(information, np.nan).rename("information_threshold")
    else:
        threshold = information_thresholds(reduction.retained_mean, policy)
    threshold = threshold.assign_attrs(units="1")
    marginalised = (information < threshold).rename("information_marginalised")
    terms = marginalisation_terms(
        retained_covariance=reduction.retained_covariance,
        effective_observation_operator=reduction.effective_observation_operator,
        marginalised=marginalised,
    )
    before = reduction.unresolved_observation_covariance
    increase = np.sum(np.square(np.asarray(terms.unresolved_factor.values, dtype=np.float64)), axis=1)
    observation_dim = str(before.dims[0])
    increase_array = xr.DataArray(
        increase,
        dims=(observation_dim,),
        coords={
            name: coord
            for name, coord in reduction.observation_intercept.coords.items()
            if set(coord.dims) <= {observation_dim}
        },
        name="unresolved_variance_increase",
        attrs={
            key: value for key, value in (("units", before.attrs.get("units")),) if value is not None
        },
    )
    return InformationSelection(
        reduction=reduction,
        information=information,
        threshold=threshold,
        marginalised=marginalised,
        unresolved_variance_increase=increase_array,
    )
