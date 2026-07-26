"""Aggregation-aware PyMC/NumPyro NUTS for one fixed full-tiling basis.

This module extends the fixed-basis reference target with the normalized
diagonal-plus-low-rank Gaussian closure from :mod:`aggregation_error_low_rank`.
The active topology remains fixed.  A Gamma root total and Dirichlet leaf
shares define positive region masses; cached fixed-partition factors provide
their conditional native-cell mean and projected aggregation covariance.

The complete correlated likelihood is one scalar PyMC ``Potential``.  Its
normalizer is retained and its scalar value is persisted with the posterior
as ``aggregation_joint_log_likelihood``.  There is deliberately no observed
diagonal ``Normal`` variable and sampling explicitly disables ArviZ
pointwise-log-likelihood extraction: the correlated joint density has no
ordinary pointwise observation decomposition.

Callers must supply one explicit native-cell alpha field and stable identity.
The bridge never derives concentration from ``K``.  Reusing the same field
and content hash across partitions is therefore a visible, auditable
cross-``K`` model choice rather than an implicit ``kappa = 2K`` convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib import import_module
import json
import math
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    PartitionSummaryFactors,
)
from .fixed_basis_nuts import (
    FixedBasisNUTSData,
    fixed_basis_nuts_initvals,
    prepare_fixed_basis_nuts,
    require_fixed_basis_nuts_float64,
)
from .full_tiling_posterior import FullTilingPosteriorState, FullTilingProblem

if TYPE_CHECKING:
    from arviz import InferenceData
    from pymc import Model

FloatArray: TypeAlias = NDArray[np.float64]
ConstrainedInitvals: TypeAlias = Mapping[str, Any] | Sequence[Mapping[str, Any] | None]
ChainMethod: TypeAlias = Literal["parallel", "vectorized"]
ScientificEvaluation: TypeAlias = tuple[float, float, FloatArray, FloatArray]
ScientificEvaluator: TypeAlias = Callable[
    [float, ArrayLike, ArrayLike],
    ScientificEvaluation,
]

__all__ = [
    "FixedBasisAggregationNUTSData",
    "build_fixed_basis_aggregation_pymc_model",
    "compile_fixed_basis_aggregation_pytensor_evaluator",
    "fixed_basis_aggregation_numpy_logp_and_gradient",
    "prepare_fixed_basis_aggregation_nuts",
    "sample_fixed_basis_aggregation_nuts",
    "validate_fixed_basis_aggregation_inference_data",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)
_SOURCE_MODEL_ID = "openghg-inversions-experimental-fixed-basis-aggregation-nuts-v1"
_MANIFEST_SCHEMA_ID = "fixed-basis-aggregation-nuts-manifest-v1"
_MANIFEST_ATTR = "fixed_basis_aggregation_manifest_json"
_MANIFEST_SHA_ATTR = "fixed_basis_aggregation_manifest_sha256"
_MODEL_DATA_OBJECT_ATTR = "_openghg_fixed_basis_aggregation_data_object"
_MODEL_TARGET_ID_ATTR = "_openghg_fixed_basis_aggregation_target_identity"
_MODEL_MANIFEST_ATTR = "_openghg_fixed_basis_aggregation_manifest_json"
_MODEL_POTENTIAL_ATTR = "_openghg_fixed_basis_aggregation_potential"
_MODEL_OUTPUT_ATTR = "_openghg_fixed_basis_aggregation_output"
_POSTERIOR_IDENTITY_RTOL = 5.0e-12
_POSTERIOR_IDENTITY_ATOL = 5.0e-12
_POSTERIOR_LIKELIHOOD_RTOL = 5.0e-10
_POSTERIOR_LIKELIHOOD_ATOL = 5.0e-10


def _readonly_float(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Return one owned finite read-only ``float64`` array."""
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _content_sha256(
    identity: str,
    arrays: Sequence[tuple[str, NDArray[Any]]],
    *,
    scalars: Sequence[tuple[str, str]] = (),
) -> str:
    """Hash named array content, shapes, dtypes, and canonical scalar text."""
    digest = hashlib.sha256()
    digest.update(identity.encode("utf-8") + b"\0")
    for name, values in arrays:
        array = np.asarray(values)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        if np.issubdtype(array.dtype, np.floating):
            content = np.ascontiguousarray(array, dtype="<f8")
        elif np.issubdtype(array.dtype, np.integer):
            content = np.ascontiguousarray(array, dtype="<i8")
        else:
            raise TypeError(f"{name} has an unsupported fingerprint dtype.")
        digest.update(content.tobytes())
    for name, value in scalars:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _array_sha256(values: FloatArray) -> str:
    """Hash one native alpha field using its public stable schema."""
    return _content_sha256(
        "fixed-native-alpha-field-v1",
        (("native_cell_alphas", values),),
    )


def _partition_factors_sha256(factors: PartitionSummaryFactors) -> str:
    """Fingerprint the complete authenticated fixed-partition cache."""
    return _content_sha256(
        "fixed-partition-summary-factors-v1",
        (
            ("labels", factors.labels),
            ("alpha_totals", factors.alpha_totals),
            ("observation_mean_design", factors.observation_mean_design),
            ("summary_mean_design", factors.summary_mean_design),
            (
                "summary_covariance_factors",
                factors.summary_covariance_factors,
            ),
        ),
    )


def _source_implementation_sha256() -> str:
    """Return the exact source-module content identity."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> str:
    """Return strict deterministic JSON for persisted target manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_partition_labels(data: FixedBasisNUTSData) -> NDArray[np.int64]:
    """Render canonical zero-based labels from validated rectangle bounds."""
    labels = np.full(data.tiling.shape, -1, dtype=np.int64)
    for label, (row_start, row_stop, col_start, col_stop) in enumerate(data.rectangle_bounds):
        block = labels[row_start:row_stop, col_start:col_stop]
        if block.size == 0 or np.any(block != -1):
            raise ValueError("rectangle_bounds must define disjoint non-empty regions.")
        block[...] = label
    if np.any(labels < 0):
        raise ValueError("rectangle_bounds must cover every native cell exactly once.")
    return labels


def _scaled_tolerance(*arrays: FloatArray) -> float:
    """Return a strict condition-scaled bridge tolerance."""
    scale = max(
        1.0,
        *(float(np.max(np.abs(array))) if array.size else 0.0 for array in arrays),
    )
    size = max(1, *(array.size for array in arrays))
    return float(512.0 * np.finfo(np.float64).eps * size * scale)


def _immutable_native_sensitivity(
    values: ArrayLike,
    *,
    n_observations: int,
    n_native_cells: int,
) -> FloatArray:
    """Validate and safely retain one immutable native sensitivity matrix."""
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("native_sensitivity must contain numeric values.") from error
    if matrix.shape != (n_observations, n_native_cells):
        raise ValueError(
            "native_sensitivity must have shape (number_of_observations, number_of_native_cells)."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("native_sensitivity must contain only finite values.")
    if matrix.flags.writeable or not matrix.flags.owndata:
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        matrix.setflags(write=False)
    return cast(FloatArray, matrix)


@dataclass(frozen=True, slots=True, eq=False)
class FixedBasisAggregationNUTSData:
    """Immutable target identity and cached factors for aggregation-aware NUTS.

    ``native_cell_alphas`` is explicit and independent of the active region
    count.  Its stable caller label and computed content hash are both
    persisted.  The cached region alpha totals must exactly match the
    Dirichlet shapes in ``fixed_basis``; this makes the allocation prior and
    conditional hidden-allocation law one coherent native model.
    """

    fixed_basis: FixedBasisNUTSData
    factors: PartitionSummaryFactors
    summary_basis: FloatArray
    native_cell_alphas: FloatArray
    native_sensitivity: FloatArray
    native_alpha_id: str
    native_alpha_sha256: str = field(init=False)
    native_alpha_total: float = field(init=False)
    summary_basis_sha256: str = field(init=False)
    partition_factors_sha256: str = field(init=False)
    reconstructed_factors_sha256: str = field(init=False)
    topology_sha256: str = field(init=False)
    native_sensitivity_sha256: str = field(init=False)
    fixed_target_sha256: str = field(init=False)
    source_implementation_sha256: str = field(init=False)
    aggregation_bridge_sha256: str = field(init=False)
    model_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate topology, coordinate order, factors, and native identity."""
        if not isinstance(self.fixed_basis, FixedBasisNUTSData):
            raise TypeError("fixed_basis must be FixedBasisNUTSData.")
        if not isinstance(self.factors, PartitionSummaryFactors):
            raise TypeError("factors must be PartitionSummaryFactors.")
        if not isinstance(self.native_alpha_id, str):
            raise TypeError("native_alpha_id must be a string.")
        identity = self.native_alpha_id.strip()
        if not identity or identity != self.native_alpha_id:
            raise ValueError("native_alpha_id must be a non-empty stripped string.")

        n_observations = self.fixed_basis.observations.size
        basis = _readonly_float(self.summary_basis, name="summary_basis", ndim=2)
        if basis.shape[0] != n_observations or basis.shape[1] > n_observations:
            raise ValueError(
                "summary_basis must have shape (number_of_observations, q) with q no larger than observations."
            )
        gram = basis.T @ basis
        basis_tolerance = float(256.0 * np.finfo(np.float64).eps * max(1, *basis.shape))
        if not np.allclose(
            gram,
            np.eye(basis.shape[1], dtype=np.float64),
            rtol=0.0,
            atol=basis_tolerance,
        ):
            raise ValueError("summary_basis columns must be orthonormal in whitened observation space.")

        alphas = _readonly_float(
            self.native_cell_alphas,
            name="native_cell_alphas",
            ndim=2,
        )
        if alphas.shape != self.fixed_basis.tiling.shape:
            raise ValueError("native_cell_alphas must have the fixed tiling's native shape.")
        if np.any(alphas <= 0.0):
            raise ValueError("native_cell_alphas must be strictly positive.")
        alpha_total = float(np.sum(alphas))
        if not math.isfinite(alpha_total) or alpha_total <= 0.0:
            raise ValueError("native_cell_alphas must have a finite positive total.")
        sensitivity = _immutable_native_sensitivity(
            self.native_sensitivity,
            n_observations=n_observations,
            n_native_cells=alphas.size,
        )

        factors = self.factors
        if factors.region_count != self.fixed_basis.k:
            raise ValueError("cached factors must have exactly one region per fixed leaf.")
        if factors.summary_dimension != basis.shape[1]:
            raise ValueError("cached factor rank must match summary_basis.")
        expected_labels = _canonical_partition_labels(self.fixed_basis)
        if not np.array_equal(factors.labels, expected_labels):
            raise ValueError("cached factor labels must match canonical fixed leaf order exactly.")
        if factors.observation_mean_design.shape != self.fixed_basis.dynamic_design.shape:
            raise ValueError("cached observation mean design shape must match fixed_basis.")
        if not np.array_equal(
            factors.observation_mean_design,
            self.fixed_basis.dynamic_design,
        ):
            raise ValueError(
                "cached observation mean design must equal the authoritative fixed-basis design exactly."
            )
        if not np.array_equal(
            factors.alpha_totals,
            self.fixed_basis.dirichlet_alpha,
        ):
            raise ValueError("cached alpha totals must equal the fixed-basis Dirichlet shapes exactly.")

        expected_summary_mean = (
            basis / self.fixed_basis.observation_sd[:, np.newaxis]
        ).T @ self.fixed_basis.dynamic_design
        if not np.array_equal(factors.summary_mean_design, expected_summary_mean):
            raise ValueError("cached summary mean design must match the whitened fixed-basis design exactly.")

        summed_alphas = np.asarray(
            [float(np.sum(alphas[expected_labels == label])) for label in range(self.fixed_basis.k)],
            dtype=np.float64,
        )
        alpha_tolerance = _scaled_tolerance(
            summed_alphas,
            factors.alpha_totals,
            alphas,
        )
        if not np.allclose(
            summed_alphas,
            factors.alpha_totals,
            rtol=0.0,
            atol=alpha_tolerance,
        ):
            raise ValueError("native_cell_alphas region sums must match cached alpha totals.")
        if not math.isclose(
            alpha_total,
            float(np.sum(factors.alpha_totals)),
            rel_tol=0.0,
            abs_tol=alpha_tolerance,
        ):
            raise ValueError("native-cell and cached regional alpha totals must identify one common prior.")

        independently_rebuilt = AdditiveDirichletAggregation(
            alphas,
            sensitivity,
            self.fixed_basis.observation_sd,
            basis,
        ).partition_factors(expected_labels)
        rebuilt_mean_tolerance = _scaled_tolerance(
            independently_rebuilt.observation_mean_design,
            self.fixed_basis.dynamic_design,
        )
        if not np.allclose(
            independently_rebuilt.observation_mean_design,
            self.fixed_basis.dynamic_design,
            rtol=0.0,
            atol=rebuilt_mean_tolerance,
        ):
            raise ValueError(
                "native alpha/sensitivity conditional means do not reconstruct the fixed target."
            )
        rebuilt_alpha_tolerance = _scaled_tolerance(
            independently_rebuilt.alpha_totals,
            self.fixed_basis.dirichlet_alpha,
        )
        if not np.allclose(
            independently_rebuilt.alpha_totals,
            self.fixed_basis.dirichlet_alpha,
            rtol=0.0,
            atol=rebuilt_alpha_tolerance,
        ):
            raise ValueError("native alpha field does not reconstruct the fixed allocation prior.")
        rebuilt_summary_mean = (
            basis / self.fixed_basis.observation_sd[:, np.newaxis]
        ).T @ self.fixed_basis.dynamic_design
        authenticated_factors = PartitionSummaryFactors(
            expected_labels,
            self.fixed_basis.dirichlet_alpha,
            self.fixed_basis.dynamic_design,
            rebuilt_summary_mean,
            independently_rebuilt.summary_covariance_factors,
        )
        factor_sha256 = _partition_factors_sha256(factors)
        reconstructed_factor_sha256 = _partition_factors_sha256(authenticated_factors)
        if factor_sha256 != reconstructed_factor_sha256:
            raise ValueError(
                "cached partition factors do not reconstruct from the authenticated native alpha, sensitivity, and basis."
            )

        native_alpha_sha256 = _array_sha256(alphas)
        basis_sha256 = _content_sha256(
            "aggregation-summary-basis-v1",
            (("summary_basis", basis),),
        )
        topology_sha256 = _content_sha256(
            "fixed-full-tiling-topology-v1",
            (("rectangle_bounds", self.fixed_basis.rectangle_bounds),),
            scalars=(
                ("rows", str(self.fixed_basis.tiling.shape[0])),
                ("columns", str(self.fixed_basis.tiling.shape[1])),
                ("k", str(self.fixed_basis.k)),
            ),
        )
        sensitivity_sha256 = _content_sha256(
            "native-physical-mass-sensitivity-v1",
            (("native_sensitivity", sensitivity),),
        )
        base = self.fixed_basis
        fixed_target_sha256 = _content_sha256(
            "fixed-basis-scientific-target-v1",
            (
                ("observations", base.observations),
                ("observation_sd", base.observation_sd),
                ("dynamic_design", base.dynamic_design),
                ("nominal_leaf_share", base.nominal_leaf_share),
                ("dirichlet_alpha", base.dirichlet_alpha),
                ("fixed_design", base.fixed_design),
                ("fixed_offset", base.fixed_offset),
                ("fixed_coefficient_prior_mean", base.fixed_coefficient_prior_mean),
                ("fixed_coefficient_prior_sd", base.fixed_coefficient_prior_sd),
            ),
            scalars=(
                ("root_shape", float(base.root_shape).hex()),
                ("root_rate", float(base.root_rate).hex()),
                ("likelihood_power", float(base.likelihood_power).hex()),
            ),
        )
        source_sha256 = _source_implementation_sha256()
        bridge_sha256 = _content_sha256(
            "fixed-basis-aggregation-bridge-v1",
            (),
            scalars=(
                ("native_alpha_id", identity),
                ("native_alpha_sha256", native_alpha_sha256),
                ("native_alpha_total", alpha_total.hex()),
                ("summary_basis_sha256", basis_sha256),
                ("partition_factors_sha256", factor_sha256),
                ("reconstructed_factors_sha256", reconstructed_factor_sha256),
                ("topology_sha256", topology_sha256),
                ("native_sensitivity_sha256", sensitivity_sha256),
                ("fixed_target_sha256", fixed_target_sha256),
            ),
        )
        model_identity_sha256 = _content_sha256(
            "fixed-basis-aggregation-model-identity-v1",
            (),
            scalars=(
                ("source_model_identity", _SOURCE_MODEL_ID),
                ("source_implementation_sha256", source_sha256),
                ("aggregation_bridge_sha256", bridge_sha256),
                ("summary_dimension", str(basis.shape[1])),
            ),
        )

        object.__setattr__(self, "summary_basis", basis)
        object.__setattr__(self, "native_cell_alphas", alphas)
        object.__setattr__(self, "native_sensitivity", sensitivity)
        object.__setattr__(self, "native_alpha_id", identity)
        object.__setattr__(self, "native_alpha_sha256", native_alpha_sha256)
        object.__setattr__(self, "native_alpha_total", alpha_total)
        object.__setattr__(self, "summary_basis_sha256", basis_sha256)
        object.__setattr__(self, "partition_factors_sha256", factor_sha256)
        object.__setattr__(
            self,
            "reconstructed_factors_sha256",
            reconstructed_factor_sha256,
        )
        object.__setattr__(self, "topology_sha256", topology_sha256)
        object.__setattr__(self, "native_sensitivity_sha256", sensitivity_sha256)
        object.__setattr__(self, "fixed_target_sha256", fixed_target_sha256)
        object.__setattr__(self, "source_implementation_sha256", source_sha256)
        object.__setattr__(self, "aggregation_bridge_sha256", bridge_sha256)
        object.__setattr__(self, "model_identity_sha256", model_identity_sha256)

    @property
    def k(self) -> int:
        """Return the fixed number of active regions."""
        return self.fixed_basis.k

    @property
    def summary_dimension(self) -> int:
        """Return the retained aggregation-summary rank."""
        return int(self.summary_basis.shape[1])

    @property
    def target_manifest(self) -> dict[str, object]:
        """Return the complete JSON-serializable target identity."""
        return {
            "schema": _MANIFEST_SCHEMA_ID,
            "source_model_identity": _SOURCE_MODEL_ID,
            "source_module": __name__,
            "source_implementation_sha256": self.source_implementation_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "aggregation_bridge_sha256": self.aggregation_bridge_sha256,
            "native_alpha_id": self.native_alpha_id,
            "native_alpha_sha256": self.native_alpha_sha256,
            "native_alpha_total": self.native_alpha_total,
            "summary_basis_sha256": self.summary_basis_sha256,
            "partition_factors_sha256": self.partition_factors_sha256,
            "reconstructed_factors_sha256": self.reconstructed_factors_sha256,
            "topology_sha256": self.topology_sha256,
            "native_sensitivity_sha256": self.native_sensitivity_sha256,
            "fixed_target_sha256": self.fixed_target_sha256,
            "k": self.k,
            "summary_dimension": self.summary_dimension,
            "observation_count": int(self.fixed_basis.observations.size),
            "fixed_coefficient_count": self.fixed_basis.n_fixed_coefficients,
        }


def prepare_fixed_basis_aggregation_nuts(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    *,
    summary_basis: ArrayLike,
    native_cell_alphas: ArrayLike,
    native_alpha_id: str,
) -> FixedBasisAggregationNUTSData:
    """Build fixed-partition cached factors from one explicit native prior.

    The caller, not this bridge, owns the cross-partition alpha-field choice.
    The field must aggregate to the existing fixed-basis Dirichlet shapes.
    No concentration is inferred from ``K``.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(initial_state, FullTilingPosteriorState):
        raise TypeError("initial_state must be a FullTilingPosteriorState.")
    if initial_state.problem is not problem:
        raise ValueError("initial_state must have been built for this exact problem.")
    fixed_basis = prepare_fixed_basis_nuts(problem, initial_state)
    basis = _readonly_float(summary_basis, name="summary_basis", ndim=2)
    alphas = _readonly_float(
        native_cell_alphas,
        name="native_cell_alphas",
        ndim=2,
    )
    if alphas.shape != problem.shape:
        raise ValueError("native_cell_alphas must have the problem's native shape.")
    if np.any(alphas <= 0.0):
        raise ValueError("native_cell_alphas must be strictly positive.")
    if not isinstance(native_alpha_id, str):
        raise TypeError("native_alpha_id must be a string.")
    if not native_alpha_id or native_alpha_id.strip() != native_alpha_id:
        raise ValueError("native_alpha_id must be a non-empty stripped string.")
    labels = _canonical_partition_labels(fixed_basis)

    aggregation = AdditiveDirichletAggregation(
        alphas,
        problem.base.sensitivity,
        problem.observation_sd,
        basis,
    )
    raw_factors = aggregation.partition_factors(labels)
    mean_tolerance = _scaled_tolerance(
        raw_factors.observation_mean_design,
        fixed_basis.dynamic_design,
    )
    if not np.allclose(
        raw_factors.observation_mean_design,
        fixed_basis.dynamic_design,
        rtol=0.0,
        atol=mean_tolerance,
    ):
        raise ValueError("native alpha conditional means do not match the fixed-basis mean design.")
    alpha_tolerance = _scaled_tolerance(
        raw_factors.alpha_totals,
        fixed_basis.dirichlet_alpha,
    )
    if not np.allclose(
        raw_factors.alpha_totals,
        fixed_basis.dirichlet_alpha,
        rtol=0.0,
        atol=alpha_tolerance,
    ):
        raise ValueError("native alpha field does not aggregate to the fixed-basis Dirichlet prior.")

    # Canonicalize the two roundoff-sensitive mean/alpha caches to the
    # authoritative fixed-basis arrays.  This makes q=0 exactly the existing
    # target while retaining covariance factors built from the explicit
    # native-cell field.
    summary_mean = (basis / fixed_basis.observation_sd[:, np.newaxis]).T @ fixed_basis.dynamic_design
    factors = PartitionSummaryFactors(
        labels,
        fixed_basis.dirichlet_alpha,
        fixed_basis.dynamic_design,
        summary_mean,
        raw_factors.summary_covariance_factors,
    )
    return FixedBasisAggregationNUTSData(
        fixed_basis=fixed_basis,
        factors=factors,
        summary_basis=basis,
        native_cell_alphas=alphas,
        native_sensitivity=problem.base.sensitivity,
        native_alpha_id=native_alpha_id,
    )


def _scientific_coordinates(
    data: FixedBasisAggregationNUTSData,
    root_total: object,
    leaf_share: ArrayLike,
    fixed_coefficient: ArrayLike,
) -> tuple[float, FloatArray, FloatArray]:
    """Validate one constrained scientific-coordinate point."""
    if not isinstance(data, FixedBasisAggregationNUTSData):
        raise TypeError("data must be FixedBasisAggregationNUTSData.")
    if isinstance(root_total, (bool, np.bool_)):
        raise TypeError("root_total must be a real number.")
    try:
        root = float(root_total)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("root_total must be a real number.") from error
    share = _readonly_float(leaf_share, name="leaf_share", ndim=1)
    fixed = _readonly_float(
        fixed_coefficient,
        name="fixed_coefficient",
        ndim=1,
    )
    if not math.isfinite(root) or root <= 0.0:
        raise ValueError("root_total must be finite and strictly positive.")
    if share.shape != (data.k,) or np.any(share <= 0.0):
        raise ValueError("leaf_share must have one strictly positive value per leaf.")
    if not math.isclose(
        float(np.sum(share)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("leaf_share must sum to one.")
    if fixed.shape != (data.fixed_basis.n_fixed_coefficients,) or np.any(fixed <= 0.0):
        raise ValueError("fixed_coefficient must be strictly positive with the fixed-design width.")
    return root, share, fixed


def fixed_basis_aggregation_numpy_logp_and_gradient(
    data: FixedBasisAggregationNUTSData,
    root_total: object,
    leaf_share: ArrayLike,
    fixed_coefficient: ArrayLike,
) -> ScientificEvaluation:
    """Evaluate the normalized target and analytic scientific-coordinate gradient.

    The simplex gradient is the ambient constrained-coordinate derivative;
    directional derivatives on the simplex are obtained by contracting it
    with zero-sum share perturbations.
    """
    root, share, fixed = _scientific_coordinates(
        data,
        root_total,
        leaf_share,
        fixed_coefficient,
    )
    base = data.fixed_basis
    masses = root * share
    mean = base.fixed_offset + data.factors.observation_mean_design @ masses + base.fixed_design @ fixed
    residual = (base.observations - mean) / base.observation_sd
    summary_residual = data.summary_basis.T @ residual
    covariance = np.zeros(
        (data.summary_dimension, data.summary_dimension),
        dtype=np.float64,
    )
    for region, mass in enumerate(masses):
        covariance += float(mass) ** 2 * data.factors.summary_covariance_factors[region]
    small_covariance = np.eye(data.summary_dimension, dtype=np.float64) + covariance
    if data.summary_dimension:
        try:
            cholesky = np.linalg.cholesky(small_covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "I + aggregation summary covariance is not numerically positive definite."
            ) from error
        inverse_summary_residual = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, summary_residual),
        )
        inverse_covariance = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(
                cholesky,
                np.eye(data.summary_dimension, dtype=np.float64),
            ),
        )
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    else:
        inverse_summary_residual = np.empty(0, dtype=np.float64)
        inverse_covariance = np.empty((0, 0), dtype=np.float64)
        log_determinant = 0.0

    diagonal_likelihood = -0.5 * (
        base.observations.size * _LOG_TWO_PI
        + 2.0 * float(np.sum(np.log(base.observation_sd)))
        + float(residual @ residual)
    )
    likelihood = diagonal_likelihood - 0.5 * (
        log_determinant
        + float(summary_residual @ inverse_summary_residual)
        - float(summary_residual @ summary_residual)
    )

    root_logp = (
        base.root_shape * math.log(base.root_rate)
        - math.lgamma(base.root_shape)
        + (base.root_shape - 1.0) * math.log(root)
        - base.root_rate * root
    )
    share_logp = (
        math.lgamma(float(np.sum(base.dirichlet_alpha)))
        - sum(math.lgamma(float(alpha)) for alpha in base.dirichlet_alpha)
        + float(np.dot(base.dirichlet_alpha - 1.0, np.log(share)))
    )
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    fixed_logp = float(
        np.sum(
            -0.5 * _LOG_TWO_PI
            - np.log(fixed_sigma)
            - np.log(fixed)
            - 0.5 * np.square((np.log(fixed) - fixed_mu) / fixed_sigma)
        )
    )
    logp = float(likelihood + root_logp + share_logp + fixed_logp)
    if not math.isfinite(logp):
        raise ValueError("scientific-coordinate target is non-finite.")

    summary_gradient = summary_residual - inverse_summary_residual
    mean_gradient = (residual - data.summary_basis @ summary_gradient) / base.observation_sd
    mass_gradient = data.factors.observation_mean_design.T @ mean_gradient
    if data.summary_dimension:
        covariance_gradient = 0.5 * (
            np.outer(inverse_summary_residual, inverse_summary_residual) - inverse_covariance
        )
        for region, mass in enumerate(masses):
            mass_gradient[region] += (
                2.0
                * float(mass)
                * float(np.sum(covariance_gradient * data.factors.summary_covariance_factors[region]))
            )
    root_gradient = float(np.dot(mass_gradient, share) + (base.root_shape - 1.0) / root - base.root_rate)
    share_gradient = root * mass_gradient + (base.dirichlet_alpha - 1.0) / share
    fixed_gradient = (
        base.fixed_design.T @ mean_gradient
        - 1.0 / fixed
        - (np.log(fixed) - fixed_mu) / (np.square(fixed_sigma) * fixed)
    )
    share_gradient = np.asarray(share_gradient, dtype=np.float64)
    fixed_gradient = np.asarray(fixed_gradient, dtype=np.float64)
    share_gradient.setflags(write=False)
    fixed_gradient.setflags(write=False)
    if (
        not math.isfinite(root_gradient)
        or not np.all(np.isfinite(share_gradient))
        or not np.all(np.isfinite(fixed_gradient))
    ):
        raise ValueError("scientific-coordinate target gradient is non-finite.")
    return (
        logp,
        root_gradient,
        cast(FloatArray, share_gradient),
        cast(
            FloatArray,
            fixed_gradient,
        ),
    )


def _numpy_joint_log_likelihood(
    data: FixedBasisAggregationNUTSData,
    root: float,
    share: FloatArray,
    fixed: FloatArray,
) -> float:
    """Recompute one normalized correlated likelihood in NumPy."""
    base = data.fixed_basis
    masses = root * share
    mean = base.fixed_offset + data.factors.observation_mean_design @ masses + base.fixed_design @ fixed
    residual = (base.observations - mean) / base.observation_sd
    likelihood = -0.5 * (
        base.observations.size * _LOG_TWO_PI
        + 2.0 * float(np.sum(np.log(base.observation_sd)))
        + float(residual @ residual)
    )
    if data.summary_dimension:
        summary_residual = data.summary_basis.T @ residual
        covariance = np.einsum(
            "k,kij->ij",
            np.square(masses),
            data.factors.summary_covariance_factors,
            optimize=False,
        )
        small_covariance = np.eye(data.summary_dimension, dtype=np.float64) + covariance
        try:
            cholesky = np.linalg.cholesky(small_covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "I + aggregation summary covariance is not numerically positive definite."
            ) from error
        solved = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, summary_residual),
        )
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        likelihood -= 0.5 * (
            log_determinant + float(summary_residual @ solved) - float(summary_residual @ summary_residual)
        )
    result = float(likelihood)
    if not math.isfinite(result):
        raise ValueError("normalized aggregation joint log likelihood is non-finite.")
    return result


def _pytensor_likelihood(
    data: FixedBasisAggregationNUTSData,
    root_total: Any,
    leaf_share: Any,
    fixed_coefficient: Any,
    pt: Any,
) -> tuple[Any, Any, Any]:
    """Return symbolic normalized likelihood, masses, and observation mean."""
    base = data.fixed_basis
    masses = root_total * leaf_share
    if base.n_fixed_coefficients:
        fixed_prediction = pt.dot(
            pt.as_tensor_variable(base.fixed_design),
            fixed_coefficient,
        )
    else:
        # A generic zero-length vector does not give PyTensor enough static
        # shape information to make dot((n, 0), vector(0)) reliable.  Retain a
        # zero-valued dependency for the scientific evaluator's empty gradient
        # while making the physical prediction explicitly zero.
        fixed_prediction = pt.zeros_like(
            pt.as_tensor_variable(base.fixed_offset),
            dtype="float64",
        ) + np.float64(0.0) * pt.sum(fixed_coefficient)
    mean = (
        pt.as_tensor_variable(base.fixed_offset)
        + pt.dot(
            pt.as_tensor_variable(data.factors.observation_mean_design),
            masses,
        )
        + fixed_prediction
    )
    residual = (pt.as_tensor_variable(base.observations) - mean) / pt.as_tensor_variable(base.observation_sd)
    likelihood = pt.sum(
        -np.float64(0.5 * _LOG_TWO_PI)
        - pt.log(pt.as_tensor_variable(base.observation_sd))
        - np.float64(0.5) * pt.square(residual)
    )
    if data.summary_dimension:
        summary_residual = pt.dot(
            pt.as_tensor_variable(data.summary_basis.T),
            residual,
        )
        covariance = pt.sum(
            pt.square(masses)[:, None, None] * pt.as_tensor_variable(data.factors.summary_covariance_factors),
            axis=0,
        )
        small_covariance = pt.eye(data.summary_dimension, dtype="float64") + covariance
        cholesky = pt.linalg.cholesky(small_covariance)
        solved = pt.linalg.solve(
            cholesky.T,
            pt.linalg.solve(cholesky, summary_residual),
        )
        log_determinant = np.float64(2.0) * pt.sum(pt.log(pt.diag(cholesky)))
        likelihood = likelihood - np.float64(0.5) * (
            log_determinant + pt.dot(summary_residual, solved) - pt.dot(summary_residual, summary_residual)
        )
    return likelihood, masses, mean


def compile_fixed_basis_aggregation_pytensor_evaluator(
    data: FixedBasisAggregationNUTSData,
) -> ScientificEvaluator:
    """Compile the target and gradient in authoritative scientific coordinates."""
    if not isinstance(data, FixedBasisAggregationNUTSData):
        raise TypeError("data must be FixedBasisAggregationNUTSData.")
    require_fixed_basis_nuts_float64()
    try:
        pytensor: Any = import_module("pytensor")
    except ImportError as error:
        raise ImportError(
            "Aggregation-aware NUTS requires PyTensor in the repository Pixi environment."
        ) from error
    pt: Any = import_module("pytensor.tensor")
    root = pt.scalar("scientific_root_total", dtype="float64")
    share = pt.vector("scientific_leaf_share", dtype="float64")
    fixed = pt.vector("scientific_fixed_coefficient", dtype="float64")
    likelihood, _, _ = _pytensor_likelihood(data, root, share, fixed, pt)
    base = data.fixed_basis
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    root_logp = (
        np.float64(base.root_shape * math.log(base.root_rate) - math.lgamma(base.root_shape))
        + np.float64(base.root_shape - 1.0) * pt.log(root)
        - np.float64(base.root_rate) * root
    )
    share_logp = np.float64(
        math.lgamma(float(np.sum(base.dirichlet_alpha)))
        - sum(math.lgamma(float(alpha)) for alpha in base.dirichlet_alpha)
    ) + pt.dot(
        pt.as_tensor_variable(base.dirichlet_alpha - 1.0),
        pt.log(share),
    )
    fixed_logp = pt.sum(
        -np.float64(0.5 * _LOG_TWO_PI)
        - pt.log(pt.as_tensor_variable(fixed_sigma))
        - pt.log(fixed)
        - np.float64(0.5)
        * pt.square((pt.log(fixed) - pt.as_tensor_variable(fixed_mu)) / pt.as_tensor_variable(fixed_sigma))
    )
    target = likelihood + root_logp + share_logp + fixed_logp
    gradient = pt.grad(target, (root, share, fixed))
    compiled = pytensor.function(
        [root, share, fixed],
        [target, gradient[0], gradient[1], gradient[2]],
        on_unused_input="raise",
    )

    def evaluate(
        root_total: float,
        leaf_share: ArrayLike,
        fixed_coefficient: ArrayLike,
    ) -> ScientificEvaluation:
        validated = _scientific_coordinates(
            data,
            root_total,
            leaf_share,
            fixed_coefficient,
        )
        result = compiled(*validated)
        logp = float(result[0])
        root_gradient = float(result[1])
        share_gradient = np.asarray(result[2], dtype=np.float64)
        fixed_gradient = np.asarray(result[3], dtype=np.float64)
        if (
            not math.isfinite(logp)
            or not math.isfinite(root_gradient)
            or not np.all(np.isfinite(share_gradient))
            or not np.all(np.isfinite(fixed_gradient))
        ):
            raise ValueError("PyTensor scientific target or gradient is non-finite.")
        share_gradient.setflags(write=False)
        fixed_gradient.setflags(write=False)
        return (
            logp,
            root_gradient,
            cast(FloatArray, share_gradient),
            cast(FloatArray, fixed_gradient),
        )

    return evaluate


def _manifest_json(data: FixedBasisAggregationNUTSData) -> str:
    """Return the authoritative serialized target manifest."""
    return _canonical_json(data.target_manifest)


def _manifest_sha256(manifest_json: str) -> str:
    """Fingerprint one canonical manifest serialization."""
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def _attach_model_binding(
    model: Any,
    data: FixedBasisAggregationNUTSData,
) -> None:
    """Bind one constructed PyMC graph to its exact data and target identity."""
    potential = model["aggregation_joint_likelihood_potential"]
    output = model["aggregation_joint_log_likelihood"]
    setattr(model, _MODEL_DATA_OBJECT_ATTR, data)
    setattr(model, _MODEL_TARGET_ID_ATTR, data.model_identity_sha256)
    setattr(model, _MODEL_MANIFEST_ATTR, _manifest_json(data))
    setattr(model, _MODEL_POTENTIAL_ATTR, potential)
    setattr(model, _MODEL_OUTPUT_ATTR, output)


def _validate_model_binding(
    model: Any,
    data: FixedBasisAggregationNUTSData,
    pm: Any,
) -> None:
    """Fail closed unless ``model`` is the exact authenticated graph for ``data``."""
    if not isinstance(model, pm.Model):
        raise TypeError("model must be a PyMC Model.")
    if getattr(model, _MODEL_DATA_OBJECT_ATTR, None) is not data:
        raise ValueError("model is not bound to this exact FixedBasisAggregationNUTSData object.")
    if getattr(model, _MODEL_TARGET_ID_ATTR, None) != data.model_identity_sha256:
        raise ValueError("model target identity does not match data.")
    if getattr(model, _MODEL_MANIFEST_ATTR, None) != _manifest_json(data):
        raise ValueError("model target manifest does not match data.")
    if model.observed_RVs:
        raise ValueError("aggregation-aware model must not contain observed RVs or a diagonal likelihood.")

    expected_free = {"root_total", "leaf_share"}
    if data.fixed_basis.n_fixed_coefficients:
        expected_free.add("fixed_coefficient")
    if {variable.name for variable in model.free_RVs} != expected_free:
        raise ValueError("aggregation-aware model free variables do not match the target.")
    if any(str(variable.dtype) != "float64" for variable in model.value_vars):
        raise ValueError("aggregation-aware model value variables must all be float64.")

    if len(model.potentials) != 1:
        raise ValueError("aggregation-aware model must contain exactly one scalar Potential.")
    potential = model.potentials[0]
    if (
        potential is not getattr(model, _MODEL_POTENTIAL_ATTR, None)
        or potential is not model["aggregation_joint_likelihood_potential"]
        or potential.name != "aggregation_joint_likelihood_potential"
        or potential.ndim != 0
    ):
        raise ValueError("aggregation-aware model has a missing or wrong scalar Potential.")
    output = model["aggregation_joint_log_likelihood"]
    if (
        output is not getattr(model, _MODEL_OUTPUT_ATTR, None)
        or output not in model.deterministics
        or output.ndim != 0
    ):
        raise ValueError(
            "aggregation-aware model must persist aggregation_joint_log_likelihood as a scalar deterministic."
        )

    initial = fixed_basis_nuts_initvals(data.fixed_basis)
    point_fn = pm.initial_point.make_initial_point_fn(
        model=model,
        overrides=initial,
        jitter_rvs=set(),
        default_strategy="support_point",
        return_transformed=True,
    )
    compiled_logp = float(model.compile_logp(jacobian=False)(point_fn(0)))
    expected_logp = fixed_basis_aggregation_numpy_logp_and_gradient(
        data,
        data.fixed_basis.initial_root_total,
        data.fixed_basis.initial_leaf_share,
        data.fixed_basis.initial_fixed_coefficient,
    )[0]
    tolerance = 5.0e-10 * max(1.0, abs(expected_logp))
    if not math.isfinite(compiled_logp) or abs(compiled_logp - expected_logp) > tolerance:
        raise ValueError(
            "aggregation-aware model graph does not reproduce its authenticated scientific target."
        )


def build_fixed_basis_aggregation_pymc_model(
    data: FixedBasisAggregationNUTSData,
) -> Model:
    """Build the fixed-partition aggregation-aware PyMC target."""
    if not isinstance(data, FixedBasisAggregationNUTSData):
        raise TypeError("data must be FixedBasisAggregationNUTSData.")
    require_fixed_basis_nuts_float64()
    try:
        import pymc as pm
    except ImportError as error:
        raise ImportError(
            "Aggregation-aware NUTS requires PyMC in the repository Pixi environment."
        ) from error
    pt: Any = import_module("pytensor.tensor")
    base = data.fixed_basis
    coords = {
        "observation": np.arange(base.observations.size, dtype=np.int64),
        "leaf": base.leaf_labels,
        "fixed": tuple(f"fixed_{position}" for position in range(base.n_fixed_coefficients)),
    }
    initvals = fixed_basis_nuts_initvals(base)
    fixed_mu, fixed_sigma = base.fixed_lognormal_mu_sigma
    with pm.Model(coords=coords) as model:
        root_total = pm.Gamma(
            "root_total",
            alpha=np.float64(base.root_shape),
            beta=np.float64(base.root_rate),
            initval=initvals["root_total"],
            dtype="float64",
        )
        leaf_share = pm.Dirichlet(
            "leaf_share",
            a=np.asarray(base.dirichlet_alpha, dtype=np.float64),
            dims="leaf",
            initval=initvals["leaf_share"],
            dtype="float64",
        )
        if base.n_fixed_coefficients:
            fixed_coefficient = pm.LogNormal(
                "fixed_coefficient",
                mu=np.asarray(fixed_mu, dtype=np.float64),
                sigma=np.asarray(fixed_sigma, dtype=np.float64),
                dims="fixed",
                initval=initvals["fixed_coefficient"],
                dtype="float64",
            )
        else:
            fixed_coefficient = pm.Deterministic(
                "fixed_coefficient",
                pt.zeros((0,), dtype="float64"),
                dims="fixed",
            )
        likelihood, leaf_mass_value, mean_value = _pytensor_likelihood(
            data,
            root_total,
            leaf_share,
            fixed_coefficient,
            pt,
        )
        leaf_mass = pm.Deterministic(
            "leaf_mass",
            leaf_mass_value,
            dims="leaf",
        )
        pm.Deterministic(
            "leaf_scaling",
            leaf_mass / pt.as_tensor_variable(base.nominal_leaf_share),
            dims="leaf",
        )
        pm.Deterministic(
            "mean_observation",
            mean_value,
            dims="observation",
        )
        pm.Deterministic(
            "aggregation_joint_log_likelihood",
            likelihood,
        )
        pm.Potential(
            "aggregation_joint_likelihood_potential",
            likelihood,
        )
    _attach_model_binding(model, data)
    return model


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _require_posterior_array(
    posterior: Any,
    *,
    name: str,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> FloatArray:
    """Return one finite float64 posterior array with exact dimensions."""
    variable = posterior[name]
    if tuple(variable.dims) != dims:
        raise ValueError(f"posterior variable {name!r} must have dimensions {dims}.")
    if tuple(variable.shape) != shape:
        raise ValueError(f"posterior variable {name!r} must have shape {shape}.")
    if np.dtype(variable.dtype) != np.dtype(np.float64):
        raise ValueError(f"posterior variable {name!r} must have dtype float64.")
    values = np.asarray(variable.values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"posterior variable {name!r} must contain only finite values.")
    return cast(FloatArray, values)


def _require_exact_coordinate(
    posterior: Any,
    *,
    name: str,
    expected: ArrayLike,
) -> None:
    """Require one dimension coordinate with exact values and order."""
    coordinate = posterior.coords[name]
    if tuple(coordinate.dims) != (name,):
        raise ValueError(f"posterior coordinate {name!r} must index only its own dimension.")
    if not np.array_equal(np.asarray(coordinate.values), np.asarray(expected)):
        raise ValueError(f"posterior coordinate {name!r} has the wrong values or order.")


def _require_close_posterior_identity(
    *,
    name: str,
    actual: FloatArray,
    expected: FloatArray,
) -> None:
    """Require one persisted deterministic to reproduce its coordinates."""
    if not np.allclose(
        actual,
        expected,
        rtol=_POSTERIOR_IDENTITY_RTOL,
        atol=_POSTERIOR_IDENTITY_ATOL,
    ):
        maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        raise ValueError(
            f"posterior scientific identity {name!r} is inconsistent; "
            f"maximum absolute difference is {maximum:.17g}."
        )


def validate_fixed_basis_aggregation_inference_data(
    data: FixedBasisAggregationNUTSData,
    result: InferenceData,
) -> dict[str, object]:
    """Validate persisted target identity and correlated-likelihood output."""
    if not isinstance(data, FixedBasisAggregationNUTSData):
        raise TypeError("data must be FixedBasisAggregationNUTSData.")
    try:
        az: Any = import_module("arviz")
    except ImportError as error:
        raise ImportError(
            "Aggregation-aware output validation requires ArviZ in the repository Pixi environment."
        ) from error
    if not isinstance(result, az.InferenceData):
        raise TypeError("result must be an ArviZ InferenceData.")
    groups = set(result.groups())
    if "posterior" not in groups:
        raise ValueError("aggregation-aware result must contain a posterior group.")
    if "log_likelihood" in groups or "observed_data" in groups:
        raise ValueError(
            "aggregation-aware result must not contain pointwise or diagonal observed-likelihood groups."
        )

    result_as_any: Any = result
    posterior = result_as_any.posterior
    expected_variables = {
        "aggregation_joint_log_likelihood",
        "fixed_coefficient",
        "leaf_mass",
        "leaf_scaling",
        "leaf_share",
        "mean_observation",
        "root_total",
    }
    observed_variables = set(posterior.data_vars)
    if observed_variables != expected_variables:
        missing = sorted(expected_variables - observed_variables)
        extra = sorted(observed_variables - expected_variables)
        raise ValueError(
            "posterior variables do not match the aggregation-aware schema; "
            f"missing={missing}, extra={extra}."
        )

    expected_coordinates = {"chain", "draw", "fixed", "leaf", "observation"}
    observed_coordinates = set(posterior.coords)
    if observed_coordinates != expected_coordinates:
        missing = sorted(expected_coordinates - observed_coordinates)
        extra = sorted(observed_coordinates - expected_coordinates)
        raise ValueError(
            "posterior coordinates do not match the aggregation-aware schema; "
            f"missing={missing}, extra={extra}."
        )
    observed_dimensions = set(posterior.sizes)
    if observed_dimensions != expected_coordinates:
        missing = sorted(expected_coordinates - observed_dimensions)
        extra = sorted(observed_dimensions - expected_coordinates)
        raise ValueError(
            "posterior dimensions do not match the aggregation-aware schema; "
            f"missing={missing}, extra={extra}."
        )

    chain_count = int(posterior.sizes["chain"])
    draw_count = int(posterior.sizes["draw"])
    base = data.fixed_basis
    if chain_count < 1 or draw_count < 1:
        raise ValueError("posterior must contain at least one chain and retained draw.")
    expected_sizes = {
        "chain": chain_count,
        "draw": draw_count,
        "leaf": data.k,
        "fixed": base.n_fixed_coefficients,
        "observation": base.observations.size,
    }
    if dict(posterior.sizes) != expected_sizes:
        raise ValueError(
            "posterior dimension sizes do not match the aggregation-aware target; "
            f"expected={expected_sizes}, observed={dict(posterior.sizes)}."
        )
    _require_exact_coordinate(
        posterior,
        name="chain",
        expected=np.arange(chain_count, dtype=np.int64),
    )
    _require_exact_coordinate(
        posterior,
        name="draw",
        expected=np.arange(draw_count, dtype=np.int64),
    )
    _require_exact_coordinate(
        posterior,
        name="leaf",
        expected=np.asarray(base.leaf_labels),
    )
    _require_exact_coordinate(
        posterior,
        name="fixed",
        expected=np.asarray(tuple(f"fixed_{position}" for position in range(base.n_fixed_coefficients))),
    )
    _require_exact_coordinate(
        posterior,
        name="observation",
        expected=np.arange(base.observations.size, dtype=np.int64),
    )

    chain_draw_shape = (chain_count, draw_count)
    root = _require_posterior_array(
        posterior,
        name="root_total",
        dims=("chain", "draw"),
        shape=chain_draw_shape,
    )
    share = _require_posterior_array(
        posterior,
        name="leaf_share",
        dims=("chain", "draw", "leaf"),
        shape=(*chain_draw_shape, data.k),
    )
    mass = _require_posterior_array(
        posterior,
        name="leaf_mass",
        dims=("chain", "draw", "leaf"),
        shape=(*chain_draw_shape, data.k),
    )
    scaling = _require_posterior_array(
        posterior,
        name="leaf_scaling",
        dims=("chain", "draw", "leaf"),
        shape=(*chain_draw_shape, data.k),
    )
    fixed = _require_posterior_array(
        posterior,
        name="fixed_coefficient",
        dims=("chain", "draw", "fixed"),
        shape=(*chain_draw_shape, base.n_fixed_coefficients),
    )
    mean = _require_posterior_array(
        posterior,
        name="mean_observation",
        dims=("chain", "draw", "observation"),
        shape=(*chain_draw_shape, base.observations.size),
    )
    likelihood = _require_posterior_array(
        posterior,
        name="aggregation_joint_log_likelihood",
        dims=("chain", "draw"),
        shape=chain_draw_shape,
    )

    if np.any(root <= 0.0):
        raise ValueError("posterior root_total must be strictly positive.")
    if np.any(share <= 0.0):
        raise ValueError("posterior leaf_share must be strictly positive.")
    if not np.allclose(
        np.sum(share, axis=-1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("posterior leaf_share must sum to one for every retained draw.")
    if np.any(mass <= 0.0) or np.any(scaling <= 0.0):
        raise ValueError("posterior leaf_mass and leaf_scaling must be strictly positive.")
    if base.n_fixed_coefficients and np.any(fixed <= 0.0):
        raise ValueError("posterior fixed_coefficient must be strictly positive.")

    expected_mass = root[..., np.newaxis] * share
    expected_scaling = expected_mass / base.nominal_leaf_share
    expected_mean = (
        base.fixed_offset
        + np.einsum(
            "...k,nk->...n",
            expected_mass,
            data.factors.observation_mean_design,
            optimize=False,
        )
        + np.einsum(
            "...f,nf->...n",
            fixed,
            base.fixed_design,
            optimize=False,
        )
    )
    _require_close_posterior_identity(
        name="leaf_mass = root_total * leaf_share",
        actual=mass,
        expected=expected_mass,
    )
    _require_close_posterior_identity(
        name="leaf_scaling = leaf_mass / nominal_leaf_share",
        actual=scaling,
        expected=expected_scaling,
    )
    _require_close_posterior_identity(
        name="mean_observation",
        actual=mean,
        expected=expected_mean,
    )

    # A4 output sizes are bounded, so certify every retained draw rather than
    # subsampling the trace.  This is intentionally independent of the
    # persisted PyTensor deterministic.
    expected_likelihood = np.empty(chain_draw_shape, dtype=np.float64)
    for index in np.ndindex(chain_draw_shape):
        expected_likelihood[index] = _numpy_joint_log_likelihood(
            data,
            float(root[index]),
            np.asarray(share[index], dtype=np.float64),
            np.asarray(fixed[index], dtype=np.float64),
        )
    if not np.allclose(
        likelihood,
        expected_likelihood,
        rtol=_POSTERIOR_LIKELIHOOD_RTOL,
        atol=_POSTERIOR_LIKELIHOOD_ATOL,
    ):
        maximum = float(np.max(np.abs(likelihood - expected_likelihood)))
        raise ValueError(
            "posterior aggregation_joint_log_likelihood does not reproduce the "
            "normalized NumPy joint likelihood for every retained draw; "
            f"maximum absolute difference is {maximum:.17g}."
        )

    expected_json = _manifest_json(data)
    expected_sha256 = _manifest_sha256(expected_json)
    attrs = result.attrs
    observed_json = attrs.get(_MANIFEST_ATTR)
    observed_sha256 = attrs.get(_MANIFEST_SHA_ATTR)
    if not isinstance(observed_json, str) or observed_json != expected_json:
        raise ValueError("InferenceData target manifest is missing or mismatched.")
    if (
        not isinstance(observed_sha256, str)
        or observed_sha256 != expected_sha256
        or _manifest_sha256(observed_json) != observed_sha256
    ):
        raise ValueError("InferenceData target manifest checksum is invalid.")
    try:
        manifest = json.loads(observed_json)
    except json.JSONDecodeError as error:
        raise ValueError("InferenceData target manifest is not valid JSON.") from error
    if manifest != data.target_manifest:
        raise ValueError("InferenceData target manifest content does not match data.")
    return cast(dict[str, object], manifest)


def sample_fixed_basis_aggregation_nuts(
    model: Model,
    data: FixedBasisAggregationNUTSData,
    *,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
    chains: int,
    cores: int,
    chain_method: ChainMethod,
    progressbar: bool,
    max_tree_depth: int = 10,
    dense_mass: bool = False,
    initvals: ConstrainedInitvals | None = None,
) -> InferenceData:
    """Run NumPyro NUTS without creating a false pointwise likelihood."""
    if not isinstance(data, FixedBasisAggregationNUTSData):
        raise TypeError("data must be FixedBasisAggregationNUTSData.")
    require_fixed_basis_nuts_float64()
    try:
        az: Any = import_module("arviz")
        pm: Any = import_module("pymc")
    except ImportError as error:
        raise ImportError(
            "Aggregation-aware NUTS requires PyMC and ArviZ in the repository Pixi environment."
        ) from error
    _validate_model_binding(model, data, pm)
    draws = _positive_integer(draws, name="draws")
    tune = _positive_integer(tune, name="tune")
    chains = _positive_integer(chains, name="chains")
    cores = _positive_integer(cores, name="cores")
    max_tree_depth = _positive_integer(max_tree_depth, name="max_tree_depth")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    target_accept = float(target_accept)
    if not math.isfinite(target_accept) or not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie strictly between zero and one.")
    if chain_method not in ("parallel", "vectorized"):
        raise ValueError("chain_method must be 'parallel' or 'vectorized'.")
    if not isinstance(progressbar, bool):
        raise TypeError("progressbar must be Boolean.")
    if not isinstance(dense_mass, bool):
        raise TypeError("dense_mass must be Boolean.")

    selected_initvals = fixed_basis_nuts_initvals(data.fixed_basis) if initvals is None else initvals
    with model:
        result = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=progressbar,
            nuts_sampler="numpyro",
            initvals=selected_initvals,
            target_accept=target_accept,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            nuts_sampler_kwargs={
                "jitter": False,
                "chain_method": chain_method,
                "nuts_kwargs": {
                    "max_tree_depth": max_tree_depth,
                    "dense_mass": dense_mass,
                },
            },
        )
    if not isinstance(result, az.InferenceData):
        raise RuntimeError("PyMC did not return an ArviZ InferenceData result.")
    manifest_json = _manifest_json(data)
    result.attrs[_MANIFEST_ATTR] = manifest_json
    result.attrs[_MANIFEST_SHA_ATTR] = _manifest_sha256(manifest_json)
    validate_fixed_basis_aggregation_inference_data(data, result)
    return result
