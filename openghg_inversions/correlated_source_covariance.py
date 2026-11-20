"""Native source covariance with cross-source correlation under a shared kernel.

:class:`~openghg_inversions.source_covariance.IndependentSourceCovariance`
represents a block-diagonal multisource native covariance. This module adds
cross-source blocks for sources that share one spatial correlation kernel,
which is the structure needed for a correlated GPP/TER biosphere prior.

Model
-----
Let ``x_s`` be the dimensionless native scaling anomaly of source ``s`` on the
shared spatial grid and let each source's own covariance action be
``B_ss = sigma_s**2 K``. For a group of sources that share the unit-variance
kernel ``K``,

``B = S (R kron K) S``,

where ``S = diag(sigma_s)`` and ``R`` is the source correlation matrix with
unit diagonal. For the GPP/TER pair ``R = [[1, rho], [rho, 1]]``, so the cross
block is ``B_GT = rho sigma_G sigma_T K``. Because ``K`` is positive
semidefinite, ``B`` is positive semidefinite exactly when ``R`` is, which for
one pair means ``|rho| <= 1``; ``B`` is invertible when ``R`` is positive
definite (``|rho| < 1``) and every source action is invertible.

Only the shared-kernel form is supported. If ``B_GG = sigma_G**2 K_G`` and
``B_TT = sigma_T**2 K_T`` with ``K_G != K_T``, the block matrix with cross block
``rho sigma_G sigma_T K_G`` is not in general positive semidefinite, so a
correlation between sources whose kernels differ is refused. Kernel identity
is decided in one place: object identity; a kernel family's canonical
``same_unit_kernel(other)`` method when it has one; for
:class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`,
the grid, resolved correlation lengths and class partition (ignoring label
values, dtype, order, name and attributes); otherwise the ``to_dataset()``
configuration with ``sigma`` removed. Any
:class:`~openghg_inversions.native_covariance.InvertibleNativeCovarianceAction`
that exposes ``sigma`` and one of these comparisons can therefore be
correlated, independent of the kernel family.

Sign convention
---------------
``rho`` is the correlation of the native *scaling anomalies*
``rho = Corr(x_G, x_T)``, not of the signed flux anomalies. The flux anomaly is
``delta f_s = f_s^0 * x_s`` with the signed prior template ``f^0``; in the
atmospheric convention GPP is negative uptake (``f_G^0 < 0``) and TER is
positive (``f_T^0 > 0``). A positive ``rho`` therefore means that when the
magnitude of GPP is scaled up, TER is also scaled up: the signed flux
anomalies are *negatively* correlated and partly cancel in the net biosphere
flux. For a country functional ``w_s = a * f_s^0``,

``Var(NET) = sigma_G**2 w_G' K w_G + sigma_T**2 w_T' K w_T
+ 2 rho sigma_G sigma_T w_G' K w_T``,

where ``w_G' K w_T < 0``, so ``rho > 0`` reduces and ``rho < 0`` inflates the
net prior variance. In the common/contrast parameterisation
``x_G = a + d``, ``x_T = a - d`` with fraction ``f`` of the variance in the
common mode, ``rho = 2 f - 1``; ``f = 0.2`` is ``rho = -0.6``, the
reinforcing (not cancelling) choice.

Reduction
---------
:func:`~openghg_inversions.coherent_reduction.reduce_native_gaussian` and
:func:`~openghg_inversions.basis.covariance_products.project_native_covariance`
consume one covariance action over the stacked ``(source, lat, lon)`` state.
Passing this action with a source-blocked restriction (for example from
:func:`~openghg_inversions.basis.covariance_products.absolute_weighted_regional_mean_restriction`)
therefore performs the joint GPP/TER reduction: the cross blocks enter
``C_alpha = Pi B Pi.T``, ``H B Pi.T`` and ``H B H.T``, and hence the
unresolved covariance ``A``. Reducing each source separately and summing the
per-source ``A`` silently drops them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions import native_covariance
from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.native_covariance import (
    InvertibleNativeCovarianceAction,
    SeparableExponentialCovariance,
)

__all__ = [
    "CorrelatedSourceCovariance",
    "ScaledKernelCovarianceAction",
    "gpp_ter_source_covariance",
]

_PSD_TOLERANCE = 1e-12
_PD_TOLERANCE = 1e-10
_COMPONENT_PREFIX = re.compile(r"^source_\d+__")
_SERIALIZED_NAMES = {"correlation_first", "correlation_second", "correlation_value", "correlation_pair"}


@runtime_checkable
class ScaledKernelCovarianceAction(InvertibleNativeCovarianceAction, Protocol):
    """Invertible action of the form ``sigma**2 K`` with a serializable kernel.

    A source taking part in a nonzero cross-source correlation must expose its
    marginal standard deviation ``sigma`` and a way to compare its unit kernel:
    a canonical ``same_unit_kernel(other)`` method, or (as a conservative
    fallback) a ``to_dataset()`` configuration, compared with ``sigma``
    removed. :class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
    is compared on its grid, lengths and class partition.
    """

    @property
    def sigma(self) -> float:
        """Native-cell marginal standard deviation of the action."""
        ...

    def to_dataset(self) -> xr.Dataset:
        """Return the reproducible configuration of the action."""
        ...


class CorrelatedSourceCovariance:
    """Multisource native covariance with correlated shared-kernel sources.

    Sources without a configured correlation act independently, exactly as in
    :class:`~openghg_inversions.source_covariance.IndependentSourceCovariance`.
    Sources joined by nonzero correlations form groups that must share one
    unit-variance kernel; within a group the covariance is
    ``S (R kron K) S`` (see the module docstring for the sign convention).

    Apply costs one component apply per source, solve one component solve per
    source; no dense native or cross-source matrix is formed. With no nonzero
    correlation, apply and solve return exactly the component results, so the
    action is bit-identical to the independent block-diagonal action.

    Args:
        source_covariances: Non-empty insertion-ordered mapping from source
            label to that source's own invertible native covariance action
            ``B_ss`` on a common spatial grid. Insertion order defines the
            canonical source order.
        correlations: Optional mapping from unordered source-label pairs to
            the correlation ``rho`` of their native scaling anomalies. Zero
            entries are ignored. A pair may be given in either order but not
            in both.
        source_dim: Native source dimension name.

    Raises:
        TypeError: If a correlated source does not expose ``sigma`` and
            ``to_dataset()``.
        ValueError: If labels, dimensions or correlations are invalid, if a
            correlation is not finite or ``|rho| > 1``, if correlated sources
            do not share one kernel, or if a group's correlation matrix is not
            positive semidefinite.
    """

    __slots__ = (
        "_correlation_matrix",
        "_correlations",
        "_groups",
        "_group_precisions",
        "_sigmas",
        "_source_covariances",
        "_source_dim",
        "_source_labels",
    )

    schema = "openghg_inversions.correlated_source_covariance"
    schema_version = 1

    def __init__(
        self,
        source_covariances: Mapping[str, InvertibleNativeCovarianceAction],
        correlations: Mapping[tuple[str, str], float] | None = None,
        *,
        source_dim: str = "native_source",
    ) -> None:
        """Validate sources, correlations and the shared-kernel requirement."""
        covariances = dict(source_covariances)
        if not covariances:
            raise ValueError("source_covariances must contain at least one source block")
        if not isinstance(source_dim, str) or not source_dim:
            raise ValueError("source_dim must be a non-empty string")
        if any(not isinstance(label, str) or not label for label in covariances):
            raise ValueError("source covariance labels must be non-empty strings")
        reference_dims = tuple(next(iter(covariances.values())).native_dims)
        for label, covariance in covariances.items():
            if tuple(covariance.native_dims) != reference_dims:
                raise ValueError(
                    f"Source {label!r} covariance uses native dimensions {tuple(covariance.native_dims)!r}; "
                    f"expected {reference_dims!r}"
                )
        if source_dim in reference_dims:
            raise ValueError("source_dim must be distinct from the spatial native dimensions")
        if source_dim in _SERIALIZED_NAMES:
            raise ValueError(f"source_dim {source_dim!r} is reserved by the serialized schema")
        _require_common_grid(covariances)

        labels = tuple(covariances)
        position = {label: index for index, label in enumerate(labels)}
        matrix = np.eye(len(labels))
        resolved: dict[tuple[str, str], float] = {}
        for pair, value in dict(correlations or {}).items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Correlation keys must be (source, source) pairs; got {pair!r}")
            first, second = pair
            if first not in position or second not in position:
                raise ValueError(f"Correlation pair {pair!r} names an unconfigured source; sources are {labels!r}")
            if first == second:
                raise ValueError(f"Correlation pair {pair!r} must name two different sources")
            rho = float(value)
            if not np.isfinite(rho):
                raise ValueError(f"Correlation for {pair!r} must be finite; got {value!r}")
            if abs(rho) > 1.0:
                raise ValueError(
                    f"Correlation for {pair!r} is {rho!r}; |rho| <= 1 is required for a positive "
                    "semidefinite covariance"
                )
            key = (first, second) if position[first] < position[second] else (second, first)
            if key in resolved:
                raise ValueError(f"Correlation for sources {key!r} is given more than once")
            resolved[key] = rho
            matrix[position[first], position[second]] = rho
            matrix[position[second], position[first]] = rho

        nonzero = {pair: rho for pair, rho in resolved.items() if rho != 0.0}
        sigmas: dict[str, float] = {}
        for first, second in nonzero:
            _require_shared_kernel(first, covariances[first], second, covariances[second])
            for label in (first, second):
                sigmas[label] = _sigma(label, covariances[label])

        groups = _connected_groups(labels, nonzero)
        group_precisions: dict[tuple[str, ...], np.ndarray | None] = {}
        for group in groups:
            indices = [position[label] for label in group]
            group_matrix = matrix[np.ix_(indices, indices)]
            smallest = float(np.linalg.eigvalsh(group_matrix).min())
            if smallest < -_PSD_TOLERANCE:
                raise ValueError(
                    f"Correlation matrix of sources {group!r} is not positive semidefinite "
                    f"(smallest eigenvalue {smallest:.3e})"
                )
            # A PSD but singular R (|rho| = 1, or a near-zero eigenvalue in a
            # larger group) gives a valid singular B: apply works, solve is undefined.
            group_precisions[group] = np.linalg.inv(group_matrix) if smallest > _PD_TOLERANCE else None

        self._source_covariances = MappingProxyType(covariances)
        self._source_dim = source_dim
        self._source_labels = labels
        self._correlations = MappingProxyType(resolved)
        self._correlation_matrix = matrix
        self._correlation_matrix.setflags(write=False)
        self._sigmas = MappingProxyType(sigmas)
        self._groups = groups
        self._group_precisions = group_precisions

    @property
    def source_covariances(self) -> Mapping[str, InvertibleNativeCovarianceAction]:
        """Return the read-only source-to-own-covariance mapping."""
        return self._source_covariances

    @property
    def source_dim(self) -> str:
        """Return the native source dimension name."""
        return self._source_dim

    @property
    def source_labels(self) -> tuple[str, ...]:
        """Return the configured source labels in canonical order."""
        return self._source_labels

    @property
    def correlations(self) -> Mapping[tuple[str, str], float]:
        """Return the configured pair correlations in canonical pair order."""
        return self._correlations

    @property
    def correlation_matrix(self) -> xr.DataArray:
        """Return a labelled copy of the source correlation matrix ``R``.

        Returns:
            Symmetric ``R`` with unit diagonal on ``(source_dim,
            f"{source_dim}_cov")``; uncorrelated sources have zero entries.
        """
        labels = list(self._source_labels)
        return xr.DataArray(
            self._correlation_matrix.copy(),
            dims=(self._source_dim, f"{self._source_dim}_cov"),
            coords={self._source_dim: labels, f"{self._source_dim}_cov": labels},
            name="source_correlation",
            attrs={"definition": "Corr(x_s, x_t) of native scaling anomalies"},
        )

    def to_dataset(self) -> xr.Dataset:
        """Serialize source order, correlations and component configurations.

        Each source's own covariance is stored through its ``to_dataset()``
        with its variables, non-native dimensions and attributes prefixed by
        ``source_<i>__``, so components of any OGI kernel family can be saved
        for provenance and restored by :meth:`from_dataset`.

        Returns:
            Versioned dataset with the configured correlation pairs, the
            derived correlation matrix ``R`` and the component configurations.

        Raises:
            TypeError: If a component lacks ``to_dataset()``.
        """
        labels = list(self._source_labels)
        pairs = list(self._correlations.items())
        parts = [
            xr.Dataset(
                {
                    "source_correlation": self.correlation_matrix,
                    "correlation_first": ("correlation_pair", [pair[0] for pair, _ in pairs]),
                    "correlation_second": ("correlation_pair", [pair[1] for pair, _ in pairs]),
                    "correlation_value": ("correlation_pair", [rho for _, rho in pairs]),
                }
            )
        ]
        attrs: dict[str, object] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "source_dim": self._source_dim,
            "sign_convention": "Corr(x_s, x_t) of native scaling anomalies; GPP negative uptake",
        }
        for index, label in enumerate(labels):
            covariance = self._source_covariances[label]
            to_dataset = getattr(covariance, "to_dataset", None)
            if not callable(to_dataset):
                raise TypeError(f"Source {label!r} covariance cannot be serialized: it has no to_dataset()")
            component = to_dataset()
            prefix = f"source_{index}__"
            native = set(covariance.native_dims)
            names = {str(name) for name in (*component.variables, *component.dims)} - native
            component = component.rename({name: prefix + name for name in names})
            attrs[prefix + "type"] = type(covariance).__name__
            attrs[prefix + "attrs"] = json.dumps(_json_value(component.attrs), sort_keys=True)
            parts.append(component.drop_attrs(deep=False))
        dataset = xr.merge(parts, compat="override", join="exact", combine_attrs="drop_conflicts")
        return dataset.assign_attrs(attrs)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> CorrelatedSourceCovariance:
        """Restore an action from :meth:`to_dataset` output.

        Component types are resolved only among the covariance classes of
        :mod:`openghg_inversions.native_covariance` that define
        ``from_dataset``.

        Args:
            dataset: Versioned correlated-source covariance dataset.

        Returns:
            The reconstructed action, revalidated (shared kernels, PSD ``R``).

        Raises:
            ValueError: If the schema, a component type or the stored
                correlation matrix is invalid.
        """
        if dataset.attrs.get("schema") != cls.schema or dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError(f"Expected covariance schema {cls.schema!r} version {cls.schema_version}")
        source_dim = str(dataset.attrs["source_dim"])
        labels = [str(label) for label in dataset.coords[source_dim].values]
        covariances: dict[str, InvertibleNativeCovarianceAction] = {}
        reserved_dims = {source_dim, f"{source_dim}_cov", "correlation_pair"}
        for index, label in enumerate(labels):
            prefix = f"source_{index}__"
            component_type = getattr(native_covariance, str(dataset.attrs.get(prefix + "type")), None)
            if not isinstance(component_type, type) or not hasattr(component_type, "from_dataset"):
                raise ValueError(f"Unsupported serialized covariance type for source {label!r}")
            # Keep this component's prefixed variables and the shared spatial
            # dimension coordinates; everything else belongs to other parts.
            keep = {
                str(name)
                for name in dataset.variables
                if str(name).startswith(prefix)
                or (name in dataset.dims and name not in reserved_dims and not _COMPONENT_PREFIX.match(str(name)))
            }
            component = dataset.drop_vars([name for name in dataset.variables if str(name) not in keep])
            component = component.rename(
                {
                    name: name[len(prefix) :]
                    for name in {str(item) for item in (*component.variables, *component.dims)}
                    if name.startswith(prefix)
                }
            )
            component.attrs = json.loads(str(dataset.attrs[prefix + "attrs"]))
            covariances[label] = component_type.from_dataset(component)
        correlations = {
            (str(first), str(second)): float(rho)
            for first, second, rho in zip(
                dataset["correlation_first"].values,
                dataset["correlation_second"].values,
                dataset["correlation_value"].values,
                strict=True,
            )
        }
        restored = cls(covariances, correlations, source_dim=source_dim)
        if not np.array_equal(restored.correlation_matrix.values, dataset["source_correlation"].values):
            raise ValueError("Serialized source_correlation contradicts the correlation pairs")
        return restored

    @property
    def native_dims(self) -> tuple[str, ...]:
        """Return the source dimension followed by the common spatial dimensions."""
        first = self._source_covariances[self._source_labels[0]]
        return (self._source_dim, *first.native_dims)

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply ``B`` to labelled right-hand sides.

        For source ``s`` the result is
        ``B_ss x_s + sum_t rho_st (sigma_s / sigma_t) B_tt x_t``, which equals
        ``sum_t rho_st sigma_s sigma_t K x_t`` because ``B_tt = sigma_t**2 K``.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                other dimensions are preserved.

        Returns:
            ``B rhs`` with the layout and labels of ``rhs``.

        Raises:
            ValueError: If source or spatial labels are missing or invalid.
        """
        return self._operate(rhs, operation="apply")

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve ``B x = rhs`` for labelled right-hand sides.

        Uses ``B^-1 = S^-1 (R^-1 kron K^-1) S^-1`` within each correlated
        group: ``x_s = sum_t (R^-1)_st (sigma_t / sigma_s) B_tt^-1 rhs_t``.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                other dimensions are preserved.

        Returns:
            ``B^-1 rhs`` with the layout and labels of ``rhs``.

        Raises:
            ValueError: If source or spatial labels are missing or invalid.
            numpy.linalg.LinAlgError: If a correlated group's correlation
                matrix is not positive definite (e.g. ``|rho| = 1``) or a
                component solve fails.
        """
        return self._operate(rhs, operation="solve")

    def _operate(self, rhs: xr.DataArray, *, operation: Literal["apply", "solve"]) -> xr.DataArray:
        """Dispatch component operations, then mix correlated sources."""
        if operation == "solve":
            singular = [group for group in self._groups if self._group_precisions[group] is None]
            if singular:
                group = singular[0]
                unit = [pair for pair, rho in self._correlations.items() if abs(rho) == 1.0 and pair[0] in group]
                cause = f"|rho| = 1 for {unit[0]!r}" if unit else "it has a (near-)zero eigenvalue"
                raise np.linalg.LinAlgError(
                    f"R is not positive definite for sources {group!r} ({cause}); the covariance is "
                    "positive semidefinite but has no inverse"
                )
        expected = xr.DataArray(
            list(self._source_labels),
            dims=self._source_dim,
            coords={self._source_dim: list(self._source_labels)},
        )
        if self._source_dim not in rhs.indexes:
            raise ValueError(f"rhs is missing indexed source coordinate {self._source_dim!r}")
        aligned_rhs, _ = xr.align(rhs, expected, join="exact", copy=False)

        original_dims = tuple(str(dim) for dim in rhs.dims)
        component: dict[str, xr.DataArray] = {}
        for label in self._source_labels:
            source_rhs = aligned_rhs.sel({self._source_dim: label})
            component[label] = getattr(self._source_covariances[label], operation)(source_rhs)

        results = dict(component)
        position = {label: index for index, label in enumerate(self._source_labels)}
        for group in self._groups:
            if operation == "apply":
                indices = [position[label] for label in group]
                mixing = self._correlation_matrix[np.ix_(indices, indices)]
            else:
                precision = self._group_precisions[group]
                assert precision is not None
                mixing = precision
            for row, target in enumerate(group):
                dims = component[target].dims
                values = np.asarray(component[target].data)
                if operation == "solve":
                    values = mixing[row, row] * values
                for column, other in enumerate(group):
                    if column == row or mixing[row, column] == 0.0:
                        continue
                    # apply: B_st x_t = rho_st (sigma_s / sigma_t) B_tt x_t.
                    # solve: (B^-1)_st y_t = (R^-1)_st (sigma_t / sigma_s) B_tt^-1 y_t.
                    if operation == "apply":
                        ratio = self._sigmas[target] / self._sigmas[other]
                    else:
                        ratio = self._sigmas[other] / self._sigmas[target]
                    other_values = np.asarray(component[other].transpose(*dims).data)
                    values = values + (mixing[row, column] * ratio) * other_values
                results[target] = component[target].copy(data=values)

        combined = xr.concat(
            [results[label] for label in self._source_labels],
            dim=aligned_rhs.coords[self._source_dim],
            coords="minimal",
            compat="override",
        )
        restored_data = combined.transpose(*original_dims, transpose_coords=False).data
        return rhs.copy(data=restored_data, deep=False)


def gpp_ter_source_covariance(
    source_covariances: Mapping[str, InvertibleNativeCovarianceAction],
    *,
    gpp_source: str,
    ter_source: str,
    gpp_ter_correlation: float = 0.0,
    source_dim: str = "native_source",
) -> CorrelatedSourceCovariance:
    """Build the multisource native covariance with a GPP/TER correlation.

    ``gpp_ter_correlation`` is ``rho = Corr(x_GPP, x_TER)`` of the native
    *scaling* anomalies (GPP negative uptake, TER positive). ``rho > 0`` makes
    the signed GPP and TER flux anomalies cancel in the net biosphere flux;
    ``rho < 0`` makes them reinforce. With the default ``0`` the result acts
    bit-identically to
    :class:`~openghg_inversions.source_covariance.IndependentSourceCovariance`
    over the same components. A nonzero ``rho`` requires that the GPP and TER
    actions share one kernel and differ at most in ``sigma``.

    Args:
        source_covariances: Ordered source-to-own-covariance mapping including
            ``gpp_source`` and ``ter_source``.
        gpp_source: Label of the GPP source, e.g. ``"co2_BASE_GPP"``.
        ter_source: Label of the TER source, e.g. ``"co2_BASE_TER"``.
        gpp_ter_correlation: Scaling-anomaly correlation ``rho`` with
            ``|rho| <= 1``.
        source_dim: Native source dimension name.

    Returns:
        Covariance action for the joint GPP/TER (and other sources) reduction.

    Raises:
        TypeError: If ``rho != 0`` and an action lacks ``sigma`` or
            ``to_dataset()``.
        ValueError: If labels are missing, ``|rho| > 1``, or the GPP and TER
            kernels differ while ``rho != 0``.
    """
    for name, label in (("gpp_source", gpp_source), ("ter_source", ter_source)):
        if label not in source_covariances:
            raise ValueError(f"{name} {label!r} is not a configured source; sources are {tuple(source_covariances)!r}")
    return CorrelatedSourceCovariance(
        source_covariances,
        {(gpp_source, ter_source): gpp_ter_correlation},
        source_dim=source_dim,
    )


def _sigma(label: str, covariance: InvertibleNativeCovarianceAction) -> float:
    """Return a correlated source's positive finite marginal standard deviation."""
    sigma = getattr(covariance, "sigma", None)
    if sigma is None:
        raise TypeError(f"Correlated source {label!r} must expose its marginal standard deviation 'sigma'")
    value = float(sigma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Correlated source {label!r} has invalid sigma {sigma!r}")
    return value


def _require_common_grid(covariances: Mapping[str, InvertibleNativeCovarianceAction]) -> None:
    """Refuse source actions whose spatial coordinate values differ."""
    reference_label, reference = next(iter(covariances.items()))
    for label, covariance in covariances.items():
        for axis in ("latitude", "longitude"):
            expected, actual = getattr(reference, axis, None), getattr(covariance, axis, None)
            if expected is None or actual is None:
                continue
            if not np.array_equal(np.asarray(expected.values), np.asarray(actual.values)):
                raise ValueError(
                    f"Source {label!r} covariance grid differs from source {reference_label!r} on {axis}"
                )


def _require_shared_kernel(
    first: str,
    first_covariance: InvertibleNativeCovarianceAction,
    second: str,
    second_covariance: InvertibleNativeCovarianceAction,
) -> None:
    """Refuse a correlation between sources whose unit kernels differ."""
    if not _same_unit_kernel(first, first_covariance, second, second_covariance):
        raise ValueError(
            f"Sources {first!r} and {second!r} do not share one unit kernel (their kernel type, grid, "
            "class partition or correlation lengths differ); B = S (R kron K) S is positive semidefinite "
            "only when the correlated sources share K. Use the same kernel, or set the correlation to 0"
        )


def _same_unit_kernel(
    first: str,
    first_covariance: InvertibleNativeCovarianceAction,
    second: str,
    second_covariance: InvertibleNativeCovarianceAction,
) -> bool:
    """Return whether two actions apply the same unit-variance kernel ``K``.

    This is the single comparison point for kernel identity. A kernel family
    that provides a canonical public ``same_unit_kernel(other)`` or
    ``kernel_identity()`` method is trusted to define it. The separable exponential kernel is compared on what
    defines ``K``: native dimensions and coordinate values, resolved
    correlation lengths and the class partition (not label values, dtype,
    order, name or attributes). Other actions fall back to their
    ``to_dataset()`` configuration with ``sigma`` removed, which can refuse
    equal kernels whose serialization differs cosmetically. This fallback
    assumes ``sigma`` is the only amplitude in the configuration: an action
    that stores another amplitude-like parameter (e.g. a per-class variance)
    must provide ``same_unit_kernel`` instead.
    """
    if first_covariance is second_covariance:
        return True
    if type(first_covariance) is not type(second_covariance):
        return False
    canonical = getattr(first_covariance, "same_unit_kernel", None)
    if callable(canonical):
        return bool(canonical(second_covariance))
    identity = getattr(first_covariance, "kernel_identity", None)
    if callable(identity):
        return bool(identity() == getattr(second_covariance, "kernel_identity")())
    if isinstance(first_covariance, SeparableExponentialCovariance):
        assert isinstance(second_covariance, SeparableExponentialCovariance)
        return _same_separable_kernel(first_covariance, second_covariance)
    return _kernel_configuration(first, first_covariance).identical(
        _kernel_configuration(second, second_covariance)
    )


def _same_separable_kernel(first: SeparableExponentialCovariance, second: SeparableExponentialCovariance) -> bool:
    """Compare the quantities that define a separable exponential unit kernel."""
    return (
        first.native_dims == second.native_dims
        and np.array_equal(first.latitude.values, second.latitude.values)
        and np.array_equal(first.longitude.values, second.longitude.values)
        and first.latitude_correlation_length == second.latitude_correlation_length
        and first.longitude_correlation_length == second.longitude_correlation_length
        and np.array_equal(_class_partition(first), _class_partition(second))
    )


def _class_partition(covariance: SeparableExponentialCovariance) -> np.ndarray:
    """Return class codes in order of first appearance; one class if unblocked.

    The blocked kernel depends only on which cells share a class, so label
    values, their dtype (``1`` and ``1.0``), order, name and attributes are
    irrelevant.
    """
    labels = covariance.class_labels
    if labels is None:
        return np.zeros(covariance.latitude.size * covariance.longitude.size, dtype=np.intp)
    values = np.asarray(labels.transpose(*covariance.native_dims).values).reshape(-1)
    codes, _ = pd.factorize(values)
    return np.asarray(codes, dtype=np.intp)


def _kernel_configuration(label: str, covariance: InvertibleNativeCovarianceAction) -> xr.Dataset:
    """Return an action's configuration with its amplitude removed."""
    to_dataset = getattr(covariance, "to_dataset", None)
    if not callable(to_dataset):
        raise TypeError(
            f"Correlated source {label!r} must expose to_dataset() or same_unit_kernel() so that its "
            "kernel can be compared"
        )
    dataset = to_dataset()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Correlated source {label!r} to_dataset() must return an xarray Dataset")
    configuration = dataset.drop_vars("sigma", errors="ignore")
    # drop_vars returns a new Dataset; replacing its attrs leaves the input intact.
    configuration.attrs = {key: value for key, value in dataset.attrs.items() if key != "sigma"}
    return configuration


def _connected_groups(
    labels: tuple[str, ...],
    pairs: Mapping[tuple[str, str], float],
) -> tuple[tuple[str, ...], ...]:
    """Return correlated source groups (size > 1) in canonical order."""
    parent = {label: label for label in labels}

    def find(label: str) -> str:
        """Return the representative of ``label``'s group."""
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for first, second in pairs:
        parent[find(second)] = find(first)
    members: dict[str, list[str]] = {}
    for label in labels:
        members.setdefault(find(label), []).append(label)
    return tuple(tuple(group) for group in members.values() if len(group) > 1)
