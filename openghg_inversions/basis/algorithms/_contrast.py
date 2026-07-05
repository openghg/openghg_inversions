"""Observation-space contrast scoring for candidate basis splits.

The score in this module is a basis-construction diagnostic. It uses a fixed
design contribution array and an optional fixed design covariance; it must not
be built from observed mole fractions or residuals. With the default
``contrast_tau=None`` and identity design covariance, the score is
uncalibrated and is suitable only for ranking/debugging proposed splits.
Calibrated DFS/EIG interpretation requires a meaningful prior contrast
standard deviation and design covariance in the same row space as the
contribution array.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import xarray as xr

GridNode: TypeAlias = tuple[int, int]
GridPartition: TypeAlias = list[GridNode]


@dataclass(frozen=True)
class SplitContrastScore:
    """Diagnostics for one mass-preserving split contrast.

    Attributes:
        contrast: The contrast column ``f_ab`` in design-observation space.
        lambda_value: ``tau**2 * f_ab.T @ S^{-1} @ f_ab``.
        delta_dfs: Incremental DFS proxy ``lambda / (1 + lambda)``.
        delta_eig: Incremental EIG proxy ``0.5 * log(1 + lambda)``.
        mu_a: Prior mass in child ``A``.
        mu_b: Prior mass in child ``B``.
        tau: Prior standard deviation of the split contrast coefficient
            ``delta = alpha_A - alpha_B``.
        uncalibrated: True when the score used the default ``tau=1`` or
            identity covariance. Such scores are useful for ranking/debugging
            but not calibrated expected information gain.
    """

    contrast: npt.NDArray[np.float64]
    lambda_value: float
    delta_dfs: float
    delta_eig: float
    mu_a: float
    mu_b: float
    tau: float
    uncalibrated: bool


@dataclass(frozen=True)
class ContrastScoreSplitAcceptance:
    """Accept proposed binary splits using an observation-space contrast score.

    ``contribution`` is a design sensitivity/contribution array with at least
    one design-observation dimension and native spatial cell dimensions. It is
    combined with ``cell_weight`` to compute
    ``h_A = sum_i H_ti * mu_i`` and ``h_B = sum_i H_ti * mu_i``. If
    ``cell_weight`` is omitted, the class-local weights passed by the greedy
    splitter are used.

    ``contrast_tau`` is the prior standard deviation of the new split contrast
    coefficient, not observation noise. If omitted, ``tau=1`` is used and the
    score is marked uncalibrated. ``contrast_sigma_design`` and
    ``contrast_s_diag`` describe a fixed design covariance in the contribution
    row space; observed mole-fraction values must not be used here.

    If both thresholds are omitted, diagnostics are computable through
    :meth:`score_split` but :meth:`__call__` accepts all valid binary splits.
    """

    contribution: xr.DataArray | npt.ArrayLike
    cell_weight: xr.DataArray | npt.ArrayLike | None = None
    min_contrast_delta_eig: float | None = None
    min_contrast_lambda: float | None = None
    contrast_tau: float | None = None
    contrast_sigma_design: float | None = None
    contrast_s_diag: xr.DataArray | npt.ArrayLike | None = None
    spatial_dims: tuple[Hashable, Hashable] | None = None

    def __post_init__(self) -> None:
        """Validate static score configuration."""
        _validate_optional_non_negative("min_contrast_delta_eig", self.min_contrast_delta_eig)
        _validate_optional_non_negative("min_contrast_lambda", self.min_contrast_lambda)
        _validate_optional_non_negative("contrast_tau", self.contrast_tau)
        if self.contrast_sigma_design is not None and self.contrast_s_diag is not None:
            raise ValueError("Specify only one of contrast_sigma_design or contrast_s_diag.")
        if self.contrast_sigma_design is not None and (
            self.contrast_sigma_design <= 0.0 or not np.isfinite(self.contrast_sigma_design)
        ):
            raise ValueError("contrast_sigma_design must be positive and finite.")

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
    ) -> bool:
        """Return true when the proposed split passes the configured threshold."""
        del parent
        if len(children) != 2:
            raise ValueError("ContrastScoreSplitAcceptance requires a binary split.")

        score = self.score_split(children[0], children[1], fallback_cell_weight=weights)
        return self.accepts(score)

    def score_split(
        self,
        child_a: GridPartition,
        child_b: GridPartition,
        *,
        fallback_cell_weight: npt.ArrayLike | None = None,
    ) -> SplitContrastScore:
        """Return contrast-score diagnostics for two child partitions."""
        cell_weight = self.cell_weight if self.cell_weight is not None else fallback_cell_weight
        if cell_weight is None:
            raise ValueError("cell_weight is required when no fallback split weights are supplied.")

        return split_contrast_score(
            contribution=self.contribution,
            cell_weight=cell_weight,
            child_a=child_a,
            child_b=child_b,
            contrast_tau=self.contrast_tau,
            contrast_sigma_design=self.contrast_sigma_design,
            contrast_s_diag=self.contrast_s_diag,
            spatial_dims=self.spatial_dims,
        )

    def accepts(self, score: SplitContrastScore) -> bool:
        """Return true when ``score`` satisfies all configured thresholds."""
        if self.min_contrast_lambda is None and self.min_contrast_delta_eig is None:
            return True
        if self.min_contrast_lambda is not None and score.lambda_value < self.min_contrast_lambda:
            return False
        if self.min_contrast_delta_eig is not None and score.delta_eig < self.min_contrast_delta_eig:
            return False
        return True


def split_contrast_score(
    *,
    contribution: xr.DataArray | npt.ArrayLike,
    cell_weight: xr.DataArray | npt.ArrayLike,
    child_a: GridPartition,
    child_b: GridPartition,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | npt.ArrayLike | None = None,
    spatial_dims: tuple[Hashable, Hashable] | None = None,
) -> SplitContrastScore:
    """Compute the mass-preserving split contrast score.

    ``contribution`` must have at least one design-observation dimension plus
    the native spatial cell dimensions. ``cell_weight`` supplies the positive
    prior flux/mass weights ``mu_i``. The score is based only on design
    sensitivities/contributions and prior mass; observed mole fractions or
    residuals are not inputs.
    """
    contribution_values, weight_values, design_shape, design_dims = _design_by_grid_values(
        contribution,
        cell_weight,
        spatial_dims=spatial_dims,
    )
    mask_a = _nodes_mask(child_a, weight_values.shape, name="child_a")
    mask_b = _nodes_mask(child_b, weight_values.shape, name="child_b")
    if np.any(mask_a & mask_b):
        raise ValueError("child_a and child_b must not overlap.")

    weights_a = weight_values[mask_a]
    weights_b = weight_values[mask_b]
    if (weights_a <= 0.0).any() or (weights_b <= 0.0).any():
        raise ValueError("cell_weight must be positive for selected child cells.")

    mu_a = float(weights_a.sum())
    mu_b = float(weights_b.sum())
    mu_g = mu_a + mu_b
    if mu_a <= 0.0 or mu_b <= 0.0 or not np.isfinite(mu_g):
        raise ValueError("child partition masses must be positive and finite.")

    h_a = (contribution_values[:, mask_a] * weights_a.reshape(1, -1)).sum(axis=1)
    h_b = (contribution_values[:, mask_b] * weights_b.reshape(1, -1)).sum(axis=1)
    contrast = (mu_b / mu_g) * h_a - (mu_a / mu_g) * h_b

    tau = 1.0 if contrast_tau is None else float(contrast_tau)
    if tau < 0.0 or not np.isfinite(tau):
        raise ValueError("contrast_tau must be non-negative and finite.")

    norm = _contrast_weighted_norm(
        contrast,
        design_shape=design_shape,
        design_dims=design_dims,
        sigma_design=contrast_sigma_design,
        s_diag=contrast_s_diag,
    )
    lambda_value = float(tau**2 * norm)
    delta_dfs = float(lambda_value / (1.0 + lambda_value))
    delta_eig = float(0.5 * np.log1p(lambda_value))
    return SplitContrastScore(
        contrast=contrast.astype(np.float64),
        lambda_value=lambda_value,
        delta_dfs=delta_dfs,
        delta_eig=delta_eig,
        mu_a=mu_a,
        mu_b=mu_b,
        tau=tau,
        uncalibrated=contrast_tau is None or (contrast_sigma_design is None and contrast_s_diag is None),
    )


def contrast_tau_from_multiplier_cv(
    multiplier_cv: float,
    *,
    approximation: Literal["additive", "log"] = "additive",
) -> float:
    """Return an approximate split-contrast ``tau`` from a multiplier CV.

    ``tau`` is the prior standard deviation of ``delta = alpha_A - alpha_B``.
    The additive approximation uses ``sqrt(2) * multiplier_cv``. The log
    approximation uses ``sqrt(2) * sqrt(log1p(multiplier_cv**2))``.
    """
    if multiplier_cv < 0.0 or not np.isfinite(multiplier_cv):
        raise ValueError("multiplier_cv must be non-negative and finite.")
    if approximation == "additive":
        return float(np.sqrt(2.0) * multiplier_cv)
    if approximation == "log":
        sigma_log = np.sqrt(np.log1p(multiplier_cv**2))
        return float(np.sqrt(2.0) * sigma_log)
    raise ValueError("approximation must be 'additive' or 'log'.")


def _design_by_grid_values(
    contribution: xr.DataArray | npt.ArrayLike,
    cell_weight: xr.DataArray | npt.ArrayLike,
    *,
    spatial_dims: tuple[Hashable, Hashable] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[int, ...], tuple[Hashable, ...]]:
    """Return contribution as ``(design_obs, row, col)`` and 2D weights."""
    if isinstance(contribution, xr.DataArray):
        return _xarray_design_by_grid_values(contribution, cell_weight, spatial_dims=spatial_dims)

    weight_values = np.asarray(cell_weight, dtype=np.float64)
    if weight_values.ndim != 2:
        raise ValueError("cell_weight must be two-dimensional.")
    if not np.isfinite(weight_values).all():
        raise ValueError("cell_weight must be finite.")

    contribution_values = np.asarray(contribution, dtype=np.float64)
    if contribution_values.ndim < 3:
        raise ValueError("contrast contribution requires at least one design-observation dimension.")
    if contribution_values.shape[-2:] != weight_values.shape:
        raise ValueError("contribution trailing dimensions must match cell_weight shape.")
    if not np.isfinite(contribution_values).all():
        raise ValueError("contribution must be finite.")

    design_shape = contribution_values.shape[:-2]
    design_dims = tuple(range(len(design_shape)))
    return contribution_values.reshape((-1, *weight_values.shape)), weight_values, design_shape, design_dims


def _xarray_design_by_grid_values(
    contribution: xr.DataArray,
    cell_weight: xr.DataArray | npt.ArrayLike,
    *,
    spatial_dims: tuple[Hashable, Hashable] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[int, ...], tuple[Hashable, ...]]:
    """Return xarray contribution values as ``(design_obs, row, col)``."""
    if spatial_dims is None:
        if isinstance(cell_weight, xr.DataArray):
            spatial_dims = tuple(cell_weight.dims)  # type: ignore[assignment]
        else:
            spatial_dims = tuple(contribution.dims[-2:])  # type: ignore[assignment]
    if len(spatial_dims) != 2:
        raise ValueError("spatial_dims must contain exactly two dimensions.")
    if any(dim not in contribution.dims for dim in spatial_dims):
        raise ValueError("spatial_dims must be dimensions of contribution.")

    if isinstance(cell_weight, xr.DataArray):
        if tuple(cell_weight.dims) != spatial_dims:
            cell_weight = cell_weight.transpose(*spatial_dims)
        contribution, cell_weight = xr.align(contribution, cell_weight, join="exact")
        weight_values = np.asarray(cell_weight.to_numpy(), dtype=np.float64)
    else:
        weight_values = np.asarray(cell_weight, dtype=np.float64)

    if weight_values.ndim != 2:
        raise ValueError("cell_weight must be two-dimensional.")
    if tuple(contribution.sizes[dim] for dim in spatial_dims) != weight_values.shape:
        raise ValueError("contribution spatial dimensions must match cell_weight shape.")
    if not np.isfinite(weight_values).all():
        raise ValueError("cell_weight must be finite.")

    design_dims = tuple(dim for dim in contribution.dims if dim not in spatial_dims)
    if not design_dims:
        raise ValueError("contrast contribution requires at least one design-observation dimension.")
    contribution_values = np.asarray(
        contribution.transpose(*design_dims, *spatial_dims).to_numpy(),
        dtype=np.float64,
    )
    if not np.isfinite(contribution_values).all():
        raise ValueError("contribution must be finite.")

    design_shape = tuple(contribution.sizes[dim] for dim in design_dims)
    return contribution_values.reshape((-1, *weight_values.shape)), weight_values, design_shape, design_dims


def _contrast_weighted_norm(
    contrast: npt.NDArray[np.float64],
    *,
    design_shape: tuple[int, ...],
    design_dims: tuple[Hashable, ...],
    sigma_design: float | None,
    s_diag: xr.DataArray | npt.ArrayLike | None,
) -> float:
    """Return ``f.T @ S^-1 @ f`` for identity, scalar, or diagonal ``S``."""
    if sigma_design is not None and s_diag is not None:
        raise ValueError("Specify only one of contrast_sigma_design or contrast_s_diag.")
    if sigma_design is not None:
        if sigma_design <= 0.0 or not np.isfinite(sigma_design):
            raise ValueError("contrast_sigma_design must be positive and finite.")
        return float(np.sum(contrast**2) / sigma_design**2)
    if s_diag is None:
        return float(np.sum(contrast**2))

    variance = _s_diag_values(s_diag, design_shape=design_shape, design_dims=design_dims)
    if (variance <= 0.0).any() or not np.isfinite(variance).all():
        raise ValueError("contrast_s_diag entries must be positive finite variances.")
    return float(np.sum(contrast**2 / variance))


def _s_diag_values(
    s_diag: xr.DataArray | npt.ArrayLike,
    *,
    design_shape: tuple[int, ...],
    design_dims: tuple[Hashable, ...],
) -> npt.NDArray[np.float64]:
    """Return flattened diagonal design variances."""
    if isinstance(s_diag, xr.DataArray):
        if len(s_diag.dims) == len(design_dims) and set(s_diag.dims) == set(design_dims):
            values = np.asarray(s_diag.transpose(*design_dims).to_numpy(), dtype=np.float64)
        else:
            values = np.asarray(s_diag.to_numpy(), dtype=np.float64)
    else:
        values = np.asarray(s_diag, dtype=np.float64)
    if values.shape != design_shape:
        values = values.reshape(-1)
        if values.shape != (int(np.prod(design_shape)),):
            raise ValueError("contrast_s_diag must match the contribution design-observation shape.")
        return values
    return values.reshape(-1)


def _nodes_mask(nodes: GridPartition, shape: tuple[int, int], *, name: str) -> npt.NDArray[np.bool_]:
    """Return a boolean mask for grid nodes."""
    mask = np.zeros(shape, dtype=bool)
    for row, col in nodes:
        if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
            raise ValueError(f"{name} contains a node outside the cell_weight grid.")
        mask[row, col] = True
    if not mask.any():
        raise ValueError(f"{name} must contain at least one cell.")
    return mask


def _validate_optional_non_negative(name: str, value: float | None) -> None:
    """Validate an optional non-negative finite scalar."""
    if value is not None and (value < 0.0 or not np.isfinite(value)):
        raise ValueError(f"{name} must be non-negative and finite.")


__all__ = [
    "ContrastScoreSplitAcceptance",
    "SplitContrastScore",
    "contrast_tau_from_multiplier_cv",
    "split_contrast_score",
]
