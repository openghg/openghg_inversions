"""Correlated Gaussian likelihoods for experimental RJMCMC inversions.

This module implements an independent-site Ornstein--Uhlenbeck (OU) model for
model--measurement mismatch with an independent observation-error nugget.  A
unit-variance latent OU process is maintained for each site; an observation's
mismatch-group amplitude scales that latent process, so changing amplitude
groups within a site does not reset its temporal correlation.  Sites may share
correlation-timescale parameters.

The frozen :class:`IndependentSiteOUData` container validates, copies, and
marks read-only all static inputs, then precomputes an O(N) stable site/time
traversal.  :func:`ou_log_likelihood_numpy` and
:func:`ou_log_likelihood_numba` evaluate the fully normalized Gaussian density
in linear time; the scalar Kalman recursion itself uses O(1) filter state.

For same-site observations ``i`` and ``j``, the correlated covariance is
``mismatch_sd[g_i] * mismatch_sd[g_j] * exp(-abs(t_i-t_j) / tau[site_i])``.
The diagonal additionally contains ``observation_sd[i]**2``, and different
sites are independent.  Residuals and both standard deviations share units;
times and correlation timescales share another arbitrary unit.  The first call
to the Numba entry point compiles a specialization and may populate Numba's
on-disk function cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, expm1, log, pi

from numba import njit
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _readonly_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite, one-dimensional, read-only float64 copy.

    Args:
        values: Values coercible to a NumPy float64 array.
        name: Field name used in validation errors.

    Returns:
        Owned one-dimensional float64 array marked read-only.

    Raises:
        ValueError: If the result is not one-dimensional or contains a
            nonfinite value.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_index_vector(values: ArrayLike, *, name: str) -> IntArray:
    """Return a one-dimensional, read-only vector of exact integers.

    Values are first converted to float64 so integral floating-point inputs are
    accepted without silently truncating fractional indices.

    Args:
        values: Numeric index values.
        name: Field name used in validation errors.

    Returns:
        Owned one-dimensional int64 array marked read-only.

    Raises:
        ValueError: If values are not a finite one-dimensional vector of exact
            integers representable under the supported int64 conversion.
    """
    numeric = np.array(values, dtype=np.float64, copy=True)
    if numeric.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain only integer values.")
    int64_info = np.iinfo(np.int64)
    if np.any(numeric < int64_info.min) or np.any(numeric >= float(int64_info.max)):
        raise ValueError(f"{name} values must fit in int64.")
    array = numeric.astype(np.int64)
    array.setflags(write=False)
    return array


def _number_of_contiguous_indices(values: IntArray, *, name: str) -> int:
    """Validate zero-based contiguous indices and return their cardinality.

    Args:
        values: Nonempty integer index vector.
        name: Field name used in validation errors.

    Returns:
        Number of distinct contiguous indices.

    Raises:
        ValueError: If indices are negative, omit an integer before their
            maximum, or cannot be a contiguous map for the vector length.
    """
    if values.size == 0 or np.any(values < 0):
        raise ValueError(f"{name} must contain zero-based non-negative indices.")
    unique = np.unique(values)
    if unique[-1] >= values.size:
        raise ValueError(f"{name} must use contiguous indices starting at zero.")
    expected = np.arange(int(unique[-1]) + 1, dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError(f"{name} must use contiguous indices starting at zero.")
    return int(expected.size)


@dataclass(frozen=True)
class IndependentSiteOUData:
    """Frozen static data for an independent-site irregular-time OU model.

    Args:
        observation_sd: Standard deviation of the independent observation
            nugget for each observation. Values must be finite and positive.
        observation_time: Numeric time for each observation. Times may be
            unsorted and observations from different sites may be interleaved.
            Within each site, times must be unique and hence strictly increasing
            after sorting.
        site_index: Zero-based contiguous site index for each observation.
        mismatch_group_index: Zero-based contiguous mismatch-amplitude group
            for each observation. Groups may change within a site's time series.
        site_tau_index: Zero-based contiguous timescale-parameter index for each
            site. Multiple sites may refer to the same parameter.

        Input vectors are copied and exposed as arrays marked read-only.

    Attributes:
        observation_order: Stable permutation that groups observations by site
            and orders each site by time.
        ordered_time_delta: Time since the preceding same-site observation;
            zero for the first observation at each site.
        ordered_site_start: Whether an ordered observation starts a new site's
            independent latent process.
        ordered_observation_variance: Nugget variance in traversal order.
        ordered_mismatch_group_index: Amplitude-group indices in traversal order.
        ordered_tau_index: Timescale-parameter indices in traversal order.
        n_sites: Number of independent site processes.
        n_mismatch_groups: Number of mismatch-amplitude parameters required.
        n_tau_parameters: Number of timescale parameters required.

    Raises:
        ValueError: If vectors have inconsistent shapes, contain invalid values,
            use malformed index maps, or repeat an observation time within a
            site.
    """

    observation_sd: FloatArray
    observation_time: FloatArray
    site_index: IntArray
    mismatch_group_index: IntArray
    site_tau_index: IntArray
    observation_order: IntArray = field(init=False)
    ordered_time_delta: FloatArray = field(init=False)
    ordered_site_start: BoolArray = field(init=False)
    ordered_observation_variance: FloatArray = field(init=False)
    ordered_mismatch_group_index: IntArray = field(init=False)
    ordered_tau_index: IntArray = field(init=False)
    n_sites: int = field(init=False)
    n_mismatch_groups: int = field(init=False)
    n_tau_parameters: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate inputs and precompute the immutable site/time traversal."""
        observation_sd = _readonly_float_vector(self.observation_sd, name="observation_sd")
        observation_time = _readonly_float_vector(
            self.observation_time,
            name="observation_time",
        )
        site_index = _readonly_index_vector(self.site_index, name="site_index")
        mismatch_group_index = _readonly_index_vector(
            self.mismatch_group_index,
            name="mismatch_group_index",
        )
        site_tau_index = _readonly_index_vector(self.site_tau_index, name="site_tau_index")

        n_observations = observation_sd.size
        if n_observations == 0:
            raise ValueError("at least one observation is required.")
        expected_shape = (n_observations,)
        if observation_time.shape != expected_shape:
            raise ValueError("observation_time must have one value per observation.")
        if site_index.shape != expected_shape:
            raise ValueError("site_index must have one value per observation.")
        if mismatch_group_index.shape != expected_shape:
            raise ValueError("mismatch_group_index must have one value per observation.")
        if np.any(observation_sd <= 0.0):
            raise ValueError("observation_sd must be strictly positive.")

        n_sites = _number_of_contiguous_indices(site_index, name="site_index")
        n_mismatch_groups = _number_of_contiguous_indices(
            mismatch_group_index,
            name="mismatch_group_index",
        )
        if site_tau_index.shape != (n_sites,):
            raise ValueError("site_tau_index must contain one value per site.")
        n_tau_parameters = _number_of_contiguous_indices(
            site_tau_index,
            name="site_tau_index",
        )

        original_position = np.arange(n_observations, dtype=np.int64)
        order = np.lexsort((original_position, observation_time, site_index)).astype(
            np.int64,
            copy=False,
        )
        ordered_sites = site_index[order]
        ordered_times = observation_time[order]
        site_start = np.empty(n_observations, dtype=np.bool_)
        site_start[0] = True
        site_start[1:] = ordered_sites[1:] != ordered_sites[:-1]
        time_delta = np.zeros(n_observations, dtype=np.float64)
        time_delta[1:] = ordered_times[1:] - ordered_times[:-1]
        if np.any(time_delta[~site_start] <= 0.0):
            raise ValueError("observation_time must be strictly increasing within each site.")
        time_delta[site_start] = 0.0

        observation_variance = np.square(observation_sd[order])
        ordered_groups = mismatch_group_index[order]
        ordered_tau = site_tau_index[ordered_sites]
        for array in (
            order,
            site_start,
            time_delta,
            observation_variance,
            ordered_groups,
            ordered_tau,
        ):
            array.setflags(write=False)

        object.__setattr__(self, "observation_sd", observation_sd)
        object.__setattr__(self, "observation_time", observation_time)
        object.__setattr__(self, "site_index", site_index)
        object.__setattr__(self, "mismatch_group_index", mismatch_group_index)
        object.__setattr__(self, "site_tau_index", site_tau_index)
        object.__setattr__(self, "observation_order", order)
        object.__setattr__(self, "ordered_time_delta", time_delta)
        object.__setattr__(self, "ordered_site_start", site_start)
        object.__setattr__(self, "ordered_observation_variance", observation_variance)
        object.__setattr__(self, "ordered_mismatch_group_index", ordered_groups)
        object.__setattr__(self, "ordered_tau_index", ordered_tau)
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "n_mismatch_groups", n_mismatch_groups)
        object.__setattr__(self, "n_tau_parameters", n_tau_parameters)

    @property
    def n_observations(self) -> int:
        """Number of observations represented by the static data."""
        return int(self.observation_sd.size)


def _validated_runtime_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
) -> FloatArray:
    """Return one validated positive runtime parameter vector.

    Args:
        values: Runtime standard deviations or timescales.
        name: Parameter name used in validation errors.
        expected_size: Required vector length.

    Returns:
        Owned one-dimensional float64 parameter vector.

    Raises:
        ValueError: If shape, finiteness, or strict positivity is invalid.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},).")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite, strictly positive values.")
    return array


def _validated_likelihood_inputs(
    residual: ArrayLike,
    data: IndependentSiteOUData,
    mismatch_sd: ArrayLike,
    correlation_timescale: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate dynamic likelihood inputs without mutating caller arrays.

    Args:
        residual: Residual vector in original observation order.
        data: Static OU likelihood data.
        mismatch_sd: Per-group mismatch amplitudes.
        correlation_timescale: OU timescale parameters.

    Returns:
        Owned float64 copies of residuals, amplitudes, and timescales.

    Raises:
        TypeError: If ``data`` has the wrong type.
        ValueError: If dynamic arrays violate required shapes or supports.
    """
    if not isinstance(data, IndependentSiteOUData):
        raise TypeError("data must be an IndependentSiteOUData instance.")
    residual_array = np.array(residual, dtype=np.float64, copy=True)
    if residual_array.shape != (data.n_observations,):
        raise ValueError("residual must have one value per observation.")
    if not np.all(np.isfinite(residual_array)):
        raise ValueError("residual must contain only finite values.")
    mismatch_array = _validated_runtime_vector(
        mismatch_sd,
        name="mismatch_sd",
        expected_size=data.n_mismatch_groups,
    )
    timescale_array = _validated_runtime_vector(
        correlation_timescale,
        name="correlation_timescale",
        expected_size=data.n_tau_parameters,
    )
    return residual_array, mismatch_array, timescale_array


def _ou_log_likelihood_numpy_kernel(
    residual: FloatArray,
    observation_order: IntArray,
    time_delta: FloatArray,
    site_start: BoolArray,
    observation_variance: FloatArray,
    mismatch_group_index: IntArray,
    tau_index: IntArray,
    mismatch_sd: FloatArray,
    correlation_timescale: FloatArray,
) -> float:
    """Evaluate the scalar OU Kalman filter using the Python/NumPy backend.

    Static vectors must be aligned in precomputed traversal order except for
    ``residual``, which remains in original observation order and is accessed
    through ``observation_order``. The filter starts from stationary unit latent
    variance at every true entry of ``site_start``.

    Args:
        residual: Residuals in original observation order.
        observation_order: Original indices in site/time traversal order.
        time_delta: Same-site time increments in traversal order.
        site_start: Flags identifying independent latent-process starts.
        observation_variance: Nugget variances in traversal order.
        mismatch_group_index: Amplitude indices in traversal order.
        tau_index: Timescale indices in traversal order.
        mismatch_sd: Per-group mismatch amplitudes.
        correlation_timescale: OU timescale parameters.

    Returns:
        Fully normalized log likelihood accumulated over all sites.
    """
    log_likelihood = 0.0
    posterior_mean = 0.0
    posterior_variance = 1.0
    log_two_pi = log(2.0 * pi)

    for ordered_index in range(observation_order.size):
        if site_start[ordered_index]:
            predicted_mean = 0.0
            predicted_variance = 1.0
        else:
            tau = correlation_timescale[tau_index[ordered_index]]
            scaled_delta = time_delta[ordered_index] / tau
            phi = exp(-scaled_delta)
            innovation_fraction = -expm1(-2.0 * scaled_delta)
            predicted_mean = phi * posterior_mean
            predicted_variance = posterior_variance + innovation_fraction * (1.0 - posterior_variance)

        amplitude = mismatch_sd[mismatch_group_index[ordered_index]]
        nugget_variance = observation_variance[ordered_index]
        innovation_variance = amplitude * amplitude * predicted_variance + nugget_variance
        observation_index = observation_order[ordered_index]
        innovation = residual[observation_index] - amplitude * predicted_mean
        log_likelihood -= 0.5 * (
            log_two_pi + log(innovation_variance) + innovation * innovation / innovation_variance
        )

        posterior_mean = predicted_mean + (predicted_variance * amplitude / innovation_variance) * innovation
        posterior_variance = predicted_variance * nugget_variance / innovation_variance

    return log_likelihood


@njit(cache=True)
def _ou_log_likelihood_numba_kernel(
    residual: FloatArray,
    observation_order: IntArray,
    time_delta: FloatArray,
    site_start: BoolArray,
    observation_variance: FloatArray,
    mismatch_group_index: IntArray,
    tau_index: IntArray,
    mismatch_sd: FloatArray,
    correlation_timescale: FloatArray,
) -> float:
    """Evaluate the scalar OU Kalman filter using the compiled Numba backend.

    Inputs obey the same ordering and stationary-start invariants as the NumPy
    kernel. Validation is intentionally performed by the public Python wrapper.

    Args:
        residual: Residuals in original observation order.
        observation_order: Original indices in site/time traversal order.
        time_delta: Same-site time increments in traversal order.
        site_start: Flags identifying independent latent-process starts.
        observation_variance: Nugget variances in traversal order.
        mismatch_group_index: Amplitude indices in traversal order.
        tau_index: Timescale indices in traversal order.
        mismatch_sd: Per-group mismatch amplitudes.
        correlation_timescale: OU timescale parameters.

    Returns:
        Fully normalized log likelihood accumulated over all sites.
    """
    log_likelihood = 0.0
    posterior_mean = 0.0
    posterior_variance = 1.0
    log_two_pi = log(2.0 * pi)

    for ordered_index in range(observation_order.size):
        if site_start[ordered_index]:
            predicted_mean = 0.0
            predicted_variance = 1.0
        else:
            tau = correlation_timescale[tau_index[ordered_index]]
            scaled_delta = time_delta[ordered_index] / tau
            phi = exp(-scaled_delta)
            innovation_fraction = -expm1(-2.0 * scaled_delta)
            predicted_mean = phi * posterior_mean
            predicted_variance = posterior_variance + innovation_fraction * (1.0 - posterior_variance)

        amplitude = mismatch_sd[mismatch_group_index[ordered_index]]
        nugget_variance = observation_variance[ordered_index]
        innovation_variance = amplitude * amplitude * predicted_variance + nugget_variance
        observation_index = observation_order[ordered_index]
        innovation = residual[observation_index] - amplitude * predicted_mean
        log_likelihood -= 0.5 * (
            log_two_pi + log(innovation_variance) + innovation * innovation / innovation_variance
        )

        posterior_mean = predicted_mean + (predicted_variance * amplitude / innovation_variance) * innovation
        posterior_variance = predicted_variance * nugget_variance / innovation_variance

    return log_likelihood


def ou_log_likelihood_numpy(
    residual: ArrayLike,
    data: IndependentSiteOUData,
    mismatch_sd: ArrayLike,
    correlation_timescale: ArrayLike,
) -> float:
    """Return the normalized OU-plus-nugget Gaussian log likelihood.

    Args:
        residual: Observation-minus-model residuals with shape
            ``(n_observations,)`` in the original order used for ``data``.
        data: Validated static likelihood data and precomputed traversal.
        mismatch_sd: Positive latent-OU amplitudes with shape
            ``(n_mismatch_groups,)``.
        correlation_timescale: Positive OU timescales with shape
            ``(n_tau_parameters,)``. Units must match ``data.observation_time``.

    Returns:
        Fully normalized multivariate Gaussian log density, including the
        log-determinant and ``n * log(2*pi)`` terms.

    Raises:
        TypeError: If ``data`` is not :class:`IndependentSiteOUData`.
        ValueError: If a dynamic vector has the wrong shape, is nonfinite, or
            contains a non-positive standard deviation or timescale.
    """
    residual_array, mismatch_array, timescale_array = _validated_likelihood_inputs(
        residual,
        data,
        mismatch_sd,
        correlation_timescale,
    )
    return float(
        _ou_log_likelihood_numpy_kernel(
            residual_array,
            data.observation_order,
            data.ordered_time_delta,
            data.ordered_site_start,
            data.ordered_observation_variance,
            data.ordered_mismatch_group_index,
            data.ordered_tau_index,
            mismatch_array,
            timescale_array,
        )
    )


def ou_log_likelihood_numba(
    residual: ArrayLike,
    data: IndependentSiteOUData,
    mismatch_sd: ArrayLike,
    correlation_timescale: ArrayLike,
) -> float:
    """Return the normalized OU-plus-nugget log likelihood using Numba.

    This function performs the same Python-side validation as
    :func:`ou_log_likelihood_numpy`, then dispatches only the linear-time Kalman
    recursion to compiled code.

    Args:
        residual: Observation-minus-model residuals with shape
            ``(n_observations,)`` in the original order used for ``data``.
        data: Validated static likelihood data and precomputed traversal.
        mismatch_sd: Positive latent-OU amplitudes with shape
            ``(n_mismatch_groups,)``.
        correlation_timescale: Positive OU timescales with shape
            ``(n_tau_parameters,)``. Units must match ``data.observation_time``.

    Returns:
        Fully normalized multivariate Gaussian log density.

    Raises:
        TypeError: If ``data`` is not :class:`IndependentSiteOUData`.
        ValueError: If a dynamic vector has the wrong shape, is nonfinite, or
            contains a non-positive standard deviation or timescale.
    """
    residual_array, mismatch_array, timescale_array = _validated_likelihood_inputs(
        residual,
        data,
        mismatch_sd,
        correlation_timescale,
    )
    return float(
        _ou_log_likelihood_numba_kernel(
            residual_array,
            data.observation_order,
            data.ordered_time_delta,
            data.ordered_site_start,
            data.ordered_observation_variance,
            data.ordered_mismatch_group_index,
            data.ordered_tau_index,
            mismatch_array,
            timescale_array,
        )
    )
