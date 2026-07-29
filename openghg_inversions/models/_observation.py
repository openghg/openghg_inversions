"""Private helpers for RHIME's transformed Johnson-SU observation model.

For enhancement ``d = value - baseline``, effective error ``e``, and
dimensionless tail scale ``sigma``, the model uses

``T(d) = asinh(sigma * d / e) / sigma``

and declares ``T(D)`` Normal with unit variance. Its transformed location is
chosen so that ``E[D] = u``. Random draws apply the matching inverse
``D = e * sinh(sigma * Z) / sigma``. The log-density includes the exact
change-of-variables Jacobian.

The elementwise parameters must be broadcastable. ``value``, ``baseline``,
``u``, and ``e`` share concentration units; ``e`` and ``sigma`` must be
strictly positive. This module has no import-time side effects. Its main entry
points are :func:`_mean_centered_johnson_su_logp`,
:func:`_mean_centered_johnson_su_random`, and
:func:`validate_johnson_su_options`.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor.variable import TensorVariable

_HALF_LOG_TWO_PI = np.asarray(0.5 * np.log(2.0 * np.pi), dtype=config.floatX)
_NEGATIVE_INFINITY = np.asarray(-np.inf, dtype=config.floatX)


def _pytensor_asinhc(value: TensorVariable) -> TensorVariable:
    """Evaluate elementwise ``asinh(value) / value`` stably.

    Args:
        value: Dimensionless PyTensor input. Array inputs are handled
            elementwise.

    Returns:
        A tensor broadcast like ``value``. A fourth-order even series is used
        when ``abs(value) < 1e-4`` so the removable singularity at zero returns
        one and retains finite gradients.
    """
    value_squared = value**2
    series = 1.0 - value_squared / 6.0 + 3.0 * value_squared**2 / 40.0
    safe_value = cast(TensorVariable, pt.where(pt.eq(value, 0.0), 1.0, value))
    arcsinh_value = cast(TensorVariable, pt.arcsinh(value))
    ratio = arcsinh_value / safe_value
    return cast(TensorVariable, pt.where(pt.lt(pt.abs(value), 1e-4), series, ratio))


def _numpy_asinhc(value: np.ndarray) -> np.ndarray:
    """Evaluate elementwise ``asinh(value) / value`` stably.

    Args:
        value: Dimensionless NumPy input, including zero-dimensional arrays.

    Returns:
        A floating array with the same shape as ``value``. A fourth-order even
        series is used when ``abs(value) < 1e-4``.
    """
    result = np.empty_like(value, dtype=float)
    small = np.abs(value) < 1e-4
    value_squared = value[small] ** 2
    result[small] = 1.0 - value_squared / 6.0 + 3.0 * value_squared**2 / 40.0
    result[~small] = np.arcsinh(value[~small]) / value[~small]
    return result


def _numpy_sinhc(value: np.ndarray) -> np.ndarray:
    """Evaluate elementwise ``sinh(value) / value`` stably.

    Args:
        value: Dimensionless NumPy input, including zero-dimensional arrays.

    Returns:
        A floating array with the same shape as ``value``. A fourth-order even
        series is used when ``abs(value) < 1e-4``.
    """
    result = np.empty_like(value, dtype=float)
    small = np.abs(value) < 1e-4
    value_squared = value[small] ** 2
    result[small] = 1.0 + value_squared / 6.0 + value_squared**2 / 120.0
    result[~small] = np.sinh(value[~small]) / value[~small]
    return result


def _johnson_su_transformed_mean(
    u: TensorVariable,
    error: TensorVariable,
    sigma: TensorVariable,
) -> TensorVariable:
    """Map a desired enhancement mean to the transformed Normal location.

    This returns
    ``m(u) = asinh((sigma * u / error) * exp(-sigma**2 / 2)) / sigma``,
    expressed with a stable ``asinhc`` form. The choice ensures
    ``E[D] = u`` after applying the inverse ``sinh`` transformation.

    Args:
        u: Desired mean enhancement in concentration units.
        error: Strictly positive effective error in the same units.
        sigma: Strictly positive dimensionless tail scale.

    Returns:
        Dimensionless transformed locations, broadcast elementwise from the
        inputs.
    """
    scaled_mean = (u / error) * pt.exp(-0.5 * sigma**2)
    return scaled_mean * _pytensor_asinhc(sigma * scaled_mean)


def _mean_centered_johnson_su_logp(
    value: TensorVariable,
    baseline: TensorVariable,
    u: TensorVariable,
    error: TensorVariable,
    sigma: TensorVariable,
) -> TensorVariable:
    """Evaluate the mean-centred transformed Johnson-SU log-density.

    Args:
        value: Observed concentration. It broadcasts elementwise with the
            distribution parameters.
        baseline: Modelled boundary-condition contribution plus any offset, in
            the same concentration units as ``value``.
        u: Modelled non-baseline enhancement in concentration units. It is the
            distribution mean after subtracting ``baseline``.
        error: Strictly positive effective error in concentration units.
        sigma: Strictly positive dimensionless transformed-tail scale.

    Returns:
        Elementwise log-density broadcast across ``value`` and the parameters.
        Invalid non-positive ``error`` or ``sigma`` values return negative
        infinity.
    """
    enhancement = value - baseline
    scaled_enhancement = enhancement / error
    transformed = scaled_enhancement * _pytensor_asinhc(sigma * scaled_enhancement)
    transformed_mean = _johnson_su_transformed_mean(u, error, sigma)
    log_scale = cast(TensorVariable, pt.log(error**2 + (sigma * enhancement) ** 2))
    log_jacobian = -0.5 * log_scale
    normal_logp = -0.5 * (transformed - transformed_mean) ** 2 - _HALF_LOG_TWO_PI
    logp = normal_logp + log_jacobian
    valid = pt.and_(pt.gt(error, 0.0), pt.gt(sigma, 0.0))
    return cast(TensorVariable, pt.where(valid, logp, _NEGATIVE_INFINITY))


def _mean_centered_johnson_su_random(
    baseline: np.ndarray | float,
    u: np.ndarray | float,
    error: np.ndarray | float,
    sigma: np.ndarray | float,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Draw from the mean-centred transformed Johnson-SU distribution.

    Args:
        baseline: Modelled boundary-condition contribution plus any offset, in
            concentration units.
        u: Modelled non-baseline enhancement in concentration units.
        error: Strictly positive effective error in concentration units.
        sigma: Strictly positive dimensionless transformed-tail scale.
        rng: NumPy random generator supplied by PyMC.
        size: Requested PyMC sample shape. Inputs broadcast elementwise, and
            PyMC supplies a size compatible with the broadcast parameter shape.

    Returns:
        Generated concentration draws with the requested sample shape and
        broadcast observation shape.

    Raises:
        ValueError: If ``error`` or ``sigma`` contains a non-finite or
            non-positive value.
    """
    rng = np.random.default_rng() if rng is None else rng
    baseline_array, u_array, error_array, sigma_array = np.broadcast_arrays(
        np.asarray(baseline, dtype=float),
        np.asarray(u, dtype=float),
        np.asarray(error, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if not np.all(np.isfinite(error_array)) or np.any(error_array <= 0.0):
        raise ValueError("Johnson-SU effective error must be finite and strictly positive.")
    if not np.all(np.isfinite(sigma_array)) or np.any(sigma_array <= 0.0):
        raise ValueError("Johnson-SU sigma must be finite and strictly positive.")

    scaled_mean = (u_array / error_array) * np.exp(-0.5 * sigma_array**2)
    transformed_mean = scaled_mean * _numpy_asinhc(sigma_array * scaled_mean)
    transformed_draw = rng.normal(loc=transformed_mean, scale=1.0, size=size)
    transformed_draw = np.asarray(transformed_draw)
    return baseline_array + error_array * transformed_draw * _numpy_sinhc(sigma_array * transformed_draw)


def validate_johnson_su_options(
    *,
    enabled: bool,
    pollution_events_from_obs: bool,
    pollution_events_from_obs_one_sided: bool,
    no_model_error: bool,
    power: dict[str, Any] | float,
) -> None:
    """Validate the supported transformed Johnson-SU configuration.

    Args:
        enabled: Whether the transformed likelihood is selected.
        pollution_events_from_obs: Whether observation-derived pollution-event
            scaling is enabled.
        pollution_events_from_obs_one_sided: Whether one-sided clipping is
            enabled.
        no_model_error: Whether explicit model error is disabled.
        power: Configured pollution-event exponent.

    Raises:
        ValueError: If the transformed likelihood is combined with an
            incompatible setting.

    Returns:
        None.
    """
    if not enabled:
        return
    if not pollution_events_from_obs:
        raise ValueError(
            "`pollution_events_from_obs_johnson_su=True` requires `pollution_events_from_obs=True`."
        )
    if pollution_events_from_obs_one_sided:
        raise ValueError(
            "`pollution_events_from_obs_johnson_su=True` is incompatible with "
            "`pollution_events_from_obs_one_sided=True`."
        )
    if no_model_error:
        raise ValueError(
            "`pollution_events_from_obs_johnson_su=True` is incompatible with `no_model_error=True`."
        )
    if isinstance(power, bool) or not isinstance(power, Real) or float(power) != 2.0:
        raise ValueError("`pollution_events_from_obs_johnson_su=True` requires fixed numeric `power=2`.")
