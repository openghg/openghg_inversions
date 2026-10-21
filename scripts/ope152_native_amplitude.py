"""Experimental native lowering of the shared fixed-OU amplitude calculation.

This prototype targets PyTensor 3.3.2 / Numba 0.63.1. Its native POTRS/TRSM
helpers are private dependency APIs; it is not a production backend selection.
All covariance arrays are explicit shared graph inputs, never closure payloads.
"""

# ruff: noqa: E402 -- configure the repository's numerical runtime before PyTensor.

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch

import numba
import numpy as np
import pytensor
import pytensor.tensor as pt
from numba.extending import overload, register_jitable
from pytensor.gradient import DisconnectedType, grad_not_implemented
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.link.numba.dispatch.linalg.solvers.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solvers.triangular import _solve_triangular

from openghg_inversions.inference import cached_sigma as inference
from openghg_inversions.models.fixed_ou import (
    _evaluate_mode_arrays,
    _fixed_ou_cho_solve,
    _fixed_ou_solve_triangular,
)


@overload(_fixed_ou_cho_solve)
def _native_cho_solve(cholesky, rhs):
    def solve(cholesky, rhs):
        return _cho_solve(cholesky, rhs, True, False)

    return solve


@overload(_fixed_ou_solve_triangular)
def _native_triangular_solve(cholesky, rhs):
    def solve(cholesky, rhs):
        return _solve_triangular(cholesky, rhs, 0, True, False, False)

    return solve


# Register the ordinary production function without wrapping its Python caller.
register_jitable(_evaluate_mode_arrays)


def _amplitude_evaluate(amplitude, residual, factor, eigenvalues, site_index, base_logdet):
    """Validate dynamic inputs, then call the common valid-mode calculation.

    Factor, eigenvalues and site indices are canonical prepared arrays; their
    static shapes and index bounds are established by the preparation boundary.
    """
    n_site = int(site_index.max()) + 1
    if residual.shape != eigenvalues.shape or residual.size != factor.shape[0]:
        raise ValueError("residual must have one value per prepared observation.")
    if amplitude.size != n_site:
        raise ValueError("site_amplitude must have one value per prepared site.")
    squared = np.empty_like(amplitude)
    invalid = False
    for i in range(amplitude.size):
        value = float(amplitude[i])
        squared[i] = value * value
        invalid = invalid or not np.isfinite(value) or value < 0.0 or not np.isfinite(squared[i])
    if invalid:
        gradient = np.full(amplitude.shape, -1.0, dtype=np.float64)
        for i in range(amplitude.size):
            if np.isfinite(amplitude[i]) and amplitude[i] > 1.0:
                gradient[i] = -1.0 / amplitude[i]
        return -np.inf, gradient
    if not np.isfinite(residual).all():
        return -np.inf, np.zeros(amplitude.size, dtype=np.float64)
    mode_variance = eigenvalues + squared[site_index]
    if not np.isfinite(mode_variance).all() or np.any(mode_variance <= 0.0):
        return -np.inf, np.zeros(amplitude.size, dtype=np.float64)
    logp, _, _, gradient = _evaluate_mode_arrays(
        residual, amplitude, mode_variance, factor, site_index, base_logdet,
    )
    return logp, gradient


native_amplitude_evaluate = numba.njit(
    _amplitude_evaluate, fastmath=False, parallel=False, error_model="numpy",
)


class NativeAmplitudeOp(Op):
    """Stateless six-input likelihood with a genuine nopython implementation."""

    __props__ = ()

    def make_node(self, amplitude, residual, factor, eigenvalues, site_index, base_logdet):
        values = [pt.cast(pt.as_tensor_variable(value), dtype) for value, dtype in zip(
            (amplitude, residual, factor, eigenvalues, site_index, base_logdet),
            ("float64", "float64", "float64", "float64", "int64", "float64"),
            strict=True,
        )]
        if tuple(value.ndim for value in values) != (1, 1, 2, 1, 1, 0):
            raise TypeError("Expected amplitude/residual vectors, factor matrix, mode vectors and scalar logdet.")
        return Apply(self, values, [pt.dscalar(), pt.dvector()])

    def perform(self, node, inputs, output_storage):
        del node
        with np.errstate(over="ignore", invalid="ignore"):
            logp, gradient = _amplitude_evaluate(*inputs)
        output_storage[0][0] = np.asarray(logp, dtype=np.float64)
        output_storage[1][0] = gradient

    def L_op(self, inputs, outputs, output_grads):
        if not isinstance(output_grads[1].type, DisconnectedType):
            return [grad_not_implemented(self, i, value) for i, value in enumerate(inputs)]
        return [output_grads[0] * outputs[1]] + [
            grad_not_implemented(self, i, inputs[i]) for i in range(1, len(inputs))
        ]

    def infer_shape(self, *args):
        return [(), args[-1][0]]


@numba_funcify.register(NativeAmplitudeOp)
def _numba_funcify_native_amplitude(op, **kwargs):
    del op, kwargs

    @numba.njit(fastmath=False, parallel=False, error_model="numpy")
    def evaluate(amplitude, residual, factor, eigenvalues, site_index, base_logdet):
        logp, gradient = native_amplitude_evaluate(
            amplitude, residual, factor, eigenvalues, site_index, base_logdet,
        )
        return np.asarray(logp), gradient

    return evaluate


class NativeAmplitudeAdapter(inference._PytensorSigmaLikelihoodOp):
    """Keep residual installation outside the stateless compiled Op."""

    def __init__(self, target):
        super().__init__(target)
        prepared = target.prepared
        self.native_op = NativeAmplitudeOp()
        # Shared values are copied on construction and stay explicit function inputs.
        self.array_inputs = tuple(pytensor.shared(np.asarray(value), name=name) for value, name in (
            (prepared.transformed_factor, "native_ou_factor"),
            (prepared.mode_eigenvalues, "native_ou_eigenvalues"),
            (prepared.mode_site_index, "native_ou_site_index"),
            (prepared.base_logdet, "native_ou_base_logdet"),
        ))

    def __call__(self, amplitude, residual):
        return self.native_op(amplitude, residual, *self.array_inputs)


@contextmanager
def native_amplitude_context():
    """Temporarily build only the private amplitude NUTS graph with native NUMBA.

    This construction-time context patches Python classes and must not be used
    concurrently in threads. Constructed chains own their residual and arrays;
    no patch remains necessary while evaluating, sampling, copying or pickling.
    """
    original_init = inference.PymcCachedSigmaNutsStep.__init__
    original_nuts = inference.pm.NUTS

    def native_nuts(*args, **kwargs):
        compile_kwargs = dict(kwargs.get("compile_kwargs", {}))
        compile_kwargs["mode"] = "NUMBA"
        kwargs["compile_kwargs"] = compile_kwargs
        return original_nuts(*args, **kwargs)

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        with (
            pytensor.config.change_flags(numba__fastmath=False),
            patch.object(inference, "_PytensorSigmaLikelihoodOp", NativeAmplitudeAdapter),
            patch.object(inference.pm, "NUTS", native_nuts),
        ):
            original_init(self, *args, **kwargs)

    with patch.object(inference.PymcCachedSigmaNutsStep, "__init__", initialize):
        yield


def self_check():
    """Bounded compile and parity check; real-data timings belong on Slurm."""
    inputs = [pt.dvector(), pt.dvector(), pt.dmatrix(), pt.dvector(), pt.lvector(), pt.dscalar()]
    outputs = NativeAmplitudeOp()(*inputs)
    with pytensor.config.change_flags(numba__fastmath=False):
        compiled = pytensor.function(inputs, outputs, mode="NUMBA")
    assert not compiled.vm.jit_fn.targetoptions["fastmath"]
    args = (
        np.array([0.7, 1.2]), np.array([0.2, -0.1, 0.4, 0.3]),
        np.arange(8, dtype=np.float64).reshape(4, 2) * 0.03,
        np.array([0.2, 0.3, 0.1, 0.4]), np.array([0, 1, 0, 1], dtype=np.int64), 0.5,
    )
    expected = _amplitude_evaluate(*args)
    actual = compiled(*args)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)
    assert native_amplitude_evaluate.nopython_signatures
    assert not any(item.objectmode for item in native_amplitude_evaluate.overloads.values())
    print("Native amplitude compile and shared-kernel parity passed.")


if __name__ == "__main__":
    self_check()
