"""Experimental native lowering of the shared fixed-OU amplitude calculation.

This prototype targets PyTensor 3.3.2 / Numba 0.63.1. Its native POTRS/TRSM
helpers are private dependency APIs; its direct SYRK pointer forbids persistent
native code caching. It is not a production backend selection.
All covariance arrays are explicit shared graph inputs, never closure payloads.
"""

# ruff: noqa: E402 -- configure the repository's numerical runtime before PyTensor.

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from contextlib import contextmanager
from functools import wraps
import ctypes
from unittest.mock import patch

import numba
import numpy as np
import pytensor
import pytensor.tensor as pt
from numba.extending import get_cython_function_address, overload, register_jitable
from pytensor.compile.mode import Mode
from pytensor.gradient import DisconnectedType, grad_not_implemented
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.linker import NumbaLinker
from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.link.numba.dispatch.linalg._LAPACK import val_to_dptr, val_to_int_ptr
from pytensor.link.numba.dispatch.linalg.solvers.cholesky import _cho_solve
from pytensor.link.numba.dispatch.linalg.solvers.triangular import _solve_triangular

from openghg_inversions.inference import cached_sigma as inference
from openghg_inversions.models.fixed_ou import (
    _evaluate_mode_arrays,
    _fixed_ou_cho_solve,
    _fixed_ou_column_norms,
    _fixed_ou_gram,
    _fixed_ou_solve_triangular,
)


# shortcut: this SciPy ABI pointer is process-local; persistent native code
# caching must stay disabled until a cache-safe BLAS binding replaces it.
_int_pointer = ctypes.POINTER(ctypes.c_int32)
_double_pointer = ctypes.POINTER(ctypes.c_double)
# The installed PyTensor BLAS wrappers pass ASCII flags through int32 pointers.
_dsyrk = ctypes.CFUNCTYPE(
    None, _int_pointer, _int_pointer, _int_pointer, _int_pointer,
    _double_pointer, _double_pointer, _int_pointer, _double_pointer,
    _double_pointer, _int_pointer,
)(get_cython_function_address("scipy.linalg.cython_blas", "dsyrk"))


@overload(_fixed_ou_gram)
def _native_gram(factor):
    def gram(factor):
        observations, rank = factor.shape
        if not observations or not rank:
            return np.zeros((rank, rank), dtype=np.float64)
        borrowed = np.ascontiguousarray(factor)
        result = np.empty((rank, rank), dtype=np.float64)
        # C-order W is Fortran-order W.T; lower Fortran output is upper C output.
        n = val_to_int_ptr(np.int32(rank))
        k = val_to_int_ptr(np.int32(observations))
        _dsyrk(val_to_int_ptr(ord("L")), val_to_int_ptr(ord("N")), n, k,
               val_to_dptr(1.0), borrowed.ctypes, n, val_to_dptr(0.0), result.ctypes, n)
        for row in range(rank):
            for column in range(row + 1, rank):
                result[column, row] = result[row, column]
        return result

    return gram


@overload(_fixed_ou_column_norms)
def _native_column_norms(value):
    def norms(value):
        result = np.empty(value.shape[1], dtype=np.float64)
        for column in range(value.shape[1]):
            total = 0.0
            for row in range(value.shape[0]):
                total += value[row, column] * value[row, column]
            result[column] = total
        return result

    return norms


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


class StrictNumbaLinker(NumbaLinker):
    """Retain exact flags and process-local pointers when PyTensor rebuilds."""

    def fgraph_convert(self, fgraph, **kwargs):
        with pytensor.config.change_flags(numba__fastmath=False, numba__cache=False):
            return super().fgraph_convert(fgraph, **kwargs)

    def jit_compile(self, fn_and_cache):
        with pytensor.config.change_flags(numba__fastmath=False, numba__cache=False):
            return super().jit_compile(fn_and_cache)


def native_amplitude_mode():
    """Return the prototype mode, including its reconstruction-time safeguards."""
    return Mode(linker=StrictNumbaLinker(), optimizer="fast_run")


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
        compile_kwargs["mode"] = native_amplitude_mode()
        kwargs["compile_kwargs"] = compile_kwargs
        return original_nuts(*args, **kwargs)

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        with (
            pytensor.config.change_flags(numba__fastmath=False, numba__cache=False),
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
    compiled = pytensor.function(inputs, outputs, mode=native_amplitude_mode())
    assert not compiled.vm.jit_fn.targetoptions["fastmath"]
    assert isinstance(compiled.vm.jit_fn._cache, numba.core.caching.NullCache)
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
