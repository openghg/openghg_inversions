"""Bounded numerical and graph checks for the experimental native amplitude Op."""

from copy import deepcopy
import math
import multiprocessing
import pickle
import warnings

import numba
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.mode import Mode
from pymc.blocking import DictToArrayBijection

from openghg_inversions.inference.cached_sigma import PymcCachedSigmaNutsStep, PytensorMarginalQuadraticCache
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import (
    _fixed_ou_column_norms, _fixed_ou_gram, prepare_fixed_ou_low_rank,
)
from scripts.ope152_native_amplitude import (
    NativeAmplitudeAdapter, NativeAmplitudeOp, native_amplitude_context, native_amplitude_evaluate,
    native_amplitude_mode,
)


def _case(width=2):
    rng = np.random.default_rng(20261021)
    factor = rng.normal(scale=0.15, size=(5, 2 if width == "duplicate" else width))
    if width == "duplicate":
        factor[:, 1] = factor[:, 0]
    diagonal = np.array([0.12, 0.18, 0.16, 0.20, 0.11])
    times = np.array([0.0, 0.0, 1.0, 2.0, 4.0])
    sites = np.array([0, 1, 0, 1, 0])
    tau = np.array([5.0, 7.0])
    prepared = prepare_fixed_ou_low_rank(factor, diagonal, times, sites, tau)
    residual = np.array([0.3, -0.2, 0.7, 0.1, -0.4])
    correlations = []
    for site in range(2):
        same_site = (sites[:, None] == site) & (sites[None, :] == site)
        correlations.append(same_site * np.exp(-np.abs(times[:, None] - times[None, :]) / tau[site]))
    return prepared, residual, factor @ factor.T + np.diag(diagonal), correlations


def _arguments(prepared, amplitude, residual_modes):
    return (
        np.asarray(amplitude, dtype=np.float64),
        np.asarray(residual_modes, dtype=np.float64),
        prepared.transformed_factor,
        prepared.mode_eigenvalues,
        prepared.mode_site_index,
        np.float64(prepared.base_logdet),
    )


def _dense_oracle(residual, amplitude, base, correlations):
    covariance = base + sum(value**2 * correlation for value, correlation in zip(amplitude, correlations))
    inverse = np.linalg.inv(covariance)
    solved = inverse @ residual
    logp = -0.5 * (
        residual.size * math.log(2.0 * math.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ solved
    )
    gradient = np.array([
        value * (solved @ correlation @ solved - np.trace(inverse @ correlation))
        for value, correlation in zip(amplitude, correlations)
    ])
    return logp, -solved, gradient


def _target():
    prepared, residual, *_ = _case()
    return FixedOuCachedSigmaTarget(
        prepared=prepared, observations=residual, fixed_contribution=np.zeros(5),
        design=np.arange(5.0)[:, None] * 0.1,
    )


def _compile_adapter(adapter):
    amplitude = pt.dvector("amplitude")
    return pytensor.function([amplitude], adapter(amplitude, adapter.residual), mode=native_amplitude_mode())


def _evaluate_spawned(adapter, compiled, amplitude, connection):
    try:
        rebuilt = _compile_adapter(adapter)
        args = _arguments(adapter.target.prepared, amplitude, adapter.residual.get_value())
        results = []
        for function, arguments in ((rebuilt, (amplitude,)), (compiled, args)):
            values = function(*arguments)
            dispatcher = function.vm.jit_fn
            results.append((values, dispatcher.targetoptions.get("fastmath", False), type(dispatcher._cache).__name__))
        connection.send(results)
    finally:
        connection.close()


@pytest.fixture(scope="module")
def compiled_array_helpers():
    # Calls through wrappers exercise the overloads used by the shared kernel.
    return (
        numba.njit(lambda value: _fixed_ou_column_norms(value), fastmath=False),
        numba.njit(lambda value: _fixed_ou_gram(value), fastmath=False),
    )


def _array_layout(value, layout):
    if layout == "strided":
        backing = np.zeros(tuple(2 * size for size in value.shape))
        backing[::2, ::2] = value
        return backing[::2, ::2]
    return np.array(value, order=layout)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_native_column_norms_match_scalar_oracle_without_mutation(compiled_array_helpers, layout):
    values = np.arange(-3.0, 4.0)[:, None] * np.array([0.0, 1e-200, 0.3, 1e150, -2.0])
    values = _array_layout(values, layout)
    before = values.copy()
    expected = np.array([math.fsum(float(value) ** 2 for value in column) for column in values.T])
    for function in (_fixed_ou_column_norms, compiled_array_helpers[0]):
        actual = function(values)
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
        assert actual.shape == (values.shape[1],)
        assert not np.shares_memory(actual, values)
        np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_native_gram_matches_scalar_oracle_without_mutation(compiled_array_helpers, layout):
    rng = np.random.default_rng(152)
    for width in (0, 1, 4):
        values = rng.normal(size=(17, width))
        if width == 4:
            values[:, 2] = values[:, 0] * 1e10
            values[:, 3] = values[:, 2]
        values = _array_layout(values, layout)
        before = values.copy()
        expected = np.empty((width, width))
        error_bound = np.empty_like(expected)
        for i in range(width):
            for j in range(width):
                terms = [float(a) * float(b) for a, b in zip(values[:, i], values[:, j])]
                expected[i, j] = math.fsum(terms)
                error_bound[i, j] = 17 * np.finfo(float).eps * math.fsum(abs(value) for value in terms)
        for function in (_fixed_ou_gram, compiled_array_helpers[1]):
            actual = function(values)
            assert actual.shape == expected.shape
            assert np.all(np.abs(actual - expected) <= error_bound)
            np.testing.assert_array_equal(actual, actual.T)
            np.testing.assert_array_equal(values, before)
            assert not np.shares_memory(actual, values)


def test_native_column_norms_preserve_zero_and_nonfinite_results(compiled_array_helpers):
    for values in (np.empty((0, 3)), np.empty((3, 0)), np.array([[np.nan, np.inf, 1e200]])):
        before = values.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            expected = _fixed_ou_column_norms(values)
            actual = compiled_array_helpers[0](values)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, before)


@pytest.fixture(scope="module")
def compiled_graphs():
    amplitude = pt.dvector("amplitude")
    residual = pt.dvector("residual_modes")
    factor = pt.dmatrix("factor")
    eigenvalues = pt.dvector("eigenvalues")
    sites = pt.lvector("sites")
    base_logdet = pt.dscalar("base_logdet")
    inputs = [amplitude, residual, factor, eigenvalues, sites, base_logdet]
    logp, gradient = NativeAmplitudeOp()(*inputs)
    outputs = [logp, gradient, pt.grad(logp, amplitude)]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Numba will use object mode.*")
        functions = {
            "cvm": pytensor.function(inputs, outputs, mode=Mode(linker="cvm", optimizer="fast_run")),
            "native": pytensor.function(inputs, outputs, mode=native_amplitude_mode()),
        }
        prepared, value, *_ = _case()
        functions["native"](*_arguments(prepared, [0.4, 0.7], prepared.mode_transform @ value))
    return functions


@pytest.mark.parametrize("width", [0, 1, 2, 5, "duplicate"])
@pytest.mark.parametrize("amplitude", [np.array([0.4, 0.7]), np.zeros(2)])
def test_native_and_public_evaluators_match_dense_gradients(compiled_graphs, width, amplitude):
    prepared, residual, base, correlations = _case(width)
    expected, residual_gradient, amplitude_gradient = _dense_oracle(residual, amplitude, base, correlations)
    public = prepared.evaluate(residual, amplitude)
    assert public.log_likelihood == pytest.approx(expected, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(public.gradient_residual, residual_gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(public.gradient_site_amplitude, amplitude_gradient, rtol=2e-12, atol=2e-12)
    args = _arguments(prepared, amplitude, prepared.mode_transform @ residual)
    before = [np.array(value, copy=True) for value in args]
    for function in compiled_graphs.values():
        logp, gradient, automatic_gradient = function(*args)
        assert logp == pytest.approx(expected, rel=2e-12, abs=2e-12)
        np.testing.assert_allclose(gradient, amplitude_gradient, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(automatic_gradient, gradient)
    for value, original in zip(args, before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("width,scale", [(1, 1e10), (2, 1e3)])
def test_dominant_and_duplicate_factor_valid_cases(compiled_graphs, width, scale):
    prepared = prepare_fixed_ou_low_rank(
        np.full((2, width), scale), np.ones(2), np.zeros(2), np.arange(2), 1.0,
    )
    residual = np.full(2, scale)
    amplitude = np.ones(2)
    large_eigenvalue = 2.0 + 2.0 * width * scale**2
    expected = -0.5 * (
        2 * math.log(2 * math.pi) + math.log(2) + math.log(large_eigenvalue)
        + 2 * scale**2 / large_eigenvalue
    )
    public = prepared.evaluate(residual, amplitude)
    args = _arguments(prepared, amplitude, prepared.mode_transform @ residual)
    for function in compiled_graphs.values():
        logp, gradient, automatic_gradient = function(*args)
        assert logp == pytest.approx(expected, rel=2e-10, abs=2e-10)
        np.testing.assert_allclose(gradient, public.gradient_site_amplitude, rtol=2e-9, atol=2e-9)
        np.testing.assert_array_equal(automatic_gradient, gradient)


@pytest.mark.parametrize("invalid", [-0.2, np.nan, np.inf, -np.inf, 1e200])
def test_invalid_amplitudes_preserve_rejection_gradient(compiled_graphs, invalid):
    prepared, residual, *_ = _case()
    amplitude = np.array([invalid, 0.7])
    public = prepared.evaluate(residual, amplitude)
    for function in compiled_graphs.values():
        logp, gradient, automatic_gradient = function(
            *_arguments(prepared, amplitude, prepared.mode_transform @ residual),
        )
        assert logp == public.log_likelihood == -np.inf
        np.testing.assert_array_equal(gradient, public.gradient_site_amplitude)
        np.testing.assert_array_equal(automatic_gradient, gradient)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_modes_preserve_rejection(compiled_graphs, invalid):
    prepared, residual, *_ = _case()
    modes = prepared.mode_transform @ residual
    modes[0] = invalid
    for function in compiled_graphs.values():
        logp, gradient, automatic_gradient = function(*_arguments(prepared, [0.4, 0.7], modes))
        assert logp == -np.inf
        np.testing.assert_array_equal(gradient, np.zeros(2))
        np.testing.assert_array_equal(automatic_gradient, gradient)


def test_structural_zero_mode_is_not_rescued_by_low_rank_factor(compiled_graphs):
    prepared = prepare_fixed_ou_low_rank(np.eye(2), np.zeros(2), np.arange(2.0), np.zeros(2, dtype=int), 5.0)
    for function in compiled_graphs.values():
        logp, gradient, automatic_gradient = function(*_arguments(prepared, [0.0], [0.25, -0.5]))
        assert logp == -np.inf
        np.testing.assert_array_equal(gradient, np.zeros(1))
        np.testing.assert_array_equal(automatic_gradient, gradient)


def test_native_execution_does_not_call_python_perform(compiled_graphs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Native lowering must not call Python perform.")

    monkeypatch.setattr(NativeAmplitudeOp, "perform", forbidden)
    prepared, residual, *_ = _case()
    value = compiled_graphs["native"](*_arguments(prepared, [0.4, 0.7], prepared.mode_transform @ residual))
    assert np.isfinite(value[0])
    assert native_amplitude_evaluate.nopython_signatures
    assert all(not result.objectmode for result in native_amplitude_evaluate.overloads.values())
    assert not native_amplitude_evaluate.targetoptions.get("fastmath", False)
    assert isinstance(native_amplitude_evaluate._cache, numba.core.caching.NullCache)
    graph_dispatcher = compiled_graphs["native"].vm.jit_fn
    assert graph_dispatcher.nopython_signatures
    assert all(not result.objectmode for result in graph_dispatcher.overloads.values())
    assert not graph_dispatcher.targetoptions.get("fastmath", False)
    assert isinstance(graph_dispatcher._cache, numba.core.caching.NullCache)


@pytest.mark.parametrize("clone", [deepcopy, lambda function: pickle.loads(pickle.dumps(function))])
def test_copied_native_graph_preserves_strict_compiler_settings(compiled_graphs, clone):
    prepared, residual, *_ = _case()
    args = _arguments(prepared, [0.4, 0.7], prepared.mode_transform @ residual)
    original = compiled_graphs["native"]
    with pytensor.config.change_flags(numba__fastmath=True, numba__cache=True):
        copied = clone(original)
        for actual, expected in zip(copied(*args), original(*args)):
            np.testing.assert_array_equal(actual, expected)
        assert pytensor.config.numba__fastmath
        assert pytensor.config.numba__cache
    dispatcher = copied.vm.jit_fn
    assert not dispatcher.targetoptions.get("fastmath", False)
    assert isinstance(dispatcher._cache, numba.core.caching.NullCache)
    assert dispatcher.nopython_signatures


@pytest.mark.parametrize("argument", [0, 1])
def test_wrong_dynamic_vector_size_raises(compiled_graphs, argument):
    prepared, residual, *_ = _case()
    args = list(_arguments(prepared, [0.4, 0.7], prepared.mode_transform @ residual))
    args[argument] = args[argument][:-1]
    for function in compiled_graphs.values():
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize("clone", [deepcopy, lambda op: pickle.loads(pickle.dumps(op))])
def test_cloned_adapter_keeps_residual_and_shared_arrays_independent(clone):
    target = _target()
    original = NativeAmplitudeAdapter(target)
    residual = np.array([0.3, -0.2, 0.7, 0.1, -0.4])
    original.install_residual(residual)
    expected_original = target.prepared.evaluate(residual, [0.4, 0.7])
    copied = clone(original)
    original_modes = original.residual.get_value().copy()
    changed = residual[::-1].copy()
    copied.install_residual(changed)
    np.testing.assert_array_equal(original.residual.get_value(), original_modes)
    for old, new in zip((original.residual, *original.array_inputs), (copied.residual, *copied.array_inputs)):
        assert not np.shares_memory(old.get_value(borrow=True), new.get_value(borrow=True))
    assert not hasattr(copied.native_op, "target")
    for adapter, expected in [
        (original, expected_original), (copied, target.prepared.evaluate(changed, [0.4, 0.7])),
    ]:
        logp, gradient = _compile_adapter(adapter)(np.array([0.4, 0.7]))
        assert logp == pytest.approx(expected.log_likelihood, rel=2e-12, abs=2e-12)
        np.testing.assert_allclose(gradient, expected.gradient_site_amplitude, rtol=2e-12, atol=2e-12)


def test_independent_adapters_refresh_only_their_installed_residual():
    target = _target()
    first, second = NativeAmplitudeAdapter(target), NativeAmplitudeAdapter(target)
    amplitude = np.array([0.4, 0.7])
    functions = [_compile_adapter(adapter) for adapter in (first, second)]
    residual = np.array([0.3, -0.2, 0.7, 0.1, -0.4])
    first.install_residual(residual)
    second.install_residual(-residual * 2)
    expected = target.prepared.evaluate(residual, amplitude)
    borrowed = residual.copy()
    first.install_residual(borrowed)
    borrowed[:] = 100.0
    assert functions[0](amplitude)[0] == pytest.approx(expected.log_likelihood, rel=2e-12, abs=2e-12)
    second_before = functions[1](amplitude)
    for value in (residual, residual[::-1]):
        first.install_residual(value)
        actual = functions[0](amplitude)
        reference = target.prepared.evaluate(value, amplitude)
        assert actual[0] == pytest.approx(reference.log_likelihood, rel=2e-12, abs=2e-12)
        np.testing.assert_allclose(actual[1], reference.gradient_site_amplitude, rtol=2e-12, atol=2e-12)
        for before, after in zip(second_before, functions[1](amplitude)):
            np.testing.assert_array_equal(after, before)
    first.install_residual(residual)
    assert functions[0](amplitude)[0] == pytest.approx(expected.log_likelihood, rel=2e-12, abs=2e-12)
    for one, two in zip(first.array_inputs, second.array_inputs):
        assert not np.shares_memory(one.get_value(borrow=True), two.get_value(borrow=True))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max])
def test_adapter_nonfinite_and_overflowing_residual_installation(invalid):
    adapter = NativeAmplitudeAdapter(_target())
    function = _compile_adapter(adapter)
    residual = np.array([invalid, -invalid, -invalid, invalid, invalid])
    adapter.install_residual(residual)
    logp, gradient = function(np.array([0.4, 0.7]))
    assert logp == -np.inf
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_native_conditional_gradient_in_log_coordinates(precision):
    target = _target()
    amplitude = np.array([0.4, 0.7], dtype=precision)
    original_nuts = pm.NUTS
    original_fastmath = pytensor.config.numba__fastmath
    original_cache = pytensor.config.numba__cache
    with pytensor.config.change_flags(floatX=precision):
        with pm.Model() as model:
            state = pm.Normal("state", shape=1)
            sigma = pm.HalfNormal(
                "sigma", sigma=pt.constant(0.75, dtype=precision), shape=2, initval=amplitude,
            )
            mean = target.fixed_contribution + target.design @ state
        point = model.initial_point()
        initial = np.exp(point[model.rvs_to_values[sigma].name])
        cache = target.refresh(initial)
        with native_amplitude_context():
            step = PymcCachedSigmaNutsStep(
                sigma, target=target, shared_cache=PytensorMarginalQuadraticCache(cache),
                initial_cache=cache, modelled_mean=mean, model=model, initial_point=point,
            )
        assert pm.NUTS is original_nuts
        assert pytensor.config.numba__fastmath == original_fastmath
        assert pytensor.config.numba__cache == original_cache
        step.likelihood_op.install_residual(target.observations)
        density = step.nuts_step._logp_dlogp_func
        density.set_extra_values(point)
        assert "Numba" in type(density._pytensor_function.maker.mode.linker).__name__
        for eta in [point[step.sigma_value_name], np.array([-0.7, 0.2], dtype=precision)]:
            physical = np.exp(eta).astype(np.float64)
            public = target.prepared.evaluate(target.observations, physical)
            logp, gradient = density(DictToArrayBijection.map({step.sigma_value_name: eta}))
            expected = physical * public.gradient_site_amplitude + 1.0 - physical**2 / 0.75**2
            tolerance = 2e-6 if precision == "float32" else 2e-12
            expected_logp = public.log_likelihood + np.sum(
                0.5 * math.log(2 / math.pi) - math.log(0.75) - 0.5 * (physical / 0.75)**2 + eta,
            )
            assert logp == pytest.approx(expected_logp, rel=tolerance, abs=tolerance)
            np.testing.assert_allclose(gradient, expected, rtol=tolerance, atol=tolerance)
        assert not density._pytensor_function.vm.jit_fn.targetoptions.get("fastmath", False)
        assert isinstance(density._pytensor_function.vm.jit_fn._cache, numba.core.caching.NullCache)


def test_adapter_and_compiled_graph_support_spawn(compiled_graphs):
    target = _target()
    adapter = NativeAmplitudeAdapter(target)
    adapter.install_residual(target.observations)
    amplitude = np.array([0.4, 0.7])
    expected = target.prepared.evaluate(target.observations, amplitude)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_evaluate_spawned, args=(adapter, compiled_graphs["native"], amplitude, send))
    process.start()
    send.close()
    try:
        process.join(timeout=60)
        assert process.exitcode == 0, "Spawned native graph did not finish successfully."
        assert receive.poll()
        for values, fastmath, cache_type in receive.recv():
            logp, gradient = values[:2]
            assert logp == pytest.approx(expected.log_likelihood, rel=2e-12, abs=2e-12)
            np.testing.assert_allclose(gradient, expected.gradient_site_amplitude, rtol=2e-12, atol=2e-12)
            assert not fastmath
            assert cache_type == "NullCache"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        receive.close()
