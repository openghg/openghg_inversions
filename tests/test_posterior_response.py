"""Equation references for local, conditional-on-model posterior response."""

import numpy as np
import pytest
import xarray as xr
from dask import delayed
import dask.array as da

from openghg_inversions.posterior_response import (
    gaussian_conditional_score,
    gaussian_prior_shift_score,
    likelihood_score,
    paired_response_summary,
    response_diagnostics,
    response_summary,
)


def test_gaussian_reference_sign_and_three_estimators():
    # Exact sigma points integrate any quadratic under this joint Gaussian.
    sx, sb, r, e = 2.0, 0.7, 1.3, -3.0
    covariance = np.linalg.inv(np.diag([1 / sx, 1 / sb]) + np.ones((2, 2)) / r)
    root = np.sqrt(2) * np.linalg.cholesky(covariance)
    samples = np.concatenate([root.T, -root.T]) + [0.4, 1.2]
    q = samples[:, 0]
    direction, design, mean, prior = np.array([[e]]), np.ones((1, 1)), np.ones(1), np.array([[sb]])
    all_scores = []
    for x, beta in samples:
        args = {"covariance_depends_on_perturbation": False}

        def solve(rhs):
            return rhs / r

        all_scores.append(
            [
                likelihood_score(np.array([x + beta]), direction, solve, **args)[0],
                gaussian_conditional_score(np.array([x]), direction, design, mean, prior, solve, **args)[0],
                gaussian_prior_shift_score(
                    np.array([x + beta]),
                    direction,
                    np.array([beta]),
                    design,
                    direction,
                    mean,
                    prior,
                    solve,
                    **args,
                )[0],
            ]
        )
    scores = np.array(all_scores)
    gain = ((q - q.mean())[:, None] * (scores - scores.mean(axis=0))).mean(axis=0)
    np.testing.assert_allclose(gain, e * sx / (sx + sb + r), atol=1e-14)


def test_matrix_free_draw_dependent_covariance_and_linear_decomposition():
    residual = np.array([2.0, -1.0, 3.0])
    error = np.array([1.0, 0.0, 5.0])
    components = np.column_stack([np.full(3, error.mean()), error - error.mean()])
    directions = np.column_stack([components.sum(axis=1), components])
    scores = []
    for diagonal in (np.array([1.0, 2.0, 3.0]), np.array([5.0, 0.5, 1.0])):

        def solve(rhs):
            return rhs / (diagonal[:, None] if rhs.ndim == 2 else diagonal)

        score = likelihood_score(residual, directions, solve, covariance_depends_on_perturbation=False)
        np.testing.assert_allclose(score, directions.T @ np.linalg.solve(np.diag(diagonal), residual))
        np.testing.assert_allclose(score[0], score[1:].sum())
        scores.append(score)
    assert not np.allclose(scores[0], scores[1])


def test_conditional_matches_integrated_covariance_and_prior_mean():
    rng = np.random.default_rng(41)
    design = rng.normal(size=(5, 2))
    directions = rng.normal(size=(5, 3))
    residual = rng.normal(size=5)
    mean = np.array([1.2, -0.7])
    prior = np.array([[0.7, 0.2], [0.2, 1.1]])
    r = np.diag(np.arange(1.0, 6.0)) + 0.2
    actual = gaussian_conditional_score(
        residual,
        directions,
        design,
        mean,
        prior,
        lambda rhs: np.linalg.solve(r, rhs),
        covariance_depends_on_perturbation=False,
    )
    expected = directions.T @ np.linalg.solve(r + design @ prior @ design.T, residual + design @ mean)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_prior_shift_retains_imperfect_projection():
    design = np.array([[1.0], [2.0]])
    directions = np.array([[3.0, 1.0], [1.0, 0.0]])
    translations = np.linalg.lstsq(design, directions, rcond=None)[0]
    residual, beta, mean = np.array([2.0, -3.0]), np.array([1.2]), np.array([0.4])
    prior = np.array([[0.5]])
    score = gaussian_prior_shift_score(
        residual,
        directions,
        beta,
        design,
        translations,
        mean,
        prior,
        lambda rhs: rhs / 2,
        covariance_depends_on_perturbation=False,
    )
    residual_term = (directions - design @ translations).T @ (residual / 2)
    prior_term = -translations.T @ np.linalg.solve(prior, beta - mean)
    assert np.linalg.norm(residual_term) > 0.1
    np.testing.assert_allclose(score, residual_term + prior_term)


@pytest.mark.parametrize(
    "function", [likelihood_score, gaussian_conditional_score, gaussian_prior_shift_score]
)
def test_rejects_covariance_perturbation(function):
    # Rejection happens before attempting any numerical solve.
    if function is likelihood_score:
        args = [None] * 3
    elif function is gaussian_conditional_score:
        args = [None] * 6
    else:
        args = [None] * 8
    with pytest.raises(ValueError, match="additional score terms"):
        function(*args, covariance_depends_on_perturbation=True)


def sample_arrays():
    rng = np.random.default_rng(5)
    q = xr.DataArray(
        rng.normal(size=(4, 100, 2)),
        dims=("chain", "draw", "functional"),
        coords={"chain": [2, 4, 6, 8], "draw": np.arange(100), "functional": ["x", "z"]},
    )
    score = xr.DataArray(
        rng.normal(size=(4, 100, 3)),
        dims=("chain", "draw", "direction"),
        coords={"chain": q.chain, "draw": q.draw, "direction": ["full", "a", "b"]},
    )
    score.loc[{"direction": "full"}] = score.sel(direction="a") + score.sel(direction="b")
    return q, score


def test_lazy_summary_chunk_invariance_decomposition_and_serialization(tmp_path):
    q, score = sample_arrays()
    expected = response_summary(q, score)
    for chunks in ({"chain": 1, "draw": 17}, {"chain": 2, "draw": 50}):
        actual = response_summary(q.chunk(chunks), score.chunk(chunks))
        assert isinstance(actual.gain.data, da.Array)
        assert isinstance(actual.influence.data, da.Array)
        xr.testing.assert_allclose(actual.compute(), expected)
    xr.testing.assert_allclose(
        expected.gain.sel(direction="full", drop=True),
        expected.gain.sel(direction="a", drop=True) + expected.gain.sel(direction="b", drop=True),
    )
    xr.testing.assert_allclose(expected.chain_contribution.mean("chain"), expected.gain)
    path = tmp_path / "response.nc"
    expected.to_netcdf(path)
    with xr.open_dataset(path) as restored:
        xr.testing.assert_allclose(restored, expected)


def test_summary_does_not_execute_lazy_inputs():
    @delayed
    def fail_if_computed():
        raise AssertionError("array execution outside explicit boundary")

    values = da.from_delayed(fail_if_computed(), shape=(2, 20), dtype=float)
    array = xr.DataArray(values, dims=("chain", "draw"), coords={"chain": [0, 1], "draw": np.arange(20)})
    summary = response_summary(array, array)
    assert isinstance(summary.gain.data, da.Array)


@pytest.mark.parametrize("change", ["chain", "draw", "missing_dimension", "missing_coordinate"])
def test_requires_exact_sample_coordinates(change):
    q, score = sample_arrays()
    if change in ("chain", "draw"):
        score = score.assign_coords({change: score[change] + 1})
    elif change == "missing_dimension":
        score = score.isel(chain=0)
    else:
        score = score.drop_vars("draw")
    with pytest.raises(ValueError):
        response_summary(q, score)


def test_paired_influence_and_chain_aware_diagnostics():
    from openghg_inversions.inference.diagnostics import posterior_summary

    q, score = sample_arrays()
    # Very correlated estimators: independent error propagation would lose this cancellation.
    score_b = score + 0.01 * score.roll(draw=2, roll_coords=False)
    paired = paired_response_summary(q, score.chunk({"draw": 25}), score_b.chunk({"draw": 20}))
    first, second = response_summary(q, score), response_summary(q, score_b)
    xr.testing.assert_allclose(paired.gain.compute(), first.gain - second.gain)
    xr.testing.assert_allclose(paired.influence.compute(), first.influence - second.influence)
    diagnostics = response_diagnostics(paired)
    reference = posterior_summary(xr.Dataset({"influence": (first.influence - second.influence)}))
    xr.testing.assert_allclose(
        diagnostics.influence.sel(metric="mcse_mean"), reference.influence.sel(metric="mcse_mean"), atol=1e-12
    )
    # Folded rank diagnostics can resolve floating-point ties differently after subtraction.
    xr.testing.assert_allclose(diagnostics.influence, reference.influence, atol=1e-3)
    assert {"mcse_mean", "ess_bulk"}.issubset(set(diagnostics.metric.values))
    independent = response_diagnostics(first).influence.sel(metric="mcse_mean")
    assert (diagnostics.influence.sel(metric="mcse_mean") < independent * 0.1).all()
    with pytest.raises(ValueError, match="same dimensions"):
        paired_response_summary(q, score, score.isel(direction=0))


def test_missing_samples_propagate():
    q, score = sample_arrays()
    q[0, 0, 0] = np.nan
    result = response_summary(q.chunk({"draw": 20}), score)
    assert result.gain.sel(functional="x").isnull().all().compute()
    assert result.gain.sel(functional="z").notnull().all().compute()


@pytest.mark.parametrize("lazy", [False, True])
def test_sparse_diagnostics_match_dense_without_mutating_inputs(lazy):
    import sparse

    q, score = sample_arrays()
    dense = response_summary(q, score)
    sparse_summary = dense.copy(deep=False)
    originals = {}
    for name in ("q", "score", "influence"):
        values = sparse.COO.from_numpy(dense[name].values)
        if lazy:
            values = da.from_array(values, chunks=(1, 25) + values.shape[2:])
        sparse_summary[name] = dense[name].copy(data=values)
        originals[name] = values
    result = response_diagnostics(sparse_summary)
    xr.testing.assert_allclose(result, response_diagnostics(dense))
    for name, original in originals.items():
        assert sparse_summary[name].data is original
        values = original.compute() if lazy else original
        assert isinstance(values, sparse.COO)
        np.testing.assert_array_equal(values.todense(), dense[name].values)


@pytest.mark.parametrize("covariance", [np.array([[-1.0]]), np.zeros((1, 1)), np.eye(2)])
def test_rejects_invalid_gaussian_prior(covariance):
    with pytest.raises(ValueError, match="covariance"):
        gaussian_conditional_score(
            np.ones(2),
            np.ones((2, 1)),
            np.ones((2, 1)),
            np.ones(1),
            covariance,
            lambda rhs: rhs,
            covariance_depends_on_perturbation=False,
        )


def test_rejects_observation_shapes():
    with pytest.raises(ValueError, match="observation"):
        likelihood_score(
            np.ones((2, 1)), np.ones((2, 1)), lambda rhs: rhs, covariance_depends_on_perturbation=False
        )


def test_import_does_not_load_model_backends():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openghg_inversions.posterior_response; "
            "assert not {'pymc', 'pytensor', 'arviz'}.intersection(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
