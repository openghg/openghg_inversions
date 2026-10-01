# Issue 769: temporary predictive integration plan

This working note accompanies a step-1 prototype. It is intended to be revised
or removed after the experiment informs the implementation plan. The prototype
does not close [issue 769](https://github.com/openghg/openghg_inversions/issues/769).

## Delivery plan

1. Investigate replacing the cached-sigma Potential with an observed PyMC
   CustomDist using the existing normalized fixed-OU density and random kernel.
   Compare numerical correctness, predictions, and cached transition cost.
2. If the direct replacement cannot retain cheap cached state trajectories,
   use an explicit predictive model sharing the numerical kernel and scientific
   state definitions with the inference recipe.
3. Route matching prior and posterior observation draws through RhimeSampler;
   preserve all sample axes, seed reproducibility, names, labels, and units.
   Reject unsupported variable names and keyword options before sampling.
4. Extend dense numerical oracles for means, full covariance (including linked
   cross-channel terms), normalized joint density, per-draw parameters, and
   independence from the final mutable cache state.
5. Verify one likelihood contribution, unchanged inference, and transition
   runtime/memory; measure prediction costs separately.
6. Update named-runner/shared-sampler documentation and release notes, run
   focused and relevant broader checks, and deliver independently of #764.

## Step-1 experiment

Keep production builders, runners, and RhimeSampler unchanged. A standalone
script builds the real cached CO2 recipe on a deterministic labelled fixture,
removes its sole cached Potential, and adds observed `y` with
`FixedOuLowRank.logp` and `random`. The complete modelled concentration and
site amplitudes are explicit distribution parameters. The retained
sigma-then-state CompoundStep allows comparison with the original graph.

The script compares state density/gradient evaluation costs and complete
compound transitions in separate processes, including covariance evaluation,
cache refresh counts, and process peak RSS. Tests cover normalized likelihood,
gradients, prior/posterior observations, reproducible seeds, and stale-cache
independence. Existing linked CO2/O2 fixtures provide cross-channel evidence.

A CustomDist that reads only the last quadratic cache is not acceptable: its
density must honor arbitrary supplied observation values and amplitudes. An
observed distribution added alongside the Potential is also not acceptable
because it counts the observation likelihood twice.

## Results and next decision

The direct observed replacement passes the scientific checks: normalized
density and physical/transformed gradients, one likelihood, alternate supplied
observations, cache independence, seeded prior and multi-chain posterior
observations, and linked cross-channel covariance. A paired two-chain compound
sampler smoke test agrees with the cached graph's state, amplitude, and mean
draws within `1e-6`; this is an execution/parity check, not convergence evidence.

However, the direct replacement **fails the cached-efficiency gate**. Each
state log-density/gradient evaluation calls the exact covariance kernel again.
Every state trajectory now performs rank-space factorizations; unchanged
`cache_refreshes` counts alone do not reveal this regression.

Measurements on a synthetic 64-observation, 8-state, rank-16 fixture:

| Measurement | Cached Potential | Observed CustomDist |
| --- | ---: | ---: |
| State logp + gradient, seed 769 | 10.9 microseconds | 247.2 microseconds |
| State logp + gradient, seed 770 | 10.4 microseconds | 262.5 microseconds |
| Cholesky calls per state evaluation | 0 | 1 |
| Cholesky calls in probed state transition, seed 769 | 0 | 32 |
| Cholesky calls in probed state transition, seed 770 | 0 | 16 |
| Warm compound transition median, seed 769 | 3.11 ms | 11.28 ms |
| Warm compound transition median, seed 770 | 3.45 ms | 7.55 ms |
| Retained cache refreshes, both seeds | 59 | 59 |
| Inference process peak RSS, seed 769 | 351.0 MiB | 374.6 MiB |
| Inference process peak RSS, seed 770 | 351.9 MiB | 368.2 MiB |

The means and covariance parameters of every predictive draw come from that
draw's parameters. The benchmark yields prior observation shape `(1, 4, 64)`
and posterior observation shape `(2, 30, 64)`; the prior's native singleton
chain is retained, and both posterior chains are preserved.

Raw results are in `issue_769_customdist_measurements.jsonl`. Runs used Python
3.13.7, PyMC 6.3.2, PyTensor 3.3.2, NumPy 2.2.6 and SciPy 1.15.3, with one
OpenBLAS/OpenMP thread. Each graph ran in a separate process, with two MCMC
chains, 30 tuning and 30 retained draws per chain. Density/gradient timings
exclude compilation and use the median of five batches of 100 evaluations.
Compound timings use 20 warmed, post-tuning sweeps without instrumentation.
Counts are collected separately. Process peak RSS includes imports, graph
preparation, compilation and inference instrumentation, and is recorded before
predictive generation. It is not a transition-allocation measurement.

Reproduce the pair with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  scripts/prototype_cached_sigma_customdist.py \
  --observations 64 --states 8 --rank 16 \
  --draws 30 --tune 30 --evaluations 100 --transitions 20 --seed 769
```

Repeat with `--seed 770`. Use a writable PyTensor cache on the cluster, following
the repository's existing `PYTENSOR_FLAGS` guidance.

These small synthetic measurements establish the repeated-covariance-work
mechanism. They do not establish production scaling, posterior convergence,
or a general memory-performance ratio.

Validation on the current devel base: 78 tests passed (nine new prototype
checks plus the cached target/sampler, CO2, linked fixed-OU, and fixed-OU kernel
suites). Changed-path Ruff and `git diff --check` passed. The PR's existing CI
provides Python 3.12 and 3.13 coverage; no local tox environment is required.

The next planning revision should therefore prioritize the separate predictive
model from step 2, sharing the prepared covariance and scientific mean/prior
construction. Preserve the cached inference graph and its ordered compound
steps; route prior and posterior prediction through the shared sampler with
explicit model inputs. Complete prediction names/options, seed handling,
units/coordinates, output metadata and broad performance validation remain
follow-up work. The prototype does not enable production cached-runner prior
observations or resolve the complete issue.
