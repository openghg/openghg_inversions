# Issue 769: temporary predictive integration plan

This working note accompanies a step-1 prototype. Revise or remove it after
the experiment informs production implementation. The prototype does not
close [issue 769](https://github.com/openghg/openghg_inversions/issues/769).

## Delivery plan

1. Expose an observed PyMC CustomDist using the existing normalized fixed-OU
   density and random kernel, while retaining the cached quadratic in state
   transitions. Compare correctness, predictions and complete transition cost.
2. Make construction of the cached sampling functions and the public observed
   graph explicit in the concrete model recipes and step factory. Preserve the
   sigma-then-state ordering and shared-cache ownership across spawned chains.
   Keep production model assembly readable; do not require callers to perform
   the prototype's post-compilation graph mutation.
3. Route prior and posterior observation draws through RhimeSampler; preserve
   sample axes, seed reproducibility, names, labels and units. Reject unsupported
   variable names and keyword options before sampling.
4. Expose normalized joint log likelihoods with explicit joint metadata. Preserve
   scientific initialization through sampling inputs so PyMC's generic
   log-likelihood transformation works. Never present the single correlated
   observation vector as independent pointwise likelihoods.
5. Extend dense numerical oracles for means, full covariance (including linked
   cross-channel terms), per-draw parameters and stale-cache independence. Check
   one likelihood contribution, inference parity and transition runtime/memory;
   measure prediction costs separately on representative cases.
6. Update named-runner/shared-sampler documentation and release notes, run
   relevant broader checks and deliver independently of #764.

## Step-1 experiment

Production builders, runners and RhimeSampler remain unchanged. The standalone
script compares three paths built from the real cached CO2 recipe:

- `cached`: the original Potential and cached compound sampler.
- `observed`: replace the Potential with an exact observed CustomDist, then
  compile the sampler. This control rebuilds covariance in state trajectories.
- `observed-cached`: compile the existing compound sampler first, then replace
  the public model's sole Potential with the same exact observed CustomDist.
  Pass the already compiled step explicitly to `pm.sample`.

The observed distribution's parameters are the complete modelled concentration
and site amplitudes. Its `FixedOuLowRank.logp` and `random` callbacks honor
arbitrary observations and amplitudes without reading the sampler's cache.
It replaces the Potential, so the public model has one observation likelihood.

The state NUTS function compiled before replacement retains the original
normalized cached quadratic, priors and transform Jacobians. PyMC stores that
compiled function inside the step; sampling with the supplied CompoundStep
does not rebuild it from the changed public model. The amplitude step still
ensures the shared coefficients match the amplitude before each state
transition. For matching amplitudes, the cached expression and exact observed
density are the same Gaussian likelihood. The prototype preserves the sampling strategy
while newly compiled public functions support prediction and exact densities.

On PyMC 6.3.2, `compute_log_likelihood` rejects non-default graph initial values
during its value-transform removal. The helper clears graph initial values
after compiling the step. Sampling receives the recipe's captured initial point
explicitly through `initvals`; distinct per-chain initial points also work.

Construction order is deliberate and prototype-only. Call the helper on a fresh
recipe, capture scientific initial values first, and pass its returned step to
sampling. Observation and design data must stay fixed after construction; their
cached coefficients will not follow arbitrary later `pm.Data` updates. A state
step newly constructed against the final observed graph takes the expensive
direct path. Production integration must own these guarantees explicitly.

## Results and next decision

The updated path retains zero covariance evaluations and zero Cholesky calls
inside state trajectories. Dense-oracle checks pass for normalized densities,
physical and transformed gradients, alternate observations, single likelihood,
seeded prior/posterior predictions and linked CO2/O2 cross-channel covariance.
The actual saved state function agrees with the exact public model at multiple
states and amplitudes. Its state step runs with covariance evaluation and
factorization patched to raise an error.

Three-mode, two-chain smoke draws agree within `1e-6`. Repeated two-core spawned
sampling with distinct chain initial points is reproducible and agrees with the
original cached sampler. These are execution checks, not convergence evidence;
very small floating-point differences can eventually change NUTS decisions.

Generic `pm.compute_log_likelihood` now works on the CO2 fixture. After poisoning the
sampler's cache, its per-draw result still matches a dense Gaussian oracle. The
output is one normalized joint density per `(chain, draw)`, not a per-observation
array. The benchmark explicitly labels it `joint_observation_vector`.

Measurements on a synthetic 64-observation, 8-state, rank-16 fixture:

| Measurement | Cached Potential | Direct observed | Observed with cached states |
| --- | ---: | ---: | ---: |
| State logp + gradient, seed 769 | 10.7 microseconds | 251.8 microseconds | 10.6 microseconds |
| State logp + gradient, seed 770 | 11.8 microseconds | 257.7 microseconds | 11.3 microseconds |
| Cholesky calls per state evaluation | 0 | 1 | 0 |
| Cholesky calls in probed state transition, seed 769 | 0 | 32 | 0 |
| Cholesky calls in probed state transition, seed 770 | 0 | 16 | 0 |
| Warm compound transition median, seed 769 | 3.21 ms | 11.38 ms | 3.19 ms |
| Warm compound transition median, seed 770 | 3.48 ms | 7.54 ms | 3.43 ms |
| Retained cache refreshes, both seeds | 59 | 59 | 59 |

Prior observation shape is `(1, 4, 64)` and posterior observation shape is
`(2, 30, 64)`; both posterior chains are preserved. Each draw uses its own
mean and amplitudes. Joint log likelihood shape is `(2, 30)`.

Raw results are in `issue_769_customdist_measurements.jsonl`. Runs used Python
3.13.7, PyMC 6.3.2, PyTensor 3.3.2, NumPy 2.2.6 and SciPy 1.15.3, with one
OpenBLAS/OpenMP thread. Each variant ran in a separate process, with two MCMC
chains, 30 tuning and 30 retained draws per chain. Density/gradient timings use
the median of five batches of 100 evaluations of the state expression captured
at sampler construction. Compound timings use 20 warmed post-tuning sweeps.
Both exclude compilation and instrumentation. Counts probe the actual saved
state step separately. Process peak RSS is recorded before prediction, includes
imports/compilation/setup and is not a transition-allocation measurement.

Reproduce all three variants with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  scripts/prototype_cached_sigma_customdist.py \
  --observations 64 --states 8 --rank 16 \
  --draws 30 --tune 30 --evaluations 100 --transitions 20 --seed 769
```

Repeat with `--seed 770`. Use a writable PyTensor cache on the cluster, following
the repository's existing `PYTENSOR_FLAGS` guidance. These small synthetic runs
demonstrate retained cached trajectories, not production scaling, convergence
or a general memory-performance ratio.

Validation: 81 tests passed (12 prototype cases plus the existing cached
target/sampler, CO2, linked fixed-OU and fixed-OU kernel suites). Changed-path
Ruff and `git diff --check` pass. The PR's CI provides Python 3.12 and 3.13 coverage.

The next planning revision should carry forward the observed model with cached
conditional sampling functions. A separate predictive model remains an option
if production assembly proves awkward, but the observed model does not require
sacrificing cached inference. Shared-sampler routing, names/options,
units/coordinates and representative performance validation remain outstanding.
