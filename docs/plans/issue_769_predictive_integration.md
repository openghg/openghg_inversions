# Issue 769: production predictive integration plan

Status: planned; prototype step 1 complete; production implementation not started.
Date: 2026-10-02.
Base: `devel` at `f6ca9aa1`; branch: `codex/gh769-predictive-integration`.

Deliver one production PR resolving [issue 769](https://github.com/openghg/openghg_inversions/issues/769).
[PR 786](https://github.com/openghg/openghg_inversions/pull/786) remains an
unmerged experiment, not a dependency. This plan incorporates its
[production review](https://github.com/openghg/openghg_inversions/pull/786#issuecomment-5954802380)
and current devel's newer CO2 staged-prior and linked prepared-input APIs.

## Step 1: completed evidence

The prototype demonstrated an exact observed fixed-OU CustomDist, joint prior
and posterior prediction, normalized scalar joint likelihoods, and efficient
cached state transitions together. Its 81 focused tests included numerical
oracles and spawned-chain checks. Warmed synthetic transitions matched the
cached baseline and performed zero state covariance factorizations.

That evidence establishes feasibility, not production integration or scaling.
Production must construct its sampling functions explicitly; it must not depend
on compiling a Potential and then deleting it from the public model. Prototype
scripts, measurement files and temporary notes stay out of the production PR.

## Step 2: construct the observed model and cached state function explicitly

Change the shared `_add_cached_likelihood` in
`openghg_inversions/rhime/co2/co2_cached_sigma_model.py` and both concrete CO2
and CO2/O2 builders. Their completed means already exist before this boundary.

- Register exactly one observed `y`, with explicit complete mean and amplitudes,
  using the existing prepared `logp`/`random` and vector support signature.
- Return the shared-cache quadratic as an ordinary, unregistered scalar
  expression through the existing recipe result. Do not add a second likelihood
  Potential or replace the density with a cache-dependent callback.
- Extend `make_cached_sigma_compound_step` with an explicit cached-expression
  input. Compile state priors and transform Jacobians plus that expression,
  replacing random variables with value variables. Retain the amplitude-prior
  constant too, so full conditional energy matches the original sampler.
- Supply the compiled function through PyMC's existing
  `pm.NUTS(logp_dlogp_func=...)` seam. `ValueGradFunction` must differentiate
  the state value variables and receive remaining free values as runtime extras.
  Use `ravel_inputs=True` and the explicit constructor initial point, and call
  `set_extra_values(constructor_point)` before handing the function to NUTS.
  Keep the existing amplitude step, sigma-then-state order and shared tensors.
- The production caller always supplies the expression. Preserve existing
  Potential-only low-level callers with an optional argument; if it is omitted
  for an observed model, raise a clear error. Never silently compile the exact
  observed likelihood for state trajectories.

Keep this local to the two recipes and existing step factory. Add no second
scientific model, sampler framework, registry or automatic invalidation system.
Prepared observations, covariance, affine designs and activity masks stay fixed;
rebuild the recipe to replace these inputs.

**Initialization boundary:** preserve scientific starts as recipe-owned sampling
inputs, not custom initial-value metadata on the final public graph. Include
initial amplitudes and lift existing boundary/offset `PriorArgs['initval']`
strategies rather than discarding or rejecting previously supported starts.
Lift non-None initial-value metadata once at recipe completion into a plain
mapping keyed by actual free-RV names, add physical amplitude defaults, and
clear only that initialization metadata before returning the model. Convert
recipe defaults and each caller mapping separately with PyMC's
`convert_str_to_rv_dict`, then merge RV-keyed mappings with caller values last.
This preserves physical/transformed override precedence. Pass unresolved
symbolic/strategy values through sampling so `prior` starts use per-chain seeds.
Resolve a separate transformed constructor point from recipe defaults using
`make_initial_point_fn`; its amplitude must match `initial_cache.sigma`.
Existing first-sweep synchronization handles distinct chain amplitudes.
Generic `compute_log_likelihood` must work without callers clearing initial values.

Gate: both observed graphs have one exact likelihood; the explicitly compiled
state density and transformed gradients match the public model with matching
cache amplitudes, and actual state transitions perform no covariance work.

## Step 3: make RhimeSampler own prediction and seed propagation

Update `openghg_inversions/rhime/sampling.py` and both cached runners.

Remove forced disabling of generic predictions and likelihood output. Remove
manual `_append_joint_outputs` generation and its request/seed helpers. Current
devel also disables generic CO2 prior prediction and merges another prior tree;
remove that duplicate route. Preserve exported `sample_co2_cached_prior_predictive`
for `co2/stages.py` and downstream callers as a thin native prediction,
coordinate-restoration and annotation wrapper. Building/prior prediction must
not require compiling a posterior sampler.

Use existing PyMC RNG utilities, with no new seed configuration object:

1. Snapshot `sample_kwargs['random_seed']` before MCMC with a copied generator.
2. Spawn fixed prior and posterior prediction streams; pass the original seed
   to MCMC unchanged. Toggling prior prediction must not shift posterior draws.
3. An explicit posterior predictive `random_seed` overrides its default stream,
   including explicit `None`. Normalize accepted non-scalar seeds with PyMC's
   utility; do not consume a caller-owned generator or forward an incompatible
   per-chain seed list to one-chain predictive sampling.

Validate requests before `pm.sample`. Cached runners retain their documented
initial contract: booleans or a sequence containing only `y`/`concentration`,
with aliases deduplicated; an empty sequence disables posterior prediction.
Predictive kwargs remain limited to `random_seed`. Reject unsupported names and
options explicitly, including options that previously went silently unused.
Generic RhimeSampler retains its broader variable/role support: resolve and
validate the effective names after kwargs precedence, including when no role
mapping is supplied. Preserve explicit likelihood disabling and prior integer
counts; `True` uses the retained posterior draw count.

Gate: both named runners use the real RhimeSampler path for seeded prior and
posterior observations, supported requests work, and invalid requests fail
before MCMC starts. Standard and custom-model sampler behavior remains covered.

## Step 4: preserve scientific outputs and staged replay

Transfer metadata from the deleted manual generator into existing CO2 and linked
annotation functions. Preserve all chains, independent prior draw lengths,
retained posterior draw labels/burn metadata, observed data, registered
observation coordinates, species/site/time labels and concentration units.

`log_likelihood.y` is one normalized scalar per `(chain, draw)`, labelled
`rhime_likelihood_scope='joint_observation_vector'` and
`rhime_normalized_log_likelihood=1`, with no concentration-unit attribute.
Both predictive `y` groups carry the existing joint predictive scope and units.
Do not fabricate pointwise densities from the correlated vector.

Preserve linked `observation_units`, `ou_species` and `ou_station` coordinates.
The cached fixed-OU linked recipe still requires common channel concentration
units; this change does not add mixed-unit covariance support. Preserve existing
input materialization and borrowed-array ownership.

Migrate staged prior tests and replay callers alongside the runner changes.
Retain NetCDF save/load metadata, saved independent-error selection and staged
output behavior. Respect explicit `idata_kwargs['log_likelihood']=False` rather
than regenerating density after sampling.

## Step 5: validate the production route and cached efficiency

Extend existing fixtures and tests; do not import prototype scripts.

| Gate | Existing test home and required additions |
| --- | --- |
| Exact observed density/random kernel | `tests/test_co2_cached_sigma.py`, `tests/test_co2_o2_fixed_ou.py`: independent dense means/covariances, arbitrary observations, linked cross-channel terms and poisoned-cache independence |
| Complete state target | Same fixtures: active/fixed/pruned flux and boundary states, per-site/channel/global offsets, all-fixed flux with active baseline, every prior and transform Jacobian; include independently evaluated positive-prior Jacobians |
| Cached execution | `tests/test_cached_sigma_sampling.py`: actual supplied state function/step, forbid kernel evaluation and Cholesky, accepted/rejected amplitude cache synchronization, distinct spawned chain starts and repeatability |
| Initialization and compatibility | Existing recipe/sampler fixtures: preserved prior starts, physical/transformed caller overrides, per-chain strategies, generic log-likelihood transformation, old Potential-only factory callers and missing-expression rejection for observed graphs |
| Real prediction routes | Both named runner smoke tests with two chains, nonzero burn and both prediction groups; supplement short MCMC with heterogeneous trace oracles; generic joint log likelihood for both families |
| Seeds and early validation | `tests/test_rhime.py` sampler cases: integer/sequence/generator seeds, input RNG immutability, overrides and prior-toggle independence; invalid effective names/options fail before sampling |
| Output/stage compatibility | Existing CO2 staged builder/contract/output/workflow and linked runner tests: unequal prior/posterior lengths, labels/units/joint metadata, save/load and prepared-input replay |

Migrate tests that expect a Potential or call the deleted manual generator.
Use numerical tolerances and distributional oracles; matching short trajectories
alone is insufficient because roundoff can change NUTS decisions.
Assert constructor amplitudes match the initial cache, public graphs retain no
custom initial-value metadata, and both sampler-generated and standalone generic
joint log likelihood work for both families.

Benchmark the old production baseline at `f6ca9aa1` against the new **production**
step factory/runners in separate processes. Include both model families, actual
available prepared-case dimensions and controlled increases of observations and
covariance rank. Record case sizes, versions and multiple seeds. Separate warm
state/compound transitions from compilation, prediction and likelihood output.
Require zero state covariance work, unchanged refresh semantics and no unresolved
material runtime or memory regression before merge. Measure
incremental warmed memory and preparation allocations alongside peak RSS; peak
RSS including imports/compilation alone is not evidence of preserved memory.

Focused commands for implementation, not checks already run on this plan:

```bash
uv run pytest tests/test_co2_cached_sigma.py tests/test_co2_o2_fixed_ou.py \
  tests/test_cached_sigma_sampling.py tests/models/test_cached_sigma.py \
  tests/models/test_fixed_ou.py tests/models/test_fixed_ou_rhime.py
uv run pytest tests/test_rhime.py -k rhime_sampler
uv run pytest tests/test_rhime_co2_configuration.py tests/test_linked_ou_configuration.py \
  tests/test_rhime_co2_o2_runner.py tests/test_co2_stage_contracts.py \
  tests/test_co2_staged_builders.py tests/test_co2_staged_outputs.py \
  tests/test_co2_staged_workflow.py
```

Run Ruff on changed Python paths and `git diff --check`. Use the repository's
Slurm runner and `slurm-wakeup` workflow for full compatibility/type checks;
never run local tox in this worktree:

```bash
sbatch scripts/slurm_tox.sh -e py312-openghgCur,py313-openghgCur,type,borrowed-types
sbatch scripts/slurm_tox.sh -e py312-pymc60 \
  tests/test_cached_sigma_sampling.py tests/test_co2_cached_sigma.py \
  tests/test_co2_o2_fixed_ou.py tests/models/test_shape_hooks.py
```

The PyMC 6.0 lower-bound gate is required: the explicit compiled-function and
initialization design has been inspected against installed PyMC 6.3.2 only.

## Step 6: document and deliver one mergeable PR

Update cached-model/runner docstrings, relevant shared-sampler documentation,
`docs/usage/co2_models.rst` and affected model-family/staged usage text. Document
prediction names/options, seeds, dimensions/units, joint likelihood meaning,
initialization precedence and fixed-data rebuilding. Add
`newsfragments/769.feature.md`; leave CHANGELOG assembly to release tooling.

Open a new PR targeting devel after implementation and its gates are complete.
Use SSH git push, the GitHub connector and the repository PR template. The PR
should close #769 and report production-route validation/performance evidence.
Keep #786 unmerged and keep namespace work from #764 out of this change. A
separate predictive model is a fallback only if explicit cached state assembly
fails a concrete compatibility gate.
