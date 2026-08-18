# PR 591: recovering a hackable RHIME orchestration architecture

This sidecar records the intended developer-facing consequence of PR 591,
``[OPE-45] Expose the visible run_rhime orchestration spine``. It is not a
replacement for user documentation: it captures the questions that future
documentation needs to answer with explicit, tested contracts.

## Claim

PR 591 changes RHIME from a convenient but largely monolithic runner into a
set of importable handoffs. A custom script can now reuse the supported
building blocks without copying private preparation or PyMC orchestration.

This makes the orchestration hackable; it does not yet make arbitrary
whole-model replacement low-ceremony. The built-in model stage accepts the
supported likelihood-builder seam, but a complete divergent PyMC graph still
owns a ``RhimeModelBuildResult`` containing semantic roles and declared output
capabilities. Colocating and simplifying that graph contract remains W5/W6
work.

The high-level runners remain the recommended path for ordinary inversions:

- ``run_rhime`` for standard single-sector runs;
- ``run_rhime_multisector`` for source-resolved multi-sector runs;
- ``run_rhime_from_prepared_inputs`` for replaying already prepared inputs.

The new stages are for an advanced developer who needs to inspect, time,
cache, replace a stage whose public contract fits, or deliberately reorder a
portion of the workflow.

## Visible spine

The standard default recipe is:

```text
resolve options
  -> retrieve or reload merged data
  -> filter observations and drop empty sites
  -> build or load a basis
  -> build flux and boundary-condition sensitivities
  -> assemble canonical xarray inputs
  -> align run metadata to retained sites
  -> materialize PyMC inputs
  -> build model
  -> sample
  -> make result and requested outputs
```

The public functions are exported from ``openghg_inversions.rhime``:

```python
from openghg_inversions.rhime import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    build_standard_rhime_model,
    filter_rhime_observations,
    make_standard_rhime_result,
    materialize_pymc_inputs,
    resolve_rhime_options,
    retrieve_or_reload_rhime_data,
    sample_rhime_model,
    with_prepared_rhime_sites,
)
```

``RhimeMergedData`` is also public from
``openghg_inversions.inversion_data``. It carries the merged per-site data and
the site-aligned acquisition metadata needed by later stages:

```python
from openghg_inversions.inversion_data import RhimeMergedData
```

## Default order is a recipe, not a state machine

The stage APIs describe their intended inputs, but they do not use a hidden
token or provenance record to enforce a single global order. The following
are real dependency constraints:

| Constraint | Reason |
| --- | --- |
| Resolve before using ``setup.data_args``, ``setup.run_spec``, or ``setup.sampler``. | Resolution owns option normalisation and validation. |
| Retrieve before filtering or basis construction. | Both consume ``RhimeMergedData``. |
| Build a basis before sensitivity construction. | Sensitivity projection needs a ``BasisFunctions`` object. |
| Assemble before materializing or model construction. | The latter consume canonical prepared inputs. |
| Build before sampling; sample before result construction. | These stages consume the previous stage's product. |
| Keep standard/multi-sector layouts consistent. | Prepared ``H``, retained basis, and model-sector layout are validated at model construction/replay. |

There is deliberately **no** corresponding rule saying that basis generation
must occur after filtering. That is the documented default and the intended
scientific workflow, but the current stage signatures permit a custom runner
to choose another recipe. That composition is not yet contract-tested; the
custom runner owns its scientific meaning and regression tests.

For example, the following derives or loads the basis from unfiltered coverage
while retaining the current filter-before-sensitivity order:

```python
setup = resolve_rhime_options(params=params, multisector=False)
merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=False)

# Deliberate custom choice: derive/load the basis from unfiltered coverage.
basis = build_rhime_basis(merged, setup.data_args)

# The actual inversion observations and sensitivities use the filtered data.
filtered = filter_rhime_observations(merged, setup.data_args)
site_data = build_rhime_sensitivities(
    filtered, basis, setup.data_args, multisector=False
)
prepared = assemble_rhime_inputs(filtered, basis, site_data, setup.data_args)
run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)
```

This does **not** reproduce the complete v0.6 execution order: v0.6 constructed
``H``/``H_bc`` before filtering, whereas this recipe constructs sensitivities
from the filtered data. It reproduces only the choice to generate the basis
from unfiltered coverage.

For a generated weighted basis this may be scientifically different from the
default: filtering can change retained observations, footprints, or retained
sites and hence change the weight field. For a fixed loaded basis it may be
equivalent, subject to compatible domain, source, and state coordinates.
Neither outcome is inferred or certified by the runner.

## Historical evidence: fixedbasisMCMC 0.6

Release ``v0.6.0`` used this order in ``fixedbasisMCMC``:

```text
retrieve/merge data
  -> basis_functions_wrapper(...)    # constructs basis and H/H_bc
  -> filtering(fp_data, filters)
  -> prepare model inputs
```

It also contained a Dask compatibility fallback:

```python
try:
    fp_data = filtering(fp_data, filters)
except ValueError:
    for site in sites:
        fp_data[site] = fp_data[site].compute()
    fp_data = filtering(fp_data, filters)
```

This was a fallback rather than a semantic requirement. The historical log
shows an earlier unconditional compute workaround, followed by a change to
retry eagerly only after lazy filtering failed. Separately, fixedbasisMCMC
computed selected downstream model variables (``H``, ``H_bc``, measurement
and error fields, and selected model fields) after filtering; that was a
model-input/shared-Dask optimisation, not the filter workaround.

Some individual filters intrinsically materialise data. For example,
``pblh_inlet_diff`` computes its boolean mask, and daily aggregation can load
the whole dataset. Thus a documentation promise of fully lazy filtering would
be inaccurate.

## Current Dask behaviour

PR 591 exposes ``filter_rhime_observations``. Internally it retains the same
lazy-first, eager-retry policy:

1. filter copied per-site merged datasets lazily;
2. on ``ValueError``, compute every per-site dataset and retry;
3. remove empty sites and keep all site metadata aligned.

Unlike 0.6, this happens before basis projection and before construction of
``H``/``H_bc``. An eager retry can still be costly because per-site footprint,
observation, and forward-model fields may be large, but it avoids unnecessarily
materializing the derived sensitivity arrays first.

``materialize_pymc_inputs`` is not a filtering workaround. It occurs after
assembly, materializes the model-owned variables, the selected aggregation-error
representation, and any matching marginal-standard-deviation diagnostic, and
preserves the canonical prepared xarray/Dask inputs. It cannot make a pre-basis
filter work.

### Deliberate eager filtering in a custom runner

When a known filter needs eager data, a custom runner can preserve the
original merged handoff while giving the filtering stage a separate eager
copy:

```python
from dataclasses import replace


def eager_for_filtering(merged):
    fp_all = dict(merged.fp_all)
    for site in merged.sites:
        fp_all[site] = merged.fp_all[site].compute()
    return replace(merged, fp_all=fp_all)


filterable = eager_for_filtering(merged)
filtered = filter_rhime_observations(filterable, setup.data_args)
```

This is the direct modern equivalent of the broad 0.6 fallback, but occurs
before sensitivity creation. A targeted variable-level eager copy is possible,
but it is necessarily filter-specific: local-influence filtering uses
footprint/release information, PBLH filters use PBLH, and aggregating filters
can touch the whole dataset. The public API does not yet expose a declared
filter-to-variable dependency map.

## Ownership and customization rules

- Treat the supplied handoffs as borrowed. Prefer a shallow copied mapping and
  a replacement ``RhimeMergedData`` rather than mutating ``merged.fp_all`` in
  place.
- ``filter_rhime_observations`` creates copied site datasets when it filters;
  ``build_rhime_basis`` does not mutate its ``RhimeMergedData`` handoff.
- Keep the filtered merged data, basis object, and sensitivity inputs paired
  deliberately. The APIs cannot prove that a custom basis was scientifically
  intended for the retained observations.
- Call ``with_prepared_rhime_sites`` after assembly so output provenance uses
  sites that survived filtering.
- Use ``materialize_pymc_inputs`` only for built-in PyMC model inputs. The
  canonical prepared inputs remain the durable object for output and replay;
  advanced complete-model builders retain that canonical context.

## Follow-up documentation and test probes

Future RHIME documentation should answer these explicitly.

1. Which stage orderings are mechanically supported, and which are the
   scientifically recommended recipes?
2. How should a developer document a deliberate mismatch between basis
   coverage and filtered inversion observations?
3. Which filters are lazy, which materialize a mask, and which aggregate/load
   whole datasets?
4. Should filtering have a public ``compute`` policy or a filter dependency
   declaration, instead of retrying on any ``ValueError``?
5. How can users distinguish a Dask failure from an invalid filter
   configuration when the current fallback retries all ``ValueError`` cases?
6. Should the project add contract tests for basis-before-filter custom
   recipes, Dask retry behaviour, and selective eager filtering?

The follow-up work has distinct owners:

- OPE-44 should turn the default copied spine into tested executable source and
  render that source into the customization guide rather than promoting the
  hand-written snippets here.
- OPE-54 should reuse that source shape for a tested project-owned basis stage,
  proving that the public spine can supply a new version of one step rather
  than merely record the default recipe.
- OPE-46 should own filtering/basis ordering, eager-retry policy, and any
  filter dependency or computation contract, including cached or externally
  supplied preparation products.
- OPE-47 should make the concrete PyMC graph and its semantic/output contract
  easier to reach and modify. The follow-on model-input requirements plan
  describes how components should derive the labelled input bill of materials
  used for validation, PyMC materialization, and replay in
  [RHIME model input requirements and prepared artifacts](rhime_model_input_requirements_and_artifacts.md).
- OPE-55 should bind prepared data to a serialized model/run specification for
  standard replay without making reusable preparation caches model-specific.
- OPE-49 should consolidate the proven examples and historical context into
  the final modification-focused developer documentation.

These questions are the value of the visible spine: customisation is now
possible in a standalone script, and its assumptions can be made explicit and
tested rather than remaining entangled in a monolithic runner.
