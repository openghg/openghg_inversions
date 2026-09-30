An explicit six-layer RHIME prototype
=====================================

This prototype makes ownership and durable handoffs visible in the existing
standard and multisector workflows. A saved posterior can now reach supported
products without rebuilding its PyMC graph. Scientific equations, prior
choices, retrieval policy, and sampling algorithms retain their existing
implementations.

The experiment implements a bounded part of
``docs/plans/rhime_end_to_end_architecture.md`` against ``devel`` at
``f708d606`` on 29 September 2026. That plan's earlier issue and code mappings
remain historical evidence. This page describes implemented behavior and
possible later delivery splits; it does not supersede
:doc:`rhime_model_development` or complete the proposed linked-channel models.

The 30 September namespace revision implements the recipe/capability layout:
``recipes/`` owns complete models and their runners, while
``model_components/`` owns reusable model-building functions. Prepared-input
values, execution from prepared values, and output-view construction have
separate names. Internal imports use these canonical owners. Established
imports remain explicit compatibility aliases to the same objects. A later
PR split is a delivery decision, not a different architectural target.

Where each layer lives
----------------------

The layers are responsibilities and handoffs. They do not require six classes
or an execution engine. Open :func:`openghg_inversions.recipes.standard.run_rhime`
or :func:`openghg_inversions.recipes.multisector.run_rhime_multisector` to read
the actual scientific sequence; each still calls ordinary functions directly.

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - Layer
     - Concrete owner
     - Handoff and limit
   * - 1. Inputs and acquisition
     - ``inversion_data/acquisition.py``
     - Owns retrieval/reload, complete site options, and ``RhimeMergedData``.
       Recipe configuration resolves policy before calling the provider.
   * - 2. Scientific preparation
     - ``recipes/preparation_adapters.py``, ``inversion_data/preparation.py``,
       ``inversion_data/prepared_inputs.py``
     - Recipe stages filter, build a basis and sensitivities, and assemble
       ``RhimePreparedInputs``. Its schema, validation and persistence have a
       separate owner. ``recipes/_domain_support.py`` keeps pure grid
       operations local to nested preparation.
   * - 3. Model construction
     - Concrete standard/multisector builders and ``model_components/``
     - Materialization remains explicit. Builders return a PyMC graph, roles,
       and an independently usable ``OutputContract``. Equations and state
       allocation remain visible in the recipe family.
   * - 4. Inference and checks
     - ``inference/sampling.py``
     - ``RhimeSampler.sample(model, variable_roles=...)`` owns sampling and
       ordinary PyMC predictive execution. Cached-sigma recipes disable its
       generic posterior prediction and generate joint replicates separately.
       ``inference/diagnostics.py`` calculates neutral posterior summaries;
       ``postprocessing/metrics.py`` contains scientific predictive scores.
       Recipes retain acceptance policy.
   * - 5. Scientific reconstruction
     - ``postprocessing/contracts.py``, ``postprocessing/output_views.py``,
       existing basis and output operations
     - Samples, prepared data, and an explicit contract form an
       ``InversionOutput`` view. Construction needs no live model and performs
       no file writes. Domain adapters select roles and state axes explicitly.
   * - 6. Products and persistence
     - Existing output writers/codecs; ``recipes/_stage_artifacts.py``
     - Writers consume scientific views. Local stage helpers handle JSON, content
       hashes and paths. ``recipes/stages.py`` retains recipe dispatch and
       artifact-schema policy, including the saved output binding.

The principal dependency direction is recipes towards these owners.
``inference`` may use reusable model-coordinate helpers; output-view
construction does not import recipes or a model backend. Local domain helpers
do not query stores or allocate latent states. Stage artifact helpers do not
choose a model or infer execution order. Keeping these two helpers local
reflects their single production consumer.

The moves contain actual implementations. Complete standard and multisector
recipes remain readable in one module each, with CO2-family recipes grouped
in a subpackage. The former ``rhime`` and ``models`` locations re-export the
canonical objects, including family and component submodules.
``rhime.RhimeSampler`` and ``rhime.sampling.RhimeSampler`` still identify the
shared inference sampler. ``inversion_data.RhimePreparedInputs`` and the older
preparation-module import identify the class now defined in
``inversion_data.prepared_inputs``. Its
on-disk schema remains version 1. Existing runner names and configuration
formats continue to work; current examples use ``recipes/config`` resources.
The installed ``rhime/config`` resource tree is retained for compatibility,
with matching template contents.

Use ``recipes.from_prepared`` for execution from durable prepared inputs and
``recipes.preparation_adapters`` for standard/multisector preparation policy.
Only established imports receive compatibility aliases; abandoned names
introduced within this unmerged prototype are removed. Compatibility aliases
preserve object identity, not private monkeypatch locations: patch dependencies where
the implementation looks them up.

Reusable cached-sigma mechanics live in ``inference/cached_sigma.py``; the
scientific graph and joint-prediction policy stay in their CO2 recipes.
Carbon-specific linked output adaptation lives beside those recipes in
``recipes/co2/outputs.py``, with ``postprocessing.linked_paris_outputs``
retained as a compatibility alias. General PARIS writers stay in
``postprocessing``.

A readable procedural route
---------------------------

The following outline shows the handoffs, omitting configuration expansion,
timing and provenance fields. The maintained runners contain the complete
calls; :doc:`/usage/staged_workflow` supplies runnable CLI examples.

.. code-block:: text

   resolve recipe options
   merged = acquire or reload observations, transport, fluxes and boundaries
   filtered = filter observations using resolved scientific policy
   basis = construct or load the basis for those inputs
   sensitivities = apply transport and boundary operators
   prepared = assemble labelled inputs with their retained basis

   arrays = materialize selected related arrays at the PyMC boundary
   built = build the concrete scientific model(arrays, resolved options)
   samples = sampler.sample(built.model, variable_roles=built.variable_roles)

   contract = built.output_contract
   view = make_inversion_output(prepared, samples, contract, provenance)
   write requested products(view)

This is a sequence of explicit choices. The output contract never chooses
equations, source sharing, a prior, or a likelihood. It records the output
meaning of the model already selected. For ordinary analysis, callers can
use :func:`openghg_inversions.postprocessing.output_views.make_inversion_output`
directly with the prepared value, trace and metadata. The existing recipe
output adapter supplies those metadata for maintained runs.

Borrowed xarray payloads remain borrowed. Related model arrays materialize
together at the existing named backend boundary. Array-valued JSON metadata
crosses an explicit serialization boundary. No property performs retrieval,
and the ownership moves do not introduce computation into prepared values.

The durable output handoff
--------------------------

Previously, staged postprocessing built a model to recover its variable roles
and supported formats. New staged samples write:

.. code-block:: text

   prepare/
     prepared-inputs.nc
     prepare-manifest.json
   sample/
     posterior.nc
     output-binding.json
     sample-manifest.json

``OutputContract`` contains roles, supported writer formats, JSON-compatible
builder provenance, and an optional explicit trace-to-basis state-dimension
mapping for one view. It contains no numerical arrays, Python closures, or
live model. The sample-side binding joins that contract to the content
identities of the exact prepared and posterior files. The sample manifest
also hashes the binding and locates it relative to its own directory.

Postprocessing validates the prepared artifact and scientific configuration,
the posterior identity, and the binding before loading samples or producing
outputs. Missing bindings, mismatched artifacts, malformed metadata and
unsupported versions are errors. A new manifest cannot silently fall back to
building a graph. Its schema version is 2; genuine old version-1 manifests
retain their explicit graph-building compatibility path. Version 1 with
version-2 binding fields is rejected as malformed. These content hashes check
artifact consistency; they are not signatures or a security trust boundary.

``RhimeResult`` can therefore contain ``output_contract`` with ``model=None``
and ``model_build_result=None``. The standard and multisector result factories
accept either saved output information or a live build result. Existing code
passing the live build result still acquires its contract automatically.

Keep the sample manifest and binding together when copying a run. Schema-2
artifacts require a reader that understands this format; old releases should
not be used to postprocess them. The preparation artifact and trace codecs
are unchanged. Staged sampling persists its trace before later product
generation; the monolithic runner's existing output-write order is unchanged.

This route reuses stored posterior quantities. Computing new posterior
predictive draws or new backend deterministics still needs a declared model
replay route. The staged CLI still supports standard and multisector. A
graph-free nested output adapter does not imply a nested staged runner.

Nested domains exercise a second view
-------------------------------------

The nested recipe now uses the same durable contract to form outer and inner
output views, including when the live model has been discarded. Each view
selects its own roles and explicitly maps its posterior state dimension to
its basis dimension. Both share the unchanged posterior, preserve every
chain, and retain distinct native grids and support provenance.

Two pure operations in its local ``_domain_support`` helper separate array
operations from OpenGHG adaptation:

* ``rectangular_extent_mask`` constructs an inclusive coordinate bounding-box
  mask on a target grid. It does not infer cell edges, interpolate, or
  approximate an irregular boundary.
* ``remove_domain_overlap`` zeros marked native cells after exact labelled
  alignment, preserving additional dimensions and lazy payloads.

The recipe still decides per-site versus union masking, adapts OpenGHG
objects, and records support policy. Removing spatial overlap does not imply
prior independence. These local operations supply a starting point for
multisector and CO2 nesting; a second production consumer can establish a
shared public contract. Those compositions are not implemented here.

What the experiment establishes
-------------------------------

Focused checks cover the responsibilities affected by the change:

* Existing RHIME preparation, model, inference, output and public-stage tests
  retain their behavior. Moved acquisition, prepared-contract and sampler
  definitions were also compared against the originals for unchanged bodies.
* Prepared artifacts preserve their schema and import identity; existing
  borrowed-array and Dask tests exercise their execution boundary.
* A fresh process reopens staged artifacts with acquisition, model-input
  materialization and model construction disabled. Distinct chains contribute
  to the reconstructed result.
* Corrupt, swapped, absent and unsupported output bindings fail before output
  processing; genuine older manifests retain their compatibility route.
* Nested views round-trip their contract without a live model. Domain masks
  preserve laziness and reordered grids and reject misaligned coordinates.
* Fresh imports of neutral reconstruction require neither PyMC nor RHIME.
  Direct inference imports preserve PyTensor initialization defaults.

These are implementation and small numerical checks, not a production
inversion benchmark or scientific validation of a new model. They do not
establish runtime improvements from moving code.

Ideal prototype and subsequent delivery
---------------------------------------

Keep the complete ideal organization in the prototype so its navigation and
extension points can be assessed together. Retain ``OutputContract``, neutral
view construction, matched saved bindings, graph-free replay and the distinct
acquisition/prepared-contract owners alongside the new namespaces.

The PR can later be split into functional handoffs and organizational moves,
with compatibility exports preserving existing callers. That split should
retain the agreed target rather than reduce the prototype in advance. The
plan's migration map distinguishes public import compatibility from changes
to implementation-level patch locations and installed resource paths.

Before promoting the wider plan, extend this route through one fixed-coupling
linked-channel model. Its adapter should obtain unequal observation supports,
prepare the declared joint covariance, and bind matching reconstruction
artifacts. That will test what the standard prototype cannot demonstrate.
Keep the existing CO2 affine-map persistence work as an implementation to
integrate with, not as a missing feature to rebuild.

Remaining architecture work includes scientific input requests and resolved
provider identities; quantity-specific output availability and units; paired
native-flux maps; linked-channel covariance/reduction bindings; and explicit
source/domain subsets with unequal basis sizes. The present output contract
still advertises writer formats, so it does not yet solve partial availability
of quantities within a format. It also does not implement coherent CH4/C2H6,
inferred coupling, or radiocarbon equations.

Inference checks and scientific scores
-------------------------------------

``inference/diagnostics.py`` owns reusable posterior convergence summaries.
``postprocessing/metrics.py`` owns Bayesian R-squared scores calculated from observations and
predictive samples, including site/time groupings. The established
``postprocessing.diagnostics`` interfaces remain adapters for existing output
callers; product naming and stage thresholds remain outside neutral inference
calculations. Moving these calculations does not change their equations or
establish new acceptance thresholds.

Cached-sigma posterior prediction still uses its dedicated joint generator.
Latent prior sampling does not supply joint prior-predictive observations, and
a complete divergence assessment must account for the separately named
sigma-step statistics. `Issue #769
<https://github.com/openghg/openghg_inversions/issues/769>`_ tracks integration
of this specialized behavior with generic predictive execution. That
functional work is not implemented by this namespace revision.

Import isolation follow-up
--------------------------

The recipe package initializer still eagerly imports its families and staged
workflow. Importing a local helper therefore also loads the model backend and
unrelated recipes. The ownership changes improve navigation but do not yet
isolate family imports or establish broader scientific composability.

A follow-up should selectively load public exports and move PyTensor setup to
the backend boundaries that currently rely on the package initializer. Verify
in fresh processes that local artifact/domain helpers load no backend or
unrelated family, a failing optional family does not prevent another family
from importing, and direct backend imports preserve default and explicitly
selected precision. Established public exports must retain object identity.
