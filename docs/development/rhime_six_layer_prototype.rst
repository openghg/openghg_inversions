An explicit six-layer RHIME prototype
====================================

This prototype makes ownership and durable handoffs visible in the existing
standard and multisector workflows. A saved posterior can now reach supported
products without rebuilding its PyMC graph. Scientific equations, prior
choices, retrieval policy, and sampling algorithms retain their existing
implementations.

The experiment implements a bounded part of
``docs/plans/rhime_end_to_end_architecture.md`` against ``devel`` at
``f708d606`` on 29 September 2026. That plan's earlier issue and code mappings
remain historical evidence. This page describes implemented behavior and
possible smaller landing steps; it does not supersede
:doc:`rhime_model_development` or complete the proposed linked-channel models.

Where each layer lives
----------------------

The layers are responsibilities and handoffs. They do not require six classes
or an execution engine. Open :func:`openghg_inversions.rhime.standard.run_rhime`
or :func:`openghg_inversions.rhime.multisector.run_rhime_multisector` to read
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
     - ``rhime/preparation.py``, ``inversion_data/preparation.py``,
       ``inversion_data/prepared.py``
     - Recipe stages filter, build a basis and sensitivities, and assemble
       ``RhimePreparedInputs``. Its schema, validation and persistence have a
       separate owner. ``forward/domain_support.py`` supplies pure grid
       operations shared with nested preparation.
   * - 3. Model construction
     - Concrete standard/multisector builders and ``models/``
     - Materialization remains explicit. Builders return a PyMC graph, roles,
       and an independently usable ``OutputContract``. Equations and state
       allocation remain visible in the recipe family.
   * - 4. Inference and checks
     - ``inference/sampling.py``
     - ``RhimeSampler.sample(model, variable_roles=...)`` owns sampling and
       predictive draws. The recipe adapter adds timing. Specialized cached
       sigma graph/step policy stays with its scientific recipe.
   * - 5. Scientific reconstruction
     - ``postprocessing/contracts.py``, ``postprocessing/reconstruction.py``,
       existing basis and output operations
     - Samples, prepared data, and an explicit contract form an
       ``InversionOutput`` view. Construction needs no live model and performs
       no file writes. Domain adapters select roles and state axes explicitly.
   * - 6. Products and persistence
     - Existing output writers/codecs; ``workflow/artifacts.py``
     - Writers consume scientific views. Shared helpers handle JSON, content
       hashes and paths. ``rhime/stages.py`` retains recipe dispatch and
       artifact-schema policy, including the saved output binding.

The principal dependency direction is recipes towards these owners.
``inference`` may use reusable model-coordinate helpers; neutral reconstruction
does not import ``rhime`` or a model backend. ``forward`` does not query stores
or allocate latent states. ``workflow/artifacts.py`` does not choose a model,
infer execution order, or know the contents of a scientific run.

The package moves contain actual implementations. The former import locations
re-export compatible objects. In particular, ``rhime.RhimeSampler`` and
``rhime.sampling.RhimeSampler`` still identify the shared sampler, while
``inversion_data.RhimePreparedInputs`` and the older preparation-module import
identify the same prepared class. Its on-disk schema remains version 1.
Existing named runners and configuration formats continue to work.

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
use :func:`openghg_inversions.postprocessing.reconstruction.make_inversion_output`
directly with the prepared value, trace and metadata. The existing recipe
output adapter supplies those metadata for maintained runs.

Borrowed xarray payloads remain borrowed. Related model arrays materialize
together at the existing named backend boundary. Array-valued JSON metadata
crosses an explicit serialization boundary. No property performs retrieval,
and the ownership moves do not introduce computation into prepared values.

The durable output handoff
-------------------------

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
------------------------------------

The nested recipe now uses the same durable contract to form outer and inner
output views, including when the live model has been discarded. Each view
selects its own roles and explicitly maps its posterior state dimension to
its basis dimension. Both share the unchanged posterior, preserve every
chain, and retain distinct native grids and support provenance.

Two public operations were extracted from that recipe:

* ``rectangular_extent_mask`` constructs an inclusive coordinate bounding-box
  mask on a target grid. It does not infer cell edges, interpolate, or
  approximate an irregular boundary.
* ``remove_domain_overlap`` zeros marked native cells after exact labelled
  alignment, preserving additional dimensions and lazy payloads.

The recipe still decides per-site versus union masking, adapts OpenGHG
objects, and records support policy. Removing spatial overlap does not imply
prior independence. This extraction supplies a concrete starting point for
multisector and CO2 nesting; those compositions are not implemented here.

What the experiment establishes
------------------------------

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

How to scale this back for delivery
----------------------------------

The most valuable first change is the output handoff: retain ``OutputContract``,
neutral view construction, matched saved bindings, and graph-free replay.
That directly removes repeated model reconstruction from a real workflow.

Then land the prepared contract/acquisition split with compatibility exports.
It gives acquisition and scientific preparation distinct owners without
changing their scientific bodies. The sampler relocation, generic artifact
namespace, and public domain-operation extraction can be reviewed separately;
the durable output contract does not require all three package moves.

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

The draft is consequently an implementation to inspect and trim. Its package
names are negotiable; explicit scientific choices, compatible handoffs, and
the ability to reopen supported products are the properties to retain.
