Developing RHIME models
=======================

RHIME is maintained for atmospheric scientists who need to read, adapt, and
cite the model code. Its development principles favour code locality and a
visible scientific narrative over a generic model framework.

The target style is a **procedural shell with a concept-oriented functional
core**. The useful quality of the release-0.6 ``fixedbasisMCMC`` workflow was
that a scientist could inspect one imperative recipe and then follow plainly
named scientific functions. Preserve that reading experience without restoring
the old monolithic function, mutable dictionaries, parallel-list state, or
implicit ``**kwargs`` forwarding.

This page gives design and review guidance. Exact supported arguments, errors,
compatibility periods, and artifact guarantees belong to their scoped
capability specifications and usage references. See :doc:`specifications` for
those sources and how to establish their review and delivery status.
A design recommendation is not evidence that an API implements it.

Keep legacy dependencies one-way
--------------------------------

The :download:`HBMCMC isolation and retirement contract
<../../openspec/specs/hbmcmc-isolation-retirement/spec.md>`
defines normal execution through supported modern interfaces without migration
translation. Modern behavior must have modern owners;
normal execution must not invoke migration tooling. Canonical configuration
rejects unsupported options before scientific work, including every explicit
``use_tracer`` occurrence.

PR 869 delivered the earlier one-way isolation boundary. The current #719 branch
removes the transitional namespace and adapters; completed revision-specific
checks are recorded in the active change's tasks and PR #974. Specification
synchronization and archival remain separate closeout work. The
:doc:`development migration guide </usage/legacy_and_migration>` records removed
interfaces and replacements. See :doc:`specifications` for capability ownership
and the scientific amendments on this branch.

Start with the scientific recipe
--------------------------------

Read a recipe as a **scientific pipeline with optional checkpoints**: acquire,
prepare, build, sample, reconstruct, and write products. The runner makes the
sequence visible; a saved-stage entry point resumes supported operations from
a declared handoff. The :ref:`pipeline and checkpoint architecture
<rhime-pipeline-checkpoints>` explains these responsibilities. A prepared-input
recipe need only implement the entry points and products it actually supports.

The core principles for new RHIME work are:

* Keep top-level orchestration procedural and readable from top to bottom.
* Construct the concrete model in mathematical and scientific order.
* Use scientifically meaningful names, explicit inputs, and ordinary callable
  components. Configuration is a convenience for selecting resolved values.
* Accept small duplication when it keeps alternative scientific recipes clear.
  Within one recipe, full and saved routes call the same step functions with the
  same objects; a saved route only loads a checkpoint before and saves one after.
* Use dataclasses for concrete scientific concepts or durable handoffs.
* Normalize configuration at its owning boundary and forward named values
  explicitly, including an explicit ``config.data.select("name", ...)`` selection
  where useful.
* Keep helpers close to their scientific owner. Tiny registries can serve
  homogeneous families such as filters without controlling the whole model.
* Make local modification possible: a scientist can copy a runner or replace
  one component without reconstructing a framework lifecycle.

The review test is:

  A scientist familiar with release-0.6 ``fixedbasisMCMC`` can read a runner,
  open one nearby concrete model function, and replace one ordinary component
  whose interface explains what RHIME supplies.

Preparation helpers should own an operation, policy, meaningful handoff, or
necessary compatibility boundary. A uniform stage name, timing, or argument
renaming alone does not justify a second forwarding signature. Keep shared
indexing reusable and model-specific preparation policy beside its recipe.
Call canonical ``prepare_basis`` directly with resolved arguments and use the
actual sensitivity, selection and assembly implementations. Do not retain
forwarding wrappers solely for uniform stage names, timing or keyword translation.
Pass per-site datasets and aligned ``SiteOptions`` between preparation steps;
acquired ``MergedData`` remains separate. Construct results from their scientific
inputs rather than modifying the result after construction.
Use :doc:`architecture_principles` to assess responsibilities and consumer
interfaces, and :doc:`validation_and_xarray` to identify validation owners.

Choose the smallest useful extension
------------------------------------

A named model recipe combines a procedural runner with a readable concrete
model builder. It may reuse preparation, numerical, sampling, and output
functions while keeping their scientific composition visible.

Use these questions when extending RHIME:

1. Can an existing component accept the option while preparation, state sharing,
   observation channels, and output meaning remain the same?
2. Does a change to those structures deserve a named recipe with its own
   assumptions and tests? Similar short orchestration is acceptable.
3. Do two production recipes use the same equations and option meanings?
   Extract that demonstrated shared component.
4. Is this still a project-specific experiment? An ordinary callable extension
   point can provide a useful incubation path.
5. Is a variant recurring in papers? A tested in-tree recipe lets the released
   package version and DOI identify its implementation.

Reconsider a general registry or framework only after real recipes show stable
repeated structure. A public recipe should have an obvious runner and builder,
a runnable example, equation tests and a smoke test, and documentation of its
inputs, outputs, assumptions, limitations, and scientific citation.

For current public selection and names, use :doc:`/usage/rhime` and
:doc:`/usage/cli`, backed by the :download:`model-selection contract
<../../openspec/specs/rhime-model-selection/spec.md>`. Selection dispatches to
concrete recipes; their input contracts can differ. The
:doc:`/usage/standard_model_family`, :doc:`/usage/nested_domain_model_family`,
and :doc:`/usage/co2_model_family` pages describe implemented family support.

Compose scientific concepts directly
------------------------------------

A component represents a concept used when discussing the model: basis
functions, a flux contribution, a baseline, pollution events, model-data
mismatch, aggregation error, or a likelihood. It need not be mathematically
atomic. A baseline can combine boundary conditions, an offset, and fixed outer
fluxes; a multisector component can combine several pollution contributions.

Prefer a normal function. Its name, options, inputs, and result should explain
the scientific concept. A composite can call nearby smaller functions; keep a
model-specific composite beside its recipe until another production recipe
uses the same equations and meanings. Inheritance, semantic graphs, and
manifests are unnecessary prerequisites for authoring a component.

Return the smallest result the consumer needs. An observed tensor can be the
whole result of a likelihood. A small dataclass can join related scientific
outputs, such as aligned observation channels or a coherent reduction. A
shorter signature, speculative flexibility, or provenance alone does not
justify a new domain object.

The :doc:`affine components prototype <affine_components>` compares the
centered boundary and coherent flux components through their values at a state
reference and their responses to state deviations. The comparison distinguishes
these state references from the observation shift used for boundary centering;
ordinary absolute boundaries retain their direct calculation. It is a design
experiment alongside these principles, rather than a general component interface.

For a concrete likelihood, show the completed mean and how measurement error,
fixed aggregation error, and inferred mismatch enter its probability model.
These quantities have different scientific meanings; the supported covariance
representations belong to the concrete likelihood and recipe. Do not infer
universal support from a common calling signature. Current equations and
restrictions are explained in :doc:`/usage/rhime` and :doc:`/usage/co2_models`.

Sampler choice becomes part of the recipe when the graph and update algorithm
are coupled, for example through a state-likelihood cache or an ordered
``CompoundStep``. Keep the graph, sampler, and update order together. An ordinary
likelihood variation can use the existing callable seam without a new runner
or a general sampler protocol. Fixed OU, IID site sigma, and global sigma are
different scientific targets even when they reuse numerical kernels. Use a
stock PyMC calculation as the correctness reference for an optimized route.

Preserve labels until a named numerical boundary. Shared alignment values such
as ``SigmaAlignment`` keep site/period interpretation close to the mismatch
component; repeated factorization or parallel lists can hide that meaning.
The :download:`model-composition contract
<../../openspec/specs/rhime-model-composition/spec.md>` owns the registered
coordinate and standard/multisector custom-likelihood interfaces. Follow
:doc:`/usage/customising_rhime` for a concrete replacement example. Covariance
axes still need their own scientific alignment; registering one dimension does
not establish that an independently supplied matrix has the right second axis.

Keep positive-support prior policy beside prior construction rather than
repeating distribution allow-lists in each likelihood. Validate independently
sourced values at the boundary that owns them, then trust locally constructed
intermediates. Ordinary labelled alignment, Pint conversion, and the numerical
operation itself often express the required check; add a custom assertion when
it protects a scientific policy or gives a materially better boundary failure.

Keep configuration with its scientific owner
--------------------------------------------

Runtime inputs and user choices are different interfaces. Function signatures
describe observations, sensitivities, priors, and existing model terms.
Configuration groups choices by the component that owns their meaning. For
example, boundary-condition prior and period choices belong to the baseline;
site/period mismatch choices belong to the error model.

Keep accepted options and defaults beside those consumers. Compose them
explicitly at resolution rather than inferring a schema from every runtime
argument or introducing a generic option registry. Let each file frontend own
its syntax, apply raw overrides before normalization, and forward resolved
values explicitly. A moderately long keyword-only signature can be clearer
than an ambient context object or implicit forwarding through the pipeline.

The current standard/multisector requests are
described in :doc:`/usage/rhime_configuration`. That reference and the reviewed
:download:`configuration change
<../../openspec/specs/rhime-configuration/spec.md>`
own requested choices and their resolution. Canonical prepared construction
belongs to the :download:`scientific-input contract
<../../openspec/specs/rhime-scientific-inputs/spec.md>`; persisted formats and replay
belong to the prepared/sample handoff contracts. The request precedes preparation;
retained-site execution facts become known later. Frozen records do not imply
recursive immutability of borrowed arrays or caller-owned containers.

Configuration groups acquisition and preparation choices under
``RhimeConfig.data: DataConfig``. The runner acquires data separately and passes
it into scientific preparation. ``MergedData`` holds acquired data only;
preparation steps pass per-site datasets and ``SiteOptions`` between them.
Standard and multisector recipes support reusable preparation and sample
manifests; see :doc:`/usage/staged_workflow`. The remaining family adoption tasks from
`PR 802 <https://github.com/openghg/openghg_inversions/pull/802>`_ remain active.
This guide does not fix private module locations, configuration class layout,
or future manifest structure. Existing templates and complete option listings
belong with their recipes and usage references.

Preserve scientific handoffs and output meaning
-----------------------------------------------

Prepared inputs join labelled, backend-neutral scientific products. A recipe
selects the arrays it consumes and materializes related values together at the
named model boundary. Extension or diagnostic variables need not become model
inputs merely because they share a Dataset. A scientifically distinct recipe
can compose reusable prepared values without pretending to be substitutable
for another family.

Moving a preparation equation, including observation-error derivation,
minimum-error calculation, or boundary-period expansion, is a scientific
change requiring evidence for unchanged behavior and explicit tests for agreed
semantic changes. Model-owned minimum-error estimates use retained observations
after invalid-row removal; historical pre-drop estimates are not a parity target.
Observation-error imputation follows selection filters and uses only retained
observations. These numerical rules belong to the
:download:`scientific-inputs contract
<../../openspec/specs/rhime-scientific-inputs/spec.md>`; configuration owns the
accepted choices and phase resolution. Supported prepared-input
contents and examples are in :doc:`/usage/rhime` and
:doc:`/usage/customising_rhime`; checkpoint limitations are in
:doc:`/usage/staged_workflow`.

Standard and multisector preparation retains raw spatial BC inputs and operands
for supported minimum-error choices. A compatible later model can change BC
periods or floor policy without repeating spatial preparation. Saved samples
retain the fitted alignment and floor operands; replay consumes those exact
values without a new graph or calculation. The active
:download:`prepared handoff contract
<../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/prepared-recipe-handoff/spec.md>`
and :download:`sampled handoff contract
<../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/sampled-recipe-handoff/spec.md>`
own those artifact guarantees. CO2's prior-dependent coherent preparations retain
their separate reuse constraints. The :download:`#876 alignment change tasks
<../../openspec/changes/move-alignments-to-model-boundary/tasks.md>` record
revision-specific validation evidence and limits. Broader handoff programme
acceptance remains pending.

Keep one scientific sample codec owner for save/load and let replay consume its
decoded result. Keep product/export saving in a nearby operational helper with
shared naming/path policy, and normalize descriptive sampler provenance through
a shared pure metadata operation. These are development recommendations; the behavioral
contracts require scientific parity and artifact meaning without fixing internal
helper names or call graphs.

Source selection and state grouping should remain plainly named operations on
labelled values and retained artifacts. Resolutions, partitions, sources, and
observation channels need explicit labels. Variable suffixes or positions alone
cannot preserve those scientific meanings.

Postprocessing needs quantity meaning as well as arrays. Small scientific-role
mappings can connect a model's local names to real output consumers without
imposing a compiler or backend naming convention. Define the quantity, units,
dimensions, and uncertainty scope with the output contract. The maintained
:doc:`/usage/retained_state_reconstruction` and :doc:`/usage/affine_flux_map`
pages explain current reconstruction and its limits; their reviewed capability
changes are linked from :doc:`specifications`. A role name alone does not
promise a storage schema or graph-free replay for every analysis.

The numerical ownership rules apply throughout: inputs are borrowed and may be
Dask-backed; copying, computation, densification, persistence, and serialization
belong at visible boundaries. See :doc:`numerical_array_ownership` for the
reasoning and :doc:`validation_and_xarray` for practical patterns.

Record evidence and scientific ownership
----------------------------------------

Public recipe documentation should cite accessible scientific sources.
Implementation-history paths and private commit hashes belong in issues,
pull requests, or internal parity records. Released code, examples, and tests
should remain understandable without private Verification Games access.

OpenGHG Inversions owns released reusable components and production recipes;
Verification Games owns calibration cases, experimental orchestration,
campaign configuration, and independent acceptance evidence. A cutover includes
the downstream consumer and removal of its duplicate implementation after
parity, as well as merging the package implementation. Track that acceptance
work in the owning issue rather than treating a historical plan as completion.
