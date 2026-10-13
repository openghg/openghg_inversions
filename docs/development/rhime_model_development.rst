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

The active :download:`HBMCMC isolation and retirement contract
<../../openspec/changes/isolate-and-retire-hbmcmc/specs/hbmcmc-isolation-retirement/spec.md>`
sets the boundary: HBMCMC may call modern APIs, while production code
outside HBMCMC must not import, re-export or execute through it. Keep historical
resource translation inside the legacy entry point. Retained RHIME spelling
aliases belong to runner frontends, which translate and warn before constructing
canonical configuration. ``RhimeConfig.from_params`` has no alias vocabulary,
translation or deprecation warnings. Give retained modern behavior modern owners;
small duplication inside HBMCMC is acceptable to preserve this direction.
The separate contract owns the exact compatibility rules and 0.9 removal timing.
The initial 0.8 implementation is under review in PR 869; complete 0.9 removal
and durable-spec promotion remain pending. See :doc:`specifications` for delivery
status.

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
* Use dataclasses for concrete scientific concepts or durable handoffs.
* Normalize configuration at its owning boundary and forward named values
  explicitly, including an explicit ``config.select("name", ...)`` selection
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

The current standard/multisector request and its compatibility adapters are
described in :doc:`/usage/rhime_configuration`. That reference and the reviewed
:download:`configuration change
<../../openspec/specs/rhime-configuration/spec.md>`
own constructor and resolution details. The request precedes preparation;
retained-site execution facts become known later. Frozen records do not imply
recursive immutability of borrowed arrays or caller-owned containers.

Configuration grouping and reusable handoffs are being reconsidered in
`PR 802 <https://github.com/openghg/openghg_inversions/pull/802>`_. Check its
current proposal and implementation status before adopting those interfaces.
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
change requiring its own parity evidence. The configuration contract records
the current observation-error-before-filtering policy. Supported prepared-input
contents and examples are in :doc:`/usage/rhime` and
:doc:`/usage/customising_rhime`; checkpoint limitations are in
:doc:`/usage/staged_workflow`.

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
