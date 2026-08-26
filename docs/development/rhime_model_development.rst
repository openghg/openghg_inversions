Developing RHIME models
=======================

Purpose
-------

RHIME is maintained for atmospheric scientists who need to read, adapt, and
cite the model code. Its production architecture therefore favours code
locality and a visible scientific narrative over a generic model framework.

The target style is a **procedural shell with a concept-oriented functional
core**. The useful quality of the release-0.6 ``fixedbasisMCMC`` workflow was
that a scientist could inspect one imperative recipe and then follow plainly
named scientific functions. New RHIME code should provide that reading
experience without restoring the old monolithic function, mutable dictionaries,
parallel-list state, or implicit ``**kwargs`` forwarding.

Core principles
---------------

The following principles are requirements for production RHIME development:

* Top-level orchestration is procedural and readable from top to bottom.
* Model construction appears in mathematical and scientific order.
* Functions use scientifically meaningful names and explicit inputs.
* Components remain directly callable. Labelled model components require the
  small ``CoordRegistry`` attached by ``registered_model()``; configuration
  remains an optional convenience.
* Small duplication is acceptable when it keeps alternative scientific
  workflows visible.
* Dataclasses are reserved for concrete scientific concepts or durable
  boundaries.
* Configuration is normalized once at the boundary. Values are forwarded
  explicitly after that boundary.
* Tiny registries are appropriate for homogeneous families such as filters.
* Private numerical helpers are acceptable, but reading the main path must not
  require reconstructing a framework.
* Scientists should be able to copy a runner or component and modify it
  locally.

The review test is:

  A scientist familiar with release-0.6 ``fixedbasisMCMC`` should be able to
  read a runner, open one nearby concrete model function, and replace one
  ordinary component whose interface explains what RHIME supplies. They should
  not need to learn a compiler, dependency-injection system, manifest, or
  framework lifecycle first.

Model recipes
-------------

A production **model recipe** contains a named procedural runner and a readable
concrete model builder. A recipe may reuse preparation, numerical, sampling,
and output functions, but it owns the scientific order in which they are
composed.

Near-term model development may produce several explicit recipes. This is an
intentional way to deliver tested and citable scientific models before their
actual similarities are known. Do not introduce a generic pipeline or model
registry to control this proliferation in advance.

Use these rules when extending RHIME:

1. Add an option to an existing component when data preparation, state sharing,
   observation channels, and outputs retain the same shape and meaning.
2. Add a named model recipe when any of those structures changes. A repeated
   orchestration sequence is acceptable.
3. Extract a shared component after the same scientific equations and option
   meanings occur in at least two production recipes.
4. Keep callable extension points, such as ``likelihood_builder``, as useful
   incubation and project-specific escape hatches.
5. Graduate a recurring paper-facing variant into a tested, named in-tree
   recipe so the released OpenGHG Inversions version and DOI identify its
   implementation.
6. Reconsider a registry or framework only after real model recipes demonstrate
   stable repeated structure.

Each public recipe should provide:

* one obvious runner and concrete model builder;
* a runnable configuration or example stored beside the recipe or in an
  obviously parallel ``rhime/config`` layout;
* focused tests of its equations and an end-to-end smoke test;
* a description of inputs, outputs, assumptions, and limitations; and
* the relevant scientific citation and model provenance.

The current model-family expansion plan is recorded in
``docs/plans/rhime_model_family_expansion.md``.

Components and composite components
-----------------------------------

A component represents a concept used when discussing the scientific model.
Examples include basis functions, a flux contribution, a baseline, pollution
events, model-data mismatch error, aggregation error, and a likelihood.

A component does not have to be mathematically atomic. A baseline may combine
mapped boundary conditions, an offset, and fixed outer-domain fluxes. A
likelihood may combine measurement error, aggregation error, model-data
mismatch, pollution-event treatment, and a probability distribution. A
multisector flux component may combine several forward terms that are all
pollution contributions to one observation model.

Prefer a normal function for a component. A composite function may call nearby
smaller functions, but its public name, options, inputs, and output should
describe the scientific concept. Keep a model-specific composite beside its
recipe until another production recipe uses the same equations and meanings.

Do not require inheritance, registration, a semantic graph, or a manifest to
author an ordinary component. A dataclass is justified when it is itself a
recognizable product, such as ``BasisFunctions`` or a prepared handoff joining
two aligned observation channels. It is not justified solely to shorten a
function signature.

Configuration ownership
-----------------------

Runtime scientific inputs and user configuration options are different
contracts:

* Function signatures should describe runtime values such as observations,
  sensitivities, basis functions, priors, and existing PyMC terms.
* Configuration documentation should group user choices by the scientific
  component that owns them.

For example, ``bc_prior`` and ``bc_freq`` belong to the baseline component,
while ``sigma_prior``, ``sigma_freq``, and ``sigma_per_site`` belong to the
model-data mismatch component. A component may publish a small explicit tuple
or table of owned option names. Do not infer the user configuration schema from
every argument in its runtime function signature.

Configuration parsing should:

* translate legacy spellings and normalize values once at the runner boundary;
* reject unknown and unused options;
* preserve model-specific sections when names repeat across channels; and
* pass resolved values explicitly to the functions that own them.

New RHIME templates should not be added to the legacy ``hbmcmc/config`` tree or
to an unrelated global template collection. Prefer
``openghg_inversions/rhime/config/<recipe-name>.ini`` so the config layout
mirrors the model recipe layout. A complex recipe implemented as a subpackage
may keep a small model-specific resolver in that subpackage. Migrate the
existing standard template only with a compatibility plan; it is not a
prerequisite for adding a new model.

It is acceptable for a stage function to have a moderately long keyword-only
signature. Do not replace honest parameters with an ambient context object or
forward ``**kwargs`` through the scientific pipeline.

Model input requirements
------------------------

Some components also require named arrays at the PyMC materialization boundary.
When an ordinary function signature cannot describe those requirements early
enough, use a small constant or pure function beside the component. This
information may drive early validation and coordinated materialization.

It is not a caller-authored manifest and must not determine execution order.
Ordinary runners should not construct or thread a requirements object through
the pipeline merely to satisfy a framework contract.

Component registration owns cross-component coordinate consistency. Data
passed through ``add_model_data`` / ``add_coords`` is checked by the model's
``CoordRegistry``, so recipes should not repeat pairwise alignment checks for
arrays that those components register. Validate scientific invariants at the
component that owns them, and explicitly validate arrays whose labels are
not shared through one registered dimension name, such as the second
observation axis of a dense aggregation-error covariance. Construct direct and
complete custom models with ``registered_model()``; ``add_coords`` rejects a
model without that registry so the ownership rule is enforced at every
labelled component boundary.

Prepared-input inventory and ownership
--------------------------------------

``RhimePreparedInputs`` is the durable, backend-neutral labelled-data
boundary. It may retain project-specific arrays that no current PyMC recipe
uses. Concrete recipes declare only the names they select, materialize those
arrays together, and pass them to components as honest named arguments.

.. list-table:: Current ``inv_inputs`` inventory
   :header-rows: 1
   :widths: 18 20 32 30

   * - Field or value
     - Kind
     - Current owner
     - Model use
   * - ``H``
     - Reusable scientific product
     - Sensitivity preparation or an external prepared-data producer
     - Selected by the standard or multisector flux component as
       ``flux_sensitivity``
   * - ``H_bc``
     - Reusable baseline sensitivity
     - Baseline preparation; its current period expansion remains there
     - Selected by the baseline component as ``boundary_sensitivity``
   * - ``mf``, ``mf_error``
     - Reusable observation data
     - Observation preparation
     - Passed as ``observations`` and ``observation_error`` to the likelihood
   * - ``min_error``
     - Reusable error-model product
     - Currently calculated by preparation
     - Selected as ``minimum_error`` by pollution-event mismatch or by another
       built-in component which explicitly opts into the historical total-error
       floor; custom likelihoods do not require it
   * - ``aggregation_error_covariance``, ``low_rank_factor``,
       ``diagonal_residual_variance``, ``aggregation_error_sd``
     - Optional reusable fixed-error products
     - Prepared data or an external scientific producer
     - The selected likelihood representation is validated as one
       ``AggregationError`` value
   * - ``mf_repeatability``, ``mf_variability``, release coordinates,
       column-prior factors, and arbitrary labelled extensions
     - Durable diagnostics or recipe extensions
     - Their scientific producer
     - Not materialized unless a concrete recipe explicitly selects them
   * - ``site_indicator``
     - Model-only derived wiring
     - ``RhimePreparedInputs`` derives and validates it from labelled
       ``(site, time)`` observations
     - The model-error recipe derives ``SigmaAlignment``; the offset component
       may also select it
   * - ``site_names``
     - Output/compatibility wiring
     - ``RhimePreparedInputs`` regenerates it from site metadata
     - Not a PyMC input
   * - ``SigmaAlignment``
     - Cohesive model-only value
     - The standard or multisector recipe derives it from ``site_indicator``,
       time, and resolved model-error options
     - Passed explicitly to the likelihood
   * - Source/sector selection and state grouping
     - Cohesive model-only mapping
     - The multisector flux component validates source labels and derives its
       state names beside the consuming recipe
     - Never stored in a generic model-input context

The current preparation ownership of ``min_error`` calculation and boundary
period expansion is explicit rather than accidental. A concrete recipe does
not materialize ``min_error`` unless its selected mismatch component owns the
floor. Moving either preparation equation requires its own scientific parity
change. An externally cached sensitivity,
including a Verification Games ``fp_x_flux`` projection, can be installed as
labelled ``H`` at this durable handoff. Its Dask graph and provenance remain
borrowed until a selected flux recipe reaches the named PyMC materialization
boundary.

State and source selection
--------------------------

Sector/source validation and state-vector grouping are genuine reusable
scientific operations. They should be implemented as plainly named functions
over labelled data and retained artifacts. They must remain independent of a
compiler plan so nested-domain, multisector, and linked-tracer recipes can use
them directly.

Different resolutions, basis partitions, sources, and observation channels
must retain explicit labels. Do not encode these meanings only in variable
suffixes, array positions, or generated backend names.

Scientific variable roles
-------------------------

Postprocessing should identify quantities by a small backend-neutral scientific
role, not by a PyMC variable name. Roles such as ``modelled_concentration``,
``pollution_concentration``, ``baseline_concentration``, ``flux_scaling``, and
``model_data_mismatch`` are analogous to project-level CF ``standard_name``
values: they describe meaning while allowing each backend to use convenient
local names.

A model or backend adapter may provide a plain mapping from these roles to its
actual variables. This is an output and postprocessing contract, not a semantic
model or compiler manifest. Keep the vocabulary small, documented, and driven
by real consumers. Document one scientific definition and the dimensional or
unit expectations needed by consumers for each role. The role identifies a
quantity; it does not prescribe a PyMC name, analytic-result field, storage
layout, or execution step.

Preparation should remain backend-neutral through labelled scientific inputs
and coherent-reduction products. Cross an explicit boundary only when a
backend-specific representation is required. PyMC and analytic-Gaussian
builders should be able to share preparation and scientific role definitions
without pretending that their internal values or execution lifecycles are the
same.

Numerical boundaries
--------------------

The repository's xarray ownership rules still apply. Inputs are borrowed and
may be Dask-backed. Copying, eager computation, densification, persistence, and
serialization must happen at named boundaries and must not be hidden behind a
property or model framework.

Further guidance is in
``docs/development/validation_and_xarray.rst``,
``docs/plans/numerical_data_ownership_and_execution_boundaries.md`` and the
active ``run_rhime`` plan in
``docs/plans/run_rhime_readability_and_modifiability.md``.
