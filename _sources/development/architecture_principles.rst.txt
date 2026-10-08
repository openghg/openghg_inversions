Architecture principles and recipe boundaries
=============================================

Use these principles when deciding where scientific code belongs and what a
new model recipe must make explicit. They complement
:doc:`rhime_model_development` and :doc:`validation_and_xarray`: the runner
shows execution order, the concrete model shows scientific composition, and
shared components own their scientific or technical contracts.

These are design and review requirements. They do not assert that every
existing route already satisfies them or that a common staged interface has
been implemented. Existing public behaviour and artifact contracts remain
subject to their explicit compatibility policy.

Responsibilities follow reasons to change
-----------------------------------------

Give each module a coherent scientific responsibility or technical contract.
Assess that responsibility through concrete changes: a prior support policy,
an observation-alignment rule, an artifact format, or a reporting convention.
Keep decisions that must change together nearby; separate independently
changing decisions when their coupling obscures either one.

The single responsibility principle applies directly to modules. One
responsibility can require several functions and equations. A recipe owns the
composition of a scientific model, even though its runner calls preparation,
construction, inference, and output operations. A large function or module is
a reason to inspect readability, not sufficient evidence of mixed ownership.

Examples of useful ownership include:

* ``positive_prior_args`` owns positive-support prior policy beside prior
  construction, so individual likelihoods do not repeat distribution lists.
* ``SigmaAlignment`` represents the relationship between observations and
  latent site/period sigma values. Its construction and application belong to
  the same concept.
* A cached-sigma recipe keeps its graph, cache-update order, matched sampler,
  and specialized prediction policy together. Shared numerical step mechanics
  can still have a separate owner.

Hide representation details that callers need not know, such as storage
encoding and backend parameter translation. Keep scientific assumptions and
composition visible. An interface should reduce the knowledge required of its
caller; shortening a signature with an unrelated bundle of context does not
achieve that.

Prefer ordinary functions for components. Use a domain object when its shared
invariants or lifecycle justify it: a coherent reduction result groups related
moments, operators, and residual covariance from one native model; a covariance
operator can own parameters and cached factors. Such objects do not require a
general component hierarchy. PyMC component functions legitimately construct
model state; the concept-oriented functional core does not promise universal
purity.

Share policy across equivalent routes
-------------------------------------

Distinguish independent scientific recipes from multiple execution routes
through one recipe:

* Different recipes may retain similar, short orchestration sequences while
  their preparation, state sharing, observation channels, or outputs evolve
  independently. Extract shared components when their equations and option
  meanings agree, as described in :doc:`rhime_model_development`.
* Equivalent full, prepared-input, and staged routes through one recipe
  should reuse its canonical scientific operations. Short wrappers can remain
  explicit about entry points, authentication, checkpoint persistence, and
  reporting.

Calling the same low-level helpers from separate orchestrators is insufficient
if each orchestrator independently decides retained-site policy, model inputs,
or result interpretation. Give that decision an authoritative owner. Preserve
intentional differences in supported routes explicitly; consolidation must
identify any behaviour correction separately from structural moves.

For example, accepting a valid retained subset of requested sites is a
scientific preparation policy. The decision and corresponding per-site
metadata alignment should not change accidentally with the execution route.
Changing that policy requires scientific regression evidence and release
documentation, even when motivated by an ownership refactor.

Six responsibility layers
-------------------------

The six layers provide a design account of a recipe's handoffs. They are not
mandatory packages, six required methods, or an adjacent-layer-only import
chain. Recipes coordinate the relevant operations; scientific operators may
serve both preparation and reconstruction. Persistence can occur at a
checkpoint between other layers.

For each supported route, identify the concrete owners and handoffs below.
An owner may be an existing shared function or a recipe-specific operation.
State unsupported capabilities explicitly; a prepared-input-only recipe need
not invent acquisition or a complete staged workflow.

.. list-table:: Responsibilities a recipe author must account for
   :header-rows: 1
   :widths: 20 45 35

   * - Layer
     - Recipe design question
     - Boundary to preserve
   * - Inputs and acquisition
     - Which scientific inputs are required, and which providers or external
       values supply them?
     - Retrieval mechanics are separate from the equations that consume the
       resolved inputs.
   * - Scientific preparation
     - Which filtering, alignment, support, basis, sensitivity, uncertainty,
       and reduction operations are required?
     - Prepared values retain labels, units, assumptions, provenance, and
       enough phase information to identify remaining work.
   * - Model construction
     - Which states, couplings, mean equations, and likelihood are built?
     - Materialization and scientific composition are visible; construction
       establishes the meaning of outputs it supplies.
   * - Inference and checks
     - Which sampling and predictive operations are supported, and which
       choices must stay coupled to the graph?
     - Sampling mechanics, diagnostic calculations, and acceptance policy
       have explicit owners. A diagnostic need not require a live model.
   * - Scientific reconstruction
     - Which posterior quantities and retained artifacts recover each
       supported scientific quantity?
     - Supported ordinary reconstruction can use samples and matched artifacts
       without rebuilding the producing graph.
   * - Products and persistence
     - Which writers and codecs consume the quantities, and what must survive
       saving and reopening?
     - Quantity meaning and artifact identity precede format adaptation;
       storage layout does not choose the scientific model.

These responsibilities do not imply that every recipe supports every product
or that all analysis is graph-free. Generating new predictions or omitted
backend deterministics can require an explicitly supported model replay route.
Specialized graph and sampler choices remain coordinated by their recipe.

Specify the semantic handoff
----------------------------

For an important boundary, document:

* what the consumer needs and what the producer guarantees, including labels,
  units, scientific assumptions, and already-completed preparation;
* which owner validates independently supplied values and what later code may
  trust;
* whether arrays are borrowed or owned and which operation computes, copies,
  mutates, or writes them;
* supported return values and failure behaviour; and
* the operations a consumer can perform without calling back into the producer.

Use existing labelled values, explicit parameters, and small justified domain
objects to carry that information. A signature alone does not establish the
contract, and every documented invariant does not need another runtime check.
Follow :doc:`validation_and_xarray` for boundary validation and numerical
ownership; ordinary access must not hide execution or copying.

Phase meaning is especially important for replay. An ordinary merged-data
cache precedes filtering, while the staged merged checkpoint has already been
filtered. Resume from known producer/provenance or an explicit phase choice;
similar file contents do not justify guessing whether filtering remains.

Durable output contracts illustrate information that must outlive its
producer. Stored scientific roles and matched prepared/posterior identities
allow supported output reconstruction without a live graph. Their schema and
identity encoding must remain explicit even when runtime configuration
records or package locations change. Serializing an incidental Python object
layout must not silently redefine a durable contract.

Make consumer interfaces explicit
---------------------------------

Define an interface around the operations an actual consumer needs. A staged
caller can depend on a small interface implemented by participating recipes,
while scientific builders and direct runners retain explicit scientific
arguments. Select the concrete implementation at the calling boundary and
forward resolved values to its operations.

A structural ``typing.Protocol`` is appropriate when it documents and checks
an existing shared calling contract. A module of ordinary functions can satisfy
such a protocol without inheriting from it or registering an implementation.
Keep the interface definition lightweight and independent of concrete recipe
imports. Type checking establishes callable shape; documentation and focused
tests establish phase meaning, scientific guarantees, ownership, and errors.

Scope the common staged interface to its adopters. For example, a consumer of
``prepare``, ``prior_predictive``, ``sample``, and ``postprocess`` can require
those operations from the workflows it supports. That does not make the whole
suite a prerequisite for an independent builder or prepared-input-only runner.
Do not require dummy methods or unrelated sampler/output settings. Introduce
smaller interfaces when an actual consumer needs a smaller capability.

Implementations must meet the common guarantees their callers rely on. A
common signature does not make all scientific models interchangeable or erase
documented family-specific behaviour. In particular, preserve intentional
differences in readiness exception handling, supported outputs, and replay
versions. Runtime discovery, a generic execution loop, and a mutable lifecycle
are not necessary consequences of an explicit interface.

Keep the scientific contract distinct from the staged calling contract. The
custom likelihood interface, for example, receives a completed mean and
explicit observation/error inputs; it need not know how the recipe assembled
flux, boundary, and offset terms. A checkpoint wrapper instead needs artifact
paths and invocation choices, then calls the recipe's scientific operations.

Make dependency boundaries observable
-------------------------------------

Record allowed dependencies by responsibility:

* Recipes select and compose reusable components. Reusable numerical and
  model-building components do not depend on concrete recipes.
* General reconstruction consumes scientific values and contracts without
  importing a recipe or requiring a live graph. Family-specific output
  interpretation stays with the family and calls shared output operations.
* Shared artifact/path/digest helpers own mechanics. A recipe owns the artifact
  versions and scientific compatibility decisions it supports.
* Neutral diagnostic calculations own their mathematical results; recipe or
  stage policy owns acceptance thresholds and reporting decisions.
* Implementations import shared mechanics from their owners, rather than from
  a public dispatcher that imports the implementations.

Check package initializers as well as direct imports. A new namespace can
improve navigation while still loading unrelated families or a model backend.
Import isolation changes must preserve required backend initialization and
established public object identity.

Validate the boundaries affected by a change with focused evidence. Useful
checks include prepared execution with acquisition/preparation disabled,
supported graph-free replay with construction/sampling disabled, in-memory
execution with checkpoint writers disabled, and fresh-process helper imports
without unrelated families or backends.

For equivalent routes, compare scientific inputs and retained metadata,
deterministic terms or log probability at controlled parameters, matched
sampling policy, and products from the same posterior. Mocked call order alone
does not establish scientific parity; independent stochastic trajectories
need not match. Use independent small numerical oracles for equations and
separate checks for ownership and lazy execution when those contracts matter.

Review a new recipe using its named runner, concrete builder, and a short
mapping of supported routes to these owners and handoffs. That mapping can be
ordinary documentation beside the recipe; it need not drive execution. Ask
what can be understood, imported, executed, or reconstructed independently,
and which evidence demonstrates that independence.

Background
----------

The longer research record, with repository examples and the distinction
between current behaviour and ongoing designs, is
``docs/plans/architecture_principles_research.md``. The six-layer design and
shared-route work are discussed in `PR 764
<https://github.com/openghg/openghg_inversions/pull/764>`_ and `PR 787
<https://github.com/openghg/openghg_inversions/pull/787>`_. Their proposed APIs
and migrations remain separate from these design principles.

The relevant foundations are `Martin's module-level single responsibility
principle
<https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html>`_,
`Parnas's information hiding
<https://www.cs.tufts.edu/comp/150FP/archive/david-parnas/criteria.pdf>`_,
`Ousterhout's modular design
<https://web.stanford.edu/~ouster/cgi-bin/cs190-winter18/lecture.php?topic=modularDesign>`_,
and `PEP 544's structural protocols
<https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols>`_.
The boundary rules above apply these ideas to this project's scientific and
numerical contracts.
