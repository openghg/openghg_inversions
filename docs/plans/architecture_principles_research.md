# Architecture principles for readable scientific software

Status: research and proposed review guidance; this does not change project policy.

The accompanying [developer guidance](../development/architecture_principles.rst)
distils these findings into review requirements. This research record retains
its dated evidence and proposed wording; it is not a second policy source.

Research date: 2 October 2026. Repository evidence: commit
`069deaf80fa05ded48318e6cb83e0c5100169516`. Source links below refer to that
worktree snapshot; line numbers may move as the implementation develops.
Ongoing design evidence also includes PR #764 at `6b1a2149` and PR #787 at
`46be188d`, inspected on the research date.

## Purpose and authority

This note investigates principles that help maintainers and atmospheric
scientists understand, change, and review OpenGHG Inversions. It concentrates
on module responsibilities, scientific composition, and numerical boundaries.
The aim is to explain and strengthen the current design, with concrete examples
of good practice and possible improvements.

The authoritative starting points are:

- [Developing RHIME models](../development/rhime_model_development.rst).
- [Validation and labelled-array patterns](../development/validation_and_xarray.rst).
- [Numerical ownership and execution guidance](numerical_data_ownership_and_execution_boundaries.md).
- The active [readability and modifiability plan](run_rhime_readability_and_modifiability.md)
  and [model-family expansion plan](rhime_model_family_expansion.md).

The older semantic-compiler plans are superseded production architecture.
Historical problem statements and proposed file trees are not evidence that
the current implementation still has those problems or that layout. Examples
here were checked against current source and selected tests. This is an
explanation and research proposal, not an exhaustive correctness audit or a
new delivery roadmap.

The main conclusion is that **change ownership, scientific readability, and
explicit boundaries form a coherent architectural approach**. Most of its
substance already exists in the developer guidance. Established principles
give us useful names and review questions, provided we retain their limits.

### Ongoing design work changes the emphasis

[PR #764](https://github.com/openghg/openghg_inversions/pull/764) is an open
draft reference prototype split into landing PRs #772–#776. At inspection,
[#772](https://github.com/openghg/openghg_inversions/pull/772) had merged;
[#773](https://github.com/openghg/openghg_inversions/pull/773),
[#774](https://github.com/openghg/openghg_inversions/pull/774),
[#775](https://github.com/openghg/openghg_inversions/pull/775), and
[#776](https://github.com/openghg/openghg_inversions/pull/776) remained open.
Its [walkthrough](https://github.com/openghg/openghg_inversions/blob/6b1a21494c334f299bdce8ea37db7f32029f4ed6/docs/development/rhime_six_layer_prototype.rst)
and [architecture plan](https://github.com/openghg/openghg_inversions/blob/6b1a21494c334f299bdce8ea37db7f32029f4ed6/docs/plans/rhime_end_to_end_architecture.md)
therefore contain a mixture of landed behaviour, prototype implementation, and
deferred design. In particular, the proposed `recipes` and `model_components`
namespaces are not the source layout of this note's worktree.

[PR #787](https://github.com/openghg/openghg_inversions/pull/787) is an open
draft planning change, with no implementation. Its
[design](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/design.md)
and [behavioural contract](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/specs/recipe-workflow-contract/spec.md)
identify a problem that namespace changes alone do not fix: equivalent routes
through one recipe independently compose scientific operations and can drift.

| Identified architectural issue | Evidence and status | Principle it sharpens |
| --- | --- | --- |
| Saved outputs previously needed a reconstructed live graph to recover roles | #764's durable output binding; the #772 slice is merged and corresponding contracts exist in this snapshot | Preserve the semantic handoff needed by the consumer, independently of the producing process. |
| Acquisition, durable input values, complete recipes, reusable graph functions, inference diagnostics, and predictive scores have mixed ownership | #764's proposed owner separation; remaining landing slices were open | Assign responsibility by the contract or scientific decision that changes. |
| Full, prepared, and staged routes independently compose the same recipe | #787's central design issue; retained-site divergence is also visible in current source | Share scientific policy across equivalent routes, even when short route wrappers remain explicit. |
| Checkpoints can represent different phases despite similar file content | #787 distinguishes ordinary pre-filter merged caches from filtered staged checkpoints | A handoff must state what work has already happened and what remains. |
| A uniform staged interface could become compulsory for every new model | #787's revised scope permits direct builders, prepared-only execution, and partial adoption | Interfaces should serve actual consumers without imposing unrelated capabilities. |
| New package names still eagerly load unrelated families and backends | #764 explicitly records import isolation as follow-up work | Verify actual dependencies and effects, rather than infer them from directory names. |

These are existing workstreams, not newly discovered tasks. The research
provides language for their architectural reasoning. It does not treat draft
acceptance criteria as implemented behaviour or repeat their migration plans.

## 1. Give each module one coherent responsibility, assessed through change

The single responsibility principle (SRP) applies directly to modules. Robert
C. Martin's formulation explicitly refers to a software module, and explains
change through the stakeholder concern to which it responds. It does not
depend on Python modules being objects. Parnas's earlier account likewise
treats a module as an assignment of responsibility rather than a particular
language construct. [Martin, *The Single Responsibility Principle*](https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html);
[Parnas, *On the Criteria To Be Used in Decomposing Systems into Modules*](https://www.cs.tufts.edu/comp/150FP/archive/david-parnas/criteria.pdf).

For this project, the useful adaptation is:

> A submodule should own one coherent scientific policy or technical contract.
> Keep decisions that change together nearby; separate independently changing
> decisions when their coupling makes either harder to understand or modify.

This is a project interpretation, not a quotation from either author. A
"reason" means a source of requirements: the meaning of a prior, observation
alignment policy, an artifact format, or a reporting convention. A bug fix and
a refactoring are kinds of edits, not additional scientific responsibilities.
The same maintainer may own several independent concerns.

**Already working well.**
[Positive-prior policy](../../openghg_inversions/models/priors.py#L77) lives
beside prior construction. Site-sigma, scalar-sigma, and fixed-OU likelihoods
reuse it. Extending the permitted prior families has an identifiable owner;
each likelihood need not maintain its own distribution allow-list.

Similarly, [SigmaAlignment](../../openghg_inversions/sigma.py#L74) owns the
relationship between observation labels and latent site/period sigma values.
Constructing indexes, retaining labels, and applying the mapping are several
operations serving one responsibility.

PR #764 adds a useful example of separation: neutral convergence summaries,
scientific predictive scores, and acceptance thresholds can all be used after
sampling, but answer different questions. The prototype assigns the first to
inference, the second to postprocessing, and leaves acceptance policy with the
recipe or stage. Being used at the same time is not a shared responsibility.
[Prototype inference boundaries](https://github.com/openghg/openghg_inversions/blob/6b1a21494c334f299bdce8ea37db7f32029f4ed6/docs/development/rhime_six_layer_prototype.rst#L274).

**The important counterexample.**
[The standard recipe](../../openghg_inversions/rhime/standard.py#L152)
contains a concrete model and a procedural runner. Retrieval, model
construction, sampling, and output calls do not automatically give this module
four conflicting responsibilities. Its responsibility is composing the
standard scientific recipe; reusable implementations retain their own owners.
"Everything related to RHIME" would be too broad a justification, but
"the composition and contract of this named recipe" is useful and reviewable.

**Review question:** name two plausible changes. Would one require knowledge
of the other's policy? If yes, is that dependence scientifically necessary?
Updating a caller, documentation, and tests together is not itself evidence of
poor modularity. Neither file length nor number of functions answers this
question.

## 2. Hide representation decisions while exposing scientific assumptions

Parnas recommends organizing modules around difficult or changeable design
decisions, shielding other modules from those decisions. The goal is to limit
what callers must know about an implementation. [Parnas's paper](https://www.cs.tufts.edu/comp/150FP/archive/david-parnas/criteria.pdf).

Our application is to hide storage details, positional encodings, legacy
spellings, and backend parameter translation behind explicit scientific
contracts. The equations, assumptions, selected covariance representation,
and composition order remain available to the scientist.

**Already working well.**
[CO2 configuration loading](../../openghg_inversions/rhime/co2/configuration.py#L612)
reads TOML into a mapping; the nearby resolver interprets a format-neutral
mapping. [The concrete CO2 model](../../openghg_inversions/rhime/co2/co2_model.py#L86)
accepts resolved scientific values. A file-format change need not change its
concentration equation. Loading and resolution can remain in the same module:
separate contracts do not always require separate files.

Another example is [OutputContract](../../openghg_inversions/postprocessing/contracts.py#L16),
which maps scientific roles to concrete variables and carries a versioned
serialization contract. [Output-view construction](../../openghg_inversions/postprocessing/output_views.py#L17)
binds the prepared data, trace, and contract without constructing a live model
or writing files. This is a useful boundary between model construction and
later reconstruction. It is not proof that every output path is completely
backend-independent.

PR #787 extends this to compatibility: a new runtime configuration record must
not redefine saved identities merely because its Python fields changed. Its
design preserves existing family identity projections rather than serializing
the incidental record structure with `asdict`. This is information hiding
across versions: a durable contract has its own owner and lifecycle.
[Draft identity and replay decision](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/design.md#6-keep-identity-authentication-and-replay-ownership-explicit).

**Useful improvement.** For a shared operation, document both the decision it
owns and the knowledge callers can ignore. For a likelihood, this might be
the distribution and covariance construction; it should not include silently
deciding which physical terms belong in the supplied mean.

**Limit.** Information hiding does not justify an opaque "build everything"
function. Whether an oxidative ratio is fixed or inferred changes the
scientific model and must remain visible. A shorter call is not an improvement
if the scientist must inspect a hidden framework to discover the assumptions.

## 3. Optimize decomposition for reader understanding and visible composition

Ousterhout's account of modular design evaluates how much useful functionality
an interface provides relative to the complexity it exposes. Very small
modules can create more interfaces without removing much knowledge from their
callers. Bernhardt's functional-core/imperative-shell design separates
value-producing logic from interactions with external state.
[Ousterhout, *Modular Design*](https://web.stanford.edu/~ouster/cgi-bin/cs190-winter18/lecture.php?topic=modularDesign);
[Bernhardt, *Functional Core, Imperative Shell*](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell).

RHIME's existing version is deliberately a **procedural shell with a
concept-oriented functional core**. The runner shows execution order; ordinary
functions express recognizable scientific operations. This reconciles visible
procedural code with Parnas's concern about decomposition: execution order
belongs in the runner, while implementation ownership follows the decisions
being made. A flowchart need not dictate one module per box.

**Already working well.**
[The standard runner](../../openghg_inversions/rhime/standard.py#L527) visibly
retrieves, filters, builds the basis and sensitivities, assembles inputs,
materializes, builds the model, samples, and creates outputs.
[The CO2 graph](../../openghg_inversions/rhime/co2/co2_model.py#L198) then tells
a different, mathematical story: retained state, sensitivity application,
affine flux contribution, boundary, offset, complete mean, likelihood.

A composite baseline or likelihood can be a good component even when it
contains several operations. Its interface should remove implementation
knowledge from the caller while preserving its scientific meaning. Extracting
a tiny unnamed step is less useful if readers must open it merely to
understand the next line.

**Review question:** can a scientist find the complete recipe and equation,
then replace the relevant component by reading one nearby implementation?
This is a task-based criterion, not a line-count limit.

**Limits.** PyMC construction adds state to a model context; these functions
are not universally pure. There is no reason to invent a pure intermediate
model language to satisfy the slogan. Nor is a "deep module" permission for
a giant context object: it may shorten a signature while concealing the real
interface.

## 4. Expose narrow extension contracts through ordinary callables

An extension should receive the scientific values it needs, with documented
assumptions and outputs. This is a practical application of information hiding
and low coupling. It supplies much of the benefit commonly sought through
dependency injection without requiring a container or inheritance hierarchy.

**Already working well.**
[The custom likelihood call](../../openghg_inversions/rhime/_model_building.py#L113)
receives the completed mean, observations, reported error, aggregation error,
output dimension, and explicit custom options. It does not have to discover
how flux, boundary, and offset terms are stored. The extension's returned
tensor and required model variables are checked where control returns to the
package.

[The existing contract test](../../tests/test_rhime.py#L987) covers both
standard and multisector models and checks that the mean includes pollution,
boundary, and offset. This is an especially useful abstraction: it removes
forward-model knowledge that a likelihood need not own.

**Concrete distinction.** Replacing an observation distribution while keeping
the forward model can use the likelihood callable. Adding a second observation
channel with shared latent states changes preparation and composition and can
justify a named recipe. The current developer guide already draws this line.

**Useful improvement.** Document each public extension with two examples:
what can change through this contract, and what structural change needs a new
recipe. This is more helpful than simply describing a function as extensible.

PR #787 supplies a further interface-segregation test: a new prepared-input-only
recipe should be able to expose its builder and direct runner without dummy
acquisition, checkpoint, or output methods. Its hypothetical MAP operation
likewise tests whether local scientific work can reuse a sampler-free builder.
Neither example requires implementing a new optimizer or recipe to prove the
architecture. A common staged contract can remain useful for its actual
adopters. [Draft design, decisions 2 and 7](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/design.md).

**Limits.** The existing likelihood contract is PyMC-specific and requires
canonical variables such as `y` and `epsilon`; an ordinary callable is not
an unrestricted callback. Explicit custom options at this boundary are also
different from forwarding an unexamined configuration dictionary through
every stage. Introduce extension points for demonstrated scientific variation,
not every conceivable internal replacement.

## 5. Share scientific knowledge; let abstractions follow demonstrated reuse

The original DRY principle concerns an authoritative representation of
knowledge. It is broader than removing repeated lines. Metz explains how a
premature abstraction can accumulate incompatible cases, while Fowler's YAGNI
argues against carrying speculative functionality and flexibility. Fowler
explicitly distinguishes that from useful refactoring and testability.
[Hunt and Thomas, *Pragmatic Programmer Tips*, tip 15](https://pragprog.com/tips/);
[Metz, *The Wrong Abstraction*](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction);
[Fowler, *Yagni*](https://martinfowler.com/bliki/Yagni.html).

**Already working well.** Standard and multisector runners preserve readable
orchestration while sharing actual preparation, materialization, likelihood,
and sampling operations. The [standard module's introduction](../../openghg_inversions/rhime/standard.py#L1)
states that this duplication is intentional. The developer guide's extraction
criterion is particularly good: the same equations **and option meanings**
must occur in at least two production recipes before extracting their shared
component.

This distinguishes three cases that superficially all look like duplication:

- Repeating a positive-prior allow-list duplicates scientific policy and risks
  inconsistent fixes. Keep the shared owner described above.
- Repeating a short prepare/build/sample sequence can preserve independent
  recipes. Similar verbs do not establish identical preparation, latent-state
  sharing, observations, or output meaning.
- Reimplementing the scientific phases for full, prepared, and staged routes
  of the same recipe duplicates policy. Those routes should enter the same
  canonical operations at their appropriate handoff, while retaining short,
  readable wrappers for their distinct I/O needs.

**An identified problem, not just a hypothetical warning.** PR #787 documents
drift in retained-site handling. The current
[staged preparation](../../openghg_inversions/rhime/_standard_stages.py#L195)
calls shared preparation helpers but independently rejects missing requested
sites. [Its test](../../tests/test_staged_workflow.py#L247) supplies only TAC
from retrieval when TAC and MHD were requested, and expects rejection. The
full runner instead supports valid retained subsets and reconciles its run
provenance to prepared sites. Sharing low-level helper names has not made the
scientific policy the same.

The draft proposes shared canonical operations and explicitly corrects the
staged behaviour for acquisition, compatible cache reload, and filtering
losses, including per-site metadata alignment and empty-set rejection. It also
preserves intentional family differences such as readiness exception handling.
Thus reuse requires naming the common semantics, not forcing all families to
behave identically. [Draft design, decisions 1, 4, and 5](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/design.md).

**Useful improvement.** Describe the shared knowledge before approving an
extraction. If the explanation is only "these blocks look similar," the
abstraction has not yet earned its place. If callers need growing sets of
unrelated flags, revisit whether their meanings have diverged.

**Limits.** Two users are evidence, not an automatic extraction trigger.
Small duplication is not permission to fork a validated numerical equation
indefinitely. Conversely, an already-required reusable numerical component
does not need to wait for an arbitrary third caller. Modifiability and
correctness remain present requirements.

## 6. Use domain objects for shared invariants and meaningful capabilities

Names and representations should follow the scientific domain. The
ubiquitous-language principle calls for a precise vocabulary shared by
developers and domain experts. We can use that practice without adopting an
entire domain-driven-design or object-oriented architecture.
[Fowler, *Ubiquitous Language*](https://martinfowler.com/bliki/UbiquitousLanguage.html).

**Already working well.**
[CoherentGaussianReduction](../../openghg_inversions/coherent_reduction.py#L31)
contains the retained prior, effective observation operator, native observation
mean, affine intercept, and unresolved covariance. These quantities are
jointly derived from one native model. They belong together because their
relationship matters scientifically, rather than because seven values would
make an inconvenient return statement.

[Native covariance actions](../../openghg_inversions/native_covariance.py#L77)
provide a useful behavioural example. Applying a covariance and solving with
its inverse are distinct capabilities. The contracts distinguish a
positive-semidefinite action from a positive-definite invertible action.
Concrete covariance objects can own the parameters, coordinate information,
and cached factors needed to implement those operations consistently.

These are useful forms of interface segregation and semantic substitutability:
a consumer should request the capability its mathematics requires, and an
implementation must preserve the labels and mathematical meaning promised by
that capability. Matching a method name alone is insufficient. This is our
interpretation of the existing contracts, not a proposal for more interfaces.

**Concrete example.** Splitting the reduction into unrelated getters for a
mean, an operator, and residual covariance would make it easier to combine
products from inconsistent native models. Keeping the result together improves
cohesion even though it has several fields.

**Limits.** This does not justify generic `Context`, `Manager`, or `Strategy`
objects to organize every workflow. Dataclasses should represent recognizable
results or durable boundaries. An ordinary class can be appropriate for a
covariance operator with owned computational state. Neither a domain name nor
`frozen=True` proves that an object or its array payloads are immutable.

## 7. Establish semantic contracts at the boundary that owns them

King's "parse, don't validate" emphasizes retaining the information gained
when accepting an input, rather than discarding it and repeating checks later.
Its strongest guarantees depend on static types; the useful Python adaptation
is boundary normalization into a clear canonical representation.
[King, *Parse, don't validate*](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).

**Already working well.**
[The public Gaussian reduction](../../openghg_inversions/coherent_reduction.py#L103)
transposes and exactly aligns independently supplied inputs before
materialization. It then passes the resulting arrays and jointly constructed
products into a nearby equation kernel. That kernel does not have to distrust
every intermediate it has just received from its own caller.

Labels and units are part of this contract. Two arrays can have identical
shapes while their sites are ordered MHD/TAC and TAC/MHD. Covariance row and
column dimensions have different names but must describe the same ordered
labels. Unit conversion must change quantities, rather than just replace a
`units` string. The current validation guide assigns coordinate alignment to
xarray, model registration to `CoordRegistry`, and dimensional conversion to
Pint.

**Useful improvement.** For each new independent input, identify where it
becomes canonical and what later code may assume. If several downstream
functions reinterpret the same labels or configuration choice, review whether
the owning boundary has preserved enough information.

PR #787 demonstrates that a contract also needs **phase meaning**. An ordinary
merged-data cache precedes filtering; a staged merged checkpoint has already
been filtered. Reusing one as the other can repeat or omit scientific work.
The draft uses known producer/provenance or an explicit Python phase choice,
with no need for a new generic workflow-state object. File shape alone is
insufficient to identify the correct resume operation.
[Draft design, decision 3](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/design.md).

**Limits.** "Once" means once for the relevant invariant at its owning
boundary. A loader, direct public API, custom extension, or composition of
independent inputs can establish a new boundary. Exact xarray alignment does
not establish unit compatibility or guarantee that every dimension has an
index. Numerical failures such as an unusable Cholesky factorization can be
left to the operation that needs the invariant. This principle does not
require a new wrapper type for every checked scalar.

## 8. Make mutation, ownership, and execution cost part of the contract

Command-query separation helps distinguish operations that observe state from
those that change it. For scientific arrays, the repository adds another
important distinction: apparently harmless access must not conceal copying,
computation, persistence, densification, or rechunking. That cost rule is our
scientific-computing adaptation, not the original definition of command-query
separation. [Fowler, *Command Query Separation*](https://martinfowler.com/bliki/CommandQuerySeparation.html).

**Already working well.**
[materialize_pymc_inputs](../../openghg_inversions/rhime/materialization.py#L18)
names the eager boundary, selects the arrays needed by the recipe, and
computes their payloads and lazy auxiliary coordinates together. It returns a
new dataset arrangement while retaining the borrowed prepared inputs and
unselected products. [Its focused test](../../tests/test_rhime.py#L1376)
checks those ownership and execution properties.

Computing related products together is more than stylistic neatness: Dask can
share intermediate work across a joint computation that separate calls may
repeat. [Dask, *Avoid calling compute repeatedly*](https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly).

**Concrete example.** Inspecting a prepared sensitivity should not unexpectedly
run its footprint graph. The workflow that knows which related arrays the
model needs should choose the execution boundary. For a small dense SciPy
problem, that choice can correctly be eager computation.

**Useful improvement.** Public array APIs should make three things discoverable:
whether data is borrowed or owned, whether it can remain lazy, and which
operation executes or copies it. This is as meaningful as documenting shape.

**Limits.** Borrowing is a supported-use contract, not deep runtime
immutability. Indexed coordinates are commonly already eager; auxiliary
coordinates may be lazy. Sparse-to-dense conversion and eager execution are
different operations. An explicit action can return useful data: there is no
need to split materialization into command/query objects or impose lazy
execution on every numerical kernel.

## 9. Demonstrate change boundaries with scientific and operational evidence

A boundary is useful when its promised behaviour can be checked without
recreating the entire application. Scientific-software guidance recommends
testing against simplified cases and other trusted evidence, and links
testability with understandable components.
[Wilson et al., *Best Practices for Scientific Computing*](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745).

**Already working well.**
[The reduction oracle test](../../tests/test_coherent_reduction.py#L116)
compares the public reduction against explicit small dense algebra for the
retained moments, operator, intercept, and residual covariance.
[The following identity test](../../tests/test_coherent_reduction.py#L175)
checks equivalent mean formulations and recovery of native observation
covariance. Separate tests verify that invalid structure fails before payload
execution and that [shared lazy inputs execute their upstream task once](../../tests/test_coherent_reduction.py#L302).

These checks establish different things. A correct numerical answer does not
prove acceptable execution or ownership behaviour. An unmodified Dask graph
does not prove the right scientific equation. Both matter to the boundary.

For the route consolidation, #787 makes parity more concrete: compare retained
metadata, selected inputs, deterministic terms and log probability at
controlled parameters, matched sampling policy, and products generated from
the same posterior. Matching mocked call order alone is insufficient, while
independent stochastic runs need not produce identical trajectories. The
retained-site correction must be documented as an intentional behaviour change
within that work. [Draft parity contract](https://github.com/openghg/openghg_inversions/blob/46be188de584b89b588c89f6b13ea96cd6409118/openspec/changes/unify-recipe-workflow-configuration/specs/recipe-workflow-contract/spec.md#requirement-scientific-parity-and-explicit-differences).

**Useful improvement.** When introducing or substantially changing a component,
state its scientific reference case and the boundary property most likely to
regress. Add focused evidence for those properties. Keep structural
refactoring separate from scientific changes where practical, so discrepancies
have an interpretable cause.

**Limits.** A reference calculation should be independent of the production
implementation being checked, even if both express the same mathematics.
Tests should not primarily freeze private helper structure or manufacture
impossible internal states. Existing orchestration-contract tests can still
be useful where ordering and supported extension behaviour are themselves the
contract. Human evidence that a scientist can locate and change a component
remains valuable; test coverage cannot establish readability by itself.

## 10. Judge decoupling by dependencies that can actually be removed

A package name communicates ownership; it does not enforce independence.
Review the imports, initialization effects, and runtime prerequisites needed
to use a capability. This is a concrete application of information hiding and
explicit effects, made especially relevant by the ongoing design work.

**An acknowledged limit.** PR #764 explicitly reports that its `recipes`
initializer still imports unrelated families and the model backend. Importing
a local artifact or domain helper therefore acquires dependencies it does not
scientifically need. The current
[RHIME initializer](../../openghg_inversions/rhime/__init__.py#L16) similarly
combines PyTensor setup with broad public imports. Renaming or relocating that
initializer would not by itself isolate those dependencies.

The prototype describes a useful acceptance test: a fresh process should be
able to import local helpers without the backend or unrelated families, and a
failure in an optional family should not prevent another family from importing.
It also requires preserving backend precision initialization and public export
identity. This makes the desired boundary observable rather than cosmetic.
[PR #764 walkthrough, import isolation follow-up](https://github.com/openghg/openghg_inversions/blob/6b1a21494c334f299bdce8ea37db7f32029f4ed6/docs/development/rhime_six_layer_prototype.rst).

**Useful improvement.** Apply the same question to execution: can prepared
execution run with acquisition unavailable, or saved-output reconstruction with
model construction unavailable? PR #787 proposes such checks alongside real
scientific parity tests. They distinguish a useful dependency boundary from
several wrappers around the same inseparable workflow.

**Limits.** This does not require making PyMC model construction independent
of PyMC, removing legitimate coupled graph/sampler behaviour, or introducing
lazy import machinery everywhere. These are targeted independence properties
for capabilities whose contracts already imply they can be used separately.

## Worked change scenarios

These are review exercises grounded in existing boundaries, not proposed code
changes. They make "one reason to change" more useful than a count of methods.

| Proposed change | Natural owner and expected impact | What the principle protects |
| --- | --- | --- |
| Permit an additional positive prior family | Prior policy beside `parse_prior`, with focused prior/likelihood checks | Several likelihoods should not acquire separate copies of the same support policy. |
| Replace the observation distribution while retaining the forward mean | An ordinary likelihood callable and its recipe selection | Acquisition and flux construction should not need to understand the new distribution. |
| Run the same standard recipe through full and staged entry points after one requested site is lost | One canonical retained-site preparation policy, with route-specific persistence wrappers | The execution route should not accidentally select different science; #787 documents the existing divergence. |
| Change a configuration file syntax without changing scientific options | Configuration loading, retaining the resolver's canonical mapping contract | File syntax should not enter the concrete model equation. |
| Add a reporting product or group components differently for a report | Output interpretation and writer, preserving component definitions | A reporting convention should not silently change the inferred mean or covariance. |
| Add linked observations with shared latent states | A named recipe, its prepared handoff, and relevant components | A structural scientific change should remain visible rather than hide in unrelated optional flags. |
| Change covariance storage or evaluation strategy | The covariance implementation, preserving the required action/solve semantics | Reduction code should depend on mathematical capabilities rather than private storage. |
| Change the optimized cached-sigma update algorithm | The matched graph, cache, and sampler composition together | An artificial layer boundary should not split choices that must change together for correctness. |

The last case is an important constraint on SRP. The
[cached-sigma runner](../../openghg_inversions/rhime/co2/co2_cached_sigma_runner.py#L93)
constructs a sigma-then-state `CompoundStep` for its matching graph and rejects
incompatible overrides. Its graph and sampler share a real reason to change:
preserving the target and cache-update semantics. Conventional labels such as
"model layer" and "sampler layer" do not make those decisions independent.

## Where formalisation would help most

The evidence supports a small addition to existing guidance, followed by links
to current examples. It does not establish a need for repository-wide
restructuring.

| Priority | Reader need and current coverage | Smallest useful addition | Review or validation |
| --- | --- | --- | --- |
| First | Maintainers need a module-boundary criterion; locality and recipes are already explicit | Define a responsibility through independent change scenarios, including the recipe and coupled-sampler counterexamples | Maintainers should try the scenarios against a real proposed change. |
| First | #787 identifies divergent routes for one recipe, while current guidance permits small duplication | Distinguish independent recipes from equivalent routes; retain one owner for shared scientific policy | Existing #787 parity work should compare retained metadata, model terms, and products, beyond call-order mocks. |
| First | Scientists need to know when a component or new recipe is appropriate; the extension rules already exist | Explain the narrow likelihood contract with one supported variation and one structural variation | Link the concrete builders and existing callable-contract tests. |
| Next | Reviewers need to reconcile SRP, DRY, and scientific readability | State that shared knowledge, meaningful interfaces, and reader effort govern extraction | Compare actual equations and option meanings before proposing shared code. |
| Next | Contributors need to distinguish good domain objects from workflow machinery | Link the reduction result and covariance action examples, including their invariants | A domain maintainer checks that the grouping reflects the mathematics. |
| Preserve | Numerical authors already have detailed ownership and validation guidance | Cross-link it rather than create a second competing checklist | Retain equation, label/unit, and execution-boundary checks where relevant. |

One bounded place to investigate during a future substantive edit is
[_model_building.py](../../openghg_inversions/rhime/_model_building.py#L355).
It contains likelihood selection and invocation as well as
[whole-model output-role mapping](../../openghg_inversions/rhime/_model_building.py#L403).
These expose plausible independent change pressures. That observation alone
does not justify a split: current graph/output locality may be useful, and
this research did not measure change history or navigation costs. A future
review can ask whether an output-contract change repeatedly forces readers
through unrelated likelihood policy, then choose the smallest useful boundary.

## Proposed short wording for the developer guide

The following is suggested policy text for discussion, not newly adopted rules:

> Give each module a coherent scientific responsibility or technical contract.
> Evaluate that responsibility through concrete reasons to change. Keep
> mathematically coupled decisions together and isolate independently changing
> policy or representation details.
>
> Keep the runner's execution order and the model's scientific composition
> visible. Extract components that reduce what the caller must understand;
> prefer explicit scientific values and ordinary callables. Use domain objects
> when their shared invariants or lifecycle justify them.
>
> Share scientific knowledge once its meaning is established. Similar-looking
> orchestration can remain separate while different recipes evolve. Equivalent
> supported routes through one recipe reuse its scientific operations; checkpoint
> wrappers add persistence at explicit handoffs. Normalize independent inputs at
> their owning boundary, preserve labels, units, and phase meaning, and expose
> materialization and ownership costs. Verify scientific equations, boundary
> behaviour, and claimed dependency isolation independently.

This wording intentionally does not adopt SOLID as a package of obligations,
require a general model framework, or impose class/function/file-size rules.
The proposed principles are useful only insofar as they improve the specific
reading and modification tasks already required by the project.

## Explicit contracts for the six-layer design

The follow-up design discussion distinguishes responsibility layers, data
handoffs, and the calling interface required by a consumer. Their shapes need
not match. The six layers describe what a recipe must account for; each answer
can name an existing shared operation, a local scientific operation, or an
unsupported capability. Persistence can occur between layers, and prepared
execution and saved reconstruction can enter at different handoffs.

Avoiding a framework does not mean leaving actual contracts implicit. A small
structural `Protocol` can make the staged caller's existing expectations
explicit without requiring recipe inheritance or registration. Python permits
modules of ordinary functions to implement protocols. Callable shape belongs
in the interface; phase meaning, scientific guarantees, effects, and failure
behaviour also need documentation and focused tests.
[PEP 544, modules as implementations of protocols](https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols).

The smallest useful formalisation is a central responsibility/boundary table,
an interface scoped to actual staged adopters, and a short mapping of each
recipe's supported routes to concrete operations and handoffs. The mapping can
be ordinary documentation. It need not become executable metadata or dictate
a universal lifecycle. The companion developer page gives the consolidated
guidance, including allowed dependencies and observable boundary checks.

## Research and validation limits

Three research subagents independently examined modularity literature,
scientific boundaries, and repository examples. Their findings were
synthesized against the current developer guidance and checked source paths.
External sources are linked next to the ideas they support; the scientific
software adaptations and proposed review rules are this note's conclusions.
The PR descriptions, pinned design documents, and #787 review discussion were
also read; the two review clarifications about retained-site losses and family
exception policy are reflected in #787's inspected draft. The initial research
did not post comments or make other changes to those PRs; the subsequent
guidance proposal can be cross-referenced in their design discussions.

Selected tests were read as evidence of intended contracts; they were not run.
No scientific result, runtime behaviour, or repository-wide compliance claim
was established by this research. The initial assessment changed no
implementation or normative guidance; the accompanying developer page is the
subsequent documentation proposal requested after discussion.
