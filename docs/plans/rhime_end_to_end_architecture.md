# RHIME end-to-end scientific integration plan

Date: 24 September 2026

Status: strengthened architectural proposal; implementation and scientific choices require their owning reviews

Audience: model authors, maintainers, scientific reviewers, and delivery owners

Implementation experiment, 29 September 2026: the
[six-layer prototype](../development/rhime_six_layer_prototype.rst) applies a
bounded part of this plan to newer `devel`. It records actual ownership moves,
graph-free staged reconstruction, compatibility, and possible smaller landing
steps. The evidence and tracker statuses below remain dated to this proposal.

## Purpose, authority, and evidence

Keep named, readable scientific recipes and make their input and output
integration explicit. The architectural problem is the work needed to connect
a new scientific composition to acquisition, preparation, inference, durable
results, and useful products. Improving graph-construction syntax alone will
not remove that repeated work.

This document combines the architectural recommendation dated 23 September
2026 with its repository-grounded review. It is a complete proposal, not an
assertion that the described interfaces or model combinations already exist.
It supports [#663](https://github.com/openghg/openghg_inversions/issues/663)
and related implementation work; it does not create a competing delivery
umbrella or assign ownership and priorities by itself.

The normative development guidance remains
[Developing RHIME models](../development/rhime_model_development.rst),
[validation and labelled arrays](../development/validation_and_xarray.rst),
and [numerical ownership and execution boundaries](numerical_data_ownership_and_execution_boundaries.md).
The [approved readability plan](run_rhime_readability_and_modifiability.md)
and [active model-family expansion plan](rhime_model_family_expansion.md)
remain the canonical detailed delivery plans. Accepted amendments from this
proposal should be incorporated there and into the existing issues. The
[proposed programme roadmap](rhime_programme_architecture_and_outcome_roadmap.md)
provides related outcome and production gates; its dated status tables are
historical evidence, not current tracker state. Linear owns current priority,
ownership, and dependencies; GitHub records implementation and review.

The static code review uses these fixed snapshots:

| Evidence | Revision | Interpretation |
| --- | --- | --- |
| Reviewed `devel` | `b09a1dab0f` | Baseline for implementation claims in this plan |
| Model-configs prototype / PR #228 | `6aeaa5f2431eeb736b3be5cc587911d2107f9d1f` | Historical experiment, separate from the later abandoned model-spec compiler |
| Nested-domain PR #704 | `38f4a7e53783e7e416e04f3510bcb9648290142e` | Includes the output-contract corrections described below |
| Authoring worktree | `fa421914ed271e2da1477431025e7dc6e5db4320` | Older than the reviewed baseline; newer code evidence must use pinned links or `git show` |

GitHub reported #704 merged on 23 September 2026 at the reviewed head. That
status update does not imply that every later `devel` change has been reviewed.
No numerical execution or benchmark underlies this architectural assessment.
Tracker mappings later in this document distinguish existing scope from
proposed acceptance additions; recording a mapping does not modify an issue.

For implementation, start with the [work packages](#delivery-work-packages)
and [acceptance matrix](#acceptance-matrix). For scientific review, use the
[composition contract](#scientific-composition-without-a-compiler) and
[coherent reduction](#coherent-reduction-as-a-shared-statistical-operation).
The [dated issue/code index](#dated-issue-and-code-evidence) connects both to
existing work and records outstanding ownership gaps.

## Recommendation and success criterion

Retain six responsibility layers, with immediate emphasis on inputs/acquisition
and products/persistence, supported by stronger preparation and reconstruction
contracts. Recipes explicitly combine scientific pieces. Companion adapters
describe the inputs those pieces need and the quantities they can provide.
Existing providers, codecs, and writers supply common mechanics.

Extract shared integration through the upcoming models. Some repeated
scientific assembly is acceptable. Repeated implementations of acquisition,
artifact verification, reconstruction adaptation, and generic output writing
should decrease. Package reorganisation and a model-spec compiler are not
prerequisites.

The success criterion is a complete supported route: coherent CH4/C2H6 and
CO2 nesting add readable scientific assembly and appropriate adapter policy
while reusing input resolution, reduction, artifact mechanics, and output
bindings. A copied prepare/build/sample/output pipeline indicates a remaining
integration gap even if each copied runner is individually readable.

Reusable public components must eventually support the same customization
performed by maintained recipes. Direct kernel calls and labelled prepared
inputs must remain possible without registration, descriptors, or a framework
lifecycle.

## Models that determine the boundaries

These are interacting scientific axes, not a promise that every combination
is meaningful. A domain describes spatial support; a source describes a
physical contribution; a channel describes an observable and its sampling
support; a state describes an inferred quantity. None is inferred from
another's name. Channel identity is not necessarily species identity: two
channels may concern the same species with different observables or support.

| Addition | Reusable composition | Explicit scientific choices | Architectural evidence |
| --- | --- | --- | --- |
| CH4 with C2H6 | Shared states, separate channel responses, coherent preparation/reconstruction when selected | Emitting sources, fixed/inferred ratios, baseline terms, prior and error assumptions | Coherent and conventional variants reuse integration without requiring separate complete workflows |
| Multisector nesting | Domain support and masking from #704, gathered source-aware basis layouts | Valid source/domain pairs, source coverage, basis per pair, state and prior relationships | Unequal basis sizes and a missing pair work without padding or private runner imports |
| CO2 nesting | Shared domain operations plus carbon source relationships and coherent products | Signed flux conventions, cross-source/domain covariance, projection and residual-error assumptions | New code mainly expresses carbon/nesting policy; native-grid output mechanics are reused |
| CO2/radiocarbon | Shared carbon states, separate sampling support and observable transformations | Abundance, delta or derived fossil CO2; isotope signatures, additional terms and error dependencies | Independent two-channel delivery with a stated measurement equation |
| Optional CO2/O2/radiocarbon | Existing carbon states plus another observable | Three-channel equations, covariance, support and supported products | Addition does not require a universal N-gas engine |
| Coherent standard/multisector | Reduction preparation, retained prior, affine response, unresolved covariance and reconstruction | Native moments/projection, exact Gaussian model or approximation, source covariance | Existing families support the chosen reduction through their established integration seams |

The proposed programme already separates CO2/radiocarbon from optional O2.
Preserve that independent delivery; do not turn a third-channel architecture
test into a prerequisite for the two-channel scientific requirement. The
programme's dated targets need confirmation by their current delivery owners.

## CH4/C2H6: reference implementations and remaining work

The existing methane/ethane prepared-input experiment is
`build_ramsden_model` and `run_ramsden_from_prepared_inputs` in
`experimental/ramsden2022/model.py`. It supplies historical comparison and
coupling semantics. It is not the proposed coherent CH4/C2H6 model and does
not supply a complete acquisition-to-output route.

The current prepared-input CO2/O2 model is the closer structural reference
for coherent linked channels. Neither reference establishes a complete new
workflow.

| Aspect | Ramsden experiment | Prepared CO2/O2 | Proposed CH4/C2H6 |
| --- | --- | --- | --- |
| States | Source states reused across methane and ethane | Shared retained GPP/TER/fossil states and channel-specific ocean states | Shared methane-source states with explicit ethane participation and channel-specific terms |
| Coupling | Fixed or sampled ratios; distinguishes molar ratios from multipliers over reference-ratio responses | Fixed signed O2-per-CO2 coupling, including spatially embedded ratios | Preserve fixed/inferred and ratio/reference-multiplier distinctions |
| Reduction/error | Separate conditional Gaussian likelihoods; no coherent joint aggregation-covariance input | Affine means and a joint Gaussian likelihood with cross-channel aggregation covariance | Coherent option follows joint structure; conventional option declares its own error assumptions |
| Observations | Unequal sites, times and counts permitted | Separate channel axes gathered into a joint layout with units | No forced intersection merely because states are shared |
| Integration | Prepared-input comparison | Preparation assembly, model and replay seam | Input adapters, durable prepared/reconstruction artifacts, products and advertised operational route |

`prepare_co2_o2_inputs` consumes already-produced sensitivities, prior
forward means, aggregation-covariance blocks and a retained prior. It validates
and assembles these products; it does not retrieve native fields and construct
the entire reduction. Extract generic channel gathering, labelled response
application and covariance handling while keeping carbon-specific validation,
signed ratios, units and GPP/TER/fossil/ocean relationships in the carbon adapter.
Renaming carbon fields is not a generic linked-channel implementation.

For fixed-coupling coherent CH4/C2H6, explicitly obtain both channels and their
transport, selected source fluxes and native prior information; apply coupling;
prepare joint reduction products; retain matching reconstruction information;
then build shared states and visible channel equations. Methane-only sources
and channel-specific boundaries remain explicit. Source participation still
requires scientific agreement.

A conventional variant may supply ordinary regional responses, priors and
basis reconstruction through the same integration seams. Omitting coherent
reduction does not establish conditional error independence. Separate
likelihood calls require an explicit independence assumption.

Deliver one chosen fixed-coupling configuration first unless the scientific
consumer requires uncertain ratios immediately. Inferred scalar/source ratios
and spatially varying inferred ratios have separate completion gates. If a
ratio changes the native response, it can change effective responses,
intercepts and unresolved covariance. Spatial coupling generally belongs
before reduction unless an equivalent reduced transformation is established.

Tracer outputs with inferred ratios use matching posterior draws of states
and ratios. A fixed spatially embedded ratio may need a paired native-flux
map; a scalar multiplier on a regional methane result is not automatically
equivalent.

## Six responsibility layers

The layers describe handoffs, not mandatory packages or a linear import chain.
Recipes coordinate them; scientific operators are used in preparation and
reconstruction.

| Layer | Owns | Existing starting points | Boundary |
| --- | --- | --- | --- |
| 1. Inputs and acquisition | Resolve choices, retrieve/reload observations, footprints, fluxes and boundaries | `rhime/params.py`, CO2 config, `inversion_data/get_data.py`, getters, scenario and xarray adapter | Explicit recipe requests; providers resolve retrieval mechanics; source-neutral labelled inputs can enter after acquisition |
| 2. Scientific preparation | Filtering/alignment, support, basis, sensitivities, uncertainty and reduction | `inversion_data/preparation.py`, `rhime/preparation.py`, family preparation, `basis/`, covariance modules | Explicit numerical inputs; prepared values retain labels, units, assumptions and provenance |
| 3. Model construction | States/couplings, responses, observable equations and likelihoods | Concrete `rhime` builders, `models/` | Visible PyMC materialization boundary; readable scientific order |
| 4. Inference and checks | Sampling, prediction, specialized steps and diagnostics | `rhime/sampling.py`, cached-sigma code, diagnostics and stages | Model/options/roles passed explicitly; recipe coordinates specialized graph/step policy |
| 5. Scientific reconstruction | Interpret posterior quantities, native-grid fields and aggregates | `InversionOutput`, basis operators, `AffineFluxMap`, country aggregation | Samples plus bound artifacts suffice for supported ordinary analysis |
| 6. Products and persistence | Save/load artifacts, basic/PARIS/legacy products | Existing codecs, object persistence, output modules and manifests | Quantities and their bindings precede format/schema adaptation |

Validate where a contract is established, then trust locally constructed
intermediates. Borrow xarray inputs, preserve labels and shared Dask graphs,
and avoid hidden computation, copying, densification or rechunking. Indexed
coordinates can normally be checked eagerly; auxiliary coordinates may be
lazy. Use `array_ops.to_dense` for dense chunk payloads and materialize related
arrays together at a named PyMC, serialization or eager-kernel boundary.

## Explicit input and output integration

### Four small responsibilities

| Responsibility | Information | Owner and limit |
| --- | --- | --- |
| Describe input needs | Scientific inputs, selection policy, species/source/domain, sampling support and metadata | Companion adapters combined explicitly by a recipe; kernels do not query stores |
| Resolve and prepare | Actual records/revisions, labelled data, units, resolved settings, alignment/support/operator products | Providers own retrieval/cache mechanics; preparation owns scientific transformation |
| Describe available quantities | Units/support, required artifacts, conditions and approximation scope | Recipe/component adapters; capabilities concern scientific quantities before formats |
| Bind quantities | State mappings, posterior variables, coordinates, reconstruction references and provenance | Recipe combines explicit bindings; reconstruction and writers consume them |

These are not four methods of a required base class. Ordinary functions,
keyword arguments and small scientific values are sufficient. Introduce a
shared type only when demonstrated consumers need the same meanings.

Input resolution may be staged. Observation metadata can determine matching
footprints, inlets and heights. Show observations -> compatible transport ->
preparation as ordinary Python calls. Reuse mature getters to preserve
platform selection, filtering, inlet handling and automatic matching. Share
retrieval only for identical resolved requests and settings, not equal channel
names. A query for the latest record is not a durable source identity.

The same preparation should accept OpenGHG-backed data, labelled xarray data
or reopened prepared products. Replay must not reconstruct OpenGHG objects
solely to pass through acquisition. Extend the current source-neutral schema
only for concrete baseline, covariance or isotope requirements.

### Handoffs and lifecycle

| Handoff | Establishes | Does not imply |
| --- | --- | --- |
| Input request | Need, selector, required support, recipe policy | A static dependency graph or resolved record |
| Resolved input | Selected records/revisions, units/coordinates, applied settings and provenance | Prepared model wiring or a live backend object |
| Prepared product | Aligned arrays, prior/error assumptions, state/operator mapping, preparation identity | That every recipe or output is compatible |
| Reconstruction binding | Required variables, support, units, artifact identity and statistical scope | That posterior samples already exist or that all native uncertainty is recovered |

Binding occurs at two moments. Before sampling, establish scientific
dependencies and validate requested capabilities as soon as inputs permit.
After sampling or reload, attach the posterior and check actual variables,
dimensions and artifact identity. Distinguish declared possibilities from
successfully bound quantities.

For example, requested native flux must fail before sampling when its missing
reconstruction map is already knowable. Additional output requests may cause
more artifacts to be retained, but must not silently alter observations,
priors or likelihood. Test that invariance explicitly.

Use explicit reconstruction functions for stored channel predictions, affine
flux, regional totals and paired tracer quantities. Do not serialize arbitrary
closures or introduce a posterior-expression language. A supported function
may have a stable adapter identity/version and explicit numerical arguments.

One posterior can expose several channel/domain/native-grid views. Preserve
common posterior identity and chain/draw coordinates across them so totals and
ratios retain dependence. Do not flatten distinct native grids into one result
array merely to satisfy a writer.

### First implementation: remove role-only graph rebuilding

The standard staged route already writes prepared products and verifies
artifact/configuration identities. Extend those mechanics rather than
introducing another manifest system.

At the reviewed baseline, `postprocess_rhime_stage` rebuilds a model to obtain
its build result. The ordinary output adapter reads variable roles and
metadata from that result. Separate this serializable information from the
model-bearing value, save it beside the posterior, and consume it during
ordinary postprocessing. This is a small first implementation with a clear
fresh-process test; it is not a complete coherent-output contract on its own.

Retain graph replay for new predictive computation and omitted deterministics
that require backend evaluation. Issue #415 explicitly includes graph replay
as well as backend-neutral reconstruction. Removing unnecessary graph
construction must not remove that legitimate capability.

## Lessons from the model-configs prototype

PR #228 recognized genuine needs: component-adjacent preparation,
channel-scoped configuration, and explicit shared dependencies. Retain that
local knowledge and sharing.

Its implementation also used constructor/signature inspection, graph-context
option inheritance, graph traversal for preparation and scientific context,
and acquisition during some data-object constructors. Observation units could
be found through surrounding graph structure. The runner saved traces,
summaries and selected attached data without establishing the complete
scientific reconstruction/product binding proposed here.

This explains where behavior becomes hard to locate; it does not establish
why scientists accepted or rejected the approach. The recorded human feedback
also welcomed parts of the format and identified missing inlet/height,
platform, filtering, error and basis behavior. Preserve those mature behaviors
through existing acquisition functions.

Avoid graph-inherited scientific defaults, signature-driven dispatch, I/O in
constructors, and reconstruction discovery through arbitrary attributes.
Declare channel transport explicitly rather than assuming a tracer can reuse
the primary gas's prepared response. The model-configs experiment and later
model-spec compiler remain separate historical attempts.

## Scientific composition without a compiler

| Responsibility | Owns | Must leave explicit |
| --- | --- | --- |
| Observation channel | Observable identity, measurements, units, times/windows, measurement uncertainty | Explanatory states and cross-channel error dependence |
| State and prior | Labelled inferred quantities, fixed/active policy, joint prior relationships | Sharing the same state versus allocating a new one |
| Forward contribution | Apply a labelled response to an existing state/expression, retaining source/domain identity | Acquisition, state allocation and inference |
| Domain support | Native grid, physical support, overlap removal and provenance | Prior relationships and statistical independence |
| Reduction products | Consistent retained prior, response, affine term, unresolved covariance and reconstruction binding | Native prior, projection, approximation and parameter dependence |
| Observable equation and likelihood | Transform physical predictions into the measured quantity and its probability model | A transport response does not dictate the observable equation |

Existing `apply_linear_sensitivity`, `add_linked_linear_component` and
`add_coherent_affine_component` supply useful seams. Build outward from them.
For a linear channel, keep the assembly visible:

$$
\mu_c=b_c+\sum_{(d,s)\in I_c}H_{c,d,s}\alpha_{d,s}.
$$

The recipe names participating domain/source pairs and the actual state
objects they use. Coupling appears explicitly in a response or state
expression. The fixed term $b_c$ includes affine contributions; sampled
baselines remain explicit contributions. This mean equation does not specify
prior or error independence.

Use gathered labelled layouts for valid pairs, with unequal basis sizes.
Extend `basis/layout.py` and existing operators; do not require a dense
channel-by-domain-by-source-by-region tensor. Logical covariance blocks do
not require every numerical representation to be dense.

Small dataclasses represent concrete scientific values; functions implement
operations; callable objects may own fitted basis state or useful caches.
Configuration selects supported choices that resolve to concrete calls. It
must not infer equations or sharing through a generic registry.

### Unobserved, fixed, and marginalized states

Make three different meanings explicit:

1. Fixed by scientific policy: a quantity has an exact stipulated value.
2. Directly unobserved: a quantity remains uncertain and may be informed through
   prior correlations or another channel.
3. Marginalized computationally: a quantity is omitted from sampling but its
   supported outputs recover the appropriate conditional distribution.

Conditioning a correlated prior on a known value also differs from replacing
that variable by a constant while retaining the marginal prior of the others.
The selected scientific meaning must determine preparation and reconstruction.

Existing `StateActivity` combines policy with compulsory zero-column removal;
correlated-state construction can subset a prior and insert fixed values into
the public state. Reusing that behavior in new correlated models needs an
explicit review. It can preserve the active likelihood but misrepresent an
omitted state's scientific output. This is not a claim that the present
CO2/O2 builder necessarily uses that pruning path.

Preserve the existing recipe's declared fixing behavior during compatibility
work. Introducing marginalization/recovery is a separately reviewed scientific
extension, not an automatic correction to every use of an activity mask.

A discriminating oracle uses zero-mean Gaussian $(x_1,x_2)$ with unit
variances and correlation $\rho$, and $y=x_1+\epsilon$ with independent
$\epsilon\sim N(0,1)$. Although $x_2$ has zero direct sensitivity,

$$
E[x_2\mid y]=\rho y/2,\qquad
\operatorname{Var}(x_2\mid y)=1-\rho^2/2.
$$

An output that fixes $x_2$ would not represent that joint posterior. For shared
states, assess participation across all contributing channel responses;
absence from ethane alone cannot remove a methane state. Zero columns alone
are insufficient grounds to discard uncertainty needed by outputs.

## Coherent reduction as a shared statistical operation

Coherent reduction changes the prior, response and likelihood together. It
is not merely a different basis matrix or an output option. Extract reusable
preparation from the CO2 workflow around the existing shared numerical
reduction; standard and multisector recipes can then assemble an explicitly
supported prepared-input alternative.

### Exact fixed linear-Gaussian contract

Let $x$ concatenate relevant native states, with $x\sim N(m,B)$, retained
state $\alpha=\Pi x$, and $C=\Pi B\Pi^\top$ invertible. Define

$$
U^*=B\Pi^\top C^{-1},\qquad Q=B-U^*C(U^*)^\top.
$$

For channel response $H_c$,

$$
\alpha\sim N(\Pi m,C),\qquad
H_{\alpha,c}=H_cU^*,\qquad
b_c=H_cm-H_{\alpha,c}\Pi m,
$$

$$
A_{cc'}=H_cQH_{c'}^\top,\qquad
y\mid\alpha\sim N(b+H_\alpha\alpha,R+A).
$$

Here $R$ is the stipulated observation-error covariance, independent of the
native state in this formulation; channel quantities are stacked consistently.
Implement inverse actions with suitable solves rather than constructing an
inverse solely because it appears in the notation. Singular retained priors
need an explicitly designed supported subspace; do not silently substitute a
pseudoinverse or jitter that changes the scientific model.

The identities underpin `reduce_native_gaussian`. They establish a consistency
contract: retained prior, effective responses, affine terms and unresolved
covariance come from the same native model and projection. Matching dimensions
alone cannot prove this. Reconstruction is bound to those same choices,
without requiring every product to share a container or file.

Current CO2/O2 accepts `CorrelatedLognormalPrior`. Its assembly is reusable,
but a retained lognormal prior plus Gaussian unresolved error is a
moment-matched closure, not exact lognormal marginalization. Record native
moments, retained prior family and approximation assumptions separately.

Four expansion rules follow:

1. A retained prior and sensitivity do not determine native covariance or
   reconstruction. Coherent preparation requires explicit native moments and
   projection; conventional non-Gaussian choices cannot silently acquire an
   exact-Gaussian interpretation.
2. Preserve joint covariance. Cross-source/domain terms affect retained and
   unresolved covariance; shared unresolved variation can correlate channels.
   Independent reductions or diagonal approximations require declared
   assumptions. Spatial non-overlap is not statistical independence.
3. Parameter-dependent responses can change every response-dependent product.
   Dependencies of native moments or projection must also be handled if present.
4. Exactness applies to fixed linear observations of Gaussian native states.
   A nonlinear isotope transformation needs its own uncertainty treatment.

### Bounded parameter-dependent coupling

First support a scientific fixed-ratio case. A subsequent scalar/source-ratio
extension can use an explicit response decomposition when valid. For example,

$$
H(\theta)=H_0+\theta H_1,
$$

with fixed $B$, $\Pi$ and $m$, prepare

$$
L_i=H_iU^*,\qquad b_i=H_im-L_i\Pi m,\qquad
A_{ij}=H_iQH_j^\top.
$$

Then assemble

$$
H_\alpha(\theta)=L_0+\theta L_1,\quad
b(\theta)=b_0+\theta b_1,
$$

$$
A(\theta)=A_{00}+\theta(A_{01}+A_{10})+\theta^2A_{11}.
$$

The same construction extends to a small sum
$H(\theta)=\sum_i g_i(\theta)H_i$, retaining all required cross-products.
Stacked responses include both primary and tracer channel blocks. This is an
implementation option, not a claim of existing support. Compare the complete
conditional log likelihood, including its parameter-dependent log determinant,
with a small native Gaussian reference. Matching means alone is insufficient.

State whether coupling parameters are independent of native flux uncertainty.
Otherwise conditional native moments can depend on them too. A spatially
varying sampled ratio is a separate milestone; do not force the first adapter
to anticipate every parameterization. Any fixed-covariance approximation must
state what it omits and have its own scientific acceptance.

### Reconstruction operation and scope

Retain the affine map

$$
\bar x(\alpha)=m+U^*(\alpha-\Pi m),\qquad
\bar f(\alpha)=F\bar x(\alpha),
$$

where $F$ applies the signed reference flux in the selected convention.
`AffineFluxMap` is the numerical starting point. Keep its reconstruction value
separate from inversion inputs and reduction products and bind it to the exact
prepared artifacts, including the authoritative retained reference state.

Record three independent descriptions:

- operation: the affine map and reference-flux convention;
- conditioning scope: the existing `retained_state_conditional` scope;
- statistical interpretation: exact Gaussian reference conditional mean, best
  affine predictor from moments, or a stipulated affine residual closure.

For an arbitrary non-Gaussian native prior, the affine expression is not
automatically its actual conditional expectation. Positive retained states
alone also do not establish every desired native-field property; such physical
constraints belong to the selected model and its review.

The map does not give the complete observation-conditioned native posterior.
For $y=Hx+\epsilon$ in the fixed Gaussian model (subtracting any known
additional offset first), let $S=R+HQH^\top$. Then

$$
E[x\mid\alpha,y]=\bar x(\alpha)
 +QH^\top S^{-1}\{y-H\bar x(\alpha)\},
$$

$$
\operatorname{Cov}(x\mid\alpha,y)=Q-QH^\top S^{-1}HQ.
$$

These equations identify additional actions/data needed by a future full-native
output. They do not expand the initial affine persistence payload or promise
that capability. Blindly adding $Q$ to uncertainty from mapped posterior
samples does not account for the observation-conditioned correction. Use a
small unresolved-contrast oracle to test any later full-native implementation.

For regional totals, contract the spatial functional with reference flux and
prolongation before broadcasting over posterior draws. The requested quantity
must state its uncertainty scope as well as grid, units and source identity.

### Numerical representations and realistic bounds

Preserve covariance actions and block structure where available. However,
the reviewed `reduce_native_gaussian` materializes native numerical inputs at
a named boundary and requests dense observation covariance; CO2/O2 assembles
dense channel blocks. Native covariance actions do not imply an entirely
matrix-free likelihood.

Specify a small dense first acceptance case and a measured memory/runtime
envelope for the intended workload. Dense observation covariance storage
grows quadratically with observation count. Add a new action/factor-based
likelihood implementation only when a demonstrated workload requires it.
Any representation change must retain covariance meaning and likelihood
normalization. Profiling and memory evidence remain work to perform.

## Nested domains and radiocarbon

### Reuse the corrected nested-domain implementation

The reviewed #704 head has distinct native grids/bases, outer-overlap removal
before projection, a visible sum of domain contributions, and two output views
over one posterior. Its one-source scope need not have expanded before landing.

The reviewed improvements are retained:

- `make_nested_inversion_outputs` lives beside the recipe and supplies ordinary
  domain views to the main PARIS path.
- Trace-to-basis state dimensions are mapped explicitly without mutating the
  original trace; a general fallback need not recognize `inner_region`.
- Domain identities, coordinate fingerprints/extents and support policy are
  retained. Provenance distinguishes caller-prepared support from overlap
  masking during acquisition.

`align_inner_merged_to_outer_observations` and
`mask_outer_merged_for_inner_domain` are already public exports. Remaining
work is to adapt these recipe-owned, `RhimeMergedData`-specific operations for
a second scientific consumer and give the reusable parts an appropriate home.
Do not describe the task as exposing nonexistent private-only operations.

The compatibility `make_nested_paris_outputs(nested_result)` wrapper still
imports the recipe adapter. Treat it as an explicit migration exception; the
main path has the intended dependency direction. Caller-supplied support
metadata still depends on a truthful account of external preparation.

For multisector nesting, select observations appropriate to the contributions,
establish native support and remove overlap, then construct bases/responses.
Keep source and domain independent, allow unequal basis counts and absent
pairs, and retain both grids in outputs. Do not assume a Cartesian product.
For CO2 nesting, combine these operations with carbon relationships and
coherent products. Independent domain priors may be a first named scientific
assumption if acceptable; common contracts must not prohibit cross-domain
covariance. Avoid private standard-nested imports and copied CO2 pipelines.

Split `rhime/nested.py` into preparation/model/runner/output modules only when
it improves ownership and reading. Preserve public imports. Reusable contracts
matter more than file counts.

### Radiocarbon observable and sampling support

Settle whether the measured quantity is radiocarbon abundance, a delta value,
a ratio, or derived fossil CO2 before freezing the model contract. Transported
total-carbon/isotope quantities, observable transformation and likelihood are
separate operations.

The forward calculation may need total CO2 at radiocarbon sampling times and
averaging windows even without CO2 observations there. A ratio of integrated
quantities generally differs from an average of instantaneous ratios. Specify
and test the measurement protocol's transformation/averaging order.

Source isotope signatures, disequilibrium, nuclear contributions and additional
parameters are model decisions, not mandatory generic channel features. If
derived fossil-CO2 data reuse measured CO2 or isotope data, represent those
dependencies and error propagation before treating the derived values as
additional evidence. Nonlinear transformations need an explicit approximation
or uncertainty propagation model; the linear-Gaussian reduction theorem alone
does not settle it.

Deliver CO2/radiocarbon independently of optional O2, then assess the concrete
three-channel combination. Shared channel operations must not require the
carbon recipe's particular source vocabulary.

## Durable, recoverable scientific artifacts

Keep scientifically distinct `RhimePreparedInputs`, `Co2PreparedInputs`,
`Co2O2PreparedInputs` and nested prepared values. Compose reusable values
instead of requiring a universal superclass. A prepared-input Python runner
does not imply that its inputs can already be reopened from a self-contained
artifact. In particular, the reviewed `Co2O2PreparedInputs` is a field-only
value and the annotated `InferenceData` returned by its runner does not supply
a native reconstruction bundle. `AffineFluxMap` likewise needs a persistence
binding rather than being assumed serializable already.

### Artifact boundaries

| Artifact | Purpose and identity | Exclusions |
| --- | --- | --- |
| Native preparation cache | Expensive basis-independent `fp_x_flux` or source-resolved products; native grid, transport/flux records, units, support and coupling identity | Not basis-bound `H`, a posterior, or a complete run bundle |
| Prepared scientific artifact | Observations/support, source/state labels, projected operators, prior/error products and assumptions | Not automatically compatible with every recipe |
| Posterior artifact | All chains/draws, sampled/effective state roles and diagnostics, prepared identity | Does not alone encode native reconstruction |
| Reconstruction artifact and binding | Exact mapping, reference flux/state, support, approximation scope and matching artifact references | No live model or serialized arbitrary closures for ordinary supported reconstruction |
| Derived product | A supported scientific quantity in a format/schema with disclosed interpretation | Not the only recoverable record of an expensive inference |

Do not duplicate high-volume raw intermediates in every output view. The
proposed programme's native-cache boundary should be reconciled with this
plan: OpenGHG owns reusable multiplication/persistence kernels, OGI owns
labelled projection and projected prepared contracts, and campaign/data
preparation owns discovery/catalog policy. Verify actual upstream capabilities
before implementation; a dated roadmap is not evidence of current upstream
support. Small runs need not persist every intermediate.

Cache identity distinguishes exact resolved records/revisions, channel and
footprint mode, domain/grid/mask, source order, site/release/time/window
selection, units/conversions, filtering, coupling/reference ratios, algorithm
and schema revision. A compatible native cache may serve more than one basis;
a cached sensitivity is basis-bound. Path existence alone is not a cache hit.
Compute payload checksums at the explicit serialization/verification boundary,
not through a property that unexpectedly computes a Dask graph.

### Reconstruction and result contract

A result binds posterior variables to scientific roles, state/operator
mappings, dimensions, units, support, prepared identity and reconstruction
scope. Avoid interpreting names such as `region`, `inner_region` or `nx` as
universal scientific roles. Agree a small canonical role vocabulary, including
boundary contributions, with compatibility aliases where needed.

One result may expose several views. Their common posterior identity and
chain/draw labels must survive persistence so cross-domain totals and tracer
ratios use paired samples. Compatible shapes are insufficient: establish exact
coordinate compatibility at the combining boundary. Preserve active/fixed and
computationally marginalized semantics in the result contract.

Supported reconstruction must work in a fresh process without acquisition or
model construction. Explicit graph replay remains supported for new predictive
work or appropriate deterministic evaluation. A custom builder declares a
truthful replay capability or opt-out. Preserve older stored-deterministic
artifacts while new supported outputs can reconstruct omitted values.

### Failure recovery and output availability

Persist the useful posterior and minimum recovery metadata before, or
atomically with, optional expensive products. Product failure must not erase
the inference result. Publish completion only after required artifacts and
their identities are durable; reject partial, corrupt or mismatched bundles.
Retain diagnostics and the approved validity assessment without inventing
universal thresholds in an architecture change.

Output availability is quantity-specific. Supported flux/state quantities
remain available when a concentration/error quantity cannot represent the
selected covariance. Joint prediction must retain joint covariance; marginal
standard deviations are not a substitute. Distinguish stored/derivable,
missing-artifact and scientifically unsupported cases with useful explanations.

Requested extra products may change artifact retention, not the resolved
probability model. Requested full-native posterior uncertainty cannot be
satisfied by relabelling a retained-state-conditional field. Writer schemas
must retain this distinction.

## Package ownership and dependency direction

Agree adapter and artifact ownership before large moves. The following are
possible destinations, not a package-creation checklist.

| Home | Responsibility | Bounded migration |
| --- | --- | --- |
| `rhime/` and family subpackages | Named recipes, configuration policy, scientific adapters and readable graph assembly | Keep standard/multisector as references; keep family-specific requests/bindings near the science |
| `inversion_data/` | Acquisition providers/resolution, labelled alignment, prepared-data contracts | Extract `RhimePreparedInputs` and schema/validation from the large preparation module when touched, preserving exports |
| `basis/` | Basis construction, fitted operators, gathered layouts, affine flux maps | Separate weights, maps, construction and I/O in `_functions.py` when justified; preserve operator APIs |
| Possible `forward/` | Backend-neutral responses, domain support and deterministic observable operations | Extract for demonstrated second consumers; no acquisition, state creation, sampling or recipe policy |
| Possible `uncertainty/` | Native covariance, prior moments, error representations/products and coherent reduction | Consolidate appropriate numerical code while preserving operator/action representations |
| `models/` | PyMC state/prior, response, baseline, coordinate and likelihood bindings | Preserve `rhime -> models`; separate backend-neutral kernels where useful |
| Possible `inference/` | Shared sampler and reusable optimized steps | Reuse `RhimeSampler.sample(model, variable_roles=...)`; keep specialized graph/cache/step coordination explicit |
| `postprocessing/` | Reconstruction, scientific output values, aggregation, diagnostics and writers | Consume explicit views/bindings; migrate reverse compatibility imports in bounded steps |
| Possible `workflow/` | Artifact/stage/check mechanics and explicit recipe dispatch | Extract when another supported family needs the mechanics; no inferred scientific execution order |
| Existing low-level/compatibility modules | Array operations, codecs, timing and legacy imports/entry points | Keep utilities free of recipe policy and preserve bounded public compatibility |

Allowed target dependencies are:

```text
entry points / staged workflow -> named recipes and adapters
recipes -> inputs/preparation, models, inference, reconstruction/writers
inputs/preparation -> scientific operators
models -> scientific operators
reconstruction/writers -> scientific operators
inference -> backend/model mechanics where needed
```

Scientific operators group several packages whose internal imports must also
remain acyclic. Principal rules: operators do not import recipes or products;
`models` does not import `rhime`; general postprocessing does not import
`rhime`. Record exact temporary compatibility exceptions and their migration
consumers. Check package initializers and re-exports, not just direct imports.
A `basis -> uncertainty -> basis` cycle through an initializer is still a
cycle. Preserve the required PyTensor initialization order.

Classify code before moving it: `models/state_activity.py` contains neutral
labelled operations; `models/fixed_ou.py` mixes kernels and backend integration.
Directory names alone do not establish ownership. Split namespace-only moves
from scientific/default/configuration/persistence-schema changes.

Keep useful existing recipe-specific configuration formats. CO2 TOML and
standard INI need not migrate together. Resolve Python and external settings
once into recipe-owned values, reject irrelevant settings, and pass resolved
values explicitly. Do not replace these with a giant global switch list.

## Existing issue responsibilities

Use existing work rather than creating another umbrella. The connections
below describe the proposal's intended integration with those responsibilities;
the dated evidence appendix supplies verified code and tracker details.

| Existing work | Relationship to this plan | Scope boundary |
| --- | --- | --- |
| [#663](https://github.com/openghg/openghg_inversions/issues/663), package ownership | Add contract owners and end-to-end linked-channel/source-domain walkthroughs | Still needs current/target maps, accepted ownership, import migration table and ordered implementation issues |
| [#661](https://github.com/openghg/openghg_inversions/issues/661), configuration/input plumbing | Recipe-owned requests, resolved identities and typed choices | Retain canonical Python values and useful external section structure; do not absorb all acquisition into config |
| [#414](https://github.com/openghg/openghg_inversions/issues/414), component preparation/reconstruction | Reuse preparation -> backend -> deterministic reconstruction | Runner-level acquisition is outside its stated scope and needs coordinated ownership |
| [#415](https://github.com/openghg/openghg_inversions/issues/415), run bundle | Durable prepared data, roles, reconstruction binding and matched identities | Preserve legitimate graph replay alongside ordinary backend-neutral output |
| [#570](https://github.com/openghg/openghg_inversions/issues/570), covariance-safe outputs | Quantity-specific capabilities and covariance provenance | Do not disable supported flux because another quantity in the same format is unsupported |
| [#533](https://github.com/openghg/openghg_inversions/issues/533), public builders | Reuse the prepared-input extension boundary | Does not alone establish upstream acquisition or downstream native reconstruction |

Amend #663's acceptance in three ways:

1. Walk a standard route, a coherent linked-channel route, and a source/domain
   composition through requests, preparation, inference, artifacts and products.
   Name each producer/consumer and identify missing providers or maps.
2. Assign ownership to requests/resolution, prepared products, capabilities,
   bindings and durable reconstruction, as well as files and stages.
3. Require a new supported composition to reuse common integration while
   allowing small scientific adapters and repeated readable assembly. Retain
   the explicit non-goals of a compiler and universal execution context.

These are proposed acceptance additions, not edits already made to trackers.
Do not infer current ownership, status or priority from an old repository plan.

## Delivery work packages

The identifiers below are local section identifiers, not new tracker issues.
Map them to existing owners before implementation. Each slice is independently
reviewable; scientific priorities determine which model route proceeds first.
Parallel work is permitted where contracts and scientific dependencies allow.

### W1. Agree integration ownership and walkthroughs

Extend #663 with the three concrete walkthroughs and the smallest demonstrated
request/binding values. Record current and target owners, exact compatibility
exceptions, and import migrations. Do not move modules in this slice.

Completion: every needed scientific input and requested quantity has a named
producer, consumer and artifact boundary. A scientist can still call kernels
or copy a runner without enrolling in a descriptor framework.

### W2. Persist standard output information independently of PyMC

Separate durable roles/mappings/metadata from `RhimeModelBuildResult`'s live
model, using existing codecs and manifest verification. Remove role-only graph
reconstruction from ordinary standard/multisector postprocessing. Preserve
explicit graph replay for genuinely backend-dependent calculations.

Completion: a fresh process reconstructs supported existing products from
matched artifacts with acquisition and model building disabled. Every chain
is preserved; old artifacts remain readable through an explicit compatibility
path. Product failure leaves a recoverable posterior.

### W3. Persist linked prepared and reconstruction artifacts

Add durable serialization for the existing CO2/O2 prepared value and the
separate affine reconstruction value. Bind exact prepared/reconstruction and
posterior identities; retain mapping, units, covariance representation,
support, reference state and approximation scope. Follow any approved affine
persistence contract through its owning issue rather than expanding it to full
native conditioning here.

Completion: reload without acquisition reproduces supported linked/native and
aggregate quantities, rejects mismatched artifacts, and preserves shared
posterior dependence. An in-memory prepared runner alone is not completion.

### W4. Complete one fixed-coupling linked-channel route

Use CO2/O2 assembly as the coherent structural reference and an explicit
methane-specific adapter/recipe. Agree source participation, units, reference
ratios, priors, baselines and error model. Adapt existing getters and shared
artifact mechanics. A source-neutral native/prepared route can be an early
milestone, but must not be advertised as complete acquisition support.

Completion: an advertised operational input route reaches saved posterior and
supported products; channel observations may differ; shared states are created
once; covariance and reconstruction match the declared statistical reference.
Add a runnable configuration/example and scientific review evidence.

### W5. Generalize demonstrated coherent preparation

Extract repeated operations exposed by linked-channel work and CO2. Add
supported prepared coherent alternatives to standard and multisector with
explicit native moments and projection. Establish correlated unobserved/fixed
state semantics before extending state-activity helpers. Keep conventional
preparation available with unchanged assumptions where promised.

Completion: a small independent native Gaussian oracle verifies prior,
intercepts, means and full covariance; non-Gaussian approximations disclose
their scope; required native reconstruction survives reload. A measured
execution envelope distinguishes current dense support from future operators.

### W6. Deliver multisector nesting through shared domain operations

Adapt #704's public alignment/support operations for source-aware consumers.
Extract only their demonstrated common numerical/mechanical work; keep basis
selection and domain/source prior policy with the recipe.

Completion: unequal basis sizes, a missing pair, source/domain permutations
and no-double-counting support all work. Both grids and one posterior survive
output/reload. Public imports remain compatible and no private nested runner
helper becomes a dependency of the new family.

### W7. Deliver coherent CO2 nesting

Combine W5 reduction and W6 domain operations with visible carbon policy and
native-data readiness. Declare any independent-domain prior assumption; do
not infer it from masking. Reuse acquisition/artifact/output mechanics.

Completion: correct source/domain mappings, coherent reference checks, durable
reconstruction and representative workload evidence. Additional duplicated
generic plumbing is a signal to revisit an earlier boundary, not to hide a
second complete pipeline.

### W8. Add parameter-dependent coupling as a scientific extension

Deliver the smallest required uncertain-ratio model, with explicit native
dependence or an approved approximation. Keep spatially varying ratios
separate if they require different preparation. Reconstruct tracer quantities
from paired posterior samples.

Completion: complete log likelihood and output quantities agree with an
independent reference for the selected model, including response-dependent
covariance and its determinant. Reference-ratio multipliers and physical
ratios cannot be confused by config or persistence.

### W9. Deliver radiocarbon independently, then optional O2

Agree the observable, units, support, averaging/transformation order, source
signatures and uncertainty dependencies early. Deliver CO2/radiocarbon as its
own route. Add the three-channel composition only when scientifically needed.

Completion: synthetic physical quantities reproduce the specified observable
on isotope sampling support; no unintended duplicate evidence; documented
nonlinear uncertainty treatment; complete advertised input/output route and
scientist acceptance.

### W10. Extend stages and migrate namespaces incrementally

Route shared stage/artifact mechanics through named scientific adapters once
another complete family needs them. Python and CLI use the same preparation
and reconstruction contracts. Move namespaces in separate changes with public
import compatibility and initializer checks.

Completion: each advertised route has tested input boundaries and products;
stage restart preserves scientific identity; namespace-only changes alter no
equations, priors, defaults or schemas. Existing legacy entry points have a
documented compatibility path.

W2 can proceed without a new model. W3 and the input/preparation side of W4
can proceed together; W4 can expose the repeated operations W5 extracts. W6
need not await all coherent work. W7 depends on the required domain and
reduction capabilities, while W9 need not await O2 or nesting. A conventional
CH4/C2H6 model can proceed independently of coherent infrastructure using the
same integration seams. These dependencies describe capabilities, not a
mandatory sequence of ten large PRs; split each work package further where
needed for review.

## Acceptance matrix

Use small scientifically discriminating cases, extending existing numerical
oracles and fixtures. Do not enumerate the Cartesian product of all options.
Each supported family needs labelled input, durable preparation, correct
forward/prior/likelihood behavior, controlled inference or an independent
oracle, reconstruction and replay evidence.

| Interaction | Required evidence |
| --- | --- |
| Providers/adapters | Equivalent acquired and labelled inputs reach equivalent preparation; observation-dependent footprint matching, inlet/platform handling and per-channel filtering are retained; only identical resolved requests share retrieval |
| Shared states | One state affects both channel means with unequal observation support; state creation and reuse have visibly different meanings |
| Conventional/coherent variants | Reuse integration; compare each to its own stated model, not to an unjustified expectation of equivalence |
| Sources/domains | Unequal bases, absent pair, reordered labels and overlap counted once; both grids retained |
| Correlated unobserved states | A zero direct response does not incorrectly fix a quantity informed through prior correlation; marginalization/recovery and scientific fixing have separate tests |
| Fixed Gaussian reduction | Nonzero-mean native oracle verifies retained moments, affine response and off-diagonal covariance; lossless/full-rank projection agrees with the same native model |
| Parameter dependence | Vary coupling and verify all required mean/covariance terms and complete log likelihood, or explicitly test the approved approximation |
| Units | Rescale channel rows by $D$: observations/means/responses transform by $D$, covariance by $D\Sigma D^\top$; posterior inference is invariant, allowing the expected parameter-independent density Jacobian |
| Radiocarbon | Observable follows the stipulated averaging/transformation order at its own support; dependent derived observations are not counted as independent evidence |
| Reconstruction scope | Affine output retains its stated Gaussian/approximation and conditioning scope; any future full-native output matches observation-conditioned native inference |
| Persistence/views | Matched reload preserves coordinates, all chains, shared posterior identity, paired state/ratio draws and native/aggregate views; deliberately mismatched maps are rejected |
| Graph-free reconstruction | In a fresh process, acquisition and model-building functions fail if called, while supported ordinary outputs still reconstruct |
| Graph replay | Separately verify supported predictive/backend-deterministic computations; do not confuse this capability with graph-free ordinary reconstruction |
| Availability | Missing required maps fail before sampling where knowable; supported flux remains available when separate concentration/error output is unsupported |
| Output-request invariance | Requesting an extra product changes retained artifacts only; resolved model inputs, prior and likelihood remain unchanged |
| Failure recovery | Optional writer failure preserves useful posterior/recovery metadata; incomplete publication and wrong identities are rejected |
| Execution/compatibility | Owned materialization respects borrowed arrays/shared graphs; public imports and PyTensor initialization remain valid; unchanged scientific assumptions retain reference behavior |
| Practical adoption | A scientist finds and changes one prior/likelihood/coupling, runs the advertised route and explains products using an example and recorded rubric |

Diagnostics and output-validity policy are production gates alongside these
architecture checks, with their own scientific owners. Multi-chain correctness
is tested using deliberately different chains, not identical fixture values.
Separate code support, operational adoption and numerical performance claims.

## Requirements for the next component-API discussion

Discuss requests, preparation lifecycles and reconstruction before polishing
graph syntax. Keras-style explicit sharing and scikit-learn-style predictable
interfaces are useful comparison ideas, not requirements to adopt either
framework or its lifecycle.

Evaluate ordinary functions and small objects against complete workflows:

- CH4/C2H6 sharing and multisector nesting use the same public pieces as
  maintained recipes.
- Creating, reusing and transforming a latent state have distinct meanings.
- Companion adapters can state needs and quantities without edits to a central
  list of every model family.
- Prepared values, graph construction, inference and reconstruction have clear
  lifetimes; fitted basis state and caches have identifiable owners.
- Labels, units, assumptions and joint covariance survive composition.
- Scientific equations remain easy to find and failures describe scientific
  incompatibilities rather than an opaque compilation stage.
- Direct kernel/prepared-value use does not require a descriptor or registry.

Use fixed-coupling CH4/C2H6 and multisector nesting first, followed by coherent
CO2 nesting. Keep named recipes as supported, citable entry points and examples
of customization. Do not require speculative descriptors for every future
component before one complete route works.

## Decisions required before implementation claims

The proposal intentionally leaves these decisions to the relevant scientific
and implementation owners:

1. First CH4/C2H6 source set, coupling convention, prior family, baseline and
   error assumptions; whether inferred ratios are immediately required.
2. Native moments/projection and approximation scope for coherent standard,
   multisector and methane preparation.
3. Fixed, conditioned, directly unobserved and computationally marginalized
   state semantics, including what each output reconstructs.
4. Exact first quantity capabilities, artifact schema/migration and approved
   affine-persistence interpretation.
5. Native-data/cache availability and memory/runtime envelope for real linked
   and CO2-nested workloads.
6. Radiocarbon observable, averaging protocol, signatures, dependencies and
   required additional sources/parameters.
7. Current owners, priorities, independent production gates and acceptance
   evidence locations in the existing trackers.

The evidence mapping below should identify an existing home for each decision
where possible, and state an unassigned gap when no verified owner exists.

## Dated issue and code evidence

Checked: 24 September 2026. Three independent subagent reviews mapped the
completed proposal to live issue bodies/relationships and fixed code snapshots.
The rows below are an evidence index, not a claim that a tracker was amended
or that a historical implementation satisfies every new acceptance condition.
An issue marked Done or closed can be a reuse foundation; it must not be
silently reopened or assigned expanded science by this document.

Code links use `b09a1dab0f` unless explicitly marked #704 or prototype.
Tracker status can post-date that snapshot. A line anchor identifies the
relevant symbol or block; read its surrounding implementation before changing
it. Proposed gaps require owner review and a bounded follow-up, not automatic
expansion of a nearby issue.

### Historical prototype evidence

| Plan connection | Specific evidence | Consequence |
| --- | --- | --- |
| Signature-driven construction | [PR #228 `Node` constructor inspection](https://github.com/openghg/openghg_inversions/blob/6aeaa5f2431eeb736b3be5cc587911d2107f9d1f/openghg_inversions/models/config/config_parser.py#L68) | Keep scientific dispatch explicit; do not infer config contracts from runtime signatures |
| Inherited preparation context | [`component_to_data_args_map` and graph-parent lookup](https://github.com/openghg/openghg_inversions/blob/6aeaa5f2431eeb736b3be5cc587911d2107f9d1f/openghg_inversions/models/config/data_parser.py#L60) | Pass resolved units/options directly rather than recovering scientific context from graph ancestry |
| Constructor acquisition | [`Flux.__init__`](https://github.com/openghg/openghg_inversions/blob/6aeaa5f2431eeb736b3be5cc587911d2107f9d1f/openghg_inversions/data_functions.py#L562) | Put retrieval in visible provider calls, leaving numerical values constructible without I/O |
| Persistence scope | [Prototype run/output block](https://github.com/openghg/openghg_inversions/blob/6aeaa5f2431eeb736b3be5cc587911d2107f9d1f/openghg_inversions/run.py#L177) | Saved traces and selected data do not establish a complete scientific reconstruction contract |
| Production behavior and reception | [Recorded human feedback](https://github.com/openghg/openghg_inversions/pull/228#issuecomment-2733964114) | Preserve inlet/height, platform, error, basis and filtering behavior; feedback also welcomed the format, so do not infer a universal rejection from code complexity |

### Ownership, nesting, stages, and production gates

| Plan connection | Verified issue home and status | Specific code/test evidence | Existing coverage and next action |
| --- | --- | --- | --- |
| W1: ownership and walkthroughs | [#663](https://github.com/openghg/openghg_inversions/issues/663) open; [OPE-128](https://linear.app/openghg-inversions/issue/OPE-128) Backlog | [`run_rhime` orchestration](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/standard.py#L452), [independent-import test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime.py#L2631) | Existing scope owns maps, stage ownership and compatibility migrations. Propose explicit request/binding owners and the linked-channel/source-domain walkthroughs; these remain additions, not completed acceptance |
| Configuration and W10 | [#661](https://github.com/openghg/openghg_inversions/issues/661) open; [OPE-162](https://linear.app/openghg-inversions/issue/OPE-162) Done, CO2 TOML slice delivered through [#687](https://github.com/openghg/openghg_inversions/pull/687) | [CO2 configuration resolver](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/configuration.py#L618), [Python/config equivalence](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime_co2_configuration.py#L99), [linked-channel settings](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime_co2_configuration.py#L233) | Reuse delivered configuration. Inventory remaining settings/recipes under the open umbrella; new acquisition requests complement configuration rather than restarting TOML design |
| Defaults and validation | [#664](https://github.com/openghg/openghg_inversions/issues/664) / [OPE-126](https://linear.app/openghg-inversions/issue/OPE-126), and [#665](https://github.com/openghg/openghg_inversions/issues/665) / [OPE-127](https://linear.app/openghg-inversions/issue/OPE-127): GitHub open, Linear Backlog | [Likelihood resolution](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/params.py#L657), [early unknown-option test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime.py#L7031) | Existing owners cover default classification and validation at owning boundaries. Add concrete channel/domain/coupling choices to those inventories; do not silently change defaults or introduce another validation framework |
| W6: nested preparation/model foundation | [#407](https://github.com/openghg/openghg_inversions/issues/407), [#408](https://github.com/openghg/openghg_inversions/issues/408) open; [OPE-71](https://linear.app/openghg-inversions/issue/OPE-71), [OPE-72](https://linear.app/openghg-inversions/issue/OPE-72) In Progress; [#704](https://github.com/openghg/openghg_inversions/pull/704) merged | At #704: [public alignment](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L579), [masking](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L638), [single-source restriction](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L452), [equation test](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/tests/test_nested_rhime.py#L338) | Reconcile merged one-source evidence against existing acceptance. W6's unequal source/domain layouts are additional scope; no dedicated multisector-nested issue was verified in this search |
| W6: nested products/acceptance | [#409](https://github.com/openghg/openghg_inversions/issues/409), [#666](https://github.com/openghg/openghg_inversions/issues/666) open; [OPE-73](https://linear.app/openghg-inversions/issue/OPE-73) In Progress; [OPE-25](https://linear.app/openghg-inversions/issue/OPE-25), [OPE-173](https://linear.app/openghg-inversions/issue/OPE-173) Todo | At #704: [recipe output adapter](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L253), [two-view test](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/tests/test_nested_rhime.py#L704), [compatibility reverse import](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/postprocessing/nested_paris_outputs.py#L279) | Existing issues include config/docs, scientist acceptance and real-data evidence decisions. OPE-173 still references [#668](https://github.com/openghg/openghg_inversions/pull/668), now closed unmerged. Reconcile it with #704; merged code alone does not satisfy every parent gate |
| W10: another staged family | [OPE-164](https://linear.app/openghg-inversions/issue/OPE-164) Todo, child of [OPE-79](https://linear.app/openghg-inversions/issue/OPE-79); standard/multisector stages delivered by [#671](https://github.com/openghg/openghg_inversions/pull/671) | [Supported model kinds](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L54), [prepared dispatch](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L450), [identity tests](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_staged_workflow.py#L165) | OPE-164 already owns ordinary/cached CO2 routing and shared manifests. Use this delivery home before creating generic workflow work; its relations include OPE-150/153/162/163/169 as blockers, some already completed |
| W10: namespace compatibility | #663 / OPE-128 above | [PyTensor initialization before exports](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/models/__init__.py#L13), [fresh-process dtype test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_pymc_config.py#L38) | Record exact compatibility exceptions and consumers. Preserve initialization, imports and scientific behavior in namespace-only changes |
| All-chain production gate | [#657](https://github.com/openghg/openghg_inversions/issues/657) closed through merged [#670](https://github.com/openghg/openghg_inversions/pull/670); umbrella [#645](https://github.com/openghg/openghg_inversions/issues/645) open | [Chain-preserving conversion](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/postprocessing/inversion_output.py#L142), [discrepant-chain product test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime.py#L710) | Treat all-chain behavior as a delivered foundation and regression obligation for new views. The explicit legacy chain-selection path does not define modern output semantics |
| Diagnostics and validity gates | [#656](https://github.com/openghg/openghg_inversions/issues/656), [#667](https://github.com/openghg/openghg_inversions/issues/667), [#637](https://github.com/openghg/openghg_inversions/issues/637) open | [Staged diagnosis](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L616), [unassessable-metric test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_staged_workflow.py#L471) | Staged metrics/check artifacts exist through #671. Automatic ordinary-run invocation and remaining contract/regression work are separate; #637 retains scientific validity questions. Do not equate scheduler success, diagnosis and publication validity |

OPE-126/127/128 had no individual assignee returned in this review. Their
related issues are not automatically blockers. The verified nesting chain
remains OPE-71 -> OPE-72 -> OPE-73, but its issue records predate #704 and
need evidence reconciliation. No direct Linear parent mirror of #661, or
dedicated mirrors for the cited diagnostics/all-chain GitHub issues, were
verified by this search; this is a search limitation rather than proof that
none exists anywhere in the workspace.

### Acquisition, persistence, reconstruction, and output integration

| Plan connection | Verified issue home and status | Specific code evidence | Existing coverage and next action |
| --- | --- | --- | --- |
| W2: graph-free ordinary postprocessing | [#415](https://github.com/openghg/openghg_inversions/issues/415) open; [OPE-106](https://linear.app/openghg-inversions/issue/OPE-106), [OPE-107](https://linear.app/openghg-inversions/issue/OPE-107) Backlog | [Live-model build result](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/builders.py#L66), [role-only rebuild](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L790), [roles/metadata consumer](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/outputs.py#L273), [existing artifact hashes](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L596) | #415 owns both graph replay and neutral reconstruction. Allocate the smaller W2 extraction under that scope, coordinated with data ownership. Historical [OPE-55](https://linear.app/openghg-inversions/issue/OPE-55) and [OPE-21](https://linear.app/openghg-inversions/issue/OPE-21) are Canceled; do not silently revive them |
| Component reconstruction | [#414](https://github.com/openghg/openghg_inversions/issues/414) open; [OPE-105](https://linear.app/openghg-inversions/issue/OPE-105), OPE-106 Backlog | [`SigmaAlignment.from_model_data`](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/sigma.py#L234), [linear/linked/affine components](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/models/components.py#L227) | #414 already includes preparation/backend/reconstruction and, in discussion, coherent/effective-state reconstruction. Reuse sigma and implement demonstrated linear/offset cases. Runner acquisition and complete bundles remain outside its scope |
| Compact, failure-safe outputs | [OPE-125](https://linear.app/openghg-inversions/issue/OPE-125) Backlog, child of OPE-107; both blocked by OPE-105/106; [#625](https://github.com/openghg/openghg_inversions/issues/625) open | [Ordinary wrapper-before-trace saving](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/outputs.py#L437), [prepared basis/site serialization](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/inversion_data/preparation.py#L239), [output serialization](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/postprocessing/inversion_output.py#L414) | OPE-125 owns trace-first recovery, failure injection and storage evidence. #625 distinguishes combined from separately reported observation errors. Reconcile ownership before broad storage changes; W2's small role extraction need not decide every operand's durable home |
| W3: linked prepared artifact | [OPE-165](https://linear.app/openghg-inversions/issue/OPE-165) Todo; [OPE-40](https://linear.app/openghg-inversions/issue/OPE-40) Backlog; #415 open | [Field-only linked prepared value](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_preparation.py#L29), [prepared-product ingress](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_preparation.py#L314), [CO2 save/load precedent](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_preparation.py#L356) | OPE-165 already owns linked persistence/stages, unequal axes, unit/ratio provenance and joint covariance. Add explicit matching reconstruction/posterior binding within accepted scope; generic larger covariance products belong to OPE-40 |
| W3: affine persistence gate | [OPE-169](https://linear.app/openghg-inversions/issue/OPE-169) Done, but [OPE-184](https://linear.app/openghg-inversions/issue/OPE-184) Todo / [#721](https://github.com/openghg/openghg_inversions/issues/721) open; [#715](https://github.com/openghg/openghg_inversions/pull/715) merged as PR1 | [`AffineFluxMap`](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/basis/affine_flux_map.py#L50), [approved PR2 tasks](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openspec/changes/persist-co2-affine-output-reconstruction/tasks.md#L18), [approved scope](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openspec/changes/persist-co2-affine-output-reconstruction/proposal.md#L7) | OPE-169's body requires PR2 and resolution of OPE-184 despite its Done status. PR2 completion was not established. Reconcile evidence/status, preserve the finalized CO2 contract and coordinate any linked extension separately |
| Quantity capabilities | [#570](https://github.com/openghg/openghg_inversions/issues/570) open; [OPE-24](https://linear.app/openghg-inversions/issue/OPE-24) Todo; [OPE-183](https://linear.app/openghg-inversions/issue/OPE-183) Backlog | [Format-wide covariance rejection](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/prepared.py#L102), [format-level builder capabilities](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/builders.py#L61) | #570/OPE-24 supply existing homes for quantity-specific checks and scientific roles. OPE-183 concerns affine quantity maps, not a persisted registry; it is not a prerequisite for first native-flux persistence |
| W4: acquisition/source-neutral foundations | [#533](https://github.com/openghg/openghg_inversions/issues/533) and [#509](https://github.com/openghg/openghg_inversions/issues/509) closed/completed | [xarray prepared-input adapter](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/inversion_data/xarray_adapter.py#L864), [retrieval/reload](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/preparation.py#L59), [observation-dependent footprint matching](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/inversion_data/getters.py#L296), [prepared-only linked runner](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_runner.py#L197) | Reuse these delivered interfaces and mature getters. They do not establish a methane-specific acquisition/output recipe; that operational delivery remains a scientific workstream gap |
| W10: linked stages and cache boundary | [OPE-164](https://linear.app/openghg-inversions/issue/OPE-164), OPE-165 Todo; [OPE-171](https://linear.app/openghg-inversions/issue/OPE-171) In Progress | [Configuration identity](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L282), [artifact verification](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/stages.py#L167), [linked trace annotations](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_runner.py#L139), [native-product projection consumer](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/inversion_data/xarray_adapter.py#L518) | Extend CO2-only then linked dispatch through existing owners. OPE-171 owns active Verification Games native-cache work, linked to [VG #65](https://github.com/openghg/verification-games/pull/65) and [openghg-run #40](https://github.com/openghg/openghg-run/issues/40); coordinate instead of duplicating discovery/cache policy |

Three concrete reconciliation points affect implementation:

- OPE-106/107 leave durable model-data placement under investigation, while
  OPE-125 prescribes `InferenceData` ownership for model operands. Resolve that
  disagreement with their owners; the artifact table above does not mandate
  `constant_data` as a universal home.
- OPE-164 still refers to `RhimePreparedInputs` in places where the implemented
  carbon handoff is `Co2PreparedInputs`. OPE-165 is formally blocked by
  OPE-162/163/164; OPE-162/163 are recorded Done. Distinguish completed
  foundations from remaining blockers.
- OPE-165's initial route is same-unit/common-scale. Its per-axis unit metadata
  does not establish the heterogeneous CO2-ppm versus delta(O2/N2)-per-meg
  measurement transformation; that scientific contract belongs to OPE-86.

The finalized affine persistence scope retains native mean, signed reference
flux and a tagged prolongation, obtaining the authoritative retained reference
from matched `Co2PreparedInputs`. It excludes raw `fp_x_flux`, native covariance,
projection payloads, precomposed flux maps, named derived maps and residual
blocks. Full native conditioning and generic quantity maps remain separate
work. This proposal does not amend those finalized specification files.

### Scientific extensions, assumptions, and discriminating tests

| Plan connection | Verified issue home and status | Specific code/test evidence | Existing coverage and next action |
| --- | --- | --- | --- |
| W4: methane/ethane route | [#205](https://github.com/openghg/openghg_inversions/issues/205), [#412](https://github.com/openghg/openghg_inversions/issues/412) open; [OPE-77](https://linear.app/openghg-inversions/issue/OPE-77) Done | [CO2/O2 assembly](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_model.py#L114), [Ramsden builder](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/experimental/ramsden2022/model.py#L780), [unequal-channel covariance test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime_co2_o2.py#L151) | #205 requests methane/ethane broadly; #412/OPE-77 supply the carbon-specific prepared shared-state seam. No dedicated live Linear methane delivery issue was verified. Agree sources, coupling, prior and operational boundary in a bounded follow-up linked to these foundations |
| W5: coherent promotion and oracles | [OPE-78](https://linear.app/openghg-inversions/issue/OPE-78) Backlog; [OPE-18](https://linear.app/openghg-inversions/issue/OPE-18) Done; [#566](https://github.com/openghg/openghg_inversions/issues/566) open; [OPE-6](https://linear.app/openghg-inversions/issue/OPE-6) Todo | [Public reduction](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/coherent_reduction.py#L58), [dense covariance selection](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/coherent_reduction.py#L129), [nonzero-mean oracle](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_coherent_reduction.py#L116), [two-basis posterior/evidence oracle](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_coherent_reduction.py#L546) | OPE-78 explicitly owns standard/multisector promotion, starting with the affine component and unchanged defaults. OPE-18 supplies numerical foundations; broader historical #566 scope is not completed merely because that child is Done. Extend existing oracles through concrete preparation adapters |
| W5: unobserved/fixed/marginalized semantics | [#565](https://github.com/openghg/openghg_inversions/issues/565) closed; [OPE-16](https://linear.app/openghg-inversions/issue/OPE-16), [OPE-119](https://linear.app/openghg-inversions/issue/OPE-119) Done | [Zero-column policy](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/models/state_activity.py#L39), [active-prior subsetting](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/models/components.py#L559), [fixed-value compatibility test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/models/test_correlated_state.py#L342), [joint-channel zero-column test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime_co2_o2.py#L205) | OPE-119 intentionally preserves fixing and removed Gaussian outer marginalization; it requires future generic nuisance marginalization to be independently proposed. No active owner was verified for the new correlated-unobserved posterior recovery policy. Preserve compatibility and allocate a separate scientific decision/oracle |
| Projection eligibility and closure | [OPE-31](https://linear.app/openghg-inversions/issue/OPE-31) Todo; [OPE-85](https://linear.app/openghg-inversions/issue/OPE-85) Duplicate of OPE-31 | [Current projection strategy](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/basis/covariance_products.py#L88), [moment-closure warning](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/models/components.py#L426), [rank-deficiency test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_covariance_products.py#L677) | OPE-31 already owns supplied-projection numerical policy, positive-state eligibility, signed conditional lift and approximation ledger. Route those requirements there, without reactivating OPE-85; each recipe still chooses its native moments and projection |
| W7: coherent CO2 nesting | [#666](https://github.com/openghg/openghg_inversions/issues/666) open / [OPE-25](https://linear.app/openghg-inversions/issue/OPE-25) Todo; [#723](https://github.com/openghg/openghg_inversions/issues/723) open | At #704: [support masking](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L638), [two-domain builder](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/openghg_inversions/rhime/nested.py#L911), [lazy overlap test](https://github.com/openghg/openghg_inversions/blob/38f4a7e53783e7e416e04f3510bcb9648290142e/tests/test_nested_rhime.py#L245) | Existing owners cover the nested family and optional readability split, not a complete coherent-carbon nested model. Allocate native-data, carbon prior/cross-domain covariance and reconstruction scope as a bounded follow-up. #723 is not scientific delivery |
| W8: uncertain ratios | [#411](https://github.com/openghg/openghg_inversions/issues/411) open; [OPE-118](https://linear.app/openghg-inversions/issue/OPE-118) Backlog | [Ratio/reference convention](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/experimental/ramsden2022/model.py#L137), [ratio tensors](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/experimental/ramsden2022/model.py#L595), [direct/reference-ratio equivalence test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/experimental/test_ramsden2022.py#L199), [response-change reduction test](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_coherent_reduction.py#L341) | Existing scope owns fixed/inferred ratio configuration, shared states, reference semantics and serialization. Conditional coherent covariance/log determinant and paired native tracer output require an accepted extension/follow-up. [OPE-9](https://linear.app/openghg-inversions/issue/OPE-9) is uncertain-transport research, not an implementation owner for this emission-ratio model |
| Mixed-unit covariance acceptance | [OPE-86](https://linear.app/openghg-inversions/issue/OPE-86) Todo | [Model unit contract](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_model.py#L128), [dense channel-block assembly](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_preparation.py#L270) | Already owns observable convention, cross-block product units, reversible rescaling, posterior invariance and evidence Jacobian. Reuse this acceptance; extend methane coverage through explicit scientific scope rather than assuming metadata proves numerical invariance |
| Full native/aggregate conditioning | [OPE-68](https://linear.app/openghg-inversions/issue/OPE-68) Backlog; [OPE-20](https://linear.app/openghg-inversions/issue/OPE-20) Todo; [#576](https://github.com/openghg/openghg_inversions/issues/576) open | [Affine-only scope](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/basis/affine_flux_map.py#L50), [affine dense oracle](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_affine_flux_map.py#L62) | OPE-68 explicitly owns observation-conditioned complete country moments; OPE-20 owns a shared-prepared analytic realization. Coordinate the unresolved-contrast oracle there. #576 still contains superseded compiler wording; neither that wording nor affine persistence proves full conditioning exists |
| W9: radiocarbon, independently of O2 | [#205](https://github.com/openghg/openghg_inversions/issues/205) open; no dedicated live Linear delivery issue verified | [Separate-channel preparation seam](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/openghg_inversions/rhime/co2/co2_o2_preparation.py#L314), [staggered-support fixture](https://github.com/openghg/openghg_inversions/blob/b09a1dab0f/tests/test_rhime_co2_o2_preparation.py#L132) | These are reusable transport/channel seams, not isotope implementation evidence. Assign observable, signatures, sampling/averaging, dependency and nonlinear-error decisions plus a complete operational route. No implemented radiocarbon oracle was verified |

### Ownership gaps and reconciliation actions

The review found related infrastructure but no verified concrete delivery owner
for the full methane/ethane route, multisector nesting, coherent CO2 nesting,
or CO2/radiocarbon. It also found no verified active scope for correlated
unobserved-state recovery or the complete conditional coherent likelihood with
inferred emission ratios. Searches included methane, ethane, radiocarbon and
isotope terms plus the related issue bodies and relationships. These are
bounded search findings, not assertions that no relevant record could exist.

Before implementation, the existing programme owners should:

1. Accept or revise the W1 contract/walkthrough additions under #663/OPE-128.
2. Allocate W2's small output-information extraction without reviving canceled
   replay issues or preempting OPE-106/107's data-placement decision.
3. Reconcile OPE-169's completion status with PR2, OPE-184/#721, and the final
   affine specification; use OPE-165 for linked persistence and OPE-164/165 for
   staged delivery.
4. Attach merged #704 evidence to the still-open nested acceptance chain and
   replace stale #668 references before assigning new source/domain scope.
5. Scope the unassigned scientific models/decisions as linked, reviewable work
   rather than silently extending completed foundations or generic umbrellas.
6. Preserve already-delivered all-chain/configuration work and distinguish it
   from remaining ordinary-run diagnostics, scientific validity and adoption
   gates.

No issue creation, comments, status changes, implementation or numerical
validation were performed as part of this document. The code/tests above are
locations for reuse and extension; citing a test is not a claim that it was
executed during this review.
