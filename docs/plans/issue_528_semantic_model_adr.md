# ADR: A semantic model and coherent reduction boundary for RHIME

- **Status:** Proposed for adoption
- **Date:** 2026-08-10
- **Decision issue:** [#528](https://github.com/openghg/openghg_inversions/issues/528)
- **Supersedes:** The speculative semantic-kernel part of
  [the issues 402/403 design note](issues_402_403_builder_strategy_design.md),
  while retaining its completed narrow compiler work and the conceptual
  distinctions that remain valid
- **Scope:** Scientific model identity, mathematical preparation, backend
  realization, numerical approximation, and output reconstruction
- **Companion documents:**
  [scientific overview](issue_528_scientific_overview.md) and
  [delivery plan](issue_528_delivery_plan.md)

## Summary

RHIME will use a small, labelled semantic model as the source of truth for the
scientific and mathematical structure of an inversion. The semantic model will
describe native and retained states, priors, named forward-model terms,
observation models, covariance components, quantities of interest, and the
declared relationships between them. It will not contain PyMC variable names,
array slices, loop-versus-stack choices, or product-specific output names.

A **forward-model term** is a scientifically named contribution to the
expected observation, such as modelled pollution enhancement (currently often
called `mu_x`), a boundary-condition contribution (`mu_bc`), or an offset. An
**observation model** comprises the declared observed data, its complete mean
expression, covariance model, and likelihood. Species or gas, tracer status,
and observation platform are recorded separately; none is implied by the term
“observation model.”

Binding a semantic model to prepared data will produce a backend-neutral
**bound mathematical model**. This object will contain the actual labelled
operators, moments, covariance representations, state-treatment record,
approximations, and reconstruction products used in a run. PyMC and future
analytic-Gaussian realizations will compile this same object. A compilation and
output manifest will map stable semantic IDs to backend variables and
saved products without making those names part of the scientific model.

The **state-treatment record** says which declared states or subspaces are
retained as random variables, fixed or conditioned, coherently integrated out,
or structurally absent, and where uncertainty from an integrated state
re-enters the reduced model. It is not necessarily a row-by-row copy of a
materialized state vector.

A **semantic identity** is the stable, backend- and product-neutral meaning of
a scientific quantity or relation, together with its definition, units, and
coordinate scope where applicable. A serialized **semantic ID** refers to that
identity. For example, “modelled pollution enhancement” has the same semantic
identity whether a PyMC variable is named `mu_x` or an output product uses a
different name. This is analogous to a CF `standard_name`, rather than the
freer descriptive `long_name`, although project-specific quantities will not
all have CF standard names.

A scientist-facing **mathematical model card** is a primary output of semantic
binding, not optional prose added after compilation. It must show the states,
equations, arithmetic prior moments, covariance decomposition, units,
approximations, quantities of interest, and reconstruction status before
sampling begins.

The existing concrete one-sector RHIME implementation will remain as a
readable reference implementation and parity oracle. It is valuable precisely
because a scientist or developer can follow the simple PyMC construction
directly. The private normalized flux compiler remains an extension and
regression-testing path; it is not promoted to the semantic model. Whether the
concrete implementation eventually becomes executable documentation can be
revisited after the new path is stable, but removing it is not part of this
decision.

The semantic model will declare quantities of interest and record whether each
is exactly representable by the retained state. General covariance-aware
reconstruction remains part of the mathematical contract. Practical guidance
for aligning routine reporting regions with the basis is given separately
below and should migrate into the user manual.

## Status language used in this ADR

This work has developed quickly across `openghg_inversions`,
`verification-games`, and `inversions-knowledge`. Calling every unavailable
feature “missing” hides important differences. This ADR uses four primary
evidence states, with three planning refinements:

| Status | Meaning |
| --- | --- |
| **Merged/implemented** | Available on `openghg_inversions/devel` and covered by repository contracts. |
| **In flight or planned** | An active pull request contains a reviewable slice, or an open issue has a stated boundary and acceptance criteria. |
| **Tested prototype** | Implemented or exercised in `verification-games` or a documented experimental branch, but not yet a supported `openghg_inversions` contract. |
| **Unplanned/research boundary** | Not yet represented by an agreed implementation issue, or still requires scientific research before an API should be promised. This is refined below into **implied but unscheduled**, **identified gap**, or **deferred pressure test** where that distinction affects delivery. |

“Planned” is not a defect report. It means the design pressure is known and
scheduled, but users cannot yet rely on it in `devel`. “Implied but
unscheduled” means an umbrella decision anticipates the work but no focused
delivery issue owns it. An “identified gap” has neither a sufficient current
contract nor a delivery owner. “Tested prototype” is evidence for a decision,
not a claim of production readiness.

## Context

### What current RHIME says mathematically

For one flux component, current RHIME first weights the native footprints by a
fixed prior flux pattern and aggregates them into basis regions. In simplified
notation, let:

- $M_0$ be the native footprint operator;
- $D$ be the diagonal prior-flux pattern;
- $A$ be the one-hot native-cell-to-region membership matrix; and
- $a$ be the basis-region scaling state.

The modeled flux field and observation contribution are

$$
x_{\mathrm{RHIME}} = D A^{\mathsf T} a,
\qquad
\mu_{\mathrm{flux}} = M_0 D A^{\mathsf T}a.
$$

The usual model assigns the same scalar prior family and parameters to every
element of $a$ within a component, regardless of basis-region area, prior
flux mass, shape, or number of native cells. Multisector RHIME repeats this
construction per sector and sums the resulting observation contributions.

This is an internally defined model at one chosen basis: within-region spatial
variation is fixed to the prior pattern. It is not, however, the exact
re-expression of a declared stochastic native-grid model. Splitting a basis
region introduces more independent scaling variables with the same marginal
prior. It therefore changes the implied native covariance, aggregate prior
uncertainty, likelihood flexibility, and often the posterior. The model is not
equivariant to a basis choice that users are otherwise free to make.

This explains several practical inconsistencies:

- small or low-flux regions can receive the same scale uncertainty as large
  regions, producing extreme native-cell marginal behaviour;
- country uncertainty changes when a basis is split, even when the scientific
  country prior was intended to remain the same;
- a region crossing a country boundary imposes the same scale on both sides,
  while splitting it and assigning ordinary independent priors changes the
  prior rather than merely changing coordinates;
- there is no unique unresolved or “aggregation-error” correction because the
  current retained priors do not identify a native-grid probability model;
- results can be difficult to compare across bases because the basis, prior,
  and effective likelihood change together.

The independent-grid prior-width projection merged in #521 is a useful special
case and diagnostic. It does not supply correlated native covariance or the
joint prior, forward, unresolved-covariance, and reconstruction transformation
needed for exact coherent reduction.

### The modelling position adopted here

When basis regions are a computational and targeting choice rather than the
definition of scientific prior uncertainty, we implicitly require a probability
model on a native state. The basis must then induce, rather than invent, the
retained prior and unresolved contribution.

This native-model-first position is an explicit design decision, not the only
practice used in atmospheric inversions. [Turner and Jacob
(2015)](https://doi.org/10.5194/acp-15-7039-2015), for example, use
similarity-informed basis construction to encode plausible spatial error
relationships that they could not specify through a realistic native-grid
covariance. In that setting, changing the basis also changes the effective
prior information rather than merely changing the representation of one fully
specified native model.

The distinction is partly practical rather than a disagreement about the
coherent linear-Gaussian result. In his [public referee
report](https://acp.copernicus.org/preprints/15/C444/2015/acpd-15-C444-2015.pdf),
Bocquet argued that the reduced operators contain an implicit scale-transfer
choice. Turner and Jacob acknowledged in their [author
response](https://acp.copernicus.org/preprints/15/1001/2015/acpd-15-1001-2015-AR1.pdf)
and final paper that, if the native covariance were correctly specified,
native resolution would have the least analysed error and aggregation would be
motivated by computational savings. We adopt native-model-first while
recognizing that constructing a credible native model is itself difficult and
that basis-informed modelling is a pragmatic alternative when that model is
incomplete. The checked internal literature synthesis is in the private-group
[Turner--Jacob
note](https://github.com/openghg/inversions-knowledge/blob/main/docs/literature/turner-and-jacob-2015.md)
and [posterior-projection research
question](https://github.com/openghg/inversions-knowledge/blob/main/docs/research-questions/posterior-projection-conundrum.md).

The native model need not always be a dense Gaussian grid field. It may be a
structured covariance action, a latent-factor model, a positive field, a
hierarchical model, or a conditional simulator. What matters is that the model
declares:

1. the native random state and its probability law;
2. the retained map;
3. the conditional law on unresolved directions; and
4. whether the reduced likelihood is exact, moment-closed, truncated, or
   otherwise approximated.

## Mathematical foundation

### General projection principle

Let $X$ be the complete native state, $Y$ the observations, and
$A=T(X)$ the retained state for a measurable map $T$. If the retained prior
is the pushforward $T_\#\mathcal L(X)$, and the reduced likelihood is

$$
\bar \ell(y\mid a)
= \int \ell(y\mid x)\,
  \mathcal L(X\in dx\mid T(X)=a),
$$

then Bayesian updating commutes with reduction:

$$
\mathcal L(A\mid Y=y)
= T_\#\mathcal L(X\mid Y=y).
$$

In words, the posterior distribution of the retained state is the projection,
or pushforward, of the native posterior distribution.

This result is not restricted to linear or Gaussian models. Its computational
usefulness is restricted: the conditional-fibre integral may itself be as
difficult as native inference.

### Linear-Gaussian specialization

Let

$$
x\sim\mathcal N(m,B),
\qquad
y=Hx+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,R),
$$

and retain

$$
\alpha=\Pi x,
\qquad
C_\alpha=\Pi B\Pi^{\mathsf T}.
$$

With labelled solves rather than an explicit inverse, define

$$
U_*=B\Pi^{\mathsf T}C_\alpha^{-1},
\qquad
B_\perp=B-B\Pi^{\mathsf T}C_\alpha^{-1}\Pi B.
$$

Define the effective retained-state observation operator once as

$$
H_\alpha:=HU_*=HB\Pi^{\mathsf T}C_\alpha^{-1}.
$$

The first implementation requires \(\Pi\) to have full row rank in the
\(B\)-metric, so that \(C_\alpha\) is positive definite. Redundant retained
coordinates must fail labelled validation. Pseudoinverse semantics for a
singular retained covariance are future work and must not be selected
silently.

The exact reduced observation model is

$$
y\mid\alpha
\sim
\mathcal N\!\left(
Hm+H_\alpha(\alpha-\Pi m),
R+HB_\perp H^{\mathsf T}
\right).
$$

The retained prior, effective forward map, centring term, and unresolved
observation covariance all derive from the same $B/H/\Pi$ ledger. Omitting
the covariance term, using an unrelated coarse prior, or substituting an
uncentred prolongation defines a different model.

For any two exact retained maps of the same native model, each reduced
posterior is the corresponding pushforward of one common native posterior and
the prior-predictive data law is unchanged. This is the precise sense in which
the posterior is equivariant with respect to basis choice. It does not mean
that all reduced state vectors have the same dimension or values, nor that
every numerical implementation has the same conditioning or cost.

A fuller derivation, including the general non-Gaussian theorem and numerical
qualifications, is maintained in the private-group
[posterior-projection and exact-marginalization
note](https://github.com/openghg/inversions-knowledge/blob/main/docs/derivations/posterior-projection-and-exact-marginalization.md).
The mathematical material needed by users should eventually move into public
OpenGHG Inversions documentation rather than remain only in that repository.

### Prior covariance is a scientific requirement

If native cells are independent, a regional total averages many independent
perturbations. Achieving a large, scientifically intended uncertainty on that
total can then require enormous cell-scale variance. In the
`verification-games` tests, calibrating independent cells to approximately
50% country/source uncertainty drove cell-scale multiplicative standard
deviations into a prior-predictive spike regime. Correlated native covariance
allowed similar aggregate uncertainty with substantially smaller marginal
cell and retained-state uncertainty.

The semantic model therefore treats native covariance and its calibration
target as scientific model declarations. Basis geometry and numerical
representation remain separate choices. A separable or matrix-free covariance
operator may avoid materializing dense native $B$, but it must produce the
same labelled product blocks required by coherent reduction.

The first production implementation is deliberately modest: configurable
one-dimensional exponential latitude and longitude factors, combined as a
separable covariance action, with a documented introductory length scale of
1.5 degrees in each direction. That value is a usable example/default, not a
universal scientific constant. Optional declared class masks retain covariance
within classes and set cross-class covariance to zero. Source-specific
amplitudes and independent source blocks are the initial multisource case;
cross-source correlation must be explicit when introduced. These are the
decisions and boundaries recorded in #493, not requirements that every future
native covariance use an exponential kernel.

### Positive states and lognormal closure

For a positive state with arithmetic mean vector $m_\alpha>0$ and arithmetic
covariance $C_\alpha$, the multivariate-lognormal latent parameters are

$$
(\Sigma_z)_{ij}
= \log\!\left(1+
\frac{(C_\alpha)_{ij}}{(m_\alpha)_i(m_\alpha)_j}
\right),
\qquad
(\mu_z)_i
= \log (m_\alpha)_i-\tfrac12(\Sigma_z)_{ii}.
$$

For mean-one states this reduces elementwise to

$$
\Sigma_z=\log^{\circ}(1+C_\alpha),
\qquad
\mu_z=-\tfrac12\operatorname{diag}(\Sigma_z).
$$

The elementwise arithmetic-to-latent moment transform is exact when it is
defined and the resulting latent covariance is positive semidefinite. A
whitened latent Gaussian is the preferred PyMC parameterization.

The distribution of an arithmetic sum or projection of native lognormal
variables is not generally lognormal. Projecting its arithmetic first two
moments and fitting a reduced multivariate lognormal is therefore a declared
**moment closure**, not exact non-Gaussian marginalization. If the direct
elementwise transformation does not yield a valid positive-semidefinite latent
covariance, the tested fallback preserves a valid latent covariance shape and
scales it to match one declared aggregate arithmetic variance. A singular
positive-semidefinite covariance is mathematically valid; whether a backend
requires a nonsingular factor is a separate numerical issue. The fallback does
not reproduce the full target arithmetic covariance and must identify the
matched functional.

For a native Lognormal field, there are at least two distinct distributional
approximations. The projected arithmetic mean and covariance are exact, but
fitting a reduced multivariate Lognormal changes the retained marginal law
because a weighted sum of Lognormal variables is not generally Lognormal.
Separately, replacing the exact conditional fibre law by a Gaussian with
matched moments adopts a Gaussian native residual model. Replacing only its
observation-space pushforward by a Gaussian is a weaker observation-space
closure and does not by itself define a Gaussian native residual. If
covariance-regression formulas supply the moments, the exact conditional mean
and covariance are additionally replaced by linear-Bayes approximations
derived from the joint first two moments. The resulting likelihood is exact
for its declared approximate model, not for the original native Lognormal
prior. A native-fibre Gaussian closure changes the joint native prior and may
assign positive probability to negative reconstructed native-cell states; an
observation-space-only closure changes the likelihood without defining a full
replacement native prior. The model card must record retained-law fitting,
conditional-moment approximation, native-fibre or observation-space Gaussian
closure, and numerical approximation separately.

This construction also requires the retained coordinates to remain strictly
positive, normally through nonnegative aggregation weights applied to a
positive native state. Signed linear functionals require another prior family
or should remain declared quantities of interest rather than being forced into
a Lognormal retained-state representation.

Prototype evidence suggests that this can nevertheless be a useful practical
model. In three fixed-sigma JJA leave-one-month-out tests in
`verification-games`, Lognormal RHIME and the exact analytic Gaussian model had
inner-PARIS total errors differing by 6.6--9.4 Tg CO2 yr$^{-1}$ and held-out
clean-signal RMSE differing by at most 0.011 ppm; both showed the same main
sector-attribution failure. These are small observed differences in the
selected summaries and cases, not comparison with an exact Lognormal fibre
marginal, a general accuracy result, or an independently defined materiality
judgement. The experiment record is
`notebooks/worklog.d/2026-07-28-230000-coherent-jja-fixed-sigma-rhime-results.md`
in `verification-games`.

Avoiding the closure remains a serious research problem. The exact Lognormal
likelihood is a normalized integral over an $\alpha$-dependent,
positivity-constrained fibre, with no Gaussian conditioning collapse. Related
non-Gaussian work has produced exact or controlled references for small cases,
but encountered tensor growth, high-dimensional integration, boundary leakage,
and production-scale memory costs. Project research therefore stopped treating
a more sophisticated non-Gaussian marginal likelihood as a near-term delivery
path. This is a research-history and prioritization statement, not a claim of
impossibility. The theory and evidence boundaries are summarized in the
private-group [Lognormal moment
note](https://github.com/openghg/inversions-knowledge/blob/main/docs/derivations/lognormal-moment-transformations.md)
and [non-Gaussian marginalization
note](https://github.com/openghg/inversions-knowledge/blob/main/docs/derivations/non-gaussian-aggregation-error-by-marginalization.md).

### Reporting functionals

Let $Lx$ be a native-grid country, region, sector, or other linear quantity
of interest. Exact reduced inference for \(\alpha\) does not by itself imply
that the conditional-mean reconstruction
$Lm+LU_*(\alpha-\Pi m)$ has the complete posterior law of $Lx$. General
reconstruction needs the unresolved functional blocks

$$
LB_\perp L^{\mathsf T}
\quad\text{and}\quad
LB_\perp H^{\mathsf T},
$$

and conditions the functional jointly with the observations.

### Model error is also a marginalization problem

The next major pressure is model-error treatment. The DUBFI work provides a
useful example: marginalizing a Gaussian uncertain affine forward operator
produces a state-dependent covariance,

$$
y\mid s
\sim
\mathcal N\!\left(H_0(s),
R_0+S_sWS_s^{\mathsf T}\right).
$$

The normalized likelihood contains both the residual quadratic and
\(\log\det R(s)\). A state-dependent covariance cannot be treated as a fixed
aggregation-error matrix, and omitting its determinant is a different
objective. Ensemble spread also does not correct a shared transport bias.

The determinant convention must be named in the model card. A determinant
weight of one is the normalized Gaussian uncertain-operator model; a weight of
zero is the Bruch-style unnormalized quadratic objective; and a fractional
weight is a calibration or sensitivity objective, not the same probability
model. Use a name such as `determinant_weight`, because \(\alpha\) already
denotes the retained state.

There is an additional interaction with coherent aggregation. Write

$$
\mu_{x\mid\alpha}:=m+U_*(\alpha-\Pi m),
\qquad
x\mid\alpha=\mu_{x\mid\alpha}+u,
\qquad
u\sim\mathcal N(0,B_\perp).
$$

Let \(\bar H:=\mathbb E(H\mid\alpha)\) and
\(\Delta H:=H-\bar H\), so
\(\mathbb E(\Delta H\mid\alpha)=0\). Assuming \(u\) and \(\Delta H\) are
conditionally independent given \(\alpha\), define

$$
\mathcal K_H(S)
=\mathbb E\!\left[\Delta H S\Delta H^{\mathsf T}\mid\alpha\right].
$$

The conditional covariance contains

$$
D_{\mathrm{obs}}
+\bar H B_\perp\bar H^{\mathsf T}
+\mathcal K_H(B_\perp)
+\mathcal K_H(\mu_{x\mid\alpha}\mu_{x\mid\alpha}^{\mathsf T}).
$$

The middle term \(\mathcal K_H(B_\perp)\) is operator uncertainty acting on
unresolved native flux. It belongs exactly once: averaging
\(HB_\perp H^{\mathsf T}\) over the operator already includes both it and the
mean-operator aggregation covariance. Because \(\Delta H u\) is bilinear, a
Gaussian likelihood for this joint construction is generally a moment
closure, even when the conditional native state and operator coefficients are
Gaussian. The bound reduction must therefore retain \(B_\perp\), or sufficient
product-space information to evaluate this interaction later, rather than
collapsing it permanently into one fixed observation covariance.

This model is not adopted as a production RHIME option by this ADR. It is an
explicit pressure test on the design: covariance components may be fixed or
state-dependent; may arise from flux aggregation, transport, boundary
conditions, temporal mismatch, or measurement; and must declare the
marginalization or discrepancy model that gives them meaning. The initial
semantic kernel must not make fixed additive covariance the only expressible
case.

## Modelling guidance for users

### Align basis regions with routine reported totals

For countries or regions that will be reported routinely, construct basis
regions so that they do not cross the corresponding reporting boundaries, and
use the same physical weights in basis construction and reported totals. If

$$
L=C\Pi
$$

for a labelled combination matrix $C$, then $L\delta=0$ for a coherent
residual satisfying \(\Pi\delta=0\). The declared reporting functional is in
the retained row space, so no unresolved contribution is required for that
total.

This requires more than visually non-crossing regions: the basis and report
must use identical mask fractions, area or flux weights, units, time
convention, and sign, and row-space membership must be checked. Coherent prior
transformation is still required when aligned regions are split; assigning the
ordinary independent prior to each new region changes the native model.

This is the recommended default, not a restriction on the model. New,
overlapping, or deliberately cross-basis quantities of interest still require
the general functional covariance and functional--observation cross-covariance
calculation. This guidance should be reused in the user manual planned under
#572; the model card should report the result of the alignment check.

## Decision

### 1. Separate six levels

The architecture will distinguish the following levels:

1. **Semantic model specification.** Product- and backend-neutral scientific
   identities, relations, options, and requested quantities of interest.
2. **Prepared inversion data.** Canonical observed and prior products with
   labelled coordinates and provenance, including `RhimePreparedInputs`.
3. **Bound mathematical model.** Concrete labelled operators, moments,
   covariance components, reduction/reconstruction products, units, and
   approximation ledger.
4. **Derived numerical views.** Dense, diagonal, LRPD, block, or matrix-free
   realizations of exact or approximated mathematical content, each with its
   own derived identity and diagnostics.
5. **Backend realization.** PyMC or analytic-Gaussian graph/system plus sampler
   and backend numerical choices.
6. **Compilation and output manifest.** Trace-variable, coordinate, artifact,
   and product-adapter mappings back to semantic IDs.

No lower level may silently create a scientific identity or change a declared
mathematical option.

### 2. Use a small relational semantic kernel

The kernel will describe relationships, not a generic component DAG. The
conceptual records below are not yet commitments to exact public Python names.

| Relation | Responsibility | Concrete example |
| --- | --- | --- |
| **Input source** | Reference to a stable acquisition/provenance identity; acquisition remains in the prepared-data layer. It is not automatically a flux component or state. | A named fossil-fuel inventory or biosphere flux product. |
| **Source group** | One or more sources plus an explicit combination and alignment policy. | Several fossil-fuel inputs aligned and combined for one model. |
| **Flux component** | Stable physical/reporting identity. This is the canonical semantic term; current RHIME APIs often call it a sector. | Fossil fuel, GPP, TER, or ocean. |
| **Native state model** | Native coordinates, mean, probability family, covariance/operator, units, and provenance. | A native-grid fossil scaling field with a correlated LogNormal prior. |
| **Basis/reduction** | Retained map, basis group, reduction semantics, and exactness/closure classification. | A reporting-aligned spatial operator $\Pi$. |
| **State block** | Stable retained degrees of freedom with prior moments, support, activity, and reconstruction identity. | The gathered retained flux-scaling state $\alpha_{\mathrm{flux}}$. |
| **Forward-model term** | A scientifically named contribution from a state block or fixed quantity to an observation-model mean. | Modelled pollution enhancement, a boundary contribution, or an offset. |
| **State-to-term coupling** | A labelled fixed transform with units, sign, direction, coordinate scope, and alignment policy. An uncertain coupling is represented by a state block. | A component-specific oxidative ratio and unit conversion coupling a fossil state to an O$_2$ term. |
| **Observation model** | Declared observed data, complete mean expression, covariance components, likelihood, and separate species/tracer/platform descriptors. | A CO$_2$ surface in-situ likelihood at labelled site--times. |
| **Covariance component** | Scientific origin and conditional dependence, separate from dense/LRPD/block/operator storage. | Temporal OU mismatch or covariance induced by coherent aggregation. |
| **Quantity of interest** | A declared native or retained functional, required weights, and reconstruction policy. | A GBR fossil-fuel total. |
| **Output view** | Reference to product-neutral grouping; concrete presentation remains in the manifest/output layer. | A country-by-flux-component product. |

“Flux component” avoids implying correspondence with inventory sectors such as
EDGAR categories and also covers processes such as GPP, TER, and ocean flux.
The established **one-sector** and **multisector** names remain valid group and
compatibility vocabulary: a multisector RHIME model is a model with multiple
flux components. Existing `SectorSpec` and configuration names need not be
renamed as part of this decision.

A simple inferred scaling of a flux belongs to a state block, not to the
state-to-term coupling relation. Likewise, if an oxidative ratio or other
coupling is uncertain, its value is another state and the forward-model term
must declare the resulting bilinear or nonlinear dependence. The coupling
relation is reserved for fixed scalar, labelled-array, or linear-operator
transforms rather than hiding prior-backed coefficients.

A state may contribute through several forward-model terms, including terms in
several observation models. Several terms or sources may share one state.
Components, basis groups, source provenance, and observation models are
orthogonal identities unless a model explicitly relates them.
The kernel references prepared input and requested output identities so their
relationships are visible, but it does not own data acquisition or product
presentation logic.

### 3. Bind coherent reduction as one scientific artifact

A coherent reduction will travel as one labelled, serializable aggregate, not
as unrelated arrays. “One artifact” means an immutable header and content
identity binding its mathematical products atomically; it does not require
large matrices, conditional-covariance actions, and functional products to be
serialized in one physical blob. It will contain or reference at least:

- native-model identity, probability family, and covariance/operator
  provenance;
- retained-state labels and arithmetic moments;
- $\Pi m$, $C_\alpha$, effective forward operator and intercept;
- unresolved observation covariance, or an exact labelled action sufficient to
  evaluate it, and its scientific component identity;
- requested functional covariance and cross-covariance products;
- a state-treatment record distinguishing structurally absent state blocks,
  deterministic conditioning/fixing, retained states, and coherent
  marginalization, including where uncertainty from an integrated state
  re-enters the reduced model;
- exactness and moment-closure labels; and
- one source-content identity tying the mathematical products to the same
  $B/H/\Pi$ inputs.

For a Gaussian native prior, the affine conditional state, retained prior, and
unresolved Gaussian likelihood in this artifact are exact. Gaussian and
LogNormal preparation nevertheless use the same labelled arithmetic-moment
pipeline: they project the native mean and covariance and derive the retained
moments, effective forward operator, and unresolved second moments together.
The affine conditional map and constant unresolved covariance are exact
Gaussian conditional moments; for a LogNormal native prior they are
linear-Bayes products derived from the first two moments. Fitting the retained
LogNormal law and closing the conditional residual distribution are further,
separate approximations. The artifact must therefore name the native family
and record each approximation or closure.
“Coherent” means that the moments and declared approximations form one
traceable chain from the native model; it does not make LogNormal reduction an
exact, distribution-free marginalization.

Dense, diagonal, LRPD, block, or matrix-free realizations are downstream
numerical views. Each view has its own derived identity, source-artifact
reference, diagnostics, and approximation label. An LRPD view may be built
from an operator without ever materializing a dense covariance, but it is
logically applied after the scientific covariance has been defined, to make
repeated likelihood evaluation and sampling feasible. It is not part of the
scientific definition of coherent reduction.

Any numerical padding used to realize ragged arrays belongs in a derived
numerical view or compilation manifest, not in the scientific state-treatment
record.

This prevents a retained prior from one native model being combined with an
aggregation covariance or reconstruction map from another.

### 4. Treat covariance origin and representation separately

An observation model may compose named covariance components such as:

$$
C(\theta)
= C_{\mathrm{measurement}}
+ C_{\mathrm{aggregation}}
+ C_{\mathrm{temporal}}
+ C_{\mathrm{transport}}(\theta)
+ C_{\mathrm{boundary}}(\theta)
+ C_{\mathrm{other}}(\theta).
$$

This additive ledger is valid only when the component residuals have declared
zero cross-covariance. Dependent mechanisms must be represented by one joint
component or by explicit named cross terms. Additivity is therefore a
scientific zero-cross-covariance assumption, not an independence statement;
the two coincide only under an appropriate joint Gaussian model.

Each component declares units, labels, provenance, whether it is fixed or
parameter-dependent, and whether it arose by marginalization or was introduced
as discrepancy. Dense, LRPD, block-diagonal, sparse, or operator form is a
separate numerical strategy. “Aggregation error” will not remain the umbrella
name for all dense model-data-mismatch covariance.

LRPD approximation must retain its rank, diagonal-tail policy, approximation
diagnostics, and validation criterion. Preserving marginal variance or a high
percentage of eigenvalue mass is not by itself evidence that the likelihood or
posterior is accurate. It is a linked derived numerical view of a declared
covariance, not a change to the native prior or reduction semantics.

### 5. Generate a mathematical model card before compilation

Every bound model must be renderable into a compact scientist-facing card. At
minimum it will contain:

- a table of native states, retained state blocks, forward-model terms, and
  observation models;
- the arithmetic prior means and covariance interpretation;
- one explicit mean equation for every observation model;
- a named covariance sum for every likelihood;
- units and coordinate domains for every term;
- state treatment: retained, fixed/conditioned, structurally absent, or
  marginalized;
- exact, moment-closed, truncated, and numerical-approximation declarations;
- quantities of interest, reporting-mask alignment checks, and reconstruction
  status; and
- backend and sampler information in a separate realization section.

An illustrative card fragment is:

```text
alpha_flux ~ LogNormal(
    arithmetic_mean=m_alpha,
    arithmetic_covariance=C_alpha,
    parameterization=whitened,
)

mu_co2 = H_m + H_alpha @ (alpha_flux - m_alpha) + boundary_co2

Cov(y_co2 | alpha_flux) =
    R_measurement + R_temporal + R_aggregation

native reduction: exact Gaussian marginalization; LogNormal uses exact arithmetic moments plus declared retained/residual closures
aggregation covariance: scientific operator identity=...
numerical view: LRPD(rank=..., diagonal_tail=preserved, gate=...)
country outputs: GBR/DEU/FRA/ITA aligned to retained row space
```

The model card is part of the saved run provenance. A model whose scientific
meaning can only be recovered from PyMC names or builder code fails this
decision.

### 6. Keep the concrete one-sector implementation

The default concrete one-sector builder remains:

- the most readable executable statement of simple RHIME;
- a regression oracle for the compiled path;
- a bridge for scientists familiar with the existing model; and
- a deliberately constrained example against which more general semantics can
  be explained.

It need not become the implementation of every new feature. New semantic
features should compile through the bound-model route. Parity tests will show
that the semantic representation can express the concrete model without
requiring the concrete code to absorb all extensions.

### 7. Keep the private flux plan private

`_FluxPlan` is a normalized PyMC flux-compilation plan. It usefully creates
states once, applies labelled terms, and provides parity between standard and
multisector construction. It also contains backend variable names, prior
dictionaries, one shared observation dimension, and an implicit total `mu`.

It is therefore not the semantic model and will not become public. During
migration it may remain a lowering target produced from a richer bound model.

### 8. Preserve compatibility through explicit manifests

Existing trace names, `InversionOutput` data, and generic/PARIS products will
remain compatible while migration proceeds. Compatibility adapters may emit
legacy `x`, `x_<suffix>`, and role mappings, but all new reconstruction will be
driven by an explicit manifest from semantic state or quantity IDs to:

- backend public and whitened variables;
- state selectors and full-state coordinates;
- prepared designs and covariance artifacts;
- deterministic output variables;
- product-neutral quantities; and
- generic, PARIS, or legacy names.

Renaming a PyMC variable must not change scientific output identity.

## Current implementation and delivery status

The table is deliberately explicit about planned work.

| Capability | Status on 2026-08-11 | Evidence or owner |
| --- | --- | --- |
| Canonical labelled prepared-input boundary, retained basis metadata, gathered state coordinates, and save/load support | **Merged/implemented** | `RhimePreparedInputs`, `BasisFunctions`, and current serialization code |
| Concrete and private compiled one/multisector PyMC construction with explicit state/term separation | **Merged/implemented** | `models/rhime.py`, `models/_rhime_compiler.py`; narrow #402/#403 work |
| Independent-cell basis-aware marginal prior-width projection and calibration | **Merged special case** | #521 |
| Fixed diagonal, dense, and LRPD aggregation-covariance consumers in RHIME likelihoods | **Merged consumer** | #564 |
| Labelled native covariance action and product blocks $C_\alpha$, $HB\Pi^{\mathsf T}$, and $HBH^{\mathsf T}$ | **In flight** | [#493](https://github.com/openghg/openghg_inversions/issues/493) |
| Backend-neutral correlated arithmetic-moment contract, LogNormal conversion/whitening, labels, and serialization | **Merged foundation** | [#571](https://github.com/openghg/openghg_inversions/pull/571), delivering the independently mergeable foundation of [#565](https://github.com/openghg/openghg_inversions/issues/565) |
| Built-in model-spec routing for one gathered correlated state | **Planned** | Remaining [#565](https://github.com/openghg/openghg_inversions/issues/565) work; preserve compatibility views without creating independent states |
| Coherent solve, centring, unresolved covariance, reconstruction products, and provenance ledger | **Planned** | [#566](https://github.com/openghg/openghg_inversions/issues/566), consuming #493 |
| Site/time OU mismatch composed with aggregation covariance | **Planned** | [#567](https://github.com/openghg/openghg_inversions/issues/567) |
| Cached conditional sampler lifecycle for structured covariance | **Planned** | [#568](https://github.com/openghg/openghg_inversions/issues/568) |
| Covariance-aware held-out conditional prediction | **Planned** | [#569](https://github.com/openghg/openghg_inversions/issues/569) |
| Covariance-safe derived outputs | **Planned** | [#570](https://github.com/openghg/openghg_inversions/issues/570) |
| User-facing covariance and coherent-reduction examples | **Planned** | [#572](https://github.com/openghg/openghg_inversions/issues/572) |
| Matrix-free separable covariance projection, class blocking, correlated country-calibrated prior covariance, exact Gaussian reduction tests, LogNormal moment conversion/fallback, LRPD construction, and reporting-aligned basis refinement | **Tested prototype** | `verification-games` source, tests, and curated result reports listed below |
| Mathematical projection theorem, exact functional correction, LogNormal feasibility, LRPD likelihood, and uncertain-operator marginalization | **Curated theory** | `inversions-knowledge` notes listed below |
| Minimal private semantic records, mathematical model cards, and typed compilation/output manifests | **Planned** | [#575](https://github.com/openghg/openghg_inversions/issues/575), a focused #528 deliverable |
| A complete public semantic/bound-model Python extension API | **Implied but unscheduled** | Deliberately waits until coherent covariance, inner/outer, and linked-observation-model pressure tests stabilize the private relations |
| Analytic-Gaussian realization of the same bound model | **Planned** | [#576](https://github.com/openghg/openghg_inversions/issues/576), consuming #493/#566/#575 as a mathematical and parity oracle |
| Validate the second axis, ordering, and uniqueness of current dense observation covariance before positional PyMC use | **Planned correctness slice** | [#573](https://github.com/openghg/openghg_inversions/issues/573); deliberately not hidden in #493 |
| Define positivity for a complete LRPD covariance assembled from zero and positive diagonal components | **Planned correctness slice** | [#573](https://github.com/openghg/openghg_inversions/issues/573), adjacent to #564/#567 |
| Make `output_format="none"` bypass gathered-state diagnostics and reconstruction | **Planned correctness slice** | [#574](https://github.com/openghg/openghg_inversions/issues/574), narrower than #570 |
| Permit a complete custom model builder to bypass built-in one-source-per-sector preflight | **Planned correctness slice** | [#574](https://github.com/openghg/openghg_inversions/issues/574); built-in builders keep their current validation |
| State-dependent ensemble transport covariance/DUBFI-like model error | **Deferred pressure test** | Scientific design and validation required before an OGI issue promises an API |
| General exact non-Gaussian conditional-fibre likelihoods | **Research boundary** | The theorem supplies a benchmark; tractable general implementation is not assumed |
| Cross-tracer transport-error covariance | **Research boundary** | Requires a joint scientific model and validation, not only a schema extension |

## Consequences

### Benefits

- Scientists can inspect the actual model before sampling without reading PyMC
  graph construction or output code.
- Basis changes can be classified as exact re-expressions, moment closures, or
  genuine changes of model.
- Priors, effective forward operators, unresolved covariance, and output
  reconstruction share one provenance identity.
- Correlated states, inner/outer partitions, grouped sources, and linked
  tracers no longer need fake sectors or suffix conventions.
- An analytic Gaussian realization becomes possible without reverse
  engineering a PyMC model.
- Covariance sources remain scientifically interpretable while numerical
  representations can evolve independently.
- Output products can preserve covariance and reconstruction semantics even if
  backend variable names change.

### Costs and risks

- There will be a period with three representations: public run/model options,
  the bound semantic model, and private backend plans. Clear ownership and
  one-way lowering are essential.
- Exact coherent preparation adds labelled solves, covariance diagnostics, and
  provenance artifacts that may be large.
- Lognormal closure, LRPD truncation, and state-dependent discrepancy introduce
  approximation and identifiability questions that a schema cannot resolve.
- Backward-compatible output adapters will temporarily duplicate metadata.
- A semantic layer can become incomprehensible if it grows into a generic class
  framework. The model card and worked pressure tests are acceptance gates
  against that failure.

## Rejected alternatives

### Treat current independent region priors as basis invariant

Rejected. Applying the same marginal prior to every basis region after a split
changes the implied native covariance and aggregate uncertainty. It is a valid
basis-specific model only when described as such.

### Make `_FluxPlan` the public semantic representation

Rejected. It mixes semantic identifiers with PyMC names and compiler metadata,
assumes one observation axis and implicit summation, and cannot carry coherent
reduction or output-functional semantics.

### Use PyMC variable suffixes as scientific identity

Rejected. Source provenance, component identity, latent state, output quantity,
and product name have different cardinalities. Future models break their
current convenient one-to-one mapping.

### Replace all builders in one refactor

Rejected. The migration must preserve working runs and outputs. The concrete
one-sector model is also intentionally retained as a readable reference and
parity oracle.

### Eliminate general functional reconstruction by requiring aligned bases

Rejected. Alignment is the default recommendation for a small declared set of
routine reporting masks, but users will request new or overlapping
functionals. The general covariance blocks are also an important correctness
guard.

### Represent fixing and marginalization with one inactive-state flag

Rejected. Fixing conditions a state on a value. Coherent marginalization
integrates its uncertainty into the remaining prior and likelihood. Confusing
them can discard or double-count uncertainty.

### Call every correlated residual “aggregation error”

Rejected. Aggregation, temporal mismatch, transport, boundary conditions,
measurement, and structural discrepancy have different scientific meanings
and dependence on the state.

### Select LRPD rank by explained variance alone

Rejected. Preserving the diagonal or a large eigenvalue fraction does not
guarantee acceptable log density, posterior, predictive score, or functional
uncertainty.

### Build a generic probabilistic component DAG or plugin registry first

Rejected. Current and foreseeable RHIME models fit a small directed relational
model. Public extension points should follow several working implementations,
not precede them.

## Pressure tests

The design is acceptable only if all of the following can be described without
inventing fake sectors, copying latent states, parsing backend suffixes, or
changing the semantic schema.

1. **Concrete standard RHIME:** one source, component, state, and observation
   model, with parity against the readable implementation.
2. **Current multisector RHIME:** several independent states and terms on a
   shared basis, summed into one observation-model mean.
3. **Ragged multisource state:** source-specific basis regions in configured
   non-lexical order without padding or positional slice inference.
4. **Coherent correlated state:** one gathered state with within- and
   cross-source covariance, a centred affine forward map, and unresolved
   observation covariance from the same native ledger.
5. **Reporting-aligned outputs:** declared countries exactly in the retained
   row space, plus a new cross-basis functional reconstructed through joint
   covariance.
6. **Inner/outer model:** basis groups orthogonal to sectors, with explicit
   support and no double counting.
7. **Linked CO2/O2:** one state contributing to CO2 and O2 observation models
   with different coordinates, fixed state-to-term couplings such as oxidative
   ratios and unit conversions, baselines, errors, and likelihoods.
8. **Temporal mismatch:** fixed aggregation covariance composed with sampled
   site amplitudes and labelled OU correlation.
9. **Conditional prediction:** fit and held-out covariance blocks reconstructed
   from saved artifacts without a live PyMC graph.
10. **Covariance-safe output:** posterior state and country products remain
    available with structured error, while joint predictive products preserve
    the complete covariance manifest.
11. **Analytic Gaussian realization:** the same bound model produces analytic
    prior predictive, posterior, and functional results matching the PyMC
    realization on tractable fixtures.
12. **Uncertain affine operator:** a DUBFI-like state-dependent covariance and
    normalized determinant can be represented without pretending it is fixed
    aggregation error, even if production implementation remains future work.
    The representation must distinguish exact fixed-state uncertain-operator
    marginalization from joint operator/unresolved-flux moment closure and
    give the operator--aggregation interaction one owner.

## Migration plan

### Phase 0: Establish the contract and visible model

1. Adopt this ADR and reserve “semantic model” for the backend-neutral
   scientific representation.
2. Update the issue 402/403 design note to point here for post-M9 semantics.
3. Define the model-card format and write cards for standard RHIME, coherent
   gathered CO2, inner/outer, and linked CO2/O2.
4. Keep a status table distinguishing merged, active planned, tested prototype,
   and unplanned work.
5. Deliver the four small correctness gaps scheduled in #573 and #574; do not
   bury them in a semantic-framework refactor.

### Phase 1: Stabilize the mathematical building blocks

1. Use the correlated-Lognormal foundation merged in PR #571 and complete #565
   routing through the ordinary model spec.
2. Complete #493: labelled native covariance actions and product-space
   projection without requiring dense native $B$.
3. Implement #566: solve-based coherent transformation, centring, unresolved
   covariance, reconstruction products, and one provenance ledger.
4. Port small dense Gaussian oracle tests from `verification-games` before
   optimizing storage or sampler behaviour.
5. Implement #567 on a generic named covariance-component boundary, while
   retaining its focused OU first case.

### Phase 2: Introduce the private semantic/bound model

1. Add the minimum records needed by the first four pressure tests; keep them
   private until names and cardinalities have survived use.
2. Normalize the current standard and multisector model specifications into
   this representation without numerical changes.
3. Lower the simple cases to the existing concrete/parity path and private
   `_FluxPlan` compiler where appropriate.
4. Implement #576's analytic-Gaussian realization and require parity of
   equations, labelled moments, and output functionals.
5. Implement #575's model card and persist it from the same bound object.

### Phase 3: Make outputs and validation semantic

1. Implement #570 with structured output/reconstruction manifests while
   retaining legacy roles and names.
2. Implement #569 conditional prediction from saved covariance descriptors and
   prepared artifacts.
3. Complete #572 with tested no-class, land/sea, and gathered multisource
   examples.
4. Replace suffix inference in new output paths; retire it from old adapters
   only after byte/schema or declared migration parity.

### Phase 4: Use new scientific shapes as design gates

1. Normalize inner/outer models without making inner/outer into sectors.
2. Normalize linked CO2/O2 with one state contributing to multiple observation
   models.
3. Introduce public component-extension protocols only after those two shapes
   and the coherent gathered state use the same internal relations.
4. Add #568's sampler lifecycle only after the bound model identifies variables
   and cache dependencies semantically.

### Phase 5: Scope improved model-error modelling

Before opening a production implementation issue, define a controlled
experiment and model card for:

- fixed ensemble transport covariance;
- state-dependent uncertain-operator covariance with a normalized determinant;
- within-ensemble and out-of-ensemble mismatch;
- separation from temporal, boundary, aggregation, and mean-bias components;
- CO2 and eventual linked-tracer behaviour; and
- dense, LRPD, localized, block, or matrix-free numerical strategies.

The result should decide what belongs in the retained state, what is exactly
marginalized, what is moment-closed, and what is an explicit discrepancy.

## Acceptance criteria for issue #528

The design part of #528 is complete when:

- this ADR is reviewed and its status is accepted;
- the distinct identities and cardinalities in the relational kernel are
  agreed;
- standard, coherent gathered, inner/outer, and linked CO2/O2 model cards are
  present and mathematically interpretable without reading PyMC code;
- the boundary between semantic specification, prepared input, bound
  mathematical model, compiler plan, and output manifest is explicit;
- exact marginalization, moment closure, covariance truncation, deterministic
  fixing, and structural absence have distinct serialized representations;
- a native coherent reduction and a basis-specific current RHIME model can both
  be represented honestly;
- covariance origin is separate from dense/LRPD/block/operator
  representation;
- a state can contribute to several forward-model terms and observation
  models, and several sources or terms can share one state;
- quantities of interest record reporting weights, alignment, covariance needs,
  and reconstruction policy;
- backend names can be changed without changing semantic output identity;
- the readable concrete one-sector implementation has an explicit continuing
  role and parity contract; and
- follow-up implementation slices have issue ownership or are explicitly
  labelled as not yet scheduled.

Implementation of every pressure test is not required to close the ADR portion
of #528. Publishing a public extension API should remain later work until the
private representation has survived coherent covariance, inner/outer, and
linked-observation-model implementations.

## Open questions

- Which semantic and bound-model records should become public, and at what
  stability level?
- Should the scientist-facing specification be Python-only, configuration
  backed, or serializable to a restricted declarative format?
- What unit system and labelled-array validation are required at every
  relation boundary?
- Which covariance operators must support cross-source blocks in their first
  public version?
- How are state-dependent covariance components represented without binding
  the semantic model to one autodiff backend?
- Which conditional-fibre products must be retained for native-grid output,
  and which can be generated on demand?
- What quantitative gates should approve an LRPD approximation: log-density
  error, posterior distance, predictive score, functional error, or a declared
  combination?
- When a coupling parameter is itself a state, how will bilinear state--state
  terms be represented and identified without overgeneralizing the first
  compiler?
- How should semantic content identities compose across prepared-data caches,
  covariance operators, model cards, traces, and derived output artifacts?

## Evidence and references

### `openghg_inversions`

- [Issues 402/403 builder strategy design](issues_402_403_builder_strategy_design.md)
  supplies the earlier source/component/state/term/channel/output vocabulary
  and the original pressure tests. This ADR replaces “channel” with the more
  explicit observation-model terminology for active design work.
- `openghg_inversions/models/rhime.py` contains the readable concrete builder,
  `RhimeModelSpec`, and current one-source-per-sector `SectorSpec`.
- `openghg_inversions/models/_rhime_compiler.py` contains the private
  normalized flux plan and loop-sum compiler.
- `openghg_inversions/basis/prior_uncertainty.py` documents and implements the
  independent-grid-cell special case.
- `openghg_inversions/observation_error.py` and
  `openghg_inversions/models/likelihoods.py` contain the current fixed dense,
  diagonal, and LRPD covariance consumers.
- Active work is tracked in
  [#493](https://github.com/openghg/openghg_inversions/issues/493),
  [#565](https://github.com/openghg/openghg_inversions/issues/565),
  [#566](https://github.com/openghg/openghg_inversions/issues/566),
  [#567](https://github.com/openghg/openghg_inversions/issues/567),
  [#568](https://github.com/openghg/openghg_inversions/issues/568),
  [#569](https://github.com/openghg/openghg_inversions/issues/569),
  [#570](https://github.com/openghg/openghg_inversions/issues/570), and
  [#572](https://github.com/openghg/openghg_inversions/issues/572). Focused
  review outcomes are scheduled in
  [#573](https://github.com/openghg/openghg_inversions/issues/573)--
  [#576](https://github.com/openghg/openghg_inversions/issues/576).

### `verification-games`

These paths identify transferable evidence without depending on private data
or result locations:

- `src/verification_games/grid_covariance.py` and
  `tests/test_grid_covariance.py`: separable covariance application and
  projection against explicit Kronecker/dense calculations;
- `src/verification_games/scale_priors.py`: arithmetic-to-latent Lognormal
  moment construction and the scaled-latent fallback;
- `src/verification_games/prior_covariance.py`: correlated basis covariance and
  aggregate-uncertainty calibration;
- `src/verification_games/covariance_approximation.py` and
  `src/verification_games/rhime_covariance.py`: dense and LRPD covariance
  prototypes;
- `tests/test_controlled_aggregation_error_coarsening.py`: reduced/native
  Gaussian posterior projection comparison;
- `docs/plans/rhime_tentative_decisions.md`: provisional modelling defaults,
  rejected IID-cell default, preferred Lognormal construction, and reporting
  requirements; and
- `docs/plans/rhime_experiment_report_index.md`: curated dense-prior,
  Lognormal, whitening, aggregation-covariance, and sampling evidence.

### `inversions-knowledge`

- `docs/derivations/posterior-projection-and-exact-marginalization.md`:
  general projection theorem, Gaussian specialization, evidence invariance,
  numerical qualifications, and reconstruction boundary;
- `docs/topics/aggregation-error-basis-functions-and-reported-functionals.md`:
  exact native-functional conditioning, reporting-aligned bases, and the
  contrast with current RHIME;
- `docs/derivations/lognormal-moment-transformations.md`: exact arithmetic and
  latent moment maps, feasibility, and closure boundary;
- `docs/derivations/low-rank-plus-diagonal-gaussian-likelihood.md`: normalized
  Woodbury likelihood, diagonal-tail choices, and validation requirements;
- `docs/topics/model-data-mismatch-with-coherent-aggregation-covariance.md`:
  separation of observation, aggregation, temporal, transport, and structural
  mismatch; and
- `docs/topics/ensemble-transport-uncertainty-and-dubfi.md` and
  `docs/derivations/uncertain-affine-operator-marginalization.md`:
  state-dependent covariance by marginalizing an uncertain affine operator,
  determinant requirements, provenance limits, and validation cautions.
