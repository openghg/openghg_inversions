# From native scientific model to trustworthy inversion

## Scientific and design overview for issue #528

This document is written to work as either a short presentation or a
standalone overview. It explains why recent verification work has changed the
requirements for the proposed semantic model, what is already available, and
what the next implementation should preserve.

Status snapshot: OpenGHG Inversions `devel` at `7665d41`, 10 August 2026.

> **Architecture status (2026-08-18):** Retain this document's scientific
> account of native covariance, coherent reduction, aggregation error, and
> linked observation channels. Its semantic-model/compiler delivery proposal
> is superseded by the production
> [`run_rhime` readability plan](../run_rhime_readability_and_modifiability.md)
> and [model-family expansion plan](../rhime_model_family_expansion.md).

The central proposal is:

> Define one probability model at its scientifically meaningful native
> resolution. Derive every reduced state, likelihood covariance, and reported
> quantity from that model. Treat basis functions as a computational and
> reporting choice, not as the place where the scientific prior is invented.

---

# 1. Why revisit the model design now?

OpenGHG Inversions has recently gained useful infrastructure:

- labelled and ragged basis-state layouts;
- serializable prepared inputs;
- explicit model and likelihood builders;
- dense and low-rank-plus-diagonal covariance likelihoods;
- variable-role metadata; and
- a private compiler plan that separates state construction from forward
  terms.

The verification-games work has applied substantially more pressure than
adding another sector, gas, or region:

- priors are correlated across space and sometimes across sources;
- one joint latent variable can represent several scientific components;
- dropping a state may mean *marginalizing it*, not fixing it;
- basis reduction changes the prior, forward operator, residual covariance,
  and output reconstruction together;
- observation errors are correlated in time; and
- future transport-error models make covariance depend on the state itself.

These are relationships between mathematical objects. They cannot be made
coherent by adding another PyMC variable name or another runner branch.

---

# 2. The current basis-level prior changes the scientific model

The familiar RHIME model gives each basis-region scaling coefficient the same
prior distribution:

$$
\alpha_r \overset{\mathrm{iid}}{\sim} p_0,
\qquad r=1,\ldots,R.
$$

This is simple to explain and implement, but it makes the chosen partition part
of the prior model.

Suppose a country is split into \(k\) equally weighted basis regions and each
coefficient has independent variance \(\sigma^2\). The variance of the mean
country scaling is

$$
\operatorname{Var}\!\left(\frac{1}{k}\sum_{r=1}^k\alpha_r\right)
=\frac{\sigma^2}{k}.
$$

Refining the basis therefore reduces country uncertainty by \(1/\sqrt{k}\),
even though no new scientific information has been supplied. Conversely,
forcing a desired country-total uncertainty using independent native cells can
require implausibly large marginal cell uncertainty.

Other inconsistencies follow:

- a small and a large basis region receive the same marginal scale prior;
- correlations implied by common processes are discarded;
- country uncertainty changes when an optimization-oriented partition changes;
- different sectors can be forced independent by software layout; and
- omitted within-region variability is treated as zero unless an additional
  covariance is supplied.

The recently merged independent-cell prior calibration utilities are a useful
special case and diagnostic. They do not yet supply the full correlated,
coherently reduced model described here.

---

# 3. Native-model-first prior construction

Start with a native state \(x\), normally aligned to the native flux grid and
source structure:

$$
x\sim p_x(m,B).
$$

Here \(m\) is the native mean and \(B\) describes scientifically meaningful
uncertainty. Its structure may include:

- marginal variability related to flux magnitude or process class;
- spatial correlation length and anisotropy;
- barriers or independent classes such as land and ocean;
- correlations between sources or sectors; and
- calibration against declared country or regional uncertainty targets.

A retained state is then defined by a labelled operator:

$$
\alpha=\Pi x.
$$

The prior on \(\alpha\) is induced by the native model. It is not independently
chosen after the basis is made.

For a Gaussian native prior,

$$
E[\alpha]=\Pi m,
\qquad
C_\alpha=\Pi B\Pi^{\mathsf T}.
$$

Region size, shape, class membership, and correlation now affect uncertainty
through the same probability model. The basis can be chosen for computational
cost and resolution where observations are informative, while the native
scientific assumptions remain fixed.

---

# 4. Exact Gaussian reduction: what “coherent” means

Let

$$
x\sim\mathcal N(m,B),
\qquad
y=Hx+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,R),
\qquad
\alpha=\Pi x.
$$

Define

$$
C_\alpha=\Pi B\Pi^{\mathsf T},
\qquad
U_*=B\Pi^{\mathsf T}C_\alpha^{-1},
\qquad
H_\alpha:=HU_*=HB\Pi^{\mathsf T}C_\alpha^{-1},
$$

and the unresolved conditional covariance

$$
B_\perp=B-U_*C_\alpha U_*^{\mathsf T}.
$$

The first implementation requires \(\Pi\) to have full row rank in the
\(B\)-metric, so \(C_\alpha\) is positive definite. Redundant retained
coordinates fail validation; generalized-inverse semantics are not an
automatic numerical fallback.

The exact reduced model is

$$
\alpha\sim\mathcal N(\Pi m,C_\alpha),
$$

$$
y\mid\alpha
\sim
\mathcal N\!\left(
Hm+H_\alpha(\alpha-\Pi m),
R+HB_\perp H^{\mathsf T}
\right).
$$

Three quantities must therefore change together:

1. the reduced prior covariance \(C_\alpha\);
2. the effective, centred forward model \(H_\alpha\); and
3. the covariance from unresolved native variability
   \(HB_\perp H^{\mathsf T}\), often called aggregation error.

Supplying only a correlated reduced prior while retaining the old basis-sum
forward operator is not the same reduced model. Supplying aggregation
covariance independently is also insufficient unless all blocks have shared
native-model provenance.

---

# 5. What is invariant, and what is not?

For exact marginalization,

$$
p(\alpha\mid y)=\Pi_*p(x\mid y).
$$

Bayesian updating commutes with reduction: each basis produces the appropriate
marginal of the same native posterior. Verification-games has confirmed this
for three genuinely different real operators and bases: across 20 independent
draws, pairwise functional posterior means differed by at most
\(8.34\times10^{-12}\) posterior standard deviations. Omitting or diagonalizing
the unresolved covariance led to differences of tens of posterior standard
deviations in the same tests.

All exact projections have the same evidence under the common native model, so
evidence cannot choose between them. A useful basis is instead chosen for
computational cost, reporting support, observation sensitivity, and the quality
of any approximations that replace exact marginalization.

This does **not** mean that a reduced posterior reconstructs every native-grid
quantity. For a reported functional \(q=Qx\):

- if \(Q=L\Pi\), then \(q=L\alpha\) and the retained state is sufficient;
- otherwise exact uncertainty also needs unresolved functional covariance
  \(QB_\perp Q^{\mathsf T}\) and functional--observation cross-covariance
  \(QB_\perp H^{\mathsf T}\).

## Practical scientific recommendation

Do not let basis regions cross countries or scientific regions that must be
reported. OpenGHG Inversions can construct reporting-aligned bases. This makes
important totals functions of the retained state and greatly simplifies exact
reporting.

This recommendation does not by itself repair an incoherent prior, and it does
not justify claims about arbitrary native-grid reconstruction. It is a strong,
simple default that keeps common reporting questions inside the retained row
space.

---

# 6. Why prior covariance is scientifically necessary

Independent native-cell perturbations average away rapidly over a country.
Obtaining, for example, a 50% prior uncertainty on a country total can then
require enormous cell-level marginal variances and pathological prior draws.

Positive spatial correlation changes the scaling:

$$
\operatorname{Var}(w^{\mathsf T}x)=w^{\mathsf T}Bw.
$$

The cross terms in \(B\) allow realistic aggregate uncertainty without
requiring every cell to vary implausibly. Correlation length and process class
then become visible scientific assumptions rather than accidental consequences
of basis size.

Verification experiments with correlated native covariance achieved large
country-total uncertainty with substantially smaller grid- and state-level
marginal uncertainty than the independent alternative. This was the practical
motivation for making native covariance, rather than regionwise prior width,
the primary object.

A useful configuration should therefore express:

- the native marginal-amplitude rule;
- correlation kernels and their physical scale;
- independent or correlated process/source blocks;
- the regional functionals used for calibration; and
- whether target uncertainty is absolute, relative, or source-specific.

These choices belong in the mathematical model card and its provenance, not in
PyMC variable construction.

---

# 7. Numerical strategy: never materialize the native covariance

A native grid covariance may be far too large to store. The required reduced
products are nevertheless limited:

$$
\Pi B\Pi^{\mathsf T},
\qquad
HB\Pi^{\mathsf T},
\qquad
HBH^{\mathsf T}.
$$

For a separable spatial kernel,

$$
K=K_{\mathrm{lat}}\otimes K_{\mathrm{lon}},
$$

its action on a reshaped field \(X\) is

$$
K\,\operatorname{vec}(X)
=\operatorname{vec}\!\left(
K_{\mathrm{lat}}XK_{\mathrm{lon}}^{\mathsf T}
\right),
$$

up to the declared vectorization convention. Class masks and source blocks can
be applied around this operator. Products can be accumulated by source and in
observation chunks.

The implementation contract should expose an operation such as “apply native
covariance to labelled columns,” rather than require one dense \(N\times N\)
matrix. Dense matrices remain valuable as small-problem test oracles.

---

# 8. Positive states: multivariate lognormal moment closure

For a positive scaling state \(a=\exp(z)\), the desired scientific inputs are
usually arithmetic moments:

$$
E[a]=m_a,
\qquad
\operatorname{Cov}(a)=C_a.
$$

When a matching multivariate lognormal distribution exists,

$$
(\Sigma_z)_{ij}
=\log\!\left(1+\frac{(C_a)_{ij}}{(m_a)_i(m_a)_j}\right),
$$

$$
(\mu_z)_i=\log (m_a)_i-\frac12(\Sigma_z)_{ii}.
$$

Use a prior-whitened realization:

$$
\eta\sim\mathcal N(0,I),
\qquad
z=\mu_z+L_z\eta,
\qquad
a=\exp(z),
$$

where \(L_zL_z^{\mathsf T}=\Sigma_z\). Verification runs found this
parameterization computationally robust for correlated states.

In one all-site prototype there were 6735 observations, 30 sites, and 100
states. The latent-scaled fallback used \(\tau\approx0.8145\); four-chain
sampling had zero divergences, maximum \(\hat R=1.01\), and minimum effective
sample size 1547. This is useful feasibility evidence, not current `devel`
coverage or proof that every non-Gaussian closure is accurate.

Important distinction:

- transforming valid arithmetic moments into latent Gaussian moments is exact;
- representing a sum of native lognormal variables by another multivariate
  lognormal is a moment closure, not exact distributional marginalization.

Gaussian and LogNormal preparation can therefore share the same labelled
arithmetic-moment reduction: project the native mean and covariance, derive
the retained moments and effective forward operator, and propagate unresolved
second moments. The probability-family interpretation is different. For a
Gaussian native prior the affine conditional reduction is exact; for a
LogNormal prior the affine conditional map and constant unresolved covariance
are linear-Bayes products derived from the first two moments. The retained
LogNormal fit and residual-distribution closure are further declared
approximations. “Coherent” means that these moments and closures remain tied to
one native model, not that the LogNormal calculation is an exact
distribution-free marginalization.

---

# 9. Lognormal fallback when exact moment matching is infeasible

Elementwise `log1p` of a requested arithmetic covariance is not guaranteed to
produce a positive-definite latent covariance.

The tested fallback is:

1. choose a scientifically interpretable, positive-definite latent correlation
   shape \(\Sigma_{z,0}\);
2. set \(\Sigma_z=\tau^2\Sigma_{z,0}\); and
3. solve for \(\tau\) so that one declared weighted aggregate has its target
   arithmetic variance.

This preserves positivity and a chosen correlation shape. It matches the
declared aggregate, not the full requested elementwise covariance.

The semantic model must record which construction was used:

- `arithmetic_moment_match`;
- `latent_scaled_to_functional`; or
- another explicit closure.

Silently falling back would erase a meaningful scientific approximation.

---

# 10. Observation covariance is a sum of scientific components

Write the likelihood covariance as a ledger:

$$
R_{\mathrm{total}}
=R_{\mathrm{measurement}}
+R_{\mathrm{aggregation}}
+R_{\mathrm{temporal}}
+R_{\mathrm{transport}}
+R_{\mathrm{boundary}}
+\cdots.
$$

Writing this as a sum is itself a model assumption: the component residuals
must have zero cross-covariance. Dependent mechanisms must instead be one
joint component or contribute explicit cross terms. The covariance ledger
records that zero-cross-covariance assumption rather than silently treating
every mechanism as additive. It is an independence assumption only under an
appropriate joint Gaussian model.

Each component needs:

- a scientific meaning and units;
- labelled observation axes;
- provenance and parameters;
- fixed or state-dependent status;
- exact, estimated, or approximate status; and
- a numerical representation.

“Aggregation error” is therefore one covariance source, not the name of the
general covariance machinery.

The representation is a separate decision. A scientifically temporal
covariance can be stored densely, in blocks, or as low rank plus diagonal; its
meaning should not change with storage.

---

# 11. Temporal correlation: empirical evidence and model choice

For irregular observations at the same site, a fixed Ornstein--Uhlenbeck
component uses elapsed time directly:

$$
(R_{\mathrm{OU},s})_{ij}
=\sigma_s^2
\exp\!\left(-\frac{|t_i-t_j|}{\tau_s}\right).
$$

In the WUR BASE diagnosis:

- the IID model left adjusted one-hour innovation correlation of \(+0.629\);
- fixed 3 h and 4 h OU models reduced it to \(-0.008\) and \(-0.067\);
- five-fold held-out joint log predictive density improved from \(-7511\) to
  \(-6214\) and \(-6187\);
- RMSE improved from \(3.923\) to \(3.627\) and \(3.593\); and
- interval coverage improved from \(0.770\) to \(0.840\) and \(0.867\).

A separate known-truth gate using a predeclared common 8 h OU model reduced a
spurious inner-PARIS total-flux error from 1688 to 662 Tg CO2 yr\(^{-1}\), about
61%.

This is strong evidence that temporal covariance is necessary in that
workflow. It does not establish one universal timescale or a final hierarchical
model. Timescales and amplitudes require blocked validation and protected-data
discipline.

---

# 12. Numerical strategy: low rank plus positive diagonal

Dense observation covariance becomes expensive when there are thousands of
observations and the likelihood is evaluated repeatedly. Approximate

$$
D:=\operatorname{diag}(d),
\qquad
F\in\mathbb R^{n\times r},
\qquad
R\approx FF^{\mathsf T}+D,
\qquad d_i>0.
$$

Woodbury identities and the matrix determinant lemma reduce repeated solves and
log determinants to the retained rank:

$$
R^{-1}
=D^{-1}-D^{-1}F(I_r+F^{\mathsf T}D^{-1}F)^{-1}F^{\mathsf T}D^{-1}.
$$

Current devel can consume fixed dense, diagonal, and LRPD covariance in the
PyMC likelihood. Verification prototypes also construct and assess the
approximation.

LRPD is a downstream numerical approximation to an already declared dense or
operator-defined observation covariance. It does not define coherent
reduction or change the scientific source of the covariance. It may be
constructed directly from covariance actions without materializing the dense
matrix, but its purpose is to make repeated likelihood evaluations and
sampling feasible. Its identity, rank, diagonal-tail policy, and validation
evidence must remain linked to the source covariance it approximates.

Two safeguards are essential:

1. the diagonal of the complete covariance representation being solved must
   remain strictly positive; an individual component may have a zero tail when
   another declared component makes the assembled covariance proper; and
2. rank must be selected using likelihood- or posterior-aware diagnostics.

Explained covariance variance is not sufficient: an all-site rank-512
prototype retained more than 99.5% of covariance variance but still failed its
KL validation gate. Moderate tested cases achieved approximately 4--8 times
speed-up with clean numerical parity.

---

# 13. A scientist-facing model card

The semantic model should render this *before* PyMC is built.

## Identity and support

| Object | Scientific identity | Support |
|---|---|---|
| Native state | `co2_flux_scale` | native grid × source |
| Retained state | `inner_paris_flux_scale` | source × reporting-aligned region |
| Observation model | `co2_concentration` | site × time; gas/tracer/platform recorded separately |
| Output | `country_flux_total` | country × source |

## Prior

$$
x\sim\mathcal N(m,B_{\mathrm{space,source}}),
\qquad
\alpha=\Pi x,
\qquad
C_\alpha=\Pi B\Pi^{\mathsf T}.
$$

- native marginal amplitude: flux-relative by source;
- spatial kernel: separable exponential, length scales shown with units;
- source cross-covariance: explicit matrix;
- calibration target: declared country-total relative uncertainty;
- basis policy: regions may not cross requested reporting countries.

## Observation mean

$$
\mu_{\mathrm{co2}}
=Hm+H_\alpha(\alpha-\Pi m)+b_{\mathrm{boundary}}.
$$

## Observation covariance

$$
R_{\mathrm{co2}}
=R_{\mathrm{measurement}}
+R_{\mathrm{temporal,OU}}
+R_{\mathrm{aggregation}}.
$$

## Numerical realization and assurance

- aggregation reduction: exact linear-Gaussian;
- positive-state realization: lognormal arithmetic-moment closure, if selected;
- temporal parameters: fixed from declared blocked validation;
- covariance representation: dense or LRPD, with approximation diagnostics;
- backend: analytic Gaussian oracle or PyMC realization;
- output status: exact from retained state, or requires unresolved-output terms.

This is the desired level: equations, model structure, and
mathematically/scientifically meaningful options. Backend names and compiler
plumbing belong in a separate compilation manifest.

---

# 14. A second model card: future uncertain transport

The next scientific pressure is uncertainty in the forward operator itself.
Let a Gaussian uncertain affine transport operator have mean \(H_0\),
coefficient covariance \(W\), and state-dependent evaluation matrix \(E_s\).
Exact marginalization gives

$$
y\mid s
\sim
\mathcal N\!\left(
H_0(s),
R_0+E_sWE_s^{\mathsf T}
\right).
$$

Because the covariance depends on \(s\), the normalized likelihood contains
both

$$
\frac12 r(s)^{\mathsf T}R(s)^{-1}r(s)
\quad\text{and}\quad
\frac12\log\det R(s).
$$

A `determinant_weight` of one gives that normalized Gaussian model. Zero gives
an unnormalized Bruch-style quadratic objective, while a fractional value is a
calibration or sensitivity objective; these are distinct model-card choices,
not interchangeable implementations of one likelihood.

This DUBFI-style pressure is another marginalization problem, but it is more
demanding than fixed aggregation or temporal covariance:

- a covariance component is state dependent;
- low-rank factors may change with the state;
- log-determinant semantics are part of the probability model;
- ensemble spread does not correct common transport bias; and
- CH4 ensemble evidence does not automatically validate a CO2/O2 model.

The semantic model should therefore permit a covariance component to depend on
named state blocks, without making DUBFI or one ensemble parameterization part
of the core abstraction.

Coherent aggregation makes this pressure test sharper. Let
\(\mu_{x\mid\alpha}:=m+U_*(\alpha-\Pi m)\),
\(x\mid\alpha=\mu_{x\mid\alpha}+u\), and
\(u\sim\mathcal N(0,B_\perp)\). Also let
\(\bar H:=\mathbb E(H\mid\alpha)\) and
\(\Delta H:=H-\bar H\). If \(u\) and \(\Delta H\) are conditionally
independent given \(\alpha\), the covariance contains

$$
D_{\mathrm{obs}}
+\bar H B_\perp\bar H^{\mathsf T}
+\mathcal K_H(B_\perp)
+\mathcal K_H(\mu_{x\mid\alpha}\mu_{x\mid\alpha}^{\mathsf T}),
\qquad
\mathcal K_H(S)=\mathbb E(\Delta H S\Delta H^{\mathsf T}\mid\alpha).
$$

The operator--aggregation interaction \(\mathcal K_H(B_\perp)\) must be
included once. The bilinear term \(\Delta H u\) also means that a Gaussian
likelihood for this joint construction is a moment closure, not the exact
fixed-state uncertain-affine result. This is why the coherent-reduction
artifact must preserve unresolved-state information rather than saving only a
single fixed aggregation covariance.

---

# 15. What the semantic model must contain

The semantic model should be a small mathematical intermediate
representation, not a generic PyMC graph and not a new plugin framework.

| Relation | Meaning |
|---|---|
| Flux component | canonical physical/reporting identity such as fossil fuel, GPP, TER, or ocean; called a sector in current RHIME compatibility vocabulary |
| Native model | mean, covariance/operator, units, labels, provenance |
| State block | stable identity, support, prior moments/family, selectors |
| State treatment | structural, retained, fixed, or marginalized |
| Reduction | \(\Pi\), induced moments, effective forward terms, unresolved covariance |
| Forward-model term | scientifically named contribution to an observation-model mean |
| State-to-term coupling | fixed labelled transform such as an oxidative ratio and unit conversion; an uncertain coupling is another state |
| Mean expression | explicit affine terms and named sums |
| Covariance component | scientific source, observation-model grouping, parameters, dependencies |
| Quantity of interest | labelled functional and reconstruction requirements |
| Approximation ledger | exact, moment closure, truncated, numerical approximation |
| Compilation manifest | serialized semantic ID → backend variables and saved products |

The coherent mathematical reduction should travel as one atomic aggregate.
Its prior, effective design, intercept, unresolved covariance,
state-treatment record, and provenance must not become unrelated arrays that
can be mixed accidentally. Downstream numerical views such as LRPD factors
have their own derived identities, diagnostics, and source-artifact reference;
they are not part of the scientific definition of the reduction. Neither the
mathematical aggregate nor its views need to store large dense arrays when
labelled operator actions provide the required products.

---

# 16. Architecture: equations first, backends second

```text
scientific specification
    sources, native priors, observation models, covariance sources, outputs
                         │
                         ▼
canonical prepared inputs
    observations, native products, labels, units, provenance
                         │
                         ▼
bound mathematical model
    labelled states, equations, reductions, covariance ledger, QoIs
                         │
                         ▼
derived numerical views/artifacts
    aligned arrays, operators, dense/LRPD factors, approximation evidence
                    ┌────┴────┐
                    ▼         ▼
          analytic Gaussian   PyMC realization
                    └────┬────┘
                         ▼
              compilation/output manifest
```

The simple concrete one-sector RHIME model should remain as an executable,
scientist-readable reference and regression oracle. Some duplication is an
acceptable hedge while the compiler matures. It can later remain as a
documented reference even if production execution converges on one semantic
normalization path.

The private `_FluxPlan` is useful compiler groundwork, but it is not yet this
semantic model: it remains flux-oriented and contains backend names and prior
arguments.

---

# 17. Status: distinguish merged, planned, and researched work

## Legend

- **MERGED** — available on current `devel`.
- **IN FLIGHT** — active implementation work that has not yet merged.
- **PLANNED** — represented by an open issue or draft PR with a stated owner;
  not yet available to users.
- **VERIFIED PROTOTYPE** — implemented and exercised in verification-games,
  but not integrated into `devel`.
- **DEFERRED PRESSURE TEST** — deliberately shapes today's interface, but has
  no promised OGI implementation until the scientific model is fixed.
- **OPEN RESEARCH** — mathematical or empirical choices still require
  investigation; not merely an implementation omission.

| Capability | Status | Qualification or next home |
|---|---|---|
| Labelled rectangular and ragged basis layouts | **MERGED** | Strong state-alignment foundation |
| Serializable prepared RHIME inputs | **MERGED** | Natural persistence boundary for reduced artifacts |
| Dense/diagonal/LRPD likelihood consumption | **MERGED** | Fixed aggregation-specific input contract today |
| Covariance-axis validation and complete LRPD positivity | **PLANNED CORRECTION** | Issue #573, before structured covariance composition |
| Complete custom PyMC model and likelihood builders | **MERGED; CORRECTION PLANNED** | Valuable escape hatches, not backend-neutral composition; #574 owns preflight/no-output boundaries |
| Concrete one-sector reference model | **MERGED; retain** | Readable reference and compiler regression oracle |
| Relational semantic model and model card/manifest | **PLANNED** | Issues #528 and #575; this overview and its companion ADR revise the scope |
| Grouped state-vector layouts and metadata | **PLANNED / PARTLY MERGED** | Issue #456; existing MultiIndex groundwork |
| Output variable-role/accessor research | **PLANNED** | Issue #444; distinct from state grouping |
| Backend namespace and component-role mappings | **PLANNED / PARTLY MERGED** | Issue #532 and manifest issue #575 |
| Backend-neutral correlated arithmetic moments, LogNormal conversion/whitening, labels, and serialization | **MERGED FOUNDATION** | PR #571, delivering the independently mergeable foundation of issue #565 |
| Built-in gathered correlated-state routing | **PLANNED** | Remaining issue #565 work |
| Exact Gaussian coherent reducer | **VERIFIED PROTOTYPE; PLANNED FOR MIGRATION** | Issue #566; must produce all linked reduced blocks atomically |
| Analytic-Gaussian semantic realization | **PLANNED** | Issue #576; independent mathematical and PyMC parity oracle |
| Matrix-free separable and cross-source native covariance | **VERIFIED PROTOTYPE; IN FLIGHT** | Issue #493; operator-first implementation |
| Lognormal arithmetic moment matching and fallback | **VERIFIED PROTOTYPE; PLANNED FOR MIGRATION** | Issue #565; record closure and fallback provenance |
| LRPD construction and rank validation | **VERIFIED PROTOTYPE; PLANNED** | Issue #566; devel currently consumes prepared factors only |
| Fixed elapsed-time temporal OU covariance | **VERIFIED PROTOTYPE; PLANNED** | Issue #567; timescale remains application-specific |
| Inner/outer and linked CO2/O2 composition | **PLANNED / PARTLY PROTOTYPED** | Issues #407--#413 and #456 provide semantic pressure tests |
| Gaussian output functionals outside retained row space | **VERIFIED PROTOTYPE; PLANNED** | Issues #566, #570, and analytic oracle #576; add unresolved functional and cross-covariance terms when required |
| State-dependent uncertain-operator covariance | **DEFERRED PRESSURE TEST / OPEN RESEARCH** | DUBFI informs the interface; CO2/O2 validation remains future work |
| Non-Gaussian output reconstruction outside retained row space | **OPEN RESEARCH** | Prefer reporting-aligned bases while the general treatment matures |
| General non-Gaussian exact marginalization | **OPEN RESEARCH** | Current lognormal approach is moment closure |

“Planned” and “not merged” are intentionally not synonyms for “forgotten.”
Some pieces already have issue-level homes; others are verified migration work;
some remain scientific research questions.

---

# 18. Design invariants for issue #528

1. **Scientific identity is not a PyMC name.** Sources, components, states,
   observation models, and outputs have stable IDs independent of suffixes.
2. **The model can be read without compiling it.** A scientist can inspect
   equations, priors, covariance components, and outputs first.
3. **Reduction is atomic.** Induced prior, effective forward model,
   unresolved covariance, and reconstruction metadata share provenance.
4. **Fixing is not marginalization.** Structural padding, deterministic fixing,
   retained state, and marginalized state are different dispositions.
5. **Covariance meaning is separate from representation.** Aggregation,
   temporal, transport, and measurement components may each be dense, blocked,
   or LRPD.
6. **Approximations are visible.** Exact Gaussian reduction, lognormal moment
   closure, fallback calibration, and LRPD truncation are recorded separately.
7. **Outputs are part of the model contract.** Every requested functional says
   whether it factors through retained state or needs unresolved terms.
8. **Backends realize the same bound model.** Analytic Gaussian and PyMC results
   provide independent parity checks.
9. **The concrete simple model remains legible.** It is the worked scientific
   reference, not architectural debris that must be removed immediately.

---

# 19. Recommended implementation sequence

## Immediate: stabilize meaning and protect current work

1. Accept the revised #528 ADR and vocabulary.
2. Publish model cards for the one-sector, correlated multisector, inner/outer,
   and linked CO2/O2 pressure tests.
3. Schedule the focused correctness fixes already identified around covariance
   alignment, custom-builder preflight, output suppression, and LRPD diagonals.
4. Add one small dense Gaussian golden test for prior-predictive identity,
   posterior projection, centring, and output functionals.

## First production slice

1. Introduce a backend-neutral coherent-reduction artifact.
2. Port the matrix-free native covariance products needed to construct it.
3. Realize the Gaussian subset analytically and through PyMC, with parity
   tests.
4. Preserve the concrete one-sector implementation as the readable oracle.

## Next scientific slice

1. Route the #565 gathered correlated state through stable semantic identities.
2. Add multivariate lognormal moment matching, whitening, and explicit fallback
   metadata.
3. Generalize aggregation error into labelled additive covariance components.
4. Add temporal OU as a first non-aggregation covariance component.
5. Replace suffix-derived output inference with quantities of interest and a
   compilation/output manifest.

## Pressure-test before publishing an extension API

1. inner/outer states with different supports;
2. shared states contributing to more than one forward term;
3. linked CO2 and O2 observation models with unequal observation axes;
4. cross-source covariance and gathered state selectors; and
5. reportable and non-reportable native functionals.

## Later research boundary

Use uncertain-operator/DUBFI work to test state-dependent covariance,
determinants, ensemble rank, bias limitations, and dynamic LRPD. This should
extend the covariance-component relation, not force transport-ensemble details
into the semantic core.

---

# 20. The intended outcome

A scientist should be able to answer, before running an inversion:

- What is random at native resolution?
- What regional and cross-source uncertainty does that imply?
- Which state has been retained, fixed, or marginalized?
- Which forward-model terms map each state to each observation-model mean?
- Which uncertainty components enter the likelihood, and why?
- Which operations are exact, moment closures, or numerical approximations?
- Which reported quantities are exact functions of the retained posterior?
- Which numerical representation and backend realize the model?

If these questions can be answered from one generated model card, the semantic
model is doing useful scientific work. If they require reading runner branches,
suffix conventions, and PyMC graph construction, it is not yet complete.

---

# Evidence and related design material

Within this repository:

- [issues 402/403 builder-strategy design](../issues_402_403_builder_strategy_design.md)
  — earlier relational design and model-builder pressure tests;
- [state-vector grouping](../state_vector_grouping.md) — labelled state grouping;
- [concrete RHIME model](../../usage/concrete_rhime_model.rst) — current concrete
  reference model;
- [issue #528](https://github.com/openghg/openghg_inversions/issues/528) —
  backend-neutral semantic-model design; and
- [issue #565](https://github.com/openghg/openghg_inversions/issues/565) and
  [merged PR #571](https://github.com/openghg/openghg_inversions/pull/571) —
  correlated gathered LogNormal state and its merged arithmetic-moment,
  whitening, label, and serialization foundation.

Verification-games evidence:

- `src/verification_games/coherent_reduction.py` and
  `tests/test_coherent_reduction.py` on `origin/main`;
- `src/verification_games/grid_covariance.py`;
- `src/verification_games/scale_priors.py`;
- `src/verification_games/covariance_approximation.py`;
- the partition-robust aggregation-error reports and worklogs on
  `origin/codex/partition-robust-aggerr-e3-country-aligned`; and
- the WUR fixed-OU diagnosis and known-truth flux gate on
  `origin/codex/wur-base-temporal-ou-cv`.

Mathematical background in `inversions-knowledge`:

- `docs/derivations/posterior-projection-and-exact-marginalization.md`;
- `docs/topics/aggregation-error-basis-functions-and-reported-functionals.md`;
- `docs/topics/time-series-residual-autocorrelation-and-whitening.md`;
- `docs/derivations/autoregressive-and-ornstein-uhlenbeck-correlation.md`;
- `docs/topics/ensemble-transport-uncertainty-and-dubfi.md`; and
- `docs/derivations/uncertain-affine-operator-marginalization.md`.
