# Design

> Status: Draft. Finalisation requires explicit approval from the specification owner. After finalisation, this document must not be modified without that owner's explicit consent.

## Context

See [proposal.md](proposal.md) for motivation and [the capability specification](specs/co2-output-reconstruction/spec.md) for required behavior.

`Co2PreparedInputs` serializes the canonical RHIME/PyMC inputs and is the model-input/replay boundary. Coherent native preparation also has enough information to form exact output maps, but some of that information is discarded after reduction and cannot be recovered from the observation-side reduced model. Output-only maps therefore need a separately bound lifetime.

`BasisOperator` represents basis geometry. Standard and multisector postprocessing already evaluates the special case \(\alpha\mapsto FU\alpha\) from the factored `BasisFunctions` data rather than flattening it into one dense matrix. The proposed value represents the same mathematical level of abstraction for a centred affine operation that can be created during preparation, persisted, loaded by staged output code, composed with later output functionals, and consumed externally. It is similar in architectural role to `BasisFunctions`, not a subtype of it or a companion field on it.

`CoherentGaussianReduction` is the likelihood/reduction boundary. Its existing observation mean and effective observation operator already determine the affine observation-side calculation needed by the model, but that special calculation cannot recover a native or scientific output operation after native preparation data are released. Output reconstruction therefore has a separately bound lifetime rather than a new generic field on the reduction.

## Goals / Non-Goals

**Goals:**

- Define one small labelled affine state-to-quantity value whose sensitivity is an exact linear action and need not be a dense matrix.
- Persist exact coherent-CO2 affine replay data outside the model inputs.
- Accept exact maps from bucket-preserving preparation and supplied-restriction producers through one public contract.
- Compose output functionals before posterior samples so aggregate outputs do not require native-grid-by-sample intermediates.
- Preserve a typed extension boundary for unresolved functional blocks without implementing their scientific consumer.
- Make retained-state conditional grid scope explicit and testable.
- Split delivery into two independently reviewable PRs.

**Non-Goals:**

- Complete functional posterior conditioning, country statistics, or residual sampling.
- Basic/PARIS adapters, staged manifest propagation, or output formatting.
- Supplied-\(\Pi\) construction.
- Standard or multisector producers for the affine value.
- A generic output container on `CoherentGaussianReduction`.
- Generic coherent-artifact persistence.
- Complete observation-conditioned native-grid uncertainty.
- A reconstruction registry, plugin system, or class hierarchy.

## Immediate Design Decisions

### 1. Represent a reconstructed quantity of interest with a centred affine value

Add a small frozen public value, provisionally `AffineStateMap`, representing

\[
q(\alpha)=q_{ref}+S_q(\alpha-\alpha_{ref}).
\]

Here \(q\) is the reconstructed quantity of interest. It may be a gridded flux or another declared quantity; it is not necessarily a contribution to modelled concentration. Public names are `reference_output`, `sensitivity`, and `reference_state`. `reference_output` is the value at the reference state, not an intercept at zero. \(S_q\) denotes an exact labelled linear action; it is mathematical notation, not a requirement to allocate or persist one dense output-by-state matrix.

The operation preserves exact labelled alignment, compatible units, and non-state dimensions. Its sensitivity has a declared representation sufficient to apply and left-compose the linear action. The initial representations are an explicit labelled array and a `flux_times_basis` factorization \(F\circ U\), in which the bucket `BasisOperator` maps retained state to native scaling before multiplication by signed reference flux. This is a closed, versioned schema rather than an arbitrary callable, registry, or plugin interface. Neither application nor composition mutates inputs or hides eager computation, densification, persistence, or rechunking. The reference state is supplied explicitly when the operation is applied.

In the current coherent-CO2 model, \(\alpha\) is a dimensionless scaling state but its prior mean is not generally one. With \(\alpha_{ref}=\mu_\alpha\), the correct expression is

\[
q(\alpha)=q_{ref}+S_q(\alpha-\mu_\alpha).
\]

A separately normalized state \(\widetilde\alpha=\alpha/\mu_\alpha\) would instead introduce the elementwise \(\mu_\alpha\) factor, but that is not the state currently sampled.

For a labelled linear output functional \(L\), left composition produces another affine operation,

\[
(L\circ q)(\alpha)=Lq_{ref}+(L\circ S_q)(\alpha-\alpha_{ref}).
\]

Consumers compose first and apply posterior samples second. A country or other aggregate output therefore has output-by-state size before chain and draw dimensions are introduced; applying the grid operation to every draw and aggregating afterwards is prohibited by the execution contract.

### 2. Keep a CO2-specific companion outside `Co2PreparedInputs`

Add a frozen CO2 reconstruction companion, provisionally `Co2OutputReconstruction`, containing a mapping of named affine quantities plus identities and provenance. It has ordinary validation, DataTree conversion, save, load, and bind functions. Each quantity carries its exact sensitivity representation rather than requiring a flattened matrix. The bound pair obtains authoritative `alpha_prior_mean` from `Co2PreparedInputs`; an imported duplicate is equality-validated and then discarded.

Persisting `reference_output` and an exact sensitivity action is the strategy-neutral replay contract. For bucket-preserving preparation the contract preserves the useful \(F,U\) factorization instead of duplicating it as dense \(FU\). For supplied restrictions the exact covariance-natural action may be irrecoverable after native preparation data are released, so this change accepts an explicit chunked representation. Consumers apply and compose both forms through the same contract and never infer the operation from a projection-strategy name.

### 3. Keep the payload minimal and identity-bound

Each named affine quantity stores its reference output, exact sensitivity representation and factors, dimensions, units, uncertainty scope, and reconstruction/projection provenance. The companion also preserves source provenance and labelled source-to-sector relationships for later consumers, without applying them.

The companion records a semantic identity for its prepared inputs and the minimum compatibility data needed by its contents. Affine-only data require state identity, units, projection provenance, and affine convention. A quantity with residual blocks additionally requires observation identity and likelihood-representation provenance. Stage-manifest propagation is deferred to OPE-164.

The payload never contains raw `fp_x_flux`, \(\Pi\), native \(B\), or dense native-by-native covariance. An explicit sensitivity is chunked by output dimensions. Factorized representations preserve their factors and sparsity where supported. Related arrays are materialized together only at an explicit serialization boundary, and that boundary does not authorize silently flattening a factorized sensitivity.

### 4. Persist an exact, strategy-neutral grid operation

For native scaling \(x\sim N(m,B)\), retained state \(\alpha=\Pi x\), covariance-natural prolongation \(U^*=B\Pi^T C_\alpha^{-1}\), and signed reference-flux operation \(F\), the exact retained-state conditional grid map is

\[
f_{ref}=Fm,\qquad S_f=F\circ U^*,\qquad
\bar f(\alpha)=f_{ref}+S_f(\alpha-\mu_\alpha)=E[f\mid\alpha].
\]

For `preserve_bucket_prolongation`, \(U^*=U_{bucket}\), so OGI persists the existing signed reference flux and bucket operator as a factorized sensitivity and compares it with the direct `BasisFunctions` route. For supplied restrictions, OPE-31 or an external producer supplies an exact explicit sensitivity. The consumer never reconstructs \(U^*\) from a strategy name and never assumes \(U^*=U_{bucket}\).

All grid results produced by this operation carry `retained_state_conditional` scope. They are the conditional mean \(E[f\mid\alpha]\), not complete observation-conditioned native-grid inference.

For a country aggregation operation \(A\), including membership, cell area, physical conversion, and any time or sector selection, the retained-state-conditional total is

\[
\bar q(\alpha)=AFm+(A\circ F\circ U^*)(\alpha-\mu_\alpha).
\]

The postprocessing order is \(q_{ref}=Af_{ref}\), \(S_q=A\circ S_f\), and only then application to retained-state samples. If a consumer merely requests a conditional aggregate and has \(A\), it may derive this compact operation from the reusable grid operation on demand. If a declared quantity must replay without reopening the data that define \(A\), its composed operation and provenance are persisted explicitly.

### 5. Treat unresolved functional blocks as extension data only

The companion may preserve \(C_{qq}\) and \(C_{qy}\) for an already-declared quantity, or an explicit `retained_exact` declaration. These are separate from the affine operation and use typed quantity and observation axes. A grid sensitivity alone can provide conditional aggregate means but cannot manufacture these residual blocks after native covariance data have been released. OPE-169 validates and round-trips producer-supplied blocks but does not construct country functionals, condition on observations, calculate moments, sample residuals, or route outputs.

This boundary preserves the current OPE-169 handoff to complete-functional work without placing likelihood and postprocessing implementations in the same PR series.

### 6. Deliver two stacked PRs

**PR 1 — labelled affine quantity operation**

- Add the public value, centred apply operation, and left-composition operation.
- Add explicit and closed factorized sensitivity representations without an arbitrary callable or class hierarchy.
- Validate labels, dimensions, units, and arbitrary reference states.
- Test sample-dimension preservation, Dask/xarray ownership, explicit/factorized parity, compose-before-samples behavior, and a direct dense oracle.
- Do not include CO2, persistence, likelihood, country, or postprocessing integration.

**PR 2 — coherent-CO2 companion and replay contract**

- Add the named payload, optional residual extension data, versioned explicit/factorized representations, serialization, and prepared-input binding.
- Add the bucket-preserving producer and a public import path for exact externally produced maps.
- Test NetCDF/Zarr round trips, mismatch/corruption failures, prohibited arrays, representative artifact size/memory without implicit densification, and bucket/supplied-restriction parity fixtures.
- Do not add complete conditioning, output adapters, or staged manifest routing.

### 7. Keep the plan owner-controlled

All four artifacts remain `Draft` during review. Explicit owner approval triggers one planning-only edit to record the date and mark all four `Final`, with no substantive change. Later conflicts require owner consent before these artifacts are edited.

## Deferred Research and Follow-up Ownership

This section preserves research that informed the boundary above. It is non-normative for OPE-169 and does not add implementation acceptance criteria to this change.

### OPE-68: complete declared-quantity moments

For a declared quantity \(q=Lx\), define

\[
q_{ref}=Lm,\qquad S_q=LU^*,\qquad
C_{qq}=LB_\perp L^T,\qquad C_{qy}=LB_\perp H^T.
\]

Here \(S_q\) again denotes composition of linear actions and need not be formed as a dense native-grid sensitivity. For a country flux total, \(L\) includes signed reference flux, membership, cell area, physical conversion, and any time or sector selection. Construct \(S_q\) in bounded quantity batches while \(B\) and \(U^*\) are available; never construct native-by-native covariance or native-grid-by-draw output. The residual blocks cannot be derived later from the affine grid operation alone.

For posterior draw \(d\), complete conditioning under the represented likelihood covariance \(S_d\) is

\[
\mu_{q,d}=q_{ref}+S_q(\alpha_d-\mu_\alpha)
             +C_{qy}S_d^{-1}(y-\mu_{y,d}),
\]

\[
\Omega_{q,d}=C_{qq}-C_{qy}S_d^{-1}C_{yq}.
\]

The complete covariance is

\[
\operatorname{Cov}_d(\mu_{q,d})+E_d[\Omega_{q,d}].
\]

Adding a static \(C_{qq}\) to covariance across affine draws is wrong when \(C_{qy}\ne0\), because both conditional means and conditional covariance change. The first important likelihood is fixed-tau OU with independently inferred additive amplitude for each site; its solve must use each draw's amplitudes and the same dense or low-rank-plus-diagonal representation as inference, preferably by reusing `FixedOuLowRank.solve`.

With a low-rank observation approximation, the functional-observation cross block must be projected into the retained observation modes and the represented joint block checked for positive semidefiniteness. Exact and approximated covariance blocks must not be mixed inconsistently.

Analytic mean, covariance, and stdev should remain authoritative. Quantiles, HDIs, medians, or modes for a general quantity require reproducible samples from the draw-conditional residual distribution; all posterior chains must be included.

A quantity is retained-exact only after proving

\[
L=W\Pi
\]

with the exact numerical weights. Basis regions not crossing a country boundary is not sufficient by itself.

For the current Verification Games country convention, the recorded research is:

- source file `/group/chem/acrg/LPDM/countries/country_EUROPE.nc`;
- SHA-256 `48e6ae0618ef1bf7b7721f04a6de7a07b9ba8b6f5db6a3d18e520a2c4779df2a`;
- canonical PARIS alpha-3 selection on the EUROPE grid;
- preserve fractional and overlapping membership exactly, although the current file is categorical and mutually exclusive;
- OGI `areagrid` with spherical radius 6,367,500 m on the exact grid coordinates;
- country functional weights use membership, cell area, signed reference flux, and declared physical conversion;
- \(\Pi\) uses absolute reference-flux times cell-area weights normalized within source/region;
- conversion records a 365-day year and the declared CO2 molar mass.

The ordinary WUR basis is expected to be the general, non-retained-exact case. A conditional country mean may be composed on demand from the grid operation and a supplied country functional. Replay of an already-declared country quantity, especially one carrying OPE-68 residual blocks, should not need the country file; its composed numerical data and provenance must be in the bound artifacts.

### OPE-164: staged workflow and output routing

OPE-164 should carry prepared/reconstruction identities through manifests, bind them at output time, compute statistics across all chains, apply labelled source-to-sector transforms, and route supported basic/PARIS products. It should validate a complete request before opening an output destination. Complete country moments remain dependent on OPE-68.

Grid values from the compact map must remain labelled `retained_state_conditional` in downstream files. Source is OpenGHG data provenance; sector is what is optimised or reported. Several sources may map to one sector, and the labelled post-reconstruction transform must preserve signs and posterior cross-source covariance.

### OPE-24: model-neutral scientific roles

The affine value is model-neutral, and standard/multisector linear maps can later use it as a special case. OPE-169 requires only the coherent-CO2 producer and payload. OPE-24 should own broader adoption and a common uncertainty-scope vocabulary distinguishing retained-exact, retained-state conditional, complete reconstructed, and approximate quantities.

### OPE-31: supplied-restriction construction

OPE-31 owns construction and validation of supplied \(\Pi\), derivation of \(U^*\), bounded-memory native products, and fresh-preparation parity. OPE-169 consumes only the exact resulting affine map through its public import contract.

### OPE-40: generic persistence

OPE-40 owns generic persistence of coherent-reduction and covariance products. The bounded OPE-169 companion should reuse compatible coordinate, schema-version, and identity conventions but should not wait for or duplicate the generic native-product artifact.

## Rejected Ideas

- **Put reconstruction arrays in `Co2PreparedInputs`:** model inputs and output replay data have different ownership and consumers.
- **Make the affine value a `BasisOperator` subtype:** it is affine, may not use bucket geometry, and need not target a grid. It may contain a `BasisOperator` as one factorized sensitivity representation without inheriting from it.
- **Store only a projection-strategy name:** it cannot recover an exact supplied-restriction sensitivity.
- **Always rederive the map from `BasisFunctions`:** that is valid only for bucket-preserving prolongation.
- **Call `reference_output` an intercept:** in the centred form it is the output at `reference_state`, not the value at zero.
- **Assume the scaling-state mean is one:** coherent reduction supports arbitrary native and retained means.
- **Call the retained lift a complete native posterior:** it omits unresolved observation-residual updates and residual uncertainty.
- **Put a generic native/output map on `CoherentGaussianReduction`:** the reduction owns the observation-side likelihood calculation, while scientific output operations have a separate lifetime and may require native information already discarded by the reduction.
- **Treat source and sector as the same state dimension:** they have distinct provenance and scientific meanings.
- **Introduce a general registry or factory:** one public value and one CO2 companion are sufficient for this change.

## Risks / Trade-offs

- **A general supplied \(U^*\) may be intrinsically dense and expensive.** Preserve bucket sensitivities as factors; allow an explicit supplied sensitivity only at a named, chunked boundary; record representative serialized size and peak apply/compose memory; and do not silently densify a factorized operation. If an explicit representation is not affordable, a later reviewed change must add a scientifically defined factorization or require narrower declared operations rather than introducing an arbitrary operator protocol here.
- **A companion can be paired with stale prepared inputs.** Bind by content identity and validate labelled state coordinates, units, convention, and projection provenance before application.
- **Optional residual blocks add schema evolution pressure.** Keep them separate from the affine value, version the companion, and require observation/likelihood identities only when those blocks are present.
- **The reusable affine value could imply unsupported model-family integration.** Documentation will state that only coherent CO2 produces the durable payload in this change.

## PR Evaluation Plan

PR 1 is ready only with focused tests for the centred equation, arbitrary reference states, exact label/unit validation, preservation of all sample/output dimensions, explicit/factorized parity, left composition before sample application, no input mutation or hidden eager compute/densification, and an independent dense oracle.

PR 2 is ready only with:

- NetCDF and Zarr round trips, including MultiIndexes, scope, provenance, and optional residual blocks;
- prepared-artifact identity and coordinate/unit mismatch tests;
- a bucket-preserving factorized operation compared with its direct `BasisFunctions` derivation without flattening \(FU\);
- a supplied-restriction/Verification Games fixture loaded through the same public API without assuming \(U^*=U_{bucket}\);
- retained-state conditional affine parity, not complete country-moment or output-format parity;
- assertions that raw `fp_x_flux`, \(\Pi\), native \(B\), and native-by-native covariance are absent;
- representative serialized-size and peak apply/compose memory evidence, including checks that factorized and sparse inputs are not implicitly densified;
- focused pytest and Ruff, `git diff --check`, appropriate registered regression coverage, and repository compatibility/full-suite/type jobs submitted through `scripts/slurm_tox.sh`;
- a Towncrier feature fragment and documentation of the equation, terminology, binding, scope, and deferred limitations.
