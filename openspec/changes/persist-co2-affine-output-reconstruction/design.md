# Design

> Status: Final. Approved by the specification owner on 2026-09-22. This document must not be modified without that owner's explicit consent.

## Context

See [proposal.md](proposal.md) for motivation and [the capability specification](specs/co2-output-reconstruction/spec.md) for required behavior.

`FluxWeightedBasis` pairs a retained-state-to-grid `BasisOperator` \(U\) with flux \(F\). Standard and multisector postprocessing keeps those ingredients separate, reconstructs gridded scaling or flux when requested, and lets `Countries` contract country membership, cell area, flux, and \(U\) before applying posterior draws. It does not persist a collection of country-specific or other output-specific state maps.

Coherent preparation needs the same architectural object with centring and a potentially non-bucket prolongation. For native scaling \(x\sim N(m,B)\), retained state \(\alpha=\Pi x\), and

\[
C_\alpha=\Pi B\Pi^T,\qquad
U^*=B\Pi^T C_\alpha^{-1},\qquad
\alpha_{ref}=\Pi m,
\]

the retained-state conditional native mean is

\[
\bar x(\alpha)=m+U^*(\alpha-\alpha_{ref}).
\]

With signed reference flux \(F\), the corresponding native-grid flux is

\[
\bar f(\alpha)=F\bar x(\alpha)=E[f\mid\alpha].
\]

`NativeCovarianceProducts` has \(U^*\) while the native mean and flux are still available, but `CoherentGaussianReduction` deliberately retains only the observation-side likelihood calculation. Output reconstruction therefore needs a separately bound lifetime.

## Goals / Non-Goals

**Goals:**

- Define one named affine native-scaling and flux reconstruction value analogous to `FluxWeightedBasis`.
- Persist \(m\), \(F\), and \(U^*\) as separate strategy-neutral ingredients.
- Support bucket-preserving and supplied-restriction preparation through one labelled application contract.
- Bind reconstruction to the exact `Co2PreparedInputs` artifact that owns the retained reference state.
- Preserve factorization and enable aggregate consumers to contract native axes before posterior sample axes are introduced.
- Make retained-state conditional scope explicit and testable.

**Non-Goals:**

- A generic affine quantity-of-interest value in this change.
- Persisted country, sector, regional, or other derived quantity maps.
- Complete functional posterior conditioning, residual blocks, country statistics, or residual sampling.
- Basic/PARIS adapters, staged manifest propagation, source-to-sector reporting transforms, or output formatting.
- Supplied-\(\Pi\) construction or generic coherent-product persistence.
- A generic output container on `CoherentGaussianReduction`.
- A registry, plugin system, arbitrary operator protocol, or class hierarchy.

## Decisions

### 1. Add one affine flux value analogous to `FluxWeightedBasis`

Add a small frozen public value, `AffineFluxMap`. It is shorter than `AffineFluxReconstruction` while retaining the important fact that the operation is affine.

The value owns:

- `native_mean` for \(m\);
- `flux` for signed reference flux \(F\);
- `prolongation` for \(U^*\);
- labelled native/state dimensions, units, projection provenance, uncertainty scope, and prepared-input identity.

The authoritative `reference_state` \(\alpha_{ref}\) comes from the bound `Co2PreparedInputs.alpha_prior_mean`; an imported duplicate is equality-validated and discarded.

The public operations are:

- `state_to_native(alpha, reference_state=...)`, returning \(m+U^*(\alpha-\alpha_{ref})\);
- `state_to_flux(alpha, reference_state=...)`, returning \(F[m+U^*(\alpha-\alpha_{ref})]\).

The term **sensitivity** is not used for \(U^*\), \(FU^*\), or a derived output action. In this repository sensitivity denotes observation-side \(H\)-like matrices. **Design matrix** is also avoided. The exact project term for \(U^*\) is **prolongation**; downstream linear actions use directional names such as `state_to_flux` and `state_to_country`.

### 2. Persist ingredients, not precomposed maps

The durable value stores \(m\), \(F\), and \(U^*\), not

- a precomputed `state_to_flux` array \(FU^*\);
- a mapping of named affine quantities;
- country-by-state or sector-by-state products;
- generic `reference_output`/linear-action pairs.

Keeping \(F\) separate is not merely stylistic. Flux may carry time and source dimensions that \(U^*\) does not. Persisting \(FU^*\) can duplicate \(U^*\) across those dimensions and lose the useful factorization needed by bounded-memory consumers.

The two initial prolongation representations are closed and explicit:

1. **Bucket representation:** the existing bucket `BasisOperator`, used when \(U^*=U_{bucket}\).
2. **Explicit representation:** a labelled, chunked native-by-retained array containing exact \(U^*\) for supplied restrictions.

The explicit representation may be intrinsically dense. This change does not disguise it as sparse or introduce a speculative generic linear-operator protocol.

### 3. Keep the value separate and identity-bound

`AffineFluxMap` is an output-reconstruction artifact, not a PyMC input. It remains outside `Co2PreparedInputs.inv_inputs` and outside `CoherentGaussianReduction` while binding to one exact prepared-input identity.

The versioned DataTree stores native mean, flux, the tagged prolongation representation, labels, units, intrinsic source coordinates, projection provenance, reconstruction scope, and identity metadata. Reporting-sector mappings are not intrinsic reconstruction data and remain output-workflow responsibility.

The payload never contains raw `fp_x_flux`, \(\Pi\), native \(B\), dense native-by-native covariance, named derived-quantity maps, or quantity-specific residual covariance blocks. Related arrays are materialized together only at a named serialization boundary; serialization does not authorize flattening \(F\) and \(U^*\).

### 4. Make output-specific composition a consumer responsibility

For a country aggregation operation \(A\), containing membership, cell area, physical conversion, and any time or sector selection,

\[
q_{ref}=AFm,\qquad
R_q=AFU^*,
\]

and

\[
\bar q(\alpha)=q_{ref}+R_q(\alpha-\alpha_{ref}).
\]

The required evaluation order is:

1. contract \(A\), \(F\), and \(U^*\) over native dimensions;
2. obtain the compact country-by-state action \(R_q\);
3. apply it to chain/draw samples.

Consumers must not reconstruct native-grid values for every draw solely to aggregate them. This mirrors the existing `Countries`/`make_x_to_country_matrix` boundary. A coherent-aware country adapter may construct and return a transient general affine quantity map, but OPE-169 neither defines that generic value nor persists its country-specific result.

When gridded output is actually requested, `state_to_native` or `state_to_flux` necessarily returns native-grid-by-sample values. That cost is explicit at the product boundary rather than incurred while constructing aggregate outputs.

### 5. Keep unresolved functional data out of the base reconstruction value

The affine value provides \(E[x\mid\alpha]\) and \(E[f\mid\alpha]\). It does not provide complete observation-conditioned native-grid uncertainty.

For a declared functional \(q=Lx\), complete moments additionally require

\[
C_{qq}=LB_\perp L^T,\qquad
C_{qy}=LB_\perp H^T,
\]

where \(B_\perp=B-U^*C_\alpha U^{*T}\). These blocks cannot be manufactured later from \(m\), \(F\), and \(U^*\). Their construction, persistence, conditioning, and scientific interpretation belong to OPE-68 or a dedicated quantity-of-interest follow-up, not to `AffineFluxMap`.

### 6. Deliver two stacked PRs

**PR 1 — affine native-flux value**

- Add `AffineFluxMap`, its closed prolongation representations, and `state_to_native`/`state_to_flux`.
- Validate labels, dimensions, units, arbitrary non-unit reference states, and borrowed-array ownership.
- Test bucket/explicit parity and a direct dense oracle.
- Do not include CO2 binding, persistence, countries, likelihood, or generic quantity maps.

**PR 2 — coherent-CO2 persistence and binding**

- Add versioned DataTree, NetCDF/Zarr round trips, prepared-input identity binding, and provenance.
- Add the bucket-preserving producer and an exact supplied-restriction import path for \(m\), \(F\), and \(U^*\).
- Test corruption/mismatch failures, representative artifact size and peak memory, and a consumer-side aggregate contraction without persisting the aggregate map.
- Do not add complete conditioning, output adapters, source-to-sector reporting policy, or staged manifest routing.

### 7. Keep the plan owner-controlled

All four artifacts are `Final` following owner approval on 2026-09-22. Later conflicts require owner consent before these artifacts are edited.

## Deferred Research and Follow-up Ownership

This section is non-normative for OPE-169. It preserves useful research without expanding the implementation scope.

### Generic affine quantities of interest

A reusable generic value remains desirable:

\[
\bar q(\alpha)=q_{ref}+R_q(\alpha-\alpha_{ref}).
\]

A future `AffineQuantityMap` could contain a labelled reference quantity, a `state_to_quantity` action, a reference state, units, provenance, and uncertainty scope. Here *quantity* is a type-level placeholder: concrete consumers should use scientific directional names such as `state_to_country`, `state_to_region`, or `state_to_flux` rather than exposing a literal `state_to_quantity` field everywhere.

This abstraction would let `Countries` own construction of the geographic functional without also owning generic affine application, sample handling, or statistics. It should be designed under OPE-24 or a new child issue after at least the country and flux use cases are concrete. It must not be introduced merely as an alias around one dense matrix, and it must keep affine conditional means separate from unresolved covariance data.

### OPE-68: complete declared-quantity moments

For posterior draw \(d\), complete conditioning under represented likelihood covariance \(S_d\) is

\[
\mu_{q,d}=q_{ref}+R_q(\alpha_d-\alpha_{ref})
             +C_{qy}S_d^{-1}(y-\mu_{y,d}),
\]

\[
\Omega_{q,d}=C_{qq}-C_{qy}S_d^{-1}C_{yq}.
\]

The complete covariance is

\[
\operatorname{Cov}_d(\mu_{q,d})+E_d[\Omega_{q,d}].
\]

Adding a static \(C_{qq}\) to covariance across affine draws is wrong when \(C_{qy}\ne0\), because both conditional means and conditional covariance change. Fixed-tau OU with independently inferred additive amplitude per site must use each draw's represented likelihood covariance, preferably through the same `FixedOuLowRank.solve` operation used by inference. Matrix right-hand sides should combine the observation residual and required \(C_{yq}\) columns so a draw does not repeat the same factorization.

Construct \(R_q\), \(C_{qq}\), and \(C_{qy}\) in bounded quantity batches while \(B\), \(U^*\), and the declared functional are available. Never construct native-by-native covariance or native-grid-by-draw values solely for an aggregate. Do not flatten country-by-time into a giant all-times covariance when only within-time blocks are required.

With a low-rank observation approximation, the functional-observation cross block must be projected into the represented observation modes and the represented joint block checked for positive semidefiniteness. Exact and approximated covariance blocks must not be mixed inconsistently.

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
- country weights use membership, cell area, signed reference flux, and declared physical conversion;
- \(\Pi\) uses absolute reference-flux times cell-area weights normalized within source/region;
- conversion records a 365-day year and the declared CO2 molar mass.

The ordinary WUR basis is expected to be the general, non-retained-exact case. A complete declared country quantity needs its affine mean map and residual blocks produced while native covariance information remains available; this is distinct from OPE-169's reusable grid reconstruction value.

### OPE-164: staged workflow and output routing

OPE-164 should carry prepared/reconstruction identities through manifests, bind `AffineFluxMap` at output time, expose it through the CO2 result/output bundle, and route supported grid reconstruction to common outputs. A coherent-aware country adapter should consume the bound value and country policy on demand. OPE-164 should apply labelled source-to-sector reporting transforms after reconstruction and validate a complete request before opening an output destination. Complete country moments remain dependent on OPE-68.

### OPE-24 or a new child issue: generic quantity maps

OPE-24 already owns backend-neutral scientific roles and covariance-aware quantities of interest. It is the natural parent for a focused follow-up defining `AffineQuantityMap`, concrete functional composition, uncertainty-scope vocabulary, and adoption by standard, multisector, coherent-PyMC, and analytic results. A separate child issue is preferable if that work would make OPE-24 too broad.

### OPE-31: supplied-restriction construction

OPE-31 owns construction and validation of supplied \(\Pi\), derivation of exact labelled \(U^*\), bounded-memory native products, and fresh-preparation parity. Its handoff to OPE-169 is \(U^*\) itself, not a projection-strategy name or precomputed \(FU^*\).

### OPE-40: generic persistence

OPE-40 owns generic persistence of coherent-reduction and covariance products. OPE-169 should reuse compatible coordinate, schema-version, identity, and prolongation-representation conventions without waiting for or duplicating the generic native-product artifact.

## Rejected Ideas

- **Persist a mapping of named affine outputs:** it conflates reusable native reconstruction with product-specific postprocessing.
- **Persist \(FU^*\):** it duplicates the prolongation across flux time/source dimensions and discards useful factorization.
- **Persist country maps in OPE-169:** country membership, area, physical conversion, and output selection belong to country postprocessing.
- **Put reconstruction arrays in `Co2PreparedInputs`:** model inputs and output replay data have different ownership and consumers.
- **Put a generic native/output map on `CoherentGaussianReduction`:** the reduction owns the observation-side likelihood calculation, not scientific output replay.
- **Make explicit \(U^*\) a `BasisOperator` without generalizing that contract:** `BasisOperator` currently promises bucket basis geometry; a closed explicit representation is more honest.
- **Store only a projection-strategy name:** it cannot recover exact supplied-restriction \(U^*\).
- **Call the operation a sensitivity or design matrix:** sensitivity means observation-side \(H\) in this project, while design matrix is unfamiliar domain language.
- **Assume the scaling-state mean is one:** coherent reduction supports arbitrary native and retained means.
- **Call the retained lift a complete native posterior:** it omits unresolved observation-residual updates and residual uncertainty.

## Risks / Trade-offs

- **A supplied \(U^*\) may be intrinsically dense and expensive.** Preserve bucket form where exact; chunk the explicit form by native dimensions; record serialized size and peak load/apply/aggregate memory; add no hidden densification.
- **A reconstruction artifact can be paired with stale prepared inputs.** Bind by content identity and validate labelled state coordinates, units, convention, and projection provenance before application.
- **Consumers may accidentally reconstruct native values before aggregation.** Provide an explicit native-contraction boundary and test that representative country aggregation forms country-by-state data before chain/draw dimensions.
- **Deferring generic quantity maps could lead to duplicate adapters.** Preserve the research above and propose a focused OPE-24 child rather than adding speculative genericity to OPE-169.

## PR Evaluation Plan

PR 1 is ready only with focused tests for arbitrary non-unit reference states, exact label/unit validation, bucket/explicit parity, `state_to_native`, `state_to_flux`, preservation of all sample dimensions, no input mutation or hidden eager compute/densification, and an independent dense oracle.

PR 2 is ready only with:

- NetCDF and Zarr round trips for \(m\), \(F\), both prolongation representations, intrinsic source labels, scope, provenance, and identity;
- prepared-artifact identity and coordinate/unit mismatch tests;
- bucket-preserving parity against direct `BasisFunctions` native-scaling and flux reconstruction;
- a supplied-restriction/Verification Games fixture loaded through the same public API without assuming \(U^*=U_{bucket}\);
- retained-state conditional grid parity, not complete country-moment or output-format parity;
- a representative country contraction demonstrating aggregate-before-samples ordering without persisting its derived map;
- assertions that raw `fp_x_flux`, \(\Pi\), native \(B\), native-by-native covariance, precomposed \(FU^*\), named derived maps, and residual blocks are absent;
- representative serialized-size and peak load/apply/aggregate memory evidence;
- focused pytest and Ruff, `git diff --check`, appropriate registered regression coverage, and repository compatibility/full-suite/type jobs submitted through `scripts/slurm_tox.sh`;
- a Towncrier feature fragment and documentation of equations, terminology, binding, representations, scope, and deferred limitations.
