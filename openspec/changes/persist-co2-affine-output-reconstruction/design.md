# Design

> Status: Draft. Finalisation requires explicit approval from the specification owner. After finalisation, this document must not be modified without that owner's explicit consent.

## Context

See [proposal.md](proposal.md) for motivation and [the capability specification](specs/co2-output-reconstruction/spec.md) for required behavior.

`Co2PreparedInputs` serializes the canonical RHIME/PyMC inputs and is the model-input/replay boundary. Coherent native preparation also has enough information to form exact output maps, but some of that information is discarded after reduction and cannot be recovered from the observation-side reduced model. Output-only maps therefore need a separately bound lifetime.

`BasisOperator` represents basis geometry. The proposed value instead represents an affine mathematical operation that can be created during preparation, persisted, loaded by staged output code, and consumed externally. It is similar in architectural role to `BasisFunctions`, not a subtype of it or a companion field on it.

## Goals / Non-Goals

**Goals:**

- Define one small labelled affine state-to-quantity value.
- Persist exact coherent-CO2 affine replay data outside the model inputs.
- Accept exact maps from bucket-preserving preparation and supplied-restriction producers through one public contract.
- Preserve a typed extension boundary for unresolved functional blocks without implementing their scientific consumer.
- Make retained-state conditional grid scope explicit and testable.
- Split delivery into two independently reviewable PRs.

**Non-Goals:**

- Complete functional posterior conditioning, country statistics, or residual sampling.
- Basic/PARIS adapters, staged manifest propagation, or output formatting.
- Supplied-\(\Pi\) construction.
- Standard or multisector producers for the affine value.
- Generic coherent-artifact persistence.
- Complete observation-conditioned native-grid uncertainty.
- A reconstruction registry, plugin system, or class hierarchy.

## Immediate Design Decisions

### 1. Represent a reconstructed quantity of interest with a centred affine value

Add a small frozen public value, provisionally `AffineStateMap`, representing

\[
q(\alpha)=q_{ref}+S_q(\alpha-\alpha_{ref}).
\]

Here \(q\) is the reconstructed quantity of interest. It may be a gridded flux or another declared quantity; it is not necessarily a contribution to modelled concentration. Public names are `reference_output`, `sensitivity`, and `reference_state`. `reference_output` is the value at the reference state, not an intercept at zero.

The operation uses ordinary exact xarray alignment and labelled dot products, preserves non-state dimensions, and does not mutate inputs or hide eager computation, densification, persistence, or rechunking. The reference state is supplied explicitly when the operation is applied.

In the current coherent-CO2 model, \(\alpha\) is a dimensionless scaling state but its prior mean is not generally one. With \(\alpha_{ref}=\mu_\alpha\), the correct expression is

\[
q(\alpha)=q_{ref}+S_q(\alpha-\mu_\alpha).
\]

A separately normalized state \(\widetilde\alpha=\alpha/\mu_\alpha\) would instead introduce the elementwise \(\mu_\alpha\) factor, but that is not the state currently sampled.

### 2. Keep a CO2-specific companion outside `Co2PreparedInputs`

Add a frozen CO2 reconstruction companion, provisionally `Co2OutputReconstruction`, containing a mapping of named affine quantities plus identities and provenance. It has ordinary validation, DataTree conversion, save, load, and bind functions. The bound pair obtains authoritative `alpha_prior_mean` from `Co2PreparedInputs`; an imported duplicate is equality-validated and then discarded.

Persisting `reference_output` and `sensitivity` is partly an optimisation for bucket-preserving preparation, but it is also the strategy-neutral replay contract. For supplied restrictions the exact covariance-natural map may be irrecoverable after native preparation data are released. The companion must therefore be usable without knowing how its producer formed the map.

### 3. Keep the payload minimal and identity-bound

Each named affine quantity stores its actual arrays, dimensions, units, uncertainty scope, and reconstruction/projection provenance. The companion also preserves source provenance and labelled source-to-sector relationships for later consumers, without applying them.

The companion records a semantic identity for its prepared inputs and the minimum compatibility data needed by its contents. Affine-only data require state identity, units, projection provenance, and affine convention. A quantity with residual blocks additionally requires observation identity and likelihood-representation provenance. Stage-manifest propagation is deferred to OPE-164.

The payload never contains raw `fp_x_flux`, \(\Pi\), native \(B\), or dense native-by-native covariance. Arrays are chunked by output dimensions, and related arrays are materialized together only at an explicit serialization boundary.

### 4. Persist an exact, strategy-neutral grid map

For native scaling \(x\sim N(m,B)\), retained state \(\alpha=\Pi x\), covariance-natural prolongation \(U^*=B\Pi^T C_\alpha^{-1}\), and signed reference-flux operation \(F\), the exact retained-state conditional grid map is

\[
f_{ref}=Fm,\qquad S_f=FU^*,\qquad
f(\alpha)=f_{ref}+S_f(\alpha-\mu_\alpha)=E[f\mid\alpha].
\]

For `preserve_bucket_prolongation`, \(U^*=U_{bucket}\), so OGI can derive the payload from existing preparation values and compare it with the `BasisFunctions` route. For supplied restrictions, OPE-31 or an external producer supplies the exact resulting map. The consumer never reconstructs \(U^*\) from a strategy name and never assumes \(U^*=U_{bucket}\).

All grid results produced by this operation carry `retained_state_conditional` scope. They are not complete observation-conditioned native-grid inference.

### 5. Treat unresolved functional blocks as extension data only

The companion may preserve \(C_{qq}\) and \(C_{qy}\) for an already-declared quantity, or an explicit `retained_exact` declaration. These are separate from the affine operation and use typed quantity and observation axes. OPE-169 validates and round-trips them but does not construct country functionals, condition on observations, calculate moments, sample residuals, or route outputs.

This boundary preserves the current OPE-169 handoff to complete-functional work without placing likelihood and postprocessing implementations in the same PR series.

### 6. Deliver two stacked PRs

**PR 1 — labelled affine quantity operation**

- Add the public value and centred apply operation.
- Validate labels, dimensions, units, and arbitrary reference states.
- Test sample-dimension preservation, Dask/xarray ownership, and a direct dense oracle.
- Do not include CO2, persistence, likelihood, country, or postprocessing integration.

**PR 2 — coherent-CO2 companion and replay contract**

- Add the named payload, optional residual extension data, schema version, serialization, and prepared-input binding.
- Add the bucket-preserving producer and a public import path for exact externally produced maps.
- Test NetCDF/Zarr round trips, mismatch/corruption failures, prohibited arrays, representative artifact size/memory, and bucket/supplied-restriction parity fixtures.
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

The ordinary WUR basis is expected to be the general, non-retained-exact case. Replay of an already-declared country quantity should not need the country file; its composed numerical data and provenance must be in the bound artifacts.

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
- **Make the affine value a `BasisOperator` subtype:** it is affine, may not use bucket geometry, and need not target a grid.
- **Store only a projection-strategy name:** it cannot recover an exact supplied-restriction sensitivity.
- **Always rederive the map from `BasisFunctions`:** that is valid only for bucket-preserving prolongation.
- **Call `reference_output` an intercept:** in the centred form it is the output at `reference_state`, not the value at zero.
- **Assume the scaling-state mean is one:** coherent reduction supports arbitrary native and retained means.
- **Call the retained lift a complete native posterior:** it omits unresolved observation-residual updates and residual uncertainty.
- **Treat source and sector as the same state dimension:** they have distinct provenance and scientific meanings.
- **Introduce a general registry or factory:** one public value and one CO2 companion are sufficient for this change.

## Risks / Trade-offs

- **Portable grid sensitivities may duplicate bucket-derivable data and be large.** Chunk and compress by output dimensions, record representative artifact size and peak replay memory, and accept the duplication as the cost of a strategy-neutral contract.
- **A companion can be paired with stale prepared inputs.** Bind by content identity and validate labelled state coordinates, units, convention, and projection provenance before application.
- **Optional residual blocks add schema evolution pressure.** Keep them separate from the affine value, version the companion, and require observation/likelihood identities only when those blocks are present.
- **The reusable affine value could imply unsupported model-family integration.** Documentation will state that only coherent CO2 produces the durable payload in this change.

## PR Evaluation Plan

PR 1 is ready only with focused tests for the centred equation, arbitrary reference states, exact label/unit validation, preservation of all sample/output dimensions, no input mutation or hidden eager compute, and an independent dense oracle.

PR 2 is ready only with:

- NetCDF and Zarr round trips, including MultiIndexes, scope, provenance, and optional residual blocks;
- prepared-artifact identity and coordinate/unit mismatch tests;
- a bucket-preserving map compared with its direct `BasisFunctions` derivation;
- a supplied-restriction/Verification Games fixture loaded through the same public API without assuming \(U^*=U_{bucket}\);
- retained-state conditional affine parity, not complete country-moment or output-format parity;
- assertions that raw `fp_x_flux`, \(\Pi\), native \(B\), and native-by-native covariance are absent;
- representative serialized-size and peak-memory evidence;
- focused pytest and Ruff, `git diff --check`, appropriate registered regression coverage, and repository compatibility/full-suite/type jobs submitted through `scripts/slurm_tox.sh`;
- a Towncrier feature fragment and documentation of the equation, terminology, binding, scope, and deferred limitations.
