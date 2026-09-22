# Proposal

> Status: Draft. Finalisation requires explicit approval from the specification owner. After finalisation, these planning artifacts must not be modified without that owner's explicit consent.

## Why

Staged coherent-CO2 inversions need a public, durable way to replay exact retained-state output reconstruction after native preparation data have been released. `Co2PreparedInputs` is deliberately the PyMC/model-input artifact, so output-only data do not belong in it. Today the required affine maps are either derivable only for one projection strategy or retained in private Verification Games code.

## What Changes

- Add a small public, labelled affine operation for a reconstructed quantity of interest,

  \[
  q(\alpha)=q_{ref}+S_q(\alpha-\alpha_{ref}),
  \]

  using the public terms `reference_output`, `sensitivity`, and `reference_state`. Here `sensitivity` is an exact labelled linear action, not necessarily a materialized dense matrix.
- Add a separate, versioned coherent-CO2 reconstruction companion bound to one exact `Co2PreparedInputs` artifact.
- Persist named affine operations in explicit or factorized form with their labels, units, uncertainty scope, projection provenance, and source/sector metadata. The companion may also carry labelled unresolved functional covariance blocks supplied by a producer for later use by OPE-68, but this change does not condition them or calculate complete posterior moments.
- Support bucket-preserving maps produced by OGI and exact supplied-restriction maps produced elsewhere through the same strategy-neutral replay contract.
- Support left composition with an output functional before applying posterior samples, so consumers can derive compact country or other aggregate maps without constructing native-grid-by-sample arrays.
- Label gridded affine reconstruction as `retained_state_conditional`; it is not complete observation-conditioned native-grid inference.
- Deliver the change as two reviewable PRs: the reusable affine value first, then the bound CO2 companion, persistence, producers/import, and parity evaluation.

Complete functional conditioning belongs to OPE-68, staged output routing to OPE-164, backend-neutral adoption by other model families to OPE-24, supplied-\(\Pi\) construction to OPE-31, and generic coherent-artifact persistence to OPE-40.

## Capabilities

### New Capabilities

- `co2-output-reconstruction`: A labelled affine quantity operation and an identity-bound, durable coherent-CO2 reconstruction payload.

### Modified Capabilities

None.

## Impact

- Adds a public mathematical value similar in role to `BasisFunctions`, but representing an affine state-to-quantity operation rather than basis geometry; its sensitivity representation may retain useful factorization.
- Adds a CO2-specific companion without changing the `Co2PreparedInputs.inv_inputs` model-input schema.
- Leaves `CoherentGaussianReduction` as the likelihood/reduction boundary rather than making it an output container.
- Adds versioned NetCDF/Zarr round trips, binding validation, dense affine oracles for small cases, factorized-operation checks, and public Verification Games parity fixtures.
- Retains useful complete-functional research as a non-normative handoff rather than expanding this implementation into likelihood or postprocessing work.
