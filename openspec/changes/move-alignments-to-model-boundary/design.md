# Design

## Context

See proposal.md for motivation and the accepted #876 issue amendment. Scientific assembly currently expands BC periods and calculates minimum error before invalid rows are removed. Sigma alignment already belongs to likelihood construction. Standard and multisector share the independent-state recipe; nested shares preparation and requested configuration. CO2 has concrete independent configuration owners and ongoing BC work in #957.

## Goals / Non-Goals

Deliver model-owned choices with one explicit numerical boundary, retained-observation floor semantics, compatible current replay and clear older-format rejection. Do not retain historical pre-drop estimator populations, add baked-value adapters, change error-before-filtering ordering, implement centered BC arithmetic, or redesign output storage.

## Decisions

- Keep raw H_bc and related labelled variables in preparation. Temporal expansion must replace only the model's selected operand, preserving the borrowed prepared dataset and avoiding destructive dimension-wide drops.
- Move bc_freq and its anchor to the model specification; configure omitted anchors from the requested start date. Place floor choices on the likelihoods that consume them. DataConfig retains BC input availability but no temporal/floor policy.
- Jointly materialize selected raw operands, enabled priors and auxiliary coordinates, then construct eager BC/floor values using ordinary functions. Fixed/custom/no-floor likelihoods do not select estimator operands. Concrete builders remain eager.
- Calculate residual/percentile estimates from retained gathered observations. Historical pre-drop numerical parity is intentionally not required. Preserve units, site alignment, scalar/mapping validation and the intended likelihood equations.
- Use exact registered trace constants for fitted operands during replay. Keep raw preparation separate; do not rebuild the graph or repeat scientific calculations when loading samples.
- Introduce new prepared and standard/multisector handoff versions, rejecting incompatible historical records with guidance. Retain independent modern InversionOutput/legacy product codecs.
- Preserve raw BC-related variables for #957. Its future A/M/reference representation will extend the same temporal operation; this change makes no claim of centered support.

## Risks / Trade-offs

- Named floors can change after row dropping: document and test that intended population directly.
- Replacing a BC state axis can align away or delete unrelated variables: use an explicit selected model view or separate derived axis and test extension preservation.
- Removing DataConfig fields affects nested/direct/xarray/saved callers: migrate concrete callers together and run relevant broader coverage.
- Artifact formats have separate owners: version affected envelopes explicitly without silently changing unrelated codec versions.

## Migration Plan

Canonical requests place periods under model and floors under their likelihood. Older incompatible preparations/samples must be reopened with matching software or regenerated. Maintained 0.7 and release readiness remain #944 responsibilities. No release number is selected here.
