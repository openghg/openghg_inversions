# Design

## Context

See proposal.md for motivation and the accepted #876 amendments. Before this change, scientific assembly expanded BC periods and calculated minimum error before invalid rows were removed. Revision `751dedd6` implements reusable raw preparation and model-time resolution. The accepted 23 October follow-up localizes floor policy and changes observation-error imputation to use the population retained by selection filters. Standard and multisector share the independent-state recipe; nested shares preparation and requested configuration. CO2 has concrete independent configuration owners and ongoing BC work in #957.

## Goals / Non-Goals

Deliver model-owned choices with one explicit numerical boundary, retained-population observation-error and floor estimates, compatible current replay and clear older-format rejection. Do not retain historical pre-drop estimator populations, add baked-value adapters, invent new missing-component/NaN or resampling uncertainty rules, implement centered BC arithmetic, or redesign output storage. The 23 October amendment supersedes the earlier exclusion of observation-error ordering changes.

## Decisions

- Keep raw H_bc and related labelled variables in preparation. Temporal expansion must replace only the model's selected operand, preserving the borrowed prepared dataset and avoiding destructive dimension-wide drops.
- Move bc_freq and its anchor to the model specification; configure omitted anchors from the requested start date. Place floor choices on the likelihoods that consume them. DataConfig retains BC input availability but no temporal/floor policy.
- Jointly materialize selected raw operands, enabled priors and auxiliary coordinates, then construct eager BC/floor values using ordinary functions. Fixed/custom/no-floor likelihoods do not select estimator operands. Concrete builders remain eager.
- Calculate residual/percentile estimates from retained gathered observations. Historical pre-drop numerical parity is intentionally not required. Preserve units, site alignment, scalar/mapping validation and the intended likelihood equations.
- Keep the floor estimator and its required operands/options together; keep likelihood-specific enablement, preparation and validation with likelihood-input preparation. The model-input coordinator visibly composes those operations with BC resolution. Numerical materialization remains an execution boundary, without a likelihood catalogue.
- Apply observation-selection filters before observation-error derivation/imputation. Nested preparation first selects common outer observations and aligns the inner observations, then prepares each retained domain's errors. Historical retrieval without selection filters keeps its own returned-population boundary; independently supplied final errors are not re-estimated.
- Distinguish resampling, which defines observations and propagates uncertainty, from selection and imputation. The maintainer explicitly confirmed retirement of `daily_median` and `six_hr_mean` on 23 October; remove both public functions and registry names with actionable rejection guidance, tracked with #954/#945. No median-to-mean substitution or new resampling algorithm is part of this design.
- Use exact registered trace constants for fitted operands during replay. Keep raw preparation separate; do not rebuild the graph or repeat scientific calculations when loading samples.
- Introduce new prepared and standard/multisector handoff versions, rejecting incompatible historical records with guidance. Retain independent modern InversionOutput/legacy product codecs.
- Preserve raw BC-related variables for #957. Its future A/M/reference representation will extend the same temporal operation; this change makes no claim of centered support.

## Risks / Trade-offs

- Named floors can change after row dropping: document and test that intended population directly.
- Observation-error fallback values intentionally change when selection rejects influential observations: verify retained-only estimates with rejected-outlier tests while preserving custom errors and existing missing-data rules.
- Replacing a BC state axis can align away or delete unrelated variables: use an explicit selected model view or separate derived axis and test extension preservation.
- Removing DataConfig fields affects nested/direct/xarray/saved callers: migrate concrete callers together and run relevant broader coverage.
- Artifact formats have separate owners: version affected envelopes explicitly without silently changing unrelated codec versions.

## Migration Plan

Canonical requests place periods under model and floors under their likelihood. Older incompatible preparations/samples must be reopened with matching software or regenerated. Maintained 0.7 and release readiness remain #944 responsibilities. No release number is selected here.
