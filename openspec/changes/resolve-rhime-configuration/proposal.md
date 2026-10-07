# Proposal

## Why

Standard and multisector RHIME runners pass partly raw options into acquisition,
which builds an aligned site record, unpacks it, and repeats expansion in the
lower-level loader. [Issue #804](https://github.com/openghg/openghg_inversions/issues/804)
needs one concrete configuration boundary before acquisition, so internal code
can consume named values without parsing convenient external syntax again.

## What Changes

- Separate raw INI loading from a format-neutral semantic resolver shared by
  file and Python adapters; retain existing INI vocabulary and override rules.
- Reuse the existing run/model/output specifications and site-options record;
  introduce only the missing typed preparation and sampling values.
- Normalize requested sites and expand site selectors once before acquisition.
  Acquisition and preparation subsequently select retained options together.
- Preserve public shorthand through adapters, caller-owned inputs, existing
  direct sampler APIs and numerical behavior.
- Add focused configuration/entry-point checks and update affected guidance.

This is planning for review, with no implementation tasks or code changes yet.
Hash and manifest policy remains with [#808](https://github.com/openghg/openghg_inversions/issues/808)
and [PR #802](https://github.com/openghg/openghg_inversions/pull/802): this change
adds no identity protocol, migration layer or historical-hash guarantee.

## Capabilities

### New Capabilities

- `rhime-configuration`: Concrete standard/multisector configuration resolved
  before acquisition, with compatible entry adapters and aligned site options.

### Modified Capabilities

None. This checkout has no synced durable capability specs.

## Impact

The existing `rhime.params` owner, ordinary runners, and acquisition/preparation
consumers change. Build on the landed #773/#774 owners:
`inversion_data.acquisition`, `inversion_data.prepared_inputs` and
`inference.sampling`, preserving their compatibility exports. Public INI/Python
helpers and low-level retrieval entry points remain adapters; independent
builders and prepared-input routes retain their contracts. Preserve #807's
early tracer rejection and consumption of omitted/false options. No new
dependency, file format, cache policy, scientific equation or staged workflow
redesign is included.
