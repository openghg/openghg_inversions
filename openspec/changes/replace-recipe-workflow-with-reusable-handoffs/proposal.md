# Proposal

## Why

OPE-207 and its unmerged implementation in [PR #798](https://github.com/openghg/openghg_inversions/pull/798)
exposed useful scientific boundaries but left callers reconciling a full
configuration, numerical paths and manifests, while preparation remained bound
to later model choices. Replace that proposal with reusable prepared results and
one saved handoff per operation, so the same inputs can support compatible priors
and likelihoods and saved samples can produce outputs without the original config.

## What Changes

- Make the public sequence `prepare(preparation) -> prepared manifest`,
  `sample(prepared manifest, model, sampler options) -> sample manifest`, and
  `postprocess(sample manifest, output policy) -> result`. Concrete families own
  their scientific functions; full runners use the same science in memory.
- Separate preparation facts, model/inference choices and product requests.
  Authenticate prepared content, then check the selected model's scientific
  requirements rather than matching the original full configuration.
- Store bounded, versioned replay information and the existing scientific output
  contract with the sample record. Use manifest-relative dependencies and explicit
  reads, without a handoff object, workflow engine or implicit copies of large data.
- Preserve the lessons from review: retained-site consistency, immutable sampler
  choices with faithful container round trips, registered model coordinates,
  borrowed/lazy inputs, matched cached-CO2 inference and authenticated graph-free
  products. Prior-dependent CO2 reductions remain one coherent preparation.
- **BREAKING:** Replace unused staged/setup Python APIs and owned CO2 configuration
  conventions instead of keeping aliases. Reset scientific-stage envelopes and
  identities; remove the filtered merged checkpoint and separate output-binding
  sidecar. Preserve numerical codecs, product schemas, raw scientific customization
  and the optional pre-filter acquisition cache.
- Keep familiar command names and configuration-file vocabulary; simplify staged
  handoffs and make phase ownership explicit. Independent diagnosis retains its
  separately declared historical support. No new inference algorithms or families.

## Capabilities

### New Capabilities

- `recipe-execution`: Shared concrete scientific execution, phase-owned choices,
  sampler/array ownership, checks and scientific parity.
- `prepared-recipe-handoff`: Portable saved preparation, production facts,
  compatible reuse and common file-reference/publication rules.
- `sampled-recipe-handoff`: Saved inference, recorded scientific output meaning
  and graph-free products from one sample handoff plus current output choices.

### Modified Capabilities

None. On the declared base `9b8cac81`, no durable workflow capability exists.
These three bounded capabilities replace the proposed `recipe-workflow-contract`
inside [the superseded change](../unify-recipe-workflow-configuration/proposal.md);
they do not supplement or sync that obsolete proposal.

## Impact

This PR contains planning artifacts only. Read [design.md](design.md) for the API,
two small diagrams, pseudocode, records and compatibility decisions; read each
capability for its behavioral acceptance; [tasks.md](tasks.md) sequences future
implementation and its review gates. All implementation tasks remain unchecked.

Future changes affect RHIME preparation/construction, configuration, stages,
artifact readers/writers, CLI, tests and user/developer documentation. No runtime
dependency is proposed. The staged reset requires a next-minor-release notice and
Towncrier fragments with implementation, not a claim that this planning PR ships it.
Keep PR #798 as source/validation history; its code and test results do not establish
that the replacement contract is implemented. Nested, linked CO2/O2 stages and MAP
remain separate work with explicit extension examples in the design.
