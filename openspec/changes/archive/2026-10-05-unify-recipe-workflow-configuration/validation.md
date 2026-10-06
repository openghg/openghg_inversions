# OPE-207 implementation validation

Implementation branch: `codex/ope-207-recipe-workflow`, based on devel after
the approved planning PR #787. Local runtime checks used Python 3.13.7.

## Evidence

| Check | Result |
| --- | --- |
| Final integrated CLI, standard/multisector stages, recipe operations, sampler choices, CO2 contract and configuration suites | 157 passed |
| Recipe operations, existing RHIME, builders and nested suites | 328 passed |
| Array operations, builders, nested execution, prepared serialization, standard/PARIS/CO2 product suites | 138 passed |
| Cached CO2 graph/dense-reference and cached sampler suites | 31 passed |
| CO2 preparation and output scientific suites | 38 passed |
| Installed ordinary/cached CO2 CLI acceptance | 3 passed |
| Changed Python paths Ruff and whitespace | Passed |
| Active change and durable capability strict OpenSpec validation | Passed |
| Full documentation regeneration/build required by repository policy | Passed on SLURM job 19221825 (exit 0); built API object links verified |

The suite counts overlap; they are not a cumulative count. NumPy/Numba and
PyTensor compilation caches were directed to writable `/tmp` paths. Initial
read-only default-cache failures were rerun successfully with those paths.

## Scientific and boundary coverage

- Standard/multisector acquisition, compatible cache reload and filtering
  retain requested-site subsets with unequal per-site choices. All site-option
  fields align; empty sets and malformed labels fail at their owning boundaries.
- Real sensitivity projection and assembly agree with independent expected
  values. Real full/merged/prepared model log probabilities agree with an
  independent Normal-prior/likelihood calculation. Graphs, sampler policy and
  common-posterior products agree across supported routes, including multisector
  diagnostics, scientific roles and separate chain/draw axes.
- A real NetCDF acquisition cache retains pre-filter timestamps; reload bypasses
  acquisition and enters real daily-median filtering and canonical preparation.
  New manifests contain no filtered merged checkpoint or retired entries.
- Full execution works with intermediate checkpoint writers forbidden; prepared
  execution avoids preparation. Borrowed Dask arrays remain lazy and unchanged.
  Existing custom-model/likelihood and nested-runner regressions pass.
- Ordinary/cached CO2 real posterior, predictive and likelihood results agree
  across direct/staged routes. Product comparisons use the same saved samples;
  unseeded ordinary prior draws are compared by schema/finite values rather than
  requiring independent trajectories to match. Cached likelihood and conditional
  prediction retain independent dense-reference evidence.
- Saved-output replay forbids graph construction and sampling. Unsupported or
  retired schema/identity/family envelopes, missing/altered/escaping/swapped
  bindings and optional CO2 affine mismatches fail before posterior loading or
  product writes. Sampling provenance comes from the sample manifest.
- Configuration snapshots isolate nested maps, lists, output/prior choices and
  NumPy sampler initial-value arrays. Public configuration constructors retain
  their previous runtime-sampler calling convention. Scientific array handoffs
  remain borrowed rather than being copied by configuration access.
- Readiness distinguishes returned empty/non-finite evidence from execution,
  authentication and serialization errors. Existing family exception boundaries,
  predictive/report writes, convergence metrics and strict CLI exits remain.

## Documentation and delivery

Updated staged usage, CO2 usage, scientific-operation ownership and API landing
documentation. Reviewed changed public operation/configuration docstrings against
source, annotations, public exports and tests, preserving the existing style.
Added a uniquely named Towncrier removal fragment for the next-minor staged
metadata reset, filtered checkpoint removal and retained-site correction.

The durable `recipe-workflow-contract` capability is synced and the completed
change is archived. All 10 implementation/delivery tasks are complete. OPE-207
remains open for pull-request review and merge; linked CO2/O2 staged integration
remains separate in OPE-165.

## Review corrections (6 October 2026)

Preserved nested list/tuple types when resolving and recreating sampler choices,
including fresh invocation containers and unchanged NumPy isolation. The shared
sampler rejects inference-data coordinate/dimension overrides before PyMC while
forwarding permitted conversion options. Updated the consumer's explicit builder
forwarding assertion, normalized preparation-option docstrings, historical
diagnosis guidance, and the release fragment. Removed redundant materialization
deduplication, the unused CO2 output default factory, and committed EOF whitespace.

The combined workflow, scientific-operation, sampler, CO2 configuration/contract,
consumer, existing RHIME, and cached-sigma sampling suites passed: **475 tests**
on Python 3.13.7. Regression coverage includes direct/resolved rejection before
sampling, nested container round trips and isolation, and successful independent
diagnosis for sample schemas 1/2/3 across standard, multisector and CO2 families.
Changed-path Ruff, `git diff devel --check`, and strict OpenSpec validation passed.
The fresh `docs-full` regeneration/build passed on SLURM job 19228601 (exit 0).
Rendered staged/CO2 compatibility guidance, sampler restrictions, and both
preparation parameter descriptions were verified in the generated HTML.
