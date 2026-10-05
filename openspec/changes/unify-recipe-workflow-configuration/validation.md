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
| Full documentation regeneration/build required by repository policy | Pending SLURM validation |

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

The durable `recipe-workflow-contract` capability is synced. Archive and PR
delivery await the full documentation build. OPE-207 remains open for review and
merge; linked CO2/O2 staged integration remains separate in OPE-165.
