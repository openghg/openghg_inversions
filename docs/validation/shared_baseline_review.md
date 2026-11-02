# Shared baseline: independent API review

Review date: 2 November 2026. Scope: the new prepared shared-baseline primitive,
ordinary and cached CO2 builders, their diagnostics and trace annotation, and
the focused synthetic tests. This review permits the dependent configuration
work to use the API; it does not establish completed scientific replay or
release acceptance.

Reviewed implementation commit: `681d1bc7f865a236b2f19e0e74d2d6ee2df82cbc`.

## Reviewed scientific contract

The implementation retains one coherent transport reduction and its existing
aggregation-error inputs. It keeps the complete fixed affine contribution and
all outer prior means once. The four explicit choices are fixed outer,
additive net-biosphere/fossil corrections, multiplicative outer departures, and
the latter with an already resolved positive biosphere calibration.

Equal-site observation weights and column means are frozen on full support.
The BC operator is period-gated before centring. Subsets select rows of the
prepared product without recomputing means or the scalar concentration
reference. The shared atmospheric delta has response one at every observation;
its physical source is unidentified.

Nonzero covariance among retained inner and sampled outer coefficients is
preserved in multiplicative modes. Removing outer states is rejected when
they have nonzero arithmetic covariance with retained states, including when
only ocean is removed. Fixed and additive modes therefore reject nonzero
inner/outer cross-covariance. These are explicit support limits, rather than
an assumption that all coherent priors have zero cross-covariance.

## Independent checks

The focused checks ran with the existing Python 3.13 BC-prior environment,
the reviewed worktree on `PYTHONPATH`, isolated writable PyTensor/Numba caches,
two Dask workers and one numerical thread. No environment installation or
archived experiment mutation was needed.

- Realistic `source`/`region_in_source` MultiIndex states retain distinct full,
  selected and physical-outer axes through the standard NetCDF codec.
- The ordinary and cached builders have equal absolute predictions, full
  densities and gradients at nonzero coefficients in all four modes. The
  refreshed cached quadratic itself, including state gradients, is compared
  against the ordinary observed likelihood; comparing only the two observed
  graphs would not check the sampler's cached target.
- Complete affine closure holds at the prior mean and at nonzero states.
  Raw outer correction minus its removed mean equals the centred correction.
  Additive mode removes the original sampled outer states. Independent gross
  additive corrections on identical support have the same covariance as
  their combined net correction.
- A separate calibrated synthetic example reproduces the selected calibrated
  covariance exactly. The full prefixed calibration payload survives NetCDF
  storage, can reconstruct the same `BiospherePrior`, and retains the full
  original prior and calibration policy. Frozen subset means are unchanged.
- `annotate_shared_baseline_trace`, `save_trace` and `load_trace` retain the
  complete calibrated prepared Dataset, including all MultiIndex axes.
- An independent Dask callback records zero tasks during preparation when
  transport and fixed/BC inputs are chunked, including secondary lazy
  observation coordinates. The transformed operators, BC column means and
  baseline remain lazy; the borrowed sensitivity is unchanged. A separate
  additive regression also covers independently chunked site/time coordinates
  on the unit-density design. Reading site labels to freeze site groups is an
  explicit preparation operation. Reading the small `outer_source` labels to
  validate six net-biosphere and six fossil patterns is also permitted; the
  additive zero-task check keeps those categorical labels eager to isolate
  observation-coordinate comparisons.
- Unsupported covariance with removed ocean states is rejected. Gated BC
  columns can become nonzero outside their original gate after centring,
  confirming that the whole-window mean is removed after period gating.

The focused test owners are `tests/test_shared_baseline.py` and
`tests/test_shared_baseline_review.py`. Additional independent
checks reconstructed the calibrated payload and exercised full trace storage,
rather than inferring those guarantees from the Dataset variable names.
The independent focused run passed 10 tests in 40.30 seconds. A separate run
passed the lazy-preparation and newly added gross/net covariance tests
(2 passed, 9 deselected in 0.78 seconds), covering all 11 current cases. The
separate additive observation-coordinate regression passed in 1.43 seconds,
bringing the reviewed focused coverage to 12 cases. Ruff checks, formatting
checks and `git diff --check` pass. The warnings concern the existing fixed-OU
operation's Numba object mode and
PyTensor gradient API; no focused check failed in the final reviewed revision.

## Findings addressed before handoff

The implementation corrected collisions between original, retained and outer
MultiIndex metadata, including covariance-column identity coordinates. It now
persists the original prior, resolved calibration, physical density widths and
BC preparation identity. Observation unit mismatches are rejected instead of
relabelled. Physical BC output uses the raw physical correction; the centred
atmospheric correction and removed mean remain separate. Repeated auxiliary
coordinates no longer trigger numerical execution during preparation. The
cached likelihood tests now exercise the actual refreshed sampler target.

## Readiness and remaining limits

The reviewed API is ready for the sequential configuration handoff. The new
recipe supports CO2 only; it does not implement a linked CO2/O2 shared-baseline
recipe. Existing linked routes retain their separate capabilities. Input
transport, fixed affine terms and additive designs must already use the BC
reference's concentration units; this API does not silently convert those
arrays or recalibrate widths on reload or subsets.

The review does not select production uncertainty defaults, validate a fresh
native reduction, or establish that concentration fit improves flux recovery.
The accepted `results-r3` prototype was read as experimental evidence, not as
proof that the new implementation has completed its four-arm replay. The parent validation run has since completed full
matched sampling, Monte Carlo uncertainty assessment, broader compatibility
tests and Slurm CI-equivalent checks; see [the replay record](shared_baseline_replay.md)
for final outcomes and the explicit multiplicative TER comparison exception.
That evidence supplements this API review without changing its scientific scope.

Reviewed source SHA256 values:

- `openghg_inversions/models/shared_baseline.py`:
  `7ccf1e0e5ce901b5304c47029ba4f1f493ba66e0ddacd962d632f5200a001f09`.
- `openghg_inversions/rhime/co2/shared_baseline_model.py`:
  `212b3a3fc2b6c14e8a0db533efee23e8dfdfb3dfbca1669912a309c942191068`.

The parent task records the final committed revision; later changes to these
scientific owners require proportionate review and validation.

## Centred-null physical-state correction and final validation

The subsequent PR review found that a nonzero constant outer response can
become zero after centring. Structural pruning then fixed its coefficient at
one, losing physical prior uncertainty even though the centred concentration
prediction still closed. Commit `37e8b29d` corrects this by retaining the full
joint physical prior in both shared builders. General builders retain their
existing pruning default and explicit activity policies.

The regression uses an inner response `[2, 6]`, raw outer response `[4, 4]`,
outer coefficient mean 1.5 and nonzero inner/outer covariance. It verifies the
model-implied arithmetic prior moments, random physical response
`4 * x_outer`, zero centred response, absolute prediction, normalized densities
and ordinary/refreshed-cache gradients. This is a synthetic equation and
uncertainty check; it does not score campaign flux recovery. The reviewer
independently rechecked the source correction and found no residual defect in
that scoped finding.

Immutable source `d9d730bc505edb61c2ccd209e420d291db444074` passed all 238
selected tests on each of Python 3.12.11 and 3.13.7 in Slurm array task
`19888803_0` (exit `0:0`, 18m19s, batch MaxRSS 5,608,128 KiB). Coverage includes
shared algebra and physical diagnostics, correlated state/activity, borrowed
sensitivity, ordinary/cached inference and boundary priors. Both compiler/header
preflights passed. The log SHA256 is
`269d1acf4949baa04149e035d2f5da8bd540980aa99ec7d919ac59b23140afd6`.

Repository lint and full Sphinx also passed on that immutable source in
`19888806_0` (exit `0:0`). The fresh build reported ten warnings about existing
SiteOptions aliases and ambiguous cross-references, with none in the changed
guides or builders. The earlier incremental build recorded four warnings, so
unchanged warning counts are not claimed. Rendered coefficient prose and its
additive-boundary cross-link were checked. These are relevant regression and
documentation checks, not a full-repository scientific test matrix or a new
campaign replay.
