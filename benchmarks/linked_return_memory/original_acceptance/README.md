# Original full-fit acceptance under 16 GiB

Both original offset fits completed sampling, returned through PyMC, and saved
complete posterior and diagnostic files under the original **16 GiB** limit.
This closes the whole-fit acceptance check that the saved-state replay could
not establish.

Slurm array `19833552` repeated the original anchored and free O2 site-offset
arms: 5,314 observations per species, 334 states, rank-256 covariance, four
chains, 1,000 tuning iterations and 1,000 retained draws. It used the same
prepared files, low-rank sidecar, Python interpreter/dependencies, seed
`20261028`, sampler settings, priors and diagnostic code as the successful
24 GiB retry. Both tasks started on 30 October 2026 at 09:49:37 UTC and finished
at 12:05:12 / 12:05:14 UTC. Both Slurm and the process report successful exits.

The frozen candidate is `c5afe448e1eefea4014fe0144273d6425e6e376f`; current
production files are unchanged from that candidate. The
[exact harness adaptation](../original_acceptance_adaptation.json) records
original/candidate script hashes and diffs: only source checkout/revision and
allocation changed. The protected validation checkout was not modified.
Each arm directory contains its original request, completion manifest and raw
process output. [Accounting](accounting.json) records allocations, scheduler
states, output sizes, process peaks and phase timings.

## Comparable whole-fit memory

| Measurement | Anchored offsets | Free offsets |
| --- | ---: | ---: |
| Original successful retry allocation, GiB | 24 | 24 |
| Original retry batch MaxRSS, KiB | 17,292,540 | 17,284,784 |
| Fixed fit allocation, GiB | 16 | 16 |
| Fixed fit batch MaxRSS, KiB | 7,050,880 | 7,010,512 |
| Batch MaxRSS reduction | 59.2% | 59.4% |
| Fixed process MaxRSS, KiB (`/usr/bin/time`) | 6,957,680 | 6,966,224 |
| Fixed process MaxRSS, GiB | 6.635 | 6.644 |
| Fixed elapsed time | 2:15:35 | 2:15:37 |
| Saved posterior file, bytes | 4,993,550,677 | 4,993,646,005 |
| Saved diagnostic file, bytes | 91,275 | 91,307 |

Batch MaxRSS is compared with batch MaxRSS from original successful retry
array `19824176`; process MaxRSS is separately labelled. These single runs used
different compute nodes, so they do not establish a speed change. The matched
[saved-state return experiment](../campaign_replay/README.md) separately
isolates recording, conversion and likelihood coexistence, with peak process
RSS 16.534 to 6.879 GiB. Its 16 GiB negative control is OOM-killed inside native
PyMC return. The historical failing process's precise inner allocation remains
unknown.

## Scientific and output checks

The original selected-parameter diagnostic gates pass in both completed fits:
maximum Rhat is 1.0063944 / 1.0090985 and minimum bulk ESS is 590.06 / 469.38.
All recorded divergence and maximum-tree-depth event counts are zero. These
values match the original successful retries. This is a storage regression
check using the existing diagnostic recipe, not approval of exploratory priors
or a new scientific convergence criterion. The original common BC/site-offset
identifiability and derived common-mode diagnostic caveats remain; they are not
changed by trace storage.

[Saved-output validation](validation.json) compares every saved array using
one chain and at most sixteen draws per read. In **both arms, all 71 non-timing
arrays are byte-for-byte identical to the original successful retry**: all
24 posterior variables, 26 non-timing sampler statistics, 19 constants, observed
data and normalized joint likelihood. Names, dtypes, units, variable
attributes, coordinates and decoded MultiIndexes agree. Posterior and
likelihood values are finite. Only the three timing-stat arrays and generated
`created_at`/`sampling_time` group attributes differ. The entire saved selected
parameter diagnostic datasets also match exactly.

The [bounded comparison script](../../../scripts/validate_linked_campaign_acceptance.py)
checks the complete result without eagerly holding two approximately 4.7 GiB
artifacts. Its JSON records per-variable hashes and errors, group-attribute
differences, diagnostic gates and validation provenance. Fixed-state numerical
checks are additionally covered by the saved-state replay and focused tests.

The complete public posterior remains approximately 4.615 GiB of array payload.
Its memory scales with observations, retained draws, chains and requested
outputs; predictive draws or eager complete-file reloads can increase peaks.

## Reproduce

The original harness is under
`/group/chem/acrg/verification_games_round_2/verification-games-work/ope225-bc-offset-20261029-r1/linked-uob-retry-r2/scripts/`.
Copy its two files, apply the exact source/revision/allocation changes recorded
in `original_acceptance_adaptation.json`, and verify the candidate SHA256s.
Use a detached checkout at the recorded candidate revision. The completed
submission used this command from the task worktree:

```bash
slurm-wakeup submit -- sbatch \
  --output=.artifacts/linked-return-memory-validation/original-acceptance-%A_%a.out \
  .artifacts/linked-return-memory-validation/original-acceptance-harness/slurm_linked_options.sh \
  "$PWD/.artifacts/linked-return-memory-validation/original-acceptance-harness" \
  "$PWD/.artifacts/linked-return-memory-validation/original-acceptance-r1" \
  "$PWD/.artifacts/linked-return-memory-validation/acceptance-source"
```

Choose a fresh output directory for another run. The wrapper loads Git, uses
node-local caches, preserves PyTensor flags, and selects the original campaign
interpreter. Register its terminal callback through `slurm-wakeup`; do not
poll through its runtime. No allocation above 16 GiB is needed for these fixed
fits.

Reproduce the bounded saved-output comparison from this repository root:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/validate_linked_campaign_acceptance.py \
  --acceptance "$PWD/.artifacts/linked-return-memory-validation/original-acceptance-r1" \
  --output /tmp/linked-original-acceptance-validation.json
```

The default comparison source is the original `linked-uob-retry-r2`; use
`--retry` to select another frozen copy. Use a fresh output path for a complete
scan. The evidence records the initial/resumed scans and the assertion-only
final checks; no fit was repeated during validation.
