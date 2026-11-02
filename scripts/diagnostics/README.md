OPE-241 temporal sampling replay
===============================

`replay_ope241_temporal_sampling.py` is an optional acceptance diagnostic using
private Verification Games archives. It is independent of ordinary pytest and
requires the repository's locked environment plus the archived Verification
Games source for `apply_endpoint_overlay`. It writes only to a new output
directory and never fits a model or changes archived inputs.

The native check covers repaired July 2021 endpoint particles at all archived
sites, CO2 and O2, and four explicit field policies: archived hold, nearest,
linear, and interval containment. Interval bounds of midpoint ±90 minutes are
an explicitly declared sensitivity hypothesis. They are not producer-confirmed
metadata. Both nearest and interval are checked against the independent native
interval oracle on this verified regular three-hour grid. Public production
sampling and normalized contraction run before comparison. The only spatial
adaptation is relabelling native coordinates after the independent oracle's
producer-rounding tolerance checks.

The diagnostic explicitly declares three-hour hold coverage after verifying
the complete native grid, and dimensionless units for the archived particle
fractions whose unit labels are empty. These owned metadata views satisfy the
production input contract without modifying the archived files. The declarations
are recorded in the output receipt and do not confirm the producer's time semantics.

The selected replay uses every entry in authoritative `selection-r2.json`,
including its replacement monthly midpoint-hold delta+A fit. The 15 cases span
three timing hypotheses and multiplicative monthly/8-day, global level
(`delta_only`), and global level plus pattern (`delta_A`) monthly/8-day models.
It preserves the archived row labels/order, scalar reference, pattern width,
period gate and priors, compares every reconstructed boundary design operand,
and replays boundary means and pointwise prior/posterior standard deviations
from hash-verified accepted posterior coefficients. Other concentration and flux
scores remain supported by the archived selected provenance; this diagnostic
does not rescore them or establish a producer-preferred temporal policy.

Run `--self-test` directly for the bounded row-alignment and cache-validation checks.
Submit the full command through the installed Slurm callback workflow. A measured
run completed 28 of 31 sites in 90 minutes with about 355 MiB peak resident memory;
allow about two and a half hours and 4 GiB for a fresh full replay, subject to
shared-storage performance. The script processes native
curtains in 24-hour batches and retains only reduced time-by-curtain operands.
From the repository root, the compute-node command is:

```bash
export PYTHONPATH="$PWD:/group/chem/acrg/brendan_for_codex/.codex-worktrees/8476/verification-games/src"
export DASK_NUM_WORKERS=1
.venv/bin/python scripts/diagnostics/replay_ope241_temporal_sampling.py \
  --timing-spec /group/chem/acrg/verification_games_round_2/forward_model_intermediates/name_boundary_time_sampling_20261031_ope241_r1/spec.json \
  --selection /group/chem/acrg/brendan_for_codex/openghg_inversions/.ad_hoc_inversion_analysis/corso-timing-inversions-20261031/selection-r2.json \
  --output /path/to/new/replay-directory
```

`replay.json` is written only after all checks pass. It records errors, accepted
selection provenance, input hashes, and limitations. Reduced native operands
are saved per site. A failed check exits nonzero and leaves partial diagnostic
outputs for inspection. Native boundary/spec/endpoint identities and imported
overlay/helper sources must match the frozen oracle receipt; changed external
sources require a new reviewed acceptance archive.

To continue an interrupted run, use the same output directory and add
`--resume --resume-origin /path/to/origin.json`. The origin JSON must declare
the original full `script_sha256` and absolute `log_path`; include the original
git revision and job ID when available. This is caller-supplied historical
provenance, recorded separately from the current validation script hash. The
origin JSON and previous log are hashed into the final receipt.

Resume checks every existing site's complete species/policy/time/curtain grid,
mass, anomaly and absolute contribution against the independently hash-verified
native oracle before reusing its reduced arrays. Boundary sources, footprint
metadata, endpoint overlays, specification and helper hashes are checked again.
Invalid cached files fail the run; only absent site files are computed. Cached
NetCDF files remain unchanged, and each site gets a `.validation.json` receipt
with its hash, oracle/spec identity and validation result. The selected 15-case
design/posterior replay always runs after all native sites pass. Without
`--resume`, an existing output directory is still refused.
