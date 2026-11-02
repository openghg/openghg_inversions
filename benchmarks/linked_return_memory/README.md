# Linked CO2/O2 result-return memory

This bounded experiment reproduces backend storage amplification and measures
the compact return path while preserving every public posterior output. It does
not reproduce the original failed allocation or replace a full campaign run.

The scientific graph is based on revision
`c913da1a2fe2f816da2ba5eff63ca15847dba987`; `full` uses its recording policy
and `compact` uses the change accompanying this report. The fixture builds the
production ordinary or cached linked fixed-OU model, with eight retained flux
states, rank-two aggregation covariance, four stations, multiplicative boundary
conditions, and optional three anchored O2 site offsets. Both species use ppm;
the O2 absolute concentration baseline is 209,300 ppm. These are constructed
inputs and repeatable perturbed parameter states, not posterior inference or
scientific convergence evidence. Centered/additive boundary equivalence is
covered by the change's focused scientific tests, rather than this fixture.

The measurements used Python 3.13.7, PyMC 6.3.2, float64, one numerical thread,
four chains, 128 warmup states and 128 retained states per chain on 30 October
2026. Every case ran in a fresh process. The original campaign's request records
PyMC's Python environment at
`/group/chem/acrg/verification_games_round_2/verification-games-work/sources/ope188-ogi-devel-20261028/.venv/bin/python`;
the local worktree environment also provides PyMC 6.3.2.

## Reproduce

The profiler is [scripts/profile_linked_return.py](../../scripts/profile_linked_return.py).
Run from the repository root after `uv sync`. The following is the exact
environment used, with no pre-existing `PYTENSOR_FLAGS`. If flags are already
present, preserve the other entries and replace only `floatX` and
`base_compiledir`, without defining either twice.

```bash
export PYTHONPATH="$PWD"
export MPLCONFIGDIR=/tmp/linked-memory-matplotlib
export ARVIZ_DATA=/tmp/linked-memory-arviz
export PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor-linked-return,floatX=float64
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
for strategy in full compact; do
    .venv/bin/python scripts/profile_linked_return.py "$strategy" \
        --rows 512 --draws 128 --tune 128 \
        --output "benchmarks/linked_return_memory/ordinary-offset-512-$strategy.json"
done
```

Repeat the pair with `--rows 1024` for `ordinary-offset-1024`; with
`--rows 512 --no-offsets` for `ordinary-no-offset-512`; and with
`--rows 512 --cached` for `cached-offset-512`, updating output filenames to
match. Each JSON records the executed profiler's SHA256, interpreter version,
all retained variable shapes/dtypes/SHA256s, likelihood SHA256, exact array byte
counts, stage timings, and memory measurements. The committed JSONs retain
their original temporary output destinations in `options.output`.
The profiler SHA256 for all eight measurements is
`e87c5d0bd3b2de577741606b885a1b18c866dd6f001d3f1b48c4033d16eab9cd`.

Compare all scientific output arrays and the normalized joint likelihood:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("benchmarks/linked_return_memory")
for name in ["ordinary-offset-512", "ordinary-offset-1024",
             "ordinary-no-offset-512", "cached-offset-512"]:
    full = json.loads((root / f"{name}-full.json").read_text())
    compact = json.loads((root / f"{name}-compact.json").read_text())
    assert full["posterior_variables"] == compact["posterior_variables"], name
    assert full["log_likelihood_sha256"] == compact["log_likelihood_sha256"], name
    assert full["posterior_bytes"] == compact["posterior_bytes"], name
    assert full["slice_borrows_backend"] and compact["slice_borrows_backend"]
print("All four pairs preserve every output byte and joint likelihood byte")
PY
```

The profiler also asserts equal posterior and likelihood datasets after a
NetCDF save/load roundtrip, including the restored joint MultiIndex. No draws
are thinned and no precision is reduced. The synthetic fixture has constant
deterministics, so diagnostic runtime warnings about zero variance are expected;
the diagnostic values are not convergence assessments of a fitted posterior.

## What the measurements demonstrate

PyMC 6.3.2's `sampling.mcmc._sample_return` retains the original warmup-plus-draw
NDArray backends while slicing retained samples and converting them. Installed
`backends.ndarray.NDArray._slice` uses NumPy views; the profiler verifies shared
storage. `backends.arviz.DataTreeConverter.posterior_to_xarray` then uses
`np.array` on retained chain values, allocating the public posterior separately.
Automatic likelihood evaluation and convergence checks occur before those
backend references leave `pm.sample`'s return path.

The compact path records free random variables only, releases backend references,
computes the normalized joint likelihood, and reconstructs all named
deterministics using the existing PyMC graph. Installed
`compute_deterministics`/`apply_function_over_dataset` preallocate final output
arrays and evaluate one state at a time. Their point list scales with state
dimension and draw count. Likelihood evaluation selects free random variables
and uses `astype(copy=False)`; it does not copy the observation-sized posterior
deterministics. Coordinate restoration borrows numerical arrays.

MiB means 2^20 bytes. Peak RSS is Linux `ru_maxrss` for the whole process,
including imports, compilation, all return stages, eager serialization reload
and final diagnostics. Stage JSONs also contain start/end RSS and a 5 ms sampled
peak. A high-water value within a stage can reflect an earlier stage; sampled
peaks can miss shorter allocations. Allocator reuse can keep RSS high after
arrays are released. These are single measurements, without uncertainty bounds.

| Fixture | Rows per species | Full backend MiB | Compact backend MiB | Public posterior MiB, both | Full peak RSS MiB | Compact peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary, O2 offsets | 512 | 76.336 | 0.211 | 38.137 | 551.51 | 497.08 |
| Ordinary, O2 offsets | 1024 | 152.336 | 0.211 | 76.137 | 810.14 | 671.37 |
| Ordinary, no offsets | 512 | 56.312 | 0.188 | 28.125 | 522.55 | 476.11 |
| Cached, O2 offsets | 512 | 84.336 | 0.211 | 42.137 | 581.01 | 506.15 |

Adding the O2 offsets at 512 rows per species increases full recording by
20,996,096 bytes (20.023 MiB), of which just 24,576 bytes are offset coefficients
over warmup plus retained states. The compact recording increase is those
24,576 coefficient bytes alone. The three expanded offset outputs still add
10,485,760 retained bytes to both final public posteriors; their coefficients add
another 12,288 retained bytes. The full recording increase is twice this final
offset contribution, before the separate retained xarray allocation.

At 512 rows per species, ordinary-model return phases through coordinate
restoration peak at 543.40 MiB in full mode and 445.51 MiB in compact mode.
The compact process subsequently peaks during eager serialization reload and
diagnostics, after reconstruction. This later peak is included in the table.

All four comparisons preserve every public posterior variable and the
scalar joint log-likelihood byte for byte. Storage still scales with the final
public posterior; reconstruction removes warmup/backend duplication and
conversion coexistence, rather than eliminating that necessary result size.
Serialization reload explicitly holds another complete result, and diagnostics
can require their own workspace. Predictive draws are outside this profiling
fixture; they remain covered by the sampling tests and can add another
observation-sized draw array.

## Relation to the supplied campaign evidence

The supplied `linked-uob-retry-r2/report.md` and `completion.json` describe
5,314 rows per species, four chains, 1,000 tuning and 1,000 retained draws:
approximately 14.1 GiB without offsets, failure after sampling under 16 GiB with
offsets, and successful retries at approximately 16.5 GiB under 24 GiB. Those
figures are supplied campaign evidence, not new measurements from this fixture.
The separate diagnostic MultiIndex write failure is not this memory failure.

At that campaign shape, the three O2 offset deterministics have
`(5314 + 10628 + 10628) * 4 * 1000 * 8 = 850240000` retained bytes, or
810.85 MiB. Warmup-plus-draw backend storage plus the copied retained posterior
can hold three times that payload, approximately 2.38 GiB, before considering
other deterministics, graph/kernel workspace, or allocator behavior. The small
offset coefficients do not explain that amplification. This accounting and the
bounded profiles demonstrate avoidable output storage; neither proves which
allocation exhausted the original job's memory limit. A 334-state, rank-256
full-campaign fit is outside this bounded fixture; the separate original-fit
acceptance results below complete that check.

## Original-case validation beyond the bounded fixture

The bounded results above establish storage amplification; they do not by
themselves establish that the original 16 GiB fits now complete.
`scripts/replay_linked_campaign_return.py` adds a separate campaign-specific
replay using the original 5,314 rows per species, 334 states, rank-256 covariance
sidecar and successful retry's saved free-variable draws. It runs real NDArray
recording and the installed PyMC `_sample_return` inside the production sampler
and linked runner. It records separate conversion, likelihood, reconstruction,
annotation, serialization and diagnostic phases, flushing progress before each
phase so a killed job still identifies its last active phase.

The replay repeats saved retained states to fill warmup-shaped buffers; the
original warmup trajectory was not saved. It bypasses sampling updates and the
unused dense covariance's validation before replacing that covariance with the
same checksum-verified rank-256 sidecar. It therefore tests the original return
dimensions and scientific graph, rather than reproducing an entire original
process or its exact failed allocation. Backend object and payload weakrefs
must be dead before deferred likelihood or deterministic evaluation, without
forced garbage collection. Source comparisons read at most 16 draws at once.

Both full-row smoke checks passed in the original campaign interpreter with
one chain, two retained states and two repeated warmup states. All public
variable names, coordinates, attributes and scientific dtypes matched the
saved source. The joint likelihood and all but one posterior variable matched
between strategies byte for byte. The absolute `modelled_concentration` differed
by at most one float64 ULP, about 2.91e-11 ppm; source comparison reports both
hashes and numerical errors instead of claiming complete byte identity.

Submit the default full draw-count pair on this cluster with:

```bash
slurm-wakeup submit --allow-high-memory \
  "Matched original-case return replay: sacct 19824176_1.batch reports 17292540K MaxRSS (16.49 GiB); 24 GiB permits measuring the original path and the compact path under identical allocations" \
  -- sbatch --output=.artifacts/campaign-replay-%A_%a.out \
  scripts/slurm_replay_linked_campaign.sh "$PWD" "$PWD/.artifacts/campaign-replay"
```

The wrapper uses the original campaign environment, four numerical threads and
node-local PyTensor/Numba caches. The default array compares full and compact
recording at four chains, 1,000 retained states and 1,000 warmup-shaped states.
Use a fresh output directory for every submission. It requires access to the
frozen campaign artifacts named in the script.

The final acceptance check separately repeats both original offset fits using
the original retry harness, dependencies, data, seed and sampling settings,
under the original 16 GiB limit. The immutable candidate checkout is
`c5afe448e1eefea4014fe0144273d6425e6e376f`; its production files contain this
fix. `original_acceptance_adaptation.json` records the source hashes and exact
harness diff: only the source checkout/revision and 24-to-16 GiB allocation
change. **Both actual fits completed and saved posterior and diagnostic files
under 16 GiB**, at process peaks of 6.635 and 6.644 GiB. Comparable batch MaxRSS
fell by 59.2% and 59.4% relative to the original successful 24 GiB retries. See
the [full-fit acceptance report](original_acceptance/README.md), raw outputs,
requests and accounting for provenance and scientific validation.

The full draw-count replay completed: peak RSS fell from 16.53 to 6.88 GiB,
and all recomputed posterior arrays and the normalized joint likelihood match
the original saved trace byte for byte. The original return path peaks during
automatic likelihood computation while both the 9.234 GiB backend and the
4.615 GiB retained posterior exist. A separate full-path replay under the
original 16 GiB cap was OOM-killed during that same native return phase, before
returning or saving. See [the campaign report](campaign_replay/README.md) and
its raw profiles, negative-control output and accounting for the measured
phases and remaining replay limitations.

The expanded 24-module validation passed all **572 tests on each of Python
3.12 and 3.13**, using locked dependencies in Slurm job `19833596`. The modules
cover compact trace behavior, sampler selection/replay/diagnostics, linked
ordinary/cached runners, centered/additive boundaries, saved-input handoffs,
serialization, configuration ownership and PARIS output. PyMC 6.0 lower-bound
validation separately passed all 40 selected cases (`19833388`); `docs-full`
passed with 10 existing warnings (`19833264`). Changed Python paths pass Ruff
and `git diff --check`.
