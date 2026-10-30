# Original-shape saved-state return replay

At the original campaign shape, the compact return path reduced measured peak
resident memory from **16.534 GiB to 6.879 GiB**, while preserving every compared
array byte, including the normalized joint likelihood. These completed
saved-state replays isolate the return path; they do not repeat MCMC inference.

`scripts/replay_linked_campaign_return.py` rebuilds the frozen campaign's
5314-observation-per-species, 334-state, rank-256 cached linked model, with the
original centered O2 boundary at 209300 ppm, additive O2 correction priors,
multiplicative CO2 boundary, and anchored or free O2 site offsets. It uses the
original prepared-manifest and sidecar checksums. The unused dense covariance is
excluded before loading; the already-validated low-rank sidecar supplies exactly
the covariance representation used in the campaign fit.

This replays saved retained physical free-RV values through their value-space
transforms, real PyMC NDArray recording, and the installed
`pymc.sampling.mcmc._sample_return`. The production cached runner and
`RhimeSampler` perform coordinate restoration, scientific annotation and, in
compact mode, deferred joint likelihood and deterministic reconstruction.
Repeated retained states stand in for the unsaved original warmup trajectory.
Sampler-stat values are also copied from the original retained draws and
repeated for warmup; they are not evidence of new sampling or new convergence.

By default each fresh-process replay records four chains of 1000 warmup and
1000 retained states. The script records phase start information before work,
flushes profiling JSON and logs between stages, asserts via weak references
that backend objects and array payloads die before deferred likelihood and
reconstruction, and serializes the complete result. It compares all saved
groups against the original file using bounded reads, including variable
names, dtypes, dimensions, attributes, coordinates and scientific root metadata.
There is no full eager reload of the saved posterior. Selected diagnostics use
the original campaign's parameter-name selection.

Full original-size runs require Slurm and registered terminal callbacks. The
24 GiB replay allocation is justified by the supplied original successful
retry's approximately 16.49 GiB MaxRSS and identical shapes in both replay
strategies. The default full replay is an allocation experiment, not a new fit
or proof of the precise allocation that killed the original 16 GiB job.

Use the original campaign interpreter and current candidate source:

```bash
export PYTHONPATH="$PWD"
# On the allocated node, set writable node-local NUMBA_CACHE_DIR,
# MPLCONFIGDIR, ARVIZ_DATA and PyTensor base_compiledir; preserve existing
# PyTensor flags and set floatX=float64 without duplicate entries.
/group/chem/acrg/verification_games_round_2/verification-games-work/sources/ope188-ogi-devel-20261028/.venv/bin/python \
    scripts/replay_linked_campaign_return.py full \
    --arm anchored_site_offsets --output-dir "$RUN_ROOT/full"
```

Run `compact` in a separate fresh process with an otherwise identical command.
Outputs include the serialized `posterior.nc`, `diagnostics.nc`, and
`profile.json`. Profiles identify the request, streamed source-trace checksum,
imported package path, candidate revision and source-file hashes, exact
interpreter/dependency versions, replay-script hash, and native PyMC return
function hash. Source comparisons use posterior `rtol=1e-12, atol=1e-9` and
normalized joint likelihood `rtol=1e-10, atol=1e-8`; hashes additionally identify
bitwise equality. The physical-to-transformed roundtrip can introduce ULP-level
changes relative to stored original physical values.

For a bounded smoke check, retain the complete scientific model and add
`--chains 1 --draws 2 --tune 2`. Reducing the observation axis can change
structural-zero state pruning and the joint likelihood, so the script does not
offer that shortcut.

## Completed original-size return replays

[full.json](full.json) and [compact.json](compact.json) record Slurm array
19833554, one fresh process per strategy, with the same original interpreter,
four numerical threads, four chains, 1000 synthetic warmup states and 1000
retained states. Both tasks received 24 GiB. The frozen candidate is
`0cc621515f6a572046ed1a572a9944db76d6d9f8`; its scientific source is based on
baseline `c913da1a2fe2f816da2ba5eff63ca15847dba987`. The candidate's imported
file paths and hashes are included in each JSON. The interpreter, dependencies,
replay script and native `_sample_return` hashes match those listed for the
smoke checks below. The frozen source trace SHA256 is
`19fd7ffd105fe2d038fe248b2c8c8a3bc46c398496ae1ab8dda67912fc30546a`.

| Quantity | Full recording | Compact recording |
| --- | ---: | ---: |
| Warmup-plus-retained variable buffers, bytes | 9,914,752,000 | 26,624,000 |
| Warmup-plus-retained sampler-stat buffers, bytes | 1,632,000 | 1,632,000 |
| Complete public posterior, bytes | 4,955,648,000 | 4,955,648,000 |
| Whole-process peak RSS, MiB | 16,931.285 | 7,043.863 |
| Whole-process peak RSS, GiB | 16.534 | 6.879 |

The peak reduction is 9.656 GiB, or 58.4%, in these individual runs. The full
path's peak occurred inside the actual installed PyMC automatic likelihood
computation, with backend and converted posterior storage still live. Its
16,931.285 MiB high-water RSS exceeds a 16 GiB allocation. The compact path's
likelihood high-water RSS was 2,731.031 MiB after backend release; deterministic
reconstruction reached 6,801.684 MiB, and a later bounded comparison of saved
outputs against the source reached the overall 7,043.863 MiB peak.

The following table gives RSS **at each phase's end**, rather than assigning an
earlier process high-water mark to a later phase. Raw profiles additionally
provide start/end RSS, 5 ms sampled stage peaks, process high-water RSS, and
timings for input loading, model construction and sampler construction.

| Phase | Full end RSS, MiB | Compact end RSS, MiB | Full seconds | Compact seconds |
| --- | ---: | ---: | ---: | ---: |
| Backend recording | 10,312.391 | 838.062 | 29.128 | 3.042 |
| Xarray conversion | 15,039.301 | 853.289 | 1.965 | 0.017 |
| Joint likelihood | 16,412.102 | 2,079.285 | 895.729 | 1,203.251 |
| Backend release | 7,006.129 | 884.781 | 0.036 | 0.005 |
| Deterministic reconstruction | included in recording | 6,801.680 | included in recording | 15.353 |
| Coordinate restoration | 7,005.148 | 6,801.684 | 0.059 | 0.078 |
| Scientific annotation | 7,005.270 | 6,801.684 | 0.057 | 0.065 |
| Serialization | 6,881.531 | 6,812.812 | 6.417 | 6.408 |
| Bounded comparison with source | 6,882.566 | 6,820.312 | 92.239 | 107.904 |
| Selected diagnostics | 6,882.762 | 6,820.516 | 4.025 | 5.048 |

In compact mode backend release precedes joint likelihood evaluation. In full
mode it follows likelihood evaluation; the table lists matching phases for
comparison rather than execution order. The approximately 896 versus 1203
seconds for joint likelihood were measured once in separate allocated
processes. They establish that the unchanged normalized likelihood remains
costly; they do not establish a speed regression or improvement.

The exact array accounting explains the avoided coexistence. Full recording
holds twice the public retained posterior plus 3,456,000 bytes of transformed
positive parameters. Xarray conversion adds another complete retained
posterior, approximately 4.615 GiB, while those approximately 9.234 GiB backend
buffers remain live. Compact recording holds only six free-RV arrays, 25.391
MiB including synthetic warmup. Recording only free RVs eliminates
9,888,128,000 backend bytes, or 9.209 GiB. Weak-reference
assertions verify that neither backend objects nor their payloads survive
into the deferred likelihood/reconstruction phase, without forced collection.

For O2 offsets alone, full backend recording holds 1,700,480,000 bytes for
`o2_offset_native`, `o2_offset`, and `offset_concentration`; conversion adds
850,240,000 retained bytes for the same outputs. Their combined coexistence is
approximately 2.376 GiB. The actual retained offset coefficients occupy
768,000 bytes. All three expanded outputs remain present in the final result.

Every compared variable is **bitwise identical between full, compact and the
original source trace** in these complete replays: 24 posterior variables,
29 original sampler-stat variables, 19 constant-data variables, observed data,
and the scalar-per-draw normalized joint likelihood. All saved names, shapes,
dtypes, variable attributes, restored coordinates/MultiIndexes and checked
scientific root metadata agree. Every recorded maximum source error is zero.
Whole-file byte lengths can differ through generated metadata and storage
layout; array hashes establish payload equality. The recomputed 24 posterior
variables and joint likelihood provide the scientific-equivalence check.
Retained sampler statistics were copied from the successful original fit;
their equality checks serialization fidelity, not new sampler behavior or
independent convergence.

This assertion reproduces the comparison without reading the large arrays:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("benchmarks/linked_return_memory/campaign_replay")
full, compact = [json.loads((root / f"{name}.json").read_text())
                 for name in ("full", "compact")]
assert full["status"] == compact["status"] == "completed"
assert full["posterior_bytes"] == compact["posterior_bytes"]
for group, variables in full["compared_variables"].items():
    assert variables.keys() == compact["compared_variables"][group].keys()
    for name, variable in variables.items():
        other = compact["compared_variables"][group][name]
        assert variable["sha256"] == other["sha256"] == variable["source_sha256"] == other["source_sha256"]
        assert variable["shape"] == other["shape"]
        assert variable["dtype"] == other["dtype"]
        assert variable["max_absolute_error"] == other["max_absolute_error"] == 0
print("All 74 saved variables match full, compact and source byte for byte")
PY
```

The complete 4.615 GiB public posterior remains a required memory floor for
these settings. Normalized likelihood computation still needs substantial
kernel/compiler workspace, and the approximately 7 GiB final stages retain the
model and runtime workspace as well as the posterior. Requested predictive
groups or an eager full-artifact reload can add further observation-sized
arrays. They were disabled in the original case and these replays. There is no
claim of memory independent of observation count, retained draw count, chain
count or requested outputs.

The full replay now demonstrates the problematic magnitude and phase under the
original retained states: automatic likelihood evaluation exceeds 16 GiB while
large backend buffers coexist with the xarray result. Its replacement remains
below 7 GiB in the return sequence and preserves outputs. The original unsaved
warmup states, the skipped unused dense-input validation, and the absence of
actual MCMC updates still distinguish this controlled replay from the original
failing job. The exact allocation that exhausted that original job's limit is
not established by these results. Actual fits and a direct 16 GiB memory-limit
control are separate acceptance checks.

## Completed original-environment smoke checks

`smoke-full.json` and `smoke-compact.json` record completed two-state checks
using the full original observation/state/rank dimensions. The environment is
Python 3.13.7, PyMC 6.3.2, NumPy 2.2.6, SciPy 1.15.3, xarray 2025.9.1,
PyTensor 3.3.2 and Numba 0.63.1. Both profiles execute replay-script SHA256
`3eb90a190173486c1eb48da644817b8625ec6a7d7e4e6a06f33b01485aa99a03` and
native `_sample_return` SHA256
`636bbc93162fee48db21cc64b096326a5a064ebdd371c2cf811427897faa8c06`.

Both save/load comparisons pass for all 24 posterior variables, 29 original
sampler-stat variables, 19 constant-data variables, observed data and normalized
joint likelihood. All coordinates, variable attributes, dtypes, names and
scientific metadata agree. The full replay's largest difference from the
original is one float64 ULP in absolute O2 `modelled_concentration`,
2.9103830456733704e-11 ppm; compact's largest difference is
5.684341886080802e-14 ppm. The joint likelihood, original statistics,
constant-data payloads and observed data are bitwise equal to the source.
The full/compact posterior hashes differ only for `modelled_concentration`;
the difference remains one ULP in absolute O2 concentration. These small smoke
runs verify the replay implementation; their roughly 2.6 GiB compilation peaks
do not measure the campaign-sized recording/conversion amplification.
