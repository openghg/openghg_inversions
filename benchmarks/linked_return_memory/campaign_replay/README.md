# Original-shape saved-state return replay

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
