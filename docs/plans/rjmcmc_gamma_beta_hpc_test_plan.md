# Gamma--Beta RJMCMC native-data HPC smoke test

## Purpose

This plan is the first real-data test of the restricted fixed-direction
Gamma--Beta tree state space. It uses the reconstructed modern PARIS May 2014
data but a Gamma--Beta tree model, not the moving-Voronoi model and not yet the
proposed full-tiling model.

The first runs answer four narrow questions:

1. does the frozen native-data coordinate conversion close exactly;
2. can every kernel execute on the full \(1382\times23424\) response;
3. do segmented checkpoint/restart and labelled outputs remain exact; and
4. what mobility and throughput do local split/merge moves achieve from low
   and high initial \(K\)?

They are wiring, durability, acceptance, and performance experiments. A
100--1,000-cycle smoke is not a convergence experiment and must not be
reported as a posterior result.

## Model under test

The predicted mole fraction is

\[
\hat y =
Y_{\mathrm{aprioriBC}}
+S X_{\mathrm{inner}}
+H_{\mathrm{outer}}x_{\mathrm{outer}},
\]

where:

- `YaprioriBC` is the archived row-aligned fixed boundary contribution;
- \(S=G/w\), with \(G=\mathtt{fp_x_flux}\) the response to unit grid-cell
  scaling and \(w\) the explicitly declared native-cell base measure;
- \(X_{\mathrm{inner}}\) is a Gamma root total distributed through active
  Beta split fractions on one canonical fixed-direction dyadic tree; and
- the six outer-region scaling coefficients are always active and have
  independent lognormal priors stated in arithmetic mean/SD units.

The smoke profile uses:

- \(K\in[5,500]\) with a uniform marginal prior on \(K\) and a uniform prior
  over the exact fixed-tree frontiers conditional on \(K\);
- root mean one when \(w\) is normalized, with an explicitly supplied root
  variance;
- constant split concentration \(\kappa\), with split means determined by
  nominal child mass;
- independent fixed Gaussian observation errors from `mf_error`;
- likelihood power one;
- arithmetic mean one and arithmetic SD one for each outer coefficient; and
- no OU correlation, inferred mismatch hierarchy, boundary inference,
  alternative split direction, or fixed-\(K\) topology move.

The validation defaults \(\kappa=2\) and root variance \(0.25\) are not yet a
scientifically calibrated replacement for the Lunt or Ganesan priors.

## Frozen input contract

The driver reads one immutable NetCDF file. The production PARIS profile has
dimensions:

| Dimension | Size | Meaning |
|---|---:|---|
| `nmeasure` | 1,382 | filtered May 2014 observations |
| `lat` | 183 | inner InTEM-label-6 rows |
| `lon` | 128 | inner InTEM-label-6 columns |
| `outer_region` | 6 | fixed-geometry inferred outer regions |

Required data variables are:

| Variable | Dimensions | Contract |
|---|---|---|
| `fp_x_flux` | `nmeasure, lat, lon` | finite response to unit cell scaling |
| `mf` | `nmeasure` | finite observations |
| `mf_error` | `nmeasure` | finite and strictly positive fixed errors |
| `nominal_weight` | `lat, lon` | finite, aligned, and strictly positive |
| `outer_design` | `nmeasure, outer_region` | finite six-column response |
| `YaprioriBC` | `nmeasure` | finite fixed concentration offset |

Dimension order may be shuffled, but dimension sets and labelled-coordinate
alignment must be exact. C-order flattening makes longitude the fastest
native-grid index.

The file should also retain meaningful `nmeasure` and `outer_region` labels,
numeric latitude/longitude coordinates, units, provenance for each source
product, and the construction script or command. The driver records the
whole-file SHA-256 and refuses a changed file on continuation.

### Nominal-weight decision

The adapter never floors or repairs zero weights. Before the first run, choose
and record one scientifically explicit policy:

- positive prior-emissions mass, if the prepared field is strictly positive;
- positive grid-cell area for an area-exchangeable computational baseline; or
- another reviewed positive base measure.

Uniform-cell or area weights define different priors from emissions-mass
weights. They are permissible baselines only when named as such. If the
intended emissions field contains zeros, the run is blocked until masking,
aggregation, or another policy is agreed; an undocumented epsilon floor is
not acceptable.

## Required closure checks

Before sampling, the driver must verify:

\[
S w = \sum_j G_j
\]

and, at the prior-mean deterministic initial state,

\[
\hat y =
Y_{\mathrm{aprioriBC}}
+\sum_jG_j
+H_{\mathrm{outer}}\mathbf 1.
\]

Both are checked row by row to tight floating-point tolerance. The dry run
also compares the frozen file with `--expected-input-sha256`, returns the
complete canonical manifest (including the input and initial-state hashes),
verifies tree shape, supported \(K\), the 14-slot schedule, and optional exact
PARIS dimensions. PARIS mode additionally requires the reviewed
`outer_region` labels in exact column order.

## Transition schedule and units

With six fixed coefficients, one cycle contains 14 atomic transitions:

1. two independently mixed split/merge opportunities;
2. one independent-prior root-total refresh;
3. five independently selected active-fraction refreshes; and
4. one deterministic Gaussian random-walk update for each outer coefficient.

Every unavailable structural direction or fraction update remains an
explicit self-transition and consumes the ordinary acceptance uniform. The
driver's preferred `--cycles` option is therefore unambiguous. The
`--iterations` option exists only to exercise exact mid-cycle restart.

The sampler rejects:

- disconnected positive support in \(p(K)\);
- a singleton fixed \(K\) with more than one admissible frontier, because the
  current schedule has no fixed-\(K\) topology move; and
- zero fraction-refresh slots when the supported state space can contain an
  active fraction.

## Durable segment contract

Every successful segment writes a new output directory containing:

- canonical `manifest.json`;
- checksummed no-pickle `checkpoint.npz`;
- labelled `trace.nc`;
- compact `summary.json`; and
- `complete.json`, written last.

The immutable manifest binds the numerical problem, frozen input SHA,
coordinate/base-measure policy, code revision, chain identity, initial-state
fingerprint, priors, \(p(K)\), kernel settings, retention, and seed. A loaded
checkpoint rebuilds every prediction and target cache from the irreducible
frontier/root/fraction/fixed coordinates before continuing. The trace records
retained states, tree/node geometry and scientific labels, every attempted
transition, raw and powered likelihood terms, all prior components, masks,
sentinels, and global transition coordinates.

Use a distinct directory per chain and segment. Never overwrite the last good
segment. A continuation must validate the preceding completion marker and
file hashes before accepting its checkpoint.

## Stage 0: local validation and performance preflight

Completed local validation includes:

- tiny-tree enumeration and exact detailed balance;
- independent product-space target cross-check;
- exact direct/adapter forward-model closure;
- exact full-versus-split continuation across a mid-cycle boundary;
- durable save/load/rebuild and corruption/mismatch rejection;
- prior-only topology/root/fraction moment recovery;
- a six-outer 14-slot integration test; and
- NetCDF output with zero retained draws and variable \(K\).

A dense synthetic test with the production shapes, \(K=250\), and six fixed
coefficients took 3.14 seconds for 100 cycles (445 atomic transitions/s) on
the local development machine. Peak resident memory was 1.16 GB. These are
capacity checks, not BP1 performance predictions.

## Stage 1: production-data dry run

Run the driver twice with the exact production file, settings, code revision,
and distinct chain identities: once from \(K=50\) and once from \(K=250\).
Add `--dry-run --require-paris-profile`, the expected whole-file SHA, and the
reviewed comma-separated outer labels.

Pass criteria:

- exact input SHA matches the frozen-input record;
- both closure errors are within the declared tolerance;
- the problem fingerprint and initial-state fingerprint are recorded;
- the resolved cycle length is 14;
- the separate \(K=50\) and \(K=250\) initial states both construct with
  finite targets and distinct recorded initial-state fingerprints;
- no output directory is created; and
- the output NetCDF backend successfully writes and reopens a probe on the
  selected filesystem.

Archive both dry-run JSON documents beside the frozen-input metadata. Record
peak memory for each dry run with `/usr/bin/time -v` or Slurm `sacct`; memory
is an external process metric rather than a Python-return field.

## Stage 2: short wiring and acceptance run

Run two chains:

| Chain | Initial \(K\) | Cycles | Atomic transitions |
|---|---:|---:|---:|
| low | 50 | 100 | 1,400 |
| high | 250 | 100 | 1,400 |

Use one complete segment per chain with `--warmup 280` (20 cycles) and
`--thin 14`. Under the global retention convention this yields 81
cycle-aligned states including the state at transition 280. Request one CPU,
8 GB, and 30 minutes per array task. Pin BLAS/OpenMP thread counts to one; do
not hard-code a cluster partition in the repository example.

Correctness gates:

- both directories end with a valid completion marker and file hashes;
- each checkpoint reloads against the same problem and manifest;
- schedule phase is zero after 100 cycles;
- each chain records 200 structural, 100 root, 500 fraction, and 600 fixed
  attempts;
- each of the six fixed coefficients has exactly 100 attempts;
- accepted implies valid, invalid stays retain \(K\), and no target or ratio
  contains NaN or positive infinity;
- every continuous kernel and each outer coefficient accepts at least once;
- accepted splits and merges both occur somewhere across the low/high pair;
  zero accepted proposals in either direction fails the smoke, while less
  than one percent acceptance among valid proposals is a warning requiring
  inspection; and
- a separate validation chain with `--warmup 0 --thin 1`, deliberately split
  into 5+9 transitions, exactly reproduces one uninterrupted 14-transition
  chain, including every retained coordinate/cache, final checkpoint arrays,
  RNG, complete attempted path, and retention coordinates.

Performance reporting must separate input/problem construction, sampling,
checkpoint output, trace output, and summary output. Record CPU time, wall
time, atomic transitions/s, and peak RSS from Slurm.

## Stage 3: medium stability and mobility run

If Stage 2 passes and its extrapolation is below 70% of the requested
walltime, run four chains alternating \(K=50\) and \(K=250\):

- 1,000 cycles per chain;
- ten immutable 100-cycle segments;
- `--warmup 2800` (200 cycles);
- thinning every 14 atomic transitions; and
- an initial request of one CPU, 8 GB, and four hours per chain.

Report by chain and segment:

- attempts, valid proposals, accepted proposals, and acceptance conditional
  on validity for every move and fixed coefficient;
- visited \(K\) range, edge flows, immediate reversals, first-passage
  distance, and net displacement;
- root total and six outer coefficient summaries;
- raw Gaussian likelihood, powered likelihood, each prior term, and complete
  log target;
- setup/sampling/I/O throughput and peak RSS; and
- exact checkpoint/attempt-coordinate continuity.

Warnings requiring profiling before a longer run include:

- I/O above 10% of wall time;
- warmed-segment throughput falling by more than 20%;
- RSS growing by more than 10% across equal segments;
- any fixed coefficient with no accepted update;
- structural acceptance concentrated at only a few \(K\) edges; or
- repeated one-step split/merge reversals with little scientific prediction
  movement.

## Convergence and promotion

Before Stage 3, commit a postprocessor that defines and calculates the
following metrics from the segment traces; the sampler driver does not itself
claim convergence. The medium run should calculate multi-chain rank-normalized split
\(\hat R\), bulk/tail ESS, and MCSE for \(K\), root total, log target, all six
outer coefficients, and scientific prediction/flux summaries. Failure is
diagnostic rather than a wiring failure.

No result should be promoted as a converged scientific inversion without, at
minimum:

- \(\hat R\le1.01\);
- bulk and tail ESS at least 400;
- MCSE/SD no greater than 0.05;
- overlapping \(K\) distributions from low/high starts; and
- repeated traversal or round-trip evidence appropriate to the posterior
  support.

If local fixed-direction split/merge remains slow, the next comparison is
with fixed-\(K\) tree rearrangements or the full tiling move set. Parallel
tempering is not the default next step: the earlier Brazil experiment had no
accepted swaps, and the fixed-power diagnostics should first establish
whether the likelihood creates a barrier that tempering could plausibly
address.

## Environment and launch template

Use the committed branch revision and the repository's pixi environment on
BP1. Validate the NumPy, xarray, h5netcdf/netCDF4, and HDF5 stack before the
array launch; exact cross-version PCG64 replay is an environment-level claim,
not implied merely by restoring the stored state.

Pin numerical threads:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Before launch, require a clean checkout and record the actual revision:

```bash
git status --porcelain
git rev-parse HEAD
```

Define reviewed launch values:

```bash
FROZEN_INPUT=/group/.../paris_may_2014_gamma_beta_native.nc
FROZEN_INPUT_ID=paris-may-2014-gamma-beta-native-v1
FROZEN_INPUT_SHA=<64-character-reviewed-sha256>
OUTER_LABELS=<six-reviewed-comma-separated-labels>
CODE_REVISION=<clean-git-rev-parse-HEAD>
WEIGHT_POLICY=<reviewed-positive-weight-policy-id>
```

The first low-\(K\) dry run is:

```bash
pixi run python examples/rjmcmc/gamma_beta_native_smoke.py \
  --input "$FROZEN_INPUT" \
  --input-id "$FROZEN_INPUT_ID" \
  --expected-input-sha256 "$FROZEN_INPUT_SHA" \
  --output-directory /group/.../dry-chain-low \
  --cycles 100 --k-min 5 --k-max 500 --start-k 50 \
  --concentration 2 --root-variance 0.25 --likelihood-power 1 \
  --fixed-prior-mean 1 --fixed-prior-sd 1 --fixed-proposal-sd 0.4 \
  --warmup 280 --thin 14 --seed 812 --chain-id paris-gb-low-0 \
  --code-revision "$CODE_REVISION" \
  --nominal-weight-policy "$WEIGHT_POLICY" \
  --expected-outer-labels "$OUTER_LABELS" \
  --input-netcdf-engine h5netcdf --netcdf-engine h5netcdf \
  --require-paris-profile --dry-run
```

Repeat with a new unused dry-run output path, `--start-k 250`, distinct seed,
and distinct chain ID. The fresh short sampling command is identical after
removing `--dry-run` and selecting a new immutable segment directory.

A continuation uses the same logical input ID, SHA, initial \(K\), seed,
chain ID, code revision, priors, schedule, retention, variable mapping, and
backends, plus:

```bash
  --resume-checkpoint /group/.../previous-segment/checkpoint.npz
```

The logical `--input-id` is stable across scratch restaging; the absolute path
is recorded only as run summary provenance, while the file SHA provides
content identity. The driver preflights the chosen NetCDF writer before
sampling, records input/problem/sampling timing in `summary.json`, and records
per-artifact output timing and hashes in `complete.json`. CPU time and MaxRSS
remain Slurm/external measurements.

The sampling command must use a new per-chain/per-segment output directory.
One Slurm array task owns one chain; immutable segment numbering belongs in
that directory name. Do not launch until `git status --porcelain` is empty and
`--code-revision` equals the recorded `git rev-parse HEAD`.

## Remaining external dependency

The implementation can be tested locally without production data. The actual
HPC launch still depends on creating and versioning the frozen native PARIS
Dataset, recording its construction command and expected SHA, documenting the
1,382-row source/filter mapping, reviewing the exact six-column outer-region
order, and choosing a strictly positive `nominal_weight` policy. Those are
deliberate scientific inputs, not code defaults.
