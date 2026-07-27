# BP1 report: exact aggregation mixture compression

## Disposition

The G0/G1/G2 development protocol passed at:

```text
candidate: d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea
branch: codex/rjmcmc-exact-mixture-compression
run root:
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_compression/d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea
```

The common direct-mixture source size is \(S=65{,}536\).  The common
compression suffix begins at \(M=256\): 256, 512, and 1,024 components passed
every inherited scientific gate in all six tiny root cases.

This is a development pass for the root-only approximation.  It is not a
confirmation-scramble pass, a PARIS result, a multi-region result, a PyMC
integration, or authority to infer partitions or \(K\).

Nothing was written to `PARIS_inversions`, and no confirmation/protected
catalogue was opened.

## Runtime identity and validation

The detached worktree used:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_candidate_d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea
```

The pinned environment reported:

```text
Python 3.10.20
NumPy 2.2.6
SciPy 1.15.2
Pixi 0.69.0
```

G0 passed:

- 43 focused experimental tests;
- Ruff formatting and checks;
- focused Pyright;
- source-stage smoke;
- byte-identical timing-free smoke replay; and
- the clean full-SHA worktree and canonical `.pixi` checks.

## G1: direct finite-mixture source

Slurm array `18188801` completed all six tasks with exit code zero in 23--38
seconds.  Peak batch-step RSS was 281--465 MiB.

All six cases passed at all three source sizes:

| Source size | Joint all-case pass |
|---:|:---:|
| 65,536 | yes |
| 262,144 | yes |
| 1,048,576 | yes |

The authenticated merger therefore selected \(S=65{,}536\).  The source-lock
internal digest is:

```text
80d631763473f7a9262ef0314b4efcab73e44a1d85697bc9aee243a4e3f41ead
```

The raw canonical lock digest is:

```text
61fa3ff7bd2aee439b532a8c70633df05cc3cc452c6a44aff377ac4ad613fa9a
```

At the selected size:

- retained exact residual rank was one for two-cell cases and three for
  four-cell cases;
- the largest absolute log-evidence error was \(1.67\times10^{-5}\) nat;
- the largest posterior-SD-relative error was \(1.21\times10^{-3}\);
- source covariance relative Frobenius error was at most
  \(8.64\times10^{-5}\);
- source artifacts occupied about 0.50 MiB at rank one and 1.50 MiB at rank
  three; and
- direct-bank likelihood throughput was 162--675 scored grid states/s.

The two higher source sizes also passed.  Their purpose was to establish the
common suffix, not to select the smallest point in isolation.

## G2: moment-preserving compression

Slurm array `18188807` completed all six tasks with exit code zero in
10:19--17:24.  Peak batch-step RSS was 696--724 MiB.

The common all-case decision was:

| Components | Joint all-case pass |
|---:|:---:|
| 16 | no |
| 32 | no |
| 64 | yes |
| 128 | no |
| 256 | yes |
| 512 | yes |
| 1,024 | yes |

The predeclared two-point suffix rule selected \(M=256\).  The raw canonical
compression-decision digest is:

```text
932efada6a7ad1894696c40fe78c515f2da661cfa780ddb7c659304c370bdf5e
```

The non-monotone low-\(M\) result is entirely the boundary-heavy four-cell
case.  Its posterior-SD-relative errors were:

| Components | Error | Threshold | Pass |
|---:|---:|---:|:---:|
| 64 | 0.00450 | 0.02 | yes |
| 128 | 0.02194 | 0.02 | no |
| 256 | 0.01311 | 0.02 | yes |
| 512 | 0.000934 | 0.02 | yes |
| 1,024 | 0.00133 | 0.02 | yes |

For the same case, the integrated compression KL upper bound decreased
monotonically from 0.353 at 64 components to 0.0397 at 1,024.  Thus the
isolated 128-component threshold crossing is finite local approximation
variation, not a failure of normalization or moment closure.

At the selected \(M=256\):

- the largest exact log-evidence error was \(6.79\times10^{-4}\) nat;
- the largest posterior-SD-relative error was 0.01311;
- the largest incremental compressed-versus-source log-evidence difference
  was \(6.62\times10^{-4}\) nat;
- the largest compression KL upper bound was 0.1187;
- source and compressed means agreed within \(1.44\times10^{-15}\);
- source and compressed covariances agreed within \(2.14\times10^{-14}\);
- the compressed numerical arrays occupied about 8.1 KiB at rank one and
  28.2 KiB at rank three, 1.58--1.84% of the selected source-bank storage;
  and
- offline \(M=256\) construction took 32.6--57.7 seconds.

## Evaluation-performance finding and follow-up

The scientific pass exposed an implementation bottleneck.  At \(M=256\), the
original compressed evaluator scored only 134--169 grid states/s, whereas
the vectorized \(S=65{,}536\) direct bank scored 162--675 states/s on these
tiny cases.  The compressed implementation looped over components and
performed one Cholesky factorization per component and state.

The next local source update caches each component covariance
eigendecomposition and evaluates all components with batched NumPy
operations.  Focused likelihood-parity and reconstruction tests pass.  A
bounded rank-three benchmark with \(S=16{,}384\), \(M=128\) measured about
19,104 compressed states/s versus 1,145 direct-bank states/s, a 16.7-fold
speedup.  This optimization was not part of the `d23e9d9` BP1 candidate and
requires its own committed confirmation preflight.

## Evidence inventory

The 46-file G0/G1/G2 evidence inventory verifies at:

```text
report/evidence-sha256sums.txt
inventory SHA-256:
06d96a0cac5b99231cbda00ec296b9cf0cd4223cf165649a5a9b20d8c45a4c2a
```

The inventory covers preflight logs and markers, all source and compression
artifacts, both merger decisions and raw-digest records, Slurm logs, and all
completion markers.

## Next gate

Freeze \(S=65{,}536\) and \(M=256\), then test all six cases at each untouched
source scramble:

```text
1,877
4,099
8,317
```

Keep clustering starts fixed at seed 731 so confirmation varies the
finite-bank scramble without retuning the compression algorithm.  All 18
shards must pass exact-vs-source and exact-vs-compressed gates.  The
confirmation protocol must authenticate the `d23e9d9` source lock and
compression decision, use the optimized evaluator only after value parity,
and publish no promotion if any shard fails.
