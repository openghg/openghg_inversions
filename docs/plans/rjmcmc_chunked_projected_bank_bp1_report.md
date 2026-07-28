# BP1 chunked projected-bank G0--G3 execution report

## Decision

The committed BP1 engineering sequence passed G0, G1, G2, and G3 on scientific
source revision
`8b21d92195529332a65fbdd78f6b45f7cb95e56a`.  The frozen selection rules chose
projection microbatch \(P=256\) and sample chunk \(C=4096\).  All 12 full-bank
G3 candidates passed their time, memory, swap, internal-replay, and scheduler
gates, and all produced one common projected-array digest and one common binary
file digest.

This is the terminal engineering result.  G4 is barred because neither a
scientific common native Dirichlet total nor the required observation-blind
source-bank threshold supplement exists.  G5 and G6 were not run.

## What was tested

The target is the conditional pushforward of one common native
Gamma--Dirichlet root model through a fixed linear projection.  The leading
projected coordinates form a source bank for the non-Gaussian residual
marginal.  Spectrum directions beyond the stored rank remain an analytic
Gaussian moment-closure complement; that Gaussian complement is a diagnostic
approximation, not a different native model.

This is not an RJ likelihood, a comparison among partitions, or a mechanism
for data-dependent basis weighting.  Computational partitioning into Sobol
blocks, projection microbatches \(P\), and sample chunks \(C\) cannot change the
exact native-model limit.  Approximate differences are leakage diagnostics,
not structural information.

There is no truth field or scoring domain in this engineering report.  No
realized observation residual was used by a bank constructor.  The common
additive Dirichlet total was fixed a priori to the dimensionless
engineering-only value \(\eta=100\), selected as the higher-variance resource
stress case.  It is not an atmospheric mole-fraction concentration and is not
a scientific concentration lock.

## Reproducibility anchors

- Branch: `codex/rjmcmc-chunked-projected-bank`
- Initial chunked-bank implementation:
  `962d51a0fcea6f95e98766326d71249fcbb2e0b6`
- Scientific source revision:
  `8b21d92195529332a65fbdd78f6b45f7cb95e56a`
- Detached source worktree:
  `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc_chunked_projected_bank_8b21d92195529332a65fbdd78f6b45f7cb95e56a`
- Preserved run root:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/8b21d92195529332a65fbdd78f6b45f7cb95e56a`
- Frozen input:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc`
- Frozen input SHA-256:
  `24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044`
- Environment: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.2, xarray
  2025.6.1, using the frozen canonical BP1 Pixi environment
- Slurm account: `chem007981`

No file was written under `PARIS_inversions`.  The protected catalogue was not
accessed, and no production output was written.

## Preserved attempt history

The table records the software-only failures that led to the final source
revision.  Each run root remains intact; no failed artifact was reused by a
later revision.

| Source revision | Preserved run root | Slurm jobs | Result |
|---|---|---|---|
| `af6b2a3e02bcd63503700de862b1762485115179` | `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/af6b2a3e02bcd63503700de862b1762485115179` | none | G0 stopped on focused test typing; no completion marker. Preflight log SHA-256 `633e6290a261dfa907d22f9a7bdb5b8f37598e3fa89b0a7a36181cf69d05daee`. |
| `9fd067885c8c14e62970613fa0f8b99009ba43f0` | `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/9fd067885c8c14e62970613fa0f8b99009ba43f0` | `18192814` and callback `18192815` cancelled without execution; G1 retry `18194997` | G0 passed. The first submission inherited denied placeholder account `default`. The corrected-account G1 retry exposed an invalid cross-\(P\) bitwise-equality requirement and stopped without a G1 report or marker. Failure stderr SHA-256 `b103807eb16ff3797186b942a7d9e12e446e7a2284f3870985f77413c0be523b`. |
| `ddad33bf886fc94981e328806456ecab305b663b` | `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/ddad33bf886fc94981e328806456ecab305b663b` | G1 `18195017`; G2 `18195028`; G2 audit `18195039`; G3a `18195041` | G1 and G2 passed. G3a authenticated and loaded the frozen input and authoritative spectrum, then stopped before either bank construction because integer control-map keys made the otherwise valid G1 JSON noncanonical after parsing. No G3a report or marker was published. Failure stderr SHA-256 `02a9eae461db696d5782616346419e1cc46f1f1bb9d3926204ad8e82660bd521`. |
| `8b21d92195529332a65fbdd78f6b45f7cb95e56a` | `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/8b21d92195529332a65fbdd78f6b45f7cb95e56a` | listed below | Final G0--G3 pass. |

The two repairs changed only execution correctness: cross-\(P\) comparisons use
the already-frozen float64 tolerance while replay remains bitwise at fixed
\(P\), and projection-size control keys are strings before canonical JSON
serialization.  Neither repair changed the native model, scientific input, or
gate threshold.

## Gate results

### G0: preflight

G0 authenticated the exact source revision, clean detached worktree, canonical
environment link, and frozen input checksum.  It passed 77 focused
experimental tests, Ruff format and lint, focused Pyright, and the tiny v3
canonical-JSON and binary replay checks.

### G1: sequencing, replay, and projection microbatch

Job `18195050` passed 59 launcher tests and the committed matrices: 48 small
cases, two finite edge cases, and three forced-multiblock cases with Sobol block
dimensions `[21201, 1]`.  Every candidate replayed bitwise at fixed \(P\).

This table shows the three-repeat elapsed-time selection in seconds.  The
projected arrays for different \(P\) values were compared using the frozen
float64 leakage tolerance, not required to be bitwise identical.

| \(P\) | Median elapsed (s) | Projected-array SHA-256 |
|---:|---:|---|
| 64 | 6.179863249068148 | `8d3ea9c1f7094b9cab2ccd8eea8e0c4c988b0c6d9697b020531608b586eb23f4` |
| 128 | 6.019593656994402 | `3a841a85f0bb1c3d921eeee2fc4776219300a7f291ce0f3eea1ead6b99e05dad` |
| 256 | 6.017959009041078 | `3a841a85f0bb1c3d921eeee2fc4776219300a7f291ce0f3eea1ead6b99e05dad` |

The maximum cross-\(P\) absolute difference was
`3.5388358909926865e-16`, against tolerance
`1.4559020655724453e-11`.  The frozen rule therefore selected \(P=256\).
The job used zero swaps and a batch-step maximum resident set size (MaxRSS) of
1,109,256 KiB.

### G2: observation-blind spectrum

The authoritative spectrum job `18195069` ran on `bp1-compute062`.  It retained
rank 1,381 of a \(1382\times1382\) observation-space covariance, with total
variance `33395.782469203696` and retained fraction
`0.9999999998775141`.

The audit job `18195080` ran on the distinct node `bp1-compute067`.  Native
context identities were exact.  The maximum eigenvalue difference was zero
against tolerance `4.1319879823640187e-7`; the maximum reconstructed-covariance
difference was zero against tolerance `7.419375495698437e-8`.  The audit
bundle is explicitly non-authoritative.

### G3a: actual-input prefix parity

Job `18195083` used \(S=256\), \(q=32\), seed 731, and locked \(P=256\).
Consequently, \(C=256\) was the only legal member of the predeclared prefix
ladder.  The v2/v3 prefix comparison was bitwise identical, with zero absolute
and ULP difference.  Sobol block dimensions were `[21201, 2222]`, and the
Sobol catalogue SHA-256 was
`549815fc5f1a2053b518d6538d7da7df95fd7ea5a46214440d231fa46ee98315`.

### G3b: full source-bank resource matrix

The excluded warm-up used \(S=4096\), \(q=128\), \(P=256\), and \(C=1024\).
The authoritative matrix used \(S=65536\), \(q=128\), \(P=256\), seed 731,
three repeats per chunk, a 2,700 s constructor limit, and a 12 GiB Slurm MaxRSS
limit.  The table shows constructor time in seconds and full-job MaxRSS in
bytes; every row had three completed jobs, zero swaps, and passing internal
binary replay.

| \(C\) | Constructor times (s) | Median (s) | Maximum MaxRSS (bytes) | Gate |
|---:|---|---:|---:|---|
| 1024 | 1467.5925878769485, 1483.354827495059, 1523.279240531032 | 1483.354827495059 | 1,942,343,680 | pass |
| 2048 | 1462.398649691022, 1429.50457989797, 1441.1819722759537 | 1441.1819722759537 | 2,214,105,088 | pass |
| 4096 | 1386.9176466560457, 1552.0254355269717, 1391.6204763409914 | 1391.6204763409914 | 3,485,929,472 | pass |
| 8192 | 1420.3498018160462, 1454.845913887024, 1530.4900452260626 | 1454.845913887024 | 6,006,095,872 | pass |

All candidates shared projected-array SHA-256
`ba55924cac5fe20f593d25b19e921009df026563c103b4cb7a18b3935d4c9b03`
and binary-file SHA-256
`0d804453be433dcb613eff542e76175ba3e0a8442d31f802f77f2f207cde2d19`.
The frozen lowest-median rule selected \(C=4096\).  This selection tunes
temporary allocation only; it is not a data-dependent model or basis weight.

## Slurm record

This table maps each final-run stage to its scientific job IDs.  All listed
jobs completed with exit code `0:0`.

| Stage | Job IDs | Nodes | Elapsed |
|---|---|---|---|
| G1 | `18195050` | `bp1-compute064` | 00:04:26 |
| G2 authoritative | `18195069` | `bp1-compute062` | 00:00:21 |
| G2 audit | `18195080` | `bp1-compute067` | 00:00:37 |
| G3a | `18195083` | `bp1-compute071` | 00:00:51 |
| G3b excluded warm-up | `18195092` | `bp1-compute097` | 00:01:55 |
| G3b \(C=1024\) | `18195093`, `18195094`, `18195095` | `bp1-compute097`, `bp1-compute058`, `bp1-compute051` | 00:24:37, 00:25:23, 00:25:35 |
| G3b \(C=2048\) | `18195096`, `18195097`, `18195098` | `bp1-compute051`, `bp1-compute071`, `bp1-compute063` | 00:24:46, 00:24:03, 00:24:21 |
| G3b \(C=4096\) | `18195099`, `18195100`, `18195101` | `bp1-compute049`, `bp1-compute100`, `bp1-compute051` | 00:23:42, 00:26:30, 00:23:27 |
| G3b \(C=8192\) | `18195102`, `18195103`, `18195104` | `bp1-compute094`, `bp1-compute127`, `bp1-compute092` | 00:24:06, 00:24:58, 00:26:12 |
| G3 certifier | `18195105` | `bp1-compute092` | 00:00:08 |

Durable callback jobs were G1 `18195051`, G2 `18195070`, G2 audit `18195081`,
G3a `18195084`, and complete G3 chain `18195106`.  The complete-chain wake
ticket was `sw-20260728T064431Z-2480bff4e241`.

The G3 candidates in this completed run were submitted as a serial dependency
chain by the frozen submitter.  Updated operational guidance requires one
Slurm array for homogeneous repeated work.  Any future resource matrix must
use an array rather than an `sbatch` loop; no further jobs are authorized by
this report.

## Primary artifact checksums

The table records the decision-driving reports and completion markers.  The
completion markers were published after their reports; for G3, the decision
timestamp preceded the completion-marker timestamp.

| Artifact relative to the final run root | SHA-256 |
|---|---|
| `g0/preflight.log` | `4593bda84d31fba0f81e3f61bf6021681082cd022974bf90f8c35a783fd4cfaa` |
| `g0/tiny_report.json` | `b31b5fe560be0eb334ca878130ba4f99ab125a5b88e1633898ab39894182050e` |
| `g0/tiny_bank.npy` | `846286daa5299a157a725bbf94c241469b1e084bf1004cc0acbffea3f7e85fb6` |
| `g0/G0_COMPLETE.txt` | `8bfe70ebdddba40731c4e8c61b3c7ee9c0f741643bee10f7512c627c9a2e11c3` |
| `g1/g1_report.json` | `79663adac62381d8fa8e0ce437c324d63df98c036f378f80bd13ddb8bff52e16` |
| `g1/G1_COMPLETE.txt` | `9866b67346e2cdfd9a37c5271fed931ab79deaac7086086f878482a8f7b73ae5` |
| `g2/authoritative/spectrum_manifest.json` | `fb671cb1dae4505e4c97404e8430d007a30c73c7486161b0e8a09fc6c80a002c` |
| `g2/authoritative/G2_AUTHORITATIVE_COMPLETE.txt` | `5a058db2cefb9f91b917c1f205cde0ca4f5b0cde36e8ce610851ab21e961d167` |
| `g2/audit/spectrum_manifest.json` | `418044e456183add4f5eead5ee5af8f2f565786677b09e683972f2de0d8a2b99` |
| `g2/audit/g2_audit_report.json` | `6a5ba81d523906f36504090e7fdc7a8751070dfd44ac23e18533ea6ef0befa90` |
| `g2/audit/G2_AUDIT_COMPLETE.txt` | `7861c99ddfd790dd28c95ade80daced18d0b3b56811c54d1e8cac311092745a5` |
| `g3/prefix/g3a_report.json` | `d060484a742656d51939629af751cbf1cfd6628b601bf24d2504cceeca52652a` |
| `g3/prefix/G3A_COMPLETE.txt` | `ac39827af0e10ccc3a1f8b821d40084b2e1dce05e5c70ca486922c17bb0e7e26` |
| `g3/warmup/warmup_report.json` | `c7ab250cb19c1f4370b5fa03e3b5105d5b8635eeba76f7d6452fe0c0feed02f4` |
| `g3/warmup/WARMUP_COMPLETE.txt` | `c5610a7a8fc686d4add1ac584d8c6707e0f7748d5e492f53c27908d9b8e70b01` |
| `g3/G3_SUBMISSION.txt` | `af6b9dc6b57e1b5cfe8292d491d3b30c848672322c9c2433561f5692e94e715c` |
| `g3/g3_decision.json` | `54c8489353f9097f6e1e9859eb527b090320411cb1af1b20f693f896b7325eba` |
| `g3/G3_COMPLETE.txt` | `7d7d949ccea2817a6e79c0507728111625765490b5c3b7c349073bcb0487a705` |

The G3 decision additionally authenticates all 12 candidate manifests.  Each
candidate directory contains its manifest, 67,108,992-byte projected array,
resource report, scheduler logs, and completion marker.  Exactly 12 of each
manifest, projected array, and candidate completion marker are present.

## Interpretation and stop

The experiment establishes the engineering claim: the single-root chunked
constructor can reproduce the fixed projected marginal approximation
independently of allocation chunk, while bounding temporary memory and
retaining a deterministic replay identity.  It does not establish scientific
source-bank adequacy, likelihood accuracy, posterior adequacy, or a preferred
native Dirichlet total.

No extra rank, samples, chunk sizes, mixture components, flow, NLE, conditional
row model, `sbi`, PyMC, clustering, or posterior screen may be selected from
these results.  Continuing to G4 requires both prerequisites to be committed
independently of protected-catalogue results:

1. a scientifically justified common native \(\eta\); and
2. an observation-blind G4 source-bank threshold supplement.

Until then, the G0--G3 artifacts are sealed as a successful engineering
certificate, and the protected catalogue remains unopened.
