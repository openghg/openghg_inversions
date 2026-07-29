# Score-Regularized NLE N1 CPU-JIT Recovery

## What Was Tested

This diagnostic records the first score-regularized neural likelihood
development launch on BP1.  It used the six public two-cell/four-cell tiny
root-model cases, training size \(S=4096\), base seed 731, two frozen flow
initializations, and the exact-oracle likelihood, posterior, evidence, and
mass-gradient checks declared in
`rjmcmc_score_regularized_nle_bp1_plan.md`.

The source revision was
`d014b7b9f021bbd47aa88d5554fee8fa760b6e13` on
`codex/rjmcmc-score-regularized-nle`.  The detached source was
`/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/source-d014b7b9f021bbd47aa88d5554fee8fa760b6e13`
and the fresh run root was
`/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/run-d014b7b9f021bbd47aa88d5554fee8fa760b6e13`.

This was an observation-blind synthetic approximation experiment.  No
scientific truth, posterior accuracy, or approximation score was evaluated:
all N1 tasks stopped inside CPU compilation before publishing a fitted model.
The Gaussian complement remained a moment-closure approximation, not an
assertion that the native projected marginal is Gaussian.

## What Happened

N0 job `18214509` completed in 5 minutes 56 seconds and published an
authenticated passing preflight.  All six tasks in the first N1 array then
failed during XLA/LLVM JIT compilation.  A single resource-only retry doubled
memory and shortened the walltime ceiling; it failed at the same compiler
boundary.

The N0 wakeup ticket was
`sw-20260729T203409Z-df9da5b8bb01` with callback job `18214510`.  The first
N1 ticket was `sw-20260729T205538Z-ab6458c0d89d` with callback job
`18214542`; the retry ticket was
`sw-20260729T211710Z-bb26c3fdcb39` with callback job `18214572`.

The following table distinguishes requested resources from measured peak
resident memory.  Memory values are per task.

| Slurm job | Request | Elapsed range | Peak RSS range | Outcome |
|---|---:|---:|---:|---|
| `18214541_[0-5]` | 8 GB, 2 h | 9:04–16:15 | 6.0–6.2 GiB | all six failed, exits 134/139 |
| `18214571_[0-5]` | 16 GB, 1 h | 9:35–15:40 | 5.9–6.1 GiB | all six failed, exits 134/139 |

The first array ran across `bp1-compute067`, `070`, `097`, `102`, and `122`;
the retry ran across `bp1-compute067`, `088`, `089`, and `097`.  The
cross-node recurrence and unchanged memory plateau rule out one bad node and
make a simple 8 GB cgroup limit an inadequate explanation.

Every diagnostic log reported an LLVM/XLA executable-section allocation
failure, including `LLVM compilation error: Cannot allocate memory` or
`LLVM ERROR: Unable to allocate section memory`.  No `.score-flow`, task
report, or task completion marker was published in `development/`.  The
twelve failure logs remain create-only in `logs/development/`.

## Evidence Checksums

The following table authenticates the passing N0 report/marker and every
preserved N1 failure log.  SHA-256 values cover the complete file bytes.

| Evidence | SHA-256 |
|---|---|
| `preflight/N0_report.json` | `4e6511db306e53bb1d110c43161ec872bda463e7cca8a5cd2f7b5fc1e24f56f5` |
| `preflight/N0_COMPLETE.json` | `b0ee296f86d5bcfde8b12f75b840240331c46a5b7f913b281d9839a3a9ffc682` |
| `near_gaussian__two_cell__root...18214541_0.log` | `4870725186f7ad2373c3946fa81ac9c0cddac1d9892a26dbacb6978e379a6bd9` |
| `near_gaussian__four_cell__root...18214541_1.log` | `7c267c9dcb0e28c59fb452979b4b24b7a4d164043c6d9c250f1a8091a58f7b46` |
| `skewed__two_cell__root...18214541_2.log` | `dd5ab0329163434798454fa9d4c06e8eb386f08f5de1ab734f116f0b37d4f4bf` |
| `skewed__four_cell__root...18214541_3.log` | `eedd6b039b7199697db068056c65e290835b9addb9a2e6004fd247fd1525dcb8` |
| `boundary_heavy__two_cell__root...18214541_4.log` | `adee849d90e90a70b284dd0c3f822591e98fcc252529d76f94a31271183229cc` |
| `boundary_heavy__four_cell__root...18214541_5.log` | `5781251939559eeed13f4ee065e6080cd8bb0110bc12eeff94e513d016ffa430` |
| `near_gaussian__two_cell__root...18214571_0.log` | `e3fdf0880820b6bcb717e1556d9a001ae6554e11a9c17986b42ff92aeb81527c` |
| `near_gaussian__four_cell__root...18214571_1.log` | `d6cd7e939556900e6a9e9e194eb6633a82d28afb33b8f6a780ec0a81ca1843ee` |
| `skewed__two_cell__root...18214571_2.log` | `9606863cc6d299f307d727b7623c7f3466c34a5163ca75b2f8e1d6baf57def9e` |
| `skewed__four_cell__root...18214571_3.log` | `0401475a115dfde2938606f44d13fc02c12a170671722254cf67be249ad78d9e` |
| `boundary_heavy__two_cell__root...18214571_4.log` | `9643ae01508b2b3c6775dd5c1ef4b2ed5828638d9da1a323b4a0fd0bf2eb6d00` |
| `boundary_heavy__four_cell__root...18214571_5.log` | `a0862a7548cb9af8602518667aa7307eda81f32088690c70e5dfdfafc631c036` |

## Interpretation

This is a technical compilation failure, not evidence for or against the
score-regularized flow approximation.  Increasing requested memory from
8 GB to 16 GB did not raise the observed process-memory plateau or move the
failure later, so a further blind memory escalation is not justified.

The installed JAX and jaxlib versions are both 0.6.2.  Its local XLA binary
exposes `xla_cpu_parallel_codegen_split_count`; current upstream XLA source
sets the CPU split-count default to 32.  The repeated errors arise while LLVM
materializes parallel JIT sections, making a split count of one the smallest
targeted recovery.  The upstream setting is visible in
[OpenXLA `debug_options_flags.cc`](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc).

## Follow-Up

The only predeclared recovery change is to append
`--xla_cpu_parallel_codegen_split_count=1` to the existing single-thread CPU
`XLA_FLAGS`.  This exact string becomes part of the protocol identity.  A new
commit, detached full-SHA source, and fresh run root must repeat N0 before
repeating the complete six-case \(S=4096\) array.

The recovery may change compilation time and peak compiler memory only.  It
must not change the native model, simulated catalogues, flow architecture,
objective, optimizer, batch/microbatch, seeds, thresholds, or selection
rule.  No larger N1 size may run until the recovered \(S=4096\) tier
publishes six authenticated task bundles.
