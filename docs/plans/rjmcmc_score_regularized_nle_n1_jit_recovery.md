# Score-Regularized NLE N1 CPU-JIT Recovery

## What Was Tested

This diagnostic records the score-regularized neural likelihood CPU
compilation recoveries on BP1.  The development launch uses the six public
two-cell/four-cell tiny root-model cases, training size \(S=4096\), base seed
731, two frozen flow initializations, and the exact-oracle likelihood,
posterior, evidence, and mass-gradient checks declared in
`rjmcmc_score_regularized_nle_bp1_plan.md`.

The first source revision was
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

The first N0 wakeup ticket was
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
| `18214645_[0-5]` | 8 GB, 1 h | 9:56–14:49 | 5.8–6.1 GiB | all six failed, exit 139 |

The first array ran across `bp1-compute067`, `070`, `097`, `102`, and `122`;
the retry ran across `bp1-compute067`, `088`, `089`, and `097`.  The
cross-node recurrence and unchanged memory plateau rule out one bad node and
make a simple 8 GB cgroup limit an inadequate explanation.

Every diagnostic log reported an LLVM/XLA executable-section allocation
failure, including `LLVM compilation error: Cannot allocate memory` or
`LLVM ERROR: Unable to allocate section memory`.  No `.score-flow`, task
report, or task completion marker was published in `development/`.  The
twelve failure logs remain create-only in `logs/development/`.

The targeted serialized-codegen revision was
`3eae1bd49d7152b82f78957cdf4db3771e4c819c`.  Its detached source and
fresh run root were:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/source-3eae1bd49d7152b82f78957cdf4db3771e4c819c
/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/run-3eae1bd49d7152b82f78957cdf4db3771e4c819c
```

N0 job `18214635` passed in 6 minutes 16 seconds.  Its wakeup ticket was
`sw-20260729T214851Z-a04f6eb60e38`, with callback job `18214636`.  The
complete recovered \(S=4096\) array was job `18214645`; its wakeup ticket was
`sw-20260729T215836Z-dd0aa797b280`, with callback job `18214646`.  All six
tasks failed with the same LLVM allocation signature across
`bp1-compute058`, `067`, `097`, and `122`.  Serializing XLA's per-module
codegen split therefore did not resolve the compiler failure.  No scientific
task bundle was published.

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

The serialized-codegen recovery evidence has these independent SHA-256
identities:

| Evidence | SHA-256 |
|---|---|
| `preflight/N0_report.json` | `5c089a391a57f5a3ff5f7431ac810319cee4f9b205701d614a09e9fbc5d6f3bf` |
| `preflight/N0_COMPLETE.json` | `df6548b95e5ce9651bca80e2682e12e23dd78f5ae936b6ee3fa43e5691d33a01` |
| `near_gaussian__two_cell__root...18214645_0.log` | `fc0d240ea93082623b9d334c3f588cc8159d63854b02c5dcd97335a08ceec291` |
| `near_gaussian__four_cell__root...18214645_1.log` | `903982effa04ad8854d29693d24aaf991a3b4567d9a7240dafff5f797e84ecbe` |
| `skewed__two_cell__root...18214645_2.log` | `ca783cfe52ba87ce3bcba32183653022b3a8738db383f31a896bf20f74be03aa` |
| `skewed__four_cell__root...18214645_3.log` | `e428050b52c252ec8a57b74858648b7f3214ecf8762397d498be03fa23a3ff16` |
| `boundary_heavy__two_cell__root...18214645_4.log` | `01eb5b85d4175de7408ec6a0b495fc0a0aa9e78e42749202896c78eac06c1237` |
| `boundary_heavy__four_cell__root...18214645_5.log` | `666e2b82a15e8c394c82dcbf2a1cf32da73e739df008e9fcf569c24d672fd0fd` |

## Interpretation

This is a technical compilation failure, not evidence for or against the
score-regularized flow approximation.  Increasing requested memory from
8 GB to 16 GB did not raise the observed process-memory plateau or move the
failure later, so a further blind memory escalation is not justified.

The installed JAX and jaxlib versions are both 0.6.2.  Setting
`xla_cpu_parallel_codegen_split_count=1` was a useful falsification test, but
the recovered array shows that per-module codegen splitting was not the
controlling cause.

The remaining common compile graph is the mixed derivative in the score
loss.  The original implementation formed
`vmap(grad(log q, tau))` inside the outer parameter gradient.  This is
reverse-over-reverse automatic differentiation through all eight spline-flow
layers.  Because raw log mass \(\tau\) is scalar, the exact same score is the
forward-mode identity

\[
\operatorname{JVP}_{\tau}\!\left[\log q_\theta(x\mid\tau);1\right]
=\partial_\tau\log q_\theta(x\mid\tau).
\]

Taking the outer reverse-mode parameter gradient then gives the same mixed
derivative \(\partial_\theta\partial_\tau\log q_\theta\).  The JVP also
returns the primal log density, so the likelihood and score terms can share
one flow evaluation.  This changes the differentiation schedule and compiler
graph, not the density, objective, model, simulated data, or thresholds.

## Follow-Up

The next recovery replaces only that inner scalar reverse derivative with
the JVP identity above and records
`forward-jvp-in-raw-log-mass-then-reverse-parameter-gradient` in the protocol.
Focused tests must compare its score to the prior direct reverse derivative
and its composite parameter gradient to the independent analytic reference.

A fresh commit, detached full-SHA source, run root, and N0 are required.
Before another complete N1 array, a committed two-task Slurm canary must
compile and execute one exact 64-row composite gradient for \(q=1\) and
\(q=3\), covering the masked-autoregressive and coupling branches.  It uses
ordinary shared nodes, 8 GB per task, and a 30-minute limit.  It evaluates no
scientific threshold.  Only if both canaries pass may the complete six-case
\(S=4096\) array run; no larger N1 size may run until that tier publishes six
authenticated task bundles.
