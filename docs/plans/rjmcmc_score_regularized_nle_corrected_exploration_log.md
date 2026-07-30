# Corrected score-regularized marginal NLE chronological log

## 2026-07-30T07:34:22Z — restart and evidence boundary

- Resolved `origin/codex/rjmcmc-score-regularized-nle` to
  `c4f674a17a587f9e2e89488b9c541c8f61667edd`.
- Resolved `origin/codex/rjmcmc-chunked-projected-bank` to
  `95483469fe8648fc65a3ae4de24930b6e4c386cb`.
- Created `codex/rjmcmc-score-nle-corrected-exploration` from the exact
  historical NLE SHA.
- Read the historical NLE plan and reports, the independent review at the
  projected-bank SHA, and the relevant learned-marginal-model derivations in
  the sibling `inversions-knowledge` repository.
- Confirmed the old run root remains unchanged and is not a source of
  corrected artifacts.
- Opened independent simulator, boundary-oracle, and code-design reviews.
- Froze the correction and exploration scope in
  `rjmcmc_score_regularized_nle_corrected_exploration_plan.md`.

No SLURM work has been launched yet. The next action is to implement and
locally verify E0 and E1 before committing the executable experiment.

## 2026-07-30T07:48:12Z — E0 and E1 corrected locally

E0 now uses three independently seeded PCG64 constructions:

- keyed PCG64 Dirichlet allocation;
- PCG64 uniform root totals transformed by the Gamma inverse CDF;
- PCG64 uniform Gaussian noise transformed by the normal inverse CDF.

The protocol, schema, evidence schema, 64-bit seed contract, generator and
draw-order identities, transforms, NumPy/SciPy versions, and latent uniforms
are authenticated under fresh v2 identities. The historical v1 run remains
unchanged.

The joint-law regression covers all six cases and all three public domains at
\(S=8192\). It checks raw cross-block correlations, latent/rank-uniform
lower-left quadrants, bounded pairwise Legendre cross-moments, and a bounded
three-way cross-moment. Exact replay, nested prefixes, public-domain
separation, global stream-seed uniqueness, and permutation invariance are
also covered.

E1 adds a public tiny-root oracle independent of executable legacy screens.
Its primary route uses endpoint-aware Gauss--Jacobi allocation quadrature
inside adaptive log-total integration. Its independent route integrates the
two native log masses directly. The corrected boundary-heavy two-cell
reference is:

- log evidence `-1.79490498759`;
- posterior total mean `0.90254144350`;
- posterior total SD `0.06508269572`;
- posterior total 2.5%, median, and 97.5%:
  `0.78466927377`, `0.89901102296`, `1.04346682078`.

Gauss--Jacobi orders 32 and 64 differ by about \(3.7\times10^{-11}\) nat.
The native-log-mass route at lower tail bound \(-120\) differs by about
\(6.2\times10^{-10}\) nat. The support envelope reports retained prior mass,
a conservative retained-posterior-mass lower bound, mode inclusion, and its
accounting method. A separate support audit refuses posterior-weighted
conditional renormalization if a subset omits more than \(10^{-6}\) posterior
mass or excludes the mode.

Focused validation:

- 60 corrected simulator/oracle tests passed;
- Ruff passed on all changed source and test files;
- focused Pyright passed with 0 errors.

Independent simulator, numerical-oracle, and code-design reviews informed
the implementation. No SLURM work or protected-data access occurred.

## 2026-07-30T08:36:59Z — E2 implementation and prelaunch review closure

The E2 driver, committed array mappings, merger, and Slurm wrappers are now
implemented locally. The implementation preserves every attempt in a fresh
create-only directory and publishes completion last. Completion markers bind
the canonical report payload, exact report-file bytes, artifact metadata, and
exact serialized artifact bytes.

Independent prelaunch reviews initially returned HOLD for concrete control
defects. Those findings were fixed before any batch submission:

- projected-coordinate observation scores now use a forward-coordinate
  linearization, so parameter gradients are reverse-over-forward rather than
  the previous reverse-over-reverse schedule;
- the scalar partial-score Fisher risk is no longer divided by retained rank;
- component row variances and parameter-gradient norms are measured at model
  initialization before auxiliary loss weights are applied;
- private initializer and optimizer streams are derived and checked against
  every simulator stream before model construction;
- paired optimizer randomness is keyed by stage position rather than
  candidate name;
- the vectorized scientific evaluator is checked against the public artifact
  evaluator on representative tail and central points;
- exact and learned scientific grid quantities are refined through 8192
  midpoint prior-CDF strata and are marked non-interpretable if the numerical
  convergence contract fails;
- the boundary certificate now gates primary posterior refinement,
  independent lower-tail refinement, and outer/inner quadrature errors;
- oracle and attempt completion markers bind exact file bytes and are the
  terminal publisher action;
- Slurm wrappers load the pinned Git module, preserve scheduler logs for
  pre-redirection failures, run on shared nodes, and end with the Python
  driver via `exec`.

The frozen E2 catalogue contains a four-task partial/observation \(q=1/q=3\)
compile canary, 16 overfit tasks, 36 \(S=4096\) tasks, eight optional
observation-score tasks, and three separate 12-task \(S=16384\) candidate
matrices. Larger matrices will be launched only when the preceding evidence
supports them.

Focused validation after integration:

- 110 corrected simulator, score-helper, oracle, driver, wrapper, and merger
  tests passed in 113.97 seconds;
- Ruff check and formatting passed;
- focused Pyright passed with 0 errors;
- Python and shell syntax checks passed.

No SLURM work or protected-data access occurred during these fixes. The next
step is independent final launch sign-off, followed by a coherent commit and
push before a fresh detached full-SHA run root is created.

## 2026-07-30T08:40:30Z — independent final launch sign-off

The independent simulator/execution, numerical-oracle, and E2
code/provenance reviewers each returned **PASS for commit and launch** after a
read-only review of the integrated working tree. Their initial findings,
dispositions, numerical evidence, and final conditions are preserved in the
three corrected review documents.

All reviewers require the same launch sequence: commit and push the complete
working tree, resolve the exact full origin SHA, create a fresh clean detached
worktree and run root, run the corrected oracle first, and submit the
four-task compile canary only after the oracle publishes a valid completion
marker.

## 2026-07-30T09:31:17Z — first corrected oracle and compile canary

The reviewed source was committed and pushed as
`c6c1156fe49d4a93c343ef77091818664b93df8f`. Its clean detached source and
fresh run root are:

- `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle_corrected/source-c6c1156fe49d4a93c343ef77091818664b93df8f`;
- `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle_corrected/run-c6c1156fe49d4a93c343ef77091818664b93df8f`.

The pinned `nle-dev` environment was copied into the detached source. No old
run artifact was copied or reused.

The corrected oracle ran as shared-node Slurm job `18215464`, registered by
slurm-wakeup ticket `sw-20260730T090316Z-e1db810ebd08`. It completed in
2 minutes 15 seconds with batch MaxRSS 218008 KiB. The authenticated bundle
passed all primary, independent, tail and support checks:

- bundle payload SHA-256
  `d6aa0980aff4094410f6de16043e97dfdb9e23f6c9227f08a3d564d72bcde944`;
- exact bundle-file SHA-256
  `f42e59b3e2f2551905225a14a57f03076e0bf8755f8f96012a947bbeb4c04c5a`;
- completion-file SHA-256
  `f41b8f926835e0f9f758183dfcd7c802b2b3782a52a52476101568147f1e8da2`;
- near-Gaussian, skewed and boundary-heavy log evidence
  `-0.6178332235028026`, `-0.11207819228000337`, and
  `-1.7949049875536425` nat;
- corresponding primary refinement differences about
  `2.4e-15`, `2.6e-9`, and `3.7e-11` nat;
- boundary independent-route agreement `6.2e-10` nat and independent-tail
  refinement `8.4e-7` nat.

The four homogeneous compile tasks then ran as one shared-node Slurm array,
job `18215490`, registered by ticket
`sw-20260730T091216Z-176f11668528`. The frozen mapping was:

- task 0: near-Gaussian \(q=1\), Fisher-scaled partial score;
- task 1: near-Gaussian \(q=1\), observation score;
- task 2: skewed \(q=3\), Fisher-scaled partial score;
- task 3: skewed \(q=3\), observation score.

All tasks trained for the requested one epoch and reached artifact replay in
2 minutes 20 seconds to 2 minutes 34 seconds. Batch MaxRSS ranged from
1608428 to 1738164 KiB. The 8 GiB request was therefore conservative; the
fresh rerun will request 3 GiB and 15 minutes.

Tasks 0 and 1 completed. Tasks 2 and 3 failed only because an exact
trained-object-versus-deserialized likelihood assertion saw last-bit
differences. Maximum absolute differences were `7.10542736e-15` and
`1.77635684e-15` nat. Artifact SHA, canonical bytes, metadata, spectrum and
fitted flow parameters were unchanged. This is classified as a software
replay-boundary failure, not a training, resource, scheduler or scientific
miss. Both failure directories and the intentionally incomplete merger
summary remain preserved under the run root.

The local correction makes the authenticated deserialized artifact
authoritative for validation, reporting, likelihood, evidence, posterior and
published artifact evaluation. It still requires bitwise identity of every
fitted floating leaf and spectrum array, exact bytes and SHA identity, and a
tight explicit trained-to-canonical roundoff diagnostic. Fresh-process replay
coverage now includes both the \(q=1\) autoregressive and \(q=3\) coupling
architectures. Independent review identified and closed the final gap in
which validation and reporting risks had initially still used the transient
trained flow.

Post-correction validation:

- the focused driver smoke passed and published completion last;
- canonical fresh-process replay passed for both \(q=1\) and \(q=3\);
- a broad adjacent run passed 144 tests; its only two failures were the
  deliberately historical N1 screen asserting its frozen v1 Sobol protocol
  hash and v1 evidence schema against the corrected v2 PCG64 domain module;
- Ruff and focused Pyright passed;
- independent code/provenance re-review returned PASS with no remaining
  concrete defect.

The two historical-screen mismatches are preserved as scope evidence. The
invalid N1 certifier and its old run were not modified or reinterpreted.

## 2026-07-30T10:22:40Z — canonical replay recovery and overfit evidence

The canonical-replay correction was committed and pushed at
`8f03c3174545a8ad73a956885520f7e856141a9e`. Its fresh detached source and run
root are:

- `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle_corrected/source-8f03c3174545a8ad73a956885520f7e856141a9e`;
- `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle_corrected/run-8f03c3174545a8ad73a956885520f7e856141a9e`.

Oracle job `18215891`, wakeup ticket
`sw-20260730T095033Z-038dc2e94201`, completed in 2 minutes 14 seconds with
1 GiB requested. The committed loader authenticated the completion marker,
exact file bytes, nested oracle identities, all three cases, and the
independent boundary certificate. Bundle payload, bundle-file, and
completion-file SHA-256 values are respectively:

- `74b154745802a829bce08c37aa08b2664edcf62f8c4d02ee9ac1df56d592ade7`;
- `5f9c12d3d2b93b8b2fb5f87ac58cc96cd72cb1c99d9dcd431d77712654cf2223`;
- `dcab7bf4dfa1ec764e504548aac93739c0799954f73bd32876b5b31507dfee33`.

Compile-canary array `18215983`, wakeup ticket
`sw-20260730T095857Z-9e5b1d997170`, completed all four tasks in 2 minutes
2 seconds to 3 minutes 4 seconds. Peak attempt RSS was 1.64--1.79 GiB under
the 3 GiB request. The committed merger authenticated all four attempts with
no failures. Its summary payload SHA-256 is
`0c3472c82dc19b02d66a95b7407423eceb4714e08890ddb2d90fc1cfe8096059`;
the exact summary-file SHA-256 is
`84dea59c7caa29ef571c913c51376e7805e91df69bfd9bf80f9f7150312f890b`.

Overfit array `18216104`, wakeup ticket
`sw-20260730T100815Z-843da52243e4`, ran all 16 tasks in 1 minute 40 seconds to
2 minutes 44 seconds. Fifteen published complete authenticated attempts.
Task 8, skewed \(q=3\) NLL-only initialization 0, finished training but hit a
software false stop in the non-authoritative trained-object versus canonical
replay diagnostic. One of three probe values differed by
`2.39808173e-14` nat, about 108 binary64 output ULPs. Exact serialized bytes
and SHA, every fitted floating leaf, the PyTree architecture, and every
spectrum array remained identical. The create-only incomplete merger
preserves payload SHA-256
`3f5a205205493d6c88ea0648c7e7fc2610104931603c4def626b221f37c8377a`;
its exact summary and incomplete-marker file SHA-256 values are
`dc4447e645a9663f204f40cad8d10b345aa737ff4ca575e0815c63bac7e56e3f`
and `be30551cf1f3151bf0b93cdaf33c0d51cdd5ad53ca97dbb03808489239bd5027`.

The 15 interpretable overfit rows show:

- NLL-only is much more stable than the jointly scaled partial-score loss;
- near-Gaussian NLL-only absolute evidence errors range from `0.00085` to
  `0.0557` nat across four initializations;
- skewed \(q=3\) NLL-only errors for the three published initializations
  remain around `0.389`--`0.441` nat, and their NLL curves are still improving
  at epoch 40;
- joint partial-score fitting is unstable and usually degrades NLL, evidence,
  and likelihood accuracy, particularly in the near-Gaussian case where the
  measured scaled score term is orders of magnitude larger than NLL.

Independent numerical and code reviews classify the task-8 difference as
layout-dependent roundoff, not artifact drift. The canonical deserialized
artifact remains authoritative. Exact tree, leaf, spectrum, byte, SHA, and
fresh-process canonical replay checks remain hard gates; the transient
comparison becomes a labelled non-gating diagnostic reporting absolute,
relative, and output-ULP differences against an advisory
\(256\epsilon\max(1,|a|,|b|)\) range.

Because skewed \(q=3\) NLL is still learning, a new committed four-task
`overfit_q3_extended` exploratory matrix extends only those four NLL fits to
160 epochs. This is a response to preserved public overfit evidence, not a
change to a promotion gate or scientific target.
