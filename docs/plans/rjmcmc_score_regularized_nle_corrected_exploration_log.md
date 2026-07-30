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
