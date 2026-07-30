# Corrected score-regularized marginal NLE development result

## What was tested

This experiment asked whether one frozen learned conditional-density
procedure could approximate the marginal likelihood of a common native Gamma
model after each of six public fixed linear projections. It used no realised
PARIS mole fractions, protected catalogue, or file in `PARIS_inversions`.

The frozen candidate was `fisher_observation_joint`. It combined likelihood
training with Fisher-scaled observation-score supervision. For every case and
catalogue size, four fixed initialisations were trained and the initialisation
with the smallest independent model-selection negative log likelihood per
retained dimension was selected. Reporting data and exact-oracle quantities
were evaluated only after that selection. `nll_only` used the same four-start
rule as a development comparator; it was not a fallback candidate.

The six public cases crossed near-Gaussian, skewed, and boundary-heavy native
regimes with two-cell and four-cell projections. Development used
`S=4096` and `S=16384` simulator catalogues with base seed `1731`.

Exact identities were:

- attempt and oracle producer:
  `3ef17c2253d5b56eda6ee5f028d704857a4e0d4b`;
- per-case evaluation and cross-size code:
  `bf94e055854c285fe5cbf8176dab263c725e1886`;
- run root:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle_corrected/run-3ef17c2253d5b56eda6ee5f028d704857a4e0d4b`.

## Terminology and truth

“Truth” here means the certified numerical conditional likelihood, evidence,
and posterior of the declared public synthetic native Gamma model under each
fixed projection. It is a model truth, not atmospheric ground truth and not a
comparison with observed methane.

The learned likelihood is the density implied by the selected neural flow
after the fixed projection. Log-likelihood and evidence errors are in natural
log units (nat). The retained-mass gradient error is dimensionless. Evidence
differences between approximations are leakage diagnostics only; they were
never used as structural or basis weights.

The principal individual gates were:

- prior-weighted median absolute log-likelihood error no greater than
  `0.05` nat;
- exact-posterior-weighted p99 absolute log-likelihood error no greater than
  `0.20` nat;
- scaled retained-mass gradient error no greater than `0.05`;
- absolute log-evidence error no greater than `0.05` nat;
- posterior mean, SD, and interval endpoint tolerances;
- finite, normalised, replayable evaluation over the complete metric grid.

## What happened

The corrected six-case oracle passed before fitting. Both 48-task development
arrays completed. The original matrix evaluator accumulated process-global
XLA/LLVM code and exhausted memory without publishing numerical summaries.
A source-controlled recovery kept all selections and scientific calculations
unchanged but evaluated one case per fresh spawned process. This succeeded at
3 GiB for `S=4096` and at 5 GiB for `S=16384`.

The recovered numerical result is a development failure:

- no selected `S=4096` candidate passed all individual gates;
- two of six selected `S=16384` candidates passed all individual gates;
- the candidate beat the NLL-only observation-score comparator by the frozen
  five-pooled-MCSE rule in only three of six cases at either size;
- only three of six cases passed the cross-size likelihood-stability gates.

The development certificate was therefore ineligible for confirmation. Its
job exited nonzero after publishing the expected failing numerical
certificate. No confirmation array or protected-data computation ran.

## Key results

The table reports the selected candidate at each catalogue size. Median,
p99, and evidence columns are absolute errors in nat; gradient is the scaled
retained-mass gradient error. “Pass” means every individual likelihood,
gradient, evidence, posterior, finiteness, normalisation, and replay gate
passed.

| Case | S | Init | Median (nat) | p99 (nat) | Gradient | Evidence (nat) | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| near-Gaussian, two cell | 4,096 | 1 | 0.0104 | 0.0410 | 0.246 | 0.00966 | no |
| near-Gaussian, two cell | 16,384 | 3 | 0.0241 | 0.0371 | 0.0127 | 0.0226 | yes |
| near-Gaussian, four cell | 4,096 | 2 | 0.0138 | 0.199 | 0.0596 | 0.0213 | no |
| near-Gaussian, four cell | 16,384 | 2 | 0.0168 | 0.0760 | 1.38 | 0.0120 | no |
| skewed, two cell | 4,096 | 0 | 0.0536 | 0.152 | 0.00121 | 0.0382 | no |
| skewed, two cell | 16,384 | 0 | 0.0259 | 0.153 | 0.0370 | 0.00931 | yes |
| skewed, four cell | 4,096 | 2 | 0.0523 | 0.194 | 0.107 | 0.0575 | no |
| skewed, four cell | 16,384 | 2 | 0.0777 | 0.129 | 0.162 | 0.0698 | no |
| boundary-heavy, two cell | 4,096 | 2 | 0.929 | 0.469 | 10.3 | 0.252 | no |
| boundary-heavy, two cell | 16,384 | 2 | 0.637 | 0.379 | 1.55 | 0.00229 | no |
| boundary-heavy, four cell | 4,096 | 1 | 1.13 | 0.731 | 5.04 | 0.371 | no |
| boundary-heavy, four cell | 16,384 | 2 | 0.761 | 0.313 | 2.70 | 0.0977 | no |

Increasing the catalogue size clearly helped some cases but did not produce a
uniform convergence pattern. In particular, the four-cell near-Gaussian
likelihood remained accurate while its selected gradient error increased
from `0.0596` to `1.38`.

The cross-size table compares the selected `S=4096` and `S=16384` learned log
likelihoods on the common exact grid. Median and p99 differences are in nat.
Evidence difference is reported as a leakage diagnostic and is not a gate or
weight in model construction.

| Case | Median difference (nat) | p99 difference (nat) | Evidence difference (nat) | Cross-size pass |
|---|---:|---:|---:|:---:|
| near-Gaussian, two cell | 0.0201 | 0.0371 | 0.0129 | yes |
| near-Gaussian, four cell | 0.0209 | 0.154 | 0.00928 | yes |
| skewed, two cell | 0.0483 | 0.0910 | 0.0475 | yes |
| skewed, four cell | 0.126 | 0.271 | 0.127 | no |
| boundary-heavy, two cell | 0.331 | 0.567 | 0.254 | no |
| boundary-heavy, four cell | 0.949 | 1.04 | 0.468 | no |

The comparator criterion passed for the two near-Gaussian cases and the
skewed four-cell case at both sizes. It failed for both boundary-heavy cases
and the skewed two-cell case at both sizes. The boundary-heavy two-cell score
candidate had worse observation-score risk than NLL-only on both held-out
domains at both sizes.

## Exploratory context

The preceding public-oracle loop tested:

- likelihood-only NLL;
- jointly scaled partial-score training;
- NLL pretraining followed by score fine-tuning/curriculum variants;
- observation-score supervision;
- small-catalogue overfit, `S=4096`, and `S=16384` regimes;
- four initialisations for optimisation diagnosis.

The full chronological attempt inventory, loss-scale diagnostics, transient
roundoff recovery, jobs, and E2 numerical ranges are recorded in
`rjmcmc_score_regularized_nle_corrected_exploration_log.md`. NLL-only was
usually more stable, while the observation-score candidate was credible
enough in the public canary to justify the frozen all-six test. The all-six
result shows that this credibility did not extend to the boundary-heavy
regimes.

## Computational evidence

The table distinguishes scientific jobs from preserved execution failures.
Requested memory and peak resident memory refer to each batch step.

| Job | Role | Request | Peak RSS | Elapsed | Outcome |
|---:|---|---:|---:|---:|---|
| 18218317 | six-case oracle | 2 GiB | 304,304 KiB | 7m33s | passed |
| 18218592 | `S=4096`, 48-task array | 3 GiB/task | recorded per task | 2m35s–5m05s | 48/48 complete |
| 18218594 | `S=16384`, 48-task array | 5 GiB/task | recorded per task | 3m31s–13m15s | 48/48 complete |
| 18218822 | original `S=4096` evaluator | 3 GiB | 3,144,528 KiB | 5m08s | technical OOM |
| 18218991 | original `S=4096` evaluator | 5 GiB | 5,241,660 KiB | 8m25s | technical OOM |
| 18219047 | original `S=4096` evaluator | 8 GiB | 5,742,244 KiB | 9m26s | LLVM allocation failure |
| 18219026 | original `S=16384` evaluator | 5 GiB | 5,241,688 KiB | 7m05s | technical OOM |
| 18219152 | per-case `S=4096` evaluator | 3 GiB | 1,772,736 KiB | 11m08s | numerical summary published |
| 18219258 | per-case `S=16384` evaluator | 3 GiB | 3,144,532 KiB | 4m56s | technical OOM |
| 18219293 | per-case `S=16384` evaluator | 5 GiB | 3,653,020 KiB | 26m16s | numerical summary published |
| 18219435 | per-case cross-size certifier | 3 GiB | 663,152 KiB | 2m21s | failing certificate published |

Scheduler stdout and stderr for the recovery phase were routed to the run
root. All technical failures and numerical artifacts remain preserved.

## Artifact identities

- `S=4096` summary payload:
  `a12583367743435fbef64f10202dca2ccf867a68a802dc17fcdce2efc895acfc`;
- `S=4096` exact summary file:
  `01249349b876d96188e751c1b6f0c4ca8c8060184c781738f8abb4761a2a5edb`;
- `S=16384` summary payload:
  `ee504aca4979b7bf7e17d63097fb039cc0fc88363d120936ab84b960052c2be9`;
- `S=16384` exact summary file:
  `0b4d62beaa9fefc6c6b3d4143331b8288549e303dd213674a3789fadc1c916f4`;
- development certificate payload:
  `3affb2a5ded446f01594998e3b388519cf14f56cb842791eedc509e897bacfcd`;
- development certificate exact file:
  `8bd9cce9c2287f0e1672f4f4dd7d66d3a2f01922405e9115e6b729c47fe5dd66`.

The compact machine-readable counterpart is
`rjmcmc_score_regularized_nle_corrected_promotion_result.json`.

## Interpretation

On the six public synthetic two-cell and four-cell root projections of the
common native Gamma model, the corrected frozen
`fisher_observation_joint` candidate—with fixed flow/training controls, four
predeclared starts selected solely by independent model-selection NLL, seed
`1731`, and `S=4096,16384`—failed the predeclared development promotion gate
and is not eligible for confirmation. More simulator draws rescue two cases
but do not solve the boundary-heavy likelihood or derivative errors, do not
stabilise the skewed four-cell likelihood, and do not make the
observation-score objective superior to NLL-only in all six regimes.

The result is stronger than a single optimisation miss because it includes
four fixed starts, independent start selection, two catalogue sizes, an
NLL-only comparator, exact public oracles, and cross-size evaluation. It is
not a claim that all neural likelihood estimation is impossible. It does not
test every support-aware transform, mixture architecture, loss schedule, or
sample regime. It does not test lognormal or other native-prior alternatives,
and it says nothing about realised PARIS observations.

The most specific evidence-backed limitation is that this frozen flow and
training objective do not learn sufficiently accurate boundary-sensitive
likelihoods and retained-mass derivatives under the common native Gamma
model. Approximate evidence disagreement corroborates leakage but does not
identify a scientifically meaningful basis weight.

## Follow-up

No confirmation or protected-data run is authorised for this candidate.
Further public synthetic work would require a newly predeclared candidate
whose design directly addresses support boundaries and derivative
calibration. It should reuse the corrected simulator and certified oracle,
retain observation-blind selection, and compare against NLL-only. The present
result does not justify tuning thresholds, choosing projections from
approximate evidence, or treating catalogue size as structural information.

The final independent scientific-interpretation review returned PASS on this
exact scope. It separately confirmed that the earlier OOM/LLVM jobs were
technical execution failures, while the recovered complete summaries and
cross-size likelihood checks constitute the genuine numerical rejection.
