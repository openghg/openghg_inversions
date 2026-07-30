# Corrected score-regularized marginal NLE exploration on BP1

## Status and identities

This is a corrected exploratory restart. It is not a continuation of the
invalid N1 scientific result and it is not a certification campaign.

- corrected branch:
  `codex/rjmcmc-score-nle-corrected-exploration`;
- historical NLE source:
  `c4f674a17a587f9e2e89488b9c541c8f61667edd`;
- independent review source:
  `95483469fe8648fc65a3ae4de24930b6e4c386cb`;
- preserved invalid run:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/run-475d5db6026a8472fd3c44eac2e0d2369686c78b`.

The preserved run is read-only evidence. None of its catalogues, learned
artifacts, or conclusions may be reused as corrected scientific results.

## Scientific target

The target is the marginal conditional likelihood of one common native
Gamma--Dirichlet model after applying a fixed linear projection. Exact
evidence is invariant to the computational partition, retained rank, and
estimator architecture. Finite-approximation evidence differences are
leakage diagnostics only; they must never become data-dependent basis or
partition weights.

## Invalidating defects

The historical simulator restarted separate scrambled Sobol engines and then
paired their rows. That deterministic row pairing did not represent the
product law

\[
T \mathrel{\perp} \xi \mathrel{\perp} \epsilon .
\]

The historical single-scramble standard-error calculation was not a valid
Monte Carlo uncertainty estimate. In addition, the boundary-heavy two-cell
validation mask retained only about \(2.96\times10^{-7}\) of exact posterior
mass and omitted its mode, while the nominal 64-node evidence oracle was not
converged.

These defects invalidate the old scientific conclusion, not merely its
completion marker.

## E0: corrected simulation law

Use independent deterministic PCG64 streams for allocation, root total,
Gaussian noise, optimizer randomness, and flow initialization. Seeds remain
domain-, case-, and stream-keyed. Repeated construction must replay bitwise,
larger catalogues must preserve exact prefixes, public domains must remain
separated, and cell/observation permutations must preserve canonical
scientific results.

For every public domain and tiny case, jointly audit:

- pairwise correlations among \(T\), selected allocation residual
  coordinates, and Gaussian noise coordinates;
- lower-left quadrant occupancy of the independently generated latent
  root-total and Gaussian uniforms;
- selected centered cross-moments;
- exact replay, prefix identity, domain separation, and authenticated hashes.

The tests are designed to reject the historical artificial copula while
allowing ordinary finite-IID fluctuation.

## E1: trustworthy boundary reference

The boundary-heavy validation view must retain essentially all exact prior and
posterior quadrature mass, include the exact posterior mode, and report both
retained masses explicitly. It must not renormalize a negligible subset
without disclosing the omitted mass.

The boundary evidence and posterior summaries require two support-aware,
independent numerical routes:

1. endpoint-aware Gauss--Jacobi allocation quadrature with generalized
   Gauss--Laguerre total-mass quadrature and an explicit order ladder;
2. an independent adaptive or separately parameterized high-order
   integration route.

The working target is agreement within 0.005 nat for log evidence plus stable
posterior summaries. Failure to obtain a converged reference is a reported
oracle blocker, not an NLE failure.

Single-net scrambled-QMC standard errors are removed. IID replicate estimates
may use between-replicate Monte Carlo standard errors.

## E2: lightweight ablation experiment

Use only:

- near-Gaussian two-cell;
- skewed four-cell;
- corrected boundary-heavy two-cell;
- \(S\in\{4096,16384\}\);
- at least four independent initialization seeds.

Separate \(q=1\) and \(q=3\) architectures. First overfit one small fixed
catalogue. Then compare:

1. NLL only;
2. NLL plus variance/Fisher-scaled partial mass-score risk;
3. NLL pretraining followed by score fine-tuning or a short curriculum;
4. direct fixed-observation mass-score supervision, if its target is
   practical;
5. observation-score supervision for \(q=1\) and \(q=3\), if practical.

Before choosing weights, record empirical component loss scales and
variances. Preserve full decomposed train/validation histories, failure
risks, value/evidence/posterior/gradient diagnostics, normalization and
replay checks, runtime, maximum RSS, exact job IDs, and checksums.

Substantial work runs as SLURM arrays through `slurm-wakeup`. Resource
requests are based on measured canaries and should not reserve exclusive
nodes or excessive walltime without evidence.

## E3: possible promotion

Promotion is permitted only after E2 identifies a credible, replayable
candidate against converged references. A promotion plan must then freeze the
candidate, largest justified sample sizes, confirmation seeds, all-six-case
matrix, and thresholds before confirmation results are viewed.

PARIS, the protected catalogue, and G3-style target-rank science remain
forbidden until such a candidate passes. A target-rank canary may be used for
engineering only and may not open or inspect protected scientific inputs.

## Safeguards and completion

- Write nothing to `PARIS_inversions`.
- Do not inspect realized PARIS mole fractions or the protected catalogue.
- Do not turn approximation leakage into structural information.
- Do not broaden this track into the separate IID-versus-QMC source-bank
  question.
- Run focused experimental tests, Ruff, focused Pyright, and committed
  scripts only; do not run the full tox matrix.
- Publish completion markers only after their reports and artifacts.
- Preserve failed attempts and record their scientific or engineering cause.

The track ends with either a defensible frozen promotion candidate, exhaustion
of the predeclared ablations, or a concrete external blocker. The final
deliverables are this plan, a chronological log, machine-readable summaries,
independent reviews, an evidence-rich final report, and a handover status.
