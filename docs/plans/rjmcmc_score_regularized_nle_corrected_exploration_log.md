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
