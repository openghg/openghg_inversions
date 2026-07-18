# NumPy/Numba trans-dimensional MCMC rewrite

## Objective

Develop a testable NumPy reference implementation and a parity-checked Numba
implementation of the legacy ACRG trans-dimensional MCMC sampler. Integrate
the numerical engine with OpenGHG Inversions through explicit input and output
adapters rather than coupling it directly to PyMC.

The primary scientific target is now the Lunt et al. (2016) paper model. Its
model, equations, reproduction profiles, and known inconsistencies are recorded
in [tdmcmc_lunt2016_background.md](tdmcmc_lunt2016_background.md). Current RHIME
is retained as a distinct secondary integration profile.

The first slice deliberately covers only the spatial, single-sector,
uncorrelated Gaussian problem. Correlated errors, temporal partitions,
multiple trans-dimensional sectors, and parallel tempering remain follow-up
work.

## Working principles

- Treat the legacy Fortran as a behavioural comparator, not as an unquestioned
  mathematical specification.
- Keep proposal probabilities, target-density terms, and acceptance decisions
  explicit and independently testable.
- Use fixed-capacity arrays plus an active count, matching the efficient state
  representation already used by the Fortran implementation.
- Own randomness at the Python API boundary so tests can inject deterministic
  draws and runs can be reproduced and checkpointed.
- Keep xarray, PyMC, and ArviZ objects outside compiled numerical kernels.
- Establish NumPy/Numba parity before attempting incremental-update or
  parallel-chain optimisations.

## Initial architecture

### Numerical problem

`TransDimensionalProblem` holds immutable numerical inputs:

- observations and observation standard deviations;
- fine-grid sensitivity matrix;
- grid-cell coordinates;
- lower and upper coefficient bounds;
- minimum and maximum active-region counts.
- an explicit normalised prior over active-region counts;
- a uniform prior over canonical unordered nucleus subsets conditional on the
  active-region count;
- lognormal coefficient-prior moments.

### Numerical state

`TransDimensionalState` holds:

- active-region count;
- fixed-capacity nuclei and region coefficients;
- fine-grid region labels;
- aggregated sensitivity matrix;
- model prediction and residual;
- cached log likelihood and log prior where appropriate.

### Transitions

Each proposal returns a transition record containing the proposed state and
the separately reported target, forward-proposal, reverse-proposal, and
Jacobian contributions. Rejection must leave the input state unchanged.

The first implementation will prioritise correctness and complete recomputation
of labels, aggregated sensitivities, and residuals. Incremental cache updates
will be introduced only behind equivalence tests.

## Decisions

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-07-18 | Use NumPy as the reference implementation and Numba as the first accelerated backend. | The legacy state already uses fixed-capacity arrays, Numba is an existing dependency, and explicit mutable kernels match the proposal mechanics. |
| 2026-07-18 | Keep literal Voronoi RJMCMC separate from the dyadic product-space experiments. | They may share likelihood, proposal-accounting, and test interfaces, but their geometries and statistical models are not interchangeable. |
| 2026-07-18 | Start with a standalone experimental numerical package rather than modifying `RhimeSampler`. | Production RHIME currently assumes a fixed design matrix and a PyMC sampler; changing those boundaries is unnecessary for the first correctness slice. |
| 2026-07-18 | Do not require a Fortran compiler for the first implementation. | Deterministic mathematical tests and NumPy/Numba parity can proceed independently; Fortran comparison can be added when a compiler-backed environment is available. |
| 2026-07-18 | Represent nuclei as a sorted subset of flattened fine-grid indices. | Canonical ordering removes label multiplicity and makes the conditional partition prior explicit as `1 / comb(n_grid, k)`. |
| 2026-07-18 | Declare the prior over `k` in the numerical problem rather than relying on implicit cancellation. | Trans-dimensional normalising constants and count priors must remain visible to detailed-balance tests. |
| 2026-07-18 | Use a uniform global unoccupied-cell nucleus move in the first slice. | Its forward and reverse probabilities are exactly auditable. The legacy floored-Gaussian local move is deferred until its discrete boundary mass is represented explicitly. |
| 2026-07-18 | Initially repeat coefficient, birth, death, and global-move attempts on a fixed schedule (superseded below). | This preserved nominal attempt counts and boundary self-transitions, but finite enumeration later showed that separate one-way birth/death kernels are not individually invariant. |
| 2026-07-18 | Preserve the filtered `fp_x_flux` field as the initial RHIME integration seam. | Current prepared datasets retain the fine-grid contribution field, so the rewrite need not alter fixed-basis production preparation in its first integration experiment. |
| 2026-07-18 | Use Lunt et al. (2016) as the primary scientific specification and current RHIME as a separate integration profile. | The paper closely matches the legacy code, while RHIME has materially different priors and model-error structure. Keeping profiles distinct avoids an undocumented hybrid target. |
| 2026-07-18 | Reproduce the Sect. 4 pseudo-data model before the full Sect. 5 hierarchy. | The pseudo-data case isolates spatial RJMCMC with fixed independent 5 ppb error and can validate model selection before adding correlated error and fixed boundary blocks. |
| 2026-07-18 | Derive acceptance from normalized targets/proposals when a printed paper equation is inconsistent. | The paper appears to omit a lognormal `1/x` ratio in Eq. (31), prints a questionable determinant power in Eq. (33), and does not define discrete boundary handling for Gaussian nucleus moves. |
| 2026-07-18 | Treat the checked legacy deterministic RJMCMC scheduler as incorrect and replace separate birth/death steps with two 50/50 mixed structural steps per cycle. | All five inspected Fortran drivers execute separate modulo-scheduled one-way structural steps. A finite fixed-coefficient counterexample to that scheduler design is not posterior-invariant, while the equal-probability mixture is and retains the first rewrite's aggregate structural-attempt frequency. |
| 2026-07-18 | Add an emissions-only, test-data-backed pseudo-data benchmark before archived paper data are recovered. | The repository contains EDGAR7/UKGHG flux and one week of hourly TAC/MHD NAME footprints on the paper's native EUROPE grid. A 56-longitude by 48-latitude crop can exercise the real forward operator while fixed outer emissions are subtracted exactly and the known-corrupt boundary-condition files remain unopened. The 56 six-hour observations support prediction validation, not a claim of native-grid or posterior-`k` recovery. |
| 2026-07-18 | Prefer `dimension-up` and `dimension-down` in user-facing discussion of reversible dimension changes. | These names describe the actual change without implying a binary geometric split/merge. Literature quotations and current internal API identifiers remain unchanged until a separate compatibility-preserving rename is justified. |

## Validation gates

- [x] Voronoi labels agree with independently calculated toy examples.
- [x] Aggregated sensitivities equal direct column-wise sums.
- [x] Log-likelihood and prior terms agree with independent formulas.
- [x] Birth/death proposal pairs expose complete forward and reverse terms.
- [x] The first global move reports normalized symmetric probabilities; the
  corrected local discrete-Gaussian move reports separately normalized forward
  and reverse probabilities at finite-grid boundaries.
- [x] A tiny enumerable fixed-`k`, fixed-coefficient location target satisfies
  proposal normalization, rejection self-mass, edgewise detailed balance, and
  stationarity.
- [x] A seven-state mixed-`k` fixed-coefficient birth/death subkernel satisfies
  count-factor accounting, mixture boundary self-mass, detailed balance, and
  stationarity. General continuous auxiliary integration remains follow-up.
- [x] Forced birth/death pairs satisfy pointwise detailed balance.
- [x] NumPy and Numba kernels agree for deterministic and randomised states.
- [x] Fixed-seed sampling is reproducible.
- [x] A small synthetic lognormal inversion recovers expected structure.
- [x] Three seeded `8 x 8` checkerboard comparison runs improve noise-free
  prediction, spatial correlation, and high/low contrast for the declared
  oracle and adaptive variants.
- [x] A matched-proposal checkerboard comparison covers an oracle fixed layout,
  independent random fixed layouts, movable fixed `k=16`, and
  trans-dimensional `k=8..28` without asserting adaptive-method superiority.
- [x] A test-data-backed checkerboard uses raw EDGAR/NAME inputs, verifies the
  fixed outer-emissions accounting independently of boundary conditions, and
  improves noise-free prediction without asserting field recovery from its
  data-limited two-site design.
- [x] Filtered RHIME-style `fp_x_flux` inputs preserve longitude-fast grid
  ordering and observation alignment.
- [x] Retained traces reconstruct on the native grid with posterior
  mean/quantiles and posterior-mean prediction RMSE.
- [x] Focused tests, Ruff checks, formatting checks, and configured type checks pass.

## Planned stages

1. Map the legacy spatial uncorrelated sampler and relevant synthetic tests.
2. Define immutable problem/state/result types and pure geometry/likelihood
   functions.
3. Implement and test NumPy birth, death, and move proposals on toy problems.
4. Add Numba kernels behind the same array contract and prove parity.
5. Add a minimal chain driver with explicit RNG state and diagnostics.
6. Add a RHIME preparation adapter retaining the filtered fine-grid design.
7. Add output conversion with fixed-capacity trace dimensions and active masks.
8. Benchmark and decide whether incremental kernels, parallel tempering, JAX,
   or a retained compiled backend are justified.

Stages 1--5 now have a working first implementation. Stage 6 has a minimal
prepared-dataset adapter, while production runner and output integration in
stages 6--7 remain follow-up work.

The first paper-specific mechanics are now implemented: native-grid posterior
projection, an opt-in normalized local discrete-Gaussian sampler move, a
calibrated two-dimensional checkerboard recovery benchmark, and exact finite
location and special birth/death subkernel oracles. Next are declared paper
profile metadata and validation of the general continuous birth/death auxiliary
proposal. Full hierarchical error and boundary blocks remain deferred.

## Progress log

### 2026-07-18

- Created branch `codex/tdmcmc-numba-rewrite` from `origin/devel` at `0308bc6`.
- Started parallel mapping of the legacy Fortran, the lognormal synthetic-data
  branch, and focused validation requirements.
- Created this planning document before implementation.
- Located the useful lognormal synthetic experiment as uncommitted work in a
  separate worktree. It will remain untouched and will be reused only as a
  read-only benchmark contract.
- Chose a canonical sorted-subset nucleus representation with explicit
  `p(k)` and `p(nuclei | k)` terms for the first implementation.
- Added the `openghg_inversions.tdmcmc` package with immutable numerical
  problem/state types, normalized target components, NumPy and Numba kernels,
  deterministic proposal accounting, and a seeded sampler with a fixed outer
  schedule and reversible mixed structural steps.
- Added a filtered RHIME-input adapter that extracts and flattens
  `fp_x_flux(nmeasure, lat, lon)` without using the already reduced fixed-basis
  `H`.
- Added focused geometry, target, proposal, detailed-balance, backend-parity,
  replay, adapter, and synthetic-recovery tests. The default focused suite has
  98 passing tests with one slow recovery test deselected; the explicitly run
  slow recovery test also passes.
- The two-cell lognormal recovery case assigns all retained posterior draws to
  `k=2` and recovers median coefficients approximately `[0.5000, 2.0135]` for
  truth `[0.5, 2.0]`.
- A small warm-state benchmark with 333 observations, 2,000 grid cells, and 50
  regions measured about 2.96 ms per NumPy rebuild and 0.64 ms per Numba
  rebuild. This is diagnostic evidence only, not a production benchmark.
- Confirmed Homebrew GNU Fortran 15.2 is installed and successfully compiled
  the legacy uncorrelated source to an object file. No Fortran compiler or Pixi
  environment is required by the new test suite.
- Reviewed the Lunt et al. (2016) main paper and supplement visually and through
  full text extraction. Added the paper-first background/specification note,
  two explicit reproduction profiles, equation-audit findings, and a server
  data-recovery checklist.
- Added a normalized discrete-Gaussian local nucleus move with exact finite-grid
  forward/reverse normalization and pointwise balance tests.
- Added an opt-in paper-style local-move sampler mode while preserving the
  existing global-move fixed-seed path as the default.
- Added an independent 12-state location-kernel oracle that verifies stochastic
  rows, rejection self-mass, detailed balance, and stationarity; it explicitly
  excludes continuous coefficient and birth/death kernels.
- Added an exact seven-state mixed-`k` birth/death oracle at a special
  fixed-coefficient unit-density proposal. It validates the trans-dimensional
  combinatorics and boundaries while explicitly excluding general continuous
  auxiliary integration.
- Used a fixed-coefficient counterexample to identify and correct the
  non-invariant deterministic birth-then-death scheduler design. Each of the
  two structural slots now uses an equal birth/death mixture, with unavailable
  boundary draws retained as self-mass.
- Confirmed the same active modulo scheduler, with random selection commented
  out, in all five spatial/temporal correlated/uncorrelated legacy Fortran
  drivers. Recorded the bounded conclusion that legacy RJMCMC outputs require
  revalidation; fixed-dimensional runs are unaffected by this specific defect.
- Added paper-defined native-grid trace reconstruction, saved-row burn-in and
  thinning selection, posterior mean/quantiles, and comparison-vector RMSE.
- Added and calibrated an `8 x 8` Lunt-inspired checkerboard benchmark with
  smooth prior-flux-weighted sensitivities, independent 5 ppb noise, exact
  native-grid reconstruction, and robust seeded Numba recovery assertions.
- Replaced its truth-informed adaptive start with a shared seeded non-oracle
  layout and added matched coefficient-opportunity comparisons against oracle,
  random fixed-basis, and movable fixed-`k` alternatives. The fixed-basis
  helper is explicitly not presented as a production RHIME/PyMC timing result.
- Mapped the repository's raw EDGAR7/UKGHG flux and TAC/MHD NAME footprint
  fixtures for a second, paper-shaped checkerboard. The paper reports grid
  sizes longitude-first, so its `56 x 48` pseudo domain is represented locally
  as `56 lon x 48 lat`, using `lon[244:300]` and `lat[157:205]`. The resulting
  56 six-hour rows have an all-ones versus checkerboard prediction RMSE of
  6.57 ppb at fixed 5 ppb error, but the sixteen-column oracle design is ill
  conditioned and several blocks are effectively unseen. Prediction is
  therefore the primary validation metric; unweighted field recovery is only
  diagnostic.
- Confirmed that the packaged EUROPE InTEM map supplies six outer labels around
  a much larger `183 x 128` inner rectangle. The paper-shaped crop lies wholly
  inside that inner class, so adapting the map gives six fixed outer blocks plus
  one fixed remainder around the crop. With all seven coefficients fixed at
  one, their contribution is exactly a known offset. Inferring fixed outer
  coefficients remains a composite-predictor follow-up; corrupted boundary
  files are not part of this benchmark.
- Added the raw NAME/EDGAR checkerboard as a structural test and a seeded slow
  prediction benchmark. The structural path builds a canonical RHIME-style
  fine-grid dataset, passes it through `problem_from_rhime_inputs`, and closes
  the seven-column InTEM fixed-offset decomposition to `1.7e-13` ppb. In the
  slow run the all-ones prior prediction RMSE is 6.57 ppb, the oracle fixed
  sixteen-block inversion reaches 1.51 ppb, a non-oracle sensitivity-weighted
  sixteen-region quadtree reaches 1.73 ppb, and the trans-dimensional
  `k=5..100` run initialized at `k=40` reaches 1.69 ppb. Each receives 5,000
  coefficient-proposal slots. The trans-dimensional visited range (`k=6..88`)
  is reported only as a mixing diagnostic, not recovery of the paper's
  posterior `k`.
- The expanded focused suite passes all 166 tests, including all three slow
  seeded recovery/comparison cases; Ruff formatting/checks and Pyright also
  pass.

## Open questions

- Which legacy variant and production-scale parameter ranges should define the
  first performance target?
- Should the first faithful port preserve legacy acceptance expressions exactly
  or implement corrected expressions when the mathematical audit identifies a
  discrepancy? The default is to document both and validate the corrected
  target independently.
- Which representation should be canonical in saved traces: nuclei and padded
  coefficients, reconstructed fine-grid fields, or both?
- Should full-scale fine-grid sensitivity remain float64, or should an audited
  mixed-precision path retain float32 `G` to reduce memory for the
  333-by-293-by-391 InTEM benchmark?
- What local nucleus-move distribution should replace the first-slice global
  move while retaining explicitly normalized forward and reverse mass?
- Can the correlated-error likelihood be reformulated using block solves or
  structured covariance operators instead of dense inverse matrices?
