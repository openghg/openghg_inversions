# NumPy/Numba trans-dimensional MCMC rewrite

## Objective

Develop a testable NumPy reference implementation and a parity-checked Numba
implementation of the legacy ACRG trans-dimensional MCMC sampler. Integrate
the numerical engine with OpenGHG Inversions through explicit input and output
adapters rather than coupling it directly to PyMC.

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
| 2026-07-18 | Repeat coefficient, birth, death, and global-move attempts on a fixed schedule. | Birth/death attempt probabilities remain equal at dimension boundaries; impossible attempts are recorded as self-transitions rather than silently renormalised. |
| 2026-07-18 | Preserve the filtered `fp_x_flux` field as the initial RHIME integration seam. | Current prepared datasets retain the fine-grid contribution field, so the rewrite need not alter fixed-basis production preparation in its first integration experiment. |

## Validation gates

- [x] Voronoi labels agree with independently calculated toy examples.
- [x] Aggregated sensitivities equal direct column-wise sums.
- [x] Log-likelihood and prior terms agree with independent formulas.
- [x] Birth/death proposal pairs expose complete forward and reverse terms.
- [x] The first global move reports normalized symmetric probabilities; the
  legacy local boundary proposal remains deferred.
- [ ] Tiny enumerable targets satisfy edgewise detailed balance.
- [x] Forced birth/death pairs satisfy pointwise detailed balance.
- [x] NumPy and Numba kernels agree for deterministic and randomised states.
- [x] Fixed-seed sampling is reproducible.
- [x] A small synthetic lognormal inversion recovers expected structure.
- [x] Filtered RHIME-style `fp_x_flux` inputs preserve longitude-fast grid
  ordering and observation alignment.
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
  deterministic proposal accounting, and a fixed-schedule seeded sampler.
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
