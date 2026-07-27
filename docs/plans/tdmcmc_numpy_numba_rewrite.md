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

The diagnosis of slow spatial-dimension mixing and the proposed direct
full-tiling alternative are recorded separately in
[rjmcmc_partition_mixing_and_full_tiling_design.md](rjmcmc_partition_mixing_and_full_tiling_design.md).
That reference distinguishes canonical leaf tilings from their many possible
construction trees and histories, and uses the sibling `inversions-knowledge`
repository at inspected revision `e77d20c` for reviewed RJMCMC, Gamma--Beta,
projection, and MCMC background.

The completed first slice covers the spatial, single-sector, independent
Gaussian problem. The active follow-up branch adds an irregular-time temporal
error model and a shared partially pooled coefficient prior. Multiple
trans-dimensional sectors and parallel tempering remain later work.

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
| 2026-07-18 | Split the replayable NAME/EDGAR checkerboard workflow into a root example script plus focused regression tests. | Recent draft PRs #502 and #506 establish this pattern for the other model-selection experiments. Data paths, crop policy, pseudo-data construction, comparators, CLI controls, and reporting are example concerns; exact accounting, a cheap smoke path, and the optional slow comparison remain tests. The directory was subsequently named `examples/rjmcmc` when the experimental package namespace was decided. A new package module is deferred until a second consumer establishes a stable reusable boundary. |
| 2026-07-19 | Publish the implementation under an explicitly experimental RJMCMC namespace rather than treating `tdmcmc` as a supported top-level API. | The implementation is specifically reversible-jump MCMC, while trans-dimensional MCMC is a broader family. An `experimental` namespace communicates stability without an import-time warning that would pollute tests, notebooks, documentation builds, and downstream `-W error` use. Example-specific orchestration remains outside the package. |
| 2026-07-19 | Model known offsets and always-active linear predictors separately from the variable Voronoi field. | This supports fixed boundary offsets and jointly inferred InTEM outer-region coefficients without assigning them artificial nuclei or allowing structural moves to remove them. |
| 2026-07-19 | Add a fifth coefficient-update slot only when an inferred fixed predictor block is present. | The four-slot no-block schedule and its seeded traces remain unchanged; fixed-block runs alternate dynamic and fixed coefficient opportunities around the two reversible structural slots and one nucleus slot. |
| 2026-07-19 | Define retention by global completed-transition number and carry the complete PCG64 and schedule state across segments. | Split execution must be exactly equivalent to an uninterrupted chain, including at schedule and thinning boundaries, and a resumed segment must not duplicate its incoming state. |
| 2026-07-19 | Infer all seven test-data-backed outer-emissions coefficients in every fair checkerboard comparator. | The oracle and non-oracle fixed inner layouts and the trans-dimensional inner field now share the same outer design, priors, proposal scale, observation vector, and total-prediction metric. Boundary-condition fixtures remain excluded. |
| 2026-07-19 | Use a strict atomic NPZ checkpoint for execution state and a separate xarray dataset for retained scientific draws. | Checkpoints must preserve exact PCG64, schedule, retention, problem identity, fixed/dynamic caches, and cache-construction backend; retained draws need labelled variable-capacity dimensions but should not be forced into the current fixed-basis `InversionOutput` contract. |
| 2026-07-19 | Treat the Lunt per-region hierarchy as log-space Normal parameters, not arithmetic lognormal moments. | The paper's `mu_x` and `sigma_x` describe `log(x)`. Existing arithmetic mean/SD inputs remain the fixed-prior pseudo-data mode and must not be relabelled as the Lunt hierarchy. Numerical hyperprior bounds and proposal scales remain unavailable. |
| 2026-07-19 | Do not copy a parent region's inferred hyperparameter pair in an upward structural proposal. | The paper does not specify how the new dimension-dependent pair is proposed. Legacy parent copying lies on an equality submanifold and lacks a valid reverse density after the pairs evolve independently. The planned auditable completion draws the new pair from its normalized bounded hyperprior. |
| 2026-07-20 | Treat the legacy structural-schedule failure and the missing hyperparameter dimension-matching rule as distinct RJMCMC correctness defects. | The former is demonstrated by finite enumeration. The latter follows because adding a region also adds two independently variable hyperparameters, but parent copying introduces no auxiliary variables or reverse density for them. Including their prior density in the target does not repair an incomplete dimension-changing proposal. |
| 2026-07-22 | Optimize one-nucleus structural proposals by deriving incremental ownership from the existing canonical state and rebuilding only membership-changed design columns. | The current production-shaped matrix makes a full `H` scan the dominant cost. Owner identities can be recovered from `nuclei[labels]` without changing the checkpoint schema; recomputing every affected column from its final members in ascending cell order preserves the full-build target and exact replay contract. The complete rebuild remains the initialization, validation, and fallback oracle. |
| 2026-07-22 | Keep speculative nearest-neighbour accelerators replaceable and outside the proposal-accounting contract. | A k-d tree, quadtree, distance transform, or other geometry engine must reproduce canonical tie handling and cannot by itself avoid the dominant `H` aggregation. The sampler should depend on a state-construction seam, while experimental geometry providers are benchmarked against the full-build oracle before becoming checkpointed run settings. |
| 2026-07-22 | Treat the historical lognormal bias as the use of `mu_log=0`, `sigma_log=1`, not `mu_log=1`, `sigma_log=1`. | That distribution has median and mode-scale parameter one but arithmetic mean `exp(1/2)`. With independent regional coefficients the aggregate country total is driven toward that larger mean as the number of regions grows. Current comparison runs instead use arithmetic mean one and arithmetic SD one, corresponding to `mu_log=-0.5*log(2)` and `sigma_log=sqrt(log(2))`. |
| 2026-07-22 | Implement Ganesan-inspired partial pooling first as one shared, dimension-invariant prior pair for all dynamic Voronoi coefficients. | Per-region prior parameters give one coefficient little information about two local hyperparameters and also require a new reversible-jump dimension match. A shared arithmetic mean and SD pool is identifiable from the coefficient population, remains fixed-dimensional as `k` changes, and resembles the earliest trans-dimensional code before per-region pairs were introduced. Fixed outer coefficients remain outside this pool. |
| 2026-07-22 | Represent shared coefficient-prior parameters internally as `eta=log(M)` and `zeta=log(S)`, where `M` and `S` are arithmetic coefficient-prior moments. | This preserves the established user-facing arithmetic mean/SD convention. Ganesan-style lognormal hyperpriors become normalized Normal densities in `eta` and `zeta`, and symmetric random walks need no proposal Jacobian. Hyperpriors are configured explicitly by median and log-SD to avoid another parameterization ambiguity. |
| 2026-07-22 | Make independent measurement error plus latent OU model mismatch the primary correlated-error implementation. | This is the independent-site reduction of Ganesan et al. (2015): `C = D + M Q M`. It preserves a genuine uncorrelated measurement nugget and remains exactly `O(n_observations)` through a scalar irregular-time Kalman likelihood. The Lunt/Ganesan-2014 `R = S Q S` correlated-total-error model is a distinct comparison profile, not an equivalent way to add a nugget. |
| 2026-07-22 | Treat every historical iteration count as revision-specific and expose new OU/hierarchy schedules under new versioned identities. | Ganesan's fixed-dimensional code swept all components, intermediate trans-dimensional revisions used five- or seven-way random scans and one shared dynamic hyperpair, and the closest paper-model revision used a six-way random scan with per-region hyperpairs. The published pseudocode does not uniquely determine mismatch or timescale proposal counts. Existing rewrite schedules and random streams therefore remain unchanged; new 16- and 17-slot profiles explicitly add one randomly selected mismatch amplitude, one randomly selected timescale, and optionally one joint shared-pool update to the existing 14-slot cycle. |
| 2026-07-22 | Treat archived Lunt coefficient-prior inputs as arithmetic mean/SD, while recording the conflict with the paper's log-space notation and the older Ganesan kernel. | Revision `6f165e68` explicitly converts `mean` and `sd` to lognormal log-location/log-scale inside `calc_pdf`; its input template uses dynamic mean 1 and SD 1. The older Ganesan density instead treats its first input as a positive median/geometric scale and its second as log-space SD. The current rewrite's arithmetic convention follows the Lunt code and current inversion API, not the incompatible Ganesan calling convention. |
| 2026-07-23 | Treat the archived Tunnicliffe Brazil production runs as configuration and opportunity-count evidence, not a posterior-correctness or parallel-tempering oracle. | The exact INIs and outputs establish a 120,000-cycle-equivalent workload, but the source uses the invalid deterministic structural cycle, omits structural selection-count terms, copies independently evolving per-region prior SDs, and accepted zero of 275,000 attempted temperature swaps. |
| 2026-07-23 | Treat the canonical leaf tiling, rather than a split tree or chronological construction history, as the scientific state of the proposed dyadic alternative. | Several decomposition trees and split orders can produce the same final tiling. An uncorrected target on histories therefore weights tilings by their representation multiplicity. Direct leaf state avoids this hidden prior; fixed-tree Gamma--Beta/product-space work remains a restricted comparator and correctness oracle. |
| 2026-07-23 | Diagnose current `k` mixing with proposal-flow, prior-only, and powered-likelihood experiments before adding parallel tempering. | Balanced accepted up/down counts are largely implied by the net change in `k` and do not show traversal. The likelihood-power ladder separates intrinsic local diffusion from energy barriers, while scientific field, prediction, and reporting-total diagnostics distinguish weak identification of raw `k` from failure to explore informed modes. |
| 2026-07-23 | Keep structural mixing diagnostics output-only, transition-indexed, and outside checkpoint kernel compatibility. | Proposal candidates, rejections, and immediate reversals cannot be reconstructed from thinned retained states. Deterministic cached-state instrumentation can collect them without consuming random numbers or changing restart semantics; a separate xarray transition dimension provides segment persistence. |
| 2026-07-23 | Use the likelihood-power family first as independent diagnostic targets, not as a commitment to parallel tempering. | The target `p(s) L(y | s)^beta` is the same likelihood tempering used by parallel-tempering replicas, but fixed-`beta` endpoint chains need no swaps. Similar poor mobility at `beta=0` and `beta=1` would argue against rebuilding a tempering system that historically accepted no swaps. |
| 2026-07-23 | After profiling Voronoi mixing, implement a restricted fixed-direction Gamma--Beta tree as the first alternative local-move baseline. | One canonical hierarchy avoids full-tiling history multiplicity and permits tiny enumeration/product-space cross-checks. It deliberately tests local active-only split/merge mobility on restricted support; it is not yet the final full-tiling model. |
| 2026-07-26 | Use fixed-basis NUTS as the continuous-kernel reference before changing the Gamma root model. | Diagonal and dense NUTS removed the slow fixed-basis root mode by two to three orders of magnitude in ESS/hour, showing that the local continuous transition was the dominant fixed-topology problem. |
| 2026-07-26 | Test topology-conditioned static Euclidean HMC before RMHMC. | H2/H2b/H2c failures varied strongly by tiling, while a curvature audit found topology effects much larger than within-topology position effects. A deterministic precision may change between tilings while remaining constant during each ordinary HMC trajectory. |
| 2026-07-26 | Use exact log-coordinate involutions for HMC-specific edge flips and resolution relocations. | The first physical split implementation admitted forward binary64 paths without representable reverse fractions. Coordinate permutations give bit-exact reverse geometry, unit Jacobian, and explicit discrete catalogue accounting. |
| 2026-07-26 | Keep the RJ Gaussian closure distinct from the non-RJ NLE/GMM marginal-likelihood track. | The Gaussian closure is an optional explicit approximation usable with RJ. The NLE/GMM alternative integrates hidden allocations for fixed partitions; its exact common-native-model limit is partition- and \(K\)-invariant and licenses no structural learning. Approximate evidence drift is leakage, not a learned structural weight. |
| 2026-07-26 | Test a frozen conditional-allocation bank before adding `sbi`. | The transported standardized-shape mixture failed its T2 development screen. Frozen within-region Dirichlet shares retain the full conditional dependence on region masses, give a deterministic normalized Gaussian mixture after measurement-noise convolution, and can be differentiated directly in NumPy/PyTensor. Partitions remain externally weighted by their structural prior. |

## Ganesan lineage and active hierarchy plan

Ganesan et al. (2014) supplied the fixed-dimensional hierarchy for emissions
prior parameters, model-mismatch amplitudes, and temporal correlation. Ganesan
et al. (2015) applied a richer separable space-time covariance to the same four
DECC sites and overlapping March 2014 data later used by Lunt et al. The 2015
observation model separated instrumental error from a latent correlated model
discrepancy. Lunt retained the temporal hierarchy, simplified sites to
independent blocks, and made the spatial partition trans-dimensional.

Historical source inspection gives three distinct state-prior conventions:

1. the Ganesan fixed-dimensional sampler gave existing coefficients evolving
   prior parameters and could update them with ordinary Metropolis-Hastings;
2. the first July 2015 trans-dimensional code used one shared parameter pair
   for all variable regions, so structural changes did not change the
   hyperparameter dimension;
3. the February 2016 change to per-region pairs copied a parent's pair during
   an upward structural proposal without the auxiliary variables or reverse
   density required by reversible-jump MCMC.

The active implementation stages are therefore:

1. add a normalized irregular-time OU likelihood with inferred grouped model
   mismatch and explicit independent measurement error;
2. add fixed-dimensional mismatch-amplitude and correlation-time transitions,
   replayable schedules, checkpoints, and labelled retained output;
3. add one shared partially pooled coefficient-prior pair with Ganesan-style
   hyperpriors, leaving the fixed outer coefficients on their existing priors.

All three stages now have target, proposal, versioned schedule, durable
checkpoint, manifest, and labelled-output support. The shared pool is
intentionally the fixed-dimensional structure from the intermediate
trans-dimensional lineage, not a claim to reproduce the paper's per-region
hierarchy.

The implemented 16- and 17-slot drivers are intentionally scoped to the
six-fixed-coefficient PARIS/Lunt-style geometry. The core target and proposal
primitives can represent OU or hierarchy layers in other geometries, but no
generic hierarchy-only or arbitrary-fixed-block schedule is exposed yet; a
caller cannot silently sample such a target with frozen optional parameters.

### Historical update-opportunity evidence

There is no repository-supported basis for treating all reported historical
"iterations" as the same amount of work:

- The closest surviving Ganesan fixed-dimensional
  `hierarchical_MCMC_fullcovariance_Kronecker.f90` at revision `bd609a39`
  performs a deterministic full component sweep. Every
  state element and its two prior parameters, every temporal mismatch
  amplitude, every site amplitude, and the scalar timescale receive an
  opportunity in one nominal iteration. Its source header postdates the paper,
  so this is lineage evidence rather than proof of the published run's exact
  schedule.
- Intermediate trans-dimensional `acrg_full_hbtdmcmc.f90` revisions used a
  random five- or seven-slot top-level scan. The dynamic cells shared one
  hyperparameter pair, and the mismatch update selected one random group.
  Revision `7c693833` explicitly changed the emissions slot to a broader sweep
  in an attempt to improve convergence, without changing the top-level random
  scan.
- The closest preserved paper-model revision, `6f165e68` (2016-02-10), selects
  one of six slots uniformly: emissions/hyperparameters, mismatch amplitude,
  timescale, split, merge, or move. A mismatch slot selects one random group; a
  timescale slot selects one random site; an emissions slot makes five random
  dynamic coefficient and five random dynamic hyperpair proposals. Thus
  600,000 nominal iterations imply about 100,000 opportunities for each
  top-level slot, 500,000 dynamic-coefficient proposals, and 500,000 dynamic
  hyperpair proposals in expectation, before division across groups/sites.

The Lunt paper's Algorithm 1 uses a deterministic modulo pseudocode but does
not identify how its generic hyperparameter step maps to the internal mismatch,
timescale, and emissions-prior loops. The exact executable revision used for
the published 90-minute timing has not been identified. Timing reports must
therefore state both atomic transition counts and the historical revision whose
proposal opportunities they intend to match.

This revises the interpretation of the current 120,000-cycle, 14-slot PARIS
comparison. Relative to 600,000 iterations of the near-submission six-way
revision, it makes about 20% more dynamic-coefficient and geometry proposals
and 44% more fixed-coefficient proposals in aggregate. Its six deterministic
fixed positions also differ from the historical five random fixed draws with
replacement. The profile remains a useful, exactly replayable production
schedule, but it is not an exact opportunity match to that revision. The new
16- and 17-slot profiles deliberately extend this existing production
schedule; a future near-submission-revision profile would instead use five
random fixed-coefficient slots and must have a distinct identity.

The same closest paper-model revision also resolves one part of the lognormal
ambiguity. Its `calc_pdf` converts an input arithmetic `mean` and `sd` into
log-space parameters before evaluating the density; the companion template
sets the dynamic pair to arithmetic mean one and SD one. This is strong
evidence against attributing the archived template's performance to a
`mu_log=1, sigma_log=1` prior. It does not identify the exact uncommitted input
file used for the published timing. The older Ganesan density accepted a
positive median/geometric scale and log-space SD directly, so the two kernels'
configuration pairs must not be compared by name alone.

The later Tunnicliffe et al. Brazil production archive supplies an exact
operational comparison. Its Fortran loop runs 100,000 burn-in plus 500,000
post-burn nominal iterations under a deterministic five-phase schedule. Across
the full run this gives 120,000 opportunities each for model error, upward
structure, downward structure, movement, and the emissions block; each
emissions block updates all nine always-active parameters and five randomly
selected dynamic coefficients. Thus 120,000 modern cycles match the common
dynamic and geometry opportunity counts much more closely than the
near-submission Lunt revision alone suggested. The kernels still differ: the
archive uses the invalid directional schedule and per-region SD inheritance,
whereas the rewrite uses invariant mixed structural slots and, optionally, one
shared hierarchy pair.

Although those Brazil INIs enable four-temperature parallel tempering, all
three archived outputs record zero accepted swaps out of exactly 275,000
attempts. Source inspection verifies that the counters are live and that no
temperature labels exchanged. The ladder `[1, 0.15874011, 0.02519842, 0.004]`
and arbitrary-pair proposals therefore provide a negative implementation
example, not evidence that tempering cured historical `k` mixing. Any rewrite
tempering stage needs adjacent exchanges, tested swap balance, acceptance
diagnostics, round-trip diagnostics, and exact checkpointed replica/RNG state.

For one shared pool, let `M` and `S` be the arithmetic mean and arithmetic SD
of the dynamic coefficient prior and define `eta = log(M)`, `zeta = log(S)`.
The normalized dynamic target is

```text
p(k) * p(c | k) * p(eta) * p(zeta)
* product_i p(x_i | M=exp(eta), S=exp(zeta)).
```

The shared pair is counted once, not once per region. It persists unchanged
through structural proposals, so existing structural proposal and Jacobian
terms remain valid. A joint symmetric random walk in `(eta, zeta)` changes
only the coefficient-prior and hyperprior target terms and must reuse all
geometry, design, prediction, residual, and likelihood caches.

This shared model does not remove the dependence of country-total prior
variance on `k`. Prior-predictive country totals must be checked at several
fixed values of `k`; that dependence is a scientific model property rather
than a reversible-jump balance error. The per-region specification below is
retained as a possible later paper-faithful extension, but it is no longer the
first hierarchy implementation target.

## Lunt per-region hierarchy implementation specification

### Status and provenance

This section is the durable implementation specification for the next
data-independent correctness stage. It separates facts recovered from the
paper and legacy code from explicit rewrite decisions. It does not define a
numerical `Lunt2016` run profile: the archived hyperprior bounds,
initialization, and proposal scales are still unavailable.

There are two separate legacy RJMCMC findings:

1. **Demonstrated scheduler defect.** Separate deterministic upward-only and
   downward-only transition steps are not individually target-invariant, and
   their composition fails on the checked finite counterexample. The rewrite
   already replaces them with reversible mixed structural kernels.
2. **Hyperparameter dimension-matching defect.** In the hierarchical model,
   changing from `k` to `k + 1` introduces `x_new`, `mu_new`, and `sigma_new`.
   The legacy code proposes `x_new` but deterministically copies the parent's
   two prior parameters. No two-dimensional auxiliary variable, invertible
   transformation, Jacobian, or ordinary reverse density is supplied for the
   new independently variable pair. Once a within-model update moves that pair
   away from the parent's values, a parent-copying reverse proposal cannot
   reconstruct the state and has zero reverse probability. Evaluating the
   hyperprior in the target is necessary but does not fix this proposal defect.

The second finding is therefore reasonably summarized as "RJMCMC was not
applied correctly to the dimension-dependent hyperparameters." It is not yet
covered by the same finite enumeration as the scheduler defect; that oracle is
a required implementation gate below.

### Confirmed facts, rewrite decisions, and unknowns

| Category | Item |
| --- | --- |
| Paper fact | Each active region has its own log-space Normal prior parameters: `log(x_i) ~ Normal(mu_i, sigma_i**2)`. |
| Paper fact | The `mu_i` and `sigma_i` values have bounded uniform hyperpriors, and hyperparameter updates use centred Gaussian proposals. |
| Paper gap | The upward structural move does not state how the new region's `(mu_i, sigma_i)` pair is generated. |
| Legacy behavior | The new region copies the parent's arithmetic-moment prior parameters; later within-model updates can make region pairs differ. |
| Rewrite decision | Keep the existing arithmetic-mean/SD lognormal prior as an unchanged non-hierarchical mode. Add a separate opt-in log-space hierarchical mode. |
| Rewrite decision | In hierarchical mode, draw a new pair independently from its normalized bounded hyperprior during an upward move. |
| Rewrite decision | Update one selected region's `(mu_i, sigma_i)` pair jointly with centred Gaussian increments; proposals outside the bounds are self-transitions. |
| Unknown | Archived numerical bounds, initial active pairs, proposal scales, seeds, and exact production grouping/configuration. |
| Ambiguity | The paper is not fully explicit about scalar versus paired hyperparameter updates. A paired update matches its acceptance expression and the relevant legacy behavior, and is the reference choice unless archived evidence contradicts it. |

### Normalized hierarchical target

For every active region `i`, the coefficient density is

```text
p(x_i | mu_i, sigma_i)
  = exp(-0.5 * ((log(x_i) - mu_i) / sigma_i)**2)
    / (x_i * sigma_i * sqrt(2*pi)),       x_i > 0.
```

For explicit bounds `[a_mu, b_mu]` and `[a_sigma, b_sigma]`, with
`0 < a_sigma < b_sigma`, the pair density is

```text
p(mu_i, sigma_i)
  = 1 / ((b_mu - a_mu) * (b_sigma - a_sigma))
```

inside the bounds and zero outside. These normalizing constants must remain in
the target because one pair is added or removed when `k` changes. The dynamic
part of the normalized target is therefore

```text
p(k) * p(c | k)
* product_i p(x_i | mu_i, sigma_i) p(mu_i) p(sigma_i),
```

multiplied by the likelihood and any always-active parameter priors. As in the
current engine, `p(c | k) = 1 / comb(n_grid, k)` for the canonical unordered
nucleus set.

### State and configuration contract

- Hierarchical mode must be opt-in and mutually exclusive with the existing
  fixed arithmetic-moment coefficient prior.
- An immutable hierarchy configuration must contain finite log-mean bounds,
  strictly positive log-standard-deviation bounds, and explicit proposal
  scales for both values.
- The state must carry padded `mu` and `sigma` arrays aligned with the padded
  nuclei and coefficients. Sorting, moving, inserting, and removing a region
  must operate on the complete `(nucleus, coefficient, mu, sigma)` record.
- Initial hierarchical states must supply one valid pair for every active
  region. Defaults inferred from unavailable paper settings are prohibited.
- The non-hierarchical target and its seeded traces must remain unchanged; it
  must not be silently routed through a newly converted parameterization.

### Reference proposal kernels

For a within-model hyperparameter update, choose one active region uniformly
and propose

```text
mu_i'    = mu_i    + Normal(0, step_mu)
sigma_i' = sigma_i + Normal(0, step_sigma).
```

The proposal is symmetric. If either value is outside its declared bounds,
retain the input state. Otherwise the prediction, residual, and likelihood
caches are unchanged, and the Metropolis ratio reduces to the change in that
region's normalized coefficient-prior and hyperprior terms.

For an upward structural move from `k` to `k + 1`:

1. select the upward direction within the 50/50 mixed structural kernel;
2. select a vacant native cell uniformly and identify its pre-move owning
   region;
3. propose `x_new` with the existing untruncated parent-centred Gaussian,
   retaining non-positive draws as self-transitions;
4. draw `mu_new ~ Uniform(a_mu, b_mu)` and
   `sigma_new ~ Uniform(a_sigma, b_sigma)` independently;
5. insert and canonically sort the complete new region record.

Away from a structural boundary, the forward density includes

```text
q_up = 0.5
       * 1 / (n_grid - k)
       * NormalPDF(x_new; x_parent, coefficient_step)
       * 1 / (b_mu - a_mu)
       * 1 / (b_sigma - a_sigma).
```

The downward move chooses the direction with probability `0.5` and one of its
active nuclei uniformly. Its reverse upward density includes the Gaussian
density of the removed coefficient and the two uniform densities of the
removed pair. The acceptance decision must always be computed from

```text
min(1, target(candidate) * q_reverse / (target(source) * q_forward)).
```

Drawing the new pair from its normalized hyperprior makes the two proposal
density factors cancel the new pair's two hyperprior factors in this ratio.
They must nevertheless be represented explicitly and tested rather than
removed by hand. At `k_min` or `k_max`, an unavailable direction remains the
selected kernel's self-transition; the 50/50 mixture is not renormalized.

A normalized parent-centred pair proposal or an invertible split/merge mapping
could also be valid, but would introduce additional scales, boundary
normalizers, or Jacobians. Neither is the reference implementation without
evidence that it is needed for mixing.

### Schedule, serialization, and output consequences

The proposed versioned schedules are:

- hierarchy without fixed predictors: dynamic coefficient, hyperparameter,
  structural, structural, nucleus;
- hierarchy with fixed predictors: dynamic coefficient, fixed coefficient,
  hyperparameter, structural, structural, nucleus.

The existing four- and five-slot non-hierarchical schedules remain unchanged.
Adding the hierarchy requires new schedule identifiers and a checkpoint schema
version, since exact continuation must include the padded pairs, hierarchy
configuration, proposal scales, and schedule position. Provenance manifests
must record `parameterization = "log_space_parameters"`, the bounds, proposal
scales, and schedule identifier. Xarray retained output gains aligned `mu` and
`sigma` variables on `(draw, region_slot)`; inactive slots remain masked.

### Required implementation and validation sequence

1. **Target/state primitives only.** Add validated hierarchy configuration,
   aligned padded state arrays, normalized NumPy/Numba density kernels, and
   target-component reporting. Do not expose hierarchical sampling yet.
2. **Independent target tests.** Compare the density with a direct formula,
   test support and normalizers, establish NumPy/Numba parity, verify common
   sorting of complete region records, and prove the current fixed-prior path
   and seeds are unchanged.
3. **Deterministic proposal primitives.** Add forced within-model and
   structural candidates with separately reported target, forward, reverse,
   and Jacobian terms.
4. **Proposal oracles.** Check the direct hyperparameter target delta,
   out-of-bounds self-mass, unchanged likelihood caches, exact reconstruction
   of complete states by paired upward/downward moves, and every hyperparameter
   proposal-density factor.
5. **RJMCMC invariance gates.** Extend the continuous forward/reverse flux
   oracle across several pairs and build a finite hierarchical analogue that
   checks normalization, detailed balance, and stationarity. A deliberately
   parent-copying kernel should fail this oracle, documenting the regression.
6. **Sampler integration.** Add versioned schedules, exact split-chain tests at
   awkward schedule boundaries, retention, diagnostics, and NumPy/Numba seeded
   parity.
7. **Serialization and interchange.** Bump checkpoint and output schemas,
   extend exact replay/fingerprint tests, and add labelled xarray variables.
8. **Synthetic calibration only.** Exercise mixing in `k`, coefficients, and
   pairs with explicitly synthetic bounds. Do not describe those values as a
   Lunt reproduction.
9. **Archived profile later.** Recover and checksum the actual bounds,
   initialization, proposal scales, seeds, fixed outer/boundary treatment, and
   any stored traces before defining a genuine paper profile.

### Compact restart point

- **Next executable task:** steps 1--2 above, adding target/state primitives and
  their independent tests without making hierarchical sampling public.
- **Data dependency:** none for steps 1--8 when all numerical values are
  explicitly labelled synthetic. Only the archived profile in step 9 is
  blocked by unavailable paper data/configuration.
- **Compatibility constraint:** do not alter the fixed arithmetic-moment prior,
  its four-/five-slot schedules, or existing seeded traces.
- **Exposure gate:** do not expose a hierarchical sampler until the complete
  structural proposal passes forward/reverse flux, detailed-balance, and
  stationarity oracles.
- **Provenance constraint:** do not call any synthetic settings a Lunt profile.

## Current offline work queue

The archived paper inputs are expected to remain unavailable for several days.
That blocks only the faithful-data reproduction stage; it does not block the
following correctness and integration work. Items are ordered by dependency.

1. **Completed:** add an independent continuous-coefficient structural-kernel
   oracle, combining varied pointwise forward/reverse flux properties with
   numerical quadrature that includes invalid negative Gaussian proposal mass
   as a self-transition.
2. **Completed:** move the NAME/EDGAR benchmark orchestration into
   `examples/rjmcmc/lunt_name_edgar_checkerboard.py`, leaving focused accounting,
   adapter-ordering, no-boundary-condition, comparator, and smoke contracts in
   pytest.
3. **Completed:** introduce immutable, validated run-profile and provenance
   primitives without silently inventing unconfirmed paper settings.
4. **Completed:** rename the package to
   `openghg_inversions.experimental.rjmcmc` and update imports atomically. No
   compatibility shim or import-time warning was retained because this branch
   has not been released.
5. **Completed:** extend the forward model, RHIME-style adapter, sampler, and
   prediction summaries with an always-active fixed block. The NAME/EDGAR
   example now jointly infers seven InTEM outer-emissions coefficients for all
   three comparator methods while excluding the corrupt boundary-condition
   fixture.
6. **Completed:** collect retained draws
   on a global transition clock and support exact split-chain continuation with
   preserved PCG64, schedule, retention, kernel, and fixed-block state. A strict
   atomic checkpoint validates exact problem/manifest fingerprints and every
   stored cache before continuation.
7. **Completed:** add an xarray retained-trace export with global transition,
   padded region-slot, active-mask, and separate fixed-parameter coordinates.
8. **Completed experimentally:** add independent-site,
   irregular-time latent OU mismatch with an explicit measurement nugget,
   inferred bounded mismatch amplitudes, and inferred bounded timescales. The
   scalar Kalman likelihood is normalized and has NumPy/Numba and dense-
   covariance oracles, versioned schedules, strict checkpoints, manifests, and
   labelled retained output.
9. **Completed experimentally:** add one shared arithmetic
   mean/SD pair for the dynamic-coefficient prior, with normalized Ganesan-style
   hyperpriors in log coordinates, a joint symmetric proposal, and full
   schedule/checkpoint/output support. The
   paper-faithful per-region hierarchy is deferred because it is weakly
   identified and its legacy structural dimension match is invalid.

This repository does not currently contain an agent-tracker configuration, so
the queue, ownership, decisions, and evidence are recorded in this planning
document and in small commits on the draft branch.

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
  stationarity.
- [x] Continuous auxiliary-coefficient structural proposals satisfy varied
  pointwise forward/reverse flux checks and independent two-cell quadrature,
  including negative-proposal self-mass.
- [x] Irregular-time OU likelihoods agree with independently constructed dense
  Gaussian covariance log densities for interleaved sites, grouped amplitudes,
  shared and site-specific timescales, singleton blocks, and extreme gaps.
- [x] Mismatch-amplitude, timescale, and shared-hierarchy proposals satisfy
  pointwise forward/reverse accepted-flux equality and NumPy/Numba parity.
- [x] Extended schedules preserve exact global phase and PCG64 state across
  awkward in-memory and durable restarts without changing legacy seeded paths.
- [x] Checkpoint schema v3 round-trips optional state and rejects altered OU
  mappings, bounds, hierarchy settings, kernel scales, arrays, and cached target
  terms while retaining v1/v2 legacy-target compatibility.
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
- [x] Explicit fixed predictors preserve observation ordering, are sampled in
  a separate schedule slot, and contribute correctly to likelihood, priors,
  total prediction, summaries, and exact continuation.
- [x] Split retained chains exactly reproduce uninterrupted chains across
  non-aligned schedule and thinning boundaries.
- [x] Durable NumPy/Numba checkpoints round-trip dynamic-only and fixed-block
  chains and reject altered targets, manifests, schemas, runtimes, arrays, and
  unsafe pickle payloads.
- [x] Retained traces reconstruct on the native grid with posterior
  mean/quantiles and posterior-mean prediction RMSE.
- [x] Retained traces export to labelled xarray datasets without conflating
  dynamic region slots, always-active parameters, and attempted-transition
  diagnostics.
- [x] Optional retained output labels mismatch groups and timescale parameters,
  and makes inactive hierarchy coordinates explicit instead of interpreting
  neutral state zeros as inferred arithmetic moments.
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

Stages 1--7 now have working experimental implementations. The RHIME adapter
still covers the independent-error target only; inferred OU grouping and time
semantics must remain an explicit production-driver responsibility until an
opt-in adapter contract is added.

The first paper-specific mechanics are now implemented: native-grid posterior
projection, an opt-in normalized local discrete-Gaussian sampler move,
continuous and finite structural-kernel oracles, declared run profiles, exact
durable continuation, labelled retained-trace interchange, and joint
dynamic-inner/fixed-outer prediction. The first correlated-error and hierarchy
extension now adds an irregular-time independent-site latent OU discrepancy,
inferred mismatch amplitudes and timescales, and one shared partially pooled
dynamic-coefficient prior pair. A separate paper-faithful Lunt `Sigma Q Sigma`
likelihood and the problematic per-region hierarchy remain follow-up profiles,
not implicit behavior of this implementation.

## Progress log

### 2026-07-26

- Completed the fixed/mobile full-tiling control. Fixed geometry materially
  improved likelihood mixing, while mobile likelihood remained the main
  common-coordinate failure. Increasing local pair updates and starting from
  a greedy/SLS tiling did not remedy mobile mixing.
- Completed the fixed-basis NumPyro NUTS reference. Diagonal NUTS gave root
  bulk ESS 1,684 at \(K=50\) and 2,071 at \(K=250\); dense NUTS at \(K=50\)
  gave root bulk ESS 6,114 with zero divergences. This overturned the tentative
  interpretation that one coherent Gamma root was intrinsically
  unsampleable.
- Implemented custom topology-plus-PyMC HMC. H2 and H2b showed that a common
  diagonal metric did not generalize to held-out \(K=250\) tilings. H2c's
  permutation-invariant total/contrast metric also hard-stopped at both
  dimensions.
- Audited frozen H2c curvature and found topology effects much larger than
  within-topology position effects. Implemented a topology-conditioned static
  Euclidean precision and dynamic PyMC potential/integrator replacement.
- Stopped the first H2d candidate `7f7b150` at D0 after finding physical
  split fractions with no representable binary64 reverse. Replaced the
  HMC-specific edge-flip and resolution-relocation maps at `e619915` with
  exact involutions of authoritative log-mass coordinates, no Beta auxiliary,
  unit Jacobian, and transformed-target plus discrete-catalogue scoring.
  The repaired `5bb4139` BP1 run subsequently passed D0 exact structural
  reversal and D1 deterministic topology-conditioned precision construction.
  D2 binding passed, but its topology-freeze Slurm job was cancelled by
  `PartitionConfig` before scientific HMC calibration; that cancellation is
  not a model failure.
- Added exact two- and four-cell aggregation-error oracles, a normalized
  low-rank Gaussian conditional-moment closure, cached fixed-partition
  factors, and a FullTiling physical-mass bridge. Deliberately excluded a
  raw-moment rectangle-prefix acceleration after it showed catastrophic
  cancellation for small covariances on large design offsets.
- Added the fixed-partition PyMC/NUTS Gaussian aggregation target at
  `a004e526`. A1 passed implementation and exact tower controls but reached a
  scientific-shape hard stop: the non-Gaussian conditional allocation law was
  not reproduced adequately in the skewed, boundary-heavy, and heterogeneous
  regimes. The Gaussian closure remains an explicit fixed-partition
  approximation rather than an exact structural likelihood.
- Added the normalized transported-mixture foundation at `6ff3afe`. Its T2
  development fitter produced three artifacts and exhausted all fixed
  restarts in five regimes; the fitted cases also failed at least one
  scientific-shape gate, so held-out evaluation did not run.
- Started the simpler conditional-allocation likelihood track. It freezes
  direct Dirichlet share draws, retains their linear dependence on the
  retained region masses, and averages the resulting exactly noise-convolved
  Gaussian components. The fixed projection is a computational
  representation, not a data-selected model; evidence drift remains a
  surrogate-error diagnostic.
- Completed the corrected PCG64 conditional-allocation C1 development screen
  at `6ee6e5375b60535ac5f00f3ce2d786a6e3ad957e`. Four of nine cases passed,
  but five failed the unchanged bank-convergence or independent-confirmation
  gates. This is a scientific hard stop for that finite random-bank method,
  not a provenance or implementation failure. The next bounded method is a
  joint scrambled-Sobol allocation bank under the same gates, followed, if
  necessary, by a normalized residual-image MDN/NLE model.
- Implemented that RQMC successor at `e0b2166597b3baa360233eb3ff63ee325a30c263`
  using canonical balanced-tree inverse-Beta splits. It preserves nested
  power-of-two prefixes, avoids inverse-Gamma underflow for tiny native-cell
  concentrations, records a strict v2 construction identity, and leaves the
  PCG64 v1 contract unchanged. Exact moment, product, multi-block, endpoint,
  replay, and permutation gates pass locally; the unchanged nine-case BP1
  C1 screen is next.
- The certified RQMC BP1 screen at
  `863353443488b7e08900a147f92039d444676d41` passed eight of nine cases.
  Boundary-heavy four-cell root passed only at the final \(S=16{,}384\)
  point, not the required two-size suffix; at \(S=1{,}024\) and \(4{,}096\)
  only posterior-SD relative error remained outside tolerance. The
  predeclared `hard_stop` therefore starts the normalized residual-image
  MDN/NLE fallback rather than a post-hoc Sobol-ladder extension.
- Added the portable residual-image context and analytically
  noise-convolved GMM/MDN likelihood at `8628369`, including complete
  residual-image rank handling, orthogonal Gaussian factors, the observation
  Jacobian, canonical region ordering, and authenticated serialization.
- Added the deterministic root-only eight-component GMM development screen
  at `5167e36`. It uses three float64 EM starts, requires at least two to
  converge, separates training/validation/simulator-test streams, binds the
  full training and runtime record into each fitted envelope, and keeps
  `scientific_pass=false` until an independent protected density holdout.
  Independent review found and the next source update restored the unchanged
  four-bank 0.05-nat evidence-range gate.
- Chose the fully restartable BP1 layout for that learned-density screen:
  24 development size shards, a pure common-lock merger, 18 lock-bound
  confirmation-seed shards, a development certificate, and a one-shot
  protected density certifier. The protected catalogue tests new residual
  draws for the same six exact contexts, promotes only the locked seed-731
  artifact, permits no retraining or retuning, and cannot license structural
  \(P/K\) inference. The mergers now regenerate the exact cases and simulator
  banks and replay likelihood, gradient, evidence, posterior, and
  generalization gates; protected inputs remain sealed until the frozen
  runtime, source, certificate, and all six nominated artifacts pass.
- Pushed the certifying candidate
  `6ad8eee8c1d02289e1b130c4204b8bbe9c86135e` on
  `codex/rjmcmc-aggregation-conditional-likelihood`. Its BP1 G0 preflight
  passed 65 focused tests, Ruff, Pyright, and smoke, and all 24 G1
  development shards completed. The merger hard-stopped before lock:
  cross-node floating replay differed by up to approximately
  \(1.1\times10^{-9}\) in NLL, after which the exact context hash exposed
  approximately \(1.72\times10^{-16}\) basis variation in the same rank-3
  residual subspace from BLAS/LAPACK. This is an implementation-portability
  hard stop, not a scientific GMM result. G2/G3 did not run and the protected
  catalogue was untouched.
- Committed the portability implementation at
  `0803137ab96240342361f08afab38acdbaf3bb28`: scale-aware replay tolerances
  with fail-closed gate margins shared by development and protected
  certification; context/basis schema v2 using stable-cell-ordered fixed-scalar
  means and modified Gram--Schmidt; and SVD rank determination only, with
  ambiguity gates. Resolve the final pushed branch head after plan-only status
  commits and repeat all of G0 and G1; none of the old shards is reusable.
- Ran the repaired eight-component candidate
  `3c91beea7836a9996d2850aadbc6892d2ed0d46a` on BP1. G0 passed 71 tests,
  Ruff, Pyright, and smoke; all 24 G1 array tasks completed. The merger
  successfully replayed all cross-node artifacts, then hard-stopped because
  there was no common two-size scientific passing suffix. Near-Gaussian and
  skewed cases passed all sizes; both boundary-heavy cases failed all sizes
  despite valid fits and passing independent-domain generalization. G2/G3 did
  not run and the protected catalogue remained sealed.
- Ran the single predeclared sixteen-component root-GMM escalation at
  `625dc3b26dcad646ee144eea2c5fdc507851cdfa`. G1 array `18187541`
  published 20 of 24 artifacts; all four skewed two-cell fits exhausted the
  three deterministic EM starts, while both boundary-heavy cases still failed
  every scientific size gate. The merger precondition was false, so no lock,
  G2, or G3 exists. This is terminal for the root-GMM architecture.
- Replaced free-EM fitting with a structure-preserving approximation to the
  known continuous Gamma--Dirichlet Gaussian location mixture. The new
  root-only implementation computes the exact analytic noise-whitened
  covariance spectrum with SciPy, constructs an equal-weight scrambled-Sobol
  source bank, and replaces hard clusters by their exact population weights,
  means, and covariances. It records separate projection and compression KL
  bounds and evaluates a normalized hybrid likelihood with an analytic
  Gaussian complement.
- Completed the first local six-case two-stage screen. Every source case
  passed at \(S=65{,}536,262{,}144,1{,}048{,}576\), selecting the common
  \(S=65{,}536\) lock. The boundary-heavy four-cell compression was
  non-monotone at low \(M\): 64 and 256 passed while 128 narrowly missed the
  0.02 posterior-SD-relative-error gate at 0.02194. Before freezing BP1, the
  ladder was extended without consulting confirmation seeds. All six cases
  then passed the common suffix \(M=256,512,1024\). Focused core, driver,
  certifier, and source-bank tests pass locally; the full-SHA BP1 G0/G1/G2
  protocol remains to run.
- Added the operational BP1 handover at
  [`rjmcmc_bp1_handover.md`](rjmcmc_bp1_handover.md).

### 2026-07-23

- Started a separate active-only fixed-direction Gamma--Beta RJMCMC baseline
  while the Voronoi Stage C profiling runs independently. The NumPy reference
  implementation uses canonical dyadic frontiers, exact \(N_K\) counts,
  normalized root Gamma and active Beta priors, a normalized marginal
  \(p(K)\), and nominal within-leaf allocation.
- Defined split/merge as insertion/deletion of one active fraction, giving a
  unit Jacobian in root-plus-fraction coordinates. The physical
  parent-mass Jacobian is intentionally absent because leaf masses are not
  the reference coordinates.
- Kept structural scheduling posterior-invariant with one mixed direction
  choice per opportunity and explicit unavailable-direction
  self-transitions. Root-total and persistent-fraction refreshes are separate
  kernels so the first benchmark isolates local structural mobility.
- Added a seeded structural-only sampler, variable-dimensional trace, and
  exact in-memory PCG64 continuation without changing the optimized Voronoi
  model, trace, or durable checkpoint schema. Detailed scope and validation
  gates are in `docs/plans/rjmcmc_gamma_beta_baseline.md`.
- Added opt-in structural proposal diagnostics without changing the transition
  kernel, retained-state schema, checkpoint schema, or default allocation.
  Diagnostics are enabled with
  `SamplerConfig.collect_structural_diagnostics` or the output-only
  `continue_sample` keyword.
- Recorded candidate and realized region-count changes, validity/reason,
  complete target and proposal-ratio accounting, stable nucleus event IDs,
  owner-changed cell count/fraction, affected design-column count, candidate
  prediction/design norms, and signed log coefficient contrast.
- Added segment endpoint nucleus sets, chronological concatenation, immediate
  opposite-`k` and exact-endpoint reversal flags, deletion lineage age, and
  complete left-/right-censored region-lineage and nucleus-cell residence
  intervals. Location moves transfer region lineages but close and reopen
  cell-residence intervals. Immediate means consecutive global atomic
  transitions, including across concatenated checkpoint segments.
- Added separate structural-transition xarray conversion and restoration for
  segment persistence, intrinsic chain/problem provenance, required
  run-metadata checks, and atomic segment-bound continuity checks.
  Region-lineage IDs are monotonic and retain a separate origin nucleus, so a
  reused cell ID cannot alias an earlier region. Observation-error
  standardization is explicitly distinguished from full covariance whitening
  under the inferred-OU model.
- Encoded invalidity reasons as compact integer categories and used
  overflow-resistant Euclidean norms, preserving the output-only contract for
  extreme finite cached predictions while keeping proposal-row storage below
  300 bytes in the regression case.
- Added focused trajectory-invariance, invalid/rejected/accepted outcome,
  target-decomposition, owner-identity, metric-oracle, restart-concatenation,
  residence/reversal, xarray, and NumPy/Numba parity tests.
- Wrote
  `docs/plans/rjmcmc_mixing_diagnostics_hpc_test_plan.md`, including paired
  diagnostics-off/on replay, the full-likelihood profiling report, an endpoint
  likelihood-power experiment, interpretation gates, and the next restricted
  fixed-direction Gamma--Beta split/merge baseline.

### 2026-07-22

- Created `codex/rjmcmc-correlated-hierarchy` from the committed incremental-
  geometry branch and kept the work in small target, proposal, schedule,
  checkpoint, manifest, and output commits.
- Added a normalized irregular-time independent-site OU likelihood with an
  explicit independent measurement nugget. NumPy and Numba scalar Kalman
  implementations agree with dense covariance oracles and run in O(N)
  observation time without materializing `R`.
- Added bounded-uniform inferred mismatch amplitudes and timescales, including
  cache-preserving one-coordinate Gaussian proposals and pointwise accepted-
  flux balance tests.
- Added one shared arithmetic mean/SD pair for all dynamic Voronoi
  coefficients. Its log coordinates have normalized lognormal-inspired
  hyperpriors, are counted once regardless of `k`, and persist unchanged
  through structural proposals. Fixed outer coefficients remain outside the
  pool.
- Added explicit 16-slot OU and 17-slot OU-plus-hierarchy schedules while
  preserving every existing schedule and seeded random stream. Exact
  continuation tests split chains at non-cycle-aligned boundaries.
- Bumped strict checkpoints to schema v3, fingerprinting all OU alignment,
  bounds, hierarchy settings, kernel scales, and optional state. Genuine v1/v2
  archive layouts and legacy problem hashes remain loadable for legacy targets.
- Bumped run manifests to schema v2 and added schedule profile, optional
  proposal scales, time units, bounded priors, and hierarchy parameterization.
  Labelled xarray output now includes mismatch groups, timescale parameters,
  log hierarchy coordinates, and derived arithmetic moments.
- Added deterministic prior-predictive contracts showing that the historical
  `mu_log=0,sigma_log=1` choice has arithmetic mean `exp(1/2)` and that IID
  aggregation concentrates country totals around that biased mean as the
  effective number of regions increases. Correct arithmetic mean/SD `(1,1)`
  maps to `mu_log=-0.5*log(2)`, `sigma_log=sqrt(log(2))`.
- Created `codex/rjmcmc-incremental-geometry` from the pushed
  `codex/tdmcmc-numba-rewrite` checkpoint `cd73231` while the unchanged HPC
  production chains continued running.
- Replaced complete state rebuilding for valid one-nucleus insertion,
  deletion, global-move, and local-move candidates with an exact incremental
  path. It remaps canonical labels by nucleus identity, updates ownership from
  the accepted source state, copies design columns whose membership is
  unchanged, and rebuilds affected columns in ascending global-cell order.
  Unsupported multi-edits and incompatible cache shapes still use the complete
  builder; an all-cells-affected candidate uses the direct full aggregation
  kernel to avoid the indirect-index worst case.
- Kept proposal probabilities, random draws, target evaluation, state schema,
  checkpoint schema, and the complete `build_state` oracle unchanged. Exact
  tests cover both numerical backends, two-dimensional geometry, canonical
  reordering and ties, empty regions, all four structural proposal routes,
  seeded-chain replay against the former full-rebuild path, and durable
  checkpoint continuation from an accepted incremental state.
- On a warmed Apple-arm64 synthetic case matching the current production
  dimensions (`H` shape `1382 x 23424`, `k_max=500`, dense float64), typical
  Numba structural-state construction was 5.8--56.1 times faster than complete
  rebuilding across `k=50,150,300`. The affected-design fractions ranged from
  0.25% to 13.73%; every timed candidate matched the full state bit-for-bit.
  Numba-compiling affected-region marking removed an approximately 6 ms Python
  overhead. Vectorizing the NumPy reference removed its initial low-`k`
  regression: assignment plus membership marking took 7--68% of complete
  NumPy assignment time for `k=5,10,50`. These are local diagnostic timings,
  not an HPC throughput claim.
- Deferred k-d tree, quadtree, and distance-transform assignment providers.
  Incremental assignment was already about 0.1 ms in a representative
  high-`k` case; affected sensitivity aggregation dominated instead. The
  internal structural-state construction boundary is the replacement seam for
  a future provider, which must retain canonical tie behavior and be recorded
  in kernel settings/checkpoints before use in replayable runs. A measured
  95--97% affected-cell aggregation crossover was also left out because it was
  hardware-dependent; only the exact 100% case is specialized.

### 2026-07-20

- Distinguished the demonstrated deterministic structural-schedule defect from
  the additional hierarchical dimension-matching defect. The latter is the
  precise sense in which the legacy RJMCMC treatment of per-region
  hyperparameters is incomplete: their prior terms alone do not define a
  reversible proposal for the extra dimensions.
- Promoted the hierarchy notes into a durable implementation specification,
  covering evidence versus decisions, normalized target terms,
  state/configuration contracts, forward and reverse proposal densities,
  schedule variants, checkpoint/xarray consequences, validation gates, and
  archived-data unknowns.

### 2026-07-19

- Pushed `codex/tdmcmc-numba-rewrite` and opened draft PR #508 against `devel`
  at the ten-commit checkpoint `3b463e6`.
- Split the first offline continuation into non-overlapping workstreams for the
  continuous-coefficient oracle, checkerboard example extraction, and run
  profile/provenance primitives. Faithful archived-input work remains deferred
  until the server data return.
- Moved the numerical package to
  `openghg_inversions.experimental.rjmcmc`, moved its focused suite to
  `tests/experimental/rjmcmc`, and updated the example and internal imports.
  The unreleased former namespace has no compatibility shim or import warning.
- Added independent continuous-coefficient balance validation: 64 varied
  pointwise forward/reverse checks plus three two-cell quadrature cases with up
  to 45.45% invalid negative-Gaussian proposal self-mass. No rewrite defect was
  found by these checks.
- Added immutable run profiles and canonical provenance manifests with explicit
  target settings, sampler settings, retention, seed, and code/input hashes.
- Added global-clock retained-draw collection and exact in-memory continuation.
  Checkpoints retain the PCG64 state, kernel and schedule identity, retention
  phase, transition count, and fixed predictor coefficients.
- Added an always-active predictor block throughout the numerical problem,
  proposal accounting, sampler, RHIME-style input adapter, and posterior
  prediction summaries. Runs without this block preserve the original seeded
  four-slot schedule exactly.
- Converted the NAME/EDGAR example from subtracting an assumed-known outer
  contribution to jointly inferring the seven InTEM outer-emissions factors.
  Oracle, non-oracle fixed, and trans-dimensional inner methods now use the same
  observations, outer design and priors, proposal opportunities, and total
  prediction RMSE. The corrupt boundary-condition fixture is still never read.
- Added strict atomic checkpoint serialization with numeric arrays and UTF-8
  canonical JSON only. It fingerprints the transformed numerical problem,
  hashes every state array, preserves exact PCG64/kernel/schedule/retention
  state, distinguishes the cache-construction backend from the configured
  kernel backend, and independently rebuilds caches before restoring them.
- Added an experimental xarray retained-trace boundary with global transition
  coordinates, padded dynamic slots and an active mask, and a distinct
  always-active parameter dimension. Attempted-transition diagnostics remain a
  separate future output because they do not align one-for-one with retained
  draws.
- Re-audited the next Lunt hierarchy stage. The paper describes a separate
  log-space `(mu_x, sigma_x)` pair for every active region, but later source
  inspection found that the near-submission Fortran actually accepts
  arithmetic mean/SD and converts internally. Neither source defines a valid
  dimension match for adding an independently evolving pair: the legacy
  parent-copy rule remains invalid. The first implemented hierarchy therefore
  uses one fixed-dimensional shared pair instead.

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
- Added the `openghg_inversions.experimental.rjmcmc` package with immutable
  numerical problem/state types, normalized target components, NumPy and Numba
  kernels, deterministic proposal accounting, and a seeded sampler with a
  fixed outer schedule and reversible mixed structural steps.
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
- Reviewed the 523-line data-backed checkerboard against draft PRs #502 and
  #506. It is a complete scientific workflow hidden in pytest, so the next
  organizational change should move its experiment-specific orchestration to
  `examples/rjmcmc/lunt_name_edgar_checkerboard.py`, add a replayable CLI and
  machine-readable provenance/results, and leave a much smaller test module
  that imports the example. The move should preserve current numerical behavior
  and should not create a new supported package API.

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
