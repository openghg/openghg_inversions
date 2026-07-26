# MH-guided local partition search: matched synthetic test plan

## Purpose

This plan tests a finite-budget algorithm, not convergence of a partition
posterior. The algorithm starts from one reasonable fixed-\(K\) basis \(P_0\),
updates continuous parameters, and uses the existing Metropolis--Hastings
(MH) edge-flip and resolution-relocation transitions as posterior-targeted
stochastic local search. Its practical output is a cycle-residence-weighted
average of the native fields visited during a predeclared finite run.

The primary question is:

> Starting from the same data-conditioned state on \(P_0\), does adding local
> MH partition moves improve finite-budget held-out prediction and native-field
> recovery relative to holding \(P_0\) fixed?

This plan does not require random tilings, agreement between dispersed
partition starts, partition \(\hat R\), or production-grid connectivity. It
does not license Bayesian marginalization over partitions, a partition
posterior, posterior interval coverage, or structural inference over \(P\) or
\(K\).

## Terminology and historical distinction

- **Offline proxy SLS** means the historical initializer at local revision
  `1eae9ce`. It used sensitivities, nominal mass, diagonal errors, and
  Gamma--Dirichlet moments, but its artifact records
  `response_values_used=false`. It is design/error-informed and is not the
  method tested here.
- **MH-guided local search** means that the mobile sampler proposes local
  partition moves during the run and accepts or rejects them using the
  declared joint posterior target.
- **Finite-run average** means the cycle-residence-weighted average of all
  post-compound-cycle native fields in the predeclared production segment. It
  is an algorithm output, not an atomic-time average or a claim of converged
  posterior expectation.
- **Truth** means the planted positive native-cell scaling field used to
  generate synthetic observations. Truth, \(P_\star\), its witness, and
  held-out observations are forbidden from the practical fixed/mobile
  initialization, conditioning, proposal construction, acceptance, tuning,
  and best-state selection. A separately labelled oracle job may use
  \(P_\star\); the post-run analyzer may use the sealed evaluation artifact.

The existing \(2\times3,\ K=4\) exact enumeration establishes connectedness,
reversibility, and stationarity of the edge-flip plus relocation mixture on
that complete eight-tiling catalogue. It is not evidence that the production
tiling graph is connected.

## Frozen implementation boundary

Start from clean pushed revision
`17e24cee95ac32c2229d62acffb26f11ed39d282` on a dedicated branch/worktree.
The first implementation must reuse:

- `FullTilingCompoundConfig(structure_mode="fixed_basis"|"mobile")`;
- `sample_full_tiling_compound`;
- the exact root slice and pair-allocation refresh kernels;
- the existing edge-flip and resolution-relocation MH kernels; and
- `FullTilingMovementDiagnostics`.

Do not change the sampler core unless a focused correctness gate proves that
the existing public interface cannot express the experiment. With no fixed
outer block, one fixed cycle contains one root slice and five allocation-pair
updates. One mobile cycle contains those same six continuous opportunities
plus two structural MH attempts. Equal cycles therefore means equal
continuous-update opportunities, not equal wall time.

Do not weaken checkpoint schedule identity or resume a fixed-basis checkpoint
as a mobile checkpoint. The experiment runner must condition on \(P_0\) and
fork the resulting immutable state in memory, or use a new purpose-specific
audited warm-state artifact.

Construct an oracle prior-mean state on arbitrary \(P_\star\) only in the
quarantined oracle/reference code, using canonical nominal leaf shares,
`TilingState`, and `build_full_tiling_posterior_state`. Do not change the
core initializer or existing fixed-basis NUTS CLI.

## Synthetic model

Use the existing fixed-\(K\) Gamma-root/Dirichlet-share full-tiling target:

- \(T\sim\operatorname{Gamma}(4,4)\);
- globally additive Dirichlet allocation with concentration \(2K\);
- positive nominal weights normalized by the existing adapter;
- fixed independent Gaussian observation errors;
- likelihood power one;
- no outer coefficients, boundary term, aggregation error, hierarchy, or
  inferred error parameter.

The initial topology \(P_0\) is the deterministic `largest-nominal` tiling.
For uniform nominal weights this is a balanced deterministic basis, not a
random draw and not a structural prior.

Every truth field must:

- be constant within the leaves of its planted topology \(P_\star\);
- have distinct positive leaf scalings;
- have nominal-weighted mean exactly one; and
- be reconstructed and hashed on the common native grid.

In canonical planted-leaf order, assign raw scaling

\[
s_j=\exp\{-0.7+1.4j/(K-1)\},\qquad j=0,\ldots,K-1,
\]

then divide all \(s_j\) by their nominal-mass-weighted mean. This formula and
canonical order are frozen for every scenario.

Create \(P_\star\) by explicit deterministic geometry witnesses:

1. `aligned`: \(P_\star=P_0\);
2. `edge-one`: exactly one recorded valid edge flip from \(P_0\);
3. `relocation-one`: exactly one recorded valid resolution relocation from
   \(P_0\); and
4. `short-path`: a later two-to-four-move witnessed path, allowed only after
   the one-move atmospheric cases pass.

The witness contains the ordered canonical rectangle bounds and move
catalogue indices. A bounded breadth-first search may certify local distance
for these planted cases. Do not infer global connectivity.

For `edge-one`, select the first valid edge-flip path in the existing exact
catalogue order whose destination differs from \(P_0\). For
`relocation-one`, select the first valid resolution-relocation path in exact
catalogue order whose destination differs from both \(P_0\) and the selected
edge destination. Every tie is resolved by existing canonical rectangle,
axis, and split ordering. The code must emit the resolved indices and
topology hashes before generating any observations.

## Common-state conditioning and branching

For each data replicate:

1. build \(P_0\) and its prior-mean continuous state;
2. run the fixed-basis compound sampler for the declared conditioning cycles
   using training observations only;
3. record the exact final-state fingerprint;
4. start fresh fixed and mobile production segments from that exact immutable
   state and fingerprint; and
5. use predeclared, arm-specific PCG64 seeds.

The practical sampler receives an immutable checksum-addressed **training
artifact only**: training operator, noisy training observations, error vector,
nominal weights, \(P_0\), and the conditioned branch state. Its CLI must have
no truth, held-out, \(P_\star\), or witness argument. Sampling runs in a
directory where the evaluation artifact is unavailable.

A separately sealed **evaluation artifact** contains the native truth,
noiseless and noisy held-out values, held-out operator, \(P_\star\), and
witness. Only the oracle-reference runner and post-run analyzer may open it.
A definition manifest may commit to the hashes of both artifacts without
making the evaluation payload available to the practical sampler. The fixed
and mobile manifests must prove that they share the same training artifact,
branch-state fingerprint, and continuous model.

## Residence-weighted estimator

Retain every complete cycle boundary. Do not thin numerical summaries. For
post-cycle native fields \(x_1,\ldots,x_B\), the primary output is

\[
\bar x_B = B^{-1}\sum_{t=1}^{B}x_t.
\]

Repeated complete cycle-boundary states and repeated topologies count each
time they occur. A rejected structural proposal can still be followed by
continuous changes in the same compound cycle, so this is not literal
atomic-transition residence time. Never de-duplicate topologies, average
unique partitions uniformly, or average leaf indices. Reconstruct and average
common native-grid fields and common native-grid projections.

Also report:

- the cumulative mean trajectory;
- the predeclared last-50% mean;
- the final state; and
- the best retained-cycle state selected by training log target only.

Best and final states are secondary diagnostics. Truth and held-out values
must not participate in their selection.

## Scoring

Use an independent held-out operator and held-out noiseless truth prediction.
Held-out noisy observations may be generated for predictive log scores but
must not enter fitting or tuning.

For held-out operator \(H_h\), native truth \(x_\star\), and all-cycle mean
\(\bar x\), the primary metric is

\[
\operatorname{RMSE}_h(\bar x)=
\sqrt{n_h^{-1}\lVert H_h\bar x-H_hx_\star\rVert_2^2}.
\]

Every mobile/fixed ratio is mobile RMSE divided by its paired fixed-arm RMSE,
so values below one favor mobile.

Primary metric:

- held-out noiseless prediction RMSE of the all-cycle finite-run mean.

Secondary metrics:

- last-50% held-out prediction RMSE;
- held-out Gaussian mixture log score
  \(\sum_r\log[B^{-1}\sum_t
  \mathcal N(y_{h,r}\mid(H_hx_t)_r,\sigma_h^2)]\), evaluated stably by
  log-mean-exp;
- nominal-weighted native scaling RMSE
  \(\sqrt{\sum_iw_i(x_i-x_{\star,i})^2/\sum_iw_i}\);
- nine common native-grid totals: whole domain, top and bottom halves, left
  and right halves, and the four quadrants, using half-open masks and nominal
  weights;
- best-training-target and final-state scores;
- valid and accepted structural proposals by move;
- structural MH ratios and standardized prediction displacement;
- same-cycle cumulative mobile-versus-fixed score curves and the first cycle
  where mobile is below the paired fixed cumulative score, as descriptive
  non-causal diagnostics;
- unique topology hashes, residence fractions, and return to \(P_0\);
- witnessed local distance/hitting time to \(P_\star\); and
- cycles per second and useful continuous opportunities per wall hour.

Never compare absolute log-target values across the fixed and mobile modes.
An accepted atomic structural proposal is associated only with the retained
boundary at the end of its containing compound cycle; no later improvement is
attributed causally to that move.

Equal-cycle results are primary. Run production in fixed 100-cycle
sample/continue chunks and record `perf_counter` wall time around the sampler
call only. For an equal-wall sensitivity, truncate both completed traces to
the largest complete 100-cycle prefixes whose cumulative sampler-call time
does not exceed the smaller arm time. Never use setup, analysis, checkpoint,
or serialization time in that budget, and never stop a run adaptively.

## R0: implementation and reference gates

Before scientific runs, require:

1. a definition-only command that resolves all topology paths, operators,
   truth fields, seeds, and exact hashes into a reviewed manifest before
   adding observation noise or running a sampler;
2. exact separation of training and held-out operator row identities;
3. exact \(P_0\), \(P_\star\), and move-witness replay;
4. exact common branch-state identity across fixed and mobile arms;
5. fixed topology invariance and two structural attempts per mobile cycle;
6. no evaluation-artifact path or truth/held-out/\(P_\star\) access in the
   practical sampler interface, process directory, initialization, sampling,
   or selection code;
7. native-field reconstruction parity with the independent rectangle oracle;
8. residence-weighted averaging including deliberate duplicate states;
9. exact sample/continue parity for each structure mode without cross-mode
   checkpoint reuse;
10. the existing tiny uniform transition-matrix test plus exhaustive
    pointwise forward/reverse MH-flow equality and log-ratio antisymmetry for
    every tiny-catalogue path, using a nonconstant likelihood and at least
    three interior allocation fractions; and
11. schema, corruption, and checksum fail-closed tests for scientific
    artifacts.

Run only focused experimental pytest, Ruff, format, and Pyright checks. Do not
run repository-wide tox.

For only the first noise replicate of each scenario, compare the local
continuous sampler on \(P_0\), and separately on \(P_\star\), with a
fixed-basis NumPyro NUTS reference. The aligned case has only one unique
topology. Different noise replicates are different targets and are never
pooled as chains. The reference uses the exact same target, float64, and
training data.

Use four NUTS chains with 1,000 warmup and 1,000 retained draws per chain,
diagonal mass, `target_accept=0.90`, and maximum tree depth 10. Use a
prior-mean start for chain zero and prior draws from PCG64 seeds
`64101..64103` for the other chains in S0 and `74101..74103` in S1/S2. Use
NumPyro sampler seeds `64100` and `74100`, respectively. If NUTS fails, the
only permitted retry uses 2,000 warmup and 2,000 retained draws,
`target_accept=0.95`, and maximum tree depth 12 with all other choices
unchanged.

For the same representative target and topology, run four local fixed-basis
chains from the audited conditioned state using seeds `64201..64204` in S0
and `74201..74204` in S1/S2. Production length matches that stage. Estimate
each common projection's local MCSE using four-chain batch means with 20
equal contiguous batches per chain. Require:

- zero divergences;
- rank-normalized \(\hat R\leq1.01\) for the root total and every active leaf
  mass of that fixed topology;
- bulk and tail ESS at least 200 for those coordinates; and
- local projection MCSE at most `0.05` NUTS posterior SD and first-half versus
  second-half local projection means within `0.10` NUTS posterior SD; and
- local-sampler late-window common projections within
  \(\max(0.05\) reference posterior SD, \(3\) combined MCSE\()\) of NUTS.

If the retried NUTS reference still fails, the cell is inconclusive and the
structural comparison stops. If NUTS passes but the fixed local sampler fails
the projection or MCSE gate, first increase conditioning, production, and
local-reference lengths by the single predeclared factor of four. Persistent
failure is an inconclusive continuous-kernel hard stop, not a local-search
failure.

## S0: strong-signal mechanism screen

Use a \(2\times4\) grid at \(K=4\).

- Training operator: the eight direct native-cell rows, each repeated twice,
  permuted once by PCG64 seed `60000`.
- Held-out operator: the ten equal-weight averages over every horizontally or
  vertically adjacent cell pair, permuted once by PCG64 seed `60001`. No
  held-out row is identical to a training row.
- Observation SD: `0.05` in scaling units.
- Data-noise seeds: `61001, 61002, 61003, 61004`.
- Conditioning seeds: `61501, 61502, 61503, 61504`.
- Oracle-conditioning seeds: `61601, 61602, 61603, 61604`.
- Conditioning cycles: `2_000`.
- Production cycles: `5_000`.
- Sampler-seed pairs: fixed `62001..62004`, mobile `63001..63004`.
- Oracle sampler seeds: `62501..62504`.
- Scenarios: `aligned`, `edge-one`, `relocation-one`.

Each production trace retains all 5,000 post-cycle boundaries. The oracle
fixed-\(P_\star\) reference uses the same conditioning and production budget,
plus the R0 NUTS reference.

S0 operational gates:

- every artifact is finite and complete;
- every mobile replicate has at least one valid proposal of each structural
  move;
- every misaligned mobile replicate has at least one accepted structural
  move;
- every one-move scenario has at least three of four mobile replicates visit
  \(P_\star\); and
- fixed and oracle conditional reference gates pass.

Before judging the mobile arm in either misaligned scenario, require that the
oracle fixed-\(P_\star\) all-cycle primary RMSE is below fixed \(P_0\) in at
least three of four paired replicates and that the median oracle/fixed ratio
is at most `0.80`. Failure is an unlearnable/inconclusive planted cell, not a
mobile-search failure.

S0 utility gates:

- `aligned`: median mobile/fixed primary held-out RMSE ratio at most `1.10`,
  with no replicate above `1.25`;
- each one-move scenario: median mobile/fixed primary held-out RMSE ratio at
  most `0.90`, at least three of four paired ratios below one, and median
  mobile/fixed native-field RMSE ratio at most `0.95`.

Failure of proposal construction, or of acceptance in a misaligned case, is
a mechanism hard stop.
Failure of the utility gate with operational and conditional-reference passes
is a finite-budget local-search failure for that scenario. Do not alter the
truth, noise, or move witness after inspecting results.

## S1: atmospheric-like synthetic screen

Run only if every S0 operational/reference, oracle-learnability, and utility
gate passes, including aligned non-degradation.

Use an \(8\times8\) grid at \(K=8\):

- 96 training Gaussian-footprint rows;
- 48 independently seeded held-out Gaussian-footprint rows;
- native-cell centers at \((r+0.5,c+0.5)\);
- footprint centers sampled independently and uniformly on `[0, 8]` in both
  coordinates;
- widths uniform on `[0.55, 1.35]` cell lengths;
- unnormalized cell amplitude
  \(\exp(-((r+0.5-\mu_r)^2+(c+0.5-\mu_c)^2)/(2w^2))\), then normalize each
  row using `math.fsum` in canonical cell order and require its float64 sum
  to agree with one within eight ULP;
- fixed observation SD `0.08`;
- operator seeds: training `71000`, held-out `71001`;
- paired data-noise seeds: `71101, 71102, 71103, 71104`;
- conditioning seeds: `71501, 71502, 71503, 71504`;
- oracle-conditioning seeds: `71601, 71602, 71603, 71604`;
- conditioning cycles: `10_000`;
- production cycles: `50_000`;
- sampler seeds: fixed `72001..72004`, mobile `73001..73004`;
- oracle sampler seeds: `72501..72504`;
- allocation-pair refreshes per cycle: five;
- scenarios: `aligned`, `edge-one`, `relocation-one`.

There is no pair-refresh pilot and no scientific tuning after artifacts are
generated. Apply the same five continuous pair-refresh opportunities to both
arms.

S1 gates are the S0 gates with these changes:

- visiting \(P_\star\) is diagnostic, not required;
- in each misaligned scenario, oracle \(P_\star\) must beat fixed \(P_0\) in
  at least three of four replicates with median oracle/fixed primary ratio at
  most `0.90`, otherwise the cell is inconclusive;
- `aligned`: median mobile/fixed primary RMSE ratio at most `1.10`;
- each one-move scenario: median primary held-out RMSE ratio at most `0.95`,
  at least three of four paired ratios below one, median native-field RMSE
  ratio at most `0.98`, and no paired held-out ratio above `1.20`.

If S1 passes, the permitted claim is limited to finite-budget improvement for
the planted local scenarios. Do not call the finite-run average a converged
partition-marginal posterior.

## S2: witnessed short-path screen

Run only if every S1 operational/reference, oracle-learnability, and utility
gate passes, including aligned non-degradation and both one-move scenarios.

Construct one \(8\times8,\ K=8\) truth whose planted topology is exactly three
witnessed legal moves from \(P_0\), with at least one edge flip and one
resolution relocation. At each step choose the first exact-catalogue valid
path that reaches a topology not already on the witnessed path. Exhaustively
expand only BFS depths zero, one, and two to prove that the destination is not
closer; do not exhaustively expand depth three.
Reuse the frozen S1 operators, noise seeds, continuous schedule, conditioning,
and paired sampler seeds. Increase production once, predeclared, to
`100_000` cycles.

Run the S2 oracle and representative fixed-topology NUTS/local reference at
the same S2 length. Require the same continuous-reference gates and an oracle
learnability gate of at least three of four oracle/fixed ratios below one
with median at most `0.90`.

Pass if the median mobile/fixed primary held-out RMSE ratio is at most `0.95`,
at least three of four paired ratios are below one, the median mobile/fixed
native-field RMSE ratio is at most `0.98`, and the aligned S1 non-degradation
gate remains satisfied. Hitting \(P_\star\) is secondary.

No further search radius, tuning grid, or real-data phase is authorized by
this plan.

## Failure localization

At the first failed hard gate, preserve artifacts and localize the failure:

1. **continuous conditioning:** fixed local sampler disagrees with NUTS;
2. **proposal availability:** required edge-flip/relocation proposals are not
   valid from visited states;
3. **MH landing:** proposals are valid but acceptance is effectively zero;
4. **movement without utility:** moves are accepted but the finite-run average
   does not improve training-independent metrics;
5. **averaging effect:** best/final states improve but residence-weighted
   averaging does not; or
6. **compute penalty:** equal-cycle improvement disappears under equal wall
   time.

Do not repair, retune, or expand a frozen scientific grid after inspecting
truth or held-out results. A source or harness defect requires a reviewed,
committed, pushed revision and a fresh full-SHA run root.

## BP1 execution and provenance

- Check login-node load and available memory first; keep aggregate login-node
  RSS below 200 GB.
- R0 and bounded sequential S0 dry runs may use the private login node while
  quiet.
- Use Slurm under account `chem007981` for the S1/S2 replicate matrix or any
  retained multi-chain NUTS reference.
- Load `git/2.45.1-pqk5` on compute nodes when Git provenance is required.
- Use the frozen Pixi `dev` environment and record the `pixi.lock` SHA-256,
  Python, NumPy, PyMC, PyTensor, NumPyro, platform, and BLAS identities.
- Build immutable launch, analysis, and reporting harnesses before science.
- Record source SHA, clean status, commands, seeds, job IDs, resource use,
  exact last successful gate, and every input/output SHA-256.
- Preserve partial and failed artifacts. Write `complete.json` last and only
  after independent checksum validation of a passing stage.
- Write nothing to `PARIS_inversions`.

## Reporting boundary

A successful report may say:

> At the predeclared finite budget, MH-guided local partition movement from
> \(P_0\) improved paired held-out prediction and native-field recovery
> relative to keeping \(P_0\) fixed for the four predeclared paired seeds and
> tested planted truths within the witnessed local move radius.

Four paired replicates bound the claim to those seeds and cases; they do not
establish population-level superiority.

It must not claim:

- convergence or correct Bayesian marginalization over partitions;
- consistent recovery of \(P_\star\);
- connectivity or irreducibility on the production tiling space;
- posterior uncertainty coverage;
- support outside the component reachable from \(P_0\); or
- improved real-data flux accuracy.
