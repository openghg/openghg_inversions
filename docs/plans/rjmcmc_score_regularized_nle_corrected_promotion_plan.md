# Corrected score-regularized marginal NLE promotion experiment

## Scientific question

Can one learned conditional density approximate the marginal likelihood of
the same native Gamma model after each of the six public fixed linear
projections? The exact marginal is invariant to computational partition and
retained rank. Approximate evidence differences are leakage diagnostics, not
structural information, and are never used as basis weights.

This is a public synthetic experiment. It does not read PARIS mole fractions,
the protected catalogue, or `PARIS_inversions`.

## Frozen candidate and comparator

The candidate is the complete `fisher_observation_joint` training algorithm,
not one realized initialization. It uses four fixed starts and selects the
start with the smallest independent model-selection NLL per retained
dimension; an exact tie selects the lowest initialization index.

`nll_only` is a mandatory development comparator under the identical
four-start rule. It is not a fallback candidate. Reporting data and oracle
evidence are not evaluated until the start selection has been fixed.

Every matrix freezes:

- learning rate `5e-4`;
- batch size `1024`;
- projection microbatch `64`;
- maximum 40 epochs;
- patience 6;
- four initialization indices `0,1,2,3`.

Each attempt separately hashes its locked package/runtime identity and its
execution identity. The latter includes JAX/XLA/compiler and thread flags,
the outer reverse parameter gradient over the forward-mode JVP used by both
score routes, canonical artifact policy, and shared-node array scheduling
policy.

The score candidate must be reporting-NLL non-inferior to NLL-only and improve
observation-score risk by more than five pooled IID MCSEs on both independent
model-selection and reporting domains.

## All-six oracle v2

The promotion oracle is a new all-six v2 bundle. The historical three-case v1
bundle and its preserved E2 artifacts remain replayable and unchanged.

Primary endpoint-aware allocation ladders are frozen as:

- near-Gaussian two-cell: `16,32`;
- near-Gaussian four-cell: `8,12,16`;
- skewed two-cell: `8,16,32`;
- skewed four-cell: `8,12,16`;
- boundary-heavy two-cell: `16,32,64`;
- boundary-heavy four-cell: `12,16,24`.

Every primary route gates evidence, posterior mean, SD, quantiles, retained
prior/posterior mass, mode inclusion, and normalizer/moment/CDF integration
errors. Four-cell references additionally compare row-first and column-first
Dirichlet charts. Skewed two-cell compares an independent native-log-mass
route. Boundary-heavy two-cell retains its native-log-mass certificate.
Boundary-heavy four-cell additionally uses fixed Gauss-Legendre integration
in log total at orders `512,1024,2048` with a column-first allocation chart.
Every hard gradient metric is preflighted over allocation order and
log-total finite-difference steps `2^-12,2^-13,2^-14`; the learned gradient
error must also be stable over the final two steps within `0.005`.

Before fitting, each exact metric grid must pass at `4096,8192` equal-prior-
probability bins. Posterior quantiles interpolate within the crossing bin
under the grid's piecewise-constant likelihood convention. Evidence,
posterior moments, posterior quantiles, pointwise errors, and gradients have
separate interpretability flags; one discretized quantile failure cannot
invalidate an otherwise sound exact oracle or suppress converged metrics.

## Frozen matrices

Development uses one public simulator seed at both sizes:

- `promotion_development_s4096`: all six cases, candidate and comparator,
  four starts, base seed `1731`, 48 array tasks;
- `promotion_development_s16384`: the same matrix at `S=16384`, base seed
  `1731`, 48 array tasks.

Only if the exact two-summary development certifier passes may confirmation
run:

- `promotion_confirmation_s16384_seed2731`;
- `promotion_confirmation_s16384_seed3731`;
- `promotion_confirmation_s16384_seed4731`.

Each confirmation matrix contains the candidate only, all six cases, and four
starts: 24 array tasks per seed. All repeated work runs as homogeneous Slurm
arrays on shared nodes.

## Frozen scientific gates

For every selected candidate artifact:

- prior-weighted median absolute log-likelihood error: at most `0.05` nat;
- exact-posterior-weighted p99 absolute log-likelihood error: at most
  `0.20` nat;
- scaled retained-mass gradient error: at most `0.05`;
- absolute log-evidence error: at most `0.05` nat;
- posterior mean error: at most `0.05` reference SD;
- posterior SD relative error: at most `0.02`;
- interval endpoint error: at most `0.05` reference SD;
- finite complete-grid likelihood and held-out risks;
- normalized density by flow/Jacobian construction;
- model-selection/reporting NLL agreement within
  `max(0.02*q nat per draw, 5*pooled IID MCSE)`.

The development pair must also pass common all-six cross-size stability:

- prior-weighted median absolute difference between selected `S=4096` and
  `S=16384` learned log likelihoods: at most `0.05` nat;
- exact-posterior-weighted p99 difference: at most `0.20` nat.

The accompanying evidence difference is recorded only as a leakage
diagnostic.

## Execution and stopping

1. Commit and push this complete plan and executable source.
2. Resolve the exact full origin SHA.
3. Create a new clean detached full-SHA source and a fresh run root.
4. Run the all-six oracle first. Stop if it does not publish a passing
   exact-byte completion marker.
5. Run the two development arrays, then their mergers, then the pure
   two-summary certificate.
6. Stop on any scientific development miss. Do not tune weights, thresholds,
   sizes, starts, or architectures from those results.
7. Only after a passing development certificate, run all three confirmation
   arrays, their mergers, and the five-summary final certificate.
8. A final miss ends this frozen candidate. It does not authorize evidence-
   weighted bases or protected-data access.

Measured E2 resource use sets initial shared-node requests:

- `S=4096`: 3 GiB and 12 minutes;
- `S=16384`: 5 GiB and 20 minutes;
- oracle: 2 GiB and 45 minutes;
- promotion merger: 3 GiB and 45 minutes;
- cross-summary certifier: 3 GiB and 30 minutes.

Use `slurm-wakeup` for every long job. Preserve failures and publish
completion markers last.
