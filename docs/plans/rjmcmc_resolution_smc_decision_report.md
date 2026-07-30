# Gamma–Beta coarse-to-fine resolution-SMC decision

## Recommendation

Stop this proposal design before medium R3 and do not start PARIS R4.

R0, the exact R1a screen, wider R1, and bounded R2 are correct and
characterized. They do not show a reproducible variance-per-cost advantage
that justifies scaling:

- R1a's best boundary-heavy bootstrap ratio was 1.66 relative to IID.
- None of 380 wider-R1 bootstrap cells beat direct IID on relative variance
  times median measured cost.
- Four of 18 R2 guide cells beat prior SMC, but none reached the provisional
  twofold target; the best ratio was 0.598.
- Wider R1 reached ESS fraction 0.02556 and as few as three unique ancestors.
- The R2 guide's maximum normalizer-audit discrepancy was 0.704, and guide
  construction cost generally removed its raw variance gain.

The scientific target \(Z\) is the fixed-root allocation-marginal normalized
native Gaussian likelihood. All primary comparisons use the
between-replicate variance of non-negative \(Z\) estimates, divided by the
squared oracle or reference value, times median measured wall time. Logs are
secondary reporting coordinates only.

## Established results

- Exact Gamma–Beta mass identities, local conditional means and covariances,
  child-swap equivariance, and terminal zero unresolved covariance pass.
- Direct IID and no-resampling SMC are pathwise identical for identical
  allocation paths.
- Two- and four-cell likelihoods agree with independently converged
  quadrature oracles. The 16-cell results use explicitly uncertain,
  replicated IID references.
- Checkpoint/restart is bitwise reproducible at R0 and R2 boundaries and
  rejects mismatched tree, schedule, input/guide, seed, particle, and source
  provenance.
- All production artifacts authenticate the exact clean, detached source SHA
  `abdce3c30c65aebd88c3c4f27c588c71aaabe2c2`.

If the method is revisited, redesign guidance so its construction is analytic
or amortized, then repeat the tiny exact experiments. Medium scaling is not
justified for the current bootstrap or piecewise-Beta proposal. PARIS R4 is
not justified because no viable R3 configuration has been demonstrated and
the current evidence trends against cost effectiveness.

No protected or realized-observation catalogue was accessed, and nothing was
written to `PARIS_inversions`.

The complete decision bundle is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc/abdce3c30c65aebd88c3c4f27c588c71aaabe2c2/report
```
