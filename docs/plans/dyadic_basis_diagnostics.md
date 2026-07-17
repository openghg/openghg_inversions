# Dyadic Basis Diagnostic Experiments

## Purpose

This operational note follows the mathematical background in
`docs/reports/rhime_bocquet_reduced_gaussian.md`. It separates three questions
that were confounded in the first SLS demonstration:

1. does stochastic local search find a good partition in its own dictionary;
2. does that dictionary compare well with an existing basis algorithm; and
3. is the TAC/MHD input and error model suitable for an inversion?

All work remains under `basis.experimental.dyadic` or `examples/basis`. It does
not change production RHIME, `run_hbmcmc.py`, or `fixedbasisMCMC`.

## Terminology correction

The current demo's command-line search block width of 8 means an
\(8\times8\) sum-preserving spatial coarsening. The native 293 by 391 grid is
represented by a 37 by 49 search grid, with partial edge blocks retained.

This is distinct from all of the following:

- the private prototype name `dyadic8`, which means dyadic depth 8 and a
  \(2^8\times2^8\) finest padded grid;
- the private spatial precomputation, whose tensor-product dyadic array is
  about four times the padded finest spatial grid, or about 9.1 times the
  unpadded 293 by 391 grid after padding to 512 squared; and
- Bocquet, Wu, and Chevallier's up-to-eightfold storage statement for a fully
  precomputed hierarchy that is multiscale in two spatial dimensions and time.

Future reports should say **8x8-block search grid** rather than only
**factor-8**.

## Prototype search provenance

The private stochastic search did not calibrate temperature to a 50% overall
acceptance rate. It used fixed-temperature chunks, often 10,000 or 100,000
iterations, and manually explored temperatures around 0.001, 0.002, 0.005,
0.01, and sometimes 0.02 under its historical quadratic tile score.

The main target was approximately 250 regions. The above-target linear penalty
was also hand tuned against observed successful split gains. Recorded values
include 0.58 for the depth-8 padded problem and roughly 0.10 to 0.15 for the
depth-9/full-domain problem. The final transition logic approximated
\(\lambda\max(0,K-250)\), but missed the reward on the threshold-crossing
merge from 251 to 250. This provenance supports tuning a penalty on the scale
of local gains, but the exact prototype rule should not be copied as a
probabilistic prior.

The current repository calibration is newer: it takes the median positive
pilot loss \(L\) and sets \(T=-L/\log p\). A setting of `p=0.5` means the
median pilot loss has 50% acceptance at the initial temperature. It does not
mean that half of all proposals should be accepted.

## Implemented diagnostics

### Exact emissions-only additive oracle

`basis.experimental.dyadic.dynamic_programming` computes the global optimum
for every fixed region count in a canonical dyadic tree. It is the correct
reference for the present emissions-only projected-Gaussian DFS score when the
baseline is known or has already been removed. It is not an oracle for the
joint problem with uncertain boundary coefficients.

The current implementation keeps complete active-node tuples for each
subproblem and examines feasible left/right count pairs. This is convenient and
fast at the tested \(K\) values, but its worst-case work is quadratic in the
region limit per internal node. A larger production solver should store scores
and compact backpointers.

### Arbitrary labelled partitions

`basis.experimental.dyadic.partition_diagnostics` evaluates any positive
integer-labelled search-grid partition under the same projected prior as the
dyadic model. It provides:

- regional summed design columns and \(\tau^2/n_k\) variances;
- directly accumulated aggregation covariance;
- innovation-covariance closure checks;
- projected DFS;
- an emissions-only precision-weighted compression score; and
- an analytic Gaussian posterior mean with an optional explicit baseline
  design and training subset.

This permits a like-for-like score for the existing quadtree output without
claiming that quadtree and dyadic search use the same partition dictionary.

### Synthetic inversion experiment

Run:

```console
python examples/basis/dyadic_basis_diagnostics.py
```

The experiment uses the aligned one-day TAC/MHD native emissions sensitivity
and the frozen 32-column boundary sensitivity. It holds out 12:00 through
17:00 UTC at both sites, constructs bases from training rows only, and compares
at requested \(K=31\):

- greedy dyadic splitting;
- exact dyadic dynamic programming;
- fixed-count SLS with representative-loss acceptance targets 0.5 and 0.1;
- the existing quadtree algorithm, constructed from its existing cellwise
  precision-weighted proxy and then evaluated with the common projected score;
  and
- the native no-reduction Gaussian reference.

Three synthetic truths are evaluated: the prior mean, a seeded Gaussian draw,
and a smooth non-basis-aligned scaling field. The baseline is represented by
the actual frozen boundary design, not by an intercept hidden in the error
covariance. Results are repeated for observation-only error, additive 5, 10,
and 20 ppb mismatch, and the existing percentile-floor rule recomputed from
training rows only.

For this one-day split, that legacy rule produces 9.48 ppb at MHD and 4.21 ppb
at TAC, not 43 ppb. These still exceed the respective median supplied errors of
3.17 and 1.97 ppb. The much larger value seen in the earlier setup demonstrates
that this empirical range statistic is dataset-dependent; it remains a
sensitivity case rather than a recommended model-error specification.

Artifacts are in `docs/plans/figures/dyadic_basis_diagnostics/`.

## Initial findings

Under observation error plus 5 ppb mismatch and the smooth truth:

- exact DP training DFS is 0.252226 at \(K=31\);
- greedy DFS is 0.252223, within \(3.2\times10^{-6}\) of the optimum;
- neither 2,000-step SLS schedule improves the greedy state in this run;
- existing quadtree DFS is 0.170797 at an actual and effective \(K=31\);
- held-out compression quality is 0.0887 for DP, 0.0885 for greedy, and
  0.0712 for quadtree; and
- known-baseline held-out emissions RMSE is about 6.58 ppb for DP/greedy and
  6.69 ppb for quadtree.

These are proof-of-concept numbers, not a ranking of production algorithms.
The quadtree gap reflects both a different construction objective and a
different partition dictionary. The synthetic metric is based on only 47
observation rows.

The blocked holdout also exposes a baseline limitation. Some boundary
directions used in the held-out time block have no training support, so their
posterior means remain at the prior. Total modeled-mole-fraction RMSE is
therefore dominated by boundary extrapolation. Emissions RMSE and compression
quality must remain separate from total RMSE.

## Real-data consistency gate

The stored one-day real observations should not yet be inverted in this
experiment. At unit emissions and boundary coefficients:

- prior emissions range from about 2.8 to 49.7 ppb;
- prior boundary contribution ranges from about 95.9 to 1772.0 ppb; and
- the observation-minus-prior residual ranges from about 128.5 to 1846.0 ppb,
  with RMSE about 589 ppb.

This is far beyond the stored observation errors and floors. The late-TAC
boundary collapse needs to be understood before using these rows for a real
posterior or observation holdout score.

## Scale and performance check

A separate full-week fixed-\(K=250\) run on the 8x8-block search grid used
10,000 paired proposals with an initial representative-loss acceptance target
of 0.1. The common Gaussian model took 0.79 seconds to construct. Measured
end-to-end and partition-solver timings were:

- SLS: 68.1 seconds end-to-end, including a model rebuild, greedy
  initialization, temperature pilot, and search;
- exact DP: 1.09 seconds end-to-end using the common model construction, of
  which 0.30 seconds was the partition recurrence;
- accepted proposals: 3023 of 10,000;
- greedy initializer DFS: 0.490334523;
- SLS best DFS: 0.490334613; and
- exact DP DFS: 0.490338939.

The candidate-level timing fields in the generated manifest have explicit
scope strings and should not be compared without those scopes. The scale check
shows that the current code can search at the prototype's region count, but
this additive Gaussian objective is both easier and much faster to solve
exactly. SLS performance becomes the relevant research target only for a
non-additive score or constraints that break the tree recurrence.

## Next experiments

1. Add a fixed \(K\) sweep over 16, 31, 64, and 250. Report actual/effective
   \(K\), DP gap, compression, and runtime.
2. Repeat at search block widths 8, 4, and 2 to separate representation loss
   from optimizer loss.
3. Add the axis-parallel constrained/greedy basis to the arbitrary-label
   comparison, followed by bucket only when its land/sea inputs are explicit.
4. Use the week fixture for emissions-only blocked compression scores. This
   score does not require observed mole fraction or a baseline.
5. Diagnose and reconstruct valid boundary contributions before fitting the
   stored real observations or reporting total holdout prediction.
6. Keep diagonal observation-plus-fixed-mismatch sensitivity as the bounded
   extension of existing models. Defer correlated error until the basis
   experiment demonstrates a need that thinning and blocked holdouts cannot
   expose.
