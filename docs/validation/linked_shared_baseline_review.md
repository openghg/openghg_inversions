# Linked shared baseline: independent scientific review

Reviewed scientific implementation:
`3e065f31c8e732c2f553827c7f3cb794f6762300`, on 2 November 2026.
The additional source-label validation fix
`4ac74b9707df249394500468e85fb343798a997f` was independently spot-reviewed
the same day. It normalizes source names before the outer-ocean guard,
matching the existing linked preparation's case-insensitive source handling.
It changes no forward equation, prior or covariance operation.
The bounded independent tests are in
[`tests/test_linked_shared_baseline_review.py`](../../tests/test_linked_shared_baseline_review.py).
This record supports the prepared-input model API handoff. It does not certify
the later acquisition, configuration, campaign sampling or generic product route.
The [design and acceptance record](linked_shared_baseline.md) owns those stages.

## Equations checked independently

The oracle starts from each channel's original sensitivity, fixed affine
contribution, period-gated boundary correction and original prior mean. It does
not evaluate the transformed shared-baseline operator to obtain its expected
mean. With `a` the frozen equal-site weights, it computes

```text
mean_centred = fixed + H_inner x_inner + H_outer m_outer
             + (H_adjustable - 1 (a.T H_adjustable)) (x_adjustable - m_adjustable)
             + A 1 + c (M 1 - 1)
             + delta + (D_BC - 1 (a.T D_BC)) v.
```

The adjustable term is absent for fixed outer fluxes. Shrinkage uses the same
forward equation with a resolved positive gross-biosphere prior calibration.
The fixture has two CO2 observations and three O2 observations, distinct native
labels and site supports, interleaved joint rows, two tracer-specific outer
ocean states, a nonzero coherent affine term, and joint aggregation covariance
with nonzero cross-channel entries.

A dense Gaussian oracle adds independent measurement variance and separate
species/site Ornstein–Uhlenbeck blocks to that joint covariance. Its normalized
likelihood and derivatives with respect to every cached state value agree with
both the ordinary observed likelihood and the refreshed
`cached.cached_state_logp` expression used by the matched sampler. This covers
`fixed`, `multiplicative` and `multiplicative_shrunk` at nonzero points.
The comparisons use relative tolerance `1e-10` and absolute tolerance `2e-8`.
The graph contains one sampled flux latent shared by both tracers.

## Precision, ownership and persistence checks

- The O2 reference is 209300 ppm; CO2 uses 400 ppm. Native anomalies are of
  order `1e-10` ppm and mass responses differ from one by order `1e-12`.
  The bound correction equals `A 1 + c (M 1 - 1)` element for element. Physical
  delta widths remain 0.003 ppm and 0.4 ppm respectively. A linked artifact
  missing native centred operands fails explicitly.
- Both model paths accept a level-only boundary with no pattern state.
  Unsupported collapsed additive biosphere corrections fail. An independent
  rank calculation shows why unequal GPP/TER oxidative ratios preserve an O2
  response along a gross direction invisible to CO2 net flux.
- A shared delayed transport payload executes once across the two channel
  preparations at the model boundary. Borrowed lazy inputs remain lazy.
- Save/reopen preserves frozen native datasets. A subsequent observation
  subset preserves the full-support column means, cached design and affine
  contribution exactly; no subset centring is performed.
- Inner ocean states remain in the joint sampled flux vector when they are
  not selected as outer. The fixed-ocean rule applies to selected outer ocean.
- A real PyMC `NDArray` backend records an exact nonzero point, including its
  deterministics. The normal coordinate restoration, annotation, trace save
  and trace load paths preserve both frozen preparations. Pure reconstruction
  reproduces native absolute predictions. Invalid labelled pattern support is
  rejected before coefficient contraction. The recorded point is a codec and
  equation check, not a sampled posterior or convergence result.
- The CO2 marginal likelihood, after removing O2 observations, agrees in
  normalized density and state gradients with the existing CO2-only shared
  recipe using the same frozen CO2 inputs and marginal covariance. Cross-channel
  covariance is removed by marginalization in this check, never silently
  discarded from the linked target.

## Validation and verdict

The independent suite ran with the existing Python 3.13 environment, PyTensor
float64, writable temporary caches, two Dask workers and one numerical thread:

```text
python -m pytest tests/test_linked_shared_baseline_review.py -q
12 passed
ruff check tests/test_linked_shared_baseline_review.py
All checks passed!
git diff --check
```

Known dependency warnings concern the fixed-OU custom operation's future
PyTensor gradient interface and Numba object-mode dispatch; no check failed.

The validation fix also passed its three focused regressions:

```text
python -m pytest tests/test_linked_shared_baseline.py -k uppercase_outer_ocean -q
3 passed, 8 deselected
```

These cases reject sampled outer states labelled `OCEAN`, `CO2_OCEAN` and
`O2_OCEAN`. The guard continues to act on `sampled_outer`; it does not force
inner ocean states into the fixed outer set. The original numerical review
and the prepared-input readiness verdict remain applicable.

Verdict: the reviewed prepared-input scientific API is ready for the separate
configuration integration. Initial support requires common concentration units
at the linked boundary, explicitly converted upstream; independently declared
channel widths and references remain physical. No new allocation, oxidative
ratio, width calibration, or heterogeneous-unit covariance conversion is
inferred here. The review does not include long-chain convergence, a new
end-to-end acquisition run, or a fresh full tox matrix.

## Review follow-through and supported-runtime validation

The later CO2 review exposed lost physical uncertainty when a constant raw
outer response became zero after centring. The shared recipe must retain the
complete correlated physical state even when a column has no centred
likelihood response. The owning correction is `37e8b29d` in #1003; its linked
propagation is `cdb55fd4` in #1007. General builders keep their existing pruning
default and explicit activity policies. A regression checks constant raw outer
responses in both tracers, a non-unit coefficient mean, inner/outer correlation
and one shared physical draw, alongside ordinary/refreshed-cache density and
gradient agreement. These are synthetic equation and uncertainty checks, not
campaign flux-recovery evidence.

Commit `4821223d` also documents the PyTensor runtime `floatX=float64`
requirement in the guide and both public shared-builder docstrings. Float64
input arrays alone do not establish the graph precision. The additive versus
multiplicative coefficient prose correction from #989 is retained separately.

Immutable source `cdb55fd46897db904113ea5794ca98ec47bad12d` passed all 292
selected tests on each of Python 3.12.11 and 3.13.7 in Slurm array task
`19888803_1` (exit `0:0`, 23m23s, batch MaxRSS 6,925,536 KiB). Coverage includes
the CO2 shared, state/activity, correlated prior, borrowing, boundary and
inference regressions plus linked shared algebra, recording and fixed-OU
likelihood checks. Both compiler/header preflights passed. The log SHA256 is
`9b1780b9e67368a0bf01d47eac200b5fd4ca8eb0d7cc4333e8fa26ea0bdf4167`.

Repository lint and full Sphinx passed on that immutable source in
`19888806_1` (exit `0:0`), with ten warnings about existing SiteOptions aliases
and ambiguous cross-references, none in the changed guides or builders.
Rendered precision guidance, both generated API links and the inherited
coefficient prose were checked. Batch MaxRSS was 4,172,188 KiB, close to the
4 GiB allocation; future fresh documentation builds need more margin. No
rerun is needed to establish this completed check. The evidence is a relevant
regression matrix, not full-repository coverage, campaign convergence or
real-input predictive acceptance.
