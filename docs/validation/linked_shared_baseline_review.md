# Linked shared baseline: independent scientific review

Reviewed scientific implementation:
`3e065f31c8e732c2f553827c7f3cb794f6762300`, on 2 November 2026.
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

Verdict: the reviewed prepared-input scientific API is ready for the separate
configuration integration. Initial support requires common concentration units
at the linked boundary, explicitly converted upstream; independently declared
channel widths and references remain physical. No new allocation, oxidative
ratio, width calibration, or heterogeneous-unit covariance conversion is
inferred here. The review does not include long-chain convergence, a new
end-to-end acquisition run, or a fresh full tox matrix.
