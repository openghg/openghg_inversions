# RHIME CO₂ precision audit · 25 November 2026

[Issue #1130](https://github.com/openghg/openghg_inversions/issues/1130) follows
up the [review of PR #1124](https://github.com/openghg/openghg_inversions/pull/1124#pullrequestreview-6059174816)
under [the precision policy in #607](https://github.com/openghg/openghg_inversions/issues/607).
This first experiment checks the process-wide float64 requirement for
centered boundaries. The first measurements simulated the proposed cache
precision. The results below were rerun on PR #1124 at `e2eca88b`, which
stores and evaluates the state quadratic in float64. The precision
prototypes run in the probe process.

## Where precision changes

| Boundary | Current operation |
| --- | --- |
| Prepared centered operands | Binding requires process-wide float64. The three operands are the anomaly sensitivity, mass sensitivity and reference concentration. |
| Registration | `add_model_data` passes arrays through `pm.Data`, which converts floating inputs to `floatX`. The reference shift used on output also passes through this boundary. |
| Forward likelihood | The recipe subtracts the reference shift from observations before registering them. The multiplicative recipe forms the boundary deviation `b - 1`. Its scale can be close to one even when the response to that small deviation is substantial. |
| Fixed-OU targets | The native log density forms the residual in the graph and then casts it to float64 for the NumPy evaluator. The state cache forms its quadratic from the prepared affine design in float64. Promoting only the cache leaves any rounding in the native graph and the physical coefficients. |
| Prediction and sampling | Replay restores absolute observations and predictive concentrations using the saved shift. The cached state NUTS step constructs a PyMC `ValueGradFunction` with the process dtype and the default casting rule. |

Sources for this snapshot are the [centered-boundary precision guard](https://github.com/openghg/openghg_inversions/blob/e2eca88b/openghg_inversions/models/boundary/contracts.py#L50-L62),
the [centered multiplicative boundary](https://github.com/openghg/openghg_inversions/blob/e2eca88b/openghg_inversions/models/boundary/multiplicative.py#L318-L350),
the [native fixed-OU density](https://github.com/openghg/openghg_inversions/blob/e2eca88b/openghg_inversions/models/fixed_ou.py#L240-L249)
and [state step construction](https://github.com/openghg/openghg_inversions/blob/e2eca88b/openghg_inversions/inference/cached_sigma.py#L449-L486).

## First experiment

The [probe](issue_1130_centered_precision_probe.py) uses the four-observation
centered-boundary test fixture with an active flux state, fixed OU amplitudes
and additive or multiplicative boundary corrections. It evaluates the
initial state and a nearby transformed state, draws two prior predictions
and restores the absolute observations. It uses a CO₂-scale reference
of 400 ppm and a separate 209,300 per-meg stress case in the same
single-channel graph. The stress case is an arithmetic check at that scale.

At the perturbed multiplicative state in the per-meg case:

| Process and prototype | Maximum error restoring observed concentrations (per meg) | Native log likelihood compiled alone | Native log likelihood in the graph with gradients | Cached log likelihood |
| --- | ---: | ---: | ---: | ---: |
| float64 | 0 | −14.05418567 | −14.05418567 | −14.05418567 |
| float32 with guard bypassed | 0.00918 | −14.05418907 | −13.99966603 | −14.05418505 |
| float32 with selected boundary data kept in float64 | 9.54e−8 | −14.05418766 | −13.99966528 | −14.05418505 |
| float32 with selected data and the boundary prior kept in float64 | 9.54e−8 | −14.05418848 | −14.05418848 | −14.05418578 |
| float32 with selected data in float64 and an `expm1` boundary deviation | 9.54e−8 | −14.05418766 | −14.05418766 | −14.05418505 |

The data-only prototype preserved the anomaly, mass, reference, absolute
boundary reference and replay shift while keeping the flux sensitivity and
state coordinates in float32. This fixed most of the restoration error.
The centered mean still changed by up to 0.00318 per meg when compiled
alongside its gradient in the multiplicative case. Changing the PyTensor
optimizer from `fast_run` to `fast_compile` did not remove that discrepancy.
The additive case showed no comparable difference between the two compilation
paths. Its cache and native likelihood agreed within 2.43e−7 at the tested
states in the float32 data-only prototype. The data-only prototype also completed a
six-draw cached PyMC smoke check on the per-meg multiplicative case and
produced finite native log-likelihood output. This checks execution only.

Promoting the multiplicative boundary prior removed the compilation
discrepancy at these points. Mixed float32 flux and float64 boundary free
variables hit a further constraint. The cached state step's
`ValueGradFunction` rejects the float64 boundary variable with its default
float32 dtype and `casting="no"`. The experiment therefore does not support
removing the binding guard by just preserving boundary arrays or priors.

## Stable deviation with float32 sampler coordinates

For the default lower-zero truncated-normal scaling prior, the inverse
interval transform is `b = exp(eta)`. The equivalent prior on the deviation
`d = b - 1` has mean parameter zero, lower bound −1 and the same standard
deviation. Its inverse transform is `d = expm1(eta)`, with forward transform
`log1p(d)` and the same log Jacobian `eta`. Sampling the deviation lets the
centered likelihood and affine cache both use it directly. The physical scale
is still available as `1 + d`.

In an isolated float32 calculation at `eta = 2 / 209300.0021`, the
concentration error at that reference scale fell from about 0.004 per meg
with `exp(eta) - 1` to 3.8e−8 per meg with `expm1`. The compiled prototype
also retained the float32 flux and boundary coordinates. At the two tested
states in the per-meg case, its centered mean agreed with the float64
reference within 1.44e−7 per meg. Its nearby-state log-density change differed
by 1.89e−6, and the maximum componentwise gradient error divided by
`max(1, abs(reference_gradient))` was 1.73e−5 for the native graph.
The native log density agreed when compiled alone and with its gradient.
A six-draw NUTS smoke check produced finite native log likelihoods and
restored posterior predictions.

The prototype covers the default prior with all boundary states active in a
single-channel fixture. The package continues to require float64 for
centered operands. The [maintained precision explanation](../development/fixed_ou_likelihood.rst)
records the roles of the cache, amplitude trajectory and model data.

## Reproduce and extend

Run from the repository root with a fresh process for every mode. The modes
are `ordinary` for float64, `guard-only` for a bypassed guard, `local` for
the data-only prototype, `local-prior` for data plus prior precision and
`local-expm1` for local data precision with the stable deviation.
Add `--sample` to construct the cached NUTS step and run a six-draw smoke
check on the per-meg multiplicative case. With `local-prior` this exposes
the mixed-dtype construction error above.

```bash
export PYTHONPATH=".:tests${PYTHONPATH:+:$PYTHONPATH}"
PYTENSOR_FLAGS=floatX=float64,linker=py,cxx= \
  uv run python docs/plans/issue_1130_centered_precision_probe.py ordinary
PYTENSOR_FLAGS=floatX=float32,linker=py,cxx= \
  uv run python docs/plans/issue_1130_centered_precision_probe.py local-expm1 --sample
```

Measured with Python 3.12.11, PyMC 6.3.2, PyTensor 3.3.8, NumPy 2.2.6,
SciPy 1.15.3 and xarray 2025.9.1. A follow-up should cover other boundary
priors and activity policies, extend the comparison to linked and inferred
amplitude routes, and measure time and memory on representative fits. The
four-row experiment measures pointwise agreement and checks that sampling
and prediction run.
