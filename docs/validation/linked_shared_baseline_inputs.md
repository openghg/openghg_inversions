# Linked CO2/O2 baseline: input evidence and linear oracle

Evidence inspected on 2026-11-02 for the follow-up to #1005. This is an
internal scientific validation record, not a released input API or a campaign
rerun. Scientific archives and catalogues were read only. Array inspection was
limited to metadata, indexed coordinates, and the small slices stated below.
No full flux fields, endpoint fields, or covariance matrices were loaded.

## Native tracer responses and campaign conventions

The maintained `openghg_inversions/rhime/co2/co2_o2_preparation.py`
(`prepare_co2_o2_inputs`, `_ratio_provenance`, and
`_materialize_and_validate_ocean_loadings_and_ratio`) treats the O2 sensitivity
as already containing the signed O2-per-CO2 ratio. It records direction
`O2 flux per CO2 flux`, sign convention
`signed; positive CO2 flux has negative O2 loading`, and convention
`embedded_signed_o2_per_co2`. Scalar ratio metadata, when available, must have
exactly the labelled shared-state index, matching `source` labels, nonempty
provenance, and finite negative values. Otherwise an explicit unavailable
reason is required. The model must not multiply this ratio again. O2 ocean is
applied directly; CO2-ocean columns are zero in O2 and conversely.

The actual VG preparation agrees with this ownership. In
`/group/chem/acrg/verification_games_round_2/verification-games/scripts/prepare_inner_o2_fp_x_flux_cache.py`,
`compute_fp_x_flux_sectoral(..., species="o2", games_scenario="BASE")`
constructs native O2 responses before reduction. Its metadata states that the
spatial ratio is embedded in the paired O2 flux before footprint convolution.
A shared scaling field connects each O2 GPP/TER/FF source to its CO2 source.
The maintained `inner_ctehr.py` production sanity checks expect CO2 GPP
negative, CO2 TER/FF positive, and the opposite O2 signs.

This is observable in the campaign source files under
`/group/chem/acrg/verification_games_round_2/verification-games-fluxes/data/raw/`:

- `PARIS_BASE.nc`: GPP, TER and FF units `micromol/m2/s`.
- `PARIS_BASE_O2.nc`: the same three units; ocean instead has `mol/m2/s`.

The Cartesian slice `time=[0,4000,5000]`, `latitude=[100,150,200]`,
`longitude=[50,100,150]` reads 27 values per source and tracer. Ratios below
are O2/CO2 on finite, nonzero CO2 entries in that slice; they are sample ranges,
not global bounds or recommended parameters.

| Source | Minimum sampled signed ratio | Maximum sampled signed ratio |
| --- | ---: | ---: |
| GPP | -1.029999852180481 | -1.0076884031295776 |
| TER | -1.0495115518569946 | -0.9902348518371582 |
| FF | -1.4447237253189087 | -1.2007440328598022 |

The sources therefore do not justify a universal net-biosphere ratio. The
historical fallback construction in
`verification-games-fluxes/notebooks/o2_flux_processing2.py` separately defines
a positive oxidation magnitude as spatial `mean(-O2_GPP/CO2_GPP)` at each time.
That notebook magnitude is a different convention from the maintained signed
model metadata, and is not a scalar replacement for the supplied native O2
operator. This inspection did not audit fallback production or modify it.

## Frozen linked campaign evidence

The frozen input is
`/group/chem/acrg/verification_games_round_2/forward_model_intermediates/rhime_o2_spatial_shared_land_split_ocean_four_site_connected_floor6_20260814/prepared/uob-base/july/joint_co2_o2_coherent.nc`.
Its labelled sources are `co2_BASE_GPP`, `co2_BASE_TER`, `co2_BASE_FF`,
`co2_BASE_ocean`, and `o2_BASE_ocean`; `state_tracer_scope(source)` and
`tracer(nmeasure)` carry the state and observation meanings. The scalar
`species="co2"` alone does not describe the linked observation channel.
The file retains `H(source,region,nmeasure)` in ppm,
`alpha_prior_covariance(source,region,source_cov,region_cov)`,
`x_prior_mean(source,region)`, `prior_forward_mf(nmeasure)` in ppm, and
`aggregation_error_covariance(nmeasure,nmeasure_cov)` in ppm squared.
The latter is the joint unresolved covariance and can contain cross-channel
blocks. These axes must retain their labels when selecting states or rows.

Its attributes explicitly describe noise-free UOB BASE sector-sum synthetic
CO2/O2 inner-only NAME enhancements, shared GPP/TER/FF scales, independent
tracer-specific ocean scales, and no outer or baseline state. That historical
inner-only scope is not evidence of existing shared-baseline acquisition.

A bounded current-package replay report at
`/group/chem/acrg/verification_games_round_2/codex-data-work/reports/co2-o2-current-ogi-smoke-20260925/current_linked_co2_o2_smoke.json`
records MHD/TAC, 70 observations per tracer, and explicitly states that no
scalar ratio exists per state because the native paired O2 operator embeds
spatially varying ratios before convolution. Its archived prior-forward
maximum absolute difference is `9.769962616701378e-15` ppm and the maximum
absolute cross covariance is `46.94102962940779` ppm squared. These are
historical checks from the recorded revision, not newly rerun results.

## Additive GPP/TER map and the limit of net collapse

Let `dg` and `dr` be signed atmospheric CO2 GPP and TER additive flux
perturbations on the same unit-density support, in common molar flux units.
They are perturbations, so Gaussian coefficients can have either sign even
though physical GPP uptake is negative and physical TER respiration positive.
For each channel `t`, let `K_tg` and `K_tr` be its native responses per unit
signed CO2 flux perturbation. The linked response is

```text
Delta y_CO2 = K_CO2,g dg + K_CO2,r dr
Delta y_O2  = K_O2,g  dg + K_O2,r  dr
```

The O2 responses already include the source conversion and transport. With a
common local transport factor and signed ratios `qg`, `qr`, its local response
is `qg*dg + qr*dr`. In a general spatial operator, multiply the native local
source-to-O2 map before convolution; a ratio of projected response columns
need not define a source ratio.

Changing coordinates to signed net `n=dg+dr` and gross `u=-dg+dr` gives
`dg=(n-u)/2`, `dr=(n+u)/2`. Therefore each tracer has columns
`(K_tg+K_tr)/2` for net and `(-K_tg+K_tr)/2` for gross. Under identical CO2
unit-density support `K_CO2,g=K_CO2,r`, its gross column vanishes. The O2 gross
column still remains when `qg != qr`. A net coefficient alone then needs an
explicit GPP/TER allocation assumption and cannot represent all source draws.

For unrestricted additive perturbations on common support, exact collapse to
CO2 net is valid only if every direction removed by the CO2 net map is also
invisible to the O2 map: `ker(N) subset ker(K_O2)`. In the two-source scalar
case this is `qg=qr`. Identical scalar ratios are insufficient if the two
spatial/temporal unit responses differ. On a deliberately restricted support,
the same kernel criterion applies only to the admissible perturbations; that
restriction must be recorded rather than inferred from a source name.

The following exact-arithmetic oracle is independent of the PyMC code and
uses illustrative ratios, not fitted campaign values. It was executed during
this inspection. It proves both the failure and the sufficient equal-ratio
case, and checks the net/gross transform.

```python
from fractions import Fraction as F


def response(row, draw):
    return sum(a * b for a, b in zip(row, draw, strict=True))


co2 = (F(1), F(1))
o2 = (F(-1), F(-6, 5))
a, b = (F(-2), F(1)), (F(-3), F(2))
assert response(co2, a) == response(co2, b) == -1
assert response(o2, a) == F(4, 5)
assert response(o2, b) == F(3, 5)
assert response(o2, a) != response(o2, b)
for draw in (a, b):
    net, gross = draw[0] + draw[1], -draw[0] + draw[1]
    for row in (co2, o2):
        transformed = ((row[0] + row[1]) / 2, (-row[0] + row[1]) / 2)
        assert response(row, draw) == response(transformed, (net, gross))
equal_ratio_o2 = (F(-11, 10), F(-11, 10))
assert response(equal_ratio_o2, a) == response(equal_ratio_o2, b) == F(11, 10)
```

## Repaired endpoint product schema and provenance

Root:
`/group/chem/acrg/verification_games_round_2/forward_model_intermediates/name_endpoint_repair_20261031_ope239_r2/`.
The inspected `sites/BIR.nc`, `sites/MHD.nc`, and `sites/LIN.nc` contain:

| Variable | Dimensions | Dtype | Units |
| --- | --- | --- | --- |
| `H_bc_anomaly` | `(species,time,bc_region)` | float64 | ppm |
| `H_bc_mass` | `(species,time,bc_region)` | float64 | 1 |
| `background` | `(species,time)` | float64 | ppm |
| `previous_normalized_background` | `(species,time)` | float64 | ppm |
| `raw_particle_mass` | `(species,time)` | float64 | 1 |

Indexed labels are `species=[co2,o2]` and `bc_region=[n,e,s,w]`.
The scalar references are recorded in attribute `centering_references_ppm`:
CO2 400 ppm and O2 209300 ppm. They are numerical centering references,
not imposed atmospheric baselines. O2 is absolute mole fraction, not per meg
or APO. No scalar reference variable was present in these files, so an adapter
must deliberately resolve that metadata into the scalar reference operand.
At MHD `time=[0,200,400]`, both channels satisfy `sum_region(M)=1` and
`background=cref+sum_region(A)` exactly in the inspected values. Hence the
native centered operands support the usual `cref + A*scale + M*offset`
construction without reconstructing anomalies by subtracting large O2 totals.

The attributes identify the original time-resolved footprint path, issue
OPE-239, spec SHA256
`282949477f6da7f9987be73ec7088f7ea2a3848442439be7ec7f4d245e9c1a86`,
and the historical CORSO midpoint-label forward-fill policy with at most six
hours hold, including year end. They record mol/mol-to-ppm conversion once
and joint float64 particle-mass normalization. This inspection does not settle
point-versus-interval-mean temporal semantics; the bundle tracks that separately.

`delivery/manifest.json` lists twelve CO2/O2 concentration products, their
paths, sizes, checksums, and source specification. Its delivery-spec SHA256 is
`adf06e195dbdf3fd8fb0b9664d19107c547028e76ec1e6769a2c00ce484e2b32`.
`catalog-registration.json` records baseline and footprint lineage separately
from concentration delivery. Corrected CORSO boundary records are 367/368;
BIR original footprint and normalized baseline records are 369/370; BASE
concentration delivery records are 585/591. The unit-repair records explicitly
say boundary numeric values and dtypes are unchanged while labels become
`mol mol-1`. Existing erroneous original labels must not trigger an additional
numeric conversion in a new adapter.

These are corrected UOB/NAME synthetic campaign products. The bundle explicitly
leaves WUR/STILT products, frozen inference inputs and fitted posteriors
unchanged. The prior single-channel WUR baseline replay is separate evidence;
it must not be presented as a linked corrected-endpoint replay.

## Acquisition-to-model contract recommendation

Preparation should own source selection, physical-unit conversion, site/time
alignment and observation filtering. It should pass per-channel native
float64 `A`, dimensionless `M`, and resolved scalar `cref` in the channel's
observation units, together with labelled observations and complete joint
covariance. Scalar metadata must keep its physical quantity and provenance.
No ratio or unit conversion should be inferred again during model binding.

Joint state selectors must use retained-state labels with `source` and
`tracer_scope`, and apply the same selection to means, sensitivity columns,
both covariance axes, and reporting maps. Shared GPP/TER/FF coefficients act
on each channel's own response; oceans remain tracer-specific. Removing outer
states must preserve any supported cross-covariance or fail explicitly.

For additive modes, retain native responses for both GPP and TER, or an exact
labelled two-column net/gross transform. A single CO2-derived net column is
scientifically sufficient only after checking the kernel condition above or
recording an explicit allocation restriction. Responses supplied in CO2 and
O2 should embed their own signs, units and paired-flux provenance exactly once.
This recommendation concerns scientific operands; exact callable names and
persisted schema remain subject to the reviewed implementation.

## Bounded saved-input preparation check

`examples/shared_baseline/linked_input_check.py` selects 28 saved rows from the
frozen linked file before loading H or either observation-covariance axis:
eight CO2 and six O2 rows at each of MHD and TAC, within
`[2021-07-12,2021-07-19)`. Its JSON result is written to
`inversion_outputs/linked-validation/real-input-check.json`; the exact parent
row positions and coherent identity are retained there. It reads a 365 by 365
retained arithmetic covariance, not a native global covariance. The active
282-state marginal is accepted unchanged by `CorrelatedLognormalPrior`, using
explicit source, tracer-scope and region labels. No covariance jitter, moment
replacement or inactive dummy prior is introduced. Inactive saved means are
zero. The coherent affine remainder `prior_forward_mf-H_active@mean_active`
is retained; its maximum absolute value on these rows is
8.353646793292402 ppm and must not be discarded.

The selected joint aggregation covariance has nonzero cross-channel blocks
(maximum absolute entry 45.33039137343721 ppm squared). The labelled saved
observations and both covariance axes survive a separate NetCDF round trip.
The native corrected endpoint A/M arrays at those exact site-times remain
float64; their scalar references are supplied explicitly by the caller as
400 and 209300 ppm. The largest baseline-reconstruction discrepancy is
5.684341886080802e-14 ppm for CO2 and zero for O2. Historical midpoint-label
forward fill remains the supplied product's policy; no temporal interpolation
or refilling occurs in this check.

The new `prepare_linked_shared_baseline` seam accepts these native operators
with `mode="fixed"`, and all-false joint `outer` and `sampled_outer` selectors.
The artifact contains only inner states, including the original independent
inner oceans. A prepared dataset for each channel survives the normal
MultiIndex serialization codec, and frozen pattern means survive smaller
unequal row subsets. Explicit 1 ppm level/delta widths and dimensionless
0.1 A-pattern width are numerical-check choices; they are not proposed
scientific calibration. No observations enter a likelihood.

The scientific limit is consequential: the frozen `mf_clean` observations
are inner-only enhancements with a zero saved baseline. The corrected endpoint
baselines are absolute mole fractions. Combining them in a conditioned model
would require a scientifically matched observation product. The check therefore
leaves the observations unchanged and stops at preparation and independent
operator closure. It does not evaluate real-input direct/cached densities or
gradients, does not sample, does not validate outer modes on real data, and does
not claim a corrected linked posterior. Separate synthetic model oracles own
those density and gradient checks.

The bounded check ran with the existing scientific environment at
`/group/chem/acrg/brendan_for_codex/openghg_inversions/.codex-worktrees/bc-pattern-priors-20261031/.venv`,
`PYTHONPATH=.` against this follow-up checkout, one numerical thread, and
writable `/tmp` caches. To reproduce from the checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_CACHE_DIR=/tmp/linked-input-check-numba \
MPLCONFIGDIR=/tmp/linked-input-check-mpl \
PYTENSOR_FLAGS=base_compiledir=/tmp/linked-input-check-pytensor \
PYTHONPATH=. \
/group/chem/acrg/brendan_for_codex/openghg_inversions/.codex-worktrees/bc-pattern-priors-20261031/.venv/bin/python \
examples/shared_baseline/linked_input_check.py
```
