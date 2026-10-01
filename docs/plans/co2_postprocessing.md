# CO2 and linked CO2/O2 postprocessing delivery

This plan extends the prepared-input CO2 recipes into scientifically labelled
postprocessing products. It is for implementers and reviewers of
[OPE-164](https://linear.app/openghg-inversions/issue/OPE-164) and
[OPE-165](https://linear.app/openghg-inversions/issue/OPE-165), under the
[OPE-79](https://linear.app/openghg-inversions/issue/OPE-79) integration gate.
The CO2-only route includes conditional native-flux and country-total summaries,
a role-based output adapter, and installed ordinary and cached fixed-OU stages.
Linked staged delivery and complete unresolved native uncertainty remain the
separate work identified below.

## Baseline and inherited decisions

The inherited baseline was reviewed against OGI `f708d606` and Linear on
2026-09-29. The staged-workflow row and delivery stage 2 below include the
subsequent OPE-164 implementation:

| Capability | Existing implementation / remaining boundary |
| --- | --- |
| Coherent CO2 handoff | OPE-153: `Co2PreparedInputs` and `prepare_co2_inputs` are available. |
| CO2 boundary and offset | OPE-150: ordinary and cached runners already include selected terms. |
| Configuration and trace serialization | OPE-162 and OPE-163 have landed; reuse their TOML and MultiIndex contracts. |
| Native reconstruction | OPE-169 and OPE-184: factorized `AffineFluxMap`, directional application methods, persistence and exact prepared-artifact binding are available. |
| Linked products | OPE-185: `linked_paris_outputs.py` already reconstructs channel concentrations and writes separate PARIS files. Its native-flux path accepts only a single-source bucket representation. |
| Linked baseline | OPE-186: tracer-specific BC and offsets are available. |
| Staged workflows | OPE-164 adds installed CO2 ordinary/cached stages; linked staging remains OPE-165. |
| Complete native uncertainty | OPE-68 remains open; affine reconstruction alone supplies retained-state-conditional means. |

Issue descriptions contain historical prerequisites. Check current issue state
and code before reopening completed foundations. The approved
[OPE-169 design](../../openspec/changes/persist-co2-affine-output-reconstruction/design.md)
and [reconstruction API design](../../openspec/changes/unify-retained-state-reconstruction/design.md)
remain authoritative; this plan does not revise them.

The implementation must follow the existing
[model locality rules](../development/rhime_model_development.rst),
[array ownership boundaries](numerical_data_ownership_and_execution_boundaries.md),
and [validation policy](../development/validation_and_xarray.rst).
Use explicit recipe adapters and ordinary functions. Do not add a model registry,
another workflow engine, or a fake multiplicative basis for a coherent map.

## Prototype evidence

The inspected checkout is
`/group/chem/acrg/verification_games_round_2/verification-games`, commit
`77959d29a1b775750e156e6d4c8f05b33fc4b689`. The files below are tracked and clean
at that revision; unrelated working-tree changes were excluded. They are
scientific reference implementations, not runtime dependencies.

* [rhime_paris_adapter.py](https://github.com/openghg/verification-games/blob/77959d29a1b775750e156e6d4c8f05b33fc4b689/src/verification_games/rhime_paris_adapter.py):
  centred affine draws, source sums before statistics, country integration,
  complete-sample covariance and compact output without grid-by-draw arrays.
* [wur_flux_projection.py](https://github.com/openghg/verification-games/blob/77959d29a1b775750e156e6d4c8f05b33fc4b689/src/verification_games/wur_flux_projection.py):
  native functionals and unresolved observation-conditioned reconstruction.
* [prepare_joint_co2_o2_coherent_input.py](https://github.com/openghg/verification-games/blob/77959d29a1b775750e156e6d4c8f05b33fc4b689/scripts/prepare_joint_co2_o2_coherent_input.py):
  tracer-qualified observations and zero CO2 response for O2-only ocean states.
* [adapter tests](https://github.com/openghg/verification-games/blob/77959d29a1b775750e156e6d4c8f05b33fc4b689/tests/test_rhime_paris_adapter.py)
  and [linked prototype fixture](https://github.com/openghg/verification-games/blob/77959d29a1b775750e156e6d4c8f05b33fc4b689/tests/fixtures/co2_o2_prototype_v1/README.md):
  independent oracle ideas, joint covariance, signed cancellation and unequal
  observation axes. The linked fixture is UOB/NAME development evidence,
  not WUR validation.

Port equations and small independent fixtures. Leave campaign discovery, truth
selection, scoring, plots and report policy in Verification Games. Older plotting
code's positional relabelling, suffix-based source inference, hard-coded ppm,
and independent predictive errors are unsuitable production contracts.

## Scientific output contract

### Flux and country quantities

For native scaling mean `m`, signed reference flux `F`, exact prolongation `U*`,
and the prepared retained mean `alpha_ref`, reconstruct

```text
f(alpha) = F [m + U* (alpha - alpha_ref)].
```

Use the bound OPE-169 artifact. Do not assume either mean equals one, derive
flux from observation sensitivity, or replace an explicit `U*` with a bucket
lift. Preserve source identity until an explicit reporting operation sums or
maps sources. GPP/TER cancellation requires signed arithmetic and joint draws.
Source totals, sector combinations and spatial/temporal aggregates must precede
standard deviations, quantiles and covariance. Applying `U*` to state quantiles
or scaling standard deviations by signed `F` is generally wrong.

For a country functional `A` (membership, cell area and physical conversion),
form `q_ref = A F m` and `R = A F U*` **before** applying chain/draw axes:

```text
q(alpha) = q_ref + R (alpha - alpha_ref).
```

Countries owns membership, region selection and area policy. Preserve fractional
and overlapping membership without normalization. Require exact geographic
coordinates; regridding is a separate explicit caller choice. The initial CO2
adapter converts flux to `mol m-2 s-1` using Pint, then country totals to grams
of CO2 per fixed 365-day year using OpenGHG's CO2 molar mass. This annualizes a
rate at each retained flux time; it does not integrate a calendar year's flux.

Keep inputs borrowed, lazy and factorized. Native-grid draws are permitted only
for an explicit native product request. Aggregate output must never construct
native-grid-by-sample intermediates. Statistical reductions may consolidate
sample chunks at their existing named boundary. Missing values must not silently
turn a source sum into a partial total; future covariance products must use one
declared complete sample set across all participating quantities.

### Concentrations, linked channels and provenance

Select scientific roles, not guessed backend variable names. The current CO2
recipe calls the complete mean role `model_mean`; the linked recipe uses
`modelled_concentration`. CO2's `co2_flux_contribution` already contains its
affine prior term. Boundary, offset and outer-flux reporting remain distinct;
a composed baseline must not double-count them.

Distinguish the expected concentration from a posterior predictive draw, which
also samples the represented observation covariance. Reuse the model's joint
aggregation/OU/measurement-error semantics. Marginal error bars cannot be used
as independent predictive noise for a correlated likelihood, and a joint log
likelihood is not pointwise predictive density.

The linked model remains one posterior with shared states and two independently
labelled observation axes. Preserve `(tracer, site, time)` identity and
cross-tracer covariance. Native flux maps must declare tracer, source, units,
ratio direction/sign and whether a coupling is already applied. CO2 functionals
must have zero response to O2-only ocean states. A sampled oxidative ratio
requires reconstruction with the matching ratio draw; a fixed affine map cannot
silently stand in for that bilinear calculation.

Retain prepared-input identity, reconstruction/projection/source provenance,
scientific roles and uncertainty scope on products. A bound reconstruction
proves its preparation identity; until staged posterior manifests exist, the
caller must supply the posterior from that same preparation. Matching state
labels alone cannot prove posterior provenance. Keep installed OGI revision in
stage/parity manifests rather than inventing a trace-root revision attribute.

## Delivery stages and acceptance

### 1. Conditional CO2 flux summaries — implemented, part of OPE-164

Add `postprocessing.co2_flux_outputs` with two public functions:

* `co2_native_flux_outputs(trace, reconstruction)` applies the bound map to
  full retained-state draws and returns source-resolved and total flux summaries.
* `co2_country_flux_outputs(trace, reconstruction, countries)` constructs compact
  country maps and returns source-resolved and total country summaries.

Both use explicit `flux_scale` roles, process prior and posterior independently,
and combine all chains only when calculating mean, population standard deviation
and the 0.159/0.841 quantiles. Products carry `retained_state_conditional` scope.
They are labelled summary datasets, not PARIS files or installed stage routes.
Linked traces are rejected until their own reconstruction binding is defined.

Acceptance: independent dense oracle with signed flux, nonunit means, nonbucket
prolongation, correlated/cancelling sources, multiple chains and unequal prior
draw counts; exact label/unit failure cases; saved artifact replay; sparse/Dask
laziness and input ownership; aggregate-before-samples evidence. Existing affine,
country, linked output and statistics coverage must remain green.

### 2. CO2 result and staged output route — implemented for OPE-164

Installed `--model co2` stages reuse the CO2 TOML resolver and public ordinary
and cached runners. Preparation validates and copies an existing coherent
`Co2PreparedInputs` handoff; it does not acquire campaign inputs or choose a
coherent reduction. Manifests bind the recipe, saved preparation, posterior and
affine reconstruction. The result adapter exposes concentration components,
full/active scaling, residuals, supported predictive products and conditional
native/country products through roles. Output capabilities are checked before
product writing. Explicit source-to-sector transforms follow reconstruction
and precede statistics, with conditional uncertainty scope preserved in PARIS.
See [the installed command examples](../usage/co2_model_family.rst).

Acceptance: a small non-VG installed
`prepare → prior-predictive → sample → diagnose → postprocess` case, a bounded
cached case, component closure with affine/BC/offset/outer terms counted once,
round-trip identities/labels/units, and early rejection of unsupported requests.
Coordinate consumer parity and retirement with OPE-151/OPE-145; do not remove
VG implementations merely because the adapter imports successfully.

### 3. Durable linked output route — OPE-165, reusing completed OPE-185

Persist `Co2O2PreparedInputs`, preserving both observation axes and cross blocks.
Define tracer-specific native reconstruction binding against that joint saved
preparation, including private-state zero response and signed coupling provenance.
Reuse stage 1's demonstrated numerical operations where meanings agree; keep
tracer selection explicit. Extend the existing separate CO2/O2 PARIS adapter
with coherent reconstruction instead of writing another concentration writer.

Acceptance: unequal-axis joint fixture, one posterior producing both products,
per-channel component closure, no O2-only contamination of CO2 flux, save/load
parity, and preservation of joint covariance in the posterior artifact. Keep
the current supported common-unit path. Heterogeneous CO2 ppm and delta(O2/N2)
per-meg transformation belongs to
[OPE-86](https://linear.app/openghg-inversions/issue/OPE-86); reject unsupported
PARIS meanings rather than treating per-meg as ordinary mole fraction.
OPE-188 owns bounded VG parity evidence.

### 4. Shared affine functionals and complete uncertainty — OPE-183 / OPE-68

[OPE-183](https://linear.app/openghg-inversions/issue/OPE-183) should extract a
shared affine quantity value only after native flux and country consumers expose
the common contract. Include ordinary/multisector and analytic consumers; do not
persist every reporting map or make that abstraction a prerequisite for stage 1.

[OPE-68](https://linear.app/openghg-inversions/issue/OPE-68) owns complete declared
quantity moments. For draw `d`, use the represented likelihood covariance `S_d`:

```text
mu_q,d = q_ref + R (alpha_d - alpha_ref) + C_qy S_d^-1 (y - mu_y,d)
Omega_q,d = C_qq - C_qy S_d^-1 C_yq
Cov(q | y) = Cov_d(mu_q,d) + E_d(Omega_q,d).
```

Adding static unresolved variance to stage 1's draws is insufficient when
`C_qy` is nonzero. Build residual blocks in bounded quantity batches while
native covariance is available, using a consistent represented observation
approximation. Reuse likelihood solves where applicable. Avoid native-by-native
covariance and all-times flattened country covariance.

Acceptance: independent dense native Gaussian oracle for both mean and covariance;
nonzero unresolved cross-block case; country covariance diagonal equals reported
stdev squared; retained-only products remain explicitly distinguished. Analytic
moments are authoritative; general quantiles require reproducible conditional
residual samples from every chain. Coordinate OPE-40 for durable residual products.

## Completion and validation boundaries

OPE-79 closes only after both installed routes, supported products, user guidance
and bounded scientific acceptance are complete. CO2 staged acceptance uses
small package-built scientific inputs and actual installed CLI commands; it does
not launch a protected VG campaign or claim full native posterior or WUR parity. Compatibility/type/full-suite jobs, when
needed for later stage integration, use the repository Slurm runner. Registered
inversion acceptance and prototype retirement require their own reviewed gates.
