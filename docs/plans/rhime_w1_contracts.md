# RHIME W1 behavioural and usability contracts

- **Status:** Characterization baseline for OPE-43
- **Scope:** Current public RHIME behaviour only. This is not a design for the
  W2--W7 workstreams, and does not make private runner helpers public API.

## Repeatable parity selection

Run the current public-path oracle with:

```bash
tox -e py310-openghgCur -- -m rhime_contract
```

The selected tests exercise both public Python entry points, their prepared
input layouts, both installed CLI commands, config override forwarding, early
unknown-parameter failure, and the current `run_hbmcmc.py` compatibility
route. Broader schema, output, and legacy formatter cases remain in the
ordinary focused RHIME test files and are listed below so later work can extend
this selection deliberately rather than infer a contract from implementation
structure.

## Public routes and order

`run_rhime(...)` and `run_rhime_multisector(...)` accept direct keyword
arguments or an INI file. With an INI file, keyword arguments override config
values; the installed `run-rhime` and `run-rhime-multisector` commands forward
their dates, output path, and JSON `--kwargs` as those keyword arguments.

The observable current order is:

```text
normalize and validate parameters
-> retrieve/reload data
-> filter sites and observations
-> construct/load retained basis functions
-> assemble canonical inversion inputs
-> build PyMC model
-> sample and generate predictive groups
-> make in-memory outputs and write requested artifacts
```

`run_rhime_from_prepared_inputs(...)` begins after canonical input assembly;
it validates the prepared layout, output compatibility, and retained basis
before model construction. The names and placement of private helpers are not
part of this contract. Timing records are emitted at runner setup, input
preparation, model build/sample, sampling, and output-bundle boundaries.

The legacy `run_hbmcmc.py` script currently translates its fixedbasis-style
INI vocabulary to modern RHIME vocabulary, validates it, checks the country
file, copies the configuration for provenance, then calls `run_rhime(...)`.
Direct `fixedbasisMCMC(...)` remains a separate legacy Python path.

## Inputs and scientific graph

Prepared inputs retain the filtered `sites`, matching `averaging_period`,
authoritative site metadata, and `BasisFunctions`. `nmeasure` carries the
observation identity (normally the indexed `(site, time)` layout); site
metadata must align with the site identity in that dimension. Inputs are
borrowed xarray objects: callers must not mutate them in place, and ordinary
inspection must not copy, compute, persist, densify, or rechunk them. PyMC
sampling and serialization/product writing are explicit execution boundaries.

Standard sensitivity `H` has no `source` dimension and uses the retained
region layout. Multi-sector `H` retains source-resolved state, including
gathered ragged source/state layouts; it must not be rectangularized. Filtering
happens before basis generation, and the basis metadata records whether the
artifact was generated or loaded.

The built-in standard graph exposes `mf`, `mf_error`, `min_error`, `x`, `mu`,
`hx`, `y`, and `epsilon`; enabling boundary conditions adds `bc`, `mu_bc`, and
`hbc`, while offsets add `offset`. Multi-sector graphs use sanitized sector
suffixes such as `x_ff`, `mu_ff`, and `hx_ff`. Custom prepared-input builders
may declare a different explicit variable-role/output manifest, so those names
are a built-in-model contract rather than a universal custom-builder rule.

## Outputs and side effects

The supported public output modes are `none`, `inv_out`, `basic`, `paris`, and
`legacy`. `legacy` is single-sector only. `none` writes no ordinary output
products, although a multi-sector run still returns in-memory sector flux
diagnostics. Derived output formats default not to save an `InversionOutput`;
`inv_out` defaults to saving one. `save_trace` and explicit output saves require
an output path where a default path is needed.

The current standard names include
`<name><start>_trace.nc`, `<name><start>_inversion_output.nc`, RHIME derived
products named from the output/species/domain/start fields, and legacy derived
products named `<SPECIES>_<domain>_<name>_<start>.nc`. Multi-sector runs can
also write `<name><start>_sector_flux_diagnostics.nc`. Output creation may make
parent directories, write NetCDF/trace artifacts, and record their paths plus
the modern inversion-output contract in `RhimeResult.output_metadata`.

The detailed output/schema oracle is in `tests/test_rhime.py` (output bundle,
filename, and serialization tests), `tests/test_prepared_inputs_serialisation.py`,
`tests/test_xarray_input_adapter.py`, and `tests/test_postprocessing.py`.

## Scientist variation baseline and review checklist

The current acquisition-to-output call accepts ordinary arguments such as
`flux_sources`, `x_prior`, and `output_path`. Changing a prior is supported:

```python
result = run_rhime(
    species="ch4",
    sites=["TAC"],
    averaging_period=["1h"],
    domain="EUROPE",
    start_date="2019-01-01",
    end_date="2019-01-02",
    flux_sources=["total-ukghg-edgar7"],
    x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
    output_path="out",
    output_name="prior_variation",
)
```

Changing a likelihood or complete model at that same boundary is not currently
supported: `model_builder` and `likelihood_builder` are accepted only by
`run_rhime_from_prepared_inputs(...)`; passing either to `run_rhime(...)` is an
early unsupported-parameter error. This is the baseline ergonomics limitation
for W2, not a reason to add a new framework in W1.

Scientific-user review:

1. Find `run_rhime(...)` or `run_rhime_multisector(...)` and see the route
   above without needing private runner knowledge.
2. Follow normalization, preparation, model, sampling, and output in order.
3. Change a normal prior using ordinary acquisition arguments and retain the
   output path/name.
4. Run the explicit contract suite, then the focused RHIME tests for the
   output mode being changed.
5. Inspect `RhimeResult`, `output_metadata`, the saved trace/inversion output,
   and any selected derived product.
