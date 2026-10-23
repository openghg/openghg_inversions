# Migration evidence for #960

These are original synthetic inputs and manually justified effective settings or
errors. They are evidence, not a converter schema. See
[the evidence note](../../../docs/development/migration_960.rst) for exact source
revisions, environments, execution status, scientific exceptions and owners.

- `requests/`: current development requests, decoding, override/alias rules,
  resolved defaults, output intent and filenames.
- `resources/`: synthetic layouts and explicit expected path selections/errors;
  each case includes its source justification. NetCDF files are generated only
  as tiny test resources from the declared layout, never as expected results.
- `historical/`: released 0.7 route/default/scientific evidence and rerun inputs.
- [Aggregation filter retirement, 23 October 2026](retirements/aggregation_filters_2026_10_23.json):
  pinned-source arithmetic and the modern removal disposition.
- [Model input and tracer ownership, 23 October 2026](retirements/model_inputs_2026_10_23.json):
  pinned-source populations, boundary reuse, artifact dispositions and current owners.

Keep source route and version with each expectation. A fixedbasis-looking INI
can enter the ordinary RHIME wrapper or the explicit legacy executor.

After runtime HBMCMC deletion, retain these files for later migration-tool tests.
Runtime-only drivers may retire with their adapters; port affected scientific
regressions in the removal PR. Do not regenerate expected values from the tool
under test. Changes since the pinned source revision need a recorded refresh.
