# Repository Guidance

## RHIME architecture and scientific model development

New RHIME work must follow the simplicity and locality rules in
`docs/development/rhime_model_development.rst`. In particular, keep model
runners procedural, keep concrete model construction readable in scientific
order, use ordinary callable components, forward resolved values explicitly,
and accept small duplication when it keeps a model recipe understandable.

The active delivery roadmap is
`docs/plans/run_rhime_readability_and_modifiability.md`. Near-term 6 km nested-
domain, CO2-family, and verification-games feature landing is recorded in
`docs/plans/rhime_model_family_expansion.md`. These documents supersede the
semantic-compiler plans as production architecture.

## Numerical array ownership and execution

Treat xarray objects as borrowed and potentially Dask-backed. Do not mutate
inputs in place or hide copying, computation, persistence, densification, or
rechunking behind properties.

Indexed dimension coordinates are normally eager and may be validated directly;
auxiliary coordinates may remain lazy. Preserve shared Dask graphs and
materialize related arrays together at a named PyMC, serialization, or
eager-kernel boundary. Use `openghg_inversions.array_ops.to_dense` when dense
chunk payloads are required; for sparse Dask inputs, this preserves outer Dask
laziness.

See `docs/plans/numerical_data_ownership_and_execution_boundaries.md` for the
full rationale, terminology, and review checklist.

Consolidate validation at the boundary which owns an input, then trust locally
constructed intermediates. Prefer ordinary xarray transpose/alignment patterns
and Pint unit conversion to repeated custom runtime assertions; see
`docs/development/validation_and_xarray.rst`.

## Testing

Run relevant tests with pytest from the project's `uv`-managed virtual
environment:

```bash
uv run pytest tests/test_array_ops.py
```

Use focused test paths while iterating and run the relevant broader pytest
coverage before handing off a change. We still support Python 3.10, so avoid
syntax, typing, and dependency features that require newer Python versions.
