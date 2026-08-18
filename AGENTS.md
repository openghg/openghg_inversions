# Repository Guidance

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

## Testing

Run relevant tests with pytest from the project's `uv`-managed virtual
environment:

```bash
uv run pytest tests/test_array_ops.py
```

Use focused test paths while iterating and run the relevant broader pytest
coverage before handing off a change. We still support Python 3.10, so avoid
syntax, typing, and dependency features that require newer Python versions.
