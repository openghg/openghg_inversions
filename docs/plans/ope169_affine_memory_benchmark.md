# OPE-169 affine artifact memory evidence

Run on `bp1-login02` (Linux x86_64, Python 3.12) against commit `904167ac`
plus the OPE-169 PR 2 worktree on 2026-09-25. The reproducible fixture and
measurement code are in `scripts/benchmark_ope169_affine_memory.py`.

```bash
UV_CACHE_DIR=/tmp/ope169-uv-cache MPLCONFIGDIR=/tmp/ope169-matplotlib uv run --frozen python scripts/benchmark_ope169_affine_memory.py --format nc
UV_CACHE_DIR=/tmp/ope169-uv-cache MPLCONFIGDIR=/tmp/ope169-matplotlib uv run --frozen python scripts/benchmark_ope169_affine_memory.py --format zarr
```

The fixture has a 72 × 96 native grid, 96 retained states, two signed flux
times, 2 chains × 128 draws, and four country-like membership regions with
latitude area weights. Native mean and flux are float64; the explicit
prolongation is float32 and genuinely dense. The bucket prolongation keeps
the operator's sparse Dask matrix. Native chunks are 24 × 24 and draw chunks
are 32. Both formats use the repository's `save_datatree` path and xarray's
default storage encoding; no custom compression was requested. Versions:
xarray 2026.1.0, Dask 2026.1.1, Zarr 2.18.3, NumPy 2.2.6.

Each stage ran once in a fresh fork after imports. `ru_maxrss` reports the
Linux process high-water resident set. The number in parentheses is peak
above that worker's baseline, approximately 292–297 MiB. Grid and aggregate
stages include artifact load. Sizes count the NetCDF file or all Zarr files.

| Representation | Format | Artifact bytes | Peak load MiB | Peak grid apply MiB | Peak aggregate contraction MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Bucket | NetCDF | 239,325 | 336.6 (+44.2) | 421.3 (+128.6) | 347.5 (+54.9) |
| Explicit | NetCDF | 2,839,218 | 308.0 (+15.6) | 576.0 (+283.6) | 328.8 (+36.2) |
| Bucket | Zarr | 151,333 | 325.9 (+33.4) | 421.6 (+124.8) | 335.3 (+42.8) |
| Explicit | Zarr | 740,256 | 294.0 (+1.3) | 505.6 (+213.1) | 315.8 (+23.3) |

The benchmark asserts Dask-backed ingredients before serialization and a
lazy grid result before the requested grid `.compute()`. For aggregate output,
it contracts membership, area, flux, and prolongation to country × time ×
state before introducing chain/draw, then materializes that compact action
and the final aggregate at named `.compute()` boundaries. The loaded explicit
array is eager by the current load API, so its compact contraction is eager.
No native-grid-by-sample array is built on the aggregate path. The final
aggregate is 4 × 2 × 2 × 128, while the requested flux grid is
2 × 72 × 96 × 2 × 128. These are single-run process measurements, not a
production-domain memory guarantee; allocator reuse can make a small stage
show little RSS growth.
