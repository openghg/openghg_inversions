# OCO2 multisector short-run summary

Command used:

```bash
MPLCONFIGDIR=/tmp/mpl-oco2-vq21425 \
PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor-oco2-vq21425,cxx= \
.tox/py310-openghgCur/bin/python -m openghg_inversions.cli \
  run-rhime-multisector \
  -c run_artifacts/oco2_multisector_short/oco2_multisector_short.ini
```

## Real-data scope

- OCO2 CO2 column observations and CAMS boundary conditions came from
  `/group/chem/acrg/object_stores/temp/OCO2_test`.
- One-day-back, time-resolved OCO2 footprints came from
  `/group/chem/acrg/object_stores/OCO2/OCO2_HR_bk1day_2022`.
- Six soundings were retained between 2022-04-01 04:07:34 and 04:08:10.
- The inversion window begins 24 hours earlier so hourly flux covers the full
  resolved `H_back` window.
- The flux sectors were `anth`, `resp`, and `gpp_atm`, exposed as
  `anthropogenic`, `respiration`, and `gpp`. Aggregate `all` and `nep` fluxes
  were excluded because they overlap these components.
- A generated weighted basis had 8 regions. The sampler used one chain, four
  tuning steps, and four retained draws; this is a pipeline smoke test, not a
  scientifically interpretable posterior.

## Observed outputs

- Preparation completed with `nmeasure=6`, `regions=8`, and `sources=3`.
- The time-resolved forward calculation ran once for the total and once for
  each sector. It retained both `fp_time_resolved` and `fp_residual`.
- PARIS v03 contains total and per-sector prior/posterior mean, standard
  deviation, percentile, province-total, and inversion-grid variables.
- PARIS dimensions are 25 hourly flux periods, 340 latitudes, 391 longitudes,
  31 Chinese provinces, 3 sectors, and 2 percentiles.
- All total prior and posterior grid cells are finite. Total gridded flux agrees
  with the sum of the three sector grids to float precision (maximum absolute
  differences about `7.3e-12` prior and `1.5e-11` posterior in
  `mol m-2 s-1`).
- `gpp` retains negative uptake, while `anthropogenic` and `respiration` remain
  non-negative in the supplied prior fluxes.
- Country totals are in `kg yr-1`. Small absolute sector-sum differences are
  float32 rounding against totals of order `1e12` to `1e13 kg yr-1`.

## Runtime observations

- Preparation: about 37 seconds.
- Build and smoke sampling: about 6 seconds.
- PARIS/diagnostics output construction: about 631 seconds.
- Peak reported RSS: about 11 GB, reached during full-grid PARIS output.
- Pint/xarray future warnings and sparse divide warnings were emitted. The
  latter occur in zero-flux-cell ratio calculations; final total grids were
  finite.
- PARIS covariance variables intentionally use repeated `country` and `sector`
  dimensions from the template. Xarray warns about those duplicate names when
  reopening the file.
- The current multisector output bundle does not save an ArviZ trace even when
  `save_trace=True`; this smoke config therefore leaves trace saving disabled.
