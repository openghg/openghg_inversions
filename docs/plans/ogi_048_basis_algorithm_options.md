# Constrained Basis Algorithm Options

This page explains the lower-level constrained basis algorithms and compares 250-target-region variants on Blue Pebble OpenGHG data.

## How The Algorithm Is Built

The constrained basis path is a small orchestration framework rather than one fixed algorithm. The caller supplies a two-dimensional importance field, `weights`, and a two-dimensional `region_classes` mask. The algorithm partitions each mapped class independently, then offsets labels so basis labels are globally unique and never cross class boundaries.

The independently variable pieces are:

- **Region classes**: no mask, land/ocean, countries, grouped countries, or any caller-supplied class field.
- **Class allocation**: explicit per-class counts, automatic allocation by class total weight, or automatic allocation by cell count.
- **Greedy orchestration**: repeatedly split the currently highest-weight partition until the target is reached or no acceptable split remains.
- **Partition step**: axis-parallel row/column splits or inertial principal-axis splits.
- **Split mode**: count-based splits or balanced splits near half parent weight.
- **Geometry**: row/column index geometry or local lat/lon metre geometry for split-shape decisions.
- **Split stopping**: optional policies that reject proposed child regions. When stopping is enabled, the requested region count is an upper target.

The comparison also includes legacy `bucketbasisfunction` and `quadtreebasisfunction` rows generated from the same training weights. Their actual region counts can differ from the 250 target.

The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting.

For the no-mask score rows below, allocation is reported as `single_class` because there is only one class. The generator uses the normal weight-allocation API internally, but no inter-class allocation decision is being tested in that case.

## Blue Pebble Cross-Validation Data

The script reads footprints and CH4 flux from OpenGHG store `shared_store_zarr` on Blue Pebble. It uses TAC 185m and MHD 10m EUROPE NAME inert footprints, and the monthly `edgarv80_wetchartsv131` CH4 flux product.

January and July 2019 are scored separately. Each site/month uses two temporal CV splits: a one-week holdout starting on days 6 and 20, with a two-day buffer excluded before and after the held-out week. Basis weights are built only from the remaining in-month footprints.

Aggregate scores are written to `docs/plans/ogi_048_basis_option_scores.csv` and split-level scores are written to `docs/plans/ogi_048_basis_option_split_scores.csv`. The split-level table contains 336 scored rows.

## Region Class Modes

The plots below use three class modes. The selected-country mode treats ocean as one class, keeps selected large European-domain countries as separate classes, and groups all remaining land as `other_land`. The selected countries are: UK, France, Germany, Spain, Italy, Poland, Ukraine, Sweden, Norway, Finland, Turkey, Romania, Russia.

![Region class modes](figures/ogi_048_basis_options/region_class_modes.png)

## Held-Out Forward-Model Compression Score

A normal multiplicative prior basis exactly reproduces the prior modelled observations when all basis coefficients are one: summing projected `H` over all regions gives the same `sum(fp * flux)` as the full grid. That RMSE is therefore a trivial zero and is not useful for comparing basis shapes.

Instead, this page uses deterministic perturbation-reconstruction diagnostics on held-out footprints. Fine-grid flux-scale perturbation fields are applied to held-out `fp * flux`, then each perturbation is projected to one training-weighted mean value per basis region. This is still an optimistic representability diagnostic, but the footprint data used for scoring are not used to construct the basis weights.

The headline score and ranking use only smooth perturbations: latitude and longitude gradients plus western-Europe and Nordic Gaussian blobs. Boundary-aligned perturbations, a land/ocean contrast and a selected-country patch, are reported separately because they can tautologically reward basis masks that hard-code the same boundaries.

A secondary score also projects the prior flux field itself to one cell-mean value per region and compares held-out modelled observations from full and projected flux. That is a prior observation-space compression score; low observation NRMSE can still coexist with poor spatial flux-field reconstruction.

Neither score is a posterior-quality metric, and neither replaces posterior or synthetic-recovery tests.

## Basis Map Contrasts

![Basis option contrasts](figures/ogi_048_basis_options/basis_option_contrasts_250.png)

## Scores For 250-Region Options

The heatmap shows the mean split score for each site/month context. The ranked plot averages the four site/month aggregate rows for each candidate.

![Score heatmaps](figures/ogi_048_basis_options/basis_option_score_heatmaps_250.png)

![Ranked scores](figures/ogi_048_basis_options/basis_option_ranked_scores_250.png)

### Best Smooth-Perturbation Scores By Site And Month

| site | month | rank | candidate | regions | splits | smooth perturb NRMSE | max smooth NRMSE | boundary perturb NRMSE | prior obs NRMSE | flux-field NRMSE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MHD | January | 1 | quadtreebasisfunction | 248.5 | 2 | 0.0350 | 0.0729 | 0.0693 | 0.1120 | 0.9321 |
| MHD | January | 2 | no_mask/single_class/axis_parallel/count/row_column | 250.0 | 2 | 0.0358 | 0.0705 | 0.0730 | 0.1052 | 0.9290 |
| MHD | January | 3 | bucketbasisfunction | 249.5 | 2 | 0.0381 | 0.0780 | 0.0648 | 0.1057 | 0.9302 |
| MHD | January | 4 | selected_countries/area/inertial/count/row_column | 250.0 | 2 | 0.0452 | 0.0737 | 0.0000 | 0.1118 | 0.9252 |
| MHD | January | 5 | selected_countries/area/inertial/count/lat_lon_metres | 250.0 | 2 | 0.0466 | 0.0689 | 0.0000 | 0.1069 | 0.9203 |
| MHD | July | 1 | bucketbasisfunction | 249.0 | 2 | 0.0112 | 0.0165 | 0.0400 | 0.2783 | 0.8705 |
| MHD | July | 2 | no_mask/single_class/axis_parallel/count/lat_lon_metres | 250.0 | 2 | 0.0136 | 0.0202 | 0.0585 | 0.2925 | 0.8724 |
| MHD | July | 3 | no_mask/single_class/axis_parallel/count/row_column | 250.0 | 2 | 0.0141 | 0.0212 | 0.0582 | 0.2926 | 0.8740 |
| MHD | July | 4 | quadtreebasisfunction | 251.5 | 2 | 0.0185 | 0.0323 | 0.0876 | 0.3082 | 0.8740 |
| MHD | July | 5 | no_mask/single_class/inertial/count/row_column | 250.0 | 2 | 0.0190 | 0.0306 | 0.0421 | 0.2668 | 0.8873 |
| TAC | January | 1 | no_mask/single_class/inertial/count/row_column | 250.0 | 2 | 0.0045 | 0.0084 | 0.0194 | 0.0446 | 0.9372 |
| TAC | January | 2 | no_mask/single_class/inertial/count/lat_lon_metres | 250.0 | 2 | 0.0085 | 0.0143 | 0.0125 | 0.0803 | 0.9410 |
| TAC | January | 3 | land_sea/weight/axis_parallel/count/row_column | 250.0 | 2 | 0.0085 | 0.0158 | 0.0072 | 0.0272 | 0.9196 |
| TAC | January | 4 | land_sea/area/axis_parallel/count/row_column | 250.0 | 2 | 0.0122 | 0.0315 | 0.0146 | 0.0449 | 0.9151 |
| TAC | January | 5 | selected_countries/weight/axis_parallel/count/lat_lon_metres | 250.0 | 2 | 0.0138 | 0.0191 | 0.0000 | 0.0525 | 0.9251 |
| TAC | July | 1 | land_sea/weight/axis_parallel/count/row_column | 250.0 | 2 | 0.0171 | 0.0367 | 0.0095 | 0.0458 | 0.8515 |
| TAC | July | 2 | land_sea/weight/axis_parallel/count/lat_lon_metres | 250.0 | 2 | 0.0173 | 0.0340 | 0.0143 | 0.0438 | 0.8501 |
| TAC | July | 3 | land_sea/area/axis_parallel/count/lat_lon_metres | 250.0 | 2 | 0.0206 | 0.0478 | 0.0180 | 0.0590 | 0.8559 |
| TAC | July | 4 | no_mask/single_class/inertial/count/row_column | 250.0 | 2 | 0.0211 | 0.0533 | 0.0227 | 0.0595 | 0.8829 |
| TAC | July | 5 | land_sea/area/axis_parallel/count/row_column | 250.0 | 2 | 0.0241 | 0.0522 | 0.0271 | 0.1156 | 0.8542 |

## Interpretation

- Lower smooth-perturbation NRMSE means the basis preserves the held-out observation response of the deterministic smooth fine-grid perturbations more efficiently.
- Lower boundary-perturbation NRMSE means the basis preserves perturbations that match land/ocean or selected-country boundaries. It is useful context, but it is not used for the headline ranking.
- Lower prior observation NRMSE means the basis preserves the prior forward model better under a region-mean flux-field approximation, not that it preserves the full spatial flux field.
- Balanced splits often help when the score is dominated by high-contribution areas, but they are not guaranteed to produce visually regular regions.
- Region classes impose hard boundaries, which can help interpretability but can also spend regions on low-contribution classes.
- Lat/lon metre geometry is a physical-coordinate correction. It can change region shapes, especially for inertial or high-latitude splits, but it is not itself a weight-balancing rule.
- Split-stopping policies are not included in the score matrix because they can return fewer than 250 actual regions, making direct comparison less clean.

## What This Does Not Prove

These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.
