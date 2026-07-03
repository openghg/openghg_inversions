# Constrained Basis Algorithm Options

This page explains the lower-level constrained basis algorithms and compares 250-target-region variants on the repository TAC/EUROPE test data.

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

The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting.

For the no-mask score rows below, allocation is reported as `single_class` because there is only one class. The generator uses the normal weight-allocation API internally, but no inter-class allocation decision is being tested in that case.

## Region Class Modes

The plots below use three class modes. The selected-country mode treats ocean as one class, keeps selected large European-domain countries as separate classes, and groups all remaining land as `other_land`. The selected countries are: UK, France, Germany, Spain, Italy, Poland, Ukraine, Sweden, Norway, Finland, Turkey, Romania, Russia.

![Region class modes](figures/ogi_048_basis_options/region_class_modes.png)

## Forward-Model Compression Score

A normal multiplicative prior basis exactly reproduces the prior modelled observations when all basis coefficients are one: summing projected `H` over all regions gives the same `sum(fp * flux)` as the full grid. That RMSE is therefore a trivial zero and is not useful for comparing basis shapes.

Instead, this page uses deterministic perturbation-reconstruction diagnostics. Fine-grid flux-scale perturbation fields are applied to `fp * flux`, then each perturbation is projected to one contribution-weighted mean value per basis region. This is an optimistic representability diagnostic: it uses the known perturbation field and the same TAC fixture, not coefficients estimated from noisy held-out observations.

The headline score and ranking use only smooth perturbations: latitude and longitude gradients plus western-Europe and Nordic Gaussian blobs. Boundary-aligned perturbations, a land/ocean contrast and a selected-country patch, are reported separately because they can tautologically reward basis masks that hard-code the same boundaries.

A secondary score also projects the prior flux field itself to one cell-mean value per region and compares modelled observations from full and projected flux. That is a prior observation-space compression score; low observation NRMSE can still coexist with poor spatial flux-field reconstruction.

Neither score is a posterior-quality metric, and neither replaces posterior or synthetic-recovery tests.

## Basis Map Contrasts

![Basis option contrasts](figures/ogi_048_basis_options/basis_option_contrasts_250.png)

## Scores For 250-Region Options

All scored combinations are written to `docs/plans/ogi_048_basis_option_scores.csv`.

The heatmap color scales are local to each panel so within-panel differences remain visible. Use the printed values and ranked score plot for cross-panel comparisons.

![Score heatmaps](figures/ogi_048_basis_options/basis_option_score_heatmaps_250.png)

![Ranked scores](figures/ogi_048_basis_options/basis_option_ranked_scores_250.png)

### Best Smooth-Perturbation Scores

| rank | class mode | allocation | split step | split mode | geometry | actual regions | smooth perturb NRMSE | max smooth NRMSE | boundary perturb NRMSE | prior obs NRMSE | flux-field NRMSE |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | no_mask | single_class | axis_parallel | count | lat_lon_metres | 250 | 0.0011 | 0.0021 | 0.0048 | 0.0175 | 0.9359 |
| 2 | no_mask | single_class | inertial | count | row_column | 250 | 0.0011 | 0.0020 | 0.0038 | 0.0331 | 0.9433 |
| 3 | no_mask | single_class | axis_parallel | count | row_column | 250 | 0.0012 | 0.0020 | 0.0049 | 0.0163 | 0.9360 |
| 4 | no_mask | single_class | inertial | count | lat_lon_metres | 250 | 0.0013 | 0.0023 | 0.0040 | 0.0309 | 0.9429 |
| 5 | selected_countries | weight | inertial | count | row_column | 250 | 0.0019 | 0.0030 | 0.0000 | 0.0243 | 0.9253 |
| 6 | land_sea | weight | inertial | count | row_column | 250 | 0.0021 | 0.0038 | 0.0009 | 0.0259 | 0.9302 |
| 7 | no_mask | single_class | inertial | balanced | row_column | 250 | 0.0022 | 0.0032 | 0.0050 | 0.0323 | 0.9420 |
| 8 | land_sea | weight | axis_parallel | count | row_column | 250 | 0.0023 | 0.0040 | 0.0019 | 0.0176 | 0.9234 |

## Interpretation

- Lower smooth-perturbation NRMSE means the basis preserves the TAC observation response of the deterministic smooth fine-grid perturbations more efficiently.
- Lower boundary-perturbation NRMSE means the basis preserves perturbations that match land/ocean or selected-country boundaries. It is useful context, but it is not used for the headline ranking.
- Lower prior observation NRMSE means the basis preserves the prior forward model better under a region-mean flux-field approximation, not that it preserves the full spatial flux field.
- Balanced splits often help when the score is dominated by high-contribution areas, but they are not guaranteed to produce visually regular regions.
- Region classes impose hard boundaries, which can help interpretability but can also spend regions on low-contribution classes.
- Lat/lon metre geometry is a physical-coordinate correction. It can change region shapes, especially for inertial or high-latitude splits, but it is not itself a weight-balancing rule.
- Split-stopping policies are not included in the score matrix because they can return fewer than 250 actual regions, making direct comparison less clean.

## What This Does Not Prove

These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, held-out observations, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.
