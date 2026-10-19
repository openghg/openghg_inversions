Missing observation uncertainty: behavior and decisions
=======================================================

This record explains how RHIME handles missing repeatability and variability,
and which changes still require scientific agreement. It is intended for
maintainers investigating lost observations or proposing an uncertainty policy.
The implementation review covers OpenGHG Inversions 0.7.4 and OpenGHG 0.16.0
and 0.18.0. Historical decisions are linked below; proposed estimators are not
configuration options that users can currently select.

Quantities and missingness
--------------------------

``mf_repeatability`` describes measurement repeatability. ``mf_variability``
can be supplied with observations or calculated during resampling. Supplied
variability commonly describes the spread of high-frequency measurements within
the provider's averaging window; for some datasets it is also the supplied
measure used as instrument uncertainty. Its interpretation therefore depends
on the source, not just the variable name. Variability calculated from input
concentrations during resampling describes their spread over the new window
and is normally added alongside repeatability.

Estimating variability from already averaged concentrations cannot reconstruct
missing within-input variability or instrument repeatability. Conversely,
absent repeatability does not necessarily mean that instrument uncertainty is
unavailable: supplied variability may serve that role. ``mf_error`` is the
prepared observation-error component supplied to the likelihood; RHIME
likelihoods may also include modeled mismatch uncertainty. These error
quantities have the same mole-fraction units as ``mf``.

An absent variable, a present variable containing NaNs, and a zero value take
different paths in the existing implementation. A synthetic zero contribution
does not establish that measurement uncertainty was measured to be zero.

Existing uncertainty construction
---------------------------------

:func:`openghg_inversions.inversion_data.observation_errors.prepare_observation_errors`
constructs missing errors from each acquired site before temporal filtering or
aggregation. Supplied custom ``mf_error`` remains unchanged. Preparation owns
``averaging_error``; acquisition retains the ``averaging_period`` resampling
selector. With averaging errors enabled, the derivation is:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Input fields
     - Result before the zero-error fallback
   * - Both fields absent
     - Raise ``ValueError``; no observation error is accepted.
   * - Repeatability absent, variability present
     - Supply zero repeatability and use variability as ``mf_error``, including
       any NaNs already in variability.
   * - Variability absent, repeatability present
     - Supply zero variability and use the repeatability contribution, with
       NaNs treated as zero during quadrature.
   * - Both fields present, one component NaN
     - Use the other component; NaNs are replaced by zero only in the
       quadrature expression, not in the original diagnostic arrays.
   * - Both fields present, both components NaN
     - Quadrature gives zero, which enters the fallback below.

For two present fields, the calculation is
``sqrt(repeatability.fillna(0)**2 + variability.fillna(0)**2)``.
With averaging errors disabled, present repeatability is used directly,
including its NaNs. If repeatability is absent, the variability-only path still
applies.

This option operates on field names; it does not distinguish supplied
variability used as instrument uncertainty from newly calculated variability.

Zero ``mf_error`` values are replaced by the maximum of the median nonzero,
non-NaN error and the standard deviation of the merged ``mf`` series passed to
the function. This is an empirical fallback over that series, not an estimate
of instrument precision or an annual average of within-window variability.
It need not be positive when concentrations are constant. NaN ``mf_error``
values are not replaced by this step, despite the current warning referring to
both zeros and NaNs.

Inversion-input preparation subsequently drops rows containing NaNs in the
required sensitivity, concentration, or ``mf_error`` variables. NaNs in the
repeatability and variability diagnostic fields alone do not remove a row.
A configured minimum total-error floor acts later; it cannot recover a site
discarded during retrieval or bypass the missing-both-fields error.

Why preserving NaNs is not an incidental bug fix
------------------------------------------------

The history contains several distinct decisions:

* `PR 267 <https://github.com/openghg/openghg_inversions/pull/267>`_ explicitly
  masked the derived error when both source uncertainties were NaN, so those
  observations would be excluded.
* The `April 15, 2025 change
  <https://github.com/openghg/openghg_inversions/commit/fbbe3013f0dec67e71691ac5b0b5cc43f4268f3f>`_
  removed that mask and introduced the zero-error fallback.
* `PR 292 <https://github.com/openghg/openghg_inversions/pull/292>`_ added NaN
  detection to warning messages, particularly when averaging errors are
  disabled. It did not change the replacement mask to include NaNs.
* `PR 306 <https://github.com/openghg/openghg_inversions/pull/306>`_ removed
  the percentage threshold for replacing zero errors. Its stated scope was
  zero values, although the combined warning refers to zeros and NaNs.

Consequently, filling all NaN errors would change which observations enter an
inversion. The warning mismatch does not establish agreement to make that
scientific change. Preserve the current distinction when fixing unrelated
retrieval errors; assess any replacement policy explicitly, including the case
where both uncertainty components are unknown.

OpenGHG resampling and retrieval
--------------------------------

There are two distinct variability paths in the reviewed OpenGHG resampler:

* **Variability supplied with observations:** resampling pools the supplied
  within-input variances and the differences between input means. The mean and
  variability are weighted by the supplied number of observations. Without
  counts, OpenGHG uses the same calculation with equal weights. Supplied
  variability is thus propagated, rather than discarded and replaced by the
  standard deviation of the input means.
* **Variability calculated during resampling:** when neither variability nor
  counts is supplied, OpenGHG calculates the standard deviation of the input
  concentrations in each window. A window with one finite observation has zero
  variability in this path; a single input with supplied variability retains
  its variability mathematically under pooling, subject to numerical precision.

For complete, finite inputs, the supplied-variability calculation is

.. math::

   N = \sum_i n_i, \qquad
   \mu = \frac{\sum_i n_i m_i}{N}, \qquad
   s^2 = \frac{\sum_i n_i(s_i^2 + m_i^2)}{N} - \mu^2.

Here ``m_i``, ``s_i``, and ``n_i`` are each input's mean, variability, and
observation count. This preserves within-input spread as well as spread between
means; it is not the standard error of the new mean. Repeatability uses a
different propagation rule, assuming independent errors: the variance of the
mean is the sum of input repeatability variances divided by the square of the
number of non-NaN repeatability records, rather than underlying measurement
counts.

The supplied PARIS methodology draft, *Inverse modelling of greenhouse gas
surface fluxes with ELRIS, InTEM and RHIME: general methodology and model
settings* (October 2025), describes these RHIME paths in section 7.1,
pages 26--27, using equations (21) and (24). Section 4.1, pages 13--14,
describes the provider's hourly variability and target-gas repeatability.
The draft is supporting scientific context, not an adopted missing-value policy.

In the reviewed OpenGHG versions, all-NaN variables are deleted before surface
resampling, so an all-NaN supplied field subsequently follows an absent-field
path. When an observation-count field remains, the weighted branch
does not create variability if its input variability field is absent. Thus
identical missing uncertainty components can produce different outputs
depending on the retained count field. See the `0.18.0 resampler
<https://github.com/openghg/openghg/blob/0.18.0/openghg/data_processing/_resampling.py>`_.

Zero observation counts in issue 765
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A matched August 2020 CBW retrieval, investigated on October 2, 2026 using
OpenGHG 0.16.0, isolated an input-data difference. Both observation stores
contained 602 concentrations at identical timestamps, with nearly identical
concentrations and finite supplied variability. ``obs_nid_2026_store`` had no
observation-count field and returned 163 four-hour observations.
``obs_paris_2026_07_store`` contained an observation-count field with all values
zero and returned an empty dataset: weighted resampling divided by zero,
producing NaN means which were subsequently dropped. No exception was raised.
The retrieval ``AttributeError`` change below does not address this path.

Those zero counts were already present as ``mf_count`` in the frozen input
created on August 18, before store ingestion on August 21. Valid CBW records
have zero counts through September 29, 2021, and positive counts from September
30, matching the source boundary with the ICOS L2 contribution. The zero
counts are numeric values, not the file's NaN fill value. Their interpretation
as unavailable historical counts is plausible but has not been established
by provider documentation or the exact freezing script.

A finite measurement represents at least one observation, so supplying a count
of zero is invalid. An absent count field instead gives every input record
equal weight; it does not mean zero underlying observations. Guard against
counts below one for valid measurements at the OpenGHG input boundary,
before weighted resampling can silently discard those measurements.

Correcting erroneous counts requires an authoritative source. An explicit
equal-weight fallback for unavailable counts could reuse existing pooling,
but must preserve valid counts in other periods and supplied variability.
Inferred record weights must not be presented as known underlying measurement
counts or used uncritically in sample-count thresholds. Agree a consistent
weighting policy for windows mixing positive counts with zero or missing counts;
replacing only unavailable counts with one would mix source-record weights with
underlying-measurement counts.

The saved-data control also exposed float32 cancellation in the supplied-
variability formula: subtracting large second moments yielded one NaN and three
zero variability estimates despite positive supplied variability. Promoting
concentrations and variability to float64 before squaring produced 163 finite,
positive estimates. A centered variance calculation is a more robust,
algebraically equivalent alternative. These numerical corrections do not impute
missing uncertainty or resolve invalid counts; clipping negative variances to
zero would conceal the loss of genuine variability.

On a saved copy, omitting the all-zero count field, promoting concentrations and
supplied variability to float64, and applying normal all-NaN repeatability
removal recovered 163 four-hour observations. Inversions supplied zero
repeatability and used the pooled variability directly: all 163 ``mf_error``
values were finite and positive. Times matched the old-store result, with
maximum concentration and variability differences below 0.00014 and 0.000084
ppb respectively. This validates a recovery path under the explicit assumption
that these historical counts are unavailable; it does not adopt that assumption
as a general policy or establish the exact successful-run environment.

Temporary recovery using an existing store fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The existing ordered-store option can recover wholly missing periods without
changing uncertainty handling:

.. code-block:: ini

   [INPUT.STORES]
   obs_store = ["obs_paris_2026_07_store", "obs_nid_2026_store"]

An actual retrieval control returned 163 CBW observations from the older store
for August 2020 after the newer store returned no observations. An August 2022
control returned 186 observations from the newer store without querying the
older one. Both used four-hour averaging and OpenGHG 0.16.0.

This selects the first nonempty dataset for the whole requested period; it does
not fill gaps within a partially populated period. In particular, a run spanning
the September 2021 source transition may still select only its valid newer
observations. The list also applies to every site, and fallback does not occur
after later uncertainty processing or filtering. A successful merged-data
reload bypasses retrieval entirely.

Using this temporary option deliberately substitutes an older observation
release. Record the selected store, source UUID and version, and audit retained
sites and uncertainties, including other sites that fall back. The final
``source_format`` metadata alone is insufficient to identify the selected
store. This option does not implement Joe's proposed hierarchy or correct the
numerical precision issue.

Retrieval outcomes
~~~~~~~~~~~~~~~~~~

Historical retrieval outcomes were also different. The `January 29, 2024
OpenGHG implementation
<https://github.com/openghg/openghg/blob/c94b9e7048ab9bcdd0c4eb43278c267d19d21050/openghg/retrieve/_access.py>`_
returned ``None`` for an empty time selection, but raised ``AttributeError``
when a dataset had no time attribute. Still earlier versions used other return
values and exception types. The `inversions catch added the following day
<https://github.com/openghg/openghg_inversions/commit/b7e811d699b52b44a66e481a262063f236041eba>`_
treated retrieval ``AttributeError`` as a reason to skip a site.

The retrieval change accompanying this record preserves handling of
``SearchError``, ``None``, and datasets with an empty time dimension. It logs
and re-raises retrieval ``AttributeError`` with site, store, and date context.
This intentionally exposes legacy missing-time failures as well as unexpected
attribute failures; it does not establish compatibility with every historical
OpenGHG return convention. Retrieval failures should be diagnosed separately
from deciding which missing uncertainties to impute.

Flask decision and proposed variability hierarchy
-------------------------------------------------

`Issue 304 <https://github.com/openghg/openghg_inversions/issues/304>`_ proposed
fixed variability values for flask data. The `closing discussion
<https://github.com/openghg/openghg_inversions/issues/304#issuecomment-3188163132>`_
records the decision not to introduce arbitrary flask variability after
HFC-143a experiments showed compensation by the inferred model-error parameter.
This was evidence for that experiment, not a decision that zero variability is
adequate for every gas and site. Current flask handling bypasses resampling;
there is no implemented gas-specific variability lookup table.

`Joe's proposal in issue 765
<https://github.com/openghg/openghg_inversions/issues/765#issuecomment-5949214874>`_
provides a possible policy for sparse observations. For a four-hour target
window containing at least one observation:

#. With at least six observations in that window, calculate its variability.
#. Otherwise, with at least three observations in the centered twelve-hour
   window, calculate variability there, retaining the four-hour mean.
#. With two observations in twelve hours, use their absolute difference,
   retaining the four-hour mean.
#. With one observation in the centered twelve-hour window, use the annual
   average variability for the site and gas.
#. If that annual reference is unavailable, use a gas-specific reference derived
   from TAC-185m observations during 2017--2024, or MHD during 2013--2024 where
   TAC data are unavailable.

This hierarchy is not implemented. Its annual reference is an average of
valid variability estimates, not the standard deviation of a year's
concentrations, which also contains seasonal variation. Implementing the
neighboring-window calculation needs observations beyond the requested run
boundaries, followed by trimming to the target period.

The draft's InTEM description (section 6.5.1, pages 24--25) also uses pooled
variability, but differs from the issue proposal: it requires more than fifty
calculated variances before using the annual reference, says "more than three"
for the extended-window threshold, and calls the two-observation difference a
variance. Agree the threshold, reference eligibility, and whether the difference
is a standard deviation before adopting either description; a concentration
difference has standard-deviation units, not variance units.

Before adopting it, agree whether observation thresholds count source records
or underlying measurements in already averaged data; weighting and standard
deviation conventions; when to preserve or impute supplied variability used as
instrument uncertainty, separately from estimating resampling variability; and
how to handle genuinely zero estimates. Missing within-sample variability cannot be
reconstructed from sample means and counts alone. Reference values also need
units, source data, averaging conventions, and a recorded period of validity.

Repeatability requires a separate policy. `Issue 212
<https://github.com/openghg/openghg_inversions/issues/212>`_ discusses using
valid repeatability from the inversion period, then a wider dataset. The
choice of reference site/instrument, mean or median, and the behavior when no
reference exists remain scientific decisions. `Issue 268
<https://github.com/openghg/openghg_inversions/issues/268>`_ and issue 765
provide further missing-uncertainty context.

Evidence required for a policy change
-------------------------------------

Record the installed versions, configuration, observation store and source
dataset, variables before and after resampling, and the resulting prepared
``mf_error``. Compare retained site and observation counts as well as errors.
Capture whether each value was supplied, estimated by resampling, or imputed
from a reference, together with that reference's provenance.

PARIS concentration products export repeatability, variability, and total
likelihood error but omit prepared ``mf_error``. A successful output can
therefore establish site retention without proving the exact fallback used.
Save merged data when that distinction matters, and retain logs with job IDs:
scheduler completion alone does not show that all requested sites were retained.
