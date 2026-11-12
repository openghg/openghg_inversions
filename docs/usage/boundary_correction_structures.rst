Boundary correction structures
==============================

Centered boundary sensitivity gives the prescribed boundary concentration as
:math:`A\mathbf1+cM\mathbf1` (see :doc:`centered_boundary_conditions`). Two
boundary families correct it with independent zero-mean Gaussian
coefficients whose widths are explicit scientific inputs; nothing is
calibrated from the data.

* :class:`~openghg_inversions.models.boundary.AdditiveChoice` adds
  concentration-unit state corrections :math:`Md`. Its ``anomalies`` option
  states the structure of :math:`d`.
* :class:`~openghg_inversions.models.boundary.AnomalyScalingChoice` adds
  :math:`\delta M\mathbf1+A'v`, scaling the prescribed anomaly by
  dimensionless :math:`v`.

Both are choices. A model recipe binds a choice to the prepared centered
operands of one fit with ``choice.bind(observations, sensitivity)`` and adds
the bound record inside its registered model; no correction artifact is
prepared or stored. The CO2 runners and configuration do not yet accept these
choices; that adoption is downstream work.

Additive structures
-------------------

States :math:`j` are boundary curtains :math:`k` and periods :math:`p`. With
:math:`K` curtains and :math:`P` periods:

.. list-table::
   :header-rows: 1

   * - Choice
     - Correction :math:`d`
     - Anomaly coordinates
   * - ``AdditiveChoice(mean, s)``
     - :math:`\delta\mathbf1+(Q-\mathbf1w^\mathsf{T}Q)z`
     - :math:`KP-1`
   * - ``AdditiveChoice(mean, s, anomalies="period_contrasts")``
     - :math:`\delta\mathbf1+` Helmert contrasts of the curtains in each period
     - :math:`P(K-1)`
   * - ``AdditiveChoice(None, s, anomalies="independent")``
     - :math:`u`, one anomaly per state
     - :math:`KP`
   * - ``AdditiveChoice(mean, 0.0, ...)``
     - :math:`\delta\mathbf1` only
     - none

Every :math:`z` and :math:`u` coordinate has standard deviation :math:`s`.
A ``None`` mean prior omits :math:`\delta` from period contrasts or
independent anomalies; weighted centering always keeps it.
``mean_by_period=True`` replaces :math:`\delta\mathbf1` with one independent
shift per boundary period, drawn from the same ``mean`` prior. ``frequency``
and ``anchor_time`` follow ``bc_freq``: ``None`` gives one period,
``"monthly"`` calendar months, and a fixed duration such as ``"7D"`` periods
anchored at ``anchor_time`` or, when it is ``None``, at the fit's earliest
observation (the RHIME parameter adapter defaults ``bc_freq_anchor`` to the
configured start date). Weighted
centering requires the prepared ``bc_centering_weights``; the other
structures ignore them.

The mean shift responds through :math:`M`, contributing :math:`\delta
M\mathbf1` to the observations. Under supported surface preparation
:math:`M\mathbf1=\mathbf1`; for other transport it is the transported shift
rather than a constant.

Temporal curtain contrasts and weighted centering are different priors.
Contrasts leave each period's equal-weight curtain mean to the mean shift;
weighted centering constrains only one weighted mean over all states and lets
whole periods move together. Independent anomalies with scale :math:`s` give
the same covariance as curtain contrasts with scale :math:`s` plus one mean
shift per period (``mean_by_period=True``) with scale :math:`s/\sqrt K`, and
no other widths. A single whole-window mean gives that covariance only with
one period; with several it adds a cross-period covariance :math:`s^2/K`
that independent anomalies do not have.

Independent anomalies are sampled directly as one coordinate per state, so
their design is :math:`M` itself and no basis is stored.

Anomaly scaling
---------------

The anomaly-scaling family adds :math:`\delta M\mathbf1+A'v`: a mean shift
transported through :math:`M`, and dimensionless :math:`v_j\sim N(0,s^2)`
scaling the reference-adjusted anomaly :math:`A'` on each state. A ``None``
mean prior omits :math:`\delta`. Gaussian :math:`v` permits negative
:math:`1+v`; no truncation is applied.

:math:`A'` is the anomaly about a chosen reference :math:`c'`. Changing the
reference leaves the prescribed concentration unchanged but changes
:math:`A'`, and therefore the covariance of :math:`A'v`, so the reference is
part of the prior. Explicitly, :math:`A'=A+(c-c')M` and
:math:`A'\mathbf1+c'M\mathbf1=A\mathbf1+cM\mathbf1`. The options are:

* ``reference="stored"`` keeps the prepared :math:`c`;
* a number in observation concentration units sets :math:`c'` directly;
* ``reference="whole_window_transport_mean"`` uses
  :math:`c'=c+\sum A/\sum M` over the fit's rows, so the recentred anomaly
  sums to zero.

The reference is computed when the choice is bound, from the rows of that fit.

Example
-------

::

   from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice

   contrasts = AdditiveChoice(
       {"pdf": "normal", "mu": 0.0, "sigma": 0.5}, 0.3,
       anomalies="period_contrasts", frequency="monthly",
   )
   scaling = AnomalyScalingChoice(
       {"pdf": "normal", "mu": 0.0, "sigma": 0.5}, 0.8,
       reference="whole_window_transport_mean",
   )

   # observations: one fit's observation rows; sensitivity: its prepared
   # RawCenteredSensitivity.
   bound = contrasts.bind(observations, sensitivity)
   # Materialize the bound record together with the recipe's other inputs,
   # then, inside the recipe's registered model:
   term = bound.add()

Prior covariance
----------------

Each bound boundary returns an
:class:`~openghg_inversions.models.components.AffineConcentrationTerm` whose
affine blocks are the observation-space designs :math:`D_k` of its Gaussian
coefficient groups. With block widths :math:`s_k`, the prior covariance of
the correction is :math:`\sum_k s_k^2D_kD_k^\mathsf{T}`. A linear functional
:math:`l` of the observations, such as a row mean or a difference of two rows,
has standard deviation :math:`\sqrt{\sum_k s_k^2\|D_k^\mathsf{T}l\|^2}`. Use
this when choosing widths; it is not applied automatically. Before model
construction, ``anomaly_factor()`` of a bound boundary returns the anomaly
design scaled to standard-normal coordinates: :math:`sMB` on
``bc_contrast`` for weighted anomalies and period contrasts, :math:`sM` on
``bc_region`` for independent anomalies, and :math:`sA'` on ``bc_region`` for
anomaly scaling.
