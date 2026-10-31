Independent boundary level and pattern priors
=============================================

The CO2 and linked CO2/O2 ordinary and cached builders accept a prepared
``boundary_correction`` Dataset. This opt-in Python API separates a
concentration-level correction from scaling the prescribed departure patterns.
It supplies physical coefficients with labelled support; it does not use the
multiplicative ``bc`` variable. The multiplicative and existing weighted
additive options retain their previous equations and defaults.

Operators and units
-------------------

For observation rows :math:`i` and boundary states :math:`j`, define

.. math::

   H=A+cM,\qquad b_0=H\mathbf1,\qquad g=M\mathbf1.

Here :math:`H` transports prescribed concentrations, :math:`M` transports a
unit concentration correction in the chosen basis, and :math:`A` transports
departures from one scalar reference :math:`c`. In particular, :math:`M` is
not an array of ones. Normalized transport and a partition-of-unity NESW
basis give :math:`g=1`; the implementation retains the general response.

``prepare_boundary_correction`` exposes these independent zero-mean Gaussian
corrections. All physical predictions are :math:`b_0+d`.

.. list-table:: Correction choices
   :header-rows: 1

   * - ``method``
     - Correction :math:`d`
     - Coefficient units
   * - ``level``
     - :math:`g\delta`
     - :math:`\delta`: concentration
   * - ``A``
     - :math:`g\delta+Av`
     - :math:`\delta`: concentration; :math:`v`: dimensionless
   * - ``curtain_contrasts``
     - :math:`g\delta+MQz`
     - :math:`\delta,z`: concentration
   * - ``independent_curtains``
     - :math:`Mu`, without a separate level
     - :math:`u`: concentration

For :math:`K` states, :math:`Q` is the :math:`K\times(K-1)` orthonormal
Helmert basis with :math:`Q^T\mathbf1=0`. The A model retains all :math:`K`
independent coefficients per period, in addition to its level: :math:`[g,A]`
need not have a redundant column. Gaussian :math:`v` permits negative
:math:`1+v`, reversing departures around :math:`c`; there is no truncation or
lognormal substitution.

The existing ``bc_mean_shift_prior``/``bc_anomaly_scale`` formulation uses its
supplied positive weights and one global weighted centering constraint across
all curtain/period states. These weights and its pre-centering covariance
semantics are unchanged. The new equal-weight Helmert choice is explicit and
does not replace that formulation. Independent iid curtain corrections with
width :math:`\sigma_u` match the equal-weight level/Helmert covariance only
when :math:`\operatorname{SD}(\delta)=\sigma_u/\sqrt K` and
:math:`\operatorname{SD}(z_j)=\sigma_u`.

Reference selection is part of the A prior
------------------------------------------

Numerical centering and scientific prior choice are separate operations.
``reference=None`` retains the stored scalar. A finite scalar selects an
explicit reference in observation concentration units. The named
``whole_window_transport_mean`` policy uses prescribed BCs over all retained
rows:

.. math::

   c={\sum_i b_{0i}\over\sum_i g_i}
     =c_0+{\sum_{ij}A_{0ij}\over\sum_{ij}M_{ij}},\qquad
   A(c)=A_0+(c_0-c)M.

The latter expressions keep float64 centered operands without reconstructing
large absolute O2 concentrations. No observations, truth, residuals, posterior
fit or outer flux enter this policy. One scalar per species covers the entire
inversion; there are no period-specific or rolling references. Reference
selection remains lazy until joint materialization/serialization.

Changing :math:`c` preserves :math:`b_0`, but changes :math:`Av` and its prior
covariance at fixed coefficient widths. It is therefore a scientific change
to the A-scaled prior, not solely a numerical reparameterization. The prepared
artifact records the resolved scalar and selection policy. Save it before
subsetting rows for a replay or model preflight.

Receptor-time coefficient periods
---------------------------------

``frequency`` uses the existing frequency utility and requires an explicit
inversion-start ``anchor_time``. Periods gate receptor observation times,
not particle exit times. With indicators :math:`I_{ip}`, the A choices are

.. math::

   d_i=g_i\delta+\sum_p I_{ip}(Av_p)_i

or, with ``level_by_period=True``,

.. math::

   d_i=\sum_p I_{ip}\{g_i\delta_p+(Av_p)_i\}.

The second choice has no additional global level. For July with an 8D
frequency and a July 1 midnight anchor, the retained periods are July 1–8,
9–16, 17–24 and 25–31. The reference and coefficient widths remain fixed
when frequency changes. Equal pointwise variance does not preserve temporal
covariance or whole-window mean uncertainty; independent period levels
typically narrow uncertainty in that mean. No hidden recalibration occurs.

Runnable Python example and replay
----------------------------------

The offline example below constructs inputs and exercises numerical prediction,
serialization and model construction:

.. code-block:: bash

   PYTENSOR_FLAGS=floatX=float64 python -m examples.boundary_pattern_priors

.. literalinclude:: ../../examples/boundary_pattern_priors.py
   :language: python
   :start-after: # BEGIN EXAMPLE
   :end-before: # END EXAMPLE

Pass the Dataset as ``boundary_correction`` to ``build_co2_model``,
``build_co2_cached_sigma_model``, ``build_co2_from_materialized`` or
``build_rhime_co2_cached_sigma``. The CO2 inference functions also accept it
in their explicit Python model argument mapping. For linked builders and
prepared-input runners, pass a mapping such as
``boundary_correction={"co2": co2_correction, "o2": o2_correction}``.
Each channel has explicit references and widths; no cross-species BC
covariance is introduced. A prepared-input runner replaces the saved boundary
on selected channels. Omit ``use_bc`` on those channels. Other channels may
retain their existing boundary formulation.

The prepared correction owns its boundary data, priors and frequency. Omit
conflicting legacy boundary options. This change exposes a Python/data-artifact
route; it does not add a TOML correction-prior configuration language. Store
the correction Dataset alongside the ordinary prepared inputs using
``encode_multiindexes_for_storage`` and restore it with
``restore_serialisation_multiindexes``. The model boundary checks exact row
alignment, unique labels, units, finite values and positive active widths.
Inputs remain borrowed; related correction arrays are jointly densified and
computed at this named boundary.

Likelihood and physical outputs
-------------------------------

Both likelihood paths evaluate :math:`A\mathbf1+d` against :math:`y-cg`.
The existing concentration-term accounting restores :math:`cg` exactly once
to absolute deterministics and, during trace annotation, predictive ``y``.
Direct model users should restore scientific coordinates, call
``annotate_boundary_correction_trace`` for coefficient units/support and
``restore_absolute_predictives`` for absolute predictive draws.

The coefficient coordinates also record native curtain order and the Helmert
convention, so contrast coefficients can be interpreted after replay.

The labelled trace retains ``mu_bc_prescribed`` (:math:`b_0`), ``mu_bc_level``,
``mu_bc_pattern``, ``mu_bc`` (total physical BC), physical ``bc_level`` and
``bc_pattern`` coefficients, operators, widths, period/state support and the
scalar reference. Linked variables have a species prefix. Trace reconstruction
and basic products retain the separate concentration components and physical
coefficients. The complete experimental baseline remains

.. math::

   b_0+d+\text{outer flux}+\text{fixed full}-\text{fixed inner}.

Outer flux stays in the flux component and is not folded into physical BC.
A free global observation offset duplicates the boundary level when
:math:`g=1`; the builders retain the warning. Anchoring a site offset removes
that exact common direction, but does not guarantee separation from BC
patterns or flux contributions.

Widths and prior diagnostics
----------------------------

The experiments used :math:`\delta\sim N(0,5\,\mathrm{ppm})` and a
pattern-only median pointwise SD of 1 ppm. These are approved experimental
widths, not production defaults or calibrated BC uncertainties. For independent
standard normals and a width-scaled factor :math:`F`,

.. math::

   \operatorname{Cov}(d)=FF^T,\qquad
   \operatorname{SD}(l^Td)=\|l^TF\|,\qquad
   \sigma={t\over\operatorname{median}_i\|L_i\|}.

``calibrate_pattern_width`` explicitly translates target :math:`t` for an
unscaled operator :math:`L`. ``functional_sd`` evaluates labelled means or
differences exactly, without allocating an observation-square covariance.
Keep direct width specification available and report level and pattern
uncertainty separately: pointwise SDs, whole-window means and within-site
temporal differences. Pointwise matching alone does not establish equivalent
priors.

Historical experiment adapter
-----------------------------

``scripts/replay_bc_priors.py --help`` describes a new-output adapter for the
three monthly and two 8D frozen designs. ``--prepare`` verifies input/design
hashes and records revision/dependencies; ``--check-only`` checks nonzero
physical coefficients and ordinary/cached densities. Full ``--sample`` replay
requires Slurm. The 8D arms use their matched shrunk-lognormal covariance;
monthly arms retain the original correlated lognormal prior. Scientific
frequency comparisons must use their matched shrunk-lognormal monthly control,
not an original-prior monthly fit.

Frozen endpoints have historical defects. This adapter verifies matched
implementation replay, not clean physical validation of BC priors. Corrected
scientific comparisons need a new versioned preparation for both implementations
using the explicit endpoint overlay, with observations, surface transport,
fluxes, row order and other terms held fixed. Retrieving an old object-store
UUID does not apply that repair. CORSO midpoint labels and the archived temporal
sampling policy are unchanged; instantaneous versus interval-average semantics
remain unconfirmed. Historical exported-BC differences between input
generations are not an OGI export/model defect.
