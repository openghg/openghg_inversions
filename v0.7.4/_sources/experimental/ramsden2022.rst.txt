Ramsden methane/ethane model
============================

Status
------

The :mod:`openghg_inversions.experimental.ramsden2022` module implements the
shared-state methane/ethane model described by `Ramsden et al. (2022)
<https://doi.org/10.5194/acp-22-3911-2022>`_. It is intended for historical
comparison and as a requirements fixture for future generic linked-tracer
work. It is not a stable public API.

The model starts at the prepared-input boundary. It accepts two canonical
RHIME-style datasets and uses the current PyMC model components and
:class:`~openghg_inversions.rhime.sampling.RhimeSampler`. It deliberately does
not port the historical ACRG/OpenGHG retrieval layer, pickle cache,
configuration parser, custom Metropolis-Hastings sampler, isotope extensions,
or bespoke country/NetCDF post-processing.

Scientific model
----------------

For fossil-fuel and non-fossil methane scaling states, the two observation
models are

.. math::

   \mu_{\mathrm{CH_4}} =
       H_{\mathrm{CH_4,FF}} x_{\mathrm{FF}}
       + H_{\mathrm{CH_4,nonFF}} x_{\mathrm{nonFF}}
       + H_{\mathrm{bc,CH_4}} b_{\mathrm{CH_4}},

.. math::

   \mu_{\mathrm{C_2H_6}} =
       H_{\mathrm{C_2H_6,FF}} (R x_{\mathrm{FF}})
       + H_{\mathrm{bc,C_2H_6}} b_{\mathrm{C_2H_6}}.

The fossil-fuel scaling state is shared between channels. Non-fossil methane
does not contribute to the ethane likelihood. The channels can have different
sites, timestamps, observation counts, boundary states, and absolute
model-error states.

For channel :math:`g`, the observation standard deviation is

.. math::

   \epsilon_g =
       \max\left(
           \sqrt{\sigma_{\mathrm{measurement},g}^2
                 + \sigma_{\mathrm{model},g}^2},
           \epsilon_{\min,g}
       \right).

Set ``min_error`` to zero for the paper-shaped likelihood. A positive value
retains the modern RHIME numerical floor intentionally.

Prepared inputs
---------------

Each channel dataset must use the canonical names below. The two observation
axes can differ, but every coupled sector must use exactly matching labelled
state coordinates.

.. list-table:: Required channel variables
   :header-rows: 1
   :widths: 20 25 55

   * - Variable
     - Required dimensions
     - Meaning
   * - ``H``
     - observation, state, source
     - Labelled sensitivity matrix mapping scaling states into observation space.
   * - ``mf``
     - observation
     - Measured mole fraction in the channel's declared numeric units.
   * - ``mf_error``
     - observation
     - Measurement contribution to likelihood uncertainty.
   * - ``min_error``
     - observation
     - Optional positive floor applied after combining measurement and model
       uncertainty.
   * - ``site_indicator``
     - observation
     - Integer site index used to align site-level model error.
   * - ``H_bc``
     - observation, boundary state
     - Boundary sensitivity; required when the channel enables boundary scaling.

If state labels are positional numbers, retain and pass both
:class:`~openghg_inversions.basis.basis_functions.BasisFunctions` objects.
The builder then verifies the spatial maps instead of assuming that two
``0..N-1`` indexes describe the same regions.

Ratio contracts
---------------

The model supports two explicit ratio conventions:

Direct physical ratio
   ``reference_ratio=None`` means the tracer sensitivity is ratio-free. The
   sampled or fixed value is the dimensionless molar emission ratio, moles of
   tracer divided by moles of primary gas.

Historical multiplier
   A positive ``reference_ratio`` means the tracer sensitivity already contains
   that ratio. The sampled or fixed value is a multiplier, and the model
   exposes

   .. math::

      R_{\mathrm{physical}} =
          R_{\mathrm{reference}} R_{\mathrm{multiplier}}.

The independent ``tracer_design_reference_ratios`` mapping records what is
already present in every tracer sensitivity. The builder rejects inconsistent
declarations to prevent applying a ratio twice or omitting it.

Units
-----

The builder validates supported unit declarations by mole-fraction scale but
does not convert numeric values. Observations, measurement errors, minimum
errors, forward sensitivities, and model-error prior values must already share
each channel's declared scale.

For the retained Ramsden case, methane values are numeric ``ppb`` and ethane
values numeric ``ppt``. The paper prints the ethane model-error bounds as ppb,
but its figures, retained data, and historical configuration support ppt.

API reference
-------------

.. autoclass:: openghg_inversions.experimental.ramsden2022.RamsdenPreparedInputs
   :members:
   :no-index:

.. autoclass:: openghg_inversions.experimental.ramsden2022.RamsdenChannelSpec
   :members:
   :no-index:

.. autoclass:: openghg_inversions.experimental.ramsden2022.RamsdenSectorSpec
   :members:
   :no-index:

.. autoclass:: openghg_inversions.experimental.ramsden2022.RamsdenModelSpec
   :members:
   :no-index:

.. autoclass:: openghg_inversions.experimental.ramsden2022.RamsdenResult
   :members:
   :no-index:

.. autofunction:: openghg_inversions.experimental.ramsden2022.build_ramsden_model
   :no-index:

.. autofunction:: openghg_inversions.experimental.ramsden2022.run_ramsden_from_prepared_inputs
   :no-index:

Design history
--------------

The detailed historical comparison and porting decision is retained in
`adding_multi_gas_model_historical_comparison.md
<https://github.com/openghg/openghg_inversions/blob/codex/ramsden-2022-multigas/docs/plans/adding_multi_gas_model_historical_comparison.md>`_.
It documents the behavior preserved from the old branch, the behavior
deliberately excluded, and known historical correctness defects.
