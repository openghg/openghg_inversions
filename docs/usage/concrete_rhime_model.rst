Concrete RHIME Model
====================

.. note::

   This is a draft description of the current RHIME PyMC model and its
   customization boundary. Code under ``openghg_inversions.models._rhime_compiler``
   is private implementation machinery, not a public model-definition API.

This page makes the model graph behind :func:`run_rhime` and
:func:`run_rhime_multisector` explicit. It has two purposes:

* show the concrete statistical model and its PyMC names;
* show how the standard model can be reconstructed from public component
  helpers.

The current builders
--------------------

:data:`~openghg_inversions.models.rhime.RhimeBuilderStrategy` defines the two
public strategy values used by ``RhimeModelSpec``. Direct composition is the
default:

.. code-block:: text

   RhimeModelSpec + canonical inversion inputs
   -> concrete standard or multisector builder
   -> public linear-component helpers
   -> total flux contribution, mu
   -> boundary, offset, error, and likelihood components
   -> PyMC model

Set ``builder_strategy="compiled"`` on the spec to opt into the alternative
path:

.. code-block:: text

   RhimeModelSpec + canonical inversion inputs
   -> private flux compilation plan
   -> flux states and forward terms
   -> total flux contribution, mu
   -> the same boundary, offset, error, and likelihood components
   -> PyMC model

The compiler is retained as experimental machinery for developing a more
general semantic representation. It currently covers only the linear flux part
of one observation channel. Both paths use the same source/sector resolution,
gathered ragged-state handling, and prior selection before constructing the
PyMC graph. They preserve the same public variable names and dimensions.

.. _rhime-builder-stability:

Stability contract
------------------

The concrete builders are the readable reference implementations. Their
explicit PyMC code is the primary model definition for scientific review,
auditing, and user confidence. ``builder_strategy="concrete"`` therefore
remains the default.

The compiled strategy is a public opt-in extension and regression-checking
path. Its plan and compiler objects remain private and may evolve, but
``builder_strategy="compiled"`` and the externally meaningful graph contract
for unchanged components are stable. That contract includes named variables,
dimensions and scientific coordinates, registered model data, deterministic
contributions, and seeded prior-predictive behaviour.

There is no automatic fallback between strategies. A failure in the selected
strategy stops model construction. If a future compiler feature intentionally
changes part of the graph, that divergence should be explicit, narrowly scoped,
and covered by a focused test; components outside that feature should continue
to match the concrete reference implementation.

Source/design resolution and the boundary, offset, error, and likelihood
components remain shared. This keeps parity meaningful and avoids independent
copies silently drifting while compiler extensions are developed.

Standard single-flux model
--------------------------

Let ``H`` be the flux sensitivity matrix with dimensions
``(region, nmeasure)`` in canonical inversion inputs. The builder transposes it
to observation-first order before registering it as PyMC data. The flux model
is

.. math::

   x &\sim p_x, \\
   \mu &= Hx.

With the default prior, ``x`` is the physical deterministic transform of a
standard-normal ``x_latent``:

.. math::

   x_{\mathrm{latent}} &\sim \mathcal{N}(0, 1), \\
   x &= \exp(\mu_x + \sigma_x x_{\mathrm{latent}}).

When boundary-condition scaling is enabled,

.. math::

   bc &\sim p_{bc}, \\
   \mu_{bc} &= H_{bc}bc.

An optional site or site-by-period offset contributes ``offset``. The mean of
the observed distribution is therefore

.. math::

   \mu_{\mathrm{obs}} = \mu + \mu_{bc} + \mathrm{offset},

where omitted components are left out of the sum.

The default RHIME error model uses observation error, a minimum error, and an
observation-aligned model-error scale ``sigma``. Unless
``pollution_events_from_obs`` is selected,

.. math::

   \epsilon =
   \max\left(
     \sqrt{
       \mathrm{error}^2 +
       \left(\left|\mu\right|\sigma\right)^{\mathrm{power}}
     },
     \mathrm{min\_error}
   \right).

The current observed distribution is

.. math::

   y \sim \mathcal{N}(\mu_{\mathrm{obs}}, \epsilon).

The default priors are:

.. list-table::
   :header-rows: 1
   :widths: 20 42 38

   * - Quantity
     - Prior
     - PyMC variables
   * - Flux scaling
     - Lognormal with mean 1 and standard deviation 1, reparameterized
     - ``x_latent`` and ``x``
   * - Boundary scaling
     - Truncated normal with mean 1, standard deviation 0.05, and lower bound 0
     - ``bc``
   * - Model error
     - Uniform from 0.1 to 3
     - ``sigma``
   * - Optional offset
     - Normal with mean 0 and standard deviation 1
     - ``offset_latent`` and ``offset``

This table describes the Python builder default used when ``x_prior`` is
omitted. The shipped RHIME config template instead supplies an explicit
``x_prior`` without ``reparameterise=True``. That config therefore creates
``x`` directly, without ``x_latent``. Add ``"reparameterise": True`` to the
config prior to request the API-default parameterization shown here.

The important default model-data and deterministic names are:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Name
     - Role
     - Canonical input
   * - ``hx``
     - Flux design data
     - ``H``
   * - ``mu``
     - Flux contribution
     - ``hx @ x``
   * - ``hbc``
     - Boundary design data
     - ``H_bc``
   * - ``mu_bc``
     - Boundary contribution
     - ``hbc @ bc``
   * - ``Y``
     - Observed mole fraction data
     - ``mf``
   * - ``error``
     - Observation error data
     - ``mf_error``
   * - ``min_error``
     - Minimum error data
     - ``min_error``
   * - ``epsilon``
     - Observation-aligned error scale
     - RHIME error model
   * - ``y``
     - Observed random variable
     - Normal likelihood

Equivalent construction from public helpers
-------------------------------------------

The default production builder constructs the graph in the same direct style
shown below. A researcher does not need to construct a private plan to write an
equivalent concrete single-flux model. The following uses public component
helpers:

.. code-block:: python

   import pymc as pm

   from openghg_inversions.models import (
       CoordRegistry,
       add_inferpymc_likelihood_component,
       add_linear_component,
       attach_coord_registry,
   )
   from openghg_inversions.sigma import SigmaAlignment

   x_prior = {
       "pdf": "lognormal",
       "mean": 1.0,
       "stdev": 1.0,
       "reparameterise": True,
   }
   bc_prior = {
       "pdf": "truncatednormal",
       "mu": 1.0,
       "sigma": 0.05,
       "lower": 0.0,
   }
   sigma_prior = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}

   sigma_alignment = SigmaAlignment.from_frequency(
       inv_inputs["site_indicator"],
       frequency=None,
       per_site=True,
   )

   with pm.Model() as model:
       attach_coord_registry(model, CoordRegistry())

       flux = add_linear_component(
           inv_inputs["H"],
           data_name="hx",
           prior_args=x_prior,
           var_name="x",
           output_name="mu",
           output_dim="nmeasure",
       )
       boundary = add_linear_component(
           inv_inputs["H_bc"],
           data_name="hbc",
           prior_args=bc_prior,
           var_name="bc",
           output_name="mu_bc",
           output_dim="nmeasure",
       )
       add_inferpymc_likelihood_component(
           inv_inputs,
           mu=flux.output,
           mu_bc=boundary.output,
           sigprior=sigma_prior,
           sigma_alignment=sigma_alignment,
           output_dim="nmeasure",
       )

This example is deliberately concrete and editable. It is suitable when a
model developer needs to change graph construction directly. It does not
automatically participate in the complete ``run_rhime`` output pipeline; that
pipeline still selects one of the built-in model builders.

Multisector model
-----------------

For the current shared-basis multisector model, each sector has an independent
state and forward contribution:

.. math::

   x_s &\sim p_s, \\
   \mu_s &= H_s x_s, \\
   \mu &= \sum_s \mu_s.

If the normalized PyMC suffix for sector ``s`` is ``ff``, its variables are
``x_ff`` and ``mu_ff`` and its design data is ``hx_ff``. Source values select
the corresponding ``H`` slices; sector names provide model identities. They
are not required to be the same strings.

Every sector state and every reparameterization-generated latent must have a
unique backend name. Concrete composition relies on PyMC to reject duplicate
generated names. The opt-in compiler performs whole-plan name and observation
layout checks before mutating the active model.

Names and generated names
-------------------------

PyMC model variables currently share one flat effective namespace. A prior can
create more names than the requested base name. In particular, a
reparameterized lognormal requested as ``x_ff`` creates both ``x_ff`` and
``x_ff_latent``.

Both names should be treated as reserved for that prior. No data variable,
other state, forward-term deterministic, or total should use either name. PyMC
enforces this during concrete composition; the opt-in compiler detects it
during flux-plan preflight. The shared observation-component helper still
relies on conventional names for boundary, offset, error, and likelihood
components; there is not yet one allocator for the complete model namespace.

Knowledge of the ``_latent`` suffix is also duplicated between the compiler
and ``parse_prior``. Generated-name reporting, whole-model allocation, and
component namespaces are not implemented. They are tracked in
`issue #532 <https://github.com/openghg/openghg_inversions/issues/532>`_.

Alternative models and likelihoods
----------------------------------

There is currently no likelihood option on ``RhimeModelSpec``. The private flux
compiler stops at ``mu``, and the shared assembler always installs
``add_inferpymc_likelihood_component``. Likewise, the prepared-input runner
selects one of the built-in standard or multisector model builders.

Public contracts for alternative likelihood builders, complete model builders,
output roles, and possible later plugin discovery are not implemented. That
design and its pressure tests are tracked in
`issue #533 <https://github.com/openghg/openghg_inversions/issues/533>`_.
The more general semantic model and observation-channel representation is
tracked in
`issue #528 <https://github.com/openghg/openghg_inversions/issues/528>`_.

Customization boundaries
------------------------

The current customization levels are:

Supported high-level options
   Priors, boundary conditions, offsets, sigma alignment, and existing error
   options supplied through the public RHIME builders and model spec.

Supported low-level components
   Public functions in ``openghg_inversions.models`` can be composed inside a
   user-owned ``pm.Model`` as shown above.

Private implementation
   ``_FluxPlan``, ``_StatePlan``, ``_ForwardTermPlan``, and
   ``_compile_loop_sum`` may change while the semantic model representation is
   developed. Set ``RhimeModelSpec(builder_strategy="compiled")`` to exercise
   that path without importing private compiler objects.
