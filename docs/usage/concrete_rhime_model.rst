Concrete RHIME Model
====================

.. note::

   This is a draft description of the current RHIME PyMC model and its
   customization boundary. Code under ``openghg_inversions.models._rhime_compiler``
   is private implementation machinery, not a public model-definition API.

This page makes the model graph behind :func:`run_rhime` and
:func:`run_rhime_multisector` explicit. It has three purposes:

* show the concrete statistical model and its PyMC names;
* show how the standard model can be reconstructed from public component
  helpers; and
* pressure-test how a future likelihood or complete model extension could fit
  without turning compiler internals into a researcher-facing API.

The current builders
--------------------

Both public builders use the same construction stages:

.. code-block:: text

   RhimeModelSpec + canonical inversion inputs
   -> private flux compilation plan
   -> flux states and forward terms
   -> total flux contribution, mu
   -> boundary, offset, error, and likelihood components
   -> PyMC model

The private compiler currently covers only the linear flux part of one
observation channel. Boundary conditions, optional offsets, the RHIME error
model, and the observed distribution are added afterwards by the shared model
assembler.

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

The production builder routes the flux component through the private compiler
so standard and multisector RHIME share one implementation. A researcher does
not need to construct a private plan to write the equivalent concrete
single-flux model. The following uses public component helpers:

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
unique backend name. The compiler also checks that all sector contributions
share an identical observation layout before summing them.

Names and generated names
-------------------------

PyMC model variables currently share one flat effective namespace. A prior can
create more names than the requested base name. In particular, a
reparameterized lognormal requested as ``x_ff`` creates both ``x_ff`` and
``x_ff_latent``.

Both names should be treated as reserved for that prior. No data variable,
other state, forward-term deterministic, or total should use either name. The
current compiler enforces this rule within the flux compilation plan. The
shared assembler still relies on conventional names for later boundary,
offset, error, and likelihood components; there is not yet one allocator for
the complete model namespace.

Knowledge of the ``_latent`` suffix is also duplicated between the compiler
and ``parse_prior``. A future pure prior-description helper should report all
generated names before graph construction, leaving ``parse_prior`` responsible
for creating them.

Component-local namespacing is another possible way to guarantee uniqueness.
An earlier model-configuration prototype used nested PyMC models to prefix
internal variables by component. That made collisions difficult but also made
the final trace names and graph construction less transparent. A future
component namespace should therefore be explicit:

* each component has a stable semantic ID;
* local roles such as ``state``, ``design``, and ``contribution`` are unique
  within that component;
* one name allocator renders those identities into stable backend names; and
* every generated companion name is reserved at the same time.

This preserves the useful uniqueness guarantee without requiring nested PyMC
models or making PyMC prefixes part of scientific model identity.

Alternative likelihood thought experiment
-----------------------------------------

There is currently no likelihood option on ``RhimeModelSpec``. The private flux
compiler stops at ``mu``, and the shared assembler always installs
``add_inferpymc_likelihood_component``.

A plausible first extension is an injected ``likelihood_builder`` on the
direct Python model builder. It would replace the whole current observation
component, including error construction and the observed distribution, rather
than only replacing ``pm.Normal``:

.. code-block:: python

   from dataclasses import dataclass
   from collections.abc import Mapping
   from typing import Protocol

   import pymc as pm
   import xarray as xr
   from pytensor.tensor.variable import TensorVariable

   from openghg_inversions.sigma import SigmaAlignment


   @dataclass(frozen=True)
   class LikelihoodContext:
       data: xr.Dataset
       mu: TensorVariable
       mu_bc: TensorVariable | None
       offset: TensorVariable | None
       sigma_alignment: SigmaAlignment
       sigma_prior: Mapping[str, object]
       power: Mapping[str, object] | float
       pollution_events_from_obs: bool
       no_model_error: bool
       output_dim: str


   @dataclass(frozen=True)
   class LikelihoodResult:
       observed: TensorVariable
       prediction: TensorVariable
       error_scale: TensorVariable | None


   class LikelihoodBuilder(Protocol):
       def __call__(self, context: LikelihoodContext) -> LikelihoodResult:
           ...


   def build_model(
       inv_inputs: xr.Dataset,
       *,
       likelihood_builder: LikelihoodBuilder = build_default_rhime_likelihood,
   ) -> pm.Model:
       ...

This is a design sketch, not current API. Passing a context object avoids a
growing callback signature as observation models evolve. Returning explicit
roles avoids requiring sampling and postprocessing to infer meaning from names
such as ``y`` or ``epsilon``.

Replacing the complete observation component is the cleanest initial contract
because the current helper combines error construction and the observed
distribution. If real alternatives differ only in distribution family, error
construction could later be extracted as a smaller reusable component before
introducing a second callback boundary.

There are two useful levels of extension:

``likelihood_builder``
   Reuses the built-in flux, boundary, and offset assembly but replaces the
   observation/error component. This is appropriate for a Student-t likelihood,
   a different model-error policy, or another single-channel observation model.

``model_builder``
   Consumes canonical prepared inputs and returns a complete PyMC model plus a
   serializable variable-role manifest. This is the broader boundary needed for
   linked tracers, several observation channels, or models distributed by an
   extension package.

The direct Python API should establish these contracts before config or plugin
discovery is added. ``RhimeModelSpec`` should remain serializable; arbitrary
Python callables should not be embedded in it. If external model packages later
need CLI or config selection, a plugin entry point can resolve a stable builder
name to a complete model factory and record the package and builder identity in
output provenance.

Customization stability
-----------------------

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
   developed. They should not be imported by research scripts.

Future extension API
   Likelihood and complete model builders should be introduced only with
   explicit input, naming, sampling, and output-role contracts.
