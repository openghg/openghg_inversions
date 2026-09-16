Standard and multisector RHIME models
=====================================

This page makes the model graph behind :func:`run_rhime` and
:func:`run_rhime_multisector` explicit. The advanced CO₂-only and linked
CO₂/O₂ graphs have their canonical home in :doc:`co2_models`. This page has two
purposes:

* show the concrete statistical model and its PyMC names;
* show how the standard model can be reconstructed from public component
  helpers.

.. _co2-coherent-reduction-model:

The CO₂ coherent-reduction model documentation moved to the
:ref:`CO₂-only model recipe <co2-only-model>`.

.. _co2-grouped-inner-and-outer-states:

The CO₂ grouped inner and outer state documentation moved to the
:ref:`grouped CO₂ state recipe <co2-grouped-states>`.

.. _co2-o2-shared-state-model:

The CO₂/O₂ shared-state model documentation moved to the
:ref:`linked CO₂/O₂ model recipe <linked-co2-o2-model>`.

The current builders
--------------------

Each recipe directly composes one readable concrete graph. The standard graph
lives beside :func:`run_rhime` in ``openghg_inversions.rhime.standard``; the
multisector graph lives beside :func:`run_rhime_multisector` in
``openghg_inversions.rhime.multisector``:

.. code-block:: text

   RhimeModelSpec + canonical inversion inputs
   -> concrete standard or multisector builder
   -> public linear-component helpers
   -> total flux contribution, mu
   -> boundary, offset, error, and likelihood components
   -> PyMC model

.. _rhime-builder-stability:

Stability contract
------------------

The concrete builders are the production reference implementations. Their
explicit PyMC code is the primary model definition for scientific review,
auditing, and user confidence. Shared source selection and ordinary scientific
components remain separate helpers, while model-specific composition stays in
the recipe that runs it.

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

An optional global, site, or site-by-period offset contributes ``offset``. The
mean of the observed distribution is therefore

.. math::

   \mu_{\mathrm{obs}} = \mu + \mu_{bc} + \mathrm{offset},

where omitted components are left out of the sum.

The configuration template explicitly selects
``mismatch_model="pollution_event"``. This preserves the fractional-error
equation used by ``run_hbmcmc.py`` for users who start from that template.
Direct runner calls must make a likelihood selection explicitly. The concrete
model recipe has no mismatch default: parameter resolution converts the
selector to ``PollutionEventSettings`` in the serializable model specification
before construction. Select
``mismatch_model="additive_sigma"`` for an absolute concentration-scale
mismatch instead; this is a resolved model option and does not use the custom
``likelihood_builder`` extension point. Additive sigma does not select the
prepared ``min_error`` input unless ``use_minimum_error_floor=True`` is also
set. Aggregation error is disabled by default. Let
:math:`P` be the pollution event and let ``sigma`` be the observation-aligned
fractional model-error parameter. With the default
``pollution_events_from_obs=False``,

.. math::

   P &= |\mu|, \\
   \epsilon &= \max\left(
     \sqrt{\mathrm{error}^2 + (P\sigma)^{\mathrm{power}}},
     \mathrm{min\_error}
   \right), \\
   y &\sim \mathcal{N}(\mu_{\mathrm{obs}}, \epsilon^2).

When ``pollution_events_from_obs=True``, the modern recipe derives the
pollution event from observations after removing the complete baseline,
including any boundary and offset contributions. Thus
:math:`P=|Y-(\mu_{bc}+\mathrm{offset})|` when a baseline exists; without one,
it uses :math:`P=|Y|+10^{-6}\operatorname{mean}(Y)`.

The ``run_hbmcmc.py`` compatibility path retains its historical boundary-only
variant: :math:`P=|Y-\mu_{bc}|`, even when an offset is also included in
:math:`\mu_{\mathrm{obs}}`. That exception preserves existing configurations;
it is not the scientific default for new RHIME recipes.

Select ``mismatch_model="fixed_error"`` to omit inferred mismatch error. Its
likelihood scale uses the reported observation error and does not select
``min_error``.
``run_hbmcmc.py`` privately preserves the different historical floor and
unused-variable details of its two ``no_model_error`` routes.

Aggregation covariance is an explicit advanced opt-in. If a caller selects a
prepared covariance :math:`C_{agg}` with marginal variance
:math:`v_{agg}=\operatorname{diag}(C_{agg})`, the marginal floor and observed
distribution become

.. math::

   v_{raw} &= \mathrm{error}^2 + (P\sigma)^{\mathrm{power}}, \\
   v_{ind} &= v_{raw} + \max\left(
     \mathrm{min\_error}^2-v_{raw}-v_{agg}, 0
   \right), \\
   \epsilon &= \sqrt{v_{ind}+v_{agg}}, \\
   y &\sim \mathcal{N}\left(
     \mu_{\mathrm{obs}}, \operatorname{diag}(v_{ind})+C_{agg}
   \right).

Selecting aggregation error says only that this fixed covariance should enter
the likelihood. It does not perform, imply, or verify a coherent prior and
forward-model transformation. Covariance obtained by marginalising native
states must therefore be supplied together with the matching transformed
prior and forward operator. Merely finding an aggregation-error array in
prepared inputs does not opt a run into this model.

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
   * - Optional site/global offset
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
     - Flux sensitivity data
     - ``H``
   * - ``mu``
     - Flux contribution
     - ``hx @ x``
   * - ``hbc``
     - Boundary sensitivity data
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

   from openghg_inversions.models import (
       add_linear_component,
       prepare_linear_sensitivity,
       registered_model,
   )
   from openghg_inversions.models.pollution_event import add_pollution_event_likelihood
   from openghg_inversions.observation_error import resolve_aggregation_error
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
   sigma_prior = {"pdf": "uniform", "lower": 0.0, "upper": 0.1}

   sigma_alignment = SigmaAlignment.from_observations(
       inv_inputs["mf"],
       frequency=None,
       per_site=True,
   )
   flux_sensitivity = prepare_linear_sensitivity(inv_inputs["H"])
   boundary_sensitivity = prepare_linear_sensitivity(inv_inputs["H_bc"])

   with registered_model() as model:
       flux = add_linear_component(
           flux_sensitivity,
           data_name="hx",
           prior_args=x_prior,
           var_name="x",
           output_name="mu",
           output_dim="nmeasure",
       )
       boundary = add_linear_component(
           boundary_sensitivity,
           data_name="hbc",
           prior_args=bc_prior,
           var_name="bc",
           output_name="mu_bc",
           output_dim="nmeasure",
       )
       pollution_mean = flux.output
       baseline_mean = boundary.output
       modelled_mean = pollution_mean + baseline_mean

       add_pollution_event_likelihood(
           observations=inv_inputs["mf"],
           observation_error=inv_inputs["mf_error"],
           minimum_error=inv_inputs["min_error"],
           aggregation_error=resolve_aggregation_error(inv_inputs, "none"),
           mean=modelled_mean,
           pollution_mean=pollution_mean,
           pollution_event_baseline=baseline_mean,
           sigma_alignment=sigma_alignment,
           sigma_prior=sigma_prior,
           power=1.99,
           pollution_events_from_obs=False,
           no_model_error=False,
           output_dim="nmeasure",
       )

This example is deliberately concrete and editable. It is suitable when a
model developer needs to change graph construction directly. It does not
automatically participate in the complete ``run_rhime`` output pipeline; that
pipeline still selects one of the built-in model builders.

``prepare_linear_sensitivity`` is the single owning boundary for exact-zero column
inspection. It removes those columns from the backend sensitivity while retaining
their full labelled-state mapping; ``state_activity=None`` therefore samples
every retained state. An explicit ``StateActivity`` fixes or groups scientific
states but cannot restore a structurally absent column. For shared or
correlated states, use ``apply_linear_sensitivity`` to apply another prepared
forward operator without constructing a second prior.

Multisector model
-----------------

For the current shared-basis multisector model, each sector has an independent
state and forward contribution:

.. math::

   x_s &\sim p_s, \\
   \mu_s &= H_s x_s, \\
   \mu &= \sum_s \mu_s.

If the normalized PyMC suffix for sector ``s`` is ``ff``, its variables are
``x_ff`` and ``mu_ff`` and its sensitivity data is ``hx_ff``. Source values select
the corresponding ``H`` slices; sector names provide model identities. They
are not required to be the same strings.

Every sector state and every reparameterization-generated latent must have a
unique backend name. Concrete composition relies on PyMC to reject duplicate
generated names.

Names and generated names
-------------------------

PyMC model variables currently share one flat effective namespace. A prior can
create more names than the requested base name. In particular, a
reparameterized lognormal requested as ``x_ff`` creates both ``x_ff`` and
``x_ff_latent``.

Both names should be treated as reserved for that prior. No data variable,
other state, forward-term deterministic, or total should use either name. PyMC
enforces this during concrete composition. The shared observation-component
helper still relies on conventional names for boundary, offset, error, and
likelihood components; there is not yet one allocator for the complete model
namespace.

Generated-name reporting, whole-model allocation, and component namespaces are
not implemented. They are tracked in
`issue #532 <https://github.com/openghg/openghg_inversions/issues/532>`_.

Alternative models and likelihoods
----------------------------------

Direct-Python likelihood builders enter through ``run_rhime``,
``run_rhime_multisector``, or ``run_rhime_from_prepared_inputs``.
Complete-model builders remain available only at the prepared-input boundary.
Callables are never read from configuration or stored on ``RhimeModelSpec`` or
``RhimeRunSpec``, so model and run specs remain serializable. There is no
entry-point or config-file plugin registry.

A concrete recipe owns the complete forward-model mean: pollution, baseline,
and optional offset contributions are composed visibly and packaged as named
forward terms before the shared built-in dispatcher is invoked. The runner
stores one typed settings value on ``RhimeModelSpec``; the dispatcher calls the
ordinary built-in equation with only that likelihood's inputs. A custom caller
instead supplies a mean-only callable. Every likelihood receives the completed
concentration, prepared observations and reported observation error, a
validated ``AggregationError``, and output dimension. Built-in pollution-event
scaling additionally receives the named pollution and baseline terms.
``likelihood_kwargs`` is reserved for custom callables.
The builder adds and returns the canonical observed variable ``y`` and also
adds the canonical marginal error scale ``epsilon``.

``likelihood_kwargs`` is valid only when a custom likelihood builder is active.
The runner expands the mapping into that callable and records it with the
callable identity in result and saved builder metadata.

The editable example in :doc:`customising_rhime` implements a fixed-error
Student-t likelihood using only those common inputs. Pass it directly to the
ordinary runner:

.. code-block:: python

   from my_project.likelihoods import likelihood_builder
   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       config_file="config.ini",
       mismatch_model=None,
       likelihood_builder=likelihood_builder,
   )

This example supports independent fixed aggregation-error representations. A
custom likelihood with additional options declares them in its own signature,
and the runner supplies only those values through ``likelihood_kwargs``.

Pass ``add_offset=True`` and ``offset_args={"per_site": False}`` as Python or
configuration options to combine it with one global scalar offset. The
default ``per_site=True`` retains the existing site or site-period offset
design.

Keep scientific customization implementations in one tested location. The
:doc:`customising_rhime` guide contains the editable Student-t example and the
minimal ordinary-runner call; this concrete-model page does not duplicate it.

Ordinary likelihood builders keep the ``y`` and ``epsilon`` names used by
sampling and postprocessing. The runner records the likelihood builder's
module and qualified name, so direct-Python likelihoods remain identifiable in
persisted inversion outputs.

Labelled per-site IID mismatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The built-in
:func:`openghg_inversions.models.site_sigma.add_site_sigma_gaussian_likelihood`
provides a labelled, run-level IID site-mismatch component. It derives stable
site labels in first-observation order and uses exactly one standard deviation
per site:

.. math::

   R = A + D_{obs}
       + \operatorname{diag}\left(\sigma_{site}[\operatorname{site}(i)]^2\right).

Here ``A`` is the selected aggregation-error covariance and ``D_obs`` is the
diagonal covariance from reported observation errors. Each is included once.
This likelihood does not apply ``min_error`` or ``fixed_model_mismatch`` and
does not add temporal correlation. In particular, it is an IID component, not
the fixed-OU model.

Select it through the existing Python-only likelihood seam. For inferred site
amplitudes, pass an explicit positive prior:

.. code-block:: python

   from openghg_inversions.models import add_site_sigma_gaussian_likelihood
   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       config_file="config.ini",
       mismatch_model=None,
       likelihood_builder=add_site_sigma_gaussian_likelihood,
       likelihood_kwargs={
           "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
       },
   )

The ``0.75`` value above is an explicit scale in ppm because all prior
parameters and fixed amplitudes are interpreted in the observations'
concentration units; OpenGHG Inversions does not choose a universal default
scale. To use known values instead, replace ``site_amplitude_prior`` with
``fixed_site_amplitudes``, a mapping covering every observed site label. Pass
exactly one of these two options.

The graph names the labelled site vector ``sigma_site`` on
``sigma_site_dim``, with ``sigma_site_index`` retaining the observation-to-site
mapping. Fixed amplitudes are stored as model data and inferred amplitudes as
posterior variables. The ordinary runner records the callable identity,
and JSON-compatible options in the saved output. The component reuses OpenGHG
Inversions' aggregation-error and Gaussian likelihood machinery.

Advanced whole-model compatibility boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A complete model builder is an advanced escape hatch available only through
``run_rhime_from_prepared_inputs``. It receives a
``RhimeModelBuilderContext`` containing the validated ``RhimePreparedInputs``,
updated ``RhimeRunSpec``, and validated single- versus multi-sector mode, and
returns a ``RhimeModelBuildResult``. Ordinary standard and multisector recipes
never construct or consume this context. Complete builders also bypass the
ordinary recipe-owned materialization step, so they must select, validate, and
materialize any lazy arrays they consume:

.. code-block:: python

   from importlib.metadata import version

   import pymc as pm

   from openghg_inversions.rhime import (
       RhimeModelBuilderContext,
       RhimeModelBuildResult,
       run_rhime_from_prepared_inputs,
   )
   from openghg_inversions.models import (
       add_coords,
       registered_model,
   )


   def complete_model(
       context: RhimeModelBuilderContext,
   ) -> RhimeModelBuildResult:
       data = context.prepared_inputs.inv_inputs
       with registered_model() as model:
           add_coords(data.coords, model_dims=("nmeasure",))
           mean = pm.Normal("custom_mean", mu=0.0, sigma=10.0)
           pm.Normal(
               "custom_y",
               mu=mean,
               sigma=data.mf_error.values,
               observed=data.mf.values,
               dims="nmeasure",
           )
       return RhimeModelBuildResult(
           model=model,
           variable_roles={
               "observation": "mf",
               "observation_error": "mf_error",
               "concentration": "custom_y",
           },
           supported_output_formats=("none", "inv_out"),
           metadata={
               "package": "my-rhime-models",
               "version": version("my-rhime-models"),
               "model": "complete_model_v1",
           },
       )


   result = run_rhime_from_prepared_inputs(
       prepared_inputs=prepared,
       run_spec=run_spec,
       model_builder=complete_model,
   )

The compatibility rules are explicit:

* a complete builder must return a concrete ``pm.Model`` and a non-empty role
  manifest; ``concentration`` is required;
* every declared role name must exist in either the model or prepared inversion
  inputs;
* builders must construct their graph with ``registered_model()`` before
  calling ``add_coords`` or public model components, so
  ``RhimeSampler`` can restore MultiIndexes and auxiliary scientific
  coordinates;
* builder metadata must be JSON serializable, and external packages should
  record their package version and stable model identity there;
* custom builders support only ``output_format="none"`` unless they explicitly
  declare more formats; declaring a format promises that the trace, roles,
  basis layout, and variables required by that output really are present;
* ``model_builder`` and ``likelihood_builder`` are mutually exclusive; and
* a component that does not exist, such as inferred model error in a fixed-error
  model, is omitted from the role manifest rather than represented by a magic
  name.

The built-in standard and multisector builders produce the same
``RhimeModelBuildResult`` contract on ``RhimeResult.model_build_result``. Their
sector roles use keys such as ``flux_scale:FF`` and
``flux_contribution:FF``. The low-level recipe builders return a plain
``pm.Model``; the corresponding ``*_model_result`` wrappers add the runner and
output metadata contract.

Structurally distinct observation-channel models belong in readable named
recipes with concrete builders. The active design guidance is
:doc:`../development/rhime_model_development`; the retired semantic-model work
in issue #528 remains research evidence rather than production architecture.

Customization boundaries
------------------------

The current customization levels are:

Supported high-level options
   Priors, boundary conditions, offsets, sigma alignment, and existing error
   options supplied through the public RHIME builders and model spec.

Supported low-level components
   Public functions in ``openghg_inversions.models`` can be composed inside a
   user-owned ``pm.Model`` as shown above.

Recipe-local model composition
   Copy or modify the readable concrete builder in
   ``openghg_inversions.rhime.standard`` or
   ``openghg_inversions.rhime.multisector`` when an existing option or shared
   component is insufficient.
