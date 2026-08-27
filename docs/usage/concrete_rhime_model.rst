Concrete RHIME Model
====================

This page makes the model graph behind :func:`run_rhime` and
:func:`run_rhime_multisector` explicit. It has two purposes:

* show the concrete statistical model and its PyMC names;
* show how the standard model can be reconstructed from public component
  helpers.

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

By default, the runner and configuration template select
``mismatch_model="pollution_event"``. This preserves the fractional-error
equation used by ``run_hbmcmc.py``. The concrete model recipe has no mismatch
default: the runner resolves this selector to ``PollutionEventSettings`` in
the serializable model specification before construction. Select
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

With ``no_model_error=True``, the sampled fractional-error contribution is
omitted and the likelihood scale is the observation error, protected only by
the historical very-small numerical floor. ``min_error`` is not applied in
that branch.

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

CO2 coherent-reduction model
----------------------------

The public :func:`openghg_inversions.rhime.build_co2_model` recipe
consumes the labelled products of a coherent state reduction. Let
``H_alpha`` be the retained-state sensitivity, ``m_alpha`` and ``C_alpha`` its
arithmetic prior mean and covariance, and ``b_fixed`` the fixed affine prior
contribution. The core retained-state terms are

.. math::

   x &\sim \operatorname{LogNormalMoments}(m_\alpha, C_\alpha), \\
   \mu_{CO_2} &= H_\alpha x.

The affine term is part of coherent prior closure; it is not an atmospheric
boundary condition. An explicit state-activity policy omits inactive elements
from the sampled correlated vector while restoring their exact fixed values
in the full public ``flux_scaling`` vector and in ``co2_flux_contribution``.

The CO2 likelihood uses the explicit :class:`~openghg_inversions.observation_error.AggregationError`
selected from prepared inputs. With reported observation standard deviation
``s_y``, optional known mismatch ``s_fixed``, and optional inferred additive
mismatch ``sigma``, its covariance is

.. math::

   R = C_{agg} + \operatorname{diag}
       (s_y^2 + s_{fixed}^2 + \sigma^2),

after applying ``min_error`` as a floor on the total marginal standard
deviation. OpenGHG Inversions does not default ``s_fixed`` to 1 ppm. The
Verification Games fixed-only policy passes ``fixed_model_mismatch=1.0`` and
``no_model_error=True`` visibly. A runnable CO2 configuration and resolver are
tracked in `OPE-79 <https://linear.app/openghg-inversions/issue/OPE-79>`_.

CO2 grouped inner and outer states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same-grid inner and outer fluxes use one ordinary retained state and one
``co2_sensitivity``. Their geography is labelled metadata, not a second model
component. The state axis retains ``basis_group``, ``basis_partition``, and
``region_in_partition`` coordinates; select ``basis_group == "outer"`` rather
than inferring outer entries from integer ranges or positions.

Fixed versus inferred outer entries use the ordinary
:class:`~openghg_inversions.models.StateActivity` contract. To preserve the
fixed-at-one behavior, pass
``StateActivity(fixed_groups=("outer",), fixed_value=1.0)``. Leaving the outer
group active infers those entries with the same correlated LogNormal state as
the inner entries. Exact-zero sensitivity pruning composes with either policy.

The builder accepts one prepared
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior` over the
complete ordered state. Group-specific arithmetic moments must be aligned and
assembled before graph construction. A full prior may contain inner/outer
cross-covariance; it remains part of that one prior and is not also added to
the observation covariance.

``co2_flux_contribution`` contains the complete flux prediction. The model does
not construct separate inner or outer terms. Outputs can reconstruct either
view by selecting the corresponding ``basis_group`` entries from the stored
sensitivity and ``flux_scaling`` state.

With optional boundary conditions and offset present, the likelihood mean is

.. math::

   \mathtt{modelled\_concentration}
   = \mathtt{fixed\_prior\_contribution}
   + \mathtt{co2\_flux\_contribution}
   + \mathtt{mu\_bc}
   + \mathtt{offset}.

An output may report a composite baseline by reconstructing the outer flux view
and adding it to ``mu_bc + offset``. This is an output policy only: boundary
conditions, offsets, and outer flux remain distinct scientific terms. Sector
combinations are likewise a general state-grouping choice rather than an
outer-region model option.

The model builder accepts explicit scientific arrays rather than a dataset.
For durable prepared artifacts, :func:`openghg_inversions.rhime.run_rhime_co2`
is the public replay seam: it validates and materializes the selected arrays,
resolves aggregation error, calls the explicit builder, samples, and stores a
JSON variable-role and model-provenance manifest on the returned
``InferenceData``. A prepared ``fixed_model_mismatch`` is preserved when the
runner argument is ``None``; an explicit scalar or labelled vector overrides
it. Persist gathered-state traces with
:func:`openghg_inversions.serialization.save_inferencedata`, which uses the
same MultiIndex-safe boundary as standard and multisector RHIME outputs.
The current prepared-input runner does not accept or construct an
outer-specific object. It constructs the complete retained prior from the
prepared arithmetic mean and covariance, then forwards the prepared activity
policy to the builder.

CO2/O2 shared-state model
-------------------------

The CO2/O2 recipe applies one retained state to both observation channels.
Its public boundaries are
:func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` for labelled
preparation, :func:`openghg_inversions.rhime.co2.build_co2_o2_model` for graph
construction, and
:func:`openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs` for
materialization, sampling, and trace metadata.

Partition that state as

.. math::

   \alpha =
   \begin{bmatrix}
      \alpha_{shared} \\
      \alpha_{CO_2,ocean} \\
      \alpha_{O_2,ocean}
   \end{bmatrix},

where :math:`\alpha_{shared}` contains the GPP, TER, and fossil-fuel states.
The joint affine model is

.. math::

   H_{joint} =
   \begin{bmatrix}
      H_{CO_2,shared} & H_{CO_2,ocean} & 0 \\
      H_{O_2,shared}^{eff} & 0 & H_{O_2,ocean}
   \end{bmatrix},
   \qquad
   b_{joint} =
   \begin{bmatrix} b_{CO_2} \\ b_{O_2} \end{bmatrix},
   \qquad
   \mu_{joint} = b_{joint} + H_{joint}\alpha.

Equivalently, coherent reduction may be written in centred or affine form,

.. math::

   \mu_{joint}
   = \mu_{prior} + H_{joint}(\alpha - m_\alpha)
   = (\mu_{prior} - H_{joint}m_\alpha) + H_{joint}\alpha.

The prepared ``fixed_prior_contribution`` is the parenthesized affine
intercept, not the complete prior-forward concentration.

Thus this is a row-stacked, block-sparse sensitivity acting on one state vector,
not two independent block-diagonal models. Its fixed-error likelihood is

.. math::

   \begin{bmatrix} y_{CO_2} \\ y_{O_2} \end{bmatrix}
   \mid \alpha
   \sim \mathcal N\!\left(
      \mu_{joint},
      \begin{bmatrix}
         A_{CO_2,CO_2} & A_{CO_2,O_2} \\
         A_{O_2,CO_2} & A_{O_2,O_2}
      \end{bmatrix}
      + \operatorname{diag}(s_{independent}^2)
   \right).

These quantities must come from one coherent reduction. With native state
mean :math:`m`, covariance :math:`B`, joint native observation sensitivity
:math:`G`, and retained-state restriction :math:`\Pi`,

.. math::

   C_\alpha &= \Pi B\Pi^\mathsf{T}, \\
   H_{joint} &= GB\Pi^\mathsf{T}C_\alpha^{-1}, \\
   b_{joint} &= Gm - H_{joint}\Pi m, \\
   A &= GBG^\mathsf{T} - H_{joint}C_\alpha H_{joint}^\mathsf{T}.

In particular, the off-diagonal :math:`A_{CO_2,O_2}` block is part of the
coherent-reduction contract. See the :doc:`full derivation
<coherent_reduction>` for its assumptions and limitations.

Preparation accepts separate native channel arrays, then gathers their rows on
one ``(species, channel_observation)`` observation index before the model
applies the joint sensitivity once. Each row still retains its declared native
units and numerical scale. Verification-game inputs may use ppm for both
channels, while real atmospheric O2 observations may use per-meg delta(O2/N2).
The prepared channel fields are named ``co2_sensitivity`` and
``o2_sensitivity``; their gathered model-data variable is
``co2_o2_sensitivity``.
``independent_error_sd`` and every covariance row and column must use the
corresponding observation-row units. Any future numerical scaling or whitening
must be a named transformation applied consistently to observations, model
mean, independent error, and all joint covariance blocks, while retaining
physical-unit outputs and provenance. The displayed row stack is the
mathematical model: every sensitivity and covariance block must still be produced
by the same reduction.

The graph names the gathered linear signal ``co2_o2_flux_contribution`` and the
complete affine sum and likelihood mean ``modelled_concentration``. In
model-variable vocabulary,

.. math::

   \mathtt{modelled\_concentration}
   = \mathtt{fixed\_prior\_contribution}
   + \mathtt{co2\_o2\_flux\_contribution}.

Persist sampled CO2/O2 results with
:func:`openghg_inversions.serialization.save_inferencedata` and restore them
with :func:`openghg_inversions.serialization.load_inferencedata`; this is the
declared boundary for preserving gathered MultiIndex coordinates.

The signed oxidation ratio is fixed in this recipe and already folded into the
shared-state O2 sensitivity. When it is representable by retained-state or
source-resolved values :math:`R`,

.. math::

   H_{O_2,shared}^{eff}
   = H_{O_2,ratio\text{-}free}\operatorname{diag}(R).

Native paired-flux construction may instead apply spatially resolved ratios
before footprint convolution, in which case no unique retained-state
:math:`R` is available. Preparation records that status and its reason rather
than inventing scalar values; the supplied effective O2 sensitivity remains the
scientific input.

Because this recipe receives the O2 sensitivity with the fixed ratio already
applied upstream, its builder applies the gathered sensitivity directly to the
unchanged shared state with ``apply_linear_sensitivity``. If a fixed or
inferred oxidation ratio were instead explicit model state, the recipe would
visibly form ``o2_state = oxidation_ratio * co2_state`` before applying the
ratio-free O2 sensitivity. The :doc:`Ramsden methane/ethane model
<../experimental/ramsden2022>` follows that explicit pattern for its emission
ratio. OPE-118 owns that future CO2/O2 coupling work.

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
   sigma_prior = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}

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

Advanced whole-model compatibility boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
