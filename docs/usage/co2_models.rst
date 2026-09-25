CO₂ model recipes
=================

This page describes the scientific graphs and current prepared-input replay
seams for the CO₂-only and linked CO₂/O₂ recipes. Start with the
:doc:`CO₂ model family <co2_model_family>` support matrix before using these
interfaces.

.. _co2-only-model:

CO2 coherent-reduction model
----------------------------

The public :func:`openghg_inversions.rhime.build_co2_model` recipe
consumes the labelled products of a coherent state reduction. Let
``H_alpha`` be the retained-state sensitivity, ``m_alpha`` and ``C_alpha`` its
arithmetic prior mean and covariance, and ``b_fixed`` the fixed affine prior
contribution. The core retained-state terms are

.. math::

   x &\sim \operatorname{LogNormalMoments}(m_\alpha, C_\alpha), \\
   \mathtt{co2\_flux\_contribution} &= b_{fixed} + H_\alpha x, \\
   \mathtt{modelled\_concentration}
      &= \mathtt{co2\_flux\_contribution} + \mu_{bc} + \mathtt{offset}.

The boundary and offset terms in the last line are optional. When either is
omitted, it is also omitted from the sum.

The covariance projection identities in this reduction are exact for a
jointly Gaussian native state. The builder reuses the resulting arithmetic
moments for a correlated LogNormal positive state, which is a moment-matched
approximation rather than an exact marginalization of a LogNormal native
state. The positivity constraint and the suitability of that approximation
therefore require separate scientific justification.

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
       (s_y^2 + s_{fixed}^2 + \sigma^2).

The lower-level builder adds inferred ``sigma`` only when the caller supplies a
``sigma_alignment``. The public :func:`~openghg_inversions.rhime.run_rhime_co2`
runner creates a site-specific alignment over one shared time period by
default, and uses an independent HalfNormal prior with ``sigma=1.0`` in the
observations' concentration units unless ``sigma_prior`` overrides it. Override
``sigma_prior`` when that scale is inappropriate for the observation units or
scientific application. Set ``no_model_error=True`` to disable this inferred
term.

The CO₂ runner does not consume ``min_error`` or apply a minimum-error floor.
Other recipes and components own their own floor settings; for example, the
standard additive-sigma likelihood can opt into ``min_error`` explicitly.
OpenGHG Inversions does not default ``s_fixed`` to 1 ppm. The
Verification Games fixed-only policy passes ``fixed_model_mismatch=1.0`` and
``no_model_error=True`` visibly. A runnable CO2 configuration and resolver are
described below; staged routing and common output integration remain tracked in
`OPE-79 <https://linear.app/openghg-inversions/issue/OPE-79>`_.

For the matched fixed-tau Ornstein--Uhlenbeck (OU) likelihood with independently
inferred site amplitudes, use :ref:`the package-supported cached-sigma CO2 runner
<co2-cached-sigma-recipe>`.

.. _co2-grouped-states:

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
not construct separate inner or outer terms. Exact-zero columns are pruned from
the model's stored ``co2_sensitivity`` and use a retained-state dimension,
while ``flux_scaling`` preserves the complete labelled state. To reconstruct a
complete inner or outer view, select matching ``basis_group`` entries from the
original prepared ``H`` and ``flux_scaling``. Do not multiply the full state by
the pruned model-data variable unless an explicit retained-to-full mapping is
also applied.

With optional boundary conditions and offset present, the likelihood mean is

.. math::

   \mathtt{co2\_flux\_contribution}
   = \mathtt{fixed\_prior\_contribution}
   + H\,\mathtt{flux\_scaling},

.. math::

   \mathtt{modelled\_concentration}
   = \mathtt{co2\_flux\_contribution}
   + \mathtt{mu\_bc}
   + \mathtt{offset}.

An output may report a composite baseline by reconstructing the outer flux view
and adding it to ``mu_bc + offset``. This is an output policy only: boundary
conditions, offsets, and outer flux remain distinct scientific terms. Sector
combinations are likewise a general state-grouping choice rather than an
outer-region model option.

Run the ordinary prepared-input CO2 runner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model builder accepts explicit scientific arrays rather than a dataset.
For durable prepared artifacts, :func:`openghg_inversions.rhime.run_rhime_co2`
is the public replay seam. It accepts only a
:class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact, validates
and materializes its selected arrays, resolves its declared aggregation-error
representation, calls the explicit builder, samples, and stores a JSON
variable-role and model-provenance manifest on the returned xarray
``DataTree``.
The artifact, rather than a runner argument, owns whether aggregation error is
stored as an exact dense covariance or as a low-rank-plus-diagonal (LRPD)
approximation. A prepared ``fixed_model_mismatch`` is preserved when the
runner argument is ``None``; an explicit scalar or labelled vector overrides
it. Persist gathered-state traces with
:func:`openghg_inversions.serialization.save_trace`, which uses the
same MultiIndex-safe boundary as standard and multisector RHIME outputs.
The current prepared-input runner does not accept or construct an
outer-specific object. It constructs the complete retained prior from the
prepared arithmetic mean and covariance, then forwards the prepared activity
policy to the builder.

Set ``use_bc=True`` to select ``H_bc`` from the prepared artifact and add its
boundary contribution. ``bc_prior`` overrides the default labelled boundary
scaling prior, while ``bc_state_activity`` can mark boundary states as active
or fixed. Supplying ``offset_prior`` adds an offset. By default the offset has
one coefficient per site; ``offset_args`` can instead select a global offset
or site-by-period terms using ``per_site``, ``offset_freq``, and ``drop_first``.
The model derives period indicators from the observation time coordinate.
Boundary and offset contributions remain separate from
``co2_flux_contribution`` and are included in ``modelled_concentration``, the
likelihood, and sampled outputs.
The location and scale parameters in ``offset_prior`` use the observations'
concentration units.

Configure prepared-input replay from TOML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CO2 family provides three installed TOML templates:
``co2.toml``, ``co2_cached_sigma.toml``, and ``co2_o2.toml``. Use
:func:`openghg_inversions.rhime.co2.co2_config_templates` to discover their
installed paths. Copy the closest template for a run; do not edit the installed
resource. The templates configure the existing prepared-input Python seams.
They do not make CO2 available through the staged CLI.

:func:`openghg_inversions.rhime.co2.load_co2_family_config` reads TOML, while
:func:`openghg_inversions.rhime.co2.resolve_co2_family_config` accepts an
ordinary mapping and returns a frozen
:class:`~openghg_inversions.rhime.co2.Co2RunSetup` or
:class:`~openghg_inversions.rhime.co2.Co2O2RunSetup`. Keeping parsing separate
from resolution makes the scientific choices independent of the file format.
The setup identifies the runner and contains explicit ``preparation_kwargs``,
``runner_kwargs``, and :class:`~openghg_inversions.rhime.RhimeSampler` values;
it does not retain an ambient configuration mapping. After preparing or
loading the appropriate artifact, ``setup.runner_arguments(prepared)`` binds
it to the exact arguments accepted by ``setup.runner``. For the linked recipe,
this step expands the two configured error scalars over the labelled joint
observation axis and verifies its species and unit labels.

For example, the ordinary TOML settings::

   format_version = 1
   recipe = "co2"
   variant = "ordinary"

   [prepared_inputs]
   path = "co2-coherent-dense.zarr"

   [likelihood]
   kind = "additive_sigma"
   sigma_prior = { pdf = "halfnormal", sigma = 0.75 }

   [sampling]
   draws = 1000
   tune = 1000
   chains = 4
   nuts_sampler = "numpyro"

resolve to the same scientific runner choices as this direct Python call::

   from openghg_inversions.rhime import RhimeSampler, run_rhime_co2
   from openghg_inversions.rhime.co2 import Co2PreparedInputs

   prepared = Co2PreparedInputs.load("co2-coherent-dense.zarr")
   idata = run_rhime_co2(
       prepared_inputs=prepared,
       sigma_prior={"pdf": "halfnormal", "sigma": 0.75},
       sampler=RhimeSampler(
           draws=1000,
           tune=1000,
           chains=4,
           nuts_sampler="numpyro",
       ),
   )

The resolved ordinary setup can execute that same call through its explicit
binding method::

   from openghg_inversions.rhime.co2 import (
       Co2PreparedInputs,
       load_co2_family_config,
       resolve_co2_family_config,
   )

   config = load_co2_family_config("my-co2.toml")
   setup = resolve_co2_family_config(config)
   prepared = Co2PreparedInputs.load(setup.preparation_kwargs["path"])
   idata = setup.runner(**setup.runner_arguments(prepared))

``prepared_inputs.path`` and ``likelihood.eigenbasis_path`` become ordinary
``pathlib.Path`` values. Relative values are interpreted from the process
working directory, not from the directory containing the TOML file. Use
absolute paths when the run may start from another directory.

The resolver deliberately supports a closed matrix rather than arbitrary
callable imports:

.. list-table::
   :header-rows: 1
   :widths: 18 22 22 38

   * - Recipe
     - Variant
     - Runner
     - Configuration boundary
   * - ``co2``
     - ``ordinary``
     - ``run_rhime_co2``
     - ``additive_sigma``, ``site_sigma``, ``fixed_ou``, or ``scalar_sigma``
       likelihood; boundary conditions and offsets are available.
   * - ``co2``
     - ``cached_fixed_ou``
     - ``run_rhime_co2_cached_sigma``
     - Fixed positive OU timescales and HalfNormal site amplitudes, using the
       runner-owned PyMC sampler.
   * - ``co2_o2``
     - ``linked``
     - ``run_rhime_co2_o2_from_prepared_inputs``
     - Fixed independent error, or ``fixed_ou`` with fixed or inferred
       species/site amplitudes; one shared channel-unit label.
   * - ``co2_o2``
     - ``cached_fixed_ou``
     - ``run_rhime_co2_o2_cached_sigma_from_prepared_inputs``
     - Fixed OU timescales and HalfNormal species/site amplitudes, using the
       runner-owned PyMC sampler; one shared channel-unit label.

For ``variant = "ordinary"``, the ``[likelihood]`` table accepts these closed
forms. Scalar-or-site-map values use either one positive number or an inline
table such as ``{ MHD = 24.0, TAC = 12.0 }``.

.. list-table:: Ordinary CO2 likelihood configuration
   :header-rows: 1
   :widths: 18 34 48

   * - ``kind``
     - Required keys
     - Optional keys and constraints
   * - ``additive_sigma``
     - None
     - ``sigma_prior``; non-negative ``fixed_model_mismatch``;
       ``no_model_error`` (default false). ``sigma_prior`` and
       ``no_model_error = true`` are mutually exclusive.
   * - ``site_sigma``
     - Exactly one of ``fixed_site_amplitudes`` or ``site_amplitude_prior``
     - Fixed amplitudes are a non-negative site map; the inferred-amplitude
       prior must have positive support.
   * - ``fixed_ou``
     - Positive scalar-or-site-map ``tau_hours`` and exactly one of
       ``fixed_site_amplitudes`` or ``site_amplitude_prior``
     - Fixed amplitudes are non-negative; the inferred-amplitude prior must
       have positive support.
   * - ``scalar_sigma``
     - ``eigenbasis_path`` and positive-support ``sigma_prior``
     - The eigenbasis must match the prepared artifact as described in
       :ref:`the scalar-sigma recipe <co2-scalar-sigma-recipe>`.

Prior tables use ``pdf`` plus the parameters for that family: ``normal`` uses
``mu`` and positive ``sigma``; ``truncatednormal`` also requires ``lower`` and
optionally accepts ``upper``; ``halfnormal`` uses positive ``sigma``;
``halfstudentt`` uses positive ``nu`` and ``sigma``; ``gamma`` uses positive
``alpha`` and ``beta``; ``exponential`` uses positive ``lam``; and ``uniform``
uses ordered ``lower`` and ``upper``. ``lognormal`` accepts exactly one of
``mu``/``sigma`` or positive ``mean``/``stdev``, plus optional boolean
``reparameterise``. Positive-support prior positions accept only HalfNormal,
HalfStudentT, Gamma, Exponential, Uniform with a non-negative lower bound, or
LogNormal.

Optional model components belong in their own tables. ``[model.boundary]``
accepts boolean ``enabled`` (default true) and ``prior``; a disabled boundary
cannot specify a prior. ``[model.offset]`` requires ``prior`` and optionally
accepts ``frequency``, ``per_site`` (default true), and ``drop_first`` (default
false). A global offset (``per_site = false``) cannot set a frequency or use
``drop_first = true``.

For ``variant = "cached_fixed_ou"``, ``[likelihood]`` requires
``kind = "fixed_ou"``, positive scalar-or-site-map ``tau_hours`` (species/site
keys such as ``"co2:MHD"`` for the linked recipe), and positive
``site_amplitude_prior_scale``. It optionally accepts positive
``initial_site_amplitudes`` and the zero-to-one controls
``sigma_target_accept`` and ``state_target_accept``. Its ``[sampling]`` table
must use ``nuts_sampler = "pymc"``; posterior-predictive name lists may contain
only ``y`` or ``concentration``.

The optional ``[sampling]`` table is shared by the supported setups.
Defaults below apply when an option is omitted; linked fixed-error sampling
uses NumPyro, while both linked OU routes default to and require PyMC.

.. list-table:: Sampling configuration
   :header-rows: 1
   :widths: 23 18 19 40

   * - Option
     - Ordinary/cached default
     - Linked default
     - Accepted value and constraint
   * - ``draws``
     - ``1000``
     - ``1000``
     - Positive integer.
   * - ``burn``
     - ``0``
     - ``0``
     - Non-negative integer strictly less than ``draws``.
   * - ``tune``
     - ``1000``
     - ``1000``
     - Non-negative integer.
   * - ``chains``
     - ``4``
     - ``4``
     - Positive integer.
   * - ``nuts_sampler``
     - ``"pymc"``
     - ``"numpyro"`` for fixed error; ``"pymc"`` for OU
     - One of ``"pymc"``, ``"nutpie"``, ``"numpyro"``, or ``"blackjax"``;
       cached variants and linked ``fixed_ou`` require ``"pymc"``.
   * - ``progressbar``
     - ``false``
     - ``false``
     - Boolean.
   * - ``sample_prior_predictive``
     - ``true``
     - ``true``
     - Boolean or non-negative integer number of prior-predictive samples.
   * - ``sample_posterior_predictive``
     - ``["y"]``
     - ``["y"]``
     - Boolean or list of non-empty variable names. For the cached variant,
       listed names are restricted to ``"y"`` and ``"concentration"``.
   * - ``target_accept``
     - Not set
     - ``0.95``
     - Number strictly between zero and one. It is not accepted for the cached
       variant; use the two likelihood target-accept controls described above.
   * - ``random_seed``
     - Not set
     - Not set
     - Non-negative integer, applied to both posterior and
       posterior-predictive sampling.

For ``recipe = "co2_o2"``, ``[channels]`` must contain exactly the two tables
``[channels.co2]`` and ``[channels.o2]``. Each table requires a non-empty
``units`` string convertible to ``mol/mol`` and a finite, positive
``independent_error_sd`` number::

   [channels.co2]
   units = "ppm"
   independent_error_sd = 1.0

   [channels.o2]
   units = "ppm"
   independent_error_sd = 2.0

Each ``independent_error_sd`` is numerically expressed in its sibling
``units`` scale: in this example the CO2 and O2 standard deviations are 1 ppm
and 2 ppm, respectively. The resolver does not convert these values. During
binding it expands each scalar over that channel's observation rows. The two
``units`` strings must currently be identical, although the two error values
may differ.

Boundary and offset options use the same equations and option names as the
CO2 recipe, nested beneath each channel. For example::

   [channels.co2.boundary]
   enabled = true
   prior = {pdf = "normal", mu = 1.0, sigma = 0.1}

   [channels.co2.boundary.activity]
   active = false
   fixed_value = 1.0

   [channels.o2.offset]
   per_site = false
   prior = {pdf = "normal", mu = 0.0, sigma = 1.0}

Supply boundary sensitivities to ``prepare_co2_o2_inputs`` as
``boundary_sensitivity={"co2": H_bc_co2, "o2": H_bc_o2}``. Each array uses its
channel's native observation axis followed by one labelled boundary-state
axis. The runner includes supplied boundaries by default; ``use_bc`` can
select channels explicitly. Direct calls pass ``bc_prior``,
``bc_state_activity``, ``offset_prior`` and ``offset_args`` as mappings keyed
by ``"co2"`` and ``"o2"``. For example,
``offset_prior={"o2": {"pdf": "normal", "mu": 0.0, "sigma": 1.0}}`` and
``offset_args={"o2": {"per_site": False}}`` reproduce the offset above.
Missing sensitivities, unconsumed options, unknown channels, and mixed units
with baseline terms fail before sampling.

Posterior ``co2_mu_bc`` and ``o2_mu_bc`` are boundary concentrations;
``co2_offset`` and ``o2_offset`` are offset concentrations. Each is recorded on
the joint observation axis with zero contribution to the opposite channel.
``co2_bc`` and ``o2_bc`` retain independent labelled scaling states and activity.
The optional ``baseline_concentration`` reporting sum contains only boundary
and offset terms. The coherent affine intercept ``fixed_prior_contribution``
and all flux contributions remain distinct. Thus ``modelled_concentration``
equals ``co2_o2_flux_contribution + fixed_prior_contribution`` plus the
baseline sum when present. Aggregation covariance and independent observation
error still enter the single joint likelihood once.

Standalone O2, arbitrary Python callables, and additional recipe or variant
names are rejected. Linked likelihood selection supports the fixed-error
default and ``fixed_ou``; it rejects ``additive_sigma``, ``site_sigma``,
``scalar_sigma``, and unequal CO2 and O2 unit labels. The linked
``cached_fixed_ou`` variant uses the matched cached sampler. The lower-level
linked fixed-error API continues to represent row-specific mixed units;
configuring a heterogeneous ppm/per-meg
run is deferred until the scaling contract tracked in `OPE-86
<https://linear.app/openghg-inversions/issue/OPE-86>`_ is available. Use the
direct Python interfaces for experimental combinations outside this matrix.
The linked :class:`~openghg_inversions.rhime.co2.Co2O2PreparedInputs` artifact
does not yet have a durable ``load`` method. Construct it through the documented
preparation boundary and bind the resulting in-memory artifact; linked staged
artifact loading remains follow-up work in `OPE-165
<https://linear.app/openghg-inversions/issue/OPE-165>`_.

The linked template therefore follows a prepare, bind, and run sequence. The
scientific array names below are the labelled inputs documented by
:func:`~openghg_inversions.rhime.co2.prepare_co2_o2_inputs`; replace them with
the products from one coherent reduction. Exactly one of
``o2_co2_flux_ratio`` and ``o2_co2_flux_ratio_unavailable_reason`` must be
non-null::

   from openghg_inversions.rhime.co2 import (
       load_co2_family_config,
       prepare_co2_o2_inputs,
       resolve_co2_family_config,
   )

   config = load_co2_family_config("my-co2-o2.toml")
   setup = resolve_co2_family_config(config)

   prepared = prepare_co2_o2_inputs(
       co2_observations=co2_observations,
       o2_observations=o2_observations,
       co2_prior_forward_mean=co2_prior_forward_mean,
       o2_prior_forward_mean=o2_prior_forward_mean,
       co2_sensitivity=co2_sensitivity,
       o2_sensitivity=o2_sensitivity,
       o2_co2_flux_ratio=o2_co2_flux_ratio,
       o2_co2_flux_ratio_unavailable_reason=None,
       co2_aggregation_covariance=co2_aggregation_covariance,
       co2_o2_aggregation_covariance=co2_o2_aggregation_covariance,
       o2_aggregation_covariance=o2_aggregation_covariance,
       retained_prior=retained_prior,
       boundary_sensitivity={"co2": H_bc_co2},
       co2_units=setup.preparation_kwargs["co2_units"],
       o2_units=setup.preparation_kwargs["o2_units"],
   )
   idata = setup.runner(**setup.runner_arguments(prepared))

Unknown or unused keys are errors, reported by their dotted path. Resolution
also rejects incompatible component choices before an artifact is loaded or a
model is built. This section defines the local CO2-family configuration
surface, not the cross-family configuration and CLI catalogue. That reference
work is tracked in `OPE-159
<https://linear.app/openghg-inversions/issue/OPE-159>`_.

Construct the CO2-specific artifact by pairing canonical RHIME inputs with all
linked products from one
:class:`~openghg_inversions.coherent_reduction.CoherentGaussianReduction`.
The :doc:`canonical RHIME workflow <rhime>` produces the durable base inputs,
and :doc:`coherent_reduction` shows how to construct ``reduction`` from the
native covariance products.
Pass ``aggregation_error_rank=None`` to keep the reduction's exact dense
unresolved covariance::

   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.rhime.co2 import prepare_co2_inputs
   from openghg_inversions.rhime import run_rhime_co2

   canonical_inputs = RhimePreparedInputs.load("base-prepared-inputs.zarr")
   prepared = prepare_co2_inputs(
       canonical_inputs,
       reduction,
       aggregation_error_rank=None,
   )
   prepared.save("co2-coherent-dense.zarr")

   idata = run_rhime_co2(
       prepared_inputs=prepared,
       use_bc=True,
       bc_prior={
           "pdf": "truncatednormal",
           "mu": 1.0,
           "sigma": 0.05,
           "lower": 0.0,
       },
       offset_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
       offset_args={"per_site": False},
   )

Reload a durable artifact with
:meth:`openghg_inversions.rhime.co2.Co2PreparedInputs.load`. The loaded
artifact retains the selected representation, so replay does not require or
accept ``aggregation_error_mode``.

The CO2 builder and prepared-input runner also accept one ordinary
``likelihood_builder`` with explicit ``likelihood_kwargs``. This selects
package components such as
:func:`openghg_inversions.models.fixed_ou.add_fixed_ou_gaussian_likelihood` and
:func:`openghg_inversions.models.add_site_sigma_gaussian_likelihood` after
``modelled_concentration`` has been completed. Selecting one replaces the
default additive-sigma likelihood; its scientific options belong in
``likelihood_kwargs`` rather than the default ``sigma_*`` or
``fixed_model_mismatch`` arguments. The returned trace records the selected
callable identity and its explicit options using the ordinary likelihood
provenance attributes.

.. _co2-scalar-sigma-recipe:

Run the global scalar-sigma CO2 likelihood
------------------------------------------

Use the scalar-sigma likelihood when one global positive mismatch amplitude
must augment a fixed, possibly non-diagonal aggregation covariance. It
evaluates the exact Gaussian covariance

.. math::

   R(\sigma_{global}) = A + D_{obs} + \sigma_{global}^2 I

from one reusable eigendecomposition of ``A + D_obs``. This differs from the
default CO2 likelihood, which infers site-aligned additive amplitudes in a
diagonal covariance. It also differs from the
:ref:`fixed-OU cached-sigma recipe <co2-cached-sigma-recipe>`, which infers
site amplitudes for a time-correlated OU covariance and owns a matched sampler
with an accepted-state runtime cache. The scalar-sigma cache is instead an
external numerical preparation artifact used through the ordinary
``run_rhime_co2`` likelihood seam.

Prepare and save that artifact once from the same prepared CO2 inputs that
sampling will use::

   from openghg_inversions.models import save_scalar_sigma_eigenbasis
   from openghg_inversions.rhime.co2 import (
       Co2PreparedInputs,
       prepare_co2_scalar_sigma_eigenbasis,
   )

   prepared = Co2PreparedInputs.load("co2-coherent-dense.zarr")
   eigenbasis = prepare_co2_scalar_sigma_eigenbasis(prepared)
   save_scalar_sigma_eigenbasis("scalar-sigma-eigenbasis.nc", eigenbasis)

Select the package likelihood through the CO2 runner::

   from openghg_inversions.models import add_scalar_sigma_eigen_likelihood
   from openghg_inversions.rhime.co2 import run_rhime_co2

   trace = run_rhime_co2(
       prepared_inputs=prepared,
       likelihood_builder=add_scalar_sigma_eigen_likelihood,
       likelihood_kwargs={
           "eigenbasis_path": "scalar-sigma-eigenbasis.nc",
           "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75},
       },
   )

The cache stores labelled eigenvectors and eigenvalues, the aggregation-error
mode, and a fingerprint of the resolved ``A + D_obs``. Loading checks its
schema, dimensions, labels, cache/observation/reported-error unit labels, mode,
and current covariance identity before model construction. Regenerate the
cache after changing ``mf_error`` values, aggregation-error values, or the
artifact's aggregation-error representation. Changing the observation order
or the unit label also makes the cache incompatible. Loading reconstructs the
current base covariance once, but does not repeat the eigendecomposition or
add work to likelihood evaluations.

Unit conversion is caller-owned. CO2 preparation validates compatible units
at the same numeric scale but does not convert values. After numerical
conversion, ``mf``, ``mf_error``, ``sigma_global``
and, when present, ``low_rank_factor`` use one concentration unit. When
present, ``aggregation_error_covariance`` and
``diagonal_residual_variance`` use that unit squared. Configure
``sigma_prior`` for the concentration unit; in the HalfNormal example,
``sigma=0.75`` is in that unit. The cache, ``mf.units``, and
``mf_error.units`` must carry the same non-empty unit label. A consistent but
incorrectly scaled aggregation covariance will otherwise be accepted and
fingerprinted.

This is a same-unit CO2-only likelihood; it does not support the linked
mixed-unit CO2/O2 vector. A direct
:func:`openghg_inversions.rhime.co2.build_co2_model` caller may load the cache
with :func:`openghg_inversions.models.load_scalar_sigma_eigenbasis` and pass
the resulting ``eigenbasis`` in ``likelihood_kwargs``. See the
:doc:`scalar-sigma API reference
<../reference/openghg_inversions.models.scalar_sigma>` for signatures and
object contracts.

.. _linked-co2-o2-model:

CO2/O2 shared-state model
-------------------------

The CO2/O2 recipe applies one retained state to both observation channels.
Its public boundaries are
:func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` for labelled
preparation, :func:`openghg_inversions.rhime.co2.build_co2_o2_model` for graph
construction, and
:func:`openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs` for
materialization, sampling, and trace metadata.

The default joint recipe uses fixed, row-labelled independent error. The
same-unit fixed-OU option below adds independent species/site mismatch blocks
while preserving the prepared cross-channel aggregation covariance.

Partition that state as

.. math::

   \alpha =
   \begin{bmatrix}
      \alpha_{shared} \\
      \alpha_{CO_2,ocean} \\
      \alpha_{O_2,ocean}
   \end{bmatrix},

where :math:`\alpha_{shared}` contains the gross primary production (GPP),
terrestrial ecosystem respiration (TER), and fossil-fuel states.
The joint coherent flux contribution is

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
   \mu_{flux} = b_{joint} + H_{joint}\alpha.

Equivalently, coherent reduction may be written in centred or affine form,

.. math::

   \mu_{flux}
   = \mu_{prior} + H_{joint}(\alpha - m_\alpha)
   = (\mu_{prior} - H_{joint}m_\alpha) + H_{joint}\alpha.

The prepared ``fixed_prior_contribution`` is the parenthesized affine
intercept, not the complete prior-forward concentration. Each channel may
add an independent boundary scaling state :math:`\beta_c` through its native
boundary sensitivity :math:`H_{bc,c}`, and an observation-aligned offset
:math:`o_c`. The complete likelihood mean is

.. math::

   \mu_{joint} = \mu_{flux}
   + \begin{bmatrix} H_{bc,CO_2}\beta_{CO_2} \\ H_{bc,O_2}\beta_{O_2} \end{bmatrix}
   + \begin{bmatrix} o_{CO_2} \\ o_{O_2} \end{bmatrix}.

An omitted channel term is zero. These boundary and offset contributions form
the optional reporting baseline; they remain distinct from the coherent affine
intercept and shared or tracer-specific flux states. Baseline terms currently
require identical channel units.

Thus this is a row-stacked, block-sparse sensitivity acting on one state vector,
not two independent block-diagonal models. Its fixed-error likelihood is

.. math::

   \begin{bmatrix} y_{CO_2} \\ y_{O_2} \end{bmatrix}
   \mid \alpha, \beta_{CO_2}, \beta_{O_2}, o_{CO_2}, o_{O_2}
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

These reduction identities are exact for a jointly Gaussian native state. As
in the CO₂-only recipe, using the projected arithmetic moments for a correlated
LogNormal positive retained state is a moment-matched approximation, not exact
LogNormal marginalization. Positivity and the scientific suitability of that
approximation must be justified independently.

In particular, the off-diagonal :math:`A_{CO_2,O_2}` block is part of the
coherent-reduction contract. See the :doc:`full derivation
<coherent_reduction>` for its assumptions and limitations.

Preparation accepts separate native channel arrays, then gathers their rows on
one ``(species, channel_observation)`` observation index before the model
applies the joint sensitivity once. Before calling
:func:`~openghg_inversions.rhime.co2.prepare_co2_o2_inputs`, callers must
numerically convert observations, prior-forward means, sensitivities, and every
covariance block into mutually consistent channel units. The ``co2_units`` and
``o2_units`` arguments only attach labels; they do not convert or validate
numerical scales, so incorrectly scaled values can pass preparation. Before
calling
:func:`~openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs`,
callers must separately convert ``independent_error_sd`` into the corresponding
observation-row units and attach matching ``observation_units`` labels. Each
row then retains its declared native units and numerical scale.
Verification-game inputs may use ppm for both channels, while real atmospheric
O2 observations may use per-meg delta(O2/N2).
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
:func:`openghg_inversions.serialization.save_trace` and restore them
with :func:`openghg_inversions.serialization.load_trace`; this is the
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
ratio. That future CO2/O2 coupling work is tracked in `OPE-118
<https://linear.app/openghg-inversions/issue/OPE-118>`_.

.. _co2-cached-sigma-recipe:

Run the package-supported cached-sigma CO2 runner
-------------------------------------------------

``run_rhime_co2_cached_sigma`` is the package-supported cached-sigma runner at
the prepared-input boundary for the fixed-tau OU likelihood with independently
inferred site amplitudes. It owns the matched PyMC graph and sampler: the
sigma-only No-U-Turn Sampler (NUTS) step runs first against
the exact conditional likelihood. The sampler then rebuilds the
state-likelihood quadratic for the site amplitudes returned by that transition,
and stock state NUTS reads that cache without refactorizing the observation
covariance during its trajectory.

The runner begins from an already assembled coherent-reduction
:class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact; it does not
perform coherent reduction. Its ``inv_inputs`` must contain ``H``,
``alpha_prior_mean``,
``alpha_prior_covariance``, ``fixed_prior_contribution``, ``mf``, and
``mf_error``. The observation arrays share one ``nmeasure`` row order, while
the prior arrays share ``H``'s state labels. A low-rank artifact contains
``low_rank_factor`` and ``diagonal_residual_variance``; a dense artifact
instead contains ``aggregation_error_covariance``. The artifact records which
one representation its runner must use. Optional ``state_is_active`` and
``state_fixed_value`` variables carry the prepared state-activity policy.
The observation array must also have observation-aligned ``site`` and ``time``
coordinates. Times must be finite and unique within each site, although rows
may be interleaved or unsorted. Datetime coordinates are converted to elapsed
hours; numeric time coordinates are interpreted directly as hours. If
``tau_hours`` is a mapping, its keys must exactly match every observed site
label.

Boundary and offset selection uses the same prepared-input contract as the
ordinary runner. Set ``use_bc=True`` to select a prepared ``H_bc``;
``bc_prior`` and ``bc_state_activity`` configure its labelled scaling state.
Supplying ``offset_prior`` adds the offset described by ``offset_args``.
Its location and scale parameters use the observations' concentration units.
These coefficients join the flux coefficients only inside the private cached
likelihood calculation; their public scientific variables and contributions
remain separate.

:doc:`coherent_reduction` describes the linked retained prior, effective
operator, affine contribution, and unresolved covariance.
:func:`openghg_inversions.rhime.co2.prepare_co2_inputs` assembles those
products with canonical observations and metadata at the durable CO2 boundary.

The reduction itself is exact under its stated Gaussian assumptions. An LRPD
artifact is a separate downstream numerical approximation of its unresolved
covariance. The cached runner can retain every mode above the scale-relative
numerical tolerance and carry the remaining marginal variance in the diagonal
tail. This is exact when every positive mode exceeds that tolerance, but it
generally produces a full numerical-rank factor and is not the intended
scaling path. The default retains at most 512 modes; override it when a
different retained rank is scientifically justified::

   from openghg_inversions.rhime.co2 import prepare_co2_inputs

   prepared = prepare_co2_inputs(
       canonical_inputs,
       reduction,
       aggregation_error_rank=40,
   )
   prepared.save("co2-coherent-low-rank.zarr")

The approximation preserves the dense covariance diagonal up to accepted
roundoff and diagonal-tail clipping, recorded by
``diagonal_preservation_error``. A chosen rank is not evidence that the
approximation is adequate for an inversion. Assess the resulting total
likelihood covariance and log density for representative observation-error,
site-amplitude, and OU profiles, especially when model-mismatch error is
small. See :doc:`coherent_reduction` for the dense preparation cost and the
other retained-spectrum and reconstruction diagnostics.

For example::

   from openghg_inversions.rhime import RhimeSampler
   from openghg_inversions.rhime.co2 import (
       Co2PreparedInputs,
       run_rhime_co2_cached_sigma,
   )

   prepared = Co2PreparedInputs.load("co2-coherent-low-rank.zarr")
   idata = run_rhime_co2_cached_sigma(
       prepared_inputs=prepared,
       tau_hours={"BSD": 24.0, "TAC": 18.0},
       site_amplitude_prior_scale=0.75,  # concentration units
       use_bc=True,
       bc_prior={
           "pdf": "truncatednormal",
           "mu": 1.0,
           "sigma": 0.05,
           "lower": 0.0,
       },
       offset_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
       offset_args={"per_site": False},
       sigma_target_accept=0.9,
       state_target_accept=0.9,
       sampler=RhimeSampler(
           draws=1000,
           tune=1000,
           chains=4,
           nuts_sampler="pymc",
           sample_kwargs={"random_seed": 20260913},
           sample_prior_predictive=False,
       ),
   )

The runner constructs the required ``site amplitude -> state`` ``CompoundStep`` and
uses process spawning for multiple chains. Do not pass another step method in
``sample_kwargs``. Set ``sigma_target_accept`` and ``state_target_accept`` on
the runner rather than putting a generic ``target_accept`` in
``RhimeSampler.sample_kwargs``. The state-likelihood quadratic cache and
fixed-OU generalized eigenbasis are runtime numerical state derived from the
materialized prepared inputs; they are not external cache artifacts.

Because the cached graph uses a normalized joint ``Potential``, it does not
invent an independent observed distribution. After sampling, the same exact
fixed-OU target adds ``log_likelihood.y`` as one scalar per complete
observation vector and, when requested, draws complete correlated vectors in
``posterior_predictive.y``. The variables carry explicit joint-scope metadata.

This route deliberately supports only independent HalfNormal site-amplitude
priors and fixed positive OU timescales. Sampled tau and alternative amplitude
priors are separate extensions. The route is PyMC-only because it constructs
and owns a PyMC ``CompoundStep``. The ordinary stock-PyMC fixed-OU likelihood
above remains the log-density/gradient oracle and fallback.

The exponential within-site covariance follows the stationary process of
`Uhlenbeck and Ornstein (1930)
<https://doi.org/10.1103/PhysRev.36.823>`_. The named recipe and its matched
runtime state-likelihood cache are implemented and versioned by OpenGHG
Inversions.

Built-in aggregation covariance relies on the guarantees of its construction
pipeline. The runner selects and validates prepared aggregation-error arrays
through :func:`openghg_inversions.observation_error.resolve_aggregation_error`.
A custom pipeline that constructs an ``AggregationError`` directly owns the
completeness, coherence, and numerical covariance guarantees of that object;
there is no separate public complete-covariance validator.


Linked fixed-OU mismatch and cached sampling
-------------------------------------------

The linked prepared-input runner accepts ``tau_hours`` together with either
``fixed_site_amplitudes`` or ``site_amplitude_prior``. It uses the same fixed-OU
numerical component as the CO2 recipe, with groups defined by both species and
site. The complete joint covariance is

.. math::

   R = A_{joint} + D_{obs}
       + \operatorname{blockdiag}_{(species, site)}
         \left(\sigma_{species,site}^{2} T_{species,site}\right),
   \qquad T_{ij}=\exp(-|t_i-t_j|/\tau_{species,site}).

``A_joint`` is the prepared aggregation covariance, including its nonzero
cross-channel blocks. ``D_obs`` is the squared reported independent error.
Each term enters exactly once. Rows may be irregular and interleaved; the
OU groups preserve their original row positions. Different channels at the
same site have separate amplitudes and no cross-channel OU contribution.
Both channels must already use the same concentration units. Heterogeneous
covariance units are not supported by this option.

Tau is fixed in hours. A scalar tau or fixed amplitude applies independently
to every observed group. Mappings must cover labels such as ``co2:MHD`` and
``o2:MHD``. Amplitudes and their prior scales use the common concentration unit.
For example, the ordinary linked configuration can include:

.. code-block:: toml

   recipe = "co2_o2"
   variant = "linked"

   [likelihood]
   kind = "fixed_ou"
   tau_hours = { "co2:MHD" = 6.0, "o2:MHD" = 8.0 }
   fixed_site_amplitudes = { "co2:MHD" = 0.4, "o2:MHD" = 0.7 }

Replace ``fixed_site_amplitudes`` with a ``site_amplitude_prior`` table to infer
amplitudes with stock PyMC. The optimized runner
:func:`openghg_inversions.rhime.co2.run_rhime_co2_o2_cached_sigma_from_prepared_inputs`
uses independent HalfNormal amplitude priors and the existing CO2
sigma-then-state ``CompoundStep``. It refreshes the exact state quadratic only
when returned amplitude values change, then updates all active flux, boundary,
and offset coefficients against that cache. Its builder is
:func:`openghg_inversions.rhime.co2.build_co2_o2_cached_sigma_model`.

Select ``variant = "cached_fixed_ou"`` and supply ``site_amplitude_prior_scale``
in ``[likelihood]`` in place of fixed amplitudes. Optional
``initial_site_amplitudes`` accepts the same scalar or species/site mapping.
Sampler options match the CO2 cached recipe, including separate
``sigma_target_accept`` and ``state_target_accept`` controls. Both linked OU
routes require ``nuts_sampler = "pymc"`` and select it by default; the ordinary
fixed-error linked default remains NumPyro.

Both routes retain species and native channel identity in the trace. The
``ou_site`` coordinate uses the species/site labels, with ``ou_species`` and
``ou_station`` coordinates describing each group. Amplitudes retain
concentration units and ``ou_tau_hours`` retains hours through serialization.
The cached runner supplies one normalized joint log likelihood per posterior
draw and optional correlated joint predictive vectors; these are not
independent per-observation likelihoods.
