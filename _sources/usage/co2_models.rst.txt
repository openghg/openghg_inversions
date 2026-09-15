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
tracked in `OPE-79 <https://linear.app/openghg-inversions/issue/OPE-79>`_.
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

For example::

   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.rhime import run_rhime_co2

   prepared = RhimePreparedInputs.load("co2-coherent-dense.zarr")
   idata = run_rhime_co2(
       prepared_inputs=prepared,
       aggregation_error_mode="dense",
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

Prepare and save that artifact once from the same prepared CO2 inputs and
aggregation-error mode that sampling will use::

   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.models import save_scalar_sigma_eigenbasis
   from openghg_inversions.rhime.co2 import prepare_co2_scalar_sigma_eigenbasis

   prepared = RhimePreparedInputs.load("co2-coherent-dense.zarr")
   eigenbasis = prepare_co2_scalar_sigma_eigenbasis(
       prepared,
       aggregation_error_mode="dense",
   )
   save_scalar_sigma_eigenbasis("scalar-sigma-eigenbasis.nc", eigenbasis)

Select the package likelihood through the CO2 runner::

   from openghg_inversions.models import add_scalar_sigma_eigen_likelihood
   from openghg_inversions.rhime.co2 import run_rhime_co2

   trace = run_rhime_co2(
       prepared_inputs=prepared,
       aggregation_error_mode="dense",
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
cache after changing ``mf_error`` values, aggregation-error values, or
``aggregation_error_mode``. Changing the observation order or the unit
label also makes the cache incompatible. Loading reconstructs the current base
covariance once, but does not repeat the eigendecomposition or add work to
likelihood evaluations.

Unit conversion is caller-owned. Neither
:func:`openghg_inversions.observation_error.resolve_aggregation_error` nor
scalar-sigma preparation converts or validates units on aggregation-error
arrays. After numerical conversion, ``mf``, ``mf_error``, ``sigma_global``
and, when present, ``aggregation_error_sd`` and ``low_rank_factor`` use one
concentration unit. When present, ``aggregation_error_covariance`` and
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

This joint recipe currently keeps its fixed, row-labelled independent error
and does not expose the CO2 ``likelihood_builder`` seam. A cross-channel
mismatch model must first define its CO2/O2 covariance, parameter sharing, and
mixed-unit behavior explicitly.

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
``RhimePreparedInputs`` artifact; it does not perform coherent reduction. Its
``inv_inputs`` must contain ``H``, ``alpha_prior_mean``,
``alpha_prior_covariance``, ``fixed_prior_contribution``, ``mf``, and
``mf_error``. The observation arrays share one ``nmeasure`` row order, while
the prior arrays share ``H``'s state labels. With the ``"low_rank"``
aggregation-error mode used below, the artifact must also contain
``low_rank_factor`` and ``diagonal_residual_variance``. A dense artifact
instead contains ``aggregation_error_covariance`` and must be selected with
``aggregation_error_mode="dense"``. Optional ``state_is_active`` and
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
operator, affine contribution, and unresolved covariance, but no public
function currently assembles those products into this durable artifact. That
handoff is tracked in `OPE-153
<https://linear.app/openghg-inversions/issue/OPE-153/add-a-public-coherent-reduction-handoff-for-co-prepared-inputs>`_.
Until it lands, callers must supply already-prepared coherent-reduction inputs.

For example::

   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.rhime import RhimeSampler
   from openghg_inversions.rhime.co2 import run_rhime_co2_cached_sigma

   prepared = RhimePreparedInputs.load("co2-coherent-low-rank.zarr")
   idata = run_rhime_co2_cached_sigma(
       prepared_inputs=prepared,
       tau_hours={"BSD": 24.0, "TAC": 18.0},
       site_amplitude_prior_scale=0.75,  # concentration units
       aggregation_error_mode="low_rank",
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
