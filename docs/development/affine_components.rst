Prototype: affine concentration components
==========================================

This design experiment makes explicit what the centered boundary and the
coherent flux reduction have in common. Each is the affine generalization of
an older linear option, and each needs handling that the linear options do
not. Recipes still compose flux, boundary and offset contributions explicitly.
See :doc:`rhime_model_development` for the component and ownership principles,
and :doc:`/usage/affine_flux_map` for the retained-state flux reconstruction
contract.

Linear and affine options
-------------------------

.. list-table::
   :header-rows: 1

   * -
     - Linear (existing)
     - Affine (new)
   * - Flux
     - ``BasisOperator``: sensitivity :math:`H` from native footprints;
       native field :math:`U x`
     - ``CoherentGaussianReduction``: :math:`H_\alpha`, intercept :math:`d`
       and unresolved covariance; ``AffineFluxMap``: native field
       :math:`m + U_*(\alpha - \alpha_{\mathrm{ref}})`
   * - Boundary
     - Sensitivity :math:`H_{bc}`, ``bind_boundary_sensitivity``
     - Anomaly :math:`A`, mass :math:`M` and reference :math:`c`,
       ``bind_boundary_operands``, ``CenteredBoundaryMap``

``AffineFluxMap`` is the affine form of a basis operator's native-field
reconstruction, and the coherent reduction is the affine form of its
sensitivity. The centered boundary has no corresponding object for its
operands; they travel as a validated ``xr.Dataset``.

Affine terms in a recipe
------------------------

Every concentration contribution is affine in its state: its value at a
reference state plus a linear response to a deviation from that state. Recipes
represent component contributions as an ``AffineTerm`` with three parts:

``output``
   The absolute concentration contribution.
``likelihood_contribution``
   What the term adds to the likelihood mean.
``observation_shift``
   A deterministic concentration subtracted from the observations, or ``None``.

Ordinary linear, offset and coherent flux components retain their tensor or
component-result return types. Recipes wrap their outputs with
``AffineTerm.absolute``: their likelihood contribution is their output and they
have no shift. ``add_centered_boundary`` instead supplies a term through
``CenteredBoundaryResult.term``, contributing :math:`Ab + cM(b-\mathbf{1})`
against observations :math:`y-q`, where :math:`q=cM\mathbf{1}`. Every recipe
then composes its likelihood the same way:

.. code-block:: python

   terms = [AffineTerm.absolute(co2_flux_contribution)]
   if boundary_operands is not None:
       terms.append(add_centered_boundary(boundary_operands, prior=bc_prior).term)
   ...
   modelled_mean = pm.Deterministic("modelled_concentration", sum_of_outputs)
   likelihood_mean, likelihood_observations, shift = likelihood_representation(
       terms, observations, modelled_mean,
   )

``likelihood_representation`` returns ``modelled_mean`` and the observations
unchanged when no term is shifted. Otherwise it sums the likelihood
contributions directly, because subtracting a large shift from the absolute
mean would lose small centered anomalies. Subtracting the same deterministic
shift from observations and mean changes neither the measurement-error nor
model-error covariance, so both forms describe the same probability model.

The term list must describe every contribution to ``modelled_mean`` in the
same component order, using common concentration units and observation row
order. The helper checks that observation shifts have exactly matching indexed
rows before combining them; a mismatch raises ``ValueError`` instead of
discarding observations through xarray's default arithmetic join. Recipes use
``term.output`` for total absolute concentration composition and grouped
boundary/baseline reports. Components retain their registered graph variables,
state, activity and design information for lowering.

Cached likelihoods lower each ``AffineTerm`` once more, to a dense fixed
contribution plus an active design times coefficients. For a centered boundary
the coefficients are :math:`b-\mathbf{1}`; for other terms they are the active
states.

Handling beyond the linear options
----------------------------------

.. list-table::
   :header-rows: 1

   * - Stage
     - Coherent flux
     - Centered boundary
   * - Preparation
     - ``reduce_native_gaussian`` then ``prepare_co2_inputs``
     - Anomaly, mass and reference acquisition; ``align_boundary_operands``
   * - State activity
     - Retained state with a correlated prior
     - Resolved on the complete design :math:`A + cM`
   * - Likelihood
     - Unresolved covariance added as aggregation error
     - Observation shift :math:`q`; ``floatX=float64`` required
   * - Cached lowering
     - Intercept plus fixed states; active :math:`H_\alpha`
     - ``centered_concentration`` at the fixed baseline; active :math:`A + cM`
   * - Trace
     - None
     - ``restore_absolute_predictives`` adds :math:`q` back to ``y``
   * - Physical output
     - ``BoundCo2AffineFluxMap.state_to_flux``
     - None; scaled curtains are not reconstructed
   * - Recipe support
     - CO2 recipe family
     - CO2 recipe family; ``standard`` and ``multisector`` reject the operands

Reference states
----------------

Let :math:`K=A+cM` be the complete boundary design. Unit boundary scaling
:math:`\mathbf{1}` provides an algebraic state reference.

.. list-table::
   :header-rows: 1

   * -
     - Centered boundary
     - Coherent flux
   * - Absolute concentration
     - :math:`K b`
     - :math:`H m + H_\alpha(\alpha - \alpha_{\mathrm{ref}}) = d + H_\alpha \alpha`
   * - State reference
     - :math:`b=\mathbf{1}`
     - :math:`\alpha=\alpha_{\mathrm{ref}}`
   * - Concentration at the state reference
     - :math:`K\mathbf{1}=A\mathbf{1}+q`
     - :math:`H m`
   * - Response to a state deviation
     - :math:`K(b-\mathbf{1})`
     - :math:`H_\alpha(\alpha-\alpha_{\mathrm{ref}})`
   * - Observation shift
     - :math:`q`
     - None; the mean retains its intercept :math:`d`

The observation shift :math:`q` is different from the boundary concentration
at unit scaling :math:`K\mathbf{1}`. Likewise,
:math:`d=Hm-H_\alpha\alpha_{\mathrm{ref}}` is the coherent intercept at
:math:`\alpha=0`, while :math:`Hm` is the concentration at its reference state.
For the boundary, the reference-state identities are

.. math::

   Kb &= K\mathbf{1} + K(b-\mathbf{1}), \\
   Kb-q &= A\mathbf{1} + K(b-\mathbf{1}).

These identities explain the shared structure; they are not an evaluation
strategy for ordinary boundaries. The ordinary absolute boundary calculation
remains the direct :math:`H_{bc}b`.

Run the :download:`offline example <../../examples/affine_components.py>`
from the repository to evaluate both components on small synthetic arrays::

   python -m examples.affine_components

It needs no store, network or posterior sampling.

Centered boundary component
---------------------------

``CenteredBoundaryMap`` in ``rhime.co2.centered_boundary`` owns the factored
boundary equations for registered PyTensor operands and dense NumPy operands:

.. math::

   \operatorname{centered\_concentration}(b) &= A b + c M (b - \mathbf{1}), \\
   \operatorname{concentration}(b) &= \operatorname{centered\_concentration}(b) + q.

``add_centered_boundary`` returns the boundary state, its ``AffineTerm`` and
the operands it was built from. The cached lowering evaluates the same
``centered_concentration`` on those operands, so one class owns the centered
equation on both routes.

Keep :math:`A`, :math:`M` and :math:`c` separate. Forming the absolute
concentration first and subtracting :math:`q` afterwards would reintroduce the
cancellation this representation avoids, for example against an O2 reference
of about 209,300 ppm. The prior and fixed/active policy still describe the
physical coefficients :math:`b`.

Coherent flux component
-----------------------

``build_co2_model`` composes the concentration contribution
:math:`d + H_\alpha\alpha` with ``apply_linear_sensitivity`` and
``add_coherent_affine_component``, then wraps it with ``AffineTerm.absolute``.
``BoundCo2AffineFluxMap.state_to_flux`` evaluates the
retained-state-conditional signed native flux mean

.. math::

   F\left[m + U_* (\alpha - \alpha_{\mathrm{ref}})\right],

where :math:`m` is the native scaling mean, :math:`F` the signed reference flux,
and :math:`U_*` the conditional prolongation. The bound reconstruction owns the
authoritative retained reference :math:`\alpha_{\mathrm{ref}}`. This state
reference and the boundary concentration reference :math:`c` have different
roles and units.

The flux reconstruction supplies a conditional mean for each retained-state
draw. Unresolved native covariance is not sampled or added to these draws.

The CO2 recipe family
---------------------

The centered boundary equations, coherent reduction and covariance products
are species-neutral. Their current module locations do not establish support
for other species in the delivered recipes and output adapters. Single-channel
CO2 outputs retain CO2 species metadata and PARIS templates; the linked recipes
additionally encode CO2/O2 relationships. Reuse for another gas needs suitable
preparation and priors, together with corresponding recipe and output support.
``standard`` and ``multisector`` currently reject centered operands because
that support has not been delivered there.

Scope of the experiment
-----------------------

``AffineTerm`` is a small result record, not a base class or protocol;
components still construct their own graphs. The centered ``floatX=float64``
guard remains in place. Prepared and sampled artifact formats retain their
existing owners and contents.
