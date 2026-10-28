Prototype: affine concentration components
==========================================

This design experiment shows a structure shared by the centered boundary and
the coherent flux reduction. Each concentration contribution can be described
as its value at a reference state plus a linear response to a deviation from
that state. Boundary centering additionally translates the observations. Each
component also has a physical output in its own space. The components share
this structure; they do not share an interface, protocol or base class.
Recipes still compose flux, boundary and offset contributions
explicitly. See :doc:`rhime_model_development` for the component and ownership
principles, and :doc:`/usage/affine_flux_map` for the retained-state flux
reconstruction contract.

For the boundary comparison, let :math:`K=A+cM` be the complete design, where
:math:`A` is the anomaly design, :math:`M` the dimensionless mass design and
:math:`c` the scalar reference concentration. Unit boundary scaling
:math:`\mathbf{1}` provides an algebraic state reference. The observation shift
is :math:`q=cM\mathbf{1}`.

.. list-table::
   :header-rows: 1

   * -
     - Centered boundary
     - Coherent flux
   * - State
     - Boundary scaling :math:`b`
     - Retained flux state :math:`\alpha`
   * - Absolute concentration
     - :math:`K b`
     - :math:`H m + H_\alpha(\alpha - \alpha_{\mathrm{ref}}) = d + H_\alpha \alpha`
   * - State reference
     - :math:`b=\mathbf{1}`
     - :math:`\alpha=\alpha_{\mathrm{ref}}`
   * - Absolute concentration at the state reference
     - :math:`K\mathbf{1}=A\mathbf{1}+q`
     - :math:`H m`
   * - Response to a state deviation
     - :math:`K(b-\mathbf{1})`
     - :math:`H_\alpha(\alpha-\alpha_{\mathrm{ref}})`
   * - Observation translation
     - Subtract :math:`q` from observations and the likelihood mean
     - None; the mean retains its intercept :math:`d`
   * - Physical output
     - Scaled boundary curtains (not reconstructed)
     - Signed native flux :math:`F[m + U_*(\alpha - \alpha_{\mathrm{ref}})]`
   * - Code
     - ``CenteredBoundaryMap``
     - ``add_coherent_affine_component`` and ``BoundCo2AffineFluxMap``

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
remains the direct :math:`H_{bc}b`. It acquires no additional shift or state
parameterization. In centered mode, the likelihood uses the factored expression
:math:`Ab+cM(b-\mathbf{1})` against observations :math:`y-q`, and absolute
outputs restore :math:`q` to that centered contribution.

Subtracting the same deterministic reference from observations and the mean
changes neither the measurement-error nor model-error covariance. Centering is
therefore a representation of the same boundary model, chosen because a large
reference such as O2 (about 209,300 ppm) would otherwise swamp the small
anomaly signal.

Run the :download:`offline example <../../examples/affine_components.py>`
from the repository to evaluate both components on small synthetic arrays::

   python -m examples.affine_components

It needs no store, network or posterior sampling.

Centered boundary component
---------------------------

``CenteredBoundaryMap`` in ``rhime.co2.centered_boundary`` owns the factored
boundary equations for registered PyTensor operands and dense NumPy operands.
Its physical boundary coefficients :math:`b` give

.. math::

   \operatorname{centered\_concentration}(b) &= A b + c M (b - \mathbf{1}), \\
   \operatorname{concentration}(b) &= \operatorname{centered\_concentration}(b) + q.

``add_centered_boundary`` builds both from registered PyTensor operands. The
cached likelihood projection evaluates the same ``centered_concentration`` on
dense NumPy operands for its fixed boundary contribution, so one class owns the
centered equation on both routes. Its active design :math:`A + cM` and the
numerical projection are otherwise unchanged.

Keep :math:`A`, :math:`M` and :math:`c` separate. Forming the absolute
concentration first and subtracting :math:`q` afterwards would reintroduce the
cancellation this representation avoids. The prior and fixed/active policy
still describe the physical coefficients :math:`b`; activity is resolved from
the full absolute design :math:`A+cM`. Trace annotation restores observed and
predictive concentrations using the retained reference shift.

Coherent flux component
-----------------------

The coherent component already has both pieces, and the prototype adds no new
object for it. ``build_co2_model`` composes the concentration contribution
:math:`d + H_\alpha\alpha` with ``apply_linear_sensitivity`` and
``add_coherent_affine_component``. ``BoundCo2AffineFluxMap.state_to_flux``
evaluates the retained-state-conditional signed native flux mean

.. math::

   F\left[m + U_* (\alpha - \alpha_{\mathrm{ref}})\right],

where :math:`m` is the native scaling mean, :math:`F` the signed reference flux,
and :math:`U_*` the conditional prolongation. The bound reconstruction owns the
authoritative retained reference :math:`\alpha_{\mathrm{ref}}`. This state
reference and the boundary concentration reference :math:`c` have different
roles and units.

The flux reconstruction supplies a conditional mean for each retained-state
draw. Unresolved native covariance is not sampled or added to these draws. The
coherent likelihood retains its prepared unresolved observation covariance.

Scope of the experiment
-----------------------

A wrapper giving the coherent component matching method names was tried and
dropped: it had no production caller and only restated an existing graph
expression in xarray. Add one only when a consumer needs concentrations
recomputed from saved retained-state draws.

The centered ``floatX=float64`` guard remains in place. Prepared and sampled
artifact formats retain their existing owners and contents.
