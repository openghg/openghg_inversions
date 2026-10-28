Prototype: affine concentration components
==========================================

This design experiment shows a structure shared by the centered boundary and
the coherent flux reduction. Each component's absolute concentration
contribution is a fixed reference contribution plus a linear response to a
physical state. Each also has a physical output in its own space. The
components share this structure; they do not share an interface, protocol or
base class. Recipes still compose flux, boundary and offset contributions
explicitly. See :doc:`rhime_model_development` for the component and ownership
principles, and :doc:`/usage/affine_flux_map` for the retained-state flux
reconstruction contract.

.. list-table::
   :header-rows: 1

   * -
     - Centered boundary
     - Coherent flux
   * - State
     - Boundary scaling :math:`b`
     - Retained flux state :math:`\alpha`
   * - Absolute concentration
     - :math:`(A + cM)\,b = q + A b + c M (b - \mathbf{1})`
     - :math:`H m + H_\alpha(\alpha - \alpha_{\mathrm{ref}}) = d + H_\alpha \alpha`
   * - Fixed contribution
     - :math:`q = c M \mathbf{1}`, moved onto the observations
     - :math:`d = H m - H_\alpha \alpha_{\mathrm{ref}}`, kept in the mean
   * - Physical output
     - Scaled boundary curtains (not reconstructed)
     - Signed native flux :math:`F[m + U_*(\alpha - \alpha_{\mathrm{ref}})]`
   * - Code
     - ``CenteredBoundaryMap``
     - ``add_coherent_affine_component`` and ``BoundCo2AffineFluxMap``

The two components differ in one choice: where the fixed contribution goes in
the likelihood. The coherent flux likelihood compares :math:`y` with a mean
containing :math:`d`. The centered boundary likelihood compares :math:`y - q`
with a mean that excludes :math:`q`. Subtracting a deterministic reference
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
boundary equations after model-data registration. Let :math:`A` denote the
anomaly sensitivity, :math:`M` the dimensionless mass sensitivity, :math:`c`
the concentration reference, and :math:`b` the physical boundary coefficients:

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
