Fixed-OU likelihood and cached quadratics
=========================================

This page explains the numerical target and the two conditional updates used
by the :ref:`cached-sigma CO2 runner <co2-cached-sigma-recipe>`. It is intended
for maintainers changing the likelihood, its gradients or cache lifecycle.
The equations describe the represented low-rank covariance, including its
diagonal remainder; choosing a different rank changes that target.

Implementation scope
--------------------

The implementation described here is in ``models/fixed_ou.py``,
``models/cached_sigma.py`` and ``inference/cached_sigma.py``. The cache routes
were introduced in PRs #922 and #926, and amplitude residual reuse in #928.
The additional evaluator screens and joint QR/SVD fallback described below
are implemented in this branch by
`PR #942 <https://github.com/openghg/openghg_inversions/pull/942>`_.
They must not be assumed present in releases or branches that omit that fix.

Earlier evaluators used an unguarded Cholesky factorization of the small
Woodbury core. Dominant duplicate factor columns could erase its identity
term, producing a finite but wrong likelihood. Even a well-conditioned
scalar core could lose the inverse-weighted residual through subtraction,
and hence corrupt gradients. A sum-of-squares likelihood alone did not fix
those gradient failures. The cache's guarded solve was a separate path and
did not establish evaluator accuracy.

Covariance and prepared coordinates
-----------------------------------

Let :math:`r = y - \mu` be the observation residual, with :math:`n`
observations and :math:`r_f` retained aggregation-factor columns. Write

.. math::

   C(a) = \operatorname{diag}(d) + F F^T
          + \operatorname{blockdiag}_s(a_s^2 R_s),
   \qquad (R_s)_{ij} = \exp(-|t_i-t_j|/\tau_s).

Here :math:`F` has shape :math:`n\times r_f`, :math:`d` contains the fixed
measurement variance and any diagonal aggregation tail, and :math:`a_s` is
the mismatch amplitude for site :math:`s`. Residuals, factor entries and
amplitudes have concentration units; :math:`d` has squared concentration
units. Times and fixed positive :math:`\tau_s` use hours internally.
Site rows may be interleaved. Preparation requires unique times within a
site and a positive-definite OU correlation template.

For each site, the generalized eigensystem gives :math:`V_s` satisfying

.. math::

   V_s^T R_s V_s = I, \qquad
   V_s^T \operatorname{diag}(d_s) V_s = \operatorname{diag}(\lambda_s).

The assembled transform :math:`T` applies :math:`V_s^T` to the appropriate
observation rows. It is generally not Euclidean-orthogonal. Preparation
stores :math:`TF`, the mode eigenvalues, mode-to-site indices and
:math:`b=\sum_s\log|R_s|`. The transform is stored sparsely across sites,
but its within-site blocks are dense.

At supplied amplitudes define

.. math::

   D_{ii}=\lambda_i+a_{s(i)}^2, \quad w_i=D_{ii}^{-1},\quad
   W=D^{-1/2}TF,\quad u=D^{-1/2}Tr,\quad A=I+WW^T.

Then

.. math::

   \log|C| = b+\sum_i\log D_{ii}+\log|A|,\qquad
   r^TC^{-1}r=u^TA^{-1}u.

The implemented route requires every :math:`D_{ii}` to be positive and
finite. A zero amplitude is valid only when the corresponding mode
variances remain positive. The log density includes the normalization
:math:`-\tfrac12(n\log(2\pi)+\log|C|+u^TA^{-1}u)`.
Preparation and numerical evaluation are explicit eager boundaries; see
:doc:`numerical_array_ownership` for input ownership rules.

Cached state quadratic
----------------------

For fixed amplitudes, the model mean is affine in the sampled flux/boundary
state :math:`x`: :math:`\mu=f+Hx`. Let :math:`r_0=y-f` and form
:math:`B=D^{-1/2}T[r_0,H]`. The joint Gram matrix
:math:`G=B^TA^{-1}B` supplies the normalized state likelihood

.. math::

   \ell(x\mid a)=c+h^Tx-\tfrac12x^TPx,\quad
   c=-\tfrac12(n\log(2\pi)+\log|C|+G_{00}),\quad
   h=G_{1:,0},\quad P=G_{1:,1:}.

This is a likelihood cache: state priors and parameter-transform Jacobians
remain in the scientific graph. State trajectories evaluate the quadratic
without refactorizing the observation covariance. The fixed transformed
right-hand sides :math:`T[r_0,H]` are computed when the target is constructed.

For :math:`0<r_f<n`, cache refresh first tries the small core
:math:`K=I+W^TW`. If :math:`Z=K^{-1}W^TB` and :math:`E=B-WZ`, then

.. math::

   G=E^TE+Z^TZ.

This sum of squares avoids subtracting two large Gram matrices. The identity
follows from :math:`KZ=W^TB`; it does not make an inaccurate solve for
:math:`Z` accurate. The implementation screens core conditioning and an
estimate of residual-rounding error relative to column energies. Rejected
cases use the general covariance solve and then form the Gram matrix.

Cache and evaluator routing
---------------------------

The cache and direct evaluator need different outputs and have distinct
fallbacks. They share the prepared covariance and transform, not one universal
factorization policy.

* **Cache/general solve:** at full factor width :math:`r_f\ge n`, try
  observation-space Cholesky on :math:`I+WW^T`. Reject a failed or poorly
  conditioned factorization. If that attempt is rejected, or for a lower-width
  general solve (including a rejected low-rank cache screen), use a reduced
  joint QR basis for the factor and right-hand sides,
  followed by an SVD in those coordinates. Solve there and include the
  factor's determinant contribution. Rank zero needs only diagonal weighting.
* **Direct/amplitude evaluator:** for nonzero rank, try Cholesky of
  :math:`K`. Check finiteness, successful factorization and a reciprocal
  condition estimate greater than :math:`\sqrt{\epsilon}`. With
  :math:`z=K^{-1}W^Tu`, form :math:`v=u-Wz`. A separate normwise roundoff
  screen protects this subtraction. Another screen protects the precision
  diagonal :math:`p_i=1-\|L^{-1}W^T_{:,i}\|^2`, where :math:`K=LL^T`.
  Accepted evaluations use :math:`\|v\|^2+\|z\|^2` for the quadratic.

For the evaluator, rejecting any screen invokes a complete QR basis of
:math:`[W,u]`. An SVD of the factor in the retained joint coordinates gives
singular values :math:`s_j`. The corresponding inverse square-root scales
are :math:`1/\sqrt{1+s_j^2}`; orthogonal directions have scale one.
The same coordinates provide the quadratic, determinant update
:math:`\sum_j\log(1+s_j^2)`, solved residual :math:`v=A^{-1}u` and
precision diagonal :math:`p=\operatorname{diag}(A^{-1})`.

The precision diagonal includes the complete orthogonal complement. It is
computed as sums of squared scaled-basis entries, including complement blocks,
so it does not subtract leverage values from one. Merely stabilizing the
quadratic or dropping that complement would leave the gradients wrong.

Amplitude gradients and trajectory-local reuse
----------------------------------------------

In either evaluator route, the physical amplitude gradient is

.. math::

   \frac{\partial\ell}{\partial a_s}
     = a_s\sum_{i:s(i)=s}w_i(v_i^2-p_i),\qquad
   \nabla_r\ell=-T^TD^{-1/2}v.

The general evaluator returns both gradients. The amplitude-only Op calls
the same numerical evaluator with a transformed residual and omits the
unused residual-gradient back-transform. The amplitude prior and its
coordinate-transform Jacobian are supplied by the PyMC graph, not these
physical-gradient equations.

The matched step compiles its private amplitude graph with PyTensor's CVM
linker. This choice applies to that graph; it does not change global compiler
settings or the separate state graph.

Each amplitude trajectory conditions on a fixed flux/boundary state. Before
calling stock amplitude NUTS, the matched step recomputes the modelled mean
from the incoming point and installs :math:`Tr` once. Every density call
within that trajectory reuses it. Changing amplitudes still changes
:math:`D`, :math:`W` and :math:`K`; those cannot be reused unchanged across
amplitude proposals.

The next trajectory installs a new residual even when the preceding amplitude
draw did not change. The state quadratic is refreshed for changed accepted
amplitudes; an incoming amplitude point that differs from the stored cache
also forces refresh. The refreshed quadratic is installed before the
subsequent state update. These are two different invalidation rules:
state changes invalidate the trajectory residual, while amplitude changes
invalidate the state quadratic.

The prepared covariance, design and observations must remain fixed after
target construction. Replacing those inputs requires rebuilding the target;
the amplitude/state invalidation rules do not track arbitrary array changes.

Residual storage and mutable sampler caches belong to the step/chain.
Borrowed input arrays are not mutated, installed residuals are copied into
shared storage, and independent steps do not share that mutable state.
Nonfinite residuals retain rejection behavior rather than being transformed
into misleading finite inputs. Invalid amplitude and variance handling is
shared with the general evaluator.

Costs, limits and regression coverage
-------------------------------------

Generalized eigendecomposition is a preparation cost of roughly
:math:`\sum_s O(n_s^3)`; storing/applying its blocks needs
:math:`\sum_s O(n_s^2)` entries/work. At low rank the amplitude core and
broad triangular solve cost :math:`O(nr_f^2+r_f^3)` per evaluation.
For :math:`q` cached right-hand sides, refresh additionally includes
:math:`O(nr_fq+r_f^2q+nq^2+r_fq^2)` matrix products and small solves.
State likelihood evaluation then costs :math:`O(p^2)` for :math:`p` state
coefficients. Full-width observation-space Cholesky costs :math:`O(n^3)`
and requires observation-sized square storage.

The exceptional evaluator fallback also constructs an :math:`n\times n`
basis. Its complement reductions are blocked, but that does not remove the
square allocation. Its QR/SVD work can be much more expensive than the fast
route. A benchmark of ordinary inputs that never trigger fallback does not
measure this exceptional cost.

The screens protect known cancellation failures, not arbitrary componentwise
relative accuracy. Cache Gram-error screening is relative to column energies,
not each small cross coefficient, and does not certify latent-solve error.
The evaluator residual screen is normwise: a large accurate component can
mask a small inaccurate one. Orthogonal projection still has absolute
roundoff at the input scale, so tiny solved components can have poor relative
accuracy even in the fallback. These limits should remain visible when
adding new extreme-scale inputs.

Regression coverage is retained independently of historical benchmarks:

For exactly aligned residuals with dominant duplicate factors beyond
float64's condition range, residual-gradient tests use a small-case absolute
rounding budget proportional to
:math:`\epsilon\|r\|_2\|C^{-1}\|_2`. This does not promise relative accuracy
for an arbitrarily tiny gradient or certify a general forward-error bound.
Nearby residual tests also require resolvable transverse components to survive;
clipping small components is not an acceptable way to satisfy the budget.
Likelihood and amplitude-gradient checks retain their separate tolerances.

* ``tests/models/test_fixed_ou.py`` checks dense and analytic likelihood/gradient
  oracles, dominant and duplicate factors, rank-zero/full-width cases,
  near-dependent mixed-scale factors using an 80-digit Decimal oracle,
  precision cancellation at zero residual, routing and input ownership.
* ``tests/models/test_cached_sigma.py`` checks cached values/gradients against
  a dense oracle, factorization counts, screen rejection, zero right-hand
  sides and borrowed-input copying.
* ``tests/test_cached_sigma_sampling.py`` checks trajectory integration,
  residual reuse, cache refresh, chain isolation, priors and posterior parity.
  CO2 runner and shape-hook tests protect the surrounding integration.

Historical evidence and further experiments
-------------------------------------------

The :doc:`fixed_ou_evidence` index links immutable reports, scripts and timing
CSVs. Those are dated experimental evidence, not maintained benchmark drivers.
The native Numba prototype remains a closed reference PR and is not the
production backend.

Unimplemented ideas include replacing a broad triangular solve with a small
triangular inverse plus multiplication, or computing only the required
site-wise gradient traces. Both require accuracy and whole-density timing
checks; neither is an established improvement. A separate proposed sampler
would draw an exact aggregation-factor latent variable for the amplitude
update alone, discard it, and retain the marginalized flux update. Earlier
experiments conditioned the flux update on that latent variable and mixed
poorly. The narrower proposal has not been tested. Its chronology and target
preservation caveats remain in the archived follow-up note linked by the
index; none of these ideas defines supported behavior.
