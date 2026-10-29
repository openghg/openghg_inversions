Boundary-condition operators
============================

Boundary conditions (BCs) contribute a transported concentration to each
observation. First, we define the transport and basis operators for standard
multiplicative BCs. Then we describe numerical centering and additive BC
corrections in separate sections, before showing the current preparation
calculations in pseudocode.

These operators apply to any concentration field. Numerical accuracy needs
particular care when the required absolute accuracy is small relative to the
background concentration. For example, resolving ppm or smaller changes in
O2 near 200,000 ppm requires retaining many significant digits. We explain
both the effect of quantized footprint weights and the order of arithmetic
operations below. See :doc:`surface_preparation` for the preparation workflow
and :doc:`co2_models` for supported runners and prior configuration.

Transport, concentrations and the boundary basis
------------------------------------------------

We distinguish boundary grid cells and native times from the states inferred
by the model. A state may cover many cells and times, such as one curtain
region over one month. Its coefficient is shared across those cells even
when their supplied concentrations differ.

We use :math:`i=1,\ldots,N` for observations, :math:`\ell=1,\ldots,L` for
boundary cells at native times, and :math:`j=1,\ldots,K` for model states.
The native field contains concentrations on the boundary grid before
observation averaging. Transport maps those concentrations to observations.
Time selection, unit conversion and observation averaging must be consistent
between the operators below.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Symbol
     - Shape and units
     - Meaning
   * - :math:`P`
     - :math:`N\times L`, dimensionless
     - Transport weights, including time selection and observation averaging.
   * - :math:`C^0`
     - :math:`L`, concentration
     - Supplied concentration at each boundary cell and native time.
   * - :math:`B`
     - :math:`L\times K`, dimensionless
     - Basis mapping model-state coefficients to the native boundary grid.
   * - :math:`H_{bc}`
     - :math:`N\times K`, concentration
     - Transported supplied concentrations weighted by the basis.
   * - :math:`M`
     - :math:`N\times K`, dimensionless
     - Transported basis functions without concentration factors.

The basis may contain binary or fractional values. A **partition basis** is
the binary special case: :math:`B_{\ell j}=1` when cell and time
:math:`\ell` belongs to state :math:`j`, and zero otherwise. Each row has
exactly one nonzero entry. For example, ``(north, January)`` and
``(north, February)`` are distinct states. A fractional basis can instead
assign weights 0.25 and 0.75 from two neighbouring state coefficients to a
cell. We call any basis satisfying :math:`B\mathbf1=\mathbf1` a
**partition of unity**; this includes both examples and preserves constant
fields. Defining the field from coefficients and averaging concentrations
into observation bins are separate operations.

We aim to choose independent native basis columns, so that
:math:`\operatorname{rank}(B)=K`. Reducing redundant columns requires
re-expressing the intended prior consistently, preserving its distribution
over native fields. Even with independent columns, transport can give
:math:`\operatorname{rank}(PB)<K`: distinct native fields may be
indistinguishable to the observing network. Native basis independence and
observation-space identifiability are separate scientific checks; the current
projection code does not impose a rank check.

Standard multiplicative boundary conditions
-------------------------------------------

For dimensionless state coefficients :math:`b`, the native boundary field is
:math:`\operatorname{diag}(C^0)Bb`. Its contribution to observations is

.. math::

   H_{bc} &= P\operatorname{diag}(C^0)B, \\
   \mu_{bc} &= H_{bc}b, \\
   H_{bc,ij} &= \sum_\ell P_{i\ell}C^0_\ell B_{\ell j}.

For a basis that partitions unity, :math:`b=\mathbf1` gives the supplied
native field and :math:`H_{bc}\mathbf1=PC^0`. Otherwise its baseline is
:math:`P\operatorname{diag}(C^0)B\mathbf1`, so the choice of basis also
changes the baseline field.

We also define the transported basis, or mass operator,

.. math::

   M=PB,\qquad M_{ij}=\sum_\ell P_{i\ell}B_{\ell j}.

The code name ``H_bc_mass`` refers to transport weight. For example,

.. math::

   M=\begin{pmatrix}0.7&0.2&0.1\\0.1&0.3&0.6\end{pmatrix}

means that a unit concentration assigned through the first basis state gives
responses 0.7 and 0.1 at the two observations. For normalized, complete
transport and a basis that partitions unity, :math:`M\mathbf1=\mathbf1`.
Incomplete transport coverage or a basis that does not partition unity can
give other row sums. We retain the actual vector :math:`M\mathbf1` in the
centering equations.

.. _boundary-numerical-centering:

Numerical centering
-------------------

We choose a scalar reference concentration :math:`c` in the same units as
:math:`C^0` and form the anomaly operator

.. math::

   A &= P\operatorname{diag}(C^0-c\mathbf1)B, \\
   A_{ij} &= \sum_\ell P_{i\ell}(C^0_\ell-c)B_{\ell j}, \\
   H_{bc} &= A+cM.

``H_bc_anomaly`` stores :math:`A`, ``H_bc_mass`` stores :math:`M`, and
``bc_reference`` stores the provided :math:`c`. The caller supplies the
reference; preparation does not estimate it from the boundary field.

Quantized transport weights and operation order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization of ``particle_locations`` when footprints are saved can change
the sum of the stored weights. For a constant 200,000 ppm field and a basis
that partitions unity, a total-weight error of :math:`10^{-4}` gives a
20 ppm error in the transported baseline. This error comes from the stored
weights' quantization, rather than ordinary float64 rounding during the
subsequent calculation.

The native producer checks that the total over all four curtains is finite,
positive and within an explicitly justified tolerance of one, then normalizes
them together. This corrects the total-weight defect under the assumption of
complete boundary coverage. It does not reconstruct individual weights before
quantization or correct other transport errors. Totals outside the supplied
tolerance are rejected.

After this normalization, we calculate each native anomaly as
``mass * (concentration - reference)`` before summation and observation
averaging. Choosing :math:`c` near the background keeps those native
departures small. This ordering evaluates differences before the transport
sums and avoids constructing :math:`A` by subtracting two accumulated
background terms. We use float64 to limit further rounding; it cannot recover
precision lost from input concentrations before preparation.

Centered observations and model means
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We choose the fixed observation-space shift

.. math::

   r=cM\mathbf1.

For the same multiplicative coefficients :math:`b`,

.. math::

   \mu_{bc}-r=Ab+cM(b-\mathbf1).

We subtract :math:`r` from observations and the total model mean. The
likelihood residual is unchanged algebraically. This calculation retains the
original BC coefficients and their prior, while evaluating departures around
the chosen reference. We restore :math:`r` once for absolute predictions.
The identity holds whether or not :math:`M\mathbf1=\mathbf1`.

Normalization addresses the total defect in stored transport weights;
forming native anomalies and centering the likelihood address the subsequent
arithmetic. The required absolute accuracy relative to the background
motivates these steps. Changing the concentration unit changes both numeric
values and the required absolute tolerance by the same factor.

.. _boundary-additive-corrections:

Additive boundary corrections
-----------------------------

An additive formulation keeps the projected baseline
:math:`H_{bc}\mathbf1` fixed and adds state corrections :math:`d` in
concentration units. We denote their dimensionless observation response by
:math:`G_{bc}`. Its entry :math:`G_{bc,ij}` is the change at observation
:math:`i` per unit concentration correction in state :math:`j`.
For corrections using the same transport and basis,

.. math::

   G_{bc}=PB=M.

When the baseline basis partitions unity, the corrected native field is
:math:`C=C^0+Bd`. With the example :math:`M` above,
a 1 ppm correction to the first state changes the first prediction by
0.7 ppm. A separately supplied :math:`G_{bc}` must describe the intended
correction field with matching observation and state labels. Its units are
``"1"`` when corrections use the observations' concentration units. The
supported additive interfaces require finite, nonnegative responses that
are not all zero.

The CO2 preparation/model paths use provided ``H_bc_mass`` as this response
when no separate ``boundary_correction_sensitivity`` is supplied. Numerical
centering and the choice of statistical prior remain independent: the
absolute and numerically centered expressions are respectively

.. math::

   \mu_{bc} &= H_{bc}\mathbf1+G_{bc}d, \\
   \mu_{bc}-r &= A\mathbf1+G_{bc}d.

The absolute expression is valid without numerical centering. Applications
requiring many significant digits still need the accuracy checks and
operation ordering in :ref:`boundary-numerical-centering`.

One whole-window correction and state anomalies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current additive prior writes

.. math::

   d_j &= \delta+u_j, \\
   w^\mathsf{T}u &= 0,\qquad w_j>0,\qquad \sum_jw_j=1.

The scalar :math:`\delta` is a constant correction to the boundary product
over the entire inversion window. States :math:`j=(\text{region},\text{period})`
have anomalies that may vary between regions and periods. There is one
weighted constraint over the whole window. Individual periods can have
nonzero mean anomalies; we do not impose a zero mean within each period.
For example, :math:`u_{rt}=\gamma_t` gives the same anomaly to every region in
period :math:`t`. It is allowed when
:math:`\sum_t(\sum_rw_{rt})\gamma_t=0`, with individual :math:`\gamma_t`
potentially nonzero. For a basis that partitions unity, :math:`\delta` gives
a constant native concentration correction. Otherwise its native pattern is
:math:`\delta B\mathbf1`.

Independent priors on the :math:`K` corrections :math:`d_j` are a valid
alternative scientific formulation. We use the mean-and-anomaly formulation
to specify uncertainty in the common boundary correction separately from
uncertainty in departures from it. The current implementation gives
:math:`\delta` its scalar prior and makes it independent of the constrained
Gaussian anomalies. Decomposing independent corrections into their mean and
departures would generally give a different joint prior. See :doc:`co2_models`
for its covariance, scale interpretation and supported configuration.

A constant observation offset represents persistent bias in observations;
the boundary correction describes bias in the supplied BC product. They enter
the model through different components. If :math:`G_{bc}\mathbf1=\mathbf1`,
both constant coefficients have the same observation-space direction, so
the likelihood alone identifies their sum. Their distinct scientific roles
do not resolve this exact confounding. Separate prior information or external
constraints are needed to distinguish them in that case.

.. warning::

   With the default NESW basis, complete jointly normalized particle weights
   and :math:`G_{bc}=M`, we know that :math:`G_{bc}\mathbf1=\mathbf1`.
   A global observation offset, or unrestricted site offsets, therefore
   duplicates the boundary mean's constant direction. Model construction
   warns for this combination. Fixing one site's offset or constraining site
   offsets to contrasts removes that one redundancy; BC anomalies can still
   reproduce site contrasts. See :doc:`../development/boundary_offset_identifiability`
   for examples and the distinction between relative site effects and physical
   bias attribution.

.. _boundary-averaging-weights:

Choosing the averaging weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The positive weights :math:`w` define which mean is assigned to
:math:`\delta`. Uniform state weights define an equal-state average. A
physical average requires a declared measure on the native boundary grid.
For corrections using basis :math:`B`, let :math:`a_\ell\geq0` be normalized
native weights with :math:`\sum_\ell a_\ell=1`, such as an explicitly
specified curtain-area and duration measure. If the basis partitions unity,
choose

.. math::

   w=B^\mathsf{T}a.

Then :math:`\sum_jw_j=1` and

.. math::

   a^\mathsf{T}B(\delta\mathbf1+u)
   =\delta+w^\mathsf{T}u=\delta.

Thus the state constraint represents the chosen native-field mean. The
current interface requires every state weight to be strictly positive, so the
native measure must give each retained correction state positive weight.
For a basis that does not partition unity, :math:`a^\mathsf{T}B\mathbf1`
need not equal one and this interpretation must be reconsidered.

The caller supplies these weights; preparation does not infer a native
measure or basis geometry. Choosing weights for a mean constraint does not
derive the full state prior from a native-field prior. That would also require
a native covariance and a justified transformation to state coefficients.

What the current surface preparation computes
---------------------------------------------

We now follow the actual calculations using labelled xarray curtain arrays.
The blocks below are pseudocode. The mathematics is general; currently
supported centered-input runners are listed in :doc:`co2_model_family`.

Native curtain products
^^^^^^^^^^^^^^^^^^^^^^^^

The local ``baseline_reference_sensitivities`` kernel, called by
:func:`~openghg_inversions.inversion_data.prepare_surface_site`, matches
supplied concentrations to footprint timestamps within declared coverage
intervals and converts concentrations and the reference to common units.

.. code-block:: text

   require finite supplied concentrations and chosen reference
   for each native observation i:
       total = sum(particle_locations[i, direction, cell]
                   over every cell on all four curtains)
       require finite, nonnegative particle weights
       require total > 0 and abs(total - 1) <= mass_tolerance

       for each direction and cell:
           mass = particle_locations[i, direction, cell] / total
           anomaly = mass * (supplied_concentration[i, direction, cell] - c)
           retain mass and anomaly on their labelled curtain grids

The resulting arrays are ``bc_mass_n/e/s/w`` and ``bc_anomaly_n/e/s/w``,
with scalar ``bc_reference``. The products use float64 arithmetic and preserve
lazy payloads.

Observation averaging and spatial basis projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``prepare_surface_site`` averages the native mass and anomaly products onto
the retained observation bins using the provider's alignment operations.
We average the products before spatial projection, using the same alignment
for both operands.

``bc_sensitivity`` then contracts them with the supplied spatial basis:

.. code-block:: text

   for each retained observation i and spatial basis region k:
       M_spatial[i, k] = sum(mass[i, direction, cell]
                             * spatial_basis[i, direction, cell, k])
       A_spatial[i, k] = sum(anomaly[i, direction, cell]
                             * spatial_basis[i, direction, cell, k])
       # Both sums cover every curtain and its spatial cells.

Without a supplied basis, the four curtains become four regions. Supplied
basis values can be fractional. A singleton basis time applies to every
observation; multiple basis times must match retained timestamps. The outputs
are ``H_bc_mass`` and ``H_bc_anomaly`` on spatial ``bc_region`` and observation
axes, plus the unchanged scalar ``bc_reference``.

Temporal state construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At model binding,
:func:`~openghg_inversions.boundary_sensitivity.align_boundary_operands`
assigns a fitted period from the retained observation timestamp.

.. code-block:: text

   for each observation i:
       p_i = 0 if frequency is None else period_containing(observation_time[i])
       for each spatial region k and fitted period p:
           j = state_label(k, p)
           D = 1 if p == p_i else 0
           M[i, j] = M_spatial[i, k] * D
           A[i, j] = A_spatial[i, k] * D

There is one period without a frequency; otherwise calendar-month or
fixed-duration rules determine membership. The current mask is binary.
It uses observation time rather than a particle's boundary-exit time.
The resulting ``bc_region`` MultiIndex has ``bc_curtain`` and ``bc_period``
levels, so four spatial regions and three periods give :math:`K=12` states.

For this implementation we can write the basis in observation-row form.
Let :math:`q` label a curtain cell, :math:`S_{iqk}` be the supplied spatial
basis at retained observation :math:`i`, and :math:`D_{ip}` be the binary
period mask above. Then

.. math::

   B_{iq,(k,p)}=S_{iqk}D_{ip}.

Boundary concentrations have already been selected and aligned to native
footprint timestamps before observation averaging. This row-wise expression
makes the retained observation's temporal assignment explicit. The native
rank discussion concerns independent basis functions over their full domain;
it does not require each masked observation slice to have rank :math:`K`.

A future temporal basis could supply fractional weights. The operator
identities would still hold when both operands use the same mapping; the
current period interface implements the binary assignment shown above.

The :ref:`centered preparation handoff <surface-centered-boundary-contract>`
explains saving raw spatial operands and binding them to supported periods.
The model then uses matrix--vector products of the observation/state
operators, centers observations and model means for the likelihood, and
restores absolute predictions through the supported trace annotation/replay
path.
