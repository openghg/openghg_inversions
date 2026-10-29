Centered boundary conditions
============================

Boundary conditions (BCs) contribute a transported concentration to each
observation. This page defines the absolute sensitivity ``H_bc`` and the
numerically centered operands ``H_bc_anomaly``, ``H_bc_mass`` and
``bc_reference``. It first describes their mathematical meaning, then the
specific calculations performed by the current surface preparation code.
See :doc:`surface_preparation` for the preparation workflow and
:doc:`co2_models` for configuring boundary priors and model runners.

Transport, concentrations and the boundary basis
------------------------------------------------

Distinguish the native boundary samples from the states inferred by the model.
A native sample has a location on a curtain, including height, and a time.
A model state may cover many such samples: for example, one curtain region
over one month. Its correction is shared across those samples, even when
their supplied concentrations differ.

Use :math:`i=1,\ldots,N` for observations, :math:`\ell=1,\ldots,L` for native
samples and :math:`j=1,\ldots,K` for model states. Observation-specific boundary
samples can be represented as separate native entries. All time selection,
unit conversion and observation averaging must be consistent between the
operators below.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Symbol
     - Shape and units
     - Meaning
   * - :math:`P`
     - :math:`N\times L`, dimensionless
     - Transport weights mapping native boundary samples to observations.
   * - :math:`C^0`
     - :math:`L`, concentration
     - Supplied concentration at each native sample.
   * - :math:`B`
     - :math:`L\times K`, dimensionless
     - Basis mapping model-state coefficients to native boundary samples.
   * - :math:`H_{bc}`
     - :math:`N\times K`, concentration
     - Transported supplied concentrations, grouped by model state.
   * - :math:`M`
     - :math:`N\times K`, dimensionless
     - Transported basis functions, without concentration factors.
   * - :math:`A`
     - :math:`N\times K`, concentration
     - Transported supplied concentration departures from a scalar reference.
   * - :math:`c`
     - scalar, concentration
     - Fixed numerical reference in the same units as :math:`C^0`.

For a partition basis, :math:`B_{\ell j}` is one when native sample
:math:`\ell` belongs to state :math:`j`, and zero otherwise. For example,
``(north, January)`` and ``(north, February)`` are distinct columns. This is a
dummy or indicator matrix. Height is part of the spatial membership rule;
there is no additional height argument on the state coefficient.

The mathematical definition also allows fractional basis entries. A native
sample might receive weights 0.25 and 0.75 from two neighbouring state
coefficients, instead of belonging wholly to one state. If every row sums to
one, the basis preserves constant fields. This property is called a partition
of unity and does not require binary entries. Such a basis defines how a field
is reconstructed from coefficients; averaging data into bins is a separate
operation. The current implementation's spatial projection and temporal
membership rules are described below; this general definition does not imply
a built-in spatial or temporal interpolation recipe.

The three observation-space operators are

.. math::

   H_{bc} &= P\operatorname{diag}(C^0)B, \\
   M &= PB, \\
   A &= P\operatorname{diag}(C^0-c\mathbf1)B.

Equivalently, their entries are sums over native samples:

.. math::

   H_{bc,ij} &= \sum_\ell P_{i\ell}C^0_\ell B_{\ell j}, \\
   M_{ij} &= \sum_\ell P_{i\ell}B_{\ell j}, \\
   A_{ij} &= \sum_\ell P_{i\ell}(C^0_\ell-c)B_{\ell j}.

Thus :math:`H_{bc}=A+cM`. These are operator identities; preparation does not
need to construct a large diagonal concentration matrix or a flattened
matrix :math:`B`.

When :math:`B\mathbf1=\mathbf1`, setting every multiplicative coefficient to
one gives the supplied native field and :math:`H_{bc}\mathbf1=PC^0`. For a
basis that does not partition unity, this baseline is instead
:math:`P\operatorname{diag}(C^0)B\mathbf1`. The identities still hold, but
the choice of basis then changes the baseline field as well.

What the mass operator means
----------------------------

:math:`M_{ij}` is the observation response to a unit additive concentration
correction in state :math:`j`, using the basis above. The name ``H_bc_mass``
refers to transport weight, not mass of the gas. For example,

.. math::

   M=\begin{pmatrix}0.7&0.2&0.1\\0.1&0.3&0.6\end{pmatrix}

means that adding 1 ppm to the first state changes the two predictions by
0.7 ppm and 0.1 ppm. It is not an array of ones. Adding 1 ppm to all three
states changes both predictions by 1 ppm in this example.

For normalized, complete transport and a basis that partitions unity,
:math:`M\mathbf1=\mathbf1`. No constant column is needed: multiplication by
:math:`\mathbf1` sums the columns. In general, retain the actual vector
:math:`M\mathbf1`; incomplete transport coverage or a basis that does not
partition unity need not give row sums of one. Numerical centering does not
require those row sums to equal one.

The additive-prior interface calls its unit-correction response :math:`G_{bc}`:
entry :math:`G_{bc,ij}` is the change in predicted concentration at observation
:math:`i` per unit additive correction in state :math:`j`. When it uses the
same transport and basis as the centered operands, :math:`G_{bc}=M`. The current
CO2 preparation/model paths use supplied ``H_bc_mass`` for this response when
no separate ``boundary_correction_sensitivity`` is supplied. A separately
supplied response must represent the intended correction field with matching
observation/state labels. The response itself has dimensionless units ``"1"``;
corrections use the observations' concentration units.

Neither :math:`H_{bc}` nor :math:`c` alone supplies the missing transport
information. Even together they do not uniquely determine :math:`M` and
:math:`A`. Dividing :math:`H_{bc}` by a nominal concentration is generally
incorrect because the supplied concentrations vary across native samples.

Numerical centering and statistical corrections
------------------------------------------------

For multiplicative state coefficients :math:`b`, the absolute boundary term is

.. math::

   \mu_{bc}=H_{bc}b=(A+cM)b.

Choose the fixed observation-space reference shift

.. math::

   r=cM\mathbf1.

Subtracting it gives the equivalent expression

.. math::

   \mu_{bc}-r=Ab+cM(b-\mathbf1).

Subtract :math:`r` from both observations and the total model mean. Their
residual is unchanged, so numerical centering preserves the multiplicative
prior and likelihood. Absolute predictions restore :math:`r` once. The
scalar :math:`c` is an arithmetic reference, not an inferred boundary mean.
Forming :math:`A` from native differences before summation helps preserve
small signals around a large background; subtracting :math:`cM` from an
already rounded :math:`H_{bc}` cannot recover precision lost earlier.

An additive prior instead fixes the projected baseline
:math:`H_{bc}\mathbf1` and infers state corrections
:math:`d_j=\delta+u_j`, in concentration units. That baseline equals the
transported supplied native field when the basis partitions unity. Here
:math:`\delta` is one common correction over the input window and
:math:`u_j` is the anomaly for state :math:`j`. For corrections using the same
transport and basis, :math:`G_{bc}=PB=M` and the native correction field is
:math:`Bd`. The observation contribution is

.. math::

   \mu_{bc} &= H_{bc}\mathbf1+G_{bc}(\delta\mathbf1+u), \\
   \mu_{bc}-r &= A\mathbf1+G_{bc}(\delta\mathbf1+u).

The common coefficient :math:`\delta` adds a uniform native concentration
only when :math:`B\mathbf1=\mathbf1`; otherwise its native pattern is
:math:`\delta B\mathbf1`.

Neither correction is multiplied by :math:`c`. This concentration-unit prior,
independently of numerical centering, avoids setting the prior width as a
percentage of the large absolute O2 concentration. It can also be used with
absolute operands.

The statistical constraint :math:`w^\mathsf{T}u=0`, with positive weights
summing to one, makes :math:`\delta` the weighted mean state correction.
These weights define the scientific average; they are distinct from transport
weights :math:`P` and basis weights :math:`B`. Equal state weights, or boundary
area times period duration, define different averages. The caller must choose
and document them. For fractional bases, relating state weights to an average
of the native field also requires accounting for that basis.

There are :math:`K` state anomalies but only :math:`K-1` independent anomaly
coordinates. The code uses a Helmert basis to sample them. These latent
coordinates are unrelated to native location or height. The precise anomaly
prior, pre-centering scale interpretation and configuration are given in
:doc:`co2_models`. Statistical centering of :math:`u` and numerical centering
of the observations are separate operations; neither resolves boundary--flux
or common-offset confounding.

What the current surface preparation computes
---------------------------------------------

The implementation keeps labelled xarray curtain arrays rather than building
all the matrices above. The following pseudocode describes its calculation;
it is not a callable Python example.

Native curtain products
^^^^^^^^^^^^^^^^^^^^^^^^

The local ``baseline_reference_sensitivities`` kernel, called by
:func:`~openghg_inversions.inversion_data.prepare_surface_site`, matches
supplied concentrations to footprint timestamps within their declared
coverage intervals and converts concentrations and the reference to common
units. All four curtains are normalized together per native observation:

.. code-block:: text

   for each native observation i:
       total = sum(particle_locations[i, direction, cell]
                   over every cell on all four curtains)
       require finite, nonnegative particle weights
       require total > 0 and abs(total - 1) <= mass_tolerance

       for each direction and cell:
           mass = particle_locations[i, direction, cell] / total
           anomaly = mass * (supplied_concentration[i, direction, cell] - c)
           retain mass and anomaly on their labelled curtain grids

The arrays are ``bc_mass_n/e/s/w`` and ``bc_anomaly_n/e/s/w``, with scalar
``bc_reference``. The tolerance is supplied explicitly and should reflect
justified transport error. This producer rejects materially incomplete
transport instead of normalizing it into apparently complete coverage. A
separately supplied incomplete unit response must not be silently normalized
to one by the model.

The products use float64 arithmetic and preserve lazy payloads. They do not
recover precision absent from the input concentrations. The native producer
does not infer a reference concentration or statistical centering weights.

Spatial basis projection
^^^^^^^^^^^^^^^^^^^^^^^^^

``prepare_surface_site`` averages the native mass and anomaly products onto
the retained observation bins using the provider's alignment operations before
spatial basis projection. It averages the products, rather than multiplying
separately averaged transport and concentration. Both operands follow the same
observation alignment.

``bc_sensitivity`` contracts both native products with the same spatial basis:

.. code-block:: text

   for each retained observation i and spatial basis region k:
       M_spatial[i, k] = sum(mass[i, direction, cell]
                             * spatial_basis[direction, cell, k])
       A_spatial[i, k] = sum(anomaly[i, direction, cell]
                             * spatial_basis[direction, cell, k])
       # Both sums cover every curtain and its spatial cells.

With no supplied spatial basis, the four curtains themselves become the four
regions. With a basis, the code multiplies its supplied values and sums; this
contraction does not require binary spatial membership. It does not choose
interpolation weights for the caller. A singleton basis time applies to all
observations; multiple basis times must match the retained timestamps.
The additive model additionally requires its unit-correction response to be
finite, nonnegative and not all zero; the general projection algebra alone
does not establish that a signed basis is accepted by that model.

The outputs are ``H_bc_mass`` and ``H_bc_anomaly`` on spatial ``bc_region``
and observation axes, plus the unchanged scalar ``bc_reference``. Gathering
and observation averaging must carry both products consistently. The stored
``H_bc_anomaly`` is a fixed prepared operand; it is not the sampled
``bc_anomaly`` :math:`u` of the additive prior.

Temporal state construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At model binding,
:func:`~openghg_inversions.boundary_sensitivity.align_boundary_operands`
uses the retained observation timestamp to assign a period. It currently
constructs a binary period mask, not weighted interpolation across periods:

.. code-block:: text

   for each observation i:
       p_i = period_containing(observation_time[i])
       for each spatial region k and fitted period p:
           j = state_label(k, p)
           D = 1 if p == p_i else 0
           M[i, j] = M_spatial[i, k] * D
           A[i, j] = A_spatial[i, k] * D

Without a frequency, there is one period. Otherwise supported calendar-month
or fixed-duration rules determine membership. This assignment uses observation
time, not a particle's boundary-exit time. The resulting ``bc_region``
MultiIndex has ``bc_curtain`` and ``bc_period`` levels, so four spatial regions
and three periods give :math:`K=12` states.

This spatial projection followed by period masking implements the particular
basis used by the current path. Conceptually, the entry of :math:`B` for a
curtain cell sampled at retained observation :math:`i` and state :math:`(k,p)`
is its supplied spatial basis value for :math:`k` times the period indicator
:math:`D_{ip}`. The code applies these factors in stages rather than assembling
that full matrix. A future temporal interpolation could replace
the binary mask with appropriate fractional weights while retaining the
operator identities above, provided both operands use the same mapping.
The current period interface does not implement that interpolation.

The :ref:`centered preparation handoff <surface-centered-boundary-contract>`
explains how to save raw spatial operands and bind them to supported periods.
The CO2 model runners then materialize related operands at their model boundary,
where model construction uses matrix--vector products of the observation/state
operators. They use centered observations and means for the likelihood and
restore absolute predictive concentrations through the supported trace
annotation/replay path.
