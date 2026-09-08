# Nested-domain RHIME workflow

The inner-domain workflow splits one emissions source into two non-overlapping
spatial contributions: a coarse outer domain for the surrounding region and a
fine 6 km domain for the area of interest.

```text
Outer footprints + flux ── mask inner area ── outer basis ── H_outer ── x_outer ─┐
                                                                                ├─ predicted observations
Inner footprints + flux ──────────────────── inner basis ── H_inner ── x_inner ─┘
```

## 1. Retrieve both domains independently

RHIME retrieves:

- the normal outer-domain footprints and emissions;
- the 6 km inner-domain footprints and emissions; and
- observations and boundary conditions through the outer preparation.

The inner domain uses the same emissions source, but has its own grid,
footprint store, emissions store, and basis definition. It is not treated as
another emissions sector.

## 2. Establish one canonical observation set

Observation filtering is performed once using the outer data. RHIME then
selects the corresponding inner footprints for exactly the same sites and
times.

Exact timestamp matching is the default. An optional tolerance allows
nearest-time matching:

```python
inner_time_tolerance="30min"
```

If an outer observation cannot be matched to an inner footprint, the run
fails instead of silently using zero sensitivity.

## 3. Remove the overlap from the outer domain

RHIME identifies the rectangular latitude–longitude extent of the inner grid.
Within that area, it sets the outer-domain footprint, footprint-times-flux
response, and prior emissions flux to zero.

Therefore, emissions inside the 6 km area contribute only through the inner
model. Emissions outside it contribute through the outer model. This prevents
double counting.

Conceptually:

\[
F_{\mathrm{outer,masked}}(s)=
\begin{cases}
0, & s \text{ inside inner domain}\\
F_{\mathrm{outer}}(s), & s \text{ outside inner domain}
\end{cases}
\]

## 4. Build two independent basis representations

RHIME constructs each basis on its native grid:

| Domain | State dimension | Sensitivity | Scaling vector |
|---|---|---|---|
| Outer | `region` | `H_outer` | `x_outer` |
| Inner | `inner_region` | `H_inner` | `x_inner` |

The inner grid is never interpolated or coerced onto the outer grid. Only the
observation-space sensitivity matrices are combined.

If `inner_nbasis` is not specified, the total `nbasis` budget is divided using
the square-root ratio of outer and inner absolute sensitivities. The inner
allocation is constrained to 35–60% of the total.

## 5. Infer two state vectors jointly

The PyMC model creates separate scaling-factor vectors:

\[
x_{\mathrm{outer}}
\quad\text{and}\quad
x_{\mathrm{inner}}
\]

Their observation-space contributions are:

\[
\mu_{\mathrm{outer},t}
=
\sum_r H_{\mathrm{outer}}(r,t)x_{\mathrm{outer},r}
\]

\[
\mu_{\mathrm{inner},t}
=
\sum_q H_{\mathrm{inner}}(q,t)x_{\mathrm{inner},q}
\]

The total pollution contribution is:

\[
\mu_{\mathrm{pollution},t}
=
\mu_{\mathrm{outer},t}
+
\mu_{\mathrm{inner},t}
\]

The posterior contains the separate variables `x_outer` and `x_inner`, plus
their contributions `mu_outer`, `mu_inner`, and combined pollution signal
`mu`.

By default, `x_inner` uses a copy of the outer scaling prior. It can be changed
independently with `inner_x_prior`.

## 6. Add the normal RHIME baseline and likelihood

After combining the two emissions terms, RHIME adds the usual components:

\[
\mu_{\mathrm{total}}
=
\mu_{\mathrm{outer}}
+
\mu_{\mathrm{inner}}
+
\mu_{\mathrm{BC}}
+
\mu_{\mathrm{offset}}
\]

The boundary-condition term comes only from the outer domain; there is no
second inner boundary condition. The resulting mean is passed through the
normal RHIME likelihood, including observation error, aggregation error, and
the selected model-data mismatch treatment.

All state vectors are sampled jointly, so the observations determine how the
scaling is allocated between outer and inner regions. Identifiability depends
on the footprints providing sufficiently distinct outer and inner sensitivity
patterns.

## Basis-file handling

The workflow can use saved basis files, but it does not expect one basis file
containing both domains. The two domains have independent basis definitions:

| Domain | Configuration options | Expected contents |
|---|---|---|
| Outer | `fp_basis_case`, `basis_directory`, `basis_output_path` | Coarse outer grid and outer region labels |
| Inner | `inner_fp_basis_case`, `inner_basis_directory`, `inner_basis_output_path` | Fine 6 km grid and inner region labels |

Each basis file must reflect its own native grid. The outer file produces
`H_outer` and `x_outer`; the inner file produces `H_inner` and `x_inner`.

Example:

```python
result = run_rhime_nested(
    domain="EUROPE",
    inner_domain="6km",
    fp_basis_case="outer_basis",
    basis_directory="/data/basis/outer",
    inner_fp_basis_case="inner_6km_basis",
    inner_basis_directory="/data/basis/inner",
    output_format="none",
)
```

The outer basis should cover the complete outer grid. During preparation,
RHIME masks the inner-domain contribution from the outer footprints and flux,
so the corresponding outer sensitivity is zero inside the inner extent. A
purpose-built outer basis that groups the remaining area sensibly may still
be preferable.

If basis files are not supplied, RHIME generates separate outer and inner
bases. The inner basis defaults to quadtree and can be saved independently.

## Results and current limitation

The result retains both native spatial representations:

```python
result.idata.posterior[["x_outer", "x_inner"]]
result.inv_inputs[["H", "H_inner"]]
result.outer_basis_functions
result.inner_basis_functions
```

Nested runs support `output_format="none"` or `output_format="paris"`; other
formats (`"basic"`, `"legacy"`, ...) are still rejected because their writers
assume a single spatial grid.

`output_format="paris"` does not merge the two grids onto one either. It
builds two ordinary, single-grid `InversionOutput` views of the shared
posterior (see `openghg_inversions.postprocessing.nested_paris_outputs`) --
one reading `x_outer`/`hx_outer` against the outer basis, one reading
`x_inner`/`hx_inner` against the inner basis -- and runs the existing
single-grid PARIS writer against each unmodified. The result is three
products: an outer PARIS flux file (zero inside the inner extent, so it never
double-counts against the inner domain), a separate native-resolution inner
PARIS flux file (including its own country totals, computed against a
nearest-neighbour-resampled country map when no inner-resolution one is
supplied), and one shared PARIS concentration file. The complete posterior
and both basis objects remain available for custom dual-grid post-processing
regardless of `output_format`.
