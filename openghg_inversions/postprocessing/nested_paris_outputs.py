"""PARIS output construction for nested (dual-grid) RHIME runs.

Nested RHIME preparation (:mod:`openghg_inversions.rhime.nested`) keeps the
outer and inner domains as independently retained basis operators and flux
grids -- see that module's docstring for why they are never merged onto one
grid. This module turns a sampled ``NestedRhimeResult`` into two ordinary,
single-grid :class:`~openghg_inversions.postprocessing.inversion_output.InversionOutput`
*views* over the same shared trace (one per domain, with variable roles
pointed at that domain's tagged ``x_outer``/``x_inner`` trace variables) and
reuses the existing single-grid PARIS/flux/country postprocessing unmodified
against each view.

The outer view already has its prior flux and footprint response masked to
zero over the inner domain's extent
(:func:`openghg_inversions.rhime.nested.mask_outer_merged_for_inner_domain`),
so its flux and country totals never double-count inner-domain emissions.
The inner view reports genuine native-resolution (e.g. 6 km) flux and
country totals on its own grid; it is a separate product and is never
regridded onto the outer grid for arithmetic.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import xarray as xr

from openghg_inversions._country_file import load_country_dataset
from openghg_inversions.postprocessing.countries import regrid_country_dataset
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_paris_outputs import (
    DEFAULT_PARIS_TEMPLATE_VERSION,
    PARIS_LATEST_COUNTRIES,
    ParisTemplateVersion,
    infer_flux_frequency,
    paris_concentration_outputs,
    paris_flux_output,
)
from openghg_inversions.rhime.outputs import _make_inversion_output
from openghg_inversions.utils import get_country_file_path

if TYPE_CHECKING:
    from openghg_inversions.rhime.builders import RhimeModelBuildResult
    from openghg_inversions.rhime.nested import NestedRhimeResult

__all__ = [
    "make_nested_inversion_outputs",
    "make_nested_paris_outputs",
]


def _domain_variable_roles(all_roles: Mapping[str, str], *, tag: str) -> dict[str, str]:
    """Return one nested domain's role mapping, stripping its ``:tag`` suffix.

    ``build_nested_rhime_model_result`` declares roles shared by both domains
    (``"observation"``, ``"concentration"``, ``"boundary"``, ...) alongside
    domain-tagged roles (``"flux_scale:outer"``, ``"flux_scale:inner"``, ...).
    Selecting one domain's tagged roles and merging them with the shared
    roles reproduces the ordinary, untagged role mapping that single-grid
    postprocessing (``make_flux_outputs``, ``make_country_outputs``, ...)
    expects.
    """
    roles: dict[str, str] = {}
    for role, name in all_roles.items():
        if ":" in role:
            base_role, role_tag = role.split(":", 1)
            if role_tag == tag:
                roles[base_role] = name
        else:
            roles[role] = name
    return roles


def make_nested_inversion_outputs(nested_result: "NestedRhimeResult") -> tuple[InversionOutput, InversionOutput]:
    """Build outer- and inner-domain ``InversionOutput`` views of a nested result.

    Both views share the same sampled trace; they differ only in which
    prepared inputs, basis functions, and (domain-tagged) trace variables
    they read. ``domain`` metadata on both views is the *outer* domain name,
    so PARIS country-region lookups (e.g. ``country_regions="paris"``)
    resolve identically for both -- the inner view's own grid resolution is
    carried separately via ``nested_result.prepared_inputs.inner_domain_label``.

    Args:
        nested_result: Sampled nested RHIME result.

    Returns:
        ``(outer_inv_out, inner_inv_out)``.

    Raises:
        ValueError: If the nested model build result is missing (the result
            was not produced by ``run_rhime_nested``/``run_rhime_nested_from_prepared_inputs``).
    """
    rhime_result = nested_result.rhime_result
    build_result = cast("RhimeModelBuildResult | None", rhime_result.model_build_result)
    if build_result is None:
        raise ValueError("Nested RHIME result is missing its model build result.")

    all_roles = build_result.variable_roles
    outer_roles = _domain_variable_roles(all_roles, tag="outer")
    inner_roles = _domain_variable_roles(all_roles, tag="inner")

    outer_inv_out = _make_inversion_output(
        result=rhime_result,
        prepared=nested_result.prepared_inputs.outer,
        variable_roles=outer_roles,
    )
    inner_inv_out = _make_inversion_output(
        result=rhime_result,
        prepared=nested_result.prepared_inputs.inner,
        variable_roles=inner_roles,
    )
    return outer_inv_out, inner_inv_out


def _regridded_inner_country_file(
    *,
    country_file: str | Path | None,
    domain: str | None,
    lat: xr.DataArray,
    lon: xr.DataArray,
    inner_domain_label: str,
    cache_dir: str | Path | None,
) -> Path:
    """Return a country file resampled onto the inner grid, caching it on disk.

    Fine (inner) nested domains rarely have a matching country-definition
    file at their native resolution. Country/region membership does not
    change between repeated runs for the same physical inner domain (e.g.
    monthly array-job tasks), so the regridded file is cached by inner-domain
    label and reused rather than recomputed; concurrent writers race safely
    because the file is written to a process-unique temporary path and moved
    into place with an atomic rename.
    """
    source_path = get_country_file_path(country_file=country_file, domain=domain)
    target_dir = Path(cache_dir) if cache_dir is not None else source_path.parent
    safe_label = str(inner_domain_label).replace("/", "_")
    target_path = target_dir / f"country_{safe_label}_regridded.nc"
    if target_path.exists():
        return target_path

    countries_ds = load_country_dataset(source_path)
    regridded = regrid_country_dataset(countries_ds, lat=lat, lon=lon)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dir / f".country_{safe_label}_regridded.{os.getpid()}.tmp.nc"
    try:
        regridded.to_netcdf(tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target_path


def make_nested_paris_outputs(
    nested_result: "NestedRhimeResult",
    *,
    country_file: str | Path | None = None,
    inner_country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: str | None = None,
    obs_avg_period: str = "4h",
    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
    country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES,
    country_file_cache_dir: str | Path | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Build PARIS concentration, outer-flux, and inner-flux products for a nested run.

    Args:
        nested_result: Sampled nested RHIME result.
        country_file: Country-definition file for the outer domain (as for
            ordinary single-grid PARIS output).
        inner_country_file: Optional country-definition file already at the
            inner domain's native resolution. If not given, ``country_file``
            (or the outer domain's default) is resampled onto the inner grid
            and cached; see ``country_file_cache_dir``.
        time_point, report_mode, inversion_grid, template_version,
            country_selections: Forwarded to ``paris_flux_output`` /
            ``paris_concentration_outputs`` for both domains.
        flux_frequency: Flux interval frequency. If ``None``, it is inferred
            independently for each domain's flux.
        obs_avg_period: Averaging period recorded in concentration metadata.
        country_file_cache_dir: Directory for the cached regridded inner
            country file. Defaults to the outer country file's directory.

    Returns:
        ``(flux_outer, flux_inner, conc_outs)``. ``flux_outer`` matches the
        single-grid PARIS flux schema exactly, with zero emissions inside the
        inner extent, so it is never a source of double-counting against
        ``flux_inner``. ``flux_inner`` reports genuine native-resolution flux
        and country totals on the inner domain's own grid.
    """
    outer_inv_out, inner_inv_out = make_nested_inversion_outputs(nested_result)

    conc_outs = paris_concentration_outputs(
        outer_inv_out,
        report_mode=report_mode,
        obs_avg_period=obs_avg_period,
        template_version=template_version,
    )

    outer_flux_frequency = flux_frequency or infer_flux_frequency(outer_inv_out.flux)
    flux_outer = paris_flux_output(
        outer_inv_out,
        country_file=country_file,
        time_point=time_point,
        report_mode=report_mode,
        inversion_grid=inversion_grid,
        flux_frequency=outer_flux_frequency,
        template_version=template_version,
        country_selections=country_selections,
    )

    inner_domain_label = nested_result.prepared_inputs.inner_domain_label
    if inner_domain_label is None:
        inner_domain_label = f"{outer_inv_out.domain}-inner"

    resolved_inner_country_file = inner_country_file
    if resolved_inner_country_file is None:
        resolved_inner_country_file = _regridded_inner_country_file(
            country_file=country_file,
            domain=outer_inv_out.domain,
            lat=inner_inv_out.flux["lat"],
            lon=inner_inv_out.flux["lon"],
            inner_domain_label=inner_domain_label,
            cache_dir=country_file_cache_dir,
        )

    inner_flux_frequency = flux_frequency or infer_flux_frequency(inner_inv_out.flux)
    flux_inner = paris_flux_output(
        inner_inv_out,
        country_file=resolved_inner_country_file,
        time_point=time_point,
        report_mode=report_mode,
        inversion_grid=inversion_grid,
        flux_frequency=inner_flux_frequency,
        template_version=template_version,
        country_selections=country_selections,
    )
    flux_inner = flux_inner.copy()
    flux_inner.attrs = dict(flux_inner.attrs)
    flux_inner.attrs["domain"] = inner_domain_label
    flux_inner.attrs["inner_domain"] = inner_domain_label
    flux_inner.attrs["spatial_resolution"] = inner_domain_label
    flux_inner.attrs["nested_output_note"] = (
        "Native-resolution inner-domain flux and country totals; never regridded "
        "onto the outer/standard grid for arithmetic."
    )

    return flux_outer, flux_inner, conc_outs
