"""Boundary validation for externally supplied RHIME stage products."""

from __future__ import annotations

from collections.abc import Sequence

from openghg_inversions.inversion_data import RhimeMergedData


def validate_external_merged_data(
    merged_data: RhimeMergedData,
    *,
    requested_sites: Sequence[str],
    multisector: bool,
) -> RhimeMergedData:
    """Validate the contents promised by a borrowed merged-data handoff."""
    if not isinstance(merged_data, RhimeMergedData):
        raise TypeError(f"`merged_data` must be a RhimeMergedData handoff; got {type(merged_data).__name__}.")

    resolved_sites = {str(site).upper() for site in requested_sites}
    unexpected_sites = [site for site in merged_data.sites if site not in resolved_sites]
    if unexpected_sites:
        raise ValueError(
            f"External RHIME merged data contains site(s) outside the resolved run: {unexpected_sites!r}."
        )

    retained_sites = set(merged_data.sites)
    stored_sites = {str(name).upper() for name in merged_data.fp_all if not str(name).startswith(".")}
    missing_site_data = sorted(retained_sites - stored_sites)
    undeclared_site_data = sorted(stored_sites - retained_sites)
    if missing_site_data or undeclared_site_data:
        raise ValueError(
            "External RHIME merged data site contents do not match its retained site metadata: "
            f"missing site data {missing_site_data!r}; undeclared site data {undeclared_site_data!r}."
        )

    if ".split_by_sectors" not in merged_data.fp_all:
        raise ValueError(
            "External RHIME merged data must declare an explicit '.split_by_sectors' sector layout."
        )
    stored_multisector = merged_data.fp_all[".split_by_sectors"]
    if type(stored_multisector) is not bool:
        raise ValueError("External RHIME merged data '.split_by_sectors' must be a boolean.")
    if stored_multisector != multisector:
        raise ValueError(
            "External RHIME merged data has an incompatible sector layout: "
            f"artifact split_by_sectors={stored_multisector!r}, "
            f"runner multisector={multisector!r}."
        )
    return merged_data
