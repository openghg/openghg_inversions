"""Guidance for unsupported historical request options."""

from collections.abc import Iterable

_REMOVED_OPTIONS = {
    "calculate_min_error": "`calculate_min_error` is not supported by RHIME runners; use `min_error`.",
    "reparameterise_log_normal": (
        "`reparameterise_log_normal` is not supported by RHIME runners; "
        "set `reparameterise` in the relevant prior dictionary if needed."
    ),
    "mcmc_type": "`mcmc_type` is not supported by RHIME runners; use `nuts_sampler` if needed.",
}


def migration_errors(names: Iterable[str]) -> list[str]:
    """Explain every removed option among ``names`` at frontend validation boundaries."""
    names = set(names)
    return resource_migration_errors(names) + [message for name, message in _REMOVED_OPTIONS.items() if name in names]


def resource_migration_errors(names: Iterable[str]) -> list[str]:
    """Explain removed resource selectors at frontend validation boundaries."""
    names = set(names)
    errors = []
    if "country_directory" in names:
        errors.append(
            "`country_directory` was removed in 0.8; use `country_file` for reporting, "
            "`landsea_path` for land/sea classes, or `outer_regions_path` for outer regions."
        )
    historical = {"fp_basis_case", "basis_directory", "bc_basis_case", "bc_basis_directory"} & names
    if historical:
        errors.append(
            f"Historical resource selectors {sorted(historical)!r} are unsupported. "
            "Use `basis_path`, `basis_partition_path`/`basis_partition_paths`, or `bc_basis_paths` "
            "in modern RHIME requests."
        )
    return errors
