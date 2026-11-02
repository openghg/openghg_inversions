"""Resolved reporting choices shared by the PARIS output recipes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Self

from .make_paris_outputs import (
    DEFAULT_PARIS_TEMPLATE_VERSION,
    PARIS_LATEST_COUNTRIES,
    ParisTemplateVersion,
    validate_paris_options,
)


@dataclass(frozen=True, kw_only=True)
class ParisOptions:
    """Passive PARIS reporting choices shared across model families.

    Attributes:
        template_version: Select the legacy or latest CDL schema.
        time_point: Place flux timestamps at interval starts or midpoints.
        flux_frequency: Explicit calendar period or fixed duration; None requests inference.
        report_mode: Report KDE modes instead of means.
        inversion_grid: Include reduced inversion-grid flux variables.
        country_selections: Country names/codes; None includes all countries in the file.
    """

    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION
    time_point: Literal["start", "midpoint"] = "midpoint"
    flux_frequency: str | None = None
    report_mode: bool = False
    inversion_grid: bool = True
    country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None, *,
        default_template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
        default_inversion_grid: bool = True,
        default_country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES,
    ) -> Self:
        """Resolve and validate a recipe's PARIS mapping without constructing products.

        Args:
            params: Raw reporting options. Unknown and recipe-owned argument names fail.
            default_template_version: Recipe default when no schema was selected.
            default_inversion_grid: Recipe default for reduced-grid variables.
            default_country_selections: Recipe default country selection. An explicit
                None in params is preserved.

        Returns:
            Validated options of the concrete class, without mutating params.

        Raises:
            ValueError: If names, template, timestamp, or explicit frequency are unsupported.
            FileNotFoundError: If a nested option selects a missing inner country file.
        """
        values = dict(params) if params is not None else {}
        if unknown := values.keys() - {field.name for field in fields(cls)}:
            raise ValueError(f"Unsupported PARIS postprocessing options: {sorted(unknown)!r}.")
        values.setdefault("template_version", default_template_version)
        values.setdefault("inversion_grid", default_inversion_grid)
        values.setdefault("country_selections", default_country_selections)
        options = cls(**values)
        options.validate()
        return options

    def validate(self) -> None:
        """Check template, timestamp and explicit frequency constraints."""
        validate_paris_options(
            template_version=self.template_version, time_point=self.time_point,
            flux_frequency=self.flux_frequency,
        )


@dataclass(frozen=True, kw_only=True)
class NestedParisOptions(ParisOptions):
    """PARIS choices with optional inner-grid reporting resources.

    Attributes:
        inner_country_file: Existing country mask on the inner grid; None requests regridding.
        country_file_cache_dir: Destination for cached regridded masks; None uses the user cache.
    """

    inner_country_file: str | Path | None = None
    country_file_cache_dir: str | Path | None = None

    def validate(self) -> None:
        """Also require an explicitly selected inner country file to exist."""
        super().validate()
        if self.inner_country_file is not None and not Path(self.inner_country_file).is_file():
            raise FileNotFoundError(self.inner_country_file)
