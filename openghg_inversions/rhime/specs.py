"""RHIME runner specification dataclasses.

This module contains lightweight immutable dataclasses used by the RHIME
runner. Model-construction specs live with the RHIME model builders; these
runner specs keep sampling and output implementation details out of the run
metadata boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from openghg_inversions.models.rhime import RhimeModelSpec

OutputFormat = Literal["none", "inv_out", "basic", "paris", "legacy"]
OutputFilenameConvention = Literal["rhime", "legacy"]


@dataclass(frozen=True)
class RhimeOutputSpec:
    """Output settings for a RHIME run.

    Args:
        output_format: Output mode. ``"inv_out"`` saves/returns the modern
            inversion output, ``"basic"`` and ``"paris"`` additionally create
            derived outputs, ``"legacy"`` creates the old HBMCMC-compatible
            NetCDF product from modern RHIME output, and ``"none"`` skips
            output products.
        output_path: Directory for saved outputs.
        output_name: Base output name.
        save_trace: Trace save setting. If true, save to ``output_path`` using
            the default trace file name; if a path, save there.
        save_inversion_output: Inversion-output save setting. Defaults to true
            for CLI-friendly behaviour.
        country_file: Optional country mask file used by derived outputs.
        paris_postprocessing_kwargs: Extra keyword arguments for PARIS output
            creation.
        output_filename_convention: Filename convention for derived products.
            Direct RHIME runs use ``"rhime"``. The ``run_hbmcmc.py``
            compatibility shim uses ``"legacy"`` for old SLURM/config
            workflows.
    """

    output_format: OutputFormat = "inv_out"
    output_path: str | None = None
    output_name: str = "rhime"
    save_trace: str | Path | bool = False
    save_inversion_output: str | Path | bool = True
    country_file: str | None = None
    paris_postprocessing_kwargs: dict[str, Any] | None = None
    output_filename_convention: OutputFilenameConvention = "rhime"


@dataclass(frozen=True)
class RhimeRunSpec:
    """Top-level run metadata for a RHIME run.

    Args:
        start_date: Inclusive inversion start date.
        end_date: Exclusive inversion end date.
        sites: Sites included after data preparation and filtering.
        averaging_period: Observation averaging period per retained site.
        model: Mathematical model specification.
        output: Output settings.
        split_by_sectors: Whether flux data were prepared in sector-resolved
            mode. Single-sector and multi-sector RHIME are runner/model modes;
            this flag records the prepared data layout.
    """

    start_date: str
    end_date: str
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    model: RhimeModelSpec
    output: RhimeOutputSpec
    split_by_sectors: bool = False


def validate_output_format(output_format: str) -> None:
    """Raise if a RHIME output format is not supported by the modern runners."""
    valid_formats = {"none", "inv_out", "basic", "paris", "legacy"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Unsupported RHIME output_format {output_format!r}; expected one of {sorted(valid_formats)!r}."
        )


def validate_output_path_settings(
    *,
    output_format: str,
    output_path: str | None,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    multisector: bool,
) -> None:
    """Raise if output settings imply a default save path but none is supplied."""
    if multisector and output_format == "legacy":
        raise ValueError("RHIME output_format 'legacy' supports only single-sector runs.")
    if output_format == "none":
        return
    if output_path is not None:
        return
    if save_trace is True:
        raise ValueError("`output_path` is required when `save_trace=True`.")
    if save_inversion_output is True:
        raise ValueError("`output_path` is required when saving the RHIME InversionOutput.")


def validate_output_filename_convention(output_filename_convention: str) -> None:
    """Raise if an output filename convention is not supported."""
    valid_conventions = {"rhime", "legacy"}
    if output_filename_convention not in valid_conventions:
        raise ValueError(
            "Unsupported RHIME output_filename_convention "
            f"{output_filename_convention!r}; expected one of {sorted(valid_conventions)!r}."
        )


def make_output_spec(
    *,
    output_format: str,
    output_path: str | None,
    output_name: str,
    save_trace: str | Path | bool,
    save_inversion_output: str | Path | bool,
    country_file: str | None,
    paris_postprocessing_kwargs: dict[str, Any] | None,
    output_filename_convention: str,
    multisector: bool,
) -> RhimeOutputSpec:
    """Create validated output settings from normalized RHIME parameters."""
    validate_output_format(output_format)
    validate_output_filename_convention(output_filename_convention)
    validate_output_path_settings(
        output_format=output_format,
        output_path=output_path,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        multisector=multisector,
    )
    return RhimeOutputSpec(
        output_format=cast(OutputFormat, output_format),
        output_path=output_path,
        output_name=output_name,
        save_trace=save_trace,
        save_inversion_output=save_inversion_output,
        country_file=country_file,
        paris_postprocessing_kwargs=paris_postprocessing_kwargs,
        output_filename_convention=cast(OutputFilenameConvention, output_filename_convention),
    )
