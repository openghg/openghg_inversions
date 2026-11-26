"""Acquire the observations and transport inputs for the surface CO2 recipe."""

from collections.abc import Mapping
from typing import Any

from openghg_inversions.inversion_data import MergedData


def acquire_co2_data(
    acquisition: Mapping[str, Any] | None, *,
    flux_sources: tuple[str, ...], domain: str, start_date: str, end_date: str,
) -> MergedData:
    """Retrieve native CO2 inputs before observation filtering or basis construction.

    Args:
        acquisition: Resolved arguments for surface retrieval.
        flux_sources: Requested flux inventory names.
        domain: Requested transport domain.
        start_date: Requested window start.
        end_date: Requested window end.

    Returns:
        Acquired observations, transport and native flux.

    Raises:
        ValueError: If no acquisition selectors are configured.
    """
    if acquisition is None:
        raise ValueError("Pass merged_data or configure data.acquisition for fresh retrieval.")
    return MergedData.from_options(
        species="co2", domain=domain, start_date=start_date, end_date=end_date,
        flux_sources=flux_sources, **acquisition,
    )


__all__ = ["acquire_co2_data"]
