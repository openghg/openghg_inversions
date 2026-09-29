"""Bind prepared numerical values and samples for scientific reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import xarray as xr

from .contracts import OutputContract
from .inversion_output import InversionOutput

if TYPE_CHECKING:
    from openghg_inversions.inversion_data import RhimePreparedInputs


def make_inversion_output(
    *,
    prepared: RhimePreparedInputs,
    trace: xr.DataTree,
    contract: OutputContract,
    run_metadata: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    output_metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> InversionOutput:
    """Create a scientific output view without constructing a model or writing files.

    Args:
        prepared: Borrowed canonical arrays, retained basis and site metadata.
        trace: Borrowed posterior and model-data groups, preserving all chains.
        contract: Explicit roles, builder metadata and optional state-axis mapping.
        run_metadata: Resolved dates, sites and other run provenance.
        model_metadata: Resolved scientific model choices.
        output_metadata: Requested product settings and sampling provenance.
        provenance: Recipe-specific description of the scientific handoff.

    Returns:
        An ordinary ``InversionOutput`` consumed by reconstruction and product
        writers. Numerical validation and labelled state normalization remain
        owned by that value; the caller's arrays and metadata are not mutated.
    """
    model_metadata = dict(model_metadata)
    model_metadata["footprint_provenance"] = {
        str(site): {
            name: str(prepared.site_metadata[name].sel(site=site).item())
            for name in ("transport_model", "transport_model_version", "met_model")
            if name in prepared.site_metadata
        }
        for site in prepared.sites
    }
    model_metadata["variable_roles"] = dict(contract.variable_roles)
    if contract.state_dimension_mapping:
        model_metadata["state_dimension_mapping"] = dict(contract.state_dimension_mapping)
    if contract.metadata:
        model_metadata["builder"] = dict(contract.metadata)
    return InversionOutput(
        inv_inputs=prepared.inv_inputs,
        basis_functions=prepared.basis_functions,
        trace=trace,
        run_metadata={
            **run_metadata,
            "basis_artifact_source": prepared.basis_artifact_source,
            "basis_artifact_path": prepared.basis_artifact_path,
        },
        model_metadata=model_metadata,
        output_metadata=dict(output_metadata),
        provenance=dict(provenance),
    )
