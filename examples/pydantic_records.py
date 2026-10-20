"""Export current schemas and roundtrip metadata without acquiring scientific data.

Run ``python -m examples.pydantic_records /tmp/pydantic-records`` from this
prototype checkout. No model graph, inference or array materialization runs.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

from pydantic import TypeAdapter, ValidationError

from openghg_inversions._record_schema import validate_json_keys
from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.inversion_data._site_options import InletSlice
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime._manifest_records import (
    Co2PreparationWriteRecord, Co2SampleWriteRecord,
    MultisectorPreparationRecord, MultisectorSampleRecord,
    StandardPreparationRecord, StandardSampleRecord,
)
from openghg_inversions.rhime._model_records import ModelDescription
from openghg_inversions.rhime.params import DataConfig, RhimeConfig
from openghg_inversions.rhime.specs import AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings


def run_example(destination: Path) -> list[Path]:
    """Write schemas and portable choices without scientific execution."""
    destination.mkdir(parents=True, exist_ok=True)
    records = (
        OutputContract, ModelDescription, InletSlice,
        StandardPreparationRecord, StandardSampleRecord,
        MultisectorPreparationRecord, MultisectorSampleRecord,
        Co2PreparationWriteRecord, Co2SampleWriteRecord,
    )
    paths = []
    for owner in records:
        path = destination / f"{owner.__name__}.schema.json"
        path.write_text(json.dumps(owner.model_json_schema(), indent=2) + "\n")
        paths.append(path)
    for settings in (PollutionEventSettings, AdditiveSigmaSettings, FixedErrorSettings, RhimeSampler, RhimeConfig):
        path = destination / f"{settings.__name__}.schema.json"
        path.write_text(json.dumps(TypeAdapter(settings).json_schema(mode="serialization"), indent=2) + "\n")
        paths.append(path)

    config = RhimeConfig.from_params({
        "species": "ch4", "domain": "EUROPE", "sites": ["tac"], "averaging_period": "1h",
        "inlet": [slice("10m", "100m")], "start_date": "2019-01-01", "end_date": "2019-02-01",
        "flux_sources": ["inventory"], "basis_path": destination / "unopened-basis.nc",
        "use_bc": False, "mismatch_model": "pollution_event", "output_name": "inspect",
        "output_format": "none", "draws": 23, "chains": 2,
        "sample_prior_predictive": 5, "sample_posterior_predictive": ["concentration"],
        "sample_kwargs": {"target_accept": .95, "random_seed": [10, 11]},
    }, multisector=False)
    assert config.model.likelihood.sigma_freq_anchor == config.data.start_date
    # This explicit projection owns Path/tuple/slice encodings; validation never computes arrays.
    data = config.data.to_params()
    validate_json_keys(data)
    data = _json_value(data)
    data_path = destination / "resolved-data.json"
    data_path.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    assert DataConfig.from_params(json.loads(data_path.read_text())) == config.data
    paths.append(data_path)

    # The sampler owns execution settings and their portable representation.
    for filename, current in (("resolved-config.json", config),
                              ("config-without-sampler.json", replace(config, sampler=None))):
        config_path = destination / filename
        config_path.write_text(json.dumps(current.to_dict(), allow_nan=False, indent=2) + "\n")
        assert RhimeConfig.from_dict(json.loads(config_path.read_text())) == current
        paths.append(config_path)

    contract = OutputContract.create(
        variable_roles={"flux_scale": "x", "concentration": "y"},
        supported_output_formats=("none", "inv_out"),
        metadata={"recipe": "standard", "sources": ["inventory"]},
    )
    contract_path = destination / "output-contract.json"
    contract_path.write_text(json.dumps(contract.to_dict(), indent=2) + "\n")
    assert OutputContract.from_dict(json.loads(contract_path.read_text())) == contract
    paths.append(contract_path)
    try:
        PollutionEventSettings(sigma_per_site="yes")
    except ValidationError as error:
        assert error.errors()[0]["loc"] == ("sigma_per_site",)
    else:
        raise AssertionError("Invalid component setting was accepted")
    return paths


if __name__ == "__main__":
    for path in run_example(Path(sys.argv[1])):
        print(path)
