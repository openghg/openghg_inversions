"""Scientific readers consume validated records without a second JSON boundary."""

import pytest

from examples.multisector_handoffs import synthetic_inputs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime import _standard_stages, multisector_stages
from openghg_inversions.rhime.co2 import handoffs
from openghg_inversions.rhime.params import DataConfig
from tests.test_co2_handoffs import FACTS, _components_fixture
from tests.test_multisector_handoffs import _save_sample
from tests.test_standard_handoffs import _saved


@pytest.mark.parametrize("family", ["standard", "multisector", "co2"])
def test_saved_readers_use_validated_values_without_writer_projection(tmp_path, monkeypatch, family):
    if family == "standard":
        stages = _standard_stages
        preparation, samples = _saved(tmp_path)
    elif family == "multisector":
        stages = multisector_stages
        merged, options, *_ = synthetic_inputs(tmp_path / "inputs")
        data = DataConfig.from_params(options, multisector=True)
        preparation = stages.prepare(data, merged_data=merged, output_dir=tmp_path / "prepared")
        samples = _save_sample(tmp_path, (preparation, stages.load_preparation(preparation)))
    else:
        stages = handoffs
        prepared, trace = _components_fixture()
        preparation = stages.save_preparation(prepared, output_dir=tmp_path / "prepared", **FACTS)
        generation = stages.load_preparation(preparation).generation
        samples = stages.save_samples(
            preparation, trace, actual_inference={"draws": 3},
            preparation_generation=generation, output_dir=tmp_path / "samples",
        )

    read_checkpoint = stages.read_checkpoint
    records = []

    def read_once(*args, **kwargs):
        path, record = read_checkpoint(*args, **kwargs)
        records.append(record)
        return path, record

    def writer_boundary(*args, **kwargs):
        pytest.fail("A scientific reader projected or revalidated an already decoded record.")

    monkeypatch.setattr(stages, "read_checkpoint", read_once)
    monkeypatch.setattr(stages, "checkpoint_dict", writer_boundary)
    monkeypatch.setattr(OutputContract, "from_dict", writer_boundary)
    stages.load_preparation(preparation)
    result = stages.load_samples(samples)
    sample = next(record for record in records if record.stage == "sample")
    assert result.output_metadata["sampler"]["draws"] == sample.actual_inference["draws"]
