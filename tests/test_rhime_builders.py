import pymc as pm

from openghg_inversions.models import registered_model
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult


def test_build_result_retains_one_validated_contract():
    with registered_model() as model:
        pm.Normal("y")
    roles = {"concentration": "y"}
    contract = OutputContract.create(variable_roles=roles)
    result = RhimeModelBuildResult(model=model, output_contract=contract)
    roles["concentration"] = "changed"
    assert result.output_contract is contract
    assert result.output_contract.variable_roles == {"concentration": "y"}
