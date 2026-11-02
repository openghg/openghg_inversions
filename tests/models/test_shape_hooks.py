"""Shape inference compatibility for the custom fixed-OU likelihood Ops."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph

from openghg_inversions.models.fixed_ou import _FixedOuLogpOp
from openghg_inversions.rhime.cached_sigma import _PytensorSigmaLikelihoodOp


@pytest.mark.parametrize("op_type", [_FixedOuLogpOp, _PytensorSigmaLikelihoodOp])
def test_shape_hooks_support_both_pytensor_signatures(op_type) -> None:
    # Shape inference must not need a scientific target evaluation.
    op = op_type(cast(Any, SimpleNamespace(n_obs=5)))
    left, right = pt.dvectors("left", "right")
    node = op.make_node(left, right)
    input_shapes = [(3,), (2,)]
    expected = [(), *input_shapes[: len(node.outputs) - 1]]
    graph = FunctionGraph([left, right], node.outputs, clone=False)

    assert op.infer_shape(node, input_shapes) == expected
    assert op.infer_shape(graph, node, input_shapes) == expected

    with pytensor.config.change_flags(on_shape_error="raise"):
        shapes = pytensor.function(
            [left, right],
            [output.shape for output in node.outputs],
            on_unused_input="ignore",
        )
    for actual, wanted in zip(shapes(np.zeros(3), np.zeros(2)), expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert all(apply.op is not op for apply in shapes.maker.fgraph.toposort())
