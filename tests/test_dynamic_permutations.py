"""Permutations preserve actual dynamic dimensions without example hints."""

import pytest
import torch
from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.func import FuncOp
from xdsl.dialects.linalg.ops import TransposeOp

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.coverage import opaque_report
from m2m.ir.permutations import lower


@pytest.mark.parametrize(
    "kind,shape,dynamic,permutation",
    (
        ("t", (3, 5), (0, 1), (1, 0)),
        ("permute", (3, 4, 5), (0, 2), (2, 0, 1)),
        ("negative", (3, 4, 5), (0, 2), (2, 0, 1)),
        ("transpose", (3, 4, 5), (0, 2), (2, 1, 0)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.int64, torch.float32))
def test_real_exports_allocate_transposes_from_original_source_axes(
    kind, shape, dynamic, permutation, dtype
):
    class Model(torch.nn.Module):
        def forward(self, x):
            if kind == "t":
                return x.t()
            if kind == "permute":
                return x.permute(2, 0, 1)
            if kind == "negative":
                return x.permute(-1, -3, -2)
            return x.transpose(0, -1)

    x = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    exported = torch.export.export(
        Model(),
        (x,),
        dynamic_shapes={
            "x": {
                axis: torch.export.Dim("axis" + str(axis), min=2, max=9)
                for axis in dynamic
            }
        },
    )
    result = m2m.convert(exported, (x,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    assert result.capture_trace["status"] == "complete"
    result.module.verify()
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
    entry = next(
        op for op in result.module.ops if isinstance(op, FuncOp) and op.body.blocks
    )
    original = entry.body.block.args[0]
    transposes = [op for op in result.module.walk() if isinstance(op, TransposeOp)]
    assert len(transposes) == 1
    transpose = transposes[0]
    assert tuple(transpose.permutation.get_values()) == permutation
    assert transpose.inputs[0] is original
    destination = transpose.outputs[0].owner
    assert isinstance(destination, tensor.EmptyOp)
    expected_axes = [axis for axis in permutation if axis in dynamic]
    assert len(destination.operands) == len(expected_axes)
    for size, axis in zip(destination.operands, expected_axes, strict=True):
        assert isinstance(size.owner, tensor.DimOp)
        assert size.owner.source is original
        assert isinstance(size.owner.index.owner, arith.ConstantOp)
        assert size.owner.index.owner.value.value.data == axis
    shape_type = original.type.get_shape()
    assert list(transpose.results[0].type.get_shape()) == [
        shape_type[axis] for axis in permutation
    ]


@pytest.mark.parametrize("shape", ((), (7,), (0, 3), (1, 4)))
def test_torch_t_retains_scalar_vector_and_empty_static_cases(shape):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.t()

    value = torch.empty(shape, dtype=torch.int64)
    result = m2m.convert(Model(), (value,), backend="fx_importer")
    assert result.ok and opaque_report(result.mlir_text) == {}
    result.module.verify()
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()


@pytest.mark.parametrize("permutation", ((0, 0), (0,), (0, 2), (-3, 0), (False, 1)))
def test_invalid_or_incomplete_axis_maps_do_not_allocate_a_destination(permutation):
    entry = FuncOp(
        "probe", ([builtin.TensorType(builtin.i64, [builtin.DYNAMIC_INDEX, 3])], [])
    )
    assert lower(entry.body.block.args[0], permutation, [-1, 3]) is None
    assert not tuple(entry.body.block.ops)


def test_metadata_cannot_specialize_an_unknown_input_axis():
    entry = FuncOp(
        "probe", ([builtin.TensorType(builtin.i64, [builtin.DYNAMIC_INDEX, 3])], [])
    )
    assert lower(entry.body.block.args[0], [1, 0], [3, 5]) is None
