"""Literal selection preserves dynamic axes, retained unit axes and bounds."""

from io import StringIO

import pytest
import torch
from torch.fx.experimental import _config
from xdsl.dialects import builtin, tensor
from xdsl.ir import Block
from xdsl.printer import Printer

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.coverage import opaque_report
from m2m.ir.tensor_selection import select


@pytest.mark.parametrize("dtype", (torch.int64, torch.float32))
def test_real_export_keeps_dynamic_selected_axes_and_scalar_results(dtype):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x[1, :, :], x[:, :, -1], x[1, 0, -1]

    x = torch.arange(15, dtype=dtype).reshape(3, 1, 5)
    with _config.patch(backed_size_oblivious=True):
        program = torch.export.export(Model(), (x,), dynamic_shapes={"x": {
            0: torch.export.Dim("rows", min=2, max=9), 2: torch.export.Dim("cols", min=2, max=17)}})
    result = m2m.convert(program, (x,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    assert result.capture_trace["status"] == "complete"
    result.module.verify()
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
    assert sum(isinstance(op, tensor.ExtractSliceOp) for op in result.module.walk()) >= 5


@pytest.mark.parametrize("shape,axis,index", (
    ((2, 1, 2, 6, 32), 0, 1), ((1, 1, 3), 0, 0), ((1, 1, 3), 1, 0),
    ((1, 1, 1), 1, -1), ((3,), 0, -1), ((0, 3), 1, 1), ((3, 0), 0, 1),
))
def test_static_unit_and_empty_axes_remain_native_verifiable(shape, axis, index):
    from xdsl.dialects import func
    from xdsl.ir import Region

    original = builtin.TensorType(builtin.i64, shape)
    wanted = builtin.TensorType(builtin.i64, [n for i, n in enumerate(shape) if i != axis])
    block = Block(arg_types=[original])
    operations, result = select(block.args[0], wanted, axis, index)
    block.add_ops([*operations, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([original], [wanted]), Region(block))])
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    _parse_mlir_text_to_xdsl(stream.getvalue()).verify()


@pytest.mark.parametrize("axis,index", ((0, 3), (0, -4), (2, 0), (-3, 0), (True, 0), (0, False), (0, 1 << 63)))
def test_invalid_literal_selections_refuse_before_emitting_or_clamping(axis, index):
    typ = builtin.TensorType(builtin.i64, [3, 1])
    source = Block(arg_types=[typ]).args[0]
    assert select(source, builtin.TensorType(builtin.i64, [1]), axis, index) is None
    assert not tuple(source.uses)


def test_metadata_cannot_specialize_a_retained_dynamic_axis():
    typ = builtin.TensorType(builtin.i64, [3, builtin.DYNAMIC_INDEX])
    source = Block(arg_types=[typ]).args[0]
    assert select(source, builtin.TensorType(builtin.i64, [5]), 0, 1) is None
    assert not tuple(source.uses)
