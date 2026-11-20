"""Runtime bounds remain source SSA, including on statically sized parameters."""

from io import StringIO
from types import SimpleNamespace

import pytest
import torch
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.ir import Block, Region
from xdsl.printer import Printer

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.coverage import opaque_report
from m2m.ir.dynamic_slices import extract_slice


@pytest.mark.parametrize("static", (False, True))
@pytest.mark.parametrize("step", (1, 3, (1 << 63) - 1))
def test_signed_runtime_bounds_keep_wide_clipping_and_empty_control(static, step):
    source_type = builtin.TensorType(builtin.i32, [7 if static else builtin.DYNAMIC_INDEX])
    result_type = builtin.TensorType(builtin.i32, [builtin.DYNAMIC_INDEX])
    block = Block(arg_types=[source_type, builtin.i64, builtin.i64])
    operations, result = extract_slice(block.args[0], result_type, (0, *block.args[1:], step))
    block.add_ops([*operations, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([source_type, builtin.i64, builtin.i64], [result_type]), Region(block))])
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    _parse_mlir_text_to_xdsl(stream.getvalue()).verify()
    # Both signed extremes are possible: adding a negative bound to a size
    # must not execute a wrapped i64 addition before selecting the valid arm.
    assert any(isinstance(op, arith.AddiOp) and op.result.type.bitwidth > 64 for op in module.walk())
    assert any(isinstance(op, arith.ExtSIOp) and op.input in block.args[1:] for op in module.walk())
    conditional = next(op for op in module.walk() if isinstance(op, scf.IfOp))
    assert isinstance(conditional.true_region.block.first_op, tensor.ExtractSliceOp)
    assert isinstance(conditional.false_region.block.first_op, tensor.EmptyOp)
    assert all(op.parent_op() is conditional for op in module.walk() if isinstance(op, tensor.ExtractSliceOp))


def test_actual_exported_shape_bounds_lower_static_parameter_slices(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(9, dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            return self.weight[:x.shape[0]].clone(), self.weight[x.shape[0]:].clone(), x

    model, inputs = Model().eval(), (torch.arange(4, dtype=torch.int64),)
    exported = torch.export.export(model, inputs,
        dynamic_shapes=({0: torch.export.Dim("n", min=2, max=7)},))
    dimension = next(node for node in exported.graph.nodes if node.op == "placeholder" and node.name == "x").meta["val"].shape[0]
    guards = list(dimension.node.shape_env.guards)
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True,
                         weights_path=str(tmp_path / "weights.safetensors"))
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    assert dimension.node.shape_env.guards == guards
    result.module.verify()
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
    queries = [op.result for op in result.module.walk() if isinstance(op, tensor.DimOp)]
    assert queries and any(query.uses for query in queries)
    slices = [op for op in result.module.walk() if isinstance(op, tensor.ExtractSliceOp)]
    assert len(slices) == 2 and all(op.result.type.get_shape() == (builtin.DYNAMIC_INDEX,) for op in slices)


def test_unresolved_source_bounds_do_not_select_runtime_lowering():
    type_ = builtin.TensorType(builtin.i64, [9])
    block = Block(arg_types=[type_])
    unresolved = SimpleNamespace(name="unobserved")
    assert extract_slice(block.args[0], type_, (0, 0, unresolved, 1)) is None
    assert not tuple(block.args[0].uses)


@pytest.mark.parametrize("arguments", ((0, 0, 3, 0), (0, 0, 3, -1), (0, True, 3, 1),
                                        (0, 0, 1 << 63, 1)))
def test_invalid_literal_contracts_keep_their_refusal(arguments):
    type_ = builtin.TensorType(builtin.i64, [9])
    block = Block(arg_types=[type_])
    assert extract_slice(block.args[0], type_, arguments) is None
