"""full_like dimensions come from source SSA, independently of fill contents."""

import pytest
import torch
from xdsl.dialects import builtin, func, tensor

import m2m
from m2m.coverage import opaque_report


@pytest.mark.parametrize("dtype,value", [(torch.int64, -17), (torch.int32, 23),
    (torch.float32, 1.25), (torch.float32, -0.0), (torch.float16, -3.5), (torch.bfloat16, 2.25)])
def test_exported_full_like_retains_runtime_extents_and_actual_typed_fill(dtype, value):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, value)

    inputs = (torch.arange(21).reshape(7, 3).to(dtype),)
    exported = torch.export.export(Model(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=19)}})
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()
    forward = next(op for op in result.module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    splat = next(op for op in forward.walk() if isinstance(op, tensor.SplatOp))
    assert splat.result.type.get_shape() == (builtin.DYNAMIC_INDEX, 3)
    assert len(splat.dynamicSizes) == 1
    dimension = splat.dynamicSizes[0].owner
    assert isinstance(dimension, tensor.DimOp) and dimension.source is forward.body.block.args[0]
    assert not any(isinstance(op, func.CallOp) for op in forward.walk())


def test_full_like_explicit_dtype_keeps_source_dimensions_and_integer_conversion():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, (1 << 53) + (1 << 29) + 1, dtype=torch.float32)

    inputs = (torch.arange(21).reshape(7, 3),)
    exported = torch.export.export(Model(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=19)}})
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()
    splat = next(op for op in result.module.walk() if isinstance(op, tensor.SplatOp))
    assert splat.result.type.element_type == builtin.f32
    assert splat.input.owner.name == "arith.sitofp"
