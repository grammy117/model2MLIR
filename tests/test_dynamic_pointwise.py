"""Actual exported pointwise models retain dynamic extents and standard bodies."""

import pytest
import torch
from xdsl.dialects import builtin, func, tensor

import m2m
from m2m.coverage import opaque_report


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("mode", ["scalar", "row", "shared"])
def test_exported_binary_models_keep_source_dynamic_axes_and_runtime_arithmetic(operation, mode):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return getattr(torch, operation)(x, y)

    dimension = torch.export.Dim("rows", min=2, max=31)
    inputs = (torch.arange(39, dtype=torch.float32).reshape(13, 3),
              torch.tensor(1.25) if mode == "scalar" else torch.full((1 if mode == "row" else 13, 3), 1.25))
    shapes = {"x": {0: dimension}, "y": {0: dimension} if mode == "shared" else {}}
    exported = torch.export.export(Model(), inputs, dynamic_shapes=shapes)
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()
    function = next(op for op in result.module.ops if isinstance(op, func.FuncOp))
    assert function.function_type.outputs.data[0] == builtin.TensorType(builtin.f32, [builtin.DYNAMIC_INDEX, 3])
    assert any(isinstance(op, tensor.DimOp) for op in function.walk())
    assert any(op.name == "linalg.generic" for op in function.walk())
    assert not any(isinstance(op, func.CallOp) for op in function.walk())
    # Reusing the exported source remains legal at independent sizes; conversion
    # must not have inserted a guard specializing the example length 13.
    for length in (2, 7, 31):
        x = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3)
        y = torch.full_like(x, 1.25) if mode == "shared" else inputs[1]
        torch.testing.assert_close(exported.module()(x, y), getattr(torch, operation)(x, y))


@pytest.mark.parametrize("bounds", [(0, 0, 1), (1, None, 2), (-8, -1, 3), (None, 5, 1)])
def test_exported_slice_with_dynamic_source_has_clipped_standard_tensor_body(bounds):
    start, end, step = bounds

    class Model(torch.nn.Module):
        def forward(self, x):
            return x[slice(start, end, step)]

    inputs = (torch.arange(39, dtype=torch.int32).reshape(13, 3),)
    exported = torch.export.export(Model(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("rows", min=9, max=31)}})
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()
    assert any(isinstance(op, tensor.ExtractSliceOp) for op in result.module.walk())
    assert not any(isinstance(op, func.CallOp) for op in result.module.walk())
    for length in (9, 13, 31):
        x = torch.arange(length * 3, dtype=torch.int32).reshape(length, 3)
        assert torch.equal(exported.module()(x), x[slice(start, end, step)])


@pytest.mark.parametrize("arguments", [(0, 0, None, 0), (0, 0, None, -1), (1, 0, None, 1),
                                      (0, 0, None, 1 << 63), (0, object(), None, 1)])
def test_unknown_or_invalid_slice_arguments_do_not_invent_runtime_bounds(arguments):
    from xdsl.ir import Block

    from m2m.ir.dynamic_slices import extract_slice

    type_ = builtin.TensorType(builtin.i32, [builtin.DYNAMIC_INDEX])
    source = Block(arg_types=[type_]).args[0]
    before = tuple(source.uses)
    assert extract_slice(source, type_, arguments) is None
    assert tuple(source.uses) == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_dynamic_half_python_scalar_preserves_existing_opmath_casts(dtype, operation):
    class Model(torch.nn.Module):
        def forward(self, x):
            return getattr(torch, operation)(x, 1.234567)

    inputs = (torch.arange(13, dtype=dtype),)
    exported = torch.export.export(Model(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=31)}})
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()
    forward = next(op for op in result.module.ops if isinstance(op, func.FuncOp))
    element = builtin.f16 if dtype == torch.float16 else builtin.bf16
    assert forward.function_type.outputs.data[0] == builtin.TensorType(element, [builtin.DYNAMIC_INDEX])
    assert any(op.name == "arith.truncf" for op in forward.walk()) == (operation in ("mul", "div"))
