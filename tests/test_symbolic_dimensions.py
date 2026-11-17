"""Backed exported dimensions must not become example-size constants."""

import json

import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType, i32
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.tensor import DimOp

import m2m
from m2m.ir.decompositions import _coerce_static_dim as decomposition_dimension
from m2m.ir.decompositions import _static_shape
from m2m.ir.import_fx import _coerce_static_dim as importer_dimension
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


@torch.library.custom_op("m2m_symbolic_test::copy", mutates_args=())
def copy_kernel(x: torch.Tensor, n: int) -> torch.Tensor:
    assert x.shape[0] == n
    return x.clone()


@copy_kernel.register_fake
def fake_copy(x, n):
    return torch.empty_like(x)


class Model(torch.nn.Module):
    def forward(self, x):
        first = copy_kernel(x, x.shape[0])
        second = copy_kernel(first, first.shape[0])
        return second, first, x


def exported():
    inputs = (torch.arange(39, dtype=torch.int32).reshape(13, 3),)
    program = torch.export.export(Model(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("n", min=2, max=31)}})
    return program, inputs


@pytest.mark.parametrize("helper,expected", [
    (importer_dimension, DYNAMIC_INDEX),
    (decomposition_dimension, DYNAMIC_INDEX),
    (lambda dim: _static_shape([dim])[0], -1),
])
def test_backed_symbolic_extent_never_uses_its_example_hint(helper, expected):
    program, _ = exported()
    placeholder = next(node for node in program.graph.nodes if node.op == "placeholder")
    dimension = placeholder.meta["val"].shape[0]
    assert isinstance(dimension, torch.SymInt)
    environment = dimension.node.shape_env
    guards = list(environment.guards)
    assert helper(dimension) == expected
    assert environment.guards == guards


def test_registered_kernel_calls_keep_symbolic_public_and_dependent_types(monkeypatch, tmp_path):
    from m2m.capture.bundle import write_bundle
    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_symbolic_test::copy", KernelSpec("m2m_symbolic_test::copy"))
    program, inputs = exported()
    result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True,
                         weights_path=str(tmp_path / "weights.safetensors"))
    assert result.ok, result.diagnostics
    result.module.verify()
    expected = TensorType(i32, [DYNAMIC_INDEX, 3])
    forward = next(op for op in result.module.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")
    assert list(forward.function_type.inputs) == [expected]
    assert list(forward.function_type.outputs) == [expected] * 3
    calls = [op for op in result.module.walk() if isinstance(op, CallOp)]
    assert len(calls) == 2
    for call in calls:
        assert call.arguments[0].type == call.results[0].type == expected
        assert isinstance(call.arguments[1].owner, DimOp)
        assert call.arguments[1].owner.source is forward.body.block.args[0]
    # Export shares the original size SSA while retaining dependent tensor data.
    assert calls[1].arguments[0] is calls[0].results[0]
    assert calls[1].arguments[1] is calls[0].arguments[1]
    assert result.capture_trace["status"] == "complete"
    assert result.capture_trace["graphs"]["original"]["range_constraints"] == {
        str(key): str(value) for key, value in program.range_constraints.items()}
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
    write_bundle(Model().eval(), inputs, tmp_path, capture_trace=True,
                 capture_regions=False, conversion_result=result, source_path=__file__)
    saved = json.loads((tmp_path / "frontend-trace.json").read_text())
    assert saved["graphs"]["original"]["range_constraints"] == result.capture_trace["graphs"]["original"]["range_constraints"]
    with np.load(tmp_path / "goldens.npz", allow_pickle=False) as goldens:
        assert goldens.files == ["out0", "out1", "out2"]
        assert all(np.array_equal(goldens[name], inputs[0].numpy()) for name in goldens.files)
    # The exported reference retains its declared shape domain after conversion.
    for count in (2, 7, 13, 31):
        x = torch.arange(count * 3, dtype=torch.int32).reshape(count, 3)
        assert all(torch.equal(value, x) for value in program.module()(x))


@pytest.mark.parametrize("value", [0, 1, 13, 31])
def test_concrete_dimensions_remain_exact(value):
    assert importer_dimension(value) == decomposition_dimension(value) == value
    assert _static_shape([value, 3]) == [value, 3]


@pytest.mark.parametrize("number", [0, 1, 7, 31])
def test_literal_symints_remain_concrete_without_new_guards(number):
    import sympy
    from torch.fx.experimental.sym_node import SymNode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    environment = ShapeEnv()
    value = torch.SymInt(SymNode(sympy.Integer(number), environment, int, number))
    assert importer_dimension(value) == decomposition_dimension(value) == number
    assert _static_shape([value]) == [number]
    assert environment.guards == []


@torch.library.custom_op("m2m_symbolic_test::control", mutates_args=())
def control_kernel(x: torch.Tensor, n: int) -> torch.Tensor:
    return x + n


@control_kernel.register_fake
def fake_control(x, n):
    return torch.empty_like(x)


@pytest.mark.parametrize("kind,opcode", [("add", "arith.addi"), ("sub", "arith.subi"),
                                       ("mul", "arith.muli"), ("floor", "arith.floordivsi")])
def test_exported_symbolic_arithmetic_keeps_actual_scalar_ssa_and_checked_transport(monkeypatch, kind, opcode):
    import operator
    from xdsl.dialects.cf import AssertOp
    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
    from m2m.coverage import opaque_report
    from m2m.ir.standard_arith import IndexCastUIOp

    arithmetic = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "floor": operator.floordiv}[kind]

    class Controlled(torch.nn.Module):
        def forward(self, x):
            return control_kernel(x, arithmetic(x.shape[0], 2))

    inputs = (torch.arange(13, dtype=torch.int32),)
    program = torch.export.export(Controlled(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("n", min=4, max=31)}})
    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_symbolic_test::control", KernelSpec("m2m_symbolic_test::control"))
    shape = next(node for node in program.graph.nodes if node.op == "placeholder").meta["val"].shape[0]
    guards = list(shape.node.shape_env.guards)
    result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert opaque_report(result.mlir_text) == {} and shape.node.shape_env.guards == guards
    call = next(op for op in result.module.walk() if isinstance(op, CallOp))
    cast = call.arguments[1].owner
    assert isinstance(cast, IndexCastUIOp) and cast.operands[0].owner.name == opcode
    assert any(isinstance(op, AssertOp) and op.msg.data == "symbolic size arithmetic fits logical index storage"
               for op in result.module.walk())
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()


def test_negative_symbolic_control_is_not_silently_transported_as_a_size(monkeypatch):
    class Negative(torch.nn.Module):
        def forward(self, x):
            return control_kernel(x, 2 - x.shape[0])

    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_symbolic_test::control", KernelSpec("m2m_symbolic_test::control"))
    inputs = (torch.arange(13, dtype=torch.int32),)
    program = torch.export.export(Negative(), inputs,
        dynamic_shapes={"x": {0: torch.export.Dim("n", min=4, max=31)}})
    result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True)
    assert not result.ok
