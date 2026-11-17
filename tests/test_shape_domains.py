"""Actual exported input domains are executable guards, not example facts."""

import pytest
import torch
from xdsl.dialects.cf import AssertOp
from xdsl.dialects.func import FuncOp

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.coverage import opaque_report
from m2m.ir.standard_arith import IndexCastUIOp


@pytest.mark.parametrize("mode", ["shared", "affine", "inverted", "zero", "unbounded"])
def test_exported_shared_and_derived_dimensions_stay_executable(mode):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x.sum(dim=0) + y.sum(dim=0)

    minimum = 0 if mode == "zero" else 2
    symbol = torch.export.Dim("n", min=minimum, **({} if mode == "unbounded" else {"max": 31}))
    a, b = (27 if mode == "inverted" else 13), (13 if mode == "shared" else 27)
    values = (torch.randn(a, 3), torch.randn(b, 3))
    program = torch.export.export(Model(), values, dynamic_shapes={
        "x": {0: 2 * symbol + 1 if mode == "inverted" else symbol},
        "y": {0: symbol if mode == "shared" else 2 * symbol + 1}})
    placeholder = next(node for node in program.graph.nodes if node.op == "placeholder")
    environment = placeholder.meta["val"].shape[0].node.shape_env
    original_guards = list(environment.guards)
    result = m2m.convert(program, values, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert opaque_report(result.mlir_text) == {} and result.capture_trace["status"] == "complete"
    assert environment.guards == original_guards
    operations = list(result.module.walk())
    assert sum(isinstance(op, IndexCastUIOp) for op in operations) == 4
    guards = [op.msg.data for op in operations if isinstance(op, AssertOp)]
    assert all(message.startswith("exported shape domain:") for message in guards)
    assert any(f"sge {minimum}" in message for message in guards)
    assert any("sle 31" in message for message in guards) == (mode != "unbounded")
    assert sum(op.name == "linalg.reduce" for op in operations) == 2
    entry = next(op for op in result.module.ops if isinstance(op, FuncOp))
    assert "prov.shape_domain" in entry.attributes
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
