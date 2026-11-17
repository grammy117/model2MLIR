"""Unbacked integer scalars retain real reads and executable signed guards."""

import pytest
import torch
from xdsl.dialects import arith, cf, func, tensor

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.coverage import opaque_report
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


@torch.library.custom_op("m2m_item_test::allocate", mutates_args=())
def allocate(x: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros((n,), dtype=x.dtype)


@allocate.register_fake
def fake_allocate(x, n):
    return torch.empty((n,), dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("dtype", (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("shape", ((), (1, 1)))
def test_integer_item_is_source_storage_not_a_symbolic_hint(monkeypatch, dtype, shape):
    class Model(torch.nn.Module):
        def forward(self, x, count):
            n = count.item()
            torch._check_is_size(n)
            torch._check(n <= 31)
            first = allocate(x, n)
            second = allocate(first, n)
            return second, first, x, count

    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_item_test::allocate", KernelSpec("m2m_item_test::allocate"))
    inputs = (torch.arange(13, dtype=torch.int32), torch.full(shape, 7, dtype=dtype))
    program = torch.export.export(Model(), inputs)
    result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert opaque_report(result.mlir_text) == {} and result.capture_trace["status"] == "complete"
    entry = next(op for op in result.module.ops if isinstance(op, func.FuncOp))
    calls = [op for op in entry.walk() if isinstance(op, func.CallOp)]
    assert len(calls) == 2 and calls[1].operands[0] is calls[0].results[0]
    value = calls[0].operands[1]
    assert value is calls[1].operands[1] and str(value.type) == "i64"
    if dtype != torch.int64:
        assert isinstance(value.owner, arith.ExtUIOp if dtype == torch.uint8 else arith.ExtSIOp)
        value = value.owner.operands[0]
    assert isinstance(value.owner, tensor.ExtractOp) and value.owner.operands[0] is entry.body.block.args[1]
    guards = [op for op in entry.walk() if isinstance(op, cf.AssertOp)]
    assert len(guards) >= 3
    assert all(isinstance(op.operands[0].owner, arith.CmpiOp) for op in guards)
    _parse_mlir_text_to_xdsl(result.mlir_text).verify()
