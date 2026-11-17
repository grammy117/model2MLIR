"""Empty clipped slices must not execute an out-of-bounds extraction."""

from io import StringIO

import pytest
from xdsl.dialects import builtin, func, scf, tensor
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.ir.dynamic_slices import extract_slice


@pytest.mark.parametrize("shape,axis,start,end,step,result_shape", (
    ((-1, 3), 0, 31, None, 2, (-1, 3)),
    ((-1, 3), 0, 0, 0, 1, (0, 3)),
    ((-1, 0), 0, 1, None, 2, (-1, 0)),
    ((0, -1), 1, -3, None, 2, (0, -1)),
    ((-1, -1), 1, 2, None, 3, (-1, -1)),
))
def test_empty_domains_have_an_exact_pure_conditional(shape, axis, start, end, step, result_shape):
    typ = lambda dimensions: builtin.TensorType(builtin.i64, [
        builtin.DYNAMIC_INDEX if size < 0 else size for size in dimensions])
    block = Block(arg_types=[typ(shape)])
    operations, result = extract_slice(block.args[0], typ(result_shape), (axis, start, end, step))
    block.add_ops([*operations, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([typ(shape)], [typ(result_shape)]), Region(block))])
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    _parse_mlir_text_to_xdsl(stream.getvalue()).verify()
    conditional = next(op for op in module.walk() if isinstance(op, scf.IfOp))
    true_ops, false_ops = tuple(conditional.true_region.block.ops), tuple(conditional.false_region.block.ops)
    assert len(true_ops) == len(false_ops) == 2
    extraction, destination = true_ops[0], false_ops[0]
    assert isinstance(extraction, tensor.ExtractSliceOp)
    assert isinstance(destination, tensor.EmptyOp)
    assert destination.tensor.type == extraction.result.type == conditional.results[0].type
    assert tuple(extraction.sizes) == tuple(destination.operands)
    assert isinstance(true_ops[1], scf.YieldOp) and tuple(true_ops[1].operands) == (extraction.result,)
    assert isinstance(false_ops[1], scf.YieldOp) and tuple(false_ops[1].operands) == (destination.tensor,)
