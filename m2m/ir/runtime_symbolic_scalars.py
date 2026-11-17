"""Materialize actual singleton integer reads used by exported scalar guards.

Tensor storage is executable input data. Neither an unbacked SymInt hint nor
its range metadata supplies the scalar value or an unchecked size conversion.
"""

import torch
from xdsl.dialects import arith, builtin, tensor


def emit_integer_item(node, value_map, block, stamp):
    if node.target not in (torch.ops.aten.item.default, torch.ops.aten._local_scalar_dense.default):
        return None
    if len(node.args) != 1:
        return None
    source = value_map.get(getattr(node.args[0], "name", ""))
    if (source is None or not isinstance(source.type, builtin.TensorType)
            or not isinstance(source.type.encoding, builtin.NoneAttr)
            or any(size != 1 for size in source.type.get_shape())
            or not isinstance(source.type.element_type, builtin.IntegerType)
            or source.type.element_type.bitwidth not in (8, 16, 32, 64)):
        return None
    metadata = node.args[0].meta.get("val")
    dtype = getattr(metadata, "dtype", None)
    if dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        return None
    operations = [arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                  for _ in source.type.get_shape()]
    extract = tensor.ExtractOp.build(operands=[source, [op.result for op in operations]],
                                    result_types=[source.type.element_type])
    operations.append(extract)
    result = extract.result
    if source.type.element_type.bitwidth < 64:
        cast = (arith.ExtUIOp if dtype == torch.uint8 else arith.ExtSIOp)(result, builtin.i64)
        operations.append(cast)
        result = cast.result
    for operation in operations:
        stamp(operation, node.meta, "runtime_integer")
        block.add_op(operation)
    return result
