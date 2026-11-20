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


def emit_dense_stride(node, value_map, block, stamp):
    """Read a proved source dense-layout stride as checked dimension products.

    Tensor values have no physical stride query. This represents the captured
    source's logical storage coordinates; consumers still need an explicit
    storage/view binding before interpreting those coordinates as addresses.
    """
    from xdsl.dialects import cf

    from .shape_domains import SIZE_MAX, Builder
    from .standard_arith import IndexCastUIOp
    from .storage_layouts import dense_axes

    if node.target != torch.ops.aten.sym_stride.int or len(node.args) != 2 or node.kwargs:
        return None
    original, axis = node.args
    source = value_map.get(getattr(original, "name", ""))
    if (source is None or not isinstance(source.type, builtin.TensorType)
            or not isinstance(source.type.encoding, builtin.NoneAttr)):
        return None
    shape = source.type.get_shape()
    if type(axis) is not int or not -len(shape) <= axis < len(shape):
        return None
    axes = dense_axes(original.meta.get("val"), len(shape))
    axis %= len(shape)
    temporary = type(block)()
    builder = Builder(temporary, stamp)
    builder.metadata = node.meta
    stride = builder.constant(1)
    for minor in axes[axes.index(axis) + 1:]:
        size = builder.extent(source, minor) if shape[minor] < 0 else builder.constant(shape[minor])
        stride = builder.binary("mul", stride, size)
    if stride.upper > SIZE_MAX:
        builder.check(stride, builder.constant(SIZE_MAX), "sle", "source dense stride fits logical i64")
    result = builder.emit(IndexCastUIOp(stride.value, builtin.IndexType()))
    restored = builder.emit(IndexCastUIOp(result, stride.value.type))
    equal = builder.emit(arith.CmpiOp(restored, stride.value, "eq"))
    builder.emit(cf.AssertOp(equal, "source dense stride fits logical index storage"))
    for operation in tuple(temporary.ops):
        temporary.detach_op(operation)
        block.add_op(operation)
    return result
