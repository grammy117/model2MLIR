"""Checked literal-axis selection preserving exact rank and retained unit axes.

LLVM a47bddccec30255619bb8c37fa59700e661d4e66 TensorOps.cpp getDroppedDims,
and tensor/canonicalize.mlir's rank_reducing_slice_preserve_dropped_dims define
the reduction map. Explicit mixed sizes avoid ambiguous adjacent static ones.
"""

from xdsl.dialects import arith, builtin, tensor
from xdsl.ir import Block

from .dynamic_slices import finish_extraction
from .shape_domains import SIZE_MAX, Builder
from .standard_arith import IndexCastUIOp


def select(source, result_type, axis, index):
    typ = source.type
    if (not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr)
            or not isinstance(result_type, builtin.TensorType) or typ.element_type != result_type.element_type
            or not isinstance(result_type.encoding, builtin.NoneAttr)):
        return None
    shape = typ.get_shape()
    if (type(axis) is not int or not -len(shape) <= axis < len(shape)
            or type(index) is not int or not -SIZE_MAX - 1 <= index <= SIZE_MAX):
        return None
    axis %= len(shape)
    expected = [size for i, size in enumerate(shape) if i != axis]
    if list(result_type.get_shape()) != expected:
        return None
    bound = shape[axis]
    if bound >= 0 and not -bound <= index < bound:
        return None
    block = Block()
    builder = Builder(block, lambda *_args: None)
    offset = index + bound if index < 0 and bound >= 0 else index
    if bound < 0:
        extent = builder.extent(source, axis)
        number = builder.constant(index)
        selected = builder.binary("add", extent, number) if index < 0 else number
        builder.check(selected, builder.constant(0), "sge", "selection index is nonnegative")
        builder.check(selected, extent, "slt", "selection index is below selected dimension")
        # Both executable comparisons precede transport to the target index.
        # This is a nonnegative offset smaller than an actual tensor extent.
        offset = builder.emit(IndexCastUIOp(selected.value, builtin.IndexType()))
    offsets, sizes, static_offsets, static_sizes, reduced = [], [], [], [], []
    for i, size in enumerate(shape):
        origin = offset if i == axis else 0
        if i == axis and bound < 0:
            offsets.append(origin)
            static_offsets.append(builtin.DYNAMIC_INDEX)
        elif size == 0:
            # The extraction branch is unreachable for this empty source axis.
            offsets.append(builder.emit(arith.ConstantOp.from_int_and_width(0, builtin.IndexType())))
            static_offsets.append(builtin.DYNAMIC_INDEX)
        else:
            static_offsets.append(origin)
        if i == axis:
            static_sizes.append(1)
        elif size == 1:
            # Keep retained unit axes distinct from the one selected axis.
            # Widen only their type; their actual mixed size remains exactly 1.
            static_sizes.append(builtin.DYNAMIC_INDEX)
            sizes.append(builder.emit(arith.ConstantOp.from_int_and_width(1, builtin.IndexType())))
            reduced.append(builtin.DYNAMIC_INDEX)
        elif size < 0:
            position = builder.emit(arith.ConstantOp.from_int_and_width(i, builtin.IndexType()))
            sizes.append(builder.emit(tensor.DimOp(source, position)))
            static_sizes.append(builtin.DYNAMIC_INDEX)
            reduced.append(builtin.DYNAMIC_INDEX)
        else:
            static_sizes.append(size)
            reduced.append(size)
    operation = tensor.ExtractSliceOp.build(operands=[source, offsets, sizes, []],
        result_types=[builtin.TensorType(typ.element_type, reduced)], properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, static_offsets),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, static_sizes),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1] * len(shape))})
    return finish_extraction(operation, block, result_type)
