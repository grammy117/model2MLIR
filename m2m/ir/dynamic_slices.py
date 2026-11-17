"""Rank-preserving positive-step slices with executable runtime clipping.

Reuse checked, interval-sized shape arithmetic. Bounds come from actual tensor
dimensions and source integer literals; symbolic example hints never participate.
"""

from xdsl.dialects import arith, builtin, tensor
from xdsl.ir import Block

from .shape_domains import SIZE_MAX, Builder, Integer
from .standard_arith import IndexCastUIOp


def extract_slice(source, result_type, arguments):
    axis, start, end, step = arguments
    step = 1 if step is None else step
    shape = tuple(source.type.get_shape())
    declared = tuple(result_type.get_shape())
    if (type(axis) is not int or not -len(shape) <= axis < len(shape)
            or len(shape) != len(declared) or type(step) is not int or not 0 < step <= SIZE_MAX
            or any(value is not None and (type(value) is not int or not -SIZE_MAX-1 <= value <= SIZE_MAX)
                   for value in (start, end))):
        return None
    axis %= len(shape)
    start = 0 if start is None else start
    for dimension, (actual, wanted) in enumerate(zip(shape, declared, strict=True)):
        if dimension != axis and wanted >= 0 and actual != wanted:
            return None
    if shape[axis] >= 0:
        begin, finish, stride = slice(start, end, step).indices(shape[axis])
        if declared[axis] >= 0 and declared[axis] != len(range(begin, finish, stride)):
            return None
    elif start == end == 0 and declared[axis] not in (0, builtin.DYNAMIC_INDEX):
        return None
    block = Block()
    builder = Builder(block, lambda *_args: None)
    static_offsets, static_sizes, static_strides = [0] * len(shape), list(shape), [1] * len(shape)
    offsets, sizes = [], []
    static_strides[axis] = step

    def select(lhs, rhs, predicate, lower, upper):
        left, right = builder.align(lhs, rhs)
        condition = builder.emit(arith.CmpiOp(left, right, predicate))
        return Integer(builder.emit(arith.SelectOp(condition, left, right)), lower, upper)

    def clipped(bound, extent):
        if bound is None:
            return extent
        literal = builder.constant(bound)
        if bound >= 0:
            return select(literal, extent, "sle", 0, min(bound, SIZE_MAX))
        shifted = builder.binary("add", extent, literal)
        return select(shifted, builder.constant(0), "sge", 0, max(0, SIZE_MAX + bound))

    for dimension, size in enumerate(shape):
        if dimension != axis:
            if size < 0:
                index = builder.emit(arith.ConstantOp.from_int_and_width(dimension, builtin.IndexType()))
                sizes.append(builder.emit(tensor.DimOp(source, index)))
            continue
        if size >= 0:
            begin, finish, stride = slice(start, end, step).indices(size)
            count = len(range(begin, finish, stride))
            static_offsets[dimension], static_sizes[dimension] = begin, count
            continue
        if start == 0 and end == 0:
            static_sizes[dimension] = 0
            continue
        extent = builder.extent(source, dimension)
        begin, finish = clipped(start, extent), clipped(end, extent)
        length = builder.binary("sub", finish, begin)
        length = select(length, builder.constant(0), "sge", 0, SIZE_MAX)
        # Both clipped endpoints lie in [0, extent], so the nonnegative
        # difference fits the checked logical i64 size domain. Narrow only
        # after that proof. Quotient/remainder avoids length+step-1 overflow
        # and an unnecessary wide integer division runtime dependency.
        bounded = length.value
        if bounded.type.bitwidth > 64:
            bounded = builder.emit(arith.TruncIOp(bounded, builtin.i64))
        elif bounded.type.bitwidth < 64:
            bounded = builder.emit(arith.ExtUIOp(bounded, builtin.i64))
        stride = builder.emit(arith.ConstantOp.from_int_and_width(step, builtin.i64))
        quotient = builder.emit(arith.DivUIOp(bounded, stride))
        remainder = builder.emit(arith.RemUIOp(bounded, stride))
        zero = builder.emit(arith.ConstantOp.from_int_and_width(0, builtin.i64))
        nonzero = builder.emit(arith.CmpiOp(remainder, zero, "ne"))
        increment = builder.emit(arith.ExtUIOp(nonzero, builtin.i64))
        count = Integer(builder.emit(arith.AddiOp(quotient, increment)), 0, SIZE_MAX)
        if declared[dimension] >= 0:
            builder.check(count, builder.constant(declared[dimension]), "eq", "slice result matches its static type")
            static_sizes[dimension] = declared[dimension]
        else:
            static_sizes[dimension] = builtin.DYNAMIC_INDEX
            sizes.append(builder.emit(IndexCastUIOp(count.value, builtin.IndexType())))
        if start != 0:
            static_offsets[dimension] = builtin.DYNAMIC_INDEX
            offsets.append(builder.emit(IndexCastUIOp(begin.value, builtin.IndexType())))

    result_shape = list(static_sizes)
    operation = tensor.ExtractSliceOp.build(operands=[source, offsets, sizes, []],
        result_types=[builtin.TensorType(source.type.element_type, result_shape)], properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, static_offsets),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, static_sizes),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, static_strides)})
    operations = list(block.ops)
    for item in operations:
        block.detach_op(item)
    operations.append(operation)
    result = operation.result
    if result.type != result_type:
        cast = tensor.CastOp(result, result_type)
        operations.append(cast)
        result = cast.results[0]
    return operations, result
