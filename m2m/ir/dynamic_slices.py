"""Rank-preserving positive-step slices with executable runtime clipping.

Reuse checked, interval-sized shape arithmetic. Bounds come from actual tensor
dimensions and source integer literals; symbolic example hints never participate.
"""

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.ir import Block, Region

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

    # LLVM a47bddcc verifyInBoundsSlice rejects offset == source extent,
    # including empty slices. Keep extraction inside the nonempty branch.
    # Dynamic zero operands also keep statically empty source axes verifiable
    # in the unreachable extraction branch; they grant no runtime access.
    dynamic_offsets = iter(offsets)
    offsets = []
    for dimension, origin in enumerate(static_offsets):
        if origin == builtin.DYNAMIC_INDEX:
            offsets.append(next(dynamic_offsets))
        elif shape[dimension] == 0:
            offsets.append(builder.emit(arith.ConstantOp.from_int_and_width(origin, builtin.IndexType())))
            static_offsets[dimension] = builtin.DYNAMIC_INDEX

    result_shape = list(static_sizes)
    operation = tensor.ExtractSliceOp.build(operands=[source, offsets, sizes, []],
        result_types=[builtin.TensorType(source.type.element_type, result_shape)], properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, static_offsets),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, static_sizes),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, static_strides)})
    return finish_extraction(operation, block, result_type)


def finish_extraction(operation, block, result_type):
    """Keep possibly empty extraction behind exact standard control flow.

    Rank reduction may drop static unit axes; dynamic sizes still enumerate
    precisely the retained result axes. Callers retain original coordinates.
    """
    result_shape = operation.result.type.get_shape()
    if any(size <= 0 for size in result_shape):
        sizes = tuple(operation.sizes)
        dynamic_sizes = iter(sizes)
        builder = Builder(block, lambda *_args: None)
        zero = builder.emit(arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
        predicate = builder.emit(arith.ConstantOp.from_int_and_width(1, builtin.i1))
        for size in result_shape:
            extent = (next(dynamic_sizes) if size == builtin.DYNAMIC_INDEX else
                      builder.emit(arith.ConstantOp.from_int_and_width(size, builtin.IndexType())))
            positive = builder.emit(arith.CmpiOp(extent, zero, "sgt"))
            predicate = builder.emit(arith.AndIOp(predicate, positive))
        empty = tensor.EmptyOp(sizes, operation.result.type)
        operation = scf.IfOp(predicate, [operation.result.type],
            Region(Block([operation, scf.YieldOp(operation.result)])),
            Region(Block([empty, scf.YieldOp(empty.tensor)])))
    operations = list(block.ops)
    for item in operations:
        block.detach_op(item)
    operations.append(operation)
    result = operation.results[0]
    if result.type != result_type:
        cast = tensor.CastOp(result, result_type)
        operations.append(cast)
        result = cast.results[0]
    return operations, result
