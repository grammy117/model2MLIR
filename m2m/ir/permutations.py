"""Word-preserving permutations sized from actual source tensor dimensions.

LLVM a47bddccec30255619bb8c37fa59700e661d4e66 LinalgStructuredOps.td's
TransposeOp and mlir/test/Dialect/Linalg/roundtrip.mlir define the static
permutation and dynamic destination contract. Shape hints grant no dimensions.
"""

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    IndexType,
    NoneAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg.ops import TransposeOp


def lower(source, dimensions, declared_shape):
    typ = source.type
    if (
        not isinstance(typ, TensorType)
        or not isinstance(typ.encoding, NoneAttr)
        or not isinstance(dimensions, (tuple, list))
        or len(dimensions) != len(typ.get_shape())
        or any(type(axis) is not int for axis in dimensions)
    ):
        return None
    rank = len(dimensions)
    if any(not -rank <= axis < rank for axis in dimensions):
        return None
    permutation = [axis % rank for axis in dimensions]
    if sorted(permutation) != list(range(rank)):
        return None
    shape = [typ.get_shape()[axis] for axis in permutation]
    if shape != [DYNAMIC_INDEX if size < 0 else size for size in declared_shape]:
        return None
    result_type = TensorType(typ.element_type, shape)
    operations, dynamic = [], []
    for axis, size in zip(permutation, shape, strict=True):
        if size == DYNAMIC_INDEX:
            index = arith.ConstantOp.from_int_and_width(axis, IndexType())
            dimension = tensor.DimOp(source, index.result)
            operations.extend((index, dimension))
            dynamic.append(dimension.result)
    destination = tensor.EmptyOp(dynamic, result_type)
    result = TransposeOp(
        input=source,
        init=destination.tensor,
        permutation=DenseArrayBase.from_list(i64, permutation),
        result=result_type,
    )
    operations.extend((destination, result))
    return operations, result.results[0]
