"""Runtime broadcasting in standard Linalg, from actual operand dimensions.

Static singleton projections retain affine input maps. Ambiguous runtime
singletons use explicit element reads inside the parallel output computation.
No symbolic example hint or caller annotation supplies an extent.
"""

from xdsl.dialects import arith, builtin, cf, linalg, tensor
from xdsl.ir.affine import AffineExpr, AffineMap


def binary(inputs, result_type, scalar_build, *, emit_elementwise, cast_scalar):
    shapes = [tuple(value.type.get_shape()) for value in inputs]
    shape = tuple(result_type.get_shape())
    rank = len(shape)
    if len(inputs) != 2 or max(map(len, shapes)) != rank:
        return None
    # Reject incompatible static contracts before creating operand uses.
    for axis, declared in enumerate(shape):
        positions = [axis - (rank - len(item)) for item in shapes]
        left, right = [1 if index < 0 else shapes[i][index] for i, index in enumerate(positions)]
        if left >= 0 and right >= 0:
            if left != right and left != 1 and right != 1:
                return None
            if declared >= 0 and declared != (right if left == 1 else left):
                return None
    ops, sizes, singleton_tests, projections = [], [], {}, [{}, {}]

    def emit(operation):
        ops.append(operation)
        return operation.results[0] if operation.results else None

    def constant(value):
        return emit(arith.ConstantOp.from_int_and_width(value, builtin.IndexType()))

    zero, one = constant(0), constant(1)

    def extent(operand, axis):
        value = shapes[operand][axis]
        return constant(value) if value >= 0 else emit(tensor.DimOp(inputs[operand], constant(axis)))

    for axis, declared in enumerate(shape):
        positions = [axis - (rank - len(item)) for item in shapes]
        dimensions = [1 if index < 0 else shapes[i][index] for i, index in enumerate(positions)]
        left, right = dimensions
        if left >= 0 and right >= 0:
            chosen = right if left == 1 else left
            size = constant(chosen)
        elif left == 1 or right == 1:
            source = 1 if left == 1 else 0
            size = extent(source, positions[source])
        elif inputs[0] is inputs[1] and positions[0] == positions[1]:
            size = extent(0, positions[0])
        else:
            a, b = (extent(i, positions[i]) for i in range(2))
            a_one, b_one = (emit(arith.CmpiOp(value, one, "eq")) for value in (a, b))
            equal = emit(arith.CmpiOp(a, b, "eq"))
            valid = emit(arith.OrIOp(equal, emit(arith.OrIOp(a_one, b_one))))
            emit(cf.AssertOp(valid, "dynamic broadcast: incompatible operand dimensions"))
            size = emit(arith.SelectOp(a_one, b, a))
            for i, test in enumerate((a_one, b_one)):
                if dimensions[i] < 0:
                    singleton_tests[i, positions[i]] = test
        if declared < 0:
            sizes.append(size)
        elif left < 0 or right < 0:
            valid = emit(arith.CmpiOp(size, constant(declared), "eq"))
            emit(cf.AssertOp(valid, "dynamic broadcast: result differs from its declared static extent"))
        for i, index in enumerate(positions):
            if index >= 0:
                projections[i][index] = AffineExpr.constant(0) if dimensions[i] == 1 else AffineExpr.dimension(axis)

    if not singleton_tests:
        maps = [AffineMap(rank, 0, tuple(projection[i] for i in range(len(shapes[n]))))
                for n, projection in enumerate(projections)]
        result = emit_elementwise(inputs, result_type, scalar_build, input_maps=maps,
                                  promote=True, dynamic_sizes=sizes)
    else:
        def body(_arguments, out_elem):
            body_ops, values, indices = [], [], {}
            for i, value in enumerate(inputs):
                coordinates = []
                for axis, dimension in enumerate(shapes[i]):
                    output_axis = rank - len(shapes[i]) + axis
                    if dimension == 1:
                        index = zero
                    else:
                        if output_axis not in indices:
                            operation = linalg.IndexOp(output_axis)
                            body_ops.append(operation)
                            indices[output_axis] = operation.result
                        index = indices[output_axis]
                        if (i, axis) in singleton_tests:
                            operation = arith.SelectOp(singleton_tests[i, axis], zero, index)
                            body_ops.append(operation)
                            index = operation.result
                    coordinates.append(index)
                extract = tensor.ExtractOp(value, coordinates, value.type.element_type)
                body_ops.append(extract)
                converted, scalar = cast_scalar(extract.result, out_elem)
                body_ops.extend(converted)
                values.append(scalar)
            calculation, answer = scalar_build(values, out_elem)
            return [*body_ops, *calculation], answer

        result = emit_elementwise([], result_type, body, input_maps=[], dynamic_sizes=sizes)
    if result is None:
        return None
    operations, value = result
    return [*ops, *operations], value
