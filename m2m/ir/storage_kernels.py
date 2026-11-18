"""Expose complete cloned storage roots and actual mutable-view coordinates.

Torch 2.10 regenerates SliceViewInfo/AsStridedViewInfo/AliasViewInfo from one
clone per base. Keep that grouping: separate extract/update/scatter calls would
lose writes observed through a second alias inside the original kernel.
"""

from xdsl.dialects import arith, builtin
from xdsl.ir import Block

from .kernel_registry import KernelLoweringError, _emit_bound_call, _is_fx_node
from .shape_domains import SIZE_MAX, Builder, Integer
from .standard_arith import IndexCastUIOp

PROFILE = "m2m.functional-kernel-storage.v1"


class Geometry(Builder):
    def __init__(self, values):
        super().__init__(Block(), lambda *_args: None)
        self.values, self.fields = values, []

    def integer(self, value):
        if type(value) is int:
            if not 0 <= value <= SIZE_MAX:
                raise KernelLoweringError("mutable-view coordinates require nonnegative logical i64 integers")
            return self.constant(value)
        actual = self.values.get(value.name) if _is_fx_node(value) else None
        if actual is not None and isinstance(actual.type, builtin.IndexType):
            index = actual
            actual = self.emit(IndexCastUIOp(index, builtin.i64))
            restored = self.emit(IndexCastUIOp(actual, builtin.IndexType()))
            equal = self.emit(arith.CmpiOp(index, restored, "eq"))
            from xdsl.dialects import cf
            self.emit(cf.AssertOp(equal, "mutable-view coordinate fits logical i64 storage"))
        if actual is None or not isinstance(actual.type, builtin.IntegerType) or actual.type.bitwidth > 64:
            raise KernelLoweringError("mutable-view coordinates require actual imported integer SSA: "
                                     f"{getattr(value, 'name', type(value).__name__)} -> {getattr(actual, 'type', None)}")
        width = actual.type.bitwidth
        result = Integer(actual, -(1 << (width-1)), (1 << (width-1))-1)
        self.check(result, self.constant(0), "sge", "mutable-view coordinate is nonnegative")
        return Integer(actual, 0, result.upper)

    def i64(self, value):
        if value.lower < 0:
            self.check(value, self.constant(0), "sge", "mutable-view coordinate is nonnegative")
        if value.upper > SIZE_MAX:
            self.check(value, self.constant(SIZE_MAX), "sle", "mutable-view coordinate fits logical i64")
        width = value.value.type.bitwidth
        if width == 64:
            return value.value
        return self.emit((arith.TruncIOp if width > 64 else arith.ExtSIOp)(value.value, builtin.i64))

    def field(self, name, value):
        self.fields.append((name, self.i64(value)))
        return builtin.StringAttr(name)

    def dense(self, node, index):
        from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

        source, metadata = self.values[node.name], node.meta.get("val")
        if metadata is None or len(source.type.get_shape()) > 8:
            raise KernelLoweringError("mutable storage requires bounded original tensor metadata")
        product = 1
        if not statically_known_true(sym_eq(metadata.storage_offset(), 0)):
            raise KernelLoweringError("mutable storage base must have a proved zero origin")
        for size, stride in reversed(list(zip(metadata.shape, metadata.stride(), strict=True))):
            if not statically_known_true(sym_eq(stride, product)):
                raise KernelLoweringError("mutable storage base needs its proved dense source layout")
            product *= size
        dimensions = [self.extent(source, axis) if size < 0 else self.constant(size)
                      for axis, size in enumerate(source.type.get_shape())]
        strides, stride = [], self.constant(1)
        for size in reversed(dimensions):
            strides.insert(0, stride)
            stride = self.binary("mul", stride, size)
        names = [self.field(f"@base{index}.size{axis}", size) for axis, size in enumerate(dimensions)]
        return dimensions, strides, names

    def view(self, kwargs, argument, base, dimensions, strides, index):
        prefix = "_" + argument
        alias = kwargs.pop(prefix + "_alias", None)
        if alias is not None and type(alias) is not bool:
            raise KernelLoweringError("mutable storage alias marker must be an actual boolean")
        kind, offset = "alias" if alias is not None else "base", self.constant(0)
        sizes, steps = dimensions, strides
        if alias is None and prefix + "_storage_offset" in kwargs:
            kind = "as_strided"
            sizes, steps = (kwargs.pop(prefix + name, None) for name in ("_size", "_stride"))
            if not isinstance(sizes, (tuple, list)) or not isinstance(steps, (tuple, list)) or len(sizes) != len(steps) or len(sizes) > 8:
                raise KernelLoweringError("mutable storage view needs complete bounded size/stride fields")
            sizes, steps = [self.integer(value) for value in sizes], [self.integer(value) for value in steps]
            offset = self.integer(kwargs.pop(prefix + "_storage_offset"))
        elif alias is None and prefix + "_slice_dim" in kwargs:
            kind = "slice"
            axis = kwargs.pop(prefix + "_slice_dim")
            if type(axis) is not int or not 0 <= axis < len(dimensions):
                raise KernelLoweringError("mutable storage slice needs an actual source axis")
            start, end = (self.integer(kwargs.pop(prefix + name, None)) for name in ("_slice_start", "_slice_end"))
            self.check(start, end, "sle", "mutable storage slice has ordered endpoints")
            self.check(end, dimensions[axis], "sle", "mutable storage slice is within its base")
            sizes = list(dimensions)
            sizes[axis] = self.binary("sub", end, start)
            offset = self.binary("mul", start, strides[axis])
        sizes = [self.field(f"@view{index}.size{axis}", value) for axis, value in enumerate(sizes)]
        steps = [self.field(f"@view{index}.stride{axis}", value) for axis, value in enumerate(steps)]
        return builtin.DictionaryAttr({"argument": builtin.StringAttr(argument), "base_index": builtin.IntegerAttr(base, builtin.i64),
            "kind": builtin.StringAttr(kind), "shape": builtin.ArrayAttr(sizes), "strides": builtin.ArrayAttr(steps),
            "offset": self.field(f"@view{index}.offset", offset)})


def emit(spec, node, value_map, bases, mutated, ordinary, types):
    geometry = Geometry(value_map)
    layouts = [geometry.dense(base, index) for index, base in enumerate(bases)]
    kwargs, views, grouped = dict(node.kwargs), [], [[] for _ in bases]
    kwargs.pop("_all_bases")
    for index, arg in enumerate(mutated):
        base = kwargs.pop("_" + arg.name + "_base_index", None)
        if type(base) is not int or not 0 <= base < len(bases):
            raise KernelLoweringError("mutable storage argument lacks an original base index")
        grouped[base].append(arg.name)
        dimensions, strides, _names = layouts[base]
        views.append(geometry.view(kwargs, arg.name, base, dimensions, strides, index))
    if any(not group for group in grouped):
        raise KernelLoweringError("mutable storage has an unaccounted cloned base")
    schema = node.args[0]._schema
    omitted = {arg.name for arg in mutated}
    if set(kwargs) - {arg.name for arg in schema.arguments if arg.name not in omitted}:
        raise KernelLoweringError("mutable storage has unknown, overridden or incomplete source fields")
    slots, result_types = [], []
    for typ in types:
        slots.append(len(result_types) if typ is not None else -1)
        if typ is not None:
            result_types.append(typ)
    result = _emit_bound_call(spec, schema, kwargs, value_map, result_types, omitted=omitted)
    call = result.ops[0]
    operands, names = list(call.operands), list(call.attributes["prov.kernel_operands"])
    ties, roots = [], []
    for index, base in enumerate(bases):
        name = builtin.StringAttr(f"@base{index}")
        ties.append(builtin.DictionaryAttr({"base_index": builtin.IntegerAttr(index, builtin.i64),
            "operand": builtin.IntegerAttr(len(operands), builtin.i64), "result": builtin.IntegerAttr(slots[ordinary+index], builtin.i64)}))
        roots.append(builtin.DictionaryAttr({"operand": name, "shape": builtin.ArrayAttr(layouts[index][2])}))
        operands.append(value_map[base.name])
        names.append(name)
    operands.extend(value for _name, value in geometry.fields)
    names.extend(builtin.StringAttr(name) for name, _value in geometry.fields)
    call.operands = operands
    call.attributes.update({"prov.kernel_operands": builtin.ArrayAttr(names),
        "prov.kernel_functionalization": builtin.StringAttr(PROFILE),
        "prov.kernel_source_schema": builtin.StringAttr(str(schema)),
        "prov.kernel_mutated": builtin.ArrayAttr(builtin.StringAttr(arg.name) for arg in mutated),
        "prov.kernel_tied": builtin.ArrayAttr(ties),
        "prov.kernel_result_slots": builtin.ArrayAttr(builtin.IntegerAttr(index, builtin.i64) for index in slots),
        "prov.kernel_storage_bases": builtin.ArrayAttr(roots), "prov.kernel_storage_views": builtin.ArrayAttr(views)})
    result.ops = [geometry.block.detach_op(op) for op in tuple(geometry.block.ops)] + [call]
    return result, slots
