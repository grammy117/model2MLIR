"""Retain Torch's base-cloning mutation as explicit tied tensor results.

Torch 2.10 _higher_order_ops/auto_functionalize.py auto_functionalized_v2_dense
returns ordinary result slots followed by complete cloned bases. Decode its
actual schema, keyword fields and metadata; caller annotations supply no effects.
"""

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
    i64,
)

from .kernel_registry import (
    KernelLoweringError,
    _emit_bound_call,
    _is_fx_node,
    functional_kernel,
)

PROFILE = "m2m.functional-kernel-bases.v1"


def storage_required(node):
    """Select a storage-root contract when source views share or reshape bases."""
    arguments = [arg for arg in node.args[0]._schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
    positions = [node.kwargs.get("_" + arg.name + "_base_index") for arg in arguments]
    fields = {"_" + arg.name + suffix for arg in arguments
              for suffix in ("_size", "_stride", "_storage_offset", "_slice_dim", "_slice_start", "_slice_end")}
    return bool(fields.intersection(node.kwargs)) or len(set(positions)) != len(positions)


def emit(spec, node, value_map, tensor_type):
    if functional_kernel(node) != spec:
        raise KernelLoweringError("mutating kernels require the actual auto_functionalized_v2 source operator")
    target, values = node.args[0], node.meta.get("val")
    schema = target._schema
    if not isinstance(values, (tuple, list)):
        raise KernelLoweringError("functional kernel lacks its complete tuple result metadata")
    mutated = [arg for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
    if not mutated or any(not isinstance(arg.type, torch.TensorType) for arg in mutated):
        raise KernelLoweringError("functional kernel requires whole mutable Tensor arguments")
    kwargs = dict(node.kwargs)
    bases = kwargs.pop("_all_bases", None)
    if (not isinstance(bases, (tuple, list)) or not bases or len(bases) > 64
            or any(not _is_fx_node(base) or base.name not in value_map
                   or not isinstance(value_map[base.name].type, TensorType) for base in bases)):
        raise KernelLoweringError("functional kernel lacks its complete source tensor bases")
    ordinary = len(schema.returns) or 1
    if len(values) != ordinary + len(bases) or not schema.returns and values[0] is not None:
        raise KernelLoweringError("functional kernel result slots differ from the original schema/base roster")
    types = [tensor_type(value) if value is not None else None for value in values]
    if any(value is not None and typ is None for value, typ in zip(values, types, strict=True)):
        raise KernelLoweringError("functional kernel has an unsupported non-tensor result")
    if any(not isinstance(ret.type, torch.TensorType) or types[index] is None
           for index, ret in enumerate(schema.returns)):
        raise KernelLoweringError("functional kernel requires every ordinary Tensor result")
    if any(ret.alias_info is not None for ret in schema.returns):
        raise KernelLoweringError("functional kernel returns cannot alias source arguments")
    if any(types[ordinary+i] != value_map[base.name].type for i, base in enumerate(bases)):
        raise KernelLoweringError("functional kernel changes a cloned base's tensor type")
    if storage_required(node):
        from .storage_kernels import emit as emit_storage
        return emit_storage(spec, node, value_map, bases, mutated, ordinary, types)
    base_arguments = {}
    for arg in mutated:
        position = kwargs.pop("_" + arg.name + "_base_index", None)
        alias = kwargs.pop("_" + arg.name + "_alias", False)
        if (type(position) is not int or not 0 <= position < len(bases) or position in base_arguments
                or type(alias) is not bool):
            raise KernelLoweringError("functional kernel needs distinct whole-base mutation bindings")
        base_arguments[position] = arg.name
        kwargs[arg.name] = bases[position]
    if set(base_arguments) != set(range(len(bases))):
        raise KernelLoweringError("functional kernel has an unaccounted mutation base")
    allowed = {arg.name for arg in schema.arguments}
    if set(kwargs) - allowed:
        raise KernelLoweringError("functional kernel view/clone fields need explicit source semantics: "
                                  + ", ".join(sorted(set(kwargs) - allowed)))
    for arg in schema.arguments:
        if arg.name not in kwargs and arg.has_default_value():
            kwargs[arg.name] = arg.default_value
    slots, result_types = [], []
    for typ in types:
        slots.append(len(result_types) if typ is not None else -1)
        if typ is not None:
            result_types.append(typ)
    result = _emit_bound_call(spec, schema, kwargs, value_map, result_types)
    call = result.ops[0]
    names = [name.data for name in call.attributes["prov.kernel_operands"]]
    ties = []
    for base_index, argument in sorted(base_arguments.items()):
        ties.append(DictionaryAttr({"argument": StringAttr(argument), "base_index": IntegerAttr(base_index, i64),
            "operand": IntegerAttr(names.index(argument), i64), "result": IntegerAttr(slots[ordinary+base_index], i64)}))
    call.attributes.update({"prov.kernel_functionalization": StringAttr(PROFILE),
        "prov.kernel_source_schema": StringAttr(str(schema)),
        "prov.kernel_mutated": ArrayAttr(StringAttr(arg.name) for arg in mutated),
        "prov.kernel_tied": ArrayAttr(ties),
        "prov.kernel_result_slots": ArrayAttr(IntegerAttr(index, i64) for index in slots)})
    return result, slots
