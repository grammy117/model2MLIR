"""Canonical storage order for complete zero-origin dense permutation roots.

Torch's auto_functionalized_v2 uses clone_preserve_strides, not a logical
contiguous clone. Recover only a proved dense axis order; explicit standard
permutations supply all value semantics. No hidden storage bytes are invented.
"""

from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import TransposeOp

from .kernel_registry import KernelLoweringError


def dense_axes(metadata, rank):
    from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

    if metadata is None or not 0 <= rank <= 8 or len(metadata.shape) != rank:
        raise KernelLoweringError("mutable storage requires bounded original tensor metadata")
    if not statically_known_true(sym_eq(metadata.storage_offset(), 0)):
        raise KernelLoweringError("mutable storage base must have a proved zero origin")
    remaining, minor_first, product = list(reversed(range(rank))), [], 1
    strides = metadata.stride()
    while remaining:
        candidates = [axis for axis in remaining if statically_known_true(sym_eq(strides[axis], product))]
        if not candidates:
            raise KernelLoweringError("mutable storage base needs its proved dense source layout or permutation")
        # Unit dimensions must precede a nonunit dimension at the same stride;
        # otherwise multiplying first can lose the remaining exact stride proof.
        units = [axis for axis in candidates if statically_known_true(sym_eq(metadata.shape[axis], 1))]
        axis = (units or candidates)[0]
        minor_first.append(axis)
        remaining.remove(axis)
        product *= metadata.shape[axis]
    return tuple(reversed(minor_first))


def checked_result_adapters(call, operations, results):
    """Permit only complete inverse storage permutations after one real call."""
    if call.attributes.get("prov.kernel_functionalization") != builtin.StringAttr("m2m.functional-kernel-storage.v1"):
        raise ValueError("post-call result adapters require the actual mutable-storage contract")
    if len(results) != len(call.results):
        raise ValueError("storage result adapters change the complete result roster")
    trailing = operations[operations.index(call) + 1:]
    allowed = {"arith.constant", "tensor.dim", "tensor.empty", "linalg.transpose"}
    if any(op.name not in allowed for op in trailing):
        raise ValueError("storage result adapters retain an unaccounted operation")
    ties = call.attributes["prov.kernel_tied"]
    by_result = {tie.data["result"].value.data: tie.data["operand"].value.data for tie in ties}
    used = set()
    for index, (raw, adapted) in enumerate(zip(call.results, results, strict=True)):
        if adapted is raw:
            continue
        if index not in by_result:
            raise ValueError("storage result adapter changes an ordinary kernel result")
        before = call.operands[by_result[index]].owner
        after = adapted.owner
        if (not isinstance(before, TransposeOp) or not isinstance(after, TransposeOp)
                or list(after.inputs) != [raw] or after not in trailing
                or adapted.type != before.inputs[0].type):
            raise ValueError("storage result needs an actual inverse of its incoming permutation")
        axes = list(before.permutation.get_values())
        if list(after.permutation.get_values()) != [axes.index(axis) for axis in range(len(axes))]:
            raise ValueError("storage result permutation differs from its original logical axes")
        pending = [after]
        while pending:
            op = pending.pop()
            if op in used or op not in trailing:
                continue
            used.add(op)
            pending.extend(value.owner for value in op.operands if hasattr(value.owner, "operands"))
    if used != set(trailing):
        raise ValueError("storage result adapters have unaccounted payload")
    for op in trailing:
        op.verify()
