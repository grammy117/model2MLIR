"""Account for actual dimension reads in the standard sym_numel decomposition.

Torch 2.10's registered sym_numel decomposition reduces operator.mul over
t.shape. Match that executable tree and anchored tensor result, not FX names,
example extents or lineage tags on otherwise unexplained dimension queries.
"""

import math
from collections import Counter


def numel_dimension_relations(source, destination, relations, unknown):
    src = {node["id"]: node for node in source["nodes"]}
    dst = {node["id"]: node for node in destination["nodes"]}
    mapped = {}
    for relation in relations:
        for identity in relation["source_ids"]:
            mapped.setdefault(identity, set()).update(relation["destination_ids"])
    reverse = {}
    for identity, targets in mapped.items():
        for target in targets:
            reverse.setdefault(target, set()).add(identity)

    def result(nodes, reference):
        if not isinstance(reference, dict) or set(reference) != {"node_id", "value_id"}:
            return None
        node = nodes.get(reference["node_id"])
        if node is None:
            return None
        matches = [
            value
            for value in node.get("results", ())
            if value.get("id") == reference["value_id"]
        ]
        return (node, matches[0]) if len(matches) == 1 else None

    proved = []
    for original in source["nodes"]:
        args = original.get("args", ())
        if (
            original.get("op") != "call_function"
            or original.get("target") != "aten.sym_numel.default"
            or len(args) != 1
            or original.get("kwargs")
        ):
            continue
        tensor = result(src, args[0])
        if tensor is None or tensor[1].get("kind") != "tensor":
            continue
        shape = tensor[1].get("shape")
        if not isinstance(shape, list) or any(
            type(n) not in (int, str) or type(n) is int and n < 0 for n in shape
        ):
            continue
        for final_id in mapped.get(original["id"], ()):
            final = dst.get(final_id)
            if final is None or len(final.get("results", ())) != 1:
                continue
            roots, axes, members = set(), [], set()

            def tree(
                reference,
                visiting,
                *,
                final=final,
                shape=shape,
                tensor=tensor,
                members=members,
                roots=roots,
                axes=axes,
            ):
                if type(reference) is int and reference >= 0:
                    return reference, Counter()
                selected = result(dst, reference)
                if selected is None:
                    raise ValueError("missing actual shape value")
                node, value = selected
                identity = node["id"]
                if (
                    identity in visiting
                    or node.get("graph_id") != final.get("graph_id")
                    or node.get("op") != "call_function"
                    or node.get("kwargs")
                    or len(node.get("results", ())) != 1
                    or value.get("kind") != "SymInt"
                ):
                    raise ValueError("unknown shape arithmetic")
                members.add(identity)
                operands = node.get("args", ())
                if (
                    node.get("target") == "<built-in function mul>"
                    and len(operands) == 2
                ):
                    left, lhs = tree(operands[0], visiting | {identity})
                    right, rhs = tree(operands[1], visiting | {identity})
                    return left * right, lhs + rhs
                if (
                    node.get("target") == "<built-in function pow>"
                    and len(operands) == 2
                    and type(operands[1]) is int
                    and 0 <= operands[1] <= len(shape)
                ):
                    factor, symbols = tree(operands[0], visiting | {identity})
                    return factor ** operands[1], Counter(
                        {
                            name: count * operands[1]
                            for name, count in symbols.items()
                            if operands[1]
                        }
                    )
                if (
                    node.get("target") != "aten.sym_size.int"
                    or len(operands) != 2
                    or type(operands[1]) is not int
                ):
                    raise ValueError("not a source dimension read")
                root = result(dst, operands[0])
                if root is None:
                    raise ValueError("missing dimension tensor")
                candidates = []
                for source_id in reverse.get(root[0]["id"], ()):
                    owner = src.get(source_id)
                    values = owner.get("results", ()) if owner else ()
                    if (
                        owner is None
                        or len(values) != 1
                        or values[0].get("kind") != "tensor"
                        or owner.get("graph_id") != tensor[0].get("graph_id")
                        or source_id != tensor[0]["id"]
                        and not (
                            owner.get("op") == tensor[0].get("op") == "placeholder"
                        )
                        or {k: v for k, v in root[1].items() if k != "id"}
                        != {k: v for k, v in values[0].items() if k != "id"}
                    ):
                        continue
                    candidates.append(values[0])
                if len(candidates) != 1:
                    raise ValueError(
                        "dimension read changes its original tensor or axis"
                    )
                root_shape = candidates[0].get("shape")
                axis = operands[1] + len(root_shape) if operands[1] < 0 else operands[1]
                if not 0 <= axis < len(root_shape):
                    raise ValueError("dimension read has an invalid axis")
                extent = root_shape[axis]
                if type(extent) not in (int, str) or type(extent) is int and extent < 0:
                    raise ValueError("dimension read lacks its exported shape contract")
                roots.add(root[0]["id"])
                axes.append({"tensor": root[0]["id"], "axis": axis})
                if type(extent) is str:
                    return 1, Counter({extent: 1})
                return extent, Counter()

            try:
                constant, symbols = tree(
                    {"node_id": final_id, "value_id": final["results"][0]["id"]}, set()
                )
            except (ValueError, KeyError, TypeError, RecursionError):
                continue
            expected_factor = math.prod(n for n in shape if type(n) is int)
            if (
                not roots
                or expected_factor != constant
                or expected_factor
                and Counter(n for n in shape if type(n) is str) != symbols
            ):
                continue
            added = sorted(members.intersection(unknown))
            if added:
                proved.append(
                    {
                        "source_ids": [original["id"]],
                        "destination_ids": added,
                        "kind": "numel_dimension_queries",
                        "proof": {
                            "tensor_source": tensor[0]["id"],
                            "tensor_destinations": sorted(roots),
                            "queried_axes": sorted(
                                axes, key=lambda row: (row["tensor"], row["axis"])
                            ),
                            "exported_symbols": dict(sorted(symbols.items())),
                            "static_factor": constant,
                            "complete_dimension_product": True,
                        },
                    }
                )
    return proved
