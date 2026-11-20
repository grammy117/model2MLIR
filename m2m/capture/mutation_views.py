"""Observe Torch's actual regenerated view metadata for source correspondence.

Geometry records describe the source; they grant no arithmetic, writer
equivalence, public alias or downstream memory permission.
"""


def expression(value):
    """Serialize actual symbolic structure without observing a backed hint."""
    import torch

    if type(value) is int:
        return ["integer", value]
    value = value.node.expr if isinstance(value, torch.SymInt) else value
    if getattr(value, "is_Integer", False):
        return ["integer", int(value)]
    if getattr(value, "is_Symbol", False):
        return ["symbol", str(value)]
    kind = {"Add": "add", "Mul": "mul", "FloorDiv": "floordiv"}.get(type(value).__name__)
    if kind is None:
        return None
    children = [expression(argument) for argument in value.args]
    return [kind, *children] if all(child is not None for child in children) else None


def geometry(value):
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        return None

    def dimension(number):
        from torch.fx.experimental.symbolic_shapes import is_concrete_int
        return int(number) if type(number) is int or is_concrete_int(number) else str(number)

    try:
        offset, strides = value.storage_offset(), value.stride()
        return {"shape": [dimension(size) for size in value.shape],
                "strides": [dimension(stride) for stride in strides],
                "offset": dimension(offset), "dtype": str(value.dtype),
                "shape_expressions": [expression(size) for size in value.shape],
                "stride_expressions": [expression(stride) for stride in strides],
                "offset_expression": expression(offset)}
    except (RuntimeError, NotImplementedError, AttributeError, TypeError):
        # Subclass storage may be intentionally unavailable. Do not invent
        # an origin/layout for unrelated captures or claim view correspondence.
        return None


def regenerated(node):
    """Use the selected Torch ViewInfo implementations, not a second decoder."""
    from torch._higher_order_ops.auto_functionalize import (
        get_mutable_args_from_schema,
        read_view_information_from_args,
    )
    from torch.fx import Node

    def actual(value):
        if isinstance(value, Node):
            return value.meta.get("val")
        if isinstance(value, (list, tuple)):
            return [actual(item) for item in value]
        return value

    kwargs = {key: actual(value) for key, value in node.kwargs.items()}
    bases = kwargs.pop("_all_bases")
    names, types = get_mutable_args_from_schema(node.args[0]._schema)
    infos = read_view_information_from_args(names, types, kwargs, bases)
    if any(info is None or isinstance(info, list) for info in infos.values()):
        return []
    return [{"argument": name, "base_index": infos[name].base_index,
             "geometry": geometry(infos[name].regenerate_view(bases))} for name in names]


def relations(source, destination, existing, consumed, unknown):
    """Account for regenerated storage views and their closed size expressions.

    Require actual source schemas/aliases, base ancestry, exact observed geometry
    and the destination's real scalar expression tree. This records view custody
    and shape decomposition; it does not certify the custom writer's arithmetic.
    """
    import sympy as sp
    from torch.utils._sympy.functions import FloorDiv

    from m2m.capture.view_trace import SourceStorageAliases

    aliases = SourceStorageAliases(source)
    if not aliases.valid or source.get("runtime_versions") != destination.get("runtime_versions"):
        return []
    dst = {node["id"]: node for node in destination["nodes"]}
    mapped = {}
    for row in existing:
        for identity in row["source_ids"]:
            mapped.setdefault(identity, set()).update(row["destination_ids"])
    pure_views = {"aten.slice.Tensor", "aten.select.int", "aten.transpose.int", "aten.permute.default",
                  "aten.as_strided.default", "aten.alias.default", "aten.detach.default"}
    result, accounted = [], set()

    def decode(tree):
        if not isinstance(tree, list) or len(tree) < 2:
            raise ValueError("missing structured source expression")
        if tree[0] == "integer" and len(tree) == 2 and type(tree[1]) is int:
            return sp.Integer(tree[1])
        if tree[0] == "symbol" and len(tree) == 2 and isinstance(tree[1], str) and tree[1]:
            return sp.Symbol(tree[1], integer=True)
        children = [decode(value) for value in tree[1:]]
        if tree[0] in ("add", "mul"):
            return (sp.Add if tree[0] == "add" else sp.Mul)(*children)
        if tree[0] == "floordiv" and len(children) == 2:
            return FloorDiv(*children)
        raise ValueError("unknown structured source expression")

    def reference(value, nodes):
        if not isinstance(value, dict) or set(value) != {"node_id", "value_id"}:
            raise ValueError("missing exact source reference")
        node = nodes[value["node_id"]]
        if value["value_id"] not in {result["id"] for result in node["results"]}:
            raise ValueError("source reference changes its actual result")
        return node

    def shape_ancestors(root):
        """Follow exact shape-preserving source operations, never storage aliases.

        A clone owns fresh storage but preserves dimensions. Torch may hoist a
        regenerated view's dimension query before that clone and a permutation.
        Check the actual selected schemas, axis maps and typed source geometry;
        equal symbolic spellings on unrelated tensors supply no ancestry.
        """
        node, visited = aliases.nodes[root], set()
        while node["id"] not in visited:
            visited.add(node["id"])
            yield node
            args, kwargs = node.get("args", ()), node.get("kwargs", {})
            schema = aliases.schema(node)
            if (schema is None or len(schema.returns) != 1 or not args
                    or any(arg.alias_info is not None and arg.alias_info.is_write for arg in schema.arguments)):
                return
            parent = aliases.tensor_ref(args[0])
            if (parent is None or parent.get("graph_id") != node.get("graph_id")
                    or parent["ordinal"] >= node["ordinal"]):
                return
            before, after = parent.get("storage_geometry", ()), node.get("storage_geometry", ())
            if len(before) != 1 or len(after) != 1:
                return
            shape = before[0].get("shape_expressions")
            if not isinstance(shape, list):
                return
            rank, op = len(shape), node["target"]
            axes = list(range(rank))
            if op in {"aten.clone.default", "aten.alias.default", "aten.detach.default"}:
                if len(args) != 1 or set(kwargs) - ({"memory_format"} if op == "aten.clone.default" else set()):
                    return
            elif op == "aten.numpy_T.default" and len(args) == 1 and not kwargs:
                axes.reverse()
            elif op == "aten.permute.default" and len(args) == 2 and not kwargs:
                axes = args[1]
                if (not isinstance(axes, list) or len(axes) != rank
                        or any(type(axis) is not int or not -rank <= axis < rank for axis in axes)):
                    return
                axes = [axis % rank for axis in axes]
                if sorted(axes) != list(range(rank)):
                    return
            elif op == "aten.transpose.int" and len(args) == 3 and not kwargs:
                if any(type(axis) is not int or not -rank <= axis < rank for axis in args[1:]):
                    return
                left, right = (axis % rank for axis in args[1:])
                axes[left], axes[right] = axes[right], axes[left]
            else:
                return
            if (after[0].get("shape_expressions") != [shape[axis] for axis in axes]
                    or after[0].get("dtype") != before[0].get("dtype")):
                return
            node = parent

    for original in source["nodes"]:
        schema = aliases.schema(original)
        if schema is None:
            continue
        mutable = [arg for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
        targets = [dst[identity] for identity in mapped.get(original["id"], ()) if identity in dst
                   and dst[identity].get("functionalization", {}).get("profile") == "m2m.functional-kernel-storage.v1"]
        if not mutable or len(targets) != 1:
            continue
        target = targets[0]
        info = target["functionalization"]
        if info.get("operator") != original["target"] or info.get("schema") != str(schema):
            continue
        views = info.get("views")
        bases = target.get("kwargs", {}).get("_all_bases")
        if not isinstance(views, list) or len(views) != len(mutable) or not isinstance(bases, list):
            continue
        selected, roots = set(), set()

        def base_identity(ref, root, visited=frozenset(), *, before=original["ordinal"]):
            node = reference(ref, dst)
            if node["id"] in mapped.get(root, ()):
                original_base = aliases.nodes[root].get("storage_geometry")
                return bool(original_base) and original_base == node.get("storage_geometry")
            if (node["id"] in visited or node["target"] != "<built-in function getitem>"
                    or len(node.get("args", ())) != 2 or type(node["args"][1]) is not int):
                return False
            parent = reference(node["args"][0], dst)
            parent_info = parent.get("functionalization", {})
            if parent_info.get("profile") not in ("m2m.functional-kernel-storage.v1", "m2m.functional-kernel-bases.v1"):
                return False
            sources = [item for item in source["nodes"] if parent["id"] in mapped.get(item["id"], ())
                       and item.get("target") == parent_info.get("operator")]
            if len(sources) != 1 or sources[0]["ordinal"] >= before:
                return False
            previous = aliases.schema(sources[0])
            if previous is None or str(previous) != parent_info.get("schema"):
                return False
            ordinal = node["args"][1] - (len(previous.returns) or 1)
            owners = parent.get("kwargs", {}).get("_all_bases", ())
            return 0 <= ordinal < len(owners) and base_identity(owners[ordinal], root, visited | {node["id"]})

        try:
            for arg, view in zip(mutable, views, strict=True):
                index = list(schema.arguments).index(arg)
                owner = aliases.tensor_ref(aliases.argument(original, index, arg.name))
                if (owner is None or view.get("argument") != arg.name
                        or type(view.get("base_index")) is not int or not 0 <= view["base_index"] < len(bases)
                        or owner.get("storage_geometry") != [view.get("geometry")]):
                    raise ValueError("functional view differs from original typed geometry")
                root = aliases.root(owner)
                if root is None or not base_identity(bases[view["base_index"]], root):
                    raise ValueError("functional view changes its original storage root")
                roots.add(root)
                while owner["id"] != root:
                    if owner["target"] not in pure_views:
                        raise ValueError("unknown original storage view")
                    selected.add(owner["id"])
                    owner = aliases.tensor_ref(aliases.argument(owner, 0, "self"))
                    if owner is None:
                        raise ValueError("storage view loses its original owner")
            uses = [edge for edge in source.get("edges", ()) if edge["producer_node_id"] in selected]
            if any(edge["consumer_node_id"] not in selected | {original["id"]} | consumed for edge in uses):
                raise ValueError("storage view retains an unaccounted external observation")
            members = set()

            def scalar(value, visiting=frozenset(), *, members=members, roots=roots, base_identity=base_identity):
                if type(value) is int:
                    return sp.Integer(value)
                node = reference(value, dst)
                if node["id"] in visiting or node.get("kwargs") or len(node.get("results", ())) != 1:
                    raise ValueError("unknown storage coordinate computation")
                args, op = node.get("args", ()), node["target"]
                members.add(node["id"])
                if op in {"aten.sym_size.int", "aten.sym_stride.int"} and len(args) == 2 and type(args[1]) is int:
                    owner = reference(args[0], dst)
                    anchored = any(base_identity(args[0], root) for root in roots)
                    if not anchored:
                        anchored = any(owner["id"] in mapped.get(ancestor["id"], ())
                            and owner.get("storage_geometry") == ancestor.get("storage_geometry")
                            for root in roots for ancestor in shape_ancestors(root))
                    if not anchored:
                        raise ValueError("coordinate dimension has a different actual owner")
                    geometry = owner.get("storage_geometry", ())
                    field = "shape_expressions" if op == "aten.sym_size.int" else "stride_expressions"
                    if len(geometry) != 1 or not -len(geometry[0][field]) <= args[1] < len(geometry[0][field]):
                        raise ValueError("coordinate dimension has no original typed axis")
                    actual = decode(geometry[0][field][args[1]])
                elif op in {"<built-in function add>", "<built-in function sub>", "<built-in function mul>",
                            "<built-in function floordiv>"} and len(args) == 2:
                    left, right = (scalar(arg, visiting | {node["id"]}) for arg in args)
                    if op == "<built-in function add>":
                        actual = left + right
                    elif op == "<built-in function sub>":
                        actual = left - right
                    elif op == "<built-in function mul>":
                        actual = left * right
                    elif right.is_Integer and right > 0:
                        actual = FloorDiv(left, right)
                    else:
                        raise ValueError("coordinate division lacks an actual positive literal divisor")
                else:
                    raise ValueError("unaccounted coordinate operator")
                if actual != decode(node.get("integer_expression")):
                    raise ValueError("actual coordinate arithmetic differs from its observed symbolic value")
                return actual

            def visit(value):
                if isinstance(value, list):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    scalar(value)
                elif type(value) is not int and type(value) is not bool:
                    raise ValueError("unknown original view field")

            for arg in mutable:
                prefix = "_" + arg.name
                for suffix in ("_size", "_stride", "_storage_offset", "_slice_dim", "_slice_start", "_slice_end", "_alias"):
                    if prefix + suffix in target.get("kwargs", {}):
                        visit(target["kwargs"][prefix + suffix])
            for arg, view in zip(mutable, views, strict=True):
                kwargs = target["kwargs"]
                prefix = "_" + arg.name
                if kwargs.get(prefix + "_base_index") != view["base_index"]:
                    raise ValueError("actual storage base differs from its regenerated view")
                base = reference(bases[view["base_index"]], dst)["storage_geometry"][0]
                shape = [decode(value) for value in base["shape_expressions"]]
                strides = [decode(value) for value in base["stride_expressions"]]
                offset = decode(base["offset_expression"])
                if prefix + "_alias" not in kwargs:
                    if prefix + "_storage_offset" in kwargs:
                        shape = [scalar(value) for value in kwargs[prefix + "_size"]]
                        strides = [scalar(value) for value in kwargs[prefix + "_stride"]]
                        offset = scalar(kwargs[prefix + "_storage_offset"])
                    elif prefix + "_slice_dim" in kwargs:
                        axis = kwargs[prefix + "_slice_dim"]
                        if type(axis) is not int or not 0 <= axis < len(shape):
                            raise ValueError("actual storage slice changes its axis")
                        start, end = (scalar(kwargs[prefix + suffix]) for suffix in ("_slice_start", "_slice_end"))
                        offset += start * strides[axis]
                        shape[axis] = end - start
                observed = view["geometry"]
                if (shape != [decode(value) for value in observed["shape_expressions"]]
                        or strides != [decode(value) for value in observed["stride_expressions"]]
                        or offset != decode(observed["offset_expression"])):
                    raise ValueError("actual regenerated view differs from original observed geometry")
        except (KeyError, TypeError, ValueError, RecursionError, AttributeError):
            continue
        proof = {"kernel_node_id": target["id"], "source_storage_roots": sorted(roots),
                 "source_view_node_ids": sorted(selected), "exact_typed_geometry": True,
                 "scope": "source view/coordinate custody; not writer or numeric equivalence"}
        for identity in sorted(selected - consumed - accounted):
            result.append({"source_ids": [identity], "destination_ids": [], "kind": "storage_view", "proof": proof})
            accounted.add(identity)
        introduced = sorted(members.intersection(unknown))
        if introduced:
            result.append({"source_ids": sorted(selected or {original["id"]}), "destination_ids": introduced,
                           "kind": "introduced_storage_coordinates", "proof": proof})
    return result
