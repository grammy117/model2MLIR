"""Explicit functional state ports from Torch's typed export signature.

The entry remains a pure tensor function. These ports describe which complete
results update caller state; they never grant writable borrowed inputs or public
physical aliases. Invocation adapters must implement copyback separately.
"""

PROFILE = "m2m.functional-state-results.v1"
ATTRIBUTE = "prov.functional_state"


def output_contract(exported):
    from torch.export.graph_signature import InputKind, OutputKind, TensorArgument

    signature = getattr(exported, "graph_signature", None)
    if signature is None:
        return None
    mutations = {OutputKind.USER_INPUT_MUTATION, OutputKind.BUFFER_MUTATION}
    if not any(spec.kind in mutations for spec in signature.output_specs):
        return None
    placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
    inputs = list(signature.input_specs)
    if len(inputs) != len(placeholders) or any(
        getattr(spec.arg, "name", None) != node.name
        for spec, node in zip(inputs, placeholders, strict=True)
    ):
        raise ValueError("functional state ports require exact exported placeholder bindings")
    outputs = [node for node in exported.graph.nodes if node.op == "output"]
    if len(outputs) != 1 or len(outputs[0].args) != 1:
        raise ValueError("functional state ports require one complete flat output")
    returned = outputs[0].args[0]
    if not isinstance(returned, (tuple, list)) or len(returned) != len(signature.output_specs):
        raise ValueError("functional state ports differ from the complete exported output roster")
    states, users, used = [], [], set()
    for ordinal, (spec, node) in enumerate(zip(signature.output_specs, returned, strict=True)):
        if not isinstance(spec.arg, TensorArgument) or spec.arg.name != getattr(node, "name", None):
            raise ValueError("functional state ports require actual Tensor output bindings")
        if spec.kind == OutputKind.USER_OUTPUT:
            users.append(ordinal)
            continue
        if spec.kind not in mutations:
            raise ValueError("unsupported exported effect in functional state results")
        kind = InputKind.USER_INPUT if spec.kind == OutputKind.USER_INPUT_MUTATION else InputKind.BUFFER
        candidates = [index for index, source in enumerate(inputs) if source.kind == kind
                      and isinstance(source.arg, TensorArgument)
                      and (source.arg.name if kind == InputKind.USER_INPUT else source.target) == spec.target]
        if len(candidates) != 1 or candidates[0] in used:
            raise ValueError("functional state result needs one distinct original input binding")
        index = candidates[0]
        used.add(index)
        states.append({"result": ordinal, "input": index, "kind": spec.kind.name, "target": spec.target})
    return {"schema": PROFILE, "state_outputs": states, "user_outputs": users,
            "output_count": len(returned), "invocation": "explicit-state-results"}


def check_types(contract, inputs, outputs):
    if contract is None:
        return
    if len(outputs) != contract["output_count"]:
        raise ValueError("functional state entry loses an original output")
    for state in contract["state_outputs"]:
        if inputs[state["input"]] != outputs[state["result"]]:
            raise ValueError("functional state result changes its source input tensor type")
