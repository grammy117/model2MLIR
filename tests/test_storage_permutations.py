"""Complete mutation roots retain source storage order as standard computation."""

import copy

import pytest
import torch
from test_functional_kernels import pair
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.linalg.ops import TransposeOp

import m2m
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


@pytest.fixture(scope="module")
def transposed():

    class Model(torch.nn.Module):
        def forward(self, x):
            state = x.T.clone()
            before = state.clone()
            a, b = pair(state[:, 1], state[:, -1])
            return state, before, a, b

    inputs = (torch.arange(24).reshape(4, 6),)
    exported = torch.export.export(Model(), inputs, dynamic_shapes=({
        0: torch.export.Dim("rows", min=2, max=7), 1: torch.export.Dim("cols", min=2, max=9)},))
    name = "m2m_state_test::pair"
    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name)
        else:
            KERNEL_REGISTRY[name] = previous
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    result.module.verify()
    return result


def test_internal_transposed_clone_keeps_one_shared_root_and_inverse_result(transposed):
    result = transposed
    calls = [op for op in result.module.walk() if isinstance(op, func.CallOp)]
    assert len(calls) == 1
    call = calls[0]
    names = [name.data for name in call.attributes["prov.kernel_operands"]]
    root = call.operands[names.index("@base0")]
    assert isinstance(root.owner, TransposeOp)
    assert list(root.owner.permutation.get_values()) == [1, 0]
    tied = call.attributes["prov.kernel_tied"].data[0]
    state = call.results[tied.data["result"].value.data]
    restorations = [op for op in result.module.walk() if isinstance(op, TransposeOp)
                    and op.inputs[0] is state]
    assert len(restorations) == 1
    assert list(restorations[0].permutation.get_values()) == [1, 0]


@pytest.mark.parametrize("damage", ("dimension-axis", "different-owner", "clone-schema", "permutation"))
def test_hoisted_dimensions_require_actual_source_shape_ancestry(transposed, damage):
    from m2m.capture.trace import graph_relation

    source = copy.deepcopy(transposed.capture_trace["graphs"]["quantized"])
    prepared = copy.deepcopy(transposed.capture_trace["graphs"]["prepared"])
    dimension = next(node for node in prepared["nodes"] if node["target"] == "aten.sym_size.int")
    if damage == "dimension-axis":
        dimension["args"][1] = 1 - dimension["args"][1]
    elif damage == "different-owner":
        snapshot = [node for node in prepared["nodes"] if node["target"] == "aten.clone.default"][-1]
        dimension["args"][0] = {"node_id": snapshot["id"], "value_id": snapshot["results"][0]["id"]}
    elif damage == "clone-schema":
        source["operator_schemas"]["aten.clone.default"] = str(torch.ops.aten.alias.default._schema)
    else:
        permutation = next(node for node in source["nodes"] if node["target"] == "aten.numpy_T.default")
        permutation["target"] = "aten.permute.default"
        permutation["args"].append([0, 1])
        source["operator_schemas"][permutation["target"]] = str(torch.ops.aten.permute.default._schema)
    assert graph_relation(source, prepared)["status"] == "diagnostic"


@pytest.mark.parametrize("axes", ((2, 0, 1), (1, 2, 0)))
def test_dynamic_rank_three_roots_restore_their_logical_axis_order(monkeypatch, axes):
    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_state_test::pair", KernelSpec("m2m_state_test::pair"))

    class Model(torch.nn.Module):
        def forward(self, x):
            state = x.permute(axes).clone()
            old = pair(state[:, 1, :], state[:, -1, :])
            return state, old

    inputs = (torch.arange(60).reshape(3, 4, 5),)
    exported = torch.export.export(Model(), inputs, dynamic_shapes=({
        axis: torch.export.Dim(f"axis{axis}", min=2, max=8) for axis in range(3)},))
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    result.module.verify()
    call = next(op for op in result.module.walk() if isinstance(op, func.CallOp))
    tied = call.attributes["prov.kernel_tied"].data[0]
    operand = call.operands[tied.data["operand"].value.data]
    assert isinstance(operand.owner, TransposeOp)
    assert list(operand.owner.permutation.get_values()) == [axes.index(axis) for axis in range(3)]
    raw = call.results[tied.data["result"].value.data]
    restored = next(op for op in result.module.walk() if isinstance(op, TransposeOp) and op.inputs[0] is raw)
    assert list(restored.permutation.get_values()) == list(axes)
    if axes == (1, 2, 0):
        from m2m.capture.trace import graph_relation
        prepared = copy.deepcopy(result.capture_trace["graphs"]["prepared"])
        stride = next(node for node in prepared["nodes"] if node["target"] == "aten.sym_stride.int")
        stride["args"][1] = (stride["args"][1] + 1) % 3
        assert graph_relation(result.capture_trace["graphs"]["quantized"], prepared)["status"] == "diagnostic"


@pytest.mark.parametrize("damage", ("inverse", "ordinary-result", "extra-payload", "profile"))
def test_result_adapters_require_complete_inverse_permutations(transposed, damage):
    from m2m.ir.storage_layouts import checked_result_adapters

    module = transposed.module.clone()
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    tied = call.attributes["prov.kernel_tied"].data[0]
    index = tied.data["result"].value.data
    restored = next(op for op in module.walk() if isinstance(op, TransposeOp) and op.inputs[0] is call.results[index])
    operations = list(call.parent.ops)
    operations = operations[:operations.index(restored) + 1]
    results = list(call.results)
    results[index] = restored.results[0]
    checked_result_adapters(call, operations, results)
    if damage == "inverse":
        restored.properties["permutation"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1])
    elif damage == "ordinary-result":
        results[0] = restored.results[0]
    elif damage == "extra-payload":
        operations.append(arith.ConstantOp.from_int_and_width(0, builtin.i64))
    else:
        call.attributes["prov.kernel_functionalization"] = builtin.StringAttr("m2m.functional-kernel-bases.v1")
    with pytest.raises(ValueError):
        checked_result_adapters(call, operations, results)


@pytest.mark.parametrize("shape,permutation", (((2, 1, 3), (2, 0, 1)), ((1, 4, 1), (1, 2, 0))))
def test_mutable_storage_retains_real_unit_axes(monkeypatch, shape, permutation):
    monkeypatch.setitem(KERNEL_REGISTRY, "m2m_state_test::pair", KernelSpec("m2m_state_test::pair"))

    class Model(torch.nn.Module):
        def forward(self, x):
            state = x.permute(permutation).clone()
            a, b = pair(state[:, 0, :], state[:, -1, :])
            return state, a, b

    result = m2m.convert(Model(), (torch.empty(shape, dtype=torch.int64),), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    result.module.verify()
    entry = next(op for op in result.module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    assert entry.function_type.outputs.data[0].get_shape() == tuple(shape[axis] for axis in permutation)
