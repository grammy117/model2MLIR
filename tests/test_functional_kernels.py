"""Actual Torch mutation exports retain state, snapshots and complete results."""

import copy
import json

import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp

import m2m
from m2m.capture.bundle import write_bundle
from m2m.capture.eager_state import observe
from m2m.ir.functional_state import ATTRIBUTE, PROFILE
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


@torch.library.custom_op("m2m_state_test::bump", mutates_args={"state"})
def bump(state: torch.Tensor, bias: int = 3) -> torch.Tensor:
    before = state.clone()
    state.add_(bias)
    return before


@bump.register_fake
def _(state, bias=3):
    return torch.empty_like(state)


@torch.library.custom_op("m2m_state_test::pair", mutates_args={"left", "right"})
def pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    before = left.clone(), right.clone()
    left.add_(2)
    right.mul_(3)
    return before


@pair.register_fake
def _(left, right):
    return torch.empty_like(left), torch.empty_like(right)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    for name in ("m2m_state_test::bump", "m2m_state_test::pair"):
        monkeypatch.setitem(KERNEL_REGISTRY, name, KernelSpec(name))


class Chain(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.weight = torch.nn.Parameter(torch.tensor(4, dtype=torch.int64), requires_grad=False)
        self.register_buffer("state", torch.arange(5, dtype=torch.int64) + 10)

    def forward(self, x):
        state = self.state if self.kind == "buffer" else x if self.kind == "public" else x + self.weight
        before = state.clone()
        first = bump(state)
        second = bump(state, -2)
        return before, first, second, state, x, self.weight


def entry(result):
    return next(op for op in result.module.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")


@pytest.mark.parametrize("kind", ("internal", "public", "buffer"))
def test_original_mutation_schema_and_complete_state_chain(kind):
    converted = m2m.convert(Chain(kind), (torch.arange(5),), backend="fx_importer", capture_trace=True)
    assert converted.ok, converted.diagnostics
    assert converted.capture_trace["status"] == "complete", converted.capture_trace["blockers"]
    calls = [op for op in converted.module.walk() if isinstance(op, CallOp)]
    assert len(calls) == 2 and all(len(call.results) == 2 for call in calls)
    assert calls[1].arguments[0] is calls[0].results[1]
    assert calls[0].attributes["prov.kernel_source_schema"].data == str(torch.ops.m2m_state_test.bump.default._schema)
    for call in calls:
        ties = call.attributes["prov.kernel_tied"]
        assert len(ties) == 1
        assert ties.data[0].data["argument"].data == "state"
        assert ties.data[0].data["result"].value.data == 1
    main = entry(converted)
    assert isinstance(main.body.block.last_op, ReturnOp)
    assert len(main.function_type.outputs) == (6 if kind == "internal" else 7)
    if kind == "internal":
        assert ATTRIBUTE not in main.attributes
    else:
        contract = json.loads(main.attributes[ATTRIBUTE].data)
        assert contract["schema"] == PROFILE
        assert contract["user_outputs"] == list(range(1, 7))
        assert contract["state_outputs"][0]["result"] == 0
        assert contract["state_outputs"][0]["kind"] == (
            "USER_INPUT_MUTATION" if kind == "public" else "BUFFER_MUTATION")
        assert main.function_type.outputs.data[0] == main.function_type.inputs.data[contract["state_outputs"][0]["input"]]
    prepared = converted.capture_trace["graphs"]["prepared"]
    hops = [node for node in prepared["nodes"] if node.get("functionalization")]
    assert len(hops) == 2 and hops[0]["functionalization"]["schema"] == str(torch.ops.m2m_state_test.bump.default._schema)
    assert len(prepared["output_specs"]) == len(main.function_type.outputs)


@pytest.mark.parametrize("kind", ("public", "buffer"))
@pytest.mark.parametrize("reuse", (False, True))
@pytest.mark.parametrize("regions", (False, True))
def test_bundle_preserves_initial_inputs_buffers_and_all_state_references(tmp_path, kind, reuse, regions):
    model, x = Chain(kind), torch.arange(5)
    initial_x, initial_buffer = x.clone(), model.state.clone()
    expected_model = copy.deepcopy(model)
    expected_x = x.clone()
    user_outputs = expected_model(expected_x)
    state = expected_x if kind == "public" else expected_model.state
    wanted = (state, *user_outputs)
    converted = None
    if reuse:
        converted = m2m.convert(model, (x,), backend="fx_importer", capture_trace=True,
                                weights_path=str(tmp_path / "weights.safetensors"))
        assert converted.ok, converted.diagnostics
    write_bundle(model, (x,), tmp_path, capture_regions=regions, capture_trace=True, conversion_result=converted)
    assert torch.equal(x, initial_x) and torch.equal(model.state, initial_buffer)
    with np.load(tmp_path / "inputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["in0"], initial_x.numpy())
    with np.load(tmp_path / "extra.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["buf::state"], initial_buffer.numpy())
    with np.load(tmp_path / "goldens.npz", allow_pickle=False) as data:
        assert len(data.files) == len(wanted)
        for index, expected in enumerate(wanted):
            np.testing.assert_array_equal(data[f"out{index}"], expected.numpy())
    meta = json.loads((tmp_path / "meta.json").read_text())
    receipt = json.loads((tmp_path / "capture_receipt.json").read_text())
    assert meta["functional_state"]["schema"] == PROFILE
    assert len(meta["output_abi"]) == receipt["materialized_abi"]["outputs"] == len(wanted)


def test_two_distinct_mutable_bases_keep_original_return_and_state_order():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            a, b = pair(x, y)
            return a, b, x, y
    converted = m2m.convert(Model(), (torch.arange(3), torch.arange(4)), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    call = next(op for op in converted.module.walk() if isinstance(op, CallOp))
    assert len(call.results) == 4
    assert [tie.data["result"].value.data for tie in call.attributes["prov.kernel_tied"]] == [2, 3]
    assert len(json.loads(entry(converted).attributes[ATTRIBUTE].data)["state_outputs"]) == 2


def test_mutated_slice_retains_complete_storage_and_source_view_coordinates():
    class Model(torch.nn.Module):
        def forward(self, x):
            old = bump(x[1:])
            return old, x
    converted = m2m.convert(Model(), (torch.arange(5),), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    call = next(op for op in converted.module.walk() if isinstance(op, CallOp))
    assert call.attributes["prov.kernel_functionalization"].data == "m2m.functional-kernel-storage.v1"
    assert call.attributes["prov.kernel_storage_views"].data[0].data["kind"].data == "slice"
    assert call.attributes["prov.kernel_storage_bases"].data[0].data["operand"].data == "@base0"
    assert len(call.results) == 2 and call.results[0].type.get_shape() == (4,) and call.results[1].type.get_shape() == (5,)


@pytest.mark.parametrize("kind", ("slice", "stride", "same", "matrix"))
def test_shared_mutable_views_remain_one_complete_base(kind):
    class Model(torch.nn.Module):
        def forward(self, x):
            if kind == "slice":
                left, right = x[1:-1], x[2:]
            elif kind == "stride":
                left, right = x[::2], x[1::2]
            elif kind == "matrix":
                left, right = x[:, 1], x[:, -1]
            else:
                left, right = x[1:-1], x[1:-1]
            a, b = pair(left, right)
            return a, b, x
    x = torch.arange(24).reshape(4, 6) if kind == "matrix" else torch.arange(12)
    dynamic = None if kind == "matrix" else ({0: torch.export.Dim("length", min=4, max=31)},)
    exported = torch.export.export(Model(), (x,), dynamic_shapes=dynamic)
    converted = m2m.convert(exported, (x,), backend="fx_importer", capture_trace=True)
    assert converted.ok, converted.diagnostics
    converted.module.verify()
    calls = [op for op in converted.module.walk() if isinstance(op, CallOp)]
    assert len(calls) == 1
    call = calls[0]
    assert call.attributes["prov.kernel_functionalization"].data == "m2m.functional-kernel-storage.v1"
    assert len(call.results) == 3 and len(call.attributes["prov.kernel_tied"]) == 1
    views = call.attributes["prov.kernel_storage_views"]
    assert len(views) == 2 and all(view.data["base_index"].value.data == 0 for view in views)
    assert len(call.attributes["prov.kernel_storage_bases"]) == 1
    assert converted.capture_trace["status"] == "complete", converted.capture_trace["blockers"]


@pytest.mark.parametrize("damage", ("geometry", "base", "offset", "schema", "missing-source"))
def test_storage_view_accounting_requires_original_geometry_and_ownership(damage):
    from m2m.capture.trace import graph_relation

    class Model(torch.nn.Module):
        def forward(self, x):
            a, b = pair(x[:, 1], x[:, -1])
            return a, b, x
    converted = m2m.convert(Model(), (torch.arange(24).reshape(4, 6),), backend="fx_importer", capture_trace=True)
    assert converted.ok and converted.capture_trace["status"] == "complete"
    source = copy.deepcopy(converted.capture_trace["graphs"]["quantized"])
    prepared = copy.deepcopy(converted.capture_trace["graphs"]["prepared"])
    hop = next(node for node in prepared["nodes"] if node.get("functionalization"))
    if damage == "geometry":
        view = next(node for node in source["nodes"] if node["target"] == "aten.select.int")
        view["storage_geometry"][0]["offset"] += 1
    elif damage == "base":
        hop["kwargs"]["_left_base_index"] = 1
    elif damage == "offset":
        hop["kwargs"]["_left_storage_offset"] = 2
    elif damage == "schema":
        hop["functionalization"]["schema"] = str(torch.ops.aten.add_.Tensor._schema)
    else:
        source["nodes"][0].pop("storage_geometry")
    assert graph_relation(source, prepared)["status"] == "diagnostic"


def test_coordinate_accounting_checks_actual_arithmetic_not_symbolic_tags():
    from m2m.capture.trace import graph_relation

    class Model(torch.nn.Module):
        def forward(self, x):
            a, b = pair(x[::2], x[1::2])
            return a, b, x
    exported = torch.export.export(Model(), (torch.arange(12),), dynamic_shapes=({0: torch.export.Dim("n", min=4, max=31)},))
    converted = m2m.convert(exported, (torch.arange(12),), backend="fx_importer", capture_trace=True)
    assert converted.ok and converted.capture_trace["status"] == "complete"
    source = converted.capture_trace["graphs"]["quantized"]
    prepared = copy.deepcopy(converted.capture_trace["graphs"]["prepared"])
    addition = next(node for node in prepared["nodes"] if node["target"] == "<built-in function add>" and 1 in node["args"])
    addition["args"] = [2 if value == 1 else value for value in addition["args"]]
    assert graph_relation(source, prepared)["status"] == "diagnostic"


def test_non_dense_mutable_base_requires_an_explicit_storage_layout_contract():
    class Model(torch.nn.Module):
        def forward(self, x):
            return pair(x[:, 1], x[:, -1])
    converted = m2m.convert(Model(), (torch.arange(24).reshape(4, 6).T,), backend="fx_importer")
    assert not converted.ok
    assert any("proved dense source layout" in diagnostic for diagnostic in converted.diagnostics)


def test_dynamic_public_state_keeps_actual_dynamic_tensor_types():
    exported = torch.export.export(Chain("public"), (torch.arange(5),),
        dynamic_shapes=({0: torch.export.Dim("length", min=2, max=20)},))
    converted = m2m.convert(exported, (torch.arange(5),), backend="fx_importer", capture_trace=True)
    assert converted.ok, converted.diagnostics
    contract = json.loads(entry(converted).attributes[ATTRIBUTE].data)
    source = entry(converted).function_type.inputs.data[contract["state_outputs"][0]["input"]]
    assert source.get_shape()[0] < 0 and entry(converted).function_type.outputs.data[0] == source


def test_bundle_rejects_forged_entry_state_roles(tmp_path):
    model, x = Chain("public"), torch.arange(5)
    converted = m2m.convert(model, (x,), backend="fx_importer", capture_trace=True,
                            weights_path=str(tmp_path / "weights.safetensors"))
    contract = json.loads(entry(converted).attributes[ATTRIBUTE].data)
    contract["state_outputs"][0]["input"] = 0
    entry(converted).attributes[ATTRIBUTE] = StringAttr(json.dumps(contract, sort_keys=True))
    with pytest.raises(ValueError, match="state contract differs"):
        write_bundle(model, (x,), tmp_path, capture_regions=False, capture_trace=True, conversion_result=converted)


def test_eager_snapshot_preserves_input_aliases_and_restores_buffer_after_failure():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(2))

        def forward(self, x, view):
            x.add_(4)
            self.state.add_(1)
            return view.clone()
    model = Model()
    x = torch.arange(7)
    seen = observe(model, (x, x[1::2]))
    assert torch.equal(seen.output, torch.tensor([5, 7, 9]))
    assert torch.equal(x, torch.arange(7)) and model.state.item() == 2
    assert seen.buffers["state"].item() == 3
    def fail(_arguments):
        model.state.add_(10)
        raise RuntimeError("reference failure")
    with pytest.raises(RuntimeError, match="reference failure"):
        observe(model, (x,), fail)
    assert model.state.item() == 2


@pytest.mark.parametrize("damage", ("missing-return", "clone-override", "duplicate-base", "changed-base-type"))
def test_functional_decoder_refuses_unproved_result_or_base_contracts(damage):
    from xdsl.ir import Block

    from m2m.ir.functional_kernels import emit
    from m2m.ir.import_fx import _tensor_type_from_meta
    from m2m.ir.kernel_registry import KernelLoweringError, functional_kernel

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return pair(x, y)
    prepared = torch.export.export(Model(), (torch.arange(3), torch.arange(4))).run_decompositions({})
    node = next(node for node in prepared.graph.nodes if node.op == "call_function" and functional_kernel(node))
    bases = node.kwargs["_all_bases"]
    block = Block(arg_types=[_tensor_type_from_meta(base.meta["val"]) for base in bases])
    values = dict(zip((base.name for base in bases), block.args, strict=True))
    if damage == "missing-return":
        node.meta["val"] = (None, *node.meta["val"][1:])
    elif damage == "changed-base-type":
        node.meta["val"] = (*node.meta["val"][:-1], torch.ones(9))
    else:
        kwargs = dict(node.kwargs)
        kwargs["_only_clone_these_bases" if damage == "clone-override" else "_right_base_index"] = [] if damage == "clone-override" else 0
        node.kwargs = kwargs
    with pytest.raises(KernelLoweringError):
        emit(KERNEL_REGISTRY["m2m_state_test::pair"], node, values, _tensor_type_from_meta)
