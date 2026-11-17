"""Actual symbolic numel decompositions need complete source dimension trees."""

from copy import deepcopy

import pytest
import torch

import m2m
from m2m.capture.trace import graph_relation


@pytest.fixture(scope="module")
def trace():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + x.numel(), y

    shape = {
        0: torch.export.Dim("rows", min=2, max=4),
        1: torch.export.Dim("cols", min=2, max=7),
    }
    inputs = (torch.ones(3, 5), torch.ones(4, 6))
    other = {
        0: torch.export.Dim("other_rows", min=2, max=6),
        1: torch.export.Dim("other_cols", min=2, max=8),
    }
    exported = torch.export.export(
        Model(), inputs, dynamic_shapes={"x": shape, "y": other}
    )
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace[
        "blockers"
    ]
    return result.capture_trace


def test_real_numel_decomposition_accounts_for_actual_dimension_reads(trace):
    relation = graph_relation(trace["graphs"]["quantized"], trace["graphs"]["prepared"])
    assert relation["status"] == "complete"
    dimensions = [
        row for row in relation["relations"] if row["kind"] == "numel_dimension_queries"
    ]
    assert len(dimensions) == 1
    assert [row["axis"] for row in dimensions[0]["proof"]["queried_axes"]] == [0, 1]
    assert dimensions[0]["proof"]["complete_dimension_product"] is True


@pytest.mark.parametrize("kind", ("shared-input-symbols", "square", "static-factor"))
def test_exported_shape_equalities_and_static_factors_survive_real_cse(kind):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + x.numel(), y

    rows = torch.export.Dim("shared_rows", min=2, max=4)
    cols = rows if kind == "square" else torch.export.Dim("shared_cols", min=2, max=7)
    shape = {0: rows, 1: cols} if kind != "static-factor" else {1: cols}
    inputs = (torch.ones(3, 3),) * 2 if kind == "square" else (torch.ones(3, 5),) * 2
    program = torch.export.export(
        Model(), inputs, dynamic_shapes={"x": shape, "y": shape}
    )
    result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok and result.capture_trace["status"] == "complete", (
        result.capture_trace["blockers"]
    )
    records = result.capture_trace["transformations"][1]["relations"]
    assert any(
        row["kind"] == "numel_dimension_queries"
        and row["proof"]["complete_dimension_product"]
        for row in records
    )


@pytest.mark.parametrize(
    "damage",
    ("axis", "root", "reference", "arithmetic", "scope", "result-kind", "kwargs"),
)
def test_unproved_dimension_products_remain_diagnostic(trace, damage):
    source, dest = trace["graphs"]["quantized"], deepcopy(trace["graphs"]["prepared"])
    queries = [node for node in dest["nodes"] if node["target"] == "aten.sym_size.int"]
    query = queries[-1]
    if damage == "axis":
        query["args"][1] = queries[0]["args"][1]
    elif damage == "root":
        other = next(
            node
            for node in dest["nodes"]
            if node["op"] == "placeholder" and node["id"] != query["args"][0]["node_id"]
        )
        query["args"][0] = {
            "node_id": other["id"],
            "value_id": other["results"][0]["id"],
        }
    elif damage == "reference":
        query["args"][0]["value_id"] += "_missing"
    elif damage == "arithmetic":
        node = next(
            node
            for node in dest["nodes"]
            if node["target"] == "<built-in function mul>"
        )
        node["target"] = "<built-in function add>"
    elif damage == "scope":
        query["graph_id"] += "_different"
    elif damage == "result-kind":
        query["results"][0]["kind"] = "tensor"
    else:
        query["kwargs"] = {"unknown": True}
    assert graph_relation(source, dest)["status"] == "diagnostic"
