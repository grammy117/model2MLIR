"""Lifted source values keep exact compiler storage and original ABI positions."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from m2m.capture.bundle import write_bundle
from m2m.capture.provenance import capture_receipt


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int8, torch.int16, torch.int32, torch.int64, torch.bool])
def test_source_constant_is_externalized_losslessly_at_its_actual_argument(tmp_path, dtype):
    value = torch.arange(6).reshape(2, 3).T.to(dtype)
    if dtype == torch.bfloat16:
        value.view(torch.int16)[0, 0] = 0x7f81

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.literal = value

        def forward(self, x):
            return x.clone(), self.literal.clone()

    write_bundle(Model().eval(), (torch.arange(2, dtype=torch.float32),), tmp_path,
                 capture_regions=False, capture_trace=True)
    receipt = capture_receipt(tmp_path)
    profile = receipt["compiler_constants"]
    assert profile["schema"] == "m2m.compiler-constants.v1" and profile["storage"] == "weights.safetensors"
    (binding,) = profile["bindings"]
    assert binding["name"] == "c_literal" and binding["target"] == "literal"
    assert set(receipt["lifted_constants"]) == {"c_literal"}
    manifest = json.loads((tmp_path / "weights.safetensors.manifest.json").read_text())
    assert manifest[str(binding["argument"])]["kind"] == "constant"
    saved = load_file(str(tmp_path / "weights.safetensors"))[binding["weight"]]
    assert saved.dtype == value.dtype and saved.shape == value.shape
    assert saved.contiguous().view(torch.uint8).numpy().tobytes() == value.contiguous().view(torch.uint8).numpy().tobytes()
    with np.load(tmp_path / "extra.npz", allow_pickle=False) as extra:
        assert extra["c_literal"].shape == (3, 2)


@pytest.mark.parametrize("change", ["target", "name", "dtype", "shape", "hash", "weight", "index"])
def test_rehashed_manifest_cannot_invent_constant_source_bindings(tmp_path, change):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.literal = torch.tensor([1., -0.])
        def forward(self, x):
            return x + self.literal
    write_bundle(Model().eval(), (torch.tensor([2., 3.]),), tmp_path, capture_regions=False, capture_trace=True)
    path = tmp_path / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_text())
    index = next(k for k, value in manifest.items() if value["kind"] == "constant")
    entry = manifest[index]
    if change == "target": entry["source_target"] = "different_source"
    elif change == "name": entry["name"] = "different_placeholder"
    elif change == "dtype": entry["dtype"] = "float64"
    elif change == "shape": entry["shape"] = [1, 2]
    elif change == "hash": entry["sha256"] = "0" * 64
    elif change == "weight": entry["weight"] = "missing"
    elif change == "index": manifest[str(int(index) + 99)] = manifest.pop(index)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        capture_receipt(tmp_path)
