"""BF16 exact storage remains separate from legacy numeric archives."""

import json

import numpy as np
import pytest
import torch

from m2m.capture.bundle import _numpy_safe, write_bundle
from m2m.capture.provenance import capture_receipt
from m2m.capture.tensor_words import FILENAME, TensorWordArchives, checked_tensor_words


def tensor(words):
    return torch.from_numpy(np.asarray(words, np.uint16).view(np.int16)).view(torch.bfloat16)


def fixture(out):
    value = tensor([0, 0x8000, 1, 0x807f, 0x3f80, 0x7f80, 0x7f81, 0xffa5])
    words = TensorWordArchives()
    legacy = _numpy_safe(value, word_archives=words, archive="inputs.npz", key="in0")
    np.savez(out / "inputs.npz", in0=legacy)
    words.write(out)
    return value


def test_all_bfloat_words_are_captured_without_numeric_conversion(tmp_path):
    value = tensor(np.arange(65536, dtype=np.uint16))
    words = TensorWordArchives()
    legacy = _numpy_safe(value, word_archives=words, archive="golden.npy")
    np.save(tmp_path / "golden.npy", legacy)
    words.write(tmp_path)
    assert legacy.dtype == np.float32
    with np.load(tmp_path / FILENAME, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["word0"], np.arange(65536, dtype=np.uint16))
    assert checked_tensor_words(tmp_path)["entries"][0]["dtype"] == "bf16"


@pytest.mark.parametrize("shape", [(), (3, 2)])
def test_scalar_and_noncontiguous_sources_keep_logical_shape_and_words(tmp_path, shape):
    value = tensor(0x7f81) if shape == () else tensor([1, 0x8000, 0x7f81, 0xffa5, 0, 0x3f80]).reshape(2, 3).T
    words = TensorWordArchives()
    np.save(tmp_path / "golden.npy", _numpy_safe(value, word_archives=words, archive="golden.npy"))
    words.write(tmp_path)
    with np.load(tmp_path / FILENAME, allow_pickle=False) as saved:
        assert saved["word0"].shape == shape
        np.testing.assert_array_equal(saved["word0"], value.contiguous().view(torch.int16).numpy().view(np.uint16))


@pytest.mark.parametrize("change", ["schema", "path", "missing-word", "extra-word", "word-type", "word-shape",
    "word-values", "binding", "duplicate", "undeclared", "missing-abi-member"])
def test_word_archive_refuses_incomplete_or_inconsistent_bindings(tmp_path, change):
    fixture(tmp_path)
    path = tmp_path / "meta.json"
    meta = json.loads(path.read_text())
    pointer = meta["tensor_words"]
    with np.load(tmp_path / FILENAME, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if change == "schema": pointer["schema"] = "unknown"
    elif change == "path": pointer["path"] = "../other.npz"
    elif change == "missing-word": arrays.clear()
    elif change == "extra-word": arrays["extra"] = arrays["word0"].copy()
    elif change == "word-type": arrays["word0"] = arrays["word0"].astype(np.float32)
    elif change == "word-shape": arrays["word0"] = arrays["word0"].reshape(2, 4)
    elif change == "word-values": arrays["word0"][0] = 0x3f80
    elif change == "binding": pointer["entries"][0]["archive"] = "weights.safetensors"
    elif change == "duplicate":
        pointer["entries"].append({**pointer["entries"][0], "word_key": "word1"})
        arrays["word1"] = arrays["word0"].copy()
    elif change == "undeclared": meta.pop("tensor_words")
    elif change == "missing-abi-member": meta["output_abi"] = [{"dtype": "bf16", "shape": [8]}]
    path.write_text(json.dumps(meta))
    np.savez(tmp_path / FILENAME, **arrays)
    with pytest.raises(ValueError):
        checked_tensor_words(tmp_path)


def test_non_bfloat_capture_does_not_add_an_archive_or_rewrite_metadata(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text('{"original": true}')
    words = TensorWordArchives()
    _numpy_safe(torch.tensor([1.], dtype=torch.float16), word_archives=words, archive="inputs.npz", key="in0")
    words.write(tmp_path)
    assert not (tmp_path / FILENAME).exists() and path.read_text() == '{"original": true}'


def test_first_result_alias_preserves_nan_words_not_only_classification(tmp_path):
    words = TensorWordArchives()
    value = tensor([0x7f81])
    first = _numpy_safe(value, word_archives=words, archive="golden.npy")
    repeated = _numpy_safe(value, word_archives=words, archive="goldens.npz", key="out0")
    np.save(tmp_path / "golden.npy", first)
    np.savez(tmp_path / "goldens.npz", out0=repeated)
    words.arrays["word1"][0] = 0x7fc2
    with pytest.raises(ValueError, match="legacy first result"):
        words.write(tmp_path)


def test_real_bundle_preserves_inputs_state_buffers_and_every_result(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(tensor([1, 0x7f81, 0x8000]), requires_grad=False)
            self.register_buffer("snapshot", tensor([0xffa5, 0x807f]))

        def forward(self, x):
            return x.clone(), self.state.clone(), self.snapshot.clone()

    model, values = Model().eval(), (tensor([0, 1, 0x7f81, 0xffa5, 0x8000]),)
    out = tmp_path / "capture"
    write_bundle(model, values, out, capture_regions=False, capture_trace=True)
    receipt = capture_receipt(out)
    assert FILENAME in receipt["artifacts"] and receipt["materialized_abi"]["outputs"] == 3
    pointer = checked_tensor_words(out)
    expected = {("inputs.npz", "in0"): values[0], ("golden.npy", None): values[0],
        ("extra.npz", "buf::snapshot"): model.snapshot,
        **{("goldens.npz", f"out{i}"): value for i, value in enumerate(model(*values))}}
    with np.load(out / FILENAME, allow_pickle=False) as saved:
        for entry in pointer["entries"]:
            value = expected[(entry["archive"], entry["key"])]
            np.testing.assert_array_equal(saved[entry["word_key"]], value.view(torch.int16).numpy().view(np.uint16))
