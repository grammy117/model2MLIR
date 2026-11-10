"""Exact BF16 storage witnesses alongside unchanged legacy value archives.

NumPy cannot carry torch.bfloat16 directly. Widened FP32 values remain useful
references, but do not identify the original signaling-NaN words. This optional
evaluator artifact records source bytes without floating-point conversion.
"""

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

FILENAME = "tensor_words.npz"
SCHEMA = "m2m.tensor-word-archives.v1"
ARCHIVES = {"inputs.npz", "extra.npz", "golden.npy", "goldens.npz"}


class TensorWordArchives:
    def __init__(self):
        self.entries = []
        self.arrays = {}

    def add(self, archive, key, tensor):
        import torch
        if tensor.dtype != torch.bfloat16:
            return
        if archive not in ARCHIVES or (key is None) != (archive == "golden.npy"):
            raise ValueError("exact tensor words require a known archive/key binding")
        if any(entry["archive"] == archive and entry["key"] == key for entry in self.entries):
            raise ValueError("duplicate exact tensor-word binding")
        word_key = f"word{len(self.entries)}"
        words = tensor.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16).copy()
        self.arrays[word_key] = words
        self.entries.append({"archive": archive, "key": key, "word_key": word_key,
            "dtype": "bf16", "storage_dtype": "uint16", "shape": list(words.shape)})

    def write(self, out):
        out = Path(out)
        path, metadata_path = out / FILENAME, out / "meta.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if self.entries:
            np.savez(path, **self.arrays)
            metadata["tensor_words"] = {"schema": SCHEMA, "path": FILENAME, "entries": self.entries}
        else:
            if not path.exists() and "tensor_words" not in metadata:
                return
            path.unlink(missing_ok=True)  # Reserved generated member from a previous capture.
            metadata.pop("tensor_words", None)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        checked_tensor_words(out)


def checked_tensor_words(out):
    """Check exact archive bindings and agreement with legacy numeric values."""
    out = Path(out)
    metadata = json.loads((out / "meta.json").read_text()) if (out / "meta.json").is_file() else {}
    pointer, path = metadata.get("tensor_words"), out / FILENAME
    if pointer is None and not path.exists():
        return None
    if (not isinstance(pointer, dict) or set(pointer) != {"schema", "path", "entries"}
            or pointer.get("schema") != SCHEMA or pointer.get("path") != FILENAME
            or not path.is_file() or path.is_symlink()
            or not isinstance(pointer.get("entries"), list) or not pointer["entries"]):
        raise ValueError("exact tensor words lack a complete archive declaration")
    bindings = set()
    with ExitStack() as stack:
        saved = stack.enter_context(np.load(path, allow_pickle=False))
        if len(saved.files) != len(pointer["entries"]) or set(saved.files) != {f"word{i}" for i in range(len(pointer["entries"]))}:
            raise ValueError("exact tensor-word inventory differs from its declaration")
        archives, first_results = {}, {}
        for i, entry in enumerate(pointer["entries"]):
            if (not isinstance(entry, dict) or set(entry) != {"archive", "key", "word_key", "dtype", "storage_dtype", "shape"}
                    or not isinstance(entry["archive"], str) or entry["archive"] not in ARCHIVES or entry["word_key"] != f"word{i}"
                    or entry["dtype"] != "bf16" or entry["storage_dtype"] != "uint16"
                    or not isinstance(entry["shape"], list) or any(type(n) is not int or n < 0 for n in entry["shape"])
                    or (entry["key"] is None) != (entry["archive"] == "golden.npy")
                    or entry["key"] is not None and (not isinstance(entry["key"], str) or not entry["key"])):
                raise ValueError("invalid exact tensor-word archive binding")
            binding = (entry["archive"], entry["key"])
            if binding in bindings:
                raise ValueError("duplicate exact tensor-word archive binding")
            bindings.add(binding)
            archive_path = out / entry["archive"]
            if archive_path.is_symlink() or not archive_path.is_file():
                raise ValueError("exact tensor words lack their original legacy archive")
            if entry["archive"] not in archives:
                archive = np.load(archive_path, allow_pickle=False)
                archives[entry["archive"]] = archive if entry["archive"] == "golden.npy" else stack.enter_context(archive)
            archive = archives[entry["archive"]]
            legacy = archive if entry["key"] is None else archive[entry["key"]]
            words = saved[entry["word_key"]]
            if binding in (("golden.npy", None), ("goldens.npz", "out0")):
                first_results[binding] = words
            if (words.dtype != np.uint16 or list(words.shape) != entry["shape"]
                    or legacy.dtype != np.float32 or legacy.shape != words.shape):
                raise ValueError("exact tensor words differ from declared storage/legacy shape")
            bits = words.astype(np.uint32) << 16
            nan = (bits & 0x7fffffff) > 0x7f800000
            if (not np.array_equal(np.isnan(legacy), nan)
                    or not np.array_equal(legacy.view(np.uint32)[~nan], bits[~nan])):
                raise ValueError("exact tensor words disagree with legacy numeric values")
        for archive, prefix, abi in (("inputs.npz", "in", metadata.get("input_abi", [])),
                                     ("goldens.npz", "out", metadata.get("output_abi", []))):
            for i, item in enumerate(abi):
                if item.get("dtype") != "bf16":
                    continue
                required = {(archive, f"{prefix}{i}")} if archive == "inputs.npz" or len(abi) > 1 else set()
                if archive == "goldens.npz" and i == 0:
                    required.add(("golden.npy", None))
                if not required <= bindings:
                    raise ValueError("exact tensor words omit a declared BF16 ABI member")
        if len(first_results) == 2 and not np.array_equal(*first_results.values()):
            raise ValueError("exact tensor words disagree for the legacy first result")
    return pointer
