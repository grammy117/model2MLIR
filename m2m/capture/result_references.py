"""Capture and validate complete flat tensor result references, outside compiler inputs."""

import json
from pathlib import Path

import numpy as np

SCHEMA = "m2m.flat-tensor-result-references.v1"
FILENAME = "goldens.npz"


def write_result_references(out, arrays):
    """Keep the legacy first result and add ordered references for tensor tuples."""
    out = Path(out)
    path = out / FILENAME
    meta_path = out / "meta.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if len(arrays) <= 1:
        path.unlink(missing_ok=True)  # Reserved generated member from a previous capture.
        if "result_references" not in metadata:
            return
        metadata.pop("result_references", None)
    else:
        if any(a.dtype.hasobject for a in arrays):
            raise ValueError("tensor result references cannot contain object arrays")
        keys = [f"out{i}" for i in range(len(arrays))]
        np.savez(path, **dict(zip(keys, arrays, strict=True)))
        metadata["result_references"] = {"schema": SCHEMA, "path": FILENAME, "keys": keys,
            "storage_dtypes": [str(a.dtype) for a in arrays], "legacy_first_result": "golden.npy"}
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def checked_result_references(out):
    """Validate observed reference order/storage; this supplies no source equivalence."""
    out = Path(out)
    metadata = json.loads((out / "meta.json").read_text()) if (out / "meta.json").is_file() else {}
    pointer = metadata.get("result_references")
    path = out / FILENAME
    if pointer is None and not path.exists():
        return None
    if (not isinstance(pointer, dict) or pointer.get("schema") != SCHEMA or pointer.get("path") != FILENAME
            or pointer.get("legacy_first_result") != "golden.npy" or not path.is_file() or path.is_symlink()
            or not isinstance(pointer.get("keys"), list) or len(pointer["keys"]) < 2
            or pointer["keys"] != [f"out{i}" for i in range(len(pointer["keys"]))]
            or not isinstance(pointer.get("storage_dtypes"), list) or len(pointer["storage_dtypes"]) != len(pointer["keys"])):
        raise ValueError("flat tensor result references lack a complete ordered capture declaration")
    with np.load(path, allow_pickle=False) as references:
        keys = pointer["keys"]
        if len(references.files) != len(keys) or set(references.files) != set(keys):
            raise ValueError("flat tensor result references do not contain every declared result exactly once")
        first = np.load(out / "golden.npy", allow_pickle=False)
        if (references["out0"].shape != first.shape or references["out0"].dtype != first.dtype
                or references["out0"].tobytes() != first.tobytes()):
            raise ValueError("flat tensor result references differ from the legacy first golden")
        for key, dtype in zip(keys, pointer["storage_dtypes"], strict=True):
            if str(references[key].dtype) != dtype:
                raise ValueError("flat tensor result reference storage dtype differs from its capture")
        abi = metadata.get("output_abi")
        if abi is not None and (len(abi) != len(keys) or any(list(references[key].shape) != item.get("shape")
                for key, item in zip(keys, abi, strict=True))):
            raise ValueError("flat tensor result reference shapes differ from the captured output ABI")
    return pointer
