"""Receipt declarations for actual exported constants in compiler-owned storage."""

from pathlib import Path
import json

from m2m.capture.trace import _tensor_bytes_sha256

SCHEMA = "m2m.compiler-constants.v1"


def checked_compiler_constants(out, manifest, lifted):
    """Bind complete constant arguments to exact lossless safetensors payloads.

    Legacy extra entries stay available to evaluators. Their presence alone
    never supplies a compiler constant binding.
    """
    entries = [(key, entry) for key, entry in manifest.items() if entry.get("kind") == "constant"]
    if not any(entry.get("weight") for _, entry in entries):
        return None
    from safetensors import safe_open
    bindings, names = [], set()
    trace_path = Path(out) / "frontend-trace.json"
    prepared = json.loads(trace_path.read_text()).get("graphs", {}).get("prepared", {}) if trace_path.is_file() else None
    placeholders = [node for node in prepared.get("nodes", [])
                    if node.get("op") == "placeholder" and node.get("graph_id") == prepared.get("graph_id")] if prepared else None
    with safe_open(Path(out) / "weights.safetensors", framework="pt") as weights:
        for index, entry in sorted(entries, key=lambda item: int(item[0])):
            name, target, key = entry.get("name"), entry.get("source_target"), entry.get("weight")
            if (str(int(index)) != index or not isinstance(name, str) or not name
                    or name in names or name not in lifted or not isinstance(target, str) or not target
                    or not isinstance(key, str) or key not in weights.keys()
                    or "error" in entry or entry.get("stub", False) is not False):
                raise ValueError("compiler constants require complete unique exported argument bindings")
            value = weights.get_tensor(key)
            if placeholders is not None:
                if int(index) >= len(placeholders):
                    raise ValueError("compiler constant index is absent from the actual source graph")
                node = placeholders[int(index)]
                result = node.get("results", [])
                if (node.get("input_kind") != "CONSTANT_TENSOR" or node.get("target") != name
                        or node.get("input_target") != target or len(result) != 1 or result[0].get("kind") != "tensor"
                        or result[0].get("shape") != entry.get("shape") or result[0].get("dtype") != entry.get("dtype")):
                    raise ValueError("compiler constant binding differs from the actual exported source")
            if (list(value.shape) != entry.get("shape") or str(value.dtype).removeprefix("torch.") != entry.get("dtype")
                    or _tensor_bytes_sha256(value) != entry.get("sha256") or lifted[name]["shape"] != list(value.shape)):
                raise ValueError("compiler constant payload differs from its captured declaration")
            names.add(name)
            bindings.append({"argument": int(index), "name": name, "target": target, "weight": key,
                             "dtype": entry["dtype"], "shape": entry["shape"], "sha256": entry["sha256"]})
    if names != set(lifted):
        raise ValueError("compiler constant profile must cover every lifted argument")
    return {"schema": SCHEMA, "storage": "weights.safetensors", "bindings": bindings}
