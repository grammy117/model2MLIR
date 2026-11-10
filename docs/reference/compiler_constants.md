# Lifted source constants in compiler storage

Exported `CONSTANT_TENSOR` placeholders now keep their exact tensors in the
existing `weights.safetensors` file. Their public argument positions and MLIR
types are unchanged. Manifest entries use `kind: constant`, the actual
placeholder `name`, `source_target`, storage `weight` key, dtype, shape and a
SHA-256 of the dense logical tensor bytes. BF16 and integer data never pass
through a floating conversion. Constants cannot use quantized parameter stubs.

`capture_receipt.json.compiler_constants` is an optional
`m2m.compiler-constants.v1` object with `storage: weights.safetensors` and ordered
`bindings`. Each binding contains `argument`, `name`, `target`, `weight`,
`dtype`, `shape` and `sha256`. Receipt construction checks complete coverage,
unique names, actual safetensors bytes and the traced source's constant
placeholder role, target, type and shape when a trace is present.

The existing `extra.npz` evaluator copy and `lifted_constants` receipt ledger
remain. Their keys now follow actual exported placeholder names. Default
`c_lifted_tensor_<i>` names retain that layout; attribute constants such as
`c_literal` use their real names rather than an invented ordinal alias. The
runtime input-order table excludes constant arguments independently of naming.
BF16 extras may retain widened value arrays plus their separate exact-word
archive, while compiler storage retains the original BF16 bytes.

This is an additive source-storage profile under `m2m.capture-receipt.v1`.
Compiler consumers check it against the typed source graph and only read
compiler-owned weights/metadata; they must not read evaluator extras or goldens
to fill missing constants. Older captures without this profile retain their
existing receipt semantics and require an explicit compiler constant binding.
These records establish byte custody and source correspondence, not numerical,
effect, alias, packing or target-execution authority.
