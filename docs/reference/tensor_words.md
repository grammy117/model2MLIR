# Exact BF16 tensor-word archives

`write_bundle` preserves legacy `inputs.npz`, `extra.npz`, `golden.npy` and
`goldens.npz` storage. Their BF16 values remain widened F32. It additionally
writes `tensor_words.npz` when any archived source tensor is BF16. Each member
is a raw uint16 array obtained by an equal-width tensor view, with no floating
conversion. This preserves signed zeros, subnormals and signaling-NaN payloads.
Safetensors weights already retain original BF16 bytes and do not need a second
copy in this archive.

`meta.json.tensor_words` declares schema `m2m.tensor-word-archives.v1`, path
`tensor_words.npz`, and ordered `entries`. Each entry contains `archive`, `key`,
`word_key`, `dtype: bf16`, `storage_dtype: uint16`, and `shape`. `word_key` is
`word0`, `word1`, …; `key` is null for `golden.npy` and otherwise exactly the
original archive key. Entries cover BF16 inputs, registered buffers, lifted
constants, tensor-subclass leaves and every flat tensor result. The first
result's legacy/tuple aliases retain identical raw words.

Receipt construction validates the full inventory, unique bindings, declared
types/shapes, agreement with finite legacy values including signed zero,
NaN classification, and observed BF16 input/output ABI completeness. The
`m2m.capture-receipt.v1` artifact map binds the actual word-archive bytes.
These are observed source values, not operator/effect or equivalence proofs.
Consumers must exclude this evaluator archive and its digests from compiler
inputs, and must match a word entry to the logical BF16 ABI before using it.

This is an additive optional artifact. Non-BF16 captures retain their existing
layout. Older BF16 captures with widened values alone cannot recover original
signaling-NaN words; consumers must not infer those words from a value conversion.
Other low-precision formats and non-flat result schemas require their own
explicit storage contracts.
