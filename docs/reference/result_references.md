# Flat tensor result references

`write_bundle` retains `golden.npy` as the original first eager model result.
For a flat tuple/list of two or more tensors it additionally writes
`goldens.npz`, keyed `out0`, `out1`, … in exported result order. Every array
uses the existing `_numpy_safe` policy: representable dtypes/words stay exact;
the existing bfloat16 storage conversion remains explicit in the source ABI
and recorded NumPy storage dtype.

`meta.json.result_references` declares
`m2m.flat-tensor-result-references.v1`, the archive path, ordered keys, storage
dtypes and the legacy first-result member. The capture receipt binds the
actual archive bytes and records its output count. Receipt construction checks
the complete key set, first-result consistency, storage dtypes and any captured
output shapes. Reference values are evaluator data; consumers must exclude
them and their digests from compiler inputs.

This is an additive optional member under `m2m.capture-receipt.v1`. Existing
single-result captures retain their original golden layout. Old tuple captures
without the archive establish only their existing first-result reference;
consumers must not invent goldens for omitted results. Nested/non-tensor
Python result structures require a separate explicit ABI contract.

The references describe one observed eager execution. Byte custody is not a
claim of source closure, compiled equivalence or all permitted outcomes of a
nondeterministic kernel.
