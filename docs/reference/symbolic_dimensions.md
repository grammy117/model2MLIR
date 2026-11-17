# Symbolic dimensions at the FX importer boundary

The FX importer and decomposition shape helpers distinguish an actual integer
literal from an exported symbolic dimension. A backed `torch.SymInt` can have
an example-size hint while still representing a range of values. Calling
`int()` on that object can introduce a specialization guard; the hint is not a
literal source dimension.

`m2m.ir.shapes.coerce_dimension` uses PyTorch's
`torch.fx.experimental.symbolic_shapes.is_concrete_int` before converting a
`SymInt`. Nonconstant symbols retain xDSL's `DYNAMIC_INDEX`, printed as `?`.
Integer literals, including literal `SymInt` expressions, remain concrete.
Legacy static-only decomposition checks retain their internal `-1` unknown
marker; this is never substituted for xDSL's dynamic tensor-type sentinel.
An unsupported dynamic decomposition keeps its existing refusal/opaque path
rather than specializing the symbol to make a static emitter applicable.

The implementation was inspected and exercised with PyTorch 2.10.0, revision
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`. The experimental helper is an upstream
integration dependency, with direct symbolic/literal regression coverage; it
does not provide a version-independent shape solver. See the pinned
[PyTorch implementation](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/fx/experimental/symbolic_shapes.py).

## Kernel calls and capture artifacts

A symbolic export such as:

```python
class Model(torch.nn.Module):
    def forward(self, x):
        first = copy_kernel(x, x.shape[0])
        second = copy_kernel(first, first.shape[0])
        return second, first, x

program = torch.export.export(
    Model(), (torch.empty(13, 3, dtype=torch.int32),),
    dynamic_shapes={"x": {0: torch.export.Dim("n", min=2, max=31)}},
)
```

keeps `tensor<?x3xi32>` in the public function, both registered kernel
declarations/calls and every model result. Actual `aten.sym_size.int` produces
standard `tensor.dim` SSA; source export may share that value between calls.
Dependent tensor SSA and complete tuple order remain unchanged. No sample
array, provenance label or kernel identity supplies a dimension constant.

The existing frontend trace retains original symbolic shapes and
`range_constraints`. These are captured source observations, not an executable
shape guard or permission to invoke outside the exported domain. A traced
bundle's `input_abi`/`output_abi` and saved arrays still describe its concrete
reference invocation; they do not specialize the symbolic MLIR or certify a
general dynamic consumer ABI. Receipt schemas and artifact rosters are unchanged.

`tests/test_symbolic_dimensions.py` covers backed symbols without new guards,
literal symbols, ordinary integer dimensions, real registered calls, typed
serialization, source ranges and all three saved result references. Its eager
exported model also runs independent sizes within the declared range. This is
frontend/capture evidence; no custom-kernel implementation or compiled dynamic
model execution is claimed by those tests.

Consumers must separately check source shape-domain relations, representable
index/scalar conversions, runtime allocation/result dimensions and their
invocation interface. In particular, the triton2MLIR live-argument profile for
static public models does not admit unknown symbolic dimensions. Complete
symbolic model binding and Merlin execution remain separate integration work.
