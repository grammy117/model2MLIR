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
`range_constraints`. These observations alone grant no executable shape fact.
The entry-domain expansion below supplies real guards. A traced
bundle's `input_abi`/`output_abi` and saved arrays still describe its concrete
reference invocation; they do not specialize the symbolic MLIR or certify a
general dynamic consumer ABI. Receipt schemas and artifact rosters are unchanged.

`tests/test_symbolic_dimensions.py` covers backed symbols without new guards,
literal symbols, ordinary integer dimensions, real registered calls, typed
serialization, source ranges and all three saved result references. Its eager
exported model also runs independent sizes within the declared range. This is
frontend/capture evidence; no custom-kernel implementation or compiled dynamic
model execution is claimed by those tests.

## Executable entry domains

`m2m.ir.shape_domains` reads actual symbolic expressions and ValueRanges from
the exported program. It emits `tensor.dim`, standard `arith.index_castui`,
integer arithmetic/comparisons and `cf.assert` before model computation.
Unsigned casts have an independent native-width round-trip check and a signed
i64 logical-size bound. Shape arithmetic uses interval-sized integers so an
intermediate coefficient/product cannot silently wrap at the size width.
Scalar cast bindings retain the native optional `nonNeg` promise; domain
expansion does not invent that promise.

Direct input dimensions establish symbolic roots. Shared symbols get actual
dimension-equality checks. Invertible affine dimensions can establish a root
without an observed direct axis, with divisibility checks before inversion.
Range and final relation checks retain actual source expressions. Expressions
over established roots include integer polynomials and positive-constant floor
division. Ambiguous roots or unsupported expressions refuse explicitly.
Data-dependent body ranges remain body obligations rather than guessed entry
values. Printed constraint strings, example hints and evaluator arrays never
produce these operations. The advisory `prov.shape_domain` entry annotation
records their origin; deleting it grants no permission to delete guards.

The finite analysis budgets are 256 symbolic input axes, 8192 expression visits
and 4096-bit arithmetic. Exceeding a budget refuses conversion. These are
resource bounds, not target-specific index widths or arbitrary source coverage.
The source index width remains undecided. Runtime failures use the existing
native assertion mechanism; Python exception text is not reproduced.

Channel sums additionally retain a dynamic reduced axis when every retained
output axis is static and the existing dtype rules apply. Original Linalg
reduction bodies and input-dependent loop extents remain executable. This
supports a real `x.sum(0) + y.sum(0)` model with shared, affine and inverse-derived
input sizes. Independent native tests in triton2MLIR execute valid sizes and
actual invalid-domain aborts; they do not qualify a generic dynamic-result ABI.

Consumers must separately check source shape-domain relations, representable
index/scalar conversions, runtime allocation/result dimensions and their
invocation interface. In particular, the triton2MLIR live-argument profile for
static public models does not admit unknown symbolic dimensions. Complete
symbolic model binding and Merlin execution remain separate integration work.
