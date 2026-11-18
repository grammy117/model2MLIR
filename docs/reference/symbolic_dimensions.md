# Symbolic dimensions at the FX importer boundary

Literal `aten.select.int` now supports actual dynamic dimensions and scalar
results. Negative indices normalize against the selected source extent;
executable bounds precede index transport and extraction. Static invalid indices
refuse the lowering, rather than being clamped. Only the selected size-one axis
is removed. Other unit axes use explicit dynamic constant-one sizes and a
compatible result cast, preserving their identity through adjacent unit axes.
The empty-slice refinement below also handles zero-sized retained dimensions.
LLVM `getDroppedDims` and its `rank_reducing_slice_preserve_dropped_dims` fixture
define the actual reduction map; no inferred physical alias permission follows.

Positive-step runtime slices keep extraction in a standard `scf.if` nonempty
branch and return a `tensor.empty` with the same actual sizes otherwise.
This preserves clipped empty slices at the source boundary without executing
an out-of-bounds subview. Both branches retain exact storage and result types;
offset/size/stride SSA stays outside the conditional for source correspondence.
The selected LLVM `verifyInBoundsSlice` rejects static offsets equal to the
source extent even for empty results. Unreachable extraction branches on
statically empty axes use explicit dynamic zero offsets and remain native
verifiable. Original clipping and size-domain checks remain executable.

`aten.permute`, `aten.t` and `aten.transpose.int` now size their standard
`linalg.transpose` destinations from the actual permuted source axes. Each
dynamic output extent uses its corresponding `tensor.dim` operand; an example
hint never becomes a constant. Mixed static/dynamic ranks and legal negative
axis names retain their original element words. Incomplete/duplicate/out-of-range
permutations and metadata that specializes an unknown source axis refuse this
lowering. Scalar/vector `t` and static empty cases retain their source behavior.
This recovers logical tensor dimensions without physical alias, mutation or
consumer placement permission.

Frontend correspondence now accounts for the registered `aten.sym_numel`
decomposition into actual dimension queries and their complete product. The
`numel_dimension_queries` relation checks the original tensor/result anchors,
static factors and exported symbolic dimension identities. Shared input symbols
and repeated square dimensions may survive common-expression elimination; the
proof preserves their multiplicity, including integer powers introduced by
symbolic simplification. These identities come from the exported shape contract,
whose actual equality/range assertions remain in the emitted MLIR. Example-size
hints and origin tags cannot supply a missing dimension or arithmetic step.
Unknown roots, result references, operators, scopes and incorrect factors remain
diagnostic. This is source graph correspondence, not layout, mutation or numeric
rewrite permission. The trace schema stays v1 with the additional relation kind.

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
static public models does not admit unknown symbolic dimensions. Triton2MLIR's
separate explicit symbolic profile now independently reconstructs selected
input-axis model signatures in Merlin. Broader result expressions, effects and
consumer composition retain their own qualification requirements.

## Dynamic surrounding pointwise computation

`dynamic_pointwise` emits binary broadcasting from actual tensor dimensions.
Static singleton axes retain affine Linalg maps, including scalar parameters
and row/column broadcasts. When a dynamic axis might be a singleton, standard
comparisons check that operand sizes match or one size is one. A parallel
Linalg output computation uses explicit indexed tensor reads for that case;
the frontend does not assume equality from two `?` markers. Empty dimensions
follow the same rules. Result storage follows executable dimension choices,
with checks for any declared static result extent. Numeric promotion and the
existing half-precision Python-scalar opmath policy remain intact.

This closes the surrounding `x + weight` call in an actual exported model with
two dependent registered Triton copies, an empty slice/call and five public
results. Its dynamic public types and original source-domain checks survive;
neither the saved input length nor a provenance annotation supplies an extent.
Triton2MLIR qualifies this enclosing model through native, Merlin host and Spike
under its separate compiler/runtime contracts.

## Dynamic positive-step slices

`dynamic_slices` emits rank-preserving `tensor.extract_slice` from actual source
dimensions and literal integer start/end/positive-step arguments. It normalizes
negative bounds, clips both endpoints into the actual extent and computes the
nonnegative length. The existing checked shape builder retains unsigned index
round trips and logical signed-i64 size bounds without fixing native index
width. Interval-sized arithmetic prevents overflow while clipping endpoints.

Both endpoints lie in `[0, extent]`, so their nonnegative difference fits the
checked logical i64 domain. Only after this proof is it narrowed, when needed.
An i64 quotient/remainder computes the ceiling count without `length+step-1`
overflow or a wide-division runtime dependency. Static result sizes require an
executable equality check unless the source itself fixes the size, as for
`x[:0]`. Other dynamic axes retain their real source dimensions. Type-preserving
casts retain dynamic metadata when the actual slice has a more precise size.
Unknown/nonliteral bounds, nonpositive steps and unsupported rank changes keep
their explicit opaque/refusal boundary. Rejected static contracts publish no
new operand uses. No physical view-alias or ownership promise is introduced.

`tests/test_dynamic_pointwise.py` covers actual exported binary/slice models,
independent source sizes, dynamic half opmath and invalid-bound refusals. The
separate triton2MLIR execution tests run emitted bodies at independent sizes,
check incompatible-broadcast aborts and include signed-i64 extreme bounds/steps.
Those primitive tests use explicit test-driver output extents for descriptor
allocation; they do not establish a general derived-result compiler signature.
The enclosing-model tests independently derive their actual Merlin signatures.

## Symbolic size arithmetic at kernel calls

Exported `aten.sym_size.int` queries and Python add/subtract/multiply/floor-divide
nodes now retain actual scalar SSA when their checked intervals establish a
nonnegative logical-i64 size. Queries compare the real `tensor.dim` with the
source expression; arithmetic consumes imported scalar operands. Unsigned
index transport keeps executable round-trip assertions. Backed symbolic hints
never supply a literal or a proof. Negative or unproved scalar controls remain
outside this size profile and follow the existing explicit import refusal.

`tests/test_symbolic_dimensions.py` covers all four arithmetic forms, the
original dependent copy model and a negative-control refusal. The triton2MLIR
derived-shape integration separately exercises shared input axes and a stepped
slice feeding two translated kernels. Consumer signatures and allocation
require their own independent source proof; capture alone supplies neither.

## Runtime integer words

`runtime_symbolic_scalars` retains `aten.item`/`aten._local_scalar_dense` on
actual static singleton integer tensors as `tensor.extract`. Signed i8/i16/i32
storage extends to Python's i64 integer carrier with sign extension; uint8 uses
zero extension, and i64 stays unchanged. The original signed size constraints,
comparisons and scalar assertions remain executable on that integer SSA.
Unsupported element types or non-singleton sources do not invent a scalar.
Out-of-range comparison literals cannot wrap into a different i64 guard.

An unbacked symbolic size is runtime input data, never its example hint or a
literal from range metadata. A model can now retain an integer count feeding
registered kernels with dynamic result lengths. This import capability grants
no allocation proof: the triton2MLIR input-word shape profile and independent
Merlin signature v5 reconstruct the actual source reads, guards and result
relationships before resolving invocation storage. General data-dependent
scalar arithmetic, effectful reads and resizing state remain separate work.
`tests/test_runtime_integer_item.py` checks signed/unsigned transport, scalar
and singleton-matrix storage, complete dependent calls and native parsing.
