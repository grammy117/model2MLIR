"""Materialize exported entry-shape relations as ordinary checked computation.

Read actual SymPy expressions and ValueRanges, never their printed spellings
or example hints. Interval-sized signless integers retain mathematical shape
arithmetic before comparisons, independently of the target's index width.
"""

from dataclasses import dataclass

import sympy as sp
import torch
from xdsl.dialects import arith, builtin, cf, tensor

from .shapes import coerce_dimension
from .standard_arith import IndexCastUIOp

SIZE_MAX = (1 << 63) - 1
MAX_INTEGER_BITS = 4096


@dataclass(frozen=True)
class Integer:
    value: object
    lower: int
    upper: int


class Builder:
    def __init__(self, block, stamp):
        self.block, self.stamp, self.metadata = block, stamp, {}
        self.expression_count = 0

    def emit(self, operation):
        for metadata in self.metadata if isinstance(self.metadata, list) else [self.metadata]:
            self.stamp(operation, metadata, "shape_guard")
        self.block.add_op(operation)
        return operation.results[0] if operation.results else None

    @staticmethod
    def width(lower, upper):
        width = max(2, max(abs(lower), abs(upper)).bit_length() + 2)
        if width > MAX_INTEGER_BITS:
            raise ValueError("exported shape arithmetic exceeds the checked integer budget")
        return width

    def constant(self, number):
        width = self.width(number, number)
        value = self.emit(arith.ConstantOp.from_int_and_width(number, width))
        return Integer(value, number, number)

    def align(self, *values):
        width = max(value.value.type.bitwidth for value in values)
        return [value.value if value.value.type.bitwidth == width else
                self.emit(arith.ExtSIOp(value.value, builtin.IntegerType(width))) for value in values]

    def binary(self, kind, lhs, rhs):
        if kind == "add":
            low, high, operation = lhs.lower + rhs.lower, lhs.upper + rhs.upper, arith.AddiOp
        elif kind == "sub":
            low, high, operation = lhs.lower - rhs.upper, lhs.upper - rhs.lower, arith.SubiOp
        elif kind == "mul":
            products = [a * b for a in (lhs.lower, lhs.upper) for b in (rhs.lower, rhs.upper)]
            low, high, operation = min(products), max(products), arith.MuliOp
        elif kind == "floor" and rhs.lower == rhs.upper and rhs.lower > 0:
            low, high, operation = lhs.lower // rhs.lower, lhs.upper // rhs.lower, arith.FloorDivSIOp
        else:
            raise ValueError("unsupported exported shape arithmetic")
        width = max(self.width(low, high), lhs.value.type.bitwidth, rhs.value.type.bitwidth)
        operands = [value.value if value.value.type.bitwidth == width else
                    self.emit(arith.ExtSIOp(value.value, builtin.IntegerType(width))) for value in (lhs, rhs)]
        return Integer(self.emit(operation(*operands)), low, high)

    def check(self, lhs, rhs, predicate, message):
        compare = self.emit(arith.CmpiOp(*self.align(lhs, rhs), predicate))
        self.emit(cf.AssertOp(compare, "exported shape domain: " + message))

    def extent(self, value, axis):
        index = self.emit(arith.ConstantOp.from_int_and_width(axis, builtin.IndexType()))
        extent = self.emit(tensor.DimOp(value, index))
        size = self.emit(IndexCastUIOp(extent, builtin.i64))
        back = self.emit(IndexCastUIOp(size, builtin.IndexType()))
        equal = self.emit(arith.CmpiOp(extent, back, "eq"))
        self.emit(cf.AssertOp(equal, "exported shape domain: dimension fits logical i64 storage"))
        limit = self.emit(arith.ConstantOp.from_int_and_width(SIZE_MAX, builtin.i64))
        valid = self.emit(arith.CmpiOp(size, limit, "ule"))
        self.emit(cf.AssertOp(valid, "exported shape domain: dimension is a nonnegative signed i64 size"))
        return Integer(size, 0, SIZE_MAX)

    def expression(self, expression, symbols):
        self.expression_count += 1
        if self.expression_count > 8192:
            raise ValueError("exported shape expression exceeds its analysis budget")
        if expression.is_Integer:
            return self.constant(int(expression))
        if expression.is_Symbol and expression in symbols:
            return symbols[expression]
        if expression.is_Add or expression.is_Mul:
            kind, identity = ("add", 0) if expression.is_Add else ("mul", 1)
            result = self.constant(identity)
            for argument in expression.args:
                result = self.binary(kind, result, self.expression(argument, symbols))
            return result
        if expression.is_Pow and expression.args[1].is_Integer and expression.args[1] >= 0:
            base, exponent = self.expression(expression.args[0], symbols), int(expression.args[1])
            if exponent.bit_length() > MAX_INTEGER_BITS:
                raise ValueError("exported shape exponent exceeds its analysis budget")
            result = self.constant(1)
            while exponent:
                if exponent & 1:
                    result = self.binary("mul", result, base)
                exponent //= 2
                if exponent:
                    base = self.binary("mul", base, base)
            return result
        if expression.func.__name__ == "FloorDiv" and len(expression.args) == 2 and expression.args[1].is_Integer:
            return self.binary("floor", self.expression(expression.args[0], symbols),
                               self.constant(int(expression.args[1])))
        raise ValueError("unsupported exported entry-shape expression: " + str(expression))

    def divisible(self, value, divisor):
        constant = self.constant(divisor)
        remainder = Integer(self.emit(arith.RemSIOp(*self.align(value, constant))), -divisor + 1, divisor - 1)
        self.check(remainder, self.constant(0), "eq", "derived dimension is divisible by its source coefficient")


def _integer_bound(value, lower):
    from torch.utils._sympy.numbers import int_oo

    if value in ((-sp.oo, -int_oo) if lower else (sp.oo, int_oo)):
        return None
    value = sp.ceiling(value) if lower else sp.floor(value)
    if not value.is_Integer:
        raise ValueError("exported shape range must have integer or infinite endpoints")
    return int(value)


def emit_entry_shape_guards(exported, placeholders, block, stamp):
    rows, report = [], []
    for index, node in enumerate(placeholders):
        metadata = node.meta.get("val")
        if not hasattr(metadata, "shape"):
            continue
        for axis, dimension in enumerate(metadata.shape):
            if not isinstance(dimension, torch.SymInt) or coerce_dimension(dimension) != builtin.DYNAMIC_INDEX:
                continue
            rows.append((index, axis, dimension.node.expr, node.meta))
    if not rows:
        return None
    if len(rows) > 256:
        raise ValueError("exported entry-shape contract exceeds its dimension budget")
    builder, symbols, observations = Builder(block, stamp), {}, []
    for index, axis, expression, metadata in rows:
        builder.metadata = metadata
        actual = builder.extent(block.args[index], axis)
        observations.append((actual, expression, metadata))
        report.append({"argument": index, "axis": axis, "expression": str(expression)})
        if expression.is_Symbol and expression not in symbols:
            symbols[expression] = actual

    # Reconstruct roots from actual direct dimensions or invertible affine
    # source axes. No example hint supplies a root, even for backed symbols.
    pending = {symbol for _, expression, _ in observations for symbol in expression.free_symbols} - set(symbols)
    while pending:
        previous = len(pending)
        for actual, expression, metadata in observations:
            missing = expression.free_symbols - set(symbols)
            if len(missing) != 1:
                continue
            symbol = next(iter(missing))
            try:
                polynomial = sp.Poly(expression, symbol)
            except sp.PolynomialError:
                continue
            coefficient = polynomial.coeff_monomial(symbol)
            if polynomial.degree() != 1 or not coefficient.is_Integer or coefficient == 0:
                continue
            builder.metadata = metadata
            rest = builder.expression(sp.expand(expression - coefficient * symbol), symbols)
            delta = builder.binary("sub", actual, rest) if coefficient > 0 else builder.binary("sub", rest, actual)
            divisor = abs(int(coefficient))
            builder.divisible(delta, divisor)
            symbols[symbol] = builder.binary("floor", delta, builder.constant(divisor))
            pending.remove(symbol)
        if len(pending) == previous:
            raise ValueError("exported entry-shape roots need a direct or invertible affine source dimension")

    ranges = getattr(exported, "range_constraints", {}) or {}
    for expression, interval in ranges.items():
        if not expression.free_symbols <= set(symbols):
            continue  # A data-dependent body range is not an entry tensor domain.
        builder.metadata = [metadata for _, source_expression, metadata in observations
                            if source_expression.free_symbols & expression.free_symbols]
        value = builder.expression(expression, symbols)
        lower, upper = _integer_bound(interval.lower, True), _integer_bound(interval.upper, False)
        for limit, predicate in ((lower, "sge"), (upper, "sle")):
            if limit is not None:
                builder.check(value, builder.constant(limit), predicate, str(expression) + " " + predicate + " " + str(limit))
        if expression.is_Symbol:
            low = max(value.lower, lower) if lower is not None else value.lower
            high = min(value.upper, upper) if upper is not None else value.upper
            if low > high:
                raise ValueError("exported entry-shape range has no representable source values")
            symbols[expression] = Integer(value.value, low, high)

    for actual, expression, metadata in observations:
        builder.metadata = metadata
        builder.check(actual, builder.expression(expression, symbols), "eq", "original input dimension relation")
    return {"scope": "entry-tensor-shapes", "origin": "actual-exported-symbolic-expressions",
            "dimensions": report, "arithmetic": "checked-logical-i64-sizes-and-interval-sized-integers"}
