"""Convert metadata dimensions without specializing exported symbolic values."""

from typing import Any

from xdsl.dialects.builtin import DYNAMIC_INDEX


def coerce_dimension(value: Any, *, dynamic: int = DYNAMIC_INDEX) -> int:
    """Keep genuine integer literals and mark nonconstant SymInts dynamic.

    A backed SymInt's ``int()`` observes its hint and adds a specialization
    guard. PyTorch's own ``is_concrete_int`` inspects whether the underlying
    expression is already a literal, without requesting that specialization.
    The separate ``dynamic`` marker preserves older static-only emitters.
    """
    from torch import SymInt

    if isinstance(value, SymInt):
        from torch.fx.experimental.symbolic_shapes import is_concrete_int

        if not is_concrete_int(value):
            return dynamic
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return dynamic
