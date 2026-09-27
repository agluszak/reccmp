"""Private frame slots held per side.

A local the compiler keeps in a stack slot on one side may live in a
register on the other. Stack stores are observables everywhere else (every
store must match), so such a pair never verifies. In promotion mode
(``SideState.frame`` is not None) each side keeps the function's private
frame to itself instead, like a register file: a store to a slot records
its value there and observes nothing, a load of the slot reads that value
back, and states join slot by slot. What reaches a callee is compared
where it does: the argument bytes on top of the stack at each call.

The private frame is the memory strictly below the entry stack pointer,
addressed at a constant offset from it. No pointer the function receives
can point there, and while none of its own pointers has left the function
(``Context.stack_escaped``), nothing else can read or write it. So every
access the model cannot place exactly, but which may reach the frame (an
indexed local, a stack pointer after a call of unknown cleanup), while any
slot is promoted, rejects the attempt instead of guessing, and so does an
escape. A load of a slot that holds nothing (uninitialized, or written by
an access the model could not place) rejects too.
"""

from __future__ import annotations

from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.verifier.addresses import (
    Init,
    Value,
    constant_offset,
    flatten_mem,
)
from reccmp.compare.asm.verifier.state import Context, SideState

_ENTRY_SP = Init("sp")


def maybe_frame_pointer(value: Value) -> bool:
    """Whether a value may point into the function's own frame: it is built
    from the entry stack or frame pointer, or from the stack pointer after
    a call, other than through a load (a load from shared memory cannot
    return a frame pointer before one has escaped)."""
    seen: set[int] = set()
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, Init):
            if node.family in ("sp", "bp"):
                return True
            continue
        if not isinstance(node, tuple) or not node or id(node) in seen:
            continue
        seen.add(id(node))
        if node[0] == "callesp":
            return True
        if node[0] == "load":
            continue
        stack.extend(node)
    return False


def frame_offset(
    state: SideState, ctx: Context, address: Value, width: int | None, *, slot: bool
) -> int | None:
    """The entry-sp offset of an access to a promoted slot, or None when the
    access is not the private frame's (promotion off, the frame escaped
    before anything was promoted, or memory the frame cannot overlap).
    ``slot``: the address is a push/pop stack pointer, not a memory
    operand. Rejects an access that may reach a promoted slot but cannot
    be placed exactly."""
    if state.frame is None:
        return None
    resolved = _entry_offset(address, slot)
    if ctx.stack_escaped:
        if state.frame:
            raise Reject
        return None
    if resolved is not None and resolved[0] == _ENTRY_SP and width is not None:
        offset = resolved[1]
        if offset >= 0:
            return None  # the caller's side of the stack
        if offset + width <= 0:
            return offset
        raise Reject  # straddles the entry stack pointer
    terms = [address] if slot else [term for term, _ in flatten_mem(address)[2]]
    if state.frame and any(maybe_frame_pointer(term) for term in terms):
        raise Reject
    return None


def _entry_offset(address: Value, slot: bool) -> tuple[Value, int] | None:
    """(root, offset) of an access at a constant offset from one root."""
    if slot:
        return constant_offset(address)
    mem = flatten_mem(address)
    if len(mem[2]) == 1 and not mem[4] and isinstance(mem[3], int):
        value, scale = mem[2][0]
        if scale == 1:
            root, offset = constant_offset(value)
            return (root, offset + mem[3])
    return None


def _overlapping(state: SideState, offset: int, width: int) -> list[int]:
    assert state.frame is not None
    return [
        start
        for start, (size, _) in state.frame.items()
        if start < offset + width and offset < start + size
    ]


def read_slot(state: SideState, offset: int, width: int) -> Value:
    """The value a promoted slot holds; rejects one it does not hold whole."""
    assert state.frame is not None
    entry = state.frame.get(offset)
    if entry is None or entry[0] != width or entry[1] is None:
        raise Reject
    return entry[1]


def write_slot(state: SideState, offset: int, width: int, value: Value) -> None:
    """Store into a promoted slot. The part of an older slot it overwrites
    only in part can no longer be read."""
    assert state.frame is not None
    for start in _overlapping(state, offset, width):
        size, _ = state.frame[start]
        state.frame[start] = (size, None)
    state.frame[offset] = (width, value)
