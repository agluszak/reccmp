"""Shape of symbolic memory addresses: flattening, stack roots, disjointness."""

from __future__ import annotations

# A symbolic value. Nested tuples of str/int; compared structurally.
Value = tuple


def _foldable_address_value(value: Value, scale: int, segment: str) -> bool:
    if scale != 1 or not isinstance(value, tuple):
        return False
    if value and value[0] == "add":
        return True
    return len(value) == 2 and value[0] == "addr" and value[1][1] in ("", segment)


def _flatten_mem(addr: Value) -> Value:
    """Fold scale-1 base registers that hold a computed address (from lea)
    into the memory expression itself, so `[esi]` with esi = &[ebx + 0x1c6]
    compares as `[ebx + 0x1c6]`."""
    _, seg, terms, disp, syms = addr
    for _ in range(8):
        folded = None
        for term in terms:
            value, scale = term
            if _foldable_address_value(value, scale, seg):
                folded = term
                break
        if folded is None:
            break
        value = folded[0]
        terms = tuple(t for t in terms if t is not folded)
        if value[0] == "addr":
            inner = value[1]
            terms += inner[2]
            disp += inner[3]
            syms = tuple(sorted(set(syms) | set(inner[4])))
            seg = seg or inner[1]
        else:
            for leaf in value[1:]:
                if isinstance(leaf, tuple) and leaf[0] == "imm":
                    disp += leaf[1]
                else:
                    terms += ((leaf, 1),)
    return ("mem", seg, tuple(sorted(terms, key=repr)), disp, syms)


def _stack_rooted(value: Value) -> bool:
    """Is the value derived from the stack pointer or frame pointer?"""
    if not isinstance(value, tuple) or not value:
        return False
    if value in (("init", "sp"), ("init", "bp")):
        return True
    tag = value[0]
    children: tuple[Value, ...] = ()
    if tag == "spadd":
        children = (value[1],)
    elif tag == "addr":
        children = tuple(child for child, _ in value[1][2])
    elif tag in ("add", "ins_r16"):
        children = value[1:]
    return any(_stack_rooted(child) for child in children)


def _is_pure_global(mem: Value) -> bool:
    return not mem[2] and bool(mem[4])


def _unwind_spadd(value: Value, offset: int = 0) -> tuple[Value, int]:
    while isinstance(value, tuple) and value and value[0] == "spadd":
        offset += value[2]
        value = value[1]
    return (value, offset)


def _abs_stack_offset(addr: Value, is_slot) -> tuple[Value, int] | None:
    """Resolve an access to (root value, byte offset) when its address is a
    plain chain of constant adjustments over one root — a push/pop slot, or
    a single-register memory operand like [ebp - 8] or [esp + 4]."""
    if is_slot:
        return _unwind_spadd(addr)
    mem = _flatten_mem(addr)
    if len(mem[2]) == 1 and not mem[4] and isinstance(mem[3], int):
        value, scale = mem[2][0]
        if scale == 1:
            root, offset = _unwind_spadd(value)
            return (root, offset + mem[3])
    return None


def _ranges_disjoint(a_disp, a_width, b_disp, b_width) -> bool:
    if isinstance(a_disp, int) and isinstance(b_disp, int):
        if a_width is None or b_width is None:
            return False
        return a_disp + a_width <= b_disp or b_disp + b_width <= a_disp
    if a_disp == b_disp:
        return False
    # Alpha-renamed frame slots: distinct slot ids are distinct locals
    # (their non-overlap is validated by _slots_consistent).
    return (
        isinstance(a_disp, tuple)
        and isinstance(b_disp, tuple)
        and a_disp[0] == b_disp[0] == "slot"
    )


def _mem_disjoint(a: tuple, b: tuple) -> bool:
    """Can the two memory accesses be proven non-overlapping?
    Accesses are (address value, width, stack_kind) where stack_kind is
    False for ordinary operands, "push" for a fresh slot below the stack
    pointer, "pop" for a read of the top of the stack."""
    # pylint: disable=too-many-return-statements
    a_addr, a_width, a_stack = a
    b_addr, b_width, b_stack = b

    if a_stack or b_stack:
        a_res = _abs_stack_offset(a_addr, a_stack)
        b_res = _abs_stack_offset(b_addr, b_stack)
        if (
            a_res is not None
            and b_res is not None
            and a_res[0] == b_res[0]
            and _ranges_disjoint(a_res[1], a_width, b_res[1], b_width)
        ):
            return True
        if a_stack and b_stack:
            return False
        other = _flatten_mem(b_addr if a_stack else a_addr)
        # A stack slot never overlaps a named global. An access through an
        # unknown pointer, however, must be assumed to alias the stack:
        # nothing proves an incoming pointer cannot equal the slot address.
        return _is_pure_global(other)

    a_mem = _flatten_mem(a_addr)
    b_mem = _flatten_mem(b_addr)

    if a_mem[1] != b_mem[1]:
        # Different segment prefixes: assume they can alias.
        return False

    # Same base values (symbolically identical registers/symbols): the two
    # accesses differ only by constant displacement.
    if a_mem[2] == b_mem[2] and a_mem[4] == b_mem[4]:
        return _ranges_disjoint(a_mem[3], a_width, b_mem[3], b_width)

    global_a = _is_pure_global(a_mem)
    global_b = _is_pure_global(b_mem)

    if global_a and global_b and a_mem[4] != b_mem[4]:
        # Two different named globals do not overlap.
        return True

    # Stack/frame memory never overlaps a named global.
    stack_a = any(_stack_rooted(v) for v, _ in a_mem[2])
    stack_b = any(_stack_rooted(v) for v, _ in b_mem[2])
    if (global_a and stack_b) or (global_b and stack_a):
        return True

    return False
