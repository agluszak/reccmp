"""Discover a function's extent from its control flow.

Used when the original function has no annotated size. The old estimate,
``min(distance to the next entity, recompiled size)``, cuts off originals
that are longer than the recompilation or keep blocks after its length,
which leaves the extent open and the comparison looking at a truncated
body.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from capstone import (  # type: ignore[import-untyped]
    CS_ARCH_X86,
    CS_MODE_16,
    CS_MODE_32,
    Cs,
)
from capstone.x86 import X86_OP_IMM, X86_OP_MEM  # type: ignore[import-untyped]

from reccmp.formats import Image
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)

_JCC_PREFIX = "j"
_TERMINALS = frozenset({"ret", "retf", "iret", "iretd", "int3", "hlt", "ud2"})
# Walking further than this without closing every path means the start is
# not a normal function (or flows into unknown code); give up.
_MAX_EXTENT = 0x10000


@dataclass(frozen=True)
class EntityExtent:
    """An entity's size on one side, and whether a record says so. An
    estimate (for instance the other side's size) bounds where the entity
    may end, but does not show that it owns every byte up to there."""

    size: int
    recorded: bool = True


@cache
def _disassembler(is_32bit: bool) -> Cs:
    cs = Cs(CS_ARCH_X86, CS_MODE_32 if is_32bit else CS_MODE_16)
    cs.detail = True
    return cs


def _read(image: Image, addr: int, size: int) -> bytes:
    try:
        return image.read(addr, size)
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return b""


def _jump_table_targets(image: Image, insn, start: int, limit: int) -> list[int] | None:
    """Targets of ``jmp dword ptr [reg*4 + table]`` that land in the window."""
    op = insn.operands[0]
    if op.type != X86_OP_MEM or op.mem.scale != 4 or not op.mem.index or op.mem.base:
        return None
    table = op.mem.disp & 0xFFFFFFFF
    targets: list[int] = []
    for i in range(256):
        raw = _read(image, table + 4 * i, 4)
        if len(raw) != 4:
            break
        target = int.from_bytes(raw, "little")
        if not start <= target < start + limit:
            break
        targets.append(target)
    return targets or None


def discover_extent(
    image: Image, start: int, limit: int | None, *, is_32bit: bool = True
) -> int | None:
    """Size of the code reachable from ``start`` by fallthrough, local jumps
    and jump tables, when every path ends in a terminal inside
    ``[start, start + limit)``. None when it cannot be closed.

    Calls fall through; jumps leaving the window are tail calls.
    """
    # pylint: disable=too-many-branches,too-many-locals
    limit = min(limit or _MAX_EXTENT, _MAX_EXTENT)
    code = _read(image, start, limit)
    while not code and limit > 16:
        # The window runs past the end of the image or section.
        limit //= 2
        code = _read(image, start, limit)
    if not code:
        return None
    cs = _disassembler(is_32bit)
    end = start
    pending = [start]
    seen: set[int] = set()
    # Switch data placed after the code belongs to the function too.
    jump_tables: list[tuple[int, int]] = []  # (address, entries)
    byte_tables: set[int] = set()
    while pending:
        addr = pending.pop()
        while addr not in seen:
            if not start <= addr < start + len(code):
                return None
            seen.add(addr)
            offset = addr - start
            insn = next(cs.disasm(code[offset : offset + 16], addr, 1), None)
            if insn is None:
                return None
            end = max(end, addr + insn.size)
            mnemonic = insn.mnemonic
            if mnemonic in _TERMINALS:
                break
            op = insn.operands[0] if insn.operands else None
            if mnemonic == "jmp":
                if op is not None and op.type == X86_OP_IMM:
                    target = op.imm & 0xFFFFFFFF
                    if start <= target < start + len(code):
                        pending.append(target)
                    # Otherwise a tail call: the path ends here.
                else:
                    targets = _jump_table_targets(image, insn, start, len(code))
                    if targets is not None and op is not None:
                        pending.extend(targets)
                        jump_tables.append((op.mem.disp & 0xFFFFFFFF, len(targets)))
                    # An indirect jump without a table is a tail call.
                break
            if (
                (
                    mnemonic.startswith(_JCC_PREFIX)
                    or mnemonic in ("loop", "loope", "loopne")
                )
                and op is not None
                and op.type == X86_OP_IMM
            ):
                target = op.imm & 0xFFFFFFFF
                if not start <= target < start + len(code):
                    return None
                pending.append(target)
            if mnemonic in ("movzx", "mov") and len(insn.operands) == 2:
                # Byte index table: [reg + table] with a single register.
                src = insn.operands[1]
                one_register = bool(src.mem.base) != bool(src.mem.index)
                if (
                    src.type == X86_OP_MEM
                    and src.size == 1
                    and one_register
                    and src.mem.scale == 1
                    and start <= (src.mem.disp & 0xFFFFFFFF) < start + len(code)
                ):
                    byte_tables.add(src.mem.disp & 0xFFFFFFFF)
            addr += insn.size
    window_end = start + len(code)
    for table, entries in jump_tables:
        if start <= table and table + 4 * entries <= window_end:
            end = max(end, table + 4 * entries)
    cases = max((entries for _, entries in jump_tables), default=0)
    for table in byte_tables:
        # An index table maps inputs to case numbers, all below the count.
        position = table
        while start <= position < window_end and code[position - start] < cases:
            position += 1
        if position > table:
            end = max(end, position)
    return end - start


def plausible_discovered_extent(
    image: Image,
    start: int,
    limit: int | None,
    counterpart_size: int,
    *,
    is_32bit: bool = True,
) -> int | None:
    """``discover_extent`` for a function whose size is not recorded,
    rejected when much larger than its counterpart in the other binary: the
    walk then usually ran past a call that does not return into the next
    function."""
    discovered = discover_extent(image, start, limit, is_32bit=is_32bit)
    if discovered is None or discovered > 2 * counterpart_size + 64:
        return None
    return discovered
