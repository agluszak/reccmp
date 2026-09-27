"""Instruction relocation: per-line effect summaries, and undoing moves that
are proven independent of everything they cross."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reccmp.compare.asm.ir import DecodedInstruction, instruction_semantic_key
from reccmp.compare.asm.model import REGISTERS, Reject
from reccmp.compare.asm.verifier.addresses import mem_disjoint
from reccmp.compare.asm.verifier.semantics import execute
from reccmp.compare.asm.verifier.state import (
    FAMILIES,
    JCC_MNEMONICS,
    Context,
    SideState,
    X87Stack,
    guard_state_size,
)
from reccmp.compare.pinned_sequences import DiffOpcode

# ---------------------------------------------------------------------------
# Per-line effect summaries for dependency-aware instruction relocation


@dataclass(frozen=True)
class LineEffects:
    """Conservative summary of one instruction's reads and writes, used to
    decide whether two instructions may be reordered. Memory accesses are
    (address value, width, is_stack_slot) with addresses resolved by
    symbolic execution (see sequence_effects), so e.g. `[esi]` after
    `lea esi, [ebx + 0x1c6]` is comparable with `[ebx + 0xb0]`."""

    # pylint: disable=too-many-instance-attributes

    regs_read: frozenset = frozenset()
    regs_written: frozenset = frozenset()
    reads_flags: bool = False
    writes_flags: bool = False
    mem_reads: tuple = ()
    mem_writes: tuple = ()
    x87: bool = False
    barrier: bool = False


BARRIER = LineEffects(barrier=True)

_X87_REGISTERS = frozenset({"fpsw", "fpcw", "fptag", "fpip", "fpdp"})


def _line_base_effects(ins: DecodedInstruction) -> LineEffects:
    """Effect summary for one instruction from Capstone's register and flag
    access: register families, flags, x87 use and barriers. Memory
    accesses are filled in by sequence_effects. Control transfers, prefixed
    instructions and registers outside the model are scheduling barriers;
    a barrier keeps its flag effects (a jcc reads the flags, a call
    clobbers them)."""
    reads_flags = ins.reads_flags or "eflags" in ins.regs_read
    writes_flags = ins.writes_flags or "eflags" in ins.regs_written
    if (
        ins.prefix
        or ins.is_jump
        or ins.is_call
        or ins.is_ret
        or not ins.register_access_known
    ):
        return LineEffects(
            reads_flags=reads_flags,
            writes_flags=writes_flags or ins.is_call or ins.is_ret,
            barrier=True,
        )
    x87 = ins.mnemonic.startswith("f")
    families: list[set[str]] = [set(), set()]
    for names, found in ((ins.regs_read, families[0]), (ins.regs_written, families[1])):
        for name in names:
            if name in REGISTERS:
                found.add(REGISTERS[name][0])
            elif name == "eflags" or (x87 and name in _X87_REGISTERS):
                continue
            else:
                return BARRIER
    return LineEffects(
        regs_read=frozenset(families[0]),
        regs_written=frozenset(families[1]),
        reads_flags=reads_flags,
        writes_flags=writes_flags,
        x87=x87,
    )


def _havoc(state: SideState, idx: int) -> None:
    """Discard everything we know about the state after an instruction
    outside the model."""
    for family in FAMILIES:
        state.regs[family] = ("havoc", idx, family)
    state.flags = ("havoc_flags", idx)
    state.carry = ("havoc_cf", idx)
    state.fpu_flags = ("havoc_fpuflags", idx)
    state.x87 = X87Stack(epoch=-idx - 1)


def sequence_effects(rows: Sequence[DecodedInstruction]) -> list[LineEffects] | None:
    """Symbolically execute one instruction sequence and return a per-line
    effect summary with memory addresses resolved to symbolic values.
    Returns None if the sequence cannot be analyzed at all."""
    state = SideState(rename_slots=False)
    ctx = Context()
    result = []
    try:
        for idx, row in enumerate(rows):
            base = _line_base_effects(row)
            ctx.trace = []
            failed = False
            try:
                execute(state, ctx, idx, row, [])
                guard_state_size(state, ctx)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                _havoc(state, idx)
                failed = True
            if base.barrier or failed:
                # Keep the flag information of modeled barriers (a jcc reads
                # the flags, a call clobbers them).
                result.append(BARRIER if failed and not base.barrier else base)
                continue
            trace = ctx.trace
            result.append(
                LineEffects(
                    regs_read=base.regs_read,
                    regs_written=base.regs_written,
                    reads_flags=base.reads_flags,
                    writes_flags=base.writes_flags,
                    mem_reads=tuple(
                        (addr, width, stack)
                        for op, addr, width, stack in trace
                        if op == "r"
                    ),
                    mem_writes=tuple(
                        (addr, width, stack)
                        for op, addr, width, stack in trace
                        if op == "w"
                    ),
                    x87=base.x87,
                    barrier=False,
                )
            )
    except (Reject, RecursionError):
        return None
    return result


def effects_conflict(moved: LineEffects, other: LineEffects) -> bool:
    """True if the moved instruction cannot be reordered across `other`."""
    # pylint: disable=too-many-return-statements
    if moved.barrier or other.barrier:
        return True
    if moved.regs_written & (other.regs_read | other.regs_written):
        return True
    if moved.regs_read & other.regs_written:
        return True
    if moved.writes_flags and other.reads_flags:
        return True
    if moved.reads_flags and other.writes_flags:
        return True
    # x87 instructions depend on the fp stack order.
    if moved.x87 and other.x87:
        return True
    for access in moved.mem_writes:
        for against in other.mem_reads + other.mem_writes:
            if not mem_disjoint(access, against):
                return True
    for access in moved.mem_reads:
        for against in other.mem_writes:
            if not mem_disjoint(access, against):
                return True
    return False


def flags_dead_at(effects_list: list[LineEffects], start: int) -> bool:
    """Are the CPU flags provably dead (rewritten before being read) from
    line `start` onward?"""
    for effects in effects_list[start:]:
        if effects.reads_flags:
            return False
        if effects.writes_flags:
            return True
        if effects.barrier:
            # Unknown control flow: assume the flags could be read.
            return False
    return True


def undo_relocations(
    codes: Sequence[DiffOpcode],
    orig: Sequence[DecodedInstruction],
    recomp: Sequence[DecodedInstruction],
) -> list[DecodedInstruction] | None:
    """If every diff insertion can be paired with an equal deletion whose
    move is proven independent of all crossed instructions, return recomp
    reordered into orig's instruction order. Returns None when the diffs
    are not (only) relocations."""
    # pylint: disable=too-many-return-statements
    if len(orig) != len(recomp):
        return None

    # Sorted for deterministic matching when several identical lines
    # could pair up. (GH #324)
    deletes = sorted(
        i for code, i1, i2, _, __ in codes for i in range(i1, i2) if code == "delete"
    )
    # `i1` is the index of the orig list where this line will be inserted.
    # This is not necessarily equal to `j1`, the index of the inserted line in recomp.
    # Therefore we need to save `i1` so that we verify each line between the start and end of the move. (GH #332)
    inserts = [
        (i1, j)
        for code, i1, __, j1, j2 in codes
        for j in range(j1, j2)
        if code == "insert"
    ]

    if not inserts or len(inserts) != len(deletes):
        return None

    effects = sequence_effects(orig)
    if effects is None:
        return None

    orig_keys = [instruction_semantic_key(row) for row in orig]
    pairs: dict[int, int] = {}
    remaining = list(deletes)
    for orig_dest, j in inserts:
        key_r = instruction_semantic_key(recomp[j])
        matched = None
        for i in remaining:
            if orig_keys[i] != key_r:
                continue
            if _can_relocate(effects, orig, i, orig_dest):
                matched = i
                break
        if matched is None:
            return None
        pairs[j] = matched
        remaining.remove(matched)

    # Sort recomp lines by their position in orig's coordinate system:
    # matching lines keep their diff-aligned position, relocated lines take
    # the position of their paired deletion.
    key: dict[int, int] = {}
    for code, i1, i2, j1, j2 in codes:
        if code in ("equal", "replace"):
            if (i2 - i1) != (j2 - j1):
                return None
            for i, j in zip(range(i1, i2), range(j1, j2)):
                key[j] = i
        elif code == "insert":
            for j in range(j1, j2):
                key[j] = pairs[j]

    if len(key) != len(recomp):
        return None

    order = sorted(range(len(recomp)), key=key.__getitem__)
    if order == list(range(len(recomp))):
        return None
    return [recomp[j] for j in order]


def _can_relocate(
    effects: list[LineEffects],
    orig: Sequence[DecodedInstruction],
    i: int,
    orig_dest: int,
) -> bool:
    """May the instruction at orig index `i` move to position `orig_dest`?
    Only if it is independent of every instruction it crosses: no register
    or flag dependency, no possibly-aliasing memory access, no x87 stack
    interaction and no control-flow barrier in between. (GH #324)"""
    moved = effects[i]
    if moved.barrier:
        return False

    # To account for a move in either direction:
    # the deleted line can precede or follow the inserted line.
    reloc_start = min(i, orig_dest)
    reloc_end = max(i, orig_dest)

    crossed_flag_writer = False
    for k in range(reloc_start, reloc_end):
        if k == i:
            continue
        other = effects[k]
        if other.barrier:
            # Exception: a forward conditional jump whose target lies within
            # the crossed region. The moved instruction then executes on
            # both the taken and the fallthrough path in both placements
            # (and it must not touch the flags the jump reads).
            if not moved.writes_flags and _forward_jcc_within(orig, k, reloc_end):
                continue
            return False
        if effects_conflict(moved, other):
            return False
        if other.writes_flags:
            crossed_flag_writer = True

    # If both the moved instruction and a crossed instruction write the
    # flags, the move changes which value the flags hold at the end of the
    # region: the flags must be dead there.
    if moved.writes_flags and crossed_flag_writer:
        after = reloc_end + 1 if reloc_end == i else reloc_end
        if not flags_dead_at(effects, after):
            return False

    return True


def _forward_jcc_within(
    orig: Sequence[DecodedInstruction], k: int, reloc_end: int
) -> bool:
    """Is orig[k] a forward conditional jump whose target is at or before
    index reloc_end?"""
    row = orig[k]
    if row.mnemonic not in JCC_MNEMONICS or row.branch_target is None:
        return False
    if row.address is None or row.branch_target <= row.address:
        return False
    target = next(
        (
            index
            for index, other in enumerate(orig)
            if other.address == row.branch_target
        ),
        None,
    )
    return target is not None and k < target <= reloc_end
