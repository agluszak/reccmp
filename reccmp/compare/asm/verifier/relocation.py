"""Per-line effect summaries for dependency-aware instruction relocation."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)
from reccmp.compare.asm.ir import (
    AsmStream,
    instruction_at,
    is_data_row,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import (
    Instruction,
    REGISTERS,
    Reject,
)
from reccmp.compare.asm.verifier.addresses import (
    _mem_disjoint,
)
from reccmp.compare.asm.verifier.semantics import (
    execute,
)
from reccmp.compare.asm.verifier.state import (
    CC_CANON,
    Context,
    FAMILIES,
    JCC_MNEMONICS,
    SideState,
    X87Stack,
    guard_state_size,
)

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

_RMW_BINOPS = frozenset(
    {"add", "sub", "and", "or", "xor", "shl", "shr", "sar", "rol", "ror", "adc", "sbb"}
)
_X87_MEM_WRITERS = frozenset({"fst", "fstp", "fist", "fistp", "fnstcw", "fbstp"})


def _line_base_effects(ins: Instruction) -> LineEffects:
    """Effect summary for one instruction: register families, flags, x87
    use and barriers. Memory accesses are filled in by sequence_effects.
    Anything not modeled (calls, jumps, string ops) is a scheduling
    barrier."""
    # pylint: disable=too-many-branches,too-many-statements,too-many-return-statements
    if ins.prefix:
        return BARRIER

    mnemonic = ins.mnemonic
    ops = ins.operands

    regs_read: set = set()
    regs_written: set = set()
    x87 = mnemonic.startswith("f")
    reads_flags = False
    writes_flags = False

    def use(op, write: bool = False) -> None:
        kind = op[0]
        if kind == "reg":
            (regs_written if write else regs_read).add(REGISTERS[op[1]][0])
        elif kind == "mem":
            for reg, _ in op[3]:
                regs_read.add(REGISTERS[reg][0])
        elif kind not in ("imm", "sym", "st"):
            raise Reject

    try:
        if mnemonic in ("mov", "movsx", "movzx") and len(ops) == 2:
            use(ops[1])
            use(ops[0], write=True)
        elif mnemonic == "lea" and len(ops) == 2 and ops[1][0] == "mem":
            for reg, _ in ops[1][3]:
                regs_read.add(REGISTERS[reg][0])
            use(ops[0], write=True)
        elif mnemonic in _RMW_BINOPS and len(ops) == 2:
            use(ops[0])
            use(ops[0], write=True)
            use(ops[1])
            writes_flags = True
            reads_flags = mnemonic in ("adc", "sbb")
        elif mnemonic == "imul" and len(ops) == 3:
            use(ops[0], write=True)
            use(ops[1])
            use(ops[2])
            writes_flags = True
        elif mnemonic == "imul" and len(ops) == 2:
            use(ops[0])
            use(ops[0], write=True)
            use(ops[1])
            writes_flags = True
        elif mnemonic in ("inc", "dec", "neg", "not") and len(ops) == 1:
            use(ops[0])
            use(ops[0], write=True)
            writes_flags = mnemonic != "not"
        elif mnemonic in ("cmp", "test") and len(ops) == 2:
            use(ops[0])
            use(ops[1])
            writes_flags = True
        elif mnemonic in ("mul", "imul", "div", "idiv") and len(ops) == 1:
            use(ops[0])
            regs_read.update(("a", "d"))
            regs_written.update(("a", "d"))
            writes_flags = True
        elif mnemonic == "cdq":
            regs_read.add("a")
            regs_written.add("d")
        elif mnemonic == "cwde":
            regs_read.add("a")
            regs_written.add("a")
        elif mnemonic == "sahf":
            regs_read.add("a")
            writes_flags = True
        elif mnemonic == "lahf":
            regs_written.add("a")
            reads_flags = True
        elif mnemonic == "push" and len(ops) == 1:
            use(ops[0])
            regs_read.add("sp")
            regs_written.add("sp")
        elif mnemonic == "pop" and len(ops) == 1:
            use(ops[0], write=True)
            regs_read.add("sp")
            regs_written.add("sp")
        elif mnemonic.startswith("set") and mnemonic[3:] in CC_CANON and len(ops) == 1:
            use(ops[0], write=True)
            reads_flags = True
        elif mnemonic in ("nop", "int3"):
            pass
        elif mnemonic in JCC_MNEMONICS or mnemonic in ("loop", "loope", "loopne"):
            return LineEffects(reads_flags=True, barrier=True)
        elif mnemonic in ("call", "ret"):
            # Calls clobber the flags; at ret they are dead.
            return LineEffects(writes_flags=True, barrier=True)
        elif x87:
            for op in ops:
                if op[0] == "mem":
                    use(op, write=mnemonic in _X87_MEM_WRITERS)
            if mnemonic == "fnstsw":
                regs_written.add("a")
        else:
            return BARRIER
    except (Reject, IndexError, KeyError):
        return BARRIER

    return LineEffects(
        regs_read=frozenset(regs_read),
        regs_written=frozenset(regs_written),
        reads_flags=reads_flags,
        writes_flags=writes_flags,
        x87=x87,
        barrier=False,
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


def sequence_effects(asm: AsmStream) -> list[LineEffects] | None:
    """Symbolically execute one instruction sequence and return a per-line
    effect summary with memory addresses resolved to symbolic values.
    Returns None if the sequence cannot be analyzed at all."""
    stream = resolve_asm_stream(asm)
    state = SideState(rename_slots=False)
    ctx = Context()
    result = []
    try:
        for idx in range(len(stream)):
            ins: Instruction | None = None
            if not is_data_row(stream, idx):
                try:
                    ins = instruction_at(stream, idx)
                except (Reject, IndexError, KeyError, ValueError, TypeError):
                    ins = None
            base = BARRIER if ins is None else _line_base_effects(ins)
            ctx.trace = []
            failed = False
            try:
                if ins is None:
                    raise Reject
                execute(state, ctx, idx, ins, [])
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
            if not _mem_disjoint(access, against):
                return True
    for access in moved.mem_reads:
        for against in other.mem_writes:
            if not _mem_disjoint(access, against):
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
