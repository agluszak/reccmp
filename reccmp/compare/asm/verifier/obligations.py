"""Rules shared by every strategy for when two diverging states may still
be equivalent: synchronization, one-sided instructions, callee-save
swaps, frame slots and the end-of-run admission checklist."""

from __future__ import annotations

import re

from reccmp.compare.asm.instgen import InstructionMeta
from reccmp.compare.asm.model import (
    REGISTERS,
    Instruction,
    Reject,
    parse_instruction,
)
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import (
    Value,
    mem_disjoint,
    unwind_spadd,
)
from reccmp.compare.asm.verifier.evidence import (
    diagnostic_summaries,
    record_observable_difference,
)
from reccmp.compare.asm.verifier.semantics import (
    esp_add,
    execute,
    mem_address,
    read_operand,
)
from reccmp.compare.asm.verifier.state import (
    ASSOCIATIVE_COMMUTATIVE_BINOPS,
    COMMUTATIVE_BINOPS,
    CONTROL_TAGS,
    FAMILIES,
    JCC_MNEMONICS,
    STRING_OPS,
    WIDTHS,
    Context,
    SideState,
    commit_clobber,
    commutative_result,
    frame_pointer_value,
    guard_state_size,
    memory_load_tag,
    vsort,
)
from reccmp.compare.diagnosis import AnalysisRecorder

# ---------------------------------------------------------------------------
# Lockstep driver

# Non-executable lines emitted by the sanitizer (jump/data tables).
DATA_LINE_RE = re.compile(r"^(Jump table:|Data table:|start \+ |0x[0-9a-f]+$)")
# Jump-table entries produced by ParseAsm for ADDR_TAB destinations.
JUMP_TABLE_ENTRY_RE = re.compile(r"^start \+ (0x[0-9a-f]+)$")


def switch_index_observation(state: SideState, ins: Instruction) -> tuple:
    """Index register values that select a recognized switch-table case."""
    op = ins.operands[0]
    assert isinstance(op, tuple) and op[0] == "mem"
    reg_terms = op[3]
    return tuple(
        (scale, state.read_reg(reg))
        for reg, scale in sorted(reg_terms, key=lambda t: (-t[1], t[0]))
    )


def fully_synced(orig: SideState, recomp: SideState) -> bool:
    return (
        orig.regs == recomp.regs
        and orig.flags == recomp.flags
        and orig.carry == recomp.carry
        and orig.fpu_flags == recomp.fpu_flags
        and orig.x87.state_key() == recomp.x87.state_key()
        # An unsupported-but-identical instruction accesses the same textual
        # frame slots on both sides, so any slot renaming so far must be
        # the identity for the states to be concretely identical.
        and orig.slot_map == recomp.slot_map
    )


def resync(states: tuple[SideState, SideState], idx: int, ctx: Context) -> None:
    """After an unsupported-but-identical instruction on a fully synced
    state, both sides are still concretely identical, but we no longer know
    which locations the instruction wrote. Give every location a fresh
    paired value so no stale claims survive."""
    for state in states:
        for family in FAMILIES:
            state.regs[family] = ("resync", idx, family)
        state.flags = ("resync_flags", idx)
        state.carry = ("resync_cf", idx)
        state.fpu_flags = ("resync_fpuflags", idx)
        state.x87.known = [("resync_st", idx, i) for i in range(len(state.x87.known))]
    commit_clobber(ctx, ("resync", idx))


def _contained(value: Value, ctx: Context) -> bool:
    return value in ctx.matched_nodes


def _dead_or_contained(value: Value, ctx: Context) -> bool:
    return _is_scratch(value) or _contained(value, ctx)


def _assembled(value: Value, ctx: Context) -> bool:
    """A partial-register insert whose pieces are both accounted for
    (`and al, 1` on a loaded value: the load and the new byte were observed;
    or a constant under the new byte), so the register holds nothing
    unobserved."""
    if not (
        isinstance(value, tuple)
        and len(value) == 3
        and str(value[0]).startswith("ins_")
    ):
        return False
    old = value[1]
    old_ok = (isinstance(old, tuple) and old[:1] == ("imm",)) or _dead_or_contained(
        old, ctx
    )
    return old_ok and _dead_or_contained(value[2], ctx)


def _ins_split_ok(value_o: Value, value_r: Value, ctx: Context) -> bool:
    """A 16/8-bit result inserted into dead upper bits on both sides:
    the inserted part must be identical; the surrounding old bits are
    garbage as long as they came from consumed computations."""
    return (
        isinstance(value_o, tuple)
        and isinstance(value_r, tuple)
        and len(value_o) == 3
        and len(value_r) == 3
        and value_o[0] == value_r[0]
        and str(value_o[0]).startswith("ins_")
        and value_o[2] == value_r[2]
        and _dead_or_contained(value_o[1], ctx)
        and _dead_or_contained(value_r[1], ctx)
    )


# Caller-saved register families: dead at function end (eax is separately
# checked as the return value at every `ret`).
CALLER_SAVED = ("a", "c", "d")


def divergences_justified(ctx: Context, orig: SideState, recomp: SideState) -> bool:
    # pylint: disable=too-many-boolean-expressions
    """At a control transfer, refuse unjustified live divergent register state.

    Linear verification cannot prove live-out on both successors of a
    conditional. Divergent values that merely appear in the branch predicate
    are not a liveness proof: the taken path may still observe them. Scratch
    pairs remain allowed; anything else must match or the CFG verifier owns
    the pair.
    """
    del ctx
    for family in FAMILIES:
        value_o, value_r = orig.regs[family], recomp.regs[family]
        if value_o == value_r:
            continue
        if (
            isinstance(value_o, tuple)
            and isinstance(value_r, tuple)
            and len(value_o) == 3
            and len(value_r) == 3
            and value_o[0] == value_r[0]
            and str(value_o[0]).startswith("ins_")
            and value_o[2] == value_r[2]
            and _is_scratch(value_o[1])
            and _is_scratch(value_r[1])
        ):
            continue
        if _is_scratch(value_o) and _is_scratch(value_r):
            continue
        if family in CALLER_SAVED and (_is_scratch(value_o) or _is_scratch(value_r)):
            continue
        return False
    if len(orig.x87.known) != len(recomp.x87.known):
        return False
    for slot_o, slot_r in zip(orig.x87.known, recomp.x87.known):
        if slot_o != slot_r and not (_is_scratch(slot_o) and _is_scratch(slot_r)):
            return False
    return True


def addrs_from_meta(
    metas: list[InstructionMeta | None] | None,
) -> list[int | None] | None:
    if metas is None:
        return None
    return [meta.address if meta is not None else None for meta in metas]


def _control_destination(
    raw_operand: object,
    meta: InstructionMeta | None,
    addrs: list[int | None] | None,
) -> object:
    """Prefer local instruction-id; never a relative displacement."""
    if meta is None:
        return raw_operand
    if meta.branch_target is not None and addrs is not None:
        try:
            return ("L", addrs.index(meta.branch_target))
        except ValueError:
            pass
    if meta.control_target is not None:
        return ("ext", meta.control_target)
    if meta.branch_target is not None:
        return ("ext", ("unresolved", None, meta.branch_target))
    return raw_operand


def rewrite_control_observables(
    obs: list,
    meta: InstructionMeta | None,
    addrs: list[int | None] | None,
) -> None:
    if meta is None or addrs is None:
        return
    for index, entry in enumerate(obs):
        if entry and entry[0] in CONTROL_TAGS - {"jmpind"}:
            obs[index] = (
                *entry[:-1],
                _control_destination(entry[-1], meta, addrs),
            )


def _is_scratch(value: Value) -> bool:
    """Values with no computational content: the untouched initial register
    value, or the clobbered result of a call or string instruction. If such
    a value is left in a caller-saved register while the other side holds
    something else, the register is simply dead."""
    return (
        isinstance(value, tuple)
        and bool(value)
        and value[0]
        in (
            "init",
            "callret",
            "strres",
            "resync",
        )
    )


CALLEE_SAVED = ("b", "si", "di")


def aligned_indices(
    codes, orig_len: int, recomp_len: int
) -> list[tuple[int | None, int | None]] | None:
    """Pair up the two sequences (by index) for lockstep verification.
    Without diff opcodes, the sequences must have equal length. With them,
    unmatched insertions/deletions become one-sided entries, which the
    verifier only accepts for whitelisted unobservable instructions."""
    if codes is None:
        if orig_len != recomp_len:
            return None
        return [(i, i) for i in range(orig_len)]
    result: list[tuple[int | None, int | None]] = []
    for tag, i1, i2, j1, j2 in codes:
        if tag in ("equal", "replace"):
            paired = min(i2 - i1, j2 - j1)
            result.extend(zip(range(i1, i1 + paired), range(j1, j1 + paired)))
            result.extend((i, None) for i in range(i1 + paired, i2))
            result.extend((None, j) for j in range(j1 + paired, j2))
        elif tag == "delete":
            result.extend((i, None) for i in range(i1, i2))
        elif tag == "insert":
            result.extend((None, j) for j in range(j1, j2))
    return result


# Mnemonics with implicit memory or environment effects that capstone's
# operand list does not surface. Never stepped over via metadata.
_META_STEP_BLACKLIST = frozenset(
    {
        "xlatb",
        "pusha",
        "pushal",
        "pushad",
        "popa",
        "popal",
        "popad",
        "pushf",
        "pushfd",
        "popf",
        "popfd",
        "enter",
        "leave",
        "int",
        "int1",
        "int3",
        "into",
        "syscall",
        "sysenter",
        "iret",
        "iretd",
        "cpuid",
        "rdtsc",
        "in",
        "out",
        "hlt",
    }
)


def _meta_step(orig: SideState, recomp: SideState, meta, idx: int) -> bool:
    """Step both sides over an identical instruction outside the symbolic
    model, using capstone's structured facts about it. Sound only when the
    instruction touches nothing but registers and flags: every register it
    reads must be cross-equal, and every register it writes gets a fresh
    paired value. Anything with memory access, control flow, x87, or
    unmapped registers falls back to the full-synchronization rule."""
    # pylint: disable=too-many-return-statements,too-many-boolean-expressions
    if not getattr(meta, "register_access_known", True):
        return False
    if (
        meta.accesses_memory
        or meta.is_jump
        or meta.is_call
        or meta.is_ret
        or meta.mnemonic in _META_STEP_BLACKLIST
        or meta.mnemonic.startswith(("f", "rep"))
    ):
        return False

    read_families: set[str] = set()
    written_families: set[str] = set()
    for names, families in (
        (meta.regs_read, read_families),
        (meta.regs_written, written_families),
    ):
        for name in names:
            if name == "eflags":
                continue
            if name not in REGISTERS:
                # x87/MMX/SSE or segment registers: not modeled.
                return False
            family, part = REGISTERS[name]
            families.add(family)
            if part != "r32" and families is written_families:
                # A partial-register write preserves the remaining bits,
                # so the family must already agree for the fresh paired
                # value to be sound.
                read_families.add(family)

    for family in read_families:
        if orig.regs[family] != recomp.regs[family]:
            return False
    if meta.reads_flags and (orig.flags != recomp.flags or orig.carry != recomp.carry):
        return False

    for family in written_families:
        value = ("metastep", idx, family)
        orig.regs[family] = value
        recomp.regs[family] = value
    if meta.writes_flags:
        orig.flags = recomp.flags = ("metastep_flags", idx)
        orig.carry = recomp.carry = ("metastep_cf", idx)

    return True


# One-sided instructions that are never unobservable: control flow, the
# stack discipline (push/pop/leave/enter), x87 (stack-shape effects), and
# instructions that can fault on operand values (division).
_ONE_SIDED_BLACKLIST = frozenset(
    {"leave", "enter", "call", "ret", "jmp", "int3", "div", "idiv"}
    | set(JCC_MNEMONICS)
    | {"loop", "loope", "loopne", "jcxz", "jecxz"}
    | set(STRING_OPS)
)


def _one_sided_push_ok(state: SideState, ctx: Context, ins, idx: int) -> bool:
    """One side spills a register the other side never needed (a save
    around a region, or a scratch spill). The slot is private: it lies
    strictly below the entry stack pointer, no frame pointer has escaped,
    and the spill must be reclaimed before any call (a pushed value still
    live at a call would be an argument). The store is committed so later
    reads of the slot alias correctly."""
    value = read_operand(state, ctx, ins.operands[0])
    new_esp = esp_add(state.read_reg("esp"), -4)
    root, offset = unwind_spadd(new_esp)
    if root != ("init", "sp") or offset >= 0 or ctx.stack_escaped:
        return False
    if frame_pointer_value(value):
        return False
    obs = [("store", new_esp, "stack", value)]
    state.write_reg("esp", new_esp)
    tag = ("mem", ("scratch", idx), 0)
    ctx.mem_events.append((tag, (new_esp, 4, "push")))
    ctx.gen = tag
    ctx.scratch_pushes.append([state, offset, value, tag])
    invalidate_save_slots(ctx, obs)
    ctx.categories.add("callee_save_substitution")
    return True


def _one_sided_pop_ok(state: SideState, ctx: Context, ins) -> bool:
    """Reclaim of a one-sided spill (or a plain scratch read): only from
    the function's own private scratch. When it provably reads back an
    intact one-sided push, the popped register regains the exact pushed
    value, so callee-save round-trips stay externally clean."""
    if ins.operands[0][0] != "reg":
        return False
    esp = state.read_reg("esp")
    root, offset = unwind_spadd(esp)
    if root != ("init", "sp") or offset >= 0 or ctx.stack_escaped:
        return False
    tag = memory_load_tag(ctx, esp, 4, "pop")
    value: Value = ("load", esp, "stack", tag)
    for k, record in enumerate(ctx.scratch_pushes):
        if record[0] is state and record[1] == offset:
            if record[3] == tag:
                value = record[2]
            del ctx.scratch_pushes[k]
            break
    state.write_reg(ins.operands[0][1], value)
    state.write_reg("esp", esp_add(esp, 4))
    ctx.categories.add("callee_save_substitution")
    return True


def one_sided_ok(
    state: SideState,
    other: SideState,
    ctx: Context,
    idx: int,
    line: str,
    *,
    ins: Instruction | None = None,
    is_data: bool = False,
) -> bool:
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-arguments
    """Execute an instruction that exists on only one side. Any instruction
    with no observable effect (no store, call, branch or return) is allowed:
    its register and flag writes are validated downstream by the observables
    that consume them, or by the end-of-run divergence rules. A memory read
    may fault, so it incurs a trap-parity obligation: the other side must
    read the same address at the same memory generation somewhere in the
    same verification scope (the folded-load case). Control flow, stack
    adjustments, x87 and potentially-faulting arithmetic stay excluded."""
    if is_data or (ins is None and DATA_LINE_RE.match(line)):
        return False
    if ins is None:
        try:
            ins = parse_instruction(line)
        except (Reject, IndexError, KeyError, ValueError, TypeError):
            return False
    if ins.prefix or ins.mnemonic in _ONE_SIDED_BLACKLIST:
        return False
    if ins.mnemonic == "nop":
        ctx.categories.add("dead_operation")
        return True
    if ins.mnemonic.startswith("f"):
        return False
    try:
        if ins.mnemonic == "push" and len(ins.operands) == 1:
            return _one_sided_push_ok(state, ctx, ins, idx)
        if ins.mnemonic == "pop" and len(ins.operands) == 1:
            return _one_sided_pop_ok(state, ctx, ins)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return False
    reads_before = len(state.load_log)
    log_snapshot = set(state.load_log) if state.load_log else set()
    obs: list = []
    try:
        execute(state, ctx, idx, ins, obs)
        guard_state_size(state, ctx)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return False
    if obs:
        return False
    if len(state.load_log) > reads_before:
        for entry in state.load_log - log_snapshot:
            ctx.load_obligations.append((other, *entry))
        ctx.categories.add("load_folding")
    else:
        ctx.categories.add("dead_operation")
    return True


def _load_obligations_met(ctx: Context) -> bool:
    """Discharge the trap-parity obligations of one-sided memory reads:
    the other side must have read the same address at the same memory
    generation somewhere in the current verification scope."""
    return all(
        (address, gen) in other.load_log for other, address, gen in ctx.load_obligations
    )


def discharge_run_obligations(
    ctx: Context,
    orig: SideState,
    recomp: SideState,
    recorder: AnalysisRecorder | None,
    last_index_o: int | None = None,
    last_index_r: int | None = None,
) -> bool:
    """Shared end-of-run admission checklist for every verifier strategy.

    Divergent caller-saved registers must be dead (matched/consumed or
    scratch); callee-saved and SP must match; callee-save swaps must balance;
    load-folding obligations and frame-slot layouts must hold; x87 depth and
    live slots must agree.
    """
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    for family in FAMILIES:
        if orig.regs[family] == recomp.regs[family]:
            ctx.add_matched(orig.regs[family])
    for slot_o, slot_r in zip(orig.x87.known, recomp.x87.known):
        if slot_o == slot_r:
            ctx.add_matched(slot_o)

    dead_register_difference = False
    for family in FAMILIES:
        value_o, value_r = orig.regs[family], recomp.regs[family]
        if value_o == value_r:
            continue
        if family not in CALLER_SAVED:
            if recorder is not None:
                summary_o, summary_r = diagnostic_summaries(value_o, value_r)
                recorder.record_difference(
                    "preserved_state",
                    last_index_o,
                    last_index_r,
                    {"register": family, "value": summary_o},
                    {"register": family, "value": summary_r},
                )
            return False
        if _ins_split_ok(value_o, value_r, ctx):
            dead_register_difference = True
            continue
        for value in (value_o, value_r):
            if _is_scratch(value):
                dead_register_difference = True
                continue
            if not (_contained(value, ctx) or _assembled(value, ctx)):
                if recorder is not None:
                    recorder.mark_inconclusive("analysis_limit")
                return False
            dead_register_difference = True
    if dead_register_difference and "register_allocation" not in ctx.categories:
        ctx.categories.add("dead_operation")

    if ctx.save_stack:
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    if not _load_obligations_met(ctx):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    if not _slots_consistent(orig, recomp):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    if _uses_frame_slot_layout(orig, recomp):
        ctx.categories.add("frame_slot_layout")

    if orig.x87.state_key()[1:] != recomp.x87.state_key()[1:]:
        return False
    if len(orig.x87.known) != len(recomp.x87.known):
        return False
    for slot_o, slot_r in zip(orig.x87.known, recomp.x87.known):
        if slot_o != slot_r:
            if not (_contained(slot_o, ctx) and _contained(slot_r, ctx)):
                return False
    return True


def admit_unsupported_identical(
    orig: SideState,
    recomp: SideState,
    ctx: Context,
    idx: int,
    meta_o: InstructionMeta | None,
    meta_r: InstructionMeta | None,
) -> bool:
    # pylint: disable=too-many-positional-arguments
    """Shared policy for identical unsupported instructions on both sides."""
    if (
        meta_o is not None
        and meta_r is not None
        and _same_meta_effects(meta_o, meta_r)
        and _meta_step(orig, recomp, meta_o, idx)
    ):
        return True
    if not fully_synced(orig, recomp):
        return False
    resync((orig, recomp), idx, ctx)
    return True


def callee_save_swap(ctx: Context, ins_o, ins_r, obs_o, obs_r, orig, recomp) -> bool:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-boolean-expressions
    """Detect a balanced callee-save substitution: one side saves and
    restores e.g. esi where the other uses edi. The pushed values are the
    untouched initial registers, so the stores differ; treat the pair as
    bookkeeping and make the matching pops restore the initial values."""
    if ins_o.mnemonic != ins_r.mnemonic:
        return False
    if (
        ins_o.mnemonic == "push"
        and len(obs_o) == 1
        and len(obs_r) == 1
        and obs_o[0][0] == obs_r[0][0] == "store"
        and obs_o[0][1] == obs_r[0][1]
        and obs_o[0][3][0] == obs_r[0][3][0] == "init"
        and obs_o[0][3][1] in CALLEE_SAVED
        and obs_r[0][3][1] in CALLEE_SAVED
        and obs_o[0][3] != obs_r[0][3]
    ):
        ctx.save_stack.append([obs_o[0][3][1], obs_r[0][3][1], obs_o[0][1], True])
        ctx.categories.add("callee_save_substitution")
        return True
    if (
        ins_o.mnemonic == "pop"
        and ctx.save_stack
        and ins_o.operands
        and ins_r.operands
        and ins_o.operands[0][0] == "reg"
        and ins_r.operands[0][0] == "reg"
    ):
        family_o = REGISTERS[ins_o.operands[0][1]][0]
        family_r = REGISTERS[ins_r.operands[0][1]][0]
        saved_o, saved_r, slot_addr, valid = ctx.save_stack[-1]
        popped_o = orig.regs.get(family_o)
        popped_r = recomp.regs.get(family_r)
        if (
            (saved_o, saved_r) == (family_o, family_r)
            and valid
            # A frame address that escaped could have reached the saved
            # slot through a pointer we cannot see.
            and not orig.slots_escaped
            and not recomp.slots_escaped
            and isinstance(popped_o, tuple)
            and popped_o
            and popped_o[0] == "load"
            and popped_o[1] == slot_addr
            and isinstance(popped_r, tuple)
            and popped_r
            and popped_r[0] == "load"
            and popped_r[1] == slot_addr
        ):
            ctx.save_stack.pop()
            orig.regs[family_o] = ("init", family_o)
            recomp.regs[family_r] = ("init", family_r)
            return True
    return False


def invalidate_save_slots(ctx: Context, obs: list) -> None:
    """Any store that cannot be proven disjoint from a pending callee-save
    slot invalidates that record: the pop can no longer be trusted to
    restore the pushed value."""
    if not ctx.save_stack:
        return
    for entry in obs:
        if entry[0] != "store":
            continue
        address, size = entry[1], entry[2]
        if size == "stack":
            access: tuple = (address, 4, "push")
        else:
            access = (address, WIDTHS.get(size), False)
        for record in ctx.save_stack:
            if record[3] and not mem_disjoint((record[2], 4, "pop"), access):
                record[3] = False


def _slots_consistent(orig: SideState, recomp: SideState) -> bool:
    # pylint: disable=too-many-return-statements
    """Validate the frame-slot alpha-renaming: the two sides must map the
    same slot ids in the same order, and — when the layouts actually differ
    — every slot must be a self-contained region (known widths, no overlap
    between distinct slots, no escaped frame addresses)."""
    orig_disps = [
        d
        for d, slot in sorted(
            orig.slot_map.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)
        )
        if slot is not None
    ]
    recomp_disps = [
        d
        for d, slot in sorted(
            recomp.slot_map.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)
        )
        if slot is not None
    ]
    if len(orig_disps) != len(recomp_disps):
        return False
    if orig_disps == recomp_disps:
        # No renaming took place; nothing to prove.
        return True
    if orig.slots_escaped or recomp.slots_escaped:
        return False
    for state in (orig, recomp):
        widths: dict[int, set] = {}
        for disp, width in state.slot_accesses:
            if width is None:
                return False
            widths.setdefault(disp, set()).add(width)
        # Every renamed slot must be accessed with a single width: a byte
        # write followed by a dword read would pull the remaining bytes
        # from different (uninitialized) stack locations on each side.
        for accessed in widths.values():
            if len(accessed) != 1:
                return False
        spans = sorted((disp, next(iter(ws))) for disp, ws in widths.items())
        for (d1, w1), (d2, _) in zip(spans, spans[1:]):
            if d1 + w1 > d2:
                return False
    return True


def _uses_frame_slot_layout(orig: SideState, recomp: SideState) -> bool:
    orig_slots = sorted(d for d, slot in orig.slot_map.items() if slot is not None)
    recomp_slots = sorted(d for d, slot in recomp.slot_map.items() if slot is not None)
    return bool(orig_slots or recomp_slots) and orig_slots != recomp_slots


def _commutative_order_used(
    before_o: SideState,
    before_r: SideState,
    ctx: Context,
    ins_o: Instruction,
    ins_r: Instruction,
) -> bool:
    # pylint: disable=too-many-return-statements
    """Whether this paired operation needed commutative-order normalization."""
    if ins_o.mnemonic != ins_r.mnemonic:
        return False
    try:
        for operand_o, operand_r in zip(ins_o.operands, ins_r.operands):
            if operand_o[0] == operand_r[0] == "mem":
                terms_o = operand_o[3]
                terms_r = operand_r[3]
                if terms_o != terms_r and sorted(terms_o) == sorted(terms_r):
                    if mem_address(before_o, operand_o) == mem_address(
                        before_r, operand_r
                    ):
                        return True
        if ins_o.mnemonic in COMMUTATIVE_BINOPS and len(ins_o.operands) == 2:
            if len(ins_r.operands) != 2:
                return False
            a_o = read_operand(before_o, ctx, ins_o.operands[0])
            b_o = read_operand(before_o, ctx, ins_o.operands[1])
            a_r = read_operand(before_r, ctx, ins_r.operands[0])
            b_r = read_operand(before_r, ctx, ins_r.operands[1])
            if a_o == b_r and b_o == a_r and (a_o != a_r or b_o != b_r):
                return True
            if ins_o.mnemonic in ASSOCIATIVE_COMMUTATIVE_BINOPS:
                pair_o = vsort(a_o, b_o)
                pair_r = vsort(a_r, b_r)
                return pair_o != pair_r and commutative_result(
                    ins_o.mnemonic, a_o, b_o
                ) == commutative_result(ins_r.mnemonic, a_r, b_r)
            return False
        if ins_o.mnemonic in ("fadd", "fmul", "fiadd", "fimul"):
            if len(ins_o.operands) != 1 or len(ins_r.operands) != 1:
                return False
            a_o = before_o.x87.read(0)
            b_o = read_operand(before_o, ctx, ins_o.operands[0])
            a_r = before_r.x87.read(0)
            b_r = read_operand(before_r, ctx, ins_r.operands[0])
            return a_o == b_r and b_o == a_r and (a_o != a_r or b_o != b_r)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return False
    return False


def _same_meta_effects(
    orig: InstructionMeta | None, recomp: InstructionMeta | None
) -> bool:
    """Whether two metadata records describe the same non-address effects.

    The textual instruction is already identical when this is used.  Still,
    consume metadata only as a pair: trusting facts collected from one binary
    to model the other would defeat the purpose of structured input.
    Incomplete Capstone register-access info must not be treated as empty.
    """
    if orig is None or recomp is None:
        return False
    if not orig.register_access_known or not recomp.register_access_known:
        return False
    if not getattr(orig, "control_flow_known", True) or not getattr(
        recomp, "control_flow_known", True
    ):
        return False
    fields = (
        "mnemonic",
        "regs_read",
        "regs_written",
        "reads_flags",
        "writes_flags",
        "accesses_memory",
        "is_jump",
        "is_call",
        "is_ret",
    )
    return all(getattr(orig, field) == getattr(recomp, field) for field in fields)


def record_pair_categories(
    ctx: Context,
    before_o: SideState,
    before_r: SideState,
    after_o: SideState,
    after_r: SideState,
    ins_o: Instruction,
    ins_r: Instruction,
    obs_o: list,
    obs_r: list,
) -> None:
    """Name the compiler entropy an agreeing instruction pair relied on."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if any(
        family_o != family_r and value_o == value_r
        for family_o, value_o in after_o.regs.items()
        if value_o is not before_o.regs[family_o]
        for family_r, value_r in after_r.regs.items()
        if value_r is not before_r.regs[family_r]
    ):
        ctx.categories.add("register_allocation")
    if _commutative_order_used(before_o, before_r, ctx, ins_o, ins_r):
        ctx.categories.add("commutative_order")
    if (
        any(entry[0] == "branch" for entry in obs_o)
        and obs_o == obs_r
        and (ins_o.mnemonic != ins_r.mnemonic or before_o.flags != before_r.flags)
    ):
        ctx.categories.add("condition_inversion")


def observations_agree(ctx: Context, obs_o: list, obs_r: list) -> bool:
    """Equal observations, or (with z3-solver installed) observations whose
    values are proven equal as bit-vectors. Such a proof is recorded as an
    algebraic identity, and both sides' values become matched evidence."""
    if obs_o == obs_r:
        return True
    if not bitvector.observations_equal(obs_o, obs_r):
        return False
    ctx.categories.add("algebraic_identity")
    for entry in obs_r:
        ctx.add_matched(entry)
    return True


def accept_agreeing_pair(
    ctx: Context,
    index_o: int,
    index_r: int,
    before: tuple[SideState, SideState],
    after: tuple[SideState, SideState],
    ins: tuple[Instruction, Instruction],
    obs: tuple[list, list],
    meta: tuple[InstructionMeta | None, InstructionMeta | None],
) -> bool:
    """CFG strategies' per-pair check: the observables must agree; then the
    pair's effects become matched evidence. Records the difference and
    returns False otherwise."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    obs_o, obs_r = obs
    if not observations_agree(ctx, obs_o, obs_r):
        record_observable_difference(
            ctx, index_o, index_r, ins[0], ins[1], obs_o, obs_r, meta[0], meta[1]
        )
        return False
    invalidate_save_slots(ctx, obs_o)
    for obs_entry in obs_o:
        ctx.add_matched(obs_entry)
    record_pair_categories(ctx, *before, *after, *ins, obs_o, obs_r)
    return True
