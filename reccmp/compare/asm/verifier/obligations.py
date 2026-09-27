"""Rules shared by every strategy for when two diverging states may still
be equivalent: synchronization, one-sided instructions, callee-save
swaps, frame slots and the end-of-run admission checklist."""

from __future__ import annotations

from collections.abc import Sequence

from reccmp.compare.asm.ir import DecodedInstruction
from reccmp.compare.asm.model import FAMILY_REGISTER, REGISTERS, Reject
from reccmp.compare.asm.operand import Mem, Reg
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import (
    Constant,
    Init,
    Insert,
    Load,
    RegisterPart,
    Resync,
    Value,
    mem_disjoint,
    unwind_spadd,
)
from reccmp.compare.asm.verifier.evidence import (
    record_observable_difference,
)
from reccmp.compare.asm.verifier.render import render
from reccmp.compare.asm.verifier.semantics import (
    esp_add,
    execute,
    mem_address,
    read_operand,
)
from reccmp.compare.asm.verifier.state import (
    Branch,
    CalleeSaveSubstitution,
    JCC_MNEMONICS,
    LoadObligation,
    Observation,
    ScratchPush,
    Store,
    STRING_OPS,
    is_scratch,
    ASSOCIATIVE_COMMUTATIVE_BINOPS,
    COMMUTATIVE_BINOPS,
    FAMILIES,
    WIDTHS,
    Context,
    SideState,
    commit_clobber,
    commutative_result,
    frame_pointer_value,
    guard_state_size,
    memory_load_tag,
    observation_values,
    vsort,
)
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    DifferenceKind,
    EffectiveReason,
    InconclusiveReason,
    Observed,
)
from reccmp.types import ImageId

# ---------------------------------------------------------------------------
# Lockstep driver


def switch_index_observation(state: SideState, ins: DecodedInstruction) -> tuple:
    """Index register values that select a recognized switch-table case."""
    op = ins.operands[0]
    assert isinstance(op, Mem)
    return tuple(
        (term.scale, state.read_reg(term.register))
        for term in sorted(op.terms, key=lambda t: (-t.scale, t.register))
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
            state.regs[family] = Resync(idx, family)
        state.flags = Resync(idx, "flags")
        state.carry = Resync(idx, "carry")
        state.fpu_flags = Resync(idx, "fpuflags")
        state.x87.known = [
            Resync(idx, ("st", index)) for index in range(len(state.x87.known))
        ]
    commit_clobber(ctx, Resync(idx, "memory"))


def _contained(value: Value, ctx: Context) -> bool:
    return value in ctx.matched_nodes


def _dead_or_contained(value: Value, ctx: Context) -> bool:
    return is_scratch(value) or _contained(value, ctx)


def _assembled(value: Value, ctx: Context) -> bool:
    """A partial-register insert whose pieces are both accounted for
    (`and al, 1` on a loaded value: the load and the new byte were observed;
    or a constant under the new byte), so the register holds nothing
    unobserved."""
    if not isinstance(value, Insert):
        return False
    old_ok = isinstance(value.old, Constant) or _dead_or_contained(value.old, ctx)
    return old_ok and _dead_or_contained(value.new, ctx)


def _returned_insert(value: Value, ctx: Context) -> bool:
    """eax holding a returned ``al`` or ``ax`` (see FunctionMetadata's
    return_kind) inserted over other bits: the bits outside the return are
    dead, and the returned part is already observed."""
    kind = ctx.metadata.return_kind if ctx.metadata is not None else "unknown"
    match value:
        case Insert(RegisterPart.LOW8, _, part) if kind == "i8":
            return _contained(part, ctx)
        case Insert(RegisterPart.LOW16, _, part) if kind == "i16":
            return _contained(part, ctx)
    return False


def _ins_split_ok(value_o: Value, value_r: Value, ctx: Context) -> bool:
    """A 16/8-bit result inserted into dead upper bits on both sides:
    the inserted part must be identical; the surrounding old bits are
    garbage as long as they came from consumed computations or are a
    constant (`xor eax, eax` before `setcc al`)."""
    return (
        isinstance(value_o, Insert)
        and isinstance(value_r, Insert)
        and value_o.part is value_r.part
        and value_o.new == value_r.new
        and _old_bits_ok(value_o.old, ctx)
        and _old_bits_ok(value_r.old, ctx)
    )


def _inserted_over(value: Value, base: Value, ctx: Context) -> bool:
    """``value`` is ``base`` with parts overwritten by constants or observed
    values (`mov cl, [...]` over the other side's ecx): the two differ in
    nothing unobserved."""
    match value:
        case _ if value == base:
            return True
        case Insert(old=old, new=new):
            return _old_bits_ok(new, ctx) and _inserted_over(old, base, ctx)
    return False


def _old_bits_ok(value: Value, ctx: Context) -> bool:
    is_constant = isinstance(value, Constant)
    return is_constant or _dead_or_contained(value, ctx)


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
            isinstance(value_o, Insert)
            and isinstance(value_r, Insert)
            and value_o.part is value_r.part
            and value_o.new == value_r.new
            and is_scratch(value_o.old)
            and is_scratch(value_r.old)
        ):
            continue
        if is_scratch(value_o) and is_scratch(value_r):
            continue
        if family in CALLER_SAVED and (is_scratch(value_o) or is_scratch(value_r)):
            continue
        return False
    if len(orig.x87.known) != len(recomp.x87.known):
        return False
    for slot_o, slot_r in zip(orig.x87.known, recomp.x87.known):
        if slot_o != slot_r and not (is_scratch(slot_o) and is_scratch(slot_r)):
            return False
    return True


CALLEE_SAVED = ("b", "si", "di")


def aligned_indices(
    codes, orig_len: int, recomp_len: int
) -> list[tuple[int | None, int | None]] | None:
    """Pair up the two sequences (by index) for lockstep verification.
    Without diff opcodes, the sequences must have equal length. With them,
    unmatched insertions/deletions become one-sided entries, which the
    verifier only accepts for whitelisted unobservable instructions. None
    unless every row of both sequences is visited once, in order."""
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
    orig_order = [i for i, _ in result if i is not None]
    recomp_order = [j for _, j in result if j is not None]
    if orig_order != list(range(orig_len)) or recomp_order != list(range(recomp_len)):
        return None
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
    if not meta.register_access_known:
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
_NEVER_ONE_SIDED = frozenset(
    {"leave", "enter", "call", "ret", "jmp", "int3", "div", "idiv"}
    | set(JCC_MNEMONICS)
    | {"loop", "loope", "loopne", "jcxz", "jecxz"}
    | set(STRING_OPS)
)


def may_be_one_sided(ins: DecodedInstruction) -> bool:
    """Whether the verifier may accept ``ins`` on one side only (see
    obligations.one_sided_ok): never control flow, stack frame setup, x87,
    string or prefixed instructions, or potentially-faulting division."""
    return not (
        ins.prefix or ins.mnemonic in _NEVER_ONE_SIDED or ins.mnemonic.startswith("f")
    )


def _one_sided_push_ok(
    side: ImageId, state: SideState, ctx: Context, ins, idx: int
) -> bool:
    """One side spills a register the other side never needed (a save
    around a region, or a scratch spill). The slot is private: it lies
    strictly below the entry stack pointer, no frame pointer has escaped,
    and the spill must be reclaimed before any call (a pushed value still
    live at a call would be an argument). The store is committed so later
    reads of the slot alias correctly."""
    value = read_operand(state, ctx, ins.operands[0])
    new_esp = esp_add(state.read_reg("esp"), -4)
    root, offset = unwind_spadd(new_esp)
    if root != Init("sp") or offset >= 0 or ctx.stack_escaped:
        return False
    if frame_pointer_value(value):
        return False
    obs: list[Observation] = [Store(new_esp, "stack", value)]
    state.write_reg("esp", new_esp)
    tag = ("mem", ("scratch", idx), 0)
    ctx.mem_events.append((tag, (new_esp, 4, "push")))
    ctx.gen = tag
    ctx.scratch_pushes.append(ScratchPush(side, offset, value, tag))
    invalidate_save_slots(ctx, obs)
    ctx.categories.add(EffectiveReason.CALLEE_SAVE_SUBSTITUTION)
    return True


def _one_sided_pop_ok(side: ImageId, state: SideState, ctx: Context, ins) -> bool:
    """Reclaim of a one-sided spill (or a plain scratch read): only from
    the function's own private scratch. When it provably reads back an
    intact one-sided push, the popped register regains the exact pushed
    value, so callee-save round-trips stay externally clean."""
    match ins.operands[0]:
        case Reg(name):
            register = name
        case _:
            return False
    esp = state.read_reg("esp")
    root, offset = unwind_spadd(esp)
    if root != Init("sp") or offset >= 0 or ctx.stack_escaped:
        return False
    tag = memory_load_tag(ctx, esp, 4, "pop")
    value: Value = Load(esp, "stack", tag)
    for k, record in enumerate(ctx.scratch_pushes):
        if record.side is side and record.offset == offset:
            if record.tag == tag:
                value = record.value
            del ctx.scratch_pushes[k]
            break
    state.write_reg(register, value)
    state.write_reg("esp", esp_add(esp, 4))
    ctx.categories.add(EffectiveReason.CALLEE_SAVE_SUBSTITUTION)
    return True


def one_sided_ok(
    side: ImageId,
    state: SideState,
    ctx: Context,
    idx: int,
    ins: DecodedInstruction,
) -> bool:
    # pylint: disable=too-many-return-statements
    """Execute an instruction that exists on only one side. Any instruction
    with no observable effect (no store, call, branch or return) is allowed:
    its register and flag writes are validated downstream by the observables
    that consume them, or by the end-of-run divergence rules. A memory read
    may fault, so it incurs a trap-parity obligation: the other side must
    read the same address at the same memory generation somewhere in the
    same verification scope (the folded-load case). Control flow, stack
    adjustments, x87 and potentially-faulting arithmetic stay excluded."""
    if not may_be_one_sided(ins):
        return False
    if ins.mnemonic == "nop":
        ctx.categories.add(EffectiveReason.DEAD_OPERATION)
        return True
    try:
        # With the frame promoted, a push or pop is a slot like any other.
        if state.frame is None and ins.mnemonic == "push" and len(ins.operands) == 1:
            return _one_sided_push_ok(side, state, ctx, ins, idx)
        if state.frame is None and ins.mnemonic == "pop" and len(ins.operands) == 1:
            return _one_sided_pop_ok(side, state, ctx, ins)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return False
    reads_before = len(state.load_log)
    log_snapshot = set(state.load_log) if state.load_log else set()
    obs: list[Observation] = []
    try:
        execute(state, ctx, idx, ins, obs)
        guard_state_size(state, ctx)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return False
    if obs:
        return False
    if len(state.load_log) > reads_before:
        other_side = ImageId.RECOMP if side is ImageId.ORIG else ImageId.ORIG
        for address, generation in state.load_log - log_snapshot:
            ctx.load_obligations.append(LoadObligation(other_side, address, generation))
        ctx.categories.add(EffectiveReason.LOAD_FOLDING)
    else:
        ctx.categories.add(EffectiveReason.DEAD_OPERATION)
    return True


def _load_obligations_met(ctx: Context, orig: SideState, recomp: SideState) -> bool:
    """Discharge the trap-parity obligations of one-sided memory reads:
    the other side must have read the same address at the same memory
    generation somewhere in the current verification scope."""
    states = {ImageId.ORIG: orig, ImageId.RECOMP: recomp}
    return all(
        (obligation.address, obligation.generation) in states[obligation.side].load_log
        for obligation in ctx.load_obligations
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
                recorder.record_difference(
                    DifferenceKind.PRESERVED_STATE,
                    last_index_o,
                    last_index_r,
                    Observed(value=render(value_o), register=FAMILY_REGISTER[family]),
                    Observed(value=render(value_r), register=FAMILY_REGISTER[family]),
                )
            return False
        if (
            _ins_split_ok(value_o, value_r, ctx)
            or _inserted_over(value_o, value_r, ctx)
            or _inserted_over(value_r, value_o, ctx)
        ):
            dead_register_difference = True
            continue
        for value in (value_o, value_r):
            if is_scratch(value):
                dead_register_difference = True
                continue
            if family == "a" and _returned_insert(value, ctx):
                continue
            if not (_contained(value, ctx) or _assembled(value, ctx)):
                if recorder is not None:
                    recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
                return False
            dead_register_difference = True
    if (
        dead_register_difference
        and EffectiveReason.REGISTER_ALLOCATION not in ctx.categories
    ):
        ctx.categories.add(EffectiveReason.DEAD_OPERATION)

    if ctx.save_stack:
        if recorder is not None:
            recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        return False
    if not _load_obligations_met(ctx, orig, recomp):
        if recorder is not None:
            recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        return False
    if not _slots_consistent(orig, recomp):
        if recorder is not None:
            recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        return False
    if _uses_frame_slot_layout(orig, recomp):
        ctx.categories.add(EffectiveReason.FRAME_SLOT_LAYOUT)

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
    meta_o: DecodedInstruction | None,
    meta_r: DecodedInstruction | None,
) -> bool:
    # pylint: disable=too-many-positional-arguments
    """Shared policy for identical unsupported instructions on both sides:
    step over them by their Capstone effects when both agree and are
    known, else only from fully synchronized states."""
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


def callee_save_swap(
    ctx: Context,
    ins_o,
    ins_r,
    obs_o: list[Observation],
    obs_r: list[Observation],
    orig,
    recomp,
) -> bool:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-boolean-expressions
    """Detect a balanced callee-save substitution: one side saves and
    restores e.g. esi where the other uses edi. The pushed values are the
    untouched initial registers, so the stores differ; treat the pair as
    bookkeeping and make the matching pops restore the initial values."""
    if ins_o.mnemonic != ins_r.mnemonic:
        return False
    stores = (
        obs_o[0] if len(obs_o) == 1 and isinstance(obs_o[0], Store) else None,
        obs_r[0] if len(obs_r) == 1 and isinstance(obs_r[0], Store) else None,
    )
    if (
        ins_o.mnemonic == "push"
        and stores[0] is not None
        and stores[1] is not None
        and stores[0].address == stores[1].address
        and isinstance(stores[0].value, Init)
        and isinstance(stores[1].value, Init)
        and stores[0].value.family in CALLEE_SAVED
        and stores[1].value.family in CALLEE_SAVED
        and stores[0].value != stores[1].value
    ):
        ctx.save_stack.append(
            CalleeSaveSubstitution(
                stores[0].value.family, stores[1].value.family, stores[0].address
            )
        )
        ctx.categories.add(EffectiveReason.CALLEE_SAVE_SUBSTITUTION)
        return True
    match ins_o.operands, ins_r.operands:
        case (Reg(name_o),), (Reg(name_r),) if (
            ins_o.mnemonic == "pop" and ctx.save_stack
        ):
            return _restores_swapped_save(
                ctx, orig, recomp, REGISTERS[name_o][0], REGISTERS[name_r][0]
            )
    return False


def _restores_swapped_save(
    ctx: Context, orig: SideState, recomp: SideState, family_o: str, family_r: str
) -> bool:
    """A pair of pops that restores the innermost substituted callee save:
    both read back its intact slot, so each register regains its initial
    value."""
    substitution = ctx.save_stack[-1]
    popped_o = orig.regs.get(family_o)
    popped_r = recomp.regs.get(family_r)
    if not (
        (substitution.orig_family, substitution.recomp_family) == (family_o, family_r)
        and substitution.valid
        # A frame address that escaped could have reached the saved
        # slot through a pointer we cannot see.
        and not orig.slots_escaped
        and not recomp.slots_escaped
        and isinstance(popped_o, Load)
        and popped_o.address == substitution.address
        and isinstance(popped_r, Load)
        and popped_r.address == substitution.address
    ):
        return False
    ctx.save_stack.pop()
    orig.regs[family_o] = Init(family_o)
    recomp.regs[family_r] = Init(family_r)
    return True


def invalidate_save_slots(ctx: Context, obs: Sequence[Observation]) -> None:
    """Any store that cannot be proven disjoint from a pending callee-save
    slot invalidates that record: the pop can no longer be trusted to
    restore the pushed value."""
    if not ctx.save_stack:
        return
    for entry in obs:
        if not isinstance(entry, Store):
            continue
        if entry.size == "stack":
            access: tuple = (entry.address, 4, "push")
        else:
            access = (entry.address, WIDTHS.get(entry.size), False)
        for record in ctx.save_stack:
            if record.valid and not mem_disjoint((record.address, 4, "pop"), access):
                record.valid = False


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
    ins_o: DecodedInstruction,
    ins_r: DecodedInstruction,
) -> bool:
    # pylint: disable=too-many-return-statements
    """Whether this paired operation needed commutative-order normalization."""
    if ins_o.mnemonic != ins_r.mnemonic:
        return False
    try:
        for operand_o, operand_r in zip(ins_o.operands, ins_r.operands):
            if isinstance(operand_o, Mem) and isinstance(operand_r, Mem):
                terms_o = operand_o.terms
                terms_r = operand_r.terms
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
    orig: DecodedInstruction | None, recomp: DecodedInstruction | None
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
    if not orig.control_flow_known or not recomp.control_flow_known:
        return False
    return (
        orig.mnemonic,
        orig.regs_read,
        orig.regs_written,
        orig.reads_flags,
        orig.writes_flags,
        orig.accesses_memory,
        orig.flow,
    ) == (
        recomp.mnemonic,
        recomp.regs_read,
        recomp.regs_written,
        recomp.reads_flags,
        recomp.writes_flags,
        recomp.accesses_memory,
        recomp.flow,
    )


def record_pair_categories(
    ctx: Context,
    before_o: SideState,
    before_r: SideState,
    after_o: SideState,
    after_r: SideState,
    ins_o: DecodedInstruction,
    ins_r: DecodedInstruction,
    obs_o: list[Observation],
    obs_r: list[Observation],
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
        ctx.categories.add(EffectiveReason.REGISTER_ALLOCATION)
    if _commutative_order_used(before_o, before_r, ctx, ins_o, ins_r):
        ctx.categories.add(EffectiveReason.COMMUTATIVE_ORDER)
    if (
        any(isinstance(entry, Branch) for entry in obs_o)
        and obs_o == obs_r
        and (ins_o.mnemonic != ins_r.mnemonic or before_o.flags != before_r.flags)
    ):
        ctx.categories.add(EffectiveReason.CONDITION_INVERSION)


def observations_agree(
    ctx: Context, obs_o: list[Observation], obs_r: list[Observation]
) -> bool:
    """Equal observations, or, where the configuration allows algebraic
    identities, observations whose values z3 proves equal as bit-vectors.
    Such a proof is recorded as an algebraic identity, and both sides'
    values become matched evidence."""
    if obs_o == obs_r:
        return True
    if ctx.metadata is not None and not ctx.metadata.algebraic_identities:
        return False
    if not bitvector.observations_equal(obs_o, obs_r):
        return False
    ctx.categories.add(EffectiveReason.ALGEBRAIC_IDENTITY)
    for entry in obs_r:
        for value in observation_values(entry):
            ctx.add_matched(value)
    return True


def accept_agreeing_pair(
    ctx: Context,
    index_o: int,
    index_r: int,
    before: tuple[SideState, SideState],
    after: tuple[SideState, SideState],
    ins: tuple[DecodedInstruction, DecodedInstruction],
    obs: tuple[list[Observation], list[Observation]],
) -> bool:
    """CFG strategies' per-pair check: the observables must agree; then the
    pair's effects become matched evidence. Records the difference and
    returns False otherwise."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    obs_o, obs_r = obs
    if not observations_agree(ctx, obs_o, obs_r):
        record_observable_difference(
            ctx, index_o, index_r, ins[0], ins[1], obs_o, obs_r
        )
        return False
    invalidate_save_slots(ctx, obs_o)
    for obs_entry in obs_o:
        for value in observation_values(obs_entry):
            ctx.add_matched(value)
    record_pair_categories(ctx, *before, *after, *ins, obs_o, obs_r)
    return True
