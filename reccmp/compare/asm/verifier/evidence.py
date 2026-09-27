"""Readable summaries of symbolic values and recorded difference facts."""

from __future__ import annotations

from reccmp.compare.asm.ir import DecodedInstruction
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.operand import Imm, Mem, Reg, Sym
from reccmp.compare.asm.verifier.semantics import mem_address
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    WIDTHS,
    Context,
    SideState,
    clone_state,
    register_arguments,
)
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.render import render
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    DifferenceKind,
    InconclusiveReason,
    Observed,
    SolverResult,
)
from reccmp.types import ImageId


def _solved(values: tuple | None) -> dict:
    """``record_difference`` arguments for a value difference: its symbolic
    values, and what the solver says about them."""
    if values is None:
        return {}
    return {"values": values, "solver": bitvector.compare(values)}


def shown(value) -> Observed:
    """A side that computes ``value``."""
    return Observed(value=render(value))


def transfer(ins: DecodedInstruction, target_index: int | None = None) -> Observed:
    """A control transfer's destination: its operand, address and row."""
    return Observed(
        operand=ins.operands[0] if ins.operands else None,
        target=ins.branch_target,
        target_index=target_index,
    )


def _target_index(
    recorder: AnalysisRecorder | None, which: ImageId, ins: DecodedInstruction
) -> int | None:
    if recorder is None:
        return None
    return recorder.image(which).index_of(ins.branch_target)


def _checked_call_registers(ctx: Context, ins: DecodedInstruction) -> list[str]:
    facts = None
    if ctx.metadata is not None and ctx.metadata.call_facts is not None:
        match ins.operands:
            case (Sym(ref), *_):
                facts = ctx.metadata.call_facts(ref.identity)
    ecx_argument, edx_argument = register_arguments(facts)
    return [
        register
        for register, used in (("ecx", ecx_argument), ("edx", edx_argument))
        if used
    ]


def _operand_addresses(
    states: tuple[SideState, SideState] | None, op_o: Mem, op_r: Mem
) -> tuple | None:
    """The two memory operands' addresses as values, in the two states
    before their instructions; None when they cannot be computed."""
    if states is None:
        return None
    try:
        # mem_address records frame-slot uses; work on copies.
        return (
            ("addr", mem_address(clone_state(states[0]), op_o)),
            ("addr", mem_address(clone_state(states[1]), op_r)),
            32,
            "value",
        )
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return None


def _stack_adjustment(ins: DecodedInstruction) -> bool:
    """`add/sub esp, N`: a frame size or an argument cleanup, which the
    stack pointer's own value accounts for."""
    return (
        ins.mnemonic in ("add", "sub")
        and len(ins.operands) == 2
        and ins.operands[0] == Reg("esp")
    )


def record_operand_candidate(
    ctx: Context,
    index_o: int,
    index_r: int,
    ins_o: DecodedInstruction,
    ins_r: DecodedInstruction,
    states: tuple[SideState, SideState] | None = None,
) -> None:
    """Record a candidate difference between two paired instructions'
    operands, for when no observable says more. ``states`` (before the
    instructions) let operands that denote the same address pass."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-return-statements
    recorder = ctx.recorder
    if recorder is None or ins_o.mnemonic != ins_r.mnemonic:
        return
    if _stack_adjustment(ins_o) and _stack_adjustment(ins_r):
        return
    if ins_o.is_conditional:
        return
    if ins_o.is_call:
        # CALL observations diagnose canonical direct/indirect targets and
        # register arguments after symbolic execution. A raw memory-operand
        # candidate here would re-expose the physical vtable register after a
        # virtual target had already been proved equivalent.
        return
    for op_o, op_r in zip(ins_o.operands, ins_r.operands):
        if op_o == op_r:
            continue
        match op_o, op_r:
            case Mem(), Mem():
                addresses = _operand_addresses(states, op_o, op_r)
                if addresses is not None and (
                    addresses[0] == addresses[1]
                    or bitvector.compare(addresses).result is SolverResult.PROVED
                ):
                    # Registers renamed, or one address spelled two ways.
                    continue
                recorder.record_difference(
                    DifferenceKind.MEMORY_ADDRESS,
                    index_o,
                    index_r,
                    Observed(operand=op_o),
                    Observed(operand=op_r),
                    candidate=True,
                    **_solved(addresses),
                )
                return
            case Imm(), Imm():
                recorder.record_difference(
                    DifferenceKind.IMMEDIATE_VALUE,
                    index_o,
                    index_r,
                    Observed(operand=op_o),
                    Observed(operand=op_r),
                    candidate=True,
                )
                return
            case Sym(), Sym():
                recorder.record_difference(
                    DifferenceKind.SYMBOL_RESOLUTION,
                    index_o,
                    index_r,
                    Observed(operand=op_o),
                    Observed(operand=op_r),
                    candidate=True,
                )
                return


def record_observable_difference(
    ctx: Context,
    index_o: int,
    index_r: int,
    ins_o: DecodedInstruction,
    ins_r: DecodedInstruction,
    obs_o: list,
    obs_r: list,
) -> None:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-return-statements,too-many-locals
    """Classify the first differing observable at a trusted paired point."""
    recorder = ctx.recorder
    if recorder is None or recorder.difference is not None:
        return
    first_o: tuple = obs_o[0] if obs_o else ()
    first_r: tuple = obs_r[0] if obs_r else ()
    tag_o = first_o[0] if first_o else None
    tag_r = first_r[0] if first_r else None

    if tag_o == tag_r == "call":
        if first_o[1] != first_r[1]:
            recorder.record_difference(
                DifferenceKind.CALL_TARGET,
                index_o,
                index_r,
                transfer(ins_o),
                transfer(ins_r),
            )
            return
        registers = _checked_call_registers(ctx, ins_o)
        for position, register in enumerate(registers, start=2):
            if position >= min(len(first_o), len(first_r)):
                break
            if first_o[position] != first_r[position]:
                recorder.record_difference(
                    DifferenceKind.CALL_ARGUMENT,
                    index_o,
                    index_r,
                    Observed(value=render(first_o[position]), register=register),
                    Observed(value=render(first_r[position]), register=register),
                )
                return
        # With the frame promoted, the arguments the callee reads from each
        # side's stack follow the call (see iso_cfg._pass_frame).
        passed_o = next((e for e in obs_o if e[0] == "frame_args"), None)
        passed_r = next((e for e in obs_r if e[0] == "frame_args"), None)
        if passed_o != passed_r:
            recorder.record_difference(
                DifferenceKind.CALL_ARGUMENT,
                index_o,
                index_r,
                Observed(value=render(passed_o)),
                Observed(value=render(passed_r)),
            )
            return

    if tag_o == tag_r == "store":
        if first_o[1] != first_r[1]:
            recorder.record_difference(
                DifferenceKind.MEMORY_ADDRESS,
                index_o,
                index_r,
                Observed(operand=ins_o.operands[0] if ins_o.operands else None),
                Observed(operand=ins_r.operands[0] if ins_r.operands else None),
                **_solved((("addr", first_o[1]), ("addr", first_r[1]), 32, "value")),
            )
            return
        if first_o[3] != first_r[3]:
            width = WIDTHS.get(first_o[2])
            recorder.record_difference(
                DifferenceKind.MEMORY_VALUE,
                index_o,
                index_r,
                shown(first_o[3]),
                shown(first_r[3]),
                **_solved(
                    (first_o[3], first_r[3], 8 * width, "value")
                    if width is not None and first_o[2] == first_r[2]
                    else None
                ),
            )
            return

    branch_tags = CONTROL_TAGS - {"jmpind"}
    if tag_o in branch_tags and tag_r in branch_tags:
        predicate_o = first_o[1] if tag_o == "branch" else None
        predicate_r = first_r[1] if tag_r == "branch" else None
        if predicate_o != predicate_r:
            recorder.record_difference(
                DifferenceKind.BRANCH_CONDITION,
                index_o,
                index_r,
                shown(predicate_o),
                shown(predicate_r),
                **_solved(
                    (predicate_o, predicate_r, None, "predicate")
                    if predicate_o is not None and predicate_r is not None
                    else None
                ),
            )
            return
        target_o = _target_index(recorder, ImageId.ORIG, ins_o)
        target_r = _target_index(recorder, ImageId.RECOMP, ins_r)
        recorder.record_difference(
            DifferenceKind.BRANCH_TARGET,
            index_o,
            index_r,
            transfer(ins_o, target_o),
            transfer(ins_r, target_r),
        )
        return

    for entry_o, entry_r in zip(obs_o, obs_r):
        if entry_o == entry_r:
            continue
        if entry_o and entry_r and entry_o[0] == entry_r[0]:
            if entry_o[0] in ("retval", "retfpu"):
                recorder.record_difference(
                    DifferenceKind.RETURN_VALUE,
                    index_o,
                    index_r,
                    shown(entry_o[1]),
                    shown(entry_r[1]),
                    **_solved(
                        (entry_o[1], entry_r[1], None, "value")
                        if entry_o[0] == "retval" and len(entry_o) == len(entry_r) == 2
                        else None
                    ),
                )
                return
            if entry_o[0] in ("retsaved", "retstack"):
                recorder.record_difference(
                    DifferenceKind.PRESERVED_STATE,
                    index_o,
                    index_r,
                    shown(entry_o[1]),
                    shown(entry_r[1]),
                )
                return
    recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
