"""Readable summaries of symbolic values and recorded difference facts."""

from __future__ import annotations

import hashlib

from reccmp.compare.asm.ir import DecodedInstruction
from reccmp.compare.asm.model import (
    format_operand,
    Reject,
    Reference,
    operand_identity,
)
from reccmp.compare.asm.verifier.semantics import mem_address
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    JCC_MNEMONICS,
    WIDTHS,
    Context,
    SideState,
    clone_state,
    register_arguments,
)
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.diagnosis import AnalysisRecorder


def _symbolic_values(values: tuple | None) -> dict:
    """``record_difference`` arguments for a value difference: its symbolic
    values, and what the solver says about them."""
    if values is None:
        return {}
    return {"values": values, "solver": bitvector.compare(values).summary()}


def _symbolic_summary(value, depth: int = 0) -> str:
    # pylint: disable=too-many-return-statements
    """Small stable rendering for diagnosis; never expands the whole DAG."""
    if not isinstance(value, tuple) or not value:
        return str(value)
    tag = str(value[0])
    if tag == "imm" and len(value) > 1:
        return str(value[1])
    if tag == "sym" and len(value) > 1:
        return str(value[1])
    if tag == "init" and len(value) > 1:
        return f"initial:{value[1]}"
    if tag == "load" and len(value) > 1:
        return f"load:{_symbolic_summary(value[1], depth + 1)}"
    if tag == "mem" and len(value) >= 5:
        symbols = value[4]
        if symbols:
            return str(symbols[0][1])
        terms = value[2]
        base = _symbolic_summary(terms[0][0], depth + 1) if terms else "absolute"
        return f"{base}{int(value[3]):+d}"
    if depth >= 1:
        return tag
    children = [
        _symbolic_summary(child, depth + 1)
        for child in value[1:3]
        if isinstance(child, tuple)
    ]
    return f"{tag}:{','.join(children)}" if children else tag


def _symbolic_fingerprint(value) -> str:
    """Bounded deterministic identity for two values with the same summary."""
    digest = hashlib.blake2s(digest_size=4)
    pending = [value]
    visited = 0
    while pending and visited < 256:
        node = pending.pop()
        visited += 1
        if isinstance(node, tuple):
            digest.update(f"tuple:{len(node)}:".encode())
            pending.extend(reversed(node))
        else:
            digest.update(f"{type(node).__name__}:{node!s}:".encode())
    if pending:
        digest.update(b"truncated")
    return digest.hexdigest()


def diagnostic_summaries(value_o, value_r) -> tuple[str, str]:
    """Readable summaries, disambiguated when shortening hides a difference."""
    summary_o = _symbolic_summary(value_o)
    summary_r = _symbolic_summary(value_r)
    if value_o != value_r and summary_o == summary_r:
        summary_o += f"#{_symbolic_fingerprint(value_o)}"
        summary_r += f"#{_symbolic_fingerprint(value_r)}"
    return summary_o, summary_r


def identity_facts(prefix: str, token) -> dict[str, str | int | bool | None]:
    """What a reference resolved to, from its proof identity (see
    asm.replacement.entity_proof_identity): its kind (``entity`` for a
    paired one, ``unresolved`` for an address nothing names, ...) and, for
    an entity, its original address and the offset into it."""
    match operand_identity(token):
        case ("entity", int() as address, int() as offset):
            return {
                f"{prefix}_kind": "entity",
                f"{prefix}_entity": address,
                f"{prefix}_offset": offset,
            }
        case (kind, *_):
            return {f"{prefix}_kind": str(kind)}
        case _:
            return {}


def _memory_facts(op) -> dict[str, str | int | bool | None]:
    """Primitive address components from one parsed memory operand."""
    if op[0] != "mem":
        return {
            "base_register": None,
            "index_register": None,
            "scale": 1,
            "displacement": 0,
            "symbol": None,
        }
    reg_terms = op[3]
    base = next((reg for reg, scale in reg_terms if scale == 1), None)
    index = next((reg for reg, scale in reg_terms if reg != base or scale != 1), None)
    index_scale = next((scale for reg, scale in reg_terms if reg == index), 1)
    symbols = op[5]
    symbol = None
    if symbols:
        symbol = " + ".join(
            ("-" if sign < 0 else "") + str(name) for sign, name in symbols
        )
    facts: dict[str, str | int | bool | None] = {
        "base_register": base,
        "index_register": index,
        "scale": index_scale,
        "displacement": op[4],
        "symbol": symbol,
    }
    if len(symbols) == 1 and symbols[0][0] == 1:
        facts.update(identity_facts("symbol", symbols[0][1]))
    return facts


def target_facts(
    ins: DecodedInstruction, target_index: int | None = None
) -> dict[str, str | int | bool | None]:
    """A control transfer's destination: its address, what it names, its
    row in the excerpt, and the identity it resolves to."""
    target_name = None
    match ins.operands:
        case (("imm", _), *_):
            pass
        case (operand, *_):
            target_name = format_operand(operand)
    facts: dict[str, str | int | bool | None] = {
        "target": ins.branch_target,
        "target_name": target_name,
        "target_instruction_index": target_index,
    }
    if ins.operands:
        operand = ins.operands[0]
        if operand[0] == "sym":
            facts.update(identity_facts("target", operand[1]))
            if isinstance(operand[1], Reference):
                facts["target_entity_type"] = operand[1].entity_type
        elif operand[0] == "mem":
            facts["target_indirect"] = True
            symbols = operand[5]
            if len(symbols) == 1 and symbols[0][0] == 1:
                reference = symbols[0][1]
                facts.update(identity_facts("target", reference))
                if isinstance(reference, Reference):
                    facts["target_entity_type"] = reference.entity_type
    return facts


def _target_index(
    recorder: AnalysisRecorder | None, which: str, ins: DecodedInstruction
) -> int | None:
    if recorder is None or ins.branch_target is None:
        return None
    addrs = recorder.orig_addrs if which == "orig" else recorder.recomp_addrs
    if addrs is None:
        return None
    try:
        return addrs.index(ins.branch_target)
    except ValueError:
        return None


def _checked_call_registers(ctx: Context, ins: DecodedInstruction) -> list[str]:
    facts = None
    if ctx.metadata is not None and ctx.metadata.call_facts is not None:
        if ins.operands and ins.operands[0][0] == "sym":
            facts = ctx.metadata.call_facts(operand_identity(ins.operands[0][1]))
    ecx_argument, edx_argument = register_arguments(facts)
    return [
        register
        for register, used in (("ecx", ecx_argument), ("edx", edx_argument))
        if used
    ]


def _operand_addresses(
    states: tuple[SideState, SideState] | None, op_o, op_r
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
        and ins.operands[0] == ("reg", "esp")
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
    if ins_o.mnemonic in JCC_MNEMONICS or ins_o.mnemonic.startswith("loop"):
        return
    if ins_o.mnemonic == "call":
        # CALL observations diagnose canonical direct/indirect targets and
        # register arguments after symbolic execution. A raw memory-operand
        # candidate here would re-expose the physical vtable register after a
        # virtual target had already been proved equivalent.
        return
    for op_o, op_r in zip(ins_o.operands, ins_r.operands):
        if op_o == op_r:
            continue
        if op_o[0] == op_r[0] == "mem":
            addresses = _operand_addresses(states, op_o, op_r)
            if addresses is not None and (
                addresses[0] == addresses[1]
                or bitvector.compare(addresses).result == "proved"
            ):
                # Registers renamed, or one address spelled two ways.
                continue
            facts_o, facts_r = _memory_facts(op_o), _memory_facts(op_r)
            if facts_o != facts_r:
                recorder.record_difference(
                    "memory_address",
                    index_o,
                    index_r,
                    facts_o,
                    facts_r,
                    candidate=True,
                    **_symbolic_values(addresses),
                )
                return
        if op_o[0] == op_r[0] == "imm":
            recorder.record_difference(
                "immediate_value",
                index_o,
                index_r,
                {"value": op_o[1]},
                {"value": op_r[1]},
                candidate=True,
            )
            return
        if op_o[0] == op_r[0] == "sym":
            recorder.record_difference(
                "symbol_resolution",
                index_o,
                index_r,
                {"symbol": str(op_o[1])},
                {"symbol": str(op_r[1])},
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
                "call_target",
                index_o,
                index_r,
                target_facts(ins_o),
                target_facts(ins_r),
            )
            return
        registers = _checked_call_registers(ctx, ins_o)
        for position, register in enumerate(registers, start=2):
            if position >= min(len(first_o), len(first_r)):
                break
            if first_o[position] != first_r[position]:
                value_o, value_r = diagnostic_summaries(
                    first_o[position], first_r[position]
                )
                recorder.record_difference(
                    "call_argument",
                    index_o,
                    index_r,
                    {
                        "register": register,
                        "value": value_o,
                    },
                    {
                        "register": register,
                        "value": value_r,
                    },
                )
                return
        # With the frame promoted, the arguments the callee reads from each
        # side's stack follow the call (see iso_cfg._pass_frame).
        passed_o = next((e for e in obs_o if e[0] == "frame_args"), None)
        passed_r = next((e for e in obs_r if e[0] == "frame_args"), None)
        if passed_o != passed_r:
            recorder.record_difference(
                "call_argument",
                index_o,
                index_r,
                {"register": "stack", "value": repr(passed_o)[:200]},
                {"register": "stack", "value": repr(passed_r)[:200]},
            )
            return

    if tag_o == tag_r == "store":
        if first_o[1] != first_r[1]:
            facts_o = _memory_facts(ins_o.operands[0]) if ins_o.operands else {}
            facts_r = _memory_facts(ins_r.operands[0]) if ins_r.operands else {}
            recorder.record_difference(
                "memory_address",
                index_o,
                index_r,
                facts_o,
                facts_r,
                **_symbolic_values(
                    (("addr", first_o[1]), ("addr", first_r[1]), 32, "value")
                ),
            )
            return
        if first_o[3] != first_r[3]:
            value_o, value_r = diagnostic_summaries(first_o[3], first_r[3])
            width = WIDTHS.get(first_o[2])
            recorder.record_difference(
                "memory_value",
                index_o,
                index_r,
                {"value": value_o},
                {"value": value_r},
                **_symbolic_values(
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
            value_o, value_r = diagnostic_summaries(predicate_o, predicate_r)
            recorder.record_difference(
                "branch_condition",
                index_o,
                index_r,
                {"predicate": value_o},
                {"predicate": value_r},
                **_symbolic_values(
                    (predicate_o, predicate_r, None, "predicate")
                    if predicate_o is not None and predicate_r is not None
                    else None
                ),
            )
            return
        target_o = _target_index(recorder, "orig", ins_o)
        target_r = _target_index(recorder, "recomp", ins_r)
        recorder.record_difference(
            "branch_target",
            index_o,
            index_r,
            target_facts(ins_o, target_o),
            target_facts(ins_r, target_r),
        )
        return

    for entry_o, entry_r in zip(obs_o, obs_r):
        if entry_o == entry_r:
            continue
        if entry_o and entry_r and entry_o[0] == entry_r[0]:
            if entry_o[0] in ("retval", "retfpu"):
                value_o, value_r = diagnostic_summaries(entry_o[1], entry_r[1])
                recorder.record_difference(
                    "return_value",
                    index_o,
                    index_r,
                    {"value": value_o},
                    {"value": value_r},
                    **_symbolic_values(
                        (entry_o[1], entry_r[1], None, "value")
                        if entry_o[0] == "retval" and len(entry_o) == len(entry_r) == 2
                        else None
                    ),
                )
                return
            if entry_o[0] in ("retsaved", "retstack"):
                value_o, value_r = diagnostic_summaries(entry_o, entry_r)
                recorder.record_difference(
                    "preserved_state",
                    index_o,
                    index_r,
                    {"value": value_o},
                    {"value": value_r},
                )
                return
    recorder.mark_inconclusive("analysis_limit")
