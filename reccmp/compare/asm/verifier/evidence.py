"""Readable summaries of symbolic values and recorded difference facts."""

from __future__ import annotations

import hashlib
import re

from reccmp.compare.asm.instgen import InstructionMeta
from reccmp.compare.asm.model import (
    Instruction,
    operand_display,
)
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    JCC_MNEMONICS,
    WIDTHS,
    Context,
    register_arguments,
)
from reccmp.compare.diagnosis import AnalysisRecorder


def _clean_symbol(text: str) -> str:
    """Sanitized symbol without the display-only entity annotation."""
    return re.sub(r"\s+\((?:DATA|STRING|FLOAT|FUNCTION|IMPORT)\)$", "", text)


def _symbolic_summary(value, depth: int = 0) -> str:
    # pylint: disable=too-many-return-statements
    """Small stable rendering for diagnosis; never expands the whole DAG."""
    if not isinstance(value, tuple) or not value:
        return str(value)
    tag = str(value[0])
    if tag == "imm" and len(value) > 1:
        return str(value[1])
    if tag == "sym" and len(value) > 1:
        return _clean_symbol(str(value[1]))
    if tag == "init" and len(value) > 1:
        return f"initial:{value[1]}"
    if tag == "load" and len(value) > 1:
        return f"load:{_symbolic_summary(value[1], depth + 1)}"
    if tag == "mem" and len(value) >= 5:
        symbols = value[4]
        if symbols:
            return _clean_symbol(str(symbols[0][1]))
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
            ("-" if sign < 0 else "") + _clean_symbol(str(name))
            for sign, name in symbols
        )
    return {
        "base_register": base,
        "index_register": index,
        "scale": index_scale,
        "displacement": op[4],
        "symbol": symbol,
    }


def target_facts(
    ins: Instruction, meta: InstructionMeta | None, target_index: int | None = None
) -> dict[str, str | int | bool | None]:
    target_name = None
    if ins.raw_operands:
        raw = ins.raw_operands[0]
        if not raw.startswith(("0x", "-0x")):
            target_name = _clean_symbol(raw)
    return {
        "target": meta.branch_target if meta is not None else None,
        "target_name": target_name,
        "target_instruction_index": target_index,
    }


def _target_index(
    recorder: AnalysisRecorder | None,
    which: str,
    meta: InstructionMeta | None,
) -> int | None:
    if recorder is None or meta is None or meta.branch_target is None:
        return None
    addrs = recorder.orig_addrs if which == "orig" else recorder.recomp_addrs
    if addrs is None:
        return None
    try:
        return addrs.index(meta.branch_target)
    except ValueError:
        return None


def _checked_call_registers(ctx: Context, ins: Instruction) -> list[str]:
    facts = None
    if ctx.metadata is not None and ctx.metadata.call_facts is not None:
        if ins.operands and ins.operands[0][0] == "sym":
            facts = ctx.metadata.call_facts(operand_display(ins.operands[0][1]))
    ecx_argument, edx_argument = register_arguments(facts)
    return [
        register
        for register, used in (("ecx", ecx_argument), ("edx", edx_argument))
        if used
    ]


def record_operand_candidate(
    ctx: Context,
    index_o: int,
    index_r: int,
    ins_o: Instruction,
    ins_r: Instruction,
) -> None:
    recorder = ctx.recorder
    if recorder is None or ins_o.mnemonic != ins_r.mnemonic:
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
            facts_o, facts_r = _memory_facts(op_o), _memory_facts(op_r)
            if facts_o != facts_r:
                recorder.record_difference(
                    "memory_address",
                    index_o,
                    index_r,
                    facts_o,
                    facts_r,
                    candidate=True,
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
                {"symbol": _clean_symbol(str(op_o[1]))},
                {"symbol": _clean_symbol(str(op_r[1]))},
                candidate=True,
            )
            return


def record_observable_difference(
    ctx: Context,
    index_o: int,
    index_r: int,
    ins_o: Instruction,
    ins_r: Instruction,
    obs_o: list,
    obs_r: list,
    meta_o: InstructionMeta | None,
    meta_r: InstructionMeta | None,
) -> None:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-return-statements
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
                target_facts(ins_o, meta_o),
                target_facts(ins_r, meta_r),
            )
            return
        registers = _checked_call_registers(ctx, ins_o)
        for position, register in enumerate(registers, start=2):
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

    if tag_o == tag_r == "store":
        if first_o[1] != first_r[1]:
            facts_o = _memory_facts(ins_o.operands[0]) if ins_o.operands else {}
            facts_r = _memory_facts(ins_r.operands[0]) if ins_r.operands else {}
            recorder.record_difference(
                "memory_address", index_o, index_r, facts_o, facts_r
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
                values=(
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
                values=(
                    (predicate_o, predicate_r, None, "predicate")
                    if predicate_o is not None and predicate_r is not None
                    else None
                ),
            )
            return
        target_o = _target_index(recorder, "orig", meta_o)
        target_r = _target_index(recorder, "recomp", meta_r)
        recorder.record_difference(
            "branch_target",
            index_o,
            index_r,
            target_facts(ins_o, meta_o, target_o),
            target_facts(ins_r, meta_r, target_r),
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
                    values=(
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
