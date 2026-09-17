"""Structured instruction keys used for alignment and normalized ratios.

Display strings remain the user-facing representation. Matching and
stack-normalized scoring operate on Instruction IR (or opaque line keys when
a line is not an instruction, e.g. jump/data tables).
"""

from __future__ import annotations

from typing import Hashable

from .effective import Instruction, Reject, parse_instruction

_MEM = "mem"

# Sentinel displacing frame/stack offsets for modulo-stack comparison keys.
_STACK_SLOT = ("stack_slot",)


def try_parse_instruction(line: str) -> Instruction | None:
    """Parse a sanitized asm line into IR, or None for non-instruction lines."""
    try:
        return parse_instruction(line)
    except Reject:
        return None


def _freeze(value) -> Hashable:
    """Make parse_operand structures hashable (reg_terms are lists today)."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def instruction_match_key(line: str) -> Hashable:
    """Hashable key for SequenceMatcher alignment.

    Prefer structured Instruction equality so operand structure drives matching.
    Fall back to the raw display string for jump/data table lines.
    """
    ins = try_parse_instruction(line)
    if ins is None:
        return ("raw", line)
    return ("ins", ins.mnemonic, ins.prefix, _freeze(ins.operands))


def _normalize_operand_stack(operand) -> object:
    """Erase ebp/esp displacements so stack-layout entropy collapses."""
    if not isinstance(operand, tuple) or not operand:
        return operand
    tag = operand[0]
    if tag == _MEM:
        # ("mem", size, seg, reg_terms, disp, syms)
        size, seg, reg_terms, _disp, syms = (
            operand[1],
            operand[2],
            operand[3],
            operand[4],
            operand[5],
        )
        regs = {name for name, _scale in reg_terms}
        if regs & {"ebp", "esp"} and not syms:
            return (_MEM, size, seg, _freeze(reg_terms), _STACK_SLOT, ())
        return _freeze(operand)
    return _freeze(operand)


def stack_normalized_key(line: str) -> Hashable:
    """Match key with frame/stack displacements abstracted away."""
    ins = try_parse_instruction(line)
    if ins is None:
        return ("raw", line)
    operands = tuple(_normalize_operand_stack(op) for op in ins.operands)
    return ("ins", ins.mnemonic, ins.prefix, operands)


def rewrite_stack_displacements(
    line: str, mapping: dict[tuple[str, int], tuple[str, int]]
) -> str:
    """Rewrite ebp/esp ± offset tokens in a display line through a slot map.

    ``mapping`` keys and values are ``(register, signed_offset)``.
    Unmapped offsets are left unchanged.
    """
    from reccmp.compare.stack_layout import STACK_ENTRY_REGEX

    def repl(match) -> str:
        register = match.group("register")
        offset = int(match.group("sign") + match.group("offset"), 16)
        target = mapping.get((register, offset))
        if target is None:
            return match.group(0)
        tgt_reg, tgt_off = target
        if tgt_off >= 0:
            return f"{tgt_reg} + {tgt_off:#x}"
        return f"{tgt_reg} - {-tgt_off:#x}"

    return STACK_ENTRY_REGEX.sub(repl, line)
