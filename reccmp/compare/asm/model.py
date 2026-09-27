"""The operand vocabulary: references, rendering operands and instructions
to Intel text, and parsing Intel text (a boundary for assembly that arrives
as text, never for text reccmp rendered).

Leaf module: no imports from ``ir`` or the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable


class Reject(Exception):
    """The two sequences could not be proven equivalent."""


@dataclass(frozen=True)
class Reference:
    """A relocated or absolute address after sanitization.

    ``display`` is the human/placeholder token (``<OFFSET1>``, a symbol).
    ``identity`` is what proofs compare: a named/paired entity, or a
    side-local unresolved address that cannot equal a placeholder from
    the other image merely by occupying the same replacement slot.
    """

    display: str
    identity: Hashable
    entity_type: str | None = None

    def __str__(self) -> str:
        return self.display


@dataclass(frozen=True)
class ResolvedAddress:
    """What an address resolves to, in one lookup: the entity's proof
    identity, and the name to show for it (None when it has no name, and
    the sanitizer shows a placeholder)."""

    name: str | None
    identity: Hashable
    entity_type: str | None = None


def operand_display(value) -> str:
    if isinstance(value, Reference):
        return value.display
    return str(value)


def operand_identity(value) -> Hashable:
    if isinstance(value, Reference):
        return value.identity
    return value


# Register families. Writing e.g. `al` produces a new value for the whole
# `a` family so that partial-register writes are never lost.
REGISTERS: dict[str, tuple[str, str]] = {
    **{f"e{r}x": (r, "r32") for r in "abcd"},
    **{f"{r}x": (r, "r16") for r in "abcd"},
    **{f"{r}l": (r, "l8") for r in "abcd"},
    **{f"{r}h": (r, "h8") for r in "abcd"},
    "esi": ("si", "r32"),
    "si": ("si", "r16"),
    "edi": ("di", "r32"),
    "di": ("di", "r16"),
    "ebp": ("bp", "r32"),
    "bp": ("bp", "r16"),
    "esp": ("sp", "r32"),
    "sp": ("sp", "r16"),
}


def format_imm(value: int) -> str:
    """Capstone Intel-syntax immediates: decimal for |n| < 10, else hex."""
    if -9 <= value <= 9:
        return str(value)
    return hex(value)


def format_operand(operand) -> str:
    # pylint: disable=too-many-return-statements
    """Render a structured operand back to Capstone-like Intel text."""
    kind = operand[0]
    if kind == "reg":
        return operand[1]
    if kind == "st":
        return f"st({operand[1]})"
    if kind == "imm":
        return format_imm(operand[1])
    if kind == "sym":
        return operand_display(operand[1])
    if kind == "opaque":
        # The canonical operand carries machine bytes; the original decoded
        # instruction retains Capstone text separately for display.
        return "?"
    if kind != "mem":
        raise Reject

    size, seg, reg_terms, disp, syms = (
        operand[1],
        operand[2],
        operand[3],
        operand[4],
        operand[5],
    )
    parts: list[str] = []
    for name, scale in reg_terms:
        token = name if scale == 1 else f"{name}*{scale}"
        if not parts:
            parts.append(token)
        else:
            parts.append(f"+ {token}")
    for sign, name in syms:
        if not parts:
            parts.append(
                operand_display(name) if sign > 0 else f"-{operand_display(name)}"
            )
        else:
            shown = operand_display(name)
            parts.append(f"+ {shown}" if sign > 0 else f"- {shown}")
    if disp or (not parts and not syms):
        if not parts:
            parts.append(format_imm(disp))
        elif disp > 0:
            parts.append(f"+ {format_imm(disp)}")
        elif disp < 0:
            parts.append(f"- {format_imm(-disp)}")

    body = " ".join(parts)
    if seg:
        body = f"{seg}:[{body}]"
    else:
        body = f"[{body}]"
    if size:
        return f"{size} ptr {body}"
    return body


def format_instruction(mnemonic: str, prefix: str, operands: tuple) -> str:
    """Build a display line from structured fields.

    Zero-operand instructions keep a trailing space (``\"nop \"``) for
    compatibility with the historical ``\" \".join((mnemonic, op_str))`` form.
    """
    head = f"{prefix} {mnemonic}".strip() if prefix else mnemonic
    if not operands:
        return f"{head} "
    return f"{head} {', '.join(format_operand(op) for op in operands)}"


def split_mnemonic_prefix(mnemonic: str) -> tuple[str, str]:
    """Split Capstone's combined ``rep movsd``-style mnemonic into prefix + op."""
    for candidate in ("repne", "repe", "rep"):
        if mnemonic == candidate:
            return candidate, ""
        prefix = candidate + " "
        if mnemonic.startswith(prefix):
            return candidate, mnemonic[len(prefix) :]
    return "", mnemonic
