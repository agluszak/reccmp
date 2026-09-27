"""The operands of a decoded instruction, as Capstone reports them and
with the sanitizer's references in place of addresses."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Reference, format_imm


@dataclass(frozen=True, slots=True)
class Reg:
    """A general, segment or other register, by Capstone's name."""

    name: str


@dataclass(frozen=True, slots=True)
class St:
    """An x87 stack register, ``st(index)``."""

    index: int


@dataclass(frozen=True, slots=True)
class Imm:
    value: int


@dataclass(frozen=True, slots=True)
class Sym:
    """An address operand the sanitizer resolved to a reference."""

    ref: Reference


@dataclass(frozen=True, slots=True, order=True)
class ScaledReg:
    """``register * scale`` inside a memory operand."""

    register: str
    scale: int


@dataclass(frozen=True, slots=True)
class SignedSymbol:
    """``+ reference`` or ``- reference`` inside a memory operand."""

    sign: int
    ref: Reference


@dataclass(frozen=True, slots=True)
class Mem:
    """``size ptr segment:[terms + symbols + displacement]``. ``size`` is
    empty for ``lea`` and ``size{N}`` for a width Capstone names nothing."""

    size: str
    segment: str
    terms: tuple[ScaledReg, ...]
    displacement: int
    symbols: tuple[SignedSymbol, ...] = ()


@dataclass(frozen=True, slots=True)
class Opaque:
    """An operand kind outside the model, distinguished by the instruction's
    bytes and the operand's position, never by Capstone's text."""

    kind: int
    raw: bytes
    index: int


Operand = Reg | St | Imm | Sym | Mem | Opaque


def format_operand(operand: Operand) -> str:
    """Render a structured operand to Capstone-like Intel text."""
    match operand:
        case Reg(name):
            return name
        case St(index):
            return f"st({index})"
        case Imm(value):
            return format_imm(value)
        case Sym(ref):
            return ref.display
        case Opaque():
            # The decoded instruction keeps Capstone's text for display.
            return "?"
        case Mem(size, segment, terms, displacement, symbols):
            parts: list[str] = []
            for term in terms:
                token = (
                    term.register
                    if term.scale == 1
                    else f"{term.register}*{term.scale}"
                )
                parts.append(f"+ {token}" if parts else token)
            for symbol in symbols:
                shown = symbol.ref.display
                if parts:
                    parts.append(f"+ {shown}" if symbol.sign > 0 else f"- {shown}")
                else:
                    parts.append(shown if symbol.sign > 0 else f"-{shown}")
            if displacement or not parts:
                if not parts:
                    parts.append(format_imm(displacement))
                elif displacement > 0:
                    parts.append(f"+ {format_imm(displacement)}")
                else:
                    parts.append(f"- {format_imm(-displacement)}")
            body = " ".join(parts)
            body = f"{segment}:[{body}]" if segment else f"[{body}]"
            return f"{size} ptr {body}" if size else body


def format_instruction(
    mnemonic: str, prefix: str, operands: tuple[Operand, ...]
) -> str:
    """Build a display line from structured fields.

    Zero-operand instructions keep a trailing space (``\"nop \"``) for
    compatibility with the historical ``\" \".join((mnemonic, op_str))`` form.
    """
    head = f"{prefix} {mnemonic}".strip() if prefix else mnemonic
    if not operands:
        return f"{head} "
    return f"{head} {', '.join(format_operand(op) for op in operands)}"
