"""A C-like spelling of the verifier's symbolic values, for people."""

from __future__ import annotations

from typing import Any

from reccmp.compare.asm.model import FAMILY_REGISTER
from reccmp.compare.asm.verifier.addresses import (
    CallResult,
    Init,
    Phi,
    Resync,
    StringResult,
)

_BINARY = {
    "and": "&",
    "or": "|",
    "xor": "^",
    "sub": "-",
    "imul": "*",
    "imul3": "*",
    "shl": "<<",
    "shr": ">>u",
    "sar": ">>s",
}
_PREDICATE = {
    "eq": "==",
    "ne": "!=",
    "lt_u": "<u",
    "le_u": "<=u",
    "lt_s": "<s",
    "le_s": "<=s",
}
_UNARY = {"inc": "{} + 1", "dec": "{} - 1", "neg": "-{}", "not": "~{}"}
_PART = {"l8": "low8", "h8": "bits8_15", "r16": "low16"}
_INSERT = {"ins_l8": "al", "ins_h8": "ah", "ins_r16": "ax"}


def render_number(value: int) -> str:
    return str(value) if -10 < value < 10 else hex(value)


_ENTRY_SP = (Init("sp"), 1)


def render(value: Any, depth: int = 0) -> str:
    """A C-like spelling of a verifier symbolic value."""
    # pylint: disable=too-many-return-statements,too-many-branches
    if depth > 8:
        return "…"

    def inner(item: Any) -> str:
        return render(item, depth + 1)

    match value:
        case ("imm", int() as constant):
            return render_number(constant)
        case Init(family):
            return f"{FAMILY_REGISTER.get(family, family)}@entry"
        case ("load", ("mem", "", (entry,), int() as displacement, ()), "dword", _) if (
            entry == _ENTRY_SP and displacement > 0 and displacement % 4 == 0
        ):
            return f"arg{displacement // 4}"
        case ("load", address, size, *_):
            return f"{size}[{_address(address, depth + 1)}]"
        case ("mem", *_):
            return _address(value, depth)
        case ("addr", address):
            return f"&[{_address(address, depth + 1)}]"
        case ("sym", identity):
            return _symbol(identity)
        case ("spadd", base, int() as offset):
            return f"{inner(base)} {'+' if offset >= 0 else '-'} {render_number(abs(offset))}"
        case (tag, whole) if tag in _PART:
            return f"{_PART[tag]}({inner(whole)})"
        case (tag, old, new) if tag in _INSERT:
            return f"({inner(old)} with {_INSERT[tag]} = {inner(new)})"
        case ("add", *terms) if len(terms) >= 2:
            return "(" + " + ".join(inner(term) for term in terms) + ")"
        case (tag, left, right) if tag in _BINARY:
            return f"({inner(left)} {_BINARY[tag]} {inner(right)})"
        case (tag, operand) if tag in _UNARY:
            return "(" + _UNARY[tag].format(inner(operand)) + ")"
        case ("movzx", _, operand):
            return f"zext({inner(operand)})"
        case ("movsx", _, operand):
            return f"sext({inner(operand)})"
        case CallResult(call, family):
            return f"{FAMILY_REGISTER.get(family, family)} after call@{call}"
        case StringResult(site, family):
            return f"{FAMILY_REGISTER.get(family, family)} after string@{site}"
        case Resync(site, location):
            return f"{location} after resync@{site}"
        case Phi(block, class_id):
            return f"join{block}#{class_id}"
        case ("eq" | "ne" as tag, (left, right), *width):
            return _comparison(tag, left, right, width, depth)
        case (tag, left, right, *width) if tag in _PREDICATE:
            return _comparison(tag, left, right, width, depth)
        case ("cc", condition, flags, *_):
            return f"{condition} of {inner(flags)}"
        case (tag, *operands):
            return f"{tag}(" + ", ".join(inner(item) for item in operands[:2]) + ")"
        case _:
            return repr(value)


def _comparison(tag: str, left: Any, right: Any, width: list, depth: int) -> str:
    match width:
        case [int() as size]:
            bits = f" ({8 * size}-bit)"
        case _:
            bits = ""
    return (
        f"{render(left, depth + 1)} {_PREDICATE[tag]} {render(right, depth + 1)}{bits}"
    )


def _symbol(identity: Any) -> str:
    match identity:
        case ("entity", int() as address, int() as offset):
            return f"entity@0x{address:x}" + (
                f"+{render_number(offset)}" if offset else ""
            )
        case ("import", name):
            return str(name)
        case (*parts,) if parts:
            return "/".join(str(part) for part in parts)
        case _:
            return str(identity)


def _address(mem: Any, depth: int) -> str:
    match mem:
        case ("mem", segment, terms, displacement, symbols):
            pass
        case _:
            return render(mem, depth)
    parts = [
        render(term, depth + 1) if scale == 1 else f"{render(term, depth + 1)}*{scale}"
        for term, scale in terms
    ]
    parts += [term.ref.display for term in symbols]
    match displacement:
        case 0 if parts:
            pass
        case int() as constant:
            parts.append(render_number(constant))
        case other:
            parts.append(str(other))
    text = " + ".join(parts).replace("+ -", "- ")
    return f"{segment}:{text}" if segment else text
