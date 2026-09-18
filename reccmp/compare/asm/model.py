"""Instruction/operand text model for sanitized assembly lines.

Leaf module: no imports from ``effective``, ``ir``, or ``stack_layout``.
Parsing lives here so IR and the symbolic executor can share it without cycles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class Reject(Exception):
    """The two sequences could not be proven equivalent."""


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

MEM_RE = re.compile(
    r"^(?:(byte|word|dword|qword|tbyte|xword|xmmword) ptr )?"
    r"(?:(cs|ds|es|fs|gs|ss):)?\[(.+)\]$"
)
SCALED_REG_RE = re.compile(r"^(e[a-d]x|e[sd]i|e[bs]p)\*([1248])$")
NUM_RE = re.compile(r"^-?(?:0x[0-9a-f]+|\d+)$")
ST_RE = re.compile(r"^st(?:\((\d)\))?$")

# ebp/esp ± offset tokens in display lines (shared by IR rewrite and stack_layout).
STACK_ENTRY_REGEX = re.compile(
    r"(?P<register>e[sb]p)\s(?P<sign>[+-])\s(?P<offset>(0x)?[0-9a-f]+)(?![0-9a-f])"
)


def split_operands(op_str: str) -> list[str]:
    """Split on top-level ', ' only: brackets and parens may contain commas."""
    operands = []
    depth = 0
    start = 0
    i = 0
    while i < len(op_str):
        char = op_str[i]
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif depth == 0 and op_str.startswith(", ", i):
            operands.append(op_str[start:i])
            start = i + 2
            i += 2
            continue
        i += 1
    operands.append(op_str[start:])
    return [op for op in (o.strip() for o in operands) if op]


def parse_operand(text: str):
    if text in REGISTERS:
        return ("reg", text)

    st_match = ST_RE.match(text)
    if st_match:
        return ("st", int(st_match.group(1) or 0))

    if NUM_RE.match(text):
        return ("imm", int(text, 0))

    mem_match = MEM_RE.match(text)
    if mem_match:
        size, seg, content = mem_match.groups()
        reg_terms: list[tuple[str, int]] = []
        disp = 0
        syms: list[tuple[int, str]] = []
        tokens = re.split(r" ([+-]) ", content)
        sign = 1
        for k, token in enumerate(tokens):
            if k % 2 == 1:
                sign = 1 if token == "+" else -1
                continue
            token = token.strip()
            if token in REGISTERS:
                if sign < 0:
                    raise Reject
                reg_terms.append((token, 1))
            elif (scaled := SCALED_REG_RE.match(token)) is not None:
                if sign < 0:
                    raise Reject
                reg_terms.append((scaled.group(1), int(scaled.group(2))))
            elif NUM_RE.match(token):
                disp += sign * int(token, 0)
            else:
                syms.append((sign, token))
        return ("mem", size or "", seg or "", reg_terms, disp, tuple(sorted(syms)))

    # Symbol, placeholder, or anything else we treat as an opaque token.
    return ("sym", text)


@dataclass(frozen=True)
class Instruction:
    mnemonic: str
    prefix: str  # rep/repe/repne or ""
    operands: tuple
    raw_operands: tuple[str, ...]


def parse_instruction(line: str) -> Instruction:
    mnemonic, _, op_str = line.partition(" ")
    prefix = ""
    if mnemonic in ("rep", "repe", "repne"):
        prefix = mnemonic
        mnemonic, _, op_str = op_str.partition(" ")
    raw = tuple(split_operands(op_str)) if op_str else ()
    return Instruction(mnemonic, prefix, tuple(parse_operand(t) for t in raw), raw)


def format_imm(value: int) -> str:
    """Capstone Intel-syntax immediates: decimal for |n| < 10, else hex."""
    if -9 <= value <= 9:
        return str(value)
    return hex(value)


def format_operand(operand) -> str:
    """Render a structured operand back to Capstone-like Intel text."""
    kind = operand[0]
    if kind == "reg":
        return operand[1]
    if kind == "st":
        return f"st({operand[1]})"
    if kind == "imm":
        return format_imm(operand[1])
    if kind == "sym":
        return operand[1]
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
            parts.append(name if sign > 0 else f"-{name}")
        else:
            parts.append(f"+ {name}" if sign > 0 else f"- {name}")
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
