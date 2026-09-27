"""Test boundary: assembly written as text, parsed once into the IR.

Production code never parses rendered text; tests that are clearest as
assembly lines build their rows here. Each line becomes a
``DecodedInstruction`` at ``start + index``; ``targets`` gives the index of
the row each jump reaches (None for none, or outside the rows).
"""

# Fixture wrappers preserve concise assembly tests while production APIs stay typed.
# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import re

from collections.abc import Sequence
from dataclasses import replace

from reccmp.compare.asm.const import JUMP_MNEMONICS
from reccmp.compare.asm.ir import (
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
)
from reccmp.compare.asm.model import REGISTERS, Reject

START = 0x1000

_ST_RE = re.compile(r"^st(?:\((\d)\))?$")
_MEM_RE = re.compile(
    r"^(?:(byte|word|dword|qword|tbyte|xword|xmmword) ptr )?"
    r"(?:(cs|ds|es|fs|gs|ss):)?\[(.+)\]$"
)
_SCALED_REG_RE = re.compile(r"^(e[a-d]x|e[sd]i|e[bs]p)\*([1248])$")
_NUM_RE = re.compile(r"^-?(?:0x[0-9a-f]+|\d+)$")


def _split_operands(op_str: str) -> list[str]:
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


def _parse_operand(text: str):
    if text in REGISTERS:
        return ("reg", text)

    st_match = _ST_RE.match(text)
    if st_match:
        return ("st", int(st_match.group(1) or 0))

    if _NUM_RE.match(text):
        return ("imm", int(text, 0))

    mem_match = _MEM_RE.match(text)
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
            elif (scaled := _SCALED_REG_RE.match(token)) is not None:
                if sign < 0:
                    raise Reject
                reg_terms.append((scaled.group(1), int(scaled.group(2))))
            elif _NUM_RE.match(token):
                disp += sign * int(token, 0)
            else:
                syms.append((sign, token))
        return ("mem", size or "", seg or "", reg_terms, disp, tuple(sorted(syms)))

    # Symbol, placeholder, or anything else we treat as an opaque token.
    return ("sym", text)


def parse_instruction(line: str) -> tuple[str, str, tuple]:
    """``(prefix, mnemonic, operands)`` of one line of Intel assembly text.

    Only test fixtures arrive as text; reccmp decodes machine code."""
    mnemonic, _, op_str = line.partition(" ")
    prefix = ""
    if mnemonic in ("rep", "repe", "repne"):
        prefix = mnemonic
        mnemonic, _, op_str = op_str.partition(" ")
    raw = tuple(_split_operands(op_str)) if op_str else ()
    return prefix, mnemonic, tuple(_parse_operand(token) for token in raw)


def rows(
    lines: Sequence[str],
    targets: Sequence[int | None] | None = None,
    *,
    start: int = START,
) -> tuple[DecodedInstruction, ...]:
    """The rows of ``lines``; jumps reach ``targets`` (row indices)."""
    result: list[DecodedInstruction] = []
    for index, line in enumerate(lines):
        address = start + index
        prefix, mnemonic, operands = parse_instruction(line)
        target = targets[index] if targets is not None else None
        is_jump = mnemonic in JUMP_MNEMONICS
        control_target = None
        if is_jump or mnemonic == "call":
            control_target = (
                ("local_insn", target)
                if target is not None
                else ("fixture", line.partition(" ")[2])
            )
        result.append(
            DecodedInstruction(
                address=address,
                size=1,
                mnemonic=mnemonic,
                prefix=prefix,
                operands=operands,
                display=line,
                is_jump=is_jump,
                is_call=mnemonic == "call",
                is_ret=mnemonic == "ret",
                branch_target=start + target if target is not None else None,
                # Text says nothing about implicit register effects.
                register_access_known=False,
                instruction_id=index,
                control_target=control_target,
            )
        )
    return tuple(result)


def image(
    lines: Sequence[str],
    targets: Sequence[int | None] | None = None,
    *,
    start: int = START,
    jump_tables: Sequence[JumpTable] = (),
    coverage_incomplete: bool = False,
    extent_closed: bool = True,
) -> FunctionImage:
    """A function image of ``lines`` (see ``rows``)."""
    return FunctionImage(
        start_addr=start,
        extent=len(lines),
        extent_kind=ExtentKind.KNOWN,
        instructions=rows(lines, targets, start=start),
        jump_tables=tuple(jump_tables),
        coverage_incomplete=coverage_incomplete,
        extent_closed=extent_closed,
    )


def fingerprint(lines: Sequence[str]):
    """A helper fingerprint from source assembly fixtures."""
    from reccmp.compare.inlines import fingerprint_of

    return fingerprint_of(rows(lines))


def image_from_bytes(code: bytes, start: int = START) -> FunctionImage:
    """Decode a real byte fixture once, including its local target topology."""
    from reccmp.compare.asm.parse import decode_function

    return decode_function(code, start)


def as_rows(
    value,
    *,
    targets: Sequence[int | None] | None = None,
    addresses: Sequence[int] | None = None,
    meta: Sequence[object | None] | None = None,
    start: int = START,
) -> tuple[DecodedInstruction, ...]:
    """Decode literal assembly at the test boundary, then attach test facts."""
    result = (
        tuple(value)
        if value and isinstance(value[0], DecodedInstruction)
        else rows(value, targets, start=start)
    )
    updated = []
    for index, row in enumerate(result):
        changes: dict[str, object] = {}
        if addresses is not None:
            changes["address"] = addresses[index]
            if (
                targets is None
                and row.is_jump
                and row.operands
                and row.operands[0][0] == "imm"
                and isinstance(row.operands[0][1], int)
            ):
                next_addr = (
                    addresses[index + 1]
                    if index + 1 < len(addresses)
                    else addresses[index] + row.size
                )
                destination = next_addr + row.operands[0][1]
                if destination in addresses:
                    target_index = addresses.index(destination)
                    changes["branch_target"] = destination
                    changes["control_target"] = ("local_insn", target_index)
        target = targets[index] if targets is not None else None
        if target is not None:
            changes["branch_target"] = (
                addresses[target]
                if addresses is not None and 0 <= target < len(addresses)
                else start + target
            )
            changes["control_target"] = ("local_insn", target)
        if meta is not None and meta[index] is not None:
            facts = meta[index]
            for name in (
                "regs_read",
                "regs_written",
                "reads_flags",
                "writes_flags",
                "accesses_memory",
                "is_jump",
                "is_call",
                "is_ret",
                "register_access_known",
                "operand_model_complete",
                "control_flow_known",
                "control_target",
                "branch_target",
            ):
                if hasattr(facts, name):
                    changes[name] = getattr(facts, name)
        updated.append(replace(row, **changes) if changes else row)
    if addresses is not None:
        for index, row in enumerate(updated[:-1]):
            following = updated[index + 1]
            if (
                row.address is not None
                and following.address is not None
                and following.address > row.address
            ):
                updated[index] = replace(row, size=following.address - row.address)
    return tuple(updated)


def verify_effective_match(
    orig,
    recomp,
    codes=None,
    metadata=None,
    recorder=None,
    *,
    orig_addrs=None,
    recomp_addrs=None,
    orig_meta=None,
    recomp_meta=None,
):
    """Call the production verifier with fixture rows."""
    from reccmp.compare.asm.verifier import verify_effective_match as verify

    return verify(
        as_rows(orig, addresses=orig_addrs, meta=orig_meta),
        as_rows(recomp, addresses=recomp_addrs, meta=recomp_meta, start=0x2000),
        codes,
        metadata,
        recorder,
    )


def verify_isomorphic_cfg_effective_match(
    orig,
    recomp,
    orig_targets=(),
    recomp_targets=(),
    metadata=None,
    recorder=None,
    *,
    orig_addrs=None,
    recomp_addrs=None,
    orig_tables=(),
    recomp_tables=(),
    unanchored=None,
):
    from reccmp.compare.asm.verifier import (
        verify_isomorphic_cfg_effective_match as verify,
    )

    if recorder is not None:
        orig_addrs = orig_addrs if orig_addrs is not None else recorder.orig_addrs
        recomp_addrs = (
            recomp_addrs if recomp_addrs is not None else recorder.recomp_addrs
        )

    def fixture_image(lines, targets, addresses, start, tables):
        fixture_rows = as_rows(
            lines, targets=targets or None, addresses=addresses, start=start
        )
        extent = max(
            (
                row.address + row.size - start
                for row in fixture_rows
                if row.address is not None
            ),
            default=0,
        )
        extent = max(
            [extent]
            + [entry + 4 - start for table in tables for entry, _ in table.entries]
        )
        return FunctionImage(
            start, extent, ExtentKind.KNOWN, fixture_rows, tuple(tables)
        )

    return verify(
        fixture_image(orig, orig_targets, orig_addrs, START, orig_tables),
        fixture_image(recomp, recomp_targets, recomp_addrs, 0x2000, recomp_tables),
        metadata,
        recorder,
        unanchored,
    )


def analyze_effective_match(
    codes,
    orig,
    recomp,
    metadata=None,
    *,
    orig_addrs=None,
    recomp_addrs=None,
    orig_meta=None,
    recomp_meta=None,
    orig_jump_tables=(),
    recomp_jump_tables=(),
    coverage_incomplete=False,
    extent_closed=True,
):
    from reccmp.compare.asm.verifier import analyze_effective_match as analyze
    from reccmp.compare.asm.verifier import compare_exact

    orig_rows = as_rows(orig, addresses=orig_addrs, meta=orig_meta)
    recomp_rows = as_rows(
        recomp, addresses=recomp_addrs, meta=recomp_meta, start=0x2000
    )
    orig_image = FunctionImage(
        START,
        len(orig_rows),
        ExtentKind.KNOWN,
        orig_rows,
        tuple(orig_jump_tables),
        coverage_incomplete,
        extent_closed,
    )
    recomp_image = FunctionImage(
        0x2000,
        len(recomp_rows),
        ExtentKind.KNOWN,
        recomp_rows,
        tuple(recomp_jump_tables),
        coverage_incomplete,
        extent_closed,
    )
    return compare_exact(orig_image, recomp_image) or analyze(
        codes, orig_image, recomp_image, metadata
    )
