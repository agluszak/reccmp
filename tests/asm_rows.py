"""Test boundary: assembly written as text, parsed once into the IR.

Production code never parses rendered text; tests that are clearest as
assembly lines build their rows here. Each line becomes a
``DecodedInstruction`` at ``start + index``; ``targets`` gives the index of
the row each jump reaches (None for none, or outside the rows).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from reccmp.compare.asm.const import JUMP_MNEMONICS
from reccmp.compare.asm.ir import (
    AsmRole,
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
)
from reccmp.compare.asm.model import parse_instruction

START = 0x1000


def _marker(line: str, address: int) -> DecodedInstruction | None:
    """A jump/data table line, as ParseAsm renders them."""
    role: AsmRole | None = None
    payload: tuple = ()
    if line == "Jump table:":
        role = AsmRole.JUMP_TABLE_HEADER
    elif line == "Data table:":
        role = AsmRole.DATA_TABLE_HEADER
    elif line.startswith("start + "):
        role = AsmRole.JUMP_TABLE_ENTRY
        payload = (("case", int(line[len("start + ") :], 16)),)
    elif line.startswith("0x") and " " not in line:
        role = AsmRole.DATA_TABLE_ENTRY
        payload = (("byte", int(line, 16)),)
    if role is None:
        return None
    return DecodedInstruction(
        address=address,
        size=0,
        mnemonic="",
        prefix="",
        operands=payload,
        display=line,
        role=role,
    )


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
        marker = _marker(line, address)
        if marker is not None:
            result.append(marker)
            continue
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
        excerpt=rows(lines, targets, start=start),
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
    from reccmp.compare.asm.parse import ParseAsm
    from reccmp.compare.asm.ir import compute_extent_closed, rebind_local_identities

    parser = ParseAsm()
    excerpt = tuple(
        replace(row, instruction_id=index)
        for index, row in enumerate(parser.parse_asm(code, start))
    )
    tables = tuple(parser.jump_tables)
    excerpt = rebind_local_identities(
        excerpt, start_addr=start, extent=len(code), jump_tables=tables
    )
    return FunctionImage(
        start,
        len(code),
        ExtentKind.KNOWN,
        excerpt,
        tables,
        parser.coverage_incomplete,
        compute_extent_closed(
            excerpt,
            start_addr=start,
            extent=len(code),
            coverage_incomplete=parser.coverage_incomplete,
            jump_tables=tables,
            extent_kind=ExtentKind.KNOWN,
        ),
        code,
    )


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
        changes = {}
        if addresses is not None:
            changes["address"] = addresses[index]
            if (
                targets is None
                and row.is_jump
                and row.operands
                and row.operands[0][0] == "imm"
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
        if targets is not None and targets[index] is not None:
            target = targets[index]
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


def verify_cfg_effective_match(
    orig, recomp, orig_targets=(), recomp_targets=(), metadata=None, recorder=None
):
    from reccmp.compare.asm.verifier import verify_cfg_effective_match as verify

    return verify(
        as_rows(
            orig,
            targets=orig_targets or None,
            addresses=recorder.orig_addrs if recorder is not None else None,
        ),
        as_rows(
            recomp,
            targets=recomp_targets or None,
            start=0x2000,
            addresses=recorder.recomp_addrs if recorder is not None else None,
        ),
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
):
    from reccmp.compare.asm.verifier import (
        verify_isomorphic_cfg_effective_match as verify,
    )

    if recorder is not None:
        orig_addrs = orig_addrs if orig_addrs is not None else recorder.orig_addrs
        recomp_addrs = (
            recomp_addrs if recomp_addrs is not None else recorder.recomp_addrs
        )

    return verify(
        as_rows(orig, targets=orig_targets or None, addresses=orig_addrs),
        as_rows(
            recomp, targets=recomp_targets or None, addresses=recomp_addrs, start=0x2000
        ),
        (),
        (),
        metadata,
        recorder,
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
    return analyze(codes, orig_image, recomp_image, metadata)
