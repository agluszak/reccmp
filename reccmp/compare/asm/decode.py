"""Capstone detail-mode decode into canonical ``DecodedInstruction`` rows.

One disassembly pass produces both the structured meta previously collected by
``collect_instruction_meta`` and the ``(addr, size, mnemonic, op_str)`` tuples
``InstructGen`` uses for section discovery.  Operand sanitization / structured
``parse_instruction`` happens later in ``ParseAsm``.
"""

from __future__ import annotations

from functools import cache
from typing import Iterable

from capstone import (  # type: ignore
    CS_ARCH_X86,
    CS_GRP_BRANCH_RELATIVE,
    CS_GRP_CALL,
    CS_GRP_JUMP,
    CS_GRP_RET,
    CS_MODE_16,
    CS_MODE_32,
    Cs,
    CsError,
)
from capstone import x86_const  # type: ignore

from .ir import AsmRole, DecodedInstruction

_EFLAGS_READ_MASK = 0
for _name in dir(x86_const):
    if _name.startswith("X86_EFLAGS_TEST_"):
        _EFLAGS_READ_MASK |= getattr(x86_const, _name)


@cache
def get_detail_disassembler(is_32: bool = True) -> Cs:
    disassembler = Cs(CS_ARCH_X86, CS_MODE_32 if is_32 else CS_MODE_16)
    disassembler.detail = True
    return disassembler


def stop_at_int3_detail(instructions) -> Iterable:
    for insn in instructions:
        if insn.mnemonic == "int3":
            break
        yield insn


def from_capstone(insn) -> DecodedInstruction:
    """Convert one Capstone detail instruction into canonical IR (unsanitized)."""
    register_access_known = True
    try:
        read_ids, write_ids = insn.regs_access()
        regs_read = tuple(sorted(insn.reg_name(r) for r in read_ids))
        regs_written = tuple(sorted(insn.reg_name(r) for r in write_ids))
    except CsError:
        # Unknown access must not be treated as "touches no registers".
        register_access_known = False
        regs_read = ()
        regs_written = ()

    is_jump = insn.group(CS_GRP_JUMP)
    is_call = insn.group(CS_GRP_CALL)
    branch_target = None
    operands = insn.operands
    if (
        (is_jump or is_call)
        and insn.group(CS_GRP_BRANCH_RELATIVE)
        and len(operands) == 1
        and operands[0].type == x86_const.X86_OP_IMM
    ):
        branch_target = operands[0].imm

    display = (
        f"{insn.mnemonic} {insn.op_str}".rstrip() if insn.op_str else insn.mnemonic
    )
    return DecodedInstruction(
        address=insn.address,
        size=insn.size,
        mnemonic=insn.mnemonic,
        prefix="",
        operands=(),
        raw_operands=(),
        display=display,
        role=AsmRole.CODE,
        regs_read=regs_read,
        regs_written=regs_written,
        reads_flags=bool(insn.eflags & _EFLAGS_READ_MASK),
        writes_flags=bool(insn.eflags & ~_EFLAGS_READ_MASK),
        accesses_memory=any(op.type == x86_const.X86_OP_MEM for op in operands),
        is_jump=is_jump,
        is_call=is_call,
        is_ret=insn.group(CS_GRP_RET),
        branch_target=branch_target,
        register_access_known=register_access_known,
        raw_op_str=insn.op_str,
    )


def disasm_detail(
    blob: bytes, start: int, is_32bit: bool = True
) -> list[DecodedInstruction]:
    """Disassemble ``blob`` once in detail mode, stopping at ``int3``."""
    disassembler = get_detail_disassembler(is_32bit)
    return [
        from_capstone(insn)
        for insn in stop_at_int3_detail(disassembler.disasm(blob, start))
    ]


def as_lite_tuple(insn: DecodedInstruction) -> tuple[int, int, str, str]:
    """Compatibility view for InstructGen section analysis."""
    assert insn.address is not None
    return (insn.address, insn.size, insn.mnemonic, insn.raw_op_str)
