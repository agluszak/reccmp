"""Capstone detail-mode decode into canonical ``DecodedInstruction`` rows.

One disassembly pass produces both the structured meta previously collected by
``collect_instruction_meta`` and the ``(addr, size, mnemonic, op_str)`` tuples
``InstructGen`` uses for section discovery.  Typed operands are filled from
Capstone detail here; address sanitization happens later in ``ParseAsm``.
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
from .model import (
    REGISTERS,
    ST_RE,
    format_operand,
    split_mnemonic_prefix,
)

_EFLAGS_READ_MASK = 0
for _name in dir(x86_const):
    if _name.startswith("X86_EFLAGS_TEST_"):
        _EFLAGS_READ_MASK |= getattr(x86_const, _name)

_SIZE_NAMES = {
    1: "byte",
    2: "word",
    4: "dword",
    8: "qword",
    10: "tbyte",
    16: "xmmword",
    32: "ymmword",
    64: "zmmword",
}


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


def _reg_operand(name: str):
    """Map a Capstone register name to the parse_operand tuple shape."""
    if name in REGISTERS:
        return ("reg", name)
    st_match = ST_RE.match(name)
    if st_match:
        return ("st", int(st_match.group(1) or 0))
    return ("sym", name)


def _mem_size_name(mnemonic: str, size: int) -> tuple[str, bool]:
    """Return ``(size_token, size_known)`` for a memory operand.

    Capstone omits the size keyword for lea; match that in structured form.
    Unknown sizes stay distinct (``size{N}``) so they cannot collapse together.
    """
    if mnemonic == "lea":
        return "", True
    known = _SIZE_NAMES.get(size)
    if known is not None:
        return known, True
    return f"size{size}", False


def capstone_operand(
    insn, op, mnemonic: str, operand_index: int
) -> tuple[object, bool]:
    """Convert one Capstone operand to the ``parse_operand`` tuple shape.

    Returns ``(operand, model_complete)``. Unsupported kinds become unique
    opaque tuples so distinct unknowns cannot share match keys.
    """
    if op.type == x86_const.X86_OP_REG:
        return _reg_operand(insn.reg_name(op.reg)), True
    if op.type == x86_const.X86_OP_IMM:
        return ("imm", int(op.imm)), True
    if op.type == x86_const.X86_OP_MEM:
        mem = op.mem
        reg_terms: list[tuple[str, int]] = []
        if mem.base:
            reg_terms.append((insn.reg_name(mem.base), 1))
        if mem.index:
            reg_terms.append((insn.reg_name(mem.index), int(mem.scale)))
        seg = insn.reg_name(mem.segment) if mem.segment else ""
        size_name, size_known = _mem_size_name(mnemonic, op.size)
        return (
            (
                "mem",
                size_name,
                seg or "",
                reg_terms,
                int(mem.disp),
                (),
            ),
            size_known,
        )
    # Rare / unsupported (e.g. invalid): keep a unique opaque identity.
    return ("opaque", op.type, str(insn.op_str), operand_index), False


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
    cs_operands = insn.operands
    if (
        (is_jump or is_call)
        and insn.group(CS_GRP_BRANCH_RELATIVE)
        and len(cs_operands) == 1
        and cs_operands[0].type == x86_const.X86_OP_IMM
    ):
        branch_target = cs_operands[0].imm

    prefix, mnemonic = split_mnemonic_prefix(insn.mnemonic)
    # Use the post-split mnemonic for lea size omission etc.
    decoded_ops = [
        capstone_operand(insn, op, mnemonic, index)
        for index, op in enumerate(cs_operands)
    ]
    operands = tuple(operand for operand, _complete in decoded_ops)
    operand_model_complete = all(complete for _operand, complete in decoded_ops)
    # Opaque operands mean jump/call targets are not fully modeled.
    control_flow_known = operand_model_complete and all(
        not (isinstance(operand, tuple) and operand and operand[0] == "opaque")
        for operand in operands
    )
    # Indirect control transfers have no absolute branch_target at decode time.
    # CFG may still recover switch tables via JumpTable metadata later.
    if control_flow_known and (is_jump or is_call) and branch_target is None:
        if is_jump:
            control_flow_known = False
        elif not any(
            isinstance(op, tuple) and op and op[0] in ("imm", "sym", "reg")
            for op in operands
        ):
            control_flow_known = False
        elif any(
            isinstance(op, tuple) and op and op[0] == "mem" for op in operands
        ):
            control_flow_known = False
    raw_operands = tuple(format_operand(op) for op in operands)

    # Preserve Capstone's own display text (including combined rep mnemonic).
    # Zero-operand lines keep a trailing space to match historical
    # ``" ".join((mnemonic, op_str))`` output used in reports/diffs.
    if insn.op_str:
        display = f"{insn.mnemonic} {insn.op_str}".rstrip()
    else:
        display = f"{insn.mnemonic} "

    return DecodedInstruction(
        address=insn.address,
        size=insn.size,
        mnemonic=mnemonic,
        prefix=prefix,
        operands=operands,
        raw_operands=raw_operands,
        display=display,
        role=AsmRole.CODE,
        regs_read=regs_read,
        regs_written=regs_written,
        reads_flags=bool(insn.eflags & _EFLAGS_READ_MASK),
        writes_flags=bool(insn.eflags & ~_EFLAGS_READ_MASK),
        accesses_memory=any(op.type == x86_const.X86_OP_MEM for op in cs_operands),
        is_jump=is_jump,
        is_call=is_call,
        is_ret=insn.group(CS_GRP_RET),
        branch_target=branch_target,
        register_access_known=register_access_known,
        operand_model_complete=operand_model_complete,
        control_flow_known=control_flow_known,
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
    # InstructGen expects Capstone's combined mnemonic (e.g. ``rep movsd``).
    mnemonic = (
        f"{insn.prefix} {insn.mnemonic}".strip() if insn.prefix else insn.mnemonic
    )
    return (insn.address, insn.size, mnemonic, insn.raw_op_str)
