"""Capstone detail-mode decode into canonical ``DecodedInstruction`` rows.

The one disassembly pass: typed operands, register and flag effects and
branch targets come from Capstone's detail here; address sanitization
happens when the function image is built.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Iterable, TypeVar

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
    CsInsn,
)
from capstone import x86_const  # type: ignore

from .ir import DecodedInstruction, FlowKind
from .model import split_mnemonic_prefix
from .operand import Imm, Mem, Opaque, Operand, Reg, ScaledReg, St

_ST_REGISTERS = {getattr(x86_const, f"X86_REG_ST{index}"): index for index in range(8)}

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

_T = TypeVar("_T")


@cache
def get_detail_disassembler(is_32: bool = True) -> Cs:
    disassembler = Cs(CS_ARCH_X86, CS_MODE_32 if is_32 else CS_MODE_16)
    disassembler.detail = True
    return disassembler


def decode_one(code: bytes, address: int, is_32: bool = True) -> CsInsn | None:
    """Decode the single instruction at the start of ``code``."""
    return next(get_detail_disassembler(is_32).disasm(code, address, 1), None)


def direct_branch_target(insn: CsInsn) -> int | None:
    """Destination of a relative ``call``/``jmp``/``jcc``, else None."""
    operands = insn.operands
    if (
        (insn.group(CS_GRP_JUMP) or insn.group(CS_GRP_CALL))
        and insn.group(CS_GRP_BRANCH_RELATIVE)
        and len(operands) == 1
        and operands[0].type == x86_const.X86_OP_IMM
    ):
        return operands[0].imm
    return None


def e9_jump_target(code: bytes, address: int) -> int | None:
    """Target of the 5-byte ``jmp rel32`` at ``address``, when present."""
    if len(code) < 5 or code[0] != 0xE9:
        return None
    return address + 5 + int.from_bytes(code[1:5], "little", signed=True)


def jump_thunk_target(
    insn: CsInsn | None, read_absolute: Callable[[int], _T | None]
) -> int | _T | None:
    """One ``jmp`` thunk step: its direct target, or ``read_absolute`` on an
    absolute memory operand's slot. ``None`` when ``insn`` is not a supported
    ``jmp`` form."""
    if insn is None or insn.id != x86_const.X86_INS_JMP or not insn.operands:
        return None
    operand = insn.operands[0]
    if operand.type == x86_const.X86_OP_IMM:
        return operand.imm
    if (
        operand.type == x86_const.X86_OP_MEM
        and not operand.mem.base
        and not operand.mem.index
    ):
        return read_absolute(operand.mem.disp & 0xFFFFFFFF)
    return None


def jump_table_targets(
    insn: CsInsn,
    read_dword: Callable[[int], int | None],
    start: int,
    limit: int,
) -> tuple[int, ...] | None:
    """Targets of ``jmp dword ptr [index*4 + table]`` in the window.

    Table entries are read until the first unreadable/out-of-window target.
    ``None`` means ``insn`` is not that dispatch or produced no cases.
    """
    if insn.id != x86_const.X86_INS_JMP or not insn.operands:
        return None
    operand = insn.operands[0]
    if (
        operand.type != x86_const.X86_OP_MEM
        or operand.mem.scale != 4
        or not operand.mem.index
        or operand.mem.base
    ):
        return None
    table = operand.mem.disp & 0xFFFFFFFF
    targets: list[int] = []
    for index in range(256):
        target = read_dword(table + 4 * index)
        if target is None or not start <= target < start + limit:
            break
        targets.append(target)
    return tuple(targets) or None


def stop_at_int3_detail(instructions) -> Iterable:
    for insn in instructions:
        if insn.mnemonic == "int3":
            break
        yield insn


def _reg_operand(insn, register: int) -> Reg | St:
    index = _ST_REGISTERS.get(register)
    return St(index) if index is not None else Reg(insn.reg_name(register))


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
) -> tuple[Operand, bool]:
    """Convert one Capstone operand to a typed operand.

    Returns ``(operand, model_complete)``. Unsupported kinds become unique
    opaque operands so distinct unknowns cannot share match keys.
    """
    if op.type == x86_const.X86_OP_REG:
        return _reg_operand(insn, op.reg), True
    if op.type == x86_const.X86_OP_IMM:
        return Imm(int(op.imm)), True
    if op.type == x86_const.X86_OP_MEM:
        mem = op.mem
        registers: list[ScaledReg] = []
        if mem.base:
            registers.append(ScaledReg(insn.reg_name(mem.base), 1))
        if mem.index:
            registers.append(ScaledReg(insn.reg_name(mem.index), int(mem.scale)))
        segment = insn.reg_name(mem.segment) if mem.segment else ""
        size_name, size_known = _mem_size_name(mnemonic, op.size)
        return (
            Mem(size_name, segment, tuple(registers), int(mem.disp)),
            size_known,
        )
    # Rare / unsupported: machine bytes distinguish unknown operands without
    # letting Capstone's display spelling influence matching.
    return Opaque(op.type, bytes(insn.bytes), operand_index), False


def _flow_kind(insn) -> FlowKind:
    if insn.group(CS_GRP_RET):
        return FlowKind.RETURN
    if insn.group(CS_GRP_CALL):
        return FlowKind.CALL
    if insn.group(CS_GRP_JUMP):
        return (
            FlowKind.JUMP if insn.id == x86_const.X86_INS_JMP else FlowKind.CONDITIONAL
        )
    if insn.id == x86_const.X86_INS_INT3:
        return FlowKind.TRAP
    return FlowKind.NORMAL


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
    branch_target = direct_branch_target(insn)
    cs_operands = insn.operands

    prefix, mnemonic = split_mnemonic_prefix(insn.mnemonic)
    # Use the post-split mnemonic for lea size omission etc.
    decoded_ops = [
        capstone_operand(insn, op, mnemonic, index)
        for index, op in enumerate(cs_operands)
    ]
    operands = tuple(operand for operand, _complete in decoded_ops)
    operand_model_complete = all(complete for _operand, complete in decoded_ops)
    # Opaque operands mean jump/call targets are not fully modeled.
    control_flow_known = operand_model_complete and not any(
        isinstance(operand, Opaque) for operand in operands
    )
    # An indirect call's destination expression is its modeled operand; it
    # needs no absolute branch_target for an exact instruction comparison.
    # An indirect jump still needs a recovered switch table or destination
    # before its local control-flow topology is known.
    if control_flow_known and (is_jump or is_call) and branch_target is None:
        if is_jump:
            control_flow_known = False
        elif not operands:
            control_flow_known = False

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
        display=display,
        regs_read=regs_read,
        regs_written=regs_written,
        reads_flags=bool(insn.eflags & _EFLAGS_READ_MASK),
        writes_flags=bool(insn.eflags & ~_EFLAGS_READ_MASK),
        accesses_memory=any(op.type == x86_const.X86_OP_MEM for op in cs_operands),
        flow=_flow_kind(insn),
        branch_target=branch_target,
        register_access_known=register_access_known,
        operand_model_complete=operand_model_complete,
        control_flow_known=control_flow_known,
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
