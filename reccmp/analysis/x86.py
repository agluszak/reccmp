"""The few x86 decoding facts catalog preparation needs.

Instruction semantics belong to Ghidra. The catalog only asks whether bytes
decode at all (vtable slot plausibility), where a ``jmp rel32`` goes (thunk
chains), which addresses a short CRT initializer mentions, and whether two
function bodies are the same code.
"""

from functools import cache
from typing import Iterator

from capstone import CS_ARCH_X86, CS_MODE_32, Cs, CsInsn  # type: ignore
from capstone import x86_const  # type: ignore


@cache
def _disassembler() -> Cs:
    disassembler = Cs(CS_ARCH_X86, CS_MODE_32)
    disassembler.detail = True
    return disassembler


def decode_one(code: bytes, address: int) -> CsInsn | None:
    """Decode the single instruction at the start of ``code``."""
    return next(_disassembler().disasm(code, address, 1), None)


def instructions(code: bytes, address: int) -> Iterator[CsInsn]:
    """Decode ``code`` in order, stopping at ``int3`` padding."""
    for insn in _disassembler().disasm(code, address):
        if insn.id == x86_const.X86_INS_INT3:
            return
        yield insn


def e9_jump_target(code: bytes, address: int) -> int | None:
    """Target of the 5-byte ``jmp rel32`` at ``address``, when present."""
    if len(code) < 5 or code[0] != 0xE9:
        return None
    return address + 5 + int.from_bytes(code[1:5], "little", signed=True)


def operand_addresses(
    insn: CsInsn, first: int = 0, last: int | None = None
) -> list[int]:
    """Values in the selected operands that may be addresses: immediates and
    memory displacements."""
    addrs = []
    for operand in insn.operands[first:last]:
        if operand.type == x86_const.X86_OP_IMM:
            addrs.append(operand.imm & 0xFFFFFFFF)
        elif operand.type == x86_const.X86_OP_MEM:
            addrs.append(operand.mem.disp & 0xFFFFFFFF)
    return addrs


def is_call(insn: CsInsn) -> bool:
    return insn.id == x86_const.X86_INS_CALL


def is_ret(insn: CsInsn) -> bool:
    return insn.id in (x86_const.X86_INS_RET, x86_const.X86_INS_RETF)


def is_jump(insn: CsInsn) -> bool:
    return insn.group(x86_const.X86_GRP_JUMP)


def pushed_immediate(insn: CsInsn) -> int | None:
    """The value of ``push imm32``."""
    if (
        insn.id == x86_const.X86_INS_PUSH
        and len(insn.operands) == 1
        and insn.operands[0].type == x86_const.X86_OP_IMM
    ):
        return insn.operands[0].imm & 0xFFFFFFFF
    return None


def direct_call_target(insn: CsInsn) -> int | None:
    """Destination of a relative ``call``."""
    if (
        is_call(insn)
        and len(insn.operands) == 1
        and insn.operands[0].type == x86_const.X86_OP_IMM
    ):
        return insn.operands[0].imm & 0xFFFFFFFF
    return None


def code_signature(code: bytes, address: int) -> tuple[bytes | tuple[str, int], ...]:
    """What a function body is, independent of where it was placed.

    Instruction bytes, except that a relative branch is described by its
    target: an offset into the body when it stays inside, the absolute
    address otherwise. Relocated operands are absolute already. Two bodies
    with one signature are the same code, which identical-code folding would
    have kept once."""
    end = address + len(code)
    signature: list[bytes | tuple[str, int]] = []
    for insn in instructions(code, address):
        relative = (
            insn.group(x86_const.X86_GRP_JUMP) or insn.group(x86_const.X86_GRP_CALL)
        ) and any(operand.type == x86_const.X86_OP_IMM for operand in insn.operands)
        if not relative:
            signature.append(bytes(insn.bytes))
            continue
        target = insn.operands[0].imm & 0xFFFFFFFF
        if address <= target < end:
            signature.append(("inside", target - address))
        else:
            signature.append(("at", target))
    return tuple(signature)
