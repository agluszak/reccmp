"""The bit-vector lowering must compute what the CPU computes.

Random straight-line programs over the operations `bitvector.py` models
run in Unicorn, and the same bytes go through the verifier's decoder,
symbolic semantics and bit-vector lowering. With the initial registers
substituted, every lowered register value must evaluate to the register
the CPU produced. A mismatch is a bug in the lowering (or in the symbolic
semantics), exactly the kind of bug that would make an algebraic proof
unsound; Z3 itself is not the risky part.
"""

import random
from typing import Any

import unicorn  # type: ignore[import-untyped]
import z3  # type: ignore[import-untyped]
from unicorn import x86_const  # type: ignore[import-untyped]

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.ir import instruction_at, resolve_asm_stream
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.semantics import execute
from reccmp.compare.asm.verifier.state import Context, SideState

BASE = 0x1000
FAMILIES = ("a", "c", "d", "b")
UC_REGISTERS = (
    x86_const.UC_X86_REG_EAX,
    x86_const.UC_X86_REG_ECX,
    x86_const.UC_X86_REG_EDX,
    x86_const.UC_X86_REG_EBX,
)
CONDITIONS = (
    0x2,
    0x3,
    0x4,
    0x5,
    0x6,
    0x7,
    0xC,
    0xD,
    0xE,
    0xF,
)  # b ae e ne be a l ge le g


def _modrm(reg: int, rm: int) -> int:
    return 0xC0 | (reg << 3) | rm


def _instruction(rng: random.Random) -> bytes:
    # pylint: disable=too-many-return-statements

    def r32() -> int:  # eax ecx edx ebx
        return rng.randrange(4)

    def r8() -> int:  # al cl dl bl ah ch dh bh
        return rng.randrange(8)

    kind = rng.randrange(14)
    if kind == 0:  # add/or/and/sub/xor r32, r32
        return bytes([rng.choice((0x01, 0x09, 0x21, 0x29, 0x31)), _modrm(r32(), r32())])
    if kind == 1:  # the same on bytes
        return bytes([rng.choice((0x00, 0x08, 0x20, 0x28, 0x30)), _modrm(r8(), r8())])
    if kind == 2:  # op r32, imm8 (sign-extended)
        digit = rng.choice((0, 1, 4, 5, 6))
        return bytes([0x83, _modrm(digit, r32()), rng.randrange(256)])
    if kind == 3:  # imul r32, r32
        return bytes([0x0F, 0xAF, _modrm(r32(), r32())])
    if kind == 4:  # shl/shr/sar r32, imm8 (counts past 31 test the masking)
        return bytes([0xC1, _modrm(rng.choice((4, 5, 7)), r32()), rng.randrange(40)])
    if kind == 5:  # shl/shr/sar r8, imm8
        return bytes([0xC0, _modrm(rng.choice((4, 5, 7)), r8()), rng.randrange(40)])
    if kind == 6:  # inc/dec r32, neg/not r32
        opcode, digit = rng.choice(((0xFF, 0), (0xFF, 1), (0xF7, 3), (0xF7, 2)))
        return bytes([opcode, _modrm(digit, r32())])
    if kind == 7:  # the same on bytes
        opcode, digit = rng.choice(((0xFE, 0), (0xFE, 1), (0xF6, 3), (0xF6, 2)))
        return bytes([opcode, _modrm(digit, r8())])
    if kind == 8:  # movzx/movsx r32, r8 / r16
        second = rng.choice((0xB6, 0xBE, 0xB7, 0xBF))
        source = r8() if second in (0xB6, 0xBE) else r32()
        return bytes([0x0F, second, _modrm(r32(), source)])
    if kind == 9:  # cmp/test, then setcc r8
        compare = rng.choice(
            (
                bytes([0x39, _modrm(r32(), r32())]),
                bytes([0x85, _modrm(r32(), r32())]),
                bytes([0x83, _modrm(7, r32()), rng.randrange(256)]),
                bytes([0x38, _modrm(r8(), r8())]),
            )
        )
        return compare + bytes([0x0F, 0x90 | rng.choice(CONDITIONS), _modrm(0, r8())])
    if kind == 10:  # mov r32, imm32
        return bytes([0xB8 + r32()]) + rng.randrange(1 << 32).to_bytes(4, "little")
    if kind == 11:  # mov r8, r8
        return bytes([0x88, _modrm(r8(), r8())])
    if kind == 12:  # 16-bit add/sub/and/xor ax..bx
        return bytes([0x66, rng.choice((0x01, 0x29, 0x21, 0x31)), _modrm(r32(), r32())])
    return bytes([rng.choice((0x99, 0x98))])  # cdq, cwde


def _cpu(code: bytes, registers: list[int]) -> list[int]:
    emulator = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_32)
    emulator.mem_map(BASE, 0x1000)
    emulator.mem_write(BASE, code)
    for register, value in zip(UC_REGISTERS, registers):
        emulator.reg_write(register, value)
    emulator.emu_start(BASE, BASE + len(code))
    return [emulator.reg_read(register) for register in UC_REGISTERS]


def _symbolic(code: bytes) -> SideState | None:
    stream = resolve_asm_stream(list(disasm_detail(code, BASE)))
    state, ctx = SideState(), Context()
    obs: list[Any] = []
    try:
        for index in range(len(stream)):
            execute(state, ctx, index, instruction_at(stream, index), obs)
    except Exception:  # pylint: disable=broad-except
        return None  # an instruction the semantics reject
    return state


def _evaluate(value, registers: list[int]) -> int | None:
    """The lowered value with the initial registers substituted, or None
    when it is not built from them alone (an opaque term) or not lowered."""
    # pylint: disable=protected-access
    lowering = bitvector._Lowering()
    try:
        expression = lowering.sized(value, 32)
    except bitvector._Unsupported:
        return None
    substitutions = []
    for key, variable in lowering.variables.items():
        term: Any = key[0]  # type: ignore[index]
        bits: Any = key[1]  # type: ignore[index]
        if bits == "bool" or term[:1] != ("init",) or term[1] not in FAMILIES:
            return None
        concrete = registers[FAMILIES.index(term[1])]
        substitutions.append((variable, z3.BitVecVal(concrete, bits)))
    result = z3.simplify(z3.substitute(expression, *substitutions))
    return result.as_long() if z3.is_bv_value(result) else None


def test_lowered_values_match_the_cpu():
    rng = random.Random(1234)
    checked = 0
    for _ in range(2000):
        code = b"".join(_instruction(rng) for _ in range(rng.randrange(1, 7)))
        registers = [rng.randrange(1 << 32) for _ in FAMILIES]
        state = _symbolic(code)
        if state is None:
            continue
        cpu = _cpu(code, registers)
        for family, expected in zip(FAMILIES, cpu):
            lowered = _evaluate(state.regs[family], registers)
            if lowered is None:
                continue
            checked += 1
            assert lowered == expected, (
                f"{code.hex()} with {[hex(r) for r in registers]}: "
                f"e{family}x lowered to {lowered:#x}, the CPU has {expected:#x}"
            )
    # The generator must exercise the lowering, not skip everything.
    assert checked > 5000
