"""The bit-vector lowering must compute what the CPU computes.

Random straight-line programs over the operations `bitvector.py` models
run in Unicorn, and the same bytes go through the verifier's decoder,
symbolic semantics and bit-vector lowering. With the initial registers
substituted, every lowered register value must evaluate to the register
the CPU produced. A mismatch is a bug in the lowering (or in the symbolic
semantics), exactly the kind of bug that would make an algebraic proof
unsound; Z3 itself is not the risky part.

Coverage is counted per operation and width, over registers the program
wrote whose value was checked: every operation must be checked often, not
merely appear in a program.
"""

import random
from collections import Counter
from typing import Any

import unicorn  # type: ignore[import-untyped]
import z3  # type: ignore[import-untyped]
from unicorn import x86_const  # type: ignore[import-untyped]

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.ir import instruction_at, resolve_asm_stream
from reccmp.compare.asm.model import Reject
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
# Register values at the edges of signed and unsigned ranges, where carries,
# sign extension and comparison results change.
BOUNDARIES = (
    0,
    1,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFE,
    0xFFFFFFFF,
)
# Operations a program writes a register with, by width.
OPERATIONS = {
    "alu r32",
    "alu r8",
    "alu r32, imm8",
    "imul r32",
    "imul r32, r32, imm8",
    "imul r32, r32, imm32",
    "shift r32, imm8",
    "shift r8, imm8",
    "shift r32, cl",
    "shift r8, cl",
    "unary r32",
    "unary r8",
    "extend r8",
    "extend r16",
    "setcc r8",
    "mov r32, imm32",
    "mov r8",
    "alu r16",
    "cdq",
    "cwde",
}


def _modrm(reg: int, rm: int) -> int:
    return 0xC0 | (reg << 3) | rm


def _family8(register: int) -> str:
    """The family of al cl dl bl ah ch dh bh."""
    return FAMILIES[register & 3]


def _instruction(rng: random.Random) -> tuple[bytes, str, str | None]:
    """An instruction, its operation, and the register family it writes."""
    # pylint: disable=too-many-return-statements,too-many-branches

    def r32() -> int:  # eax ecx edx ebx
        return rng.randrange(4)

    def r8() -> int:  # al cl dl bl ah ch dh bh
        return rng.randrange(8)

    kind = rng.randrange(18)
    if kind == 0:  # add/or/and/sub/xor r32, r32
        dest = r32()
        opcode = rng.choice((0x01, 0x09, 0x21, 0x29, 0x31))
        return bytes([opcode, _modrm(r32(), dest)]), "alu r32", FAMILIES[dest]
    if kind == 1:  # the same on bytes
        dest = r8()
        opcode = rng.choice((0x00, 0x08, 0x20, 0x28, 0x30))
        return bytes([opcode, _modrm(r8(), dest)]), "alu r8", _family8(dest)
    if kind == 2:  # op r32, imm8 (sign-extended)
        dest = r32()
        digit = rng.choice((0, 1, 4, 5, 6))
        code = bytes([0x83, _modrm(digit, dest), rng.randrange(256)])
        return code, "alu r32, imm8", FAMILIES[dest]
    if kind == 3:  # imul r32, r32
        dest = r32()
        return bytes([0x0F, 0xAF, _modrm(dest, r32())]), "imul r32", FAMILIES[dest]
    if kind == 4:  # shl/shr/sar r32, imm8 (counts past 31 test the masking)
        dest = r32()
        code = bytes([0xC1, _modrm(rng.choice((4, 5, 7)), dest), rng.randrange(40)])
        return code, "shift r32, imm8", FAMILIES[dest]
    if kind == 5:  # shl/shr/sar r8, imm8
        dest = r8()
        code = bytes([0xC0, _modrm(rng.choice((4, 5, 7)), dest), rng.randrange(40)])
        return code, "shift r8, imm8", _family8(dest)
    if kind == 6:  # inc/dec r32, neg/not r32
        dest = r32()
        opcode, digit = rng.choice(((0xFF, 0), (0xFF, 1), (0xF7, 3), (0xF7, 2)))
        return bytes([opcode, _modrm(digit, dest)]), "unary r32", FAMILIES[dest]
    if kind == 7:  # the same on bytes
        dest = r8()
        opcode, digit = rng.choice(((0xFE, 0), (0xFE, 1), (0xF6, 3), (0xF6, 2)))
        return bytes([opcode, _modrm(digit, dest)]), "unary r8", _family8(dest)
    if kind == 8:  # movzx/movsx r32, r8 / r16
        dest = r32()
        second = rng.choice((0xB6, 0xBE, 0xB7, 0xBF))
        byte = second in (0xB6, 0xBE)
        source = r8() if byte else r32()
        operation = "extend r8" if byte else "extend r16"
        return bytes([0x0F, second, _modrm(dest, source)]), operation, FAMILIES[dest]
    if kind == 9:  # cmp/test, then setcc r8
        compare = rng.choice(
            (
                bytes([0x39, _modrm(r32(), r32())]),
                bytes([0x85, _modrm(r32(), r32())]),
                bytes([0x83, _modrm(7, r32()), rng.randrange(256)]),
                bytes([0x38, _modrm(r8(), r8())]),
            )
        )
        dest = r8()
        setcc = bytes([0x0F, 0x90 | rng.choice(CONDITIONS), _modrm(0, dest)])
        return compare + setcc, "setcc r8", _family8(dest)
    if kind == 10:  # mov r32, imm32
        dest = r32()
        immediate = rng.choice((rng.randrange(1 << 32), rng.choice(BOUNDARIES)))
        code = bytes([0xB8 + dest]) + immediate.to_bytes(4, "little")
        return code, "mov r32, imm32", FAMILIES[dest]
    if kind == 11:  # mov r8, r8
        dest = r8()
        return bytes([0x88, _modrm(r8(), dest)]), "mov r8", _family8(dest)
    if kind == 12:  # 16-bit add/sub/and/xor ax..bx
        dest = r32()
        opcode = rng.choice((0x01, 0x29, 0x21, 0x31))
        return bytes([0x66, opcode, _modrm(r32(), dest)]), "alu r16", FAMILIES[dest]
    if kind == 13:  # shl/shr/sar r32, cl (the count is masked to five bits)
        dest = r32()
        code = bytes([0xD3, _modrm(rng.choice((4, 5, 7)), dest)])
        return code, "shift r32, cl", FAMILIES[dest]
    if kind == 14:  # shl/shr/sar r8, cl
        dest = r8()
        code = bytes([0xD2, _modrm(rng.choice((4, 5, 7)), dest)])
        return code, "shift r8, cl", _family8(dest)
    if kind == 15:  # imul r32, r32, imm8 (sign-extended)
        dest = r32()
        code = bytes([0x6B, _modrm(dest, r32()), rng.randrange(256)])
        return code, "imul r32, r32, imm8", FAMILIES[dest]
    if kind == 16:  # imul r32, r32, imm32
        dest = r32()
        immediate = rng.choice((rng.randrange(1 << 32), rng.choice(BOUNDARIES)))
        code = bytes([0x69, _modrm(dest, r32())]) + immediate.to_bytes(4, "little")
        return code, "imul r32, r32, imm32", FAMILIES[dest]
    if rng.randrange(2):
        return b"\x99", "cdq", "d"
    return b"\x98", "cwde", "a"


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
    except Reject:
        return None  # an instruction the semantics do not model
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


def _registers(rng: random.Random) -> list[int]:
    return [
        rng.choice(BOUNDARIES) if rng.randrange(2) else rng.randrange(1 << 32)
        for _ in FAMILIES
    ]


def test_lowered_values_match_the_cpu():
    rng = random.Random(1234)
    checked: Counter[str] = Counter()
    rejected = 0
    for _ in range(2000):
        program = [_instruction(rng) for _ in range(rng.randrange(1, 7))]
        code = b"".join(code for code, _, _ in program)
        # The operation that last wrote each register decides what its
        # final value checks.
        writer = {family: operation for _, operation, family in program if family}
        registers = _registers(rng)
        state = _symbolic(code)
        if state is None:
            rejected += 1
            continue
        cpu = _cpu(code, registers)
        for family, expected in zip(FAMILIES, cpu):
            lowered = _evaluate(state.regs[family], registers)
            if lowered is None:
                continue
            assert lowered == expected, (
                f"{code.hex()} with {[hex(r) for r in registers]}: "
                f"e{family}x lowered to {lowered:#x}, the CPU has {expected:#x}"
            )
            if family in writer:
                checked[writer[family]] += 1
    # Every modelled operation's written result is checked, many times, and
    # the semantics reject none of these programs.
    assert rejected == 0
    assert set(checked) == OPERATIONS
    assert min(checked.values()) >= 50, checked
