"""Decode path fills structured operands from Capstone detail."""

from reccmp.compare.asm.decode import disasm_detail, from_capstone
from reccmp.compare.asm.model import parse_instruction
from capstone import Cs, CS_ARCH_X86, CS_MODE_32


def test_from_capstone_mem_operand_without_display_parse():
    """Indexed memory operands come from Capstone detail, not op_str reparse."""
    # mov eax, dword ptr [ebx + ecx*4 + 0x10]
    blob = bytes.fromhex("8b448b10")
    rows = disasm_detail(blob, 0x1000)
    assert len(rows) == 1
    insn = rows[0]
    assert insn.mnemonic == "mov"
    assert insn.operands == (
        ("reg", "eax"),
        ("mem", "dword", "", [("ebx", 1), ("ecx", 4)], 0x10, ()),
    )
    # Display may be Capstone's text; operands must not depend on reparsing it.
    assert insn.operands != ()
    assert insn.raw_operands == ("eax", "dword ptr [ebx + ecx*4 + 0x10]")


def test_from_capstone_matches_parse_of_display_for_mem():
    blob = bytes.fromhex("8b4508")  # mov eax, [ebp+8]
    cs = Cs(CS_ARCH_X86, CS_MODE_32)
    cs.detail = True
    (cs_insn,) = cs.disasm(blob, 0x1000)
    decoded = from_capstone(cs_insn)
    parsed = parse_instruction(decoded.display)
    assert decoded.mnemonic == parsed.mnemonic
    assert decoded.prefix == parsed.prefix
    assert decoded.operands == parsed.operands


def test_from_capstone_lea_omits_mem_size():
    blob = bytes.fromhex("8d4508")  # lea eax, [ebp+8]
    rows = disasm_detail(blob, 0x1000)
    assert rows[0].operands[1] == ("mem", "", "", [("ebp", 1)], 8, ())


def test_from_capstone_st_register():
    blob = bytes.fromhex("d9c1")  # fld st(1)
    rows = disasm_detail(blob, 0x1000)
    assert rows[0].operands == (("st", 1),)


def test_from_capstone_rep_prefix():
    blob = bytes.fromhex("f3a5")  # rep movsd
    rows = disasm_detail(blob, 0x1000)
    assert rows[0].prefix == "rep"
    assert rows[0].mnemonic == "movsd"
    assert rows[0].operands[0][0] == "mem"
