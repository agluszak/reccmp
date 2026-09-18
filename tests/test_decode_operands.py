"""Decode path fills structured operands from Capstone detail."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from capstone import x86_const

from reccmp.compare.asm.decode import capstone_operand, disasm_detail, from_capstone
from reccmp.compare.asm.ir import AsmRole, DecodedInstruction, instruction_match_key
from reccmp.compare.asm.model import parse_instruction
from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.functions import FunctionComparator


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
    assert insn.operand_model_complete is True


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
    assert rows[0].operand_model_complete is True


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


def test_opaque_operands_remain_distinct_in_match_keys():
    """Unsupported Capstone kinds must not collapse to a shared ``("sym", "?")``."""
    insn_a = SimpleNamespace(op_str="mystery_a")
    insn_b = SimpleNamespace(op_str="mystery_b")
    op = SimpleNamespace(type=x86_const.X86_OP_INVALID)
    left, left_ok = capstone_operand(insn_a, op, "ud2", 0)
    right, right_ok = capstone_operand(insn_b, op, "ud2", 0)
    assert left_ok is False and right_ok is False
    assert left != right
    assert left[0] == "opaque" and right[0] == "opaque"

    row_a = DecodedInstruction(
        address=0x1000,
        size=2,
        mnemonic="ud2",
        prefix="",
        operands=(left,),
        raw_operands=("mystery_a",),
        display="ud2 mystery_a",
        role=AsmRole.CODE,
        operand_model_complete=False,
        control_flow_known=False,
    )
    row_b = DecodedInstruction(
        address=0x2000,
        size=2,
        mnemonic="ud2",
        prefix="",
        operands=(right,),
        raw_operands=("mystery_b",),
        display="ud2 mystery_b",
        role=AsmRole.CODE,
        operand_model_complete=False,
        control_flow_known=False,
    )
    assert instruction_match_key(row_a) != instruction_match_key(row_b)


def test_unknown_mem_size_stays_distinct_and_incomplete():
    insn = SimpleNamespace(op_str="byte ptr [eax]", reg_name=lambda _r: "eax")
    mem = SimpleNamespace(base=1, index=0, scale=1, segment=0, disp=0)
    op = SimpleNamespace(type=x86_const.X86_OP_MEM, size=3, mem=mem)
    operand, complete = capstone_operand(insn, op, "mov", 0)
    assert complete is False
    assert operand[0] == "mem"
    assert operand[1] == "size3"


def test_incomplete_operand_model_blocks_exact_from_collapsed_keys():
    """ratio==1.0 from IR keys must not yield EXACT when models are incomplete
    and displays differ (the old collapsed ``?`` failure mode)."""
    opaque_a = ("opaque", x86_const.X86_OP_INVALID, "a", 0)
    opaque_b = ("opaque", x86_const.X86_OP_INVALID, "a", 0)
    # Same opaque identity → same match key, but different displays.
    row_a = DecodedInstruction(
        address=0x1000,
        size=1,
        mnemonic="nop",
        prefix="",
        operands=(opaque_a,),
        raw_operands=("a",),
        display="nop a",
        role=AsmRole.CODE,
        operand_model_complete=False,
    )
    row_b = DecodedInstruction(
        address=0x2000,
        size=1,
        mnemonic="nop",
        prefix="",
        operands=(opaque_b,),
        raw_operands=("b",),
        display="nop b",
        role=AsmRole.CODE,
        operand_model_complete=False,
    )
    assert instruction_match_key(row_a) == instruction_match_key(row_b)

    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=MagicMock(),
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
    )
    result = comparator._compare_function_assembly(
        [row_a],
        [row_b],
        [],
        include_diff=False,
    )
    assert result.match_ratio == 1.0
    assert result.analysis.status != ComparisonStatus.EXACT
