"""Decode path fills structured operands from Capstone detail."""

from types import SimpleNamespace
from dataclasses import replace

from capstone import x86_const  # type: ignore[import-untyped]

from reccmp.compare.asm.decode import capstone_operand, disasm_detail
from reccmp.compare.asm.ir import (
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
    instruction_match_key,
)
from reccmp.compare.asm.verifier import analyze_effective_match
from reccmp.compare.diagnosis import ComparisonStatus


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
    assert insn.operand_model_complete is True


def test_display_does_not_change_decoded_memory_semantics():
    blob = bytes.fromhex("8b4508")  # mov eax, [ebp+8]
    decoded = disasm_detail(blob, 0x1000)[0]
    assert decoded.operands[1] == ("mem", "dword", "", [("ebp", 1)], 8, ())
    assert instruction_match_key(decoded) == instruction_match_key(
        replace(decoded, display="diagnostic text changed")
    )


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
    insn_a = SimpleNamespace(op_str="mystery_a", bytes=b"\x00\x01")
    insn_b = SimpleNamespace(op_str="mystery_b", bytes=b"\x00\x02")
    op = SimpleNamespace(type=x86_const.X86_OP_INVALID)
    left, left_ok = capstone_operand(insn_a, op, "ud2", 0)
    right, right_ok = capstone_operand(insn_b, op, "ud2", 0)
    same_bytes, _ = capstone_operand(
        SimpleNamespace(op_str="another rendering", bytes=b"\x00\x01"),
        op,
        "ud2",
        0,
    )
    assert left_ok is False and right_ok is False
    assert left != right
    assert left == same_bytes
    assert isinstance(left, tuple) and isinstance(right, tuple)
    assert left[0] == "opaque" and right[0] == "opaque"

    row_a = DecodedInstruction(
        address=0x1000,
        size=2,
        mnemonic="ud2",
        prefix="",
        operands=(left,),
        display="ud2 mystery_a",
        operand_model_complete=False,
        control_flow_known=False,
    )
    row_b = DecodedInstruction(
        address=0x2000,
        size=2,
        mnemonic="ud2",
        prefix="",
        operands=(right,),
        display="ud2 mystery_b",
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
    assert isinstance(operand, tuple)
    assert operand[0] == "mem"
    assert operand[1] == "size3"


def test_indirect_call_destination_expression_is_modeled():
    row = disasm_detail(bytes.fromhex("ff5208"), 0x1000)[0]
    assert row.is_call
    assert row.branch_target is None
    assert row.operands == (("mem", "dword", "", [("edx", 1)], 8, ()),)
    assert row.operand_model_complete
    assert row.control_flow_known


def test_recognized_switch_table_completes_indirect_jump():
    row = disasm_detail(bytes.fromhex("ff248500100000"), 0x1000)[0]
    assert row.is_jump and not row.control_flow_known
    incomplete = FunctionImage(0x1000, 7, ExtentKind.KNOWN, (row,))
    assert not incomplete.control_flow_complete
    table = JumpTable(
        0x1000,
        ((0x1000, 0x1007),),
        dispatch_address=0x1000,
        index_register="eax",
    )
    image = FunctionImage(0x1000, 7, ExtentKind.KNOWN, (row,), (table,))
    assert image.control_flow_complete


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
        display="nop a",
        operand_model_complete=False,
    )
    row_b = DecodedInstruction(
        address=0x2000,
        size=1,
        mnemonic="nop",
        prefix="",
        operands=(opaque_b,),
        display="nop b",
        operand_model_complete=False,
    )
    assert instruction_match_key(row_a) == instruction_match_key(row_b)

    original = FunctionImage(0x1000, 1, ExtentKind.KNOWN, (row_a,))
    recompiled = FunctionImage(0x2000, 1, ExtentKind.KNOWN, (row_b,))
    analysis = analyze_effective_match([], original, recompiled)
    assert analysis.status != ComparisonStatus.EXACT
