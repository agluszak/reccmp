"""Address sanitization operates on decoded operands, not assembly text."""

from reccmp.compare.asm.model import ResolvedAddress
from reccmp.compare.asm.parse import ParseAsm
from tests.asm_rows import rows


def _row(line: str):
    return rows([line])[0]


def _resolve(addr: int, exact: bool = False, indirect: bool = False):
    del exact, indirect
    if addr == 0x1234:
        return ResolvedAddress("g_data", ("entity", 0x100, 0))
    return None


def test_register_operands_keep_their_values():
    row = _row("cmp eax, edx")
    assert ParseAsm().sanitize_row(row).operands == row.operands


def test_absolute_memory_uses_one_resolved_identity():
    parser = ParseAsm(resolver=_resolve)
    first = parser.sanitize_row(_row("mov eax, dword ptr [0x1234]"))
    second = parser.sanitize_row(_row("mov dword ptr [0x1234], edx"))
    first_ref = first.operands[1][5][0][1]
    second_ref = second.operands[0][5][0][1]
    assert first_ref.identity == second_ref.identity == ("entity", 0x100, 0)
    assert first_ref.display == second_ref.display == "g_data"


def test_unproven_absolute_memory_stays_numeric():
    row = _row("mov eax, dword ptr [0x1234]")
    assert ParseAsm().sanitize_row(row).operands == row.operands


def test_relocated_displacement_becomes_reference():
    parser = ParseAsm(addr_test=lambda addr: addr == 0x1234)
    row = parser.sanitize_row(_row("mov eax, dword ptr [ecx + 0x1234]"))
    memory = row.operands[1]
    assert memory[4] == 0
    assert memory[5][0][1].identity == ("unresolved", "unknown", 0x1234)


def test_nonrelocated_displacement_stays_numeric():
    row = _row("mov eax, dword ptr [ecx + 0x1234]")
    assert ParseAsm().sanitize_row(row).operands == row.operands


def test_segment_relative_address_does_not_become_image_reference():
    row = _row("mov eax, dword ptr fs:[0x1234]")
    parser = ParseAsm(addr_test=lambda _addr: True, resolver=_resolve)
    assert parser.sanitize_row(row).operands == row.operands


def test_relocated_immediate_becomes_reference():
    parser = ParseAsm(addr_test=lambda addr: addr == 0x1234, resolver=_resolve)
    row = parser.sanitize_row(_row("mov eax, 0x1234"))
    assert row.operands[1][0] == "sym"
    assert row.operands[1][1].identity == ("entity", 0x100, 0)


def test_cmp_uses_only_named_relocated_reference():
    unnamed = ParseAsm(addr_test=lambda _addr: True)
    named = ParseAsm(addr_test=lambda _addr: True, resolver=_resolve)
    row = _row("cmp eax, 0x1234")
    assert unnamed.sanitize_row(row).operands == row.operands
    assert named.sanitize_row(row).operands[1][1].identity == ("entity", 0x100, 0)


def test_direct_call_keeps_target_identity_independent_of_display():
    # call rel32 from 0x1000 to 0x1234
    code = b"\xe8" + (0x1234 - 0x1005).to_bytes(4, "little", signed=True)
    row = ParseAsm(resolver=_resolve).parse_asm(code, 0x1000)[0]
    assert row.control_target == ("entity", 0x100, 0)
    assert row.operands[0][1].display == "g_data"
