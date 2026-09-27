"""Address sanitization operates on decoded operands, not assembly text."""

from reccmp.compare.asm.model import ResolvedAddress
from reccmp.compare.asm.operand import Sym
from reccmp.types import ImageId
from reccmp.compare.asm.parse import AddressSanitizer, decode_function
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
    assert AddressSanitizer().sanitize_row(row).operands == row.operands


def test_absolute_memory_uses_one_resolved_identity():
    parser = AddressSanitizer(resolver=_resolve)
    first = parser.sanitize_row(_row("mov eax, dword ptr [0x1234]"))
    second = parser.sanitize_row(_row("mov dword ptr [0x1234], edx"))
    first_ref = first.operands[1].symbols[0].ref
    second_ref = second.operands[0].symbols[0].ref
    assert first_ref.identity == second_ref.identity == ("entity", 0x100, 0)
    assert first_ref.display == second_ref.display == "g_data"


def test_unresolved_absolute_memory_has_side_local_identity():
    """Equal numeric addresses nothing names may hold different data on
    each side, so they never share an identity."""
    row = _row("mov eax, dword ptr [0x1234]")
    orig = AddressSanitizer(image_id=ImageId.ORIG).sanitize_row(row)
    recomp = AddressSanitizer(image_id=ImageId.RECOMP).sanitize_row(row)
    assert orig.operands[1].symbols[0].ref.identity == ("unresolved", "orig", 0x1234)
    assert recomp.operands[1].symbols[0].ref.identity == (
        "unresolved",
        "recomp",
        0x1234,
    )


def test_relocated_displacement_becomes_reference():
    parser = AddressSanitizer(addr_test=lambda addr: addr == 0x1234)
    row = parser.sanitize_row(_row("mov eax, dword ptr [ecx + 0x1234]"))
    memory = row.operands[1]
    assert memory.displacement == 0
    assert memory.symbols[0].ref.identity == ("unresolved", "unknown", 0x1234)


def test_nonrelocated_displacement_stays_numeric():
    row = _row("mov eax, dword ptr [ecx + 0x1234]")
    assert AddressSanitizer().sanitize_row(row).operands == row.operands


def test_segment_relative_address_does_not_become_image_reference():
    row = _row("mov eax, dword ptr fs:[0x1234]")
    parser = AddressSanitizer(addr_test=lambda _addr: True, resolver=_resolve)
    assert parser.sanitize_row(row).operands == row.operands


def test_relocated_immediate_becomes_reference():
    parser = AddressSanitizer(addr_test=lambda addr: addr == 0x1234, resolver=_resolve)
    row = parser.sanitize_row(_row("mov eax, 0x1234"))
    assert isinstance(row.operands[1], Sym)
    assert row.operands[1].ref.identity == ("entity", 0x100, 0)


def test_cmp_uses_only_named_relocated_reference():
    unnamed = AddressSanitizer(addr_test=lambda _addr: True)
    named = AddressSanitizer(addr_test=lambda _addr: True, resolver=_resolve)
    row = _row("cmp eax, 0x1234")
    assert unnamed.sanitize_row(row).operands == row.operands
    assert named.sanitize_row(row).operands[1].ref.identity == ("entity", 0x100, 0)


def test_direct_call_keeps_target_identity_independent_of_display():
    # call rel32 from 0x1000 to 0x1234
    code = b"\xe8" + (0x1234 - 0x1005).to_bytes(4, "little", signed=True)
    row = decode_function(code, 0x1000, resolver=_resolve).instructions[0]
    assert row.control_target == ("entity", 0x100, 0)
    assert row.operands[0].ref.display == "g_data"
