"""Differential execution witnesses (reccmp.compare.witness).

Each test builds two tiny binaries whose code and data sit at different
addresses, the way a recompiled image differs from the original, and checks
that a witness is found only for a real observable difference.
"""

from types import SimpleNamespace

import pytest

from reccmp.compare.db import EntityDb
from reccmp.compare.diagnosis import ComparisonAnalysis, ComparisonStatus
from reccmp.formats.image import ImageSection, ImageSectionFlags
from reccmp.types import EntityType, ImageId

pytest.importorskip("unicorn")

# pylint: disable=wrong-import-position
from reccmp.compare.witness import SideMachine, Translator, find_witness

CODE = 0x401000
DATA = 0x402000
# The recompiled image places the function and its table elsewhere.
ORIG_FUNC, RECOMP_FUNC = CODE, CODE + 0x100
ORIG_TABLE, RECOMP_TABLE = DATA, DATA + 0x40
TABLE = bytes(range(16)) * 4  # 16 dwords
# A paired callee (just ``ret``) at different addresses on the two sides.
ORIG_CALLEE, RECOMP_CALLEE = CODE + 0x800, CODE + 0x900


def _image(code_at: int, code: bytes, table_at: int):
    code_page = bytearray(0x1000)
    code_page[code_at - CODE : code_at - CODE + len(code)] = code
    callee = ORIG_CALLEE if code_at == ORIG_FUNC else RECOMP_CALLEE
    code_page[callee - CODE] = 0xC3
    data_page = bytearray(0x1000)
    data_page[table_at - DATA : table_at - DATA + len(TABLE)] = TABLE

    def section(start: int, data: bytearray, flags) -> ImageSection:
        view = memoryview(bytes(data))
        return ImageSection(
            virtual_range=range(start, start + len(data)),
            physical_range=range(0, len(data)),
            view=view,
            flags=flags,
        )

    return SimpleNamespace(
        sections=[
            section(CODE, code_page, ImageSectionFlags.EXECUTE),
            section(DATA, data_page, ImageSectionFlags.READ),
        ]
    )


def _search(orig_code: bytes, recomp_code: bytes, return_kind: str = "i32"):
    db = EntityDb()
    with db.batch() as batch:
        for image_id, func, table in (
            (ImageId.ORIG, ORIG_FUNC, ORIG_TABLE),
            (ImageId.RECOMP, RECOMP_FUNC, RECOMP_TABLE),
        ):
            batch.set(image_id, func, type=EntityType.FUNCTION, size=0x100)
            batch.set(image_id, table, type=EntityType.DATA, size=len(TABLE))
            callee = ORIG_CALLEE if image_id == ImageId.ORIG else RECOMP_CALLEE
            batch.set(image_id, callee, type=EntityType.FUNCTION, size=1)
        batch.match(ORIG_FUNC, RECOMP_FUNC)
        batch.match(ORIG_TABLE, RECOMP_TABLE)
        batch.match(ORIG_CALLEE, RECOMP_CALLEE)
    translator = Translator(
        db,
        SideMachine(_image(ORIG_FUNC, orig_code, ORIG_TABLE)),  # type: ignore[arg-type]
        SideMachine(
            _image(RECOMP_FUNC, recomp_code, RECOMP_TABLE)  # type: ignore[arg-type]
        ),
    )
    return find_witness(
        translator,
        range(ORIG_FUNC, ORIG_FUNC + len(orig_code)),
        range(RECOMP_FUNC, RECOMP_FUNC + len(recomp_code)),
        return_kind=return_kind,
    )


def _abs32(value: int) -> bytes:
    return value.to_bytes(4, "little")


# mov eax, [esp+4]
LOAD_ARG = bytes.fromhex("8b442404")
RET = bytes.fromhex("c3")


def test_different_constant_is_refuted():
    result = _search(
        LOAD_ARG + bytes.fromhex("83c001") + RET,  # add eax, 1
        LOAD_ARG + bytes.fromhex("83c002") + RET,  # add eax, 2
    )
    assert result.witness is not None
    assert result.witness.kind == "return_value"


def test_equivalent_arithmetic_is_not_refuted():
    result = _search(
        LOAD_ARG + bytes.fromhex("01c0") + RET,  # add eax, eax
        LOAD_ARG + bytes.fromhex("d1e0") + RET,  # shl eax, 1
    )
    assert result.witness is None
    assert result.agreeing_seeds > 0


def test_void_return_value_is_not_compared():
    result = _search(
        LOAD_ARG + bytes.fromhex("83c001") + RET,
        LOAD_ARG + bytes.fromhex("83c002") + RET,
        return_kind="void",
    )
    assert result.witness is None


def test_different_stored_value_is_refuted():
    # mov ecx, [esp+4]; mov dword ptr [ecx+8], imm32; ret
    store = bytes.fromhex("8b4c2404c74108")
    result = _search(store + _abs32(5) + RET, store + _abs32(6) + RET)
    assert result.witness is not None
    assert result.witness.kind == "memory_value"


def test_same_table_entry_at_different_addresses_agrees():
    # mov eax, [esp+4]; and eax, 15; mov eax, [eax*4 + table]; ret
    def body(table: int) -> bytes:
        return LOAD_ARG + bytes.fromhex("83e00f8b0485") + _abs32(table) + RET

    result = _search(body(ORIG_TABLE), body(RECOMP_TABLE))
    assert result.witness is None
    assert result.agreeing_seeds > 0


def test_unbounded_table_index_is_not_a_witness():
    """Indexing past the table reads whatever each binary placed there."""

    # mov eax, [esp+4]; mov eax, [eax*4 + table]; ret
    def body(table: int) -> bytes:
        return LOAD_ARG + bytes.fromhex("8b0485") + _abs32(table) + RET

    assert _search(body(ORIG_TABLE), body(RECOMP_TABLE)).witness is None


def test_uninitialized_locals_are_not_a_witness():
    """Stale stack contents depend on each side's frame layout."""
    # sub esp, 8; mov eax, [esp + k]; add esp, 8; ret
    orig = bytes.fromhex("83ec088b0424") + bytes.fromhex("83c408") + RET
    recomp = bytes.fromhex("83ec088b442404") + bytes.fromhex("83c408") + RET
    assert _search(orig, recomp).witness is None


def test_pointer_to_paired_data_compares_by_identity():
    # mov eax, table; ret
    result = _search(
        bytes.fromhex("b8") + _abs32(ORIG_TABLE) + RET,
        bytes.fromhex("b8") + _abs32(RECOMP_TABLE) + RET,
    )
    assert result.witness is None


def test_witness_turns_inconclusive_into_refuted_mismatch():
    result = _search(
        LOAD_ARG + bytes.fromhex("83c001") + RET,
        LOAD_ARG + bytes.fromhex("83c002") + RET,
    )
    assert result.witness is not None
    refuted = ComparisonAnalysis.inconclusive("non_isomorphic_cfg").with_witness(
        result.witness
    )
    assert refuted.status == ComparisonStatus.MISMATCH
    assert refuted.is_refuted
    assert refuted.difference is not None
    assert refuted.difference.kind == "return_value"
    with pytest.raises(ValueError):
        ComparisonAnalysis.exact().with_witness(result.witness)


def test_agreeing_runs_record_reaching_the_location():
    result = _search(
        LOAD_ARG + bytes.fromhex("01c0") + RET,  # add eax, eax
        LOAD_ARG + bytes.fromhex("d1e0") + RET,  # shl eax, 1
    )
    assert result.witness is None
    assert result.agreeing_runs_through(ORIG_FUNC + 4, RECOMP_FUNC + 4) == (
        result.agreeing_seeds
    )
    assert result.agreeing_runs_through(ORIG_FUNC + 0x80, None) == 0


def _call(at: int, target: int) -> bytes:
    return b"\xe8" + _abs32((target - (at + 5)) & 0xFFFFFFFF)


def _store_then_call(value: int, func: int, callee: int) -> bytes:
    # mov ecx, [esp+4]; mov dword ptr [ecx+8], imm32; call callee; ret
    prefix = bytes.fromhex("8b4c2404c74108") + _abs32(value)
    return prefix + _call(func + len(prefix), callee) + RET


def test_store_before_a_call_is_not_settled():
    """A callee may overwrite a store made before it (e.g. a base destructor
    resetting the vtable pointer), so such a difference is no witness."""
    result = _search(
        _store_then_call(5, ORIG_FUNC, ORIG_CALLEE),
        _store_then_call(6, RECOMP_FUNC, RECOMP_CALLEE),
    )
    assert result.witness is None
    assert result.agreeing_seeds > 0


def test_paired_call_is_matched_by_identity():
    result = _search(
        _store_then_call(5, ORIG_FUNC, ORIG_CALLEE),
        _store_then_call(5, RECOMP_FUNC, RECOMP_CALLEE),
    )
    assert result.witness is None
    assert "call_structure" not in result.skipped
    assert result.agreeing_seeds > 0


def test_store_before_a_tail_call_is_not_settled():
    # mov ecx, [esp+4]; mov dword ptr [ecx+8], imm32; jmp callee
    def body(value: int, func: int, callee: int) -> bytes:
        prefix = bytes.fromhex("8b4c2404c74108") + _abs32(value)
        at = func + len(prefix)
        return prefix + b"\xe9" + _abs32((callee - (at + 5)) & 0xFFFFFFFF)

    result = _search(
        body(5, ORIG_FUNC, ORIG_CALLEE), body(6, RECOMP_FUNC, RECOMP_CALLEE)
    )
    assert result.witness is None
    assert result.agreeing_seeds > 0


def test_return_differing_only_above_al_is_not_a_witness():
    """Retail may return a bool in al; the declared width is the reconstruction's."""
    result = _search(
        bytes.fromhex("b800530000") + RET,  # mov eax, 0x5300
        bytes.fromhex("31c0") + RET,  # xor eax, eax
    )
    assert result.witness is None
    assert result.skipped.get("return_upper_bits")
