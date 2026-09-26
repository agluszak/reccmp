"""Differential execution witnesses (reccmp.compare.witness).

Each test builds two tiny binaries whose code and data sit at different
addresses, the way a recompiled image differs from the original, and checks
that a witness is found only for a real observable difference.
"""

from collections import deque
from types import SimpleNamespace

import pytest

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.ir import ExtentKind, FunctionImage
from reccmp.call_facts import CallFacts
from reccmp.compare.db import EntityDb
from reccmp.compare.diagnosis import ComparisonAnalysis, ComparisonStatus
from reccmp.formats.image import ImageSection, ImageSectionFlags
from reccmp.types import EntityType, ImageId

pytest.importorskip("unicorn")

# pylint: disable=wrong-import-position
from reccmp.compare.witness import SideMachine, Translator, find_witness
from reccmp.compare.witness.machine import (
    PAGE,
    STACK_BASE,
    STACK_TOP,
    RunInput,
    import_registry,
    page_contents,
)
from reccmp.compare.witness.search import _excerpt_constants

CODE = 0x401000
DATA = 0x402000
# The recompiled image places the function and its table elsewhere.
ORIG_FUNC, RECOMP_FUNC = CODE, CODE + 0x100
ORIG_TABLE, RECOMP_TABLE = DATA, DATA + 0x40
TABLE = bytes(range(16)) * 4  # 16 dwords
# A paired callee (just ``ret``) at different addresses on the two sides.
ORIG_CALLEE, RECOMP_CALLEE = CODE + 0x800, CODE + 0x900
RET_BYTE = b"\xc3"


def _image(code_at: int, code: bytes, table_at: int, callee_body: bytes = RET_BYTE):
    code_page = bytearray(0x1000)
    code_page[code_at - CODE : code_at - CODE + len(code)] = code
    callee = ORIG_CALLEE if code_at == ORIG_FUNC else RECOMP_CALLEE
    code_page[callee - CODE : callee - CODE + len(callee_body)] = callee_body
    data_page = bytearray(0x2000)
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


def _function_image(start: int, code: bytes) -> FunctionImage:
    return FunctionImage(
        start_addr=start,
        extent=len(code),
        extent_kind=ExtentKind.KNOWN,
        excerpt=tuple(disasm_detail(code, start)),
    )


def _search(
    orig_code: bytes,
    recomp_code: bytes,
    return_kind: str = "i32",
    *,
    callee_body: bytes = RET_BYTE,
    call_facts=None,
):
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

    def sizes(image_id: ImageId):
        def size(address: int) -> int | None:
            entity = db.get(image_id, address)
            return entity.size(image_id) if entity is not None else None

        return size

    translator = Translator(
        db,
        SideMachine(
            _image(ORIG_FUNC, orig_code, ORIG_TABLE, callee_body),  # type: ignore[arg-type]
            function_size=sizes(ImageId.ORIG),
        ),
        SideMachine(
            _image(RECOMP_FUNC, recomp_code, RECOMP_TABLE, callee_body),  # type: ignore[arg-type]
            function_size=sizes(ImageId.RECOMP),
        ),
        call_facts=call_facts,
    )
    return find_witness(
        translator,
        _function_image(ORIG_FUNC, orig_code),
        _function_image(RECOMP_FUNC, recomp_code),
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
    assert result.skipped.get("truncated")


def test_instruction_facts_come_from_capstone_detail():
    # push 7; call callee; add esp, 4; cmp eax, 0x1234; je +0; ret 0xc
    body = bytes.fromhex("6a07") + _call(ORIG_FUNC + 2, ORIG_FUNC + 0x40)
    body += bytes.fromhex("83c4043d341200007400c20c00")
    # A jmp thunk at +0x40 to a callee ending in ``ret 8``.
    thunk = b"\xe9" + _abs32((0x50 - 0x45) & 0xFFFFFFFF)
    code = body.ljust(0x40, b"\x90") + thunk.ljust(0x10, b"\x90") + b"\xc2\x08\x00"
    machine = SideMachine(
        _image(ORIG_FUNC, code, ORIG_TABLE),  # type: ignore[arg-type]
        function_size=lambda address: 3 if address == ORIG_FUNC + 0x50 else None,
    )

    # Branch targets and the ``ret`` immediate are not constants.
    image = _function_image(ORIG_FUNC, body)
    assert _excerpt_constants(image, machine) == {4, 7, 0x1234}
    assert machine.callee_pop_bytes(ORIG_FUNC + 0x40) == 8
    # pylint: disable-next=protected-access
    assert machine._caller_cleanup(ORIG_FUNC + 7) == 4


def test_reset_clears_every_stack_store_of_the_previous_run():
    # sub esp, 0x200; mov edi, esp; mov ecx, 0x80; xor eax, eax; dec eax;
    # rep stosd; push eax; pop eax; add esp, 0x200; ret
    code = bytes.fromhex("81ec000200008bfcb98000000031c048f3ab505881c400020000c3")
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    below_frame = (STACK_BASE, STACK_TOP - STACK_BASE)

    trace = machine.run(range(ORIG_FUNC, ORIG_FUNC + len(code)), RunInput.from_seed(0))
    assert trace.end == "return"
    assert any(machine.uc.mem_read(*below_frame))
    machine._reset(1)  # pylint: disable=protected-access
    assert not any(machine.uc.mem_read(*below_frame))


def test_return_differing_only_above_al_is_not_a_witness():
    """Retail may return a bool in al; the declared width is the reconstruction's."""
    result = _search(
        bytes.fromhex("b800530000") + RET,  # mov eax, 0x5300
        bytes.fromhex("31c0") + RET,  # xor eax, eax
    )
    assert result.witness is None
    assert result.skipped.get("return_upper_bits")


def _store_after_call(value: int, func: int, callee: int) -> bytes:
    # mov ecx, [esp+4]; push 1; call callee; mov dword ptr [ecx+8], imm32; ret
    prefix = bytes.fromhex("8b4c24046a01")
    return (
        prefix
        + _call(func + len(prefix), callee)
        + bytes.fromhex("c74108")
        + (_abs32(value) + RET)
    )


def test_nothing_after_a_guessed_call_cleanup_is_a_witness():
    """The callee (``jmp [eax]``) states no cleanup and the caller removes
    nothing, so the model guesses it pops the push. The guess may be wrong,
    and everything after it may then be a model artifact."""
    unknown_cleanup = bytes.fromhex("ff20")
    result = _search(
        _store_after_call(5, ORIG_FUNC, ORIG_CALLEE),
        _store_after_call(6, RECOMP_FUNC, RECOMP_CALLEE),
        callee_body=unknown_cleanup,
    )
    assert result.witness is None
    assert result.skipped.get("assumed_call_cleanup")
    # With a callee that states its cleanup, the same difference refutes.
    result = _search(
        _store_after_call(5, ORIG_FUNC, ORIG_CALLEE),
        _store_after_call(6, RECOMP_FUNC, RECOMP_CALLEE),
        callee_body=bytes.fromhex("c20400"),  # ret 4
    )
    assert result.witness is not None
    assert result.witness.kind == "memory_value"


def _ecx_then_call(value: int, func: int, callee: int, tail: bytes = RET) -> bytes:
    # mov ecx, imm32; call callee; <tail>
    prefix = b"\xb9" + _abs32(value)
    return prefix + _call(func + len(prefix), callee) + tail


def test_register_arguments_are_compared_when_the_callee_reads_them():
    orig = _ecx_then_call(5, ORIG_FUNC, ORIG_CALLEE)
    recomp = _ecx_then_call(6, RECOMP_FUNC, RECOMP_CALLEE)
    # Unknown convention: ecx may be dead, so a difference proves nothing.
    assert _search(orig, recomp, return_kind="void").witness is None
    thiscall = CallFacts(uses_ecx=True, uses_edx=False)
    result = _search(
        orig, recomp, return_kind="void", call_facts=lambda _identity: thiscall
    )
    assert result.witness is not None
    assert result.witness.kind == "call_argument"
    assert result.witness.location.endswith("ecx")


def test_calls_clobber_the_caller_saved_registers():
    """A value left in ecx before a call does not survive it."""
    mov_eax_ecx = bytes.fromhex("8bc1") + RET
    result = _search(
        _ecx_then_call(5, ORIG_FUNC, ORIG_CALLEE, mov_eax_ecx),
        _ecx_then_call(6, RECOMP_FUNC, RECOMP_CALLEE, mov_eax_ecx),
    )
    assert result.witness is None
    assert result.agreeing_seeds > 0


def test_a_read_straddling_the_end_of_an_object_is_not_known():
    """Three of the four bytes come from whatever each binary placed there."""

    # mov eax, [table + 63]; ret
    def body(table: int) -> bytes:
        return b"\xa1" + _abs32(table + len(TABLE) - 1) + RET

    result = _search(body(ORIG_TABLE), body(RECOMP_TABLE))
    assert result.witness is None
    assert result.skipped.get("unknown_image_read") == result.runs


def test_a_write_across_a_page_boundary_is_undone():
    boundary = DATA + 0x1000
    # mov dword ptr [boundary - 2], 0x11223344; ret
    code = b"\xc7\x05" + _abs32(boundary - 2) + _abs32(0x11223344) + RET
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    before = bytes(machine.uc.mem_read(boundary - 2, 4))
    trace = machine.run(range(ORIG_FUNC, ORIG_FUNC + len(code)), RunInput.from_seed(0))
    assert trace.end == "return"
    assert bytes(machine.uc.mem_read(boundary - 2, 4)) != before
    machine._reset(1)  # pylint: disable=protected-access
    assert bytes(machine.uc.mem_read(boundary - 2, 4)) == before


def test_code_that_writes_code_gives_no_verdict():
    # mov byte ptr [ORIG_CALLEE], 0x90; ret
    code = b"\xc6\x05" + _abs32(ORIG_CALLEE) + b"\x90" + RET
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    trace = machine.run(range(ORIG_FUNC, ORIG_FUNC + len(code)), RunInput.from_seed(0))
    assert trace.end == "code_write"


def test_pushed_bytes_stop_at_any_other_stack_pointer_write():
    # push 1; xchg eax, esp; push 2; push ax; push 0; call $+5
    code = bytes.fromhex("6a01946a0266506a00e800000000")
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    addresses = [ORIG_FUNC + offset for offset in (0, 2, 3, 5, 7, 9)]
    # pylint: disable-next=protected-access
    assert machine._pushed_bytes(deque(addresses)) == 4 + 2 + 4


def test_imports_share_one_collision_free_registry():
    def image(*imports: tuple[str, str]):
        return SimpleNamespace(
            get_imports=lambda: [
                SimpleNamespace(module=module, name=name, ordinal=0, addr=0)
                for module, name in imports
            ]
        )

    registry = import_registry(
        image(("KERNEL32.dll", "Sleep"), ("USER32.dll", "MessageBoxA")),
        image(("kernel32.DLL", "Sleep"), ("GDI32.dll", "TextOutA")),
    )
    assert list(registry) == [
        "gdi32.dll!TextOutA",
        "kernel32.dll!Sleep",
        "user32.dll!MessageBoxA",
    ]
    assert len(set(registry.values())) == 3


def test_generated_pages_are_deterministic_and_shared():
    page = 0x20000000
    first = page_contents(3, page)
    assert len(first) == PAGE
    assert page_contents(3, page) is first  # the other side reuses it
    assert page_contents(4, page) != first
    assert page_contents(3, 0x1000) == bytes(PAGE)


def test_a_callee_without_ret_does_not_lend_the_next_functions():
    # callee: mov eax, ecx; jmp eax (never returns here); next function: ret 8
    tail_jump = bytes.fromhex("8bc1ffe0")
    padded = tail_jump + b"\xcc" * 14 + bytes.fromhex("c20800")
    machine = SideMachine(_image(ORIG_FUNC, RET, ORIG_TABLE, padded))  # type: ignore[arg-type]
    assert machine.callee_pop_bytes(ORIG_CALLEE) is None
    # Without padding in between, the callee's known extent bounds the scan.
    adjacent = tail_jump + bytes.fromhex("c20800")
    sized = SideMachine(
        _image(ORIG_FUNC, RET, ORIG_TABLE, adjacent),  # type: ignore[arg-type]
        function_size=lambda address: 4 if address == ORIG_CALLEE else None,
    )
    assert sized.callee_pop_bytes(ORIG_CALLEE) is None
    # No known extent: unknown, never a scan into the next function.
    unsized = SideMachine(_image(ORIG_FUNC, RET, ORIG_TABLE, adjacent))  # type: ignore[arg-type]
    assert unsized.callee_pop_bytes(ORIG_CALLEE) is None
    # A thunk is followed, and its destination needs a known extent too.
    thunk = b"\xe9" + _abs32(0x10 - 5)  # jmp +0x10
    body = thunk.ljust(0x10, b"\x90") + bytes.fromhex("c20800")
    for known, expected in ((False, None), (True, 8)):

        def destination_size(address: int, known: bool = known) -> int | None:
            return 3 if known and address == ORIG_CALLEE + 0x10 else None

        machine = SideMachine(
            _image(ORIG_FUNC, RET, ORIG_TABLE, body),  # type: ignore[arg-type]
            function_size=destination_size,
        )
        assert machine.callee_pop_bytes(ORIG_CALLEE) == expected
