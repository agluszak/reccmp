"""Differential execution witnesses (reccmp.compare.witness).

Each test builds two tiny binaries whose code and data sit at different
addresses, the way a recompiled image differs from the original, and checks
that a witness is found only for a real observable difference.
"""

# pylint: disable=too-many-lines

import dataclasses
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.ir import ExtentKind, FunctionImage
from reccmp.call_facts import CallFacts
from reccmp.compare.db import EntityDb, ReccmpEntity
from reccmp.compare.refutation import RefutationMixin
from reccmp.compare.asm.verifier import verify_effective_match
from reccmp.compare.comparison_json import analysis_json, parse_analysis
from reccmp.compare.extent import EntityExtent
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    RefutationWitness,
    ComparisonAnalysis,
    ComparisonStatus,
)
from reccmp.compare.refutation import _hint_fate, _solver_hints
from reccmp.formats.image import ImageSection, ImageSectionFlags
from reccmp.types import EntityType, ImageId

pytest.importorskip("unicorn")

# pylint: disable=wrong-import-position
from reccmp.compare.witness import (
    SearchResult,
    SideMachine,
    Translator,
    find_witness,
    replay,
)
from reccmp.compare.witness.cleanup import Cleanup
from reccmp.compare.witness.machine import (
    WITNESS_MODEL,
    HEAP_BASE,
    PAGE,
    STACK_BASE,
    STACK_TOP,
    RunInput,
    import_registry,
    page_contents,
)
from reccmp.compare.witness.hints import Rejection, input_from_assignment
from reccmp.compare.witness.search import HINT_SEED, UNRESOLVED, _excerpt_constants

CODE = 0x401000
DATA = 0x402000
# The recompiled image places the function and its table elsewhere.
ORIG_FUNC, RECOMP_FUNC = CODE, CODE + 0x100
ORIG_TABLE, RECOMP_TABLE = DATA, DATA + 0x40
TABLE = bytes(range(16)) * 4  # 16 dwords
# A paired callee (just ``ret``) at different addresses on the two sides.
ORIG_CALLEE, RECOMP_CALLEE = CODE + 0x800, CODE + 0x900
RET_BYTE = b"\xc3"


def _image(
    code_at: int,
    code: bytes,
    table_at: int,
    callee_body: bytes = RET_BYTE,
    relocated: range = range(0),
):
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
        ],
        is_relocated_addr=lambda addr: addr in relocated,
    )


def _function_image(start: int, code: bytes) -> FunctionImage:
    return FunctionImage(
        start_addr=start,
        extent=len(code),
        extent_kind=ExtentKind.KNOWN,
        excerpt=tuple(disasm_detail(code, start)),
    )


def _translator(
    orig_code: bytes,
    recomp_code: bytes,
    *,
    callee_body: bytes = RET_BYTE,
    call_facts=None,
    orig_table_size: int | None = len(TABLE),
    extent=None,
    objects: tuple[tuple[int, int, int], ...] = (),
    pair_table: bool = True,
    relocated_table: bool = False,
) -> Translator:
    """``objects``: more paired data, (original address, recompiled
    address, size)."""
    # pylint: disable=too-many-arguments
    db = EntityDb()
    with db.batch() as batch:
        for image_id, func, table in (
            (ImageId.ORIG, ORIG_FUNC, ORIG_TABLE),
            (ImageId.RECOMP, RECOMP_FUNC, RECOMP_TABLE),
        ):
            batch.set(image_id, func, type=EntityType.FUNCTION, size=0x100)
            table_size = orig_table_size if image_id == ImageId.ORIG else len(TABLE)
            batch.set(image_id, table, type=EntityType.DATA, size=table_size)
            callee = ORIG_CALLEE if image_id == ImageId.ORIG else RECOMP_CALLEE
            batch.set(image_id, callee, type=EntityType.FUNCTION, size=len(callee_body))
        batch.match(ORIG_FUNC, RECOMP_FUNC)
        if pair_table:
            batch.match(ORIG_TABLE, RECOMP_TABLE)
        batch.match(ORIG_CALLEE, RECOMP_CALLEE)
        for orig, recomp, size in objects:
            batch.set(ImageId.ORIG, orig, type=EntityType.DATA, size=size)
            batch.set(ImageId.RECOMP, recomp, type=EntityType.DATA, size=size)
            batch.match(orig, recomp)

    def sizes(image_id: ImageId):
        def size(address: int) -> EntityExtent | None:
            entity = db.get(image_id, address)
            if entity is None or entity.size(image_id) is None:
                return None
            return EntityExtent(entity.size(image_id))

        return size

    def relocations(table: int) -> range:
        return range(table, table + len(TABLE)) if relocated_table else range(0)

    return Translator(
        db,
        SideMachine(
            _image(ORIG_FUNC, orig_code, ORIG_TABLE, callee_body, relocations(ORIG_TABLE)),  # type: ignore[arg-type]
            function_window=sizes(ImageId.ORIG),
        ),
        SideMachine(
            _image(RECOMP_FUNC, recomp_code, RECOMP_TABLE, callee_body, relocations(RECOMP_TABLE)),  # type: ignore[arg-type]
            function_window=sizes(ImageId.RECOMP),
        ),
        call_facts=call_facts,
        extent=extent,
    )


def _search(
    orig_code: bytes,
    recomp_code: bytes,
    return_kind: str = "i32",
    *,
    hints=(),
    **setup,
):
    return find_witness(
        _translator(orig_code, recomp_code, **setup),
        _function_image(ORIG_FUNC, orig_code),
        _function_image(RECOMP_FUNC, recomp_code),
        return_kind=return_kind,
        hints=hints,
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
    )

    # Branch targets and the ``ret`` immediate are not constants.
    image = _function_image(ORIG_FUNC, body)
    assert _excerpt_constants(image, machine) == {4, 7, 0x1234}
    assert machine.callee_pop_bytes(ORIG_FUNC + 0x40) == Cleanup(8, True)
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
    # The diagnostics name the read, its instruction and the object it overran.
    detail = result.skipped_details["unknown_image_read"]
    assert detail["side"] == "orig" and detail["size"] == 4
    assert detail["instruction"] == f"{ORIG_FUNC:#x}"
    assert detail["first"]["why"] == "entity"
    assert detail["last"]["why"] == "outside_extent"
    assert detail["last"]["entity"]["address"] == f"{ORIG_TABLE:#x}"
    assert detail["counterpart_reads"] == [
        {
            "address": f"{RECOMP_TABLE + len(TABLE) - 1:#x}",
            "size": 4,
            "instruction": f"{RECOMP_FUNC:#x}",
            "identity": "('unresolved',)",
        }
    ]


def test_an_unresolved_call_is_described():
    """A call into the image where the database knows no function."""
    unknown = CODE + 0x600
    # mov eax, unknown; call eax; ret
    code = b"\xb8" + _abs32(unknown) + b"\xff\xd0" + RET
    result = _search(code, code)
    assert result.witness is None
    assert result.skipped.get("unresolved_call") == result.runs
    detail = result.skipped_details["unresolved_call"]
    assert detail["index"] == 0
    orig = detail["orig"]
    assert orig["kind"] == "register" and not orig["resolved"]
    assert orig["call_site"] == f"{ORIG_FUNC + 5:#x}"
    assert orig["target"]["why"] == "outside_extent"
    assert orig["target"]["entity"]["address"] == f"{ORIG_FUNC:#x}"
    assert orig["target"]["entity"]["canonical"] == f"{ORIG_FUNC:#x}"


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
    write = b"\xc6\x05" + _abs32(ORIG_CALLEE) + b"\x90" + RET
    # mov eax, ORIG_CALLEE; mov al, [eax]; ret: the callee's first byte
    probe = b"\xb8" + _abs32(ORIG_CALLEE) + b"\x8a\x00" + RET
    code = write.ljust(0x40, b"\x90") + probe
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    trace = machine.run(range(ORIG_FUNC, ORIG_FUNC + len(write)), RunInput.from_seed(0))
    assert trace.end == "code_write"
    # The stopped write still happened; the next run sees the code again.
    probe_range = range(ORIG_FUNC + 0x40, ORIG_FUNC + 0x40 + len(probe))
    after = machine.run(probe_range, RunInput.from_seed(0))
    assert after.end == "return" and after.eax & 0xFF == RET_BYTE[0]


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


def _callee_pop(
    body: bytes, window: int | None = None, *, recorded: bool = True
) -> Cleanup | None:
    extent = EntityExtent(window, recorded) if window is not None else None
    machine = SideMachine(
        _image(ORIG_FUNC, RET, ORIG_TABLE, body),  # type: ignore[arg-type]
        function_window=lambda address: extent if address == ORIG_CALLEE else None,
    )
    return machine.callee_pop_bytes(ORIG_CALLEE)


def test_a_callee_without_ret_does_not_lend_the_next_functions():
    # callee: mov eax, ecx; jmp eax (never returns here); next function: ret 8
    tail_jump = bytes.fromhex("8bc1ffe0")
    assert _callee_pop(tail_jump + b"\xcc" * 14 + bytes.fromhex("c20800")) is None
    assert _callee_pop(tail_jump + bytes.fromhex("c20800")) is None
    assert _callee_pop(tail_jump + bytes.fromhex("c20800"), window=4) is None


def test_callee_cleanup_comes_from_the_returns_control_flow_reaches():
    # nop; jmp +3; ret 8 (unreachable); ret 4
    assert _callee_pop(bytes.fromhex("90eb03c20800c20400")) == Cleanup(4, True)
    # A thunk is followed wherever it leads; no extent is needed.
    thunk = b"\xe9" + _abs32(0x10 - 5)  # jmp +0x10
    thunked = thunk.ljust(0x10, b"\x90") + bytes.fromhex("c20800")
    assert _callee_pop(thunked) == Cleanup(8, True)
    # Every instruction must lie whole inside the window.
    assert _callee_pop(bytes.fromhex("c20800"), window=2) is None


def test_a_return_reached_only_past_a_call_is_not_certain():
    """A call may not return: without a recorded size, the bytes after it
    may be the next function. With one, they are the callee's own."""
    call = _call(ORIG_CALLEE, ORIG_CALLEE + 0x40)
    # call; ret 8
    assert _callee_pop(call + bytes.fromhex("c20800")) == Cleanup(8, False)
    assert _callee_pop(call + bytes.fromhex("c20800"), window=8) == Cleanup(8, True)
    assert _callee_pop(
        call + bytes.fromhex("c20800"), window=8, recorded=False
    ) == Cleanup(8, False)
    # test ecx, ecx; jz +8; call; ret 8; ...; ret 4: they disagree.
    body = bytes.fromhex("85c97408") + call + bytes.fromhex("c20800c20400")
    assert _callee_pop(body) is None
    # The same return past the call and on a certain path agrees.
    body = bytes.fromhex("85c97408") + call + bytes.fromhex("c20800c20800")
    assert _callee_pop(body) == Cleanup(8, True)


def test_an_uncertain_cleanup_needs_the_paired_callees_to_agree():
    """The original callee's size is not recorded and its return lies past
    a call: its cleanup counts only when the recompiled callee's own,
    certain, cleanup is the same."""
    call = _call(ORIG_CALLEE, ORIG_CALLEE + 0x40)
    orig_callee = call + bytes.fromhex("c20400")  # call; ret 4

    # push 7; call callee; ret
    def caller(at: int, callee: int) -> bytes:
        return bytes.fromhex("6a07") + _call(at + 2, callee) + RET

    def translator(recomp_callee: bytes) -> Translator:
        translator = _translator(
            caller(ORIG_FUNC, ORIG_CALLEE), caller(RECOMP_FUNC, RECOMP_CALLEE)
        )
        for side, body, callee in (
            (ImageId.ORIG, orig_callee, ORIG_CALLEE),
            (ImageId.RECOMP, recomp_callee, RECOMP_CALLEE),
        ):
            machine = translator.machines[side]
            machine.uc.mem_write(callee, body)
            # pylint: disable-next=protected-access
            machine._pristine = bytes(
                machine.uc.mem_read(machine.image_range.start, len(machine.image_range))
            )
            machine.function_window = (
                (lambda _a: None)
                if side == ImageId.ORIG
                else (lambda _a, n=len(body): EntityExtent(n))
            )
            machine.insn_at.cache_clear()
            machine.callee_pop_bytes.cache_clear()
        return translator

    agreeing = translator(bytes.fromhex("c20400"))  # ret 4
    orig = agreeing.machines[ImageId.ORIG]
    assert orig.stack_cleanup(ORIG_CALLEE, ORIG_FUNC + 7) == 4
    differing = translator(bytes.fromhex("c20800"))  # ret 8
    orig = differing.machines[ImageId.ORIG]
    assert orig.stack_cleanup(ORIG_CALLEE, ORIG_FUNC + 7) is None


def test_jump_table_cases_must_agree_with_the_certain_returns():
    def switch(case_return: bytes) -> bytes:
        # cmp eax, 1; ja default; jmp [eax*4 + table]; case 0; case 1;
        # default: ret 8; table
        table = ORIG_CALLEE + 21
        head = bytes.fromhex("83f801770dff2485") + _abs32(table)
        cases = case_return + case_return + bytes.fromhex("c20800")
        return head + cases + _abs32(ORIG_CALLEE + 12) + _abs32(ORIG_CALLEE + 15)

    assert _callee_pop(switch(bytes.fromhex("c20800"))) == Cleanup(8, True)
    assert _callee_pop(switch(bytes.fromhex("c20400"))) is None


def test_the_solver_suggests_the_input_the_seeds_miss():
    """`arg + 5 < 0x1234` against `arg + 5 <= 0x1234` differ only at
    arg == 0x122f. Z3 finds it from the verifier's values, and the run on
    that input reproduces the divergence on both machines."""
    # mov eax, [esp+4]; add eax, 5; cmp eax, 0x1234; jb/jbe +6;
    # mov eax, 1; ret; xor eax, eax; ret
    head = LOAD_ARG + bytes.fromhex("83c005") + bytes.fromhex("3d34120000")
    tail = bytes.fromhex("06") + bytes.fromhex("b801000000c3") + bytes.fromhex("31c0c3")
    orig, recomp = head + b"\x72" + tail, head + b"\x76" + tail

    recorder = AnalysisRecorder()
    assert not verify_effective_match(
        list(disasm_detail(orig, ORIG_FUNC)),
        list(disasm_detail(recomp, RECOMP_FUNC)),
        recorder=recorder,
    )
    difference = recorder.difference or recorder.candidate_difference
    assert difference is not None and difference.kind == "branch_condition"
    hints, _ = _solver_hints(ComparisonAnalysis.mismatch(difference))
    assert [hint.stack_args[0] for hint in hints] == [0x122F]

    assert _search(orig, recomp).witness is None  # the seeds miss it
    witness = _search(orig, recomp, hints=hints).witness
    assert witness is not None
    assert (witness.seed, witness.kind) == (HINT_SEED, "return_value")


def _field(offset: int, size: str = "word", register: str = "c") -> tuple:
    """A load of `[register + offset]` as memory was at entry."""
    return ("load", ("mem", "", (((("init", register)), 1),), offset, ()), size, 0)


def test_solver_assignments_only_become_inputs_when_the_witness_sets_them():
    base = RunInput.from_seed(1)
    argument = ("load", ("mem", "", ((("init", "sp"), 1),), 8, ()), "dword", 0)
    hint = input_from_assignment({("init", "c"): 7, argument: 9}, base)
    assert isinstance(hint, RunInput)
    assert hint.registers["ecx"] == 7 and hint.stack_args[1] == 9
    # this->field at entry: modelled memory, set byte by byte when its page
    # is mapped
    hint = input_from_assignment({("init", "c"): 0x20000000, _field(4): 0xBEEF}, base)
    assert isinstance(hint, RunInput)
    assert hint.memory == ((0x20000004, 0xEF), (0x20000005, 0xBE))
    # an argument after a store cannot be set; a symbol's address is ignored
    stored = ("load", argument[1], "dword", 3)
    assert input_from_assignment({stored: 1}, base) == Rejection(
        "uncontrollable_leaf", repr(stored)
    )
    symbol = input_from_assignment({("sym", ("entity", 0x5000, 0)): 1}, base)
    assert isinstance(symbol, RunInput) and symbol.registers == base.registers


def test_overlapping_solver_loads_must_agree_on_every_byte():
    base = RunInput.from_seed(1)
    this = {("init", "c"): 0x20000000}
    # a dword at +4 and a word at +5 share bytes 5 and 6
    agree = input_from_assignment(
        {**this, _field(4, "dword"): 0x11223344, _field(5): 0x2233}, base
    )
    assert isinstance(agree, RunInput) and len(agree.memory) == 4
    disagree = input_from_assignment(
        {**this, _field(4, "dword"): 0x11223344, _field(5): 0x9999}, base
    )
    assert isinstance(disagree, Rejection)
    assert disagree.reason == "conflicting_memory"
    # memory an input cannot preset: the stack is mapped from the start
    stack = input_from_assignment({("init", "c"): STACK_BASE, _field(4): 1}, base)
    assert isinstance(stack, Rejection) and stack.reason == "invalid_destination"


def test_a_preset_dword_across_a_page_boundary_is_written_whole():
    # mov eax, [ecx + 0xffe]; ret
    code = bytes.fromhex("8b81fe0f0000") + RET
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    hint = input_from_assignment(
        {("init", "c"): 0x20000000, _field(0xFFE, "dword"): 0x11223344},
        RunInput.from_seed(1),
    )
    assert isinstance(hint, RunInput)
    trace = machine.run(range(ORIG_FUNC, ORIG_FUNC + len(code)), hint)
    assert trace.end == "return" and trace.eax == 0x11223344


# mov eax, [ecx + 8]; add eax, 5; cmp eax, 0x1234; jb/jbe +6;
# mov eax, 1; ret; xor eax, eax; ret
_FIELD_COMPARE = bytes.fromhex("8b410883c0053d34120000")
_FIELD_TAIL = bytes.fromhex("06b801000000c331c0c3")
FIELD_BELOW = _FIELD_COMPARE + b"\x72" + _FIELD_TAIL
FIELD_BELOW_OR_EQUAL = _FIELD_COMPARE + b"\x76" + _FIELD_TAIL


def test_a_solver_input_can_set_this_fields():
    """`this->count + 5 < 0x1234` against `<=`: differs only when the field
    holds 0x122f, which the input sets in modelled memory."""
    orig, recomp = FIELD_BELOW, FIELD_BELOW_OR_EQUAL
    recorder = AnalysisRecorder()
    verify_effective_match(
        list(disasm_detail(orig, ORIG_FUNC)),
        list(disasm_detail(recomp, RECOMP_FUNC)),
        recorder=recorder,
    )
    difference = recorder.difference or recorder.candidate_difference
    assert difference is not None
    hints, _ = _solver_hints(ComparisonAnalysis.mismatch(difference))
    assert hints and hints[0].memory
    assert _search(orig, recomp).witness is None
    witness = _search(orig, recomp, hints=hints).witness
    assert witness is not None and witness.seed == HINT_SEED


_REPLAY_IN_A_FRESH_PROCESS = """
import json, sys
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.diagnosis import RefutationWitness
from reccmp.compare.witness import replay
from tests.test_witness import FIELD_BELOW, FIELD_BELOW_OR_EQUAL, _translator

def no_solver(*_args, **_kwargs):
    raise AssertionError("replay ran the solver")

bitvector.distinguishing_assignment = no_solver
bitvector._query = no_solver
witness = RefutationWitness.from_json(json.loads(sys.stdin.read()))
result = replay(_translator(FIELD_BELOW, FIELD_BELOW_OR_EQUAL), witness)
print(json.dumps([result.reproduced, list(result.problems)]))
"""


def test_a_solver_witness_replays_in_a_fresh_process():
    """The record is the whole witness: another process reproduces the
    divergence from it, with no solver and no search."""
    recorder = AnalysisRecorder()
    verify_effective_match(
        list(disasm_detail(FIELD_BELOW, ORIG_FUNC)),
        list(disasm_detail(FIELD_BELOW_OR_EQUAL, RECOMP_FUNC)),
        recorder=recorder,
    )
    difference = recorder.difference or recorder.candidate_difference
    assert difference is not None
    hints, _ = _solver_hints(ComparisonAnalysis.mismatch(difference))
    witness = _search(FIELD_BELOW, FIELD_BELOW_OR_EQUAL, hints=hints).witness
    assert witness is not None and witness.replay is not None
    assert witness.seed == HINT_SEED and witness.replay.input.memory
    serialized = json.dumps(dataclasses.asdict(witness))
    assert RefutationWitness.from_json(json.loads(serialized)) == witness
    refuted = ComparisonAnalysis.mismatch(difference).with_witness(witness)
    in_report = json.loads(json.dumps(analysis_json(refuted)))
    assert parse_analysis(in_report).witness == witness
    process = subprocess.run(
        [sys.executable, "-c", _REPLAY_IN_A_FRESH_PROCESS],
        input=serialized,
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert json.loads(process.stdout) == [True, []]


def test_a_replay_says_why_it_does_not_reproduce():
    witness = _search(
        LOAD_ARG + bytes.fromhex("83c001") + RET,
        LOAD_ARG + bytes.fromhex("83c002") + RET,
    ).witness
    assert witness is not None and witness.replay is not None
    same = _translator(
        LOAD_ARG + bytes.fromhex("83c001") + RET,
        LOAD_ARG + bytes.fromhex("83c001") + RET,
    )
    result = replay(same, witness)
    assert not result.reproduced and result.problems == ("agreed",)
    stale = dataclasses.replace(
        witness, replay=dataclasses.replace(witness.replay, model=WITNESS_MODEL + 1)
    )
    assert replay(same, stale).problems == ("model",)


def _difference(orig: bytes, recomp: bytes):
    recorder = AnalysisRecorder()
    verify_effective_match(
        list(disasm_detail(orig, ORIG_FUNC)),
        list(disasm_detail(recomp, RECOMP_FUNC)),
        recorder=recorder,
    )
    difference = recorder.difference or recorder.candidate_difference
    assert difference is not None
    return difference


def test_a_value_difference_says_what_the_solver_found():
    difference = _difference(FIELD_BELOW, FIELD_BELOW_OR_EQUAL)
    assert difference.solver is not None
    assert difference.solver["result"] == "differs"
    assert isinstance(difference.solver["rlimit"], int)
    in_report = json.loads(
        json.dumps(analysis_json(ComparisonAnalysis.mismatch(difference)))
    )
    assert parse_analysis(in_report).difference.solver == difference.solver  # type: ignore[union-attr]


def test_the_fate_of_a_solver_hint_is_recorded():
    """Why a solver answer led to no run: the assignment constrains a leaf
    no input sets (memory after a store), or a term could not be lowered;
    and what a run from it did."""
    difference = _difference(FIELD_BELOW, FIELD_BELOW_OR_EQUAL)
    analysis = ComparisonAnalysis.mismatch(difference)
    hints, failure = _solver_hints(analysis)
    assert hints and failure is None
    stored = ("load", ("mem", "", (((("init", "c")), 1),), 8, ()), "dword", 3)
    rejected = dataclasses.replace(difference, values=(stored, ("imm", 1), 32, "value"))
    assert _solver_hints(ComparisonAnalysis.mismatch(rejected)) == (
        [],
        "rejected: uncontrollable_leaf",
    )
    unsupported = dataclasses.replace(
        difference, values=(("imm", 1), ("imm", 2), None, "value")
    )
    assert _solver_hints(ComparisonAnalysis.mismatch(unsupported)) == (
        [],
        "solver unsupported: constant operands",
    )
    located = ComparisonAnalysis.mismatch(
        dataclasses.replace(
            difference,
            orig=dataclasses.replace(difference.orig, address=ORIG_FUNC + 11),
            recomp=dataclasses.replace(difference.recomp, address=RECOMP_FUNC + 11),
        )
    )
    both = (frozenset({ORIG_FUNC + 11}), frozenset({RECOMP_FUNC + 11}))
    ran = SearchResult(hint_runs=[("call_structure", *both)])
    assert _hint_fate(ran, located) == "run call_structure"
    missed = SearchResult(hint_runs=[("agreed", frozenset(), frozenset())])
    assert _hint_fate(missed, located) == "run agreed, did not reach the difference"


def test_the_start_of_a_paired_entity_needs_no_extent():
    """A call target or pointer equal to a paired entity's start is that
    entity; only addresses past the start need its extent."""
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, ORIG_CALLEE, type=EntityType.FUNCTION)
        batch.set(ImageId.RECOMP, RECOMP_CALLEE, type=EntityType.FUNCTION, size=8)
        batch.match(ORIG_CALLEE, RECOMP_CALLEE)
    translator = Translator(
        db,
        SideMachine(_image(ORIG_FUNC, RET, ORIG_TABLE)),  # type: ignore[arg-type]
        SideMachine(_image(RECOMP_FUNC, RET, RECOMP_TABLE)),  # type: ignore[arg-type]
    )
    start = ("entity", ORIG_CALLEE, 0)
    assert translator.classify(ImageId.ORIG, ORIG_CALLEE) == (start, "entity")
    assert translator.identity(ImageId.RECOMP, RECOMP_CALLEE) == start
    assert translator.classify(ImageId.ORIG, ORIG_CALLEE + 4) == (
        UNRESOLVED,
        "unknown_extent",
    )


def test_a_paired_object_borrows_only_an_exactly_fitting_extent():
    """Without an original size, the recompiled size is used only when it is
    exactly the gap to the next known original entity."""

    def extent(orig_gap: int, *, matched: bool = True) -> int | None:
        entity = ReccmpEntity(
            DATA,
            DATA + 0x100 if matched else None,
            {"type": EntityType.DATA, "recomp_size": 8, "orig_max_size": orig_gap},
        )
        comparator = SimpleNamespace(_witness_extents={})
        # pylint: disable-next=protected-access
        return RefutationMixin._witness_extent(comparator, ImageId.ORIG, entity)  # type: ignore[arg-type]

    assert extent(8) == EntityExtent(8, recorded=False)
    assert extent(12) is None  # room for an unknown object after it
    assert extent(4) is None  # does not fit
    assert extent(8, matched=False) is None


def _store_to_table(offset: int, value: int, table: int) -> bytes:
    # mov dword ptr [table + offset], imm32; ret
    return b"\xc7\x05" + _abs32(table + offset) + _abs32(value) + RET


def test_a_difference_through_an_estimated_extent_is_not_a_witness():
    """The original table has no recorded size, only an estimate: its bytes
    past the start may be padding or an unrecorded object."""

    def estimated(side: ImageId, entity: ReccmpEntity) -> EntityExtent | None:
        if side == ImageId.ORIG and entity.orig_addr == ORIG_TABLE:
            return EntityExtent(len(TABLE), recorded=False)
        size = entity.size(side)
        return EntityExtent(size) if size is not None else None

    result = _search(
        _store_to_table(4, 5, ORIG_TABLE),
        _store_to_table(4, 6, RECOMP_TABLE),
        orig_table_size=None,
        extent=estimated,
    )
    assert result.witness is None
    assert result.skipped["estimated_extent"] == result.runs
    assert result.skipped_details["estimated_extent"]["kind"] == "memory_value"
    # With the size recorded, the same difference refutes.
    result = _search(
        _store_to_table(4, 5, ORIG_TABLE), _store_to_table(4, 6, RECOMP_TABLE)
    )
    assert result.witness is not None


def test_a_location_one_side_cannot_identify_is_not_compared():
    """Only the recompiled store identifies <table+4>; without an original
    size, nothing shows the original's table+4 is the same object."""
    result = _search(
        _store_to_table(4, 5, ORIG_TABLE),
        _store_to_table(4, 6, RECOMP_TABLE),
        orig_table_size=None,
    )
    assert result.witness is None


def test_an_index_past_the_table_into_another_object_is_not_a_witness():
    """Index 17 of a 16-dword table stores into whatever each binary laid
    out after the table: another paired object on each side, but not the
    same one."""

    # mov eax, [esp+4]; mov dword ptr [eax*4 + table], 1; ret
    def body(table: int) -> bytes:
        return LOAD_ARG + b"\xc7\x04\x85" + _abs32(table) + _abs32(1) + RET

    after = len(TABLE)
    objects = (
        (ORIG_TABLE + after, RECOMP_TABLE + 0x200, 0x40),
        (ORIG_TABLE + 0x200, RECOMP_TABLE + after, 0x40),
    )
    base = RunInput.from_seed(HINT_SEED)
    past = dataclasses.replace(base, stack_args=(17, *base.stack_args[1:]))
    result = _search(
        body(ORIG_TABLE), body(RECOMP_TABLE), hints=[past], objects=objects
    )
    assert result.witness is None
    assert result.skipped["out_of_object_access"] >= 1
    detail = result.skipped_details["out_of_object_access"]
    assert detail["access"] == "write" and detail["meant"] == f"<{ORIG_TABLE:#x}+0x0>"


def test_an_index_that_wraps_out_of_the_image_is_not_a_witness():
    """A huge index wraps `table + 8 * index` around into modelled memory,
    a different spot on each side since the tables differ."""

    # mov eax, [esp+4]; mov eax, [eax*8 + table]; ret
    def body(table: int) -> bytes:
        return LOAD_ARG + b"\x8b\x04\xc5" + _abs32(table) + RET

    index = ((HEAP_BASE - ORIG_TABLE) % (1 << 32)) // 8
    base = RunInput.from_seed(HINT_SEED)
    wraps = dataclasses.replace(base, stack_args=(index, *base.stack_args[1:]))
    result = _search(body(ORIG_TABLE), body(RECOMP_TABLE), hints=[wraps])
    assert result.witness is None
    assert result.skipped["out_of_object_access"] >= 1


def _load_table(offset: int, table: int) -> bytes:
    # mov eax, dword ptr [table + offset]; ret
    return b"\xa1" + _abs32(table + offset) + RET


def test_unpaired_read_only_constants_are_read_as_what_they_hold():
    """No pair identifies the table, but its bytes are the same wherever
    each binary put it: the same element agrees, another one refutes."""
    same = _search(
        _load_table(4, ORIG_TABLE), _load_table(4, RECOMP_TABLE), pair_table=False
    )
    assert same.witness is None and same.agreeing_seeds > 0
    other = _search(
        _load_table(4, ORIG_TABLE), _load_table(8, RECOMP_TABLE), pair_table=False
    )
    assert other.witness is not None and other.witness.kind == "return_value"
    # Relocated bytes are pointers, whose values depend on the layout.
    relocated = _search(
        _load_table(4, ORIG_TABLE),
        _load_table(8, RECOMP_TABLE),
        pair_table=False,
        relocated_table=True,
    )
    assert relocated.witness is None
    assert relocated.skipped["unknown_image_read"] == relocated.runs


def test_an_index_past_an_unpaired_table_is_not_a_constant_read():
    # mov eax, [esp+4]; mov eax, [eax*4 + table]; ret
    def body(table: int) -> bytes:
        return LOAD_ARG + b"\x8b\x04\x85" + _abs32(table) + RET

    after = len(TABLE)
    objects = (
        (ORIG_TABLE + after, RECOMP_TABLE + 0x200, 0x40),
        (ORIG_TABLE + 0x200, RECOMP_TABLE + after, 0x40),
    )
    base = RunInput.from_seed(HINT_SEED)
    past = dataclasses.replace(base, stack_args=(17, *base.stack_args[1:]))
    result = _search(
        body(ORIG_TABLE),
        body(RECOMP_TABLE),
        hints=[past],
        objects=objects,
        pair_table=False,
    )
    assert result.witness is None
    assert result.skipped["out_of_object_access"] >= 1


def test_a_run_does_not_inherit_the_previous_runs_fpu_state():
    """A reused machine and a new one (as in replay) run alike: the x87
    stack a run leaves behind does not reach the next run."""
    # fld1; fld1; ret  /  fnstsw ax; ret
    push = bytes.fromhex("d9e8d9e8") + RET
    status = bytes.fromhex("dfe0") + RET
    code = push.ljust(0x10, b"\x90") + status
    machine = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    fresh = SideMachine(_image(ORIG_FUNC, code, ORIG_TABLE))  # type: ignore[arg-type]
    run_input = RunInput.from_seed(1)
    machine.run(range(ORIG_FUNC, ORIG_FUNC + len(push)), run_input)
    status_range = range(ORIG_FUNC + 0x10, ORIG_FUNC + 0x10 + len(status))
    after = machine.run(status_range, run_input)
    alone = fresh.run(status_range, run_input)
    assert after.end == alone.end == "return"
    assert after.eax & 0xFFFF == alone.eax & 0xFFFF
