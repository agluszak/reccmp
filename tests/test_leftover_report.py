"""Leftover architectural-review items after the P0 proof fixes."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from reccmp.compare.asm.effective import _extract_switch_tables
from reccmp.compare.asm.ir import (
    AsmRole,
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
    compute_extent_closed,
    rebind_local_identities,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.asm.replacement import create_name_lookup
from reccmp.compare.db import EntityDb, FrozenEntityDbError, ReccmpMatch
from reccmp.compare.diff import EntityCompareResult
from reccmp.compare.diagnosis import ComparisonAnalysis, ComparisonStatus
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.compare.report import (
    ReccmpStatusReport,
    deserialize_reccmp_report,
    serialize_reccmp_report,
)
from reccmp.compare.source_capability import (
    load_source_index_for_target,
    resolve_source_index_path,
    source_index_abi_compatible,
)
from reccmp.compare.verification import (
    admit_effective,
    admit_exact_analysis,
    admit_proof,
)
from reccmp.cvdump import Cvdump, CvdumpError
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.source import (
    SourceAbi,
    SourceClass,
    SourceField,
    SourceIndex,
    SourceVariable,
)
from reccmp.source.batch import _file_digest_factory
from reccmp.tools.find_inlines import _resolve_helper
from reccmp.types import EntityType, ImageId
from tests.raw_image import RawImage


def test_instruction_ids_survive_slice_and_reorder():
    blob = bytes.fromhex("B80100000083C001C3")  # mov eax,1; add eax,1; ret
    rows = ParseAsm().parse_asm(blob, 0x1000)
    stamped = tuple(replace(row, instruction_id=100 + i) for i, row in enumerate(rows))
    stream = resolve_asm_stream(stamped)
    assert stream.instruction_ids == (100, 101, 102)
    sliced = stream.slice(2)
    assert sliced.instruction_ids == (100, 101)
    reordered = stream.reorder([2, 0, 1])
    assert reordered.instruction_ids == (102, 100, 101)
    assert reordered.displays[0] == stream.displays[2]


def test_estimated_extent_without_terminal_is_open():
    blob = bytes.fromhex("B801000000B802000000")  # mov eax,1; mov eax,2
    excerpt = tuple(ParseAsm().parse_asm(blob, 0x1000))
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=len(blob),
            extent_kind=ExtentKind.ESTIMATED,
        )
        is False
    )


def test_known_extent_with_plain_fallthrough_is_open():
    blob = bytes.fromhex("B801000000B901000000")  # mov eax,1; mov ecx,1
    excerpt = tuple(ParseAsm().parse_asm(blob, 0x1000))
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=len(blob),
            extent_kind=ExtentKind.KNOWN,
        )
        is False
    )


def test_known_extent_ending_in_ret_is_closed():
    blob = bytes.fromhex("B801000000C3")  # mov eax,1; ret
    sanitizer = ParseAsm()
    excerpt = tuple(
        replace(row, instruction_id=i)
        for i, row in enumerate(sanitizer.parse_asm(blob, 0x1000))
    )
    image = FunctionImage(
        start_addr=0x1000,
        extent=len(blob),
        extent_kind=ExtentKind.KNOWN,
        excerpt=excerpt,
        extent_closed=compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=len(blob),
            extent_kind=ExtentKind.KNOWN,
        ),
        raw=blob,
    )
    assert image.extent_closed is True


def _ret_at(addr: int, iid: int) -> DecodedInstruction:
    row = ParseAsm().parse_asm(b"\xc3", addr)[0]
    return replace(row, instruction_id=iid)


def _table_entry(addr: int, display: str, iid: int) -> DecodedInstruction:
    return replace(
        ParseAsm().parse_asm(b"\xc3", addr)[0],
        address=addr,
        size=4,
        mnemonic="",
        operands=(),
        display=display,
        role=AsmRole.JUMP_TABLE_ENTRY,
        instruction_id=iid,
    )


def test_jump_table_dispatch_closes_indirect_switch_extent():
    dispatch = DecodedInstruction(
        address=0x1000,
        size=2,
        mnemonic="jmp",
        prefix="",
        operands=(("mem", "dword", "", (("eax", 4),), 0, ()),),
        raw_operands=("dword ptr [eax*4]",),
        display="jmp dword ptr [eax*4]",
        role=AsmRole.CODE,
        is_jump=True,
        branch_target=None,
        control_flow_known=False,
        instruction_id=0,
    )
    case0 = _ret_at(0x1010, 1)
    case1 = _ret_at(0x1020, 2)
    table = JumpTable(
        address=0x1004,
        entries=((0x1004, 0x1010), (0x1008, 0x1020)),
        dispatch_address=0x1000,
    )
    excerpt = (dispatch, case0, case1)
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=0x21,
            jump_tables=(table,),
            extent_kind=ExtentKind.KNOWN,
        )
        is True
    )
    other_table = JumpTable(
        address=0x2000,
        entries=((0x2000, 0x1010),),
        dispatch_address=0x9999,
    )
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=0x21,
            jump_tables=(other_table,),
            extent_kind=ExtentKind.KNOWN,
        )
        is False
    )
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=0x21,
            extent_kind=ExtentKind.KNOWN,
        )
        is False
    )


def test_admit_proof_refuses_open_extent_and_incomplete_coverage():
    exact = ComparisonAnalysis.exact()
    assert (
        admit_proof(exact, coverage_incomplete=False, extent_closed=False).status
        == ComparisonStatus.INCONCLUSIVE
    )
    assert (
        admit_proof(exact, coverage_incomplete=True, extent_closed=True).status
        == ComparisonStatus.INCONCLUSIVE
    )
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=True,
            extent_closed=False,
        )
        is None
    )


def test_entity_db_freeze_rejects_later_pairing():
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x10, name="orig", size=4)
        batch.set(ImageId.RECOMP, 0x20, name="recomp", size=4)
        batch.match(0x10, 0x20)
    db.freeze()
    with pytest.raises(FrozenEntityDbError):
        with db.batch() as batch:
            batch.set(ImageId.ORIG, 0x30, name="later")


def test_source_index_is_not_auto_discovered(tmp_path: Path):
    planted = tmp_path / "reccmp-source" / "source-index.json"
    planted.parent.mkdir()
    planted.write_text("{}", encoding="utf-8")
    target = MagicMock()
    target.target_id = "GAME"
    target.recompiled_path = tmp_path / "game.exe"
    target.source_paths = ()
    assert resolve_source_index_path(target) is None
    assert load_source_index_for_target(target) is None


def test_source_index_rejects_incompatible_abi(tmp_path: Path):
    document = SourceIndex(
        declarations=(),
        markers=(),
        classes=(),
        abi=SourceAbi(
            target_triple="x86_64-pc-windows-msvc",
            pointer_width=8,
            ms_abi=True,
        ),
    ).to_dict()
    path = tmp_path / "source-index.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    target = MagicMock()
    target.target_id = "GAME"
    target.recompiled_path = tmp_path / "game.exe"
    target.source_paths = ()
    assert (
        source_index_abi_compatible(SourceAbi("x86_64-pc-windows-msvc", 8, True))
        is False
    )
    assert load_source_index_for_target(target, explicit=path) is None


def test_source_index_scopes_variable_only_targets(tmp_path: Path):
    document = SourceIndex(
        declarations=(),
        markers=(),
        classes=(),
        abi=SourceAbi("i386-pc-windows-msvc", 4, True),
        variables=(
            SourceVariable(
                semantic_id="gOnly",
                qualified_name="gOnly",
                type="int",
                linkage="external",
                storage_class="none",
                definition_kind="definition",
                source_file="a.cpp",
                line=1,
                end_line=1,
                target="OTHER",
            ),
        ),
    ).to_dict()
    path = tmp_path / "source-index.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    target = MagicMock()
    target.target_id = "GAME"
    target.recompiled_path = tmp_path / "game.exe"
    target.source_paths = ()
    assert load_source_index_for_target(target, explicit=path) is None


def test_array_field_resolves_later_elements():
    index = SourceIndex(
        declarations=(),
        markers=(),
        classes=(
            SourceClass(
                semantic_id="record:Child",
                qualified_name="Child",
                bases=(),
                fields=(
                    SourceField(
                        name="value",
                        type="int",
                        source_file="a.h",
                        line=2,
                        offset=0,
                        size=4,
                        storage_kind="scalar",
                    ),
                ),
                virtual_declarations=(),
                source_file="a.h",
                line=1,
                end_line=3,
                size=4,
                alignment=4,
                layout_trusted=True,
            ),
            SourceClass(
                semantic_id="record:Parent",
                qualified_name="Parent",
                bases=(),
                fields=(
                    SourceField(
                        name="children",
                        type="Child [4]",
                        source_file="a.h",
                        line=6,
                        offset=0,
                        size=16,
                        record_semantic_id="record:Child",
                        storage_kind="array",
                        array_element_type="Child",
                        array_stride=4,
                        array_count=4,
                    ),
                ),
                virtual_declarations=(),
                source_file="a.h",
                line=5,
                end_line=7,
                size=16,
                alignment=4,
                layout_trusted=True,
            ),
        ),
    )
    first = index.resolve_field("Parent", 0)
    later = index.resolve_field("Parent", 8)
    assert first is not None
    assert first.path == ("children[0]", "value")
    assert later is not None
    assert later.path == ("children[2]", "value")
    assert later.leaf.name == "value"


def test_entity_compare_result_derives_normalizations_at_construction():
    result = EntityCompareResult(
        analysis=ComparisonAnalysis.effective(("register_allocation",))
    )
    assert result.analysis.status == ComparisonStatus.EFFECTIVE
    assert result.diagnostic_normalizations


def test_first_class_jump_table_requires_scale4_indexed_jmp():
    dispatch = DecodedInstruction(
        address=0x1000,
        size=3,
        mnemonic="jmp",
        prefix="",
        operands=(("mem", "dword", "", (("eax", 1),), 0x1004, ()),),
        raw_operands=("dword ptr [eax+0x1004]",),
        display="jmp dword ptr [eax+0x1004]",
        role=AsmRole.CODE,
        is_jump=True,
        instruction_id=0,
    )
    excerpt = (
        dispatch,
        _table_entry(0x1004, "start + 0x10", 3),
        _table_entry(0x1008, "start + 0x20", 4),
        _ret_at(0x1010, 1),
        _ret_at(0x1020, 2),
    )
    table = JumpTable(
        address=0x1004,
        entries=((0x1004, 0x1010), (0x1008, 0x1020)),
        dispatch_address=0x1000,
        scale=4,
        entry_width=4,
        index_register="eax",
    )
    stream = resolve_asm_stream(excerpt, jump_tables=(table,))
    extracted = _extract_switch_tables(
        stream,
        ["jmp", "data", "data", "ret", "ret"],
        [0x1000, 0x1004, 0x1008, 0x1010, 0x1020],
    )
    assert extracted is not None
    dests, _owned = extracted
    assert dests == {}


def test_first_class_jump_table_accepts_scale4_indexed_jmp():
    dispatch = DecodedInstruction(
        address=0x1000,
        size=3,
        mnemonic="jmp",
        prefix="",
        operands=(("mem", "dword", "", (("eax", 4),), 0x1004, ()),),
        raw_operands=("dword ptr [eax*4+0x1004]",),
        display="jmp dword ptr [eax*4+0x1004]",
        role=AsmRole.CODE,
        is_jump=True,
        instruction_id=0,
    )
    excerpt = (
        dispatch,
        _table_entry(0x1004, "start + 0x10", 3),
        _table_entry(0x1008, "start + 0x20", 4),
        _ret_at(0x1010, 1),
        _ret_at(0x1020, 2),
    )
    table = JumpTable(
        address=0x1004,
        entries=((0x1004, 0x1010), (0x1008, 0x1020)),
        dispatch_address=0x1000,
        scale=4,
        entry_width=4,
        index_register="eax",
    )
    stream = resolve_asm_stream(excerpt, jump_tables=(table,))
    extracted = _extract_switch_tables(
        stream,
        ["jmp", "data", "data", "ret", "ret"],
        [0x1000, 0x1004, 0x1008, 0x1010, 0x1020],
    )
    assert extracted is not None
    dests, owned = extracted
    assert dests[0] == [3, 4]
    assert owned == {1, 2}


def test_gate_mints_verification_result_not_strategy():
    minted = admit_effective(
        {"register_allocation"},
        coverage_incomplete=False,
        extent_closed=True,
    )
    assert minted is not None
    assert minted.proof_kind == ComparisonStatus.EFFECTIVE
    assert minted.analysis.status == ComparisonStatus.EFFECTIVE
    assert "extent_closed" in minted.assumptions
    assert (
        admit_effective(
            {"register_allocation"},
            coverage_incomplete=False,
            extent_closed=False,
        )
        is None
    )


def test_name_lookup_invalidates_when_entity_db_generation_changes():
    db = EntityDb()
    lookup = create_name_lookup(
        db, ImageId.ORIG, lambda _addr: None, lambda _type, _offset: ""
    )
    assert lookup(0x100) is None
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x100, type=EntityType.FUNCTION, name="foo")
    assert db.generation > 0
    assert lookup(0x100) is not None


def test_alias_equivalent_reuses_proved_results_not_active_frames():
    comparator = object.__new__(FunctionComparator)
    calls: list[tuple[int, int]] = []

    def body(orig_addr, recomp_addr, _size, _depth, active, proved):
        calls.append((orig_addr, recomp_addr))
        if orig_addr == 1:
            first = comparator.raw_pair_alias_equivalent(
                2, 20, 4, _depth=1, _active=active, _proved=proved
            )
            again = comparator.raw_pair_alias_equivalent(
                2, 20, 4, _depth=1, _active=active, _proved=proved
            )
            return first and again
        return True

    comparator._raw_pair_alias_equivalent_body = body  # type: ignore[method-assign]
    assert comparator.raw_pair_alias_equivalent(1, 10, 4)
    assert calls == [(1, 10), (2, 20)]


def test_alias_cycle_is_not_treated_as_proved():
    comparator = object.__new__(FunctionComparator)

    def body(orig_addr, recomp_addr, size, depth, active, proved):
        return comparator.raw_pair_alias_equivalent(
            orig_addr,
            recomp_addr,
            size,
            _depth=depth + 1,
            _active=active,
            _proved=proved,
        )

    comparator._raw_pair_alias_equivalent_body = body  # type: ignore[method-assign]
    assert not comparator.raw_pair_alias_equivalent(1, 10, 4)


def test_include_diff_false_still_computes_stack_diagnostics():
    orig = bytes.fromhex("558BEC8945FC5DC3")  # mov [ebp-4]
    recomp = bytes.fromhex("558BEC8945F85DC3")  # mov [ebp-8]
    orig_bin = Mock(spec=[])
    orig_bin.read = Mock(return_value=orig)
    orig_bin.imagebase = 0
    orig_bin.is_relocated_addr = Mock(return_value=False)
    orig_bin.is_debug = Mock(return_value=False)
    recomp_bin = Mock(spec=[])
    recomp_bin.read = Mock(return_value=recomp)
    recomp_bin.imagebase = 0
    recomp_bin.is_relocated_addr = Mock(return_value=False)
    recomp_bin.is_debug = Mock(return_value=False)
    comparator = FunctionComparator(
        EntityDb(),
        LinesDb(),
        orig_bin,
        recomp_bin,
        Mock(spec=ReccmpReportProtocol),
        CvdumpTypesParser(),
    )
    result = comparator.compare_function(
        ReccmpMatch(
            0x200,
            0x400,
            {
                "type": EntityType.FUNCTION,
                "name": "stack",
                "orig_size": len(orig),
                "recomp_size": len(recomp),
            },
        ),
        include_diff=False,
    )
    assert not result.diff.codes
    assert result.stack_permutation


def test_pointer_array_does_not_descend_into_pointee():
    index = SourceIndex(
        declarations=(),
        markers=(),
        classes=(
            SourceClass(
                semantic_id="record:Child",
                qualified_name="Child",
                bases=(),
                fields=(
                    SourceField(
                        name="value",
                        type="int",
                        source_file="a.h",
                        line=2,
                        offset=0,
                        size=4,
                        storage_kind="scalar",
                    ),
                ),
                virtual_declarations=(),
                source_file="a.h",
                line=1,
                end_line=3,
                size=4,
                alignment=4,
                layout_trusted=True,
            ),
            SourceClass(
                semantic_id="record:Parent",
                qualified_name="Parent",
                bases=(),
                fields=(
                    SourceField(
                        name="children",
                        type="Child *[4]",
                        source_file="a.h",
                        line=6,
                        offset=0,
                        size=16,
                        record_semantic_id="record:Child",
                        storage_kind="array",
                        array_element_type="Child *",
                        array_stride=4,
                        array_count=4,
                        array_element_kind="pointer",
                    ),
                ),
                virtual_declarations=(),
                source_file="a.h",
                line=5,
                end_line=7,
                size=16,
                alignment=4,
                layout_trusted=True,
            ),
        ),
    )
    later = index.resolve_field("Parent", 8)
    assert later is not None
    assert later.path == ("children[2]",)
    assert later.leaf.name == "children"
    assert later.relative_offset == 0


def test_scalar_array_uses_element_stride():
    index = SourceIndex(
        declarations=(),
        markers=(),
        classes=(
            SourceClass(
                semantic_id="record:Parent",
                qualified_name="Parent",
                bases=(),
                fields=(
                    SourceField(
                        name="values",
                        type="int [4]",
                        source_file="a.h",
                        line=2,
                        offset=0,
                        size=16,
                        storage_kind="array",
                        array_element_type="int",
                        array_stride=4,
                        array_count=4,
                        array_element_kind="scalar",
                    ),
                ),
                virtual_declarations=(),
                source_file="a.h",
                line=1,
                end_line=3,
                size=16,
                alignment=4,
                layout_trusted=True,
            ),
        ),
    )
    later = index.resolve_field("Parent", 8)
    assert later is not None
    assert later.path == ("values[2]",)
    assert later.leaf.name == "values"


def test_rebind_local_identities_uses_instruction_and_table_ids():
    lea = DecodedInstruction(
        address=0x1000,
        size=5,
        mnemonic="lea",
        prefix="",
        operands=(
            "eax",
            Reference("<OFFSET>", ("local", 8)),
        ),
        raw_operands=("eax", "<OFFSET>"),
        display="lea eax, <OFFSET>",
        instruction_id=0,
    )
    ret = _ret_at(0x1008, 1)
    rebound = rebind_local_identities(
        (lea, ret),
        start_addr=0x1000,
        extent=9,
        image_id="orig",
    )
    assert rebound[0].operands[1].identity == ("local_insn", 1)

    data_ref = replace(
        lea,
        operands=("eax", Reference("<OFFSET>", ("local", 4))),
    )
    rebound_data = rebind_local_identities(
        (data_ref, ret),
        start_addr=0x1000,
        extent=16,
        image_id="orig",
    )
    assert rebound_data[0].operands[1].identity == ("unresolved", "orig", 0x1004)

    table = JumpTable(address=0x1004, entries=((0x1004, 0x1008),))
    rebound_table = rebind_local_identities(
        (data_ref, ret),
        start_addr=0x1000,
        extent=16,
        jump_tables=(table,),
        image_id="orig",
    )
    assert rebound_table[0].operands[1].identity == ("table", 0)


def test_find_inlines_resolves_helper_via_public_get_match():
    match = ReccmpMatch(
        0x100,
        0x200,
        {"type": EntityType.FUNCTION, "name": "helper"},
    )
    compare = MagicMock()
    compare.get_match.return_value = match
    assert _resolve_helper(compare, "0x100") is match
    compare.get_match.assert_called_once_with(0x100)
    compare.db.get_one_match.assert_not_called()


def test_cvdump_run_raises_on_nonzero_status():
    proc = MagicMock()
    proc.stdout = io.BytesIO(b"")
    proc.wait.return_value = 2
    with patch.object(Cvdump, "cmd_line", return_value=["cvdump", "App.pdb"]):
        with patch("reccmp.cvdump.runner.subprocess.Popen") as popen:
            popen.return_value.__enter__.return_value = proc
            popen.return_value.__exit__.return_value = False
            with pytest.raises(CvdumpError):
                Cvdump("App.pdb").run()


def test_file_digest_factory_caches_sha256_bytes(tmp_path: Path):
    path = tmp_path / "a.cpp"
    path.write_bytes(b"abc")
    digest = _file_digest_factory()
    first = digest(path)
    path.write_bytes(b"changed")
    second = digest(path)
    assert first == second == hashlib.sha256(b"abc").digest()
    assert len(first) == 32


def test_report_identity_prefers_source_digest():
    same = ReccmpStatusReport(filename="a.exe", source_digest="aaa")
    other_name = ReccmpStatusReport(filename="b.exe", source_digest="aaa")
    mismatch = ReccmpStatusReport(filename="a.exe", source_digest="bbb")
    filename_only = ReccmpStatusReport(filename="a.exe")
    assert same.has_same_source(other_name)
    assert not same.has_same_source(mismatch)
    assert not same.has_same_source(filename_only)
    serialized = serialize_reccmp_report(same)
    restored = deserialize_reccmp_report(serialized)
    assert restored.source_digest == "aaa"


def test_to_report_records_orig_image_digest():
    from reccmp.compare import Compare
    from reccmp.cvdump import CvdumpAnalysis

    payload = b"orig-bytes"
    compare = Compare(
        RawImage.from_memory(payload),
        RawImage.from_memory(b"recomp"),
        Mock(spec=CvdumpAnalysis),
        "HELLO",
    )
    report = compare.to_report("hello.exe")
    assert report.source_digest == hashlib.sha256(payload).hexdigest()
    assert compare.orig_source_digest == report.source_digest
