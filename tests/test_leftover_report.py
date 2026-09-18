"""Leftover architectural-review items after the P0 proof fixes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reccmp.compare.asm.ir import (
    AsmRole,
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
    compute_extent_closed,
    resolve_asm_stream,
)
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.db import EntityDb, FrozenEntityDbError
from reccmp.compare.diff import EntityCompareResult
from reccmp.compare.diagnosis import ComparisonAnalysis, ComparisonStatus
from reccmp.compare.source_capability import (
    load_source_index_for_target,
    resolve_source_index_path,
    source_index_abi_compatible,
)
from reccmp.compare.verification import admit_exact_analysis, admit_proof
from reccmp.source import (
    SourceAbi,
    SourceClass,
    SourceField,
    SourceIndex,
    SourceVariable,
)
from reccmp.types import ImageId


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
    assert admit_proof(exact, extent_closed=False).status == ComparisonStatus.INCONCLUSIVE
    assert (
        admit_proof(exact, coverage_incomplete=True).status
        == ComparisonStatus.INCONCLUSIVE
    )
    assert (
        admit_exact_analysis(
            displays_equal=True, topology_equal=True, extent_closed=False
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
    assert result.equivalence_level is not None
    assert result.diagnostic_normalizations
