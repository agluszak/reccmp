"""Tests for Clang layout lookup and memory_address enrichment."""

from __future__ import annotations

import dataclasses

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceSide,
)
from reccmp.compare.functions import FunctionComparator
from reccmp.source import (
    SourceBaseOffset,
    SourceClass,
    SourceCollector,
    SourceField,
    SourceIndex,
)
from reccmp.source import SourceAbi, keyed
from reccmp.source.index import (
    DeclarationKey,
    SourceComparison,
    SourceComparisonOperand,
    SourceDeclaration,
    SourceFunctionFacts,
    SourceMarker,
)


def _index_with_layout() -> SourceIndex:
    return SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Foo",
                    qualified_name="Foo",
                    bases=(),
                    fields=(
                        SourceField(
                            name="bar",
                            type="Bar",
                            source_file="foo.h",
                            line=10,
                            offset=0,
                            size=8,
                        ),
                        SourceField(
                            name="flag",
                            type="int",
                            source_file="foo.h",
                            line=11,
                            offset=8,
                            size=4,
                        ),
                        SourceField(
                            name="tail",
                            type="char",
                            source_file="foo.h",
                            line=12,
                            offset=12,
                            size=1,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="foo.h",
                    line=1,
                    end_line=20,
                    size=16,
                    alignment=4,
                    base_offsets=(),
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Bar",
                    qualified_name="Bar",
                    bases=(),
                    fields=(
                        SourceField(
                            name="baz",
                            type="int",
                            source_file="foo.h",
                            line=3,
                            offset=0,
                            size=4,
                        ),
                        SourceField(
                            name="qux",
                            type="int",
                            source_file="foo.h",
                            line=4,
                            offset=4,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="foo.h",
                    line=1,
                    end_line=5,
                    size=8,
                    alignment=4,
                    layout_trusted=True,
                ),
            )
        ),
    )


def test_field_at_requires_covering_size_and_searches_bases():
    index = _index_with_layout()
    at0 = index.field_at("Foo", 0)
    assert at0 is not None and at0.name == "baz"
    at7 = index.field_at("Foo", 7)
    assert at7 is not None and at7.name == "qux"
    at8 = index.field_at("Foo", 8)
    assert at8 is not None and at8.name == "flag"
    at12 = index.field_at("Foo", 12)
    assert at12 is not None and at12.name == "tail"
    # Past the end / padding: do not clamp to the last field.
    assert index.field_at("Foo", 100) is None
    assert index.field_at("Foo", 13) is None
    assert index.field_at("Missing", 0) is None


def test_field_path_at_descends_into_nested_layout():
    index = _index_with_layout()
    assert index.field_path_at("Foo", 4) == "bar.qux"
    assert index.field_path_at("Foo", 8) == "flag"


def test_resolve_field_absolute_offsets_for_nested_and_bases():
    index = _index_with_layout()
    nested = index.resolve_field("Foo", 4)
    assert nested is not None
    assert nested.root_class == "Foo"
    assert nested.path == ("bar", "qux")
    assert nested.leaf.name == "qux"
    assert nested.absolute_offset == 4
    assert nested.relative_offset == 0
    assert nested.base_chain == ()

    derived_index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Base",
                    qualified_name="Base",
                    bases=(),
                    fields=(
                        SourceField(
                            name="base_x",
                            type="int",
                            source_file="a.h",
                            line=1,
                            offset=0,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="a.h",
                    line=1,
                    end_line=2,
                    size=4,
                    alignment=4,
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Derived",
                    qualified_name="Derived",
                    bases=("Base",),
                    fields=(
                        SourceField(
                            name="derived_y",
                            type="int",
                            source_file="a.h",
                            line=5,
                            offset=4,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="a.h",
                    line=3,
                    end_line=6,
                    size=8,
                    alignment=4,
                    base_offsets=(SourceBaseOffset(name="Base", offset=0),),
                    layout_trusted=True,
                ),
            )
        ),
    )
    base = derived_index.resolve_field("Derived", 0)
    assert base is not None
    assert base.leaf.name == "base_x"
    assert base.path == ("Base", "base_x")
    assert base.absolute_offset == 0
    assert base.relative_offset == 0
    assert base.base_chain == ("Base",)
    derived = derived_index.resolve_field("Derived", 4)
    assert derived is not None
    assert derived.leaf.name == "derived_y"
    assert derived.absolute_offset == 4
    assert derived.base_chain == ()


def test_field_at_searches_base_subobjects():
    index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Base",
                    qualified_name="Base",
                    bases=(),
                    fields=(
                        SourceField(
                            name="base_x",
                            type="int",
                            source_file="a.h",
                            line=1,
                            offset=0,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="a.h",
                    line=1,
                    end_line=2,
                    size=4,
                    alignment=4,
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Derived",
                    qualified_name="Derived",
                    bases=("Base",),
                    fields=(
                        SourceField(
                            name="derived_y",
                            type="int",
                            source_file="a.h",
                            line=5,
                            offset=4,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="a.h",
                    line=3,
                    end_line=6,
                    size=8,
                    alignment=4,
                    base_offsets=(SourceBaseOffset(name="Base", offset=0),),
                    layout_trusted=True,
                ),
            )
        ),
    )
    field = index.field_at("Derived", 0)
    assert field is not None
    assert field.name == "base_x"
    assert index.field_path_at("Derived", 0) == "Base.base_x"
    derived = index.field_at("Derived", 4)
    assert derived is not None and derived.name == "derived_y"


def test_resolve_field_rejects_overlapping_bitfields():
    index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Bits",
                    qualified_name="Bits",
                    bases=(),
                    fields=(
                        SourceField(
                            name="a",
                            type="unsigned",
                            source_file="b.h",
                            line=2,
                            offset=0,
                            size=1,
                            bitfield_width=3,
                            bitfield_offset=0,
                        ),
                        SourceField(
                            name="b",
                            type="unsigned",
                            source_file="b.h",
                            line=3,
                            offset=0,
                            size=1,
                            bitfield_width=5,
                            bitfield_offset=3,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="b.h",
                    line=1,
                    end_line=4,
                    size=1,
                    alignment=1,
                    layout_trusted=True,
                ),
            )
        ),
    )
    assert index.resolve_field("Bits", 0) is None
    assert index.field_at("Bits", 0) is None
    assert index.field_path_at("Bits", 0) is None


def test_layout_conflict_marks_untrusted():
    collector = SourceCollector(Path("/repo"))
    collector.collect_record(
        {
            "record": "class",
            "semantic_id": "record:Foo",
            "qualified_name": "Foo",
            "bases": [],
            "fields": [
                {
                    "name": "x",
                    "type": "int",
                    "source_file": "a.h",
                    "line": 1,
                    "offset": 0,
                    "size": 4,
                }
            ],
            "virtual_declarations": [],
            "source_file": "a.h",
            "line": 1,
            "end_line": 2,
            "size": 4,
            "alignment": 4,
        },
        unit_id="a.cpp",
    )
    collector.collect_record(
        {
            "record": "class",
            "semantic_id": "record:Foo",
            "qualified_name": "Foo",
            "bases": [],
            "fields": [
                {
                    "name": "x",
                    "type": "int",
                    "source_file": "b.h",
                    "line": 1,
                    "offset": 0,
                    "size": 4,
                }
            ],
            "virtual_declarations": [],
            "source_file": "b.h",
            "line": 1,
            "end_line": 2,
            "size": 8,
            "alignment": 4,
        },
        unit_id="b.cpp",
    )
    namespace = collector.derive()
    assert len(namespace.classes) == 1
    assert list(namespace.classes.values())[0].layout_trusted is False
    assert any(c.record_kind == "class_layout" for c in namespace.conflicts)
    index = SourceIndex(
        declarations={},
        markers=(),
        classes=namespace.classes,
        conflicts=namespace.conflicts,
    )
    assert not index.has_layout("Foo")


def test_asserted_size_mismatch_untrusts_layout():
    collector = SourceCollector(Path("/repo"))
    collector.collect_record(
        {
            "record": "class",
            "semantic_id": "record:Foo",
            "qualified_name": "Foo",
            "bases": [],
            "fields": [
                {
                    "name": "x",
                    "type": "int",
                    "source_file": "a.h",
                    "line": 1,
                    "offset": 0,
                    "size": 4,
                }
            ],
            "virtual_declarations": [],
            "source_file": "a.h",
            "line": 1,
            "end_line": 2,
            "size": 4,
            "alignment": 4,
        },
        unit_id="a.cpp",
    )
    collector.collect_record(
        {
            "record": "size-assertion",
            "qualified_name": "Foo",
            "asserted_size": 16,
        },
        unit_id="a.cpp",
    )
    namespace = collector.derive()
    assert list(namespace.classes.values())[0].asserted_size == 16
    assert list(namespace.classes.values())[0].layout_trusted is False


def test_source_index_reader_fails_when_a_field_is_missing():
    document = SourceIndex(declarations={}, classes={}, markers=()).to_dict()
    del document["member_uses"]

    with pytest.raises(KeyError, match="member_uses"):
        SourceIndex.from_dict(document)


def test_unit_abi_round_trips_in_index_projection():

    index = SourceIndex(
        declarations={},
        classes={},
        markers=(),
        abi=SourceAbi(
            target_triple="i386-pc-windows-msvc",
            pointer_width=4,
            ms_abi=True,
        ),
    )
    revived = SourceIndex.from_dict(index.to_dict())
    assert revived.abi is not None
    assert revived.abi.target_triple == "i386-pc-windows-msvc"
    assert revived.abi.pointer_width == 4
    assert revived.abi.ms_abi is True


def test_record_semantic_id_preferred_for_nested_lookup():

    index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Bar",
                    qualified_name="Bar",
                    bases=(),
                    fields=(
                        SourceField(
                            name="x",
                            type="int",
                            source_file="b.h",
                            line=1,
                            offset=0,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="b.h",
                    line=1,
                    end_line=2,
                    size=4,
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Foo",
                    qualified_name="Foo",
                    bases=(),
                    fields=(
                        SourceField(
                            name="inner",
                            # Embedded record: peeling would leave ``Bar``; the
                            # semantic id must still win over the spelling.
                            type="volatile const struct WeirdSpelling",
                            source_file="f.h",
                            line=1,
                            offset=0,
                            size=4,
                            record_semantic_id="record:Bar",
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="f.h",
                    line=1,
                    end_line=2,
                    size=4,
                    layout_trusted=True,
                ),
            )
        ),
    )
    resolved = index.resolve_field("Foo", 0)
    assert resolved is not None
    assert resolved.path == ("inner", "x")
    assert resolved.leaf.name == "x"


def test_pointer_and_reference_fields_are_layout_leaves():
    """Pointer/reference storage must not descend into the pointee layout."""
    index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Node",
                    qualified_name="Node",
                    bases=(),
                    fields=(
                        SourceField(
                            name="next",
                            type="Node *",
                            source_file="n.h",
                            line=2,
                            offset=0,
                            size=4,
                            pointer_depth=1,
                            record_semantic_id="record:Node",
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="n.h",
                    line=1,
                    end_line=3,
                    size=4,
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Child",
                    qualified_name="Child",
                    bases=(),
                    fields=(
                        SourceField(
                            name="value",
                            type="int",
                            source_file="c.h",
                            line=2,
                            offset=0,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="c.h",
                    line=1,
                    end_line=3,
                    size=4,
                    layout_trusted=True,
                ),
                SourceClass(
                    semantic_id="record:Holder",
                    qualified_name="Holder",
                    bases=(),
                    fields=(
                        SourceField(
                            name="ptr",
                            type="Child *",
                            source_file="h.h",
                            line=2,
                            offset=0,
                            size=4,
                            pointer_depth=1,
                            record_semantic_id="record:Child",
                        ),
                        SourceField(
                            name="ref",
                            type="Child &",
                            source_file="h.h",
                            line=3,
                            offset=4,
                            size=4,
                            record_semantic_id="record:Child",
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="h.h",
                    line=1,
                    end_line=4,
                    size=8,
                    layout_trusted=True,
                ),
            )
        ),
    )
    # Recursive Node* must terminate at the pointer field.
    node = index.resolve_field("Node", 0)
    assert node is not None
    assert node.path == ("next",)
    assert node.leaf.name == "next"
    assert (node.leaf.pointer_depth or 0) == 1

    holder_ptr = index.resolve_field("Holder", 0)
    assert holder_ptr is not None
    assert holder_ptr.path == ("ptr",)
    assert holder_ptr.leaf.name == "ptr"

    holder_ref = index.resolve_field("Holder", 4)
    assert holder_ref is not None
    assert holder_ref.path == ("ref",)
    assert holder_ref.leaf.name == "ref"


def test_untrusted_nested_layout_is_not_published():
    index = SourceIndex(
        declarations={},
        markers=(),
        classes=keyed(
            (
                SourceClass(
                    semantic_id="record:Child",
                    qualified_name="Child",
                    bases=(),
                    fields=(
                        SourceField(
                            name="value",
                            type="int",
                            source_file="c.h",
                            line=2,
                            offset=0,
                            size=4,
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="c.h",
                    line=1,
                    end_line=3,
                    size=4,
                    layout_trusted=False,
                ),
                SourceClass(
                    semantic_id="record:Parent",
                    qualified_name="Parent",
                    bases=(),
                    fields=(
                        SourceField(
                            name="child",
                            type="Child",
                            source_file="p.h",
                            line=2,
                            offset=0,
                            size=4,
                            record_semantic_id="record:Child",
                        ),
                    ),
                    virtual_declarations=(),
                    source_file="p.h",
                    line=1,
                    end_line=3,
                    size=4,
                    layout_trusted=True,
                ),
            )
        ),
    )
    assert index.has_layout("Parent") is True
    assert index.has_layout("Child") is False
    assert index.resolve_field("Parent", 0) is None


def test_cross_target_class_name_is_not_last_wins():
    index = SourceIndex(
        declarations={},
        markers=(),
        classes={
            **keyed(
                (
                    SourceClass(
                        semantic_id="record:Foo",
                        qualified_name="Foo",
                        bases=(),
                        fields=(
                            SourceField(
                                name="a",
                                type="char",
                                source_file="a.h",
                                line=1,
                                offset=0,
                                size=1,
                            ),
                        ),
                        virtual_declarations=(),
                        source_file="a.h",
                        line=1,
                        end_line=2,
                        size=8,
                        layout_trusted=True,
                    ),
                ),
                "EXE",
            ),
            **keyed(
                (
                    SourceClass(
                        semantic_id="record:Foo",
                        qualified_name="Foo",
                        bases=(),
                        fields=(
                            SourceField(
                                name="a",
                                type="char",
                                source_file="a.h",
                                line=1,
                                offset=0,
                                size=1,
                            ),
                        ),
                        virtual_declarations=(),
                        source_file="a.h",
                        line=1,
                        end_line=2,
                        size=4,
                        layout_trusted=True,
                    ),
                ),
                "DLL",
            ),
        },
    )
    assert index.class_named("Foo") is None
    exe = index.for_target("EXE")
    dll = index.for_target("DLL")
    exe_foo = exe.class_named("Foo")
    dll_foo = dll.class_named("Foo")
    assert exe_foo is not None and exe_foo.size == 8
    assert dll_foo is not None and dll_foo.size == 4


def test_enrich_memory_address_with_layout_facts():
    index = _index_with_layout()
    lines_db = MagicMock()
    lines_db.find_line_of_recomp_address.return_value = None
    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=lines_db,
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
        source_index=index,
    )

    match = MagicMock()
    match.orig_addr = 0x401000
    match.name = "Foo::setFlag"
    match.best_name.return_value = "Foo::setFlag"

    analysis = ComparisonAnalysis.mismatch(
        ComparisonDifference(
            "memory_address",
            DifferenceSide(0, 0x401010, {"displacement": 8, "base_register": "ecx"}),
            DifferenceSide(1, 0x501010, {"displacement": 12, "base_register": "ecx"}),
        )
    )
    enriched = (
        comparator._enrich_analysis_with_source(  # pylint: disable=protected-access
            analysis, match=match
        )
    )
    assert enriched.difference is not None
    assert enriched.difference.orig.facts["class_name"] == "Foo"
    assert enriched.difference.orig.facts["field_name"] == "flag"
    assert enriched.difference.orig.facts["field_offset"] == 8
    assert enriched.difference.recomp.facts["field_name"] == "tail"


def test_branch_condition_shows_the_source_comparisons_on_its_line():
    """A signedness or predicate difference is explained by the type the
    recompiled source compares in."""
    key = DeclarationKey("TEST", "?f@@YAHF@Z")
    comparison = SourceComparison(
        operator="<",
        type="int",
        operands=(
            SourceComparisonOperand("short", field="c:@S@Foo@FI@s"),
            SourceComparisonOperand("int", constant=65),
        ),
        line=12,
        offset=None,
        bits=32,
        signed=True,
    )
    other_line = dataclasses.replace(comparison, line=13, operator="==")
    index = SourceIndex(
        declarations={},
        classes={},
        markers=[
            SourceMarker(
                address=0x401000,
                marker_kind="FUNCTION",
                source_file="f.cpp",
                line=10,
                declaration=SourceDeclaration(
                    key.semantic_id,
                    "f",
                    "free_function",
                    "__cdecl",
                    "int",
                    ("short",),
                    None,
                    False,
                    False,
                    "f.cpp",
                    11,
                    14,
                    True,
                ),
                target="TEST",
                declaration_key=key,
            )
        ],
        function_facts={
            key: SourceFunctionFacts(key.semantic_id, (), (comparison, other_line))
        },
    )
    lines_db = MagicMock()
    lines_db.find_line_of_recomp_address.return_value = None
    lines_db.find_line_containing_recomp_address.return_value = (Path("f.cpp"), 12)
    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=lines_db,
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
        source_index=index,
    )
    match = MagicMock()
    match.orig_addr = 0x401000
    match.recomp_addr = 0x501000
    analysis = ComparisonAnalysis.mismatch(
        ComparisonDifference(
            "branch_condition",
            DifferenceSide(0, 0x401010, {"predicate": "lt_u:..."}),
            DifferenceSide(0, 0x501010, {"predicate": "lt_s:..."}),
        )
    )

    enriched = (
        comparator._enrich_analysis_with_source(  # pylint: disable=protected-access
            analysis, match=match
        )
    )

    assert enriched.difference is not None
    assert (
        enriched.difference.recomp.facts["source_comparisons"]
        == "short field < 65 as int, signed"
    )
    assert "source_comparisons" not in enriched.difference.orig.facts
    lines_db.find_line_containing_recomp_address.assert_called_with(0x501010, 0x501000)
