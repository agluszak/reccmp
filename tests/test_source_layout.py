"""Tests for Clang layout lookup and memory_address enrichment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


def _index_with_layout() -> SourceIndex:
    return SourceIndex(
        declarations=(),
        markers=(),
        classes=(
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
        declarations=(),
        markers=(),
        classes=(
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
        declarations=(),
        markers=(),
        classes=(
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
        declarations=(),
        markers=(),
        classes=(
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
    assert namespace.classes[0].layout_trusted is False
    assert any(c.record_kind == "class_layout" for c in namespace.conflicts)
    index = SourceIndex(
        declarations=(),
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
    assert namespace.classes[0].asserted_size == 16
    assert namespace.classes[0].layout_trusted is False


def test_v3_schema_loads_without_layout_fields():
    document = {
        "schema": "reccmp-source-index-v3",
        "declarations": [],
        "markers": [],
        "classes": [
            {
                "semantic_id": "record:Foo",
                "qualified_name": "Foo",
                "bases": [],
                "fields": [
                    {
                        "name": "x",
                        "type": "int",
                        "source_file": "a.h",
                        "line": 1,
                    }
                ],
                "virtual_declarations": [],
                "source_file": "a.h",
                "line": 1,
                "end_line": 2,
            }
        ],
    }
    index = SourceIndex.from_dict(document)
    assert index.classes[0].size is None
    assert index.classes[0].fields[0].offset is None
    assert index.classes[0].layout_trusted is None
    assert index.to_dict()["schema"] == "reccmp-source-index-v5"


def test_enrich_memory_address_with_layout_facts():
    index = _index_with_layout()
    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=MagicMock(),
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
        source_index=index,
    )
    comparator.lines_db.find_line_of_recomp_address.return_value = None

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
    enriched = comparator._enrich_analysis_with_source(analysis, match=match)
    assert enriched.difference is not None
    assert enriched.difference.orig.facts["class_name"] == "Foo"
    assert enriched.difference.orig.facts["field_name"] == "flag"
    assert enriched.difference.orig.facts["field_offset"] == 8
    assert enriched.difference.recomp.facts["field_name"] == "tail"
