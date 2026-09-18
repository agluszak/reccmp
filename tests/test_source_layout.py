"""Tests for Clang layout lookup and memory_address enrichment."""

from __future__ import annotations

from unittest.mock import MagicMock

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceSide,
)
from reccmp.compare.functions import FunctionComparator
from reccmp.source import SourceClass, SourceField, SourceIndex


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
            ),
        ),
    )


def test_field_at_picks_largest_offset_not_exceeding_request():
    index = _index_with_layout()
    assert index.field_at("Foo", 0).name == "bar"
    assert index.field_at("Foo", 7).name == "bar"
    assert index.field_at("Foo", 8).name == "flag"
    assert index.field_at("Foo", 12).name == "tail"
    assert index.field_at("Foo", 100).name == "tail"
    assert index.field_at("Missing", 0) is None


def test_field_path_at_descends_into_nested_layout():
    index = _index_with_layout()
    assert index.field_path_at("Foo", 4) == "bar.qux"
    assert index.field_path_at("Foo", 8) == "flag"


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
    assert index.to_dict()["schema"] == "reccmp-source-index-v4"


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
    assert enriched.difference.recomp.facts["field_name"] == "tail"
