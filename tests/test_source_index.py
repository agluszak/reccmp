"""Marker join and link-namespace derivation from direct compiler records."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from reccmp.source import (
    DeclarationKey,
    SourceCollector,
    SourceIndex,
    SourceIndexError,
    keyed,
)
from reccmp.source.index import source_digest
from reccmp.source.records import SourceDeclaration
from reccmp.source.observations import RecordPool, TranslationUnitRecords


def _declaration(**fields) -> dict:
    base = {
        "record": "declaration",
        "semantic_kind": "free_function",
        "calling_convention": "__cdecl",
        "return_type": "void",
        "parameter_types": [],
        "owning_class": None,
        "is_definition": True,
        "linkage": "external",
        "storage_class": "none",
        "end_line": fields.get("line", 1),
    }
    base.update(fields)
    return base


def _class(**fields) -> dict:
    base = {
        "record": "class",
        "bases": [],
        "fields": [],
        "virtual_declarations": [],
        "end_line": fields.get("line", 1),
    }
    base.update(fields)
    return base


def _marker_block(
    source_file: str,
    first_line: int,
    *comments: str,
    candidates: tuple[dict, ...] = (),
) -> dict:
    """A marker block as the indexer reports it."""
    return {
        "record": "marker-block",
        "source_file": source_file,
        "comments": [
            {
                "text": text,
                "line": first_line + index,
                "column": 1,
                "offset": 100 * (first_line + index),
            }
            for index, text in enumerate(comments)
        ],
        "anchor": {
            "line": first_line + len(comments),
            "column": 1,
            "candidates": list(candidates),
            "string": None,
        },
    }


def _function_candidate(semantic_id: str, name: str, line: int) -> dict:
    return {
        "kind": "function",
        "semantic_id": semantic_id,
        "qualified_name": name,
        "is_definition": True,
        "line": line,
        "end_line": line,
    }


def _class_candidate(name: str) -> dict:
    return {"kind": "class", "semantic_id": f"record:{name}", "qualified_name": name}


def test_source_index_joins_markers_to_clang_semantics(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:N::Widget",
            qualified_name="N::Widget",
            bases=["N::Base"],
            fields=[
                {
                    "name": "value",
                    "type": "int",
                    "pointer_depth": 0,
                    "storage_kind": "scalar",
                    "source_file": "sample.cpp",
                    "line": 6,
                }
            ],
            virtual_declarations=["?Run@Widget@N@@UAEHF@Z"],
            source_file="sample.cpp",
            line=4,
            end_line=8,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _declaration(
            semantic_id="?Run@Widget@N@@UAEHF@Z",
            qualified_name="N::Widget::Run",
            semantic_kind="instance_method",
            calling_convention="__thiscall",
            return_type="int",
            parameter_types=["short"],
            owning_class="N::Widget",
            source_file="sample.cpp",
            line=7,
            end_line=7,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2000",
            candidates=(_class_candidate("N::Widget"),),
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            6,
            "// FUNCTION: TEST 0x1000",
            candidates=(
                _function_candidate("?Run@Widget@N@@UAEHF@Z", "N::Widget::Run", 7),
            ),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})

    assert len(index.markers) == 1
    declaration = index.markers[0].declaration
    assert declaration is not None
    assert declaration.qualified_name == "N::Widget::Run"
    assert declaration.semantic_kind == "instance_method"
    assert declaration.calling_convention == "__thiscall"
    assert declaration.parameter_types == ("short",)
    assert declaration.owning_class == "N::Widget"
    assert list(index.classes.values())[0].bases == ("N::Base",)
    assert [
        (field.name, field.type) for field in list(index.classes.values())[0].fields
    ] == [("value", "int")]
    assert list(index.classes.values())[0].vtable_address == 0x2000
    assert len(index.marker_blocks) == 2


def test_source_index_records_isle_style_base_vtables(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Widget",
            qualified_name="Widget",
            bases=["Primary", "Secondary"],
            source_file="sample.cpp",
            line=5,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2000 Widget",
            "// VTABLE: TEST 0x2100 Secondary",
            candidates=(_class_candidate("Widget"),),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})

    assert len(index.classes) == 1
    assert list(index.classes.values())[0].vtable_address == 0x2000
    assert [
        (item.address, item.base_class)
        for item in list(index.classes.values())[0].base_vtables
    ] == [(0x2100, "Secondary")]


def test_source_index_primary_vtable_can_name_first_base(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Widget",
            qualified_name="Widget",
            bases=["Primary", "Secondary"],
            source_file="sample.cpp",
            line=5,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2000 Primary",
            "// VTABLE: TEST 0x2100 Secondary",
            candidates=(_class_candidate("Widget"),),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})
    widget = next(iter(index.classes.values()))
    assert widget.vtable_address == 0x2000
    assert [(item.address, item.base_class) for item in widget.base_vtables] == [
        (0x2100, "Secondary")
    ]


def test_source_index_primary_vtable_can_name_first_base_ancestor(
    tmp_path: Path,
) -> None:
    collector = SourceCollector(tmp_path)
    for name, bases in (
        ("Widget", ["Middle", "Secondary"]),
        ("Middle", ["Root"]),
        ("Root", []),
    ):
        collector.collect_record(
            _class(
                semantic_id=f"record:{name}",
                qualified_name=name,
                bases=bases,
                source_file="sample.cpp",
                line=5,
            ),
            unit_id="sample.cpp",
        )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2000 Root",
            "// VTABLE: TEST 0x2100 Secondary",
            candidates=(_class_candidate("Widget"),),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})
    widget = index.class_named("Widget", target="TEST")
    assert widget is not None and widget.vtable_address == 0x2000
    assert [(item.address, item.base_class) for item in widget.base_vtables] == [
        (0x2100, "Secondary")
    ]


def test_source_index_first_base_after_primary_is_secondary(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Widget",
            qualified_name="Widget",
            bases=["Primary"],
            source_file="sample.cpp",
            line=5,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2000 Widget",
            "// VTABLE: TEST 0x2100 Primary",
            candidates=(_class_candidate("Widget"),),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})
    widget = next(iter(index.classes.values()))
    assert widget.vtable_address == 0x2000
    assert [(item.address, item.base_class) for item in widget.base_vtables] == [
        (0x2100, "Primary")
    ]


def test_source_index_first_base_before_explicit_primary_is_secondary(
    tmp_path: Path,
) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Widget",
            qualified_name="Widget",
            bases=["Primary"],
            source_file="sample.cpp",
            line=5,
        ),
        unit_id="sample.cpp",
    )
    collector.collect_record(
        _marker_block(
            "sample.cpp",
            3,
            "// VTABLE: TEST 0x2100 Primary",
            "// VTABLE: TEST 0x2000 Widget",
            candidates=(_class_candidate("Widget"),),
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"sample.cpp"})
    widget = next(iter(index.classes.values()))
    assert widget.vtable_address == 0x2000
    assert [(item.address, item.base_class) for item in widget.base_vtables] == [
        (0x2100, "Primary")
    ]


def test_source_index_preserves_template_specialization_owner(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _declaration(
            semantic_id="?Convert@?$Vec@M@@QAEPAV1@N@Z",
            qualified_name="Vec<float>::Convert",
            semantic_kind="instance_method",
            calling_convention="__thiscall",
            return_type="Vec<float> *",
            parameter_types=["double"],
            owning_class="Vec<float>",
            source_file="vector.cpp",
            line=2,
            end_line=2,
        ),
        unit_id="vector.cpp",
    )
    collector.collect_record(
        _marker_block(
            "vector.cpp",
            1,
            "// FUNCTION: TEST 0x3000",
            candidates=(
                _function_candidate(
                    "?Convert@?$Vec@M@@QAEPAV1@N@Z", "Vec<float>::Convert", 2
                ),
            ),
        ),
        unit_id="vector.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"vector.cpp"})

    declaration = index.markers[0].declaration
    assert declaration is not None
    assert declaration.qualified_name == "Vec<float>::Convert"
    assert declaration.owning_class == "Vec<float>"


def test_source_index_refuses_an_ambiguous_function_marker(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    for semantic_id in ("?Get@?$Vec@H@@QAEHXZ", "?Get@?$Vec@M@@QAEMXZ"):
        collector.collect_record(
            _declaration(
                semantic_id=semantic_id,
                qualified_name="Vec::Get",
                source_file="vec.h",
                line=2,
            ),
            unit_id="a.cpp",
        )
    collector.collect_record(
        _marker_block(
            "vec.h",
            1,
            "// FUNCTION: TEST 0x3000",
            candidates=(
                _function_candidate("?Get@?$Vec@H@@QAEHXZ", "Vec<int>::Get", 2),
                _function_candidate("?Get@?$Vec@M@@QAEMXZ", "Vec<float>::Get", 2),
            ),
        ),
        unit_id="a.cpp",
    )
    with pytest.raises(SourceIndexError, match="binds to 2 function definitions"):
        SourceIndex.from_collector("TEST", collector, unit_ids={"a.cpp"})


def test_source_index_joins_standalone_template_vtable_by_name(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Vec<float>",
            qualified_name="Vec<float>",
            source_file="vector.cpp",
            line=1,
        ),
        unit_id="vector.cpp",
    )
    collector.collect_record(
        _marker_block("vector.cpp", 2, "// VTABLE: TEST 0x2000", "// class Vec<float>"),
        unit_id="vector.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"vector.cpp"})

    assert len(index.classes) == 1
    assert list(index.classes.values())[0].qualified_name == "Vec<float>"
    assert list(index.classes.values())[0].vtable_address == 0x2000


def test_source_index_pointer_template_vtable_is_primary(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _class(
            semantic_id="record:Vec<Item *>",
            qualified_name="Vec<Item *>",
            source_file="vector.cpp",
            line=1,
        ),
        unit_id="vector.cpp",
    )
    collector.collect_record(
        _marker_block("vector.cpp", 2, "// VTABLE: TEST 0x2000", "// class Vec<Item*>"),
        unit_id="vector.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"vector.cpp"})
    record = next(iter(index.classes.values()))
    assert record.vtable_address == 0x2000
    assert record.base_vtables == ()


def test_source_index_preserves_standalone_template_vtable_without_compiler_record(
    tmp_path: Path,
) -> None:
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _marker_block("vector.cpp", 1, "// VTABLE: TEST 0x2000", "// class Vec<float>"),
        unit_id="vector.cpp",
    )

    index = SourceIndex.from_collector("TEST", collector, unit_ids={"vector.cpp"})

    assert len(index.classes) == 1
    assert list(index.classes.values())[0].semantic_id == "record:Vec<float>"
    assert list(index.classes.values())[0].qualified_name == "Vec<float>"
    assert list(index.classes.values())[0].source_file == "vector.cpp"
    assert list(index.classes.values())[0].vtable_address == 0x2000


def test_source_index_combines_distinct_marker_targets(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    for unit, target, address, name in (
        ("first.cpp", "FIRST", 0x1000, "One"),
        ("second.cpp", "SECOND", 0x2000, "Two"),
    ):
        semantic_id = f"?{name}@@YAXXZ"
        collector.collect_record(
            _declaration(
                semantic_id=semantic_id, qualified_name=name, source_file=unit, line=2
            ),
            unit_id=unit,
        )
        collector.collect_record(
            _marker_block(
                unit,
                1,
                f"// FUNCTION: {target} 0x{address:x}",
                candidates=(_function_candidate(semantic_id, name, 2),),
            ),
            unit_id=unit,
        )

    indexes = [
        SourceIndex.from_collector("FIRST", collector, unit_ids={"first.cpp"}),
        SourceIndex.from_collector("SECOND", collector, unit_ids={"second.cpp"}),
    ]
    index = SourceIndex(
        declarations=keyed(
            (item for part in indexes for item in part.declarations.values())
        ),
        classes={},
        markers=(item for part in indexes for item in part.markers),
    )

    marker_names: list[tuple[int, str]] = []
    for marker in index.markers:
        assert marker.declaration is not None
        marker_names.append((marker.address, marker.declaration.qualified_name))

    assert marker_names == [
        (0x1000, "One"),
        (0x2000, "Two"),
    ]


def test_marker_blocks_and_source_digests_survive_a_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "a.cpp"
    source.write_text("// FUNCTION: TEST 0x1000\nvoid f() {}\n", encoding="utf-8")
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _declaration(
            semantic_id="?f@@YAXXZ", qualified_name="f", source_file="a.cpp", line=2
        ),
        unit_id="a.cpp",
    )
    collector.collect_record(
        _marker_block(
            "a.cpp",
            1,
            "// FUNCTION: TEST 0x1000",
            candidates=(_function_candidate("?f@@YAXXZ", "f", 2),),
        ),
        unit_id="a.cpp",
    )
    derived = SourceIndex.from_collector("TEST", collector, unit_ids={"a.cpp"})
    index = SourceIndex(
        declarations=derived.declarations,
        classes=derived.classes,
        markers=derived.markers,
        marker_blocks=derived.marker_blocks,
        source_digests={"a.cpp": source_digest(source)},
    )
    revived = SourceIndex.from_dict(json.loads(json.dumps(index.to_dict())))
    assert revived.marker_blocks == index.marker_blocks
    assert revived.source_digests == {"a.cpp": source_digest(source)}
    assert not revived.stale_sources([source])
    source.write_text("// FUNCTION: TEST 0x2000\nvoid f() {}\n", encoding="utf-8")
    other = tmp_path / "b.cpp"
    other.write_text("", encoding="utf-8")
    assert revived.stale_sources([source, other]) == [source, other]


def test_conflicts_are_derived_inside_one_link_namespace(tmp_path: Path) -> None:
    """Cross-target collisions must not become same-target one-variant conflicts."""

    game = tmp_path / "game.cpp"
    editor = tmp_path / "editor.cpp"
    game.write_text("int gThing;\n", encoding="utf-8")
    editor.write_text("float gThing;\n", encoding="utf-8")

    def variable(name: str, path: Path, type_name: str) -> dict:
        return {
            "record": "variable",
            "semantic_id": f"_{name}",
            "qualified_name": name,
            "type": type_name,
            "storage_kind": "scalar",
            "linkage": "external",
            "storage_class": "none",
            "definition_kind": "definition",
            "source_file": path.name,
            "line": 1,
            "end_line": 1,
        }

    collector = SourceCollector(tmp_path)
    collector.collect_record(variable("gThing", game, "int"), unit_id="game.cpp")
    collector.collect_record(variable("gThing", editor, "int"), unit_id="editor.cpp")
    collector.collect_record(
        {
            **variable("gThing", editor, "float"),
            "definition_kind": "declaration",
            "line": 2,
            "end_line": 2,
        },
        unit_id="editor.cpp",
    )

    game_index = SourceIndex.from_collector("GAME", collector, unit_ids={"game.cpp"})
    editor_index = SourceIndex.from_collector(
        "EDITOR", collector, unit_ids={"editor.cpp"}
    )

    assert [item.type for item in game_index.variables.values()] == ["int"]
    assert not game_index.conflicts
    assert list(editor_index.variables.values())[0].type in {"int", "float"}
    assert len(editor_index.conflicts) == 1
    assert {variant.signature for variant in editor_index.conflicts[0].variants} == {
        ("int", "external"),
        ("float", "external"),
    }


def test_internal_functions_are_distinct_per_translation_unit(tmp_path: Path) -> None:
    collector = SourceCollector(tmp_path)
    for unit, return_type in (("a.cpp", "int"), ("b.cpp", "long")):
        collector.collect_record(
            _declaration(
                semantic_id="_helper",
                qualified_name="helper",
                return_type=return_type,
                linkage="internal",
                storage_class="static",
                source_file=unit,
                line=1,
                end_line=3,
            ),
            unit_id=unit,
        )

    namespace = collector.derive(target="GAME", unit_ids={"a.cpp", "b.cpp"})
    assert len(namespace.declarations) == 2
    assert not namespace.conflicts


def test_shared_header_facts_preserve_local_keys_targets_and_conflicts(
    tmp_path: Path,
) -> None:
    external = _declaration(
        semantic_id="_shared", qualified_name="shared", source_file="common.h", line=1
    )
    local = _declaration(
        semantic_id="_helper",
        qualified_name="helper",
        source_file="common.h",
        line=2,
        linkage="internal",
        storage_class="static",
    )
    pool = RecordPool()
    units = []
    for name, records in (
        ("a.cpp", [external, local]),
        ("b.cpp", [external, local]),
        ("c.cpp", [{**external, "return_type": "long"}]),
    ):
        path = tmp_path / name
        path.write_text(
            " \t\n" + "\n".join(json.dumps(record) for record in records) + "\n"
        )
        units.append(TranslationUnitRecords.load(path, name, pool))
    index = SourceIndex.from_units(
        units, {"GAME": {"a.cpp", "b.cpp", "c.cpp"}, "EDITOR": {"a.cpp"}}
    )
    assert {key for key in index.declarations if key.semantic_id == "_helper"} == {
        DeclarationKey("GAME", "_helper", "a.cpp"),
        DeclarationKey("GAME", "_helper", "b.cpp"),
        DeclarationKey("EDITOR", "_helper", "a.cpp"),
    }
    assert DeclarationKey("GAME", "_shared") in index.declarations
    assert DeclarationKey("EDITOR", "_shared") in index.declarations
    [conflict] = index.conflicts
    assert conflict.target == "GAME"
    assert {variant.signature[2] for variant in conflict.variants} == {"void", "long"}


def test_tu_local_functions_with_one_mangled_name_bind_by_location(
    tmp_path: Path,
) -> None:
    """Two files each define ``static void* copyMemory(...)``: the mangled
    names are equal, the definitions are not."""
    collector = SourceCollector(tmp_path)
    for unit, line, address in (
        ("huffman.cpp", 565, 0x1000),
        ("renderer.cpp", 79, 0x2000),
    ):
        collector.collect_record(
            _declaration(
                semantic_id="?copyMemory@@YAPAXPAXPBXJ@Z",
                qualified_name="copyMemory",
                linkage="internal",
                storage_class="static",
                source_file=unit,
                line=line,
            ),
            unit_id=unit,
        )
        collector.collect_record(
            _marker_block(
                unit,
                line - 1,
                f"// FUNCTION: TEST 0x{address:x}",
                candidates=(
                    _function_candidate(
                        "?copyMemory@@YAPAXPAXPBXJ@Z", "copyMemory", line
                    ),
                ),
            ),
            unit_id=unit,
        )
    # A header's static function has one identical winner per including unit.
    for unit in ("a.cpp", "b.cpp"):
        collector.collect_record(
            _declaration(
                semantic_id="?helper@@YAXXZ",
                qualified_name="helper",
                linkage="internal",
                source_file="util.h",
                line=3,
            ),
            unit_id=unit,
        )
        collector.collect_record(
            _marker_block(
                "util.h",
                2,
                "// FUNCTION: TEST 0x3000",
                candidates=(_function_candidate("?helper@@YAXXZ", "helper", 3),),
            ),
            unit_id=unit,
        )

    derived = SourceIndex.from_collector("TEST", collector)
    # Identity survives the JSON projection.
    index = SourceIndex.from_dict(json.loads(json.dumps(derived.to_dict())))

    assert {
        marker.address: (marker.declaration.source_file, marker.declaration_key)
        for marker in index.markers
        if marker.declaration is not None
    } == {
        0x1000: (
            "huffman.cpp",
            DeclarationKey("TEST", "?copyMemory@@YAPAXPAXPBXJ@Z", "huffman.cpp"),
        ),
        0x2000: (
            "renderer.cpp",
            DeclarationKey("TEST", "?copyMemory@@YAPAXPAXPBXJ@Z", "renderer.cpp"),
        ),
        # A header's TU-local function: the first including unit's copy.
        0x3000: ("util.h", DeclarationKey("TEST", "?helper@@YAXXZ", "a.cpp")),
    }


def test_a_targets_markers_come_from_its_own_source_files(tmp_path: Path) -> None:
    """A header of another target's sources may carry this target's markers;
    as with every other marker reader, they are not this target's markers."""
    collector = SourceCollector(tmp_path)
    for source_file, address in (("game/main.cpp", 0x1000), ("lib/math.h", 0x2000)):
        collector.collect_record(
            _declaration(
                semantic_id=f"?f{address:x}@@YAXXZ",
                qualified_name=f"f{address:x}",
                source_file=source_file,
                line=2,
            ),
            unit_id="game/main.cpp",
        )
        collector.collect_record(
            _marker_block(
                source_file,
                1,
                f"// FUNCTION: GAME 0x{address:x}",
                candidates=(
                    _function_candidate(f"?f{address:x}@@YAXXZ", f"f{address:x}", 2),
                ),
            ),
            unit_id="game/main.cpp",
        )
    units = tuple(collector.units.values())

    scoped = SourceIndex.from_units(
        units, {"GAME": None}, target_files={"GAME": {"game/main.cpp"}}
    )
    assert [marker.address for marker in scoped.markers] == [0x1000]
    assert [
        marker.address
        for marker in SourceIndex.from_units(units, {"GAME": None}).markers
    ] == [0x1000, 0x2000]


def test_identity_changes_with_what_clang_reported(tmp_path: Path) -> None:
    """The prepared-analysis cache keys on the index's identity: any change
    to the collected facts must change it, even with unchanged sources."""
    key = DeclarationKey("GAME", "?f@@YAXXZ")
    declaration = SourceDeclaration(
        key.semantic_id,
        "f",
        "free_function",
        "__cdecl",
        "void",
        (),
        None,
        "f.cpp",
        2,
        3,
        True,
    )
    index = SourceIndex(declarations={key: declaration}, classes={}, markers=())
    path = tmp_path / "source-index.json"
    index.write(path)

    first = SourceIndex.read(path)
    assert first.identity() == SourceIndex.read(path).identity()
    assert first.for_target("GAME").identity() == f"{first.identity()}:GAME"
    # Built in memory: the projection's digest, stable across calls
    assert index.identity() == index.identity()

    moved = SourceIndex(
        declarations={key: dataclasses.replace(declaration, line=5)},
        classes={},
        markers=(),
    )
    moved.write(path)
    assert SourceIndex.read(path).identity() != first.identity()
    assert moved.identity() != index.identity()


def test_cached_read_observes_replacement_with_preserved_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source-index.json"
    index = SourceIndex(
        declarations={}, classes={}, markers=(), source_digests={"a.cpp": "aaa"}
    )
    index.write(path)
    first = SourceIndex.read(path)
    stat = path.stat()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes().replace(b"aaa", b"bbb"))
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    second = SourceIndex.read(path)
    assert second.source_digests == {"a.cpp": "bbb"}
    assert second.identity() != first.identity()


def test_cached_read_keeps_reader_maps_independent(tmp_path: Path) -> None:
    path = tmp_path / "source-index.json"
    SourceIndex(
        declarations={}, classes={}, markers=(), source_digests={"a.cpp": "aaa"}
    ).write(path)
    first = SourceIndex.read(path)
    first.source_digests["a.cpp"] = "mutated"
    first.unit_dependencies["a.cpp"] = ("invented.h",)
    second = SourceIndex.read(path)
    assert second.source_digests == {"a.cpp": "aaa"}
    assert second.unit_dependencies == {}


def test_source_marker_keeps_display_and_recomp_selector_separate(tmp_path: Path):
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _marker_block(
            "a.cpp",
            1,
            "// TEMPLATE: TEST 0x1000",
            "// NAME: Vec<T>::Grow",
            "// RECOMP: ?Grow@?$Vec@H@@QAEXH@Z",
        ),
        unit_id="a.cpp",
    )
    index = SourceIndex.from_collector("TEST", collector, unit_ids={"a.cpp"})
    marker = index.markers[0]
    assert marker.marker_name == "Vec<T>::Grow"
    assert marker.recomp_selector == "?Grow@?$Vec@H@@QAEXH@Z"
    assert marker.selector_is_symbol
    assert marker.declaration is None
    path = tmp_path / "index.json"
    index.write(path)
    assert SourceIndex.read(path).markers == index.markers
