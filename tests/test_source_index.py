"""Marker join and link-namespace derivation from direct compiler records."""

from pathlib import Path

from reccmp.source import SourceCollector, SourceIndex


def _declaration(**fields) -> dict:
    base = {
        "record": "declaration",
        "semantic_kind": "free_function",
        "calling_convention": "__cdecl",
        "return_type": "void",
        "parameter_types": [],
        "owning_class": None,
        "has_this": False,
        "is_virtual": False,
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


def test_source_index_joins_markers_to_clang_semantics(tmp_path: Path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text(
        "namespace N {\n"
        "class Base {};\n"
        "// VTABLE: TEST 0x2000\n"
        "class Widget : public Base {\n"
        "public:\n"
        "  // FUNCTION: TEST 0x1000\n"
        "  virtual int Run(short value) { return value; }\n"
        "};\n"
        "}\n",
        encoding="utf-8",
    )
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
            has_this=True,
            is_virtual=True,
            source_file="sample.cpp",
            line=7,
            end_line=7,
        ),
        unit_id="sample.cpp",
    )

    index = SourceIndex.from_collector(
        tmp_path, "TEST", [source], collector, unit_ids={"sample.cpp"}
    )

    assert len(index.markers) == 1
    declaration = index.markers[0].declaration
    assert declaration is not None
    assert declaration.qualified_name == "N::Widget::Run"
    assert declaration.semantic_kind == "instance_method"
    assert declaration.calling_convention == "__thiscall"
    assert declaration.parameter_types == ("short",)
    assert declaration.owning_class == "N::Widget"
    assert declaration.is_virtual
    assert index.classes[0].bases == ("N::Base",)
    assert [(field.name, field.type) for field in index.classes[0].fields] == [
        ("value", "int")
    ]
    assert index.classes[0].vtable_address == 0x2000


def test_source_index_records_isle_style_base_vtables(tmp_path: Path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text(
        "class Primary {};\n"
        "class Secondary {};\n"
        "// VTABLE: TEST 0x2000 Widget\n"
        "// VTABLE: TEST 0x2100 Secondary\n"
        "class Widget : public Primary, public Secondary {};\n",
        encoding="utf-8",
    )
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

    index = SourceIndex.from_collector(
        tmp_path, "TEST", [source], collector, unit_ids={"sample.cpp"}
    )

    assert len(index.classes) == 1
    assert index.classes[0].vtable_address == 0x2000
    assert [
        (item.address, item.base_class) for item in index.classes[0].base_vtables
    ] == [(0x2100, "Secondary")]


def test_source_index_preserves_template_specialization_owner(tmp_path: Path) -> None:
    source = tmp_path / "vector.cpp"
    source.write_text(
        "// FUNCTION: TEST 0x3000\n"
        "template<> Vec<float>* Vec<float>::Convert(double) { return this; }\n",
        encoding="utf-8",
    )
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
            has_this=True,
            source_file="vector.cpp",
            line=2,
            end_line=2,
        ),
        unit_id="vector.cpp",
    )

    index = SourceIndex.from_collector(
        tmp_path, "TEST", [source], collector, unit_ids={"vector.cpp"}
    )

    declaration = index.markers[0].declaration
    assert declaration is not None
    assert declaration.qualified_name == "Vec<float>::Convert"
    assert declaration.owning_class == "Vec<float>"


def test_source_index_joins_standalone_template_vtable_by_name(tmp_path: Path) -> None:
    source = tmp_path / "vector.cpp"
    source.write_text(
        "template<class T> class Vec {};\n"
        "// VTABLE: TEST 0x2000\n"
        "// class Vec<float>\n",
        encoding="utf-8",
    )
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

    index = SourceIndex.from_collector(
        tmp_path, "TEST", [source], collector, unit_ids={"vector.cpp"}
    )

    assert len(index.classes) == 1
    assert index.classes[0].qualified_name == "Vec<float>"
    assert index.classes[0].vtable_address == 0x2000


def test_source_index_preserves_standalone_template_vtable_without_compiler_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vector.cpp"
    source.write_text(
        "// VTABLE: TEST 0x2000\n// class Vec<float>\n",
        encoding="utf-8",
    )
    collector = SourceCollector(tmp_path)

    index = SourceIndex.from_collector(
        tmp_path, "TEST", [source], collector, unit_ids={"vector.cpp"}
    )

    assert len(index.classes) == 1
    assert index.classes[0].semantic_id == "record:Vec<float>"
    assert index.classes[0].qualified_name == "Vec<float>"
    assert index.classes[0].source_file == "vector.cpp"
    assert index.classes[0].vtable_address == 0x2000


def test_source_index_combines_distinct_marker_targets(tmp_path: Path) -> None:
    first = tmp_path / "first.cpp"
    second = tmp_path / "second.cpp"
    first.write_text("// FUNCTION: FIRST 0x1000\nvoid One() {}\n", encoding="utf-8")
    second.write_text("// FUNCTION: SECOND 0x2000\nvoid Two() {}\n", encoding="utf-8")
    collector = SourceCollector(tmp_path)
    collector.collect_record(
        _declaration(
            semantic_id="?One@@YAXXZ",
            qualified_name="One",
            source_file="first.cpp",
            line=2,
            end_line=2,
        ),
        unit_id="first.cpp",
    )
    collector.collect_record(
        _declaration(
            semantic_id="?Two@@YAXXZ",
            qualified_name="Two",
            source_file="second.cpp",
            line=2,
            end_line=2,
        ),
        unit_id="second.cpp",
    )

    indexes = [
        SourceIndex.from_collector(
            tmp_path, "FIRST", [first], collector, unit_ids={"first.cpp"}
        ),
        SourceIndex.from_collector(
            tmp_path, "SECOND", [second], collector, unit_ids={"second.cpp"}
        ),
    ]
    index = SourceIndex(
        declarations=(item for part in indexes for item in part.declarations),
        classes=(),
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

    game_index = SourceIndex.from_collector(
        tmp_path, "GAME", [game], collector, unit_ids={"game.cpp"}
    )
    editor_index = SourceIndex.from_collector(
        tmp_path, "EDITOR", [editor], collector, unit_ids={"editor.cpp"}
    )

    assert [item.type for item in game_index.variables] == ["int"]
    assert not game_index.conflicts
    assert editor_index.variables[0].type in {"int", "float"}
    assert len(editor_index.conflicts) == 1
    assert {
        variant.signature for variant in editor_index.conflicts[0].variants
    } == {("int", "external"), ("float", "external")}


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
