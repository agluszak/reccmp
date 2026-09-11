import json
from pathlib import Path

import pytest
from reccmp.source.index import SourceIndexError
from reccmp.source import SourceCollector
from reccmp.source.batch import record_command

DECLARATION = {
    "record": "declaration",
    "semantic_id": "?Grow@Vector@@QAEHH@Z",
    "qualified_name": "Vector::Grow",
    "semantic_kind": "instance_method",
    "calling_convention": "__thiscall",
    "return_type": "int",
    "parameter_types": ["int"],
    "source_signature": "int Vector::Grow(int value)",
    "parameter_references": [False],
    "owning_class": "Vector",
    "has_this": True,
    "is_virtual": False,
    "source_file": "include/wiz8/vector.h",
    "line": 20,
    "end_line": 20,
    "is_definition": False,
}
CLASS = {
    "record": "class",
    "semantic_id": "record:Vector",
    "qualified_name": "Vector",
    "bases": [],
    "fields": [
        {
            "name": "count",
            "type": "int",
            "source_file": "include/wiz8/vector.h",
            "line": 24,
        }
    ],
    "virtual_declarations": [],
    "source_file": "include/wiz8/vector.h",
    "line": 12,
    "end_line": 30,
}


def _records(*records: dict) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def test_definition_replaces_a_declaration_from_another_unit() -> None:
    collector = SourceCollector(Path("/repo"))
    definition = {**DECLARATION, "is_definition": True, "line": 105, "end_line": 118}
    collector.collect_records(_records(DECLARATION))
    collector.collect_records(_records(definition))
    collector.collect_records(_records(DECLARATION))

    kept = collector.declarations["?Grow@Vector@@QAEHH@Z"]
    assert kept.is_definition
    assert (kept.line, kept.end_line) == (105, 118)
    assert kept.parameter_types == ("int",)
    assert kept.source_signature == "int Vector::Grow(int value)"
    assert kept.parameter_references == (False,)


def test_class_is_kept_from_the_first_unit_that_located_it() -> None:
    collector = SourceCollector(Path("/repo"))
    collector.collect_records(_records({**CLASS, "line": 0, "end_line": 0}))
    collector.collect_records(_records(CLASS))
    collector.collect_records(_records({**CLASS, "line": 99, "end_line": 99}))

    kept = collector.classes["record:Vector"]
    assert (kept.line, kept.end_line) == (12, 30)
    assert kept.fields[0].name == "count"


def test_conflicting_size_assertions_are_refused() -> None:
    collector = SourceCollector(Path("/repo"))
    assertion = {
        "record": "size-assertion",
        "qualified_name": "Vector",
        "asserted_size": 16,
    }
    collector.collect_records(_records(assertion))
    collector.collect_records(_records(assertion))
    assert collector.size_assertions == {"Vector": 16}

    with pytest.raises(SourceIndexError, match="conflicting size assertions"):
        collector.collect_records(_records({**assertion, "asserted_size": 20}))


def test_an_unknown_record_is_refused_rather_than_ignored() -> None:
    collector = SourceCollector(Path("/repo"))
    with pytest.raises(SourceIndexError, match="unknown record"):
        collector.collect_records(
            _records({"record": "enum", "qualified_name": "Slot"})
        )


def test_the_index_command_keeps_the_build_arguments_and_drops_the_ast_dump() -> None:
    command = record_command(
        {
            "directory": "/out",
            "file": "/repo/src/wiz8/vector.cpp",
            "command": (
                "/usr/bin/clang-cl /nologo -Xclang -fno-wchar /Fovector.obj /c "
                "-- /repo/src/wiz8/vector.cpp"
            ),
        },
        "/indexer/indexer",
        "/usr/bin/clang-cl",
    )

    assert command[:2] == ["/indexer/indexer", "/usr/bin/clang-cl"]
    assert "-ast-dump=json" not in command
    # The build's own -Xclang option and its argument both survive, and the
    # source file stays behind the driver's end-of-options separator.
    assert command.count("-Xclang") == 1
    assert command[command.index("-Xclang") + 1] == "-fno-wchar"
    assert command[-2:] == ["--", "/repo/src/wiz8/vector.cpp"]
    assert not any(
        argument.startswith("/Fo") or argument == "/c" for argument in command
    )


VARIABLE = {
    "record": "variable",
    "semantic_id": "_gThing",
    "qualified_name": "gThing",
    "type": "Foo *",
    "linkage": "external",
    "storage_class": "none",
    "definition_kind": "declaration",
    "source_file": "src/wiz8/a.cpp",
    "line": 3,
    "end_line": 3,
}


def test_variable_rank_prefers_initialized_definitions() -> None:
    from reccmp.source import SourceCollector

    collector = SourceCollector(Path("/repo"))
    collector.collect_records(_records(VARIABLE))
    collector.collect_records(
        _records(
            {
                **VARIABLE,
                "definition_kind": "tentative",
                "source_file": "src/wiz8/b.cpp",
            }
        )
    )
    kept = collector.variables["_gThing"]
    assert kept.definition_kind == "tentative"

    collector.collect_records(
        _records(
            {
                **VARIABLE,
                "definition_kind": "definition",
                "source_file": "src/wiz8/c.cpp",
            }
        )
    )
    assert collector.variables["_gThing"].definition_kind == "definition"

    # A later tentative definition does not displace the initialized one.
    collector.collect_records(
        _records(
            {
                **VARIABLE,
                "definition_kind": "tentative",
                "source_file": "src/wiz8/d.cpp",
            }
        )
    )
    assert collector.variables["_gThing"].source_file == "src/wiz8/c.cpp"
    assert not collector.conflicts


def test_conflicting_global_spellings_are_retained() -> None:
    from reccmp.source import SourceCollector

    collector = SourceCollector(Path("/repo"))
    collector.collect_records(_records(VARIABLE))
    collector.collect_records(
        _records(
            {
                **VARIABLE,
                "type": "int",
                "definition_kind": "definition",
                "source_file": "src/wiz8/b.cpp",
            }
        )
    )

    # The definition still wins the merged index, but the disagreement survives.
    assert collector.variables["_gThing"].type == "int"
    (conflict,) = collector.conflicts.values()
    assert conflict.semantic_id == "_gThing"
    assert conflict.record_kind == "variable"
    assert [
        (variant.signature, variant.locations) for variant in conflict.variants
    ] == [
        (("Foo *", "external"), ("src/wiz8/a.cpp:3",)),
        (("int", "external"), ("src/wiz8/b.cpp:3",)),
    ]


def test_identical_header_spellings_do_not_conflict() -> None:
    from reccmp.source import SourceCollector

    collector = SourceCollector(Path("/repo"))
    for unit in ("a.cpp", "b.cpp", "c.cpp"):
        collector.collect_records(
            _records({**VARIABLE, "source_file": f"src/wiz8/{unit}"})
        )
    assert not collector.conflicts
    assert collector.variables["_gThing"].source_file == "src/wiz8/a.cpp"


def test_static_and_external_linkage_conflict() -> None:
    from reccmp.source import SourceCollector

    collector = SourceCollector(Path("/repo"))
    collector.collect_records(
        _records(
            {
                "record": "declaration",
                "semantic_id": "_helper",
                "qualified_name": "helper",
                "semantic_kind": "free_function",
                "calling_convention": "__cdecl",
                "return_type": "void",
                "parameter_types": [],
                "owning_class": None,
                "has_this": False,
                "is_virtual": False,
                "source_file": "src/wiz8/a.c",
                "line": 10,
                "end_line": 12,
                "is_definition": True,
                "linkage": "internal",
                "storage_class": "static",
            }
        )
    )
    collector.collect_records(
        _records(
            {
                "record": "declaration",
                "semantic_id": "_helper",
                "qualified_name": "helper",
                "semantic_kind": "free_function",
                "calling_convention": "__cdecl",
                "return_type": "void",
                "parameter_types": [],
                "owning_class": None,
                "has_this": False,
                "is_virtual": False,
                "source_file": "src/wiz8/b.c",
                "line": 4,
                "end_line": 4,
                "is_definition": False,
                "linkage": "external",
                "storage_class": "none",
            }
        )
    )
    assert len(collector.conflicts) == 1


def test_conflicts_survive_a_json_round_trip() -> None:
    from reccmp.source import SourceCollector, SourceIndex

    collector = SourceCollector(Path("/repo"))
    collector.collect_records(_records(VARIABLE))
    collector.collect_records(
        _records({**VARIABLE, "type": "int", "definition_kind": "definition"})
    )
    index = SourceIndex(
        declarations=(),
        classes=(),
        markers=(),
        variables=collector.variables.values(),
        conflicts=collector.conflicts.values(),
    )
    import json

    revived = SourceIndex.from_dict(json.loads(json.dumps(index.to_dict())))
    assert revived.to_dict() == index.to_dict()
    assert revived.variables[0].is_external
    assert revived.variables[0].definition_kind == "definition"
