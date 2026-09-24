"""Exercise the actual collector inside the pinned LLVM 19 analysis environment."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from reccmp.parser import DecompCodebase
from reccmp.parser.error import AlertCode
from reccmp.parser.marker import MarkerType
from reccmp.source import SourceIndex, SourceIndexError
from reccmp.tools.decomplint import DecomplintTarget, lint_all_targets


def _require_collector() -> None:
    if (
        not os.environ.get("RECCMP_SOURCE_INDEXER")
        and not Path("/usr/lib/llvm-19/include/clang/AST/ASTConsumer.h").is_file()
    ):
        pytest.skip(
            "run inside the pinned analysis image (LLVM 19 + reccmp-source-indexer)"
        )


def _clang_cl(repository: Path) -> str:
    for candidate in ("/usr/bin/clang-cl", "/usr/bin/clang-cl-19"):
        if Path(candidate).is_file():
            return candidate
    # Debian's clang package may omit the cl driver name; the indexer still
    # selects CL mode from a path that ends in clang-cl.
    clang_cl = repository / "clang-cl"
    clang_cl.symlink_to("/usr/bin/clang-19")
    return str(clang_cl)


_MARKED_SOURCE = """\
#include "widget.h"
#define EXPORT
namespace N {
// GLOBAL: TEST 0x3000
int g_count = 0, g_other;

// STRING: TEST 0x4000
const char* g_hello = "hello\\tworld";

// FUNCTION: TEST 0x1020
// clang-format off
int Widget::Declared()
{
  // GLOBAL: TEST 0x3010
  static int s_calls = 0;
  // STRING: TEST 0x4010
  return (int)(long)L"wide" + s_calls;
}
}

// SYNTHETIC: TEST 0x5000
// N::Widget::`scalar deleting destructor'

// FUNCTION: TEST 0x1030
EXPORT void Exported() {}

// FUNCTION: TEST 0x1040
extern "C" void CFunction() {}

template <class T> struct Vec { T Get() { return T(); } };
// FUNCTION: TEST 0x1050
template <> float Vec<float>::Get() { return 1.0f; }

#if 0
// FUNCTION: TEST 0x9999
void Dead() {}
#endif

void Lines() {
  // LINE: TEST 0x6000
  Exported();
}
"""

_MARKED_HEADER = """\
namespace N {
// VTABLE: TEST 0x2000
// VTABLE: TEST 0x2100 Base
class Widget {
public:
  // FUNCTION: TEST 0x1000
  virtual int Run(short value) { return value; }
  int Declared();
};
}
"""


def test_markers_come_from_the_compiler(tmp_path: Path) -> None:
    # pylint: disable=too-many-locals
    _require_collector()
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "widget.cpp"
    header = repository / "widget.h"
    orphan = repository / "orphan.h"
    source.write_text(_MARKED_SOURCE, encoding="utf-8")
    header.write_text(_MARKED_HEADER, encoding="utf-8")
    orphan.write_text("// FUNCTION: TEST 0x7000\nvoid Orphan() {}\n", encoding="utf-8")
    database = repository / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {
                    "directory": str(repository),
                    "file": str(source),
                    "arguments": [
                        _clang_cl(repository),
                        "--target=i686-pc-windows-msvc",
                        "/c",
                        str(source),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    files = [source, header, orphan]
    previous_root = os.environ.get("RECCMP_SOURCE_ROOT")
    os.environ["RECCMP_SOURCE_ROOT"] = str(repository)
    try:
        index = SourceIndex.from_compile_database(
            repository,
            database,
            {"TEST": files},
            cache_dir=tmp_path / "cache",
            jobs=1,
        )
    finally:
        if previous_root is None:
            os.environ.pop("RECCMP_SOURCE_ROOT", None)
        else:
            os.environ["RECCMP_SOURCE_ROOT"] = previous_root

    functions = index.functions_by_address(target="TEST")
    assert {address: marker.name for address, marker in functions.items()} == {
        0x1000: "N::Widget::Run",
        0x1020: "N::Widget::Declared",
        0x1030: "Exported",
        0x1040: "CFunction",
        0x1050: "Vec<float>::Get",
        0x5000: "N::Widget::`scalar deleting destructor'",
    }
    widget = index.class_named("N::Widget", target="TEST")
    assert widget is not None and widget.vtable_address == 0x2000
    assert [(item.address, item.base_class) for item in widget.base_vtables] == [
        (0x2100, "Base")
    ]

    codebase = DecompCodebase.from_source_index(
        index.for_target("TEST"), "TEST", files, encoding="latin1"
    )
    line_functions = {f.offset: f for f in codebase.iter_line_functions()}
    assert (line_functions[0x1020].filename, line_functions[0x1020].line_number) == (
        source,
        12,
    )
    assert line_functions[0x1020].end_line == 18
    assert line_functions[0x1000].filename == header
    variables = {v.offset: v for v in codebase.iter_variables()}
    assert (variables[0x3000].name, variables[0x3000].is_static) == (
        "N::g_count",
        False,
    )
    assert (variables[0x3010].name, variables[0x3010].parent_function) == (
        "s_calls",
        0x1020,
    )
    strings = {s.offset: (s.name, s.is_widechar) for s in codebase.iter_strings()}
    assert strings == {0x4000: ("hello\tworld", False), 0x4010: ("wide", True)}
    lines = [(s.offset, s.line_number) for s in codebase.iter_line_symbols()]
    assert lines == [(0x6000, 40)]
    assert {s.type for s in codebase.iter_name_functions()} == {MarkerType.SYNTHETIC}
    assert 0x9999 not in codebase.symbols_for_offsets([0x9999])

    alerts = lint_all_targets(
        (
            DecomplintTarget(
                tuple(files), "TEST", "utf-8", source_index=_write(index, tmp_path)
            ),
        )
    )
    assert sorted(
        (alert.path.name, alert.line_number)
        for alert in alerts
        if alert.code == AlertCode.MARKER_NOT_COMPILED
    ) == [("orphan.h", 1), ("widget.cpp", 35)]


def _write(index: SourceIndex, directory: Path) -> Path:
    path = directory / "source-index.json"
    index.write(path)
    return path


def test_native_batch_records_cache_and_errors(tmp_path: Path) -> None:
    # pylint: disable=too-many-statements,too-many-locals
    _require_collector()
    repository = tmp_path / "source with spaces"
    repository.mkdir()
    header = repository / "owner.h"
    header.write_text(
        "struct Owner {\n"
        "  int **pointers;\n"
        "  int (*callback)(int);\n"
        "  int *elements[2];\n"
        "  int &reference;\n"
        '};\nstatic_assert(sizeof(Owner) == 20, "size");\n'
        "struct W8First { int unknown_04; int values[4]; };\n"
        "struct W8Second { int unknown_04; };\n"
        "struct W8Payload { int value; };\n"
        "struct W8Record { W8Payload payload; };\n"
        "struct SurrenderOnly { int unknown_04; };\n"
        "template<class T> struct DependentRecord { T value; };\n"
        "inline int ReadW8Field(W8First& object, int index) {\n"
        "  int value = object.unknown_04;\n"
        "  object.values[index] = value;\n"
        "  return value;\n"
        "}\n"
        "inline int ReadW8Second(const W8Second& object) { return object.unknown_04; }\n"
        "inline double ConvertW8Field(const W8First& object) {\n"
        "  return static_cast<double>(object.unknown_04);\n"
        "}\n"
        "inline int* AddressW8Field(W8First& object) { return &object.unknown_04; }\n"
        "inline int ReadW8Element(const W8First& object) { return object.values[2]; }\n"
        "inline void CopyW8Record(W8Record& dest, const W8Record& source) {\n"
        "  dest.payload = source.payload;\n"
        "}\n"
        "template<class T> int ReadDependentMember(T& object) { return object.unknown_dependent; }\n",
        encoding="utf-8",
    )
    sources = [
        repository / name
        for name in ("first.cpp", "second.cpp", "empty.cpp", "repeat.cpp")
    ]
    clang_cl = _clang_cl(repository)
    for path, target in zip(sources, ("WIZ8", "SURRENDER")):
        local_type = "int" if target == "WIZ8" else "long"
        path.write_text(
            '#include "owner.h"\n'
            "extern int gShared;\n"
            "static int gLocal = 1;\n"
            f"int g{target} = 0;\n"
            f"// FUNCTION: {target} 0x00401000\n"
            f"int {target}() {{ {local_type} scratch = 0; "
            + (
                "SurrenderOnly surrender{}; "
                "return gShared + gLocal + gSURRENDER + surrender.unknown_04 + (int)scratch; }\n"
                if target == "SURRENDER"
                else "return gShared + gLocal + gWIZ8 + (int)scratch; }\n"
            ),
            encoding="utf-8",
        )
    sources[2].write_text(
        "// A successful unit may emit no records.\n", encoding="utf-8"
    )
    sources[3].write_text(
        '#include "owner.h"\n'
        "// FUNCTION: WIZ8_REPEAT 0x00401004\n"
        "int WIZ8_REPEAT() { return 0; }\n",
        encoding="utf-8",
    )
    database = repository / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {
                    "directory": str(repository),
                    "file": str(path),
                    "arguments": [
                        clang_cl,
                        "--target=i686-pc-windows-msvc",
                        "/c",
                        str(path),
                    ],
                }
                for path in sources
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache with spaces"
    previous_root = os.environ.get("RECCMP_SOURCE_ROOT")
    os.environ["RECCMP_SOURCE_ROOT"] = str(repository)

    def collect():
        return SourceIndex.from_compile_database(
            repository,
            database,
            {
                "WIZ8": [header, sources[0], sources[2], sources[3]],
                "SURRENDER": [sources[1]],
            },
            cache_dir=cache,
            jobs=2,
        )

    try:
        index = collect()
        profile = json.loads((cache / "profile.json").read_text())
        assert profile["miss_reasons"] == {"new": 4}
        assert profile["records"]["declaration"] > 0
        assert profile["indexer_totals_ms"]["frontend_ms"] > 0
        assert "owner.h" in index.unit_dependencies["first.cpp"]
        owners = [item for item in index.classes if item.qualified_name == "Owner"]
        assert len(owners) == 2
        assert {item.target for item in owners} == {"WIZ8", "SURRENDER"}
        assert all(item.asserted_size == 20 for item in owners)
        assert [field.pointer_depth for field in owners[0].fields] == [2, 1, 0, 0]
        dependent_records = [
            item for item in index.classes if item.qualified_name == "DependentRecord"
        ]
        assert len(dependent_records) == 2
        assert all(item.size is None for item in dependent_records)
        assert all(item.fields[0].offset is None for item in dependent_records)
        assert index.functions_by_address(target="WIZ8")[0x401000].name == "WIZ8"
        assert (
            index.functions_by_address(target="SURRENDER")[0x401000].name == "SURRENDER"
        )
        variables = {
            (item.target, item.qualified_name): item for item in index.variables
        }
        assert variables[("WIZ8", "gWIZ8")].definition_kind == "definition"
        assert variables[("WIZ8", "gWIZ8")].is_external
        assert variables[("WIZ8", "gShared")].definition_kind == "declaration"
        assert variables[("WIZ8", "gShared")].is_external
        assert variables[("SURRENDER", "gSURRENDER")].is_external
        assert not any(item.qualified_name == "gLocal" for item in index.variables)
        assert not index.conflicts
        declaration = index.functions_by_address(target="WIZ8")[0x401000].declaration
        assert declaration is not None
        assert declaration.linkage == "external"
        wiz8_uses = index.for_target("WIZ8").member_uses
        surrender_uses = index.for_target("SURRENDER").member_uses
        first_uses = [
            item
            for item in wiz8_uses
            if item.owner == "W8First"
            and item.name == "unknown_04"
            and item.function == "ReadW8Field"
        ]
        assert len(first_uses) == 1
        assert first_uses[0].owner_identity
        assert first_uses[0].field_usr
        assert first_uses[0].offset_bits == 0
        assert first_uses[0].extent_bits == 32
        assert first_uses[0].operations == ("read",)
        first_field_identity = first_uses[0].field_identity
        assert any(
            item.owner == "W8Second" and item.name == "unknown_04" for item in wiz8_uses
        )
        assert not any(item.owner == "SurrenderOnly" for item in wiz8_uses)
        assert any(item.owner == "SurrenderOnly" for item in surrender_uses)
        assert any(item.owner_status == "unknown" for item in wiz8_uses)
        assert any(
            "array-index" in item.operations and "write" in item.operations
            for item in wiz8_uses
        )
        assert any(
            item.owner == "W8First"
            and item.name == "values"
            and item.array_indices
            and not item.array_indices[0].constant
            for item in wiz8_uses
        )
        assert any(
            item.owner == "W8First"
            and item.name == "values"
            and item.array_indices
            and item.array_indices[0].constant
            and item.array_indices[0].value == "2"
            for item in wiz8_uses
        )
        assert any(
            item.function == "ConvertW8Field"
            and any(
                conversion.destination_type == "double"
                for conversion in item.conversions
            )
            for item in wiz8_uses
        )
        assert any(
            item.function == "AddressW8Field" and "address-taken" in item.operations
            for item in wiz8_uses
        )
        assert any(
            "copy-memory-source" in item.operations
            for item in wiz8_uses
            if item.owner == "W8Record"
        )
        assert any(
            "copy-memory-destination" in item.operations
            for item in wiz8_uses
            if item.owner == "W8Record"
        )
        assert (
            SourceIndex.from_dict(json.loads(json.dumps(index.to_dict()))).to_dict()
            == index.to_dict()
        )
        tu_cache = cache / "tu"
        before = sorted(path.stat().st_mtime_ns for path in tu_cache.glob("*.ndjson"))
        assert before
        assert collect().to_dict() == index.to_dict()
        after = sorted(path.stat().st_mtime_ns for path in tu_cache.glob("*.ndjson"))
        assert after == before
        assert json.loads((cache / "profile.json").read_text())["units"] == {
            "hits": 4,
            "misses": 0,
        }
        # Collectors sharing one cache do not wait for each other.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: collect().to_dict(), range(2)))
        assert results == [index.to_dict()] * 2
        header.write_text(
            header.read_text().replace("**pointers", "*pointers"), encoding="utf-8"
        )
        refreshed = collect()
        reasons = json.loads((cache / "profile.json").read_text())["misses"]
        assert set(reasons) == {"first.cpp", "second.cpp", "repeat.cpp"}
        assert all(
            reason.startswith("dependency_changed:") and reason.endswith("owner.h")
            for reason in reasons.values()
        )
        assert all(
            item.fields[0].pointer_depth == 1
            for item in refreshed.classes
            if item.qualified_name == "Owner"
        )
        header.write_text(
            header.read_text()
            .replace(
                "struct W8First { int unknown_04; int values[4]; }",
                "struct W8First { int state; int values[4]; }",
            )
            .replace("int value = object.unknown_04;", "int value = object.state;")
            .replace(
                "static_cast<double>(object.unknown_04)",
                "static_cast<double>(object.state)",
            )
            .replace("&object.unknown_04", "&object.state"),
            encoding="utf-8",
        )
        renamed = collect()
        renamed_uses = [
            item
            for item in renamed.for_target("WIZ8").member_uses
            if item.owner == "W8First"
            and item.name == "state"
            and item.function == "ReadW8Field"
        ]
        assert len(renamed_uses) == 1
        assert renamed_uses[0].field_identity == first_field_identity
        sources[0].write_text("this is not valid C++;\n", encoding="utf-8")
        with pytest.raises(SourceIndexError, match="first.cpp"):
            collect()
    finally:
        if previous_root is None:
            os.environ.pop("RECCMP_SOURCE_ROOT", None)
        else:
            os.environ["RECCMP_SOURCE_ROOT"] = previous_root
