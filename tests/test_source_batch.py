"""Exercise the actual collector inside the pinned LLVM 19 analysis environment."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from reccmp.call_facts import CallFacts
from reccmp.parser import DecompCodebase
from reccmp.parser.error import AlertCode
from reccmp.parser.marker import MarkerType
from reccmp.source import DeclarationKey, SourceIndex, SourceIndexError
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
// GLOBAL: TEST 0x3020
/* a block comment and a pragma are not blank lines */
#pragma bss_seg(".data")
int g_pragma = 0;
#pragma bss_seg()

// GLOBAL: TEST 0x3030

int g_spaced = 0;
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
    assert [
        (alert.path.name, alert.line_number)
        for alert in alerts
        if alert.code == AlertCode.UNEXPECTED_BLANK_LINE
    ] == [("widget.cpp", 50)]


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
        owner_keys = {
            key: item
            for key, item in index.classes.items()
            if item.qualified_name == "Owner"
        }
        owners = list(owner_keys.values())
        assert {key.target for key in owner_keys} == {"WIZ8", "SURRENDER"}
        assert all(item.asserted_size == 20 for item in owners)
        assert [field.pointer_depth for field in owners[0].fields] == [2, 1, 0, 0]
        dependent_records = [
            item
            for item in index.classes.values()
            if item.qualified_name == "DependentRecord"
        ]
        assert len(dependent_records) == 2
        assert all(item.size is None for item in dependent_records)
        assert all(item.fields[0].offset is None for item in dependent_records)
        assert index.functions_by_address(target="WIZ8")[0x401000].name == "WIZ8"
        assert (
            index.functions_by_address(target="SURRENDER")[0x401000].name == "SURRENDER"
        )
        variables = {
            (key.target, item.qualified_name): item
            for key, item in index.variables.items()
        }
        assert variables[("WIZ8", "gWIZ8")].definition_kind == "definition"
        assert variables[("WIZ8", "gWIZ8")].is_external
        assert variables[("WIZ8", "gShared")].definition_kind == "declaration"
        assert variables[("WIZ8", "gShared")].is_external
        assert variables[("SURRENDER", "gSURRENDER")].is_external
        assert not any(
            item.qualified_name == "gLocal" for item in index.variables.values()
        )
        assert not index.conflicts
        declaration = index.functions_by_address(target="WIZ8")[0x401000].declaration
        assert declaration is not None
        assert declaration.linkage == "external"
        wiz8_uses = [
            use
            for uses in index.for_target("WIZ8").member_uses.values()
            for use in uses
        ]
        surrender_uses = [
            use
            for uses in index.for_target("SURRENDER").member_uses.values()
            for use in uses
        ]
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
            for item in refreshed.classes.values()
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
            for uses in renamed.for_target("WIZ8").member_uses.values()
            for item in uses
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


_FACTS_SOURCE = """\
struct Point { int x; short y; unsigned char flags; };
struct Big { int a, b, c; };
class Base { public: virtual int Run(int value); int field; };
class Widget : public Base {
public:
  int Run(int value) override;
  static int __stdcall Stat(Point p, double d, char c);
  int __fastcall Fast(double d, int a, int b);
  int Vararg(int n, ...);
  Big MakeBig();
  void operator delete(void* block);
  Point point;
  int count;
};
int Helper(int);
int __fastcall One(int a);
int Widget::Run(int value) {
  int wide = point.flags;
  int sign = point.y;
  Helper(count);
  Base::Run(value);
  Base* self = this;
  return self->Run(wide + sign) + Fast(1.0, 2, 3);
}
int Use(Widget& w) { return w.count; }
struct Small { short a; short b; };
Small MakeSmall(int);
struct NonTrivial { NonTrivial(const NonTrivial&); int v; };
int __stdcall TakesNonTrivial(NonTrivial n, int x);
Big __stdcall MakeBigStd(int x);
int __vectorcall Vec(int a, double b);
struct VBase { int v; };
struct Derived : virtual VBase { Derived(int x); };
struct A { virtual void f(); };
struct B { virtual void f(); };
struct C : A, B { void f() override; };
void CallF(C* c) { c->f(); }
struct Inner { int x; short s; };
struct Outer { Inner inner; Inner* ptr; void M(); };
Outer g_outer;
void Outer::M() {
  inner.x = 1;
  ptr->x = 2;
  g_outer.inner.x = 3;
  unsigned char narrowed = (unsigned char)(inner.s + 1);
  int picked = narrowed ? inner.s : inner.x;
  Helper(inner.s);
}
int Compares(Outer* o, unsigned int n, int k) {
  if (o->inner.s < 65) return 1;
  if (n <= 64u) return 2;
  if (k < n) return 3;
  if (n == 0xffffffffu) return 4;
  return 0;
}
"""


def test_function_facts_come_from_the_compiler(tmp_path: Path) -> None:
    _require_collector()
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "facts.cpp"
    source.write_text(_FACTS_SOURCE, encoding="utf-8")
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
    index = SourceIndex.from_compile_database(
        repository, database, {"TEST": [source]}, cache_dir=tmp_path / "cache", jobs=1
    )

    def facts(name: str) -> CallFacts | None:
        return next(
            item.call
            for item in index.declarations.values()
            if item.qualified_name == name
        )

    # Microsoft x86 conventions, from the parameter types.
    assert facts("Widget::Stat") == CallFacts(False, False, 8 + 8 + 4, "i32")
    # fastcall member: this in ecx, the double on the stack, a in edx, b on the stack
    assert facts("Widget::Fast") == CallFacts(True, True, 12, "i32")
    # variadic member: __cdecl with this on the stack
    assert facts("Widget::Vararg") == CallFacts(False, False, 0, "i32")
    # Clang's ABI lowering decides record returns and arguments:
    # a 12-byte record comes back through a hidden pointer the callee pops,
    assert facts("Widget::MakeBig") == CallFacts(True, False, 4, "unknown")
    assert facts("MakeBigStd") == CallFacts(False, False, 4 + 4, "unknown")
    # a 4-byte one in eax,
    assert facts("MakeSmall") == CallFacts(False, False, 0, "i32")
    # a non-trivial one is passed by value in the argument block (inalloca).
    assert facts("TakesNonTrivial") == CallFacts(False, False, 4 + 4, "i32")
    # Not modelled: unknown rather than guessed.
    assert facts("Vec") is None  # __vectorcall
    assert facts("Derived::Derived") is None  # hidden virtual-base argument
    assert facts("Widget::Run") == CallFacts(True, False, 4, "i32")
    # operator delete is implicitly static: no this, the default convention
    assert facts("Widget::operator delete") == CallFacts(False, False, 0, "void")
    # a fastcall function with one argument leaves edx dead
    assert facts("One") == CallFacts(True, False, 0, "i32")

    run = index.function_facts_for(DeclarationKey("TEST", "?Run@Widget@@UAEHH@Z"))
    assert run is not None
    assert run.call == CallFacts(True, False, 4, "i32")
    extensions = {
        use.name: use.conversions[-1]
        for use in run.accesses
        if use.name in ("flags", "y") and use.conversions
    }
    assert (extensions["flags"].source_bits, extensions["flags"].source_signed) == (
        8,
        False,
    )
    assert (extensions["y"].source_bits, extensions["y"].source_signed) == (16, True)
    assert {use.base.kind for use in run.accesses if use.name == "count"} == {"this"}
    calls = {(call.callee, call.virtual) for call in run.calls}
    assert ("?Helper@@YAHH@Z", False) in calls
    assert ("?Run@Base@@UAEHH@Z", False) in calls  # Base::Run(value): qualified
    (virtual,) = [call for call in run.calls if call.virtual]
    assert virtual.slots == ("?Run@Base@@UAEHH@Z",)
    assert virtual.object_class == "record:Base"
    helper = next(call for call in run.calls if call.callee == "?Helper@@YAHH@Z")
    assert helper.field_arguments[0] is not None  # Helper(count)
    use = index.function_facts_for(DeclarationKey("TEST", "?Use@@YAHAAVWidget@@@Z"))
    assert use is not None and use.accesses[0].base.kind == "parameter"
    assert use.accesses[0].base.index == 0

    # One override can fill a slot of each base.
    call_f = index.function_facts_for(DeclarationKey("TEST", "?CallF@@YAXPAUC@@@Z"))
    assert call_f is not None
    (call,) = call_f.calls
    assert call.slots == ("?f@A@@UAEXXZ", "?f@B@@UAEXXZ")

    method = index.function_facts_for(DeclarationKey("TEST", "?M@Outer@@QAEXXZ"))
    assert method is not None
    leaves = [use for use in method.accesses if use.name in ("x", "s")]

    def line_of(text: str) -> int:
        return next(
            number
            for number, line in enumerate(_FACTS_SOURCE.splitlines(), 1)
            if text in line
        )

    lines = [line_of(text) for text in ("inner.x = 1", "ptr->x", "g_outer.inner")]
    # Roots and field paths: inner.x, ptr->x, g_outer.inner.x
    assert {
        (use.use_line, use.base.kind, use.base.identity, len(use.base.path), use.arrow)
        for use in leaves
        if use.name == "x" and use.use_line in lines
    } == {
        (lines[0], "this", None, 1, False),
        (lines[1], "this", None, 1, True),
        (lines[2], "global", "?g_outer@@3UOuter@@A", 1, False),
    }
    # A field's conversions are its own value's: the promotion of inner.s to
    # int, not the cast of (inner.s + 1) to unsigned char; the same inside a
    # conditional and as a call argument.
    assert {
        tuple(f"{c.source_type}->{c.destination_type}" for c in use.conversions)
        for use in leaves
        if use.name == "s"
    } == {("short->int",)}

    # Comparisons: the type compared in (after the usual conversions) and the
    # operands as written.
    compares = index.function_facts_for(
        DeclarationKey("TEST", "?Compares@@YAHPAUOuter@@IH@Z")
    )
    assert compares is not None
    short_field, unsigned, mixed, all_ones = sorted(
        compares.comparisons, key=lambda c: c.line
    )
    # an unsigned constant keeps its unsigned value
    assert all_ones.operands[1].constant == 0xFFFFFFFF
    assert (short_field.operator, short_field.type, short_field.bits) == (
        "<",
        "int",
        32,
    )
    assert short_field.signed is True
    assert short_field.operands[0].type == "short" and short_field.operands[0].field
    assert short_field.operands[1].constant == 65
    assert (unsigned.operator, unsigned.signed, unsigned.operands[1].constant) == (
        "<=",
        False,
        64,
    )
    # int < unsigned int compares unsigned: the source of a jb, not a jl
    assert (mixed.type, mixed.signed) == ("unsigned int", False)
    assert [operand.type for operand in mixed.operands] == ["int", "unsigned int"]
    assert compares.comparisons_on_line(short_field.line) == (short_field,)
