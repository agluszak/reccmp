"""Call facts from calling conventions, decorated and mangled names."""

import pytest

from reccmp.compare.call_facts import (
    CallFacts,
    convention_facts,
    import_facts,
    mangled_facts,
)


@pytest.mark.parametrize(
    "symbol, cleanup",
    [
        ("?getChildCount@srNode@@QBEJXZ", 0),  # thiscall, no parameters
        ("?setPos@srNode@@QAEXMMM@Z", 12),  # thiscall, three floats
        ("?postProcess@srMaterial@@UAEXAAVsrVertexPipe@@@Z", 4),  # class &
        ("?foo@@YGXHPAD0@Z", 12),  # stdcall int, char *, back-ref to char *
        ("?make@A@@SGPAV1@HN@Z", 12),  # static stdcall int, double
        ("?f@@YGXPAUtagRECT@@H@Z", 8),  # struct pointer, int
        ("?bar@@YAXH@Z", 0),  # cdecl: the caller cleans up
        ("?g@@YGXVValue@@@Z", None),  # class by value: size unknown
        ("?t@?$List@H@@QAEXH@Z", 4),  # member of a template class
        ("?d@@YGXN_J@Z", 16),  # double, __int64
        ("?v@@YGXHZZ", None),  # varargs
        ("_RegOpenKeyExA@20", 20),  # stdcall C decoration
        ("__wtoi", 0),  # cdecl C decoration
        ("@Fast@8", None),  # fastcall: registers carry part of it
    ],
)
def test_mangled_cleanup(symbol: str, cleanup: int | None):
    assert mangled_facts(symbol).stack_cleanup == cleanup


def test_mangled_registers_and_return_kind():
    assert mangled_facts("?setPos@srNode@@QAEXMMM@Z") == CallFacts(
        True, False, 12, "void"
    )
    assert mangled_facts("?count@@YGJXZ") == CallFacts(False, False, 0, "i32")
    assert mangled_facts("@Fast@8") == CallFacts(True, True, None, "unknown")
    assert mangled_facts("plain") == CallFacts()


def test_stronger_facts_win_field_by_field():
    pdb = CallFacts(uses_ecx=True, uses_edx=False, return_kind="i8")
    assert pdb.merged(CallFacts(False, True, 8, "i32")) == CallFacts(
        True, False, 8, "i8"
    )
    assert convention_facts("C Near") == CallFacts(False, False, 0)
    assert convention_facts("ThisCall") == CallFacts(True, False, None)


def test_import_facts_from_publics():
    facts = import_facts(
        [
            "__imp__RegOpenKeyExA@20",
            "__imp___wtoi",
            "__imp_?setPos@srNode@@QAEXMMM@Z",
            "_NotAnImport@8",
        ]
    )
    assert {name: item.stack_cleanup for name, item in facts.items()} == {
        "RegOpenKeyExA": 20,
        "_wtoi": 0,
        "?setPos@srNode@@QAEXMMM@Z": 12,
    }
    assert facts["?setPos@srNode@@QAEXMMM@Z"].uses_ecx


@pytest.mark.parametrize(
    "symbol, return_kind, registers",
    [
        ("?setPos@srNode@@QAEXMMM@Z", "void", (True, False)),
        ("?make@A@@SGPAV1@HN@Z", "i32", (False, False)),  # returns a pointer
        ("?getChildCount@srNode@@QBEJXZ", "i32", (True, False)),
        ("??0srNode@@QAE@XZ", "unknown", (True, False)),  # constructor
        ("?f@@YANH@Z", "float", (False, False)),  # double
        ("?g@@YG_JH@Z", "i64", (False, False)),
        ("?b@@YA_NXZ", "i8", (False, False)),  # bool
        ("?c@@YAVValue@@XZ", "unknown", (False, False)),  # class by value
        ("_foo@8", "unknown", (False, False)),
        ("_bar", "unknown", (False, False)),
    ],
)
def test_mangled_return_kind(symbol: str, return_kind: str, registers):
    facts = mangled_facts(symbol)
    assert (facts.return_kind, (facts.uses_ecx, facts.uses_edx)) == (
        return_kind,
        registers,
    )
