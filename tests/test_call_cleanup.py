"""Callee stack cleanup from decorated and mangled import names."""

import pytest

from reccmp.compare.call_cleanup import import_cleanup, mangled_cleanup


@pytest.mark.parametrize(
    "symbol, cleanup",
    [
        ("?getChildCount@srNode@@QBEJXZ", 0),  # thiscall, no parameters
        ("?setPos@srNode@@QAEXMMM@Z", 12),  # thiscall, three floats
        ("?postProcess@srMaterial@@UAEXAAVsrVertexPipe@@@Z", 4),  # class &
        ("?foo@@YGXHPAD0@Z", 12),  # stdcall int, char *, back-ref to int
        ("?make@A@@SGPAV1@HN@Z", 12),  # static stdcall int, double
        ("?f@@YGXPAUtagRECT@@H@Z", 8),  # struct pointer, int
        ("?bar@@YAXH@Z", 0),  # cdecl: the caller cleans up
        ("?g@@YGXVValue@@@Z", None),  # class by value: size unknown
        ("?t@?$List@H@@QAEXH@Z", None),  # template: not handled
    ],
)
def test_mangled_cleanup(symbol: str, cleanup: int | None):
    assert mangled_cleanup(symbol) == cleanup


def test_import_cleanup_from_publics():
    assert import_cleanup(
        [
            "__imp__RegOpenKeyExA@20",
            "__imp___wtoi",
            "__imp_?setPos@srNode@@QAEXMMM@Z",
            "_NotAnImport@8",
        ]
    ) == {"RegOpenKeyExA": 20, "_wtoi": 0, "?setPos@srNode@@QAEXMMM@Z": 12}
