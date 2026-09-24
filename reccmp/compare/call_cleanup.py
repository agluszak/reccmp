"""How many argument bytes a callee removes from the stack.

The witness model does not run callees, so it must apply their stack
cleanup itself. For imports this comes from the decorated names in the
recompiled PDB (``__imp__Name@N``) or, for C++ exports, from the mangled
name's calling convention and parameter list.
"""

from __future__ import annotations

from typing import Iterable

# Parameter sizes on the stack for MSVC mangled type codes.
_SIMPLE = {
    "C": 4,  # signed char
    "D": 4,  # char
    "E": 4,  # unsigned char
    "F": 4,  # short
    "G": 4,  # unsigned short
    "H": 4,  # int
    "I": 4,  # unsigned int
    "J": 4,  # long
    "K": 4,  # unsigned long
    "M": 4,  # float
    "N": 8,  # double
    "O": 8,  # long double (MSVC: same as double)
}
_EXTENDED = {"_J": 8, "_K": 8, "_N": 4, "_W": 4}
# Member functions with a this-pointer cv qualifier before the convention.
_INSTANCE_ACCESS = frozenset("ABEFIJMNQRUV")
_STATIC_ACCESS = frozenset("CDKLST")
_CONVENTIONS = {
    "A": "cdecl",
    "B": "cdecl",
    "E": "thiscall",
    "F": "thiscall",
    "G": "stdcall",
    "H": "stdcall",
}


class _Unsupported(Exception):
    pass


def _name_end(code: str, pos: int) -> int:
    """Position after a qualified name: fragments ``name@`` or back
    references ``0``-``9``, terminated by ``@``."""
    while True:
        char = code[pos]
        if char == "@":
            return pos + 1
        if char.isdigit():
            pos += 1
        elif char == "?":
            raise _Unsupported  # template or special name
        else:
            end = code.find("@", pos)
            if end < 0:
                raise _Unsupported
            pos = end + 1


def _type(code: str, pos: int, params: list[int]) -> tuple[int, int]:
    """Size of the type at ``pos`` and the position after it."""
    # pylint: disable=too-many-return-statements
    char = code[pos]
    if char in _SIMPLE:
        return _SIMPLE[char], pos + 1
    if char == "_":
        size = _EXTENDED.get(code[pos : pos + 2])
        if size is None:
            raise _Unsupported
        return size, pos + 2
    if char.isdigit():
        index = int(char)
        if index >= len(params):
            raise _Unsupported
        return params[index], pos + 1
    if char in "PQRSAB":
        # Pointer or reference: cv qualifier, then the pointee.
        if pos + 1 >= len(code) or code[pos + 1] not in "ABCD":
            raise _Unsupported
        if code[pos + 2] in "VU":
            return 4, _name_end(code, pos + 3)
        _, after = _type(code, pos + 2, params)
        return 4, after
    if char in "VU":
        raise _Unsupported  # class or struct by value: size unknown here
    if char == "W":
        return 4, _name_end(code, pos + 2)  # enum
    if char == "X":
        return 0, pos + 1
    raise _Unsupported


def mangled_cleanup(symbol: str) -> int | None:
    """Argument bytes the callee pops, for an MSVC-mangled function name."""
    # pylint: disable=too-many-return-statements
    if not symbol.startswith("?") or "?$" in symbol:
        return None
    _, sep, code = symbol.partition("@@")
    if not sep or not code:
        return None
    try:
        access, pos = code[0], 1
        if access in _INSTANCE_ACCESS:
            pos += 1  # cv qualifier of this
        elif access not in _STATIC_ACCESS and access != "Y":
            return None
        convention = _CONVENTIONS.get(code[pos])
        if convention is None:
            return None
        if convention == "cdecl":
            return 0
        pos += 1
        if code[pos] == "?":  # return type qualifier (?A / ?B)
            pos += 2
        _, pos = _type(code, pos, [])
        if code[pos] == "X":
            return 0
        params: list[int] = []
        while code[pos] not in "@Z":
            size, pos = _type(code, pos, params)
            params.append(size)
        return sum(params)
    except (_Unsupported, IndexError):
        return None


def import_cleanup(decorated_names: Iterable[str]) -> dict[str, int]:
    """Cleanup bytes per import name, from ``__imp_`` public symbols."""
    result: dict[str, int] = {}
    for decorated in decorated_names:
        if not decorated.startswith("__imp_"):
            continue
        name = decorated[len("__imp_") :]
        if name.startswith("?"):
            cleanup = mangled_cleanup(name)
            if cleanup is not None:
                result[name] = cleanup
        elif name.startswith("_"):
            base, at, count = name[1:].partition("@")
            if at and count.isdigit():
                result[base] = int(count)
            elif not at:
                result[base] = 0  # cdecl: the caller cleans up
    return result
