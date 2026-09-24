"""How many argument bytes a callee removes from the stack.

The witness model does not run callees, so it must apply their stack
cleanup itself. For imports this comes from the decorated names in the
recompiled PDB: ``__imp__Name@N`` for C functions, and for C++ exports the
demangled calling convention and parameter types.
"""

from __future__ import annotations

import re
from typing import Iterable

from reccmp.cvdump.demangler import demangle_function

_EIGHT_BYTES = frozenset({"double", "long double", "__int64", "unsigned __int64"})
_TEMPLATE_ARGS = re.compile(r"<[^<>]*>")


def _parameter_bytes(type_name: str) -> int | None:
    """Stack bytes of one parameter; None when its size is unknown."""
    name = " ".join(w for w in type_name.split() if w not in ("const", "volatile"))
    bare = name
    while "<" in bare:
        stripped = _TEMPLATE_ARGS.sub("", bare)
        if stripped == bare:
            return None
        bare = stripped
    if "*" in bare or "&" in bare:
        return 4  # pointers, references, function pointers
    if name == "...":
        return None  # varargs: the callee cannot pop them
    if name.startswith(("class ", "struct ", "union ")):
        return None  # by value: size unknown here
    if name in _EIGHT_BYTES:
        return 8
    return 4  # int-sized scalars, char/short/bool promoted, enums, float


def mangled_cleanup(symbol: str) -> int | None:
    """Argument bytes the callee pops, for an MSVC-mangled function name."""
    function = demangle_function(symbol)
    if function is None:
        return None
    if function.convention == "cdecl":
        return 0  # the caller cleans up
    if function.convention == "fastcall":
        return None  # the first two arguments travel in registers
    sizes = [_parameter_bytes(parameter) for parameter in function.parameters]
    if any(size is None for size in sizes):
        return None
    return sum(size for size in sizes if size is not None)


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
                result[base] = 0  # cdecl
    return result
