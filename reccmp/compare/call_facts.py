"""Call facts stated by MSVC decorated names."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from reccmp.call_facts import CallFacts, convention_facts
from reccmp.cvdump.demangler import demangle_function, type_return_kind

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


def _c_decoration_facts(symbol: str) -> CallFacts:
    """_name (cdecl), _name@N (stdcall), @name@N (fastcall)."""
    if symbol.startswith("_"):
        _, at, count = symbol[1:].partition("@")
        if not at:
            return convention_facts("cdecl")
        facts = convention_facts("stdcall")
        return replace(facts, stack_cleanup=int(count)) if count.isdigit() else facts
    if symbol.startswith("@") and "@" in symbol[1:]:
        return convention_facts("fastcall")
    return CallFacts()


def _is_record_value(type_name: str) -> bool:
    """Whether a demangled type is a class, struct or union by value."""
    name = " ".join(w for w in type_name.split() if w not in ("const", "volatile"))
    return name.startswith(("class ", "struct ", "union ")) and not name.endswith(
        ("*", "&")
    )


def mangled_facts(symbol: str) -> CallFacts:
    """Everything a decorated function name states about calling it."""
    if not symbol.startswith("?"):
        return _c_decoration_facts(symbol)
    function = demangle_function(symbol)
    if function is None:
        return CallFacts()
    facts = convention_facts(function.convention)
    if function.return_type is not None:
        facts = replace(facts, return_kind=type_return_kind(function.return_type))
    returns_record = function.return_type is not None and _is_record_value(
        function.return_type
    )
    # A record returned by value may come back through a hidden pointer
    # argument the callee pops, depending on the record: unknown here.
    if (
        facts.stack_cleanup is None
        and function.convention != "fastcall"
        and not returns_record
    ):
        sizes = [_parameter_bytes(parameter) for parameter in function.parameters]
        if all(size is not None for size in sizes):
            facts = replace(
                facts, stack_cleanup=sum(size for size in sizes if size is not None)
            )
    return facts


def import_facts(decorated_names: Iterable[str]) -> dict[str, CallFacts]:
    """Facts per import name (as the import table spells it), from the
    ``__imp_`` public symbols of the recompiled PDB."""
    result: dict[str, CallFacts] = {}
    for decorated in decorated_names:
        if not decorated.startswith("__imp_"):
            continue
        symbol = decorated[len("__imp_") :]
        facts = mangled_facts(symbol)
        if symbol.startswith("?"):
            result[symbol] = facts
        elif symbol.startswith("_"):
            result[symbol[1:].partition("@")[0]] = facts
    return result
