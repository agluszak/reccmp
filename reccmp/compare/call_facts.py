"""What a caller may assume about a callee: its register arguments, how
many argument bytes it removes from the stack, and how it returns a value.

One record serves the effective-match verifier and the witness model. Each
field is filled from the strongest producer that knows it: PDB type records
first, decorated names second. ``None`` means unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from reccmp.cvdump.demangler import demangle_function, type_return_kind


@dataclass(frozen=True)
class CallFacts:
    uses_ecx: bool | None = None
    uses_edx: bool | None = None
    stack_cleanup: int | None = None
    return_kind: str = "unknown"

    def merged(self, weaker: "CallFacts") -> "CallFacts":
        """These facts, with unknown fields taken from a weaker producer."""
        return CallFacts(
            self.uses_ecx if self.uses_ecx is not None else weaker.uses_ecx,
            self.uses_edx if self.uses_edx is not None else weaker.uses_edx,
            (
                self.stack_cleanup
                if self.stack_cleanup is not None
                else weaker.stack_cleanup
            ),
            (self.return_kind if self.return_kind != "unknown" else weaker.return_kind),
        )

    def agreed(self, other: "CallFacts") -> "CallFacts":
        """The facts two callees sharing one name both have."""
        return CallFacts(
            self.uses_ecx if self.uses_ecx == other.uses_ecx else None,
            self.uses_edx if self.uses_edx == other.uses_edx else None,
            self.stack_cleanup if self.stack_cleanup == other.stack_cleanup else None,
            self.return_kind if self.return_kind == other.return_kind else "unknown",
        )


# Register arguments by calling convention: cdecl and stdcall take every
# argument on the stack; thiscall reads the receiver from ecx; fastcall reads
# its first two register-sized arguments from ecx and edx. Keys are the PDB
# spellings and the ones recovered from decorated names.
_REGISTERS = {
    "C Near": (False, False),
    "STD Near": (False, False),
    "ThisCall": (True, False),
    "Fast Near": (True, True),
    "cdecl": (False, False),
    "stdcall": (False, False),
    "thiscall": (True, False),
    "fastcall": (True, True),
}
_CALLER_CLEANS = frozenset({"C Near", "cdecl"})


def convention_facts(convention: str | None) -> CallFacts:
    """Register usage (and cdecl's zero cleanup) implied by a convention."""
    if convention is None or convention not in _REGISTERS:
        return CallFacts()
    uses_ecx, uses_edx = _REGISTERS[convention]
    return CallFacts(uses_ecx, uses_edx, 0 if convention in _CALLER_CLEANS else None)


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
    if facts.stack_cleanup is None and function.convention != "fastcall":
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
