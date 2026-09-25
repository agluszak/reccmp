"""What a caller may assume about a callee: its register arguments, how
many argument bytes it removes from the stack, and how it returns a value.

One record serves the effective-match verifier and the witness model. Each
field is filled from the strongest producer that knows it: PDB type records,
then the reconstruction's Clang declarations, then decorated names
(``reccmp.compare.call_facts``). ``None`` means unknown.
"""

from __future__ import annotations

from dataclasses import dataclass


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
# argument on the stack; thiscall reads the receiver from ecx. fastcall reads
# its first two register-sized arguments from ecx and edx, so which of them
# carry arguments depends on the parameters: unknown from the convention
# alone. Keys are the PDB spellings and the ones recovered from decorated
# names.
_REGISTERS: dict[str, tuple[bool | None, bool | None]] = {
    "C Near": (False, False),
    "STD Near": (False, False),
    "ThisCall": (True, False),
    "Fast Near": (None, None),
    "cdecl": (False, False),
    "stdcall": (False, False),
    "thiscall": (True, False),
    "fastcall": (None, None),
}
_CALLER_CLEANS = frozenset({"C Near", "cdecl"})


def convention_facts(convention: str | None) -> CallFacts:
    """Register usage (and cdecl's zero cleanup) implied by a convention."""
    if convention is None or convention not in _REGISTERS:
        return CallFacts()
    uses_ecx, uses_edx = _REGISTERS[convention]
    return CallFacts(uses_ecx, uses_edx, 0 if convention in _CALLER_CLEANS else None)
