"""References, register families and Intel-text formatting helpers shared
by the decoder, the operands and the verifier.

Leaf module: no imports from ``ir`` or the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable


class Reject(Exception):
    """The two sequences could not be proven equivalent."""


@dataclass(frozen=True)
class Reference:
    """A relocated or absolute address after sanitization.

    ``display`` is the human/placeholder token (``<OFFSET1>``, a symbol).
    ``identity`` is what proofs compare: a named/paired entity, or a
    side-local unresolved address that cannot equal a placeholder from
    the other image merely by occupying the same replacement slot.
    """

    display: str
    identity: Hashable
    entity_type: str | None = None

    def __str__(self) -> str:
        return self.display


@dataclass(frozen=True)
class ResolvedAddress:
    """What an address resolves to, in one lookup: the entity's proof
    identity, and the name to show for it (None when it has no name, and
    the sanitizer shows a placeholder)."""

    name: str | None
    identity: Hashable
    entity_type: str | None = None


# Register families. Writing e.g. `al` produces a new value for the whole
# `a` family so that partial-register writes are never lost.
REGISTERS: dict[str, tuple[str, str]] = {
    **{f"e{r}x": (r, "r32") for r in "abcd"},
    **{f"{r}x": (r, "r16") for r in "abcd"},
    **{f"{r}l": (r, "l8") for r in "abcd"},
    **{f"{r}h": (r, "h8") for r in "abcd"},
    "esi": ("si", "r32"),
    "si": ("si", "r16"),
    "edi": ("di", "r32"),
    "di": ("di", "r16"),
    "ebp": ("bp", "r32"),
    "bp": ("bp", "r16"),
    "esp": ("sp", "r32"),
    "sp": ("sp", "r16"),
}


def format_imm(value: int) -> str:
    """Capstone Intel-syntax immediates: decimal for |n| < 10, else hex."""
    if -9 <= value <= 9:
        return str(value)
    return hex(value)


def split_mnemonic_prefix(mnemonic: str) -> tuple[str, str]:
    """Split Capstone's combined ``rep movsd``-style mnemonic into prefix + op."""
    for candidate in ("repne", "repe", "rep"):
        if mnemonic == candidate:
            return candidate, ""
        prefix = candidate + " "
        if mnemonic.startswith(prefix):
            return candidate, mnemonic[len(prefix) :]
    return "", mnemonic
