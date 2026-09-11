"""Variable records and cross-TU type conflicts for the source index."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceVariable:
    """One semantic variable declaration or definition emitted by Clang."""

    semantic_id: str
    qualified_name: str
    type: str
    linkage: str
    storage_class: str
    definition_kind: str
    source_file: str
    line: int
    end_line: int
    # The link namespace (reccmp target) this record belongs to. Unmangled C
    # symbols with the same spelling can be unrelated across separate binaries,
    # so consistency must join on (target, semantic_id), never the spelling.
    target: str | None = None

    @property
    def signature(self) -> tuple[str, ...]:
        """The type identity a cross-TU consistency gate compares."""
        return (self.type, self.linkage)

    @property
    def is_external(self) -> bool:
        """Genuinely cross-TU linkage. Internal, unique-external (anonymous
        namespace) and unlinked storage never join across units."""
        return self.linkage == "external"


@dataclass(frozen=True)
class SourceConflictVariant:
    """One spelling of a contested symbol and where it was first seen."""

    signature: tuple[str, ...]
    locations: tuple[str, ...]


@dataclass(frozen=True)
class SourceConflict:
    """One symbol whose collected declarations disagree about its type.

    Deduplication keeps one winning record per identity (a definition beats a
    declaration, an initialized definition beats a tentative one), so the
    merged index alone cannot show that two units disagreed: an unmangled
    ``_gThing`` carries no type in the symbol. The collector therefore retains
    every distinct spelling it saw, and cross-TU consistency gates report these
    instead of re-deriving them from the winners.
    """

    semantic_id: str
    qualified_name: str
    record_kind: str
    variants: tuple[SourceConflictVariant, ...]
    target: str | None = None
