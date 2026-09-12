"""Variable records and cross-TU type conflicts for the source index."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceVariable:
    """One semantic variable declaration or definition emitted by Clang.

    Only external-linkage variables are collected: TU-local storage has no
    legitimate cross-unit writer/reader disagreement for the consistency gate.
    """

    semantic_id: str
    qualified_name: str
    type: str
    linkage: str
    storage_class: str
    definition_kind: str
    source_file: str
    line: int
    end_line: int
    # Compilation unit that observed this variable (repo-relative main file).
    unit_id: str = ""
    # Link namespace (reccmp target) assigned when observations are partitioned.
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
    """One spelling of a contested symbol and where it was observed."""

    signature: tuple[str, ...]
    locations: tuple[str, ...]


@dataclass(frozen=True)
class SourceConflict:
    """One symbol whose observations inside a link namespace disagree.

    Conflicts are derived after partitioning observations by target: each
    distinct signature retained among that namespace's units becomes a variant.
    Deduplication still keeps one winning record per identity, but the
    disagreement is not a side-channel of a global merge.
    """

    semantic_id: str
    qualified_name: str
    record_kind: str
    variants: tuple[SourceConflictVariant, ...]
    target: str | None = None
