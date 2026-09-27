"""Compiler-backed source ownership model."""

from .records import (
    SourceAbi,
    SourceBaseOffset,
    SourceBaseVtable,
    DeclarationKey,
    SourceClass,
    SourceDeclaration,
    SourceField,
    SourceMemberUse,
    SourceArrayIndex,
    SourceConversion,
    SourceMarker,
    ResolvedField,
)
from .observations import SourceIndexError, TranslationUnitRecords
from .index import SourceCollector, SourceIndex, keyed
from .commands import record_command
from .variables import SourceConflict, SourceConflictVariant, SourceVariable

__all__ = [
    "DeclarationKey",
    "keyed",
    "SourceAbi",
    "SourceBaseOffset",
    "SourceBaseVtable",
    "SourceClass",
    "SourceCollector",
    "SourceConflict",
    "SourceConflictVariant",
    "SourceDeclaration",
    "SourceField",
    "SourceIndex",
    "SourceIndexError",
    "SourceMemberUse",
    "SourceArrayIndex",
    "SourceConversion",
    "SourceMarker",
    "ResolvedField",
    "SourceVariable",
    "TranslationUnitRecords",
    "record_command",
]
