"""Immutable compiler fact records used by the source index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DeclarationKey:
    """The identity of a declared entity inside the index.

    A mangled name alone is not one: TU-local functions of different units
    can share it. External entities are identified by (target, semantic id);
    TU-local ones also by the unit that defines them."""

    target: str | None
    semantic_id: str
    unit_id: str | None = None

    def sort_key(self) -> tuple[str, str, str]:
        return (self.target or "", self.semantic_id, self.unit_id or "")

    def to_json(self) -> list[str | None]:
        return [self.target, self.semantic_id, self.unit_id]

    @classmethod
    def from_json(cls, values: Sequence[str | None]) -> "DeclarationKey":
        target, semantic_id, unit_id = values
        assert semantic_id is not None
        return cls(target, semantic_id, unit_id)


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceDeclaration:
    """One semantic function declaration emitted by Clang. A compiler fact:
    which target and unit it belongs to is its DeclarationKey's business."""

    semantic_id: str
    qualified_name: str
    semantic_kind: str
    calling_convention: str
    return_type: str
    parameter_types: tuple[str, ...]
    owning_class: str | None
    source_file: str
    line: int
    end_line: int
    is_definition: bool
    linkage: str = ""
    storage_class: str = ""
    is_variadic: bool = False

    @property
    def is_external(self) -> bool:
        """Genuinely cross-TU linkage."""
        return self.linkage == "external"

    @property
    def signature(self) -> tuple[str, ...]:
        """The type identity a cross-TU consistency gate compares."""
        return (
            self.semantic_kind,
            self.calling_convention,
            self.return_type,
            *self.parameter_types,
            self.linkage,
            "..." if self.is_variadic else "",
        )

    def key(self, target: str | None, unit_id: str) -> DeclarationKey:
        """Its identity when observed by ``unit_id`` in ``target``."""
        return DeclarationKey(
            target, self.semantic_id, None if self.is_external else unit_id
        )


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceField:
    """One direct non-static source field emitted by Clang."""

    name: str
    type: str
    source_file: str
    line: int
    pointer_depth: int
    # Physical storage: scalar, pointer, reference, embedded_record, array.
    storage_kind: str
    offset: int | None = None
    size: int | None = None
    bitfield_width: int | None = None
    bitfield_offset: int | None = None
    # ``record:Qualified::Name`` when the field type is (or points to) a record.
    record_semantic_id: str | None = None
    array_element_type: str | None = None
    array_stride: int | None = None
    array_count: int | None = None
    # Physical storage of one array element: scalar, pointer, reference,
    # embedded_record, array.
    array_element_kind: str | None = None


@dataclass(frozen=True)
class SourceAbi:
    """Compilation ABI facts shared by translation units in this index."""

    target_triple: str
    pointer_width: int
    ms_abi: bool


@dataclass(frozen=True)
class SourceBaseOffset:
    """Byte offset of one direct base subobject within a complete class."""

    name: str
    semantic_id: str
    offset: int


@dataclass(frozen=True)
class SourceBaseVtable:
    """One vtable installed for a polymorphic base subobject."""

    address: int
    base_class: str


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceClass:
    """One complete C++ record definition emitted by Clang."""

    semantic_id: str
    qualified_name: str
    bases: tuple[str, ...]
    fields: tuple[SourceField, ...]
    virtual_declarations: tuple[str, ...]
    source_file: str
    line: int
    end_line: int
    asserted_size: int | None = None
    vtable_address: int | None = None
    base_vtables: tuple[SourceBaseVtable, ...] = ()
    size: int | None = None
    alignment: int | None = None
    base_offsets: tuple[SourceBaseOffset, ...] = ()
    # True when Clang layout is present and consistent across observations;
    # False on layout conflict or asserted_size mismatch; None when unknown.
    layout_trusted: bool | None = None


@dataclass(frozen=True)
class ResolvedField:
    """A byte offset resolved to a leaf field within a class hierarchy."""

    root_class: str
    path: tuple[str, ...]
    leaf: SourceField
    absolute_offset: int
    relative_offset: int  # within leaf storage
    base_chain: tuple[str, ...]


@dataclass(frozen=True)
class SourceMarker:
    """A reccmp marker and its compiler-owned declaration, when applicable."""

    address: int
    marker_kind: str
    source_file: str
    line: int
    declaration: SourceDeclaration | None
    marker_name: str | None = None
    folded: bool = False
    target: str | None = None
    declaration_key: DeclarationKey | None = None

    @property
    def name(self) -> str:
        """Compiler identity, or the name attached to a non-body marker."""
        return (
            self.declaration.qualified_name
            if self.declaration
            else self.marker_name or ""
        )
