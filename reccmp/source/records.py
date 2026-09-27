"""Immutable compiler fact records used by the source index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reccmp.call_facts import CallFacts


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
    has_this: bool
    is_virtual: bool
    source_file: str
    line: int
    end_line: int
    is_definition: bool
    source_signature: str | None = None
    parameter_references: tuple[bool, ...] = ()
    parameter_reference_forms: tuple[str, ...] = ()
    linkage: str = ""
    storage_class: str = ""
    is_variadic: bool = False
    # How callers call it, under the Microsoft x86 ABI.
    call: CallFacts | None = None

    @property
    def prototype(self) -> str:
        """Render compiler-owned types for display, not ABI synchronization."""
        parameters = ", ".join(self.parameter_types) or "void"
        if self.is_variadic:
            parameters = f"{parameters}, ..." if self.parameter_types else "..."
        prefix = f"{self.return_type} " if self.return_type else ""
        return f"{prefix}{self.qualified_name}({parameters})"

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
class SourceArrayIndex:
    """One source array selector observed at a member use."""

    constant: bool
    value: str | None = None


@dataclass(frozen=True)
class SourceConversion:
    """One Clang conversion surrounding a member expression. For integer,
    enumeration and pointer values it states the widths and whether the
    source is signed: a widening from a signed source sign-extends."""

    kind: str
    source_type: str
    destination_type: str
    source_bits: int | None = None
    source_signed: bool | None = None
    destination_bits: int | None = None


@dataclass(frozen=True)
class SourceAccessStep:
    """One field on the way from an access's root to its object."""

    field: str  # field identity
    arrow: bool  # reached through a pointer (->)


@dataclass(frozen=True)
class SourceAccessBase:
    """The object of a member access or call: its root (``this``,
    ``parameter`` with its index, ``local`` or ``global`` with the
    declaration's identity, ``call``, ``other``) and the fields leading from
    the root to it. ``this->a.b.c`` has root ``this`` and path ``a, b``."""

    kind: str
    index: int | None = None
    identity: str | None = None
    path: tuple[SourceAccessStep, ...] = ()


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceMemberUse:
    """One compiler-resolved field use in a source function body."""

    owner_identity: str | None
    owner_status: str
    owner: str
    field_identity: str
    field_usr: str | None
    name: str
    declaration_file: str
    declaration_line: int
    declaration_column: int
    declaration_offset: int | None
    offset_bits: int | None
    extent_bits: int | None
    offset_bytes: int | None
    extent_bytes: int | None
    declared_type: str
    function_identity: str
    function: str
    function_file: str
    function_line: int
    use_file: str
    use_line: int
    use_column: int
    use_offset: int | None
    operations: tuple[str, ...]
    array_indices: tuple[SourceArrayIndex, ...]
    conversions: tuple[SourceConversion, ...]
    base: SourceAccessBase
    arrow: bool = False  # the access dereferences its base (->)


@dataclass(frozen=True)
class SourceCall:
    """One call in a source function body."""

    callee: str | None  # semantic id of the called declaration
    virtual: bool
    # Per argument: the field identity when it is a plain field read.
    field_arguments: tuple[str | None, ...]
    line: int
    offset: int | None
    # Virtual calls: every declaration introducing a vtable slot the call
    # may use (more than one under multiple inheritance), and the static
    # class of the object.
    slots: tuple[str, ...] = ()
    object_class: str | None = None
    object: SourceAccessBase | None = None


@dataclass(frozen=True)
class SourceComparisonOperand:
    """One side of a comparison, as written (before conversions)."""

    type: str
    field: str | None = None  # field identity, for a plain field read
    constant: int | None = None  # its value, when it is an integer constant


@dataclass(frozen=True)
class SourceComparison:
    """One built-in comparison in a source function body."""

    # pylint: disable=too-many-instance-attributes
    operator: str  # <, <=, >, >=, ==, !=
    # The type compared in, after the usual arithmetic conversions: this is
    # what decides a signed or an unsigned machine comparison.
    type: str
    operands: tuple[SourceComparisonOperand, SourceComparisonOperand]
    line: int
    offset: int | None
    bits: int | None = None  # integers, enumerations and pointers
    signed: bool | None = None
    floating: bool = False


@dataclass(frozen=True)
class SourceFunctionFacts:
    """Facts about one function body beyond its field uses."""

    function: str  # semantic id
    # The explicit calls (CallExpr nodes) in the body: not constructors,
    # destructors or other implicit calls, so not a complete call graph.
    calls: tuple[SourceCall, ...]
    # Built-in comparisons; overloaded comparison operators are calls.
    comparisons: tuple[SourceComparison, ...] = ()


@dataclass(frozen=True)
class FunctionFacts:
    """What the reconstruction's compiler says about one function: how it is
    called, the fields it accesses and the calls it makes. These explain the
    recompiled side; they never prove the original equivalent."""

    key: DeclarationKey
    call: CallFacts | None
    accesses: tuple[SourceMemberUse, ...]
    calls: tuple[SourceCall, ...]  # explicit calls only
    comparisons: tuple[SourceComparison, ...] = ()

    def comparisons_on_line(self, line: int) -> tuple[SourceComparison, ...]:
        return tuple(item for item in self.comparisons if item.line == line)


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
