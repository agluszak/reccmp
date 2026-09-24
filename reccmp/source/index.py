"""Join reccmp markers to semantic declarations from the Clang AST.

The marker grammar (``reccmp.parser``) owns annotation syntax and addresses.
Clang owns C++ names, function and variable kinds, types, linkage, class
membership, inheritance, virtual declarations, and which declaration each
marker block annotates. This module keeps the indexer's marker blocks, joins
them to the declaration records by semantic id, and writes disposable JSON
projections for downstream tools.

Compiler records arrive as per-TU observations. Link-namespace partitioning,
winner selection, and conflict derivation happen after collection — never by
globally collapsing bare ``semantic_id`` values first.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

# The optional execution backend imports this record model when first used.
# pylint: disable=cyclic-import

import hashlib
import json
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence, TextIO

from reccmp.parser.marker import MarkerType, ProjectAliases
from reccmp.parser.node import ParserFunction, ParserVtable
from reccmp.parser.reader import MarkerBlock, local_paths, read_marker_blocks
from .variables import SourceConflict, SourceConflictVariant, SourceVariable

_VARIABLE_RANK = {"declaration": 0, "tentative": 1, "definition": 2}
_SCHEMA = "reccmp-source-index-v8"


class SourceIndexError(ValueError):
    """The source markers and compiler model cannot be joined unambiguously."""


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceDeclaration:
    """One semantic function declaration emitted by Clang."""

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
    unit_id: str = ""
    target: str | None = None

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

    @property
    def merge_key(self) -> tuple[str, ...]:
        """Identity used when grouping observations inside one link namespace."""
        if self.is_external:
            return (self.semantic_id,)
        return (self.unit_id, self.semantic_id)


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceField:
    """One direct non-static source field emitted by Clang."""

    name: str
    type: str
    source_file: str
    line: int
    pointer_depth: int | None = None
    offset: int | None = None
    size: int | None = None
    bitfield_width: int | None = None
    bitfield_offset: int | None = None
    # ``record:Qualified::Name`` when the field type is (or points to) a record.
    record_semantic_id: str | None = None
    # Physical storage: scalar, pointer, reference, embedded_record, array.
    storage_kind: str | None = None
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
    """One Clang conversion surrounding a member expression."""

    kind: str
    source_type: str
    destination_type: str


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
    unit_id: str = ""
    target: str | None = None


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
    unit_id: str = ""
    target: str | None = None
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

    @property
    def name(self) -> str:
        """Compiler identity, or the name attached to a non-body marker."""
        return (
            self.declaration.qualified_name
            if self.declaration
            else self.marker_name or ""
        )


@dataclass(frozen=True)
class _SizeAssertion:
    unit_id: str
    qualified_name: str
    asserted_size: int


@dataclass(frozen=True)
class _NamespaceRecords:
    """Winners and conflicts derived inside one link namespace."""

    declarations: tuple[SourceDeclaration, ...]
    variables: tuple[SourceVariable, ...]
    classes: tuple[SourceClass, ...]
    member_uses: tuple[SourceMemberUse, ...]
    conflicts: tuple[SourceConflict, ...]
    size_assertions: dict[str, int]
    abi: SourceAbi | None = None


@dataclass
class TranslationUnitRecords:
    """Raw compiler observations from one translation unit.

    No winner selection or conflict tracking happens here — that is derived
    after observations are grouped by link namespace.
    """

    unit_id: str
    declarations: list[SourceDeclaration] = field(default_factory=list)
    variables: list[SourceVariable] = field(default_factory=list)
    classes: list[SourceClass] = field(default_factory=list)
    member_uses: list[SourceMemberUse] = field(default_factory=list)
    size_assertions: list[_SizeAssertion] = field(default_factory=list)
    marker_blocks: list[MarkerBlock] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    abi: SourceAbi | None = None
    # Where the indexer's time went for this unit; not part of the index.
    profile: dict[str, Any] | None = None

    def add(self, record: Mapping[str, Any]) -> None:
        """Store one compiler observation from this unit."""
        values = dict(record)
        kind = values.pop("record")
        if kind in _UNIT_RECORDS:
            self._add_unit_record(kind, values)
        else:
            self._append(kind, self._fact(kind, values))

    def _add_unit_record(self, kind: str, values: dict[str, Any]) -> None:
        """A record about the unit itself, never shared with other units."""
        if kind == "profile":
            self.profile = values
        elif kind == "dependency":
            self.dependencies = [str(path) for path in values.get("files") or ()]
        elif kind == "unit-abi":
            self.abi = SourceAbi(
                target_triple=str(values["target_triple"]),
                pointer_width=int(values["pointer_width"]),
                ms_abi=bool(values["ms_abi"]),
            )
        else:
            self.size_assertions.append(
                _SizeAssertion(
                    unit_id=self.unit_id,
                    qualified_name=str(values["qualified_name"]),
                    asserted_size=int(values["asserted_size"]),
                )
            )

    def _fact(self, kind: str, values: dict[str, Any]) -> Any:
        """A record about source code: identical in every unit that sees it."""
        if kind == "marker-block":
            return MarkerBlock.from_dict(values)
        if kind == "declaration":
            return _declaration_from_dict({**values, "unit_id": self.unit_id})
        if kind == "variable":
            return _variable_from_dict({**values, "unit_id": self.unit_id})
        if kind == "class":
            return _class_from_dict({**values, "unit_id": self.unit_id})
        if kind == "member-use":
            return _member_use_from_dict({**values, "unit_id": self.unit_id})
        raise SourceIndexError(
            f"the source indexer emitted an unknown record: {kind!r}"
        )

    def _append(self, kind: str, fact: Any) -> None:
        if kind == "marker-block":
            self.marker_blocks.append(fact)
        elif kind == "declaration":
            self.declarations.append(fact)
        elif kind == "variable":
            if fact.is_external:
                self.variables.append(fact)
        elif kind == "class":
            self.classes.append(fact)
        else:
            self.member_uses.append(fact)

    @classmethod
    def load(
        cls, path: Path, unit_id: str, pool: RecordPool | None = None
    ) -> "TranslationUnitRecords":
        """Stream one NDJSON artifact into a TU record set. With a pool,
        records already read from another unit are shared, not parsed again:
        most of a unit's records describe headers many units include."""
        unit = cls(unit_id=unit_id)
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                shared = pool.facts.get(line) if pool is not None else None
                if shared is not None:
                    unit._append(*shared)
                    continue
                values = json.loads(line)
                kind = values.pop("record")
                if kind in _UNIT_RECORDS:
                    unit._add_unit_record(kind, values)
                    continue
                fact = unit._fact(kind, values)
                if pool is not None:
                    pool.facts[line] = (kind, fact)
                unit._append(kind, fact)
        return unit

    def extend_stream(self, handle: TextIO) -> None:
        for line in handle:
            if line.strip():
                self.add(json.loads(line))


# Records about a translation unit rather than about source code.
_UNIT_RECORDS = frozenset({"profile", "dependency", "unit-abi", "size-assertion"})


class RecordPool:
    """Parsed source records by their exact artifact line, shared across
    every unit loaded with the pool. A shared record keeps the unit id of
    the first unit that read it."""

    def __init__(self) -> None:
        self.facts: dict[bytes, tuple[str, Any]] = {}


def relative_unit_id(
    repository: Path, main_file: str | Path, compilation_root: Path | None = None
) -> str:
    """Repo-relative identity of a translation unit's main file."""
    path = Path(main_file)
    if compilation_root is not None:
        try:
            path = repository / path.relative_to(compilation_root)
        except ValueError:
            pass
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def derive_namespace(
    units: Sequence[TranslationUnitRecords],
    *,
    target: str | None = None,
    unit_ids: set[str] | None = None,
) -> _NamespaceRecords:
    """Partition TU observations, then derive winners and conflicts."""

    def belongs(unit_id: str) -> bool:
        return unit_ids is None or unit_id in unit_ids

    selected = [unit for unit in units if belongs(unit.unit_id)]

    def distinct(items: Iterable[Any]) -> list[Any]:
        """Each record once: units loaded with a RecordPool share them."""
        return list({id(item): item for item in items}.values())

    declarations = distinct(item for unit in selected for item in unit.declarations)
    variables = distinct(item for unit in selected for item in unit.variables)
    classes = distinct(item for unit in selected for item in unit.classes)
    member_uses = distinct(item for unit in selected for item in unit.member_uses)
    assertions = [item for unit in selected for item in unit.size_assertions]

    derived_declarations, declaration_conflicts = _derive_entities(
        declarations,
        key=lambda item: item.merge_key,
        rank=lambda item: 1 if item.is_definition else 0,
        record_kind="declaration",
        target=target,
    )
    derived_variables, variable_conflicts = _derive_entities(
        variables,
        key=lambda item: (item.semantic_id,),
        rank=lambda item: _VARIABLE_RANK.get(item.definition_kind, 0),
        record_kind="variable",
        target=target,
    )
    derived_classes, class_conflicts = _derive_classes(classes, target=target)
    derived_member_uses = _derive_member_uses(member_uses, target=target)
    size_assertions = _derive_size_assertions(assertions)
    abi = _derive_abi(selected)
    return _NamespaceRecords(
        declarations=derived_declarations,
        variables=derived_variables,
        classes=tuple(
            _apply_asserted_size(item, size_assertions.get(item.qualified_name))
            for item in derived_classes
        ),
        member_uses=derived_member_uses,
        conflicts=declaration_conflicts + variable_conflicts + class_conflicts,
        size_assertions=size_assertions,
        abi=abi,
    )


def merge_marker_blocks(blocks: Iterable[MarkerBlock]) -> tuple[MarkerBlock, ...]:
    """One block per source position; a header is seen by many units."""
    merged: dict[tuple[str, int], MarkerBlock] = {}
    for block in blocks:
        previous = merged.get(block.key)
        merged[block.key] = block if previous is None else previous.merged(block)
    return tuple(merged[key] for key in sorted(merged))


class _RepositoryPaths:
    """Repository-relative spellings of absolute paths, each resolved once:
    every unit lists the same headers."""

    def __init__(self, repository: Path):
        self.root = repository.resolve()
        self._known: dict[str, str | None] = {}

    def relative(self, raw: str) -> str | None:
        if raw not in self._known:
            try:
                self._known[raw] = Path(raw).resolve().relative_to(self.root).as_posix()
            except ValueError:
                self._known[raw] = None
        return self._known[raw]

    def files(self, paths: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted({path for raw in paths if (path := self.relative(raw)) is not None})
        )


def _with_target(item: Any, target: str) -> Any:
    """A copy of a frozen record with ``target`` set, without re-running its
    constructor (``dataclasses.replace`` does, which dominates deriving)."""
    copy = object.__new__(type(item))
    copy.__dict__.update(item.__dict__)
    object.__setattr__(copy, "target", target)
    return copy


def _plain(value: Any) -> Any:
    """JSON-ready form of a record (``dataclasses.asdict`` without its deep
    copies, which dominate writing the index)."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: _plain(item) for key, item in value.__dict__.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_use_key(item: SourceMemberUse) -> tuple[Any, ...]:
    """Deduplicate a repeated header observation without collapsing functions."""
    return (
        item.target or "",
        item.owner_identity or "",
        item.field_identity,
        item.function_identity,
        item.use_file,
        item.use_line,
        item.use_column,
        item.use_offset if item.use_offset is not None else -1,
        item.operations,
        tuple((index.constant, index.value or "") for index in item.array_indices),
        tuple(
            (conversion.kind, conversion.source_type, conversion.destination_type)
            for conversion in item.conversions
        ),
    )


def _derive_member_uses(
    observations: Sequence[SourceMemberUse], *, target: str | None
) -> tuple[SourceMemberUse, ...]:
    """Attach the target namespace and collapse identical repeated-TU rows."""
    unique: dict[tuple[Any, ...], SourceMemberUse] = {}
    for item in observations:
        projected = _with_target(item, target) if target is not None else item
        unique.setdefault(_member_use_key(projected), projected)
    return tuple(unique[key] for key in sorted(unique))


# Test/fixture helper: accumulate observations across units without merging.
class SourceCollector:
    """Fixture helper that gathers ``TranslationUnitRecords`` by unit id."""

    def __init__(self, repository: Path, compilation_root: Path | None = None) -> None:
        self.repository = repository.resolve()
        self.compilation_root = compilation_root
        self.units: dict[str, TranslationUnitRecords] = {}

    def unit_id_for(self, main_file: str | Path) -> str:
        return relative_unit_id(self.repository, main_file, self.compilation_root)

    def collect_record(self, record: Mapping[str, Any], *, unit_id: str = "") -> None:
        self.units.setdefault(unit_id, TranslationUnitRecords(unit_id)).add(record)

    def collect_records(self, records: str, *, unit_id: str = "") -> None:
        for line in records.splitlines():
            if line.strip():
                self.collect_record(json.loads(line), unit_id=unit_id)

    @property
    def variables(self) -> list[SourceVariable]:
        return [item for unit in self.units.values() for item in unit.variables]

    @property
    def member_uses(self) -> list[SourceMemberUse]:
        return [item for unit in self.units.values() for item in unit.member_uses]

    def derive(
        self, *, target: str | None = None, unit_ids: set[str] | None = None
    ) -> _NamespaceRecords:
        return derive_namespace(
            tuple(self.units.values()), target=target, unit_ids=unit_ids
        )


def _derive_entities(
    observations: Sequence[Any],
    *,
    key,
    rank,
    record_kind: str,
    target: str | None,
) -> tuple[tuple[Any, ...], tuple[SourceConflict, ...]]:
    groups: dict[tuple[str, ...], list[Any]] = {}
    for item in observations:
        groups.setdefault(key(item), []).append(item)

    winners: list[Any] = []
    conflicts: list[SourceConflict] = []
    for group in groups.values():
        winner = group[0]
        for item in group[1:]:
            if rank(item) > rank(winner):
                winner = item
        winners.append(_with_target(winner, target) if target is not None else winner)

        variants: dict[tuple[str, ...], list[str]] = {}
        for item in group:
            location = f"{item.source_file}:{item.line}"
            variants.setdefault(item.signature, [])
            if location not in variants[item.signature]:
                variants[item.signature].append(location)
        if len(variants) > 1:
            sample = group[0]
            conflicts.append(
                SourceConflict(
                    semantic_id=sample.semantic_id,
                    qualified_name=sample.qualified_name,
                    record_kind=record_kind,
                    variants=tuple(
                        SourceConflictVariant(
                            signature=signature, locations=tuple(locations)
                        )
                        for signature, locations in variants.items()
                    ),
                    target=target,
                )
            )
    return tuple(winners), tuple(conflicts)


def _layout_identity(source_class: SourceClass) -> tuple:
    """Fingerprint of Clang layout facts used for cross-TU consistency."""
    return (
        source_class.size,
        source_class.alignment,
        tuple(
            (
                field.name,
                field.type,
                field.offset,
                field.size,
                field.bitfield_width,
                field.bitfield_offset,
            )
            for field in source_class.fields
        ),
        tuple((base.name, base.offset) for base in source_class.base_offsets),
    )


def _has_layout_evidence(source_class: SourceClass) -> bool:
    return (
        source_class.size is not None
        or source_class.alignment is not None
        or bool(source_class.base_offsets)
        or any(field.offset is not None for field in source_class.fields)
    )


def _layout_identity_signature(identity: tuple) -> tuple[str, ...]:
    """Flatten a layout identity into a conflict-variant signature."""
    size, alignment, fields, base_offsets = identity
    parts = [f"size={size}", f"align={alignment}"]
    for name, type_name, offset, field_size, bit_width, bit_offset in fields:
        parts.append(
            f"field:{name}:{type_name}:{offset}:{field_size}:{bit_width}:{bit_offset}"
        )
    for name, offset in base_offsets:
        parts.append(f"base:{name}:{offset}")
    return tuple(parts)


def _apply_asserted_size(
    source_class: SourceClass, asserted_size: int | None
) -> SourceClass:
    updated = replace(source_class, asserted_size=asserted_size)
    if (
        updated.size is not None
        and asserted_size is not None
        and updated.size != asserted_size
    ):
        return replace(updated, layout_trusted=False)
    return updated


def _derive_classes(
    observations: Sequence[SourceClass], *, target: str | None
) -> tuple[tuple[SourceClass, ...], tuple[SourceConflict, ...]]:
    groups: dict[str, list[SourceClass]] = {}
    for item in observations:
        groups.setdefault(item.semantic_id, []).append(item)

    winners: list[SourceClass] = []
    conflicts: list[SourceConflict] = []
    for group in groups.values():
        winner = group[0]
        for item in group[1:]:
            if not winner.line and item.line:
                winner = item

        variants: dict[tuple, list[str]] = {}
        for item in group:
            identity = _layout_identity(item)
            location = f"{item.source_file}:{item.line}"
            variants.setdefault(identity, [])
            if location not in variants[identity]:
                variants[identity].append(location)

        layout_trusted: bool | None = None
        if len(variants) > 1:
            layout_trusted = False
            sample = group[0]
            conflicts.append(
                SourceConflict(
                    semantic_id=sample.semantic_id,
                    qualified_name=sample.qualified_name,
                    record_kind="class_layout",
                    variants=tuple(
                        SourceConflictVariant(
                            signature=_layout_identity_signature(identity),
                            locations=tuple(locations),
                        )
                        for identity, locations in variants.items()
                    ),
                    target=target,
                )
            )
        elif _has_layout_evidence(winner):
            layout_trusted = True

        if target is not None:
            winner = replace(winner, target=target, layout_trusted=layout_trusted)
        else:
            winner = replace(winner, layout_trusted=layout_trusted)
        winners.append(winner)
    return tuple(winners), tuple(conflicts)


def _derive_size_assertions(assertions: Sequence[_SizeAssertion]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in assertions:
        previous = sizes.get(item.qualified_name)
        if previous is not None and previous != item.asserted_size:
            raise SourceIndexError(
                f"{item.qualified_name} has conflicting size assertions: "
                f"{previous:#x} and {item.asserted_size:#x}"
            )
        sizes[item.qualified_name] = item.asserted_size
    return sizes


def _derive_abi(units: Sequence[TranslationUnitRecords]) -> SourceAbi | None:
    """Pick a representative ABI; disagreeing units yield no trusted ABI."""
    seen: dict[tuple[str, int, bool], SourceAbi] = {}
    for unit in units:
        if unit.abi is None:
            continue
        key = (unit.abi.target_triple, unit.abi.pointer_width, unit.abi.ms_abi)
        seen[key] = unit.abi
    if len(seen) == 1:
        return next(iter(seen.values()))
    return None


def _command_arguments(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if arguments:
        return [str(item) for item in arguments]
    return shlex.split(str(entry["command"]), posix=True)


def record_command(
    entry: dict[str, Any], indexer: str, clang: str | None = None
) -> list[str]:
    """Normalize a compile-database entry into an indexer driver command."""
    arguments = _command_arguments(entry)
    compiler = clang or arguments[0]
    filtered: list[str] = []
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-c", "/c", "-o", "-MF", "-MT", "-MQ"}:
            skip_next = argument in {"-o", "-MF", "-MT", "-MQ"}
            continue
        # codespell:ignore-begin
        if argument.startswith(("/Fo", "/Fd", "-o")):
            # codespell:ignore-end
            continue
        filtered.append(argument)
    try:
        separator = filtered.index("--")
    except ValueError:
        separator = len(filtered)
    return [
        indexer,
        compiler,
        *filtered[:separator],
        # clang-cl defers primary template bodies until instantiation by
        # default. The source-use index needs the original dependent
        # expressions too, so collect from Clang's eager AST while keeping
        # function ownership rules separate from body availability.
        "-fno-delayed-template-parsing",
        "-fsyntax-only",
        *filtered[separator:],
    ]


def _declaration_from_dict(values: Mapping[str, Any]) -> SourceDeclaration:
    data = dict(values)
    for key in ("parameter_types", "parameter_references", "parameter_reference_forms"):
        data[key] = tuple(data.get(key) or ())
    data.pop("declaration_key", None)
    return SourceDeclaration(**data)


def _variable_from_dict(values: Mapping[str, Any]) -> SourceVariable:
    return SourceVariable(**dict(values))


def _member_use_from_dict(values: Mapping[str, Any]) -> SourceMemberUse:
    data = dict(values)
    data["operations"] = tuple(data.get("operations") or ())
    data["array_indices"] = tuple(
        SourceArrayIndex(
            constant=bool(item.get("constant")),
            value=(str(item["value"]) if item.get("value") is not None else None),
        )
        for item in data.get("array_indices") or ()
    )
    data["conversions"] = tuple(
        SourceConversion(
            kind=str(item.get("kind") or ""),
            source_type=str(item.get("source_type") or ""),
            destination_type=str(item.get("destination_type") or ""),
        )
        for item in data.get("conversions") or ()
    )
    for key in (
        "owner_identity",
        "field_usr",
        "declaration_offset",
        "offset_bits",
        "extent_bits",
        "offset_bytes",
        "extent_bytes",
        "use_offset",
    ):
        if (
            key in data
            and data[key] is not None
            and key not in ("owner_identity", "field_usr")
        ):
            data[key] = int(data[key])
    data.pop("record", None)
    return SourceMemberUse(**data)


def _conflict_from_dict(values: Mapping[str, Any]) -> SourceConflict:
    return SourceConflict(
        semantic_id=str(values["semantic_id"]),
        qualified_name=str(values["qualified_name"]),
        record_kind=str(values["record_kind"]),
        target=values.get("target"),
        variants=tuple(
            SourceConflictVariant(
                signature=tuple(variant.get("signature") or ()),
                locations=tuple(variant.get("locations") or ()),
            )
            for variant in values.get("variants") or ()
        ),
    )


def strip_type_qualifiers(type_name: str) -> str:
    """Reduce a Clang type spelling to a record name usable for layout lookup."""
    name = type_name.strip()
    for prefix in ("const ", "volatile ", "struct ", "class ", "union "):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
    while name.endswith("*") or name.endswith("&"):
        name = name[:-1].rstrip()
    return name


def type_spelling_is_indirection(type_name: str) -> bool:
    """True when the outermost type is a pointer or reference (not an array)."""
    name = type_name.strip()
    for prefix in ("const ", "volatile "):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
    # Array spellings look like ``T [N]`` or ``T[]``; those are containment.
    if name.endswith("]") and "[" in name:
        return False
    return name.endswith("*") or name.endswith("&")


def field_is_indirection(source_field: SourceField) -> bool:
    """Pointer/reference storage is a layout leaf; do not descend physically."""
    if source_field.storage_kind in ("pointer", "reference"):
        return True
    if source_field.storage_kind == "array":
        return source_field.array_element_kind in ("pointer", "reference")
    if (source_field.pointer_depth or 0) > 0:
        return True
    return type_spelling_is_indirection(source_field.type)


def variable_type_is_indirection(type_name: str) -> bool:
    """A ``T*`` / ``T&`` variable stores an address, not a ``T`` aggregate."""
    return type_spelling_is_indirection(type_name)


def _field_from_dict(values: Mapping[str, Any]) -> SourceField:
    data = dict(values)
    for key in (
        "offset",
        "size",
        "bitfield_width",
        "bitfield_offset",
        "pointer_depth",
        "record_semantic_id",
        "storage_kind",
        "array_element_type",
        "array_stride",
        "array_count",
        "array_element_kind",
    ):
        if key in data and data[key] is None:
            continue
        if key not in data:
            data[key] = None
    return SourceField(
        name=str(data["name"]),
        type=str(data["type"]),
        source_file=str(data["source_file"]),
        line=int(data["line"]),
        pointer_depth=data.get("pointer_depth"),
        offset=data.get("offset"),
        size=data.get("size"),
        bitfield_width=data.get("bitfield_width"),
        bitfield_offset=data.get("bitfield_offset"),
        record_semantic_id=data.get("record_semantic_id"),
        storage_kind=data.get("storage_kind"),
        array_element_type=data.get("array_element_type"),
        array_stride=data.get("array_stride"),
        array_count=data.get("array_count"),
        array_element_kind=data.get("array_element_kind"),
    )


def _class_from_dict(values: Mapping[str, Any]) -> SourceClass:
    return SourceClass(
        semantic_id=str(values["semantic_id"]),
        qualified_name=str(values["qualified_name"]),
        bases=tuple(values["bases"]),
        fields=tuple(_field_from_dict(field) for field in values["fields"]),
        virtual_declarations=tuple(values["virtual_declarations"]),
        source_file=str(values["source_file"]),
        line=int(values["line"]),
        end_line=int(values["end_line"]),
        asserted_size=values.get("asserted_size"),
        vtable_address=values.get("vtable_address"),
        base_vtables=tuple(
            SourceBaseVtable(**item) for item in values.get("base_vtables", ())
        ),
        unit_id=str(values.get("unit_id") or ""),
        target=values.get("target"),
        size=values.get("size"),
        alignment=values.get("alignment"),
        base_offsets=tuple(
            SourceBaseOffset(name=str(item["name"]), offset=int(item["offset"]))
            for item in values.get("base_offsets") or ()
        ),
        layout_trusted=values.get("layout_trusted"),
    )


def _marker_projection(marker: SourceMarker) -> dict[str, Any]:
    """Serialize a marker with a declaration key instead of a nested copy."""
    payload = {
        "address": marker.address,
        "marker_kind": marker.marker_kind,
        "source_file": marker.source_file,
        "line": marker.line,
        "marker_name": marker.marker_name,
        "folded": marker.folded,
        "target": marker.target,
        "declaration_key": None,
    }
    if marker.declaration is not None:
        payload["declaration_key"] = [
            marker.declaration.target,
            marker.declaration.semantic_id,
        ]
    return payload


def _join_markers(
    target: str,
    namespace: _NamespaceRecords,
    blocks: Sequence[MarkerBlock],
    *,
    aliases: ProjectAliases | None,
) -> tuple[list[SourceClass], list[SourceMarker]]:
    identity = {block.source_file: PurePath(block.source_file) for block in blocks}
    symbols = [
        symbol
        for result in read_marker_blocks(blocks, identity, aliases=aliases)
        for symbol in result.tokens
        if symbol.module == target.upper()
    ]
    # By location as well as identity: TU-local functions of different files
    # can share a mangled name.
    definitions: dict[tuple[str, str, int], list[SourceDeclaration]] = {}
    for declaration in namespace.declarations:
        if declaration.is_definition:
            definitions.setdefault(
                (declaration.semantic_id, declaration.source_file, declaration.line), []
            ).append(declaration)

    markers: list[SourceMarker] = []
    for method_symbol in symbols:
        if not isinstance(method_symbol, ParserFunction):
            continue
        relative = method_symbol.filename.as_posix()
        marker_declaration: SourceDeclaration | None = None
        # Name-reference markers (TEMPLATE/SYNTHETIC/LIBRARY, and FUNCTION with a
        # name comment e.g. `FUNCTION: X 0x... SYMBOL` + `// ??0foo@@QAE@XZ`)
        # name their entity instead of annotating a definition.
        if (
            method_symbol.type in {MarkerType.FUNCTION, MarkerType.STUB}
            and not method_symbol.is_nameref()
        ):
            # One per definition: a TU-local function in a header has one
            # (identical) winner per including unit.
            candidates = [
                found[0]
                for semantic_id in method_symbol.definitions
                if (
                    found := definitions.get(
                        (semantic_id, relative, method_symbol.line_number)
                    )
                )
            ]
            if len(candidates) != 1:
                raise SourceIndexError(
                    f"{relative}:{method_symbol.line_number}: {method_symbol.type.name} "
                    f"0x{method_symbol.offset:08x} "
                    f"binds to {len(candidates)} function definitions"
                )
            marker_declaration = candidates[0]
        markers.append(
            SourceMarker(
                address=method_symbol.offset,
                marker_kind=method_symbol.type.name,
                source_file=relative,
                line=method_symbol.line_number,
                declaration=marker_declaration,
                folded=method_symbol.is_folded,
                target=target,
                marker_name=(
                    method_symbol.name if marker_declaration is None else None
                ),
            )
        )

    classes = list(namespace.classes)
    class_by_name = {item.qualified_name: index for index, item in enumerate(classes)}
    for vtable_symbol in symbols:
        if not isinstance(vtable_symbol, ParserVtable):
            continue
        relative = vtable_symbol.filename.as_posix()
        index = class_by_name.get(vtable_symbol.name)
        if index is None:
            source_class = SourceClass(
                semantic_id=f"record:{vtable_symbol.name}",
                qualified_name=vtable_symbol.name,
                bases=(),
                fields=(),
                virtual_declarations=(),
                source_file=relative,
                line=vtable_symbol.line_number,
                end_line=vtable_symbol.line_number,
                vtable_address=vtable_symbol.offset,
                target=target,
            )
            classes.append(source_class)
            class_by_name[source_class.qualified_name] = len(classes) - 1
            continue
        source_class = classes[index]
        base_class = vtable_symbol.base_class
        class_names = {
            source_class.qualified_name,
            source_class.qualified_name.rsplit("::", 1)[-1],
        }
        if base_class is not None and base_class not in class_names:
            base_vtable = SourceBaseVtable(vtable_symbol.offset, base_class)
            if base_vtable in source_class.base_vtables:
                raise SourceIndexError(
                    f"{relative}:{vtable_symbol.line_number}: duplicate VTABLE marker "
                    f"for base {base_class}"
                )
            classes[index] = replace(
                source_class,
                base_vtables=(*source_class.base_vtables, base_vtable),
            )
            continue
        if source_class.vtable_address is not None:
            raise SourceIndexError(
                f"{relative}:{vtable_symbol.line_number}: class has more than one "
                "primary VTABLE marker"
            )
        classes[index] = replace(source_class, vtable_address=vtable_symbol.offset)
    return classes, markers


def _unique_class_map(
    classes: Sequence[SourceClass],
    *,
    key,
) -> dict[str, SourceClass]:
    """Build an unscoped lookup that drops names colliding across targets."""
    by_key: dict[str, SourceClass] = {}
    ambiguous: set[str] = set()
    for item in classes:
        map_key = key(item)
        if map_key in ambiguous:
            continue
        previous = by_key.get(map_key)
        if previous is None:
            by_key[map_key] = item
            continue
        if previous.target != item.target:
            del by_key[map_key]
            ambiguous.add(map_key)
    return by_key


class SourceIndex:
    """Canonical marker plus Clang semantic source index."""

    def __init__(
        self,
        *,
        declarations: Iterable[SourceDeclaration],
        classes: Iterable[SourceClass],
        markers: Iterable[SourceMarker],
        variables: Iterable[SourceVariable] = (),
        member_uses: Iterable[SourceMemberUse] = (),
        conflicts: Iterable[SourceConflict] = (),
        abi: SourceAbi | None = None,
        target_abis: Mapping[str, SourceAbi] | None = None,
        marker_blocks: Iterable[MarkerBlock] = (),
        source_digests: Mapping[str, str] | None = None,
        unit_dependencies: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        # Every marker block the compiler saw, for all targets: the marker
        # grammar picks out each target's markers when reading them.
        self.marker_blocks = merge_marker_blocks(marker_blocks)
        # sha256 of every target source file when the index was collected.
        self.source_digests: dict[str, str] = dict(
            sorted((source_digests or {}).items())
        )
        # Repository files each translation unit includes, by unit id.
        self.unit_dependencies: dict[str, tuple[str, ...]] = {
            unit: tuple(sorted(paths))
            for unit, paths in sorted((unit_dependencies or {}).items())
        }
        self.declarations = tuple(
            sorted(declarations, key=lambda item: item.semantic_id)
        )
        self.classes = tuple(sorted(classes, key=lambda item: item.semantic_id))
        self.markers = tuple(
            sorted(markers, key=lambda item: (item.address, item.source_file))
        )
        self.variables = tuple(sorted(variables, key=lambda item: item.semantic_id))
        self.member_uses = tuple(sorted(member_uses, key=_member_use_key))
        self.conflicts = tuple(sorted(conflicts, key=lambda item: item.semantic_id))
        self.abi = abi
        self.target_abis: dict[str, SourceAbi] = dict(target_abis or {})
        # Unscoped name/id maps only retain unambiguous entries. Cross-target
        # collisions must not silently pick a last-wins layout.
        self._classes_by_name = _unique_class_map(
            self.classes, key=lambda item: item.qualified_name
        )
        self._classes_by_semantic_id = _unique_class_map(
            self.classes, key=lambda item: item.semantic_id
        )
        self._classes_by_target_name: dict[tuple[str | None, str], SourceClass] = {
            (item.target, item.qualified_name): item for item in self.classes
        }
        self._classes_by_target_semantic_id: dict[
            tuple[str | None, str], SourceClass
        ] = {(item.target, item.semantic_id): item for item in self.classes}

    def for_target(self, target: str) -> "SourceIndex":
        """Return a view restricted to one link-namespace / reccmp target."""
        abi = self.target_abis.get(target)
        if abi is None and self.abi is not None:
            # Single-ABI indexes (and legacy JSON) may only carry ``abi``.
            targets_present = {
                item.target for item in self.classes if item.target is not None
            }
            targets_present.update(
                item.target for item in self.markers if item.target is not None
            )
            targets_present.update(
                item.target for item in self.variables if item.target is not None
            )
            targets_present.update(
                item.target for item in self.declarations if item.target is not None
            )
            targets_present.update(
                item.target for item in self.member_uses if item.target is not None
            )
            if not targets_present or targets_present == {target}:
                abi = self.abi
        return SourceIndex(
            declarations=(item for item in self.declarations if item.target == target),
            classes=(item for item in self.classes if item.target == target),
            markers=(item for item in self.markers if item.target == target),
            variables=(item for item in self.variables if item.target == target),
            member_uses=(item for item in self.member_uses if item.target == target),
            conflicts=(item for item in self.conflicts if item.target == target),
            abi=abi,
            target_abis={target: abi} if abi is not None else {},
            marker_blocks=self.marker_blocks,
            source_digests=self.source_digests,
            unit_dependencies=self.unit_dependencies,
        )

    def stale_sources(self, paths: Iterable[PurePath]) -> list[PurePath]:
        """Source files that changed, or appeared, since the index was collected."""
        paths = list(paths)
        known = {
            path: relative
            for relative, path in local_paths(self.source_digests, paths).items()
        }
        return [
            path
            for path in paths
            if path not in known
            or source_digest(Path(path)) != self.source_digests[known[path]]
        ]

    def class_named(
        self, qualified_name: str, *, target: str | None = None
    ) -> SourceClass | None:
        """Return the indexed class with this qualified name, if present."""
        if target is not None:
            return self._classes_by_target_name.get((target, qualified_name))
        return self._classes_by_name.get(qualified_name)

    def class_for_semantic_id(
        self, semantic_id: str, *, target: str | None = None
    ) -> SourceClass | None:
        """Return the indexed class for a ``record:…`` semantic id."""
        if target is not None:
            return self._classes_by_target_semantic_id.get((target, semantic_id))
        return self._classes_by_semantic_id.get(semantic_id)

    def _lookup_nested_class(
        self, source_field: SourceField, type_spelling: str
    ) -> str | None:
        """Prefer Clang ``record_semantic_id``; fall back to qualifier stripping.

        Only for *embedded* record storage. Pointer/reference fields keep the
        pointee id for metadata but are not physical containment.
        """
        if (
            source_field.storage_kind == "array"
            and source_field.array_element_kind
            not in (
                None,
                "embedded_record",
            )
        ):
            return None
        if field_is_indirection(source_field):
            return None
        if source_field.record_semantic_id:
            nested = self._classes_by_semantic_id.get(source_field.record_semantic_id)
            if nested is not None:
                return nested.qualified_name
        stripped = strip_type_qualifiers(type_spelling)
        if stripped in self._classes_by_name:
            return stripped
        return None

    def field_at(self, qualified_name: str, offset: int) -> SourceField | None:
        """Leaf field covering ``offset`` bytes within the class layout."""
        resolved = self.resolve_field(qualified_name, offset)
        return resolved.leaf if resolved is not None else None

    def field_path_at(self, qualified_name: str, offset: int) -> str | None:
        """Dotted field path for a byte offset, including base subobjects."""
        resolved = self.resolve_field(qualified_name, offset)
        if resolved is None:
            return None
        return ".".join(resolved.path) if resolved.path else resolved.leaf.name

    def resolve_field(self, qualified_name: str, offset: int) -> ResolvedField | None:
        """Resolve ``offset`` to a leaf field with absolute hierarchy offsets.

        Overlapping bitfields at the same byte return ``None`` — byte-level
        queries cannot pick a unique leaf without bit-position information.
        """
        return self._resolve_field(
            qualified_name,
            offset,
            root_class=qualified_name,
            base_chain=(),
            path_prefix=(),
            abs_base=0,
        )

    def _bitfields_covering(
        self, source_class: SourceClass, offset: int
    ) -> list[SourceField]:
        covering: list[SourceField] = []
        for item in source_class.fields:
            if item.bitfield_width is None or item.offset is None:
                continue
            if item.size is None:
                covers = item.offset == offset
            else:
                covers = item.offset <= offset < item.offset + item.size
            if covers:
                covering.append(item)
        return covering

    def _layout_usable(self, source_class: SourceClass) -> bool:
        """Trusted layout evidence for this class (checked on every visit)."""
        return (
            source_class.layout_trusted is not False
            and source_class.size is not None
            and (
                any(field.offset is not None for field in source_class.fields)
                or bool(source_class.base_offsets)
                or source_class.alignment is not None
            )
        )

    def _resolve_field(
        self,
        qualified_name: str,
        offset: int,
        *,
        root_class: str,
        base_chain: tuple[str, ...],
        path_prefix: tuple[str, ...],
        abs_base: int,
    ) -> ResolvedField | None:
        # pylint: disable=too-many-return-statements
        source_class = self._classes_by_name.get(qualified_name)
        if source_class is None or not self._layout_usable(source_class):
            return None

        # Ambiguous bitfield packing: refuse a byte-level answer.
        if len(self._bitfields_covering(source_class, offset)) > 1:
            return None

        for item in source_class.fields:
            if item.offset is None:
                continue
            if item.size is None:
                covers = item.offset == offset
            else:
                covers = item.offset <= offset < item.offset + item.size
            if not covers:
                continue
            remaining = offset - item.offset
            nested_type = item.array_element_type or item.type
            nested = self._lookup_nested_class(item, nested_type)
            absolute = abs_base + item.offset
            if (
                item.storage_kind == "array"
                and item.array_stride
                and item.array_stride > 0
            ):
                element_index = remaining // item.array_stride
                if item.array_count is not None and element_index >= item.array_count:
                    continue
                remaining = remaining % item.array_stride
                path = path_prefix + (f"{item.name}[{element_index}]",)
                absolute = abs_base + item.offset + element_index * item.array_stride
            else:
                path = path_prefix + (item.name,)
            if nested is not None:
                nested_resolved = self._resolve_field(
                    nested,
                    remaining,
                    root_class=root_class,
                    base_chain=base_chain,
                    path_prefix=path,
                    abs_base=absolute,
                )
                if nested_resolved is not None:
                    return nested_resolved
                # Nested layout unavailable/untrusted: treat this field as an
                # opaque leaf rather than publishing an untrusted path.
                return None
            return ResolvedField(
                root_class=root_class,
                path=path,
                leaf=item,
                absolute_offset=absolute,
                relative_offset=remaining,
                base_chain=base_chain,
            )

        for base in source_class.base_offsets:
            if offset < base.offset:
                continue
            nested_name = strip_type_qualifiers(base.name)
            base_label = nested_name.rsplit("::", 1)[-1]
            found = self._resolve_field(
                nested_name,
                offset - base.offset,
                root_class=root_class,
                base_chain=base_chain + (nested_name,),
                path_prefix=path_prefix + (base_label,),
                abs_base=abs_base + base.offset,
            )
            if found is not None:
                return found
        return None

    def has_layout(self, qualified_name: str, *, target: str | None = None) -> bool:
        """True when trusted Clang layout evidence is available for the class."""
        source_class = self.class_named(qualified_name, target=target)
        if source_class is None:
            return False
        return self._layout_usable(source_class)

    @classmethod
    def from_units(
        cls,
        units: Sequence[TranslationUnitRecords],
        targets: Mapping[str, set[str] | None],
        *,
        aliases: ProjectAliases | None = None,
        source_digests: Mapping[str, str] | None = None,
        repository: Path | None = None,
    ) -> "SourceIndex":
        """Derive every target's link namespace from TU observations in one
        pass, then join markers. ``targets`` maps each target to the units
        compiled into it (None: all of them).

        Marker blocks come from every unit and are merged once: a header's
        markers for one target may only be compiled by another target's
        translation units."""
        blocks = merge_marker_blocks(
            block for unit in units for block in unit.marker_blocks
        )
        declarations: list[SourceDeclaration] = []
        classes: list[SourceClass] = []
        markers: list[SourceMarker] = []
        variables: list[SourceVariable] = []
        member_uses: list[SourceMemberUse] = []
        conflicts: list[SourceConflict] = []
        abis: dict[str, SourceAbi] = {}
        for target, unit_ids in targets.items():
            namespace = derive_namespace(units, target=target, unit_ids=unit_ids)
            target_classes, target_markers = _join_markers(
                target, namespace, blocks, aliases=aliases
            )
            declarations.extend(namespace.declarations)
            classes.extend(target_classes)
            markers.extend(target_markers)
            variables.extend(namespace.variables)
            member_uses.extend(namespace.member_uses)
            conflicts.extend(namespace.conflicts)
            if namespace.abi is not None:
                abis[target] = namespace.abi
        distinct = set(abis.values())
        dependencies = None
        if repository is not None:
            paths = _RepositoryPaths(repository)
            dependencies = {
                unit.unit_id: paths.files(unit.dependencies) for unit in units
            }
        return cls(
            declarations=declarations,
            classes=classes,
            markers=markers,
            variables=variables,
            member_uses=member_uses,
            conflicts=conflicts,
            abi=distinct.pop() if len(distinct) == 1 else None,
            target_abis=abis,
            marker_blocks=blocks,
            source_digests=source_digests,
            unit_dependencies=dependencies,
        )

    @classmethod
    def from_collector(
        cls,
        target: str,
        collector: SourceCollector,
        *,
        unit_ids: set[str] | None = None,
        aliases: ProjectAliases | None = None,
    ) -> "SourceIndex":
        """Derive one target from a fixture ``SourceCollector`` (tests)."""
        return cls.from_units(
            tuple(collector.units.values()), {target: unit_ids}, aliases=aliases
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "SourceIndex":
        """Read the public JSON projection back into its canonical records."""
        if document.get("schema") != _SCHEMA:
            raise SourceIndexError(
                f"source index schema {document.get('schema')!r} is not {_SCHEMA!r}; "
                "collect the index again"
            )
        declarations = tuple(
            _declaration_from_dict(item) for item in document["declarations"]
        )
        by_key = {(item.target, item.semantic_id): item for item in declarations}
        markers: list[SourceMarker] = []
        for item in document["markers"]:
            values = dict(item)
            if "declaration_key" in values:
                key = values.pop("declaration_key")
                values.pop("declaration", None)
                declaration = by_key.get(tuple(key)) if key else None
            elif values.get("declaration"):
                declaration = _declaration_from_dict(values.pop("declaration"))
            else:
                values.pop("declaration", None)
                declaration = None
            markers.append(SourceMarker(**values, declaration=declaration))
        return cls(
            declarations=declarations,
            classes=(_class_from_dict(item) for item in document["classes"]),
            markers=markers,
            variables=(_variable_from_dict(item) for item in document["variables"]),
            member_uses=(
                _member_use_from_dict(item) for item in document["member_uses"]
            ),
            conflicts=(_conflict_from_dict(item) for item in document["conflicts"]),
            abi=(
                SourceAbi(**document["abi"])
                if isinstance(document.get("abi"), Mapping)
                else None
            ),
            target_abis={
                str(target): SourceAbi(**values)
                for target, values in (document.get("target_abis") or {}).items()
                if isinstance(values, Mapping)
            },
            marker_blocks=(
                MarkerBlock.from_dict(item) for item in document["marker_blocks"]
            ),
            source_digests=document["source_digests"],
            unit_dependencies=document["unit_dependencies"],
        )

    def functions_by_address(
        self, *, target: str | None = None
    ) -> dict[int, SourceMarker]:
        """Return one owner per address, preferring an unfolded body over aliases."""
        functions: dict[int, SourceMarker] = {}
        for marker in self.markers:
            if target is not None and marker.target != target:
                continue
            if not marker.name:
                raise SourceIndexError(
                    f"{marker.source_file}:{marker.line}: marker has no semantic identity"
                )
            previous = functions.get(marker.address)
            if previous is not None:
                if marker.folded and not previous.folded:
                    continue
                if marker.folded == previous.folded:
                    raise SourceIndexError(
                        f"0x{marker.address:08x} has more than one source owner"
                    )
            functions[marker.address] = marker
        return functions

    @classmethod
    def from_compile_database(
        cls,
        repository: Path,
        compilation_database: Path,
        targets: Mapping[str, Sequence[Path]],
        *,
        clang: str | None = None,
        jobs: int | None = None,
        cache_dir: Path | None = None,
        force: bool = False,
        aliases: ProjectAliases | None = None,
        paranoid: bool = False,
    ) -> "SourceIndex":
        # pylint: disable=too-many-arguments
        """Collect direct AST records natively, once for all marker targets.

        Expects to run in the same filesystem as the compile database (typically
        inside the pinned analysis image). ``RECCMP_SOURCE_INDEXER`` or
        ``reccmp-source-indexer`` on ``PATH`` supplies a prebuilt collector;
        otherwise the collector is built once into ``cache_dir`` against LLVM 19.
        """
        # pylint: disable=import-outside-toplevel
        from .batch import collect_compile_database

        return collect_compile_database(
            repository,
            compilation_database,
            targets,
            clang=clang,
            jobs=jobs,
            cache_dir=cache_dir,
            force=force,
            aliases=aliases,
            paranoid=paranoid,
        )

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": _SCHEMA,
            "markers": [_marker_projection(item) for item in self.markers],
            "declarations": [_plain(item) for item in self.declarations],
            "classes": [_plain(item) for item in self.classes],
            "variables": [_plain(item) for item in self.variables],
            "member_uses": [_plain(item) for item in self.member_uses],
            "conflicts": [_plain(item) for item in self.conflicts],
            "marker_blocks": [item.to_dict() for item in self.marker_blocks],
            "source_digests": self.source_digests,
            "unit_dependencies": {
                unit: list(paths) for unit, paths in self.unit_dependencies.items()
            },
        }
        if self.abi is not None:
            document["abi"] = _plain(self.abi)
        if self.target_abis:
            document["target_abis"] = {
                target: _plain(abi) for target, abi in sorted(self.target_abis.items())
            }
        return document

    @classmethod
    def read(cls, path: Path) -> "SourceIndex":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(document)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, SourceIndexError):
                raise
            raise SourceIndexError(
                f"source index at {path} is unusable: {exc}"
            ) from exc

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), separators=(",", ":")) + "\n"
        encoded = content.encode("utf-8")
        if not path.is_file() or path.read_bytes() != encoded:
            path.write_bytes(encoded)
