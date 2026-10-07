"""Parse per-translation-unit Clang observations into compiler fact records."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TextIO

from reccmp.parser.reader import MarkerBlock
from .records import (
    SourceAbi,
    SourceBaseOffset,
    SourceBaseVtable,
    SourceClass,
    SourceDeclaration,
    SourceField,
)
from .variables import SourceConflict, SourceConflictVariant, SourceVariable


class SourceIndexError(ValueError):
    """The source markers and compiler model cannot be joined unambiguously."""


@dataclass(frozen=True)
class _SizeAssertion:
    unit_id: str
    qualified_name: str
    asserted_size: int


@dataclass
# pylint: disable=too-many-instance-attributes
class TranslationUnitRecords:
    """Raw compiler observations from one translation unit.

    No winner selection or conflict tracking happens here — that is derived
    after observations are grouped by link namespace.
    """

    unit_id: str
    declarations: list[SourceDeclaration] = field(default_factory=list)
    variables: list[SourceVariable] = field(default_factory=list)
    classes: list[SourceClass] = field(default_factory=list)
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
            return _declaration_from_dict(values)
        if kind == "variable":
            return _variable_from_dict(values)
        if kind == "class":
            return _class_from_dict(values)
        raise SourceIndexError(
            f"the source indexer emitted an unknown record: {kind!r}"
        )

    def _append(self, kind: str, fact: Any) -> None:
        if kind == "marker-block":
            self.marker_blocks.append(fact)
        elif kind == "declaration":
            self.declarations.append(fact)
        elif kind == "variable":
            self.variables.append(fact)
        else:
            self.classes.append(fact)

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
                if line.isspace():
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
            if not line.isspace():
                self.add(json.loads(line))


# Records about a translation unit rather than about source code.
_UNIT_RECORDS = frozenset({"profile", "dependency", "unit-abi", "size-assertion"})


class RecordPool:
    """Parsed source records by their exact artifact line, shared across
    every unit loaded with the pool. Records are compiler facts without unit
    or target: a unit observes a fact by listing it."""

    def __init__(self) -> None:
        self.facts: dict[bytes, tuple[str, Any]] = {}


_DECLARATION_FIELDS = frozenset(
    item.name for item in dataclasses.fields(SourceDeclaration)
)


def _declaration_from_dict(values: Mapping[str, Any]) -> SourceDeclaration:
    # Indexes collected by earlier indexers carry facts nothing reads now.
    data = {key: value for key, value in values.items() if key in _DECLARATION_FIELDS}
    data["parameter_types"] = tuple(data.get("parameter_types") or ())
    return SourceDeclaration(**data)


def _variable_from_dict(values: Mapping[str, Any]) -> SourceVariable:
    return SourceVariable(**dict(values))


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


def _field_from_dict(values: Mapping[str, Any]) -> SourceField:
    data = dict(values)
    for key in (
        "offset",
        "size",
        "bitfield_width",
        "bitfield_offset",
        "record_semantic_id",
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
        pointer_depth=int(data["pointer_depth"]),
        offset=data.get("offset"),
        size=data.get("size"),
        bitfield_width=data.get("bitfield_width"),
        bitfield_offset=data.get("bitfield_offset"),
        record_semantic_id=data.get("record_semantic_id"),
        storage_kind=str(data["storage_kind"]),
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
        size=values.get("size"),
        alignment=values.get("alignment"),
        base_offsets=tuple(
            SourceBaseOffset(
                name=str(item["name"]),
                semantic_id=str(item["semantic_id"]),
                offset=int(item["offset"]),
            )
            for item in values.get("base_offsets") or ()
        ),
        layout_trusted=values.get("layout_trusted"),
    )
