"""Derive link-namespace winners and conflicts from per-unit observations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .observations import SourceIndexError, TranslationUnitRecords, _SizeAssertion
from .records import (
    DeclarationKey,
    SourceAbi,
    SourceClass,
    SourceDeclaration,
)
from .variables import SourceConflict, SourceConflictVariant, SourceVariable

_VARIABLE_RANK = {"declaration": 0, "tentative": 1, "definition": 2}


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class _NamespaceRecords:
    """Winners and conflicts derived inside one link namespace."""

    declarations: dict[DeclarationKey, SourceDeclaration]
    variables: dict[DeclarationKey, SourceVariable]
    classes: dict[DeclarationKey, SourceClass]
    conflicts: tuple[SourceConflict, ...]
    size_assertions: dict[str, int]
    abi: SourceAbi | None = None


def derive_namespace(
    units: Sequence[TranslationUnitRecords],
    *,
    target: str | None = None,
    unit_ids: set[str] | None = None,
) -> _NamespaceRecords:
    """Partition TU observations, then derive winners and conflicts."""
    selected = [unit for unit in units if unit_ids is None or unit.unit_id in unit_ids]
    declarations: dict[DeclarationKey, list[SourceDeclaration]] = {}
    variables: dict[DeclarationKey, list[SourceVariable]] = {}
    classes: dict[DeclarationKey, list[SourceClass]] = {}
    shared_declarations: set[int] = set()
    shared_variables: set[int] = set()
    shared_classes: set[int] = set()
    for unit in selected:
        # Units share fact objects; each key keeps each object once.
        for declaration in unit.declarations:
            if declaration.is_external:
                identity = id(declaration)
                if identity in shared_declarations:
                    continue
                shared_declarations.add(identity)
            _add_observation(
                declarations, declaration.key(target, unit.unit_id), declaration
            )
        for variable in unit.variables:
            if variable.is_external:
                identity = id(variable)
                if identity in shared_variables:
                    continue
                shared_variables.add(identity)
            _add_observation(
                variables,
                DeclarationKey(
                    target,
                    variable.semantic_id,
                    None if variable.is_external else unit.unit_id,
                ),
                variable,
            )
        for source_class in unit.classes:
            identity = id(source_class)
            if identity in shared_classes:
                continue
            shared_classes.add(identity)
            _add_observation(
                classes, DeclarationKey(target, source_class.semantic_id), source_class
            )
    assertions = [item for unit in selected for item in unit.size_assertions]

    derived_declarations, declaration_conflicts = _derive_entities(
        declarations,
        rank=lambda item: 1 if item.is_definition else 0,
        record_kind="declaration",
        target=target,
    )
    derived_variables, variable_conflicts = _derive_entities(
        variables,
        rank=lambda item: _VARIABLE_RANK.get(item.definition_kind, 0),
        record_kind="variable",
        target=target,
    )
    derived_classes, class_conflicts = _derive_classes(classes, target=target)
    size_assertions = _derive_size_assertions(assertions)
    return _NamespaceRecords(
        declarations=derived_declarations,
        variables=derived_variables,
        classes={
            key: _apply_asserted_size(item, size_assertions.get(item.qualified_name))
            for key, item in derived_classes.items()
        },
        conflicts=declaration_conflicts + variable_conflicts + class_conflicts,
        size_assertions=size_assertions,
        abi=_derive_abi(selected),
    )


def _add_observation(
    groups: dict[DeclarationKey, list], key: DeclarationKey, fact
) -> None:
    group = groups.setdefault(key, [])
    if not any(item is fact for item in group):
        group.append(fact)


def _derive_entities(
    groups: Mapping[DeclarationKey, Sequence[Any]],
    *,
    rank,
    record_kind: str,
    target: str | None,
) -> tuple[dict[DeclarationKey, Any], tuple[SourceConflict, ...]]:
    winners: dict[DeclarationKey, Any] = {}
    conflicts: list[SourceConflict] = []
    for key, group in groups.items():
        winner = group[0]
        for item in group[1:]:
            if rank(item) > rank(winner):
                winner = item
        winners[key] = winner

        variants: dict[tuple[str, ...], list[str]] = {}
        for item in group:
            location = f"{item.source_file}:{item.line}"
            variants.setdefault(item.signature, [])
            if location not in variants[item.signature]:
                variants[item.signature].append(location)
        if len(variants) > 1:
            conflicts.append(
                SourceConflict(
                    semantic_id=winner.semantic_id,
                    qualified_name=winner.qualified_name,
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
    return winners, tuple(conflicts)


def _layout_identity(source_class: SourceClass) -> tuple:
    """Fingerprint of Clang layout facts used for cross-TU consistency."""
    return (
        source_class.size,
        source_class.alignment,
        tuple(
            (
                field.name,
                field.storage_kind,
                field.pointer_depth,
                field.record_semantic_id,
                field.array_element_kind,
                field.array_stride,
                field.array_count,
                field.offset,
                field.size,
                field.bitfield_width,
                field.bitfield_offset,
            )
            for field in source_class.fields
        ),
        tuple((base.semantic_id, base.offset) for base in source_class.base_offsets),
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
    for (
        name,
        kind,
        depth,
        record_id,
        element_kind,
        stride,
        count,
        offset,
        field_size,
        bit_width,
        bit_offset,
    ) in fields:
        parts.append(
            f"field:{name}:{kind}:{depth}:{record_id}:{element_kind}:"
            f"{stride}:{count}:{offset}:{field_size}:{bit_width}:{bit_offset}"
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
    groups: Mapping[DeclarationKey, Sequence[SourceClass]], *, target: str | None
) -> tuple[dict[DeclarationKey, SourceClass], tuple[SourceConflict, ...]]:
    winners: dict[DeclarationKey, SourceClass] = {}
    conflicts: list[SourceConflict] = []
    for key, group in groups.items():
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

        winners[key] = replace(winner, layout_trusted=layout_trusted)
    return winners, tuple(conflicts)


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
