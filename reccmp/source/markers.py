"""Bind source markers to Clang declarations and serialize marker facts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePath
from typing import Any, Iterable, Mapping, Sequence

from reccmp.parser.marker import MarkerType, ProjectAliases
from reccmp.parser.node import ParserFunction, ParserVtable
from reccmp.parser.reader import MarkerBlock, read_marker_blocks
from .observations import SourceIndexError
from .records import (
    DeclarationKey,
    SourceBaseVtable,
    SourceClass,
    SourceDeclaration,
    SourceMarker,
)


def merge_marker_blocks(blocks: Iterable[MarkerBlock]) -> tuple[MarkerBlock, ...]:
    """One block per source position; a header is seen by many units."""
    merged: dict[tuple[str, int], MarkerBlock] = {}
    for block in blocks:
        previous = merged.get(block.key)
        merged[block.key] = block if previous is None else previous.merged(block)
    return tuple(merged[key] for key in sorted(merged))


def _marker_projection(marker: SourceMarker) -> dict[str, Any]:
    """Serialize a marker with a declaration key instead of a nested copy."""
    return {
        "address": marker.address,
        "marker_kind": marker.marker_kind,
        "source_file": marker.source_file,
        "line": marker.line,
        "marker_name": marker.marker_name,
        "folded": marker.folded,
        "target": marker.target,
        "declaration_key": (
            marker.declaration_key.to_json()
            if marker.declaration_key is not None
            else None
        ),
    }


def _compact_pointer_spacing(name: str) -> str:
    return name.replace(" *", "*").replace(" &", "&")


def _join_markers(
    target: str,
    declarations: Mapping[DeclarationKey, SourceDeclaration],
    source_classes: Mapping[DeclarationKey, SourceClass],
    blocks: Sequence[MarkerBlock],
    *,
    aliases: ProjectAliases | None,
) -> tuple[dict[DeclarationKey, SourceClass], list[SourceMarker]]:
    symbols = [
        symbol
        for result in read_marker_blocks(
            blocks,
            {block.source_file: PurePath(block.source_file) for block in blocks},
            aliases=aliases,
        )
        for symbol in result.tokens
        if symbol.module == target.upper()
    ]
    # By location as well as identity: TU-local functions of different files
    # can share a mangled name.
    definitions: dict[tuple[str, str, int], list[DeclarationKey]] = {}
    for key, declaration in declarations.items():
        if declaration.is_definition:
            definitions.setdefault(
                (declaration.semantic_id, declaration.source_file, declaration.line), []
            ).append(key)

    markers: list[SourceMarker] = []
    for method_symbol in symbols:
        if not isinstance(method_symbol, ParserFunction):
            continue
        relative = method_symbol.filename.as_posix()
        marker_key: DeclarationKey | None = None
        # Name-reference markers (TEMPLATE/SYNTHETIC/LIBRARY, and FUNCTION with a
        # name comment e.g. `FUNCTION: X 0x... SYMBOL` + `// ??0foo@@QAE@XZ`)
        # name their entity instead of annotating a definition.
        if (
            method_symbol.type in {MarkerType.FUNCTION, MarkerType.STUB}
            and not method_symbol.is_nameref()
        ):
            # One per definition. A TU-local function defined in a header
            # has an identical copy in each including unit, and nothing here
            # says which copy the marker's address is: it binds the first
            # unit's.
            candidates = [
                min(found, key=DeclarationKey.sort_key)
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
            marker_key = candidates[0]
        marker_declaration = (
            declarations[marker_key] if marker_key is not None else None
        )
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
                declaration_key=marker_key,
            )
        )

    classes = dict(source_classes)
    class_by_name = {item.qualified_name: key for key, item in classes.items()}
    explicitly_primary = {
        symbol.name
        for symbol in symbols
        if isinstance(symbol, ParserVtable)
        and (
            symbol.base_class is None
            or _compact_pointer_spacing(symbol.base_class)
            == _compact_pointer_spacing(symbol.name)
        )
    }
    for vtable_symbol in symbols:
        if not isinstance(vtable_symbol, ParserVtable):
            continue
        relative = vtable_symbol.filename.as_posix()
        class_key = class_by_name.get(vtable_symbol.name)
        if class_key is None:
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
            )
            class_key = DeclarationKey(target, source_class.semantic_id)
            classes[class_key] = source_class
            class_by_name[source_class.qualified_name] = class_key
            continue
        source_class = classes[class_key]
        base_class = vtable_symbol.base_class
        class_names = {
            source_class.qualified_name,
            source_class.qualified_name.rsplit("::", 1)[-1],
        }
        # Clang spells a pointer template argument as ``T *`` while a marker
        # may spell the same class as ``T*``. That whitespace is not a base.
        # MSVC may name the primary table for an inherited first-base vtable.
        # Follow only the first-base chain; other bases own secondary tables.
        primary_names = class_names
        if (
            source_class.vtable_address is None
            and source_class.qualified_name not in explicitly_primary
        ):
            ancestor = source_class
            seen_bases: set[str] = set()
            while ancestor.bases:
                first_base = ancestor.bases[0]
                if first_base in seen_bases:
                    break
                seen_bases.add(first_base)
                primary_names.add(first_base)
                parent_key = class_by_name.get(first_base)
                if parent_key is None:
                    break
                ancestor = classes[parent_key]
        if base_class is not None and _compact_pointer_spacing(base_class) not in {
            _compact_pointer_spacing(name) for name in primary_names
        }:
            base_vtable = SourceBaseVtable(vtable_symbol.offset, base_class)
            if base_vtable in source_class.base_vtables:
                raise SourceIndexError(
                    f"{relative}:{vtable_symbol.line_number}: duplicate VTABLE marker "
                    f"for base {base_class}"
                )
            classes[class_key] = replace(
                source_class,
                base_vtables=(*source_class.base_vtables, base_vtable),
            )
            continue
        if source_class.vtable_address is not None:
            raise SourceIndexError(
                f"{relative}:{vtable_symbol.line_number}: class has more than one "
                "primary VTABLE marker"
            )
        classes[class_key] = replace(source_class, vtable_address=vtable_symbol.offset)
    return classes, markers
