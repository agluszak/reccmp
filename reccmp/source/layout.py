"""Queries over trusted Clang class layout and field ownership."""

from __future__ import annotations

from .records import SourceClass, SourceField, ResolvedField


def field_is_indirection(source_field: SourceField) -> bool:
    """Pointer/reference storage is a layout leaf; do not descend physically."""
    if source_field.storage_kind in ("pointer", "reference"):
        return True
    if source_field.storage_kind == "array":
        return source_field.array_element_kind in ("pointer", "reference")
    return False


class SourceLayoutQueries:
    """Layout lookup behavior shared by each scoped ``SourceIndex`` view."""

    _classes_by_name: dict[str, SourceClass]
    _classes_by_semantic_id: dict[str, SourceClass]
    _classes_by_target_name: dict[tuple[str | None, str], SourceClass]
    _classes_by_target_semantic_id: dict[tuple[str | None, str], SourceClass]

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

    def _lookup_nested_class(self, source_field: SourceField) -> str | None:
        """Look up a physically contained record by Clang semantic ID."""
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
        semantic_id = source_field.record_semantic_id
        nested = (
            self._classes_by_semantic_id.get(semantic_id)
            if semantic_id is not None
            else None
        )
        return nested.qualified_name if nested is not None else None

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
            nested = self._lookup_nested_class(item)
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
            base_class = self._classes_by_semantic_id.get(base.semantic_id)
            if base_class is None:
                continue
            nested_name = base_class.qualified_name
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
