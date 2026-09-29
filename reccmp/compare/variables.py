import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple
from struct import unpack, error as StructError
from reccmp.formats import Image
from reccmp.analysis.string_const import is_likely_latin1, is_likely_widechar
from reccmp.formats.exceptions import (
    InvalidVirtualReadError,
    InvalidVirtualAddressError,
    InvalidStringError,
)
from reccmp.compare.db import EntityDb, ReccmpMatch
from reccmp.cvdump.cvinfo import CvdumpTypeKey
from reccmp.cvdump.types import (
    CvdumpTypesParser,
    CvdumpKeyError,
    CvdumpIntegrityError,
)
from reccmp.types import ImageId

if TYPE_CHECKING:
    from reccmp.source import SourceIndex

logger = logging.getLogger(__name__)


class CompareResult(Enum):
    MATCH = 1
    DIFF = 2
    ERROR = 3
    WARN = 4


class DataOffset(NamedTuple):
    offset: int
    name: str
    pointer: bool


class ComparedOffset(NamedTuple):
    offset: int
    # name is None for scalar types
    name: str | None
    match: bool
    values: tuple[str, str]


class ComparisonItem(NamedTuple):
    """Each variable that was compared"""

    orig_addr: int
    recomp_addr: int
    name: str

    # The list of items that were compared.
    # For a complex type, these are the members.
    # For a scalar type, this is a list of size one.
    # If we could not retrieve type information, this is
    # a list of size one but without any specific type.
    compared: list[ComparedOffset]

    # If present, the error message from the types parser.
    error: str | None = None

    # If true, there is no type specified for this variable. (i.e. non-public)
    # In this case, we can only compare the raw bytes.
    # This is different from the situation where a type id _is_ given, but
    # we could not retrieve it for some reason. (This is an error.)
    raw_only: bool = False

    @property
    def result(self) -> CompareResult:
        if self.error is not None:
            return CompareResult.ERROR

        if all(c.match for c in self.compared):
            return CompareResult.MATCH

        # Prefer WARN for a diff without complete type information.
        return CompareResult.WARN if self.raw_only else CompareResult.DIFF


def create_comparison_item(
    var: ReccmpMatch,
    compared: list[ComparedOffset] | None = None,
    error: str | None = None,
    raw_only: bool = False,
) -> ComparisonItem:
    """Helper to create the ComparisonItem from the fields in the reccmp database."""
    if compared is None:
        compared = []
    assert var.name is not None

    return ComparisonItem(
        orig_addr=var.orig_addr,
        recomp_addr=var.recomp_addr,
        name=var.name,
        compared=compared,
        error=error,
        raw_only=raw_only,
    )


def pointer_display(
    db: EntityDb, types: CvdumpTypesParser, img: ImageId, addr: int
) -> str:
    """Helper to streamline pointer textual display."""
    if addr == 0:
        return "nullptr"

    entity = db.get(img, addr, exact=False)

    if entity is not None:
        name = None

        base_addr = entity.addr(img)
        assert isinstance(base_addr, int)

        offset = addr - base_addr
        if offset == 0:
            name = entity.match_name()
        else:
            type_key = entity.get("data_type")
            if type_key:
                suffix = types.get_name_for_offset(CvdumpTypeKey(type_key), offset)
                name = entity.match_name(suffix)
            else:
                name = entity.match_name(f"+{offset}")

        if name:
            return f"Pointer to {name}"

    # This variable did not match if we do not have
    # the pointer target in our DB.
    return f"Unknown pointer 0x{addr:x}"


@dataclass
class VariableComparator:
    db: EntityDb
    types: CvdumpTypesParser
    orig_bin: Image
    recomp_bin: Image
    source_index: "SourceIndex | None" = None

    def _source_type_name(self, var: ReccmpMatch) -> str | None:
        """Resolve a Clang layout type name for this variable, when indexed.

        Pointer/reference variables store an address; do not treat their
        pointee ``record_semantic_id`` as the variable's physical layout.
        """
        if self.source_index is None:
            return None

        semantic_id = var.get("semantic_id")
        if semantic_id:
            candidates = [
                item
                for item in self.source_index.variables.values()
                if item.semantic_id == semantic_id
            ]
        elif var.name:
            candidates = [
                item
                for item in self.source_index.variables.values()
                if item.qualified_name == var.name
                or item.qualified_name.endswith(f"::{var.name}")
            ]
        else:
            return None
        # A source index may contain the same declaration in several units.
        # Resolve only when every candidate has the same physical type facts.
        types = {(item.storage_kind, item.record_semantic_id) for item in candidates}
        if len(types) != 1:
            return None
        storage_kind, record_id = next(iter(types))
        if storage_kind in ("pointer", "reference") or record_id is None:
            return None
        nested = self.source_index.class_for_semantic_id(record_id)
        if nested is not None and self.source_index.has_layout(nested.qualified_name):
            return nested.qualified_name
        return None

    def _source_layout_members(
        self, type_name: str, data_size: int
    ) -> list[tuple[DataOffset, int]] | None:
        """Build ``(DataOffset, size)`` rows from trusted Clang layout.

        Used when PDB cannot provide a typed format string (``raw_only``).
        Returns ``None`` when layout is missing or untrusted so callers fall
        back to byte-wise raw comparison.
        """
        if self.source_index is None or not self.source_index.has_layout(type_name):
            return None
        members: list[tuple[DataOffset, int]] = []
        offset = 0
        while offset < data_size:
            resolved = self.source_index.resolve_field(type_name, offset)
            if resolved is None or resolved.relative_offset != 0:
                members.append((DataOffset(offset=offset, name="", pointer=False), 1))
                offset += 1
                continue
            leaf = resolved.leaf
            size = leaf.size
            is_pointer = (leaf.pointer_depth or 0) > 0
            if (
                leaf.storage_kind == "array"
                and leaf.array_stride
                and leaf.array_stride > 0
            ):
                size = leaf.array_stride
                is_pointer = leaf.array_element_kind in ("pointer", "reference")
            if size is None or size <= 0:
                members.append((DataOffset(offset=offset, name="", pointer=False), 1))
                offset += 1
                continue
            if is_pointer:
                pointer_width = 4
                if self.source_index is not None and self.source_index.abi is not None:
                    pointer_width = max(1, self.source_index.abi.pointer_width // 8)
                size = pointer_width
            # Do not overrun the variable extent.
            if resolved.absolute_offset + size > data_size:
                size = data_size - resolved.absolute_offset
            if size <= 0:
                break
            path = ".".join(resolved.path) if resolved.path else leaf.name
            members.append(
                (
                    DataOffset(
                        offset=resolved.absolute_offset,
                        name=path,
                        pointer=is_pointer,
                    ),
                    size,
                )
            )
            offset = resolved.absolute_offset + size
        return members or None

    @staticmethod
    def _unpack_layout_members(
        data: bytes, members: list[tuple[DataOffset, int]]
    ) -> tuple:
        values: list[int] = []
        for item, size in members:
            chunk = data[item.offset : item.offset + size]
            if len(chunk) < size:
                raise StructError(
                    f"short read at offset {item.offset:#x} need {size} got {len(chunk)}"
                )
            values.append(int.from_bytes(chunk, "little"))
        return tuple(values)

    def _member_display_name(
        self, member: DataOffset, type_name: str | None
    ) -> str | None:
        """Prefer SourceIndex field paths when layout is known; else PDB name."""
        if type_name is not None and self.source_index is not None:
            path = self.source_index.field_path_at(type_name, member.offset)
            if path:
                return path
        if member.name:
            return member.name
        if type_name is None:
            return None
        return f"+{member.offset:#x}" if member.offset else None

    def is_pointer_match(self, orig_addr: int, recomp_addr: int) -> bool:
        """Check whether these pointers point at the same thing"""

        # Identical absolute values match: NULL, INVALID_HANDLE_VALUE (-1), and
        # same-base matching builds where the pointed-to VA is unchanged.
        if orig_addr == recomp_addr:
            return True

        if self.db.is_match(orig_addr, recomp_addr):
            return True

        # MSVC string pooling can leave orig pointing at another string's NUL
        # while recomp has a distinct "". Wide strings are also easy to mis-
        # identify as short Latin1 fragments during PE analysis; compare the
        # decoded contents as a last resort.
        return self.is_string_content_match(orig_addr, recomp_addr)

    def _decode_string_at(self, img: Image, addr: int) -> tuple[str, bool] | None:
        """Return (text, is_wide) for the best string decode at addr, or None."""
        wide_text: str | None = None
        try:
            wide_text = img.read_widechar(addr).decode("utf-16-le")
        except (InvalidStringError, UnicodeDecodeError, InvalidVirtualAddressError):
            pass

        narrow_text: str | None = None
        try:
            narrow_text = img.read_string(addr).decode("latin1")
        except (InvalidStringError, UnicodeDecodeError, InvalidVirtualAddressError):
            pass

        if wide_text is None and narrow_text is None:
            return None

        # Prefer wide when it continues past a Latin1 truncation (embedded NUL).
        if wide_text is not None and (
            narrow_text is None
            or len(wide_text) > len(narrow_text)
            or wide_text == narrow_text
        ):
            return wide_text, True

        assert narrow_text is not None
        return narrow_text, False

    def is_string_content_match(self, orig_addr: int, recomp_addr: int) -> bool:
        """True when both addresses decode to the same C or wide string text."""
        orig = self._decode_string_at(self.orig_bin, orig_addr)
        recomp = self._decode_string_at(self.recomp_bin, recomp_addr)
        if orig is None or recomp is None:
            return False

        orig_text, orig_wide = orig
        recomp_text, recomp_wide = recomp

        # Empty narrow and empty wide both mean "".
        if orig_text == "" and recomp_text == "":
            return True

        if orig_wide != recomp_wide:
            return False

        if orig_text != recomp_text:
            return False

        # Reject binary blobs that merely share a decode (e.g. int payloads).
        if orig_wide:
            return is_likely_widechar(orig_text)
        return is_likely_latin1(orig_text)

    def is_pointer_match_to_offset(self, orig_addr: int, recomp_addr: int) -> bool:
        """Check whether these pointers point at the same offset of the same matched entity."""
        orig_ent = self.db.get(ImageId.ORIG, orig_addr, exact=False)
        recomp_ent = self.db.get(ImageId.RECOMP, recomp_addr, exact=False)

        if orig_ent is None or recomp_ent is None:
            return False

        # Are both entities matched?
        if not isinstance(orig_ent, ReccmpMatch) or not isinstance(
            recomp_ent, ReccmpMatch
        ):
            return False

        # Are they matched to each other?
        if orig_ent.orig_addr != recomp_ent.orig_addr:
            return False

        # Are we at the same offset?
        return (orig_addr - orig_ent.orig_addr) == (
            recomp_addr - recomp_ent.recomp_addr
        )

    def compare_variable(self, var: ReccmpMatch) -> ComparisonItem:
        # pylint: disable=too-many-locals
        assert var.name is not None
        type_key = CvdumpTypeKey(var.get("data_type")) if var.get("data_type") else None

        # Start by assuming we can only compare the raw bytes
        data_size = var.any_size()
        raw_only = True

        if type_key is not None:
            try:
                # If we are type-aware, we can get the precise
                # data size for the variable.
                data_type = self.types.get(type_key)
                assert data_type.size is not None
                data_size = data_type.size

                # Make sure we can retrieve struct or array members.
                if self.types.get_format_string(type_key):
                    raw_only = False
                else:
                    logger.info(
                        "No struct members for type '0x%x' used by variable '%s' (0x%x). Comparing raw data.",
                        type_key,
                        var.name,
                        var.orig_addr,
                    )

            except (CvdumpKeyError, CvdumpIntegrityError):
                # This may occur even when nothing is wrong, so permit a raw comparison here.
                # For example: we do not handle bitfields and this complicates fieldlist parsing
                # where they are used. (GH #299)
                logger.error(
                    "Could not materialize type '0x%x' used by variable '%s' (0x%x). Comparing raw data.",
                    type_key,
                    var.name,
                    var.orig_addr,
                )

        assert data_size is not None
        source_type_name = self._source_type_name(var)

        try:
            orig_bytes = self.orig_bin.read(var.orig_addr, data_size)
        except InvalidVirtualReadError as ex:
            # Reading from orig can fail if the recomp variable is too large
            return create_comparison_item(var, error=repr(ex))

        # Reading from recomp should never fail, so if it does, raising an exception is correct
        recomp_bytes = self.recomp_bin.read(var.recomp_addr, data_size)

        used_source_layout = False
        if raw_only:
            source_members = (
                self._source_layout_members(source_type_name, data_size)
                if source_type_name
                else None
            )
            if source_members is not None:
                # Trusted Clang layout as an alternate type provider when PDB
                # cannot materialize a format string.
                used_source_layout = True
                compare_items = [item for item, _size in source_members]
                try:
                    orig_data = self._unpack_layout_members(orig_bytes, source_members)
                    recomp_data = self._unpack_layout_members(
                        recomp_bytes, source_members
                    )
                except StructError as e:
                    return create_comparison_item(
                        var, error=f"Failed to unpack data: {e}"
                    )
            else:
                # If there is no specific type information available
                # (i.e. if this is a static or non-public variable)
                # then we can only compare the raw bytes.
                compare_items = [
                    DataOffset(offset=i, name="", pointer=False)
                    for i in range(data_size)
                ]
                orig_data = tuple(orig_bytes)
                recomp_data = tuple(recomp_bytes)
        else:
            assert type_key is not None
            compare_items = [
                DataOffset(offset=sc.offset, name=sc.name or "", pointer=sc.is_pointer)
                for sc in self.types.get_scalars_gapless(type_key)
            ]
            format_str = self.types.get_format_string(type_key)

            try:
                orig_data = unpack(format_str, orig_bytes)
                recomp_data = unpack(format_str, recomp_bytes)
            except StructError as e:
                return create_comparison_item(var, error=f"Failed to unpack data: {e}")

        compared = []
        for orig_val, recomp_val, member in zip(orig_data, recomp_data, compare_items):
            if member.pointer:
                match = self.is_pointer_match(orig_val, recomp_val)

                if not match:
                    match = self.is_pointer_match_to_offset(orig_val, recomp_val)

                value_a = pointer_display(self.db, self.types, ImageId.ORIG, orig_val)
                value_b = pointer_display(
                    self.db, self.types, ImageId.RECOMP, recomp_val
                )
            else:
                match = orig_val == recomp_val
                value_a = str(orig_val)
                value_b = str(recomp_val)

            compared.append(
                ComparedOffset(
                    offset=member.offset,
                    name=(
                        member.name
                        if used_source_layout and member.name
                        else self._member_display_name(member, source_type_name)
                    ),
                    match=match,
                    values=(value_a, value_b),
                )
            )

        return create_comparison_item(
            var,
            compared=compared,
            # Source-layout typed compare is not "raw only" for reporting.
            raw_only=raw_only and not used_source_layout,
        )
