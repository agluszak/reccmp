import struct

from reccmp.compare.db import EntityDb
from reccmp.compare.thunk_resolve import (
    effective_orig_vtable_size,
    is_plausible_vtable_target,
)
from reccmp.formats.image import ImageSection, ImageSectionFlags
from reccmp.types import EntityType, ImageId

from .raw_image import RawImage


def _with_executable_range(
    image: RawImage, start: int, end: int, *, name: str = ".text"
) -> RawImage:
    """Attach a single EXECUTE section covering ``[start, end)``."""
    offset = start - image.base_addr
    size = end - start
    section = ImageSection(
        name=name,
        virtual_range=range(start, end),
        physical_range=range(offset, offset + size),
        view=memoryview(image.data[offset : offset + size]),
        flags=ImageSectionFlags.EXECUTE | ImageSectionFlags.READ,
    )
    image.sections = (section,)
    return image


def test_null_slot_is_plausible_vtable_target() -> None:
    image = RawImage.from_memory(b"\x00", base_addr=0x401000)
    assert is_plausible_vtable_target(image, 0)


def test_missing_section_info_is_not_blindly_accepted() -> None:
    """Without section metadata, do not fall back to an opcode whitelist."""
    image = RawImage.from_memory(b"\x55\xc3", base_addr=0x401000)
    assert not image.sections
    assert not is_plausible_vtable_target(image, 0x401000)


def test_import_address_table_jump_is_a_plausible_vtable_target() -> None:
    image = _with_executable_range(
        RawImage.from_memory(b"\xff\x25\x00\x20\x40\x00", base_addr=0x401000),
        0x401000,
        0x401006,
    )
    assert is_plausible_vtable_target(image, 0x401000)


def test_non_executable_section_is_not_a_plausible_vtable_target() -> None:
    """Decodable bytes in a non-executable section (e.g. .rdata) are refused."""
    image = RawImage.from_memory(b"\x55\xc3", base_addr=0x401000)
    section = ImageSection(
        name=".rdata",
        virtual_range=range(0x401000, 0x401002),
        physical_range=range(0, 2),
        view=memoryview(image.data),
        flags=ImageSectionFlags.READ,
    )
    image.sections = (section,)
    assert not is_plausible_vtable_target(image, 0x401000)


def test_undecodable_bytes_in_executable_section_are_rejected() -> None:
    # Capstone refuses to decode a lone 0xFF as a complete instruction.
    image = _with_executable_range(
        RawImage.from_memory(b"\xff", base_addr=0x401000),
        0x401000,
        0x401001,
    )
    assert not is_plausible_vtable_target(image, 0x401000)


def test_known_function_entity_is_plausible_without_section_info() -> None:
    image = RawImage.from_memory(b"\x00\x00", base_addr=0x401000)
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x401000, type=EntityType.FUNCTION, size=2)
    assert is_plausible_vtable_target(image, 0x401000, db=db, image_id=ImageId.ORIG)


def test_vtable_keeps_trailing_callee_cleanup_noops() -> None:
    # A normal method followed by folded thiscall no-ops and non-code data.
    # Executable coverage stops before the trailing ASCII junk so the last
    # slot (pointing at "AB") is not treated as a code target.
    base = 0x401000
    table = struct.pack("<4I", base + 16, base + 18, base + 18, base + 21)
    image = _with_executable_range(
        RawImage.from_memory(table + b"\x55\xc3\xc2\x04\x00AB", base_addr=base),
        base + 16,
        base + 21,
    )

    assert is_plausible_vtable_target(image, base + 18)
    assert not is_plausible_vtable_target(image, base + 21)
    assert effective_orig_vtable_size(image, base, len(table)) == 12


def test_vtable_stops_at_first_implausible_not_last_plausible() -> None:
    """Contiguous prefix: an early non-code hole must not extend to a later method."""
    base = 0x401000
    # File layout: [12-byte table][2-byte non-exec pad][two push/ret methods].
    # slot0 → first method, slot1 → pad (not executable), slot2 → second method.
    table = struct.pack("<3I", base + 14, base + 12, base + 16)
    image = RawImage.from_memory(table + b"\x00\x00\x55\xc3\x55\xc3", base_addr=base)
    image = _with_executable_range(image, base + 14, base + 18)
    assert is_plausible_vtable_target(image, base + 14)
    assert not is_plausible_vtable_target(image, base + 12)
    assert is_plausible_vtable_target(image, base + 16)
    # Old "last plausible anywhere" would return 12; contiguous prefix stops at 4.
    assert effective_orig_vtable_size(image, base, len(table)) == 4
