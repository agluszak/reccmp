import struct

from reccmp.compare.thunk_resolve import (
    effective_orig_vtable_size,
    is_plausible_vtable_target,
)

from .raw_image import RawImage


def test_import_address_table_jump_is_a_plausible_vtable_target() -> None:
    image = RawImage.from_memory(b"\xff\x25\x00\x20\x40\x00", base_addr=0x401000)

    assert is_plausible_vtable_target(image, 0x401000)


def test_other_ff_instruction_is_not_a_plausible_vtable_target() -> None:
    image = RawImage.from_memory(b"\xff\xd0", base_addr=0x401000)

    assert not is_plausible_vtable_target(image, 0x401000)


def test_vtable_keeps_trailing_callee_cleanup_noops() -> None:
    # A normal method followed by folded thiscall no-ops and non-code data.
    base = 0x401000
    table = struct.pack("<4I", base + 16, base + 18, base + 18, base + 21)
    image = RawImage.from_memory(table + b"\x55\xc3\xc2\x04\x00AB", base_addr=base)

    assert is_plausible_vtable_target(image, base + 18)
    assert effective_orig_vtable_size(image, base, len(table)) == 12
