from reccmp.compare.thunk_resolve import is_plausible_vtable_target

from .raw_image import RawImage


def test_import_address_table_jump_is_a_plausible_vtable_target() -> None:
    image = RawImage.from_memory(b"\xff\x25\x00\x20\x40\x00", base_addr=0x401000)

    assert is_plausible_vtable_target(image, 0x401000)


def test_other_ff_instruction_is_not_a_plausible_vtable_target() -> None:
    image = RawImage.from_memory(b"\xff\xd0", base_addr=0x401000)

    assert not is_plausible_vtable_target(image, 0x401000)
