"""Control-flow extent discovery for functions without an annotated size."""

import struct

from reccmp.compare.extent import discover_extent
from tests.raw_image import RawImage

BASE = 0x401000


def _image(code: bytes) -> RawImage:
    return RawImage.from_memory(code + b"\xcc" * 64, base_addr=BASE)


def test_branches_and_ret():
    # test eax, eax; je +1; inc eax; ret; int3 padding
    code = bytes.fromhex("85c0740140c3")
    assert discover_extent(_image(code), BASE, None) == len(code)


def test_block_after_the_first_ret_is_included():
    # test eax, eax; jne +1; ret; inc eax; ret
    code = bytes.fromhex("85c07501c340c3")
    assert discover_extent(_image(code), BASE, None) == len(code)


def test_tail_jump_ends_the_path():
    # jmp far away (tail call)
    code = b"\xe9" + struct.pack("<i", 0x10000)
    assert discover_extent(_image(code), BASE, None) == len(code)


def test_call_falls_through():
    # call +0; ret
    code = b"\xe8" + struct.pack("<i", 0) + b"\xc3"
    assert discover_extent(_image(code), BASE, None) == len(code)


def test_branch_leaving_the_window_cannot_be_closed():
    # je +0x40; ret -- the branch target lies past the limit
    code = bytes.fromhex("7440c3")
    assert discover_extent(_image(code), BASE, 8) is None


def test_switch_tables_after_the_code_are_included():
    """mov bl, [eax + bytes]; jmp [ebx*4 + jumps]; case blocks; the jump
    table and the byte index table follow the code."""
    code_len = 6 + 7 + 2  # mov bl; jmp; two one-byte cases (ret, ret)
    jumps = BASE + code_len
    index = jumps + 8
    code = (
        b"\x8a\x98"
        + struct.pack("<I", index)  # mov bl, [eax + index]
        + b"\xff\x24\x9d"
        + struct.pack("<I", jumps)  # jmp [ebx*4 + jumps]
        + b"\xc3\xc3"
    )
    tables = struct.pack("<II", BASE + 13, BASE + 14) + bytes([0, 1, 1, 0])
    assert discover_extent(_image(code + tables), BASE, None) == len(code) + len(tables)
