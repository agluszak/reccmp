from pathlib import Path
from types import SimpleNamespace
from typing import cast
import struct

import pytest

from reccmp.compare.exact import (
    compare_relocation_masked,
    mask_relocations,
    original_absolute_relocation_targets,
    parse_coff_functions,
    stable_ranges,
)
from reccmp.formats.coff import (
    CoffObject,
    CoffRelocation,
    CoffSection,
    CoffSymbol,
    parse_coff_object,
)
from reccmp.formats.pe import PEImage


def test_relocation_masked_exact_comparison_reports_mode() -> None:
    result = compare_relocation_masked(
        b"\x55\x11\x22\x33\x44\xc3",
        b"\x55\xaa\xbb\xcc\xdd\xc3",
        recompiled_relocations=(1,),
    )

    assert result.exact
    assert result.status == "exact"
    assert result.exact_mode == "relocation-masked-object"
    assert result.stable_bytes == 2


def test_exact_comparison_honours_function_extent() -> None:
    result = compare_relocation_masked(b"\x90\xc3", b"\x90\xc3\x00\x01", size=2)
    assert result.exact
    assert result.original_size == 2
    assert result.recompiled_size == 4


def test_stable_ranges_and_masking_clip_invalid_relocations() -> None:
    assert stable_ranges(8, (-2, 2, 7)) == [(0, 2), (6, 8)]
    assert mask_relocations(b"abcdefgh", (-2, 2, 7)) == b"ab\0\0\0\0gh"


def test_data_only_object_keeps_storage_and_relocation_targets(tmp_path) -> None:
    # Ordinary i386 COFF: one data section, four symbols, one DIR32 relocation.
    path = tmp_path / "data.obj"
    symbol_table = 20 + 40 + 12 + 10
    header = struct.pack("<HHIIIHH", 0x14C, 1, 0, symbol_table, 4, 0, 0)
    section = struct.pack(
        "<8sIIIIIIHHI", b".data", 0, 0, 12, 60, 72, 0, 1, 0, 0xC0000040
    )
    symbols = b"".join(
        struct.pack("<8sIhHBB", name, value, index, 0, storage, 0)
        for name, value, index, storage in (
            (b"_table", 0, 1, 2),
            (b"local", 8, 1, 3),
            (b"_extern", 0, 0, 2),
            (b"_common", 16, 0, 2),
        )
    )
    path.write_bytes(
        header
        + section
        + b"abcdefghijkl"
        + struct.pack("<IIH", 4, 2, 6)
        + symbols
        + struct.pack("<I", 4)
    )
    obj = parse_coff_object(path)
    table = obj.contribution("_table")
    assert table.data == b"abcdefgh"
    assert table.relocations[0].offset == 4
    assert obj.symbols[table.relocations[0].symbol_index].name == "_extern"
    assert obj.contribution("local").data == b"ijkl"
    assert obj.contribution("_common").data == bytes(16)
    assert parse_coff_functions(path) == []


def test_static_function_auxiliary_extent_preserves_nop_and_alias(tmp_path) -> None:
    path = tmp_path / "static.obj"
    header = struct.pack("<HHIIIHH", 0x14C, 1, 0, 64, 3, 0, 0)
    section = struct.pack("<8sIIIIIIHHI", b".text", 0, 0, 4, 60, 0, 0, 0, 0, 0x60000020)
    function = struct.pack("<8sIhHBB", b"_local", 0, 1, 0x20, 3, 1)
    auxiliary = struct.pack("<IIIIH", 0, 3, 0, 0, 0)
    alias = struct.pack("<8sIhHBB", b"_alias", 0, 1, 0x20, 2, 0)
    path.write_bytes(
        header
        + section
        + b"\x55\xc3\x90\xcc"
        + function
        + auxiliary
        + alias
        + struct.pack("<I", 4)
    )
    obj = parse_coff_object(path)
    assert obj.contribution("_local").data == b"\x55\xc3\x90"
    assert obj.contribution("_alias").data == b"\x55\xc3\x90\xcc"
    assert {f.name for f in parse_coff_functions(path)} == {"local", "alias"}


@pytest.mark.parametrize("addend", [12, -8])
def test_original_absolute_target_accounts_for_addend_and_symbol_index(addend) -> None:
    body = b"\xa1" + struct.pack("<I", addend & 0xFFFFFFFF) + b"\xc3"
    source = CoffSymbol(0, "_read", 0, 1, 0x20, 2, (bytes(18),))
    target = CoffSymbol(2, "_global", 0, 0, 0, 2, ())
    obj = CoffObject(
        Path("read.obj"),
        (
            CoffSection(
                1, ".text", len(body), body, 0x60000020, (CoffRelocation(1, 2, 6),)
            ),
        ),
        (source, target),
    )
    linked = b"\xa1" + struct.pack("<I", 0x402000 + addend) + b"\xc3"
    original = cast(
        PEImage,
        SimpleNamespace(
            read=lambda address, size: linked[
                address - 0x401000 : address - 0x401000 + size
            ],
            relocations={0x401001},
        ),
    )
    observations = original_absolute_relocation_targets(
        original, obj, "_read", 0x401000, 6
    )
    assert len(observations) == 1
    assert observations[0].symbol is target
    assert observations[0].offset == 1
    assert observations[0].address == 0x402000

    linked = b"\x90" + linked[1:]
    with pytest.raises(ValueError, match="exact paired contribution"):
        original_absolute_relocation_targets(original, obj, "_read", 0x401000, 6)
