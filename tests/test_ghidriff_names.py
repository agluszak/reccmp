"""Names both programs receive before decompilation."""

from pathlib import Path

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import (
    BinaryInput,
    Manifest,
    NamedObject,
    UnpairedEntity,
)
from reccmp.ghidriff.engine import (  # pylint: disable=protected-access
    _Extents,
    canonical_names,
    unpaired_names,
)
from reccmp.types import EntityType, ImageId


def _object(orig_addr: int, name: str) -> NamedObject:
    return NamedObject(
        orig_addr=orig_addr,
        recomp_addr=orig_addr + 0x1000,
        name=name,
        entity_type=EntityType.FUNCTION,
        orig_size=None,
        recomp_size=None,
        basis=PairBasis.ANNOTATION,
    )


def test_shared_names_are_qualified_by_the_pair_identity():
    names = canonical_names(
        (_object(0x10, "Unique"), _object(0x20, "Twice"), _object(0x30, "Twice"))
    )
    assert names == {0x10: "Unique", 0x20: "Twice@0x20", 0x30: "Twice@0x30"}


def _manifest(*unpaired: UnpairedEntity) -> Manifest:
    binary = BinaryInput(Path("x"), "0")
    return Manifest("T", binary, binary, (), (), unpaired)


def test_unpaired_names_never_look_like_a_correspondence():
    manifest = _manifest(
        UnpairedEntity(ImageId.RECOMP, 0x100, None, EntityType.FUNCTION, "PLLength"),
        UnpairedEntity(ImageId.ORIG, 0x200, None, EntityType.FUNCTION, "ILLength"),
        UnpairedEntity(ImageId.ORIG, 0x300, None, EntityType.FUNCTION, "Dup"),
        UnpairedEntity(ImageId.ORIG, 0x400, None, EntityType.FUNCTION, "Dup"),
        UnpairedEntity(ImageId.ORIG, 0x500, 3, EntityType.STRING, '"hi"'),
    )
    names = unpaired_names(manifest, shared={"ILLength"})
    assert names == {
        (ImageId.RECOMP, 0x100): "PLLength",
        # A shared name elsewhere would suggest this is that pair.
        (ImageId.ORIG, 0x200): "ILLength@0x200",
        (ImageId.ORIG, 0x300): "Dup@0x300",
        (ImageId.ORIG, 0x400): "Dup@0x400",
    }


def _data(orig_addr: int, name: str, size: int, entity_type=EntityType.DATA):
    return NamedObject(
        orig_addr=orig_addr,
        recomp_addr=orig_addr + 0x1000,
        name=name,
        entity_type=entity_type,
        orig_size=size,
        recomp_size=size,
        basis=PairBasis.ANNOTATION,
    )


def _extents(*objects: NamedObject) -> _Extents:
    binary = BinaryInput(Path("x"), "0")
    return _Extents(Manifest("T", binary, binary, (), objects, ()), ImageId.ORIG)


def test_a_loop_bound_at_an_array_end_belongs_to_the_array():
    table = _data(0x100, "g_table", 0x24)
    flag = _data(0x124, "g_flag", 4)
    bound = _extents(table, flag).bound_at(0x124)
    # The array's end, not the object the linker placed after it.
    assert bound is not None and bound.named == table and bound.offset == 0x24


def test_a_field_bound_past_an_array_belongs_to_the_array():
    # Six 12-byte elements; the loop steps through the field at offset 8.
    table = _data(0x100, "g_animations", 0x48)
    text = _data(0x148, "s_text", 0x24, EntityType.STRING)
    bound = _extents(table, text).bound_at(0x150)
    assert bound is not None and bound.named == table and bound.offset == 0x50


def test_a_paired_object_at_a_bound_past_the_end_is_itself():
    table = _data(0x100, "g_table", 0x48)
    other = _data(0x150, "g_other", 8)
    extents = _extents(table, other)
    assert extents.bound_at(0x150) is None
    assert extents.bound_at(0x154) is None
    assert extents.bound_at(0x100 + 2 * 0x48) is None
