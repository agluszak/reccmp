"""Names both programs receive before decompilation."""

from pathlib import Path

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import (
    BinaryInput,
    Manifest,
    NamedObject,
    UnpairedEntity,
)
from reccmp.ghidriff.engine import canonical_names, unpaired_names
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
