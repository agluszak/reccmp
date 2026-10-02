"""Names both programs receive before decompilation."""

from pathlib import Path

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import (
    BinaryInput,
    Manifest,
    NamedObject,
    UnpairedEntity,
)
from reccmp.ghidriff.engine import _canonical_parameter_names
from reccmp.ghidriff.names import (
    canonical_names,
    paired_reference_tokens,
    replace_paired_raw_addresses,
    unquoted_raw_addresses,
    unpaired_names,
)
from reccmp.ghidriff.locations import Extents
from reccmp.ghidriff.results import DataReference, ObjectOffset, UnknownExtent
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


def test_only_shared_paired_data_references_normalize_raw_addresses():
    shared = ObjectOffset(0x617584, "g_format_s_space_s", 0)
    orig_only = ObjectOffset(0x605880, "g_other", 0)
    original = (
        DataReference(0x617584, shared, UnknownExtent()),
        DataReference(0x605880, orig_only, UnknownExtent()),
    )
    recomp = (DataReference(0x627C04, shared, UnknownExtent()),)
    orig_tokens, recomp_tokens = paired_reference_tokens(original, recomp)
    assert orig_tokens == {0x617584: "PAIRED_DATA_617584_0"}
    assert recomp_tokens == {0x627C04: "PAIRED_DATA_617584_0"}

    old_code = [
        '  swprintf(buffer,0x617584,"0x617584");\n',
        "  use(0x605880);\n",
        "/* 0x617584 */\n",
    ]
    new_code = ['  swprintf(buffer,0x627c04,"0x627c04");\n']
    replace_paired_raw_addresses(old_code, orig_tokens)
    replace_paired_raw_addresses(new_code, recomp_tokens)
    assert old_code == [
        '  swprintf(buffer,PAIRED_DATA_617584_0,"0x617584");\n',
        "  use(0x605880);\n",
        "/* 0x617584 */\n",
    ]
    assert new_code == ['  swprintf(buffer,PAIRED_DATA_617584_0,"0x627c04");\n']


def test_paired_address_normalizes_when_ghidra_misses_one_reference():
    shared = ObjectOffset(0x689B34, "g_empty_wide_string", 0)
    original = (DataReference(0x689B34, shared, UnknownExtent()),)
    recomp = (
        DataReference(0x6550A0, shared, UnknownExtent()),
        DataReference(0x6550A2, shared, UnknownExtent()),
    )
    orig_tokens, recomp_tokens = paired_reference_tokens(
        original, recomp, {0x689B34: 0x6550A0}
    )
    assert orig_tokens == {0x689B34: "PAIRED_DATA_689b34_0"}
    code = ["  swprintf(buffer,0x6550a0,text);\n"]
    replace_paired_raw_addresses(code, recomp_tokens)
    assert code == ["  swprintf(buffer,PAIRED_DATA_689b34_0,text);\n"]


def test_raw_address_candidates_ignore_strings_and_comments():
    code = '  swprintf(buffer,0x6550a0,"0x689b34");\n/* 0x689b34 */\n'
    assert unquoted_raw_addresses(code) == {0x6550A0}


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


def _extents(*objects: NamedObject) -> Extents:
    binary = BinaryInput(Path("x"), "0")
    return Extents(Manifest("T", binary, binary, (), objects, ()), ImageId.ORIG)


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


def test_default_parameter_names_use_reccmp_numbering():
    code = [
        "void __cdecl F(int param_1,int param0)\n",
        "{\n",
        '  puts("param_1 stays");\n',
        "  return param_1 + param_12 + xparam_1;\n",
    ]
    _canonical_parameter_names(code)
    assert code == [
        "void __cdecl F(int param0,int param0)\n",
        "{\n",
        '  puts("param_1 stays");\n',
        "  return param0 + param11 + xparam_1;\n",
    ]
