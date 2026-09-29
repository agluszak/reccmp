"""Slot-by-slot vtable comparison."""

from reccmp.compare.db import EntityDb, PairBasis, ReccmpMatch
from reccmp.compare.vtables import SlotStatus, compare_vtable
from reccmp.types import EntityType, ImageId
from .raw_image import RawImage


def _vtable(db: EntityDb, orig_addr: int) -> ReccmpMatch:
    match = db.get_one_match(orig_addr)
    assert match is not None
    return match


def test_vtable_match():
    orig_bin = RawImage.from_memory(b"\x00" * 0x1004 + b"\x00\x10\x00\x00")
    recomp_bin = RawImage.from_memory(b"\x00" * 0x1004 + b"\x00\x10\x00\x00")

    db = EntityDb()
    with db.batch() as batch:
        batch.set(
            ImageId.RECOMP, 0x1000, type=EntityType.FUNCTION, name="hello", size=1
        )
        batch.set(ImageId.RECOMP, 0x1004, type=EntityType.VTABLE, name="test", size=4)
        batch.match(0x1000, 0x1000, basis=PairBasis.ANNOTATION)
        batch.match(0x1004, 0x1004, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, 0x1004))
    assert result.matches
    [slot] = result.slots
    assert slot.offset == 0
    assert slot.orig is not None and slot.orig.best_name() == "hello"


def test_vtable_diff():
    functions = b"\xc3\x00\x00\x00" * 3
    slot0 = b"\x00\x10\x00\x00"
    slot1 = b"\x04\x10\x00\x00"
    slot2 = b"\x08\x10\x00\x00"
    orig_mem = b"\x00" * 0x1000 + functions + slot2 + slot0 * 30 + slot1
    recomp_mem = b"\x00" * 0x1000 + functions + slot1 + slot0 * 30 + slot2

    orig_bin = RawImage.from_memory(orig_mem)
    recomp_bin = RawImage.from_memory(recomp_mem)

    db = EntityDb()
    with db.batch() as batch:
        for addr, name in ((0x1000, "func0"), (0x1004, "func1"), (0x1008, "func2")):
            batch.set(ImageId.RECOMP, addr, type=EntityType.FUNCTION, name=name, size=1)
            batch.match(addr, addr, basis=PairBasis.ANNOTATION)
        batch.set(
            ImageId.RECOMP,
            0x100C,
            type=EntityType.VTABLE,
            name="hello",
            size=len(orig_mem) - len(functions) - 0x1000,
        )
        batch.match(0x100C, 0x100C, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, 0x100C))
    assert not result.matches
    assert len(result.slots) == 32
    mismatched = [slot.offset for slot in result.slots if not slot.matches]
    assert mismatched == [0, 31 * 4]


def test_vtable_thunk_resolution():
    """A slot pointing at an incremental-link jmp thunk equals a slot that
    points directly at the thunk's target."""
    orig_mem = (
        b"\xc3\x00\x00\x00"  # function@0
        + b"\xe9\xf7\xff\xff\xff"  # thunk@4: jmp -9 (-> 0)
        + b"\xcc\xcc\xcc"
        + b"\x04\x00\x00\x00"  # vtable@12: slot -> thunk@4
    )
    recomp_mem = b"\xc3\x00\x00\x00" + b"\x00\x00\x00\x00"

    orig_bin = RawImage.from_memory(orig_mem)
    recomp_bin = RawImage.from_memory(recomp_mem)

    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0, type=EntityType.FUNCTION, name="func", size=1)
        batch.set(ImageId.ORIG, 4, type=EntityType.THUNK, name="func_thunk", ref=0)
        batch.set(ImageId.RECOMP, 4, type=EntityType.VTABLE, name="hello", size=4)
        batch.match(0, 0, basis=PairBasis.ANNOTATION)
        batch.match(12, 4, basis=PairBasis.ANNOTATION)

    assert compare_vtable(db, orig_bin, recomp_bin, _vtable(db, 12)).matches


def test_vtable_thunk_chain_resolution():
    """Bad ILT aliases may jmp through an unregistered mid-.text stub before
    the real ILT entry and function body."""
    orig_mem = (
        b"\xc3\x00\x00\x00"  # function@0
        + b"\xe9\x07\x00\x00\x00"  # bad ILT@4: jmp 0x10
        + b"\x90\x90\x90\x90\x90\x90"
        + b"\xe9\x03\x00\x00\x00"  # mid@0x10: jmp 0x18
        + b"\xe9\xe3\xff\xff\xff"  # good ILT@0x18: jmp 0x00
        + b"\x00" * (0x3C - 0x1D)
        + b"\x04\x00\x00\x00"  # vtable@0x3c
        + b"\x00" * 4
    )
    recomp_mem = b"\xcc" * 4 + b"\x08\x00\x00\x00" + b"\xc3"

    orig_bin = RawImage.from_memory(orig_mem)
    recomp_bin = RawImage.from_memory(recomp_mem)

    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x00, type=EntityType.FUNCTION, name="func", size=1)
        batch.set(ImageId.RECOMP, 0x08, type=EntityType.FUNCTION, name="func", size=1)
        batch.set(ImageId.ORIG, 0x04, type=EntityType.THUNK, name="bad", ref=0x10)
        batch.set(ImageId.ORIG, 0x18, type=EntityType.THUNK, name="good", ref=0x00)
        batch.set(ImageId.RECOMP, 0x04, type=EntityType.VTABLE, name="hello", size=4)
        batch.match(0x00, 0x08, basis=PairBasis.ANNOTATION)
        batch.match(0x3C, 0x04, basis=PairBasis.ANNOTATION)

    assert compare_vtable(db, orig_bin, recomp_bin, _vtable(db, 0x3C)).matches


def test_vtable_null_orig_slot():
    """Literal NULL slots in the orig vtable accept any recomp slot."""
    orig_mem = b"\x00\x00\x00\x00" + b"\x04\x00\x00\x00" + b"\xc3\x00\x00\x00"
    recomp_mem = b"\xcc" * 4 + b"\xde\xad\xbe\xef" + b"\x08\x00\x00\x00" + b"\xc3"

    orig_bin = RawImage.from_memory(orig_mem)
    recomp_bin = RawImage.from_memory(recomp_mem)

    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x04, type=EntityType.FUNCTION, name="func", size=1)
        batch.set(ImageId.RECOMP, 0x08, type=EntityType.FUNCTION, name="func", size=1)
        batch.set(ImageId.ORIG, 0x00, type=EntityType.VTABLE, name="hello", size=8)
        batch.set(ImageId.RECOMP, 0x04, type=EntityType.VTABLE, name="hello", size=8)
        batch.match(0x04, 0x08, basis=PairBasis.ANNOTATION)
        batch.match(0x00, 0x04, basis=PairBasis.ANNOTATION)

    assert compare_vtable(db, orig_bin, recomp_bin, _vtable(db, 0)).matches


def test_vtable_recomp_longer():
    """An extra virtual function on the recomp side is a mismatching slot."""
    base_addr = 0x400000
    functions = b"\xc3\x00\x00\x00" * 2
    func0_ptr = base_addr.to_bytes(4, "little")
    func1_ptr = (base_addr + 4).to_bytes(4, "little")

    orig_bin = RawImage.from_memory(functions + func0_ptr, base_addr=base_addr)
    recomp_bin = RawImage.from_memory(
        functions + func0_ptr + func1_ptr, base_addr=base_addr
    )

    db = EntityDb()
    with db.batch() as batch:
        for addr, name in ((base_addr, "func0"), (base_addr + 4, "func1")):
            batch.set(ImageId.RECOMP, addr, type=EntityType.FUNCTION, name=name, size=1)
            batch.match(addr, addr, basis=PairBasis.ANNOTATION)
        batch.set(
            ImageId.ORIG, base_addr + 8, type=EntityType.VTABLE, name="hello", size=4
        )
        batch.set(
            ImageId.RECOMP, base_addr + 8, type=EntityType.VTABLE, name="hello", size=8
        )
        batch.match(base_addr + 8, base_addr + 8, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base_addr + 8))
    assert not result.matches
    extra = result.slots[1]
    assert extra.orig_raw is None
    assert extra.recomp is not None and extra.recomp.best_name() == "func1"


def test_vtable_recomp_trailing_padding():
    """Alignment padding after the recomp vtable is not an extra slot."""
    base_addr = 0x400000
    func0_ptr = base_addr.to_bytes(4, "little")
    mem = b"\xc3\x00\x00\x00" + func0_ptr + b"\x00\x00\x00\x00"

    orig_bin = RawImage.from_memory(mem, base_addr=base_addr)
    recomp_bin = RawImage.from_memory(mem, base_addr=base_addr)

    db = EntityDb()
    with db.batch() as batch:
        batch.set(
            ImageId.RECOMP, base_addr, type=EntityType.FUNCTION, name="func0", size=1
        )
        batch.set(
            ImageId.ORIG, base_addr + 4, type=EntityType.VTABLE, name="hello", size=4
        )
        batch.set(
            ImageId.RECOMP, base_addr + 4, type=EntityType.VTABLE, name="hello", size=8
        )
        batch.match(base_addr, base_addr, basis=PairBasis.ANNOTATION)
        batch.match(base_addr + 4, base_addr + 4, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base_addr + 4))
    assert result.matches
    assert len(result.slots) == 1


def test_slot_at_an_unpaired_function_is_unpaired_not_different():
    """The original linker may fold identical functions: a recompiled slot
    at a function reccmp has not paired is unresolved, not wrong."""
    base = 0x1000
    ret = b"\xc3\x00\x00\x00"
    orig_bin = RawImage.from_memory(ret + base.to_bytes(4, "little"), base_addr=base)
    recomp_bin = RawImage.from_memory(
        ret + ret + (base + 4).to_bytes(4, "little"), base_addr=base
    )

    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, base, type=EntityType.FUNCTION, name="Base::f")
        batch.set(ImageId.RECOMP, base + 4, type=EntityType.FUNCTION, name="Derived::f")
        batch.set(
            ImageId.ORIG, base + 4, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.set(
            ImageId.RECOMP, base + 8, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.match(base, base, basis=PairBasis.ANNOTATION)
        batch.match(base + 4, base + 8, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base + 4))
    assert [slot.status for slot in result.slots] == [SlotStatus.UNPAIRED]


def test_unpaired_return_only_slot_is_code_equivalent():
    base = 0x1000
    ret4 = b"\xc2\x04\x00"
    orig_bin = RawImage.from_memory(
        ret4 + b"\x90" + base.to_bytes(4, "little"), base_addr=base
    )
    recomp_bin = RawImage.from_memory(
        ret4 + b"\x90" + ret4 + b"\x90" + (base + 4).to_bytes(4, "little"),
        base_addr=base,
    )

    db = EntityDb()
    with db.batch() as batch:
        batch.set(
            ImageId.RECOMP, base, type=EntityType.FUNCTION, name="Base::f", size=3
        )
        batch.set(
            ImageId.RECOMP,
            base + 4,
            type=EntityType.FUNCTION,
            name="Derived::f",
            size=3,
        )
        batch.set(
            ImageId.ORIG, base + 4, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.set(
            ImageId.RECOMP, base + 8, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.match(base, base, basis=PairBasis.ANNOTATION)
        batch.match(base + 4, base + 8, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base + 4))
    assert [slot.status for slot in result.slots] == [SlotStatus.CODE_EQUIVALENT]
    assert result.matches


def test_unpaired_return_only_slot_requires_same_stack_cleanup():
    base = 0x1000
    orig_bin = RawImage.from_memory(
        b"\xc2\x04\x00\x90" + base.to_bytes(4, "little"), base_addr=base
    )
    recomp_bin = RawImage.from_memory(
        b"\xc2\x04\x00\x90\xc2\x08\x00\x90" + (base + 4).to_bytes(4, "little"),
        base_addr=base,
    )

    db = EntityDb()
    with db.batch() as batch:
        batch.set(
            ImageId.RECOMP, base, type=EntityType.FUNCTION, name="Base::f", size=3
        )
        batch.set(
            ImageId.RECOMP,
            base + 4,
            type=EntityType.FUNCTION,
            name="Derived::f",
            size=3,
        )
        batch.set(
            ImageId.ORIG, base + 4, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.set(
            ImageId.RECOMP, base + 8, type=EntityType.VTABLE, name="Derived", size=4
        )
        batch.match(base, base, basis=PairBasis.ANNOTATION)
        batch.match(base + 4, base + 8, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base + 4))
    assert [slot.status for slot in result.slots] == [SlotStatus.UNPAIRED]


def test_slots_at_different_pairs_are_different():
    base = 0x1000
    ret = b"\xc3\x00\x00\x00"
    orig_bin = RawImage.from_memory(
        ret + ret + base.to_bytes(4, "little"), base_addr=base
    )
    recomp_bin = RawImage.from_memory(
        ret + ret + (base + 4).to_bytes(4, "little"), base_addr=base
    )

    db = EntityDb()
    with db.batch() as batch:
        for addr, name in ((base, "A::f"), (base + 4, "B::f")):
            batch.set(ImageId.RECOMP, addr, type=EntityType.FUNCTION, name=name)
            batch.match(addr, addr, basis=PairBasis.ANNOTATION)
        batch.set(ImageId.RECOMP, base + 8, type=EntityType.VTABLE, name="T", size=4)
        batch.match(base + 8, base + 8, basis=PairBasis.ANNOTATION)

    result = compare_vtable(db, orig_bin, recomp_bin, _vtable(db, base + 8))
    assert [slot.status for slot in result.slots] == [SlotStatus.DIFFERENT]
