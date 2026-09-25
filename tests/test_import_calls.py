"""A call through an import thunk is a call through the import slot."""

from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.replacement import entity_proof_identity
from reccmp.compare.asm.verifier.semantics import _import_call
from reccmp.compare.db import EntityDb
from reccmp.types import EntityType, ImageId

IMPORT = "SR.dll::??1srNode@@MAE@XZ"


def _thunk_identity(db: EntityDb):
    thunk = db.get(ImageId.RECOMP, 0x1000)
    assert thunk is not None
    return entity_proof_identity(db, ImageId.RECOMP, thunk)


def _db() -> EntityDb:
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0x5000, type=EntityType.IMPORT, name=IMPORT)
        batch.set(ImageId.RECOMP, 0x1000, type=EntityType.IMPORT_THUNK, size=6)
        batch.set_ref(ImageId.RECOMP, 0x1000, ref=0x5000)
    return db


def test_an_import_thunk_is_identified_by_the_slot_it_jumps_through():
    db = _db()
    assert _thunk_identity(db) == ("jmp_through", ("import", IMPORT))
    # A paired slot has its pair's identity, and so does the thunk.
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x7000, type=EntityType.IMPORT, name=IMPORT)
    db.bulk_match([(0x7000, 0x5000)])
    assert _thunk_identity(db) == ("jmp_through", ("entity", 0x7000, 0))


def test_calls_through_the_thunk_and_the_slot_have_one_callee():
    slot = ("entity", 0x7000, 0)
    reference = Reference(display=IMPORT, identity=slot)
    through_slot = ("load", ("mem", "", (), 0, ((1, reference),)), "dword", 0)
    through_thunk = ("sym", ("jmp_through", slot))
    assert _import_call(through_slot) == _import_call(through_thunk)
    assert _import_call(through_thunk) == ("call_through", slot)
    # [slot + 4], [reg + slot], a word load and other symbols keep their identity
    offset = ("load", ("mem", "", (), 4, ((1, reference),)), "dword", 0)
    indexed = (
        "load",
        ("mem", "", ((("init", "a"), 4),), 0, ((1, reference),)),
        "dword",
        0,
    )
    word = ("load", ("mem", "", (), 0, ((1, reference),)), "word", 0)
    for value in (offset, indexed, word, ("sym", ("entity", 0x1000, 0))):
        assert _import_call(value) == value
