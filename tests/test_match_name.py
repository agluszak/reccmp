"""Name spellings that carry no identity (reccmp.compare.match_msvc.match_name)."""

import pytest

from reccmp.compare.db import EntityDb
from reccmp.compare.match_msvc import (
    match_name,
    match_vtables,
    match_annotation_selectors,
    match_symbols,
    match_functions,
)
from reccmp.compare.event import ReccmpEvent
from reccmp.types import EntityType, ImageId


@pytest.fixture(name="db")
def fixture_db():
    return EntityDb()


def test_match_vtables_of_template_classes(db):
    """The demangler spells template arguments with elaborated keywords and
    spaces; an annotation spells them tight. Different arguments still differ."""
    with db.batch() as batch:
        batch.set(
            ImageId.ORIG,
            100,
            name="srClassSupport<Trigger,srClass,1,65544>",
            type=EntityType.VTABLE,
        )
        batch.set(
            ImageId.ORIG,
            110,
            name="W8GrowableVector<srVector3T<float>>",
            type=EntityType.VTABLE,
        )
        batch.set(
            ImageId.ORIG,
            120,
            name="srClassSupport<Trigger,srClass,1,65545>",
            type=EntityType.VTABLE,
        )
        batch.set(
            ImageId.RECOMP,
            200,
            name="srClassSupport<Trigger, class srClass, 1, 65544>::`vftable'",
            type=EntityType.VTABLE,
        )
        batch.set(
            ImageId.RECOMP,
            210,
            name="W8GrowableVector<srVector3T<float> >::`vftable'",
            type=EntityType.VTABLE,
        )

    match_vtables(db)

    assert db.get(ImageId.ORIG, 100).recomp_addr == 200
    assert db.get(ImageId.ORIG, 110).recomp_addr == 210
    assert db.get(ImageId.ORIG, 120).recomp_addr is None


def test_match_name_spells_value_arguments_in_decimal():
    assert match_name("srClassSupport<srPalette, srClass, true, 0x2900>::vClone") == (
        "srClassSupport<srPalette,srClass,1,10496>::vClone"
    )
    assert match_name("S<false,0X10>") == "S<0,16>"
    assert match_name("f<my_true,0x10u>") == "f<my_true,0x10u>"


def test_match_name_spells_template_arguments_tight():
    assert match_name("std::basic_string<char, struct std::char_traits<char> >") == (
        "std::basic_string<char,std::char_traits<char>>"
    )
    assert match_name("f<T *, U &>") == "f<T*,U&>"
    # An identifier that merely ends in a keyword is left alone.
    assert match_name("Holder<myclass X>") == "Holder<myclass X>"


def test_recovered_selector_preserves_original_export_and_overload(db):
    with db.batch() as batch:
        batch.set(
            ImageId.ORIG,
            100,
            type=EntityType.FUNCTION,
            symbol="original_export",
            source_name="f<T>::member",
            recomp_selector="?f@@YAXH@Z",
            selector_is_symbol=True,
        )
        batch.set(
            ImageId.RECOMP, 200, type=EntityType.FUNCTION, name="f", symbol="?f@@YAXH@Z"
        )
        batch.set(
            ImageId.RECOMP, 210, type=EntityType.FUNCTION, name="f", symbol="?f@@YAXM@Z"
        )
    match_annotation_selectors(db)
    entity = db.get(ImageId.ORIG, 100)
    assert entity.recomp_addr == 200
    assert entity.fact(ImageId.ORIG, "symbol") == "original_export"
    assert entity.fact(ImageId.RECOMP, "symbol") == "?f@@YAXH@Z"


@pytest.mark.parametrize(
    "duplicate_orig,duplicate_recomp", [(True, False), (False, True)]
)
def test_ambiguous_selector_has_no_guessed_fallback(
    db, duplicate_orig, duplicate_recomp
):
    with db.batch() as batch:
        batch.set(
            ImageId.ORIG,
            100,
            type=EntityType.FUNCTION,
            name="f",
            symbol="?f@@YAXH@Z",
            recomp_selector="f",
        )
        if duplicate_orig:
            batch.set(ImageId.ORIG, 110, type=EntityType.FUNCTION, recomp_selector="f")
        batch.set(
            ImageId.RECOMP, 200, type=EntityType.FUNCTION, name="f", symbol="?f@@YAXH@Z"
        )
        if duplicate_recomp:
            batch.set(
                ImageId.RECOMP,
                210,
                type=EntityType.FUNCTION,
                name="f",
                symbol="?f@@YAXM@Z",
            )
    events = []

    def report(event: ReccmpEvent, addr: int, /, msg: str = "") -> None:
        events.append((event, addr, msg))

    match_annotation_selectors(db, report)
    match_symbols(db)
    match_functions(db)
    assert any(event == ReccmpEvent.AMBIGUOUS_MATCH for event, _, _ in events)
    assert db.get(ImageId.ORIG, 100).recomp_addr is None
