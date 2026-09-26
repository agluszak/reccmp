"""Name spellings that carry no identity (reccmp.compare.match_msvc.match_name)."""

import pytest

from reccmp.compare.db import EntityDb
from reccmp.compare.match_msvc import match_name, match_vtables
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


def test_match_name_spells_template_arguments_tight():
    assert match_name("std::basic_string<char, struct std::char_traits<char> >") == (
        "std::basic_string<char,std::char_traits<char>>"
    )
    assert match_name("f<T *, U &>") == "f<T*,U&>"
    # An identifier that merely ends in a keyword is left alone.
    assert match_name("Holder<myclass X>") == "Holder<myclass X>"
