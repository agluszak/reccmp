"""Tests for creating/matching entities using code annotations.

The markers are given as the symbols the marker reader produces from the
Clang source index; reading them from source is tested elsewhere.
"""

from pathlib import PurePath, PureWindowsPath
from unittest.mock import Mock
import pytest
from reccmp.types import EntityType, ImageId
from reccmp.formats import PEImage, TextFile
from reccmp.compare.ingest import load_markers
from reccmp.compare.db import EntityDb
from reccmp.compare.lines import LinesDb
from reccmp.compare.match_folded import match_folded_function_aliases
from reccmp.parser import DecompCodebase
from reccmp.parser.marker import MarkerType
from reccmp.parser.node import (
    ParserFunction,
    ParserLineSymbol,
    ParserString,
    ParserSymbol,
    ParserVariable,
    ParserVtable,
)

CPP = PurePath("test.cpp")
HEADER = PurePath("test.h")


@pytest.fixture(name="db")
def fixture_db():
    return EntityDb()


@pytest.fixture(name="lines_db")
def fixture_lines_db():
    return LinesDb()


def _load(db, lines_db, orig_bin, *symbols: ParserSymbol, encoding="latin1"):
    codebase = DecompCodebase(symbols, "TEST")
    files = [TextFile(path, "") for path in dict.fromkeys(s.filename for s in symbols)]
    load_markers(files, lines_db, orig_bin, codebase, db, encoding)
    return codebase


def _nameref(
    offset: int,
    name: str,
    marker: MarkerType = MarkerType.FUNCTION,
    *,
    path: PurePath = HEADER,
    symbol: bool = False,
    folded: bool = False,
) -> ParserFunction:
    return ParserFunction(
        type=marker,
        line_number=2,
        module="TEST",
        offset=offset,
        name=name,
        filename=path,
        lookup_by_name=True,
        name_is_symbol=symbol,
        is_folded=folded,
    )


def _lineref(
    offset: int,
    line: int = 2,
    end_line: int = 4,
    *,
    marker: MarkerType = MarkerType.FUNCTION,
    folded: bool = False,
) -> ParserFunction:
    return ParserFunction(
        type=marker,
        line_number=line,
        module="TEST",
        offset=offset,
        name="Pizza::Start",
        filename=CPP,
        end_line=end_line,
        is_folded=folded,
    )


def _string(offset: int, text: str, wide: bool = False) -> ParserString:
    return ParserString(
        type=MarkerType.STRING,
        line_number=2,
        module="TEST",
        offset=offset,
        name=text,
        filename=CPP,
        is_widechar=wide,
    )


def _vtable(
    offset: int, name: str, base: str | None = None, folded: bool = False
) -> ParserVtable:
    return ParserVtable(
        type=MarkerType.VTABLE,
        line_number=2,
        module="TEST",
        offset=offset,
        name=name,
        filename=HEADER,
        base_class=base,
        is_folded=folded,
    )


def _variable(offset: int, name: str, parent: int | None = None) -> ParserVariable:
    return ParserVariable(
        type=MarkerType.GLOBAL,
        line_number=2,
        module="TEST",
        offset=offset,
        name=name,
        filename=CPP,
        is_static=parent is not None,
        parent_function=parent,
    )


def test_load_code_invalid_addr(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should not create entity for an invalid address."""
    _load(db, lines_db, binfile, _lineref(0x11001000))

    # No exception raised
    assert db.get(ImageId.ORIG, 0x11001000) is None


def test_load_code_duplicate_addr(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Each address can only be used once.
    Create the entity from the annotation that appears first."""
    _load(
        db,
        lines_db,
        binfile,
        _nameref(0x1001DDE0, "_Lockit::~_Lockit"),
        _nameref(0x1001DDE0, "Hello", path=PurePath("zzz.h")),
    )

    entity = db.get(ImageId.ORIG, 0x1001DDE0)
    assert entity is not None
    assert entity.get("name") == "_Lockit::~_Lockit"


def test_load_code_cpp_symbol_function(
    db: EntityDb, lines_db: LinesDb, binfile: PEImage
):
    """Function namerefs that begin with '?' are assumed to refer to the entity symbol."""
    _load(
        db, lines_db, binfile, _nameref(0x10086240, "??2@YAPAXI@Z", MarkerType.LIBRARY)
    )

    entity = db.get(ImageId.ORIG, 0x10086240)
    assert entity is not None
    assert entity.get("symbol") == "??2@YAPAXI@Z"
    assert entity.get("name") is None


def test_load_code_c_symbol_implicit(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Namerefs that begin with '_' are NOT assumed to be the symbol.
    This would cause problems for (e.g.) STL entities like '_Tree...'"""
    _load(db, lines_db, binfile, _nameref(0x1008C410, "_strlwr", MarkerType.LIBRARY))

    entity = db.get(ImageId.ORIG, 0x1008C410)
    assert entity is not None
    assert entity.get("symbol") is None
    assert entity.get("name") == "_strlwr"


def test_load_code_c_symbol_explicit(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """If the SYMBOL annotation modifier is used, set the entity symbol instead of the name."""
    _load(
        db,
        lines_db,
        binfile,
        _nameref(0x1008C410, "_strlwr", MarkerType.LIBRARY, symbol=True),
    )

    entity = db.get(ImageId.ORIG, 0x1008C410)
    assert entity is not None
    assert entity.get("symbol") == "_strlwr"
    assert entity.get("name") is None


def test_load_code_function_nameref_variants(
    db: EntityDb, lines_db: LinesDb, binfile: PEImage
):
    """Should set extra properties for STUB and LIBRARY annotations."""
    _load(
        db,
        lines_db,
        binfile,
        _nameref(0x1001DDE0, "_Lockit::~_Lockit"),
        _nameref(
            0x1001C050,
            "Vector<unsigned char *>::~Vector<unsigned char *>",
            MarkerType.TEMPLATE,
        ),
        _nameref(0x1008B400, "_atol", MarkerType.LIBRARY),
        _lineref(0x1008B4B0, marker=MarkerType.STUB),
        _nameref(
            0x100380E0, "Pizza::`scalar deleting destructor'", MarkerType.SYNTHETIC
        ),
    )

    # n.b. These fields are always set.
    # We don't need to protect against None by using: entity.get("stub", False)

    # FUNCTION
    entity = db.get(ImageId.ORIG, 0x1001DDE0)
    assert entity is not None
    assert entity.get("type") == EntityType.FUNCTION
    assert entity.get("library") is False
    assert entity.get("stub") is False
    assert entity.get("name") == "_Lockit::~_Lockit"

    # TEMPLATE
    entity = db.get(ImageId.ORIG, 0x1001C050)
    assert entity is not None
    assert entity.get("type") == EntityType.FUNCTION
    assert entity.get("library") is False
    assert entity.get("stub") is False
    assert entity.get("name") == "Vector<unsigned char *>::~Vector<unsigned char *>"

    # LIBRARY
    entity = db.get(ImageId.ORIG, 0x1008B400)
    assert entity is not None
    assert entity.get("type") == EntityType.FUNCTION
    assert entity.get("library") is True
    assert entity.get("stub") is False
    assert entity.get("name") == "_atol"

    # STUB
    entity = db.get(ImageId.ORIG, 0x1008B4B0)
    assert entity is not None
    assert entity.get("type") == EntityType.FUNCTION
    assert not entity.get("library")
    assert entity.get("stub") is True

    # SYNTHETIC
    entity = db.get(ImageId.ORIG, 0x100380E0)
    assert entity is not None
    assert entity.get("type") == EntityType.FUNCTION
    assert entity.get("library") is False
    assert entity.get("stub") is False
    assert entity.get("name") == "Pizza::`scalar deleting destructor'"


def test_load_code_lineref(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should create a function entity for a line-based annotation."""
    _load(db, lines_db, binfile, _lineref(0x10038220))

    entity = db.get(ImageId.ORIG, 0x10038220)
    assert entity is not None

    # Nothing in the lines database. No match.
    assert entity.recomp_addr is None
    assert entity.get("type") == EntityType.FUNCTION


def test_load_code_match_line(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should match the function based on its file path and line number."""
    # Mock reading from PDB to set up the lines database.
    lines_db.add_line(PureWindowsPath("test.cpp"), 3, 0x1234)
    lines_db.mark_function_starts([0x1234])

    # Establish recomp entities as if we read the PDB first.
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0x1234)
    _load(db, lines_db, binfile, _lineref(0x10038220))

    entity = db.get(ImageId.ORIG, 0x10038220)
    assert entity is not None
    assert entity.recomp_addr == 0x1234

    # Should assign FUNCTION type as directed by the annotation.
    # The recomp entity had no type.
    assert entity.get("type") == EntityType.FUNCTION


def test_load_code_no_match_line(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Don't match the function if the line number does not match."""
    # Mock reading from PDB to set up the lines database.
    lines_db.add_line(PureWindowsPath("test.cpp"), 8, 0x1234)
    lines_db.mark_function_starts([0x1234])

    # Establish recomp entities as if we read the PDB first.
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0x1234)
    _load(db, lines_db, binfile, _lineref(0x10038220))

    entity = db.get(ImageId.ORIG, 0x10038220)
    assert entity is not None
    assert entity.recomp_addr is None
    assert entity.get("type") == EntityType.FUNCTION


def test_load_code_string(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should create a string entity from a STRING annotation."""
    _load(db, lines_db, binfile, _string(0x100F038C, "Pizza"))

    entity = db.get(ImageId.ORIG, 0x100F038C)
    assert entity is not None
    assert entity.get("type") == EntityType.STRING
    assert entity.any_size() == 6
    assert entity.get("name") == '"Pizza"'


def test_load_code_string_no_match(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Do not add the string entity if the text does not match the bytes at the address."""
    _load(db, lines_db, binfile, _string(0x100F038C, "Jetski"))

    entity = db.get(ImageId.ORIG, 0x100F038C)
    assert entity is None


def test_load_code_widechar(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should create a widechar entity from a STRING annotation."""
    _load(db, lines_db, binfile, _string(0x100DAAA0, "(null)", wide=True))

    entity = db.get(ImageId.ORIG, 0x100DAAA0)
    assert entity is not None
    assert entity.get("type") == EntityType.STRING
    assert entity.any_size() == 14
    assert entity.get("name") == 'L"(null)"'


def test_read_gb2312_string(db: EntityDb, lines_db: LinesDb):
    """Make sure we read the full length of the string in GB 2312 encoding and create the entity. GH #364"""
    string_text = "你吃饭了吗"
    string_bytes = string_text.encode("gb2312") + b"\x00"

    # Can't use RawImage here; it is missing some functions from PEImage.
    orig_bin = Mock(spec=[])
    orig_bin.read = lambda _, size: string_bytes[:size]
    orig_bin.imagebase = 0
    orig_bin.is_valid_vaddr = Mock(return_value=True)

    _load(db, lines_db, orig_bin, _string(0x1000, string_text), encoding="gb2312")

    entity = db.get(ImageId.ORIG, 0x1000)
    assert entity is not None
    assert entity.get("type") == EntityType.STRING
    assert entity.size(ImageId.ORIG) == len(string_bytes)
    assert entity.name == f'"{string_text}"'.encode("unicode_escape").decode()


def test_load_code_string_with_nulls(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should read string with nulls included.
    Using the unicode string '(null)' from the above example."""
    _load(db, lines_db, binfile, _string(0x100DAAA0, "(\x00n\x00u\x00l\x00l\x00)"))

    entity = db.get(ImageId.ORIG, 0x100DAAA0)
    assert entity is not None
    assert entity.get("type") == EntityType.STRING
    assert entity.any_size() == 12
    assert entity.get("name") == '"(\\x00n\\x00u\\x00l\\x00l\\x00)"'


def test_load_code_widechar_invalid(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should not create entity if we cannot read a widechar.
    Decoding from this address throws a UnicodeDecodeError."""
    _load(db, lines_db, binfile, _string(0x100DDA7B, "test", wide=True))

    entity = db.get(ImageId.ORIG, 0x100DDA7B)
    assert entity is None


def test_load_code_vtable(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    _load(db, lines_db, binfile, _vtable(0x100D7380, "Pizza"))

    entity = db.get(ImageId.ORIG, 0x100D7380)
    assert entity is not None
    assert entity.get("type") == EntityType.VTABLE

    # Uses the class name as the entity name. We could add the `vftable' suffix.
    assert entity.get("name") == "Pizza"
    assert entity.get("base_class") is None

    assert entity.get("folded_vtables") is None


def test_load_code_vtable_vbase(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should set base_class for VTABLE entities with virtual inheritance."""
    _load(
        db,
        lines_db,
        binfile,
        _vtable(0x100D9EC8, "Pizza", "Lunch"),
        _vtable(0x100D7380, "Pizza", "Pizza"),
    )

    entity = db.get(ImageId.ORIG, 0x100D9EC8)
    assert entity is not None
    assert entity.get("type") == EntityType.VTABLE
    assert entity.get("name") == "Pizza"
    assert entity.get("base_class") == "Lunch"

    # Should assign the base class even if it is the same as the main class.
    entity = db.get(ImageId.ORIG, 0x100D7380)
    assert entity is not None
    assert entity.get("type") == EntityType.VTABLE
    assert entity.get("name") == "Pizza"
    assert entity.get("base_class") == "Pizza"


def test_load_code_vtable_folded(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    _load(
        db,
        lines_db,
        binfile,
        _vtable(0x100D7380, "Pizza", folded=True),
        _vtable(0x100D7380, "Lunch", folded=True),
    )

    entity = db.get(ImageId.ORIG, 0x100D7380)
    assert entity is not None
    assert entity.get("type") == EntityType.VTABLE

    assert entity.get("name") == "Pizza"
    assert entity.get("base_class") is None

    assert entity.get("folded_vtables") == [("Pizza", None), ("Lunch", None)]


def test_load_code_variable(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    _load(db, lines_db, binfile, _variable(0x10102048, "g_strACTION"))

    entity = db.get(ImageId.ORIG, 0x10102048)
    assert entity is not None
    assert entity.get("type") == EntityType.DATA
    assert entity.get("name") == "g_strACTION"


def test_load_code_static_variable(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should create a static variable entity if the function is also annotated."""
    _load(
        db,
        lines_db,
        binfile,
        _lineref(0x1009DA20),
        _variable(0x10109594, "g_dwStyle", parent=0x1009DA20),
    )

    entity = db.get(ImageId.ORIG, 0x10109594)
    assert entity is not None
    assert entity.get("type") == EntityType.DATA
    assert entity.get("name") == "g_dwStyle"
    assert entity.get("static_var") is True
    assert entity.get("parent_function") == 0x1009DA20


def test_load_code_line_marker(db: EntityDb, lines_db: LinesDb, binfile: PEImage):
    """Should create a LINE entity with the local file path and line number."""
    line = ParserLineSymbol(
        type=MarkerType.LINE,
        line_number=3,
        module="TEST",
        offset=0x10001038,
        name="test.cpp:3",
        filename=CPP,
    )
    _load(db, lines_db, binfile, line)

    entity = db.get(ImageId.ORIG, 0x10001038)
    assert entity is not None
    assert entity.get("type") == EntityType.LINE
    assert entity.get("filename") == "test.cpp"
    assert entity.get("line") == 3


def test_load_code_folded(db: EntityDb, lines_db: LinesDb):
    """Bind the canonical annotation first, then record the folded recomp body as an alias."""
    orig_bin = Mock(spec=PEImage)
    orig_bin.is_valid_vaddr.return_value = True

    # Each recomp body has its own line entry; one is the canonical match and the
    # other is recorded as a recomp-side alias after that match exists.
    lines_db.add_line(PureWindowsPath("test.cpp"), 3, 0x1234)
    lines_db.add_line(PureWindowsPath("test.cpp"), 7, 0x5678)
    lines_db.mark_function_starts([0x1234, 0x5678])

    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0x1234, type=EntityType.FUNCTION)
        batch.set(ImageId.RECOMP, 0x5678, type=EntityType.FUNCTION)
    codebase = _load(
        db,
        lines_db,
        orig_bin,
        _lineref(0x10001000, 2, 4),
        _lineref(0x10001000, 6, 8, folded=True),
    )
    match_folded_function_aliases(db, codebase, lines_db)

    entity = db.get(ImageId.ORIG, 0x10001000)
    assert entity is not None
    assert entity.recomp_addr == 0x1234
    assert entity.get("type") == EntityType.FUNCTION
    assert db.alias_canonical_orig(ImageId.RECOMP, 0x5678) == 0x10001000
