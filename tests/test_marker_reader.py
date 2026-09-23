"""The marker grammar applied to the comment blocks the Clang indexer reports.

Each block is built the way the indexer emits it: consecutive comment lines,
and the declarations that begin at the first code token after them.
"""

from pathlib import PurePath

from reccmp.parser.error import AlertCode
from reccmp.parser.marker import MarkerType
from reccmp.parser.node import (
    ParserFunction,
    ParserLineSymbol,
    ParserString,
    ParserVariable,
    ParserVtable,
)
from reccmp.parser.reader import (
    AnchorCandidate,
    MarkerAnchor,
    MarkerBlock,
    MarkerComment,
    MarkerString,
    local_paths,
    read_marker_blocks,
    uncompiled_markers,
)

PATH = PurePath("/repo/src/test.cpp")


def _block(
    *comments: str,
    first_line: int = 1,
    gap: int = 0,
    blank: bool = True,
    candidates: tuple[AnchorCandidate, ...] = (),
    string: MarkerString | None = None,
    anchored: bool = True,
) -> MarkerBlock:
    """``gap`` lines separate the block from its code; ``blank`` says
    whether the first of them is empty (else a block comment or pragma)."""
    lines = tuple(
        MarkerComment(text, first_line + index, 1, (first_line + index) * 100)
        for index, text in enumerate(comments)
    )
    anchor = (
        MarkerAnchor(
            first_line + len(comments) + gap,
            1,
            candidates,
            string,
            first_line + len(comments) if gap and blank else None,
        )
        if anchored
        else None
    )
    return MarkerBlock("src/test.cpp", lines, anchor)


def _function(
    semantic_id: str = "?f@@YAXXZ",
    name: str = "f",
    line: int = 2,
    end_line: int = 4,
    is_definition: bool = True,
) -> AnchorCandidate:
    return AnchorCandidate(
        "function", semantic_id, name, is_definition, line=line, end_line=end_line
    )


def _variable(
    name: str = "g_x",
    qualified: str | None = None,
    *,
    local_static: bool = False,
    enclosing: str | None = None,
) -> AnchorCandidate:
    return AnchorCandidate(
        "variable",
        f"?{name}@@3HA",
        qualified or name,
        name=name,
        local_static=local_static,
        enclosing_function=enclosing,
    )


def _read(*blocks: MarkerBlock, aliases=None, encoding: str = "latin1"):
    results = read_marker_blocks(
        blocks, {"src/test.cpp": PATH}, aliases=aliases, encoding=encoding
    )
    assert len(results) <= 1
    if not results:
        return [], []
    return list(results[0].tokens), [
        (alert.code, alert.line_number) for alert in results[0].alerts
    ]


def test_function_binds_to_the_definition_after_it():
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000",
            "// FUNCTION: OTHER 0x2000",
            candidates=(_function(line=3, end_line=5),),
        )
    )
    assert not alerts
    assert [(s.type, s.module, s.offset) for s in symbols] == [
        (MarkerType.FUNCTION, "TEST", 0x1000),
        (MarkerType.FUNCTION, "OTHER", 0x2000),
    ]
    function = symbols[0]
    assert isinstance(function, ParserFunction)
    assert (function.line_number, function.end_line) == (3, 5)
    assert function.definitions == ("?f@@YAXXZ",)
    assert not function.is_nameref()
    assert function.filename == PATH


def test_function_needs_a_definition():
    _, alerts = _read(
        _block("// FUNCTION: TEST 0x1000", candidates=(_function(is_definition=False),))
    )
    assert alerts == [(AlertCode.NO_IMPLEMENTATION, 1)]
    _, alerts = _read(_block("// FUNCTION: TEST 0x1000", candidates=(_variable(),)))
    assert alerts == [(AlertCode.NO_DECLARATION, 1)]


def test_every_template_instantiation_is_a_candidate_definition():
    symbols, _ = _read(
        _block(
            "// FUNCTION: TEST 0x1000",
            candidates=(
                _function("a"),
                _function("b"),
                _function("c", is_definition=False),
            ),
        )
    )
    assert symbols[0].definitions == ("a", "b")


def test_clang_format_directive_between_marker_and_function():
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000", "// clang-format off", candidates=(_function(),)
        )
    )
    assert not alerts
    assert not symbols[0].is_nameref()


def test_blank_line_before_the_declaration_warns():
    symbols, alerts = _read(
        _block("// FUNCTION: TEST 0x1000", gap=1, candidates=(_function(),))
    )
    assert len(symbols) == 1
    assert alerts == [(AlertCode.UNEXPECTED_BLANK_LINE, 2)]


def test_comments_and_pragmas_before_the_declaration_are_not_blank_lines():
    # // GLOBAL: ...  then  #pragma bss_seg(".data")  then the definition
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000", gap=2, blank=False, candidates=(_function(),)
        )
    )
    assert len(symbols) == 1
    assert not alerts


def test_function_named_by_a_comment_is_looked_up_by_name():
    symbols, alerts = _read(
        _block("// FUNCTION: TEST 0x1000", "// Foo::Bar", candidates=(_function(),)),
        _block("// FUNCTION: TEST 0x2000 SYMBOL", "// ?Bar@Foo@@QAEXXZ", first_line=10),
    )
    assert not alerts
    assert [
        (s.name, s.is_nameref(), s.name_is_symbol, s.line_number) for s in symbols
    ] == [
        ("Foo::Bar", True, False, 2),
        ("?Bar@Foo@@QAEXXZ", True, True, 11),
    ]


def test_symbol_option_needs_a_name():
    _, alerts = _read(
        _block("// FUNCTION: TEST 0x1000 SYMBOL", candidates=(_function(),))
    )
    assert alerts == [(AlertCode.SYMBOL_OPTION_IGNORED, 1)]


def test_nameref_markers():
    symbols, alerts = _read(
        _block(
            "// SYNTHETIC: TEST 0x1000",
            "// Foo::`scalar deleting destructor'",
            "// SYNTHETIC: TEST 0x2000",
            "// Bar::`scalar deleting destructor'",
            "// TEMPLATE: TEST 0x3000",
            "// Vec<int>::Get",
            "// LIBRARY: TEST 0x4000",
            "// _strlen",
        )
    )
    assert not alerts
    assert [(s.type, s.name, s.is_library()) for s in symbols] == [
        (MarkerType.SYNTHETIC, "Foo::`scalar deleting destructor'", False),
        (MarkerType.SYNTHETIC, "Bar::`scalar deleting destructor'", False),
        (MarkerType.TEMPLATE, "Vec<int>::Get", False),
        (MarkerType.LIBRARY, "_strlen", True),
    ]
    _, alerts = _read(_block("// SYNTHETIC: TEST 0x1000", candidates=(_function(),)))
    assert alerts == [(AlertCode.BAD_NAMEREF, 1)]


def test_incompatible_markers_drop_the_run():
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000",
            "// GLOBAL: TEST 0x2000",
            "// FUNCTION: TEST 0x3000",
            candidates=(_function(),),
        )
    )
    assert alerts == [(AlertCode.INCOMPATIBLE_MARKER, 2)]
    assert [s.offset for s in symbols] == [0x3000]


def test_duplicate_module_keeps_the_first_marker():
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000",
            "// FUNCTION: TEST 0x2000",
            candidates=(_function(),),
        )
    )
    assert alerts == [(AlertCode.DUPLICATE_MODULE, 2)]
    assert [s.offset for s in symbols] == [0x1000]


def test_folded_function():
    symbols, _ = _read(
        _block("// FUNCTION: TEST 0x1000 FOLDED", candidates=(_function(),))
    )
    assert symbols[0].is_folded


def test_global_variable_uses_the_qualified_name():
    symbols, alerts = _read(
        _block(
            "// GLOBAL: TEST 0x1000",
            candidates=(_variable("g_x", "N::g_x"), _variable("g_y")),
        )
    )
    assert not alerts
    variable = symbols[0]
    assert isinstance(variable, ParserVariable)
    assert (variable.name, variable.is_static, variable.line_number) == (
        "N::g_x",
        False,
        2,
    )


def test_static_local_belongs_to_the_marked_function():
    function = _block("// FUNCTION: TEST 0x1000", candidates=(_function("?f@@YAXXZ"),))
    static = _block(
        "// GLOBAL: TEST 0x2000",
        first_line=3,
        candidates=(
            _variable("s_x", "Foo::s_x", local_static=True, enclosing="?f@@YAXXZ"),
        ),
    )
    symbols, alerts = _read(function, static)
    assert not alerts
    variable = symbols[1]
    assert isinstance(variable, ParserVariable)
    assert (variable.name, variable.is_static, variable.parent_function) == (
        "s_x",
        True,
        0x1000,
    )


def test_static_local_without_a_marked_function():
    _, alerts = _read(
        _block(
            "// GLOBAL: TEST 0x2000",
            candidates=(_variable("s_x", local_static=True, enclosing="?g@@YAXXZ"),),
        )
    )
    assert alerts == [(AlertCode.ORPHANED_STATIC_VARIABLE, 2)]


def test_global_must_annotate_a_variable():
    _, alerts = _read(_block("// GLOBAL: TEST 0x1000", candidates=(_function(),)))
    assert alerts == [(AlertCode.GLOBAL_NOT_VARIABLE, 1)]
    _, alerts = _read(
        _block("// GLOBAL: TEST 0x1000", candidates=(_variable(enclosing="?f@@YAXXZ"),))
    )
    assert alerts == [(AlertCode.GLOBAL_NOT_VARIABLE, 1)]


def test_global_named_by_a_comment():
    symbols, _ = _read(_block("// GLOBAL: TEST 0x1000", "// g_variable"))
    assert symbols[0].name == "g_variable"


def test_strings_are_the_bytes_the_compiler_emits():
    symbols, alerts = _read(
        _block("// STRING: TEST 0x1000", string=MarkerString(b"r\xe9sum\xe9")),
        _block(
            "// STRING: TEST 0x2000",
            first_line=5,
            string=MarkerString("wide".encode("utf-16-le"), 2),
        ),
        _block("// STRING: TEST 0x3000", '// "hello \\"world\\""', first_line=9),
        _block("// STRING: TEST 0x4000", first_line=13),
        encoding="cp1252",
    )
    assert alerts == [(AlertCode.NO_SUITABLE_NAME, 14)]
    assert [
        (s.name, s.is_widechar) for s in symbols if isinstance(s, ParserString)
    ] == [
        ("résumé", False),
        ("wide", True),
        ('hello "world"', False),
    ]


def test_global_and_string_share_a_declaration():
    symbols, alerts = _read(
        _block(
            "// GLOBAL: TEST 0x1000",
            "// STRING: TEST 0x2000",
            candidates=(_variable("g_str"),),
            string=MarkerString(b"text"),
        )
    )
    assert not alerts
    assert {type(s) for s in symbols} == {ParserVariable, ParserString}


def test_vtables():
    record = AnchorCandidate("class", "record:N::Widget", "N::Widget")
    symbols, alerts = _read(
        _block(
            "// VTABLE: TEST 0x1000",
            "// VTABLE: TEST 0x1100 Base",
            candidates=(record,),
        ),
        _block("// VTABLE: TEST 0x2000", "// class Vec<int*>", first_line=5),
        _block("// VTABLE: TEST 0x3000 FOLDED", first_line=8, candidates=(record,)),
        _block("// VTABLE: TEST 0x4000", first_line=11, candidates=(_function(),)),
    )
    assert alerts == [(AlertCode.NO_DECLARATION, 11)]
    assert [
        (s.name, s.base_class, s.is_folded)
        for s in symbols
        if isinstance(s, ParserVtable)
    ] == [
        ("N::Widget", None, False),
        ("N::Widget", "Base", False),
        ("Vec<int *>", None, False),
        ("N::Widget", None, True),
    ]


def test_line_markers_stand_alone():
    symbols, alerts = _read(
        _block(
            "// FUNCTION: TEST 0x1000",
            "// LINE: TEST 0x1010",
            candidates=(_function(),),
        )
    )
    assert not alerts
    lines = [s for s in symbols if isinstance(s, ParserLineSymbol)]
    assert [(s.offset, s.line_number, s.name) for s in lines] == [
        (0x1010, 2, "test.cpp:2")
    ]


def test_unknown_markers_and_aliases():
    block = _block("// FUNC: TEST 0x1000", "// Foo::Bar")
    symbols, alerts = _read(block)
    assert not symbols
    assert alerts == [(AlertCode.UNKNOWN_ANNOTATION, 1)]
    symbols, alerts = _read(block, aliases={"TEST": {"FUNC": "FUNCTION"}})
    assert not alerts
    assert symbols[0].name == "Foo::Bar"


def test_loose_marker_format_warns():
    symbols, alerts = _read(
        _block("//FUNCTION: TEST 0x1000", candidates=(_function(),))
    )
    assert len(symbols) == 1
    assert alerts == [(AlertCode.NOT_STRICT_FORMAT, 1)]


def test_blocks_outside_the_local_files_are_ignored():
    assert not read_marker_blocks([_block("// FUNCTION: TEST 0x1000")], {})


def test_local_paths_match_by_trailing_components():
    files = [
        PurePath("/repo/src/a/util.cpp"),
        PurePath("/repo/src/b/util.cpp"),
        PurePath("/repo/src/main.cpp"),
    ]
    assert local_paths(["src/a/util.cpp", "src/main.cpp", "vendor/x.h"], files) == {
        "src/a/util.cpp": files[0],
        "src/main.cpp": files[2],
    }


def test_markers_the_compiler_never_saw():
    block = _block("// FUNCTION: TEST 0x1000", first_line=2, anchored=False)
    text = (
        "#if 0\n"
        "// FUNCTION: TEST 0x1000\n"
        "#endif\n"
        "// FUNCTION: TEST 0x2000\n"
        "int x; // FUNCTION: TEST 0x3000\n"
    )
    alerts = list(uncompiled_markers([(PATH, text)], [block], {"src/test.cpp": PATH}))
    assert [(a.code, a.line_number, a.target) for a in alerts] == [
        (AlertCode.MARKER_NOT_COMPILED, 4, "TEST")
    ]


def test_block_round_trip():
    block = _block(
        "// STRING: TEST 0x1000",
        candidates=(_variable("g"),),
        string=MarkerString(b"\x00\xff", 1),
    )
    assert MarkerBlock.from_dict(block.to_dict()) == block


def test_blocks_seen_by_several_units_merge_their_candidates():
    first = _block("// FUNCTION: TEST 0x1000", candidates=(_function("a"),))
    second = _block("// FUNCTION: TEST 0x1000", candidates=(_function("b"),))
    merged = first.merged(second)
    assert merged.anchor is not None
    assert [c.semantic_id for c in merged.anchor.candidates] == ["a", "b"]
