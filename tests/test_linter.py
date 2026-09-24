from pathlib import PurePath
from reccmp.parser import ReccmpParserResult
from reccmp.parser.error import AlertCode
from reccmp.parser.linter import (
    check_byname_allowed,
    check_function_order,
    check_offset_uniqueness,
    check_string_text,
)
from reccmp.parser.marker import MarkerType
from reccmp.parser.node import ParserFunction, ParserString, ParserSymbol, ParserVtable


def create_parser_result(path: PurePath, *symbols: ParserSymbol) -> ReccmpParserResult:
    """The linter interprets marker results; build them directly."""
    return ReccmpParserResult(tokens=tuple(symbols), alerts=(), path=path)


def function(
    offset: int,
    line: int,
    module: str = "TEST",
    *,
    marker: MarkerType = MarkerType.FUNCTION,
    by_name: bool = False,
    folded: bool = False,
) -> ParserFunction:
    return ParserFunction(
        type=marker,
        line_number=line,
        module=module,
        offset=offset,
        name="f",
        filename=PurePath("test.cpp"),
        lookup_by_name=by_name,
        is_folded=folded,
    )


def string(offset: int, line: int, text: str) -> ParserString:
    return ParserString(
        type=MarkerType.STRING,
        line_number=line,
        module="TEST",
        offset=offset,
        name=text,
        filename=PurePath("test.h"),
    )


def vtable(offset: int, line: int, *, folded: bool = False) -> ParserVtable:
    return ParserVtable(
        type=MarkerType.VTABLE,
        line_number=line,
        module="TEST",
        offset=offset,
        name="Class",
        filename=PurePath("test.cpp"),
        is_folded=folded,
    )


def test_order_in_order():
    """Functions from the same module are in order. No problems here."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x1000, 2),
        function(0x2000, 4),
        function(0x3000, 6),
    )
    assert not check_function_order(result)


def test_order_out_of_order():
    """Detect functions that are out of order."""
    path = PurePath("test.cpp")
    result = create_parser_result(
        path,
        function(0x1000, 2),
        function(0x3000, 4),
        function(0x2000, 6),
    )
    alerts = check_function_order(result)

    assert len(alerts) == 1
    assert alerts[0].code == AlertCode.FUNCTION_OUT_OF_ORDER
    # N.B. Line number given is the start of the function, not the marker
    assert alerts[0].line_number == 6
    # Identifying details of the alert's origin are now embedded.
    assert alerts[0].target == "TEST"
    assert alerts[0].path == PurePath("test.cpp")


def test_order_ignore_lookup_by_name():
    """Should ignore lookup-by-name markers when checking order."""
    result = create_parser_result(
        PurePath("test.h"),
        function(0x1000, 2),
        function(0x3000, 4, by_name=True),
        function(0x2000, 6),
    )
    assert not check_function_order(result)


def test_order_reports_all_modules():
    """Any ordering problems from any module are reported. The caller should filter for display."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x0003, 3, "ALPHA"),
        function(0x1000, 3),
        function(0x0002, 6, "ALPHA"),
        function(0x2000, 6),
        function(0x0001, 9, "ALPHA"),
        function(0x3000, 9),
    )
    alerts = check_function_order(result)

    # ALPHA markers are out of order.
    assert {alert.target for alert in alerts} == {"ALPHA"}


def test_implicit_byname_headers_only():
    """Implementation markers that fall back to name lookup belong in headers."""
    symbol = function(0x1000, 2, by_name=True)
    result_cpp = create_parser_result(PurePath("test.cpp"), symbol)
    result_h = create_parser_result(PurePath("test.h"), symbol)

    assert not check_byname_allowed(result_h)

    alerts = check_byname_allowed(result_cpp)
    assert alerts[0].code == AlertCode.BYNAME_FUNCTION_IN_CPP


def test_explicit_byname_markers_allowed_in_cpp():
    """Explicit name-reference marker kinds are valid without an adjacent body."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x1000, 2, marker=MarkerType.TEMPLATE, by_name=True),
        function(0x2000, 4, marker=MarkerType.SYNTHETIC, by_name=True),
        function(0x3000, 6, marker=MarkerType.LIBRARY, by_name=True),
    )

    assert not check_byname_allowed(result)


def test_duplicate_offsets_module_scope():
    """All duplicate offsets from all modules are reported. The caller should filter for display."""
    result = create_parser_result(
        PurePath("test.h"),
        function(0x1000, 3, by_name=True),
        function(0x1000, 3, "HELLO", by_name=True),
    )

    # Should not fail for duplicate offset 0x1000 because the modules are unique.
    assert not check_offset_uniqueness([result])

    # Simulate a failure by reading the same file twice.
    alerts = check_offset_uniqueness([result, result])

    # Duplicate addresses from both modules are reported.
    assert len(alerts) == 2
    assert {alert.target for alert in alerts} == {"HELLO", "TEST"}


def test_duplicate_strings():
    """Duplicate string markers are okay if the string value is the same."""
    string_hello = create_parser_result(
        PurePath("test.h"), string(0x1000, 2, "hello world")
    )

    assert not check_string_text([string_hello])
    assert not check_string_text([string_hello, string_hello])
    assert not check_offset_uniqueness([string_hello])
    assert not check_offset_uniqueness([string_hello, string_hello])

    # Same address but the string is different
    string_hi = create_parser_result(
        PurePath("greeting.h"), string(0x1000, 2, "hi there")
    )
    alerts = check_string_text([string_hello, string_hi])
    assert len(alerts) == 1
    assert alerts[0].code == AlertCode.WRONG_STRING

    # Strings are skipped by the uniqueness check.
    assert not check_offset_uniqueness([string_hello, string_hi])


def test_ignore_folded_duplicate():
    """Do not alert to folded functions that reuse an address."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x1000, 2, folded=True),
        function(0x1000, 5, folded=True),
    )
    assert not check_offset_uniqueness([result, result])


def test_ignore_folded_and_regular_duplicate():
    """Should alert when folded and non-folded functions reuse an address."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x1000, 2, folded=True),
        function(0x1000, 5),
    )
    alerts = check_offset_uniqueness([result, result])
    assert alerts[0].code == AlertCode.DUPLICATE_OFFSET


def test_ignore_folded_duplicate_vtable():
    """Do not alert to folded vtables that reuse an address."""
    result = create_parser_result(
        PurePath("test.cpp"),
        vtable(0x1000, 2, folded=True),
        vtable(0x1000, 6, folded=True),
    )
    assert not check_offset_uniqueness([result, result])


def test_ignore_folded_and_regular_duplicate_vtable():
    """Should alert when folded and non-folded vtables reuse an address."""
    result = create_parser_result(
        PurePath("test.cpp"),
        vtable(0x1000, 2, folded=True),
        vtable(0x1000, 6),
    )
    alerts = check_offset_uniqueness([result, result])
    assert alerts[0].code == AlertCode.DUPLICATE_OFFSET


def test_ignore_folded_order():
    """Skip folded functions and do not check their order."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x2000, 2, folded=True),
        function(0x1000, 5),
    )
    assert not check_function_order(result)


def test_folded_with_real_order_error():
    """Folded functions should not prevent us from reporting
    that regular functions are out of order."""
    result = create_parser_result(
        PurePath("test.cpp"),
        function(0x3000, 2),
        function(0x2000, 5, folded=True),
        function(0x1000, 8),
    )
    alerts = check_function_order(result)
    assert alerts
