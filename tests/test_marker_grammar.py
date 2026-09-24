"""The marker grammar: marker lines, aliases, and name comments."""

import pytest
from reccmp.parser.marker import (
    MarkerType,
    match_marker,
    is_marker_exact,
    normalize_project_aliases,
)
from reccmp.parser.reader import get_class_name, get_string_contents

marker_samples = [
    # (can_parse: bool, exact_match: bool, line: str)
    (True, True, "// FUNCTION: LEGO1 0xdeadbeef"),
    (True, True, "// FUNCTION: ISLE 0x12345678"),
    # No trailing spaces allowed
    (True, False, "// FUNCTION: LEGO1 0xdeadbeef  "),
    # Must have exactly one space between elements
    (True, False, "//FUNCTION: ISLE 0xdeadbeef"),
    (True, False, "// FUNCTION:ISLE 0xdeadbeef"),
    (True, False, "//  FUNCTION: ISLE 0xdeadbeef"),
    (True, False, "// FUNCTION:  ISLE 0xdeadbeef"),
    (True, False, "// FUNCTION: ISLE  0xdeadbeef"),
    # Must have 0x prefix for hex number to match at all
    (False, False, "// FUNCTION: ISLE deadbeef"),
    # Offset, module name, and STUB must be uppercase
    (True, False, "// function: ISLE 0xdeadbeef"),
    (True, False, "// function: isle 0xdeadbeef"),
    # Hex string must be lowercase
    (True, False, "// FUNCTION: ISLE 0xDEADBEEF"),
    # TODO: How flexible should we be with matching the module name?
    (True, True, "// FUNCTION: OMNI 0x12345678"),
    (True, True, "// FUNCTION: LEG01 0x12345678"),
    (True, False, "// FUNCTION: hello 0x12345678"),
    # Not close enough to match
    (False, False, "// FUNCTION: ISLE0x12345678"),
    (False, False, "// FUNCTION: 0x12345678"),
    (False, False, "// LEGO1: 0x12345678"),
    # Hex string shorter than 8 characters
    (True, True, "// FUNCTION: LEGO1 0x1234"),
    # TODO: These match but shouldn't.
    # (False, False, '// FUNCTION: LEGO1 0'),
    # (False, False, '// FUNCTION: LEGO1 0x'),
    # Extra field
    (True, True, "// VTABLE: HELLO 0x1234 Extra"),
    # Extra with spaces
    (True, True, "// VTABLE: HELLO 0x1234 Whatever<SubClass *>"),
    # Extra, no space (if the first non-hex character is not in [a-f])
    (True, False, "// VTABLE: HELLO 0x1234Hello"),
    # Extra, many spaces
    (True, False, "// VTABLE: HELLO 0x1234    Hello"),
]


@pytest.mark.parametrize("match, _, line", marker_samples)
def test_marker_match(line: str, match: bool, _):
    did_match = match_marker(line) is not None
    assert did_match is match


@pytest.mark.parametrize("_, exact, line", marker_samples)
def test_marker_exact(line: str, exact: bool, _):
    assert is_marker_exact(line) is exact


class_name_match_cases = [
    ("struct MxString {", "MxString"),
    ("class MxString {", "MxString"),
    ("// class MxString", "MxString"),
    ("class MxString : public MxCore {", "MxString"),
    ("class MxPtrList<MxPresenter>", "MxPtrList<MxPresenter>"),
    # If it is possible to match the symbol MxList<LegoPathController *>::`vftable'
    # we should get the correct class name if possible. If the template type is a pointer,
    # the asterisk and class name are separated by one space.
    ("// class MxList<LegoPathController *>", "MxList<LegoPathController *>"),
    ("// class MxList<LegoPathController*>", "MxList<LegoPathController *>"),
    ("// class MxList<LegoPathController* >", "MxList<LegoPathController *>"),
    # I don't know if this would ever come up, but sure, why not?
    ("// class MxList<LegoPathController**>", "MxList<LegoPathController **>"),
    ("// class Many::Name::Spaces", "Many::Name::Spaces"),
]


@pytest.mark.parametrize("line, class_name", class_name_match_cases)
def test_get_class_name(line: str, class_name: str):
    assert get_class_name(line) == class_name


class_name_no_match_cases = [
    "MxString { ",
    "clas MxString",
    "// MxPtrList<MxPresenter>::`scalar deleting destructor'",
]


@pytest.mark.parametrize("line", class_name_no_match_cases)
def test_get_class_name_none(line: str):
    assert get_class_name(line) is None


string_match_cases = [
    ('return "hello world";', "hello world"),
    ('"hello\\\\"', "hello\\"),
    ('"hello \\"world\\""', 'hello "world"'),
    ('"hello\\nworld"', "hello\nworld"),
    # Only match first string if there are multiple options
    ('Method("hello", "world");', "hello"),
]


@pytest.mark.parametrize("line, expected", string_match_cases)
def test_get_string_contents(line: str, expected: str):
    string = get_string_contents(line)
    assert string is not None
    assert string.text == expected
    assert string.is_widechar is False


def test_marker_extra_spaces():
    """The extra field can contain spaces"""
    marker = match_marker("// VTABLE: TEST 0x1234 S p a c e s")
    assert marker is not None
    assert marker.extra == "S p a c e s"

    # Trailing spaces removed
    marker = match_marker("// VTABLE: TEST 0x8888 spaces    ")
    assert marker is not None
    assert marker.extra == "spaces"

    # Trailing newline removed if present
    marker = match_marker("// VTABLE: TEST 0x5555 newline\n")
    assert marker is not None
    assert marker.extra == "newline"


def test_marker_trailing_spaces():
    """Should ignore trailing spaces. (Invalid extra field)
    Offset field not truncated, extra field set to None."""

    marker = match_marker("// VTABLE: TEST 0x1234     ")
    assert marker is not None
    assert marker.offset == 0x1234
    assert marker.extra is None


def test_marker_aliases():
    """Testing with aliases that have been normalized by normalize_project_aliases()"""
    # No alias
    marker = match_marker("// FUNC: TEST 0x1234")
    assert marker and marker.type == MarkerType.UNKNOWN

    # Alias demo
    marker = match_marker("// FUNC: TEST 0x1234", {"TEST": {"FUNC": "FUNCTION"}})
    assert marker and marker.type == MarkerType.FUNCTION

    # Aliases are scoped by target
    marker = match_marker("// FUNC: TEST 0x1234", {"HELLO": {"FUNC": "FUNCTION"}})
    assert marker and marker.type == MarkerType.UNKNOWN

    # Alias goes nowhere
    marker = match_marker("// FUNC: TEST 0x1234", {"TEST": {"FUNC": "HELLO"}})
    assert marker and marker.type == MarkerType.UNKNOWN

    # Double alias not followed
    marker = match_marker(
        "// FUNC: TEST 0x1234", {"TEST": {"FUNC": "HELLO", "HELLO": "FUNCTION"}}
    )
    assert marker and marker.type == MarkerType.UNKNOWN


def test_normalize_project_aliases():
    assert not normalize_project_aliases({})

    # Normalize target and alias names for lookup
    assert normalize_project_aliases({"test": {"func": "FUNCTION"}}) == {
        "TEST": {"FUNC": "FUNCTION"}
    }

    # Drop empty alias lists
    assert not normalize_project_aliases({"TEST": {}})

    # Drop aliases that point to nothing
    assert not normalize_project_aliases({"TEST": {"func": "FUNC"}})

    # Drop aliases that try to reassign a built-in type
    assert not normalize_project_aliases({"TEST": {"FUNCTION": "GLOBAL"}})

    # Drop duplicated alias
    assert normalize_project_aliases(
        {"TEST": {"func": "FUNCTION", "FUNC": "TEMPLATE"}}
    ) == {"TEST": {"FUNC": "FUNCTION"}}
