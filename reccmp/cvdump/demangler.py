"""For demangling a subset of MSVC mangled symbols.
Some unofficial information about the mangling scheme is here:
https://en.wikiversity.org/wiki/Visual_C%2B%2B_name_mangling
"""

import re
from typing import NamedTuple
from pydemumble import demangle as _demangle  # type: ignore


def msvc_demangle(symbol: str) -> str:
    """Wrapper for demumbler. Converts MSVC C++ symbol to a name
    more similar to what appears in the code.
    If no conversion is possible, return empty string."""
    return _demangle(symbol) or ""


class InvalidEncodedNumberError(Exception):
    pass


_encoded_number_translate = str.maketrans("ABCDEFGHIJKLMNOP", "0123456789ABCDEF")


def parse_encoded_number(string: str) -> int:
    # TODO: assert string ends in "@"?
    if string.endswith("@"):
        string = string[:-1]

    try:
        return int(string.translate(_encoded_number_translate), 16)
    except ValueError as e:
        raise InvalidEncodedNumberError(string) from e


string_const_regex = re.compile(
    r"\?\?_C@\_(?P<is_utf16>[0-1])(?P<len>\d|[A-P]+@)(?P<hash>\w+)@(?P<value>.+)@"
)


class StringConstInfo(NamedTuple):
    len: int
    is_utf16: bool


def demangle_string_const(symbol: str) -> StringConstInfo | None:
    """Don't bother to decode the string text from the symbol.
    We can just read it from the binary once we have the length."""
    match = string_const_regex.match(symbol)
    if match is None:
        return None

    try:
        strlen = (
            parse_encoded_number(match.group("len"))
            if "@" in match.group("len")
            else int(match.group("len"))
        )
    except (ValueError, InvalidEncodedNumberError):
        return None

    is_utf16 = match.group("is_utf16") == "1"
    return StringConstInfo(len=strlen, is_utf16=is_utf16)


def get_vtordisp_name(symbol: str) -> str | None:
    # pylint: disable=c-extension-no-member
    """For adjuster thunk functions, the PDB will sometimes use a name
    that contains "vtordisp" but often will just reuse the name of the
    function being thunked. We want to use the vtordisp name if possible."""
    name = msvc_demangle(symbol)
    if not name:
        return None

    if "`vtordisp" not in name:
        return None

    # Now we remove the parts of the friendly name that we don't need
    try:
        # Assuming this is the last of the function prefixes
        thiscall_idx = name.index("__thiscall")
        # To match the end of the `vtordisp{x,y}' string
        end_idx = name.index("}'")
        return name[thiscall_idx + 11 : end_idx + 2]
    except ValueError:
        return name


def get_function_arg_string(symbol: str) -> str | None:
    # pylint: disable=c-extension-no-member
    """Demangle the given symbol and return its parameters.
    We can use this to distinguish functions with the same name."""
    raw = msvc_demangle(symbol)
    if not raw:
        return None

    try:
        # Just get what's in the parens
        return raw[raw.index("(") : raw.rindex(")") + 1]
    except ValueError:
        return None


def demangle_vtable(symbol: str) -> str:
    # pylint: disable=c-extension-no-member
    """Get the class name referenced in the vtable symbol."""
    raw = msvc_demangle(symbol)

    if not raw:
        pass  # TODO: This shouldn't happen if MSVC behaves

    # Remove storage class and other stuff we don't care about
    return (
        raw.replace("<class ", "<")
        .replace("<struct ", "<")
        .replace("const ", "")
        .replace("volatile ", "")
    )


def demangle_vtable_ourselves(symbol: str) -> str:
    """Parked implementation of MSVC symbol demangling.
    We only use this for vtables and it works okay with the simple cases or
    templates that refer to other classes/structs. Some namespace support.
    Does not support backrefs, primitive types, or vtables with
    virtual inheritance."""

    # Seek ahead 4 chars to strip off "??_7" prefix
    t = symbol[4:].split("@")
    # "?$" indicates a template class
    if t[0].startswith("?$"):
        class_name = t[0][2:]
        # PA = Pointer/reference
        # V or U = class or struct
        if t[1].startswith("PA"):
            generic = f"{t[1][3:]} *"
        else:
            generic = t[1][1:]

        return f"{class_name}<{generic}>::`vftable'"

    # If we have two classes listed, it is a namespace hierarchy.
    # @@6B@ is a common generic suffix for these vtable symbols.
    if t[1] != "" and t[1] != "6B":
        return t[1] + "::" + t[0] + "::`vftable'"

    return t[0] + "::`vftable'"


class DemangledFunction(NamedTuple):
    """The parts of a demangled MSVC function name that matter for calls."""

    convention: str  # cdecl / stdcall / thiscall / fastcall
    return_type: str | None  # None for constructors and destructors
    parameters: tuple[str, ...]  # empty for (void)


_CONVENTION_RE = re.compile(r"\b__(cdecl|stdcall|thiscall|fastcall)\b")
_SPECIFIERS = frozenset(
    {"public:", "protected:", "private:", "static", "virtual", "[thunk]:"}
)


def _split_top_level(text: str) -> list[str]:
    """Split on commas outside <>, () and []."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def demangle_function(symbol: str) -> DemangledFunction | None:
    """Convention, return type and parameter types of a mangled function,
    from demumble's output. None if it is not a demangleable function."""
    text = msvc_demangle(symbol)
    match = _CONVENTION_RE.search(text) if text else None
    if match is None or not text.endswith((")", ") const", ") volatile")):
        return None
    prefix = " ".join(
        word for word in text[: match.start()].split() if word not in _SPECIFIERS
    )
    # The parameter list is the last top-level parenthesised group.
    end = text.rindex(")")
    depth = 0
    for start in range(end, -1, -1):
        if text[start] == ")":
            depth += 1
        elif text[start] == "(":
            depth -= 1
            if depth == 0:
                break
    params = _split_top_level(text[start + 1 : end])
    if params == ["void"]:
        params = []
    return DemangledFunction(match.group(1), prefix or None, tuple(params))


def _strip_cv(type_name: str) -> str:
    words = [word for word in type_name.split() if word not in ("const", "volatile")]
    return " ".join(words)


_SCALAR_KINDS = {
    "void": "void",
    "bool": "i8",
    "char": "i8",
    "signed char": "i8",
    "unsigned char": "i8",
    "short": "i16",
    "unsigned short": "i16",
    "wchar_t": "i16",
    "int": "i32",
    "unsigned int": "i32",
    "long": "i32",
    "unsigned long": "i32",
    "__int64": "i64",
    "unsigned __int64": "i64",
    "float": "float",
    "double": "float",
    "long double": "float",
}


def type_return_kind(type_name: str) -> str:
    """Register footprint of a value of this (demangled) type."""
    name = _strip_cv(type_name)
    if name.endswith(("*", "&")) or name.startswith("enum "):
        return "i32"
    return _SCALAR_KINDS.get(name, "unknown")
