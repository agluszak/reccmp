"""Read reccmp markers from the comment blocks the Clang source indexer reports.

The indexer owns C++: it finds every run of `//` comments containing a marker
in active code and lists the declarations that begin at the first code token
after it (functions with their extent, variables, classes) plus the first
string literal on that line. This module owns only the marker grammar: which
marker types may share a block, what a name comment completes, and which kind
of declaration each marker type annotates.
"""

from __future__ import annotations

import re
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Sequence

from .error import AlertCode, ParserAlert
from .marker import (
    DecompMarker,
    MarkerCategory,
    MarkerType,
    ProjectAliases,
    is_marker_exact,
    match_marker,
)
from .node import (
    ParserFunction,
    ParserLineSymbol,
    ParserString,
    ParserSymbol,
    ParserVariable,
    ParserVtable,
)

# -- records from the indexer ----------------------------------------------


@dataclass(frozen=True)
class MarkerComment:
    text: str
    line: int
    column: int
    offset: int


@dataclass(frozen=True)
class AnchorCandidate:
    """One declaration that begins where a marker block's code starts."""

    # pylint: disable=too-many-instance-attributes
    kind: str  # function, variable or class
    semantic_id: str
    qualified_name: str
    is_definition: bool = False
    line: int = 0
    end_line: int = 0
    name: str = ""
    local_static: bool = False
    enclosing_function: str | None = None


@dataclass(frozen=True)
class MarkerString:
    """The bytes the compiler emits for a string literal."""

    data: bytes
    char_width: int = 1

    def text(self, encoding: str) -> str:
        if self.char_width == 2:
            return self.data.decode("utf-16-le", errors="replace")
        if self.char_width == 4:
            return self.data.decode("utf-32-le", errors="replace")
        return self.data.decode(encoding, errors="replace")


@dataclass(frozen=True)
class MarkerAnchor:
    line: int
    column: int
    candidates: tuple[AnchorCandidate, ...] = ()
    string: MarkerString | None = None

    def of_kind(self, kind: str) -> list[AnchorCandidate]:
        return [item for item in self.candidates if item.kind == kind]


@dataclass(frozen=True)
class MarkerBlock:
    """Consecutive `//` comment lines, at least one shaped like a marker."""

    source_file: str  # relative to the indexed repository
    comments: tuple[MarkerComment, ...]
    anchor: MarkerAnchor | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_file, self.comments[0].offset)

    def merged(self, other: "MarkerBlock") -> "MarkerBlock":
        """Combine the same block seen by several translation units: each
        may instantiate different templates at the anchor."""
        if self.anchor is None or other.anchor is None:
            return self if self.anchor is not None else other
        candidates = dict.fromkeys(self.anchor.candidates)
        candidates.update(dict.fromkeys(other.anchor.candidates))
        return MarkerBlock(
            self.source_file,
            self.comments,
            MarkerAnchor(
                self.anchor.line,
                self.anchor.column,
                tuple(candidates),
                self.anchor.string or other.anchor.string,
            ),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MarkerBlock":
        anchor = values.get("anchor")
        return cls(
            source_file=str(values["source_file"]),
            comments=tuple(MarkerComment(**item) for item in values["comments"]),
            anchor=(
                None
                if anchor is None
                else MarkerAnchor(
                    line=int(anchor["line"]),
                    column=int(anchor["column"]),
                    candidates=tuple(
                        AnchorCandidate(**item) for item in anchor.get("candidates", ())
                    ),
                    string=(
                        None
                        if anchor.get("string") is None
                        else MarkerString(
                            bytes.fromhex(anchor["string"]["hex"]),
                            int(anchor["string"]["char_width"]),
                        )
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        anchor: dict[str, Any] | None = None
        if self.anchor is not None:
            anchor = {
                "line": self.anchor.line,
                "column": self.anchor.column,
                "candidates": [vars(item) for item in self.anchor.candidates],
                "string": (
                    None
                    if self.anchor.string is None
                    else {
                        "hex": self.anchor.string.data.hex(),
                        "char_width": self.anchor.string.char_width,
                    }
                ),
            }
        return {
            "source_file": self.source_file,
            "comments": [vars(item) for item in self.comments],
            "anchor": anchor,
        }


# -- name comments ----------------------------------------------------------

_NAME_COMMENT = re.compile(r"\s*//\s*(.*)")
_CLANG_FORMAT_DIRECTIVE = re.compile(r"^\s*//\s*clang-format\s+(?:on|off)\s*$", re.I)
_DOUBLE_QUOTED = re.compile(r'(L)?("(?:[^"\\]|\\.)*")')
_CLASS_NAME = re.compile(r"\s*(?:\/\/)?\s*(?:class|struct) ((?:\w+(?:<.+>)?(?:::)?)+)")
_TEMPLATE_ARGUMENT = re.compile(r"<(?P<type>[\w]+)\s*(?P<asterisks>\*+)?\s*>")


def get_synthetic_name(line: str) -> str | None:
    """The name a comment line gives to the markers above it."""
    match = _NAME_COMMENT.match(line)
    return match.group(1).strip() if match is not None else None


def is_ignorable_marker_adjacent_comment(line: str) -> bool:
    return _CLANG_FORMAT_DIRECTIVE.match(line) is not None


def get_class_name(line: str) -> str | None:
    """A VTABLE name comment such as ``// class Vec<int *>``, spelled the way
    cvdump spells template arguments."""
    match = _CLASS_NAME.match(line)
    if match is None:
        return None

    def spaced(argument: re.Match) -> str:
        name, asterisks = argument.groups()
        return f"<{name}>" if asterisks is None else f"<{name} {asterisks}>"

    return _TEMPLATE_ARGUMENT.sub(spaced, match.group(1))


class ParserCodeString(NamedTuple):
    text: str
    is_widechar: bool


def get_string_contents(line: str) -> ParserCodeString | None:
    """The first C string on a STRING name comment."""
    match = _DOUBLE_QUOTED.search(line)
    if match is None:
        return None
    try:
        text = literal_eval(match.group(2))
    except (SyntaxError, ValueError):
        return None
    return ParserCodeString(text=text, is_widechar=match.group(1) is not None)


# -- the grammar ----------------------------------------------------------


@dataclass(frozen=True)
class ReccmpParserResult:
    """Markers and alerts for one source file."""

    tokens: tuple[ParserSymbol, ...]
    alerts: tuple[ParserAlert, ...]
    path: PurePath


_NAMEREF_TYPES = (MarkerType.SYNTHETIC, MarkerType.TEMPLATE, MarkerType.LIBRARY)


def _compatible(first: MarkerType, second: MarkerType) -> bool:
    """Marker types that may annotate the same thing in one block."""
    groups = (
        {MarkerType.FUNCTION, MarkerType.STUB},
        {MarkerType.GLOBAL, MarkerType.STRING},
        {MarkerType.VTABLE},
        {MarkerType.SYNTHETIC},
        {MarkerType.TEMPLATE},
        {MarkerType.LIBRARY},
    )
    return any(first in group and second in group for group in groups)


def _extra_is(marker: DecompMarker, word: str) -> bool:
    return marker.extra is not None and marker.extra.lower() == word


@dataclass
class _Pending:
    markers: dict[tuple[MarkerCategory, str, str | None], DecompMarker] = field(
        default_factory=dict
    )
    last_line: int = 0

    def types(self) -> set[MarkerType]:
        return {marker.type for marker in self.markers.values()}


@dataclass
class _StaticVariable:
    symbol: ParserVariable
    enclosing_function: str


class _FileReader:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, path: PurePath, aliases: ProjectAliases, encoding: str):
        self.path = path
        self.aliases = aliases
        self.encoding = encoding
        self.symbols: list[ParserSymbol] = []
        self.alerts: list[ParserAlert] = []
        self.static_locals: list[_StaticVariable] = []
        self.function_ids: dict[tuple[str, str], int] = {}

    def alert(self, code: AlertCode, line: int, detail: str | None = None) -> None:
        self.alerts.append(
            ParserAlert(code=code, path=self.path, line_number=line, detail=detail)
        )

    def read(self, block: MarkerBlock) -> None:
        pending = _Pending()
        for comment in block.comments:
            marker = match_marker(comment.text, aliases=self.aliases)
            if marker is None:
                if pending.markers and not is_ignorable_marker_adjacent_comment(
                    comment.text
                ):
                    self._complete_by_name(pending, comment)
                    pending = _Pending()
                continue
            if not is_marker_exact(comment.text):
                self.alert(AlertCode.NOT_STRICT_FORMAT, comment.line, comment.text)
            if marker.type == MarkerType.LINE:
                self.symbols.append(
                    ParserLineSymbol(
                        type=marker.type,
                        line_number=comment.line,
                        module=marker.module,
                        offset=marker.offset,
                        name=f"{self.path.name}:{comment.line}",
                        filename=self.path,
                    )
                )
                continue
            if marker.type == MarkerType.UNKNOWN:
                self.alert(AlertCode.UNKNOWN_ANNOTATION, comment.line, comment.text)
                continue
            if any(not _compatible(marker.type, known) for known in pending.types()):
                # Drop the whole run; a later marker starts afresh.
                self.alert(AlertCode.INCOMPATIBLE_MARKER, comment.line, comment.text)
                pending = _Pending()
                continue
            if marker.key in pending.markers:
                self.alert(AlertCode.DUPLICATE_MODULE, comment.line, comment.text)
                continue
            pending.markers[marker.key] = marker
            pending.last_line = comment.line
        if pending.markers:
            self._complete_by_anchor(pending, block.anchor, block.comments[-1].line)

    # -- completion by a name comment ---------------------------------------

    def _complete_by_name(self, pending: _Pending, comment: MarkerComment) -> None:
        name = get_synthetic_name(comment.text) or ""
        for marker in pending.markers.values():
            if marker.type in (MarkerType.FUNCTION, MarkerType.STUB, *_NAMEREF_TYPES):
                self._function(
                    marker,
                    line=comment.line,
                    name=name,
                    lookup_by_name=True,
                    name_is_symbol=_extra_is(marker, "symbol"),
                )
            elif marker.type == MarkerType.GLOBAL:
                self._variable(marker, comment.line, name)
            elif marker.type == MarkerType.STRING:
                string = get_string_contents(comment.text)
                if string is None:
                    self.alert(AlertCode.NO_SUITABLE_NAME, comment.line, comment.text)
                else:
                    self._string(marker, comment.line, string.text, string.is_widechar)
            elif marker.type == MarkerType.VTABLE:
                class_name = get_class_name(comment.text)
                if class_name is None:
                    self.alert(AlertCode.NO_SUITABLE_NAME, comment.line, comment.text)
                else:
                    self._vtable(marker, comment.line, class_name)

    # -- completion by the declaration that follows --------------------------

    def _complete_by_anchor(
        self, pending: _Pending, anchor: MarkerAnchor | None, block_end: int
    ) -> None:
        line = anchor.line if anchor is not None else pending.last_line
        if anchor is not None and anchor.line > block_end + 1:
            self.alert(AlertCode.UNEXPECTED_BLANK_LINE, block_end + 1)
        for marker in pending.markers.values():
            if marker.type in _NAMEREF_TYPES:
                self.alert(AlertCode.BAD_NAMEREF, pending.last_line)
            elif marker.type in (MarkerType.FUNCTION, MarkerType.STUB):
                self._anchored_function(marker, anchor, pending.last_line)
            elif marker.type == MarkerType.GLOBAL:
                self._anchored_variable(marker, anchor, pending.last_line)
            elif marker.type == MarkerType.STRING:
                if anchor is None or anchor.string is None:
                    self.alert(AlertCode.NO_SUITABLE_NAME, line)
                else:
                    self._string(
                        marker,
                        line,
                        anchor.string.text(self.encoding),
                        anchor.string.char_width > 1,
                    )
            elif marker.type == MarkerType.VTABLE:
                classes = anchor.of_kind("class") if anchor is not None else []
                if not classes:
                    self.alert(AlertCode.NO_DECLARATION, pending.last_line)
                else:
                    self._vtable(marker, line, classes[0].qualified_name)

    def _anchored_function(
        self, marker: DecompMarker, anchor: MarkerAnchor | None, marker_line: int
    ) -> None:
        functions = anchor.of_kind("function") if anchor is not None else []
        if not functions:
            self.alert(AlertCode.NO_DECLARATION, marker_line)
            return
        definitions = [item for item in functions if item.is_definition]
        if not definitions:
            self.alert(AlertCode.NO_IMPLEMENTATION, marker_line)
            return
        if _extra_is(marker, "symbol"):
            self.alert(AlertCode.SYMBOL_OPTION_IGNORED, marker_line)
        first = definitions[0]
        self._function(
            marker,
            line=first.line,
            end_line=first.end_line,
            name=first.qualified_name,
            definitions=tuple(item.semantic_id for item in definitions),
        )

    def _anchored_variable(
        self, marker: DecompMarker, anchor: MarkerAnchor | None, marker_line: int
    ) -> None:
        variables = anchor.of_kind("variable") if anchor is not None else []
        if not variables:
            self.alert(AlertCode.GLOBAL_NOT_VARIABLE, marker_line)
            return
        variable = variables[0]
        assert anchor is not None
        if variable.enclosing_function is None:
            self._variable(marker, anchor.line, variable.qualified_name)
        elif variable.local_static:
            symbol = self._variable(marker, anchor.line, variable.name, is_static=True)
            self.static_locals.append(
                _StaticVariable(symbol, variable.enclosing_function)
            )
        else:
            self.alert(AlertCode.GLOBAL_NOT_VARIABLE, marker_line)

    # -- symbols ------------------------------------------------------------

    def _function(  # pylint: disable=too-many-arguments
        self,
        marker: DecompMarker,
        *,
        line: int,
        name: str,
        end_line: int | None = None,
        lookup_by_name: bool = False,
        name_is_symbol: bool = False,
        definitions: tuple[str, ...] = (),
    ) -> None:
        self.symbols.append(
            ParserFunction(
                type=marker.type,
                line_number=line,
                module=marker.module,
                offset=marker.offset,
                name=name,
                filename=self.path,
                end_line=end_line if end_line is not None else line,
                lookup_by_name=lookup_by_name,
                name_is_symbol=name_is_symbol,
                is_folded=_extra_is(marker, "folded"),
                definitions=definitions,
            )
        )
        for semantic_id in definitions:
            self.function_ids.setdefault((marker.module, semantic_id), marker.offset)

    def _variable(
        self, marker: DecompMarker, line: int, name: str, *, is_static: bool = False
    ) -> ParserVariable:
        symbol = ParserVariable(
            type=marker.type,
            line_number=line,
            module=marker.module,
            offset=marker.offset,
            name=name,
            filename=self.path,
            is_static=is_static,
        )
        self.symbols.append(symbol)
        return symbol

    def _string(self, marker: DecompMarker, line: int, text: str, wide: bool) -> None:
        self.symbols.append(
            ParserString(
                type=marker.type,
                line_number=line,
                module=marker.module,
                offset=marker.offset,
                name=text,
                filename=self.path,
                is_widechar=wide,
            )
        )

    def _vtable(self, marker: DecompMarker, line: int, class_name: str) -> None:
        folded = _extra_is(marker, "folded")
        self.symbols.append(
            ParserVtable(
                type=marker.type,
                line_number=line,
                module=marker.module,
                offset=marker.offset,
                name=class_name,
                filename=self.path,
                base_class=None if folded else marker.extra,
                is_folded=folded,
            )
        )


def local_paths(
    relative_paths: Iterable[str], files: Iterable[PurePath]
) -> dict[str, PurePath]:
    """Map repository-relative index paths onto the local source files that
    end with them. Index paths without a local file are left out."""
    by_name: dict[str, list[PurePath]] = {}
    for path in files:
        by_name.setdefault(path.name.lower(), []).append(path)
    result: dict[str, PurePath] = {}
    for relative in relative_paths:
        parts = PurePath(relative).parts
        matches = [
            path
            for path in by_name.get(PurePath(relative).name.lower(), [])
            if path.parts[-len(parts) :] == parts
        ]
        if len(matches) == 1:
            result[relative] = matches[0]
    return result


def read_marker_blocks(
    blocks: Iterable[MarkerBlock],
    paths: Mapping[str, PurePath],
    *,
    aliases: ProjectAliases | None = None,
    encoding: str = "latin1",
) -> list[ReccmpParserResult]:
    """Markers and alerts per local file, for blocks in files of ``paths``."""
    readers: dict[str, _FileReader] = {}
    ordered = sorted(blocks, key=lambda block: block.key)
    for block in ordered:
        path = paths.get(block.source_file)
        if path is None:
            continue
        reader = readers.get(block.source_file)
        if reader is None:
            reader = readers[block.source_file] = _FileReader(
                path, aliases or {}, encoding
            )
        reader.read(block)

    function_ids: dict[tuple[str, str], int] = {}
    for reader in readers.values():
        for key, offset in reader.function_ids.items():
            function_ids.setdefault(key, offset)
    for reader in readers.values():
        for static in reader.static_locals:
            parent = function_ids.get((static.symbol.module, static.enclosing_function))
            if parent is None:
                reader.alert(
                    AlertCode.ORPHANED_STATIC_VARIABLE, static.symbol.line_number
                )
                reader.symbols.remove(static.symbol)
            else:
                static.symbol.parent_function = parent

    return [
        ReccmpParserResult(
            tokens=tuple(sorted(reader.symbols, key=lambda item: item.line_number)),
            alerts=tuple(reader.alerts),
            path=reader.path,
        )
        for reader in readers.values()
    ]


def uncompiled_markers(
    files: Iterable[tuple[PurePath, str]],
    blocks: Sequence[MarkerBlock],
    paths: Mapping[str, PurePath],
    aliases: ProjectAliases | None = None,
) -> Iterator[ParserAlert]:
    """Marker lines the compiler never saw: in files no translation unit
    includes, or in code the preprocessor skipped."""
    seen: set[tuple[PurePath, int]] = set()
    for block in blocks:
        path = paths.get(block.source_file)
        if path is not None:
            seen.update((path, comment.line) for comment in block.comments)
    for path, text in files:
        for number, line in enumerate(text.splitlines(), 1):
            if (path, number) in seen:
                continue
            marker = match_marker(line, aliases=aliases)
            if marker is not None:
                yield ParserAlert(
                    code=AlertCode.MARKER_NOT_COMPILED,
                    path=path,
                    line_number=number,
                    detail=line.strip(),
                    target=marker.module,
                )
