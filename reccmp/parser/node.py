from dataclasses import dataclass
from pathlib import PurePath
from .marker import MarkerType


@dataclass
class ParserSymbol:
    """Exported decomp marker with all information (except the code filename) required to
    cross-reference with cvdump data."""

    type: MarkerType
    line_number: int
    module: str
    offset: int
    name: str

    # The parser doesn't (currently) know about the code filename, but if you
    # wanted to set it here after the fact, here's the spot.
    filename: PurePath

    def should_skip(self) -> bool:
        """The default is to compare any symbols we have"""
        return False

    def is_library(self) -> bool:
        """The default is to assume that arbitrary symbols are not library functions"""
        return False

    def is_nameref(self) -> bool:
        """All symbols default to name lookup"""
        return True


@dataclass
class ParserFunction(ParserSymbol):
    # We are able to detect the closing line of a function with some reliability.
    # This isn't used for anything right now, but perhaps later it will be.
    end_line: int | None = None

    # All marker types are referenced by name except FUNCTION/STUB. These can also be
    # referenced by name, but only if this flag is true.
    lookup_by_name: bool = False

    # SYMBOL selects a recomp linker spelling, never independent retail provenance.
    name_is_symbol: bool = False

    # Explicit RECOMP selector; name may instead describe a template family/member.
    recomp_selector: str | None = None

    @property
    def selector(self) -> str:
        return self.recomp_selector if self.recomp_selector is not None else self.name

    @property
    def selector_is_symbol(self) -> bool:
        return self.name_is_symbol or self.selector.startswith("?")

    # True if this address is used by many identical functions.
    is_folded: bool = False

    # Semantic ids of the definitions a line-based marker annotates: one, or
    # one per template instantiation of the same source.
    definitions: tuple[str, ...] = ()

    def should_skip(self) -> bool:
        return self.type == MarkerType.STUB

    def is_library(self) -> bool:
        return self.type == MarkerType.LIBRARY

    def is_nameref(self) -> bool:
        return self.lookup_by_name or (
            not self.definitions
            and self.type
            in (MarkerType.SYNTHETIC, MarkerType.TEMPLATE, MarkerType.LIBRARY)
        )


@dataclass
class ParserVariable(ParserSymbol):
    is_static: bool = False
    parent_function: int | None = None
    # The enclosing function's symbol, when it has no marker of its own
    # (for example a function every caller inlines).
    parent_symbol: str | None = None
    semantic_id: str | None = None


@dataclass
class ParserVtable(ParserSymbol):
    base_class: str | None = None

    # True if this address is shared by many identical VMTs.
    is_folded: bool = False


@dataclass
class ParserString(ParserSymbol):
    is_widechar: bool = False


@dataclass
class ParserLineSymbol(ParserSymbol):
    pass
