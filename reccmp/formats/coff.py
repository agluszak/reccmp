"""i386 COFF object contributions, including data-only translation units.

Keep linker names, storage classes, auxiliary records and relocation targets.
Section/symbol extents are not a claim about an original function's size.
"""

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class CoffRelocation:
    offset: int
    symbol_index: int
    type: int


@dataclass(frozen=True)
class CoffSymbol:
    index: int
    name: str
    value: int
    section: int
    type: int
    storage_class: int
    auxiliaries: tuple[bytes, ...]

    @property
    def is_function(self) -> bool:
        return self.section > 0 and self.type & 0x30 == 0x20

    @property
    def is_common(self) -> bool:
        return self.section == 0 and self.value > 0 and self.storage_class == 2


@dataclass(frozen=True)
class CoffSection:
    index: int
    name: str
    size: int
    data: bytes
    characteristics: int
    relocations: tuple[CoffRelocation, ...]


@dataclass(frozen=True)
class CoffContribution:
    symbol: CoffSymbol
    section: CoffSection | None
    data: bytes
    relocations: tuple[CoffRelocation, ...]


@dataclass(frozen=True)
class CoffObject:
    path: Path
    sections: tuple[CoffSection, ...]
    symbols: tuple[CoffSymbol, ...]

    def contribution(self, name: str) -> CoffContribution:
        """Select a defined symbol without guessing aliases or demangled names.

        Function auxiliary records supply the compiler's extent when present;
        otherwise the next distinct symbol offset bounds the contribution.
        Same-address aliases share that extent. Section descriptors and debug
        labels do not bound it. BSS/common symbols retain zero-initialized storage.
        """
        matches = [
            s for s in self.symbols if s.name == name and (s.section > 0 or s.is_common)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one defined {name!r} in {self.path}, found {len(matches)}"
            )
        symbol = matches[0]
        if symbol.is_common:
            return CoffContribution(symbol, None, bytes(symbol.value), ())
        section = self.sections[symbol.section - 1]
        ends = [
            s.value
            for s in self.symbols
            if s.section == symbol.section
            and s.value > symbol.value
            and s.storage_class in (2, 3)
            and s.name != section.name
        ]
        end = min(ends, default=section.size)
        if symbol.is_function and symbol.auxiliaries:
            size = struct.unpack_from("<I", symbol.auxiliaries[0], 4)[0]
            if size:
                if symbol.value + size > end:
                    raise ValueError(
                        f"Function extent exceeds COFF contribution: {symbol.name}"
                    )
                end = symbol.value + size
        if not 0 <= symbol.value <= end <= section.size:
            raise ValueError(f"Invalid COFF contribution extent: {symbol.name}")
        data = (section.data if section.data else bytes(section.size))[
            symbol.value : end
        ]
        relocations = tuple(
            CoffRelocation(r.offset - symbol.value, r.symbol_index, r.type)
            for r in section.relocations
            if symbol.value <= r.offset < end
        )
        return CoffContribution(symbol, section, data, relocations)


def parse_coff_object(path: Path) -> CoffObject:
    """Read one ordinary i386 COFF object; no executable or PDB is required."""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Truncated COFF header: {path}")
    machine, count, _, table, symbols_count, optional, _ = struct.unpack_from(
        "<HHIIIHH", data
    )
    if machine != 0x14C or optional:
        raise ValueError(f"Not an ordinary i386 COFF object: {path}")
    strings = table + symbols_count * 18

    def read(offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > len(data):
            raise ValueError(f"Truncated COFF record at {offset:#x}: {path}")
        return data[offset : offset + size]

    def string(offset: int) -> str:
        length = struct.unpack("<I", read(strings, 4))[0]
        if not 4 <= offset < length:
            raise ValueError(f"Invalid COFF string offset {offset}: {path}")
        return (
            read(strings + offset, length - offset)
            .split(b"\0", 1)[0]
            .decode("utf-8", errors="replace")
        )

    def name(raw: bytes, section: bool = False) -> str:
        if raw[:4] == b"\0" * 4:
            return string(struct.unpack_from("<I", raw, 4)[0])
        value = raw.rstrip(b"\0").decode("utf-8", errors="replace")
        return string(int(value[1:])) if section and value.startswith("/") else value

    sections = []
    for index in range(1, count + 1):
        header = read(20 + (index - 1) * 40, 40)
        size, raw, relocs, _, nrelocs, _, flags = struct.unpack_from(
            "<IIIIHHI", header, 16
        )
        if flags & 0x01000000:
            raise ValueError(f"Extended COFF relocation counts are unsupported: {path}")
        relocations = tuple(
            CoffRelocation(*struct.unpack("<IIH", read(relocs + i * 10, 10)))
            for i in range(nrelocs)
        )
        sections.append(
            CoffSection(
                index,
                name(header[:8], True),
                size,
                read(raw, size) if raw else b"",
                flags,
                relocations,
            )
        )
    symbols = []
    index = 0
    while index < symbols_count:
        raw = read(table + index * 18, 18)
        value, section, kind, storage, aux = struct.unpack_from("<IhHBB", raw, 8)
        if index + aux >= symbols_count or section > count:
            raise ValueError(f"Invalid COFF symbol {index}: {path}")
        auxiliaries = tuple(read(table + (index + i + 1) * 18, 18) for i in range(aux))
        symbols.append(
            CoffSymbol(index, name(raw[:8]), value, section, kind, storage, auxiliaries)
        )
        index += aux + 1
    return CoffObject(path, tuple(sections), tuple(symbols))
