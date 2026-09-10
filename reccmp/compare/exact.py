"""Relocation-masked comparison of COFF contributions against an original PE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reccmp.formats.coff import CoffObject, CoffRelocation, parse_coff_object
from reccmp.formats.pe import PEImage

RELOCATION_TYPES = frozenset({0x06, 0x07, 0x14})


@dataclass(frozen=True)
class CoffFunction:
    """One i386 COFF function and its contributing relocations."""

    name: str
    body: bytes
    relocation_offsets: tuple[int, ...]
    source_object: str = ""

    @property
    def masked_body(self) -> bytes:
        return mask_relocations(self.body, self.relocation_offsets)


@dataclass(frozen=True)
class ExactComparison:
    """A recomputed exact-body verdict; no persistent digest is involved."""

    status: str
    exact_mode: str
    original_size: int
    recompiled_size: int
    stable_bytes: int

    @property
    def exact(self) -> bool:
        return self.status == "exact"


def _source_name(name: str) -> str:
    value = name.removeprefix("_")
    if "@" in value and value.rsplit("@", 1)[-1].isdigit():
        value = value.rsplit("@", 1)[0]
    return value


def parse_coff_functions(path: Path) -> list[CoffFunction]:
    """Compatibility function view of the complete object, including statics.

    Do not strip bytes which could be code or data. Callers supply an original
    extent when comparing; section alignment is not a function-size oracle.
    A data-only object legitimately returns an empty function view.
    """
    obj = parse_coff_object(path)
    functions: list[CoffFunction] = []
    for symbol in obj.symbols:
        if not symbol.is_function or symbol.storage_class not in (2, 3):
            continue
        contribution = obj.contribution(symbol.name)
        offsets = _contribution_relocations(
            contribution.relocations, len(contribution.data)
        )
        functions.append(
            CoffFunction(
                _source_name(symbol.name),
                contribution.data,
                offsets,
                source_object=str(path),
            )
        )
    return sorted(functions, key=lambda item: item.name.casefold())


def _contribution_relocations(
    relocations: Iterable[CoffRelocation], size: int
) -> tuple[int, ...]:
    offsets = []
    for relocation in relocations:
        if relocation.offset >= size or relocation.type == 0:
            continue
        if relocation.type not in RELOCATION_TYPES:
            raise ValueError(f"Unsupported i386 relocation type {relocation.type:#x}")
        if relocation.offset < 0 or relocation.offset + 4 > size:
            raise ValueError("Comparison extent cuts a COFF relocation operand")
        offsets.append(relocation.offset)
    return tuple(sorted(set(offsets)))


def compare_object_to_original(
    original: PEImage, obj: CoffObject, symbol: str, address: int, size: int
) -> ExactComparison:
    """Compare a code or data contribution at an independently known extent.

    This proves only relocation-masked byte identity, not relocation target
    identity, source syntax, function ownership, or semantic equivalence.
    Static functions and globals work without extraction from an archive or
    retention in a linked comparison executable.
    """
    if size <= 0:
        raise ValueError("An independently known positive original extent is required")
    contribution = obj.contribution(symbol)
    offsets = _contribution_relocations(
        contribution.relocations, min(size, len(contribution.data))
    )
    original_offsets = tuple(
        r - address for r in original.relocations if address <= r < address + size
    )
    if any(offset + 4 > size for offset in original_offsets):
        raise ValueError("Comparison extent cuts a PE relocation operand")
    return compare_relocation_masked(
        original.read(address, size),
        contribution.data,
        original_relocations=original_offsets,
        recompiled_relocations=offsets,
        size=size,
    )


def mask_relocations(body: bytes, offsets: Iterable[int]) -> bytes:
    masked = bytearray(body)
    for offset in offsets:
        if 0 <= offset and offset + 4 <= len(masked):
            masked[offset : offset + 4] = b"\0\0\0\0"
    return bytes(masked)


def stable_ranges(length: int, offsets: Iterable[int]) -> list[tuple[int, int]]:
    holes = [False] * length
    for offset in offsets:
        if offset < 0 or offset + 4 > length:
            continue
        for index in range(offset, offset + 4):
            holes[index] = True
    ranges: list[tuple[int, int]] = []
    start = None
    for index, hole in enumerate([*holes, True]):
        if not hole and start is None:
            start = index
        elif hole and start is not None:
            ranges.append((start, index))
            start = None
    return ranges


def compare_relocation_masked(
    original: bytes,
    recompiled: bytes,
    *,
    original_relocations: Iterable[int] = (),
    recompiled_relocations: Iterable[int] = (),
    size: int | None = None,
) -> ExactComparison:
    """Compare current bodies after masking all relocatable four-byte operands."""

    original_size = len(original) if size is None else size
    recompiled_size = len(recompiled)
    offsets = tuple(sorted(set(original_relocations) | set(recompiled_relocations)))
    comparable = original[:original_size]
    candidate = recompiled[:original_size]
    ranges = stable_ranges(original_size, offsets)
    exact = (
        len(original) >= original_size > 0
        and recompiled_size >= original_size
        and all(comparable[start:end] == candidate[start:end] for start, end in ranges)
    )
    return ExactComparison(
        status="exact" if exact else "different",
        exact_mode="relocation-masked-object",
        original_size=original_size,
        recompiled_size=recompiled_size,
        stable_bytes=sum(end - start for start, end in ranges),
    )
