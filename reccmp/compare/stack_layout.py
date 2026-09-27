"""Stack-layout pairing shared by the comparator and reccmp-stackcmp.

Infers an orig↔recomp mapping of ebp/esp-relative offsets from the paired
instructions of a diff, and scores how much of a mismatch collapses once
that map is applied.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Sequence

from reccmp.compare.asm.ir import DecodedInstruction, instruction_match_key
from reccmp.compare.diagnosis import StackPermutationEntry
from reccmp.compare.pinned_sequences import DiffOpcode, SequenceMatcherWithPins
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.cvdump.types import CvdumpTypeKey

StackRefKind = Literal["argument", "local", "spill", "saved", "unknown"]


@dataclass(frozen=True)
class CanonicalStackRef:
    """Shared vocabulary for ebp/esp-relative slots across stackcmp and diagnosis."""

    kind: StackRefKind
    key: str | int
    physical: tuple[str, int] | None = None
    # Byte offset within a multi-slot argument/local (e.g. double / struct).
    within: int = 0

    def label(self) -> str:
        suffix = f"+{self.within:#x}" if self.within else ""
        if self.kind == "argument":
            return f"arg[{self.key}]{suffix}"
        if self.kind == "local":
            if isinstance(self.key, str):
                return f"local[{self.key}]{suffix}"
            return f"local[{self.key}]{suffix}"
        if self.kind == "spill":
            return f"spill[{self.key}]"
        if self.kind == "saved":
            return f"saved[{self.key}]"
        return f"stack[{self.key}]"


def canonical_stack_ref(
    register: str,
    offset: int,
    *,
    known_spills: set[int] | frozenset[int] | None = None,
    pdb_slots: Sequence[tuple[int, int, str, StackRefKind]] | None = None,
) -> CanonicalStackRef:
    # pylint: disable=too-many-return-statements
    """Map a physical (reg, offset) pair to a canonical stack reference.

    When ``pdb_slots`` is provided (``(start, size, name, kind)`` covering
    ranges from S_BPREL32 + type size), prefer named multi-slot coverage over
    the dword-index heuristic.

    Heuristics (MSVC thiscall/cdecl frame):
    - ``ebp`` with positive offset → argument (``ebp+8`` is arg 0)
    - ``ebp`` with negative offset → local
    - ``ebp``/``ebp+4`` → saved frame / return address
    - ``esp`` → spill when listed in ``known_spills``, else unknown
    """
    reg = register.lower()
    physical = (reg, offset)
    if pdb_slots is not None and reg == "ebp":
        for start, size, name, kind in pdb_slots:
            if start <= offset < start + max(size, 1):
                return CanonicalStackRef(kind, name, physical, within=offset - start)
    if reg == "ebp":
        if offset == 0:
            return CanonicalStackRef("saved", "ebp", physical)
        if offset == 4:
            return CanonicalStackRef("saved", "return", physical)
        if offset > 0:
            # Only dword-aligned ebp+8+4k slots get argument indices.
            if offset >= 8 and offset % 4 == 0:
                arg_index = (offset - 8) // 4
                return CanonicalStackRef("argument", arg_index, physical)
            return CanonicalStackRef("unknown", offset, physical)
        return CanonicalStackRef("local", offset, physical)
    if reg == "esp":
        if known_spills is not None and offset in known_spills:
            return CanonicalStackRef("spill", offset, physical)
        return CanonicalStackRef("unknown", offset, physical)
    return CanonicalStackRef("unknown", offset, physical)


@dataclass
class StackSymbol:
    name: str
    data_type: CvdumpTypeKey


@dataclass
class StackRegisterOffset:
    register: str
    offset: int
    symbol: StackSymbol | None = None
    canonical: CanonicalStackRef | None = None

    def __str__(self) -> str:
        first_part = (
            f"{self.register} + {self.offset:#04x}"
            if self.offset > 0
            else f"{self.register} - {-self.offset:#04x}"
        )
        second_part = f"  {self.symbol.name}" if self.symbol else ""
        canonical_part = ""
        if self.canonical is not None:
            canonical_part = f"  ({self.canonical.label()})"
        return first_part + second_part + canonical_part

    def __hash__(self) -> int:
        return hash((self.register, self.offset))

    def copy(self) -> "StackRegisterOffset":
        return StackRegisterOffset(
            self.register, self.offset, self.symbol, self.canonical
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StackRegisterOffset)
            and self.register == other.register
            and self.offset == other.offset
        )

    def as_key(self) -> tuple[str, int]:
        return (self.register, self.offset)

    def label(self) -> str:
        if self.canonical is not None:
            return self.canonical.label()
        if self.offset >= 0:
            return f"{self.register}+{self.offset:#x}"
        return f"{self.register}-{ -self.offset:#x}"

    def attach_canonical(
        self,
        *,
        known_spills: set[int] | frozenset[int] | None = None,
        pdb_slots: Sequence[tuple[int, int, str, StackRefKind]] | None = None,
    ) -> "StackRegisterOffset":
        self.canonical = canonical_stack_ref(
            self.register,
            self.offset,
            known_spills=known_spills,
            pdb_slots=pdb_slots,
        )
        return self


class StackPair(NamedTuple):
    orig: StackRegisterOffset
    recomp: StackRegisterOffset


StackPairs = set[StackPair]


@dataclass
class Warnings:
    structural_mismatches_present: bool = False
    error_map_not_bijective: bool = False


@dataclass
class StackLayoutResult:
    """Stack analysis attached to a function comparison."""

    permutation: tuple[StackPermutationEntry, ...] = ()
    accuracy_modulo_stack: float | None = None
    bijective: bool = True
    structural_mismatch: bool = False
    pairs: StackPairs = field(default_factory=set)


def extract_stack_offset_from_operands(
    operands: Sequence,
) -> StackRegisterOffset | None:
    """The first ebp/esp-relative displacement among structured operands."""
    for op in operands:
        match op:
            case ("mem", _, _, [(("ebp" | "esp") as register, _)], int() as disp, ()):
                slot = StackRegisterOffset(register, disp)
                slot.attach_canonical()
                return slot
    return None


def annotate_canonical_refs(
    stack_pairs: StackPairs,
    *,
    known_spills: set[int] | frozenset[int] | None = None,
    pdb_slots: Sequence[tuple[int, int, str, StackRefKind]] | None = None,
) -> None:
    """Attach or refresh CanonicalStackRef labels on every observed slot."""
    for orig, recomp in stack_pairs:
        orig.attach_canonical(known_spills=known_spills, pdb_slots=pdb_slots)
        recomp.attach_canonical(known_spills=known_spills, pdb_slots=pdb_slots)


def pdb_stack_slots(
    fn_symbol: SymbolsEntry | None,
    types: object | None = None,
) -> list[tuple[int, int, str, StackRefKind]]:
    """Build ``(start, size, name, kind)`` coverage from S_BPREL32 + type sizes.

    ``types`` is an optional ``CvdumpTypesParser``; without it every symbol is
    treated as a 4-byte slot.
    """
    slots: list[tuple[int, int, str, StackRefKind]] = []
    if fn_symbol is None:
        return slots
    for symbol in fn_symbol.symbols:
        if symbol.frame_offset is None:
            continue
        stack_offset = symbol.frame_offset
        size = 4
        if types is not None:
            try:
                info = types.get(symbol.data_type)  # type: ignore[attr-defined]
                if info is not None and getattr(info, "size", None):
                    size = int(info.size)
            except Exception:  # pylint: disable=broad-exception-caught
                size = 4
        kind: StackRefKind = "argument" if stack_offset >= 8 else "local"
        if stack_offset in (0, 4):
            kind = "saved"
        slots.append((stack_offset, size, symbol.name, kind))
    slots.sort(key=lambda item: (item[0], -item[1]))
    return slots


def _stack_slot(row: DecodedInstruction) -> StackRegisterOffset | None:
    return extract_stack_offset_from_operands(row.operands)


def collect_stack_pairs(
    orig: Sequence[DecodedInstruction],
    recomp: Sequence[DecodedInstruction],
    opcodes: Sequence[DiffOpcode],
) -> tuple[StackPairs, Warnings]:
    """The stack slots paired rows use: the same slot where the diff says
    the rows are equal, the two sides' slots where it pairs a replacement
    row by row. A replacement of unequal length, or a slot one side uses
    where the other uses none, is a structural mismatch."""
    warnings = Warnings()
    stack_pairs: StackPairs = set()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for row in orig[i1:i2]:
                if (slot := _stack_slot(row)) is not None:
                    stack_pairs.add(StackPair(slot, slot.copy()))
            continue
        if (i2 - i1) != (j2 - j1):
            warnings.structural_mismatches_present = True
            continue
        block: StackPairs = set()
        for row_o, row_r in zip(orig[i1:i2], recomp[j1:j2]):
            if (slot_o := _stack_slot(row_o)) is None:
                continue
            if (slot_r := _stack_slot(row_r)) is None:
                warnings.structural_mismatches_present = True
                block = set()
                break
            block.add(StackPair(slot_o, slot_r))
        stack_pairs |= block
    return stack_pairs, warnings


def annotate_recomp_symbols(
    stack_pairs: StackPairs, fn_symbol: SymbolsEntry | None
) -> dict[int, StackSymbol]:
    """Attach PDB S_BPREL32 names to ebp-relative recomp offsets."""
    stack_symbols: dict[int, StackSymbol] = {}
    if fn_symbol is None:
        return stack_symbols
    for symbol in fn_symbol.symbols:
        if symbol.frame_offset is not None:
            stack_symbols[symbol.frame_offset] = StackSymbol(
                symbol.name, symbol.data_type
            )
    for _, recomp in stack_pairs:
        if recomp.register == "ebp":
            recomp.symbol = stack_symbols.get(recomp.offset)
    return stack_symbols


def build_slot_bijection(
    stack_pairs: StackPairs,
) -> tuple[dict[tuple[str, int], tuple[str, int]], bool]:
    """Build orig→recomp offset map when the correspondence is 1:1.

    Returns (mapping, is_bijective). Multi-maps yield an empty mapping and False.
    """
    by_orig: dict[tuple[str, int], set[tuple[str, int]]] = {}
    by_recomp: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for orig, recomp in stack_pairs:
        by_orig.setdefault(orig.as_key(), set()).add(recomp.as_key())
        by_recomp.setdefault(recomp.as_key(), set()).add(orig.as_key())

    if any(len(v) != 1 for v in by_orig.values()) or any(
        len(v) != 1 for v in by_recomp.values()
    ):
        return {}, False

    return {orig: next(iter(recomps)) for orig, recomps in by_orig.items()}, True


def permutation_entries(
    stack_pairs: StackPairs,
) -> tuple[StackPermutationEntry, ...]:
    mapping, bijective = build_slot_bijection(stack_pairs)
    if not bijective:
        # Still surface the observed pairs for diagnosis, even if not 1:1.
        entries: list[StackPermutationEntry] = []
        seen: set[tuple[str, str]] = set()
        for orig, recomp in sorted(
            stack_pairs,
            key=lambda p: (
                p.orig.offset,
                p.recomp.offset,
                p.orig.label(),
                p.recomp.label(),
                p.recomp.symbol.name if p.recomp.symbol else "",
            ),
        ):
            key = (orig.label(), recomp.label())
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                StackPermutationEntry(
                    orig.label(),
                    recomp.label(),
                    recomp.symbol.name if recomp.symbol else None,
                )
            )
        return tuple(entries)

    entries = []
    # Offset first, then the full slot identity so equal offsets on
    # different base registers have a stable order across processes.
    for orig_key, recomp_key in sorted(
        mapping.items(), key=lambda kv: (kv[0][1], repr(kv[0]), repr(kv[1]))
    ):
        orig = StackRegisterOffset(*orig_key)
        recomp = StackRegisterOffset(*recomp_key)
        # Recover symbol from any matching pair.
        symbol = min(
            (
                p.recomp.symbol.name
                for p in stack_pairs
                if p.orig == orig and p.recomp == recomp and p.recomp.symbol
            ),
            default=None,
        )
        entries.append(StackPermutationEntry(orig.label(), recomp.label(), symbol))
    return tuple(entries)


def _remapped_key(
    row: DecodedInstruction, mapping: dict[tuple[str, int], tuple[str, int]]
) -> Hashable:
    """A row's match key with its stack slot moved through ``mapping``."""
    operands = []
    for op in row.operands:
        match op:
            case (
                "mem",
                size,
                seg,
                [(("ebp" | "esp") as register, 1)],
                int() as disp,
                (),
            ) if (register, disp) in mapping:
                new_register, new_disp = mapping[(register, disp)]
                operands.append(("mem", size, seg, [(new_register, 1)], new_disp, ()))
            case _:
                operands.append(op)
    return instruction_match_key(
        DecodedInstruction(
            address=row.address,
            size=row.size,
            mnemonic=row.mnemonic,
            prefix=row.prefix,
            operands=tuple(operands),
            display="",
        )
    )


def accuracy_after_stack_map(
    orig: Sequence[DecodedInstruction],
    recomp: Sequence[DecodedInstruction],
    mapping: dict[tuple[str, int], tuple[str, int]],
) -> float:
    """SequenceMatcher ratio after moving orig stack slots toward recomp's."""
    orig_keys = [_remapped_key(row, mapping) for row in orig]
    recomp_keys = [instruction_match_key(row) for row in recomp]
    return SequenceMatcherWithPins(orig_keys, recomp_keys, []).ratio()


def analyze_stack_layout(
    orig: Sequence[DecodedInstruction],
    recomp: Sequence[DecodedInstruction],
    opcodes: Sequence[DiffOpcode],
    fn_symbol: SymbolsEntry | None = None,
    types: object | None = None,
) -> StackLayoutResult | None:
    """Infer stack permutation and modulo-stack accuracy from paired rows.

    Returns None when there is no diff or no stack offsets to analyze.
    """
    if not opcodes:
        return None

    stack_pairs, warnings = collect_stack_pairs(orig, recomp, opcodes)
    if not stack_pairs:
        return None

    annotate_recomp_symbols(stack_pairs, fn_symbol)
    slots = pdb_stack_slots(fn_symbol, types)
    annotate_canonical_refs(stack_pairs, pdb_slots=slots or None)
    mapping, bijective = build_slot_bijection(stack_pairs)
    # Identity pairs do not need rewriting; only non-identity matter for score.
    non_identity = {k: v for k, v in mapping.items() if k != v}
    modulo = None
    if bijective and non_identity:
        modulo = accuracy_after_stack_map(orig, recomp, non_identity)

    return StackLayoutResult(
        permutation=permutation_entries(stack_pairs),
        accuracy_modulo_stack=modulo,
        bijective=bijective,
        structural_mismatch=warnings.structural_mismatches_present,
        pairs=stack_pairs,
    )
