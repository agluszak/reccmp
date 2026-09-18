"""Stack-layout pairing shared by the comparator and reccmp-stackcmp.

Infers an orig↔recomp mapping of ebp/esp-relative offsets from a unified
assembly diff, and scores how much of a mismatch collapses once that map is
applied.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Sequence

from reccmp.compare.asm.model import STACK_ENTRY_REGEX
from reccmp.compare.diagnosis import StackPermutationEntry
from reccmp.compare.diff import (
    CombinedDiffOutput,
    MatchingOrMismatchingBlock,
    RawDiffOutput,
)
from reccmp.compare.diff import raw_diff_to_udiff
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.cvdump.types import CvdumpTypeKey

StackRefKind = Literal["argument", "local", "spill", "saved", "unknown"]


@dataclass(frozen=True)
class CanonicalStackRef:
    """Shared vocabulary for ebp/esp-relative slots across stackcmp and diagnosis."""

    kind: StackRefKind
    key: str | int
    physical: tuple[str, int] | None = None

    def label(self) -> str:
        if self.kind == "argument":
            return f"arg[{self.key}]"
        if self.kind == "local":
            return f"local[{self.key}]"
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
) -> CanonicalStackRef:
    """Map a physical (reg, offset) pair to a canonical stack reference.

    Heuristics (MSVC thiscall/cdecl frame):
    - ``ebp`` with positive offset → argument (``ebp+8`` is arg 0)
    - ``ebp`` with negative offset → local
    - ``ebp``/``ebp+4`` → saved frame / return address
    - ``esp`` → spill when listed in ``known_spills``, else unknown
    """
    reg = register.lower()
    physical = (reg, offset)
    if reg == "ebp":
        if offset == 0:
            return CanonicalStackRef("saved", "ebp", physical)
        if offset == 4:
            return CanonicalStackRef("saved", "return", physical)
        if offset > 0:
            arg_index = (offset - 8) // 4 if offset >= 8 else offset
            return CanonicalStackRef("argument", arg_index, physical)
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
        self, *, known_spills: set[int] | frozenset[int] | None = None
    ) -> "StackRegisterOffset":
        self.canonical = canonical_stack_ref(
            self.register, self.offset, known_spills=known_spills
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


def extract_stack_offset_from_instruction(
    instruction: str,
) -> StackRegisterOffset | None:
    match = STACK_ENTRY_REGEX.search(instruction)
    if not match:
        return None
    offset = int(match.group("sign") + match.group("offset"), 16)
    slot = StackRegisterOffset(match.group("register"), offset)
    slot.attach_canonical()
    return slot


def annotate_canonical_refs(
    stack_pairs: StackPairs,
    *,
    known_spills: set[int] | frozenset[int] | None = None,
) -> None:
    """Attach or refresh CanonicalStackRef labels on every observed slot."""
    for orig, recomp in stack_pairs:
        orig.attach_canonical(known_spills=known_spills)
        recomp.attach_canonical(known_spills=known_spills)


def analyze_diff_block(
    diff: MatchingOrMismatchingBlock, warnings: Warnings
) -> StackPairs:
    stack_pairs: StackPairs = set()
    if "both" in diff:
        for line in diff["both"]:
            instruction = line[1]
            if match := extract_stack_offset_from_instruction(instruction):
                stack_pairs.add(StackPair(match, match.copy()))
        return stack_pairs

    assert "orig" in diff
    assert "recomp" in diff
    orig = diff["orig"]
    recomp = diff["recomp"]
    if len(orig) != len(recomp):
        warnings.structural_mismatches_present = True
        return set()

    for orig_line, recomp_line in zip(orig, recomp):
        if orig_match := extract_stack_offset_from_instruction(orig_line[1]):
            recomp_match = extract_stack_offset_from_instruction(recomp_line[1])
            if not recomp_match:
                warnings.structural_mismatches_present = True
                return set()
            stack_pairs.add(StackPair(orig_match, recomp_match))
    return stack_pairs


def collect_stack_pairs(udiff: CombinedDiffOutput) -> tuple[StackPairs, Warnings]:
    warnings = Warnings()
    stack_pairs: StackPairs = set()
    for block in udiff:
        for diff in block[1]:
            stack_pairs |= analyze_diff_block(diff, warnings)
    return stack_pairs, warnings


def annotate_recomp_symbols(
    stack_pairs: StackPairs, fn_symbol: SymbolsEntry | None
) -> dict[int, StackSymbol]:
    """Attach PDB S_BPREL32 names to ebp-relative recomp offsets."""
    stack_symbols: dict[int, StackSymbol] = {}
    if fn_symbol is None:
        return stack_symbols
    for symbol in fn_symbol.symbols:
        if symbol.symbol_type != "S_BPREL32":
            continue
        hex_bytes = bytes.fromhex(symbol.location[1:-1])
        stack_offset = struct.unpack(">l", hex_bytes)[0]
        stack_symbols[stack_offset] = StackSymbol(symbol.name, symbol.data_type)
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
            stack_pairs, key=lambda p: (p.orig.offset, p.recomp.offset)
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
    for orig_key, recomp_key in sorted(mapping.items(), key=lambda kv: kv[0][1]):
        orig = StackRegisterOffset(*orig_key)
        recomp = StackRegisterOffset(*recomp_key)
        # Recover symbol from any matching pair.
        symbol = next(
            (
                p.recomp.symbol.name
                for p in stack_pairs
                if p.orig == orig and p.recomp == recomp and p.recomp.symbol
            ),
            None,
        )
        entries.append(StackPermutationEntry(orig.label(), recomp.label(), symbol))
    return tuple(entries)


def accuracy_after_stack_map(
    orig_asm: Sequence[str],
    recomp_asm: Sequence[str],
    mapping: dict[tuple[str, int], tuple[str, int]],
) -> float:
    """SequenceMatcher ratio after rewriting orig stack offsets toward recomp."""
    from reccmp.compare.asm.ir import rewrite_stack_displacements
    from reccmp.compare.pinned_sequences import SequenceMatcherWithPins

    if not mapping:
        return SequenceMatcherWithPins(list(orig_asm), list(recomp_asm), []).ratio()

    rewritten = [rewrite_stack_displacements(line, mapping) for line in orig_asm]
    return SequenceMatcherWithPins(rewritten, list(recomp_asm), []).ratio()


def analyze_stack_layout(
    rdiff: RawDiffOutput | None,
    orig_asm: Sequence[str],
    recomp_asm: Sequence[str],
    fn_symbol: SymbolsEntry | None = None,
) -> StackLayoutResult | None:
    """Infer stack permutation and modulo-stack accuracy from a raw diff.

    Returns None when there is no diff or no stack offsets to analyze.
    """
    if rdiff is None or not rdiff.codes:
        return None

    udiff = raw_diff_to_udiff(rdiff, grouped=False)
    stack_pairs, warnings = collect_stack_pairs(udiff)
    if not stack_pairs:
        return None

    annotate_recomp_symbols(stack_pairs, fn_symbol)
    annotate_canonical_refs(stack_pairs)
    mapping, bijective = build_slot_bijection(stack_pairs)
    # Identity pairs do not need rewriting; only non-identity matter for score.
    non_identity = {k: v for k, v in mapping.items() if k != v}
    modulo = None
    if bijective and non_identity:
        modulo = accuracy_after_stack_map(orig_asm, recomp_asm, non_identity)
    elif bijective and not non_identity:
        # All observed stack offsets already agree.
        modulo = None

    return StackLayoutResult(
        permutation=permutation_entries(stack_pairs),
        accuracy_modulo_stack=modulo,
        bijective=bijective,
        structural_mismatch=warnings.structural_mismatches_present,
        pairs=stack_pairs,
    )
