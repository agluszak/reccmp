"""Detect probable inline expansions of known helper bodies.

Given a helper fingerprint (normalized mnemonic/operand shape), search larger
functions for contiguous subsequences that match the helper body.  Hits are
diagnostic proposals — not automatic semantic proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Literal, Sequence

from reccmp.compare.pinned_sequences import SequenceMatcherWithPins

Fingerprint = tuple[tuple[str, str], ...]
FingerprintFn = Callable[[int, int], Fingerprint | None]
Counterpart = Literal["call", "inline", "absent"]
InlineSide = Literal["orig", "recomp", "both"]

# Trailing opcodes that belong to a standalone helper epilog, not an inline site.
_EPILOG_MNEMONICS = frozenset({"ret", "retn", "retf"})


@dataclass(frozen=True)
class InlineHit:
    """One probable expansion of a helper inside a larger function."""

    host_addr: int
    host_name: str
    match_offset: int  # instruction index within the host body
    match_length: int
    confidence: float


@dataclass(frozen=True)
class InlineExpansionEvidence:
    """One helper expansion observed while comparing a paired function."""

    helper_name: str
    helper_orig_addr: int
    helper_recomp_addr: int
    side: InlineSide
    match_offset: int
    match_length: int
    counterpart: Counterpart
    counterpart_offset: int | None = None
    confidence: float = 0.0


@dataclass
class InlineLayoutResult:
    expansions: tuple[InlineExpansionEvidence, ...] = ()
    accuracy_modulo_inline: float | None = None


@dataclass(frozen=True)
class HelperCatalogEntry:
    """Cached fingerprint for a paired helper usable as an inline needle."""

    orig_addr: int
    recomp_addr: int
    name: str
    fingerprint: Fingerprint  # epilog already stripped
    byte_size: int


@dataclass(frozen=True)
class _Span:
    offset: int
    length: int
    helper_orig: int
    prefer_call: bool
    confidence: float


def strip_helper_epilog(fingerprint: Fingerprint) -> Fingerprint:
    """Drop a trailing ret so the needle matches an inlined body."""
    if not fingerprint:
        return fingerprint
    mnemonic, _ = fingerprint[-1]
    if mnemonic.lower() in _EPILOG_MNEMONICS:
        return fingerprint[:-1]
    return fingerprint


def find_fingerprint_spans(haystack: Fingerprint, needle: Fingerprint) -> list[int]:
    """Return every start index where ``needle`` appears contiguously."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return []
    starts: list[int] = []
    for start in range(len(haystack) - n + 1):
        if haystack[start : start + n] == needle:
            starts.append(start)
    return starts


def _subsequence_index(haystack: Fingerprint, needle: Fingerprint) -> int | None:
    starts = find_fingerprint_spans(haystack, needle)
    return starts[0] if starts else None


def select_nonoverlapping(spans: Sequence[_Span]) -> list[_Span]:
    """Greedy non-overlapping selection: longer, call-backed, higher confidence first."""
    ordered = sorted(
        spans,
        key=lambda s: (s.length, s.prefer_call, s.confidence, -s.offset),
        reverse=True,
    )
    chosen: list[_Span] = []
    occupied: list[tuple[int, int]] = []
    for span in ordered:
        end = span.offset + span.length
        if any(not (end <= a or span.offset >= b) for a, b in occupied):
            continue
        chosen.append(span)
        occupied.append((span.offset, end))
    chosen.sort(key=lambda s: s.offset)
    return chosen


def elide_spans(
    keys: Sequence[Hashable], spans: Sequence[tuple[int, int, Hashable]]
) -> list[Hashable]:
    """Replace each ``(offset, length, placeholder)`` span with one placeholder."""
    if not spans:
        return list(keys)
    ordered = sorted(spans, key=lambda s: s[0])
    out: list[Hashable] = []
    cursor = 0
    for offset, length, placeholder in ordered:
        if offset < cursor:
            continue
        out.extend(keys[cursor:offset])
        out.append(placeholder)
        cursor = offset + length
    out.extend(keys[cursor:])
    return out


def collapse_indices(
    keys: Sequence[Hashable], indices: Sequence[tuple[int, Hashable]]
) -> list[Hashable]:
    """Replace individual instruction indices with placeholders."""
    if not indices:
        return list(keys)
    replace = {index: placeholder for index, placeholder in indices}
    return [replace.get(i, key) for i, key in enumerate(keys)]


def accuracy_after_inline_elision(
    orig_keys: Sequence[Hashable],
    recomp_keys: Sequence[Hashable],
    *,
    orig_elide: Sequence[tuple[int, int, Hashable]] = (),
    recomp_elide: Sequence[tuple[int, int, Hashable]] = (),
    orig_collapse: Sequence[tuple[int, Hashable]] = (),
    recomp_collapse: Sequence[tuple[int, Hashable]] = (),
) -> float:
    """SequenceMatcher ratio after collapsing CALL↔inline asymmetries.

    Collapse (single-instruction CALL → placeholder) runs first on each side,
    then span elision with offsets remapped past collapsed indices.
    """
    left = collapse_indices(orig_keys, orig_collapse)
    right = collapse_indices(recomp_keys, recomp_collapse)
    left = elide_spans(left, _remap_elisions(orig_elide, orig_collapse))
    right = elide_spans(right, _remap_elisions(recomp_elide, recomp_collapse))
    return SequenceMatcherWithPins(left, right, []).ratio()


def _remap_elisions(
    elisions: Sequence[tuple[int, int, Hashable]],
    collapses: Sequence[tuple[int, Hashable]],
) -> list[tuple[int, int, Hashable]]:
    if not collapses:
        return list(elisions)
    collapsed = sorted(index for index, _ in collapses)
    remapped: list[tuple[int, int, Hashable]] = []
    for offset, length, placeholder in elisions:
        if any(offset <= index < offset + length for index in collapsed):
            continue
        shift = sum(1 for index in collapsed if index < offset)
        remapped.append((offset - shift, length, placeholder))
    return remapped


def asm_fingerprint_from_lines(lines: Sequence[str]) -> Fingerprint:
    """Build a cheap (mnemonic, operand) fingerprint from sanitized asm lines."""
    result: list[tuple[str, str]] = []
    for line in lines:
        if not line or line.startswith("Jump table:") or line.startswith("Data table:"):
            continue
        if line.startswith("start + ") or (
            line.startswith("0x") and " " not in line.strip()
        ):
            continue
        mnemonic, _, operand = line.partition(" ")
        if mnemonic in ("rep", "repe", "repne"):
            rest_m, _, rest_o = operand.partition(" ")
            mnemonic = f"{mnemonic} {rest_m}"
            operand = rest_o
        result.append((mnemonic, operand))
    return tuple(result)


def find_call_indices_in_fingerprint(
    fingerprint: Fingerprint, helper_names: Sequence[str]
) -> list[int]:
    """Indices of ``call`` entries whose operand mentions a helper name."""
    needles = [name for name in helper_names if name]
    if not needles:
        return []
    indices: list[int] = []
    for i, (mnemonic, operand) in enumerate(fingerprint):
        if mnemonic != "call":
            continue
        for name in needles:
            if name in operand:
                indices.append(i)
                break
    return indices


def find_call_indices(lines: Sequence[str], helper_names: Sequence[str]) -> list[int]:
    """Indices of ``call`` instructions whose operand mentions a helper name."""
    return find_call_indices_in_fingerprint(
        asm_fingerprint_from_lines(lines), helper_names
    )


def find_inline_expansions(
    helper_fingerprint: Fingerprint,
    hosts: Sequence[tuple[int, str, int]],
    fingerprint_of: FingerprintFn,
    *,
    min_helper_ops: int = 3,
) -> list[InlineHit]:
    """Search host functions for contiguous expansions of ``helper_fingerprint``.

    ``hosts`` entries are ``(addr, name, size)``.  Hosts whose fingerprint is
    shorter than or equal to the helper are skipped (they cannot contain it as
    a strict expansion).
    """
    needle = strip_helper_epilog(helper_fingerprint)
    if len(needle) < min_helper_ops:
        return []

    hits: list[InlineHit] = []
    helper_len = len(needle)
    for addr, name, size in hosts:
        host_fp = fingerprint_of(addr, size)
        if host_fp is None or len(host_fp) <= helper_len:
            continue
        offset = _subsequence_index(host_fp, needle)
        if offset is None:
            continue
        confidence = helper_len / len(host_fp)
        hits.append(
            InlineHit(
                host_addr=addr,
                host_name=name,
                match_offset=offset,
                match_length=helper_len,
                confidence=confidence,
            )
        )
    hits.sort(key=lambda h: (-h.confidence, h.host_addr))
    return hits


@dataclass
class _HelperPlan:
    entry: HelperCatalogEntry
    orig_span: tuple[int, int] | None = None
    recomp_span: tuple[int, int] | None = None
    orig_call: int | None = None
    recomp_call: int | None = None


def analyze_inline_layout(
    orig_asm: Sequence[str],
    recomp_asm: Sequence[str],
    helpers: Sequence[HelperCatalogEntry],
    *,
    min_helper_ops: int = 3,
    exclude_orig_addrs: Sequence[int] = (),
) -> InlineLayoutResult:
    """Detect CALL↔inline asymmetries and score the streams after collapsing them."""
    orig_fp = asm_fingerprint_from_lines(orig_asm)
    recomp_fp = asm_fingerprint_from_lines(recomp_asm)
    if not orig_fp or not recomp_fp:
        return InlineLayoutResult()

    # Fingerprint lines skip jump/data tables, so indices must map back carefully.
    # For sanitized CODE-only excerpts (the common case) indices align 1:1 with
    # asm lines.  When tables are present, fall back to line-index search on the
    # raw asm fingerprint built only from instruction-shaped lines — we still
    # use the same filtered stream for SequenceMatcher keys below.
    orig_keys = list(orig_fp)
    recomp_keys = list(recomp_fp)

    excluded = set(exclude_orig_addrs)
    plans: list[_HelperPlan] = []
    for helper in helpers:
        if helper.orig_addr in excluded:
            continue
        needle = helper.fingerprint
        if len(needle) < min_helper_ops:
            continue
        if len(needle) >= len(orig_fp) and len(needle) >= len(recomp_fp):
            continue

        names = [helper.name]
        plan = _HelperPlan(entry=helper)
        orig_starts = find_fingerprint_spans(orig_fp, needle)
        recomp_starts = find_fingerprint_spans(recomp_fp, needle)
        if orig_starts:
            plan.orig_span = (orig_starts[0], len(needle))
        if recomp_starts:
            plan.recomp_span = (recomp_starts[0], len(needle))
        plan.orig_call = next(
            iter(find_call_indices_in_fingerprint(orig_fp, names)), None
        )
        plan.recomp_call = next(
            iter(find_call_indices_in_fingerprint(recomp_fp, names)), None
        )

        interesting = False
        if plan.orig_span and plan.recomp_call is not None and plan.recomp_span is None:
            interesting = True
        elif plan.recomp_span and plan.orig_call is not None and plan.orig_span is None:
            interesting = True
        elif plan.orig_span and plan.recomp_span:
            interesting = True
        if interesting:
            plans.append(plan)

    if not plans:
        return InlineLayoutResult()

    # Resolve overlapping spans per side.
    orig_span_objs = [
        _Span(
            plan.orig_span[0],
            plan.orig_span[1],
            plan.entry.orig_addr,
            prefer_call=plan.recomp_call is not None,
            confidence=plan.orig_span[1] / max(len(orig_fp), 1),
        )
        for plan in plans
        if plan.orig_span is not None
    ]
    recomp_span_objs = [
        _Span(
            plan.recomp_span[0],
            plan.recomp_span[1],
            plan.entry.orig_addr,
            prefer_call=plan.orig_call is not None,
            confidence=plan.recomp_span[1] / max(len(recomp_fp), 1),
        )
        for plan in plans
        if plan.recomp_span is not None
    ]
    kept_orig = {(s.helper_orig, s.offset) for s in select_nonoverlapping(orig_span_objs)}
    kept_recomp = {
        (s.helper_orig, s.offset) for s in select_nonoverlapping(recomp_span_objs)
    }

    expansions: list[InlineExpansionEvidence] = []
    orig_elide: list[tuple[int, int, Hashable]] = []
    recomp_elide: list[tuple[int, int, Hashable]] = []
    orig_collapse: list[tuple[int, Hashable]] = []
    recomp_collapse: list[tuple[int, Hashable]] = []

    for plan in plans:
        helper = plan.entry
        placeholder: Hashable = ("inline", helper.orig_addr)
        orig_kept = (
            plan.orig_span is not None
            and (helper.orig_addr, plan.orig_span[0]) in kept_orig
        )
        recomp_kept = (
            plan.recomp_span is not None
            and (helper.orig_addr, plan.recomp_span[0]) in kept_recomp
        )

        if orig_kept and plan.recomp_call is not None and not recomp_kept:
            assert plan.orig_span is not None
            orig_elide.append((*plan.orig_span, placeholder))
            recomp_collapse.append((plan.recomp_call, placeholder))
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="orig",
                    match_offset=plan.orig_span[0],
                    match_length=plan.orig_span[1],
                    counterpart="call",
                    counterpart_offset=plan.recomp_call,
                    confidence=plan.orig_span[1] / max(len(orig_fp), 1),
                )
            )
        elif recomp_kept and plan.orig_call is not None and not orig_kept:
            assert plan.recomp_span is not None
            recomp_elide.append((*plan.recomp_span, placeholder))
            orig_collapse.append((plan.orig_call, placeholder))
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="recomp",
                    match_offset=plan.recomp_span[0],
                    match_length=plan.recomp_span[1],
                    counterpart="call",
                    counterpart_offset=plan.orig_call,
                    confidence=plan.recomp_span[1] / max(len(recomp_fp), 1),
                )
            )
        elif orig_kept and recomp_kept:
            assert plan.orig_span is not None and plan.recomp_span is not None
            orig_elide.append((*plan.orig_span, placeholder))
            recomp_elide.append((*plan.recomp_span, placeholder))
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="both",
                    match_offset=plan.orig_span[0],
                    match_length=plan.orig_span[1],
                    counterpart="inline",
                    counterpart_offset=plan.recomp_span[0],
                    confidence=min(
                        plan.orig_span[1] / max(len(orig_fp), 1),
                        plan.recomp_span[1] / max(len(recomp_fp), 1),
                    ),
                )
            )

    if not expansions:
        return InlineLayoutResult()

    modulo = accuracy_after_inline_elision(
        orig_keys,
        recomp_keys,
        orig_elide=orig_elide,
        recomp_elide=recomp_elide,
        orig_collapse=orig_collapse,
        recomp_collapse=recomp_collapse,
    )
    return InlineLayoutResult(
        expansions=tuple(expansions),
        accuracy_modulo_inline=modulo,
    )
