"""Detect probable inline expansions of known helper bodies.

Given a helper fingerprint (normalized mnemonic/operand shape), search larger
functions for contiguous subsequences that match the helper body.  Hits are
diagnostic proposals — not automatic semantic proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable
from typing import Callable, Literal, Sequence

from reccmp.compare.asm.ir import DecodedInstruction, instruction_match_key
from reccmp.compare.asm.model import REGISTERS
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins


@dataclass(frozen=True)
class FingerprintRow:
    """One instruction of a fingerprint: its shape (operands as the match
    key freezes them: references by identity, side-local ones by the
    placeholder they show), and for a direct call the callee's identity."""

    prefix: str
    mnemonic: str
    operands: tuple
    callee: Hashable | None = None


Fingerprint = tuple[FingerprintRow, ...]
FingerprintFn = Callable[[int, int], Fingerprint | None]
Counterpart = Literal["call", "inline", "absent"]
InlineSide = Literal["orig", "recomp", "both"]
MatchKind = Literal["literal", "register", "summary"]

# Trailing opcodes that belong to a standalone helper epilog, not an inline site.
_EPILOG_MNEMONICS = frozenset({"ret", "retn", "retf"})
_STORE_MNEMONICS = frozenset(
    {"mov", "movzx", "movsx", "lea", "add", "sub", "or", "xor", "and", "xchg"}
)
_STORE_SIZES = frozenset({"dword", "word", "byte", "qword"})
_UNSUPPORTED_FOR_SUMMARY = frozenset(
    {
        "call",
        "int",
        "syscall",
        "sysenter",
        "cpuid",
        "rdtsc",
        "in",
        "out",
        "ins",
        "outs",
    }
)


def fingerprint_of(rows: Sequence[DecodedInstruction]) -> Fingerprint:
    """The fingerprint of an excerpt's instructions (table rows skipped)."""
    return tuple(
        FingerprintRow(
            row.prefix,
            row.mnemonic,
            instruction_match_key(row)[3],  # type: ignore[index]
            row.control_target if row.is_call else None,
        )
        for row in rows
        if row.is_code and row.mnemonic
    )


@dataclass(frozen=True)
class StoreEffect:
    """One observed store through this/arg + constant displacement."""

    base: Literal["this", "arg", "stack"]
    displacement: int


@dataclass(frozen=True)
class HelperEffectSummary:
    """Lightweight net-effect fingerprint for a short helper body."""

    inputs: tuple[str, ...]
    stores: tuple[StoreEffect, ...]
    return_kind: Literal["void", "register", "stack", "unknown"] = "unknown"


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
    # pylint: disable=too-many-instance-attributes
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
    semantic: bool = False


@dataclass
class InlineLayoutResult:
    expansions: tuple[InlineExpansionEvidence, ...] = ()
    accuracy_modulo_inline: float | None = None


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class HelperCatalogEntry:
    """Cached fingerprint for a paired helper usable as an inline needle."""

    orig_addr: int
    recomp_addr: int
    name: str
    fingerprint: Fingerprint  # epilog already stripped
    # The proof identity a call to the helper carries.
    identity: Hashable
    byte_size: int
    # How many helpers share this exact fingerprint across the catalog.
    # Used as an inverse-frequency confidence weight (1.0 = unique).
    uniqueness: float = 1.0
    effect_summary: HelperEffectSummary | None = None


def _registers(operand) -> list[str]:
    match operand:
        case ("reg", name):
            return [name]
        case ("mem", _, _, reg_terms, _, _):
            return [name for name, _scale in reg_terms]
    return []


def summarize_helper_effects(
    fingerprint: Fingerprint,
    *,
    max_length: int = 32,
) -> HelperEffectSummary | None:
    """Derive a cheap effect summary from a helper fingerprint: which of
    ecx/edx it reads, whether it returns in eax, and the stores it makes
    through ``this`` (ecx) or the stack at a constant displacement.

    Returns ``None`` when the helper is too long or cannot be summarized
    safely."""
    if not fingerprint or len(fingerprint) > max_length:
        return None
    if any(row.mnemonic in _UNSUPPORTED_FOR_SUMMARY for row in fingerprint):
        return None

    inputs: set[str] = set()
    stores: list[StoreEffect] = []
    return_kind: Literal["void", "register", "stack", "unknown"] = "void"

    for row in fingerprint:
        for operand in row.operands:
            inputs.update(
                name for name in _registers(operand) if name in ("ecx", "edx")
            )
        match row.mnemonic, row.operands:
            case mnemonic, (("reg", "eax"), _) if mnemonic.startswith("mov"):
                return_kind = "register"
            case mnemonic, (("mem", size, _, ((base, 1),), int() as disp, _), _) if (
                mnemonic in _STORE_MNEMONICS and size in _STORE_SIZES
            ):
                if base == "ecx":
                    stores.append(StoreEffect("this", disp))
                elif base in ("esp", "ebp"):
                    kind: Literal["arg", "stack"] = (
                        "arg" if base == "ebp" and disp >= 8 else "stack"
                    )
                    stores.append(StoreEffect(kind, disp))

    # Deduplicate while preserving order.
    unique_stores = list(dict.fromkeys(stores))
    if not unique_stores and return_kind == "void" and not inputs:
        return None

    return HelperEffectSummary(
        inputs=tuple(sorted(inputs)),
        stores=tuple(unique_stores),
        return_kind=return_kind,
    )


@dataclass(frozen=True)
class _Span:
    offset: int
    length: int
    helper_orig: int
    prefer_call: bool
    confidence: float
    call_index: int | None = None
    occurrence: int = 0  # which body/call pairing slot


def strip_helper_epilog(fingerprint: Fingerprint) -> Fingerprint:
    """Drop a trailing ret so the needle matches an inlined body."""
    if fingerprint and fingerprint[-1].mnemonic in _EPILOG_MNEMONICS:
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
    """Weighted interval selection: prefer call-backed, unique, longer spans.

    Uses the classic weighted-interval DP ordered by interval end.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.offset + s.length, s.offset))
    n = len(ordered)
    # p(i) = rightmost j < i that does not overlap i
    prev: list[int] = [-1] * n
    for i, span in enumerate(ordered):
        for j in range(i - 1, -1, -1):
            if ordered[j].offset + ordered[j].length <= span.offset:
                prev[i] = j
                break

    def weight(span: _Span) -> float:
        return span.confidence + (1.0 if span.prefer_call else 0.0) + span.length * 0.01

    best = [0.0] * n
    take = [False] * n
    for i, span in enumerate(ordered):
        alone = weight(span) + (best[prev[i]] if prev[i] >= 0 else 0.0)
        skip = best[i - 1] if i else 0.0
        if alone >= skip:
            best[i] = alone
            take[i] = True
        else:
            best[i] = skip

    chosen: list[_Span] = []
    i = n - 1
    while i >= 0:
        if take[i]:
            chosen.append(ordered[i])
            i = prev[i]
        else:
            i -= 1
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
    replace = dict(indices)
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


def _canon_reg_family(name: str, mapping: dict[str, str]) -> str | None:
    """A GP register as an occurrence-ordered placeholder (``R0.r32``)."""
    info = REGISTERS.get(name)
    if info is None:
        return None
    family, width = info
    if family not in mapping:
        mapping[family] = f"R{len(mapping)}"
    return f"{mapping[family]}.{width}"


def _rename_operand(operand, mapping: dict[str, str]):
    match operand:
        case ("reg", name) if (renamed := _canon_reg_family(name, mapping)) is not None:
            return ("reg", renamed)
        case ("mem", size, seg, reg_terms, disp, syms):
            terms = tuple(
                (_canon_reg_family(name, mapping) or name, scale)
                for name, scale in reg_terms
            )
            return ("mem", size, seg, terms, disp, syms)
    return operand


def register_normalized(fingerprint: Fingerprint) -> Fingerprint:
    """Rewrite GP registers to occurrence-ordered placeholders (R0.r32, …).

    Used as a second-level span key when the literal fingerprint misses.
    Relative register identity is preserved; absolute names are not.
    """
    mapping: dict[str, str] = {}
    return tuple(
        FingerprintRow(
            row.prefix,
            row.mnemonic,
            tuple(_rename_operand(op, mapping) for op in row.operands),
            row.callee,
        )
        for row in fingerprint
    )


def find_fingerprint_spans_tolerant(
    haystack: Fingerprint, needle: Fingerprint
) -> tuple[list[int], MatchKind]:
    """Literal span search, then per-window register-normalized fallback.

    Register slots are renumbered independently for each candidate window so
    surrounding host instructions do not shift placeholder indices.
    """
    starts = find_fingerprint_spans(haystack, needle)
    if starts:
        return starts, "literal"
    n = len(needle)
    if n == 0 or n > len(haystack):
        return [], "literal"
    norm_needle = register_normalized(needle)
    # Normalizing renames registers only: a window can match only where the
    # instructions themselves (prefix and mnemonic) already do.
    heads = [(row.prefix, row.mnemonic) for row in haystack]
    needle_heads = [(row.prefix, row.mnemonic) for row in needle]
    norm_starts = [
        start
        for start in range(len(haystack) - n + 1)
        if heads[start : start + n] == needle_heads
        and register_normalized(haystack[start : start + n]) == norm_needle
    ]
    if norm_starts:
        return norm_starts, "register"
    return [], "literal"


def _span_match_kind(
    host_fp: Fingerprint,
    needle: Fingerprint,
    span: tuple[int, int],
    *,
    helper_summary: HelperEffectSummary | None,
) -> MatchKind:
    """Classify how ``needle`` relates to ``host_fp[span]``."""
    offset, length = span
    host_slice = host_fp[offset : offset + length]
    if host_slice == needle:
        return "literal"
    if register_normalized(host_slice) == register_normalized(needle):
        return "register"
    if helper_summary is not None:
        host_summary = summarize_helper_effects(host_slice)
        if host_summary is not None and host_summary == helper_summary:
            return "summary"
    return "literal"


def find_call_sites(fingerprint: Fingerprint) -> list[tuple[int, Hashable]]:
    """``(index, callee identity)`` of every direct call in ``fingerprint``."""
    return [
        (i, row.callee)
        for i, row in enumerate(fingerprint)
        if row.mnemonic == "call" and row.callee is not None
    ]


def find_call_indices_for_helper(
    fingerprint: Fingerprint, helper: HelperCatalogEntry
) -> list[int]:
    """Indices of the calls to ``helper``, by the callee's identity."""
    return [
        index
        for index, callee in find_call_sites(fingerprint)
        if callee == helper.identity
    ]


def _inline_confidence(
    helper: HelperCatalogEntry, host_len: int, *, call_backed: bool
) -> float:
    """Confidence from uniqueness and relative size — not host_len alone."""
    if host_len <= 0 or not helper.fingerprint:
        return 0.0
    length_factor = min(1.0, len(helper.fingerprint) / 8.0)
    coverage = len(helper.fingerprint) / host_len
    score = helper.uniqueness * (0.5 * length_factor + 0.5 * coverage)
    if call_backed:
        score = min(1.0, score + 0.25)
    return round(score, 4)


def _evidence_confidence(
    helper: HelperCatalogEntry,
    host_fp: Fingerprint,
    span: tuple[int, int] | None,
    *,
    call_backed: bool,
    match_kind: MatchKind = "literal",
) -> tuple[float, bool]:
    """Confidence from uniqueness/size; semantic only for non-literal matches.

    ``semantic=True`` means the sequences differ at the literal fingerprint
    level but still match via register-normalized fingerprints or equal effect
    summaries. Exact literal fingerprint hits are never marked semantic —
    re-summarizing an identical needle is vacuous.
    """
    del span  # fingerprint identity already established by the caller
    confidence = _inline_confidence(helper, len(host_fp), call_backed=call_backed)
    semantic = match_kind in ("register", "summary")
    return confidence, semantic


def find_inline_expansions(
    helper_fingerprint: Fingerprint,
    hosts: Sequence[tuple[int, str, int]],
    host_fingerprint: FingerprintFn,
    *,
    min_helper_ops: int = 3,
) -> list[InlineHit]:
    """Search host functions for contiguous expansions of ``helper_fingerprint``."""
    needle = strip_helper_epilog(helper_fingerprint)
    if len(needle) < min_helper_ops:
        return []

    hits: list[InlineHit] = []
    helper_len = len(needle)
    for addr, name, size in hosts:
        host_fp = host_fingerprint(addr, size)
        if host_fp is None or len(host_fp) <= helper_len:
            continue
        for offset in find_fingerprint_spans(host_fp, needle):
            # uniqueness unknown here; length/coverage only
            confidence = min(1.0, helper_len / 8.0) * 0.5 + 0.5 * (
                helper_len / len(host_fp)
            )
            hits.append(
                InlineHit(
                    host_addr=addr,
                    host_name=name,
                    match_offset=offset,
                    match_length=helper_len,
                    confidence=round(confidence, 4),
                )
            )
    hits.sort(key=lambda h: (-h.confidence, h.host_addr, h.match_offset))
    return hits


@dataclass(frozen=True)
class _Pairing:
    helper: HelperCatalogEntry
    orig_span: tuple[int, int] | None
    recomp_span: tuple[int, int] | None
    orig_call: int | None
    recomp_call: int | None
    confidence: float
    match_kind: MatchKind = "literal"

    @property
    def kind(self) -> str:
        if self.orig_span and self.recomp_span:
            return "both"
        if self.orig_span and self.recomp_call is not None:
            return "orig_inline"
        if self.recomp_span and self.orig_call is not None:
            return "recomp_inline"
        return "none"


def analyze_inline_layout(
    orig_rows: Sequence[DecodedInstruction],
    recomp_rows: Sequence[DecodedInstruction],
    helpers: Sequence[HelperCatalogEntry],
    *,
    min_helper_ops: int = 3,
    exclude_orig_addrs: Sequence[int] = (),
) -> InlineLayoutResult:
    # pylint: disable=too-many-locals,too-many-statements
    """Detect CALL↔inline asymmetries; handle every occurrence of each helper."""
    orig_fp = fingerprint_of(orig_rows)
    recomp_fp = fingerprint_of(recomp_rows)
    if not orig_fp or not recomp_fp:
        return InlineLayoutResult()

    excluded = set(exclude_orig_addrs)
    pairings: list[_Pairing] = []

    for helper in helpers:
        if helper.orig_addr in excluded:
            continue
        needle = helper.fingerprint
        if len(needle) < min_helper_ops:
            continue
        if len(needle) >= max(len(orig_fp), len(recomp_fp)):
            continue

        orig_starts, orig_kind = find_fingerprint_spans_tolerant(orig_fp, needle)
        recomp_starts, recomp_kind = find_fingerprint_spans_tolerant(recomp_fp, needle)
        # Worst (most semantic) kind across sides for the body hits.
        body_kind: MatchKind = (
            "register"
            if "register" in (orig_kind, recomp_kind)
            else ("summary" if "summary" in (orig_kind, recomp_kind) else "literal")
        )
        orig_calls = find_call_indices_for_helper(orig_fp, helper)
        recomp_calls = find_call_indices_for_helper(recomp_fp, helper)

        both_n = min(len(orig_starts), len(recomp_starts))
        for i in range(both_n):
            pairings.append(
                _Pairing(
                    helper,
                    (orig_starts[i], len(needle)),
                    (recomp_starts[i], len(needle)),
                    None,
                    None,
                    _inline_confidence(helper, len(orig_fp), call_backed=False),
                    match_kind=body_kind,
                )
            )

        # Remaining orig bodies ↔ recomp CALLs
        remaining_orig = orig_starts[both_n:]
        for i, start in enumerate(remaining_orig):
            if i >= len(recomp_calls):
                break
            span = (start, len(needle))
            kind = _span_match_kind(
                orig_fp, needle, span, helper_summary=helper.effect_summary
            )
            pairings.append(
                _Pairing(
                    helper,
                    span,
                    None,
                    None,
                    recomp_calls[i],
                    _inline_confidence(helper, len(orig_fp), call_backed=True),
                    match_kind=kind,
                )
            )

        # Remaining recomp bodies ↔ orig CALLs
        remaining_recomp = recomp_starts[both_n:]
        for i, start in enumerate(remaining_recomp):
            if i >= len(orig_calls):
                break
            span = (start, len(needle))
            kind = _span_match_kind(
                recomp_fp, needle, span, helper_summary=helper.effect_summary
            )
            pairings.append(
                _Pairing(
                    helper,
                    None,
                    span,
                    orig_calls[i],
                    None,
                    _inline_confidence(helper, len(recomp_fp), call_backed=True),
                    match_kind=kind,
                )
            )

    if not pairings:
        return InlineLayoutResult()

    # Build occupancy spans for WIS on each side independently, then keep
    # pairings whose body spans survived (CALL-only side has no body span).
    orig_spans = [
        _Span(
            p.orig_span[0],
            p.orig_span[1],
            p.helper.orig_addr,
            prefer_call=p.recomp_call is not None,
            confidence=p.confidence,
            call_index=p.recomp_call,
        )
        for p in pairings
        if p.orig_span is not None
    ]
    recomp_spans = [
        _Span(
            p.recomp_span[0],
            p.recomp_span[1],
            p.helper.orig_addr,
            prefer_call=p.orig_call is not None,
            confidence=p.confidence,
            call_index=p.orig_call,
        )
        for p in pairings
        if p.recomp_span is not None
    ]
    kept_orig = {(s.helper_orig, s.offset) for s in select_nonoverlapping(orig_spans)}
    kept_recomp = {
        (s.helper_orig, s.offset) for s in select_nonoverlapping(recomp_spans)
    }

    expansions: list[InlineExpansionEvidence] = []
    orig_elide: list[tuple[int, int, Hashable]] = []
    recomp_elide: list[tuple[int, int, Hashable]] = []
    orig_collapse: list[tuple[int, Hashable]] = []
    recomp_collapse: list[tuple[int, Hashable]] = []
    used_orig_calls: set[int] = set()
    used_recomp_calls: set[int] = set()

    # Prefer call-backed pairings first, then both-inline.
    ordered = sorted(
        pairings,
        key=lambda p: (
            0 if p.kind in ("orig_inline", "recomp_inline") else 1,
            -p.confidence,
            -(p.orig_span or p.recomp_span or (0, 0))[1],
        ),
    )

    for pairing in ordered:
        helper = pairing.helper
        placeholder: Hashable = ("inline", helper.orig_addr)
        if pairing.kind == "both":
            assert pairing.orig_span and pairing.recomp_span
            if (helper.orig_addr, pairing.orig_span[0]) not in kept_orig:
                continue
            if (helper.orig_addr, pairing.recomp_span[0]) not in kept_recomp:
                continue
            # Mark consumed so a later CALL pairing cannot reuse these offsets.
            kept_orig.discard((helper.orig_addr, pairing.orig_span[0]))
            kept_recomp.discard((helper.orig_addr, pairing.recomp_span[0]))
            orig_elide.append((*pairing.orig_span, placeholder))
            recomp_elide.append((*pairing.recomp_span, placeholder))
            confidence, semantic = _evidence_confidence(
                helper,
                orig_fp,
                pairing.orig_span,
                call_backed=False,
                match_kind=pairing.match_kind,
            )
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="both",
                    match_offset=pairing.orig_span[0],
                    match_length=pairing.orig_span[1],
                    counterpart="inline",
                    counterpart_offset=pairing.recomp_span[0],
                    confidence=confidence,
                    semantic=semantic,
                )
            )
        elif pairing.kind == "orig_inline":
            assert pairing.orig_span and pairing.recomp_call is not None
            if (helper.orig_addr, pairing.orig_span[0]) not in kept_orig:
                continue
            if pairing.recomp_call in used_recomp_calls:
                continue
            kept_orig.discard((helper.orig_addr, pairing.orig_span[0]))
            used_recomp_calls.add(pairing.recomp_call)
            orig_elide.append((*pairing.orig_span, placeholder))
            recomp_collapse.append((pairing.recomp_call, placeholder))
            confidence, semantic = _evidence_confidence(
                helper,
                orig_fp,
                pairing.orig_span,
                call_backed=True,
                match_kind=pairing.match_kind,
            )
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="orig",
                    match_offset=pairing.orig_span[0],
                    match_length=pairing.orig_span[1],
                    counterpart="call",
                    counterpart_offset=pairing.recomp_call,
                    confidence=confidence,
                    semantic=semantic,
                )
            )
        elif pairing.kind == "recomp_inline":
            assert pairing.recomp_span and pairing.orig_call is not None
            if (helper.orig_addr, pairing.recomp_span[0]) not in kept_recomp:
                continue
            if pairing.orig_call in used_orig_calls:
                continue
            kept_recomp.discard((helper.orig_addr, pairing.recomp_span[0]))
            used_orig_calls.add(pairing.orig_call)
            recomp_elide.append((*pairing.recomp_span, placeholder))
            orig_collapse.append((pairing.orig_call, placeholder))
            confidence, semantic = _evidence_confidence(
                helper,
                recomp_fp,
                pairing.recomp_span,
                call_backed=True,
                match_kind=pairing.match_kind,
            )
            expansions.append(
                InlineExpansionEvidence(
                    helper_name=helper.name,
                    helper_orig_addr=helper.orig_addr,
                    helper_recomp_addr=helper.recomp_addr,
                    side="recomp",
                    match_offset=pairing.recomp_span[0],
                    match_length=pairing.recomp_span[1],
                    counterpart="call",
                    counterpart_offset=pairing.orig_call,
                    confidence=confidence,
                    semantic=semantic,
                )
            )

    if not expansions:
        return InlineLayoutResult()

    modulo = accuracy_after_inline_elision(
        list(orig_fp),
        list(recomp_fp),
        orig_elide=orig_elide,
        recomp_elide=recomp_elide,
        orig_collapse=orig_collapse,
        recomp_collapse=recomp_collapse,
    )
    return InlineLayoutResult(
        expansions=tuple(expansions),
        accuracy_modulo_inline=modulo,
    )
