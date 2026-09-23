"""Detect probable inline expansions of known helper bodies.

Given a helper fingerprint (normalized mnemonic/operand shape), search larger
functions for contiguous subsequences that match the helper body.  Hits are
diagnostic proposals — not automatic semantic proofs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Callable, Literal, Sequence, Union

from reccmp.compare.asm.ir import AsmRole, DecodedInstruction
from reccmp.compare.asm.model import (
    REGISTERS,
    Reject,
    format_operand,
    parse_instruction,
)
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins

Fingerprint = tuple[tuple[str, str], ...]
FingerprintFn = Callable[[int, int], Fingerprint | None]
Counterpart = Literal["call", "inline", "absent"]
InlineSide = Literal["orig", "recomp", "both"]
MatchKind = Literal["literal", "register", "summary"]
AsmInput = Union[Sequence[str], Sequence[DecodedInstruction]]

# Trailing opcodes that belong to a standalone helper epilog, not an inline site.
_EPILOG_MNEMONICS = frozenset({"ret", "retn", "retf"})
_STORE_MNEMONICS = frozenset(
    {"mov", "movzx", "movsx", "lea", "add", "sub", "or", "xor", "and", "xchg"}
)
# Dest-memory forms with ecx/this + optional displacement (thiscall helpers).
_THIS_STORE = re.compile(
    r"^(?:dword|word|byte|qword)\s+ptr\s+"
    r"\[(?:ecx)(?:\s*([+-])\s*(?:0x)?([0-9a-f]+))?\]\s*,",
    re.IGNORECASE,
)
_ARG_STORE = re.compile(
    r"^(?:dword|word|byte|qword)\s+ptr\s+"
    r"\[(?:esp|ebp)(?:\s*([+-])\s*(?:0x)?([0-9a-f]+))?\]\s*,",
    re.IGNORECASE,
)
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
class HelperCatalogEntry:
    """Cached fingerprint for a paired helper usable as an inline needle."""

    orig_addr: int
    recomp_addr: int
    name: str
    fingerprint: Fingerprint  # epilog already stripped
    byte_size: int
    # How many helpers share this exact fingerprint across the catalog.
    # Used as an inverse-frequency confidence weight (1.0 = unique).
    uniqueness: float = 1.0
    effect_summary: HelperEffectSummary | None = None


def summarize_helper_effects(
    fingerprint: Fingerprint,
    *,
    max_length: int = 32,
) -> HelperEffectSummary | None:
    """Derive a cheap effect summary from a helper fingerprint.

    Prefer short bodies without unsupported ops. Returns ``None`` when the
    helper is too long or cannot be summarized safely. Full symbolic
    ``execute()`` is optional future work; displacement-pattern parsing is the MVP.
    """
    if not fingerprint or len(fingerprint) > max_length:
        return None
    if any(mnemonic.lower() in _UNSUPPORTED_FOR_SUMMARY for mnemonic, _ in fingerprint):
        return None

    inputs: set[str] = set()
    stores: list[StoreEffect] = []
    return_kind: Literal["void", "register", "stack", "unknown"] = "void"

    for mnemonic, operand in fingerprint:
        lower = mnemonic.lower()
        operand_l = operand.lower()
        if "ecx" in operand_l:
            inputs.add("ecx")
        if "edx" in operand_l:
            inputs.add("edx")
        if "eax" in operand_l and lower.startswith("mov") and "," in operand_l:
            # mov eax, ... counts as producing a register return.
            dest = operand_l.split(",", 1)[0].strip()
            if dest == "eax":
                return_kind = "register"

        if lower not in _STORE_MNEMONICS:
            if lower in _EPILOG_MNEMONICS:
                continue
            continue

        this_match = _THIS_STORE.match(operand)
        if this_match and "," in operand:
            # Only count when the memory operand is the destination.
            dest = operand.split(",", 1)[0].strip().lower()
            if dest.startswith(("dword", "word", "byte", "qword")) and "ecx" in dest:
                sign, digits = this_match.group(1), this_match.group(2)
                disp = int(f"{sign or '+'}{digits or '0'}", 16)
                stores.append(StoreEffect("this", disp))
                inputs.add("ecx")
                continue

        arg_match = _ARG_STORE.match(operand)
        if arg_match and "," in operand:
            dest = operand.split(",", 1)[0].strip().lower()
            if dest.startswith(("dword", "word", "byte", "qword")):
                sign, digits = arg_match.group(1), arg_match.group(2)
                disp = int(f"{sign or '+'}{digits or '0'}", 16)
                base: Literal["arg", "stack"] = (
                    "arg" if "ebp" in dest and disp >= 8 else "stack"
                )
                stores.append(StoreEffect(base, disp))

    # Deduplicate while preserving order.
    unique_stores: list[StoreEffect] = []
    seen: set[tuple[str, int]] = set()
    for store in stores:
        key = (store.base, store.displacement)
        if key in seen:
            continue
        seen.add(key)
        unique_stores.append(store)

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


def call_operand_identities(operand: str) -> set[str]:
    """Canonical identity tokens extractable from a sanitized call operand."""
    text = operand.strip()
    # Drop trailing entity annotations: "Foo (FUNCTION)"
    text = re.sub(r"\s+\((?:DATA|STRING|FLOAT|FUNCTION|IMPORT)\)$", "", text)
    identities = {text}
    # Bare hex address
    if text.lower().startswith("0x"):
        try:
            identities.add(f"{int(text, 16):#x}")
        except ValueError:
            pass
    return identities


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


def asm_fingerprint_from_ir(rows: Sequence[DecodedInstruction]) -> Fingerprint:
    """Build a fingerprint from structured IR operands (not display splits).

    Table markers (``AsmRole`` non-CODE) are skipped. Operand text is rendered
    from the typed Capstone/sanitizer tuples via ``format_operand``.
    """
    result: list[tuple[str, str]] = []
    for row in rows:
        if row.role != AsmRole.CODE:
            continue
        mnemonic = (
            f"{row.prefix} {row.mnemonic}".strip() if row.prefix else row.mnemonic
        )
        if not mnemonic:
            continue
        if row.operands:
            try:
                operand = ", ".join(format_operand(op) for op in row.operands)
            except Reject:
                operand = ", ".join(row.raw_operands) if row.raw_operands else ""
        else:
            operand = ""
        result.append((mnemonic, operand))
    return tuple(result)


def fingerprint_from_asm(asm: AsmInput) -> Fingerprint:
    """Prefer IR fingerprints when ``DecodedInstruction`` rows are available."""
    if asm and isinstance(asm[0], DecodedInstruction):
        return asm_fingerprint_from_ir(asm)  # type: ignore[arg-type]
    return asm_fingerprint_from_lines(asm)  # type: ignore[arg-type]


def _canon_reg_family(name: str, mapping: dict[str, str]) -> tuple[str, str] | None:
    """Map a GP register to a stable ``(placeholder, width)`` slot, or None."""
    info = REGISTERS.get(name)
    if info is None:
        return None
    family, width = info
    if family not in mapping:
        mapping[family] = f"R{len(mapping)}"
    return mapping[family], width


def _rename_structured_operand(operand, mapping: dict[str, str]):
    if not isinstance(operand, tuple) or not operand:
        return operand
    kind = operand[0]
    if kind == "reg":
        renamed = _canon_reg_family(operand[1], mapping)
        if renamed is None:
            return operand
        placeholder, width = renamed
        return ("sym", f"{placeholder}.{width}")
    if kind == "mem":
        size, seg, reg_terms, disp, syms = (
            operand[1],
            operand[2],
            operand[3],
            operand[4],
            operand[5],
        )
        new_terms = []
        for reg, scale in reg_terms:
            renamed = _canon_reg_family(reg, mapping)
            if renamed is None:
                new_terms.append((reg, scale))
            else:
                placeholder, width = renamed
                new_terms.append((f"{placeholder}.{width}", scale))
        return ("mem", size, seg, tuple(new_terms), disp, syms)
    return operand


def register_normalized_fingerprint(fingerprint: Fingerprint) -> Fingerprint:
    """Rewrite GP registers to occurrence-ordered placeholders (R0.r32, …).

    Used as a second-level span key when the literal IR fingerprint misses.
    Relative register identity is preserved; absolute names are not.
    """
    mapping: dict[str, str] = {}
    result: list[tuple[str, str]] = []
    for mnemonic, operand in fingerprint:
        line = f"{mnemonic} {operand}".rstrip() if operand else mnemonic
        try:
            ins = parse_instruction(line)
        except (Reject, IndexError, KeyError, ValueError, TypeError):
            result.append((mnemonic, operand))
            continue
        new_ops = tuple(_rename_structured_operand(op, mapping) for op in ins.operands)
        head = f"{ins.prefix} {ins.mnemonic}".strip() if ins.prefix else ins.mnemonic
        try:
            op_text = ", ".join(format_operand(op) for op in new_ops) if new_ops else ""
        except Reject:
            op_text = operand
        result.append((head, op_text))
    return tuple(result)


def register_normalized_from_ir(rows: Sequence[DecodedInstruction]) -> Fingerprint:
    """Register-normalize directly from IR operands (no display reparse)."""
    mapping: dict[str, str] = {}
    result: list[tuple[str, str]] = []
    for row in rows:
        if row.role != AsmRole.CODE:
            continue
        head = f"{row.prefix} {row.mnemonic}".strip() if row.prefix else row.mnemonic
        if not head:
            continue
        new_ops = tuple(_rename_structured_operand(op, mapping) for op in row.operands)
        try:
            op_text = ", ".join(format_operand(op) for op in new_ops) if new_ops else ""
        except Reject:
            op_text = ", ".join(row.raw_operands) if row.raw_operands else ""
        result.append((head, op_text))
    return tuple(result)


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
    norm_needle = register_normalized_fingerprint(needle)
    norm_starts: list[int] = []
    for start in range(len(haystack) - n + 1):
        window = haystack[start : start + n]
        if register_normalized_fingerprint(window) == norm_needle:
            norm_starts.append(start)
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
    if register_normalized_fingerprint(host_slice) == register_normalized_fingerprint(
        needle
    ):
        return "register"
    if helper_summary is not None:
        host_summary = summarize_helper_effects(host_slice)
        if host_summary is not None and host_summary == helper_summary:
            return "summary"
    return "literal"


def find_call_sites(
    fingerprint: Fingerprint,
) -> list[tuple[int, set[str]]]:
    """Return ``(index, identity_set)`` for every call in ``fingerprint``."""
    sites: list[tuple[int, set[str]]] = []
    for i, (mnemonic, operand) in enumerate(fingerprint):
        if mnemonic == "call":
            sites.append((i, call_operand_identities(operand)))
    return sites


def find_call_indices_for_helper(
    fingerprint: Fingerprint, helper: HelperCatalogEntry
) -> list[int]:
    """Call indices that resolve to ``helper`` by name or address identity."""
    wanted = {
        helper.name,
        f"{helper.orig_addr:#x}",
        f"{helper.recomp_addr:#x}",
        f"{helper.orig_addr:x}",
        f"{helper.recomp_addr:x}",
    }
    # Also accept demangled/decorated substrings via exact identity match only.
    indices: list[int] = []
    for index, identities in find_call_sites(fingerprint):
        if identities & wanted:
            indices.append(index)
            continue
        # Substring match on the rendered name as a last resort for decorated
        # forms that still contain the helper's best_name.
        if any(helper.name and helper.name in identity for identity in identities):
            indices.append(index)
    return indices


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
        identities = call_operand_identities(operand)
        for name in needles:
            if name in identities or any(name in identity for identity in identities):
                indices.append(i)
                break
    return indices


def find_call_indices(lines: Sequence[str], helper_names: Sequence[str]) -> list[int]:
    """Indices of ``call`` instructions whose operand mentions a helper name."""
    return find_call_indices_in_fingerprint(
        asm_fingerprint_from_lines(lines), helper_names
    )


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
    fingerprint_of: FingerprintFn,
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
        host_fp = fingerprint_of(addr, size)
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
    orig_asm: AsmInput,
    recomp_asm: AsmInput,
    helpers: Sequence[HelperCatalogEntry],
    *,
    min_helper_ops: int = 3,
    exclude_orig_addrs: Sequence[int] = (),
) -> InlineLayoutResult:
    # pylint: disable=too-many-locals,too-many-statements
    """Detect CALL↔inline asymmetries; handle every occurrence of each helper.

    Accepts sanitized display lines or ``DecodedInstruction`` excerpts; IR is
    preferred when available.
    """
    orig_fp = fingerprint_from_asm(orig_asm)
    recomp_fp = fingerprint_from_asm(recomp_asm)
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
