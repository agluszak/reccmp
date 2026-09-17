"""Detect probable inline expansions of known helper bodies.

Given a helper fingerprint (normalized mnemonic/operand shape), search larger
functions for contiguous subsequences that match the helper body.  Hits are
diagnostic proposals — not automatic matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


Fingerprint = tuple[tuple[str, str], ...]
FingerprintFn = Callable[[int, int], Fingerprint | None]


@dataclass(frozen=True)
class InlineHit:
    """One probable expansion of a helper inside a larger function."""

    host_addr: int
    host_name: str
    match_offset: int  # instruction index within the host body
    match_length: int
    confidence: float


def _subsequence_index(
    haystack: Fingerprint, needle: Fingerprint
) -> int | None:
    """Return the first index where ``needle`` appears contiguously in ``haystack``."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    for start in range(len(haystack) - n + 1):
        if haystack[start : start + n] == needle:
            return start
    return None


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
    if len(helper_fingerprint) < min_helper_ops:
        return []

    hits: list[InlineHit] = []
    helper_len = len(helper_fingerprint)
    for addr, name, size in hosts:
        host_fp = fingerprint_of(addr, size)
        if host_fp is None or len(host_fp) <= helper_len:
            continue
        offset = _subsequence_index(host_fp, helper_fingerprint)
        if offset is None:
            continue
        # Confidence scales with how much of the host the helper occupies.
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
