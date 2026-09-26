"""Cluster comparison verdicts to find the verifier's repeated shortcomings.

The most useful signal is a candidate mismatch that the witness executed
through the reported difference without the two sides diverging: the
verifier said "different" at an instruction pair that behaved the same on
every input tried. Many such cases share one shape (a comparison spelled two
ways, a call through an import thunk, the same value at two widths), and one
verifier change fixes the whole cluster.

Input is the JSON report of ``reccmp-reccmp --json`` (with ``--witness`` for
the execution buckets); instruction shapes need the two binaries.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from reccmp.compare.asm.decode import decode_one

# Buckets, most useful first.
AGREED_THROUGH_DIFFERENCE = "candidate: witness agreed through the difference"
AGREED_THROUGH_BLOCKER = "inconclusive: witness agreed through the blocker"
NEVER_REACHED = "candidate: witness never reached the difference"
NOT_EXECUTED = "candidate: not executed"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"
BUCKET_ORDER = (
    AGREED_THROUGH_DIFFERENCE,
    AGREED_THROUGH_BLOCKER,
    NEVER_REACHED,
    NOT_EXECUTED,
    REFUTED,
    INCONCLUSIVE,
)

Shape = Callable[[str, int], str | None]  # (image, address) -> shape


@dataclass(frozen=True)
class TriageKey:
    """What one cluster has in common."""

    bucket: str
    kind: str  # difference kind or inconclusive reason
    strategy: str  # the verifier strategy that reported it
    orig: str  # instruction shape, or the summary shape of a fact
    recomp: str
    detail: str = ""  # callee class, source facts, ...


@dataclass
class TriageCluster:
    key: TriageKey
    samples: list[tuple[str, str]] = field(default_factory=list)  # (address, name)

    @property
    def count(self) -> int:
        return len(self.samples)


def bucket_of(comparison: Mapping[str, Any]) -> str | None:
    status = comparison.get("status")
    execution = comparison.get("execution") or {}
    reached = execution.get("reached_location") or 0
    if status == "mismatch":
        if comparison.get("witness"):
            return REFUTED
        if not execution:
            return NOT_EXECUTED
        return AGREED_THROUGH_DIFFERENCE if reached else NEVER_REACHED
    if status == "inconclusive":
        return AGREED_THROUGH_BLOCKER if reached else INCONCLUSIVE
    return None


_OPERAND_KIND = {1: "reg", 2: "imm", 3: "mem"}  # capstone x86 operand types


def instruction_shape(code: bytes, address: int) -> str | None:
    """`cmp reg, imm`, `jbe imm`, `mov mem, reg`: a mnemonic and the kinds of
    its operands."""
    insn = decode_one(code, address)
    if insn is None:
        return None
    kinds = [_OPERAND_KIND.get(op.type, "?") for op in insn.operands]
    return f"{insn.mnemonic} {', '.join(kinds)}".strip()


_SUMMARY_HEAD = re.compile(r"^[('\s]*([A-Za-z_][A-Za-z_0-9]*)")


def _fact_shape(facts: Mapping[str, Any]) -> str:
    """The leading tags of a side's facts: `lt_u` for a predicate, `load`
    for a value, `IMPORT_THUNK` for a callee."""
    parts = []
    for name in ("predicate", "value", "register"):
        value = facts.get(name)
        if isinstance(value, str):
            head = _SUMMARY_HEAD.match(value)
            parts.append(f"{name}={head.group(1) if head else '?'}")
    target = facts.get("target_name")
    if isinstance(target, str):
        callee = re.search(
            r"\((IMPORT_THUNK|IMPORT|FUNCTION|THUNK|VTORDISP|UNK)\)", target
        )
        parts.append(f"callee={callee.group(1) if callee else 'other'}")
        if target.startswith("dword ptr ["):
            parts.append("indirect")
    return " ".join(parts)


def _side(side: Mapping[str, Any] | None, image: str, shape: Shape | None) -> str:
    if not side:
        return "-"
    address = side.get("address")
    decoded = (
        shape(image, address)
        if shape is not None and isinstance(address, int)
        else None
    )
    facts = _fact_shape(side.get("facts") or {})
    return " | ".join(part for part in (decoded, facts) if part) or "-"


def _strategy(comparison: Mapping[str, Any]) -> str:
    """The strategy whose attempt reported the final difference or blocker."""
    final = comparison.get("difference") or comparison.get("inconclusive_location")
    for attempt in comparison.get("attempts") or ():
        reported = attempt.get("difference") or attempt.get("location")
        if reported == final:
            return str(attempt.get("strategy"))
    return "-"


def _detail(comparison: Mapping[str, Any]) -> str:
    difference = comparison.get("difference") or {}
    recomp = (difference.get("recomp") or {}).get("facts") or {}
    details = []
    if "field_name" in recomp or "field_name" in (
        (difference.get("orig") or {}).get("facts") or {}
    ):
        details.append("field facts")
    if "source_comparisons" in recomp:
        details.append("source comparisons")
    return ", ".join(details)


def triage_key(
    comparison: Mapping[str, Any], shape: Shape | None = None
) -> TriageKey | None:
    bucket = bucket_of(comparison)
    if bucket is None:
        return None
    difference = comparison.get("difference")
    if difference is not None:
        kind = str(difference.get("kind"))
        orig = _side(difference.get("orig"), "orig", shape)
        recomp = _side(difference.get("recomp"), "recomp", shape)
    else:
        kind = str(comparison.get("inconclusive_reason"))
        location = comparison.get("inconclusive_location")
        image = (location or {}).get("image") or "orig"
        orig = _side(location, image, shape) if image == "orig" else "-"
        recomp = _side(location, image, shape) if image == "recomp" else "-"
    return TriageKey(
        bucket, kind, _strategy(comparison), orig, recomp, _detail(comparison)
    )


def triage(
    entities: Iterable[Mapping[str, Any]], shape: Shape | None = None
) -> list[TriageCluster]:
    """Clusters, by bucket (most useful first), then by size."""
    clusters: dict[TriageKey, TriageCluster] = {}
    for entity in entities:
        comparison = entity.get("comparison") or {}
        key = triage_key(comparison, shape)
        if key is None:
            continue
        cluster = clusters.setdefault(key, TriageCluster(key))
        cluster.samples.append((str(entity.get("address")), str(entity.get("name"))))
    return sorted(
        clusters.values(),
        key=lambda item: (
            BUCKET_ORDER.index(item.key.bucket),
            -item.count,
            repr(item.key),
        ),
    )


def bucket_counts(clusters: Iterable[TriageCluster]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        counts[cluster.key.bucket] += cluster.count
    return {bucket: counts[bucket] for bucket in BUCKET_ORDER if counts[bucket]}


def triage_text(
    clusters: list[TriageCluster], *, limit: int = 20, samples: int = 3
) -> str:
    lines = []
    for bucket, count in bucket_counts(clusters).items():
        lines.append(f"{bucket}: {count}")
    current = None
    shown = 0
    for cluster in clusters:
        key = cluster.key
        if key.bucket != current:
            current, shown = key.bucket, 0
            lines.append(f"\n== {key.bucket}")
        if shown >= limit:
            continue
        shown += 1
        lines.append(f"{cluster.count:5}  {key.kind} [{key.strategy}]")
        lines.append(f"         orig:   {key.orig}")
        lines.append(f"         recomp: {key.recomp}")
        if key.detail:
            lines.append(f"         {key.detail}")
        for address, name in cluster.samples[:samples]:
            lines.append(f"         e.g. {address} {name}")
    return "\n".join(lines)
