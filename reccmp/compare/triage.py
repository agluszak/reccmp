"""Cluster comparison verdicts to find the verifier's repeated shortcomings.

The most useful signal is a candidate mismatch that the witness executed
through the reported difference without the two sides diverging: the
verifier said "different" at an instruction pair that behaved the same on
every input tried. Many such cases share one shape (a comparison spelled two
ways, a call through an import thunk, the same value at two widths), and one
verifier change fixes the whole cluster.

Input is a deserialized ``reccmp-reccmp --json`` report (with ``--witness``
for the execution buckets); instruction shapes need the two binaries.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.operand import Mem, Operand, SignedSymbol, Sym
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DifferenceKind,
    DifferenceSide,
    InconclusiveReason,
    StopDetail,
    StopLocation,
    Strategy,
    StrategyAttempt,
)
from reccmp.compare.report import ReccmpComparedEntity
from reccmp.types import ImageId


class Bucket(Enum):
    """What differential execution said about a verdict, most useful first."""

    AGREED_THROUGH_DIFFERENCE = "candidate: witness agreed through the difference"
    AGREED_THROUGH_BLOCKER = "inconclusive: witness agreed through the blocker"
    NEVER_REACHED = "candidate: witness never reached the difference"
    NOT_EXECUTED = "candidate: not executed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class InstructionShape:
    """`cmp reg, imm`: a mnemonic and the kinds of its operands."""

    mnemonic: str
    operands: tuple[type[Operand], ...]


# (image, address) -> the shape of the instruction there
Shape = Callable[[ImageId, int], InstructionShape | None]


@dataclass(frozen=True)
class SideShape:
    """What one side of a cluster has in common."""

    instruction: InstructionShape | None = None
    register: str | None = None
    callee_type: str | None = None
    indirect: bool = False


@dataclass(frozen=True)
class ProductStop:
    """Where the product under a guessed block pairing stopped."""

    difference: DifferenceKind | None = None
    blocker: InconclusiveReason | None = None
    detail: StopDetail | None = None


@dataclass(frozen=True)
class _SourceShape:
    field: bool = False
    comparisons: bool = False


@dataclass(frozen=True)
class TriageKey:
    """What one cluster has in common."""

    bucket: Bucket
    verdict: DifferenceKind | InconclusiveReason
    strategy: Strategy | None  # the verifier strategy that reported it
    orig: SideShape | None
    recomp: SideShape | None
    source: _SourceShape = _SourceShape()
    product: ProductStop | None = None


@dataclass
class TriageCluster:
    key: TriageKey
    samples: list[ReccmpComparedEntity] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)


def bucket_of(analysis: ComparisonAnalysis) -> Bucket | None:
    execution = analysis.execution
    reached = bool(execution is not None and execution.reached_location)
    match analysis.status:
        case ComparisonStatus.MISMATCH if analysis.witness is not None:
            return Bucket.REFUTED
        case ComparisonStatus.MISMATCH if execution is None:
            return Bucket.NOT_EXECUTED
        case ComparisonStatus.MISMATCH:
            return Bucket.AGREED_THROUGH_DIFFERENCE if reached else Bucket.NEVER_REACHED
        case ComparisonStatus.INCONCLUSIVE:
            return Bucket.AGREED_THROUGH_BLOCKER if reached else Bucket.INCONCLUSIVE
    return None


def instruction_shape(code: bytes, address: int) -> InstructionShape | None:
    """The shape of the instruction at the start of ``code``."""
    rows = disasm_detail(code[:16], address)
    if not rows:
        return None
    return InstructionShape(
        rows[0].mnemonic, tuple(type(operand) for operand in rows[0].operands)
    )


def _callee_type(operand: Operand | None) -> str | None:
    match operand:
        case Sym(ref) | Mem(symbols=(SignedSymbol(1, ref),)):
            return ref.entity_type
    return None


def _side_shape(side: DifferenceSide | StopLocation, shape: Shape | None) -> SideShape:
    instruction = (
        shape(side.image, side.address)
        if shape is not None and side.address is not None
        else None
    )
    if isinstance(side, StopLocation):
        return SideShape(instruction)
    operand = side.observed.operand
    return SideShape(
        instruction,
        side.observed.register,
        _callee_type(operand),
        isinstance(operand, Mem),
    )


def _reporting_strategy(analysis: ComparisonAnalysis) -> Strategy | None:
    """The strategy whose attempt reported the final difference or blocker."""
    for attempt in analysis.attempts:
        if (
            attempt.difference is not None and attempt.difference == analysis.difference
        ) or (
            attempt.location is not None
            and attempt.location == analysis.inconclusive_location
        ):
            return attempt.strategy
    return None


def _product_stop(attempts: Iterable[StrategyAttempt]) -> ProductStop | None:
    for attempt in attempts:
        if attempt.strategy is not Strategy.UNANCHORED_PRODUCT:
            continue
        if attempt.difference is not None:
            return ProductStop(difference=attempt.difference.kind)
        return ProductStop(
            blocker=attempt.blocker,
            detail=attempt.location.detail if attempt.location is not None else None,
        )
    return None


def triage_key(
    analysis: ComparisonAnalysis, shape: Shape | None = None
) -> TriageKey | None:
    bucket = bucket_of(analysis)
    if bucket is None:
        return None
    strategy = _reporting_strategy(analysis)
    product = _product_stop(analysis.attempts)
    difference = analysis.difference
    if difference is not None:
        return TriageKey(
            bucket,
            difference.kind,
            strategy,
            _side_shape(difference.orig, shape),
            _side_shape(difference.recomp, shape),
            source=_SourceShape(
                field=difference.orig.field is not None
                or difference.recomp.field is not None,
                comparisons=bool(difference.recomp.source_comparisons),
            ),
            product=product,
        )
    assert analysis.inconclusive_reason is not None
    location = analysis.inconclusive_location
    located = _side_shape(location, shape) if location is not None else None
    on_recomp = location is not None and location.image is ImageId.RECOMP
    return TriageKey(
        bucket,
        analysis.inconclusive_reason,
        strategy,
        None if on_recomp else located,
        located if on_recomp else None,
        product=product,
    )


def triage(
    entities: Iterable[ReccmpComparedEntity], shape: Shape | None = None
) -> list[TriageCluster]:
    """Clusters, by bucket (most useful first), then by size."""
    clusters: dict[TriageKey, TriageCluster] = {}
    for entity in entities:
        key = triage_key(entity.analysis, shape)
        if key is None:
            continue
        clusters.setdefault(key, TriageCluster(key)).samples.append(entity)
    order = list(Bucket)
    return sorted(
        clusters.values(),
        key=lambda item: (order.index(item.key.bucket), -item.count, repr(item.key)),
    )


def bucket_counts(clusters: Iterable[TriageCluster]) -> dict[Bucket, int]:
    counts: dict[Bucket, int] = defaultdict(int)
    for cluster in clusters:
        counts[cluster.key.bucket] += cluster.count
    return {bucket: counts[bucket] for bucket in Bucket if counts[bucket]}


def _instruction_text(instruction: InstructionShape | None) -> str:
    if instruction is None:
        return "-"
    kinds = ", ".join(kind.__name__.lower() for kind in instruction.operands)
    return f"{instruction.mnemonic} {kinds}".strip()


def side_text(side: SideShape | None) -> str:
    if side is None:
        return "-"
    parts = [_instruction_text(side.instruction)]
    if side.register is not None:
        parts.append(f"register={side.register}")
    if side.callee_type is not None:
        parts.append(f"callee={side.callee_type}")
    if side.indirect:
        parts.append("indirect")
    return " | ".join(parts)


def details_text(key: TriageKey) -> str:
    details = []
    if key.source.field:
        details.append("field facts")
    if key.source.comparisons:
        details.append("source comparisons")
    match key.product:
        case ProductStop(difference=DifferenceKind() as kind):
            details.append(f"product: {kind.value}")
        case ProductStop(blocker=InconclusiveReason() as blocker, detail=detail):
            stage = f" at {detail.value}" if detail is not None else ""
            details.append(f"product: {blocker.value}{stage}")
    return ", ".join(details)


def triage_text(
    clusters: list[TriageCluster], *, limit: int = 20, samples: int = 3
) -> str:
    lines = [
        f"{bucket.value}: {count}" for bucket, count in bucket_counts(clusters).items()
    ]
    current = None
    shown = 0
    for cluster in clusters:
        key = cluster.key
        if key.bucket is not current:
            current, shown = key.bucket, 0
            lines.append(f"\n== {key.bucket.value}")
        if shown >= limit:
            continue
        shown += 1
        strategy = key.strategy.value if key.strategy is not None else "-"
        lines.append(f"{cluster.count:5}  {key.verdict.value} [{strategy}]")
        lines.append(f"         orig:   {side_text(key.orig)}")
        lines.append(f"         recomp: {side_text(key.recomp)}")
        details = details_text(key)
        if details:
            lines.append(f"         {details}")
        for entity in cluster.samples[:samples]:
            lines.append(f"         e.g. {entity.orig_addr:#x} {entity.name}")
    return "\n".join(lines)
