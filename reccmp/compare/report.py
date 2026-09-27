from datetime import datetime
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import from_json

from reccmp.compare.diagnosis import InconclusiveReason
from reccmp.types import EntityType

from .comparison_json import (
    analysis_json,
    parse_analysis,
    parse_diagnostic_normalizations,
    parse_inline_expansions,
    parse_stack_permutation,
)
from .diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DiagnosticNormalization,
    StackPermutationEntry,
    derive_diagnostic_normalizations,
)
from .diff import (
    CombinedDiffOutput,
    RawDiffOutput,
    raw_diff_to_udiff,
)
from .inlines import InlineExpansionEvidence

if TYPE_CHECKING:
    from .stack_layout import StackLayoutResult


def format_address(addr: int) -> str:
    """This is here just to document each spot where we
    convert an int address into a string.
    In the future, the format may be customizable. (GH #370)"""
    return f"{addr:#x}"


class ReccmpReportDeserializeError(Exception):
    """The given file is not a serialized reccmp report file"""


class ReccmpReportSameSourceError(Exception):
    """Tried to aggregate reports derived from different source files."""


@dataclass
class ReccmpComparedEntity:
    # pylint:disable=too-many-instance-attributes
    orig_addr: int
    name: str
    accuracy: float
    type: EntityType = EntityType.FUNCTION
    recomp_addr: int | None = None
    """The meaning of `None` depends on `recomp_addr_varies`:
    recomp_addr_varies is False: This entity is unmatched.
    recomp_addr_varies is True:  This entity has no fixed recomp addr."""

    analysis: ComparisonAnalysis = ComparisonAnalysis.inconclusive(
        InconclusiveReason.ANALYSIS_LIMIT
    )
    is_stub: bool = False
    is_library: bool = False
    rdiff: RawDiffOutput | None = None
    report_diff: CombinedDiffOutput | None = None

    recomp_addr_varies: bool = False
    """True if this entity had no fixed recomp address across the
    samples combined by reccmp-aggregate."""

    display_similarity: float | None = None
    stack_permutation: tuple[StackPermutationEntry, ...] = ()
    accuracy_modulo_stack: float | None = None
    inline_expansions: tuple[InlineExpansionEvidence, ...] = ()
    accuracy_modulo_inline: float | None = None
    diagnostic_normalizations: tuple[DiagnosticNormalization, ...] = ()
    # The stack slots the paired instructions use (not serialized).
    stack_layout: "StackLayoutResult | None" = None

    def is_matched(self) -> bool:
        return self.recomp_addr is not None or self.recomp_addr_varies

    def is_function(self) -> bool:
        return self.type == EntityType.FUNCTION

    @property
    def is_effective_match(self) -> bool:
        return self.analysis.status == ComparisonStatus.EFFECTIVE

    @property
    def is_proven_match(self) -> bool:
        return self.analysis.status in (
            ComparisonStatus.EXACT,
            ComparisonStatus.EFFECTIVE,
        )

    @property
    def effective_accuracy(self) -> float:
        return 1.0 if self.is_effective_match else self.accuracy

    def refresh_diagnostic_normalizations(self) -> None:
        self.diagnostic_normalizations = derive_diagnostic_normalizations(
            self.analysis,
            accuracy_modulo_stack=self.accuracy_modulo_stack,
            accuracy_modulo_inline=self.accuracy_modulo_inline,
        )


class ReccmpStatusReport:
    filename: str
    """The filename of the original binary.
    This is here to avoid comparing reports derived from different files.
    TODO: in the future, we may want to use the hash instead"""

    timestamp: datetime
    """Creation date of the report file."""

    entities: dict[int, ReccmpComparedEntity]
    """Using orig addr as the key."""

    source_digest: str | None = None
    """SHA-256 of the original binary, when known. Two reports with
    different digests are not aggregate-compatible even if they share a
    filename."""

    function_count: int = 0
    """Function count used to determine progress percentage and other statistics.
    We can compute this value from the report's entities or use a user-provided value.
    We will use whichever is higher so progress cannot exceed 100%."""

    def __init__(
        self,
        filename: str,
        timestamp: datetime | None = None,
        source_digest: str | None = None,
    ) -> None:
        self.filename = filename
        self.source_digest = source_digest
        self.function_count = 0
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().replace(microsecond=0)

        self.entities = {}

    def add_match(self, match: ReccmpComparedEntity):
        self.entities[match.orig_addr] = match

    def has_same_source(self, other: "ReccmpStatusReport") -> bool:
        """Were both reports derived from the same original binary?"""
        if self.source_digest is not None or other.source_digest is not None:
            return (
                self.source_digest is not None
                and self.source_digest == other.source_digest
            )
        return self.filename.lower() == other.filename.lower()

    def update_function_count(self) -> None:
        counted_type = sum(1 for ent in self.entities.values() if ent.is_function())
        self.function_count = max(self.function_count, counted_type)

    def filter_entities(
        self, filter_fn: Callable[[ReccmpComparedEntity], bool]
    ) -> None:
        """Delete entities that return False from the provided filter function."""
        # Set the count in case it has never been set.
        self.update_function_count()

        discarded = [
            key for key, value in self.entities.items() if not filter_fn(value)
        ]

        # Only functions contribute to function_count.
        functions_removed = sum(
            1 for key in discarded if self.entities[key].is_function()
        )

        for key in discarded:
            del self.entities[key]

        # Manually decrease it because recalculating will use
        # the higher of either the previous or current count.
        self.function_count -= functions_removed

    def asmcmp_filtering(self, nolib: bool, ignore_functions: list[str]) -> None:
        """Helper to filter the report using the current filter options from `reccmp-reccmp`.
        Compare with `entity_filter` in asmcmp.py that acts on the `ReccmpEntity` object.
        """

        def entity_filter(entity: ReccmpComparedEntity) -> bool:
            if entity.is_function() and entity.name in ignore_functions:
                return False

            if nolib and entity.is_library:
                return False

            return True

        self.filter_entities(entity_filter)


def report_function_alignment(report: ReccmpStatusReport) -> int:
    """Report the count of all (non-contiguous) functions where
    the address is the same in both binaries."""
    count = 0
    for ent in report.entities.values():
        if ent.is_function() and ent.orig_addr == ent.recomp_addr:
            count += 1

    return count


def report_function_accuracy(report: ReccmpStatusReport) -> tuple[int, float, float]:
    """Collects the accuracy and effective accuracy of all compared functions in the report.
    Returns (implemented_count, total_accuracy, total_effective_accuracy).
    Stubs are not compared so they are excluded.
    The accuracy scores are raw score values. Divide by the implemented_count to get the percentage.
    """
    implemented_count = 0
    total_accuracy = 0.0
    total_effective_accuracy = 0.0

    for ent in report.entities.values():
        if ent.is_function() and ent.is_matched() and not ent.is_stub:
            implemented_count += 1
            total_accuracy += ent.accuracy
            total_effective_accuracy += ent.effective_accuracy

    return (implemented_count, total_accuracy, total_effective_accuracy)


def _get_entity_for_addr(
    samples: Iterable[ReccmpStatusReport], addr: int
) -> Iterator[ReccmpComparedEntity]:
    """Helper to return entities from xreports that have the given address."""
    for sample in samples:
        if addr in sample.entities:
            yield sample.entities[addr]


def _accuracy_sort_key(entity: ReccmpComparedEntity) -> float:
    """Helper to sort entity samples by accuracy score.
    Proven exact match is preferred over effective.
    Effective match is preferred over any unproven accuracy.
    Stubs rank lower than any accuracy score."""
    if entity.is_stub:
        return -1.0

    if entity.analysis.status == ComparisonStatus.EXACT:
        return 1000.0

    if entity.is_effective_match:
        return 1.0

    return entity.accuracy


def combine_reports(samples: list[ReccmpStatusReport]) -> ReccmpStatusReport:
    """Combines the sample reports into a single report.
    The current strategy is to use the entity with the highest
    accuracy score from any report."""
    assert len(samples) > 0

    if not all(samples[0].has_same_source(s) for s in samples):
        raise ReccmpReportSameSourceError

    output = ReccmpStatusReport(
        filename=samples[0].filename, source_digest=samples[0].source_digest
    )

    # Use the highest function total across all samples.
    # Some functions may have been inlined in some reports.
    output.function_count = max(sample.function_count for sample in samples)

    # Combine every orig addr used in any of the reports.
    orig_addr_set = {key for sample in samples for key in sample.entities.keys()}

    all_orig_addrs = sorted(list(orig_addr_set))

    for addr in all_orig_addrs:
        e_list = list(_get_entity_for_addr(samples, addr))
        assert len(e_list) > 0

        # Our aggregate accuracy score is the highest from any report.
        e_list.sort(key=_accuracy_sort_key, reverse=True)

        chosen = replace(e_list[0])
        output.entities[addr] = chosen

        # Keep the recomp_addr if it is the same across all samples.
        # i.e. to detect where function alignment ends
        if not all(e_list[0].recomp_addr == e.recomp_addr for e in e_list):
            output.entities[addr] = replace(
                chosen, recomp_addr=None, recomp_addr_varies=True
            )

    # Recalculate the count against the functions we actually have.
    # This may be higher than the count from any one sample.
    output.update_function_count()

    return output


def _render_entity_diff(entity: ReccmpComparedEntity) -> CombinedDiffOutput | None:
    """Use a stored report diff or render the raw comparison diff."""
    if entity.report_diff is not None:
        return entity.report_diff

    if entity.rdiff is None:
        # We need data to create the unified diff.
        return None

    if entity.type == EntityType.VTABLE:
        # Complete diff is always shown for vtables, even if they match.
        return raw_diff_to_udiff(entity.rdiff, grouped=False)

    if entity.is_effective_match or entity.accuracy != 1.0:
        # Show grouped diff for effective match.
        return raw_diff_to_udiff(entity.rdiff, grouped=True)

    # Display nothing for matching functions.
    return None


#### JSON schema and conversion functions ####


class JSONEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # pylint:disable=too-many-instance-attributes
    address: str
    name: str
    matching: float
    type: int
    comparison: dict[str, object]
    recomp: str | None = None
    recomp_varies: bool = False
    stub: bool = False
    library: bool = False
    diff: CombinedDiffOutput | None = None
    accuracy_modulo_stack: float | None = None
    stack_permutation: list[dict[str, object]] | None = None
    accuracy_modulo_inline: float | None = None
    inline_expansions: list[dict[str, object]] | None = None
    diagnostic_normalizations: list[str] | None = None


class JSONReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    format: Literal[2]
    timestamp: float
    data: list[JSONEntity]
    function_count: int
    source_digest: str | None = None


def _serialize_current(
    report: ReccmpStatusReport,
    diff_included: bool = False,
) -> JSONReport:
    """The JSON report can exclude the diff to make deserialization faster."""
    entities = []

    for addr, entity in report.entities.items():
        if not entity.is_matched():
            continue

        assert addr == entity.orig_addr
        entities.append(
            JSONEntity(
                address=format_address(addr),
                name=entity.name,
                matching=entity.accuracy,
                comparison=analysis_json(entity.analysis),
                recomp=(
                    format_address(entity.recomp_addr)
                    if entity.recomp_addr is not None
                    else None
                ),
                recomp_varies=entity.recomp_addr_varies,
                stub=entity.is_stub,
                library=entity.is_library,
                diff=(_render_entity_diff(entity) if diff_included else None),
                type=int(entity.type),
                accuracy_modulo_stack=entity.accuracy_modulo_stack,
                stack_permutation=(
                    [
                        {
                            "orig": entry.orig,
                            "recomp": entry.recomp,
                            **({"symbol": entry.symbol} if entry.symbol else {}),
                        }
                        for entry in entity.stack_permutation
                    ]
                    if entity.stack_permutation
                    else None
                ),
                accuracy_modulo_inline=entity.accuracy_modulo_inline,
                inline_expansions=(
                    [
                        {
                            "helper": entry.helper_name,
                            "helper_orig": format_address(entry.helper_orig_addr),
                            "helper_recomp": format_address(entry.helper_recomp_addr),
                            "side": entry.side,
                            "offset": entry.match_offset,
                            "length": entry.match_length,
                            "counterpart": entry.counterpart,
                            "counterpart_offset": entry.counterpart_offset,
                            "confidence": entry.confidence,
                            **({"semantic": True} if entry.semantic else {}),
                        }
                        for entry in entity.inline_expansions
                    ]
                    if entity.inline_expansions
                    else None
                ),
                diagnostic_normalizations=(
                    [tag.value for tag in entity.diagnostic_normalizations]
                    if entity.diagnostic_normalizations
                    else None
                ),
            )
        )

    report.update_function_count()
    return JSONReport(
        file=report.filename,
        format=2,
        timestamp=report.timestamp.timestamp(),
        data=entities,
        function_count=report.function_count,
        source_digest=report.source_digest,
    )


def _deserialize_current(obj: JSONReport) -> ReccmpStatusReport:
    report = ReccmpStatusReport(
        filename=obj.file,
        timestamp=datetime.fromtimestamp(obj.timestamp),
        source_digest=obj.source_digest,
    )
    report.function_count = obj.function_count

    for e in obj.data:
        entity_type = EntityType(e.type)
        orig_addr = int(e.address, 16)
        recomp_addr = int(e.recomp, 16) if e.recomp is not None else None
        if e.recomp_varies and recomp_addr is not None:
            raise ValueError("A varying recomp address cannot be fixed")
        analysis = parse_analysis(e.comparison)

        report.entities[orig_addr] = ReccmpComparedEntity(
            orig_addr=orig_addr,
            name=e.name,
            accuracy=e.matching,
            type=entity_type,
            recomp_addr=recomp_addr,
            analysis=analysis,
            is_stub=bool(e.stub),
            is_library=bool(e.library),
            report_diff=e.diff,
            recomp_addr_varies=e.recomp_varies,
            accuracy_modulo_stack=e.accuracy_modulo_stack,
            stack_permutation=parse_stack_permutation(e.stack_permutation),
            accuracy_modulo_inline=e.accuracy_modulo_inline,
            inline_expansions=parse_inline_expansions(e.inline_expansions),
            diagnostic_normalizations=parse_diagnostic_normalizations(
                e.diagnostic_normalizations,
                analysis,
                e.accuracy_modulo_stack,
                e.accuracy_modulo_inline,
            ),
        )

    report.update_function_count()
    return report


def deserialize_reccmp_report(json_str: str) -> ReccmpStatusReport:
    """Read only the current structured report schema."""
    try:
        obj = JSONReport.model_validate(from_json(json_str))
        return _deserialize_current(obj)
    except (ValidationError, ValueError) as ex:
        raise ReccmpReportDeserializeError from ex


def serialize_reccmp_report(
    report: ReccmpStatusReport,
    diff_included: bool = False,
) -> str:
    """Create a JSON string for the report so it can be written to a file."""
    obj = _serialize_current(report, diff_included=diff_included)

    return obj.model_dump_json(exclude_defaults=True)
