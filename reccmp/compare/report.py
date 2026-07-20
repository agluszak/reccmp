from datetime import datetime
from dataclasses import dataclass
import json
from typing import Any, Iterable, Iterator, cast
from reccmp.types import EntityType
from .diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DifferenceSide,
)
from .diff import (
    CombinedDiffOutput,
    DiffReport,
    MatchingOrMismatchingBlock,
    RawDiffOutput,
    raw_diff_to_udiff,
)


class ReccmpReportDeserializeError(Exception):
    """The given file is not a serialized reccmp report file"""


class ReccmpReportSameSourceError(Exception):
    """Tried to aggregate reports derived from different source files."""


@dataclass
class ReccmpComparedEntity:
    # pylint:disable=too-many-instance-attributes
    orig_addr: str
    name: str
    accuracy: float
    type: EntityType | None = None
    recomp_addr: str | None = None
    analysis: ComparisonAnalysis = ComparisonAnalysis.inconclusive("analysis_limit")
    is_stub: bool = False
    rdiff: RawDiffOutput | None = None
    report_diff: CombinedDiffOutput | None = None

    @property
    def is_effective_match(self) -> bool:
        return self.analysis.status == ComparisonStatus.EFFECTIVE

    @property
    def effective_accuracy(self) -> float:
        return 1.0 if self.is_effective_match else self.accuracy


class ReccmpStatusReport:
    # The filename of the original binary.
    # This is here to avoid comparing reports derived from different files.
    # TODO: in the future, we may want to use the hash instead
    filename: str

    # Creation date of the report file.
    timestamp: datetime

    # Using orig addr as the key.
    entities: dict[str, ReccmpComparedEntity]

    def __init__(
        self,
        filename: str,
        timestamp: datetime | None = None,
    ) -> None:
        self.filename = filename
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().replace(microsecond=0)

        self.entities = {}

    def add_match(self, match: DiffReport):
        orig_addr = f"0x{match.orig_addr:x}"
        recomp_addr = f"0x{match.recomp_addr:x}"

        self.entities[orig_addr] = ReccmpComparedEntity(
            orig_addr=orig_addr,
            name=match.name,
            type=match.match_type,
            accuracy=match.ratio,
            recomp_addr=recomp_addr,
            analysis=match.result.analysis,
            is_stub=match.is_stub,
            rdiff=match.result.diff,
        )

    def has_same_source(self, other: "ReccmpStatusReport") -> bool:
        """Were both reports derived from the same reccmp target?"""
        return self.filename.lower() == other.filename.lower()


def report_function_alignment(report: ReccmpStatusReport) -> int:
    """Report the count of all (non-contiguous) functions where
    the address is the same in both binaries."""
    count = 0
    for ent in report.entities.values():
        if ent.type == EntityType.FUNCTION and ent.orig_addr == ent.recomp_addr:
            count += 1

    return count


def report_function_accuracy(report: ReccmpStatusReport) -> tuple[int, float, float]:
    """Collects the accuracy and effective accuracy of all compared functions in the report.
    Returns (function_count, total_accuracy, total_effective_accuracy).
    Stubs are not compared so they are excluded.
    The accuracy scores are raw score values. Divide by the function_count to get the percentage.
    """
    function_count = 0
    total_accuracy = 0.0
    total_effective_accuracy = 0.0

    for ent in report.entities.values():
        if ent.type == EntityType.FUNCTION and not ent.is_stub:
            function_count += 1
            total_accuracy += ent.accuracy
            total_effective_accuracy += ent.effective_accuracy

    return (function_count, total_accuracy, total_effective_accuracy)


def report_progress_stats(report: ReccmpStatusReport) -> tuple[int, float]:
    """Count comparable functions in the report and sum their effective-match accuracy.

    Returns (implemented_funcs, raw_accuracy). Stubs and non-FUNCTION entities are excluded.
    Entities without a type are treated as functions."""
    implemented = 0
    raw_accuracy = 0.0
    for entity in report.entities.values():
        if entity.is_stub:
            continue
        if entity.type is not None and entity.type != EntityType.FUNCTION:
            continue
        implemented += 1
        raw_accuracy += entity.effective_accuracy
    return implemented, raw_accuracy


def _get_entity_for_addr(
    samples: Iterable[ReccmpStatusReport], addr: str
) -> Iterator[ReccmpComparedEntity]:
    """Helper to return entities from xreports that have the given address."""
    for sample in samples:
        if addr in sample.entities:
            yield sample.entities[addr]


def _accuracy_sort_key(entity: ReccmpComparedEntity) -> float:
    """Helper to sort entity samples by accuracy score.
    100% match is preferred over effective match.
    Effective match is preferred over any accuracy.
    Stubs rank lower than any accuracy score."""
    if entity.is_stub:
        return -1.0

    if entity.accuracy == 1.0:
        if not entity.is_effective_match:
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

    output = ReccmpStatusReport(filename=samples[0].filename)

    # Combine every orig addr used in any of the reports.
    orig_addr_set = {key for sample in samples for key in sample.entities.keys()}

    all_orig_addrs = sorted(list(orig_addr_set))

    for addr in all_orig_addrs:
        e_list = list(_get_entity_for_addr(samples, addr))
        assert len(e_list) > 0

        # Our aggregate accuracy score is the highest from any report.
        e_list.sort(key=_accuracy_sort_key, reverse=True)

        output.entities[addr] = e_list[0]

        # Keep the recomp_addr if it is the same across all samples.
        # i.e. to detect where function alignment ends
        if not all(e_list[0].recomp_addr == e.recomp_addr for e in e_list):
            output.entities[addr].recomp_addr = "various"

    return output


def get_udiff_for_entity(entity: ReccmpComparedEntity) -> CombinedDiffOutput | None:
    """Create a unified diff for this entity to add to a version 1 report.

    If the entity was imported from a version 1 report and we already have a unified diff, use it.
    This can occur with `reccmp-aggregate` where we copy the entity with the highest accuracy score.

    If there is no unified diff, create a new one using the entity's raw diff, if it exists.

    If we return None, no diff is possible because the entity matches 100%, is a stub,
    or was created from a deserialized report without diff data."""
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


def _side_json(side: DifferenceSide) -> dict[str, object]:
    return {
        "instruction_index": side.instruction_index,
        "address": side.address,
        "facts": side.facts,
    }


def _analysis_json(analysis: ComparisonAnalysis) -> dict[str, object]:
    value: dict[str, object] = {"status": analysis.status.value}
    if analysis.effective_reasons:
        value["effective_reasons"] = list(analysis.effective_reasons)
    if analysis.difference is not None:
        value["difference"] = {
            "kind": analysis.difference.kind,
            "orig": _side_json(analysis.difference.orig),
            "recomp": _side_json(analysis.difference.recomp),
        }
    if analysis.inconclusive_reason is not None:
        value["inconclusive_reason"] = analysis.inconclusive_reason
    if analysis.inconclusive_location is not None:
        value["inconclusive_location"] = _side_json(
            analysis.inconclusive_location
        )
    return value


def _parse_side(value: object) -> DifferenceSide:
    if not isinstance(value, dict) or not isinstance(value.get("facts"), dict):
        raise ReccmpReportDeserializeError
    instruction_index = value.get("instruction_index")
    address = value.get("address")
    if instruction_index is not None and not isinstance(instruction_index, int):
        raise ReccmpReportDeserializeError
    if address is not None and not isinstance(address, int):
        raise ReccmpReportDeserializeError
    facts = value["facts"]
    if not all(
        isinstance(key, str) and (fact is None or isinstance(fact, (str, int, bool)))
        for key, fact in facts.items()
    ):
        raise ReccmpReportDeserializeError
    return DifferenceSide(instruction_index, address, facts)


def _parse_analysis(value: object) -> ComparisonAnalysis:
    if not isinstance(value, dict):
        raise ReccmpReportDeserializeError
    try:
        status = ComparisonStatus(value["status"])
        difference_value = value.get("difference")
        difference = None
        if difference_value is not None:
            difference = ComparisonDifference(
                kind=difference_value["kind"],
                orig=_parse_side(difference_value["orig"]),
                recomp=_parse_side(difference_value["recomp"]),
            )
        return ComparisonAnalysis(
            status=status,
            effective_reasons=tuple(value.get("effective_reasons", ())),
            difference=difference,
            inconclusive_reason=value.get("inconclusive_reason"),
            inconclusive_location=(
                _parse_side(value["inconclusive_location"])
                if value.get("inconclusive_location") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as ex:
        raise ReccmpReportDeserializeError from ex


def _parse_report_diff(value: object) -> CombinedDiffOutput | None:
    """Restore the tuple-shaped in-memory diff representation from JSON lists."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ReccmpReportDeserializeError
    output: CombinedDiffOutput = []
    try:
        for group in value:
            if not isinstance(group, list) or len(group) != 2:
                raise ReccmpReportDeserializeError
            slug, blocks = group
            if not isinstance(slug, str) or not isinstance(blocks, list):
                raise ReccmpReportDeserializeError
            parsed_blocks: list[MatchingOrMismatchingBlock] = []
            for block in blocks:
                if not isinstance(block, dict):
                    raise ReccmpReportDeserializeError
                if not set(block).issubset({"both", "orig", "recomp"}):
                    raise ReccmpReportDeserializeError
                parsed_block = cast(
                    MatchingOrMismatchingBlock,
                    {
                        key: [tuple(item) for item in lines]
                        for key, lines in block.items()
                    },
                )
                parsed_blocks.append(parsed_block)
            output.append((slug, parsed_blocks))
    except (TypeError, ValueError) as ex:
        raise ReccmpReportDeserializeError from ex
    return output


def deserialize_reccmp_report(json_str: str) -> ReccmpStatusReport:
    """Read only the current structured format-1 schema."""
    try:
        obj = json.loads(json_str)
        if obj.get("format") != 1 or not isinstance(obj.get("data"), list):
            raise ReccmpReportDeserializeError
        report = ReccmpStatusReport(
            filename=obj["file"],
            timestamp=datetime.fromtimestamp(obj["timestamp"]),
        )
        for value in obj["data"]:
            entity_type = (
                EntityType(value["type"]) if value.get("type") is not None else None
            )
            address = value["address"]
            report.entities[address] = ReccmpComparedEntity(
                orig_addr=address,
                name=value["name"],
                accuracy=value["matching"],
                type=entity_type,
                recomp_addr=value.get("recomp"),
                analysis=_parse_analysis(value["comparison"]),
                is_stub=value.get("stub", False),
                report_diff=_parse_report_diff(value.get("diff")),
            )
        return report
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as ex:
        raise ReccmpReportDeserializeError from ex


def serialize_reccmp_report(
    report: ReccmpStatusReport, diff_included: bool = False
) -> str:
    """Create a JSON report whose comparison object is authoritative."""
    report.timestamp = datetime.now().replace(microsecond=0)
    entities = []
    for address, entity in report.entities.items():
        value: dict[str, Any] = {
            "address": address,
            "name": entity.name,
            "matching": entity.accuracy,
            "comparison": _analysis_json(entity.analysis),
        }
        if entity.recomp_addr is not None:
            value["recomp"] = entity.recomp_addr
        if entity.is_stub:
            value["stub"] = True
        if entity.type is not None:
            value["type"] = int(entity.type)
        if diff_included:
            difference = get_udiff_for_entity(entity)
            if difference is not None:
                value["diff"] = difference
        entities.append(value)
    return json.dumps(
        {
            "file": report.filename,
            "format": 1,
            "timestamp": report.timestamp.timestamp(),
            "data": entities,
        },
        separators=(",", ":"),
    )
