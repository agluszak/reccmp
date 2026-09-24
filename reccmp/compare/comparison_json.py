"""JSON encoding of structured comparison results (analysis, differences,
strategy attempts, stack and inline diagnostics) inside a report entity."""

import dataclasses

from .diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DiagnosticNormalization,
    DifferenceSide,
    ExecutionEvidence,
    RefutationWitness,
    StackPermutationEntry,
    StrategyAttempt,
    derive_diagnostic_normalizations,
)
from .inlines import InlineExpansionEvidence


class ComparisonJsonError(ValueError):
    """Malformed comparison data; deserialize_reccmp_report reports it as
    ReccmpReportDeserializeError."""


def _side_json(side: DifferenceSide) -> dict[str, object]:
    value: dict[str, object] = {
        "instruction_index": side.instruction_index,
        "address": side.address,
        "facts": side.facts,
    }
    if side.image is not None:
        value["image"] = side.image
    return value


def _difference_json(difference: ComparisonDifference) -> dict[str, object]:
    return {
        "kind": difference.kind,
        "orig": _side_json(difference.orig),
        "recomp": _side_json(difference.recomp),
    }


def _attempt_json(attempt: StrategyAttempt) -> dict[str, object]:
    value: dict[str, object] = {"strategy": attempt.strategy}
    if attempt.difference is not None:
        value["difference"] = _difference_json(attempt.difference)
    if attempt.blocker is not None:
        value["blocker"] = attempt.blocker
    if attempt.location is not None:
        value["location"] = _side_json(attempt.location)
    return value


def analysis_json(analysis: ComparisonAnalysis) -> dict[str, object]:
    value: dict[str, object] = {"status": analysis.status.value}
    if analysis.effective_reasons:
        value["effective_reasons"] = list(analysis.effective_reasons)
    if analysis.difference is not None:
        value["difference"] = _difference_json(analysis.difference)
    if analysis.inconclusive_reason is not None:
        value["inconclusive_reason"] = analysis.inconclusive_reason
    if analysis.inconclusive_location is not None:
        value["inconclusive_location"] = _side_json(analysis.inconclusive_location)
    if analysis.attempts:
        value["attempts"] = [_attempt_json(attempt) for attempt in analysis.attempts]
    if analysis.witness is not None:
        value["witness"] = dataclasses.asdict(analysis.witness)
    if analysis.execution is not None:
        value["execution"] = dataclasses.asdict(analysis.execution)
    return value


def _parse_side(value: object) -> DifferenceSide:
    if not isinstance(value, dict):
        raise ComparisonJsonError
    instruction_index = value.get("instruction_index")
    address = value.get("address")
    if instruction_index is not None and not isinstance(instruction_index, int):
        raise ComparisonJsonError
    if address is not None and not isinstance(address, int):
        raise ComparisonJsonError
    facts = value.get("facts", {})
    if not isinstance(facts, dict):
        raise ComparisonJsonError
    if not all(
        isinstance(key, str) and (fact is None or isinstance(fact, (str, int, bool)))
        for key, fact in facts.items()
    ):
        raise ComparisonJsonError
    image = value.get("image")
    if image not in (None, "orig", "recomp"):
        raise ComparisonJsonError
    return DifferenceSide(instruction_index, address, facts, image)


def _parse_difference(value: dict) -> ComparisonDifference:
    return ComparisonDifference(
        kind=value["kind"],
        orig=_parse_side(value["orig"]),
        recomp=_parse_side(value["recomp"]),
    )


def _parse_attempt(value: object) -> StrategyAttempt:
    if not isinstance(value, dict):
        raise ComparisonJsonError
    return StrategyAttempt(
        strategy=value["strategy"],
        difference=(
            _parse_difference(value["difference"])
            if value.get("difference") is not None
            else None
        ),
        blocker=value.get("blocker"),
        location=(
            _parse_side(value["location"])
            if value.get("location") is not None
            else None
        ),
    )


def parse_analysis(value: object) -> ComparisonAnalysis:
    if not isinstance(value, dict):
        raise ComparisonJsonError
    try:
        status = ComparisonStatus(value["status"])
        difference_value = value.get("difference")
        difference = None
        if difference_value is not None:
            difference = _parse_difference(difference_value)
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
            attempts=tuple(_parse_attempt(item) for item in value.get("attempts", ())),
            witness=(
                RefutationWitness(**value["witness"])
                if value.get("witness") is not None
                else None
            ),
            execution=(
                ExecutionEvidence(**value["execution"])
                if value.get("execution") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as ex:
        raise ComparisonJsonError from ex


def parse_stack_permutation(
    value: list[dict[str, object]] | None,
) -> tuple[StackPermutationEntry, ...]:
    if not value:
        return ()
    entries: list[StackPermutationEntry] = []
    for item in value:
        if not isinstance(item, dict):
            raise ComparisonJsonError
        orig = item.get("orig")
        recomp = item.get("recomp")
        symbol = item.get("symbol")
        if not isinstance(orig, str) or not isinstance(recomp, str):
            raise ComparisonJsonError
        if symbol is not None and not isinstance(symbol, str):
            raise ComparisonJsonError
        entries.append(StackPermutationEntry(orig, recomp, symbol))
    return tuple(entries)


def parse_inline_expansions(
    value: list[dict[str, object]] | None,
) -> tuple[InlineExpansionEvidence, ...]:
    if not value:
        return ()
    entries: list[InlineExpansionEvidence] = []
    for item in value:
        if not isinstance(item, dict):
            raise ComparisonJsonError
        helper = item.get("helper")
        helper_orig = item.get("helper_orig")
        helper_recomp = item.get("helper_recomp")
        side = item.get("side")
        offset = item.get("offset")
        length = item.get("length")
        counterpart = item.get("counterpart")
        counterpart_offset = item.get("counterpart_offset")
        confidence = item.get("confidence", 0.0)
        semantic = item.get("semantic", False)
        if not isinstance(helper, str) or not isinstance(helper_orig, str):
            raise ComparisonJsonError
        if not isinstance(helper_recomp, str):
            raise ComparisonJsonError
        if side not in ("orig", "recomp", "both"):
            raise ComparisonJsonError
        if counterpart not in ("call", "inline", "absent"):
            raise ComparisonJsonError
        if not isinstance(offset, int) or not isinstance(length, int):
            raise ComparisonJsonError
        if counterpart_offset is not None and not isinstance(counterpart_offset, int):
            raise ComparisonJsonError
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ComparisonJsonError
        if not isinstance(semantic, bool):
            raise ComparisonJsonError
        entries.append(
            InlineExpansionEvidence(
                helper_name=helper,
                helper_orig_addr=int(helper_orig, 16),
                helper_recomp_addr=int(helper_recomp, 16),
                side=side,
                match_offset=offset,
                match_length=length,
                counterpart=counterpart,
                counterpart_offset=counterpart_offset,
                confidence=float(confidence),
                semantic=semantic,
            )
        )
    return tuple(entries)


def parse_diagnostic_normalizations(
    value: list[str] | None,
    analysis: ComparisonAnalysis,
    accuracy_modulo_stack: float | None,
    accuracy_modulo_inline: float | None,
) -> tuple[DiagnosticNormalization, ...]:
    if not value:
        return derive_diagnostic_normalizations(
            analysis,
            accuracy_modulo_stack=accuracy_modulo_stack,
            accuracy_modulo_inline=accuracy_modulo_inline,
        )
    tags: set[DiagnosticNormalization] = set()
    for item in value:
        if not isinstance(item, str):
            raise ComparisonJsonError
        try:
            tags.add(DiagnosticNormalization(item.removesuffix("_equivalent")))
        except ValueError as ex:
            raise ComparisonJsonError from ex
    return tuple(tag for tag in DiagnosticNormalization if tag in tags)
