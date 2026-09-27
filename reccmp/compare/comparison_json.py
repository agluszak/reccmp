"""JSON encoding of structured comparison results (analysis, differences,
strategy attempts, stack and inline diagnostics) inside a report entity."""

import dataclasses
from collections.abc import Hashable
from enum import Enum

from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.operand import (
    Imm,
    Mem,
    Opaque,
    Operand,
    Reg,
    ScaledReg,
    SignedSymbol,
    St,
    Sym,
)
from reccmp.source.records import SourceComparison, SourceComparisonOperand
from reccmp.types import ImageId

from .diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DiagnosticNormalization,
    DifferenceKind,
    DifferenceSide,
    EffectiveReason,
    ExecutionEvidence,
    FieldAt,
    InconclusiveReason,
    Observed,
    RefutationWitness,
    SolverOutcome,
    SolverResult,
    SourceLine,
    StackPermutationEntry,
    StopDetail,
    StopLocation,
    Strategy,
    StrategyAttempt,
    WitnessInput,
    WitnessKind,
    WitnessReplay,
    derive_diagnostic_normalizations,
)
from .explain import Explanation, explanation
from .inlines import InlineExpansionEvidence


class ComparisonJsonError(ValueError):
    """Malformed comparison data; deserialize_reccmp_report reports it as
    ReccmpReportDeserializeError."""


_IMAGES = {"orig": ImageId.ORIG, "recomp": ImageId.RECOMP}


def _image_json(image: ImageId) -> str:
    return "orig" if image is ImageId.ORIG else "recomp"


def _parse_image(value: object) -> ImageId:
    match value:
        case "orig" | "recomp":
            return _IMAGES[value]
    raise ComparisonJsonError


def _optional_int(value: object) -> int | None:
    match value:
        case None:
            return None
        case bool():
            raise ComparisonJsonError
        case int():
            return value
    raise ComparisonJsonError


def _json_value(value: object) -> object:
    match value:
        case Enum():
            return value.value
        case dict():
            return {str(key): _json_value(item) for key, item in value.items()}
        case list() | tuple():
            return [_json_value(item) for item in value]
        case _:
            return value


def _identity_json(identity: object) -> object:
    match identity:
        case tuple():
            return [_identity_json(item) for item in identity]
        case str() | int() | None:
            return identity
    raise ComparisonJsonError


def _parse_identity(value: object) -> Hashable:
    match value:
        case list():
            return tuple(_parse_identity(item) for item in value)
        case str() | int() | None:
            return value
    raise ComparisonJsonError


def _reference_json(ref: Reference) -> dict[str, object]:
    return {
        "display": ref.display,
        "identity": _identity_json(ref.identity),
        "entity_type": ref.entity_type,
    }


def _parse_reference(value: object) -> Reference:
    match value:
        case {"display": str() as display, "identity": identity, **rest}:
            entity_type = rest.get("entity_type")
            if entity_type is not None and not isinstance(entity_type, str):
                raise ComparisonJsonError
            return Reference(display, _parse_identity(identity), entity_type)
    raise ComparisonJsonError


def _operand_json(operand: Operand) -> dict[str, object]:
    match operand:
        case Reg(name):
            return {"reg": name}
        case St(index):
            return {"st": index}
        case Imm(value):
            return {"imm": value}
        case Sym(ref):
            return {"sym": _reference_json(ref)}
        case Mem(size, segment, terms, displacement, symbols):
            return {
                "mem": {
                    "size": size,
                    "segment": segment,
                    "terms": [[term.register, term.scale] for term in terms],
                    "displacement": displacement,
                    "symbols": [
                        {"sign": term.sign, "ref": _reference_json(term.ref)}
                        for term in symbols
                    ],
                }
            }
        case Opaque(kind, raw, index):
            return {"opaque": {"kind": kind, "raw": raw.hex(), "index": index}}


def _parse_operand(value: object) -> Operand:
    # pylint: disable=too-many-return-statements
    match value:
        case {"reg": str() as name}:
            return Reg(name)
        case {"st": int() as index}:
            return St(index)
        case {"imm": int() as number}:
            return Imm(number)
        case {"sym": ref}:
            return Sym(_parse_reference(ref))
        case {
            "mem": {
                "size": str() as size,
                "segment": str() as segment,
                "terms": list() as terms,
                "displacement": int() as displacement,
                "symbols": list() as symbols,
            }
        }:
            return Mem(
                size,
                segment,
                tuple(_parse_term(term) for term in terms),
                displacement,
                tuple(_parse_signed_symbol(symbol) for symbol in symbols),
            )
        case {
            "opaque": {
                "kind": int() as kind,
                "raw": str() as raw,
                "index": int() as index,
            }
        }:
            return Opaque(kind, bytes.fromhex(raw), index)
    raise ComparisonJsonError


def _parse_term(value: object) -> ScaledReg:
    match value:
        case [str() as register, int() as scale]:
            return ScaledReg(register, scale)
    raise ComparisonJsonError


def _parse_signed_symbol(value: object) -> SignedSymbol:
    match value:
        case {"sign": 1 | -1 as sign, "ref": ref}:
            return SignedSymbol(sign, _parse_reference(ref))
    raise ComparisonJsonError


def _source_json(source: SourceLine | None) -> object:
    return None if source is None else {"path": source.path, "line": source.line}


def _parse_source(value: object) -> SourceLine | None:
    match value:
        case None:
            return None
        case {"path": str() as path, "line": int() as line}:
            return SourceLine(path, line)
    raise ComparisonJsonError


def _field_json(field_at: FieldAt | None) -> object:
    if field_at is None:
        return None
    return {
        "class_name": field_at.class_name,
        "path": list(field_at.path),
        "offset": field_at.offset,
        "type": field_at.type,
    }


def _parse_field(value: object) -> FieldAt | None:
    match value:
        case None:
            return None
        case {
            "class_name": str() as class_name,
            "path": list() as path,
            "offset": int() as offset,
            "type": str() as type_name,
        } if all(isinstance(part, str) for part in path):
            return FieldAt(class_name, tuple(path), offset, type_name)
    raise ComparisonJsonError


def _comparison_json(comparison: SourceComparison) -> dict[str, object]:
    return {
        "operator": comparison.operator,
        "type": comparison.type,
        "operands": [
            {"type": item.type, "field": item.field, "constant": item.constant}
            for item in comparison.operands
        ],
        "line": comparison.line,
        "offset": comparison.offset,
        "bits": comparison.bits,
        "signed": comparison.signed,
        "floating": comparison.floating,
    }


def _parse_comparison(value: object) -> SourceComparison:
    match value:
        case {
            "operator": str() as operator,
            "type": str() as type_name,
            "operands": [left, right],
            "line": int() as line,
            "offset": offset,
            "bits": bits,
            "signed": bool() | None as signed,
            "floating": bool() as floating,
        }:
            return SourceComparison(
                operator,
                type_name,
                (_parse_comparison_operand(left), _parse_comparison_operand(right)),
                line,
                _optional_int(offset),
                _optional_int(bits),
                signed,
                floating,
            )
    raise ComparisonJsonError


def _parse_comparison_operand(value: object) -> SourceComparisonOperand:
    match value:
        case {
            "type": str() as type_name,
            "field": str() | None as field_id,
            "constant": constant,
        }:
            return SourceComparisonOperand(type_name, field_id, _optional_int(constant))
    raise ComparisonJsonError


def _observed_json(observed: Observed) -> dict[str, object]:
    return {
        "operand": (
            _operand_json(observed.operand) if observed.operand is not None else None
        ),
        "target": observed.target,
        "target_index": observed.target_index,
        "value": observed.value,
        "register": observed.register,
    }


def _parse_observed(value: object) -> Observed:
    match value:
        case {
            "operand": operand,
            "target": target,
            "target_index": target_index,
            "value": str() | None as shown,
            "register": str() | None as register,
        }:
            return Observed(
                _parse_operand(operand) if operand is not None else None,
                _optional_int(target),
                _optional_int(target_index),
                shown,
                register,
            )
    raise ComparisonJsonError


def _side_json(side: DifferenceSide) -> dict[str, object]:
    return {
        "image": _image_json(side.image),
        "instruction_index": side.instruction_index,
        "address": side.address,
        "observed": _observed_json(side.observed),
        "source": _source_json(side.source),
        "field": _field_json(side.field),
        "source_comparisons": [
            _comparison_json(item) for item in side.source_comparisons
        ],
    }


def _parse_side(value: object) -> DifferenceSide:
    match value:
        case {
            "image": image,
            "instruction_index": index,
            "address": address,
            "observed": observed,
            "source": source,
            "field": field_at,
            "source_comparisons": list() as comparisons,
        }:
            return DifferenceSide(
                _parse_image(image),
                _optional_int(index),
                _optional_int(address),
                _parse_observed(observed),
                _parse_source(source),
                _parse_field(field_at),
                tuple(_parse_comparison(item) for item in comparisons),
            )
    raise ComparisonJsonError


def _location_json(location: StopLocation) -> dict[str, object]:
    return {
        "image": _image_json(location.image),
        "instruction_index": location.instruction_index,
        "address": location.address,
        "counterpart_address": location.counterpart_address,
        "detail": location.detail.value if location.detail is not None else None,
        "source": _source_json(location.source),
    }


def _parse_location(value: object) -> StopLocation:
    match value:
        case {
            "image": image,
            "instruction_index": index,
            "address": address,
            "counterpart_address": counterpart,
            "detail": str() | None as detail,
            "source": source,
        }:
            return StopLocation(
                _parse_image(image),
                _optional_int(index),
                _optional_int(address),
                _optional_int(counterpart),
                StopDetail(detail) if detail is not None else None,
                _parse_source(source),
            )
    raise ComparisonJsonError


def _solver_json(outcome: SolverOutcome) -> dict[str, object]:
    return {
        "result": outcome.result.value,
        "reason": outcome.reason,
        "rlimit": outcome.rlimit,
    }


def _parse_solver(value: object) -> SolverOutcome | None:
    match value:
        case None:
            return None
        case {
            "result": str() as result,
            "reason": str() | None as reason,
            "rlimit": rlimit,
        }:
            return SolverOutcome(SolverResult(result), reason, _optional_int(rlimit))
    raise ComparisonJsonError


def _difference_json(difference: ComparisonDifference) -> dict[str, object]:
    return {
        "kind": difference.kind.value,
        "orig": _side_json(difference.orig),
        "recomp": _side_json(difference.recomp),
        "solver": (
            _solver_json(difference.solver) if difference.solver is not None else None
        ),
    }


def _parse_difference(value: object) -> ComparisonDifference:
    match value:
        case {"kind": str() as kind, "orig": orig, "recomp": recomp, **rest}:
            return ComparisonDifference(
                DifferenceKind(kind),
                _parse_side(orig),
                _parse_side(recomp),
                solver=_parse_solver(rest.get("solver")),
            )
    raise ComparisonJsonError


def _attempt_json(attempt: StrategyAttempt) -> dict[str, object]:
    return {
        "strategy": attempt.strategy.value,
        "difference": (
            _difference_json(attempt.difference)
            if attempt.difference is not None
            else None
        ),
        "blocker": attempt.blocker.value if attempt.blocker is not None else None,
        "location": (
            _location_json(attempt.location) if attempt.location is not None else None
        ),
    }


def _parse_attempt(value: object) -> StrategyAttempt:
    match value:
        case {
            "strategy": str() as strategy,
            "difference": difference,
            "blocker": str() | None as blocker,
            "location": location,
        }:
            return StrategyAttempt(
                Strategy(strategy),
                _parse_difference(difference) if difference is not None else None,
                InconclusiveReason(blocker) if blocker is not None else None,
                _parse_location(location) if location is not None else None,
            )
    raise ComparisonJsonError


def _explanation_json(found: Explanation) -> dict[str, object]:
    return {
        "group": found.group.value,
        "causes": [cause.code.value for cause in found.causes],
        "orig_value": found.orig_value,
        "recomp_value": found.recomp_value,
        "example": found.example,
        "confirmed": found.confirmed,
    }


def _witness_json(witness: RefutationWitness) -> dict[str, object]:
    return {**dataclasses.asdict(witness), "kind": witness.kind.value}


def _parse_witness_input(value: object) -> WitnessInput:
    match value:
        case {
            "seed": int() as seed,
            "registers": list() as registers,
            "stack_args": list() as stack_args,
            "pool": list() as pool,
            "memory": list() as memory,
        }:
            return WitnessInput(
                seed,
                tuple((str(name), int(number)) for name, number in registers),
                tuple(int(item) for item in stack_args),
                tuple(int(item) for item in pool),
                tuple((int(address), int(byte)) for address, byte in memory),
            )
    raise ComparisonJsonError


def _optional_str(value: object) -> str | None:
    match value:
        case str() | None:
            return value
    raise ComparisonJsonError


def _parse_replay(value: object) -> WitnessReplay | None:
    match value:
        case None:
            return None
        case {
            "input": run_input,
            "orig_function": [int() as orig_start, int() as orig_extent],
            "recomp_function": [int() as recomp_start, int() as recomp_extent],
            "return_kind": str() as return_kind,
            "model": int() as model,
            "images": [orig_digest, recomp_digest],
        }:
            return WitnessReplay(
                _parse_witness_input(run_input),
                (orig_start, orig_extent),
                (recomp_start, recomp_extent),
                return_kind,
                model,
                (_optional_str(orig_digest), _optional_str(recomp_digest)),
            )
    raise ComparisonJsonError


def _parse_witness(value: object) -> RefutationWitness:
    match value:
        case {
            "seed": int() as seed,
            "kind": str() as kind,
            "location": str() as location,
            "orig_value": str() as orig_value,
            "recomp_value": str() as recomp_value,
            "orig_address": orig_address,
            "recomp_address": recomp_address,
            "replay": replay,
        }:
            return RefutationWitness(
                seed,
                WitnessKind(kind),
                location,
                orig_value,
                recomp_value,
                _optional_int(orig_address),
                _optional_int(recomp_address),
                _parse_replay(replay),
            )
    raise ComparisonJsonError


def _parse_execution(value: object) -> ExecutionEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ComparisonJsonError
    fields = dict(value)
    details = fields.get("no_verdict_details", {})
    if not isinstance(details, dict):
        raise ComparisonJsonError
    normalized = {str(reason): dict(detail) for reason, detail in details.items()}
    estimated = normalized.get("estimated_extent")
    if estimated is not None and isinstance(estimated.get("kind"), str):
        estimated["kind"] = DifferenceKind(estimated["kind"])
    fields["no_verdict_details"] = normalized
    try:
        return ExecutionEvidence(**fields)
    except TypeError as ex:
        raise ComparisonJsonError from ex


def analysis_json(analysis: ComparisonAnalysis) -> dict[str, object]:
    value: dict[str, object] = {"status": analysis.status.value}
    if analysis.effective_reasons:
        value["effective_reasons"] = [
            reason.value for reason in analysis.effective_reasons
        ]
    if analysis.difference is not None:
        value["difference"] = _difference_json(analysis.difference)
        found = explanation(analysis)
        if found is not None:
            value["explanation"] = _explanation_json(found)
    if analysis.inconclusive_reason is not None:
        value["inconclusive_reason"] = analysis.inconclusive_reason.value
    if analysis.inconclusive_location is not None:
        value["inconclusive_location"] = _location_json(analysis.inconclusive_location)
    if analysis.attempts:
        value["attempts"] = [_attempt_json(attempt) for attempt in analysis.attempts]
    if analysis.witness is not None:
        value["witness"] = _witness_json(analysis.witness)
    if analysis.execution is not None:
        value["execution"] = _json_value(dataclasses.asdict(analysis.execution))
    return value


def parse_analysis(value: object) -> ComparisonAnalysis:
    if not isinstance(value, dict):
        raise ComparisonJsonError
    try:
        difference = value.get("difference")
        reason = value.get("inconclusive_reason")
        location = value.get("inconclusive_location")
        return ComparisonAnalysis(
            status=ComparisonStatus(value["status"]),
            effective_reasons=tuple(
                EffectiveReason(item) for item in value.get("effective_reasons", ())
            ),
            difference=(
                _parse_difference(difference) if difference is not None else None
            ),
            inconclusive_reason=(
                InconclusiveReason(reason) if reason is not None else None
            ),
            inconclusive_location=(
                _parse_location(location) if location is not None else None
            ),
            attempts=tuple(_parse_attempt(item) for item in value.get("attempts", ())),
            witness=(
                _parse_witness(value["witness"])
                if value.get("witness") is not None
                else None
            ),
            execution=_parse_execution(value.get("execution")),
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
            tags.add(DiagnosticNormalization(item))
        except ValueError as ex:
            raise ComparisonJsonError from ex
    return tuple(tag for tag in DiagnosticNormalization if tag in tags)
