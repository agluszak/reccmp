"""Text rendering of structured comparison results for reccmp-reccmp."""

from collections import Counter
from typing import Iterable

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DifferenceSide,
)
from reccmp.compare.report import ReccmpComparedEntity, format_address
from reccmp.utils import percent_string


def triage_status_note(analysis: ComparisonAnalysis) -> str | None:
    """A one-line reminder of what a triage status means, so the semantics
    travel with the output and a reader does not misread the result. Returned
    for the two statuses that are routinely misread; None for `exact` (needs no
    gloss) and `mismatch` (the actionable case, whose diff speaks for itself)."""
    if analysis.status == ComparisonStatus.EFFECTIVE:
        return "effective: proved semantically harmless — no action needed"
    if analysis.status == ComparisonStatus.INCONCLUSIVE:
        return (
            "inconclusive: verifier could not prove either outcome — "
            "NOT evidence of a source defect; investigate verifier/metadata/alignment"
        )
    return None


def inconclusive_diagnostic_text(analysis: ComparisonAnalysis) -> str | None:
    """Render the structured reason, location, and facts for an inconclusive result."""
    if analysis.status != ComparisonStatus.INCONCLUSIVE:
        return None
    reason = (analysis.inconclusive_reason or "analysis_limit").replace("_", " ")
    lines = [f"semantic analysis inconclusive: {reason}"]
    location = analysis.inconclusive_location
    if location is not None:
        if location.address is not None:
            lines.append(f"  location: {format_address(location.address)}")
        elif location.instruction_index is not None:
            lines.append(f"  instruction index: {location.instruction_index}")
        for key, value in sorted(location.facts.items()):
            lines.append(f"  {key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def _side_text(side: DifferenceSide) -> str:
    if side.address is not None:
        text = format_address(side.address)
    elif side.instruction_index is not None:
        text = f"instruction {side.instruction_index}"
    else:
        text = "function exit"
    path = side.facts.get("source_path")
    line = side.facts.get("source_line")
    if isinstance(path, str) and isinstance(line, int):
        text += f" ({path}:{line})"
    return text


def strategy_attempts_text(analysis: ComparisonAnalysis) -> str | None:
    """One line per verifier strategy: where it stopped and why."""
    if not analysis.attempts:
        return None
    lines = ["verifier strategies:"]
    for attempt in analysis.attempts:
        name = attempt.strategy.replace("_", " ")
        if attempt.difference is not None:
            kind = attempt.difference.kind.replace("_", " ")
            where = _side_text(attempt.difference.orig)
            note = "" if attempt.trusted_alignment else " (heuristic pairing)"
            lines.append(f"  {name}: {kind} difference at {where}{note}")
        else:
            reason = (attempt.blocker or "analysis_limit").replace("_", " ")
            where = (
                f" at {_side_text(attempt.location)}"
                if attempt.location is not None
                else ""
            )
            lines.append(f"  {name}: blocked by {reason}{where}")
    return "\n".join(lines)


def mismatch_source_pin_text(match: ReccmpComparedEntity) -> str | None:
    """First recomp source line attached to a structured mismatch, if any."""
    difference = match.analysis.difference
    if difference is None:
        return None
    facts = difference.recomp.facts
    path = facts.get("source_path")
    line = facts.get("source_line")
    if isinstance(path, str) and isinstance(line, int):
        return f"probable first source-level discrepancy: {path}:{line}"
    return None


def stack_layout_text(match: ReccmpComparedEntity) -> str | None:
    """Human-readable stack permutation / modulo-stack score."""
    if not match.stack_permutation and match.accuracy_modulo_stack is None:
        return None
    lines: list[str] = []
    if match.accuracy_modulo_stack is not None:
        raw = percent_string(match.accuracy)
        modulo = percent_string(match.accuracy_modulo_stack)
        lines.append(f"{raw} raw / {modulo} modulo stack allocation")
    if match.stack_permutation:
        lines.append("stack permutation:")
        for entry in match.stack_permutation:
            if entry.orig == entry.recomp:
                continue
            symbol = f"  {entry.symbol}" if entry.symbol else ""
            lines.append(f"    {entry.orig} -> {entry.recomp}{symbol}")
    return "\n".join(lines) if lines else None


def inline_layout_text(match: ReccmpComparedEntity) -> str | None:
    """Human-readable known-inline expansions / modulo-inline score."""
    if not match.inline_expansions and match.accuracy_modulo_inline is None:
        return None
    lines: list[str] = []
    if match.accuracy_modulo_inline is not None:
        raw = percent_string(match.accuracy)
        modulo = percent_string(match.accuracy_modulo_inline)
        lines.append(f"{raw} raw / {modulo} modulo known inline expansion")
    if match.inline_expansions:
        lines.append("known inline expansions:")
        for entry in match.inline_expansions:
            where = entry.side
            counterpart = entry.counterpart
            detail = f"insn@{entry.match_offset}+{entry.match_length}"
            if entry.counterpart_offset is not None:
                detail += f" ↔ {counterpart}@{entry.counterpart_offset}"
            else:
                detail += f" ({counterpart})"
            lines.append(
                f"    {format_address(entry.helper_orig_addr)}  {entry.helper_name}  "
                f"on {where}: {detail}"
            )
    return "\n".join(lines) if lines else None


def diagnostic_normalizations_text(match: ReccmpComparedEntity) -> str | None:
    """Render non-proof diagnostic tags without claiming equivalence."""
    if not match.diagnostic_normalizations:
        return None
    joined = ", ".join(
        tag.value.replace("_", " ") for tag in match.diagnostic_normalizations
    )
    return f"diagnostic normalizations: {joined}"


def witness_text(analysis: ComparisonAnalysis) -> str | None:
    """The concrete difference behind a refuted mismatch."""
    witness = analysis.witness
    if witness is None:
        return None
    where = witness.location
    if witness.orig_address is not None and witness.recomp_address is not None:
        where += (
            f" (orig {format_address(witness.orig_address)},"
            f" recomp {format_address(witness.recomp_address)})"
        )
    return (
        f"refuted by execution (seed {witness.seed}): "
        f"{witness.kind.replace('_', ' ')} at {where}: "
        f"orig {witness.orig_value}, recomp {witness.recomp_value}\n"
        "  callees modelled, input not checked for reachability"
    )


def execution_text(analysis: ComparisonAnalysis) -> str | None:
    """Differential execution that ran without finding a witness."""
    evidence = analysis.execution
    if evidence is None:
        return None
    text = f"execution: {evidence.agreeing} of {evidence.runs} runs agreed"
    if evidence.reached_location:
        what = "difference" if analysis.difference is not None else "blocker"
        text += (
            f"; {evidence.reached_location} reached the reported {what} "
            "without an observable divergence"
        )
    if evidence.no_verdict:
        reasons = ", ".join(
            f"{reason.replace('_', ' ')} {count}"
            for reason, count in sorted(
                evidence.no_verdict.items(), key=lambda item: -item[1]
            )
        )
        text += f"\n  no verdict: {reasons}"
    return text


def verdict_summary_text(entities: Iterable[ReccmpComparedEntity]) -> str:
    counts: Counter[str] = Counter()
    for entity in entities:
        if not entity.is_function() or not entity.is_matched() or entity.is_stub:
            continue
        status = entity.analysis.status
        if status == ComparisonStatus.MISMATCH:
            counts["refuted" if entity.analysis.is_refuted else "candidate"] += 1
            execution = entity.analysis.execution
            if execution is not None and execution.reached_location:
                counts["reached"] += 1
        else:
            counts[status.value] += 1
    return (
        f"Verdicts:     {counts['exact']} exact, {counts['effective']} effective, "
        f"{counts['refuted']} refuted, {counts['candidate']} candidate mismatch, "
        f"{counts['inconclusive']} inconclusive"
    ) + (
        f"\n              ({counts['reached']} candidate mismatches were executed "
        "through the difference without diverging)"
        if counts["reached"]
        else ""
    )
