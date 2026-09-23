"""Text rendering of structured comparison results for reccmp-reccmp."""

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
