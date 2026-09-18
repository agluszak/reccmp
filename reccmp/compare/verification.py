"""Single admission layer for comparison proofs.

Strategies may propose exact displays, IR keys, effective-match reasons,
or alias evidence.  This module is the only place that turns a successful
proposal into ``ComparisonAnalysis`` EXACT/EFFECTIVE.
"""

from __future__ import annotations

from dataclasses import dataclass

from reccmp.compare.diagnosis import ComparisonAnalysis, ComparisonStatus


@dataclass(frozen=True)
class VerificationResult:
    """A strategy proposal after shared admission.

    ``assumptions`` records coverage/extent obligations that were required
    for the proof. ``reasons`` are effective-match categories. ``obligations``
    are leftover checks a later strategy would still need to discharge.
    """

    analysis: ComparisonAnalysis
    assumptions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()

    @property
    def proof_kind(self) -> ComparisonStatus | None:
        if self.analysis.status in (
            ComparisonStatus.EXACT,
            ComparisonStatus.EFFECTIVE,
        ):
            return self.analysis.status
        return None


def admit_proof(
    analysis: ComparisonAnalysis,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool = True,
) -> ComparisonAnalysis:
    """Refuse EXACT/EFFECTIVE when reachable coverage or extent is open."""
    if not analysis.is_effective:
        return analysis
    if coverage_incomplete:
        return ComparisonAnalysis.inconclusive("incomplete_coverage")
    if not extent_closed:
        return ComparisonAnalysis.inconclusive("open_extent")
    return analysis


def admit_exact_analysis(
    *,
    displays_equal: bool,
    topology_equal: bool,
    keys_equal: bool | None = None,
    operands_complete: bool = True,
    control_flow_complete: bool = True,
    coverage_incomplete: bool = False,
    extent_closed: bool = True,
) -> ComparisonAnalysis | None:
    """Shared EXACT admission policy for function comparison.

    Strategies may propose identical displays or IR keys; this is the only
    gate that mints ``ComparisonStatus.EXACT``. Incomplete reachable coverage
    or an unclosed estimated extent never admits EXACT. Identical display text
    is not enough: local branch destinations (instruction ids, not encodings)
    must also agree, and match keys (which include reference identities) must
    not disagree. Unresolved ``<OFFSET>`` placeholders from opposite images
    therefore cannot become EXACT merely by sharing a replacement slot.
    """
    if coverage_incomplete or not extent_closed:
        return None
    if not topology_equal:
        return None
    if displays_equal:
        if keys_equal is False:
            return None
        return ComparisonAnalysis.exact()
    if keys_equal and operands_complete and control_flow_complete:
        return ComparisonAnalysis.exact()
    return None


def admit_effective_analysis(
    reasons,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool = True,
) -> ComparisonAnalysis | None:
    """Shared EFFECTIVE admission; same coverage/extent obligations as exact."""
    if coverage_incomplete or not extent_closed:
        return None
    return ComparisonAnalysis.effective(reasons)
