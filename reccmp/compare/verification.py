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
    coverage_incomplete: bool,
    extent_closed: bool,
) -> ComparisonAnalysis:
    """Refuse EXACT/EFFECTIVE when reachable coverage or extent is open."""
    if not analysis.is_effective:
        return analysis
    if coverage_incomplete:
        return ComparisonAnalysis.inconclusive("incomplete_coverage")
    if not extent_closed:
        return ComparisonAnalysis.inconclusive("open_extent")
    return analysis


def _closed_assumptions(
    *, coverage_incomplete: bool, extent_closed: bool
) -> tuple[str, ...]:
    assumptions: list[str] = []
    if not coverage_incomplete:
        assumptions.append("coverage_complete")
    if extent_closed:
        assumptions.append("extent_closed")
    return tuple(assumptions)


def admit_exact(
    *,
    displays_equal: bool,
    topology_equal: bool,
    keys_equal: bool,
    operands_complete: bool = True,
    control_flow_complete: bool = True,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> VerificationResult | None:
    # pylint: disable=too-many-arguments
    """Mint an EXACT ``VerificationResult``; strategies only propose evidence."""
    if coverage_incomplete or not extent_closed:
        return None
    if not topology_equal:
        return None
    if not keys_equal:
        return None
    if not (displays_equal or (operands_complete and control_flow_complete)):
        return None
    return VerificationResult(
        analysis=ComparisonAnalysis.exact(),
        assumptions=_closed_assumptions(
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        ),
    )


def admit_exact_analysis(
    *,
    displays_equal: bool,
    topology_equal: bool,
    keys_equal: bool,
    operands_complete: bool = True,
    control_flow_complete: bool = True,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> ComparisonAnalysis | None:
    # pylint: disable=too-many-arguments
    """Shared EXACT admission policy for function comparison.

    Strategies may propose identical displays or IR keys; this is the only
    gate that mints ``ComparisonStatus.EXACT``. Incomplete reachable coverage
    or an unclosed estimated extent never admits EXACT. Identical display text
    is not enough: local branch destinations (instruction ids, not encodings)
    must also agree, and match keys (which include reference identities) must
    not disagree. Callers must pass ``keys_equal`` and ``extent_closed``
    explicitly so a forgotten semantic or extent obligation cannot default
    into a proof.
    """
    minted = admit_exact(
        displays_equal=displays_equal,
        topology_equal=topology_equal,
        keys_equal=keys_equal,
        operands_complete=operands_complete,
        control_flow_complete=control_flow_complete,
        coverage_incomplete=coverage_incomplete,
        extent_closed=extent_closed,
    )
    return None if minted is None else minted.analysis


def admit_effective(
    reasons,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> VerificationResult | None:
    """Mint an EFFECTIVE ``VerificationResult``; strategies only propose reasons."""
    if coverage_incomplete or not extent_closed:
        return None
    normalized = tuple(sorted(set(reasons)))
    return VerificationResult(
        analysis=ComparisonAnalysis.effective(reasons),
        assumptions=_closed_assumptions(
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        ),
        reasons=normalized,
    )


def admit_effective_analysis(
    reasons,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> ComparisonAnalysis | None:
    """Shared EFFECTIVE admission; same coverage/extent obligations as exact."""
    minted = admit_effective(
        reasons,
        coverage_incomplete=coverage_incomplete,
        extent_closed=extent_closed,
    )
    return None if minted is None else minted.analysis
