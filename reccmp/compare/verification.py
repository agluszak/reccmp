"""Single admission layer for comparison proofs.

Strategies may propose exact displays, IR keys, effective-match reasons,
or alias evidence.  This module is the only place that turns a successful
proposal into ``ComparisonAnalysis`` EXACT/EFFECTIVE.
"""

from __future__ import annotations

from reccmp.compare.diagnosis import ComparisonAnalysis, InconclusiveReason


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
        return ComparisonAnalysis.inconclusive(InconclusiveReason.INCOMPLETE_COVERAGE)
    if not extent_closed:
        return ComparisonAnalysis.inconclusive(InconclusiveReason.OPEN_EXTENT)
    return analysis


def admit_exact_analysis(
    *,
    bytes_equal: bool,
    topology_equal: bool,
    keys_equal: bool,
    operands_complete: bool = True,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> ComparisonAnalysis | None:
    # pylint: disable=too-many-arguments
    """Shared EXACT admission policy for function comparison.

    This is the only gate that mints ``ComparisonStatus.EXACT``. Incomplete
    models require identical machine bytes; diagnostic text cannot substitute
    for missing operand or control-flow facts. Local branch topology and
    reference identities must also agree.
    """
    if coverage_incomplete or not extent_closed:
        return None
    if not (topology_equal and keys_equal and (bytes_equal or operands_complete)):
        return None
    return ComparisonAnalysis.exact()


def admit_effective(
    reasons,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool,
) -> ComparisonAnalysis | None:
    """Mint an EFFECTIVE analysis; strategies only propose reasons."""
    if coverage_incomplete or not extent_closed:
        return None
    return ComparisonAnalysis.effective(reasons)
