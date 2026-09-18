"""Stage 2: shared verification admission contract."""

from __future__ import annotations

from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.verification import admit_exact_analysis


def test_admit_exact_requires_coverage_topology_and_models_or_displays():
    assert admit_exact_analysis(displays_equal=True, topology_equal=True) is not None
    assert (
        admit_exact_analysis(displays_equal=True, topology_equal=False) is None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True, topology_equal=True, coverage_incomplete=True
        )
        is None
    )
    assert (
        admit_exact_analysis(
            displays_equal=False,
            topology_equal=True,
            keys_equal=True,
            operands_complete=False,
            control_flow_complete=True,
        )
        is None
    )
    admitted = admit_exact_analysis(
        displays_equal=False,
        topology_equal=True,
        keys_equal=True,
        operands_complete=True,
        control_flow_complete=True,
    )
    assert admitted is not None
    assert admitted.status == ComparisonStatus.EXACT
