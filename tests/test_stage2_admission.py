"""Stage 2: shared verification admission contract."""

from __future__ import annotations

import pytest

from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.verification import admit_effective, admit_exact_analysis


def test_admit_exact_requires_coverage_topology_and_models_or_displays():
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=True,
            extent_closed=True,
        )
        is not None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=False,
            keys_equal=True,
            extent_closed=True,
        )
        is None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=True,
            coverage_incomplete=True,
            extent_closed=True,
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
            extent_closed=True,
        )
        is None
    )
    admitted = admit_exact_analysis(
        displays_equal=False,
        topology_equal=True,
        keys_equal=True,
        operands_complete=True,
        control_flow_complete=True,
        extent_closed=True,
    )
    assert admitted is not None
    assert admitted.status == ComparisonStatus.EXACT


def test_admit_exact_does_not_default_keys_equal():
    with pytest.raises(TypeError):
        admit_exact_analysis(
            displays_equal=True, topology_equal=True, extent_closed=True
        )


def test_admit_effective_requires_closed_extent():
    with pytest.raises(TypeError):
        admit_effective({"register_allocation"})
    minted = admit_effective(
        {"register_allocation"}, coverage_incomplete=False, extent_closed=True
    )
    assert minted is not None
    assert minted.proof_kind == ComparisonStatus.EFFECTIVE
    assert (
        admit_effective(
            {"register_allocation"}, coverage_incomplete=True, extent_closed=True
        )
        is None
    )
