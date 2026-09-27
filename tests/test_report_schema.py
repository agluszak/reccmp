"""The one supported report schema and its required semantic fields."""

import json

import pytest

from reccmp.compare.diagnosis import ComparisonAnalysis
from reccmp.compare.report import (
    ReccmpComparedEntity,
    ReccmpReportDeserializeError,
    ReccmpStatusReport,
    deserialize_reccmp_report,
    serialize_reccmp_report,
)
from reccmp.types import EntityType


def test_current_report_round_trip_keeps_type_and_varying_address():
    report = ReccmpStatusReport("game.exe")
    report.add_match(
        ReccmpComparedEntity(
            0x100,
            "function",
            0.75,
            type=EntityType.FUNCTION,
            recomp_addr_varies=True,
            analysis=ComparisonAnalysis.effective({"register_allocation"}),
        )
    )
    encoded = json.loads(serialize_reccmp_report(report))
    assert encoded["format"] == 2
    assert encoded["data"][0]["type"] == int(EntityType.FUNCTION)
    assert encoded["data"][0]["recomp_varies"] is True
    decoded = deserialize_reccmp_report(json.dumps(encoded))
    assert decoded.entities[0x100].analysis == report.entities[0x100].analysis
    assert decoded.entities[0x100].recomp_addr_varies


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update(format=1),
        lambda payload: payload["data"][0].pop("type"),
        lambda payload: payload["data"][0].pop("comparison"),
        lambda payload: payload["data"][0].update(effective=True),
        lambda payload: payload["data"][0].update(recomp="various"),
        lambda payload: payload["data"][0].update(recomp_varies=True),
    ],
)
def test_old_or_ambiguous_report_facts_are_rejected(change):
    report = ReccmpStatusReport("game.exe")
    report.add_match(
        ReccmpComparedEntity(
            0x100,
            "function",
            0.5,
            recomp_addr=0x200,
            analysis=ComparisonAnalysis.inconclusive("analysis_limit"),
        )
    )
    encoded = json.loads(serialize_reccmp_report(report))
    change(encoded)
    with pytest.raises(ReccmpReportDeserializeError):
        deserialize_reccmp_report(json.dumps(encoded))
