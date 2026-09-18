"""Tests for memory_address mismatch clustering."""

from types import SimpleNamespace

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceSide,
)
from reccmp.compare.mismatch_clusters import cluster_memory_address_mismatches


def _entity(addr: int, class_name: str, orig_disp: int, recomp_disp: int, field: str):
    analysis = ComparisonAnalysis.mismatch(
        ComparisonDifference(
            "memory_address",
            DifferenceSide(
                0,
                addr,
                {
                    "displacement": orig_disp,
                    "class_name": class_name,
                    "field_name": field,
                },
            ),
            DifferenceSide(
                1,
                addr + 0x1000,
                {
                    "displacement": recomp_disp,
                    "class_name": class_name,
                    "field_name": field,
                },
            ),
        )
    )
    return SimpleNamespace(orig_addr=addr, analysis=analysis)


def test_cluster_memory_address_mismatches_groups_by_layout_key():
    entities = [
        _entity(0x100, "Foo", 0x94, 0x98, "x"),
        _entity(0x200, "Foo", 0x94, 0x98, "x"),
        _entity(0x300, "Foo", 0x10, 0x14, "y"),
        _entity(0x400, "Bar", 0x94, 0x98, "z"),
        # No enrichment — ignored.
        SimpleNamespace(
            orig_addr=0x500,
            analysis=ComparisonAnalysis.mismatch(
                ComparisonDifference(
                    "memory_address",
                    DifferenceSide(0, 0x500, {"displacement": 4}),
                    DifferenceSide(1, 0x1500, {"displacement": 8}),
                )
            ),
        ),
    ]
    clusters = cluster_memory_address_mismatches(entities)
    assert len(clusters) == 3
    top = clusters[0]
    assert top.class_name == "Foo"
    assert top.orig_disp == 0x94
    assert top.recomp_disp == 0x98
    assert top.count == 2
    assert top.field_name == "x"
    assert set(top.sample_addrs) == {0x100, 0x200}
