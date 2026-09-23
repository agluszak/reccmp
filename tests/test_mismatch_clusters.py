"""Tests for memory_address mismatch clustering."""

from types import SimpleNamespace

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceSide,
)
from reccmp.compare.mismatch_clusters import (
    cluster_layout_shifts,
    cluster_memory_address_mismatches,
)


def _entity(
    addr: int,
    class_name: str,
    orig_disp: int,
    recomp_disp: int,
    field: str,
    *,
    recomp_field: str | None = None,
):
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
                    "field_name": recomp_field or field,
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


def test_cluster_layout_shifts_groups_by_class_and_delta():
    """A shared +4 shift across fields of Foo collapses to one shift cluster."""
    entities = [
        _entity(0x100, "Foo", 0x10, 0x14, "a"),
        _entity(0x200, "Foo", 0x20, 0x24, "b"),
        _entity(0x300, "Foo", 0x30, 0x34, "c"),
        # Different delta — separate cluster.
        _entity(0x400, "Foo", 0x40, 0x48, "d"),
        # Different class.
        _entity(0x500, "Bar", 0x10, 0x14, "e"),
    ]
    shifts = cluster_layout_shifts(entities)
    assert len(shifts) == 3
    top = shifts[0]
    assert top.class_name == "Foo"
    assert top.delta == 4
    assert top.count == 3
    assert top.fields == ("a", "b", "c")
    assert top.earliest_orig_disp == 0x10
    # Exact clusters still available separately.
    exact = cluster_memory_address_mismatches(entities)
    assert len(exact) == 5
