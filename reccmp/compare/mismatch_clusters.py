"""Group repeated memory_address mismatches that share layout enrichment facts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol

from reccmp.compare.diagnosis import ComparisonStatus


class _ComparedEntity(Protocol):
    orig_addr: int
    analysis: object


@dataclass(frozen=True)
class MemoryAddressCluster:
    """One (class, orig_disp, recomp_disp) bucket of layout-enriched mismatches."""

    class_name: str
    orig_disp: int
    recomp_disp: int
    count: int
    sample_addrs: tuple[int, ...]
    field_name: str | None = None
    recomp_field_name: str | None = None


def cluster_memory_address_mismatches(
    entities: Iterable[_ComparedEntity],
    *,
    sample_limit: int = 8,
) -> list[MemoryAddressCluster]:
    """Cluster mismatch entities that carry class/field layout enrichment.

    Only ``memory_address`` differences with ``class_name`` and integer
    displacements on both sides are grouped. Entities without enrichment are
    ignored.
    """
    buckets: dict[tuple[str, int, int], list[tuple[int, str | None, str | None]]] = (
        defaultdict(list)
    )
    for entity in entities:
        analysis = getattr(entity, "analysis", None)
        if analysis is None:
            continue
        if getattr(analysis, "status", None) != ComparisonStatus.MISMATCH:
            continue
        difference = getattr(analysis, "difference", None)
        if difference is None or difference.kind != "memory_address":
            continue
        orig_facts = difference.orig.facts
        recomp_facts = difference.recomp.facts
        class_name = orig_facts.get("class_name") or recomp_facts.get("class_name")
        orig_disp = orig_facts.get("displacement")
        recomp_disp = recomp_facts.get("displacement")
        if not isinstance(class_name, str) or not class_name:
            continue
        if not isinstance(orig_disp, int) or not isinstance(recomp_disp, int):
            continue
        field_name = orig_facts.get("field_name")
        recomp_field = recomp_facts.get("field_name")
        buckets[(class_name, orig_disp, recomp_disp)].append(
            (
                entity.orig_addr,
                field_name if isinstance(field_name, str) else None,
                recomp_field if isinstance(recomp_field, str) else None,
            )
        )

    clusters: list[MemoryAddressCluster] = []
    for (class_name, orig_disp, recomp_disp), samples in sorted(
        buckets.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        addrs = tuple(addr for addr, _, _ in samples[:sample_limit])
        field_name = next((name for _, name, _ in samples if name), None)
        recomp_field = next((name for _, _, name in samples if name), None)
        clusters.append(
            MemoryAddressCluster(
                class_name=class_name,
                orig_disp=orig_disp,
                recomp_disp=recomp_disp,
                count=len(samples),
                sample_addrs=addrs,
                field_name=field_name,
                recomp_field_name=recomp_field,
            )
        )
    return clusters
