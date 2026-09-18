"""Group repeated memory_address mismatches that share layout enrichment facts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class LayoutShiftCluster:
    """Fields of one class that share the same orig→recomp displacement delta.

    Contiguous/affected field names are aggregated so a single inserted member
    (``delta`` bytes) can be reported as one layout-shift root cause.
    """

    class_name: str
    delta: int
    count: int
    fields: tuple[str, ...] = ()
    earliest_orig_disp: int = 0
    sample_addrs: tuple[int, ...] = ()


@dataclass
class _ShiftBucket:
    samples: list[tuple[int, int, str | None]] = field(default_factory=list)
    # (orig_addr, orig_disp, field_name)


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
        parsed = _layout_facts(entity)
        if parsed is None:
            continue
        class_name, orig_disp, recomp_disp, field_name, recomp_field = parsed
        buckets[(class_name, orig_disp, recomp_disp)].append(
            (entity.orig_addr, field_name, recomp_field)
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


def cluster_layout_shifts(
    entities: Iterable[_ComparedEntity],
    *,
    sample_limit: int = 8,
) -> list[LayoutShiftCluster]:
    """Second pass: group by ``(class_name, recomp_disp - orig_disp)``.

    Exact ``(class, orig, recomp)`` clusters from
    :func:`cluster_memory_address_mismatches` remain useful for pinpointing a
    single field; this pass surfaces a shared insertion/deletion delta across
    many fields of the same class.
    """
    buckets: dict[tuple[str, int], _ShiftBucket] = defaultdict(_ShiftBucket)
    for entity in entities:
        parsed = _layout_facts(entity)
        if parsed is None:
            continue
        class_name, orig_disp, recomp_disp, field_name, _recomp_field = parsed
        delta = recomp_disp - orig_disp
        buckets[(class_name, delta)].samples.append(
            (entity.orig_addr, orig_disp, field_name)
        )

    clusters: list[LayoutShiftCluster] = []
    for (class_name, delta), bucket in sorted(
        buckets.items(), key=lambda item: (-len(item[1].samples), item[0])
    ):
        samples = sorted(bucket.samples, key=lambda item: (item[1], item[0]))
        field_names: list[str] = []
        seen: set[str] = set()
        for _addr, _disp, name in samples:
            if name and name not in seen:
                seen.add(name)
                field_names.append(name)
        earliest = samples[0][1]
        addrs = tuple(addr for addr, _, _ in samples[:sample_limit])
        clusters.append(
            LayoutShiftCluster(
                class_name=class_name,
                delta=delta,
                count=len(samples),
                fields=tuple(field_names),
                earliest_orig_disp=earliest,
                sample_addrs=addrs,
            )
        )
    return clusters


def _layout_facts(
    entity: _ComparedEntity,
) -> tuple[str, int, int, str | None, str | None] | None:
    analysis = getattr(entity, "analysis", None)
    if analysis is None:
        return None
    if getattr(analysis, "status", None) != ComparisonStatus.MISMATCH:
        return None
    difference = getattr(analysis, "difference", None)
    if difference is None or difference.kind != "memory_address":
        return None
    orig_facts = difference.orig.facts
    recomp_facts = difference.recomp.facts
    class_name = orig_facts.get("class_name") or recomp_facts.get("class_name")
    orig_disp = orig_facts.get("displacement")
    recomp_disp = recomp_facts.get("displacement")
    if not isinstance(class_name, str) or not class_name:
        return None
    if not isinstance(orig_disp, int) or not isinstance(recomp_disp, int):
        return None
    field_name = orig_facts.get("field_name")
    recomp_field = recomp_facts.get("field_name")
    return (
        class_name,
        orig_disp,
        recomp_disp,
        field_name if isinstance(field_name, str) else None,
        recomp_field if isinstance(recomp_field, str) else None,
    )
