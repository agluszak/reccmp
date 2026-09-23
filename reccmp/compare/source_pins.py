"""Attach recomp source lines and class-field facts to reported locations."""

import dataclasses

from reccmp.compare.comparator_state import ComparatorState
from reccmp.compare.db import ReccmpMatch
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DifferenceSide,
    FactValue,
    StrategyAttempt,
)
from reccmp.source.index import SourceIndexError


class SourcePinMixin(ComparatorState):
    """Part of FunctionComparator; relies on its attributes."""

    def _enrich_analysis_with_source(
        self,
        analysis: ComparisonAnalysis,
        *,
        match: ReccmpMatch | None = None,
    ) -> ComparisonAnalysis:
        """Pin mismatch / inconclusive locations to recomp source lines.

        When layout is available, also annotate ``memory_address`` diffs with
        class/field facts for the owning class of the compared function.
        """
        if (
            analysis.status == ComparisonStatus.MISMATCH
            and analysis.difference is not None
        ):
            diff = analysis.difference
            orig_side = diff.orig
            recomp_side = self._enrich_side_with_source(diff.recomp, recomp=True)
            if diff.kind == "memory_address":
                class_name = self._owning_class_for_match(match)
                orig_layout = self._layout_facts_for_displacement(
                    class_name, orig_side.facts
                )
                recomp_layout = self._layout_facts_for_displacement(
                    class_name, recomp_side.facts
                )
                if orig_layout:
                    orig_side = dataclasses.replace(
                        orig_side, facts={**orig_side.facts, **orig_layout}
                    )
                if recomp_layout:
                    recomp_side = dataclasses.replace(
                        recomp_side, facts={**recomp_side.facts, **recomp_layout}
                    )
            enriched = ComparisonDifference(diff.kind, orig_side, recomp_side)
            return dataclasses.replace(
                analysis,
                difference=enriched,
                attempts=self._enrich_attempts_with_source(analysis.attempts),
            )
        if analysis.status == ComparisonStatus.INCONCLUSIVE:
            location = analysis.inconclusive_location
            return dataclasses.replace(
                analysis,
                inconclusive_location=(
                    self._enrich_side_with_source(location)
                    if location is not None
                    else None
                ),
                attempts=self._enrich_attempts_with_source(analysis.attempts),
            )
        return analysis

    def _enrich_attempts_with_source(
        self, attempts: tuple[StrategyAttempt, ...]
    ) -> tuple[StrategyAttempt, ...]:
        enriched: list[StrategyAttempt] = []
        for attempt in attempts:
            if attempt.difference is not None:
                diff = attempt.difference
                attempt = dataclasses.replace(
                    attempt,
                    difference=ComparisonDifference(
                        diff.kind,
                        diff.orig,
                        self._enrich_side_with_source(diff.recomp, recomp=True),
                    ),
                )
            elif attempt.location is not None:
                attempt = dataclasses.replace(
                    attempt, location=self._enrich_side_with_source(attempt.location)
                )
            enriched.append(attempt)
        return tuple(enriched)

    def _enrich_side_with_source(
        self, side: DifferenceSide, *, recomp: bool = False
    ) -> DifferenceSide:
        """Attach PDB line info for the recomp address of this location.

        Orig-side locations are pinned through their recorded recomp
        counterpart; an orig address is never looked up in the recomp PDB.
        """
        recomp_address = (
            side.address
            if recomp or side.image == "recomp"
            else side.facts.get("recomp_address")
        )
        if not isinstance(recomp_address, int) or isinstance(recomp_address, bool):
            return side
        extra = self._source_facts_of_recomp_addr(recomp_address)
        if not extra:
            return side
        return dataclasses.replace(side, facts={**side.facts, **extra})

    def _source_facts_of_recomp_addr(
        self, recomp_addr: int | None
    ) -> dict[str, FactValue]:
        if recomp_addr is None:
            return {}
        path_line_pair = self.lines_db.find_line_of_recomp_address(recomp_addr)
        if path_line_pair is None:
            return {}
        return {
            "source_path": path_line_pair[0].name,
            "source_line": path_line_pair[1],
        }

    def _owning_class_for_match(self, match: ReccmpMatch | None) -> str | None:
        """Resolve the class that owns ``this`` for layout enrichment."""
        if match is None:
            return None
        if self.source_index is not None:
            try:
                owners = self.source_index.functions_by_address()
            except SourceIndexError:
                owners = {}
            marker = owners.get(match.orig_addr)
            if (
                marker is not None
                and marker.declaration is not None
                and marker.declaration.owning_class
            ):
                return marker.declaration.owning_class
            for declaration in self.source_index.declarations:
                if declaration.owning_class and declaration.qualified_name in {
                    match.name,
                    match.best_name(),
                }:
                    return declaration.owning_class
        name = match.best_name() or match.name or ""
        if "::" in name:
            return name.rsplit("::", 1)[0]
        return None

    def _layout_facts_for_displacement(
        self, class_name: str | None, facts: dict[str, FactValue]
    ) -> dict[str, FactValue]:
        if (
            self.source_index is None
            or not class_name
            or not isinstance(facts.get("displacement"), int)
        ):
            return {}
        if not self.source_index.has_layout(class_name):
            return {}
        displacement = facts["displacement"]
        assert isinstance(displacement, int)
        resolved = self.source_index.resolve_field(class_name, displacement)
        if resolved is None:
            return {}
        return {
            "class_name": resolved.root_class,
            "field_name": resolved.leaf.name,
            "field_offset": resolved.absolute_offset,
            "field_type": resolved.leaf.type,
            "field_path": ".".join(resolved.path) or resolved.leaf.name,
        }
