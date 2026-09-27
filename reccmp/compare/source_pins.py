"""Attach recomp source lines and class-field facts to reported locations."""

import dataclasses

from reccmp.compare.comparator_state import ComparatorState
from reccmp.compare.db import ReccmpMatch
from reccmp.types import ImageId
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DifferenceKind,
    DifferenceSide,
    FieldAt,
    SourceLine,
    StopLocation,
    StrategyAttempt,
)
from reccmp.compare.asm.operand import Mem
from reccmp.source.observations import SourceIndexError


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
            recomp_side = self._with_recomp_source_line(diff.recomp)
            if diff.kind == DifferenceKind.BRANCH_CONDITION:
                recomp_side = self._with_source_comparisons(recomp_side, match)
            if diff.kind == DifferenceKind.MEMORY_ADDRESS:
                class_name = self._owning_class_for_match(match)
                orig_side = dataclasses.replace(
                    orig_side, field=self._field_at(class_name, orig_side)
                )
                recomp_side = dataclasses.replace(
                    recomp_side, field=self._field_at(class_name, recomp_side)
                )
            enriched = dataclasses.replace(diff, orig=orig_side, recomp=recomp_side)
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
                    self._with_source_line(location) if location is not None else None
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
                    difference=dataclasses.replace(
                        diff,
                        recomp=self._with_recomp_source_line(diff.recomp),
                    ),
                )
            elif attempt.location is not None:
                attempt = dataclasses.replace(
                    attempt, location=self._with_source_line(attempt.location)
                )
            enriched.append(attempt)
        return tuple(enriched)

    def _with_recomp_source_line(self, side: DifferenceSide) -> DifferenceSide:
        """A recompiled side with the source line of its address."""
        return dataclasses.replace(side, source=self._source_line(side.address))

    def _with_source_line(self, location: StopLocation) -> StopLocation:
        """The location with the recompiled source line of it, or of its
        recompiled counterpart; an orig address is never looked up in the
        recomp PDB."""
        source = self._source_line(
            location.address
            if location.image is ImageId.RECOMP
            else location.counterpart_address
        )
        return (
            location if source is None else dataclasses.replace(location, source=source)
        )

    def _source_line(self, recomp_addr: int | None) -> SourceLine | None:
        if recomp_addr is None:
            return None
        path_line_pair = self.lines_db.find_line_of_recomp_address(recomp_addr)
        if path_line_pair is None:
            return None
        return SourceLine(path_line_pair[0].name, path_line_pair[1])

    def _with_source_comparisons(
        self, side: DifferenceSide, match: ReccmpMatch | None
    ) -> DifferenceSide:
        """The comparisons the recompiled source makes on the differing
        branch's line, with the type each compares in: whether the source asks
        for a signed or an unsigned comparison, and at what width."""
        if (
            self.source_index is None
            or match is None
            or match.recomp_addr is None
            or not isinstance(side.address, int)
        ):
            return side
        # A branch rarely starts a statement: it belongs to the line entry
        # before it in its function.
        located = self.lines_db.find_line_containing_recomp_address(
            side.address, match.recomp_addr
        )
        if located is None:
            return side
        line = located[1]
        key = self.source_index.declaration_key_at(match.orig_addr)
        facts = self.source_index.function_facts_for(key) if key else None
        comparisons = facts.comparisons_on_line(line) if facts else ()
        if not comparisons:
            return side
        return dataclasses.replace(side, source_comparisons=tuple(comparisons))

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
            for declaration in self.source_index.declarations.values():
                if declaration.owning_class and declaration.qualified_name in {
                    match.name,
                    match.best_name(),
                }:
                    return declaration.owning_class
        name = match.best_name() or match.name or ""
        if "::" in name:
            return name.rsplit("::", 1)[0]
        return None

    def _field_at(self, class_name: str | None, side: DifferenceSide) -> FieldAt | None:
        """The field of ``class_name`` a side's memory operand displacement
        reaches, in the recovered layout."""
        match side.observed.operand:
            case Mem(displacement=displacement):
                pass
            case _:
                return None
        if (
            self.source_index is None
            or not class_name
            or not self.source_index.has_layout(class_name)
        ):
            return None
        resolved = self.source_index.resolve_field(class_name, displacement)
        if resolved is None:
            return None
        return FieldAt(
            resolved.root_class,
            tuple(resolved.path) or (resolved.leaf.name,),
            resolved.absolute_offset,
            resolved.leaf.type,
        )
