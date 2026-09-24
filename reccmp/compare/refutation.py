"""Try to refute unproven comparisons by differential execution.

Enabled with ``FunctionComparator.witness_search``; needs the optional
``unicorn`` dependency. A found witness turns a candidate mismatch into a
refuted one, and an inconclusive result into a refuted mismatch.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from reccmp.compare.call_cleanup import import_cleanup
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    ExecutionEvidence,
)
from reccmp.compare.function_metadata import FunctionMetadataMixin
from reccmp.compare.source_pins import SourcePinMixin

if TYPE_CHECKING:
    from reccmp.compare.db import ReccmpMatch
    from reccmp.compare.witness import SearchResult, Translator


def _reached(result: SearchResult, analysis: ComparisonAnalysis) -> int | None:
    """Agreeing runs that executed the reported location on both sides."""
    if analysis.difference is not None:
        orig = analysis.difference.orig.address
        recomp = analysis.difference.recomp.address
    elif analysis.inconclusive_location is not None:
        location = analysis.inconclusive_location
        recomp_fact = location.facts.get("recomp_address")
        if location.image == "recomp":
            orig, recomp = None, location.address
        else:
            orig = location.address
            recomp = recomp_fact if isinstance(recomp_fact, int) else None
    else:
        return None
    if orig is None and recomp is None:
        return None
    return result.agreeing_runs_through(orig, recomp)


class RefutationMixin(FunctionMetadataMixin, SourcePinMixin):
    """Part of FunctionComparator; relies on its attributes."""

    witness_search: bool
    _witness_translator: Translator | None

    def _witness_machines(self) -> Translator:
        if self._witness_translator is None:
            # Imported here: unicorn is optional and only needed when enabled.
            # pylint: disable-next=import-outside-toplevel
            from reccmp.compare.witness import SideMachine, Translator

            # Import names are the same in both binaries; the recompiled PDB
            # carries their decorations.
            cleanup = import_cleanup(
                node.decorated_name
                for node in self.func_nodes.values()
                if node.decorated_name is not None
            )
            self._witness_translator = Translator(
                self.db,
                SideMachine(self.orig_bin, cleanup),
                SideMachine(self.recomp_bin, cleanup),
            )
        return self._witness_translator

    def _refute(
        self,
        match: ReccmpMatch,
        analysis: ComparisonAnalysis,
        orig_size: int,
        recomp_size: int,
    ) -> ComparisonAnalysis:
        if not self.witness_search or analysis.status not in (
            ComparisonStatus.MISMATCH,
            ComparisonStatus.INCONCLUSIVE,
        ):
            return analysis
        # pylint: disable-next=import-outside-toplevel
        from reccmp.compare.witness import find_witness

        metadata = self._function_metadata(match)
        result = find_witness(
            self._witness_machines(),
            range(match.orig_addr, match.orig_addr + orig_size),
            range(match.recomp_addr, match.recomp_addr + recomp_size),
            return_kind=metadata.return_kind if metadata is not None else "unknown",
        )
        if result.witness is None:
            return dataclasses.replace(
                analysis,
                execution=ExecutionEvidence(
                    runs=result.runs,
                    agreeing=result.agreeing_seeds,
                    reached_location=_reached(result, analysis),
                    no_verdict=dict(result.skipped),
                ),
            )
        refuted = analysis.with_witness(result.witness)
        if analysis.status == ComparisonStatus.INCONCLUSIVE:
            refuted = self._enrich_analysis_with_source(refuted, match=match)
        return refuted
