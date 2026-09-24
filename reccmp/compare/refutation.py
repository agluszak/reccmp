"""Try to refute unproven comparisons by differential execution.

Enabled with ``FunctionComparator.witness_search``; needs the optional
``unicorn`` dependency. A found witness turns a candidate mismatch into a
refuted one, and an inconclusive result into a refuted mismatch.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from reccmp.compare.call_facts import CallFacts, import_facts
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    ExecutionEvidence,
)
from reccmp.compare.function_metadata import FunctionMetadataMixin
from reccmp.compare.source_pins import SourcePinMixin

if TYPE_CHECKING:
    from reccmp.compare.asm.ir import FunctionImage
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

            # pylint: disable-next=import-outside-toplevel
            from reccmp.compare.witness.machine import import_registry

            # Import names are the same in both binaries; the recompiled PDB
            # carries their decorations.
            by_import = import_facts(
                node.decorated_name
                for node in self.func_nodes.values()
                if node.decorated_name is not None
            )
            registry = import_registry(self.orig_bin, self.recomp_bin)

            def callee_facts(identity) -> CallFacts | None:
                if identity[0] == "import":
                    return by_import.get(identity[1].split("!", 1)[1])
                if identity[0] == "entity" and identity[2] == 0:
                    match = self.db.get_one_match(identity[1])
                    if match is not None and match.recomp_addr is not None:
                        return self._call_facts_at(match.recomp_addr)
                return None

            self._witness_translator = Translator(
                self.db,
                SideMachine(self.orig_bin, registry, by_import),
                SideMachine(self.recomp_bin, registry, by_import),
                call_facts=callee_facts,
            )
        return self._witness_translator

    def _refute(
        self,
        match: ReccmpMatch,
        analysis: ComparisonAnalysis,
        orig_image: FunctionImage,
        recomp_image: FunctionImage,
    ) -> ComparisonAnalysis:
        if not self.witness_search or analysis.status not in (
            ComparisonStatus.MISMATCH,
            ComparisonStatus.INCONCLUSIVE,
        ):
            return analysis
        # pylint: disable-next=import-outside-toplevel
        from reccmp.compare.witness import find_witness

        facts = self._call_facts_at(match.recomp_addr)
        result = find_witness(
            self._witness_machines(),
            orig_image,
            recomp_image,
            return_kind=facts.return_kind if facts is not None else "unknown",
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
