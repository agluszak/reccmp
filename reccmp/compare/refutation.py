"""Try to refute unproven comparisons by differential execution.

Enabled with ``FunctionComparator.witness_search``; needs the optional
``unicorn`` dependency. A found witness turns a candidate mismatch into a
refuted one, and an inconclusive result into a refuted mismatch.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from reccmp.call_facts import CallFacts
from reccmp.compare.call_facts import import_facts
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    ExecutionEvidence,
)
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.function_metadata import FunctionMetadataMixin
from reccmp.compare.source_pins import SourcePinMixin
from reccmp.types import ImageId

if TYPE_CHECKING:
    from reccmp.compare.asm.ir import FunctionImage
    from reccmp.compare.db import ReccmpMatch
    from reccmp.compare.witness import SearchResult, Translator
    from reccmp.compare.witness.machine import RunInput


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


def _solver_hints(analysis: ComparisonAnalysis) -> list[RunInput]:
    """An input under which the reported values differ, when Z3 finds one
    over leaves a run input can set (see witness.hints)."""
    # The witness package needs unicorn, an optional extra.
    # pylint: disable=import-outside-toplevel
    from reccmp.compare.witness.hints import input_from_assignment
    from reccmp.compare.witness.machine import RunInput
    from reccmp.compare.witness.search import HINT_SEED

    difference = analysis.difference
    if difference is None or difference.values is None:
        return []
    assignment = bitvector.distinguishing_assignment(difference.values)
    if not assignment:
        return []
    hint = input_from_assignment(assignment, RunInput.from_seed(HINT_SEED))
    return [hint] if hint is not None else []


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
                if identity[0] == "jmp_through":
                    return callee_facts(identity[1])
                if identity[0] == "import":
                    return by_import.get(identity[1].split("!", 1)[1])
                if identity[0] == "entity" and identity[2] == 0:
                    match = self.db.get_one_match(identity[1])
                    if match is not None and match.recomp_addr is not None:
                        return self._call_facts_at(match.recomp_addr, match.orig_addr)
                return None

            def sizes(image_id: ImageId):
                def size(address: int) -> int | None:
                    entity = self.db.get(image_id, address)
                    return entity.size(image_id) if entity is not None else None

                return size

            self._witness_translator = Translator(
                self.db,
                SideMachine(self.orig_bin, registry, by_import, sizes(ImageId.ORIG)),
                SideMachine(
                    self.recomp_bin, registry, by_import, sizes(ImageId.RECOMP)
                ),
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

        facts = self._call_facts_at(match.recomp_addr, match.orig_addr)
        result = find_witness(
            self._witness_machines(),
            orig_image,
            recomp_image,
            return_kind=facts.return_kind if facts is not None else "unknown",
            hints=_solver_hints(analysis),
        )
        if result.witness is None:
            return dataclasses.replace(
                analysis,
                execution=ExecutionEvidence(
                    runs=result.runs,
                    agreeing=result.agreeing_seeds,
                    reached_location=_reached(result, analysis),
                    no_verdict=dict(result.skipped),
                    no_verdict_details=result.skipped_details,
                ),
            )
        refuted = analysis.with_witness(result.witness)
        if analysis.status == ComparisonStatus.INCONCLUSIVE:
            refuted = self._enrich_analysis_with_source(refuted, match=match)
        return refuted
