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
from reccmp.compare.extent import EntityExtent
from reccmp.types import EntityType, ImageId

_CODE_TYPES = (EntityType.FUNCTION, EntityType.THUNK, EntityType.VTORDISP)

if TYPE_CHECKING:
    from reccmp.compare.asm.ir import FunctionImage
    from reccmp.compare.db import ReccmpEntity, ReccmpMatch
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


def _solver_hints(analysis: ComparisonAnalysis) -> tuple[list[RunInput], str | None]:
    """An input under which the reported values differ, when Z3 finds one
    over leaves a run input can set (see witness.hints), and, when there is
    none, why: the solver's answer, or why its assignment is no input."""
    # The witness package needs unicorn, an optional extra.
    # pylint: disable=import-outside-toplevel
    from reccmp.compare.witness.hints import input_from_assignment
    from reccmp.compare.witness.machine import RunInput
    from reccmp.compare.witness.search import HINT_SEED

    difference = analysis.difference
    if difference is None or difference.values is None:
        return [], None
    outcome = bitvector.compare(difference.values)
    if outcome.result != "differs":
        detail = f": {outcome.reason}" if outcome.reason else ""
        return [], f"solver {outcome.result}{detail}"
    hint = input_from_assignment(
        dict(outcome.assignment), RunInput.from_seed(HINT_SEED)
    )
    if not isinstance(hint, RunInput):
        return [], f"rejected: {hint.reason}"
    return [hint], None


def _hint_fate(result: SearchResult, analysis: ComparisonAnalysis) -> str:
    """What the solver-suggested run did, when it refuted nothing."""
    reason, executed_o, executed_r = result.hint_runs[0]
    difference = analysis.difference
    assert difference is not None
    reached = (
        difference.orig.address is None or difference.orig.address in executed_o
    ) and (difference.recomp.address is None or difference.recomp.address in executed_r)
    return f"run {reason}" if reached else f"run {reason}, did not reach the difference"


class RefutationMixin(FunctionMetadataMixin, SourcePinMixin):
    """Part of FunctionComparator; relies on its attributes."""

    witness_search: bool
    _witness_translator: Translator | None
    # (side, address) -> extent from _witness_extent; filled on demand.
    _witness_extents: dict[tuple[ImageId, int], EntityExtent | None]

    def witness_translator(self) -> Translator:
        """The witness machines of this comparison, e.g. for replay."""
        return self._witness_machines()

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

            def windows(image_id: ImageId):
                def window(address: int) -> EntityExtent | None:
                    """Bytes a function at ``address`` may occupy: its
                    recorded size, else the gap to the next known entity."""
                    entity = self.db.get(image_id, address)
                    if entity is None:
                        return None
                    if (size := entity.size(image_id)) is not None:
                        return EntityExtent(size)
                    if (gap := entity.max_size(image_id)) is not None:
                        return EntityExtent(gap, recorded=False)
                    return None

                return window

            self._witness_translator = Translator(
                self.db,
                SideMachine(self.orig_bin, registry, by_import, windows(ImageId.ORIG)),
                SideMachine(
                    self.recomp_bin, registry, by_import, windows(ImageId.RECOMP)
                ),
                call_facts=callee_facts,
                extent=self._witness_extent,
            )
        return self._witness_translator

    def _witness_extent(
        self, image_id: ImageId, entity: ReccmpEntity
    ) -> EntityExtent | None:
        """An entity's size on one side, for the witness: the recorded one,
        else, for data of a pair, an estimate: the other side's size when it
        is exactly the gap to the next known entity on this side. The gap
        bounds the object without showing it owns every byte (there may be
        padding or an unrecorded object), so no witness may depend on it.

        A function's size is not needed: calls and pointers reach functions
        at their start, and callee cleanup comes from control flow."""
        size = entity.size(image_id)
        base = entity.addr(image_id)
        if size is not None:
            return EntityExtent(size)
        if base is None or not entity.matched:
            return None
        key = (image_id, base)
        if key not in self._witness_extents:
            other = ImageId.RECOMP if image_id == ImageId.ORIG else ImageId.ORIG
            other_size = entity.size(other)
            gap = entity.max_size(image_id)
            extent: EntityExtent | None = None
            if (
                other_size is not None
                and gap == other_size
                and entity.get("type") not in _CODE_TYPES
            ):
                extent = EntityExtent(other_size, recorded=False)
            self._witness_extents[key] = extent
        return self._witness_extents[key]

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
        hints, hint_failure = _solver_hints(analysis)
        result = find_witness(
            self._witness_machines(),
            orig_image,
            recomp_image,
            return_kind=facts.return_kind if facts is not None else "unknown",
            hints=hints,
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
                    solver_hint=(
                        _hint_fate(result, analysis) if hints else hint_failure
                    ),
                ),
            )
        refuted = analysis.with_witness(result.witness)
        if analysis.status == ComparisonStatus.INCONCLUSIVE:
            refuted = self._enrich_analysis_with_source(refuted, match=match)
        return refuted
