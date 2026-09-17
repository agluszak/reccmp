"""Stable structured results for semantic comparison diagnosis.

This module intentionally contains only the small, generic vocabulary shared by
the verifier, JSON reports, and downstream tools.  Symbolic execution and report
formatting remain in their existing layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

FactValue: TypeAlias = str | int | bool | None


class ComparisonStatus(Enum):
    EXACT = "exact"
    EFFECTIVE = "effective"
    MISMATCH = "mismatch"
    INCONCLUSIVE = "inconclusive"


class DiagnosticNormalization(Enum):
    """Non-proof accounting tags explaining residual compiler entropy.

    These must never be read as semantic equivalence.  Only
    ``ComparisonStatus.EXACT`` / ``EFFECTIVE`` are proofs.  Multiple tags may
    apply at once (register allocation and scheduling are orthogonal).
    """

    STACK_LAYOUT = "stack_layout"
    REGISTER_ALLOCATION = "register_allocation"
    INSTRUCTION_SCHEDULING = "instruction_scheduling"
    CFG_LAYOUT = "cfg_layout"
    KNOWN_INLINE = "known_inline"
    FOLDED_SYMBOL_ALIAS = "folded_symbol_alias"


# Backward-compatible aliases for the previous single-valued lattice.
# Kept so older reports/tests that still mention EquivalenceLevel compile;
# new code should use DiagnosticNormalization + ComparisonStatus.
class EquivalenceLevel(Enum):
    """Deprecated single-valued view; prefer ComparisonStatus + normalizations."""

    EXACT_INSTRUCTIONS = "exact_instructions"
    STACK_LAYOUT_EQUIVALENT = "stack_layout"  # renamed: not a proof
    REGISTER_ALLOCATION_EQUIVALENT = "register_allocation"
    INSTRUCTION_SCHEDULING_EQUIVALENT = "instruction_scheduling"
    CFG_LAYOUT_EQUIVALENT = "cfg_layout"
    KNOWN_INLINE_EQUIVALENT = "known_inline"
    FOLDED_SYMBOL_ALIAS = "folded_symbol_alias"
    UNKNOWN_DIFFERENCE = "unknown_difference"


_LEGACY_LEVEL_TO_NORMALIZATION = {
    "stack_layout_equivalent": DiagnosticNormalization.STACK_LAYOUT,
    "stack_layout": DiagnosticNormalization.STACK_LAYOUT,
    "register_allocation_equivalent": DiagnosticNormalization.REGISTER_ALLOCATION,
    "register_allocation": DiagnosticNormalization.REGISTER_ALLOCATION,
    "instruction_scheduling_equivalent": DiagnosticNormalization.INSTRUCTION_SCHEDULING,
    "instruction_scheduling": DiagnosticNormalization.INSTRUCTION_SCHEDULING,
    "cfg_layout_equivalent": DiagnosticNormalization.CFG_LAYOUT,
    "cfg_layout": DiagnosticNormalization.CFG_LAYOUT,
    "known_inline_equivalent": DiagnosticNormalization.KNOWN_INLINE,
    "known_inline": DiagnosticNormalization.KNOWN_INLINE,
    "folded_symbol_alias": DiagnosticNormalization.FOLDED_SYMBOL_ALIAS,
}


def derive_diagnostic_normalizations(
    analysis: "ComparisonAnalysis",
    *,
    accuracy_modulo_stack: float | None = None,
    accuracy_modulo_inline: float | None = None,
) -> tuple[DiagnosticNormalization, ...]:
    """Collect orthogonal diagnostic tags.  Never implies a proof."""
    tags: set[DiagnosticNormalization] = set()

    if analysis.status == ComparisonStatus.EFFECTIVE:
        reasons = set(analysis.effective_reasons)
        if "condition_inversion" in reasons:
            tags.add(DiagnosticNormalization.CFG_LAYOUT)
        if reasons & {"instruction_reorder", "commutative_order", "load_folding"}:
            tags.add(DiagnosticNormalization.INSTRUCTION_SCHEDULING)
        if reasons & {"register_allocation", "callee_save_substitution"}:
            tags.add(DiagnosticNormalization.REGISTER_ALLOCATION)
        if "frame_slot_layout" in reasons:
            tags.add(DiagnosticNormalization.STACK_LAYOUT)
        if "folded_symbol_alias" in reasons:
            tags.add(DiagnosticNormalization.FOLDED_SYMBOL_ALIAS)

    # Modulo scores are diagnostic collapses, not proofs of equivalence.
    if accuracy_modulo_inline is not None and accuracy_modulo_inline >= 1.0:
        tags.add(DiagnosticNormalization.KNOWN_INLINE)
    if accuracy_modulo_stack is not None and accuracy_modulo_stack >= 1.0:
        tags.add(DiagnosticNormalization.STACK_LAYOUT)

    order = (
        DiagnosticNormalization.STACK_LAYOUT,
        DiagnosticNormalization.REGISTER_ALLOCATION,
        DiagnosticNormalization.INSTRUCTION_SCHEDULING,
        DiagnosticNormalization.CFG_LAYOUT,
        DiagnosticNormalization.KNOWN_INLINE,
        DiagnosticNormalization.FOLDED_SYMBOL_ALIAS,
    )
    return tuple(tag for tag in order if tag in tags)


def derive_equivalence_level(
    analysis: "ComparisonAnalysis",
    *,
    accuracy_modulo_stack: float | None = None,
    accuracy_modulo_inline: float | None = None,
) -> EquivalenceLevel:
    """Deprecated compatibility shim over proof status + normalizations."""
    if analysis.status == ComparisonStatus.EXACT:
        return EquivalenceLevel.EXACT_INSTRUCTIONS

    norms = derive_diagnostic_normalizations(
        analysis,
        accuracy_modulo_stack=accuracy_modulo_stack,
        accuracy_modulo_inline=accuracy_modulo_inline,
    )
    if not norms:
        return EquivalenceLevel.UNKNOWN_DIFFERENCE

    # Prefer a primary tag for legacy single-valued consumers.  folded_symbol
    # is never reported as known_inline.
    primary = norms[0]
    mapping = {
        DiagnosticNormalization.STACK_LAYOUT: EquivalenceLevel.STACK_LAYOUT_EQUIVALENT,
        DiagnosticNormalization.REGISTER_ALLOCATION: (
            EquivalenceLevel.REGISTER_ALLOCATION_EQUIVALENT
        ),
        DiagnosticNormalization.INSTRUCTION_SCHEDULING: (
            EquivalenceLevel.INSTRUCTION_SCHEDULING_EQUIVALENT
        ),
        DiagnosticNormalization.CFG_LAYOUT: EquivalenceLevel.CFG_LAYOUT_EQUIVALENT,
        DiagnosticNormalization.KNOWN_INLINE: EquivalenceLevel.KNOWN_INLINE_EQUIVALENT,
        DiagnosticNormalization.FOLDED_SYMBOL_ALIAS: EquivalenceLevel.FOLDED_SYMBOL_ALIAS,
    }
    # Prefer known_inline / folded over stack when both present for legacy primary.
    for preferred in (
        DiagnosticNormalization.FOLDED_SYMBOL_ALIAS,
        DiagnosticNormalization.KNOWN_INLINE,
        DiagnosticNormalization.CFG_LAYOUT,
        DiagnosticNormalization.INSTRUCTION_SCHEDULING,
        DiagnosticNormalization.REGISTER_ALLOCATION,
        DiagnosticNormalization.STACK_LAYOUT,
    ):
        if preferred in norms:
            return mapping[preferred]
    return mapping[primary]


EFFECTIVE_REASON_ORDER = (
    "register_allocation",
    "frame_slot_layout",
    "callee_save_substitution",
    "instruction_reorder",
    "commutative_order",
    "condition_inversion",
    "load_folding",
    "dead_operation",
    "padding",
    # The original function is a stale incremental-link jmp island whose fold
    # chain lands on a proven-equivalent shared body (configured via the
    # project's equivalence-groups metadata); the recomp emits the real body.
    "folded_symbol_alias",
)

EFFECTIVE_REASONS = frozenset(EFFECTIVE_REASON_ORDER)

MISMATCH_KINDS = frozenset(
    {
        "call_target",
        "call_argument",
        "memory_address",
        "memory_value",
        "immediate_value",
        "branch_condition",
        "branch_target",
        "return_value",
        "preserved_state",
        "symbol_resolution",
    }
)

INCONCLUSIVE_REASONS = frozenset(
    {
        "unsupported_instruction",
        # Legacy report compatibility. New analyses emit a specific control-flow
        # reason below instead of this umbrella value.
        "unsupported_control_flow",
        "empty_control_flow",
        "control_flow_metadata_mismatch",
        "invalid_control_flow_target",
        "jump_table_data",
        "non_isomorphic_cfg",
        "indirect_jump",
        "external_control_flow_state",
        "function_fallthrough",
        "state_join_failure",
        "alignment_failure",
        "missing_metadata",
        "analysis_limit",
    }
)


def normalize_effective_reasons(reasons) -> tuple[str, ...]:
    """Validate, deduplicate, and order the fixed reason vocabulary."""
    values = set(reasons)
    unknown = values - EFFECTIVE_REASONS
    if unknown:
        raise ValueError(f"Unknown effective reasons: {sorted(unknown)}")
    return tuple(reason for reason in EFFECTIVE_REASON_ORDER if reason in values)


@dataclass(frozen=True)
class StackPermutationEntry:
    """One orig → recomp local slot correspondence."""

    orig: str
    recomp: str
    symbol: str | None = None


@dataclass(frozen=True)
class DifferenceSide:
    instruction_index: int | None = None
    address: int | None = None
    facts: dict[str, FactValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonDifference:
    kind: str
    orig: DifferenceSide
    recomp: DifferenceSide

    def __post_init__(self) -> None:
        if self.kind not in MISMATCH_KINDS:
            raise ValueError(f"Unknown mismatch kind: {self.kind}")


@dataclass(frozen=True)
class ComparisonAnalysis:
    status: ComparisonStatus
    effective_reasons: tuple[str, ...] = ()
    difference: ComparisonDifference | None = None
    inconclusive_reason: str | None = None
    inconclusive_location: DifferenceSide | None = None
    # Diagnostic repair-distance estimate.  This is deliberately separate
    # from status: only EXACT/EFFECTIVE are proofs of semantic equivalence.
    semantic_similarity: float | None = None

    def __post_init__(self) -> None:
        normalized = normalize_effective_reasons(self.effective_reasons)
        object.__setattr__(self, "effective_reasons", normalized)
        if self.status != ComparisonStatus.EFFECTIVE and normalized:
            raise ValueError("Only effective results carry proof reasons")
        if self.status == ComparisonStatus.MISMATCH and self.difference is None:
            raise ValueError("A mismatch must include a concrete difference")
        if self.status != ComparisonStatus.MISMATCH and self.difference is not None:
            raise ValueError("Only mismatch results carry a difference")
        if self.status == ComparisonStatus.INCONCLUSIVE:
            if self.inconclusive_reason not in INCONCLUSIVE_REASONS:
                raise ValueError(
                    f"Unknown inconclusive reason: {self.inconclusive_reason}"
                )
        elif self.inconclusive_reason is not None:
            raise ValueError("Only inconclusive results carry an inconclusive reason")
        if (
            self.status != ComparisonStatus.INCONCLUSIVE
            and self.inconclusive_location is not None
        ):
            raise ValueError("Only inconclusive results carry an analysis location")
        if self.semantic_similarity is not None:
            if not 0.0 <= self.semantic_similarity <= 1.0:
                raise ValueError("Semantic similarity must be between zero and one")
            if self.status == ComparisonStatus.INCONCLUSIVE:
                raise ValueError(
                    "Inconclusive results cannot carry semantic similarity"
                )
            if (
                self.status == ComparisonStatus.MISMATCH
                and self.semantic_similarity == 1.0
            ):
                raise ValueError("A mismatch cannot have 100% semantic similarity")

    @property
    def is_effective(self) -> bool:
        return self.status in (ComparisonStatus.EXACT, ComparisonStatus.EFFECTIVE)

    @classmethod
    def exact(cls) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.EXACT, semantic_similarity=1.0)

    @classmethod
    def effective(cls, reasons) -> "ComparisonAnalysis":
        return cls(
            ComparisonStatus.EFFECTIVE,
            tuple(reasons),
            semantic_similarity=1.0,
        )

    @classmethod
    def mismatch(
        cls,
        difference: ComparisonDifference,
        semantic_similarity: float | None = None,
    ) -> "ComparisonAnalysis":
        return cls(
            ComparisonStatus.MISMATCH,
            difference=difference,
            semantic_similarity=semantic_similarity,
        )

    @classmethod
    def inconclusive(
        cls, reason: str, location: DifferenceSide | None = None
    ) -> "ComparisonAnalysis":
        return cls(
            ComparisonStatus.INCONCLUSIVE,
            inconclusive_reason=reason,
            inconclusive_location=location,
        )


@dataclass
class AnalysisRecorder:
    """Mutable evidence sink used by one speculative verifier strategy."""

    orig_addrs: list[int | None] | None = None
    recomp_addrs: list[int | None] | None = None
    reasons: set[str] = field(default_factory=set)
    difference: ComparisonDifference | None = None
    candidate_difference: ComparisonDifference | None = None
    inconclusive_reason: str | None = None
    inconclusive_location: DifferenceSide | None = None

    def side(
        self, which: str, instruction_index: int | None, facts: dict[str, FactValue]
    ) -> DifferenceSide:
        addrs = self.orig_addrs if which == "orig" else self.recomp_addrs
        address = None
        if addrs is not None and instruction_index is not None:
            if 0 <= instruction_index < len(addrs):
                address = addrs[instruction_index]
        return DifferenceSide(instruction_index, address, facts)

    def record_difference(
        self,
        kind: str,
        orig_index: int | None,
        recomp_index: int | None,
        orig_facts: dict[str, FactValue],
        recomp_facts: dict[str, FactValue],
        *,
        candidate: bool = False,
    ) -> None:
        difference = ComparisonDifference(
            kind,
            self.side("orig", orig_index, orig_facts),
            self.side("recomp", recomp_index, recomp_facts),
        )
        if candidate:
            if self.candidate_difference is None:
                self.candidate_difference = difference
        elif self.difference is None:
            self.difference = difference

    def mark_inconclusive(
        self,
        reason: str,
        orig_index: int | None = None,
        recomp_index: int | None = None,
        facts: dict[str, FactValue] | None = None,
    ) -> None:
        if reason not in INCONCLUSIVE_REASONS:
            raise ValueError(f"Unknown inconclusive reason: {reason}")
        if self.inconclusive_reason is None:
            self.inconclusive_reason = reason
            detail = dict(facts or {})
            if orig_index is not None or (recomp_index is None and detail):
                self.inconclusive_location = self.side("orig", orig_index, detail)
            elif recomp_index is not None:
                self.inconclusive_location = self.side("recomp", recomp_index, detail)

    @property
    def best_difference(self) -> ComparisonDifference | None:
        if self.difference is None:
            return self.candidate_difference
        if self.candidate_difference is None:
            return self.difference
        if self.difference.kind in {
            "call_target",
            "call_argument",
            "branch_condition",
            "branch_target",
        }:
            return self.difference
        if (
            self.difference.kind == "return_value"
            and self.candidate_difference.kind == "immediate_value"
        ):
            return self.difference
        concrete_index = self.difference.orig.instruction_index
        candidate_index = self.candidate_difference.orig.instruction_index
        if candidate_index is not None and (
            concrete_index is None or candidate_index < concrete_index
        ):
            return self.candidate_difference
        return self.difference

    def effective_analysis(self, extra_reasons=()) -> ComparisonAnalysis:
        return ComparisonAnalysis.effective(self.reasons | set(extra_reasons))

    def failure_analysis(self) -> ComparisonAnalysis:
        if self.best_difference is not None:
            return ComparisonAnalysis.mismatch(self.best_difference)
        return ComparisonAnalysis.inconclusive(
            self.inconclusive_reason or "analysis_limit", self.inconclusive_location
        )
