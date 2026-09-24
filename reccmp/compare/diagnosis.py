"""Stable structured results for semantic comparison diagnosis.

This module intentionally contains only the small, generic vocabulary shared by
the verifier, JSON reports, and downstream tools.  Symbolic execution and report
formatting remain in their existing layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal, TypeAlias

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


EFFECTIVE_REASON_ORDER = (
    "register_allocation",
    "frame_slot_layout",
    "callee_save_substitution",
    "instruction_reorder",
    "commutative_order",
    "condition_inversion",
    "load_folding",
    "dead_operation",
    # Values proven equal as bit-vectors (z3) though computed differently.
    "algebraic_identity",
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
        "incomplete_coverage",
        "open_extent",
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
    # Which binary instruction_index/address refer to, when known.
    image: Literal["orig", "recomp"] | None = None


@dataclass(frozen=True)
class ComparisonDifference:
    kind: str
    orig: DifferenceSide
    recomp: DifferenceSide
    # The verifier's symbolic values behind a value difference: (value_orig,
    # value_recomp, bits, "value" or "predicate"). In memory only (for the
    # witness to ask a solver for a distinguishing input), never reported.
    values: tuple | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in MISMATCH_KINDS:
            raise ValueError(f"Unknown mismatch kind: {self.kind}")


STRATEGIES = ("lockstep", "diff_aligned", "relocation", "cfg", "isomorphic_cfg")

# Strategies whose instruction pairing is anchored by position or by matched
# CFG blocks. The others pair instructions heuristically (diff opcodes,
# undone relocations), so a difference they report may be an artifact of the
# pairing rather than of the code.
TRUSTED_ALIGNMENT_STRATEGIES = frozenset({"lockstep", "cfg", "isomorphic_cfg"})


@dataclass(frozen=True)
class StrategyAttempt:
    """Where one verifier strategy stopped: a difference or a blocker."""

    strategy: str
    difference: ComparisonDifference | None = None
    blocker: str | None = None
    location: DifferenceSide | None = None

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if (self.difference is None) == (self.blocker is None):
            raise ValueError("An attempt has exactly one of difference or blocker")
        if self.blocker is not None and self.blocker not in INCONCLUSIVE_REASONS:
            raise ValueError(f"Unknown inconclusive reason: {self.blocker}")
        if self.location is not None and self.blocker is None:
            raise ValueError("Only blocked attempts carry a location")

    @property
    def trusted_alignment(self) -> bool:
        return self.strategy in TRUSTED_ALIGNMENT_STRATEGIES


WITNESS_KINDS = {
    "memory_value": "memory_value",
    "return_value": "return_value",
    "call_argument": "call_argument",
    "stack_cleanup": "preserved_state",
}


@dataclass(frozen=True)
class RefutationWitness:
    """Concrete inputs under which the two functions observably differ.

    Found by executing both bodies from the same state (see
    reccmp.compare.witness). Callees are modelled, not run, and the input
    need not be reachable from the program's real callers.
    """

    seed: int
    kind: str  # key of WITNESS_KINDS
    location: str
    orig_value: str
    recomp_value: str
    orig_address: int | None = None
    recomp_address: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in WITNESS_KINDS:
            raise ValueError(f"Unknown witness kind: {self.kind}")


@dataclass(frozen=True)
class ExecutionEvidence:
    """What differential execution saw when it found no witness.

    Not a proof: agreeing runs only cover the inputs tried. Runs that reached
    the reported difference or blocker and still agreed suggest the verifier
    is missing an equivalence there.
    """

    runs: int
    agreeing: int
    # Agreeing runs that executed the reported location on both sides; None
    # when the result has no location.
    reached_location: int | None = None
    # Why the other runs gave no verdict, e.g. {"foreign_access": 3}.
    no_verdict: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonAnalysis:
    # pylint: disable=too-many-instance-attributes
    status: ComparisonStatus
    effective_reasons: tuple[str, ...] = ()
    difference: ComparisonDifference | None = None
    inconclusive_reason: str | None = None
    inconclusive_location: DifferenceSide | None = None
    # Every strategy that ran, in execution order. The primary difference or
    # reason above is chosen from these; the rest show what else blocked.
    attempts: tuple[StrategyAttempt, ...] = ()
    # Only on mismatches: a demonstrated difference. A mismatch without one
    # is a candidate located by the verifier, not a refutation.
    witness: RefutationWitness | None = None
    # Differential execution that found no witness (mismatch/inconclusive).
    execution: ExecutionEvidence | None = None

    def __post_init__(self) -> None:
        if self.witness is not None and self.status != ComparisonStatus.MISMATCH:
            raise ValueError("Only mismatch results carry a witness")
        if self.execution is not None and self.is_effective:
            raise ValueError("Proven results do not carry execution evidence")
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
        if self.is_effective and self.attempts:
            raise ValueError("Proven results do not carry failed attempts")

    @property
    def is_refuted(self) -> bool:
        return self.witness is not None

    def with_witness(self, witness: RefutationWitness) -> "ComparisonAnalysis":
        """Attach a witness; an inconclusive result becomes a mismatch."""
        if self.status == ComparisonStatus.MISMATCH:
            return replace(self, witness=witness)
        if self.status != ComparisonStatus.INCONCLUSIVE:
            raise ValueError("Proven results cannot be refuted")
        facts: dict[str, FactValue] = {"location": witness.location}
        difference = ComparisonDifference(
            WITNESS_KINDS[witness.kind],
            DifferenceSide(
                None,
                witness.orig_address,
                {**facts, "value": witness.orig_value},
                "orig",
            ),
            DifferenceSide(
                None,
                witness.recomp_address,
                {**facts, "value": witness.recomp_value},
                "recomp",
            ),
        )
        return ComparisonAnalysis(
            ComparisonStatus.MISMATCH,
            difference=difference,
            attempts=self.attempts,
            witness=witness,
        )

    @property
    def is_effective(self) -> bool:
        return self.status in (ComparisonStatus.EXACT, ComparisonStatus.EFFECTIVE)

    @classmethod
    def exact(cls) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.EXACT)

    @classmethod
    def effective(cls, reasons) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.EFFECTIVE, tuple(reasons))

    @classmethod
    def mismatch(cls, difference: ComparisonDifference) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.MISMATCH, difference=difference)

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

    def address(
        self, which: Literal["orig", "recomp"], instruction_index: int | None
    ) -> int | None:
        addrs = self.orig_addrs if which == "orig" else self.recomp_addrs
        if addrs is not None and instruction_index is not None:
            if 0 <= instruction_index < len(addrs):
                return addrs[instruction_index]
        return None

    def side(
        self,
        which: Literal["orig", "recomp"],
        instruction_index: int | None,
        facts: dict[str, FactValue],
    ) -> DifferenceSide:
        return DifferenceSide(
            instruction_index, self.address(which, instruction_index), facts, which
        )

    def record_difference(
        self,
        kind: str,
        orig_index: int | None,
        recomp_index: int | None,
        orig_facts: dict[str, FactValue],
        recomp_facts: dict[str, FactValue],
        *,
        candidate: bool = False,
        values: tuple | None = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        difference = ComparisonDifference(
            kind,
            self.side("orig", orig_index, orig_facts),
            self.side("recomp", recomp_index, recomp_facts),
            values,
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
                # Keep the counterpart so the location can be source-pinned.
                recomp_address = self.address("recomp", recomp_index)
                if recomp_address is not None:
                    detail.setdefault("recomp_address", recomp_address)
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

    def effective_reasons(self, extra_reasons=()) -> frozenset[str]:
        return frozenset(self.reasons | set(extra_reasons))

    def attempt(self, strategy: str) -> StrategyAttempt:
        """Summarize where this recorder's strategy stopped."""
        if self.best_difference is not None:
            return StrategyAttempt(strategy, difference=self.best_difference)
        return StrategyAttempt(
            strategy,
            blocker=self.inconclusive_reason or "analysis_limit",
            location=self.inconclusive_location,
        )

    def failure_analysis(self) -> ComparisonAnalysis:
        if self.best_difference is not None:
            return ComparisonAnalysis.mismatch(self.best_difference)
        return ComparisonAnalysis.inconclusive(
            self.inconclusive_reason or "analysis_limit", self.inconclusive_location
        )
