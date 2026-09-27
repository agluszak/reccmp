"""Stable structured results for semantic comparison diagnosis.

This module intentionally contains only the small, generic vocabulary shared by
the verifier, JSON reports, and downstream tools.  Symbolic execution and report
formatting remain in their existing layers.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from reccmp.compare.asm.operand import Operand
from reccmp.source.records import SourceComparison
from reccmp.types import ImageId

if TYPE_CHECKING:
    from reccmp.compare.asm.ir import FunctionImage


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
        if EffectiveReason.CONDITION_INVERSION in reasons:
            tags.add(DiagnosticNormalization.CFG_LAYOUT)
        if reasons & {
            EffectiveReason.INSTRUCTION_REORDER,
            EffectiveReason.COMMUTATIVE_ORDER,
            EffectiveReason.LOAD_FOLDING,
        }:
            tags.add(DiagnosticNormalization.INSTRUCTION_SCHEDULING)
        if reasons & {
            EffectiveReason.REGISTER_ALLOCATION,
            EffectiveReason.CALLEE_SAVE_SUBSTITUTION,
            EffectiveReason.FRAME_SLOT_PROMOTION,
        }:
            tags.add(DiagnosticNormalization.REGISTER_ALLOCATION)
        if EffectiveReason.FRAME_SLOT_LAYOUT in reasons:
            tags.add(DiagnosticNormalization.STACK_LAYOUT)
        if EffectiveReason.FOLDED_SYMBOL_ALIAS in reasons:
            tags.add(DiagnosticNormalization.FOLDED_SYMBOL_ALIAS)

    # Modulo scores are diagnostic collapses, not proofs of equivalence.
    if accuracy_modulo_inline is not None and accuracy_modulo_inline >= 1.0:
        tags.add(DiagnosticNormalization.KNOWN_INLINE)
    if accuracy_modulo_stack is not None and accuracy_modulo_stack >= 1.0:
        tags.add(DiagnosticNormalization.STACK_LAYOUT)

    return tuple(tag for tag in DiagnosticNormalization if tag in tags)


class EffectiveReason(Enum):
    """What differs between two functions proven equivalent, in order."""

    REGISTER_ALLOCATION = "register_allocation"
    FRAME_SLOT_LAYOUT = "frame_slot_layout"
    # A local kept in a stack slot on one side and in a register (or
    # another slot) on the other, each side's private frame held apart.
    FRAME_SLOT_PROMOTION = "frame_slot_promotion"
    CALLEE_SAVE_SUBSTITUTION = "callee_save_substitution"
    INSTRUCTION_REORDER = "instruction_reorder"
    COMMUTATIVE_ORDER = "commutative_order"
    CONDITION_INVERSION = "condition_inversion"
    LOAD_FOLDING = "load_folding"
    DEAD_OPERATION = "dead_operation"
    # Values proven equal as bit-vectors (z3) though computed differently.
    ALGEBRAIC_IDENTITY = "algebraic_identity"
    PADDING = "padding"
    # The original function is a stale incremental-link jmp island whose fold
    # chain lands on a proven-equivalent shared body (configured via the
    # project's equivalence-groups metadata); the recomp emits the real body.
    FOLDED_SYMBOL_ALIAS = "folded_symbol_alias"


class DifferenceKind(Enum):
    CALL_TARGET = "call_target"
    CALL_ARGUMENT = "call_argument"
    MEMORY_ADDRESS = "memory_address"
    MEMORY_VALUE = "memory_value"
    IMMEDIATE_VALUE = "immediate_value"
    BRANCH_CONDITION = "branch_condition"
    BRANCH_TARGET = "branch_target"
    RETURN_VALUE = "return_value"
    PRESERVED_STATE = "preserved_state"
    SYMBOL_RESOLUTION = "symbol_resolution"


class InconclusiveReason(Enum):
    UNSUPPORTED_INSTRUCTION = "unsupported_instruction"
    EMPTY_CONTROL_FLOW = "empty_control_flow"
    CONTROL_FLOW_METADATA_MISMATCH = "control_flow_metadata_mismatch"
    INVALID_CONTROL_FLOW_TARGET = "invalid_control_flow_target"
    JUMP_TABLE_DATA = "jump_table_data"
    EMBEDDED_DATA_MISMATCH = "embedded_data_mismatch"
    NON_ISOMORPHIC_CFG = "non_isomorphic_cfg"
    INDIRECT_JUMP = "indirect_jump"
    EXTERNAL_CONTROL_FLOW_STATE = "external_control_flow_state"
    FUNCTION_FALLTHROUGH = "function_fallthrough"
    STATE_JOIN_FAILURE = "state_join_failure"
    ALIGNMENT_FAILURE = "alignment_failure"
    MISSING_METADATA = "missing_metadata"
    ANALYSIS_LIMIT = "analysis_limit"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    OPEN_EXTENT = "open_extent"


def normalize_effective_reasons(
    reasons: Iterable[EffectiveReason],
) -> tuple[EffectiveReason, ...]:
    """Deduplicate and order effective reasons."""
    values = set(reasons)
    if not all(isinstance(reason, EffectiveReason) for reason in values):
        raise ValueError("Unknown effective reason")
    return tuple(reason for reason in EffectiveReason if reason in values)


class StopDetail(Enum):
    """Which part of a strategy gave up, within its inconclusive reason."""

    STREAM_ALIGNMENT = "stream_alignment"
    ONE_SIDED_INSTRUCTION = "one_sided_instruction"
    BLOCK_ALIGNMENT = "block_alignment"
    BLOCK_TERMINATOR_ALIGNMENT = "block_terminator_alignment"
    UNRESOLVED_SWITCH_TABLE = "unresolved_switch_table"
    INDIRECT_TARGET = "indirect_target"
    BLOCK_MAPPING_CONFLICT = "block_mapping_conflict"
    EDGE_ROLES = "edge_roles"
    EXTERNAL_EDGE = "external_edge"
    BRANCH_ORIENTATION = "branch_orientation"


@dataclass(frozen=True)
class SourceLine:
    path: str
    line: int


@dataclass(frozen=True)
class StopLocation:
    """Where a strategy stopped without finding a difference."""

    image: ImageId
    instruction_index: int | None = None
    address: int | None = None
    # The recompiled instruction paired with an original location.
    counterpart_address: int | None = None
    detail: StopDetail | None = None
    # The recompiled source line of the location (or of its counterpart).
    source: SourceLine | None = None


@dataclass(frozen=True)
class StackPermutationEntry:
    """One orig → recomp local slot correspondence."""

    orig: str
    recomp: str
    symbol: str | None = None


@dataclass(frozen=True)
class Observed:
    """What one side has where the two differ."""

    # The differing operand: an address, immediate, symbol or transfer target.
    operand: Operand | None = None
    # A control transfer's destination, and the instruction it reaches.
    target: int | None = None
    target_index: int | None = None
    # What the side computes there, as shown to a person.
    value: str | None = None
    # The register holding it (a call's argument, a preserved register).
    register: str | None = None


@dataclass(frozen=True)
class FieldAt:
    """The class field a displacement reaches in the recovered layout."""

    class_name: str
    path: tuple[str, ...]
    offset: int
    type: str


@dataclass(frozen=True)
class DifferenceSide:
    image: ImageId
    instruction_index: int | None = None
    address: int | None = None
    observed: Observed = Observed()
    # From the recompiled program's debug and source facts.
    source: SourceLine | None = None
    field: FieldAt | None = None
    source_comparisons: tuple[SourceComparison, ...] = ()


class SolverResult(Enum):
    PROVED = "proved"  # equal for every input
    # Z3 found leaf values under which they differ: in the verifier's
    # abstraction, where unlowered terms and loads are independent leaves,
    # so not a refutation.
    DIFFERS = "differs"
    UNSUPPORTED = "unsupported"  # a term could not be lowered
    UNKNOWN = "unknown"  # the budget ran out


@dataclass(frozen=True)
class SolverOutcome:
    """What one equivalence query found. ``reason`` says which term could
    not be lowered, or why Z3 gave up; ``rlimit`` is the resource units the
    query used, a deterministic measure of its cost."""

    result: SolverResult
    reason: str | None = None
    rlimit: int | None = None
    # For DIFFERS: (leaf term, value) for each leaf Z3 constrained.
    assignment: tuple[tuple[Hashable, int], ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class ComparisonDifference:
    kind: DifferenceKind
    orig: DifferenceSide
    recomp: DifferenceSide
    # The verifier's symbolic values behind a value difference: (value_orig,
    # value_recomp, bits, "value" or "predicate"). In memory only (for the
    # witness to ask a solver for a distinguishing input), never reported.
    values: tuple | None = field(default=None, compare=False, repr=False)
    # What Z3 said about those values.
    solver: SolverOutcome | None = field(default=None, compare=False)


class Strategy(Enum):
    LOCKSTEP = "lockstep"
    DIFF_ALIGNED = "diff_aligned"
    RELOCATION = "relocation"
    ISOMORPHIC_CFG = "isomorphic_cfg"
    UNANCHORED_PRODUCT = "unanchored_product"

    @property
    def trusted_alignment(self) -> bool:
        """Whether its instruction pairing is anchored by position or by
        matched blocks. The others pair instructions heuristically (diff
        opcodes, undone relocations, a guessed block pairing), so a
        difference they report may be an artifact of the pairing."""
        return self in (Strategy.LOCKSTEP, Strategy.ISOMORPHIC_CFG)


@dataclass(frozen=True)
class StrategyAttempt:
    """Where one verifier strategy stopped: a difference or a blocker."""

    strategy: Strategy
    difference: ComparisonDifference | None = None
    blocker: InconclusiveReason | None = None
    location: StopLocation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, Strategy):
            raise ValueError("Unknown strategy")
        if self.blocker is not None and not isinstance(
            self.blocker, InconclusiveReason
        ):
            raise ValueError("Unknown inconclusive reason")
        if (self.difference is None) == (self.blocker is None):
            raise ValueError("An attempt has exactly one of difference or blocker")
        if self.location is not None and self.blocker is None:
            raise ValueError("Only blocked attempts carry a location")


class WitnessKind(Enum):
    """What a witness run shows differ."""

    MEMORY_VALUE = "memory_value"
    RETURN_VALUE = "return_value"
    CALL_ARGUMENT = "call_argument"
    STACK_CLEANUP = "stack_cleanup"

    @property
    def difference(self) -> DifferenceKind:
        if self is WitnessKind.STACK_CLEANUP:
            return DifferenceKind.PRESERVED_STATE
        return DifferenceKind(self.value)


@dataclass(frozen=True)
class WitnessInput:
    """The state both functions start from in one witness run. The seed
    fixes everything else the model generates (memory pages, call results),
    so this reproduces the run; see reccmp.compare.witness.machine.RunInput."""

    seed: int
    registers: tuple[tuple[str, int], ...]
    stack_args: tuple[int, ...]
    pool: tuple[int, ...] = ()
    memory: tuple[tuple[int, int], ...] = ()  # (address, byte) presets


@dataclass(frozen=True)
class WitnessReplay:
    """What replaying a witness needs besides the two binaries and their
    entity database: no solver, no search."""

    input: WitnessInput
    orig_function: tuple[int, int]  # (start, extent)
    recomp_function: tuple[int, int]
    return_kind: str
    # reccmp.compare.witness.machine.WITNESS_MODEL the run used.
    model: int
    # SHA-256 of the original and recompiled images, when known.
    images: tuple[str | None, str | None] = (None, None)


@dataclass(frozen=True)
class RefutationWitness:
    """Concrete inputs under which the two functions observably differ.

    Found by executing both bodies from the same state (see
    reccmp.compare.witness). Callees are modelled, not run, and the input
    need not be reachable from the program's real callers.
    """

    # pylint: disable=too-many-instance-attributes
    seed: int
    kind: WitnessKind
    location: str
    orig_value: str
    recomp_value: str
    orig_address: int | None = None
    recomp_address: int | None = None
    replay: WitnessReplay | None = None


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
    # For unknown_image_read and unresolved_call: the first unidentified
    # read or call (addresses, sections, database entities on each side).
    # For estimated_extent: the difference a run would have shown if
    # estimated extents were evidence.
    no_verdict_details: dict[str, dict[str, object]] = field(default_factory=dict)
    # What came of asking the solver for an input under which the reported
    # values differ: its answer when it gave none (``solver unsupported:
    # ...``), why its assignment is no run input (``rejected: ...``), or
    # what the run from it did (``run agreed``, ``run call_structure``, and
    # whether it reached the difference). None without a value difference.
    solver_hint: str | None = None


@dataclass(frozen=True)
class ComparisonAnalysis:
    # pylint: disable=too-many-instance-attributes
    status: ComparisonStatus
    effective_reasons: tuple[EffectiveReason, ...] = ()
    difference: ComparisonDifference | None = None
    inconclusive_reason: InconclusiveReason | None = None
    inconclusive_location: StopLocation | None = None
    # Every strategy that ran, in execution order. The primary difference or
    # reason above is chosen from these; the rest show what else blocked.
    attempts: tuple[StrategyAttempt, ...] = ()
    # Only on mismatches: a demonstrated difference. A mismatch without one
    # is a candidate located by the verifier, not a refutation.
    witness: RefutationWitness | None = None
    # Differential execution that found no witness (mismatch/inconclusive).
    execution: ExecutionEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ComparisonStatus):
            raise ValueError("Unknown comparison status")
        if self.inconclusive_reason is not None and not isinstance(
            self.inconclusive_reason, InconclusiveReason
        ):
            raise ValueError("Unknown inconclusive reason")
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
        if (self.status == ComparisonStatus.INCONCLUSIVE) != (
            self.inconclusive_reason is not None
        ):
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
        difference = ComparisonDifference(
            witness.kind.difference,
            DifferenceSide(
                ImageId.ORIG,
                address=witness.orig_address,
                observed=Observed(value=witness.orig_value),
            ),
            DifferenceSide(
                ImageId.RECOMP,
                address=witness.recomp_address,
                observed=Observed(value=witness.recomp_value),
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
    def effective(cls, reasons: Iterable[EffectiveReason]) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.EFFECTIVE, tuple(reasons))

    @classmethod
    def mismatch(cls, difference: ComparisonDifference) -> "ComparisonAnalysis":
        return cls(ComparisonStatus.MISMATCH, difference=difference)

    @classmethod
    def inconclusive(
        cls, reason: InconclusiveReason, location: StopLocation | None = None
    ) -> "ComparisonAnalysis":
        return cls(
            ComparisonStatus.INCONCLUSIVE,
            inconclusive_reason=reason,
            inconclusive_location=location,
        )


@dataclass
class AnalysisRecorder:
    """Mutable evidence sink used by one speculative verifier strategy on
    two images. Locations are recorded as instruction positions in them."""

    orig: FunctionImage
    recomp: FunctionImage
    reasons: set[EffectiveReason] = field(default_factory=set)
    difference: ComparisonDifference | None = None
    candidate_difference: ComparisonDifference | None = None
    inconclusive_reason: InconclusiveReason | None = None
    inconclusive_location: StopLocation | None = None

    def image(self, which: ImageId) -> FunctionImage:
        return self.orig if which is ImageId.ORIG else self.recomp

    def address(self, which: ImageId, instruction_index: int | None) -> int | None:
        rows = self.image(which).instructions
        if instruction_index is not None and 0 <= instruction_index < len(rows):
            return rows[instruction_index].address
        return None

    def record_difference(
        self,
        kind: DifferenceKind,
        orig_index: int | None,
        recomp_index: int | None,
        orig: Observed,
        recomp: Observed,
        *,
        candidate: bool = False,
        values: tuple | None = None,
        solver: SolverOutcome | None = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        difference = ComparisonDifference(
            kind,
            DifferenceSide(
                ImageId.ORIG,
                orig_index,
                self.address(ImageId.ORIG, orig_index),
                orig,
            ),
            DifferenceSide(
                ImageId.RECOMP,
                recomp_index,
                self.address(ImageId.RECOMP, recomp_index),
                recomp,
            ),
            values,
            solver,
        )
        if candidate:
            if self.candidate_difference is None:
                self.candidate_difference = difference
        elif self.difference is None:
            self.difference = difference

    def mark_inconclusive(
        self,
        reason: InconclusiveReason,
        orig_index: int | None = None,
        recomp_index: int | None = None,
        detail: StopDetail | None = None,
        *,
        image: ImageId | None = None,
    ) -> None:
        """Record why the strategy stopped, where: at the paired
        instructions, or on ``image`` when no instruction is to blame."""
        if self.inconclusive_reason is not None:
            return
        self.inconclusive_reason = reason
        if orig_index is not None:
            self.inconclusive_location = StopLocation(
                ImageId.ORIG,
                orig_index,
                self.address(ImageId.ORIG, orig_index),
                self.address(ImageId.RECOMP, recomp_index),
                detail,
            )
        elif recomp_index is not None:
            self.inconclusive_location = StopLocation(
                ImageId.RECOMP,
                recomp_index,
                self.address(ImageId.RECOMP, recomp_index),
                detail=detail,
            )
        elif image is not None or detail is not None:
            self.inconclusive_location = StopLocation(
                image or ImageId.ORIG, detail=detail
            )

    @property
    def best_difference(self) -> ComparisonDifference | None:
        if self.difference is None:
            return self.candidate_difference
        if self.candidate_difference is None:
            return self.difference
        if self.difference.kind in (
            DifferenceKind.CALL_TARGET,
            DifferenceKind.CALL_ARGUMENT,
            DifferenceKind.BRANCH_CONDITION,
            DifferenceKind.BRANCH_TARGET,
        ):
            return self.difference
        if (
            self.difference.kind is DifferenceKind.RETURN_VALUE
            and self.candidate_difference.kind is DifferenceKind.IMMEDIATE_VALUE
        ):
            return self.difference
        concrete_index = self.difference.orig.instruction_index
        candidate_index = self.candidate_difference.orig.instruction_index
        if candidate_index is not None and (
            concrete_index is None or candidate_index < concrete_index
        ):
            return self.candidate_difference
        return self.difference

    def effective_reasons(self, extra_reasons=()) -> frozenset[EffectiveReason]:
        return frozenset(self.reasons | set(extra_reasons))

    def attempt(self, strategy: Strategy) -> StrategyAttempt:
        """Summarize where this recorder's strategy stopped."""
        if self.best_difference is not None:
            return StrategyAttempt(strategy, difference=self.best_difference)
        return StrategyAttempt(
            strategy,
            blocker=self.inconclusive_reason or InconclusiveReason.ANALYSIS_LIMIT,
            location=self.inconclusive_location,
        )

    def failure_analysis(self) -> ComparisonAnalysis:
        if self.best_difference is not None:
            return ComparisonAnalysis.mismatch(self.best_difference)
        return ComparisonAnalysis.inconclusive(
            self.inconclusive_reason or InconclusiveReason.ANALYSIS_LIMIT,
            self.inconclusive_location,
        )
