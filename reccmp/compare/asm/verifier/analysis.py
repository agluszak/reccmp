"""Function comparison: the one EXACT admission, then each verifier
strategy in turn, and the choice of the reported difference or blocker when none proves
equivalence."""

import dataclasses
import logging
from typing import Sequence

from reccmp.compare.asm.ir import (
    DecodedInstruction,
    FunctionImage,
    instruction_semantic_key,
)
from reccmp.compare.asm.verifier.iso_cfg import verify_isomorphic_cfg_effective_match
from reccmp.compare.asm.verifier.lockstep import verify_effective_match
from reccmp.compare.asm.verifier.relocation import undo_relocations
from reccmp.compare.asm.verifier.state import FunctionMetadata
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    ComparisonAnalysis,
    ComparisonStatus,
    EffectiveReason,
    InconclusiveReason,
    Strategy,
)
from reccmp.compare.pinned_sequences import DiffOpcode
from reccmp.compare.verification import admit_effective, admit_exact_analysis

logger = logging.getLogger(__name__)


# Alignment padding emitted between functions. int3 traps if executed, so
# it is only trimmed as trailing padding behind an instruction that does
# not fall through — never excused as a one-sided instruction.
_PADDING = ("nop", "int3")


def _trim_padding(rows: Sequence[DecodedInstruction]) -> Sequence[DecodedInstruction]:
    """Strip trailing nop/int3 alignment padding, but only behind an
    instruction that does not fall through into it."""
    end = len(rows)
    while end > 0 and rows[end - 1].mnemonic in _PADDING:
        end -= 1
    if 0 < end < len(rows) and rows[end - 1].mnemonic in ("ret", "jmp"):
        return rows[:end]
    return rows


def compare_exact(
    orig: FunctionImage, recomp: FunctionImage
) -> ComparisonAnalysis | None:
    """EXACT when both images have the same semantic keys, data shape and
    graph shape (every edge, including switch cases, known), and either the
    same bytes or complete operand models. This is the only place a function
    comparison is admitted EXACT."""
    orig_rows, recomp_rows = orig.instructions, recomp.instructions
    orig_shape = orig.control_graph().shape()
    return admit_exact_analysis(
        bytes_equal=(
            orig.raw is not None and recomp.raw is not None and orig.raw == recomp.raw
        ),
        topology_equal=(
            orig_shape is not None and orig_shape == recomp.control_graph().shape()
        ),
        keys_equal=(
            [instruction_semantic_key(row) for row in orig_rows]
            == [instruction_semantic_key(row) for row in recomp_rows]
            and orig.data_shape == recomp.data_shape
        ),
        operands_complete=all(
            row.operand_model_complete for row in (*orig_rows, *recomp_rows)
        ),
        coverage_incomplete=orig.coverage_incomplete or recomp.coverage_incomplete,
        extent_closed=orig.extent_closed and recomp.extent_closed,
    )


def analyze_effective_match(
    codes: Sequence[DiffOpcode],
    orig: FunctionImage,
    recomp: FunctionImage,
    metadata: FunctionMetadata | None = None,
) -> ComparisonAnalysis:
    # pylint: disable=too-many-locals,too-many-return-statements
    """Semantic analysis of two decoded functions ``compare_exact`` did not
    admit: EFFECTIVE, MISMATCH or INCONCLUSIVE, never EXACT.

    The relational verifier (see the verifier package) proves equivalence modulo
    register allocation, commutative-operand order and inverted compare/jump
    conditions. Instruction-scheduling differences are handled by undoing
    relocations that are proven independent of everything they cross, then
    running the verifier on the reordered sequence — so relocations compose
    with register renames and operand swaps. `metadata` (optional) provides
    PDB-derived return-type and callee-convention facts that widen what the
    verifier can prove."""
    coverage_incomplete = orig.coverage_incomplete or recomp.coverage_incomplete
    extent_closed = orig.extent_closed and recomp.extent_closed
    orig_rows, recomp_rows = orig.instructions, recomp.instructions

    # Embedded bytes can be read through indexed operands without appearing
    # as instruction effects. Until those reads are modeled against regions,
    # a change to the data cannot be admitted by a code-only proof.
    embedded_data_differs = orig.data_shape[0] != recomp.data_shape[0]

    def finish_effective(reasons) -> ComparisonAnalysis:
        if embedded_data_differs:
            return ComparisonAnalysis.inconclusive(
                InconclusiveReason.EMBEDDED_DATA_MISMATCH
            )
        reason_set = set(reasons)
        if not reason_set:
            reason_set.add(EffectiveReason.INSTRUCTION_REORDER)
        admitted = admit_effective(
            reason_set,
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        )
        if admitted is None:
            return ComparisonAnalysis.inconclusive(
                InconclusiveReason.INCOMPLETE_COVERAGE
                if coverage_incomplete
                else InconclusiveReason.OPEN_EXTENT
            )
        return admitted

    # Plain lockstep pairing first (with trailing alignment padding
    # trimmed): for equal-length sequences the diff's insert/delete blocks
    # can misalign lines that pair up fine positionally.
    trimmed_orig = _trim_padding(orig_rows)
    trimmed_recomp = _trim_padding(recomp_rows)
    padding = len(trimmed_orig) != len(orig_rows) or len(trimmed_recomp) != len(
        recomp_rows
    )
    relocated = undo_relocations(codes, orig_rows, recomp_rows)
    lockstep = AnalysisRecorder(orig, recomp)
    if verify_effective_match(
        trimmed_orig, trimmed_recomp, metadata=metadata, recorder=lockstep
    ):
        logger.debug("effective match: lockstep")
        extra_reasons = {EffectiveReason.PADDING} if padding else set()
        if relocated is not None:
            extra_reasons.add(EffectiveReason.INSTRUCTION_REORDER)
        return finish_effective(lockstep.effective_reasons(extra_reasons))

    # Diff-aligned pairing: handles length differences (one-sided entries
    # for whitelisted unobservable instructions, e.g. a redundant
    # register copy) and transposed independent lines.
    diff_aligned = AnalysisRecorder(orig, recomp)
    if verify_effective_match(
        orig_rows, recomp_rows, codes, metadata=metadata, recorder=diff_aligned
    ):
        logger.debug("effective match: diff-aligned")
        return finish_effective(diff_aligned.effective_reasons())

    # Its instruction positions are those of the reordered sequence.
    relocation = AnalysisRecorder(
        orig, recomp.with_instructions(relocated) if relocated is not None else recomp
    )
    if relocated is not None and verify_effective_match(
        orig_rows, relocated, metadata=metadata, recorder=relocation
    ):
        logger.debug("effective match: instruction relocation")
        return finish_effective(
            relocation.effective_reasons({EffectiveReason.INSTRUCTION_REORDER})
        )

    # Isomorphic-CFG verification: per-side block graphs matched by
    # structure. Tolerates different instruction counts (folded loads,
    # elided copies) and the shifted branch displacements they cause.
    product = verify_isomorphic_cfg_effective_match(orig, recomp, metadata)
    iso = product.recorder
    if product.proved:
        logger.debug("effective match: isomorphic cfg")
        return finish_effective(iso.effective_reasons())

    attempts = [
        lockstep.attempt(Strategy.LOCKSTEP),
        diff_aligned.attempt(Strategy.DIFF_ALIGNED),
    ]
    if relocated is not None:
        attempts.append(relocation.attempt(Strategy.RELOCATION))
    attempts.append(iso.attempt(Strategy.ISOMORPHIC_CFG))
    if product.unanchored is not None:
        attempts.append(product.unanchored.attempt(Strategy.UNANCHORED_PRODUCT))

    def failed(recorder: AnalysisRecorder) -> ComparisonAnalysis:
        return dataclasses.replace(
            recorder.failure_analysis(), attempts=tuple(attempts)
        )

    # Only positional lockstep and the product CFG pairing establish trusted
    # program points. Diff alignment and relocation are proof-only. Of
    # those, report an observed difference (a value, store or control
    # transfer that differs) before an operand candidate, and the product
    # pairing's, which follows both layouts, before lockstep's, which also
    # stops at a mere layout difference.
    trusted = (iso, lockstep)
    for recorder in trusted:
        if recorder.difference is not None:
            return failed(recorder)
    for recorder in trusted:
        if recorder.best_difference is not None:
            return failed(recorder)
    for candidate in (iso, lockstep):
        if candidate.inconclusive_reason is not None:
            inconclusive = candidate
            break
    else:
        inconclusive = lockstep
    analysis = failed(inconclusive)
    assert analysis.status == ComparisonStatus.INCONCLUSIVE
    return analysis
