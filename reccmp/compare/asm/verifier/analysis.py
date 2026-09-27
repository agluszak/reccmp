"""Strategy orchestration: EXACT admission, then each verifier strategy in
turn, and the choice of the reported difference or blocker when none proves
equivalence."""

import dataclasses
import logging
from typing import Sequence

from reccmp.compare.asm.ir import (
    DecodedInstruction,
    FunctionImage,
    instruction_semantic_key,
    local_branch_targets,
)
from reccmp.compare.asm.verifier.cfg import verify_cfg_effective_match
from reccmp.compare.asm.verifier.iso_cfg import verify_isomorphic_cfg_effective_match
from reccmp.compare.asm.verifier.lockstep import verify_effective_match
from reccmp.compare.asm.verifier.relocation import undo_relocations
from reccmp.compare.asm.verifier.state import FunctionMetadata
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    ComparisonAnalysis,
    ComparisonStatus,
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
    while end > 0 and rows[end - 1].is_code and rows[end - 1].mnemonic in _PADDING:
        end -= 1
    if 0 < end < len(rows) and rows[end - 1].mnemonic in ("ret", "jmp"):
        return rows[:end]
    return rows


def analyze_effective_match(
    codes: Sequence[DiffOpcode],
    orig: FunctionImage,
    recomp: FunctionImage,
    metadata: FunctionMetadata | None = None,
) -> ComparisonAnalysis:
    # pylint: disable=too-many-locals,too-many-return-statements
    """Canonical semantic analysis of two decoded functions.

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
    exact = admit_exact_analysis(
        bytes_equal=(
            orig.raw is not None and recomp.raw is not None and orig.raw == recomp.raw
        ),
        topology_equal=local_branch_targets(orig_rows)
        == local_branch_targets(recomp_rows),
        keys_equal=(
            [instruction_semantic_key(row) for row in orig_rows]
            == [instruction_semantic_key(row) for row in recomp_rows]
            and orig.data_shape == recomp.data_shape
        ),
        operands_complete=all(
            row.operand_model_complete for row in (*orig_rows, *recomp_rows)
        ),
        control_flow_complete=(
            orig.control_flow_complete and recomp.control_flow_complete
        ),
        coverage_incomplete=coverage_incomplete,
        extent_closed=extent_closed,
    )
    if exact is not None:
        return exact

    def finish_effective(reasons) -> ComparisonAnalysis:
        reason_set = set(reasons)
        if not reason_set:
            reason_set.add("instruction_reorder")
        admitted = admit_effective(
            reason_set,
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        )
        if admitted is None:
            return ComparisonAnalysis.inconclusive(
                "incomplete_coverage" if coverage_incomplete else "open_extent"
            )
        return admitted.analysis

    addrs = ([row.address for row in orig_rows], [row.address for row in recomp_rows])

    def new_recorder() -> AnalysisRecorder:
        return AnalysisRecorder(*addrs)

    # Plain lockstep pairing first (with trailing alignment padding
    # trimmed): for equal-length sequences the diff's insert/delete blocks
    # can misalign lines that pair up fine positionally.
    trimmed_orig = _trim_padding(orig_rows)
    trimmed_recomp = _trim_padding(recomp_rows)
    padding = len(trimmed_orig) != len(orig_rows) or len(trimmed_recomp) != len(
        recomp_rows
    )
    relocated = undo_relocations(codes, orig_rows, recomp_rows)
    lockstep = new_recorder()
    if verify_effective_match(
        trimmed_orig, trimmed_recomp, metadata=metadata, recorder=lockstep
    ):
        logger.debug("effective match: lockstep")
        extra_reasons = {"padding"} if padding else set()
        if relocated is not None:
            extra_reasons.add("instruction_reorder")
        return finish_effective(lockstep.effective_reasons(extra_reasons))

    # Diff-aligned pairing: handles length differences (one-sided entries
    # for whitelisted unobservable instructions, e.g. a redundant
    # register copy) and transposed independent lines.
    diff_aligned = new_recorder()
    if verify_effective_match(
        orig_rows, recomp_rows, codes, metadata=metadata, recorder=diff_aligned
    ):
        logger.debug("effective match: diff-aligned")
        return finish_effective(diff_aligned.effective_reasons())

    relocation = new_recorder()
    if relocated is not None and verify_effective_match(
        orig_rows, relocated, metadata=metadata, recorder=relocation
    ):
        logger.debug("effective match: instruction relocation")
        return finish_effective(relocation.effective_reasons({"instruction_reorder"}))

    # CFG-aware verification with the lines paired by position.
    cfg = new_recorder()
    if verify_cfg_effective_match(
        trimmed_orig, trimmed_recomp, metadata=metadata, recorder=cfg
    ):
        logger.debug("effective match: cfg")
        return finish_effective(cfg.effective_reasons({"padding"} if padding else ()))

    # Isomorphic-CFG verification: per-side block graphs matched by
    # structure. Tolerates different instruction counts (folded loads,
    # elided copies) and the shifted branch displacements they cause.
    iso = new_recorder()
    if verify_isomorphic_cfg_effective_match(
        orig_rows,
        recomp_rows,
        orig.jump_tables,
        recomp.jump_tables,
        metadata=metadata,
        recorder=iso,
    ):
        logger.debug("effective match: isomorphic cfg")
        return finish_effective(iso.effective_reasons())

    attempts = [lockstep.attempt("lockstep"), diff_aligned.attempt("diff_aligned")]
    if relocated is not None:
        attempts.append(relocation.attempt("relocation"))
    attempts.append(cfg.attempt("cfg"))
    attempts.append(iso.attempt("isomorphic_cfg"))

    def failed(recorder: AnalysisRecorder) -> ComparisonAnalysis:
        return dataclasses.replace(
            recorder.failure_analysis(), attempts=tuple(attempts)
        )

    # Only positional lockstep and the two CFG strategies establish trusted
    # program points. Diff alignment and relocation are proof-only. Of
    # those, report an observed difference (a value, store or control
    # transfer that differs) before an operand candidate, and the product
    # pairing's, which follows both layouts, before the positional ones',
    # which also stop at a mere layout difference.
    trusted = (iso, lockstep, cfg)
    for recorder in trusted:
        if recorder.difference is not None:
            return failed(recorder)
    for recorder in trusted:
        if recorder.best_difference is not None:
            return failed(recorder)
    for candidate in (iso, cfg, lockstep):
        if candidate.inconclusive_reason is not None:
            inconclusive = candidate
            break
    else:
        inconclusive = cfg
    analysis = failed(inconclusive)
    assert analysis.status == ComparisonStatus.INCONCLUSIVE
    return analysis
