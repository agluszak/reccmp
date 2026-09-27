"""Straight-line strategy: positional or diff-aligned paired execution."""

from __future__ import annotations

from collections.abc import Sequence

from reccmp.compare.asm.ir import DecodedInstruction, instruction_semantic_key
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.verifier.evidence import (
    record_observable_difference,
    record_operand_candidate,
)
from reccmp.compare.asm.verifier.obligations import (
    admit_unsupported_identical,
    aligned_indices,
    callee_save_swap,
    discharge_run_obligations,
    divergences_justified,
    invalidate_save_slots,
    observations_agree,
    one_sided_ok,
    record_pair_categories,
    rewrite_control_observables,
)
from reccmp.compare.asm.verifier.semantics import execute
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    Context,
    FunctionMetadata,
    SideState,
    clone_state,
    commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import AnalysisRecorder


def verify_effective_match(
    orig_rows: Sequence[DecodedInstruction],
    recomp_rows: Sequence[DecodedInstruction],
    codes=None,
    metadata: FunctionMetadata | None = None,
    recorder: AnalysisRecorder | None = None,
) -> bool:
    """True if the two instruction sequences can be proven equivalent
    modulo register allocation, frame-slot layout, commutative-operand
    order and inverted compare/jump conditions, pairing them by position
    or by the diff's ``codes``. The rows' Capstone effects let an unmodeled
    register-only instruction be stepped over precisely instead of
    requiring full synchronization."""
    # pylint: disable=too-many-branches,too-many-return-statements,too-many-statements
    # pylint: disable=too-many-locals
    aligned = aligned_indices(codes, len(orig_rows), len(recomp_rows))
    if aligned is None:
        if recorder is not None:
            recorder.mark_inconclusive(
                "alignment_failure",
                facts={
                    "stage": "stream_alignment",
                    "orig_instruction_count": len(orig_rows),
                    "recomp_instruction_count": len(recomp_rows),
                },
            )
        return False

    orig = SideState()
    recomp = SideState()
    ctx = Context(metadata=metadata, recorder=recorder)
    last_index_o: int | None = None
    last_index_r: int | None = None
    orig_cf_addrs = [row.address for row in orig_rows]
    recomp_cf_addrs = [row.address for row in recomp_rows]

    try:
        for idx, (index_o, index_r) in enumerate(aligned):
            last_index_o, last_index_r = index_o, index_r
            if index_o is None or index_r is None:
                side = recomp if index_o is None else orig
                other_side = orig if index_o is None else recomp
                row = recomp_rows[index_r] if index_o is None else orig_rows[index_o]  # type: ignore[index]
                if not one_sided_ok(side, other_side, ctx, idx, row):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "alignment_failure",
                            index_o,
                            index_r,
                            {"stage": "one_sided_instruction"},
                        )
                    return False
                continue

            ins_o, ins_r = orig_rows[index_o], recomp_rows[index_r]
            same = instruction_semantic_key(ins_o) == instruction_semantic_key(ins_r)
            if not ins_o.is_code or not ins_r.is_code:
                if not same:
                    return False
                continue

            try:
                record_operand_candidate(
                    ctx, index_o, index_r, ins_o, ins_r, (orig, recomp)
                )
                obs_o: list = []
                obs_r: list = []
                before_o = dict(orig.regs)
                before_r = dict(recomp.regs)
                state_before_o = clone_state(orig)
                state_before_r = clone_state(recomp)
                execute(orig, ctx, idx, ins_o, obs_o)
                execute(recomp, ctx, idx, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                # Unsupported instruction: only allowed if both sides are
                # the same instruction, and then only when its precise
                # effects are known (capstone) or the two symbolic states
                # are fully synchronized.
                if not same:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "unsupported_instruction", index_o, index_r
                        )
                    return False
                if admit_unsupported_identical(orig, recomp, ctx, idx, ins_o, ins_r):
                    continue
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "unsupported_instruction", index_o, index_r
                    )
                return False

            guard_state_size(orig, ctx)
            guard_state_size(recomp, ctx)

            rewrite_control_observables(obs_o, ins_o, orig_cf_addrs)
            rewrite_control_observables(obs_r, ins_r, recomp_cf_addrs)

            if callee_save_swap(ctx, ins_o, ins_r, obs_o, obs_r, orig, recomp):
                # The pushed values differ (that is the point of the swap),
                # but the slot and width agree: commit from the orig side.
                commit_memory(ctx, obs_o, idx)
                continue

            if not observations_agree(ctx, obs_o, obs_r):
                record_observable_difference(
                    ctx, index_o, index_r, ins_o, ins_r, obs_o, obs_r
                )
                return False
            invalidate_save_slots(ctx, obs_o)
            for entry in obs_o:
                ctx.add_matched(entry)

            # The same value written by both sides in this step (even to
            # different registers) is proven correspondence: remember it so
            # that dead leftovers of its computation are recognized at the
            # end of the run.
            written_o = [
                value
                for family, value in orig.regs.items()
                if value is not before_o[family]
            ]
            written_r = [
                value
                for family, value in recomp.regs.items()
                if value is not before_r[family]
            ]
            for value in written_o:
                if value in written_r:
                    ctx.add_matched(value)
            record_pair_categories(
                ctx,
                state_before_o,
                state_before_r,
                orig,
                recomp,
                ins_o,
                ins_r,
                obs_o,
                obs_r,
            )

            # Linear execution cannot prove both successors of a conditional.
            # Divergent registers at a jcc are a CFG problem even when both
            # values appear in the predicate (xchg + inverted compare) or are
            # "scratch" initials of different families.
            if any(entry[0] in CONTROL_TAGS for entry in obs_o):
                conditional = any(
                    entry[0] in {"branch", "loop", "loope", "loopne", "jcxz", "jecxz"}
                    for entry in obs_o
                )
                if conditional and orig.regs != recomp.regs:
                    return False
                if not divergences_justified(ctx, orig, recomp):
                    return False

            commit_memory(ctx, obs_o, idx)

        if not discharge_run_obligations(
            ctx, orig, recomp, recorder, last_index_o, last_index_r
        ):
            return False
    except (Reject, RecursionError):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False

    if recorder is not None:
        recorder.reasons.update(ctx.categories)
    return True
