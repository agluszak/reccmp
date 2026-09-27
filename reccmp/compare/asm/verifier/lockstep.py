"""Straight-line strategy: positional or diff-aligned paired execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from reccmp.compare.asm.ir import (
    DecodedInstruction,
    instruction_ids,
    instruction_semantic_key,
)
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
)
from reccmp.compare.asm.verifier.semantics import execute
from reccmp.compare.asm.verifier.state import (
    Branch,
    Context,
    Destination,
    ExternalDestination,
    FunctionMetadata,
    Jump,
    LocalDestination,
    Loop,
    Observation,
    SideState,
    UnresolvedDestination,
    clone_state,
    commit_memory,
    guard_state_size,
    is_conditional_observation,
    is_control_observation,
    observation_values,
)
from reccmp.compare.diagnosis import AnalysisRecorder, InconclusiveReason, StopDetail
from reccmp.types import ImageId


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
                InconclusiveReason.ALIGNMENT_FAILURE,
                detail=StopDetail.STREAM_ALIGNMENT,
            )
        return False

    orig = SideState()
    recomp = SideState()
    ctx = Context(metadata=metadata, recorder=recorder)
    last_index_o: int | None = None
    last_index_r: int | None = None
    orig_ids, recomp_ids = instruction_ids(orig_rows), instruction_ids(recomp_rows)

    try:
        for idx, (index_o, index_r) in enumerate(aligned):
            last_index_o, last_index_r = index_o, index_r
            if index_o is None or index_r is None:
                side = recomp if index_o is None else orig
                which = ImageId.RECOMP if index_o is None else ImageId.ORIG
                row = recomp_rows[index_r] if index_o is None else orig_rows[index_o]  # type: ignore[index]
                if not one_sided_ok(which, side, ctx, idx, row):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            InconclusiveReason.ALIGNMENT_FAILURE,
                            index_o,
                            index_r,
                            StopDetail.ONE_SIDED_INSTRUCTION,
                        )
                    return False
                continue

            ins_o, ins_r = orig_rows[index_o], recomp_rows[index_r]
            same = instruction_semantic_key(ins_o) == instruction_semantic_key(ins_r)
            try:
                record_operand_candidate(
                    ctx, index_o, index_r, ins_o, ins_r, (orig, recomp)
                )
                obs_o: list[Observation] = []
                obs_r: list[Observation] = []
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
                            InconclusiveReason.UNSUPPORTED_INSTRUCTION, index_o, index_r
                        )
                    return False
                if admit_unsupported_identical(orig, recomp, ctx, idx, ins_o, ins_r):
                    continue
                if recorder is not None:
                    recorder.mark_inconclusive(
                        InconclusiveReason.UNSUPPORTED_INSTRUCTION, index_o, index_r
                    )
                return False

            guard_state_size(orig, ctx)
            guard_state_size(recomp, ctx)

            _rewrite_control_observables(obs_o, ins_o, orig_ids)
            _rewrite_control_observables(obs_r, ins_r, recomp_ids)

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
                for value in observation_values(entry):
                    ctx.add_matched(value)

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
            if any(is_control_observation(entry) for entry in obs_o):
                conditional = any(is_conditional_observation(entry) for entry in obs_o)
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
            recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        return False

    if recorder is not None:
        recorder.reasons.update(ctx.categories)
    return True


def _rewrite_control_observables(
    obs: list[Observation], row: DecodedInstruction, ids: dict[int, int]
) -> None:
    """A branch observation's destination as a row index or an identity,
    never a relative displacement."""
    local = ids.get(row.branch_target) if row.branch_target is not None else None
    for index, entry in enumerate(obs):
        if not isinstance(entry, (Branch, Jump, Loop)):
            continue
        destination: Destination | None
        if local is not None:
            destination = LocalDestination(local)
        elif row.control_target is not None:
            destination = ExternalDestination(row.control_target)
        elif row.branch_target is not None:
            destination = UnresolvedDestination(row.branch_target)
        else:
            destination = entry.destination
        obs[index] = replace(entry, destination=destination)
