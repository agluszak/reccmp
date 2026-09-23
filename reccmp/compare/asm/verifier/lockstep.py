"""Straight-line strategy: positional or diff-aligned paired execution."""

from __future__ import annotations

from reccmp.compare.asm.instgen import (
    InstructionMeta,
)
from reccmp.compare.asm.ir import (
    AsmStream,
    instruction_at,
    is_data_row,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import (
    Reject,
)
from reccmp.compare.asm.verifier.evidence import (
    _record_observable_difference,
    _record_operand_candidate,
)
from reccmp.compare.asm.verifier.obligations import (
    _record_pair_categories,
    _addrs_from_meta,
    _aligned_indices,
    _callee_save_swap,
    _discharge_run_obligations,
    _divergences_justified,
    _invalidate_save_slots,
    _one_sided_ok,
    _rewrite_control_observables,
    admit_unsupported_identical,
)
from reccmp.compare.asm.verifier.semantics import (
    execute,
)
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    Context,
    FunctionMetadata,
    SideState,
    _clone_state,
    _commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
)


def verify_effective_match(
    orig_asm: AsmStream,
    recomp_asm: AsmStream,
    codes=None,
    metadata: FunctionMetadata | None = None,
    orig_meta: list[InstructionMeta | None] | None = None,
    recomp_meta: list[InstructionMeta | None] | None = None,
    recorder: AnalysisRecorder | None = None,
) -> bool:
    """True if the two instruction sequences can be proven equivalent
    modulo register allocation, frame-slot layout, commutative-operand
    order and inverted compare/jump conditions.

    Prefer ``DecodedInstruction`` / ``ResolvedAsm`` streams so structured
    operands are used directly. Legacy ``list[str]`` still reparses.

    `orig_meta` (optional, aligned with orig_asm) provides structured
    capstone facts; with them, an unmodeled register-only instruction can
    be stepped over precisely instead of requiring full synchronization."""
    # pylint: disable=too-many-branches,too-many-return-statements,too-many-statements
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    orig_stream = resolve_asm_stream(orig_asm)
    recomp_stream = resolve_asm_stream(recomp_asm)
    aligned = _aligned_indices(codes, len(orig_stream), len(recomp_stream))
    if aligned is None:
        if recorder is not None:
            recorder.mark_inconclusive(
                "alignment_failure",
                facts={
                    "stage": "stream_alignment",
                    "orig_instruction_count": len(orig_stream),
                    "recomp_instruction_count": len(recomp_stream),
                },
            )
        return False

    orig = SideState()
    recomp = SideState()
    ctx = Context(metadata=metadata, recorder=recorder)
    last_index_o: int | None = None
    last_index_r: int | None = None
    orig_cf_addrs = (
        recorder.orig_addrs
        if recorder is not None and recorder.orig_addrs is not None
        else _addrs_from_meta(orig_meta)
    )
    recomp_cf_addrs = (
        recorder.recomp_addrs
        if recorder is not None and recorder.recomp_addrs is not None
        else _addrs_from_meta(recomp_meta)
    )

    try:
        for idx, (index_o, index_r) in enumerate(aligned):
            last_index_o, last_index_r = index_o, index_r
            line_o = orig_stream.displays[index_o] if index_o is not None else None
            line_r = recomp_stream.displays[index_r] if index_r is not None else None
            if line_o is None or line_r is None:
                side = recomp if line_o is None else orig
                other_side = orig if line_o is None else recomp
                line = line_r if line_o is None else line_o
                side_stream = recomp_stream if line_o is None else orig_stream
                side_index = index_r if line_o is None else index_o
                assert line is not None and side_index is not None
                side_ins = None
                side_data = is_data_row(side_stream, side_index)
                if not side_data:
                    try:
                        side_ins = instruction_at(side_stream, side_index)
                    except (Reject, IndexError, KeyError, ValueError, TypeError):
                        side_ins = None
                if not _one_sided_ok(
                    side,
                    other_side,
                    ctx,
                    idx,
                    line,
                    ins=side_ins,
                    is_data=side_data,
                ):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "alignment_failure",
                            index_o,
                            index_r,
                            {"stage": "one_sided_instruction"},
                        )
                    return False
                continue

            assert index_o is not None and index_r is not None
            if is_data_row(orig_stream, index_o) or is_data_row(recomp_stream, index_r):
                if line_o != line_r:
                    return False
                continue

            try:
                ins_o = instruction_at(orig_stream, index_o)
                ins_r = instruction_at(recomp_stream, index_r)
                _record_operand_candidate(ctx, index_o, index_r, ins_o, ins_r)
                obs_o: list = []
                obs_r: list = []
                before_o = dict(orig.regs)
                before_r = dict(recomp.regs)
                state_before_o = _clone_state(orig)
                state_before_r = _clone_state(recomp)
                execute(orig, ctx, idx, ins_o, obs_o)
                execute(recomp, ctx, idx, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                # Unsupported or malformed instruction: only allowed if
                # both sides are textually identical, and then only when
                # its precise effects are known (capstone metadata) or the
                # two symbolic states are fully synchronized.
                if line_o != line_r:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "unsupported_instruction", index_o, index_r
                        )
                    return False
                meta_o = (
                    orig_meta[index_o]
                    if orig_meta is not None and index_o is not None
                    else None
                )
                meta_r = (
                    recomp_meta[index_r]
                    if recomp_meta is not None and index_r is not None
                    else None
                )
                if admit_unsupported_identical(orig, recomp, ctx, idx, meta_o, meta_r):
                    continue
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "unsupported_instruction", index_o, index_r
                    )
                return False

            guard_state_size(orig, ctx)
            guard_state_size(recomp, ctx)

            meta_o = orig_meta[index_o] if orig_meta is not None else None
            meta_r = recomp_meta[index_r] if recomp_meta is not None else None
            _rewrite_control_observables(obs_o, meta_o, orig_cf_addrs)
            _rewrite_control_observables(obs_r, meta_r, recomp_cf_addrs)

            if _callee_save_swap(ctx, ins_o, ins_r, obs_o, obs_r, orig, recomp):
                # The pushed values differ (that is the point of the swap),
                # but the slot and width agree: commit from the orig side.
                _commit_memory(ctx, obs_o, idx)
                continue

            if obs_o != obs_r:
                meta_o = (
                    orig_meta[index_o]
                    if orig_meta is not None and index_o is not None
                    else None
                )
                meta_r = (
                    recomp_meta[index_r]
                    if recomp_meta is not None and index_r is not None
                    else None
                )
                _record_observable_difference(
                    ctx,
                    index_o,
                    index_r,
                    ins_o,
                    ins_r,
                    obs_o,
                    obs_r,
                    meta_o,
                    meta_r,
                )
                return False
            _invalidate_save_slots(ctx, obs_o)
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
            _record_pair_categories(
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
                if not _divergences_justified(ctx, orig, recomp):
                    return False

            _commit_memory(ctx, obs_o, idx)

        if not _discharge_run_obligations(
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
