"""Positional CFG strategy: equal-shape graphs with dataflow joins."""

from __future__ import annotations

from collections.abc import Sequence

from reccmp.compare.asm.ir import (
    DecodedInstruction,
    instruction_semantic_key,
    local_branch_targets,
)
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.verifier.dataflow import (
    CfgState,
    capture_cfg_state,
    clone_cfg_state,
    converged,
    join_failure_facts,
    join_states,
    seed_context_from_cfg,
    states_equal,
)
from reccmp.compare.asm.verifier.evidence import (
    record_operand_candidate,
    target_facts,
)
from reccmp.compare.asm.verifier.obligations import (
    accept_agreeing_pair,
    admit_unsupported_identical,
    callee_save_swap,
    discharge_run_obligations,
)
from reccmp.compare.asm.verifier.semantics import execute
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    JCC_MNEMONICS,
    Context,
    FunctionMetadata,
    SideState,
    clone_state,
    commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import AnalysisRecorder

# ---------------------------------------------------------------------------
# CFG-aware verification


def verify_cfg_effective_match(
    orig_rows: Sequence[DecodedInstruction],
    recomp_rows: Sequence[DecodedInstruction],
    metadata: FunctionMetadata | None = None,
    recorder: AnalysisRecorder | None = None,
) -> bool:
    """CFG-aware verification: split both sequences into basic blocks
    (which must be structurally identical under the line pairing), then
    verify every block with state pairs flowing along the edges — forked
    at conditional branches and joined at merge points. This proves pairs
    that linear verification cannot (a divergence that is live across a
    branch and consumed after the join) and rejects pairs it must
    (a divergence created in one arm and overwritten after the join)."""
    # pylint: disable=too-many-branches,too-many-statements,too-many-return-statements
    # pylint: disable=too-many-locals
    orig_targets = local_branch_targets(orig_rows)
    recomp_targets = local_branch_targets(recomp_rows)
    total = len(orig_rows)
    if len(recomp_rows) != total or total == 0:
        if recorder is not None:
            recorder.mark_inconclusive(
                "alignment_failure",
                facts={
                    "stage": "positional_cfg_precondition",
                    "orig_instruction_count": total,
                    "recomp_instruction_count": len(recomp_rows),
                },
            )
        return False
    if orig_targets != recomp_targets:
        if recorder is not None:
            differing = next(
                index
                for index, pair in enumerate(zip(orig_targets, recomp_targets))
                if pair[0] != pair[1]
            )
            recorder.record_difference(
                "branch_target",
                differing,
                differing,
                target_facts(orig_rows[differing], orig_targets[differing]),
                target_facts(recomp_rows[differing], recomp_targets[differing]),
            )
        return False

    def classify(rows: Sequence[DecodedInstruction]) -> list[str]:
        kinds = []
        for row in rows:
            if not row.is_code:
                kinds.append("data")
            elif row.mnemonic in JCC_MNEMONICS or row.mnemonic in (
                "loop",
                "loope",
                "loopne",
                "jcxz",
                "jecxz",
            ):
                kinds.append("jcc")
            elif row.mnemonic in ("jmp", "ret"):
                kinds.append(row.mnemonic)
            else:
                kinds.append("code")
        return kinds

    kinds = classify(orig_rows)
    recomp_kinds = classify(recomp_rows)
    if kinds != recomp_kinds:
        if recorder is not None:
            differing = next(
                i
                for i, (kind_o, kind_r) in enumerate(zip(kinds, recomp_kinds))
                if kind_o != kind_r
            )
            recorder.mark_inconclusive(
                "non_isomorphic_cfg",
                differing,
                differing,
                {
                    "failure": "instruction_control_kind",
                    "orig_kind": kinds[differing],
                    "recomp_kind": recomp_kinds[differing],
                },
            )
        return False
    leaders = {0}
    for i in range(total):
        target = orig_targets[i]
        if target is not None:
            if not 0 <= target < total:
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "invalid_control_flow_target",
                        i,
                        i,
                        {"target_instruction_index": target},
                    )
                return False
            if kinds[target] == "data":
                # A control-flow edge into bytes classified as a table is an
                # inconsistent disassembly, not a block this verifier can run.
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "jump_table_data",
                        i,
                        i,
                        {
                            "target_instruction_index": target,
                            "data_line": orig_rows[target].display,
                        },
                    )
                return False
            leaders.add(target)
        if kinds[i] in ("jcc", "jmp", "ret", "data") and i + 1 < total:
            leaders.add(i + 1)
    order = sorted(leaders)
    starts = {start: n for n, start in enumerate(order)}
    ends = [order[n + 1] if n + 1 < len(order) else total for n in range(len(order))]

    def successors(block: int) -> list[int]:
        last = ends[block] - 1
        kind = kinds[last]
        if kind in ("ret", "data"):
            return []
        if kind == "jmp":
            target = orig_targets[last]
            return [starts[target]] if target is not None else []
        if kind == "jcc":
            result = []
            target = orig_targets[last]
            if target is not None:
                result.append(starts[target])
            if ends[block] < total:
                result.append(starts[ends[block]])
            return result
        return [starts[ends[block]]] if ends[block] < total else []

    entry: dict[int, CfgState] = {
        0: CfgState(
            SideState(rename_slots=False),
            SideState(rename_slots=False),
            ("cfg_mem_init",),
        )
    }
    pending = [0]
    visits = 0

    def run_block(block: int) -> bool:
        flow = clone_cfg_state(entry[block])
        orig_state, recomp_state = flow.orig, flow.recomp
        ctx = Context(gen=flow.memory, metadata=metadata, recorder=recorder)
        seed_context_from_cfg(ctx, flow)
        for i in range(order[block], ends[block]):
            ins_o, ins_r = orig_rows[i], recomp_rows[i]
            same = instruction_semantic_key(ins_o) == instruction_semantic_key(ins_r)
            if kinds[i] == "data":
                if not same:
                    return False
                continue
            try:
                record_operand_candidate(
                    ctx, i, i, ins_o, ins_r, (orig_state, recomp_state)
                )
                obs_o: list = []
                obs_r: list = []
                state_before_o = clone_state(orig_state)
                state_before_r = clone_state(recomp_state)
                execute(orig_state, ctx, i, ins_o, obs_o)
                execute(recomp_state, ctx, i, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                if not same:
                    if recorder is not None:
                        recorder.mark_inconclusive("unsupported_instruction", i, i)
                    return False
                if admit_unsupported_identical(
                    orig_state, recomp_state, ctx, i, ins_o, ins_r
                ):
                    continue
                if recorder is not None:
                    recorder.mark_inconclusive("unsupported_instruction", i, i)
                return False

            guard_state_size(orig_state, ctx)
            guard_state_size(recomp_state, ctx)

            # Canonicalize internal branch targets to block ids: with the
            # structural check done, differing displacement text (from
            # different instruction encodings) is irrelevant.
            target = orig_targets[i]
            if target is not None:
                for entries in (obs_o, obs_r):
                    for k, obs_entry in enumerate(entries):
                        if obs_entry[0] in CONTROL_TAGS - {"jmpind"}:
                            entries[k] = (*obs_entry[:-1], ("L", starts[target]))

            if callee_save_swap(
                ctx, ins_o, ins_r, obs_o, obs_r, orig_state, recomp_state
            ):
                commit_memory(ctx, obs_o, i)
                continue

            if not accept_agreeing_pair(
                ctx,
                i,
                i,
                (state_before_o, state_before_r),
                (orig_state, recomp_state),
                (ins_o, ins_r),
                (obs_o, obs_r),
            ):
                return False
            # We do not yet pair jump-table destinations. A computed jump
            # therefore cannot be justified by this CFG proof. The lockstep
            # verifier still handles identical/converged cases before this
            # path is attempted.
            if any(obs_entry[0] == "jmpind" for obs_entry in obs_o):
                if recorder is not None:
                    recorder.mark_inconclusive("indirect_jump", i, i)
                return False

            # A direct edge outside the excerpt exposes the complete machine
            # state to code this proof does not inspect. Be conservative for
            # both unconditional exits and the taken edge of a conditional.
            if kinds[i] in ("jmp", "jcc") and target is None:
                if not converged(orig_state, recomp_state):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "external_control_flow_state",
                            i,
                            i,
                            {"edge_kind": kinds[i]},
                        )
                    return False
            commit_memory(ctx, obs_o, i)

        last_kind = kinds[ends[block] - 1]
        if last_kind == "ret":
            if not discharge_run_obligations(
                ctx,
                orig_state,
                recomp_state,
                recorder,
                ends[block] - 1,
                ends[block] - 1,
            ):
                return False
        if last_kind == "code" and ends[block] == total:
            # Falling out of the disassembled function is not a modeled exit.
            return False

        outgoing = capture_cfg_state(orig_state, recomp_state, ctx)
        if recorder is not None:
            recorder.reasons.update(ctx.categories)
        # Propagate to successors.
        for successor in successors(block):
            if successor not in entry:
                entry[successor] = clone_cfg_state(outgoing)
                pending.append(successor)
            else:
                joined = join_states(entry[successor], outgoing, successor)
                if joined is None:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "state_join_failure",
                            order[successor],
                            order[successor],
                            join_failure_facts(entry[successor], outgoing),
                        )
                    return False
                if not states_equal(joined, entry[successor]):
                    entry[successor] = joined
                    pending.append(successor)
        return True

    try:
        while pending:
            block = pending.pop()
            visits += 1
            if visits > 8 * len(order) + 64:
                if recorder is not None:
                    recorder.mark_inconclusive("analysis_limit")
                return False
            if not run_block(block):
                return False
    except (Reject, RecursionError):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    return True
