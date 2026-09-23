"""Isomorphic-CFG strategy: independently built graphs paired by structure,
with block-local alignment."""

from __future__ import annotations

from reccmp.compare.asm.instgen import InstructionMeta
from reccmp.compare.asm.ir import (
    AsmRole,
    AsmStream,
    ResolvedAsm,
    instruction_at,
    is_data_row,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.verifier.block_align import (
    align_block_lines,
    dp_line,
)
from reccmp.compare.asm.verifier.cfg_build import (
    block_terminator,
    build_side_cfg,
    canonicalize_side_cfg,
    pair_cfg_blocks,
)
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
from reccmp.compare.asm.verifier.evidence import record_operand_candidate
from reccmp.compare.asm.verifier.obligations import (
    accept_agreeing_pair,
    admit_unsupported_identical,
    callee_save_swap,
    discharge_run_obligations,
    one_sided_ok,
    switch_index_observation,
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


def verify_isomorphic_cfg_effective_match(
    orig_asm: AsmStream,
    recomp_asm: AsmStream,
    orig_targets: list[int | None],
    recomp_targets: list[int | None],
    metadata: FunctionMetadata | None = None,
    orig_meta: list[InstructionMeta | None] | None = None,
    recomp_meta: list[InstructionMeta | None] | None = None,
    recorder: AnalysisRecorder | None = None,
    orig_addrs: list[int | None] | None = None,
    recomp_addrs: list[int | None] | None = None,
    orig_roles: list[AsmRole] | None = None,
    recomp_roles: list[AsmRole] | None = None,
) -> bool:
    """CFG verification that tolerates different instruction counts:
    per-side block graphs matched structurally, block contents aligned
    locally, one-sided unobservable instructions allowed. This proves
    register-allocation wobble in its full generality — renames composed
    with folded loads, elided copies and shifted branch displacements.

    Recognized switch jump tables become ``caseN`` edges so isomorphic
    pairing compares entry count and case→block topology.
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-return-statements
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    orig_stream = resolve_asm_stream(orig_asm)
    recomp_stream = resolve_asm_stream(recomp_asm)
    if orig_roles is not None and len(orig_roles) == len(orig_stream):
        orig_stream = ResolvedAsm(
            orig_stream.displays,
            orig_stream.instructions,
            list(orig_roles),
            from_ir=True,
            jump_tables=orig_stream.jump_tables,
            instruction_ids=orig_stream.instruction_ids,
        )
    if recomp_roles is not None and len(recomp_roles) == len(recomp_stream):
        recomp_stream = ResolvedAsm(
            recomp_stream.displays,
            recomp_stream.instructions,
            list(recomp_roles),
            from_ir=True,
            jump_tables=recomp_stream.jump_tables,
            instruction_ids=recomp_stream.instruction_ids,
        )
    orig_asm = orig_stream.displays
    recomp_asm = recomp_stream.displays
    cfg_o = build_side_cfg(
        orig_stream,
        orig_targets,
        recorder=recorder,
        side="orig",
        addrs=orig_addrs,
    )
    cfg_r = build_side_cfg(
        recomp_stream,
        recomp_targets,
        recorder=recorder,
        side="recomp",
        addrs=recomp_addrs,
    )
    if cfg_o is None or cfg_r is None:
        return False
    cfg_o = canonicalize_side_cfg(cfg_o, orig_asm)
    cfg_r = canonicalize_side_cfg(cfg_r, recomp_asm)
    pairs = pair_cfg_blocks(cfg_o, cfg_r, recorder)
    if pairs is None:
        return False
    pair_ids = {pair: n for n, pair in enumerate(pairs)}

    # Align every paired block's lines once, up front.
    alignments: dict[tuple[int, int], list[tuple[int | None, int | None]]] = {}
    any_shifted = False
    for block_o, block_r in pairs:
        start_o, end_o = cfg_o.starts[block_o], cfg_o.ends[block_o]
        start_r, end_r = cfg_r.starts[block_r], cfg_r.ends[block_r]
        indices_o = [i for i in range(start_o, end_o) if i not in cfg_o.owned_data]
        indices_r = [i for i in range(start_r, end_r) if i not in cfg_r.owned_data]
        aligned = align_block_lines(
            [dp_line(orig_stream, i) for i in indices_o],
            [dp_line(recomp_stream, i) for i in indices_r],
        )
        if aligned is None:
            if recorder is not None:
                recorder.mark_inconclusive(
                    "alignment_failure",
                    start_o,
                    start_r,
                    {
                        "stage": "block_alignment",
                        "orig_block_length": len(indices_o),
                        "recomp_block_length": len(indices_r),
                        "orig_block_count": len(cfg_o.starts),
                        "recomp_block_count": len(cfg_r.starts),
                    },
                )
            return False
        # The block terminators (control lines) must pair with each other:
        # a one-sided branch or return breaks the matched structure.
        for local_o, local_r in aligned:
            kind_o = cfg_o.kinds[indices_o[local_o]] if local_o is not None else "code"
            kind_r = cfg_r.kinds[indices_r[local_r]] if local_r is not None else "code"
            if kind_o != kind_r or (local_o is None and kind_r != "code"):
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "alignment_failure",
                        indices_o[local_o] if local_o is not None else None,
                        indices_r[local_r] if local_r is not None else None,
                        {
                            "stage": "block_terminator_alignment",
                            "orig_kind": kind_o,
                            "recomp_kind": kind_r,
                        },
                    )
                return False
        if any(local_o is None or local_r is None for local_o, local_r in aligned):
            any_shifted = True
        alignments[(block_o, block_r)] = [
            (
                indices_o[local_o] if local_o is not None else None,
                indices_r[local_r] if local_r is not None else None,
            )
            for local_o, local_r in aligned
        ]

    entry: dict[tuple[int, int], CfgState] = {
        (0, 0): CfgState(
            SideState(rename_slots=False),
            SideState(rename_slots=False),
            ("cfg_mem_init",),
        )
    }
    pending: list[tuple[int, int]] = [(0, 0)]
    visits = 0

    def run_pair(pair: tuple[int, int]) -> bool:
        # pylint: disable=too-many-branches,too-many-statements
        # pylint: disable=too-many-return-statements,too-many-locals
        block_o, block_r = pair
        flow = clone_cfg_state(entry[pair])
        orig_state, recomp_state = flow.orig, flow.recomp
        ctx = Context(gen=flow.memory, metadata=metadata, recorder=recorder)
        seed_context_from_cfg(ctx, flow)
        edges_o = cfg_o.succ[block_o]
        edges_r = cfg_r.succ[block_r]
        for index_o, index_r in alignments[pair]:
            if index_o is None or index_r is None:
                if index_o is None:
                    assert index_r is not None
                    side, other_side = recomp_state, orig_state
                    line, position = recomp_asm[index_r], index_r
                    side_stream = recomp_stream
                else:
                    side, other_side = orig_state, recomp_state
                    line, position = orig_asm[index_o], index_o
                    side_stream = orig_stream
                # A one-sided instruction can never store (any observable
                # rejects it), so the memory generation is unaffected.
                side_ins = None
                side_data = is_data_row(side_stream, position)
                if not side_data:
                    try:
                        side_ins = instruction_at(side_stream, position)
                    except (Reject, IndexError, KeyError, ValueError, TypeError):
                        side_ins = None
                if not one_sided_ok(
                    side,
                    other_side,
                    ctx,
                    position,
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
            line_o, line_r = orig_asm[index_o], recomp_asm[index_r]
            try:
                ins_o = instruction_at(orig_stream, index_o)
                ins_r = instruction_at(recomp_stream, index_r)
                record_operand_candidate(ctx, index_o, index_r, ins_o, ins_r)
                obs_o: list = []
                obs_r: list = []
                state_before_o = clone_state(orig_state)
                state_before_r = clone_state(recomp_state)
                execute(orig_state, ctx, index_o, ins_o, obs_o)
                execute(recomp_state, ctx, index_o, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                if line_o != line_r:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "unsupported_instruction", index_o, index_r
                        )
                    return False
                meta_o = orig_meta[index_o] if orig_meta is not None else None
                meta_r = recomp_meta[index_r] if recomp_meta is not None else None
                if admit_unsupported_identical(
                    orig_state, recomp_state, ctx, index_o, meta_o, meta_r
                ):
                    continue
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "unsupported_instruction", index_o, index_r
                    )
                return False

            guard_state_size(orig_state, ctx)
            guard_state_size(recomp_state, ctx)

            if callee_save_swap(
                ctx,
                ins_o,
                ins_r,
                obs_o,
                obs_r,
                orig_state,
                recomp_state,
            ):
                commit_memory(ctx, obs_o, index_o)
                continue

            # Canonicalize local control-flow targets: the block pairing
            # already proved that both sides' edges lead to the same matched
            # blocks, so the displacement text is irrelevant. External
            # targets keep their raw text — those must match exactly.
            # Recognized switch tables: caseN edges encode destinations; keep
            # only the index-register values so table-base placeholders may differ.
            kind = cfg_o.kinds[index_o]
            if (
                kind in ("jcc", "jmp")
                and edges_o.get("taken" if kind == "jcc" else "jmp") != "external"
            ):
                for entries in (obs_o, obs_r):
                    for k, obs_entry in enumerate(entries):
                        if obs_entry[0] in CONTROL_TAGS - {"jmpind"}:
                            entries[k] = (*obs_entry[:-1], ("L", kind))
            if any(role.startswith("case") for role in edges_o):
                idx_o = switch_index_observation(state_before_o, ins_o)
                idx_r = switch_index_observation(state_before_r, ins_r)
                for entries, idx in ((obs_o, idx_o), (obs_r, idx_r)):
                    for k, obs_entry in enumerate(entries):
                        if obs_entry[0] == "jmpind":
                            entries[k] = ("jmpind", ("L", "switch"), idx)

            if not accept_agreeing_pair(
                ctx,
                index_o,
                index_r,
                (state_before_o, state_before_r),
                (orig_state, recomp_state),
                (ins_o, ins_r),
                (obs_o, obs_r),
                (
                    orig_meta[index_o] if orig_meta is not None else None,
                    recomp_meta[index_r] if recomp_meta is not None else None,
                ),
            ):
                return False
            if any(obs_entry[0] == "jmpind" for obs_entry in obs_o):
                # Recognized switch tables already expanded to caseN edges.
                if not any(role.startswith("case") for role in edges_o):
                    if recorder is not None:
                        recorder.mark_inconclusive("indirect_jump", index_o, index_r)
                    return False

            # A direct edge outside the excerpt exposes the complete machine
            # state to code this proof does not inspect.
            if kind in ("jmp", "jcc") and (
                edges_o.get("taken" if kind == "jcc" else "jmp") == "external"
            ):
                if not converged(orig_state, recomp_state):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "external_control_flow_state",
                            index_o,
                            index_r,
                            {"edge_kind": kind},
                        )
                    return False
            commit_memory(ctx, obs_o, index_o)

        last_o = block_terminator(
            cfg_o.starts[block_o],
            cfg_o.ends[block_o],
            cfg_o.kinds,
            cfg_o.owned_data,
        )
        last_r = block_terminator(
            cfg_r.starts[block_r],
            cfg_r.ends[block_r],
            cfg_r.kinds,
            cfg_r.owned_data,
        )
        last_kind = cfg_o.kinds[last_o]
        if last_kind == "ret":
            if not discharge_run_obligations(
                ctx,
                orig_state,
                recomp_state,
                recorder,
                last_o,
                last_r,
            ):
                return False
        if "fallout" in edges_o:
            # Falling out of the disassembled function is not a modeled exit.
            if recorder is not None:
                recorder.mark_inconclusive(
                    "function_fallthrough",
                    last_o,
                    last_r,
                )
            return False

        outgoing = capture_cfg_state(orig_state, recomp_state, ctx)
        if recorder is not None:
            recorder.reasons.update(ctx.categories)
        for role, to_o in edges_o.items():
            if to_o == "external":
                continue
            to_r = edges_r[role]
            assert isinstance(to_o, int) and isinstance(to_r, int)
            successor = (to_o, to_r)
            if successor not in entry:
                entry[successor] = clone_cfg_state(outgoing)
                pending.append(successor)
            else:
                joined = join_states(entry[successor], outgoing, pair_ids[successor])
                if joined is None:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "state_join_failure",
                            cfg_o.starts[to_o],
                            cfg_r.starts[to_r],
                            join_failure_facts(entry[successor], outgoing),
                        )
                    return False
                if not states_equal(joined, entry[successor]):
                    entry[successor] = joined
                    pending.append(successor)
        return True

    try:
        while pending:
            pair = pending.pop()
            visits += 1
            if visits > 8 * len(pairs) + 64:
                if recorder is not None:
                    recorder.mark_inconclusive("analysis_limit")
                return False
            if not run_pair(pair):
                return False
    except (Reject, RecursionError):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    if any_shifted and recorder is not None:
        recorder.reasons.add("instruction_reorder")
    return True
