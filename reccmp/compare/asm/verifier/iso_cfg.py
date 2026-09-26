"""Isomorphic-CFG strategy: independently built graphs paired by structure,
with block-local alignment."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from reccmp.compare.asm.instgen import InstructionMeta
from reccmp.compare.asm.ir import (
    AsmRole,
    AsmStream,
    ResolvedAsm,
    instruction_at,
    is_data_row,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import Instruction, Reject
from reccmp.compare.asm.verifier.addresses import unwind_spadd
from reccmp.compare.asm.verifier.frame import maybe_frame_pointer
from reccmp.compare.asm.verifier.block_align import (
    DpLine,
    align_block_lines,
    dp_line,
)
from reccmp.compare.asm.verifier.cfg_build import (
    block_terminator,
    _SideCfg,
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
from reccmp.compare.asm.verifier.schedule import schedule_like
from reccmp.compare.asm.verifier.semantics import canon_condition, esp_add, execute
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    JCC_MNEMONICS,
    STRING_OPS,
    Context,
    FunctionMetadata,
    SideState,
    clone_state,
    commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import AnalysisRecorder, FactValue

if TYPE_CHECKING:
    from reccmp.compare.callee_cleanup import CallStackEffect


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
    # Without a one-to-one pairing of the blocks, which blocks the product
    # pairs where the graphs differ is a guess: enough to prove the two
    # equal, but a difference found under it may be the guess's own (a
    # redundant test on one side paired with the other's next branch). Such
    # a failure reports why the blocks do not pair one to one instead.
    anchored = pair_cfg_blocks(cfg_o, cfg_r) is not None
    product_recorder = recorder
    if recorder is not None and not anchored:
        product_recorder = AnalysisRecorder(
            orig_addrs=recorder.orig_addrs, recomp_addrs=recorder.recomp_addrs
        )
    proved = _verify_product(
        cfg_o,
        cfg_r,
        orig_stream,
        recomp_stream,
        metadata,
        orig_meta,
        recomp_meta,
        product_recorder,
    )
    if not proved:
        # A local may live in a stack slot on one side and in a register
        # on the other: try again with each side's frame its own (see
        # verifier.frame). Only a proof counts; the first attempt says why
        # the pair failed.
        promoted = AnalysisRecorder(orig_addrs=orig_addrs, recomp_addrs=recomp_addrs)
        if _verify_product(
            cfg_o,
            cfg_r,
            orig_stream,
            recomp_stream,
            metadata,
            orig_meta,
            recomp_meta,
            promoted,
            promote=True,
            addrs=(orig_addrs, recomp_addrs),
        ):
            if recorder is not None:
                recorder.reasons |= promoted.reasons | {"frame_slot_promotion"}
            return True
    if recorder is not None and product_recorder is not recorder:
        assert product_recorder is not None
        if proved:
            recorder.reasons |= product_recorder.reasons
        else:
            pair_cfg_blocks(cfg_o, cfg_r, recorder, _product_stop(product_recorder))
    return proved


def _verify_product(
    cfg_o: _SideCfg,
    cfg_r: _SideCfg,
    orig_stream: ResolvedAsm,
    recomp_stream: ResolvedAsm,
    metadata: FunctionMetadata | None,
    orig_meta: list[InstructionMeta | None] | None,
    recomp_meta: list[InstructionMeta | None] | None,
    recorder: AnalysisRecorder | None,
    *,
    promote: bool = False,
    addrs: tuple[list[int | None] | None, list[int | None] | None] = (None, None),
) -> bool:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    """Verify the product of the two graphs from the entry blocks: every
    reachable node's aligned instructions, with state pairs flowing along
    the paired edges and joined where nodes meet. ``promote``: each side
    keeps its private frame to itself (see verifier.frame); ``addrs`` give
    the instructions' addresses, for the stack effects of calls."""
    orig_asm = orig_stream.displays
    recomp_asm = recomp_stream.displays
    blocks = len(cfg_o.starts) + len(cfg_r.starts)
    node_limit = 4 * blocks + 16

    # Nodes of the product, in creation order (the order ids join states);
    # each node's aligned instruction pairs; whether its branch pairs the
    # other side's successors swapped.
    nodes: dict[_Node, int] = {}
    alignments: dict[_Node, list[tuple[int | None, int | None]]] = {}
    orientation: dict[_Node, bool] = {}
    any_shifted = False

    def align(node: _Node) -> list[tuple[int | None, int | None]] | None:
        nonlocal any_shifted
        indices_o = _run_indices(cfg_o, node[0])
        indices_r = _run_indices(cfg_r, node[1])
        start_o, start_r = cfg_o.starts[node[0][0]], cfg_r.starts[node[1][0]]
        scheduled = schedule_like(orig_stream, recomp_stream, indices_o, indices_r)
        if scheduled != indices_r:
            indices_r = scheduled
            any_shifted = True
        aligned = align_block_lines(
            [_dp_line(orig_stream, i, promote) for i in indices_o],
            [_dp_line(recomp_stream, i, promote) for i in indices_r],
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
            return None
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
                return None
        if any(local_o is None or local_r is None for local_o, local_r in aligned):
            any_shifted = True
        return [
            (
                indices_o[local_o] if local_o is not None else None,
                indices_r[local_r] if local_r is not None else None,
            )
            for local_o, local_r in aligned
        ]

    def node_at(block_o: int, block_r: int) -> _Node | None:
        """The node entered at this pair of blocks, created (and aligned)
        on first entry; None, recorded, when the two sides cannot pair."""
        node = _form_node(cfg_o, cfg_r, block_o, block_r)
        if node in nodes:
            return node
        exits_o = _exits(cfg_o, node[0][-1])
        exits_r = _exits(cfg_r, node[1][-1])
        if not (
            _exits_correspond(exits_o, exits_r, swapped=False)
            or _exits_correspond(exits_o, exits_r, swapped=True)
        ):
            if recorder is not None:
                recorder.mark_inconclusive(
                    "non_isomorphic_cfg",
                    orig_index=cfg_o.starts[node[0][-1]],
                    recomp_index=cfg_r.starts[node[1][-1]],
                    facts={
                        "failure": (
                            "edge_roles"
                            if set(exits_o) != set(exits_r)
                            else "external_edge"
                        ),
                        "orig_block_count": len(cfg_o.starts),
                        "recomp_block_count": len(cfg_r.starts),
                        "orig_edge_roles": ",".join(sorted(exits_o)),
                        "recomp_edge_roles": ",".join(sorted(exits_r)),
                    },
                )
            return None
        if len(nodes) >= node_limit:
            if recorder is not None:
                recorder.mark_inconclusive("analysis_limit")
            return None
        aligned = align(node)
        if aligned is None:
            return None
        nodes[node] = len(nodes)
        alignments[node] = aligned
        return node

    first = node_at(0, 0)
    if first is None:
        return False
    entry: dict[_Node, CfgState] = {
        first: CfgState(
            SideState(rename_slots=False, frame={} if promote else None),
            SideState(rename_slots=False, frame={} if promote else None),
            ("cfg_mem_init",),
        )
    }
    pending: list[_Node] = [first]
    visits = 0

    def run_node(node: _Node) -> bool:
        # pylint: disable=too-many-branches,too-many-statements
        # pylint: disable=too-many-return-statements,too-many-locals
        flow = clone_cfg_state(entry[node])
        orig_state, recomp_state = flow.orig, flow.recomp
        ctx = Context(gen=flow.memory, metadata=metadata, recorder=recorder)
        seed_context_from_cfg(ctx, flow)
        edges_o = _exits(cfg_o, node[0][-1])
        edges_r = _exits(cfg_r, node[1][-1])
        swapped = False
        last_o = last_r = None
        for index_o, index_r in alignments[node]:
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
            last_o, last_r = index_o, index_r
            line_o, line_r = orig_asm[index_o], recomp_asm[index_r]
            try:
                ins_o = instruction_at(orig_stream, index_o)
                ins_r = instruction_at(recomp_stream, index_r)
                record_operand_candidate(
                    ctx, index_o, index_r, ins_o, ins_r, (orig_state, recomp_state)
                )
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

            if promote and any(entry[0] == "call" for entry in obs_o):
                # What reaches the callee from each side's own frame.
                effects = _agreed_effects(
                    _call_effect(metadata, addrs, 0, index_o),
                    _call_effect(metadata, addrs, 1, index_r),
                )
                for state, before, entries, effect in (
                    (orig_state, state_before_o, obs_o, effects[0]),
                    (recomp_state, state_before_r, obs_r, effects[1]),
                ):
                    entries.append(_pass_frame(state, before, index_o, effect))
            kind = cfg_o.kinds[index_o]
            if kind == "jcc":
                swapped = _inverted_branch(
                    obs_o,
                    obs_r,
                    ins_r=ins_r,
                    state_before_r=state_before_r,
                    exits_o=edges_o,
                    exits_r=edges_r,
                )
            # Canonicalize local control-flow targets: the product pairs
            # both sides' successors, so the displacement text is
            # irrelevant. External targets keep their raw text — those
            # must match exactly. (Jumps inside the function are not run:
            # their edge is the node's ``next``.)
            # Recognized switch tables: caseN edges encode destinations; keep
            # only the index-register values so table-base placeholders may differ.
            if kind == "jcc" and edges_o.get("taken") != "external":
                for entries in (obs_o, obs_r):
                    for k, obs_entry in enumerate(entries):
                        if obs_entry[0] in CONTROL_TAGS - {"jmpind"}:
                            entries[k] = (*obs_entry[:-1], ("L", kind))
            if kind == "jmp" and any(role.startswith("case") for role in edges_o):
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
            if ctx.stack_escaped and (orig_state.frame or recomp_state.frame):
                raise Reject  # a pointer into a promoted frame left the function

        if orientation.setdefault(node, swapped) != swapped or not _exits_correspond(
            edges_o, edges_r, swapped=swapped
        ):
            # The branch's two sides pair their successors one way on one
            # visit and another way (or neither) on this one.
            if recorder is not None:
                recorder.mark_inconclusive(
                    "non_isomorphic_cfg",
                    last_o,
                    last_r,
                    {"failure": "branch_orientation"},
                )
            return False
        if last_o is not None and cfg_o.kinds[last_o] == "ret":
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
            to_r = edges_r[_SWAPPED_ROLE[role] if swapped else role]
            assert isinstance(to_o, int) and isinstance(to_r, int)
            successor = node_at(to_o, to_r)
            if successor is None:
                return False
            if successor not in entry:
                entry[successor] = clone_cfg_state(outgoing)
                pending.append(successor)
            else:
                joined = join_states(entry[successor], outgoing, nodes[successor])
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
            node = pending.pop()
            visits += 1
            if visits > 8 * blocks + 64:
                if recorder is not None:
                    recorder.mark_inconclusive("analysis_limit")
                return False
            if not run_node(node):
                return False
    except (Reject, RecursionError):
        if recorder is not None:
            recorder.mark_inconclusive("analysis_limit")
        return False
    if any_shifted and recorder is not None:
        recorder.reasons.add("instruction_reorder")
    return True


# ---------------------------------------------------------------------------
# The product of the two graphs
#
# A node pairs a run of blocks on each side. A run continues along an
# unconditional edge inside the function (fall-through or jump) while the
# two runs' exits do not correspond: the other side has that code inline (a
# duplicated tail, a jump over a block). A block may belong to several
# nodes, so a block one side shares between paths pairs with each of the
# other side's copies, each visit checked on its own state. Only the edges
# between runs are paired; which instructions run is unchanged, so
# verifying every reachable node verifies both functions.

_Node = tuple[tuple[int, ...], tuple[int, ...]]

_MAX_RUN_BLOCKS = 8

_SWAPPED_ROLE = {"taken": "fall", "fall": "taken"}

# A conditional jump and the one taken exactly when it is not.
_COMPLEMENT_JCC = {
    a: b
    for x, y in (
        ("je", "jne"),
        ("jl", "jge"),
        ("jle", "jg"),
        ("jb", "jae"),
        ("jbe", "ja"),
        ("js", "jns"),
        ("jo", "jno"),
        ("jp", "jnp"),
    )
    for a, b in ((x, y), (y, x))
}


def _exits(cfg: _SideCfg, block: int) -> dict[str, int | str]:
    """A block's successor edges by role; an unconditional edge to a block
    inside the function is ``next``, whether it falls through or jumps."""
    edges = cfg.succ[block]
    if len(edges) == 1:
        ((role, dest),) = edges.items()
        if role in ("fall", "jmp") and dest != "external":
            return {"next": dest}
    return dict(edges)


def _exits_correspond(
    exits_o: dict[str, int | str], exits_r: dict[str, int | str], *, swapped: bool
) -> bool:
    """Whether the two sides' exits pair role for role (a branch's taken
    and fall-through edges crosswise when ``swapped``), each leaving the
    function or staying inside on both sides."""
    if set(exits_o) != set(exits_r):
        return False
    if swapped and set(exits_o) != {"taken", "fall"}:
        return False
    return all(
        (dest == "external")
        == (exits_r[_SWAPPED_ROLE[role] if swapped else role] == "external")
        for role, dest in exits_o.items()
    )


def _form_node(cfg_o: _SideCfg, cfg_r: _SideCfg, block_o: int, block_r: int) -> _Node:
    """The runs of blocks entered at ``block_o`` and ``block_r``: a side
    whose run ends in ``next`` takes that block in too while the exits do
    not correspond."""
    run_o, run_r = [block_o], [block_r]
    while len(run_o) + len(run_r) < _MAX_RUN_BLOCKS:
        exits_o, exits_r = _exits(cfg_o, run_o[-1]), _exits(cfg_r, run_r[-1])
        if _exits_correspond(exits_o, exits_r, swapped=False) or _exits_correspond(
            exits_o, exits_r, swapped=True
        ):
            break
        if set(exits_o) == {"next"} and exits_o["next"] not in run_o:
            run_o.append(int(exits_o["next"]))
        elif set(exits_r) == {"next"} and exits_r["next"] not in run_r:
            run_r.append(int(exits_r["next"]))
        else:
            break
    return tuple(run_o), tuple(run_r)


def _run_indices(cfg: _SideCfg, run: tuple[int, ...]) -> list[int]:
    """The instructions a run executes: its blocks' own lines, without the
    jumps inside the function (their edge is the run's ``next``)."""
    indices: list[int] = []
    for block in run:
        start, end = cfg.starts[block], cfg.ends[block]
        last = block_terminator(start, end, cfg.kinds, cfg.owned_data)
        internal_jump = cfg.kinds[last] == "jmp" and set(_exits(cfg, block)) == {"next"}
        indices += [
            i
            for i in range(start, end)
            if i not in cfg.owned_data and not (internal_jump and i == last)
        ]
    return indices


def _inverted_branch(
    obs_o: list,
    obs_r: list,
    *,
    ins_r: Instruction,
    state_before_r: SideState,
    exits_o: dict[str, int | str],
    exits_r: dict[str, int | str],
) -> bool:
    """Whether the recompiled branch is the original's with the condition
    inverted and its successors swapped: its predicates differ, and the
    complement of the recompiled condition (on the same flags) is the
    original's. If so, observe the recompiled branch as that complement,
    so the pair's predicates compare as equal."""
    branch_o = next((k for k, e in enumerate(obs_o) if e[0] == "branch"), None)
    branch_r = next((k for k, e in enumerate(obs_r) if e[0] == "branch"), None)
    if branch_o is None or branch_r is None:
        return False
    if obs_o[branch_o][1] == obs_r[branch_r][1]:
        return False
    complement = _COMPLEMENT_JCC.get(ins_r.mnemonic)
    if complement is None or not _exits_correspond(exits_o, exits_r, swapped=True):
        return False
    inverted = canon_condition(JCC_MNEMONICS[complement], state_before_r)
    if inverted != obs_o[branch_o][1]:
        return False
    obs_r[branch_r] = ("branch", inverted, *obs_r[branch_r][2:])
    return True


def _product_stop(recorder: AnalysisRecorder) -> dict[str, FactValue]:
    """Where and why the product stopped, as facts of the failure it does
    not report: a difference's kind, or a blocker with its stage."""
    difference = recorder.best_difference
    if difference is not None:
        return {
            "product_stop": difference.kind,
            "product_orig_address": difference.orig.address,
            "product_recomp_address": difference.recomp.address,
        }
    stop = recorder.inconclusive_reason or "analysis_limit"
    location = recorder.inconclusive_location
    if location is None:
        return {"product_stop": stop}
    detail = location.facts.get("stage") or location.facts.get("failure")
    recomp_address = (
        location.facts.get("recomp_address")
        if location.image == "orig"
        else location.address
    )
    return {
        "product_stop": f"{stop}/{detail}" if detail else stop,
        "product_orig_address": location.address if location.image == "orig" else None,
        "product_recomp_address": recomp_address,
    }


def _dp_line(stream: ResolvedAsm, index: int, promote: bool) -> DpLine:
    """The alignment view of an instruction. With the frame promoted, a
    store to a frame slot is no observable: it may pair with any
    instruction, or with none."""
    line = dp_line(stream, index)
    if not promote or line.line_class != "store":
        return line
    try:
        ins = instruction_at(stream, index)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        return line
    op = ins.operands[0] if ins.operands else None
    if (  # pylint: disable=too-many-boolean-expressions
        ins.prefix
        or ins.mnemonic in STRING_OPS
        or op is None
        or op[0] != "mem"
        or op[2]
        or op[5]
        or op[3] not in ([("esp", 1)], [("ebp", 1)])
    ):
        return line
    return dataclasses.replace(line, line_class="none")


def _call_effect(
    metadata: FunctionMetadata | None,
    addrs: tuple[list[int | None] | None, list[int | None] | None],
    side: int,
    index: int,
) -> CallStackEffect | None:
    """The stack effect of one side's call at ``index``, from its binary."""
    side_addrs = addrs[side]
    if metadata is None or metadata.stack_effects is None or side_addrs is None:
        return None
    address = side_addrs[index] if index < len(side_addrs) else None
    return metadata.stack_effects[side](address) if address is not None else None


def _agreed_effects(
    orig: CallStackEffect | None, recomp: CallStackEffect | None
) -> tuple[CallStackEffect | None, CallStackEffect | None]:
    """A paired call's stack effects, each side's from its own binary. One
    read from returns that may not be the callee's own counts only when
    the other side's certain one removes the same bytes."""

    def agreed(effect, other):
        if effect is None or effect.certain:
            return effect
        if (
            other is not None
            and other.certain
            and other.callee_pops == effect.callee_pops
        ):
            return effect
        return None

    return agreed(orig, recomp), agreed(recomp, orig)


def _pass_frame(
    state: SideState,
    before: SideState,
    index: int,
    effect: CallStackEffect | None,
) -> tuple:
    """A call with the frame promoted: the observation of the promoted
    slots the callee may read (its arguments, by offset from the stack
    pointer at the call; every slot above it when their extent is
    unknown). The callee may write them, so they hold its results after
    the call, and its ``ret N`` sets the stack pointer, when known."""
    assert state.frame is not None
    esp = before.read_reg("esp")
    root, top = unwind_spadd(esp)
    if root != ("init", "sp"):
        if state.frame:
            raise Reject  # the arguments cannot be placed
        return ("frame_args",)
    arguments = effect.arguments if effect is not None else None
    for offset in [offset for offset in state.frame if offset < top]:
        # The callee's own frame, from the return address down.
        del state.frame[offset]
    passed = []
    for offset, (width, value) in sorted(state.frame.items()):
        if arguments is not None and offset >= top + arguments:
            continue
        if value is None or maybe_frame_pointer(value):
            # An unreadable argument, or a pointer into the frame: the
            # callee could reach the promoted slots through it.
            raise Reject
        passed.append((offset - top, width, value))
        state.frame[offset] = (width, ("callarg", index, offset - top, width))
    if effect is not None and effect.callee_pops is not None:
        state.write_reg("esp", esp_add(esp, effect.callee_pops))
    return ("frame_args", tuple(passed))
