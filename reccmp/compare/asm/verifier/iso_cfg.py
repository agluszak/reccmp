"""Isomorphic-CFG strategy: independently built graphs paired by structure,
with block-local alignment."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from collections.abc import Sequence

from reccmp.compare.asm.graph import EdgeRole
from reccmp.compare.asm.ir import (
    DecodedInstruction,
    FlowKind,
    FunctionImage,
    instruction_semantic_key,
)
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.operand import Mem, ScaledReg
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import Value, unwind_spadd
from reccmp.compare.asm.verifier.frame import maybe_frame_pointer
from reccmp.compare.asm.verifier.block_align import (
    DpLine,
    align_block_lines,
    dp_line,
)
from reccmp.compare.asm.verifier.blocks import (
    FALL,
    JUMP,
    NEXT,
    SWAPPED,
    TAKEN,
    Blocks,
    Exits,
    pair_heads,
    unsupported_control_flow,
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
from reccmp.compare.diagnosis import AnalysisRecorder

if TYPE_CHECKING:
    from reccmp.compare.callee_cleanup import CallStackEffect


def verify_isomorphic_cfg_effective_match(
    orig: FunctionImage,
    recomp: FunctionImage,
    metadata: FunctionMetadata | None = None,
    recorder: AnalysisRecorder | None = None,
    unanchored: AnalysisRecorder | None = None,
) -> bool:
    """CFG verification that tolerates different instruction counts:
    per-side block graphs matched structurally, block contents aligned
    locally, one-sided unobservable instructions allowed. This proves
    register-allocation wobble in its full generality — renames composed
    with folded loads, elided copies and shifted branch displacements.

    Recognized switch jump tables become ``caseN`` edges so isomorphic
    pairing compares entry count and case→block topology.

    When the blocks do not pair one to one, ``recorder`` gets why, and
    ``unanchored`` where the product stopped under its guessed pairing.
    """
    orig_rows, recomp_rows = orig.instructions, recomp.instructions
    if unsupported_control_flow(orig, recorder, "orig") or unsupported_control_flow(
        recomp, recorder, "recomp"
    ):
        return False
    cfg_o, cfg_r = Blocks(orig), Blocks(recomp)
    addrs = ([row.address for row in orig_rows], [row.address for row in recomp_rows])
    # Without a one-to-one pairing of the blocks, which blocks the product
    # pairs where the graphs differ is a guess: enough to prove the two
    # equal, but a difference found under it may be the guess's own (a
    # redundant test on one side paired with the other's next branch). Such
    # a failure reports why the blocks do not pair one to one, and the
    # difference goes to ``unanchored``: a lead, not a verdict.
    anchored = pair_heads(cfg_o, cfg_r) is not None
    product_recorder = recorder
    if recorder is not None and not anchored:
        product_recorder = unanchored or AnalysisRecorder(*addrs)
    proved = _verify_product(
        cfg_o, cfg_r, orig_rows, recomp_rows, metadata, product_recorder
    )
    if not proved:
        # A local may live in a stack slot on one side and in a register
        # on the other: try again with each side's frame its own (see
        # verifier.frame). Only a proof counts; the first attempt says why
        # the pair failed.
        promoted = AnalysisRecorder(*addrs)
        if _verify_product(
            cfg_o, cfg_r, orig_rows, recomp_rows, metadata, promoted, promote=True
        ):
            if recorder is not None:
                recorder.reasons |= promoted.reasons | {"frame_slot_promotion"}
            return True
    if recorder is not None and product_recorder is not recorder:
        assert product_recorder is not None
        if proved:
            recorder.reasons |= product_recorder.reasons
        else:
            pair_heads(cfg_o, cfg_r, recorder, (orig_rows, recomp_rows))
    return proved


def _verify_product(
    cfg_o: Blocks,
    cfg_r: Blocks,
    orig_rows: Sequence[DecodedInstruction],
    recomp_rows: Sequence[DecodedInstruction],
    metadata: FunctionMetadata | None,
    recorder: AnalysisRecorder | None,
    *,
    promote: bool = False,
) -> bool:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    """Verify the product of the two graphs from the entry blocks: every
    reachable node's aligned instructions, with state pairs flowing along
    the paired edges and joined where nodes meet. ``promote``: each side
    keeps its private frame to itself (see verifier.frame)."""
    blocks = len(cfg_o.heads) + len(cfg_r.heads)
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
        indices_o = cfg_o.instructions(node[0])
        indices_r = cfg_r.instructions(node[1])
        start_o, start_r = cfg_o.start(node[0][0]), cfg_r.start(node[1][0])
        lines_o = [_dp_line(orig_rows[i], promote) for i in indices_o]
        # The recompiled block as emitted, and reordered towards the
        # original where that only swaps independent instructions: the
        # cheaper alignment of the two pairs the instructions.
        candidates = [
            (order, alignment)
            for order in (
                indices_r,
                schedule_like(orig_rows, recomp_rows, indices_o, indices_r),
            )
            if (
                alignment := align_block_lines(
                    lines_o, [_dp_line(recomp_rows[i], promote) for i in order]
                )
            )
            is not None
        ]
        best = min(candidates, key=lambda item: item[1][1], default=None)
        if best is not None and best[0] != indices_r:
            any_shifted = True
        if best is None:
            if recorder is not None:
                recorder.mark_inconclusive(
                    "alignment_failure",
                    start_o,
                    start_r,
                    {
                        "stage": "block_alignment",
                        "orig_block_length": len(indices_o),
                        "recomp_block_length": len(indices_r),
                        "orig_block_count": len(cfg_o.heads),
                        "recomp_block_count": len(cfg_r.heads),
                    },
                )
            return None
        indices_r, (aligned, _cost) = best
        # The block terminators (control lines) must pair with each other:
        # a one-sided branch or return breaks the matched structure.
        for local_o, local_r in aligned:
            kind_o = (
                _control(orig_rows[indices_o[local_o]])
                if local_o is not None
                else FlowKind.NORMAL
            )
            kind_r = (
                _control(recomp_rows[indices_r[local_r]])
                if local_r is not None
                else FlowKind.NORMAL
            )
            if kind_o != kind_r or (local_o is None and kind_r is not FlowKind.NORMAL):
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "alignment_failure",
                        indices_o[local_o] if local_o is not None else None,
                        indices_r[local_r] if local_r is not None else None,
                        {
                            "stage": "block_terminator_alignment",
                            "orig_kind": kind_o.value,
                            "recomp_kind": kind_r.value,
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
        exits_o = cfg_o.exits(node[0][-1])
        exits_r = cfg_r.exits(node[1][-1])
        if not (
            _exits_correspond(exits_o, exits_r, swapped=False)
            or _exits_correspond(exits_o, exits_r, swapped=True)
        ):
            if recorder is not None:
                recorder.mark_inconclusive(
                    "non_isomorphic_cfg",
                    orig_index=cfg_o.start(node[0][-1]),
                    recomp_index=cfg_r.start(node[1][-1]),
                    facts={
                        "failure": (
                            "edge_roles"
                            if set(exits_o) != set(exits_r)
                            else "external_edge"
                        ),
                        "orig_block_count": len(cfg_o.heads),
                        "recomp_block_count": len(cfg_r.heads),
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

    def equal(value_o: Value, value_r: Value) -> bool:
        """Values a join may give one phi though they differ as terms."""
        if metadata is not None and not metadata.algebraic_identities:
            return False
        if not bitvector.values_equal(value_o, value_r):
            return False
        if recorder is not None:
            recorder.reasons.add("algebraic_identity")
        return True

    first = node_at(cfg_o.entry, cfg_r.entry)
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
        edges_o = cfg_o.exits(node[0][-1])
        edges_r = cfg_r.exits(node[1][-1])
        swapped = False
        last_o = last_r = None
        for index_o, index_r in alignments[node]:
            if index_o is None or index_r is None:
                if index_o is None:
                    assert index_r is not None
                    side, other_side = recomp_state, orig_state
                    row, position = recomp_rows[index_r], index_r
                else:
                    side, other_side = orig_state, recomp_state
                    row, position = orig_rows[index_o], index_o
                # A one-sided instruction can never store (any observable
                # rejects it), so the memory generation is unaffected.
                if not one_sided_ok(side, other_side, ctx, position, row):
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
            ins_o, ins_r = orig_rows[index_o], recomp_rows[index_r]
            try:
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
                if instruction_semantic_key(ins_o) != instruction_semantic_key(ins_r):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "unsupported_instruction", index_o, index_r
                        )
                    return False
                if admit_unsupported_identical(
                    orig_state, recomp_state, ctx, index_o, ins_o, ins_r
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
                    _call_effect(metadata, 0, ins_o), _call_effect(metadata, 1, ins_r)
                )
                for state, before, entries, effect in (
                    (orig_state, state_before_o, obs_o, effects[0]),
                    (recomp_state, state_before_r, obs_r, effects[1]),
                ):
                    entries.append(_pass_frame(state, before, index_o, effect))
            kind = _control(ins_o)
            switch = any(role.role is EdgeRole.CASE for role in edges_o)
            if kind is FlowKind.CONDITIONAL:
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
            if kind is FlowKind.CONDITIONAL and edges_o.get(TAKEN) is not None:
                for entries in (obs_o, obs_r):
                    for k, obs_entry in enumerate(entries):
                        if obs_entry[0] in CONTROL_TAGS - {"jmpind"}:
                            entries[k] = (*obs_entry[:-1], ("L", "jcc"))
            if kind is FlowKind.JUMP and switch:
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
            ):
                return False
            if any(obs_entry[0] == "jmpind" for obs_entry in obs_o):
                # Recognized switch tables already expanded to caseN edges.
                if not switch:
                    if recorder is not None:
                        recorder.mark_inconclusive("indirect_jump", index_o, index_r)
                    return False

            # A direct edge outside the excerpt exposes the complete machine
            # state to code this proof does not inspect.
            leaving = TAKEN if kind is FlowKind.CONDITIONAL else JUMP
            if (
                kind in (FlowKind.CONDITIONAL, FlowKind.JUMP)
                and leaving in edges_o
                and edges_o[leaving] is None
            ):
                if not converged(orig_state, recomp_state):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "external_control_flow_state",
                            index_o,
                            index_r,
                            {"edge_kind": kind.value},
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
        if last_o is not None and orig_rows[last_o].is_ret:
            if not discharge_run_obligations(
                ctx,
                orig_state,
                recomp_state,
                recorder,
                last_o,
                last_r,
            ):
                return False
        if cfg_o.falls_out(node[0][-1]):
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
            to_r = edges_r[SWAPPED[role] if swapped else role]
            if to_o is None or to_r is None:
                continue
            successor = node_at(to_o, to_r)
            if successor is None:
                return False
            if successor not in entry:
                entry[successor] = clone_cfg_state(outgoing)
                pending.append(successor)
            else:
                joined = join_states(
                    entry[successor], outgoing, nodes[successor], equal
                )
                if joined is None:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "state_join_failure",
                            cfg_o.start(to_o),
                            cfg_r.start(to_r),
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


def _control(row: DecodedInstruction) -> FlowKind:
    """How an instruction ends a block: a branch, a jump, a return, or not
    at all (``NORMAL``, which includes calls)."""
    if row.flow in (FlowKind.CONDITIONAL, FlowKind.JUMP, FlowKind.RETURN):
        return row.flow
    return FlowKind.NORMAL


def _exits_correspond(exits_o: Exits, exits_r: Exits, *, swapped: bool) -> bool:
    """Whether the two sides' exits pair role for role (a branch's taken
    and fall-through edges crosswise when ``swapped``), each leaving the
    function or staying inside on both sides."""
    if set(exits_o) != set(exits_r):
        return False
    if swapped and set(exits_o) != {TAKEN, FALL}:
        return False
    return all(
        (dest is None) == (exits_r[SWAPPED[role] if swapped else role] is None)
        for role, dest in exits_o.items()
    )


def _form_node(cfg_o: Blocks, cfg_r: Blocks, head_o: int, head_r: int) -> _Node:
    """The runs of heads entered at ``head_o`` and ``head_r``: a side whose
    run ends in ``NEXT`` takes that head in too while the exits do not
    correspond."""
    run_o, run_r = [head_o], [head_r]
    while len(run_o) + len(run_r) < _MAX_RUN_BLOCKS:
        exits_o, exits_r = cfg_o.exits(run_o[-1]), cfg_r.exits(run_r[-1])
        if _exits_correspond(exits_o, exits_r, swapped=False) or _exits_correspond(
            exits_o, exits_r, swapped=True
        ):
            break
        next_o, next_r = exits_o.get(NEXT), exits_r.get(NEXT)
        if set(exits_o) == {NEXT} and next_o is not None and next_o not in run_o:
            run_o.append(next_o)
        elif set(exits_r) == {NEXT} and next_r is not None and next_r not in run_r:
            run_r.append(next_r)
        else:
            break
    return tuple(run_o), tuple(run_r)


def _inverted_branch(
    obs_o: list,
    obs_r: list,
    *,
    ins_r: DecodedInstruction,
    state_before_r: SideState,
    exits_o: Exits,
    exits_r: Exits,
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


def _dp_line(row: DecodedInstruction, promote: bool) -> DpLine:
    """The alignment view of an instruction. With the frame promoted, a
    store to a frame slot is no observable: it may pair with any
    instruction, or with none."""
    line = dp_line(row)
    if not promote or line.line_class != "store":
        return line
    match row.operands:
        case (
            Mem(segment="", terms=(ScaledReg("esp" | "ebp", 1),), symbols=()),
            *_,
        ) if (
            not row.prefix and row.mnemonic not in STRING_OPS
        ):
            return dataclasses.replace(line, line_class="none")
    return line


def _call_effect(
    metadata: FunctionMetadata | None, side: int, row: DecodedInstruction
) -> CallStackEffect | None:
    """The stack effect of one side's call, from its binary."""
    if metadata is None or metadata.stack_effects is None or row.address is None:
        return None
    return metadata.stack_effects[side](row.address)


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
