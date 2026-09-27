"""Isomorphic-CFG strategy: independently built graphs paired by structure,
with block-local alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import Init, Value, unwind_spadd
from reccmp.compare.asm.verifier.frame import maybe_frame_pointer
from reccmp.compare.asm.verifier.block_align import AlignedPair, align_block_lines
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
    Branch,
    Call,
    FrameArguments,
    IndirectJump,
    JCC_MNEMONICS,
    Jump,
    Loop,
    Observation,
    Context,
    FunctionMetadata,
    SideState,
    clone_state,
    commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    EffectiveReason,
    InconclusiveReason,
    StopDetail,
)
from reccmp.types import ImageId

if TYPE_CHECKING:
    from reccmp.compare.callee_cleanup import CallStackEffect


@dataclass
class ProductResult:
    """What the product of the two graphs found."""

    proved: bool
    # Why and where it failed, or the categories of its proof.
    recorder: AnalysisRecorder
    # When the blocks do not pair one to one and the product failed: where
    # it stopped under its guessed pairing, a lead rather than a verdict.
    unanchored: AnalysisRecorder | None = None


def verify_isomorphic_cfg_effective_match(
    orig: FunctionImage,
    recomp: FunctionImage,
    metadata: FunctionMetadata | None = None,
) -> ProductResult:
    """CFG verification that tolerates different instruction counts:
    per-side block graphs matched structurally, block contents aligned
    locally, one-sided unobservable instructions allowed. This proves
    register-allocation wobble in its full generality — renames composed
    with folded loads, elided copies and shifted branch displacements.

    Recognized switch jump tables become case edges, so pairing compares
    entry count and case→block topology."""
    recorder = AnalysisRecorder(orig, recomp)
    if unsupported_control_flow(
        orig, recorder, ImageId.ORIG
    ) or unsupported_control_flow(recomp, recorder, ImageId.RECOMP):
        return ProductResult(False, recorder)
    orig_rows, recomp_rows = orig.instructions, recomp.instructions
    cfg_o, cfg_r = Blocks(orig), Blocks(recomp)
    # Without a one-to-one pairing of the blocks, which blocks the product
    # pairs where the graphs differ is a guess: enough to prove the two
    # equal, but a difference found under it may be the guess's own (a
    # redundant test on one side paired with the other's next branch). Such
    # a failure reports why the blocks do not pair one to one, and the
    # difference is only a lead.
    anchored = pair_heads(cfg_o, cfg_r)
    product_recorder = recorder if anchored else AnalysisRecorder(orig, recomp)
    proved = _verify_product(
        cfg_o, cfg_r, orig_rows, recomp_rows, metadata, product_recorder
    )
    if not proved:
        # A local may live in a stack slot on one side and in a register
        # on the other: try again with each side's frame its own (see
        # verifier.frame). Only a proof counts; the first attempt says why
        # the pair failed.
        promoted = AnalysisRecorder(orig, recomp)
        if _verify_product(
            cfg_o, cfg_r, orig_rows, recomp_rows, metadata, promoted, promote=True
        ):
            recorder.reasons |= promoted.reasons | {
                EffectiveReason.FRAME_SLOT_PROMOTION
            }
            return ProductResult(True, recorder)
    if anchored:
        return ProductResult(proved, recorder)
    if proved:
        recorder.reasons |= product_recorder.reasons
        return ProductResult(True, recorder)
    pair_heads(cfg_o, cfg_r, recorder)
    stopped = (
        product_recorder.best_difference is not None
        or product_recorder.inconclusive_reason is not None
    )
    return ProductResult(False, recorder, product_recorder if stopped else None)


def _verify_product(
    cfg_o: Blocks,
    cfg_r: Blocks,
    orig_rows: Sequence[DecodedInstruction],
    recomp_rows: Sequence[DecodedInstruction],
    metadata: FunctionMetadata | None,
    recorder: AnalysisRecorder,
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
    alignments: dict[_Node, tuple[AlignedPair, ...]] = {}
    orientation: dict[_Node, bool] = {}
    any_shifted = False

    def align(node: _Node) -> tuple[AlignedPair, ...] | None:
        nonlocal any_shifted
        indices_o = cfg_o.instructions(node[0])
        indices_r = cfg_r.instructions(node[1])
        start_o, start_r = cfg_o.start(node[0][0]), cfg_r.start(node[1][0])
        lines_o = [orig_rows[i] for i in indices_o]
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
                    lines_o,
                    [recomp_rows[i] for i in order],
                    promote=promote,
                )
            )
            is not None
        ]
        best = min(candidates, key=lambda item: item[1].cost, default=None)
        if best is not None and best[0] != indices_r:
            any_shifted = True
        if best is None:
            recorder.mark_inconclusive(
                InconclusiveReason.ALIGNMENT_FAILURE,
                start_o,
                start_r,
                StopDetail.BLOCK_ALIGNMENT,
            )
            return None
        indices_r, alignment = best
        # The block terminators (control lines) must pair with each other:
        # a one-sided branch or return breaks the matched structure.
        for pair in alignment.pairs:
            kind_o = (
                _control(orig_rows[indices_o[pair.orig]])
                if pair.orig is not None
                else FlowKind.NORMAL
            )
            kind_r = (
                _control(recomp_rows[indices_r[pair.recomp]])
                if pair.recomp is not None
                else FlowKind.NORMAL
            )
            if kind_o != kind_r or (
                pair.orig is None and kind_r is not FlowKind.NORMAL
            ):
                recorder.mark_inconclusive(
                    InconclusiveReason.ALIGNMENT_FAILURE,
                    indices_o[pair.orig] if pair.orig is not None else None,
                    indices_r[pair.recomp] if pair.recomp is not None else None,
                    StopDetail.BLOCK_TERMINATOR_ALIGNMENT,
                )
                return None
        if any(pair.orig is None or pair.recomp is None for pair in alignment.pairs):
            any_shifted = True
        return tuple(
            AlignedPair(
                indices_o[pair.orig] if pair.orig is not None else None,
                indices_r[pair.recomp] if pair.recomp is not None else None,
            )
            for pair in alignment.pairs
        )

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
            recorder.mark_inconclusive(
                InconclusiveReason.NON_ISOMORPHIC_CFG,
                cfg_o.start(node[0][-1]),
                cfg_r.start(node[1][-1]),
                (
                    StopDetail.EDGE_ROLES
                    if set(exits_o) != set(exits_r)
                    else StopDetail.EXTERNAL_EDGE
                ),
            )
            return None
        if len(nodes) >= node_limit:
            recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
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
        recorder.reasons.add(EffectiveReason.ALGEBRAIC_IDENTITY)
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
        for pair in alignments[node]:
            index_o, index_r = pair.orig, pair.recomp
            if index_o is None or index_r is None:
                if index_o is None:
                    assert index_r is not None
                    side, which = recomp_state, ImageId.RECOMP
                    row, position = recomp_rows[index_r], index_r
                else:
                    side, which = orig_state, ImageId.ORIG
                    row, position = orig_rows[index_o], index_o
                # A one-sided instruction can never store (any observable
                # rejects it), so the memory generation is unaffected.
                if not one_sided_ok(which, side, ctx, position, row):
                    recorder.mark_inconclusive(
                        InconclusiveReason.ALIGNMENT_FAILURE,
                        index_o,
                        index_r,
                        StopDetail.ONE_SIDED_INSTRUCTION,
                    )
                    return False
                continue
            last_o, last_r = index_o, index_r
            ins_o, ins_r = orig_rows[index_o], recomp_rows[index_r]
            try:
                record_operand_candidate(
                    ctx, index_o, index_r, ins_o, ins_r, (orig_state, recomp_state)
                )
                obs_o: list[Observation] = []
                obs_r: list[Observation] = []
                state_before_o = clone_state(orig_state)
                state_before_r = clone_state(recomp_state)
                execute(orig_state, ctx, index_o, ins_o, obs_o)
                execute(recomp_state, ctx, index_o, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                if instruction_semantic_key(ins_o) != instruction_semantic_key(ins_r):
                    recorder.mark_inconclusive(
                        InconclusiveReason.UNSUPPORTED_INSTRUCTION, index_o, index_r
                    )
                    return False
                if admit_unsupported_identical(
                    orig_state, recomp_state, ctx, index_o, ins_o, ins_r
                ):
                    continue
                recorder.mark_inconclusive(
                    InconclusiveReason.UNSUPPORTED_INSTRUCTION, index_o, index_r
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

            if promote and any(isinstance(entry, Call) for entry in obs_o):
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
                        if isinstance(obs_entry, (Branch, Jump, Loop)):
                            entries[k] = replace(obs_entry, destination=("L", "jcc"))
            if kind is FlowKind.JUMP and switch:
                idx_o = switch_index_observation(state_before_o, ins_o)
                idx_r = switch_index_observation(state_before_r, ins_r)
                for entries, idx in ((obs_o, idx_o), (obs_r, idx_r)):
                    for k, obs_entry in enumerate(entries):
                        if isinstance(obs_entry, IndirectJump):
                            entries[k] = IndirectJump(("L", "switch"), idx)

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
            if any(isinstance(obs_entry, IndirectJump) for obs_entry in obs_o):
                # Recognized switch tables already expanded to caseN edges.
                if not switch:
                    recorder.mark_inconclusive(
                        InconclusiveReason.INDIRECT_JUMP, index_o, index_r
                    )
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
                    recorder.mark_inconclusive(
                        InconclusiveReason.EXTERNAL_CONTROL_FLOW_STATE, index_o, index_r
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
            recorder.mark_inconclusive(
                InconclusiveReason.NON_ISOMORPHIC_CFG,
                last_o,
                last_r,
                StopDetail.BRANCH_ORIENTATION,
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
            recorder.mark_inconclusive(
                InconclusiveReason.FUNCTION_FALLTHROUGH,
                last_o,
                last_r,
            )
            return False

        outgoing = capture_cfg_state(orig_state, recomp_state, ctx)
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
                    recorder.mark_inconclusive(
                        InconclusiveReason.STATE_JOIN_FAILURE,
                        cfg_o.start(to_o),
                        cfg_r.start(to_r),
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
                recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
                return False
            if not run_node(node):
                return False
    except (Reject, RecursionError):
        recorder.mark_inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        return False
    if any_shifted:
        recorder.reasons.add(EffectiveReason.INSTRUCTION_REORDER)
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
    obs_o: list[Observation],
    obs_r: list[Observation],
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
    branch_o = next(
        (k for k, entry in enumerate(obs_o) if isinstance(entry, Branch)), None
    )
    branch_r = next(
        (k for k, entry in enumerate(obs_r) if isinstance(entry, Branch)), None
    )
    if branch_o is None or branch_r is None:
        return False
    original = obs_o[branch_o]
    recompiled = obs_r[branch_r]
    assert isinstance(original, Branch) and isinstance(recompiled, Branch)
    if original.predicate == recompiled.predicate:
        return False
    complement = _COMPLEMENT_JCC.get(ins_r.mnemonic)
    if complement is None or not _exits_correspond(exits_o, exits_r, swapped=True):
        return False
    inverted = canon_condition(JCC_MNEMONICS[complement], state_before_r)
    if inverted != original.predicate:
        return False
    obs_r[branch_r] = replace(recompiled, predicate=inverted)
    return True


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
) -> FrameArguments:
    """A call with the frame promoted: the observation of the promoted
    slots the callee may read (its arguments, by offset from the stack
    pointer at the call; every slot above it when their extent is
    unknown). The callee may write them, so they hold its results after
    the call, and its ``ret N`` sets the stack pointer, when known."""
    assert state.frame is not None
    esp = before.read_reg("esp")
    root, top = unwind_spadd(esp)
    if root != Init("sp"):
        if state.frame:
            raise Reject  # the arguments cannot be placed
        return FrameArguments(None)
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
    return FrameArguments(tuple(passed))
