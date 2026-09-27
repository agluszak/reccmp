"""One side's control-flow graph as the product walks it.

The graph (``FunctionImage.control_graph()``) stays the function's factual
control flow. Three equivalence rules say which of its blocks the product
treats as one:

* a block that is only a ``jmp`` to another block (a trampoline) is
  transparent: an edge to it leads to where it jumps;
* a block that falls into the contiguous block after it, which nothing else
  reaches, runs on into that block;
* empty ``ret`` blocks with the same instruction are one exit.

A *head* is a block the product can enter: not a trampoline, not run into
from the block before it, and the first of its identical returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from reccmp.compare.asm.graph import (
    EdgeKind,
    EdgeRole,
    FunctionGraph,
    GraphBlock,
)
from reccmp.compare.asm.ir import FlowKind, FunctionImage, instruction_semantic_key
from reccmp.compare.asm.operand import Mem
from reccmp.compare.asm.verifier.evidence import transfer
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    DifferenceKind,
    InconclusiveReason,
    StopDetail,
)
from reccmp.types import ImageId


@dataclass(frozen=True, slots=True)
class Exit:
    """How control leaves a head: along an edge role (and switch case), or
    ``NEXT``, the one unconditional edge to a head of the function."""

    role: EdgeRole | None
    case: int | None = None


NEXT = Exit(None)
TAKEN = Exit(EdgeRole.TAKEN)
FALL = Exit(EdgeRole.FALL)
JUMP = Exit(EdgeRole.JUMP)
SWAPPED = {TAKEN: FALL, FALL: TAKEN}

# A head's exits: where each leads, None for outside the function.
Exits = dict[Exit, int | None]


class Blocks:
    """The heads of one side and how control moves between them."""

    def __init__(self, image: FunctionImage) -> None:
        self.rows = image.instructions
        self.graph: FunctionGraph = image.control_graph()
        self._dispatches = {index for index, _ in self.graph.table_dests}

    @property
    def blocks(self) -> tuple[GraphBlock, ...]:
        return self.graph.blocks

    def _target(self, edge_target: int | None) -> int | None:
        return None if edge_target is None else self.graph.block_at(edge_target)

    def _jump_only(self, block: int) -> int | None:
        """Where a block that is only a local ``jmp`` jumps."""
        graph_block = self.blocks[block]
        if graph_block.end - graph_block.start != 1:
            return None
        if self.rows[graph_block.start].flow is not FlowKind.JUMP:
            return None
        if graph_block.start in self._dispatches:
            return None
        match graph_block.successors:
            case (edge,) if edge.role is EdgeRole.JUMP and edge.kind is EdgeKind.LOCAL:
                return self._target(edge.target)
        return None

    def _trampoline_target(self, block: int) -> int | None:
        """Where a trampoline leads: a jump-only block not jumping to itself."""
        target = self._jump_only(block)
        return None if target == block else target

    @cached_property
    def _threaded(self) -> dict[int, int]:
        """Trampoline → the block its chain of trampolines reaches. A chain
        that loops threads nothing."""
        hops = {
            block: target
            for block in range(len(self.blocks))
            if (target := self._trampoline_target(block)) is not None
        }
        threaded = {}
        for block in hops:
            seen = set()
            target = block
            while target in hops and target not in seen:
                seen.add(target)
                target = hops[target]
            if target not in hops:
                threaded[block] = target
        return threaded

    def _resolve_trampolines(self, block: int) -> int:
        return self._threaded.get(block, block)

    def _raw_exits(self, block: int) -> dict[Exit, int | None]:
        """A block's successors by role, through trampolines."""
        return {
            Exit(edge.role, edge.case): (
                self._resolve_trampolines(target)
                if edge.kind is EdgeKind.LOCAL
                and (target := self._target(edge.target)) is not None
                else None
            )
            for edge in self.blocks[block].successors
        }

    @cached_property
    def _run_on(self) -> dict[int, int]:
        """Block → the contiguous block it runs into (its only successor,
        a fall-through nothing else reaches)."""
        kept = [
            block for block in range(len(self.blocks)) if block not in self._threaded
        ]
        predecessors = dict.fromkeys(kept, 0)
        for block in kept:
            for target in self._raw_exits(block).values():
                if target is not None:
                    predecessors[target] += 1
        run_on = {}
        for block in kept:
            exits = self._raw_exits(block)
            target = exits.get(FALL)
            if (
                len(exits) == 1
                and target is not None
                and predecessors[target] == 1
                and self._may_run_into(block, target)
            ):
                run_on[block] = target
        return run_on

    def _may_run_into(self, block: int, target: int) -> bool:
        """Whether ``target`` directly follows ``block`` and is plain code:
        not a jump-only block, not ending in a switch dispatch."""
        return (
            target != block
            and self.blocks[block].end == self.blocks[target].start
            and self._jump_only(target) is None
            and self.blocks[target].last not in self._dispatches
        )

    def _chain(self, head: int) -> list[int]:
        chain = [head]
        while chain[-1] in self._run_on and self._run_on[chain[-1]] not in chain:
            chain.append(self._run_on[chain[-1]])
        return chain

    def chain(self, head: int) -> tuple[GraphBlock, ...]:
        """The graph blocks a head runs through, in order."""
        return tuple(self.blocks[block] for block in self._chain(head))

    @cached_property
    def _same_return(self) -> dict[int, int]:
        """Empty ``ret`` block → the first head with the same instruction."""
        absorbed = set(self._run_on.values()) | set(self._threaded)
        first: dict[object, int] = {}
        same = {}
        for block, graph_block in enumerate(self.blocks):
            if block in absorbed or not self._empty_return(graph_block):
                continue
            key = instruction_semantic_key(self.rows[graph_block.start])
            same[block] = first.setdefault(key, block)
        return same

    def _empty_return(self, block: GraphBlock) -> bool:
        return (
            block.end - block.start == 1
            and self.rows[block.start].is_ret
            and not block.successors
        )

    def head(self, block: int) -> int:
        """The head control reaches when it goes to ``block``."""
        block = self._resolve_trampolines(block)
        return self._same_return.get(block, block)

    @cached_property
    def entry(self) -> int:
        return self.head(0)

    @cached_property
    def heads(self) -> tuple[int, ...]:
        """Every head, in block order."""
        run_into = set(self._run_on.values())
        return tuple(
            block
            for block in range(len(self.blocks))
            if self.head(block) == block and block not in run_into
        )

    def edges(self, head: int) -> Exits:
        """A head's successors by edge role, through its chain."""
        return {
            role: None if target is None else self.head(target)
            for role, target in self._raw_exits(self._chain(head)[-1]).items()
        }

    def exits(self, head: int) -> Exits:
        """A head's edges, where one unconditional edge to a head of the
        function is ``NEXT``, whether it falls through or jumps."""
        exits = self.edges(head)
        if len(exits) == 1:
            ((role, target),) = exits.items()
            if role in (FALL, JUMP) and target is not None:
                return {NEXT: target}
        return exits

    def falls_out(self, head: int) -> bool:
        """The head's last block runs past the function's bytes."""
        return any(edge.falls_out for edge in self.chain(head)[-1].successors)

    def start(self, head: int) -> int:
        return self.chain(head)[0].start

    def last(self, head: int) -> int:
        return self.chain(head)[-1].last

    def instructions(self, heads: tuple[int, ...]) -> list[int]:
        """The instructions a run of heads executes, without the jumps
        inside the function (their edge is the run's ``NEXT``)."""
        indices: list[int] = []
        for head in heads:
            last = self.last(head)
            internal_jump = self.rows[last].flow is FlowKind.JUMP and set(
                self.exits(head)
            ) == {NEXT}
            indices += [
                index
                for block in self.chain(head)
                for index in range(block.start, block.end)
                if not (internal_jump and index == last)
            ]
        return indices


def unsupported_control_flow(
    image: FunctionImage, recorder: AnalysisRecorder | None, which: ImageId
) -> bool:
    """Whether the product cannot walk this side: it has no instructions, or
    a jump whose destinations are unknown. Records why."""
    rows = image.instructions
    if not rows:
        if recorder is not None:
            recorder.mark_inconclusive(
                InconclusiveReason.EMPTY_CONTROL_FLOW, image=which
            )
        return True
    for index, successors in enumerate(image.control_graph().edges):
        if rows[index].flow is not FlowKind.JUMP or not any(
            edge.kind is EdgeKind.UNKNOWN for edge in successors
        ):
            continue
        switch_candidate = any(
            isinstance(op, Mem) and any(term.scale == 4 for term in op.terms)
            for op in rows[index].operands
        )
        if recorder is not None:
            recorder.mark_inconclusive(
                (
                    InconclusiveReason.JUMP_TABLE_DATA
                    if switch_candidate
                    else InconclusiveReason.INDIRECT_JUMP
                ),
                index if which is ImageId.ORIG else None,
                index if which is ImageId.RECOMP else None,
                (
                    StopDetail.UNRESOLVED_SWITCH_TABLE
                    if switch_candidate
                    else StopDetail.INDIRECT_TARGET
                ),
            )
        return True
    return False


@dataclass(frozen=True, slots=True)
class _Reached:
    """A pair of heads to match, and the paired heads whose same-role exit
    reached them (None for the entries)."""

    orig: int
    recomp: int
    source_orig: int | None = None
    source_recomp: int | None = None
    exit: Exit | None = None


def pair_heads(
    orig: Blocks, recomp: Blocks, recorder: AnalysisRecorder | None = None
) -> bool:
    """Whether the two sides' reachable heads match one to one, starting
    from the entries and following same-role edges. When they do not,
    ``recorder`` gets where: a branch whose same-role edges reach heads
    paired elsewhere is a branch-target difference."""
    paired_o: dict[int, int] = {}
    paired_r: dict[int, int] = {}
    pending = [_Reached(orig.entry, recomp.entry)]
    while pending:
        reached = pending.pop()
        head_o, head_r = reached.orig, reached.recomp
        if head_o in paired_o or head_r in paired_r:
            if paired_o.get(head_o) == head_r and paired_r.get(head_r) == head_o:
                continue
            if recorder is None:
                return False
            if (
                reached.source_orig is not None
                and reached.source_recomp is not None
                and reached.exit != FALL
            ):
                branch_o = orig.last(reached.source_orig)
                branch_r = recomp.last(reached.source_recomp)
                recorder.record_difference(
                    DifferenceKind.BRANCH_TARGET,
                    branch_o,
                    branch_r,
                    transfer(orig.rows[branch_o], orig.start(head_o)),
                    transfer(recomp.rows[branch_r], recomp.start(head_r)),
                )
            else:
                recorder.mark_inconclusive(
                    InconclusiveReason.NON_ISOMORPHIC_CFG,
                    orig.start(head_o),
                    recomp.start(head_r),
                    StopDetail.BLOCK_MAPPING_CONFLICT,
                )
            return False
        paired_o[head_o] = head_r
        paired_r[head_r] = head_o
        edges_o, edges_r = orig.edges(head_o), recomp.edges(head_r)
        if set(edges_o) != set(edges_r):
            if recorder is not None:
                recorder.mark_inconclusive(
                    InconclusiveReason.NON_ISOMORPHIC_CFG,
                    orig.start(head_o),
                    recomp.start(head_r),
                    StopDetail.EDGE_ROLES,
                )
            return False
        for role, to_o in edges_o.items():
            to_r = edges_r[role]
            if to_o is None and to_r is None:
                continue
            if to_o is None or to_r is None:
                if recorder is not None:
                    recorder.mark_inconclusive(
                        InconclusiveReason.NON_ISOMORPHIC_CFG,
                        orig.start(head_o),
                        recomp.start(head_r),
                        StopDetail.EXTERNAL_EDGE,
                    )
                return False
            pending.append(_Reached(to_o, to_r, head_o, head_r, role))
    return True
