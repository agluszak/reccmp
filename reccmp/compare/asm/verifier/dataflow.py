"""Dataflow state carried along CFG edges by both CFG strategies: per-block
entry states, their joins at merge points, and convergence checks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from reccmp.compare.asm.verifier.addresses import Value
from reccmp.compare.asm.verifier.frame import maybe_frame_pointer
from reccmp.compare.asm.verifier.state import (
    CalleeSaveSubstitution,
    FAMILIES,
    LoadObligation,
    ScratchPush,
    Context,
    SideState,
    clone_state,
    is_scratch,
)

# ---------------------------------------------------------------------------
# CFG-aware verification


@dataclass
# pylint: disable=too-many-instance-attributes
class CfgState:
    """Paired machine state at a basic-block boundary.

    Memory is one relational value rather than one value per side: every
    store and call is already an observable that must match, so equal input
    memories remain equal.  Carrying the value through the CFG is important;
    a process-global generation would let visiting one path change loads on
    another path and is neither a concrete execution nor a sound join.

    Relational proof obligations that genuinely cross block boundaries
    (matched nodes, callee-save stacks, one-sided spills, trap-parity)
    live here so CFG/iso discharge can use the same checklist as linear.
    """

    orig: SideState
    recomp: SideState
    memory: int | Value
    receiver_values: dict[tuple[Value, int | None], tuple[Value, Value]] = field(
        default_factory=dict
    )
    # Whether a pointer into the function's own frame may have escaped on
    # some path reaching this point (see Context.stack_escaped).
    escaped: bool = False
    matched_nodes: set[Value] = field(default_factory=set)
    matched_ids: set[int] = field(default_factory=set)
    keepalive: list = field(default_factory=list)
    save_stack: list[CalleeSaveSubstitution] = field(default_factory=list)
    scratch_pushes: list[ScratchPush] = field(default_factory=list)
    load_obligations: list[LoadObligation] = field(default_factory=list)


def _scratch_keys(state: CfgState) -> tuple:
    return tuple(
        (record.side, record.offset, record.value, record.tag)
        for record in state.scratch_pushes
    )


def _obligation_keys(state: CfgState) -> tuple:
    return tuple(
        sorted(
            (
                (
                    record.side,
                    record.address,
                    record.generation,
                )
                for record in state.load_obligations
            ),
            key=repr,
        )
    )


def _save_keys(state: CfgState) -> tuple:
    return tuple(
        (record.orig_family, record.recomp_family, record.address, record.valid)
        for record in state.save_stack
    )


def clone_cfg_state(state: CfgState) -> CfgState:
    orig = clone_state(state.orig)
    recomp = clone_state(state.recomp)
    return CfgState(
        orig,
        recomp,
        state.memory,
        dict(state.receiver_values),
        state.escaped,
        matched_nodes=set(state.matched_nodes),
        matched_ids={id(node) for node in state.matched_nodes},
        keepalive=list(state.matched_nodes),
        save_stack=[replace(record) for record in state.save_stack],
        scratch_pushes=list(state.scratch_pushes),
        load_obligations=list(state.load_obligations),
    )


def seed_context_from_cfg(ctx: Context, flow: CfgState) -> None:
    ctx.receiver_values = dict(flow.receiver_values)
    ctx.stack_escaped = flow.escaped
    ctx.matched_nodes = set(flow.matched_nodes)
    ctx.matched_ids = {id(node) for node in flow.matched_nodes}
    ctx.keepalive = list(flow.matched_nodes)
    ctx.save_stack = [replace(record) for record in flow.save_stack]
    ctx.scratch_pushes = list(flow.scratch_pushes)
    ctx.load_obligations = list(flow.load_obligations)


def capture_cfg_state(orig: SideState, recomp: SideState, ctx: Context) -> CfgState:
    return CfgState(
        orig,
        recomp,
        ctx.gen,
        dict(ctx.receiver_values),
        ctx.stack_escaped,
        matched_nodes=set(ctx.matched_nodes),
        matched_ids={id(node) for node in ctx.matched_nodes},
        keepalive=list(ctx.matched_nodes),
        save_stack=[replace(record) for record in ctx.save_stack],
        scratch_pushes=list(ctx.scratch_pushes),
        load_obligations=list(ctx.load_obligations),
    )


_JOIN_ATTRS = ("flags", "carry", "fpu_flags")


def _settled(value: Value, state: CfgState) -> bool:
    """No content, or matched on both sides along the edge ``state`` flows."""
    return is_scratch(value) or value in state.matched_nodes


def join_states(
    entry: CfgState,
    incoming: CfgState,
    block: int,
    equal: Callable[[Value, Value], bool] | None = None,
) -> CfgState | None:
    # pylint: disable=too-many-return-statements,too-many-locals
    # pylint: disable=too-many-branches
    """Merge an incoming state pair into a block's entry pair. Returns the
    (possibly new) entry pair, or None if the states cannot be merged
    (differing x87 shapes).

    Every storage node (each side's register families, flag values and x87
    slots) is keyed by its vector of values across the merge: nodes whose
    vectors are identical held provably equal values on every incoming
    edge, so they share one phi symbol — including nodes on *different*
    sides and in *different* registers. This keeps relational knowledge
    alive across joins when a live range is allocated to different
    registers on the two sides. A node whose value agrees on all edges
    keeps that value. Phi symbols are keyed by the class's canonical node
    index; classes can only refine as more edges arrive, so the fixpoint
    terminates. ``equal``, when given, also puts a node in the class of an
    earlier one whose values it holds equal edge for edge (e.g. ``x - 32``
    and ``x + -32``): their phi is then one value too."""
    entry_o, entry_r = entry.orig, entry.recomp
    in_o, in_r = incoming.orig, incoming.recomp
    if len(entry_o.x87.known) != len(entry_r.x87.known):
        return None
    if len(in_o.x87.known) != len(in_r.x87.known):
        return None
    if len(entry_o.x87.known) != len(in_o.x87.known):
        return None
    if entry_o.x87.deep_pops != entry_r.x87.deep_pops:
        return None
    if in_o.x87.deep_pops != in_r.x87.deep_pops:
        return None
    if entry_o.x87.deep_pops != in_o.x87.deep_pops:
        return None
    if entry_o.x87.epoch != entry_r.x87.epoch:
        return None
    if in_o.x87.epoch != in_r.x87.epoch:
        return None
    out_o = clone_state(entry_o)
    out_r = clone_state(entry_r)
    if entry_o.x87.epoch != in_o.x87.epoch:
        # Paths through different call sites reach this block with different
        # x87 epochs. Control flow is paired, so both sides always arrive
        # via corresponding paths: a joined epoch keyed by the block keeps
        # deep-stack reads cross-equal (same reasoning as the memory phi).
        joined_epoch = ("x87_epoch_phi", block)
        out_o.x87.epoch = joined_epoch  # type: ignore[assignment]
        out_r.x87.epoch = joined_epoch  # type: ignore[assignment]

    # (entry value, incoming value, setter on the joined state)
    nodes: list[tuple[Value, Value, Callable[[Value], None]]] = []

    def reg_setter(state: SideState, family: str) -> Callable[[Value], None]:
        return lambda value: state.regs.__setitem__(family, value)

    def attr_setter(state: SideState, attr: str) -> Callable[[Value], None]:
        return lambda value: setattr(state, attr, value)

    def slot_setter(state: SideState, index: int) -> Callable[[Value], None]:
        return lambda value: state.x87.known.__setitem__(index, value)

    def frame_setter(
        frame: dict[int, tuple[int, Value | None]], offset: int, width: int
    ) -> Callable[[Value], None]:
        return lambda value: frame.__setitem__(offset, (width, value))

    for entry_state, in_state, out_state in (
        (entry_o, in_o, out_o),
        (entry_r, in_r, out_r),
    ):
        for family in FAMILIES:
            nodes.append(
                (
                    entry_state.regs[family],
                    in_state.regs[family],
                    reg_setter(out_state, family),
                )
            )
        for attr in _JOIN_ATTRS:
            nodes.append(
                (
                    getattr(entry_state, attr),
                    getattr(in_state, attr),
                    attr_setter(out_state, attr),
                )
            )
        for index, entry_slot in enumerate(entry_state.x87.known):
            nodes.append(
                (
                    entry_slot,
                    in_state.x87.known[index],
                    slot_setter(out_state, index),
                )
            )
        if out_state.frame is not None:
            # Promoted slots join like registers; one that does not hold a
            # value of the same width on both edges can no longer be read.
            assert entry_state.frame is not None and in_state.frame is not None
            for offset in set(entry_state.frame) | set(in_state.frame):
                old = entry_state.frame.get(offset)
                new = in_state.frame.get(offset)
                if old is None or new is None or old[0] != new[0]:
                    width = max(entry[0] for entry in (old, new) if entry is not None)
                    out_state.frame[offset] = (width, None)
                elif old[1] is None or new[1] is None:
                    out_state.frame[offset] = (old[0], None)
                else:
                    nodes.append(
                        (old[1], new[1], frame_setter(out_state.frame, offset, old[0]))
                    )

    classes: dict[tuple[Value, Value], int] = {}
    for n, (entry_value, in_value, setter) in enumerate(nodes):
        if entry_value == in_value:
            setter(entry_value)
            continue
        if out_o.frame is not None and (
            maybe_frame_pointer(entry_value) or maybe_frame_pointer(in_value)
        ):
            # A phi would hide which slot a pointer into the frame reaches.
            return None
        key = (entry_value, in_value)
        if key not in classes and equal is not None:
            classes[key] = next(
                (
                    class_id
                    for (entry_other, in_other), class_id in classes.items()
                    if equal(entry_other, entry_value) and equal(in_other, in_value)
                ),
                n,
            )
        class_id = classes.setdefault(key, n)
        # A join of values that each have no content (see is_scratch) or
        # were already observed on their edge holds nothing unobserved.
        tag = (
            "scratch_phi"
            if _settled(entry_value, entry) and _settled(in_value, incoming)
            else "phi"
        )
        setter((tag, block, class_id))

    if entry.memory == incoming.memory:
        memory = entry.memory
    else:
        memory = ("cfg_mem_phi", block)
    receiver_values = {
        key: value
        for key, value in entry.receiver_values.items()
        if incoming.receiver_values.get(key) == value
    }
    if _save_keys(entry) != _save_keys(incoming):
        return None
    if _scratch_keys(entry) != _scratch_keys(incoming):
        return None
    entry_obl = frozenset(_obligation_keys(entry))
    in_obl = frozenset(_obligation_keys(incoming))
    # Opposite-arm trap histories are incomparable and must not join.
    # A loop header's empty first visit is a subset of the body's
    # obligations; keep the superset so folded loads can stabilize.
    if entry_obl != in_obl and not entry_obl < in_obl and not in_obl < entry_obl:
        return None
    chosen_obl = incoming if in_obl > entry_obl else entry
    # Never union trap histories. When obligations refine along a loop,
    # take the more specific predecessor's logs rather than mixing paths.
    out_o.load_log = set(chosen_obl.orig.load_log)
    out_r.load_log = set(chosen_obl.recomp.load_log)
    return CfgState(
        out_o,
        out_r,
        memory,
        receiver_values,
        entry.escaped or incoming.escaped,
        matched_nodes=set(entry.matched_nodes) | set(incoming.matched_nodes),
        matched_ids={
            id(node) for node in (entry.matched_nodes | incoming.matched_nodes)
        },
        keepalive=list(entry.matched_nodes | incoming.matched_nodes),
        save_stack=[replace(record) for record in entry.save_stack],
        scratch_pushes=list(entry.scratch_pushes),
        load_obligations=list(chosen_obl.load_obligations),
    )


def states_equal(a: CfgState, b: CfgState) -> bool:
    return (
        a.memory == b.memory
        and a.receiver_values == b.receiver_values
        and a.escaped == b.escaped
        and a.matched_nodes == b.matched_nodes
        and _save_keys(a) == _save_keys(b)
        and _scratch_keys(a) == _scratch_keys(b)
        and _obligation_keys(a) == _obligation_keys(b)
        and all(
            x.regs == y.regs
            and x.flags == y.flags
            and x.carry == y.carry
            and x.fpu_flags == y.fpu_flags
            and x.x87.state_key() == y.x87.state_key()
            and x.load_log == y.load_log
            and x.frame == y.frame
            for x, y in ((a.orig, b.orig), (a.recomp, b.recomp))
        )
    )


def converged(orig: SideState, recomp: SideState) -> bool:
    return (
        orig.regs == recomp.regs
        and orig.flags == recomp.flags
        and orig.carry == recomp.carry
        and orig.fpu_flags == recomp.fpu_flags
        and orig.x87.state_key() == recomp.x87.state_key()
    )
