"""Positional CFG strategy: equal-shape graphs with dataflow joins."""

from __future__ import annotations

from typing import Callable
from dataclasses import (
    dataclass,
    field,
)
from reccmp.compare.asm.instgen import (
    InstructionMeta,
)
from reccmp.compare.asm.ir import (
    AsmStream,
    ResolvedAsm,
    instruction_at,
    is_data_row,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import (
    Reject,
)
from reccmp.compare.asm.verifier.addresses import (
    Value,
)
from reccmp.compare.asm.verifier.evidence import (
    _record_operand_candidate,
    _target_facts,
)
from reccmp.compare.asm.verifier.obligations import (
    _accept_agreeing_pair,
    _callee_save_swap,
    _discharge_run_obligations,
    admit_unsupported_identical,
)
from reccmp.compare.asm.verifier.semantics import (
    execute,
)
from reccmp.compare.asm.verifier.state import (
    CONTROL_TAGS,
    Context,
    FAMILIES,
    FunctionMetadata,
    JCC_MNEMONICS,
    SideState,
    _clone_state,
    _commit_memory,
    guard_state_size,
)
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    FactValue,
)

# ---------------------------------------------------------------------------
# CFG-aware verification


@dataclass
# pylint: disable=too-many-instance-attributes
class _CfgState:
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
    save_stack: list[list] = field(default_factory=list)
    scratch_pushes: list[list] = field(default_factory=list)
    load_obligations: list[tuple] = field(default_factory=list)


def _remap_scratch(
    records: list[list],
    old_orig: SideState,
    old_recomp: SideState,
    new_orig: SideState,
    new_recomp: SideState,
) -> list[list]:
    remapped = []
    for record in records:
        cloned = list(record)
        if cloned and cloned[0] is old_orig:
            cloned[0] = new_orig
        elif cloned and cloned[0] is old_recomp:
            cloned[0] = new_recomp
        remapped.append(cloned)
    return remapped


def _scratch_keys(state: _CfgState) -> tuple:
    keys = []
    for record in state.scratch_pushes:
        side = "orig" if record[0] is state.orig else "recomp"
        keys.append((side, record[1], record[2], record[3]))
    return tuple(keys)


def _remap_obligations(
    records: list[tuple],
    old_orig: SideState,
    old_recomp: SideState,
    new_orig: SideState,
    new_recomp: SideState,
) -> list[tuple]:
    remapped = []
    for other, *rest in records:
        if other is old_orig:
            other = new_orig
        elif other is old_recomp:
            other = new_recomp
        remapped.append((other, *rest))
    return remapped


def _obligation_keys(state: _CfgState) -> tuple:
    keys = []
    for record in state.load_obligations:
        other = record[0]
        side = "orig" if other is state.orig else "recomp"
        keys.append((side, *record[1:]))
    return tuple(sorted(keys, key=repr))


def _save_keys(state: _CfgState) -> tuple:
    return tuple(tuple(record) for record in state.save_stack)


def _clone_cfg_state(state: _CfgState) -> _CfgState:
    orig = _clone_state(state.orig)
    recomp = _clone_state(state.recomp)
    return _CfgState(
        orig,
        recomp,
        state.memory,
        dict(state.receiver_values),
        state.escaped,
        matched_nodes=set(state.matched_nodes),
        matched_ids={id(node) for node in state.matched_nodes},
        keepalive=list(state.matched_nodes),
        save_stack=[list(record) for record in state.save_stack],
        scratch_pushes=_remap_scratch(
            state.scratch_pushes, state.orig, state.recomp, orig, recomp
        ),
        load_obligations=_remap_obligations(
            state.load_obligations, state.orig, state.recomp, orig, recomp
        ),
    )


def _seed_context_from_cfg(ctx: Context, flow: _CfgState) -> None:
    ctx.receiver_values = dict(flow.receiver_values)
    ctx.stack_escaped = flow.escaped
    ctx.matched_nodes = set(flow.matched_nodes)
    ctx.matched_ids = {id(node) for node in flow.matched_nodes}
    ctx.keepalive = list(flow.matched_nodes)
    ctx.save_stack = [list(record) for record in flow.save_stack]
    ctx.scratch_pushes = [list(record) for record in flow.scratch_pushes]
    ctx.load_obligations = list(flow.load_obligations)


def _capture_cfg_state(orig: SideState, recomp: SideState, ctx: Context) -> _CfgState:
    return _CfgState(
        orig,
        recomp,
        ctx.gen,
        dict(ctx.receiver_values),
        ctx.stack_escaped,
        matched_nodes=set(ctx.matched_nodes),
        matched_ids={id(node) for node in ctx.matched_nodes},
        keepalive=list(ctx.matched_nodes),
        save_stack=[list(record) for record in ctx.save_stack],
        scratch_pushes=[list(record) for record in ctx.scratch_pushes],
        load_obligations=list(ctx.load_obligations),
    )


_JOIN_ATTRS = ("flags", "carry", "fpu_flags")


def _join_states(
    entry: _CfgState,
    incoming: _CfgState,
    block: int,
) -> _CfgState | None:
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
    terminates."""
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
    out_o = _clone_state(entry_o)
    out_r = _clone_state(entry_r)
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

    classes: dict[tuple[Value, Value], int] = {}
    for n, (entry_value, in_value, setter) in enumerate(nodes):
        if entry_value == in_value:
            setter(entry_value)
            continue
        class_id = classes.setdefault((entry_value, in_value), n)
        setter(("phi", block, class_id))

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
    return _CfgState(
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
        save_stack=[list(record) for record in entry.save_stack],
        scratch_pushes=_remap_scratch(
            entry.scratch_pushes, entry.orig, entry.recomp, out_o, out_r
        ),
        load_obligations=_remap_obligations(
            chosen_obl.load_obligations,
            chosen_obl.orig,
            chosen_obl.recomp,
            out_o,
            out_r,
        ),
    )


def _join_failure_facts(entry: _CfgState, incoming: _CfgState) -> dict[str, FactValue]:
    """Compact state-shape evidence for a failed CFG join."""
    return {
        "entry_orig_x87_depth": len(entry.orig.x87.known),
        "entry_recomp_x87_depth": len(entry.recomp.x87.known),
        "incoming_orig_x87_depth": len(incoming.orig.x87.known),
        "incoming_recomp_x87_depth": len(incoming.recomp.x87.known),
        "entry_x87_deep_pops_equal": (
            entry.orig.x87.deep_pops == entry.recomp.x87.deep_pops
        ),
        "incoming_x87_deep_pops_equal": (
            incoming.orig.x87.deep_pops == incoming.recomp.x87.deep_pops
        ),
        "entry_x87_epochs_equal": entry.orig.x87.epoch == entry.recomp.x87.epoch,
        "incoming_x87_epochs_equal": (
            incoming.orig.x87.epoch == incoming.recomp.x87.epoch
        ),
    }


def _states_equal(a: _CfgState, b: _CfgState) -> bool:
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
            for x, y in ((a.orig, b.orig), (a.recomp, b.recomp))
        )
    )


def _converged(orig: SideState, recomp: SideState) -> bool:
    return (
        orig.regs == recomp.regs
        and orig.flags == recomp.flags
        and orig.carry == recomp.carry
        and orig.fpu_flags == recomp.fpu_flags
        and orig.x87.state_key() == recomp.x87.state_key()
    )


def verify_cfg_effective_match(
    orig_asm: AsmStream,
    recomp_asm: AsmStream,
    orig_targets: list[int | None],
    recomp_targets: list[int | None],
    metadata: FunctionMetadata | None = None,
    orig_meta: list[InstructionMeta | None] | None = None,
    recomp_meta: list[InstructionMeta | None] | None = None,
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
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    orig_stream = resolve_asm_stream(orig_asm)
    recomp_stream = resolve_asm_stream(recomp_asm)
    orig_asm = orig_stream.displays
    recomp_asm = recomp_stream.displays
    total = len(orig_asm)
    if (
        len(recomp_asm) != total
        or len(orig_targets) != total
        or len(recomp_targets) != total
        or total == 0
    ):
        if recorder is not None:
            recorder.mark_inconclusive(
                "alignment_failure",
                facts={
                    "stage": "positional_cfg_precondition",
                    "orig_instruction_count": total,
                    "recomp_instruction_count": len(recomp_asm),
                    "orig_target_count": len(orig_targets),
                    "recomp_target_count": len(recomp_targets),
                },
            )
        return False
    if orig_targets != recomp_targets:
        if recorder is not None:
            differing = next(
                (
                    index
                    for index, pair in enumerate(zip(orig_targets, recomp_targets))
                    if pair[0] != pair[1]
                ),
                None,
            )
            if differing is not None:
                meta_o = orig_meta[differing] if orig_meta is not None else None
                meta_r = recomp_meta[differing] if recomp_meta is not None else None
                try:
                    facts_o = _target_facts(
                        instruction_at(orig_stream, differing),
                        meta_o,
                        orig_targets[differing],
                    )
                    facts_r = _target_facts(
                        instruction_at(recomp_stream, differing),
                        meta_r,
                        recomp_targets[differing],
                    )
                except (Reject, IndexError, KeyError, ValueError, TypeError):
                    facts_o = facts_r = {}
                recorder.record_difference(
                    "branch_target",
                    differing,
                    differing,
                    facts_o,
                    facts_r,
                )
        return False

    def classify(stream: ResolvedAsm) -> list[str]:
        kinds = []
        for index in range(len(stream)):
            if is_data_row(stream, index):
                kinds.append("data")
                continue
            try:
                mnemonic = instruction_at(stream, index).mnemonic
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                mnemonic = stream.displays[index].partition(" ")[0]
            if mnemonic in JCC_MNEMONICS:
                kinds.append("jcc")
            elif mnemonic in ("jmp", "ret"):
                kinds.append(mnemonic)
            elif mnemonic in ("loop", "loope", "loopne", "jcxz", "jecxz"):
                kinds.append("jcc")
            else:
                kinds.append("code")
        return kinds

    kinds = classify(orig_stream)
    recomp_kinds = classify(recomp_stream)
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
                            "data_line": orig_asm[target],
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

    entry: dict[int, _CfgState] = {
        0: _CfgState(
            SideState(rename_slots=False),
            SideState(rename_slots=False),
            ("cfg_mem_init",),
        )
    }
    pending = [0]
    visits = 0

    def run_block(block: int) -> bool:
        flow = _clone_cfg_state(entry[block])
        orig_state, recomp_state = flow.orig, flow.recomp
        ctx = Context(gen=flow.memory, metadata=metadata, recorder=recorder)
        _seed_context_from_cfg(ctx, flow)
        for i in range(order[block], ends[block]):
            line_o, line_r = orig_asm[i], recomp_asm[i]
            if kinds[i] == "data":
                if line_o != line_r:
                    return False
                continue
            try:
                ins_o = instruction_at(orig_stream, i)
                ins_r = instruction_at(recomp_stream, i)
                _record_operand_candidate(ctx, i, i, ins_o, ins_r)
                obs_o: list = []
                obs_r: list = []
                state_before_o = _clone_state(orig_state)
                state_before_r = _clone_state(recomp_state)
                execute(orig_state, ctx, i, ins_o, obs_o)
                execute(recomp_state, ctx, i, ins_r, obs_r)
            except (Reject, IndexError, KeyError, ValueError, TypeError):
                if line_o != line_r:
                    if recorder is not None:
                        recorder.mark_inconclusive("unsupported_instruction", i, i)
                    return False
                meta_o = orig_meta[i] if orig_meta is not None else None
                meta_r = recomp_meta[i] if recomp_meta is not None else None
                if admit_unsupported_identical(
                    orig_state, recomp_state, ctx, i, meta_o, meta_r
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

            if _callee_save_swap(
                ctx, ins_o, ins_r, obs_o, obs_r, orig_state, recomp_state
            ):
                _commit_memory(ctx, obs_o, i)
                continue

            if not _accept_agreeing_pair(
                ctx,
                i,
                i,
                (state_before_o, state_before_r),
                (orig_state, recomp_state),
                (ins_o, ins_r),
                (obs_o, obs_r),
                (
                    orig_meta[i] if orig_meta is not None else None,
                    recomp_meta[i] if recomp_meta is not None else None,
                ),
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
                if not _converged(orig_state, recomp_state):
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "external_control_flow_state",
                            i,
                            i,
                            {"edge_kind": kinds[i]},
                        )
                    return False
            _commit_memory(ctx, obs_o, i)

        last_kind = kinds[ends[block] - 1]
        if last_kind == "ret":
            if not _discharge_run_obligations(
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

        outgoing = _capture_cfg_state(orig_state, recomp_state, ctx)
        if recorder is not None:
            recorder.reasons.update(ctx.categories)
        # Propagate to successors.
        for successor in successors(block):
            if successor not in entry:
                entry[successor] = _clone_cfg_state(outgoing)
                pending.append(successor)
            else:
                joined = _join_states(entry[successor], outgoing, successor)
                if joined is None:
                    if recorder is not None:
                        recorder.mark_inconclusive(
                            "state_join_failure",
                            order[successor],
                            order[successor],
                            _join_failure_facts(entry[successor], outgoing),
                        )
                    return False
                if not _states_equal(joined, entry[successor]):
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
