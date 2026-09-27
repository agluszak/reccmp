"""Per-side basic-block graphs: control kinds, switch tables,
canonicalization and structural block pairing."""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Hashable, Sequence

from reccmp.compare.asm.ir import (
    DecodedInstruction,
    FunctionImage,
    instruction_semantic_key,
)
from reccmp.compare.asm.verifier.evidence import target_facts
from reccmp.compare.asm.verifier.state import JCC_MNEMONICS
from reccmp.compare.diagnosis import AnalysisRecorder

# Each side's basic-block graph is built independently; blocks pair by
# control-flow structure, so branch targets compare as matched blocks, not
# as displacements or instruction positions.


_LOOPS = frozenset({"loop", "loope", "loopne", "jcxz", "jecxz"})


def _control_kind(row: DecodedInstruction) -> str:
    if row.mnemonic in JCC_MNEMONICS or row.mnemonic in _LOOPS:
        return "jcc"
    if row.mnemonic in ("jmp", "ret"):
        return row.mnemonic
    return "code"


@dataclass
class _SideCfg:
    starts: list[int]
    ends: list[int]
    # Per block: role ("taken"/"fall"/"jmp"/"fallout"/"caseN") -> successor
    # block index, or "external" for a target outside the excerpt.
    succ: list[dict[str, int | str]]
    kinds: list[str]
    # jmp line index -> case destination line indices (recognized switches).
    table_dests: dict[int, list[int]]


def _mark_side_inconclusive(
    recorder: AnalysisRecorder | None,
    side: str,
    reason: str,
    index: int | None,
    facts: dict[str, str | int | bool | None],
) -> None:
    if recorder is None:
        return
    if side == "orig":
        recorder.mark_inconclusive(reason, orig_index=index, facts=facts)
    else:
        recorder.mark_inconclusive(reason, recomp_index=index, facts=facts)


def block_terminator(_start: int, end: int) -> int:
    """Last instruction in a code-only block."""
    return end - 1


def build_side_cfg(
    image: FunctionImage,
    *,
    recorder: AnalysisRecorder | None = None,
    side: str = "orig",
) -> _SideCfg | None:
    """Project the function's canonical graph into verifier block indices."""
    rows = image.instructions
    graph = image.control_graph()
    total = len(rows)
    if total == 0:
        _mark_side_inconclusive(
            recorder,
            side,
            "empty_control_flow",
            None,
            {"side": side, "instruction_count": 0},
        )
        return None
    kinds = [_control_kind(row) for row in rows]
    for i, successors in enumerate(graph.edges):
        if kinds[i] == "jmp" and any(edge.kind == "unknown" for edge in successors):
            switch_candidate = any(
                op[0] == "mem" and any(scale == 4 for _reg, scale in op[3])
                for op in rows[i].operands
            )
            _mark_side_inconclusive(
                recorder,
                side,
                "jump_table_data" if switch_candidate else "indirect_jump",
                i,
                {
                    "side": side,
                    "failure": (
                        "unresolved_switch_table"
                        if switch_candidate
                        else "indirect_target"
                    ),
                },
            )
            return None
    order = [block.start for block in graph.blocks]
    ends = [block.end for block in graph.blocks]
    index = {start: n for n, start in enumerate(order)}
    succ = [
        {
            edge.label: (
                index[edge.target]
                if edge.kind == "local" and edge.target is not None
                else "external"
            )
            for edge in block.successors
        }
        for block in graph.blocks
    ]
    return _SideCfg(
        starts=list(order),
        ends=ends,
        succ=succ,
        kinds=kinds,
        table_dests={i: list(dests) for i, dests in graph.table_dests},
    )


def _block_code_indices(cfg: _SideCfg, block: int) -> list[int]:
    return list(range(cfg.starts[block], cfg.ends[block]))


def _rebuild_cfg_keeping(
    cfg: _SideCfg,
    keep: list[int],
    *,
    redirect: dict[int, int] | None = None,
) -> _SideCfg:
    """Return a CFG containing only ``keep`` blocks (in that order).

    ``redirect`` maps removed/old block ids onto a surviving old id before
    the keep-list remapping is applied. Edge targets that cannot be
    resolved become ``\"external\"``.
    """
    redirect = dict(redirect or {})
    old_to_new = {old: new for new, old in enumerate(keep)}

    def map_target(target: int | str) -> int | str:
        if not isinstance(target, int):
            return target
        seen: set[int] = set()
        while target in redirect and target not in seen:
            seen.add(target)
            target = redirect[target]
        return old_to_new.get(target, "external")

    return _SideCfg(
        starts=[cfg.starts[b] for b in keep],
        ends=[cfg.ends[b] for b in keep],
        succ=[
            {role: map_target(dest) for role, dest in cfg.succ[b].items()} for b in keep
        ],
        kinds=cfg.kinds,
        table_dests=cfg.table_dests,
    )


def _is_empty_jump_block(cfg: _SideCfg, block: int) -> bool:
    """Single internal ``jmp`` with no other code — a jump-only trampoline."""
    indices = _block_code_indices(cfg, block)
    if len(indices) != 1:
        return False
    insn = indices[0]
    if cfg.kinds[insn] != "jmp" or insn in cfg.table_dests:
        return False
    edges = cfg.succ[block]
    if set(edges) != {"jmp"}:
        return False
    return isinstance(edges["jmp"], int)


def _is_empty_ret_block(cfg: _SideCfg, block: int) -> bool:
    indices = _block_code_indices(cfg, block)
    return len(indices) == 1 and cfg.kinds[indices[0]] == "ret" and not cfg.succ[block]


def _remove_empty_jump_blocks(cfg: _SideCfg) -> tuple[_SideCfg, bool]:
    """Thread A → empty-jmp → B into A → B. Conservative: leave unsure alone."""
    n = len(cfg.starts)
    redirect: dict[int, int] = {}
    removed: set[int] = set()
    for block in range(n):
        if not _is_empty_jump_block(cfg, block):
            continue
        target = cfg.succ[block]["jmp"]
        assert isinstance(target, int)
        # Don't create a self-loop trampoline or remove a block that jumps
        # into another trampoline we're already collapsing onto itself.
        if target == block:
            continue
        redirect[block] = target
        removed.add(block)

    if not removed:
        return cfg, False

    # Resolve redirect chains (empty jmp → empty jmp → real).
    def resolve(block: int) -> int:
        seen: set[int] = set()
        while block in redirect and block not in seen:
            seen.add(block)
            block = redirect[block]
        return block

    redirect = {b: resolve(t) for b, t in redirect.items()}
    # Drop any redirect that still lands on a removed block (cycle).
    redirect = {b: t for b, t in redirect.items() if t not in removed}
    removed = {b for b in removed if b in redirect}
    if not removed:
        return cfg, False

    # If the entry block is removed, its ultimate target becomes the new entry.
    entry = 0
    if entry in removed:
        entry = redirect[entry]
    keep = [entry] + [b for b in range(n) if b not in removed and b != entry]
    return _rebuild_cfg_keeping(cfg, keep, redirect=redirect), True


def _predecessor_counts(cfg: _SideCfg) -> list[int]:
    counts = [0] * len(cfg.starts)
    for edges in cfg.succ:
        for dest in edges.values():
            if isinstance(dest, int):
                counts[dest] += 1
    return counts


def _merge_trivial_fallthrough_splits(cfg: _SideCfg) -> tuple[_SideCfg, bool]:
    """Merge A --fall--> B when B has a single predecessor and ranges abut."""
    preds = _predecessor_counts(cfg)
    n = len(cfg.starts)
    # child block -> parent that absorbs it (only direct pairs this pass)
    absorb: dict[int, int] = {}
    claimed_parents: set[int] = set()
    for block_a in range(n):
        if block_a in absorb or block_a in claimed_parents:
            continue
        edges = cfg.succ[block_a]
        if set(edges) != {"fall"}:
            continue
        block_b = edges["fall"]
        if not isinstance(block_b, int):
            continue
        if block_b == block_a or block_b in absorb or block_b in claimed_parents:
            continue
        if preds[block_b] != 1:
            continue
        if cfg.ends[block_a] != cfg.starts[block_b]:
            continue
        if _is_empty_jump_block(cfg, block_b):
            continue
        last_b = block_terminator(cfg.starts[block_b], cfg.ends[block_b])
        if last_b in cfg.table_dests:
            continue
        absorb[block_b] = block_a
        claimed_parents.add(block_a)

    if not absorb:
        return cfg, False

    new_starts = list(cfg.starts)
    new_ends = list(cfg.ends)
    new_succ = [dict(edges) for edges in cfg.succ]
    for child, parent in absorb.items():
        new_ends[parent] = cfg.ends[child]
        new_succ[parent] = dict(cfg.succ[child])

    keep = [b for b in range(n) if b not in absorb]
    old_to_new = {old: new for new, old in enumerate(keep)}
    remapped_succ: list[dict[str, int | str]] = []
    for old in keep:
        remapped: dict[str, int | str] = {}
        for role, dest in new_succ[old].items():
            if isinstance(dest, int):
                # Absorbed children are gone; edges to them should already
                # have been rewritten on the parent. If any remain, external.
                remapped[role] = old_to_new.get(dest, "external")
            else:
                remapped[role] = dest
        remapped_succ.append(remapped)
    return (
        _SideCfg(
            starts=[new_starts[b] for b in keep],
            ends=[new_ends[b] for b in keep],
            succ=remapped_succ,
            kinds=cfg.kinds,
            table_dests=cfg.table_dests,
        ),
        True,
    )


def _collapse_duplicate_ret_blocks(
    cfg: _SideCfg, rows: Sequence[DecodedInstruction]
) -> tuple[_SideCfg, bool]:
    """Collapse empty ``ret`` blocks that share identical terminator text.

    ``ret`` and ``ret 4`` must not be treated as the same exit — collapsing
    them would let stdcall/cdecl differences prove EFFECTIVE.
    """
    groups: dict[Hashable, list[int]] = {}
    for block in range(len(cfg.starts)):
        if not _is_empty_ret_block(cfg, block):
            continue
        indices = _block_code_indices(cfg, block)
        groups.setdefault(instruction_semantic_key(rows[indices[0]]), []).append(block)

    redirect: dict[int, int] = {}
    for blocks in groups.values():
        if len(blocks) < 2:
            continue
        canonical = blocks[0]
        for block in blocks[1:]:
            redirect[block] = canonical
    if not redirect:
        return cfg, False
    keep = [b for b in range(len(cfg.starts)) if b not in redirect]
    return _rebuild_cfg_keeping(cfg, keep, redirect=redirect), True


def canonicalize_side_cfg(
    cfg: _SideCfg, rows: Sequence[DecodedInstruction]
) -> _SideCfg:
    """Conservative CFG cleanup before isomorphic block pairing.

    Removes empty jump-only trampolines, merges trivial fallthrough splits,
    and collapses duplicated empty ``ret`` blocks that share the same text.
    Transforms that are not clearly safe are skipped.
    """
    current = cfg
    for _ in range(len(cfg.starts) + 2):
        current, jumped = _remove_empty_jump_blocks(current)
        current, fell = _merge_trivial_fallthrough_splits(current)
        current, rets = _collapse_duplicate_ret_blocks(current, rows)
        if not (jumped or fell or rets):
            break
    return current


def pair_cfg_blocks(
    cfg_o: _SideCfg,
    cfg_r: _SideCfg,
    recorder: AnalysisRecorder | None = None,
    rows: tuple[Sequence[DecodedInstruction], Sequence[DecodedInstruction]] = ((), ()),
) -> list[tuple[int, int]] | None:
    """Match the two sides' reachable blocks into a structural bijection,
    starting from the entry blocks and following same-role edges. Returns
    the matched pairs in discovery order, or None if the reachable graphs
    are not isomorphic. With the
    ``rows``, a branch whose same-role edges reach blocks paired elsewhere
    is recorded as a branch-target difference."""
    map_o: dict[int, int] = {}
    map_r: dict[int, int] = {}
    order: list[tuple[int, int]] = []
    # (orig block, recomp block, the paired blocks and role that reach them)
    queue: list[tuple[int, int, tuple[int, int, str] | None]] = [(0, 0, None)]
    while queue:
        block_o, block_r, edge = queue.pop()
        seen_o = block_o in map_o
        seen_r = block_r in map_r
        if seen_o or seen_r:
            if map_o.get(block_o) != block_r or map_r.get(block_r) != block_o:
                if recorder is None:
                    return None
                if edge is not None and edge[2] != "fall" and all(rows):
                    branch_o = cfg_o.ends[edge[0]] - 1
                    branch_r = cfg_r.ends[edge[1]] - 1
                    recorder.record_difference(
                        "branch_target",
                        branch_o,
                        branch_r,
                        target_facts(rows[0][branch_o], cfg_o.starts[block_o]),
                        target_facts(rows[1][branch_r], cfg_r.starts[block_r]),
                    )
                    return None
                recorder.mark_inconclusive(
                    "non_isomorphic_cfg",
                    orig_index=cfg_o.starts[block_o],
                    recomp_index=cfg_r.starts[block_r],
                    facts={
                        "failure": "block_mapping_conflict",
                        "orig_block_count": len(cfg_o.starts),
                        "recomp_block_count": len(cfg_r.starts),
                        "orig_block": block_o,
                        "recomp_block": block_r,
                    },
                )
                return None
            continue
        map_o[block_o] = block_r
        map_r[block_r] = block_o
        order.append((block_o, block_r))
        edges_o = cfg_o.succ[block_o]
        edges_r = cfg_r.succ[block_r]
        if set(edges_o) != set(edges_r):
            if recorder is not None:
                recorder.mark_inconclusive(
                    "non_isomorphic_cfg",
                    orig_index=cfg_o.starts[block_o],
                    recomp_index=cfg_r.starts[block_r],
                    facts={
                        "failure": "edge_roles",
                        "orig_block_count": len(cfg_o.starts),
                        "recomp_block_count": len(cfg_r.starts),
                        "orig_edge_roles": ",".join(sorted(edges_o)),
                        "recomp_edge_roles": ",".join(sorted(edges_r)),
                    },
                )
            return None
        for role, to_o in edges_o.items():
            to_r = edges_r[role]
            if (to_o == "external") != (to_r == "external"):
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "non_isomorphic_cfg",
                        orig_index=cfg_o.starts[block_o],
                        recomp_index=cfg_r.starts[block_r],
                        facts={
                            "failure": "external_edge",
                            "edge_role": role,
                            "orig_external": to_o == "external",
                            "recomp_external": to_r == "external",
                            "orig_block_count": len(cfg_o.starts),
                            "recomp_block_count": len(cfg_r.starts),
                        },
                    )
                return None
            if to_o != "external":
                assert isinstance(to_o, int) and isinstance(to_r, int)
                queue.append((to_o, to_r, (block_o, block_r, role)))
    return order
