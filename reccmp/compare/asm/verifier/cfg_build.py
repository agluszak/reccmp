"""Per-side basic-block graphs: control kinds, switch tables,
canonicalization and structural block pairing."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from collections.abc import Hashable, Sequence

from reccmp.compare.asm.ir import (
    AsmRole,
    DecodedInstruction,
    JumpTable,
    instruction_semantic_key,
    local_branch_targets,
)
from reccmp.compare.asm.verifier.state import JCC_MNEMONICS
from reccmp.compare.diagnosis import AnalysisRecorder, FactValue

# ---------------------------------------------------------------------------
# Isomorphic-CFG verification (structure-matched, alignment-free)
#
# The positional CFG verifier above requires the two sequences to have equal
# length and identical line-index branch structure. Register-allocation
# entropy breaks both: a folded load or an elided register copy shifts every
# following line and every crossing branch displacement. This verifier
# instead builds each side's basic-block graph independently, pairs blocks
# by control-flow structure (so branch targets compare as matched blocks,
# not displacement text), aligns each block pair's instructions locally,
# and runs the same paired symbolic execution with dataflow joins.


_LOOPS = frozenset({"loop", "loope", "loopne", "jcxz", "jecxz"})


def _control_kind(row: DecodedInstruction) -> str:
    if not row.is_code:
        return "data"
    if row.mnemonic in JCC_MNEMONICS or row.mnemonic in _LOOPS:
        return "jcc"
    if row.mnemonic in ("jmp", "ret"):
        return row.mnemonic
    return "code"


def _scale4_mem_jmp(row: DecodedInstruction) -> bool:
    """True for ``jmp dword ptr [idx*4 + …]`` regardless of table base."""
    match row.mnemonic, row.operands:
        case "jmp", (("mem", _, _, reg_terms, _, _),):
            return any(scale == 4 for _reg, scale in reg_terms)
    return False


def _is_recognized_switch_jmp(row: DecodedInstruction) -> bool:
    """True for ``jmp dword ptr [idx*4 + table]`` (scale-4 mem with a base)."""
    match row.operands:
        case (("mem", _, _, _, disp, syms),) if _scale4_mem_jmp(row):
            # Require an identifiable table base (symbol and/or displacement).
            return bool(syms) or disp != 0
    return False


def _case_offset(row: DecodedInstruction) -> int | None:
    """A jump-table entry's target, as an offset from the function start."""
    match row.role, row.operands:
        case AsmRole.JUMP_TABLE_ENTRY, (("case", int() as offset),):
            return offset
    return None


def extract_switch_tables(
    rows: Sequence[DecodedInstruction],
    kinds: list[str],
    jump_tables: Sequence[JumpTable] = (),
) -> tuple[dict[int, list[int]], set[int]] | None:
    # pylint: disable=too-many-nested-blocks,too-many-branches
    """Map recognized switch jmps to case destination indices.

    Returns ``(jmp_index -> dest line indices, owned data line indices)``.
    ``None`` means a candidate table could not be resolved conservatively.

    A table is recognized when a switch jmp is followed by a jump-table
    header row and then contiguous entry rows.
    """
    total = len(rows)
    table_dests: dict[int, list[int]] = {}
    owned: set[int] = set()
    addr_index: dict[int, int] = {}
    func_start: int | None = None
    for i, row in enumerate(rows):
        if row.address is None or kinds[i] == "data":
            continue
        if func_start is None:
            func_start = row.address
        addr_index.setdefault(row.address, i)
    entry_index = {row.address: i for i, row in enumerate(rows) if not row.is_code}

    def _is_table_header(index: int) -> bool:
        return rows[index].role == AsmRole.JUMP_TABLE_HEADER

    # Prefer the JumpTables InstructGen found.
    for table in jump_tables:
        if table.dispatch_address is None:
            continue
        dispatch_i = addr_index.get(table.dispatch_address)
        if dispatch_i is None or kinds[dispatch_i] != "jmp":
            continue
        # First-class tables still have to be a scale-4 indexed mem jmp.
        # Metadata alone (a dispatch address on any jmp) is not enough.
        if not table.is_recognized_switch() or not _scale4_mem_jmp(rows[dispatch_i]):
            continue
        dests: list[int] = []
        entry_indices: list[int] = []
        for entry_addr, target_va in table.entries:
            dest_i = addr_index.get(target_va)
            if dest_i is None:
                dests = []
                break
            entry_i = entry_index.get(entry_addr)
            if entry_i is not None:
                entry_indices.append(entry_i)
            dests.append(dest_i)
        if not dests:
            continue
        table_dests[dispatch_i] = dests
        owned.update(entry_indices)
        if entry_indices:
            header_i = min(entry_indices) - 1
            if header_i >= 0 and _is_table_header(header_i):
                owned.add(header_i)

    i = 0
    while i < total:
        if i in table_dests:
            i += 1
            continue
        if (
            kinds[i] == "jmp"
            and _is_recognized_switch_jmp(rows[i])
            and i + 1 < total
            and _is_table_header(i + 1)
        ):
            entries: list[int] = []
            j = i + 2
            while j < total and _case_offset(rows[j]) is not None:
                entries.append(j)
                j += 1
            if entries:
                if func_start is None:
                    return None
                dests = []
                for entry_i in entries:
                    offset = _case_offset(rows[entry_i])
                    assert offset is not None
                    dest_i = addr_index.get(func_start + offset)
                    if dest_i is None:
                        return None
                    dests.append(dest_i)
                table_dests[i] = dests
                owned.update(range(i + 1, j))
                i = j
                continue
        i += 1
    return table_dests, owned


@dataclass
class _SideCfg:
    starts: list[int]
    ends: list[int]
    # Per block: role ("taken"/"fall"/"jmp"/"fallout"/"caseN") -> successor
    # block index, or "external" for a target outside the excerpt.
    succ: list[dict[str, int | str]]
    kinds: list[str]
    # Jump-table header/entry lines attached to recognized switches; excluded
    # from block bodies during alignment / symbolic execution.
    owned_data: frozenset[int] = frozenset()
    # jmp line index -> case destination line indices (recognized switches).
    table_dests: dict[int, list[int]] = field(default_factory=dict)


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


def block_terminator(
    start: int, end: int, kinds: list[str], owned_data: set[int] | frozenset[int]
) -> int:
    """Last non-owned instruction in ``[start, end)``, preferring a control op."""
    last_code = start
    for i in range(end - 1, start - 1, -1):
        if i in owned_data:
            continue
        if kinds[i] in ("jcc", "jmp", "ret"):
            return i
        last_code = i
        break
    return last_code


def build_side_cfg(
    rows: Sequence[DecodedInstruction],
    jump_tables: Sequence[JumpTable] = (),
    *,
    recorder: AnalysisRecorder | None = None,
    side: str = "orig",
) -> _SideCfg | None:
    """One side's basic-block structure, or None when the shape is outside
    this verifier's model (unrecognized jump/data tables, invalid targets).

    Recognized ``jmp [idx*4 + table]`` + jump-table rows become ``caseN``
    CFG edges; other table/data rows still bail.
    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements
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
    targets = local_branch_targets(rows)
    kinds = [_control_kind(row) for row in rows]
    extracted = extract_switch_tables(rows, kinds, jump_tables)
    if extracted is None:
        first_data = next((i for i, k in enumerate(kinds) if k == "data"), 0)
        _mark_side_inconclusive(
            recorder,
            side,
            "jump_table_data",
            first_data,
            {
                "side": side,
                "data_line_count": kinds.count("data"),
                "data_line": rows[first_data].display,
                "failure": "unresolved_switch_table",
            },
        )
        return None
    table_dests, owned_data = extracted
    unowned_data = [
        i for i, kind in enumerate(kinds) if kind == "data" and i not in owned_data
    ]
    if unowned_data:
        first_data = unowned_data[0]
        _mark_side_inconclusive(
            recorder,
            side,
            "jump_table_data",
            first_data,
            {
                "side": side,
                "data_line_count": len(unowned_data),
                "data_line": rows[first_data].display,
            },
        )
        return None
    leaders = {0}
    for i in range(total):
        if i in owned_data:
            continue
        target = targets[i]
        if target is not None:
            if not 0 <= target < total:
                _mark_side_inconclusive(
                    recorder,
                    side,
                    "invalid_control_flow_target",
                    i,
                    {"side": side, "target_instruction_index": target},
                )
                return None
            leaders.add(target)
        if i in table_dests:
            for dest in table_dests[i]:
                leaders.add(dest)
            continue
        if kinds[i] in ("jcc", "jmp", "ret") and i + 1 < total:
            nxt = i + 1
            while nxt < total and nxt in owned_data:
                nxt += 1
            if nxt < total:
                leaders.add(nxt)
    order = sorted(leaders)
    index = {start: n for n, start in enumerate(order)}
    ends = [order[n + 1] if n + 1 < len(order) else total for n in range(len(order))]
    succ: list[dict[str, int | str]] = []
    for n, start in enumerate(order):
        last = block_terminator(start, ends[n], kinds, owned_data)
        kind = kinds[last]
        edges: dict[str, int | str] = {}
        if kind == "jcc":
            target = targets[last]
            edges["taken"] = index[target] if target is not None else "external"
            if ends[n] < total:
                edges["fall"] = index[ends[n]]
            else:
                edges["fallout"] = "external"
        elif kind == "jmp":
            if last in table_dests:
                for case_i, dest in enumerate(table_dests[last]):
                    edges[f"case{case_i}"] = index[dest]
            else:
                target = targets[last]
                edges["jmp"] = index[target] if target is not None else "external"
        elif kind == "ret":
            pass
        else:
            if ends[n] < total:
                edges["fall"] = index[ends[n]]
            else:
                edges["fallout"] = "external"
        succ.append(edges)
    return _SideCfg(
        starts=list(order),
        ends=ends,
        succ=succ,
        kinds=kinds,
        owned_data=frozenset(owned_data),
        table_dests=table_dests,
    )


def _block_code_indices(cfg: _SideCfg, block: int) -> list[int]:
    return [
        i for i in range(cfg.starts[block], cfg.ends[block]) if i not in cfg.owned_data
    ]


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
        owned_data=cfg.owned_data,
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
        last_b = block_terminator(
            cfg.starts[block_b], cfg.ends[block_b], cfg.kinds, cfg.owned_data
        )
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
            owned_data=cfg.owned_data,
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
    facts: dict[str, FactValue] | None = None,
) -> list[tuple[int, int]] | None:
    """Match the two sides' reachable blocks into a structural bijection,
    starting from the entry blocks and following same-role edges. Returns
    the matched pairs in discovery order, or None if the reachable graphs
    are not isomorphic; ``facts`` join the ones recorded then."""
    map_o: dict[int, int] = {}
    map_r: dict[int, int] = {}
    order: list[tuple[int, int]] = []
    queue: list[tuple[int, int]] = [(0, 0)]
    while queue:
        block_o, block_r = queue.pop()
        seen_o = block_o in map_o
        seen_r = block_r in map_r
        if seen_o or seen_r:
            if map_o.get(block_o) != block_r or map_r.get(block_r) != block_o:
                if recorder is not None:
                    recorder.mark_inconclusive(
                        "non_isomorphic_cfg",
                        orig_index=cfg_o.starts[block_o],
                        recomp_index=cfg_r.starts[block_r],
                        facts={
                            **(facts or {}),
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
                        **(facts or {}),
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
                            **(facts or {}),
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
                queue.append((to_o, to_r))
    return order
