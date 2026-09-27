"""Control flow derived once from a function's decoded code and switch tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .ir import DecodedInstruction, ExtentKind, JumpTable
from .operand import Imm, Mem, Sym

EdgeKind = Literal["local", "external", "fallout", "unknown"]


@dataclass(frozen=True)
class GraphEdge:
    label: str
    kind: EdgeKind
    target: int | None = None  # instruction index for a local edge
    address: int | None = None
    modeled_external: bool = False


@dataclass(frozen=True)
class GraphBlock:
    start: int
    end: int
    successors: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class FunctionGraph:
    """Instruction-index graph; an absent successor means a local terminal."""

    extent: int
    blocks: tuple[GraphBlock, ...]
    edges: tuple[tuple[GraphEdge, ...], ...]
    reachable: frozenset[int]
    returns: frozenset[int]
    table_dests: tuple[tuple[int, tuple[int, ...]], ...]

    def extent_closed(
        self, *, extent_kind: ExtentKind, coverage_incomplete: bool = False
    ) -> bool:
        """Every reachable path ends locally or at an admissible external jump."""
        if coverage_incomplete:
            return False
        if not self.edges:
            return self.extent <= 0
        for index in self.reachable:
            for edge in self.edges[index]:
                if edge.kind == "local":
                    continue
                if edge.kind == "external" and (
                    edge.modeled_external or extent_kind is ExtentKind.KNOWN
                ):
                    continue
                return False
        return True


_MODELED_EXTERNAL = frozenset(
    {"entity", "import", "jmp_through", "unmatched", "symbol"}
)


def _modeled_external(row: DecodedInstruction) -> bool:
    identity = row.control_target
    return (
        isinstance(identity, tuple)
        and bool(identity)
        and identity[0] in _MODELED_EXTERNAL
    )


def _switch_table(
    row: DecodedInstruction, jump_tables: Sequence[JumpTable]
) -> JumpTable | None:
    if row.address is None:
        return None
    match row.operands:
        case (Mem(terms=terms),) if any(term.scale == 4 for term in terms):
            return next(
                (
                    table
                    for table in jump_tables
                    if table.dispatch_address == row.address
                    and table.is_recognized_switch()
                ),
                None,
            )
    return None


def build_function_graph(
    instructions: Sequence[DecodedInstruction],
    jump_tables: Sequence[JumpTable] = (),
    *,
    start_addr: int,
    extent: int,
) -> FunctionGraph:
    """Build block boundaries, successors and reachable instructions."""
    rows = tuple(instructions)
    if any(row.address is None for row in rows):
        raise ValueError("FunctionGraph requires decoded instructions only")
    by_addr = {row.address: index for index, row in enumerate(rows)}
    window = range(start_addr, start_addr + extent)
    table_dests: list[tuple[int, tuple[int, ...]]] = []

    def edge(label: str, address: int, row: DecodedInstruction) -> GraphEdge:
        target = by_addr.get(address)
        if target is not None:
            return GraphEdge(label, "local", target, address)
        if address in window:
            return GraphEdge(label, "unknown", address=address)
        if label == "fall":
            return GraphEdge("fallout", "fallout", address=address)
        return GraphEdge(
            label, "external", address=address, modeled_external=_modeled_external(row)
        )

    edges: list[tuple[GraphEdge, ...]] = []
    returns: set[int] = set()
    for index, row in enumerate(rows):
        assert row.address is not None
        if row.is_ret:
            returns.add(index)
        if not (row.is_jump or row.falls_through):
            edges.append(())
            continue
        fall = row.address + row.size
        if row.is_jump:
            branches: tuple[GraphEdge, ...]
            if row.branch_target is not None:
                branches = (
                    edge(
                        "taken" if row.is_conditional else "jmp",
                        row.branch_target,
                        row,
                    ),
                )
            else:
                table = _switch_table(row, jump_tables)
                direct_external = bool(
                    row.operands and isinstance(row.operands[0], (Imm, Sym))
                )
                branches = (
                    tuple(
                        (
                            edge(f"case{case}", target, row)
                            if target in by_addr
                            else GraphEdge(f"case{case}", "unknown", address=target)
                        )
                        for case, (_entry, target) in enumerate(table.entries)
                    )
                    if table is not None and table.entries
                    else (
                        GraphEdge(
                            "taken" if row.is_conditional else "jmp",
                            "external" if direct_external else "unknown",
                        ),
                    )
                )
                if (
                    table is not None
                    and branches
                    and all(branch.kind == "local" for branch in branches)
                ):
                    table_dests.append(
                        (
                            index,
                            tuple(
                                branch.target
                                for branch in branches
                                if branch.target is not None
                            ),
                        )
                    )
            edges.append(
                (*branches, edge("fall", fall, row)) if row.falls_through else branches
            )
            continue
        edges.append((edge("fall", fall, row),))

    reachable: set[int] = set()
    pending = [0] if rows else []
    while pending:
        index = pending.pop()
        if index in reachable:
            continue
        reachable.add(index)
        pending.extend(
            successor.target
            for successor in edges[index]
            if successor.kind == "local" and successor.target is not None
        )

    leaders = {0} if rows else set()
    for index, successors in enumerate(edges):
        leaders.update(
            successor.target
            for successor in successors
            if successor.kind == "local"
            and (successor.label != "fall" or successor.target != index + 1)
            and successor.target is not None
        )
        if (rows[index].is_jump or not rows[index].falls_through) and index + 1 < len(
            rows
        ):
            leaders.add(index + 1)
    order = sorted(leaders)
    blocks = tuple(
        GraphBlock(
            start,
            order[position + 1] if position + 1 < len(order) else len(rows),
            edges[
                (order[position + 1] if position + 1 < len(order) else len(rows)) - 1
            ],
        )
        for position, start in enumerate(order)
    )
    return FunctionGraph(
        extent=extent,
        blocks=blocks,
        edges=tuple(edges),
        reachable=frozenset(reachable),
        returns=frozenset(returns),
        table_dests=tuple(table_dests),
    )
