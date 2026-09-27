"""Control flow derived once from a function's decoded code and switch tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property

from .ir import DecodedInstruction, ExtentKind, JumpTable
from .operand import Imm, Mem, Sym


class EdgeRole(Enum):
    """How control leaves an instruction along an edge."""

    FALL = "fall"  # to the next instruction
    TAKEN = "taken"  # a conditional jump's target
    JUMP = "jump"  # an unconditional jump's target
    CASE = "case"  # one entry of a switch table


class EdgeKind(Enum):
    """Where an edge leads."""

    LOCAL = "local"  # an instruction of this function
    EXTERNAL = "external"  # outside the function's bytes
    UNKNOWN = "unknown"  # inside the bytes but not an instruction, or unknown


@dataclass(frozen=True)
class GraphEdge:
    role: EdgeRole
    kind: EdgeKind
    target: int | None = None  # instruction index for a local edge
    address: int | None = None
    modeled_external: bool = False
    case: int | None = None  # the table entry of a CASE edge

    @property
    def falls_out(self) -> bool:
        """Execution runs past the end of the function's bytes."""
        return self.role is EdgeRole.FALL and self.kind is EdgeKind.EXTERNAL


@dataclass(frozen=True)
class GraphBlock:
    start: int
    end: int
    successors: tuple[GraphEdge, ...]

    @property
    def last(self) -> int:
        """The block's last instruction, the one its successors leave."""
        return self.end - 1


@dataclass(frozen=True)
class FunctionGraph:
    """Instruction-index graph; an absent successor means a local terminal."""

    extent: int
    blocks: tuple[GraphBlock, ...]
    edges: tuple[tuple[GraphEdge, ...], ...]
    reachable: frozenset[int]
    returns: frozenset[int]
    table_dests: tuple[tuple[int, tuple[int, ...]], ...]

    @cached_property
    def _block_starting_at(self) -> dict[int, int]:
        return {block.start: number for number, block in enumerate(self.blocks)}

    def block_at(self, index: int) -> int:
        """The number of the block that starts at instruction ``index``."""
        return self._block_starting_at[index]

    def shape(self) -> tuple[tuple[GraphEdge, ...], ...] | None:
        """Every instruction's edges with addresses erased: two functions
        with the same shape transfer control between the same positions.
        None when an edge leads somewhere unknown."""
        if any(edge.kind is EdgeKind.UNKNOWN for edges in self.edges for edge in edges):
            return None
        return tuple(
            tuple(replace(edge, address=None) for edge in edges) for edges in self.edges
        )

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
                if edge.kind is EdgeKind.LOCAL:
                    continue
                if (
                    edge.kind is EdgeKind.EXTERNAL
                    and not edge.falls_out
                    and (edge.modeled_external or extent_kind is ExtentKind.KNOWN)
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

    def edge(
        role: EdgeRole, address: int, row: DecodedInstruction, case: int | None = None
    ) -> GraphEdge:
        target = by_addr.get(address)
        if target is not None:
            return GraphEdge(role, EdgeKind.LOCAL, target, address, case=case)
        if address in window:
            return GraphEdge(role, EdgeKind.UNKNOWN, address=address, case=case)
        return GraphEdge(
            role,
            EdgeKind.EXTERNAL,
            address=address,
            modeled_external=role is not EdgeRole.FALL and _modeled_external(row),
            case=case,
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
                        EdgeRole.TAKEN if row.is_conditional else EdgeRole.JUMP,
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
                            edge(EdgeRole.CASE, target, row, case)
                            if target in by_addr
                            else GraphEdge(
                                EdgeRole.CASE,
                                EdgeKind.UNKNOWN,
                                address=target,
                                case=case,
                            )
                        )
                        for case, (_entry, target) in enumerate(table.entries)
                    )
                    if table is not None and table.entries
                    else (
                        GraphEdge(
                            EdgeRole.TAKEN if row.is_conditional else EdgeRole.JUMP,
                            EdgeKind.EXTERNAL if direct_external else EdgeKind.UNKNOWN,
                        ),
                    )
                )
                if (
                    table is not None
                    and branches
                    and all(branch.kind is EdgeKind.LOCAL for branch in branches)
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
                (*branches, edge(EdgeRole.FALL, fall, row))
                if row.falls_through
                else branches
            )
            continue
        edges.append((edge(EdgeRole.FALL, fall, row),))

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
            if successor.kind is EdgeKind.LOCAL and successor.target is not None
        )

    leaders = {0} if rows else set()
    for index, successors in enumerate(edges):
        leaders.update(
            successor.target
            for successor in successors
            if successor.kind is EdgeKind.LOCAL
            and (successor.role is not EdgeRole.FALL or successor.target != index + 1)
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
