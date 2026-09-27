"""Canonical instruction IR for the compare pipeline.

``DecodedInstruction`` is the single representation, produced by Capstone
detail-mode decode (typed operands) and sanitization. Every analysis reads
its structured fields; ``display`` is rendered once for humans and JSON
diffs and is never read back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

from .model import Reference

if TYPE_CHECKING:
    from .graph import FunctionGraph

_STACK_SLOT = ("stack_slot",)


class ExtentKind(Enum):
    """How the compared byte window was chosen."""

    KNOWN = "known"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class JumpTable:
    """One switch address table discovered during function decoding.

    ``entries`` are ``(entry_address, target_address)`` pairs. Optional
    ``dispatch_address`` is the ``jmp`` that indexes the table when known.
    ``scale`` / ``entry_width`` / ``index_register`` describe the dispatch
    addressing form; a table is only a recognized MSVC switch when the
    dispatch is ``jmp dword ptr [index*4 + table]``.
    """

    address: int
    entries: tuple[tuple[int, int], ...]
    dispatch_address: int | None = None
    scale: int = 4
    entry_width: int = 4
    index_register: str | None = None

    def is_recognized_switch(self) -> bool:
        return (
            self.scale == 4
            and self.entry_width == 4
            and self.dispatch_address is not None
            and self.index_register is not None
        )


@dataclass(frozen=True)
class DataRegion:
    """Bytes embedded in a function body but not decoded as instructions."""

    address: int
    data: bytes


def rebind_local_identities(
    excerpt: Sequence[DecodedInstruction],
    *,
    start_addr: int,
    extent: int,
    jump_tables: Sequence[JumpTable] = (),
    image_id: str | None = None,
) -> tuple[DecodedInstruction, ...]:
    """Rewrite in-body OFFSET identities to instruction/table ids.

    A raw byte offset is not proof identity unless the destination is a
    decoded instruction or a discovered jump table. Unpaired local data
    stays side-local.
    """
    insn_ids = {
        row.address: (row.instruction_id if row.instruction_id is not None else index)
        for index, row in enumerate(excerpt)
        if row.address is not None
    }
    table_ids = {table.address: index for index, table in enumerate(jump_tables)}
    entry_ids: dict[int, tuple[int, int]] = {}
    for table_index, table in enumerate(jump_tables):
        for entry_index, (entry_addr, _target) in enumerate(table.entries):
            entry_ids[entry_addr] = (table_index, entry_index)
    window = range(start_addr, start_addr + max(extent, 0))
    side = image_id

    def bind(ident: Hashable) -> Hashable:
        match ident:
            case ("local", int() as offset):
                abs_addr = start_addr + offset
            case ("unresolved", _, int() as address, *_):
                abs_addr = address
            case _:
                return ident
        insn_id = insn_ids.get(abs_addr)
        if insn_id is not None:
            return ("local_insn", insn_id)
        table_id = table_ids.get(abs_addr)
        if table_id is not None:
            return ("table", table_id)
        entry = entry_ids.get(abs_addr)
        if entry is not None:
            return ("table_entry", entry[0], entry[1])
        if abs_addr in window:
            return ("unresolved", side, abs_addr)
        return ident

    def rewrite_value(value):
        match value:
            case Reference(display=display, identity=identity, entity_type=entity_type):
                bound = bind(identity)
                return (
                    value
                    if bound == identity
                    else Reference(display, bound, entity_type)
                )
            case tuple():
                return tuple(rewrite_value(item) for item in value)
            case list():
                return [rewrite_value(item) for item in value]
            case _:
                return value

    rebound: list[DecodedInstruction] = []
    for row in excerpt:
        operands = rewrite_value(row.operands)
        control = bind(row.control_target) if row.control_target is not None else None
        if operands != row.operands or control != row.control_target:
            rebound.append(replace(row, operands=operands, control_target=control))
        else:
            rebound.append(row)
    return tuple(rebound)


@dataclass(frozen=True)
class FunctionImage:
    # pylint: disable=too-many-instance-attributes
    """Lossless view of one decoded function for comparison.

    Owns the extent evidence, decoded instructions, jump tables, and coverage
    flag for a single side (original or recompiled). Callers should not
    read state from the decoder after construction.
    """

    start_addr: int
    extent: int
    extent_kind: ExtentKind
    instructions: tuple[DecodedInstruction, ...]
    jump_tables: tuple[JumpTable, ...] = ()
    coverage_incomplete: bool = False
    extent_closed: bool = True
    raw: bytes | None = None
    data_regions: tuple[DataRegion, ...] = ()
    graph: FunctionGraph | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if any(row.size <= 0 or not row.mnemonic for row in self.instructions):
            raise ValueError("FunctionImage.instructions must contain instructions")

    @property
    def instruction_ids(self) -> tuple[int, ...]:
        """Stable program-point ids owned by this image."""
        return tuple(
            row.instruction_id if row.instruction_id is not None else index
            for index, row in enumerate(self.instructions)
        )

    def with_instructions(
        self, instructions: Sequence[DecodedInstruction]
    ) -> "FunctionImage":
        """Return a copy with the supplied instructions and their ids."""
        return replace(self, instructions=tuple(instructions), graph=None)

    def control_graph(self) -> FunctionGraph:
        # pylint: disable=import-outside-toplevel
        """The graph built with this image, or derived for a synthetic fixture."""
        if self.graph is not None:
            return self.graph
        from .graph import build_function_graph

        return build_function_graph(
            self.instructions,
            self.jump_tables,
            start_addr=self.start_addr,
            extent=self.extent,
        )

    @property
    def data_shape(self) -> tuple:
        """Embedded data and table positions relative to the function start."""
        return (
            tuple(
                (region.address - self.start_addr, region.data)
                for region in self.data_regions
            ),
            tuple(
                (
                    table.address - self.start_addr,
                    tuple(entry - self.start_addr for entry, _target in table.entries),
                )
                for table in self.jump_tables
            ),
        )

    @property
    def control_flow_complete(self) -> bool:
        """Every transfer is decoded or backed by a recognized switch table."""
        switches = {
            table.dispatch_address
            for table in self.jump_tables
            if table.is_recognized_switch()
        }
        return all(
            row.control_flow_known or (row.is_jump and row.address in switches)
            for row in self.instructions
        )


@dataclass(frozen=True)
class DecodedInstruction:
    """One canonical instruction in a function image."""

    # pylint: disable=too-many-instance-attributes

    address: int | None
    size: int
    mnemonic: str
    prefix: str
    # Typed operands: ("reg", name), ("imm", value), ("st", index),
    # ("sym", Reference), ("mem", size, segment, reg_terms, displacement,
    # symbols), ("opaque", ...).
    operands: tuple
    # Rendered once, for humans and diffs; never read back.
    display: str
    # Capstone detail facts.
    regs_read: tuple[str, ...] = ()
    regs_written: tuple[str, ...] = ()
    reads_flags: bool = False
    writes_flags: bool = False
    accesses_memory: bool = False
    is_jump: bool = False
    is_call: bool = False
    is_ret: bool = False
    branch_target: int | None = None
    # False when Capstone could not report register access (CsError). Empty
    # regs_read/regs_written then means "unknown", not "touches nothing".
    register_access_known: bool = True
    # False when any operand is opaque or has an unknown memory size — match
    # keys must not be treated as a complete semantic model of the instruction.
    operand_model_complete: bool = True
    # False when jump/call target modeling is incomplete (e.g. opaque operands).
    control_flow_known: bool = True
    # Immutable program-point identity assigned by ``FunctionImage``.
    instruction_id: int | None = None
    # Proof identity of a jump/call destination. Display may be a relative
    # displacement; this is never that displacement.
    control_target: Hashable | None = None


# Identities private to one image: across images such references match by
# the placeholder or name they show, for scoring only.
_SIDE_LOCAL = frozenset({"local", "unresolved", "unmatched"})


def _freeze(value, *, semantic: bool = False) -> Hashable:
    match value:
        case list() | tuple():
            return tuple(_freeze(item, semantic=semantic) for item in value)
        case Reference(identity=identity) if semantic:
            return identity
        case Reference(display=display, identity=(kind, *_)) if kind in _SIDE_LOCAL:
            return display
        case Reference(identity=identity):
            return identity
        case _:
            return value


def instruction_match_key(row: DecodedInstruction) -> Hashable:
    """Hashable SequenceMatcher key for scoring and the diff.

    A resolved reference contributes its identity; a side-local one the
    placeholder or name it shows, so ``<OFFSET1>`` on both sides lines up.
    Proofs use ``instruction_semantic_key`` instead.
    """
    return ("ins", row.mnemonic, row.prefix, _freeze(row.operands))


def instruction_semantic_key(row: DecodedInstruction) -> Hashable:
    """Proof key: every reference compares by identity."""
    return (
        "ins",
        row.mnemonic,
        row.prefix,
        _freeze(row.operands, semantic=True),
        row.control_target,
    )


def _operand_identity(row: DecodedInstruction) -> Hashable:
    match row.operands:
        case (("sym", Reference(identity=identity)), *_):
            return identity
        case (operand, *_):
            return _freeze(operand, semantic=True)
        case _:
            return None


def _local_destination_id(
    target: int | None, addr_to_id: dict[int, int]
) -> Hashable | None:
    if target is None:
        return None
    return addr_to_id.get(target)


def _branch_proof_identity(
    row: DecodedInstruction,
    addr_to_id: dict[int, int],
) -> Hashable | None:
    """Proof identity of a direct branch, never a relative displacement."""
    local_id = _local_destination_id(row.branch_target, addr_to_id)
    if local_id is not None:
        return ("local", local_id)
    if row.control_target is not None:
        return ("ext", row.control_target)
    if row.branch_target is not None:
        return ("ext", ("unresolved", None, row.branch_target))
    ident = _operand_identity(row)
    if ident is not None:
        return ("ext", ident)
    return None


def _switch_destination_key(
    row: DecodedInstruction,
    addr_to_id: dict[int, int],
    jump_tables: Sequence[JumpTable],
) -> Hashable | None:
    for table in jump_tables:
        if table.dispatch_address != row.address:
            continue
        cases = tuple(
            (
                ("L", addr_to_id[target])
                if target in addr_to_id
                else ("ext", ("unresolved", None, target))
            )
            for _entry, target in table.entries
        )
        if cases:
            return ("switch", cases)
    return None


def control_flow_topology_keys(
    excerpt: Sequence[DecodedInstruction],
    jump_tables: Sequence[JumpTable] = (),
) -> tuple[Hashable, ...] | None:
    """Per-row exact control-flow identities, or None if a transfer is unmodeled.

    Ordinary instructions contribute ``()``. Local branches contribute the
    destination instruction id in this excerpt. External branches contribute
    a ``ControlTarget`` identity (entity, import, unmatched, or side-local
    address) — never a relative displacement. Switch tables contribute the
    tuple of case destination ids.
    """
    addr_to_id = {
        row.address: (row.instruction_id if row.instruction_id is not None else index)
        for index, row in enumerate(excerpt)
        if row.address is not None
    }
    keys: list[Hashable] = []
    for row in excerpt:
        if row.is_call:
            keys.append(("call", _operand_identity(row)))
            continue
        if not (row.is_jump or row.is_ret):
            keys.append(())
            continue
        if row.is_ret:
            keys.append(("ret",))
            continue
        proof = _branch_proof_identity(row, addr_to_id)
        if isinstance(proof, tuple) and proof[0] == "local":
            keys.append(proof)
            continue
        if proof is not None and row.branch_target is not None:
            keys.append(proof)
            continue
        switch_key = _switch_destination_key(row, addr_to_id, jump_tables)
        if switch_key is not None:
            keys.append(switch_key)
            continue
        return None
    return tuple(keys)


def local_destination_keys(
    excerpt: Sequence[DecodedInstruction],
    *,
    start_addr: int,
    extent: int,
) -> tuple[Hashable, ...] | None:
    """Per-row instruction ids of local branch and switch-case destinations.

    Rows without a local destination contribute ``()``. Displays show local
    branches as byte displacements, which identify the same instruction on
    both sides only if every crossed instruction has the same encoding
    length; these keys make the destination explicit. Returns None when a
    destination falls inside the extent but not on an instruction boundary.
    """
    addr_to_id = {
        row.address: (row.instruction_id if row.instruction_id is not None else index)
        for index, row in enumerate(excerpt)
        if row.address is not None
    }

    def key(target: int | None, tag: str) -> Hashable | None:
        if target is None or not start_addr <= target < start_addr + extent:
            return ()
        local_id = addr_to_id.get(target)
        return None if local_id is None else (tag, local_id)

    keys: list[Hashable] = []
    for row in excerpt:
        if row.is_jump:
            item = key(row.branch_target, "local")
        else:
            item = ()
        if item is None:
            return None
        keys.append(item)
    return tuple(keys)


def compute_extent_closed(
    excerpt: Sequence[DecodedInstruction],
    *,
    start_addr: int,
    extent: int,
    coverage_incomplete: bool = False,
    jump_tables: Sequence[JumpTable] = (),
    extent_kind: ExtentKind = ExtentKind.KNOWN,
) -> bool:
    # pylint: disable=import-outside-toplevel
    """Extent closure from the same control-flow graph used by comparison."""
    from .graph import build_function_graph

    return build_function_graph(
        excerpt, jump_tables, start_addr=start_addr, extent=extent
    ).extent_closed(extent_kind=extent_kind, coverage_incomplete=coverage_incomplete)


def _normalize_operand_stack(operand) -> Hashable:
    """An operand with a stack slot's displacement erased."""
    match operand:
        case ("mem", size, seg, reg_terms, _, ()) if {
            name for name, _scale in reg_terms
        } & {"ebp", "esp"}:
            return ("mem", size, seg, _freeze(reg_terms), _STACK_SLOT, ())
        case _:
            return _freeze(operand)


def stack_normalized_key(row: DecodedInstruction) -> Hashable:
    """The match key of a row with its stack slots' displacements erased."""
    operands = tuple(_normalize_operand_stack(op) for op in row.operands)
    return ("ins", row.mnemonic, row.prefix, operands)


def local_branch_targets(rows: Sequence[DecodedInstruction]) -> list[int | None]:
    """Each row's local branch destination, as the index of the row it
    reaches; None for a row that is not a jump inside the rows (calls are
    not local control flow)."""
    index_of = {row.address: i for i, row in enumerate(rows) if row.address is not None}
    return [
        (
            index_of.get(row.branch_target)
            if row.branch_target is not None and not row.is_call
            else None
        )
        for row in rows
    ]
