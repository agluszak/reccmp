"""Canonical instruction IR for the compare pipeline.

``DecodedInstruction`` is the single representation, produced by Capstone
detail-mode decode (typed operands) and sanitization. Every analysis reads
its structured fields; ``display`` is rendered once for humans and JSON
diffs and is never read back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

from .model import Reference
from .operand import Mem, Operand, SignedSymbol, Sym

if TYPE_CHECKING:
    from .graph import FunctionGraph


class FlowKind(Enum):
    """Where execution goes after an instruction."""

    NORMAL = "normal"  # the next instruction
    CALL = "call"  # a callee, then the next instruction
    CONDITIONAL = "conditional"  # a target or the next instruction (jcc, loop)
    JUMP = "jump"  # a target only
    RETURN = "return"  # the caller
    TRAP = "trap"  # nowhere this function models (int3)


class ExtentKind(Enum):
    """How the compared byte window was chosen."""

    KNOWN = "known"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class JumpTable:
    """One switch address table discovered during function decoding.

    ``entries`` are ``(entry_address, target_address)`` dword pairs.
    ``dispatch_address`` and ``index_register`` are set only when decoding
    found the ``jmp dword ptr [index*4 + table]`` that indexes the table
    (see ``parse.switch_index_register``); only then is it a recognized switch.
    """

    address: int
    entries: tuple[tuple[int, int], ...]
    dispatch_address: int | None = None
    index_register: str | None = None

    def is_recognized_switch(self) -> bool:
        return self.dispatch_address is not None and self.index_register is not None


@dataclass(frozen=True)
class DataRegion:
    """Bytes embedded in a function body but not decoded as instructions."""

    address: int
    data: bytes


def instruction_ids(rows: Sequence[DecodedInstruction]) -> dict[int, int]:
    """Instruction position by address."""
    return {
        row.address: index for index, row in enumerate(rows) if row.address is not None
    }


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
    insn_ids = instruction_ids(excerpt)
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

    def rewrite_reference(ref: Reference) -> Reference:
        bound = bind(ref.identity)
        return ref if bound == ref.identity else replace(ref, identity=bound)

    def rewrite_operand(operand: Operand) -> Operand:
        match operand:
            case Sym(ref):
                return Sym(rewrite_reference(ref))
            case Mem(symbols=symbols) if symbols:
                return replace(
                    operand,
                    symbols=tuple(
                        SignedSymbol(term.sign, rewrite_reference(term.ref))
                        for term in symbols
                    ),
                )
        return operand

    rebound: list[DecodedInstruction] = []
    for row in excerpt:
        operands = tuple(rewrite_operand(operand) for operand in row.operands)
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

    @cached_property
    def _ids_by_address(self) -> dict[int, int]:
        return instruction_ids(self.instructions)

    def index_of(self, address: int | None) -> int | None:
        """The instruction at ``address``, by its position; None when no
        instruction starts there."""
        return None if address is None else self._ids_by_address.get(address)

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


@dataclass(frozen=True)
class DecodedInstruction:
    """One canonical instruction in a function image."""

    # pylint: disable=too-many-instance-attributes

    address: int | None
    size: int
    mnemonic: str
    prefix: str
    operands: tuple[Operand, ...]
    # Rendered once, for humans and diffs; never read back.
    display: str
    # Capstone detail facts.
    regs_read: tuple[str, ...] = ()
    regs_written: tuple[str, ...] = ()
    reads_flags: bool = False
    writes_flags: bool = False
    accesses_memory: bool = False
    flow: FlowKind = FlowKind.NORMAL
    branch_target: int | None = None
    # False when Capstone could not report register access (CsError). Empty
    # regs_read/regs_written then means "unknown", not "touches nothing".
    register_access_known: bool = True
    # False when any operand is opaque or has an unknown memory size — match
    # keys must not be treated as a complete semantic model of the instruction.
    operand_model_complete: bool = True
    # False when jump/call target modeling is incomplete (e.g. opaque operands).
    control_flow_known: bool = True
    # Proof identity of a jump/call destination. Display may be a relative
    # displacement; this is never that displacement.
    control_target: Hashable | None = None

    @property
    def is_jump(self) -> bool:
        return self.flow in (FlowKind.CONDITIONAL, FlowKind.JUMP)

    @property
    def is_conditional(self) -> bool:
        return self.flow is FlowKind.CONDITIONAL

    @property
    def is_call(self) -> bool:
        return self.flow is FlowKind.CALL

    @property
    def is_ret(self) -> bool:
        return self.flow is FlowKind.RETURN

    @property
    def falls_through(self) -> bool:
        """Whether execution may continue at the next instruction."""
        return self.flow in (FlowKind.NORMAL, FlowKind.CALL, FlowKind.CONDITIONAL)


# Identities private to one image: across images such references match by
# the placeholder or name they show, for scoring only.
_SIDE_LOCAL = frozenset({"local", "unresolved", "unmatched"})


def _reference_key(ref: Reference, semantic: bool) -> Reference:
    """The reference reduced to what it compares by: its identity, or for
    scoring, the placeholder or name a side-local reference shows."""
    match ref.identity:
        case (str() as kind, *_) if not semantic and kind in _SIDE_LOCAL:
            return Reference("", ("shown", ref.display))
    return Reference("", ref.identity)


def operand_key(operand: Operand, *, semantic: bool = False) -> Operand:
    """The operand with its references reduced to what they compare by."""
    match operand:
        case Sym(ref):
            return Sym(_reference_key(ref, semantic))
        case Mem(symbols=symbols) if symbols:
            return replace(
                operand,
                symbols=tuple(
                    SignedSymbol(term.sign, _reference_key(term.ref, semantic))
                    for term in symbols
                ),
            )
    return operand


def instruction_match_key(row: DecodedInstruction) -> Hashable:
    """Hashable SequenceMatcher key for scoring and the diff.

    A resolved reference contributes its identity; a side-local one the
    placeholder or name it shows, so ``<OFFSET1>`` on both sides lines up.
    Proofs use ``instruction_semantic_key`` instead.
    """
    return ("ins", row.mnemonic, row.prefix, operand_match_key(row))


def operand_match_key(row: DecodedInstruction) -> tuple[Operand, ...]:
    """The operands' part of ``instruction_match_key``."""
    return tuple(operand_key(operand) for operand in row.operands)


def instruction_semantic_key(row: DecodedInstruction) -> Hashable:
    """Proof key: every reference compares by identity."""
    return (
        "ins",
        row.mnemonic,
        row.prefix,
        tuple(operand_key(operand, semantic=True) for operand in row.operands),
        row.control_target,
    )
