"""Canonical instruction IR for the compare pipeline.

``DecodedInstruction`` is the single representation produced by Capstone
detail-mode decode (typed operands from detail) + sanitization. Display
strings exist only for humans and JSON diffs. Matching, stack scoring,
inline fingerprints, and the effective verifier consume structured fields
via ``ResolvedAsm`` / ``instruction_at``; ``parse_instruction`` remains a
legacy fallback for string-only callers and incomplete rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Hashable, Sequence, Union

from .model import (
    STACK_ENTRY_REGEX,
    Instruction,
    Reference,
    Reject,
    operand_identity,
    parse_instruction,
)

_MEM = "mem"
_STACK_SLOT = ("stack_slot",)


class AsmRole(Enum):
    """What kind of row this is in a function excerpt."""

    CODE = auto()
    JUMP_TABLE_HEADER = auto()
    JUMP_TABLE_ENTRY = auto()
    DATA_TABLE_HEADER = auto()
    DATA_TABLE_ENTRY = auto()


class ExtentKind(Enum):
    """How the compared byte window was chosen."""

    KNOWN = "known"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class JumpTable:
    """One switch address table discovered by ``InstructGen``.

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
        row.address: (
            row.instruction_id if row.instruction_id is not None else index
        )
        for index, row in enumerate(excerpt)
        if row.address is not None and row.is_code
    }
    table_ids = {table.address: index for index, table in enumerate(jump_tables)}
    entry_ids: dict[int, tuple[int, int]] = {}
    for table_index, table in enumerate(jump_tables):
        for entry_index, (entry_addr, _target) in enumerate(table.entries):
            entry_ids[entry_addr] = (table_index, entry_index)
    window = range(start_addr, start_addr + max(extent, 0))
    side = image_id

    def bind(ident: Hashable) -> Hashable:
        abs_addr: int | None = None
        if isinstance(ident, tuple) and ident:
            if ident[0] == "local" and len(ident) == 2 and isinstance(ident[1], int):
                abs_addr = start_addr + ident[1]
            elif (
                ident[0] == "unresolved"
                and len(ident) >= 3
                and isinstance(ident[2], int)
            ):
                abs_addr = ident[2]
        if abs_addr is None:
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
        if isinstance(value, Reference):
            new_ident = bind(value.identity)
            if new_ident != value.identity:
                return Reference(value.display, new_ident)
            return value
        if isinstance(value, tuple):
            return tuple(rewrite_value(item) for item in value)
        if isinstance(value, list):
            return [rewrite_value(item) for item in value]
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
    """Lossless view of one decoded function for comparison.

    Owns the extent evidence, decoded excerpt, jump tables, and coverage
    flag for a single side (original or recompiled). Callers should not
    read long-lived state off a shared ``ParseAsm`` after construction.
    """

    start_addr: int
    extent: int
    extent_kind: ExtentKind
    excerpt: tuple[DecodedInstruction, ...]
    jump_tables: tuple[JumpTable, ...] = ()
    coverage_incomplete: bool = False
    extent_closed: bool = True
    raw: bytes | None = None

    @property
    def instruction_ids(self) -> tuple[int, ...]:
        """Stable program-point ids owned by this image."""
        return tuple(
            row.instruction_id if row.instruction_id is not None else index
            for index, row in enumerate(self.excerpt)
        )

    def with_excerpt(
        self, excerpt: Sequence[DecodedInstruction]
    ) -> "FunctionImage":
        """Return a copy whose excerpt (and ids) come from ``excerpt``."""
        return replace(self, excerpt=tuple(excerpt))


@dataclass(frozen=True)
class DecodedInstruction:
    """One canonical instruction (or table marker) in a function excerpt."""

    # pylint: disable=too-many-instance-attributes

    address: int | None
    size: int
    mnemonic: str
    prefix: str
    operands: tuple
    raw_operands: tuple[str, ...]
    display: str
    role: AsmRole = AsmRole.CODE
    # Capstone detail facts (empty for table markers).
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
    # Unsanitized Capstone op_str (useful for debug / jump-table discovery).
    raw_op_str: str = ""
    # Immutable program-point identity assigned by ``FunctionImage``.
    instruction_id: int | None = None
    # Proof identity of a jump/call destination. Display may be a relative
    # displacement; this is never that displacement.
    control_target: Hashable | None = None

    @property
    def is_code(self) -> bool:
        return self.role == AsmRole.CODE

    def as_effective(self):
        """View used by the effective-match verifier."""
        return Instruction(
            self.mnemonic,
            self.prefix,
            self.operands,
            self.raw_operands,
            control_target=self.control_target,
        )

    def with_display(self, display: str) -> "DecodedInstruction":
        """Replace the display string and refresh structured operands from it."""
        if self.role != AsmRole.CODE:
            return replace(
                self,
                display=display,
                mnemonic="",
                prefix="",
                operands=(),
                raw_operands=(),
            )
        try:
            parsed = parse_instruction(display)
        except Reject:
            return replace(self, display=display)
        return replace(
            self,
            display=display,
            mnemonic=parsed.mnemonic,
            prefix=parsed.prefix,
            operands=parsed.operands,
            raw_operands=parsed.raw_operands,
        )


def marker(
    display: str,
    *,
    address: int | None = None,
    role: AsmRole,
) -> DecodedInstruction:
    """Build a non-code excerpt row (jump/data table header or entry)."""
    return DecodedInstruction(
        address=address,
        size=0,
        mnemonic="",
        prefix="",
        operands=(),
        raw_operands=(),
        display=display,
        role=role,
    )


def from_effective(
    address: int | None,
    size: int,
    instruction,
    display: str,
    *,
    raw_op_str: str = "",
    meta: object | None = None,
) -> DecodedInstruction:
    """Assemble a code row from a parsed Instruction plus optional Capstone meta."""
    kwargs: dict = {
        "address": address,
        "size": size,
        "mnemonic": instruction.mnemonic,
        "prefix": instruction.prefix,
        "operands": instruction.operands,
        "raw_operands": instruction.raw_operands,
        "display": display,
        "role": AsmRole.CODE,
        "raw_op_str": raw_op_str,
        "control_target": getattr(instruction, "control_target", None),
    }
    if meta is not None:
        kwargs.update(
            regs_read=getattr(meta, "regs_read", ()),
            regs_written=getattr(meta, "regs_written", ()),
            reads_flags=getattr(meta, "reads_flags", False),
            writes_flags=getattr(meta, "writes_flags", False),
            accesses_memory=getattr(meta, "accesses_memory", False),
            is_jump=getattr(meta, "is_jump", False),
            is_call=getattr(meta, "is_call", False),
            is_ret=getattr(meta, "is_ret", False),
            branch_target=getattr(meta, "branch_target", None),
            register_access_known=getattr(meta, "register_access_known", True),
            operand_model_complete=getattr(meta, "operand_model_complete", True),
            control_flow_known=getattr(meta, "control_flow_known", True),
            control_target=getattr(meta, "control_target", None)
            or kwargs.get("control_target"),
        )
    return DecodedInstruction(**kwargs)


def _freeze(value, *, semantic: bool = False) -> Hashable:
    if isinstance(value, list):
        return tuple(_freeze(item, semantic=semantic) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item, semantic=semantic) for item in value)
    if semantic:
        return operand_identity(value)
    if isinstance(value, Reference):
        return value.display
    return value


def instruction_match_key(row: DecodedInstruction | str) -> Hashable:
    """Hashable SequenceMatcher key from IR (or legacy display string).

    Reference operands contribute their display token so scoring can treat
    ``<OFFSET1>`` as the same placeholder. Proofs use
    ``instruction_semantic_key`` instead.
    """
    if isinstance(row, str):
        try:
            ins = parse_instruction(row)
        except Reject:
            return ("raw", row)
        return ("ins", ins.mnemonic, ins.prefix, _freeze(ins.operands))
    if row.role != AsmRole.CODE:
        return ("raw", row.display)
    return ("ins", row.mnemonic, row.prefix, _freeze(row.operands))


def instruction_semantic_key(row: DecodedInstruction | str | Instruction) -> Hashable:
    """Proof key: unresolved references compare by identity, not placeholder."""
    if isinstance(row, str):
        try:
            ins = parse_instruction(row)
        except Reject:
            return ("raw", row)
        return (
            "ins",
            ins.mnemonic,
            ins.prefix,
            _freeze(ins.operands, semantic=True),
            ins.control_target,
        )
    if isinstance(row, Instruction):
        return (
            "ins",
            row.mnemonic,
            row.prefix,
            _freeze(row.operands, semantic=True),
            row.control_target,
        )
    if row.role != AsmRole.CODE:
        return ("raw", row.display)
    return (
        "ins",
        row.mnemonic,
        row.prefix,
        _freeze(row.operands, semantic=True),
        row.control_target,
    )


def _operand_identity(row: DecodedInstruction) -> Hashable:
    if row.operands:
        op = row.operands[0]
        if isinstance(op, tuple) and op and op[0] == "sym" and len(op) > 1:
            return operand_identity(op[1])
        return _freeze(op, semantic=True)
    if row.raw_operands:
        return row.raw_operands[0]
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
        row.address: (
            row.instruction_id if row.instruction_id is not None else index
        )
        for index, row in enumerate(excerpt)
        if row.address is not None
    }
    keys: list[Hashable] = []
    for row in excerpt:
        if row.role == AsmRole.JUMP_TABLE_ENTRY:
            target_id = None
            for table in jump_tables:
                for entry_addr, target in table.entries:
                    if entry_addr == row.address:
                        target_id = _local_destination_id(target, addr_to_id)
                        keys.append(
                            ("case", target_id)
                            if target_id is not None
                            else ("case_ext", ("unresolved", None, target))
                        )
                        break
                else:
                    continue
                break
            else:
                keys.append(("table_entry", row.display))
            continue
        if not row.is_code:
            keys.append(())
            continue
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
        if proof is not None and proof[0] == "local":
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


_NO_FALLTHROUGH = frozenset({"ret", "jmp", "int3"})
_MODELED_EXTERNAL = frozenset({"entity", "import", "unmatched", "symbol"})


def _is_modeled_external_target(row: DecodedInstruction) -> bool:
    """True when the destination is independently known as another entity."""
    ident = row.control_target
    if ident is None:
        ident = _operand_identity(row)
    if not isinstance(ident, tuple) or not ident:
        return False
    return ident[0] in _MODELED_EXTERNAL


def _enqueue_or_close_target(
    target: int,
    *,
    addr_to_row: dict[int, DecodedInstruction],
    window: range,
    extent_kind: ExtentKind,
    row: DecodedInstruction,
    pending: list[int],
) -> bool:
    """Enqueue an in-window target, accept a modeled external, or reject.

    Returns False when an estimated extent jumps into unknown bytes.
    """
    if target in addr_to_row or target in window:
        pending.append(target)
        return True
    if _is_modeled_external_target(row):
        return True
    if extent_kind is ExtentKind.ESTIMATED:
        return False
    pending.append(target)
    return True


def compute_extent_closed(
    excerpt: Sequence[DecodedInstruction],
    *,
    start_addr: int,
    extent: int,
    coverage_incomplete: bool = False,
    jump_tables: Sequence[JumpTable] = (),
    extent_kind: ExtentKind = ExtentKind.KNOWN,
) -> bool:
    """True when every reachable path ends inside a modeled terminal.

    A coverage walk can only prove the supplied byte window. Implicit
    fallthrough past the window is never a modeled terminal, even for a
    known annotated size. Explicit ``jmp`` to a known entity, import, or
    unmatched symbol can close a path. For estimated extents, jumping into
    unknown bytes just past the guessed window is not a terminal.
    """
    if coverage_incomplete:
        return False
    if extent <= 0:
        return True
    code = [row for row in excerpt if row.is_code and row.address is not None]
    if not code:
        return False
    addr_to_row = {row.address: row for row in code}
    window = range(start_addr, start_addr + extent)
    pending = [code[0].address]
    seen: set[int] = set()
    while pending:
        addr = pending.pop()
        if addr in seen:
            continue
        seen.add(addr)
        row = addr_to_row.get(addr)
        if row is None:
            if addr in window:
                return False
            if extent_kind is ExtentKind.ESTIMATED:
                return False
            continue
        nxt = addr + row.size
        mnemonic = row.mnemonic
        if mnemonic in _NO_FALLTHROUGH:
            if mnemonic == "jmp":
                if row.branch_target is not None:
                    if not _enqueue_or_close_target(
                        row.branch_target,
                        addr_to_row=addr_to_row,
                        window=window,
                        extent_kind=extent_kind,
                        row=row,
                        pending=pending,
                    ):
                        return False
                else:
                    table = _table_for_dispatch(row.address, jump_tables)
                    if table is None:
                        return False
                    for _entry, target in table.entries:
                        if not _enqueue_or_close_target(
                            target,
                            addr_to_row=addr_to_row,
                            window=window,
                            extent_kind=extent_kind,
                            row=row,
                            pending=pending,
                        ):
                            return False
            continue
        if row.is_jump and row.branch_target is not None:
            if not _enqueue_or_close_target(
                row.branch_target,
                addr_to_row=addr_to_row,
                window=window,
                extent_kind=extent_kind,
                row=row,
                pending=pending,
            ):
                return False
        elif row.is_jump and row.branch_target is None:
            table = _table_for_dispatch(row.address, jump_tables)
            if table is None:
                return False
            for _entry, target in table.entries:
                if not _enqueue_or_close_target(
                    target,
                    addr_to_row=addr_to_row,
                    window=window,
                    extent_kind=extent_kind,
                    row=row,
                    pending=pending,
                ):
                    return False
            if mnemonic == "jmp":
                continue
        if nxt not in window:
            return False
        pending.append(nxt)
    return True


def _table_for_dispatch(
    address: int | None, jump_tables: Sequence[JumpTable]
) -> JumpTable | None:
    if address is None:
        return None
    for table in jump_tables:
        if table.dispatch_address == address:
            return table
    return None


def _normalize_operand_stack(operand) -> object:
    if not isinstance(operand, tuple) or not operand:
        return operand
    if operand[0] != _MEM:
        return _freeze(operand)
    size, seg, reg_terms, _disp, syms = (
        operand[1],
        operand[2],
        operand[3],
        operand[4],
        operand[5],
    )
    regs = {name for name, _scale in reg_terms}
    if regs & {"ebp", "esp"} and not syms:
        return (_MEM, size, seg, _freeze(reg_terms), _STACK_SLOT, ())
    return _freeze(operand)


def stack_normalized_key(row: DecodedInstruction | str) -> Hashable:
    if isinstance(row, str):
        try:
            ins = parse_instruction(row)
        except Reject:
            return ("raw", row)
        operands = tuple(_normalize_operand_stack(op) for op in ins.operands)
        return ("ins", ins.mnemonic, ins.prefix, operands)
    if row.role != AsmRole.CODE:
        return ("raw", row.display)
    operands = tuple(_normalize_operand_stack(op) for op in row.operands)
    return ("ins", row.mnemonic, row.prefix, operands)


def rewrite_stack_displacements(
    line: str, mapping: dict[tuple[str, int], tuple[str, int]]
) -> str:
    """Rewrite ebp/esp ± offset tokens in a display line through a slot map."""

    def repl(match) -> str:
        register = match.group("register")
        offset = int(match.group("sign") + match.group("offset"), 16)
        target = mapping.get((register, offset))
        if target is None:
            return match.group(0)
        tgt_reg, tgt_off = target
        if tgt_off >= 0:
            return f"{tgt_reg} + {tgt_off:#x}"
        return f"{tgt_reg} - {-tgt_off:#x}"

    return STACK_ENTRY_REGEX.sub(repl, line)


def excerpt_displays(excerpt: list[DecodedInstruction]) -> list[str]:
    return [row.display for row in excerpt]


def excerpt_addrs(excerpt: list[DecodedInstruction]) -> list[int | None]:
    return [row.address for row in excerpt]


def as_addr_display_pairs(
    excerpt: list[DecodedInstruction],
) -> list[tuple[int | None, str]]:
    """Tuple form for APIs that still consume ``(addr, display)`` pairs."""
    return [(row.address, row.display) for row in excerpt]


@dataclass(frozen=True)
class ResolvedAsm:
    """Parallel display / Instruction / role views of one function excerpt.

    When ``from_ir`` is True, ``instructions[i]`` is already decoded for every
    CODE row and must not be rebuilt via ``parse_instruction``.  Legacy
    ``list[str]`` callers get ``from_ir=False`` and fall back to text parse.
    """

    displays: list[str]
    instructions: list[Instruction | None]
    roles: list[AsmRole]
    from_ir: bool = False
    jump_tables: tuple[JumpTable, ...] = ()
    instruction_ids: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.displays)

    def __getitem__(self, index: int) -> str:
        return self.displays[index]

    def slice(self, end: int) -> "ResolvedAsm":
        return ResolvedAsm(
            self.displays[:end],
            self.instructions[:end],
            self.roles[:end],
            from_ir=self.from_ir,
            jump_tables=self.jump_tables,
            instruction_ids=self.instruction_ids[:end],
        )

    def reorder(self, order: list[int]) -> "ResolvedAsm":
        ids = self.instruction_ids
        return ResolvedAsm(
            [self.displays[i] for i in order],
            [self.instructions[i] for i in order],
            [self.roles[i] for i in order],
            from_ir=self.from_ir,
            jump_tables=self.jump_tables,
            instruction_ids=tuple(ids[i] for i in order) if ids else (),
        )


AsmStream = Union[Sequence[str], Sequence[DecodedInstruction], ResolvedAsm]


def resolve_asm_stream(
    asm: AsmStream, *, jump_tables: Sequence[JumpTable] = ()
) -> ResolvedAsm:
    """Normalize display lines or ``DecodedInstruction`` rows to ``ResolvedAsm``."""
    if isinstance(asm, ResolvedAsm):
        if jump_tables and not asm.jump_tables:
            return ResolvedAsm(
                asm.displays,
                asm.instructions,
                asm.roles,
                from_ir=asm.from_ir,
                jump_tables=tuple(jump_tables),
                instruction_ids=asm.instruction_ids,
            )
        return asm
    tables = tuple(jump_tables)
    if not asm:
        return ResolvedAsm([], [], [], from_ir=False, jump_tables=tables)
    first = asm[0]
    if isinstance(first, DecodedInstruction):
        rows: Sequence[DecodedInstruction] = asm  # type: ignore[assignment]
        return ResolvedAsm(
            displays=[row.display for row in rows],
            instructions=[row.as_effective() if row.is_code else None for row in rows],
            roles=[row.role for row in rows],
            from_ir=True,
            jump_tables=tables,
            instruction_ids=tuple(
                row.instruction_id if row.instruction_id is not None else index
                for index, row in enumerate(rows)
            ),
        )
    lines: Sequence[str] = asm  # type: ignore[assignment]
    return ResolvedAsm(
        displays=list(lines),
        instructions=[None] * len(lines),
        roles=[AsmRole.CODE] * len(lines),
        from_ir=False,
        jump_tables=tables,
        instruction_ids=tuple(range(len(lines))),
    )


def instruction_at(stream: ResolvedAsm, index: int) -> Instruction:
    """Return the structured instruction at ``index``, parsing only as fallback."""
    cached = stream.instructions[index]
    if cached is not None:
        return cached
    return parse_instruction(stream.displays[index])


def is_data_row(stream: ResolvedAsm, index: int) -> bool:
    if stream.from_ir:
        return stream.roles[index] != AsmRole.CODE
    display = stream.displays[index]
    return (
        display.startswith("Jump table:")
        or display.startswith("Data table:")
        or display.startswith("start + ")
        or (display.startswith("0x") and " " not in display)
    )
