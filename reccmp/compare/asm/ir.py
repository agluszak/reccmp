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
    Reject,
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


@dataclass(frozen=True)
class JumpTable:
    """One switch address table discovered by ``InstructGen``.

    ``entries`` are ``(entry_address, target_address)`` pairs. Optional
    ``dispatch_address`` is the ``jmp`` that indexes the table when known.
    """

    address: int
    entries: tuple[tuple[int, int], ...]
    dispatch_address: int | None = None


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

    @property
    def is_code(self) -> bool:
        return self.role == AsmRole.CODE

    def as_effective(self):
        """View used by the effective-match verifier."""
        return Instruction(self.mnemonic, self.prefix, self.operands, self.raw_operands)

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
        )
    return DecodedInstruction(**kwargs)


def _freeze(value) -> Hashable:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def instruction_match_key(row: DecodedInstruction | str) -> Hashable:
    """Hashable SequenceMatcher key from IR (or legacy display string)."""
    if isinstance(row, str):
        try:
            ins = parse_instruction(row)
        except Reject:
            return ("raw", row)
        return ("ins", ins.mnemonic, ins.prefix, _freeze(ins.operands))
    if row.role != AsmRole.CODE:
        return ("raw", row.display)
    return ("ins", row.mnemonic, row.prefix, _freeze(row.operands))


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
        )

    def reorder(self, order: list[int]) -> "ResolvedAsm":
        return ResolvedAsm(
            [self.displays[i] for i in order],
            [self.instructions[i] for i in order],
            [self.roles[i] for i in order],
            from_ir=self.from_ir,
            jump_tables=self.jump_tables,
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
        )
    lines: Sequence[str] = asm  # type: ignore[assignment]
    return ResolvedAsm(
        displays=list(lines),
        instructions=[None] * len(lines),
        roles=[AsmRole.CODE] * len(lines),
        from_ir=False,
        jump_tables=tables,
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
