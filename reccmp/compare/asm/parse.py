"""Converts x86 machine code into canonical ``DecodedInstruction`` rows.

Capstone detail-mode decode and section discovery are local to this module. It
sanitizes addresses into references (a name or placeholder to show, and
the identity proofs compare) on the structured operands, and renders each
row's display from them. Nothing here reads a display back.
"""

from __future__ import annotations

import bisect
import struct
from dataclasses import replace
from enum import Enum, auto
from typing import Literal, NamedTuple
from collections.abc import Callable, Hashable
from typing_extensions import Buffer

from reccmp.types import ImageId

from .decode import disasm_detail
from .graph import build_function_graph
from .ir import (
    DataRegion,
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
    rebind_local_identities,
)
from .model import Reference, ResolvedAddress
from .operand import format_instruction, Imm, Mem, Operand, ScaledReg, SignedSymbol, Sym
from .replacement import AddrTestProtocol, ReferenceResolver


class _SectionType(Enum):
    CODE = auto()
    DATA_TAB = auto()
    ADDR_TAB = auto()


class _CodeSection(NamedTuple):
    type: Literal[_SectionType.CODE]
    contents: list[DecodedInstruction]


_TabSectionType = Literal[_SectionType.DATA_TAB] | Literal[_SectionType.ADDR_TAB]


class _TabSection(NamedTuple):
    type: _TabSectionType
    contents: list[tuple[int, int]]


_FuncSection = _CodeSection | _TabSection


def switch_index_register(insn: DecodedInstruction, table_addr: int) -> str | None:
    """The index register of ``jmp dword ptr [index*4 + table_addr]``.

    This is the only dispatch form a switch table is recognized for: one
    index term at scale 4, no base, no segment, no symbol, and the table's
    address as displacement. Any other register in the address would take
    part in every target, so the table would not describe the jump.
    """
    if insn.mnemonic != "jmp" or insn.prefix:
        return None
    match insn.operands:
        case (Mem("dword", "", (ScaledReg(index, 4),), displacement, ()),) if (
            displacement == table_addr
        ):
            return index
    return None


def _table_displacement(insn: DecodedInstruction) -> int | None:
    """The displacement of an indexed memory operand, ``[reg*4 + table]``:
    where a jump or a load reads a table of addresses or bytes."""
    for operand in insn.operands:
        match operand:
            case Mem(terms=terms, displacement=displacement) if (
                terms and displacement > 0
            ):
                return displacement
    return None


class _SectionDiscovery:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, blob: bytes, start: int, is_32bit: bool = True) -> None:
        self.is_32bit = is_32bit
        self.blob = blob
        self.start = start
        self.end = len(blob) + start
        self.section_end: int = self.end
        self.code_tracks: list[list[DecodedInstruction]] = []
        # Canonical IR from the same Capstone detail pass as code_tracks.
        self.decoded_by_addr: dict[int, DecodedInstruction] = {}

        self.cur_addr: int = 0
        self.cur_section_type: _SectionType = _SectionType.CODE
        self.section_start = start

        self.sections: list[_FuncSection] = []

        self.confirmed_addrs: dict[int, _SectionType] = {}
        self.jump_tables: list[JumpTable] = []
        self.coverage_incomplete: bool = False
        self.analysis()

    def _finish_code_section(self, contents: list[DecodedInstruction]):
        self.sections.append(_CodeSection(_SectionType.CODE, contents))

    def _finish_tab_section(self, type_: _TabSectionType, stuff: list[tuple[int, int]]):
        self.sections.append(_TabSection(type_, stuff))
        if type_ == _SectionType.ADDR_TAB and stuff:
            table_addr = stuff[0][0]
            dispatch, index_reg = self._dispatch_for_table(table_addr)
            self.jump_tables.append(
                JumpTable(
                    address=table_addr,
                    entries=tuple(stuff),
                    dispatch_address=dispatch,
                    index_register=index_reg,
                )
            )

    def _dispatch_for_table(self, table_addr: int) -> tuple[int | None, str | None]:
        """The ``jmp`` that indexes the table at ``table_addr``, and its index."""
        for addr, insn in self.decoded_by_addr.items():
            if (index := switch_index_register(insn, table_addr)) is not None:
                return addr, index
        return None, None

    def _insert_confirmed_addr(self, addr: int, type_: _SectionType):
        # Ignore address outside the bounds of the function
        if not self.start <= addr < self.end:
            return

        self.confirmed_addrs[addr] = type_

        # This newly inserted address might signal the end of this section.
        # For example, a jump table at the end of the function means we should
        # stop reading instructions once we hit that address.
        # However, if there is a jump table in between code sections, we might
        # read a jump to an address back to the beginning of the function
        # (e.g. a loop that spans the entire function)
        # so ignore this address because we have already passed it.
        if type_ != self.cur_section_type and addr > self.cur_addr:
            self.section_end = min(self.section_end, addr)

    def _next_section(self, addr: int) -> _SectionType | None:
        """We have reached the start of a new section. Tell what kind of
        data we are looking at (code or other) and how much we should read."""

        # Assume the start of every function is code.
        if addr == self.start:
            self.section_end = self.end
            return _SectionType.CODE

        # The start of a new section must be an address that we've seen.
        new_type = self.confirmed_addrs.get(addr)
        if new_type is None:
            return None

        self.cur_section_type = new_type

        # The confirmed addrs dict is sorted by insertion order
        # i.e. the order in which we read the addresses
        # So we have to sort and then find the next item
        # to see where this section should end.

        # If we are in a CODE section, ignore contiguous CODE addresses.
        # These are not the start of a new section.
        # However: if we are not in CODE, any upcoming address is a new section.
        # Do this so we can detect contiguous non-CODE sections.
        confirmed = [
            conf_addr
            for (conf_addr, conf_type) in sorted(self.confirmed_addrs.items())
            if self.cur_section_type != _SectionType.CODE
            or conf_type != self.cur_section_type
        ]

        index = bisect.bisect_right(confirmed, addr)
        if index < len(confirmed):
            self.section_end = confirmed[index]
        else:
            self.section_end = self.end

        return new_type

    def _get_code_for(self, addr: int) -> list[DecodedInstruction]:
        """Start disassembling at the given address (single Capstone detail pass)."""
        # If we are reading a code block beyond the first, see if we already
        # have disassembled instructions beginning at the specified address.
        for track in self.code_tracks:
            for i, inst in enumerate(track):
                if inst.address == addr:
                    return track[i:]

        blob_cropped = self.blob[addr - self.start :]
        decoded = disasm_detail(blob_cropped, addr, self.is_32bit)
        for insn in decoded:
            assert insn.address is not None
            self.decoded_by_addr[insn.address] = insn
        self.code_tracks.append(decoded)
        return decoded

    def _handle_jump(self, insn: DecodedInstruction):
        # A direct jump inside the function's bytes starts code there.
        if insn.branch_target is not None:
            self._insert_confirmed_addr(insn.branch_target, _SectionType.CODE)
        # An indexed jump reads a table of addresses there.
        elif (table := _table_displacement(insn)) is not None:
            self._insert_confirmed_addr(table, _SectionType.ADDR_TAB)

    def analysis(self):
        self.cur_addr = self.start
        self.coverage_incomplete = False
        visited_code: set[int] = set()

        while True:
            sect_type = self._next_section(self.cur_addr)
            if sect_type is None:
                # Drain pending confirmed addresses beyond the current cursor.
                # A forward jmp can end a CODE section at its target while
                # leaving the cursor on skipped padding (e.g. int3); the
                # target must still be visited.
                pending = sorted(
                    addr
                    for addr, kind in self.confirmed_addrs.items()
                    if addr >= self.cur_addr
                    and kind == _SectionType.CODE
                    and addr not in visited_code
                )
                if not pending:
                    pending = sorted(
                        addr
                        for addr in self.confirmed_addrs
                        if addr >= self.cur_addr and addr not in visited_code
                    )
                if not pending:
                    break
                self.cur_addr = pending[0]
                continue

            self.section_start = self.cur_addr

            if sect_type == _SectionType.CODE:
                visited_code.add(self.cur_addr)
                instructions = self._get_code_for(self.cur_addr)

                # If we didn't get any instructions back, something is wrong.
                # i.e. We can only read part of the full instruction that is up next.
                if len(instructions) == 0:
                    self.coverage_incomplete = True
                    # Nudge the current addr so we will eventually move on to the
                    # next section.
                    self.cur_addr += 1
                    continue

                for insn in instructions:
                    # section_end is updated as we read instructions.
                    # If we are into a jump/data table and would read
                    # a junk instruction, stop here.
                    if self.cur_addr >= self.section_end:
                        break

                    if insn.is_jump:
                        self._handle_jump(insn)
                    elif insn.mnemonic in ("mov", "movzx"):
                        # An indexed load reads a table of bytes there.
                        if (table := _table_displacement(insn)) is not None:
                            self._insert_confirmed_addr(table, _SectionType.DATA_TAB)

                    self.cur_addr += insn.size

                instruction_slice = [
                    inst
                    for inst in instructions
                    if inst.address is not None and inst.address < self.section_end
                ]
                self._finish_code_section(instruction_slice)

            elif sect_type == _SectionType.ADDR_TAB:
                # Clamp to multiple of 4 (dwords)
                read_size = ((self.section_end - self.cur_addr) // 4) * 4
                offsets = range(self.section_start, self.section_start + read_size, 4)
                dwords = self.blob[
                    self.cur_addr - self.start : self.cur_addr - self.start + read_size
                ]
                addrs: list[int] = [addr for addr, in struct.iter_unpack("<L", dwords)]
                for addr in addrs:
                    self._insert_confirmed_addr(addr, _SectionType.CODE)

                jump_table = list(zip(offsets, addrs))
                self._finish_tab_section(_SectionType.ADDR_TAB, jump_table)
                self.cur_addr = self.section_end

            else:
                read_size = self.section_end - self.cur_addr
                offsets = range(self.section_start, self.section_start + read_size)
                bytes_ = self.blob[
                    self.cur_addr - self.start : self.cur_addr - self.start + read_size
                ]
                data = [b for b, in struct.iter_unpack("<B", bytes_)]

                data_table = list(zip(offsets, data))
                self._finish_tab_section(_SectionType.DATA_TAB, data_table)
                self.cur_addr = self.section_end

        # Any confirmed CODE address never visited means incomplete coverage.
        for addr, kind in self.confirmed_addrs.items():
            if kind == _SectionType.CODE and addr not in visited_code:
                # Visited if any finished CODE section contains this address.
                if not any(
                    section.type == _SectionType.CODE
                    and any(inst.address == addr for inst in section.contents)
                    for section in self.sections
                ):
                    self.coverage_incomplete = True
                    break


class AddressSanitizer:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        addr_test: AddrTestProtocol | None = None,
        resolver: ReferenceResolver | None = None,
        is_32bit: bool = True,
        image_id: ImageId | None = None,
        body_bounds: tuple[int, int] | None = None,
    ) -> None:
        self.addr_test = addr_test
        self.resolver = resolver
        self.is_32bit = is_32bit
        self.image_id = image_id

        # Address -> name shown for it, so one address keeps one name (and
        # placeholders are numbered by first use) within a function.
        self.replacements: dict[int, str] = {}
        self.indirect_replacements: dict[int, str] = {}
        self.number_placeholders = True
        self._body_start: int | None = body_bounds[0] if body_bounds else None
        self._body_end: int | None = body_bounds[1] if body_bounds else None

    def is_addr(self, value: int) -> bool:
        """Whether the image says ``value`` is an address (a relocation)."""
        return self.addr_test(value) if self.addr_test is not None else False

    def resolve(
        self, addr: int, exact: bool = False, indirect: bool = False
    ) -> ResolvedAddress | None:
        if self.resolver is None:
            return None
        return self.resolver(addr, exact=exact, indirect=indirect)

    def _next_placeholder(self) -> str:
        """The placeholder number corresponds to the number of addresses we have
        already replaced. This is so the number will be consistent across the diff
        if we can replace some symbols with actual names in recomp but not orig."""
        number = len(self.replacements) + len(self.indirect_replacements) + 1
        return f"<OFFSET{number}>" if self.number_placeholders else "<OFFSET>"

    def _side_name(self) -> str:
        if self.image_id is None:
            return "unknown"
        return self.image_id.name.lower()

    def _local_identity(self, addr: int) -> Hashable:
        """The identity of an address nothing names: a byte of this body,
        or an address private to this image."""
        start, end = self._body_start, self._body_end
        if start is not None and end is not None and start <= addr < end:
            return ("local", addr - start)
        return ("unresolved", self._side_name(), addr)

    def reference(
        self, addr: int, exact: bool = False, indirect: bool = False
    ) -> Reference:
        """The reference an address operand becomes: its entity's name and
        identity, or a placeholder and a local identity."""
        resolved = self.resolve(addr, exact=exact, indirect=indirect)
        names = self.indirect_replacements if indirect else self.replacements
        if addr not in names:
            name = resolved.name if resolved is not None else None
            names[addr] = name if name is not None else self._next_placeholder()
        identity = (
            resolved.identity if resolved is not None else self._local_identity(addr)
        )
        return Reference(
            names[addr],
            identity,
            resolved.entity_type if resolved is not None else None,
        )

    def named_reference(self, addr: int, exact: bool = False) -> Reference | None:
        """The reference of an address with a name; None without one (no
        placeholder is made)."""
        resolved = self.resolve(addr, exact=exact)
        if resolved is None or resolved.name is None:
            return None
        return Reference(resolved.name, resolved.identity, resolved.entity_type)

    def control_identity(self, addr: int) -> Hashable:
        """Proof identity of a jump destination without creating a placeholder."""
        resolved = self.resolve(addr, exact=True)
        return resolved.identity if resolved is not None else self._local_identity(addr)

    def _sanitize_mem_operand(self, operand: Mem, *, indirect: bool) -> Mem:
        """An address in a memory operand becomes a reference. An absolute
        operand is an address by its syntax: one nothing names gets this
        side's unresolved identity, never a numeric key the other side could
        share. A displacement is one only when the image relocates it. A
        segment-relative operand (``fs:[0]``) never is."""
        match operand:
            case Mem(segment="", terms=(), displacement=disp, symbols=()):
                return replace(
                    operand,
                    displacement=0,
                    symbols=(SignedSymbol(1, self.reference(disp, indirect=indirect)),),
                )
            case Mem(
                segment="", displacement=disp, symbols=()
            ) if disp and self.is_addr(abs(disp)):
                return replace(
                    operand,
                    displacement=0,
                    symbols=(
                        SignedSymbol(1 if disp >= 0 else -1, self.reference(abs(disp))),
                    ),
                )
        return operand

    def _sanitize_imm_operand(self, mnemonic: str, operand: Imm) -> Operand:
        """An immediate the image says is an address becomes a reference;
        one a `cmp` compares only when it names an entity."""
        if not self.is_addr(operand.value):
            return operand
        if mnemonic == "cmp":
            named = self.named_reference(operand.value)
            return Sym(named) if named is not None else operand
        return Sym(self.reference(operand.value))

    def _direct_transfer(
        self, insn: DecodedInstruction
    ) -> tuple[Operand, Hashable, int | None]:
        """(operand, control target, displacement shown for a local jump) of
        a direct call or jump."""
        assert insn.address is not None and insn.branch_target is not None
        target = insn.branch_target
        if insn.is_call:
            ref = self.reference(target, exact=True)
            return Sym(ref), ref.identity, None
        if not insn.is_conditional:
            # The unwind section jumps to other functions: name the target
            # when it has a name.
            named = self.named_reference(target, exact=True)
            if named is not None:
                return Sym(named), named.identity, None
        # A local jump shows its displacement, not its absolute target.
        displacement = target - (insn.address + insn.size)
        return Imm(displacement), self.control_identity(target), displacement

    def sanitize_row(self, insn: DecodedInstruction) -> DecodedInstruction:
        """Replace address operands by references; render the display."""
        assert insn.address is not None
        mnemonic = insn.mnemonic
        operands = list(insn.operands)
        control_target = None
        displacement = None
        if insn.branch_target is not None and (insn.is_call or insn.is_jump):
            operands[0], control_target, displacement = self._direct_transfer(insn)
        else:
            for i, op in enumerate(operands):
                match op:
                    case Mem() if insn.is_call:
                        # Absolute indirect only; leave [reg+disp] alone.
                        if not op.terms:
                            operands[i] = self._sanitize_mem_operand(op, indirect=True)
                    case Mem():
                        operands[i] = self._sanitize_mem_operand(op, indirect=False)
                    case Imm(value) if mnemonic == "push" and len(operands) == 1:
                        if self.is_addr(value):
                            operands[i] = Sym(self.reference(value))
                    case Imm():
                        operands[i] = self._sanitize_imm_operand(mnemonic, op)

        ops_tuple = tuple(operands)
        if displacement is not None:
            head = f"{insn.prefix} {mnemonic}".strip() if insn.prefix else mnemonic
            display = f"{head} {hex(displacement)}"
        elif ops_tuple == insn.operands:
            display = insn.display
        else:
            display = format_instruction(mnemonic, insn.prefix, ops_tuple)
        return replace(
            insn, operands=ops_tuple, display=display, control_target=control_target
        )


def decode_function(
    data: Buffer,
    start_addr: int,
    *,
    extent_kind: ExtentKind = ExtentKind.KNOWN,
    addr_test: AddrTestProtocol | None = None,
    resolver: ReferenceResolver | None = None,
    is_32bit: bool = True,
    image_id: ImageId | None = None,
) -> FunctionImage:
    # pylint: disable=too-many-arguments
    """Decode one byte window into instructions and separate embedded data.

    All discovery and sanitization state is local to this call. The returned
    image is the only value subsequent analysis needs.
    """
    blob = bytes(data)
    sanitizer = AddressSanitizer(
        addr_test, resolver, is_32bit, image_id, (start_addr, start_addr + len(blob))
    )
    sections = _SectionDiscovery(blob, start_addr, is_32bit)
    instructions: list[DecodedInstruction] = []
    data_regions: list[DataRegion] = []
    for section in sections.sections:
        if section.type == _SectionType.CODE:
            instructions.extend(sanitizer.sanitize_row(row) for row in section.contents)
        elif section.type == _SectionType.DATA_TAB and section.contents:
            data_regions.append(
                DataRegion(
                    section.contents[0][0],
                    bytes(value for _address, value in section.contents),
                )
            )
    tables = tuple(sections.jump_tables)
    stamped = rebind_local_identities(
        instructions,
        start_addr=start_addr,
        extent=len(blob),
        jump_tables=tables,
        image_id=image_id.name.lower() if image_id is not None else "unknown",
    )
    graph = build_function_graph(
        stamped, tables, start_addr=start_addr, extent=len(blob)
    )
    return FunctionImage(
        start_addr=start_addr,
        extent=len(blob),
        extent_kind=extent_kind,
        instructions=stamped,
        jump_tables=tables,
        coverage_incomplete=sections.coverage_incomplete,
        extent_closed=graph.extent_closed(
            coverage_incomplete=sections.coverage_incomplete,
            extent_kind=extent_kind,
        ),
        raw=blob,
        data_regions=tuple(data_regions),
        graph=graph,
    )


# The operands `assert` receives in its line and file arguments: the macros,
# not this build's numbers.
_ASSERT_LINE = Sym(Reference("__LINE__", ("assert_macro", "__LINE__")))
_ASSERT_FILE = Sym(Reference("__FILE__", ("assert_macro", "__FILE__")))


def _with_operand(row: DecodedInstruction, operand: Operand) -> DecodedInstruction:
    return replace(
        row,
        operands=(operand,),
        display=format_instruction(row.mnemonic, row.prefix, (operand,)),
    )


def assert_fixup(asm: list[DecodedInstruction], is_assert: Callable[[int], bool]):
    """Replace the line and file arguments of each call ``is_assert`` says
    reaches the CRT ``_assert`` with the macros (from assert.h)."""
    for i, row in enumerate(asm):
        if (
            i >= 3
            and row.is_call
            and row.branch_target is not None
            and is_assert(row.branch_target)
        ):
            asm[i - 3] = _with_operand(asm[i - 3], _ASSERT_LINE)
            asm[i - 2] = _with_operand(asm[i - 2], _ASSERT_FILE)
