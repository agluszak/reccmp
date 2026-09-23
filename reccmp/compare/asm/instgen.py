"""Pre-parser for x86 instructions. Will identify data/jump tables used with
switch statements and local jump/call destinations."""

import re
import bisect
import struct
from dataclasses import dataclass
from functools import cache
from enum import Enum, auto
from collections.abc import Hashable
from typing import Iterable, Literal, NamedTuple
from capstone import (  # type: ignore
    CS_ARCH_X86,
    CS_MODE_16,
    CS_MODE_32,
    Cs,
)
from .const import JUMP_MNEMONICS
from .decode import as_lite_tuple, disasm_detail
from .ir import DecodedInstruction, JumpTable


@cache
def get_disassembler(is_32: bool = True) -> Cs:
    return Cs(CS_ARCH_X86, CS_MODE_32 if is_32 else CS_MODE_16)


DisasmLiteTuple = tuple[int, int, str, str]
"""Raw tuple returned by capstone's disasm_lite() function.
The fields are:
    - address
    - size (of instruction, in bytes)
    - mnemonic
    - op_str (all operands, comma-delimited)
"""

displacement_regex = re.compile(r".*\+ (0x[0-9a-f]+)\]")


class SectionType(Enum):
    CODE = auto()
    DATA_TAB = auto()
    ADDR_TAB = auto()


class CodeSection(NamedTuple):
    type: Literal[SectionType.CODE]
    contents: list[DisasmLiteTuple]


TabSectionType = Literal[SectionType.DATA_TAB] | Literal[SectionType.ADDR_TAB]


class TabSection(NamedTuple):
    type: TabSectionType
    contents: list[tuple[int, int]]


FuncSection = CodeSection | TabSection


def stop_at_int3(
    disasm_lite_gen: Iterable[DisasmLiteTuple],
) -> Iterable[DisasmLiteTuple]:
    """Wrapper for capstone disasm_lite generator. We want to stop reading
    instructions if we hit the int3 instruction."""
    for inst in disasm_lite_gen:
        # inst[2] is the mnemonic
        if inst[2] == "int3":
            break

        yield inst


class InstructGen:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, blob: bytes, start: int, is_32bit: bool = True) -> None:
        self.is_32bit = is_32bit
        self.blob = blob
        self.start = start
        self.end = len(blob) + start
        self.section_end: int = self.end
        self.code_tracks: list[list[DisasmLiteTuple]] = []
        # Canonical IR from the same Capstone detail pass as code_tracks.
        self.decoded_by_addr: dict[int, DecodedInstruction] = {}

        # Todo: Could be refactored later
        self.cur_addr: int = 0
        self.cur_section_type: SectionType = SectionType.CODE
        self.section_start = start

        self.sections: list[FuncSection] = []

        self.confirmed_addrs: dict[int, SectionType] = {}
        self.jump_tables: list[JumpTable] = []
        self.coverage_incomplete: bool = False
        self.analysis()

    def _finish_code_section(self, contents: list[DisasmLiteTuple]):
        self.sections.append(CodeSection(SectionType.CODE, contents))

    def _finish_tab_section(self, type_: TabSectionType, stuff: list[tuple[int, int]]):
        self.sections.append(TabSection(type_, stuff))
        if type_ == SectionType.ADDR_TAB and stuff:
            table_addr = stuff[0][0]
            dispatch, index_reg = self._dispatch_for_table(table_addr)
            self.jump_tables.append(
                JumpTable(
                    address=table_addr,
                    entries=tuple(stuff),
                    dispatch_address=dispatch,
                    scale=4,
                    entry_width=4,
                    index_register=index_reg,
                )
            )

    def _dispatch_for_table(self, table_addr: int) -> tuple[int | None, str | None]:
        """Find a scale-4 ``jmp dword ptr [idx*4 + table]`` at ``table_addr``."""
        for addr, insn in self.decoded_by_addr.items():
            if not insn.is_jump or insn.mnemonic != "jmp":
                continue
            for op in insn.operands:
                if not isinstance(op, tuple) or op[0] != "mem":
                    continue
                _size, _seg, reg_terms, disp, _syms = (
                    op[1],
                    op[2],
                    op[3],
                    op[4],
                    op[5],
                )
                index_reg = next((reg for reg, scale in reg_terms if scale == 4), None)
                if index_reg is None or disp != table_addr:
                    continue
                return addr, index_reg
        return None, None

    def _insert_confirmed_addr(self, addr: int, type_: SectionType):
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

    def _next_section(self, addr: int) -> SectionType | None:
        """We have reached the start of a new section. Tell what kind of
        data we are looking at (code or other) and how much we should read."""

        # Assume the start of every function is code.
        if addr == self.start:
            self.section_end = self.end
            return SectionType.CODE

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
            if self.cur_section_type != SectionType.CODE
            or conf_type != self.cur_section_type
        ]

        index = bisect.bisect_right(confirmed, addr)
        if index < len(confirmed):
            self.section_end = confirmed[index]
        else:
            self.section_end = self.end

        return new_type

    def _get_code_for(self, addr: int) -> list[DisasmLiteTuple]:
        """Start disassembling at the given address (single Capstone detail pass)."""
        # If we are reading a code block beyond the first, see if we already
        # have disassembled instructions beginning at the specified address.
        for track in self.code_tracks:
            for i, inst in enumerate(track):
                if inst[0] == addr:
                    return track[i:]

        blob_cropped = self.blob[addr - self.start :]
        decoded = disasm_detail(blob_cropped, addr, self.is_32bit)
        for insn in decoded:
            assert insn.address is not None
            self.decoded_by_addr[insn.address] = insn
        instructions = [as_lite_tuple(insn) for insn in decoded]
        self.code_tracks.append(instructions)
        return instructions

    def _handle_jump(self, op_str: str):
        # If this is a regular jump and its destination is within the
        # bounds of the binary data (i.e. presumed function size)
        # add it to our list of confirmed addresses.
        if op_str[0] == "0":
            value = int(op_str, 16)
            self._insert_confirmed_addr(value, SectionType.CODE)

        # If this is jumping into a table of addresses, save the destination
        elif (match := displacement_regex.match(op_str)) is not None:
            value = int(match.group(1), 16)
            self._insert_confirmed_addr(value, SectionType.ADDR_TAB)

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
                    and kind == SectionType.CODE
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

            if sect_type == SectionType.CODE:
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

                for _, inst_size, inst_mnemonic, inst_op_str in instructions:
                    # section_end is updated as we read instructions.
                    # If we are into a jump/data table and would read
                    # a junk instruction, stop here.
                    if self.cur_addr >= self.section_end:
                        break

                    if inst_mnemonic in JUMP_MNEMONICS:
                        self._handle_jump(inst_op_str)
                    elif inst_mnemonic in ("mov", "movzx"):
                        if (match := displacement_regex.match(inst_op_str)) is not None:
                            value = int(match.group(1), 16)
                            self._insert_confirmed_addr(value, SectionType.DATA_TAB)

                    self.cur_addr += inst_size

                instruction_slice = [
                    inst for inst in instructions if inst[0] < self.section_end
                ]
                self._finish_code_section(instruction_slice)

            elif sect_type == SectionType.ADDR_TAB:
                # Clamp to multiple of 4 (dwords)
                read_size = ((self.section_end - self.cur_addr) // 4) * 4
                offsets = range(self.section_start, self.section_start + read_size, 4)
                dwords = self.blob[
                    self.cur_addr - self.start : self.cur_addr - self.start + read_size
                ]
                addrs: list[int] = [addr for addr, in struct.iter_unpack("<L", dwords)]
                for addr in addrs:
                    self._insert_confirmed_addr(addr, SectionType.CODE)

                jump_table = list(zip(offsets, addrs))
                self._finish_tab_section(SectionType.ADDR_TAB, jump_table)
                self.cur_addr = self.section_end

            else:
                read_size = self.section_end - self.cur_addr
                offsets = range(self.section_start, self.section_start + read_size)
                bytes_ = self.blob[
                    self.cur_addr - self.start : self.cur_addr - self.start + read_size
                ]
                data = [b for b, in struct.iter_unpack("<B", bytes_)]

                data_table = list(zip(offsets, data))
                self._finish_tab_section(SectionType.DATA_TAB, data_table)
                self.cur_addr = self.section_end

        # Any confirmed CODE address never visited means incomplete coverage.
        for addr, kind in self.confirmed_addrs.items():
            if kind == SectionType.CODE and addr not in visited_code:
                # Visited if any finished CODE section contains this address.
                if not any(
                    section.type == SectionType.CODE
                    and any(inst[0] == addr for inst in section.contents)
                    for section in self.sections
                ):
                    self.coverage_incomplete = True
                    break


@dataclass(frozen=True)
class InstructionMeta:
    """Structured facts about one instruction, captured from capstone's
    detail mode at disassembly time: register accesses including implicit
    ones, flags effects, memory access, control-flow class and the branch
    target. Consumed privately by the effective-match verifier.

    Prefer reading these fields from ``DecodedInstruction`` directly; this
    dataclass remains as a projection for callers that still pass parallel
    meta lists into the verifier.
    """

    # pylint: disable=too-many-instance-attributes

    address: int
    size: int
    mnemonic: str
    regs_read: tuple[str, ...]
    regs_written: tuple[str, ...]
    reads_flags: bool
    writes_flags: bool
    accesses_memory: bool
    is_jump: bool
    is_call: bool
    is_ret: bool
    branch_target: int | None
    register_access_known: bool = True
    operand_model_complete: bool = True
    control_flow_known: bool = True
    control_target: Hashable | None = None


def meta_from_decoded(insn: DecodedInstruction) -> InstructionMeta:
    assert insn.address is not None
    return InstructionMeta(
        address=insn.address,
        size=insn.size,
        mnemonic=insn.mnemonic,
        regs_read=insn.regs_read,
        regs_written=insn.regs_written,
        reads_flags=insn.reads_flags,
        writes_flags=insn.writes_flags,
        accesses_memory=insn.accesses_memory,
        is_jump=insn.is_jump,
        is_call=insn.is_call,
        is_ret=insn.is_ret,
        branch_target=insn.branch_target,
        register_access_known=insn.register_access_known,
        operand_model_complete=insn.operand_model_complete,
        control_flow_known=insn.control_flow_known,
        control_target=insn.control_target,
    )


def collect_instruction_meta(
    blob: bytes, start: int, sections: list[FuncSection], is_32bit: bool = True
) -> dict[int, InstructionMeta]:
    """Project CODE-section IR into the legacy meta map via ``InstructGen``.

    Honours embedded jump/data tables the same way as ``parse_asm``.
    """
    del sections  # InstructGen rediscovers section bounds from the blob.
    ig = InstructGen(blob, start, is_32bit)
    return {
        addr: meta_from_decoded(insn)
        for addr, insn in ig.decoded_by_addr.items()
        if insn.address is not None
    }
