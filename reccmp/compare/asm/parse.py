"""Converts x86 machine code into canonical ``DecodedInstruction`` rows.

Capstone detail-mode decode happens once in ``InstructGen``.  This module
sanitizes addresses into symbols/placeholders and materializes structured
operands on each ``DecodedInstruction``.  Display strings remain for UI only.
"""

import re
from functools import cache
from typing_extensions import Buffer
from .const import JUMP_MNEMONICS, SINGLE_OPERAND_INSTS
from .instgen import (
    DisasmLiteTuple,
    FuncSection,
    InstructGen,
    InstructionMeta,
    SectionType,
    meta_from_decoded,
)
from .ir import AsmRole, DecodedInstruction, marker
from .model import Reject, parse_instruction
from .replacement import AddrTestProtocol, NameReplacementProtocol

AsmExcerpt = list[DecodedInstruction]

ptr_replace_regex = re.compile(r"(?<=\[)(0x[0-9a-f]+)(?=\])")

displace_replace_regex = re.compile(r"(?<= )(0x[0-9a-f]+)(?=\])")

# For matching an immediate value operand
immediate_replace_regex = re.compile(r"(?<=, )(0x[0-9a-f]+)")


@cache
def from_hex(string: str) -> int | None:
    try:
        return int(string, 16)
    except ValueError:
        pass

    return None


class ParseAsm:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        addr_test: AddrTestProtocol | None = None,
        name_lookup: NameReplacementProtocol | None = None,
        is_32bit: bool = True,
        collect_meta: bool = False,
    ) -> None:
        self.addr_test = addr_test
        self.name_lookup = name_lookup
        self.is_32bit = is_32bit
        self.collect_meta = collect_meta

        self.replacements: dict[int, str] = {}
        self.indirect_replacements: dict[int, str] = {}
        self.number_placeholders = True
        # Structured facts for the most recent parse_asm() call,
        # keyed by instruction address. Populated from the single decode pass.
        self.meta: dict[int, InstructionMeta] = {}
        self._sections: list[FuncSection] = []
        self._decoded_by_addr: dict[int, DecodedInstruction] = {}

    def reset(self):
        self.replacements = {}
        self.indirect_replacements = {}

    def is_addr(self, value: int) -> bool:
        """Wrapper for user-provided address test"""
        if callable(self.addr_test):
            return self.addr_test(value)

        return False

    def lookup(
        self, addr: int, exact: bool = False, indirect: bool = False
    ) -> str | None:
        """Wrapper for user-provided name lookup"""
        if callable(self.name_lookup):
            return self.name_lookup(addr, exact=exact, indirect=indirect)

        return None

    def _next_placeholder(self) -> str:
        """The placeholder number corresponds to the number of addresses we have
        already replaced. This is so the number will be consistent across the diff
        if we can replace some symbols with actual names in recomp but not orig."""
        number = len(self.replacements) + len(self.indirect_replacements) + 1
        return f"<OFFSET{number}>" if self.number_placeholders else "<OFFSET>"

    def replace(self, addr: int, exact: bool = False) -> str:
        """Provide a replacement name for the given address."""
        if addr in self.replacements:
            return self.replacements[addr]

        if (name := self.lookup(addr, exact=exact)) is not None:
            self.replacements[addr] = name
            return name

        placeholder = self._next_placeholder()
        self.replacements[addr] = placeholder
        return placeholder

    def indirect_replace(self, addr: int) -> str:
        if addr in self.indirect_replacements:
            return self.indirect_replacements[addr]

        if (name := self.lookup(addr, exact=True, indirect=True)) is not None:
            self.indirect_replacements[addr] = name
            return name

        placeholder = self._next_placeholder()
        self.indirect_replacements[addr] = placeholder
        return placeholder

    def hex_replace_always(self, match: re.Match) -> str:
        """If a pointer value was matched, always insert a placeholder"""
        value = int(match.group(1), 16)
        return self.replace(value)

    def hex_replace_relocated(self, match: re.Match) -> str:
        """For replacing immediate value operands. We only want to
        use the placeholder if we are certain that this is a valid address.
        We can check the relocation table to find out."""
        value = int(match.group(1), 16)
        if self.is_addr(value):
            return self.replace(value)

        return match.group(0)

    def hex_replace_annotated(self, match: re.Match) -> str:
        """For replacing immediate value operands. Here we replace the value
        only if the name lookup returns something. Do not use a placeholder."""
        value = int(match.group(1), 16)
        placeholder = self.lookup(value)
        if placeholder is not None:
            return placeholder

        return match.group(0)

    def hex_replace_indirect(self, match: re.Match) -> str:
        """Edge case for hex_replace_always. The context of the instruction
        tells us that the pointer value is an absolute indirect.
        So we go to that location in the binary to get the address.
        If we cannot identify the indirect address, fall back to a lookup
        on the original pointer value so we might display something useful."""
        value = int(match.group(1), 16)
        return self.indirect_replace(value)

    def sanitize(self, inst: DisasmLiteTuple) -> tuple[str, str]:
        # For jumps or calls, if the entire op_str is a hex number, the value
        # is a relative offset.
        # Otherwise (i.e. it looks like `dword ptr [address]`) it is an
        # absolute indirect that we will handle below.
        # Providing the starting address of the function to capstone.disasm has
        # automatically resolved relative offsets to an absolute address.
        # We will have to undo this for some of the jumps or they will not match.
        inst_address, inst_size, inst_mnemonic, inst_op_str = inst

        if (
            inst_mnemonic in SINGLE_OPERAND_INSTS
            and (op_str_address := from_hex(inst_op_str)) is not None
        ):
            if inst_mnemonic == "call":
                return (inst_mnemonic, self.replace(op_str_address, exact=True))

            if inst_mnemonic == "push":
                if self.is_addr(op_str_address):
                    return (inst_mnemonic, self.replace(op_str_address))

                # To avoid falling into jump handling
                return (inst_mnemonic, inst_op_str)

            if inst_mnemonic == "jmp":
                # The unwind section contains JMPs to other functions.
                # If we have a name for this address, use it. If not,
                # do not create a new placeholder. We will instead
                # fall through to generic jump handling below.
                potential_name = self.lookup(op_str_address, exact=True)
                if potential_name is not None:
                    return (inst_mnemonic, potential_name)

            # Else: this is any jump
            # Show the jump offset rather than the absolute address
            jump_displacement = op_str_address - (inst_address + inst_size)
            return (inst_mnemonic, hex(jump_displacement))

        if inst_mnemonic == "call":
            # Special handling for absolute indirect CALL.
            op_str = ptr_replace_regex.sub(self.hex_replace_indirect, inst_op_str)
        else:
            op_str = ptr_replace_regex.sub(self.hex_replace_always, inst_op_str)

            # We only want relocated addresses for pointer displacement.
            # i.e. ptr [register + something]
            # Otherwise we would use a placeholder for every stack variable,
            # vtable call, or this->member access.
            op_str = displace_replace_regex.sub(self.hex_replace_relocated, op_str)

        # In the event of pointer comparison, only replace the immediate value
        # if it is a known address.
        if inst_mnemonic == "cmp":
            op_str = immediate_replace_regex.sub(self.hex_replace_annotated, op_str)
        else:
            op_str = immediate_replace_regex.sub(self.hex_replace_relocated, op_str)

        return (inst_mnemonic, op_str)

    def _finalize_code_row(
        self, lite: DisasmLiteTuple, display: str
    ) -> DecodedInstruction:
        """Attach sanitized display + structured operands onto the decoded row."""
        addr, size, _mnemonic, raw_op = lite
        base = self._decoded_by_addr.get(addr)
        try:
            parsed = parse_instruction(display)
        except Reject:
            if base is not None:
                return base.with_display(display)
            return DecodedInstruction(
                address=addr,
                size=size,
                mnemonic=display.split()[0] if display else "",
                prefix="",
                operands=(),
                raw_operands=(),
                display=display,
                role=AsmRole.CODE,
                raw_op_str=raw_op,
            )
        if base is not None:
            from dataclasses import replace

            return replace(
                base,
                display=display,
                mnemonic=parsed.mnemonic,
                prefix=parsed.prefix,
                operands=parsed.operands,
                raw_operands=parsed.raw_operands,
            )
        return DecodedInstruction(
            address=addr,
            size=size,
            mnemonic=parsed.mnemonic,
            prefix=parsed.prefix,
            operands=parsed.operands,
            raw_operands=parsed.raw_operands,
            display=display,
            role=AsmRole.CODE,
            raw_op_str=raw_op,
        )

    def parse_asm(self, data: Buffer, start_addr: int) -> AsmExcerpt:
        self.reset()
        asm: AsmExcerpt = []

        ig = InstructGen(bytes(data), start_addr, self.is_32bit)
        self._sections = ig.sections
        self._decoded_by_addr = dict(ig.decoded_by_addr)

        # Project meta from the single decode pass (no second Capstone walk).
        self.meta = {
            addr: meta_from_decoded(insn)
            for addr, insn in self._decoded_by_addr.items()
        }

        for section in ig.sections:
            if section.type == SectionType.CODE:
                for inst in section.contents:
                    inst_address, inst_size, inst_mnemonic, inst_op_str = inst
                    if "0x" in inst_op_str and (
                        inst_mnemonic in JUMP_MNEMONICS
                        or inst_size > 4
                        or not self.is_32bit
                    ):
                        result = self.sanitize(inst)
                    else:
                        result = (inst_mnemonic, inst_op_str)

                    display = " ".join(result)
                    asm.append(self._finalize_code_row(inst, display))
            elif section.type == SectionType.ADDR_TAB:
                asm.append(marker("Jump table:", role=AsmRole.JUMP_TABLE_HEADER))
                for ofs, target in section.contents:
                    target_relative_to_function_start = target - start_addr
                    asm.append(
                        marker(
                            f"start + 0x{(target_relative_to_function_start):x}",
                            address=ofs,
                            role=AsmRole.JUMP_TABLE_ENTRY,
                        )
                    )

            elif section.type == SectionType.DATA_TAB:
                asm.append(marker("Data table:", role=AsmRole.DATA_TABLE_HEADER))
                for ofs, b in section.contents:
                    asm.append(
                        marker(hex(b), address=ofs, role=AsmRole.DATA_TABLE_ENTRY)
                    )

        return asm

    def collect_instruction_meta(
        self, data: Buffer, start_addr: int
    ) -> dict[int, InstructionMeta]:
        """Return meta from the most recent ``parse_asm`` decode, or decode now.

        The hot path already populated ``self.meta`` during ``parse_asm``.
        This method remains for callers that only need meta without a full
        sanitize pass.
        """
        if self.meta and self._decoded_by_addr:
            return self.meta
        ig = InstructGen(bytes(data), start_addr, self.is_32bit)
        self._sections = ig.sections
        self._decoded_by_addr = dict(ig.decoded_by_addr)
        self.meta = {
            addr: meta_from_decoded(insn)
            for addr, insn in self._decoded_by_addr.items()
        }
        return self.meta
