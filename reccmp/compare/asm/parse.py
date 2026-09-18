"""Converts x86 machine code into canonical ``DecodedInstruction`` rows.

Capstone detail-mode decode happens once in ``InstructGen``.  This module
sanitizes addresses into symbols/placeholders on structured operands and
refreshes the display string from that form.  ``parse_instruction`` is only
a fallback when Capstone IR is missing or a line cannot be handled
structurally.
"""

from __future__ import annotations

import re
from dataclasses import replace
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
from .model import Reject, format_instruction, format_operand, parse_instruction
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


def _hex_style_addr(value: int) -> bool:
    """True when Capstone-style text would emit a ``0x...`` token for ``value``."""
    return abs(value) >= 10


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

    def _sanitize_mem_operand(self, operand, *, indirect: bool):
        """Apply absolute/displacement address replacement to a mem operand."""
        size, seg, reg_terms, disp, syms = (
            operand[1],
            operand[2],
            operand[3],
            operand[4],
            operand[5],
        )
        if syms:
            return operand

        # Absolute pointer: ``[0x1234]`` (hex-style only; ``[8]`` is left alone).
        if not reg_terms:
            if _hex_style_addr(disp):
                name = self.indirect_replace(disp) if indirect else self.replace(disp)
                return ("mem", size, seg, [], 0, ((1, name),))
            return operand

        # Register + displacement: only replace relocated addresses.
        if _hex_style_addr(disp) and self.is_addr(abs(disp)):
            value = abs(disp)
            sign = 1 if disp >= 0 else -1
            name = self.replace(value)
            return ("mem", size, seg, list(reg_terms), 0, ((sign, name),))

        return operand

    def _sanitize_imm_operand(self, mnemonic: str, operand):
        value = operand[1]
        if not _hex_style_addr(value):
            return operand
        if mnemonic == "cmp":
            name = self.lookup(value)
            if name is not None:
                return ("sym", name)
            return operand
        if self.is_addr(value):
            return ("sym", self.replace(value))
        return operand

    def sanitize_row(self, insn: DecodedInstruction) -> DecodedInstruction:
        """Transform Capstone operands structurally; refresh display from them."""
        assert insn.address is not None
        mnemonic = insn.mnemonic
        operands = list(insn.operands)
        jump_disp_hex = False

        if (
            mnemonic in SINGLE_OPERAND_INSTS
            and len(operands) == 1
            and operands[0][0] == "imm"
        ):
            addr_val = operands[0][1]
            if mnemonic == "call":
                operands[0] = ("sym", self.replace(addr_val, exact=True))
            elif mnemonic == "push":
                if self.is_addr(addr_val):
                    operands[0] = ("sym", self.replace(addr_val))
            elif mnemonic == "jmp":
                potential_name = self.lookup(addr_val, exact=True)
                if potential_name is not None:
                    operands[0] = ("sym", potential_name)
                else:
                    operands[0] = (
                        "imm",
                        addr_val - (insn.address + insn.size),
                    )
                    jump_disp_hex = True
            else:
                # Other jumps: show relative displacement via hex().
                operands[0] = ("imm", addr_val - (insn.address + insn.size))
                jump_disp_hex = True
        else:
            for i, op in enumerate(operands):
                if op[0] == "mem":
                    if mnemonic == "call":
                        # Absolute indirect only; leave [reg+disp] alone.
                        if not op[3]:
                            operands[i] = self._sanitize_mem_operand(op, indirect=True)
                    else:
                        operands[i] = self._sanitize_mem_operand(op, indirect=False)
                elif op[0] == "imm":
                    operands[i] = self._sanitize_imm_operand(mnemonic, op)

        ops_tuple = tuple(operands)
        if jump_disp_hex and ops_tuple and ops_tuple[0][0] == "imm":
            raw = (hex(ops_tuple[0][1]),)
            head = f"{insn.prefix} {mnemonic}".strip() if insn.prefix else mnemonic
            display = f"{head} {raw[0]}"
        else:
            raw = tuple(format_operand(op) for op in ops_tuple)
            display = format_instruction(mnemonic, insn.prefix, ops_tuple)
        return replace(
            insn,
            operands=ops_tuple,
            raw_operands=raw,
            display=display,
        )

    def _should_sanitize(self, mnemonic: str, op_str: str, size: int) -> bool:
        return "0x" in op_str and (
            mnemonic in JUMP_MNEMONICS or size > 4 or not self.is_32bit
        )

    def _finalize_code_row(
        self, lite: DisasmLiteTuple, display: str
    ) -> DecodedInstruction:
        """Fallback: attach sanitized display via parse_instruction."""
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
                    # Strip combined rep prefix for JUMP_MNEMONICS membership;
                    # lite tuples still carry Capstone's combined mnemonic.
                    check_mnemonic = inst_mnemonic
                    if check_mnemonic.startswith(("rep ", "repe ", "repne ")):
                        check_mnemonic = check_mnemonic.split(" ", 1)[1]

                    base = self._decoded_by_addr.get(inst_address)
                    if base is not None:
                        if self._should_sanitize(
                            check_mnemonic, inst_op_str, inst_size
                        ):
                            asm.append(self.sanitize_row(base))
                        else:
                            asm.append(base)
                        continue

                    # Fallback when Capstone IR is missing.
                    if self._should_sanitize(check_mnemonic, inst_op_str, inst_size):
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
