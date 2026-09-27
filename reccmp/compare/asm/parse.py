"""Converts x86 machine code into canonical ``DecodedInstruction`` rows.

Capstone detail-mode decode happens once in ``InstructGen``. This module
sanitizes addresses into references (a name or placeholder to show, and
the identity proofs compare) on the structured operands, and renders each
row's display from them. Nothing here reads a display back.
"""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Hashable
from typing_extensions import Buffer

from reccmp.types import ImageId

from .instgen import InstructGen, SectionType
from .ir import (
    AsmRole,
    DataRegion,
    DecodedInstruction,
    ExtentKind,
    FunctionImage,
    JumpTable,
    compute_extent_closed,
    marker,
    rebind_local_identities,
)
from .model import Reference, ResolvedAddress, format_instruction
from .replacement import AddrTestProtocol, ReferenceResolver

AsmExcerpt = list[DecodedInstruction]


class ParseAsm:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        addr_test: AddrTestProtocol | None = None,
        resolver: ReferenceResolver | None = None,
        is_32bit: bool = True,
        image_id: ImageId | None = None,
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
        self.jump_tables: tuple[JumpTable, ...] = ()
        self.coverage_incomplete: bool = False
        self._body_start: int | None = None
        self._body_end: int | None = None

    def reset(self):
        self.replacements = {}
        self.indirect_replacements = {}

    def is_addr(self, value: int) -> bool:
        """Whether the image says ``value`` is an address (a relocation)."""
        return self.addr_test(value) if self.addr_test is not None else False

    def _image_address(self, value: int) -> bool:
        """Whether an absolute value is an address in the image."""
        return self.is_addr(value) or self.resolve(value) is not None

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

    def _sanitize_mem_operand(self, operand, *, indirect: bool):
        """An address in a memory operand becomes a reference: an absolute
        or displacement the image says is an address (a relocation or a
        known entity). A segment-relative one (``fs:[0]``) never is."""
        match operand:
            case ("mem", size, "", [], int() as disp, ()) if self._image_address(disp):
                return (
                    "mem",
                    size,
                    "",
                    [],
                    0,
                    ((1, self.reference(disp, indirect=indirect)),),
                )
            case (
                "mem",
                size,
                "",
                reg_terms,
                int() as disp,
                (),
            ) if disp and self.is_addr(abs(disp)):
                sign = 1 if disp >= 0 else -1
                return (
                    "mem",
                    size,
                    "",
                    list(reg_terms),
                    0,
                    ((sign, self.reference(abs(disp))),),
                )
        return operand

    def _sanitize_imm_operand(self, mnemonic: str, operand):
        """An immediate the image says is an address becomes a reference;
        one a `cmp` compares only when it names an entity."""
        value = operand[1]
        if not self.is_addr(value):
            return operand
        if mnemonic == "cmp":
            named = self.named_reference(value)
            return ("sym", named) if named is not None else operand
        return ("sym", self.reference(value))

    def _direct_transfer(self, insn: DecodedInstruction):
        """(operand, control target, relative?) of a direct call or jump."""
        assert insn.address is not None and insn.branch_target is not None
        target = insn.branch_target
        if insn.is_call:
            ref = self.reference(target, exact=True)
            return ("sym", ref), ref.identity, False
        if insn.mnemonic == "jmp":
            # The unwind section jumps to other functions: name the target
            # when it has a name.
            named = self.named_reference(target, exact=True)
            if named is not None:
                return ("sym", named), named.identity, False
        # A local jump shows its displacement, not its absolute target.
        displacement = target - (insn.address + insn.size)
        return ("imm", displacement), self.control_identity(target), True

    def sanitize_row(self, insn: DecodedInstruction) -> DecodedInstruction:
        """Replace address operands by references; render the display."""
        assert insn.address is not None
        mnemonic = insn.mnemonic
        operands = list(insn.operands)
        control_target = None
        relative = False
        if insn.branch_target is not None and (insn.is_call or insn.is_jump):
            operands[0], control_target, relative = self._direct_transfer(insn)
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
                    if mnemonic == "push" and len(operands) == 1:
                        if self.is_addr(op[1]):
                            operands[i] = ("sym", self.reference(op[1]))
                    else:
                        operands[i] = self._sanitize_imm_operand(mnemonic, op)

        ops_tuple = tuple(operands)
        if relative:
            head = f"{insn.prefix} {mnemonic}".strip() if insn.prefix else mnemonic
            display = f"{head} {hex(ops_tuple[0][1])}"
        elif ops_tuple == insn.operands:
            display = insn.display
        else:
            display = format_instruction(mnemonic, insn.prefix, ops_tuple)
        return replace(
            insn, operands=ops_tuple, display=display, control_target=control_target
        )

    def parse_asm(self, data: Buffer, start_addr: int) -> AsmExcerpt:
        self.reset()
        asm: AsmExcerpt = []
        blob = bytes(data)
        self._body_start = start_addr
        self._body_end = start_addr + len(blob)

        ig = InstructGen(blob, start_addr, self.is_32bit)
        self.jump_tables = tuple(ig.jump_tables)
        self.coverage_incomplete = ig.coverage_incomplete

        for section in ig.sections:
            if section.type == SectionType.CODE:
                asm.extend(self.sanitize_row(insn) for insn in section.contents)
            elif section.type == SectionType.ADDR_TAB:
                asm.append(marker("Jump table:", role=AsmRole.JUMP_TABLE_HEADER))
                for ofs, target in section.contents:
                    asm.append(
                        marker(
                            f"start + 0x{target - start_addr:x}",
                            address=ofs,
                            role=AsmRole.JUMP_TABLE_ENTRY,
                            payload=(("case", target - start_addr),),
                        )
                    )
            elif section.type == SectionType.DATA_TAB:
                asm.append(marker("Data table:", role=AsmRole.DATA_TABLE_HEADER))
                for ofs, b in section.contents:
                    asm.append(
                        marker(
                            hex(b),
                            address=ofs,
                            role=AsmRole.DATA_TABLE_ENTRY,
                            payload=(("byte", b),),
                        )
                    )

        return asm


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
    """Decode one byte window into instructions and separate embedded data.

    All discovery and sanitization state is local to this call. The returned
    image is the only value subsequent analysis needs.
    """
    blob = bytes(data)
    sanitizer = ParseAsm(addr_test, resolver, is_32bit, image_id)
    sanitizer._body_start = start_addr
    sanitizer._body_end = start_addr + len(blob)
    sections = InstructGen(blob, start_addr, is_32bit)
    instructions: list[DecodedInstruction] = []
    data_regions: list[DataRegion] = []
    for section in sections.sections:
        if section.type == SectionType.CODE:
            instructions.extend(sanitizer.sanitize_row(row) for row in section.contents)
        elif section.type == SectionType.DATA_TAB and section.contents:
            data_regions.append(
                DataRegion(
                    section.contents[0][0],
                    bytes(value for _address, value in section.contents),
                )
            )
    stamped = tuple(
        replace(row, instruction_id=index) for index, row in enumerate(instructions)
    )
    tables = tuple(sections.jump_tables)
    stamped = rebind_local_identities(
        stamped,
        start_addr=start_addr,
        extent=len(blob),
        jump_tables=tables,
        image_id=image_id.name.lower() if image_id is not None else "unknown",
    )
    return FunctionImage(
        start_addr=start_addr,
        extent=len(blob),
        extent_kind=extent_kind,
        instructions=stamped,
        jump_tables=tables,
        coverage_incomplete=sections.coverage_incomplete,
        extent_closed=compute_extent_closed(
            stamped,
            start_addr=start_addr,
            extent=len(blob),
            coverage_incomplete=sections.coverage_incomplete,
            jump_tables=tables,
            extent_kind=extent_kind,
        ),
        raw=blob,
        data_regions=tuple(data_regions),
    )


# The operands `assert` receives in its line and file arguments: the macros,
# not this build's numbers.
_ASSERT_LINE = ("sym", Reference("__LINE__", ("assert_macro", "__LINE__")))
_ASSERT_FILE = ("sym", Reference("__FILE__", ("assert_macro", "__FILE__")))


def _calls_assert(row: DecodedInstruction) -> bool:
    match row.operands:
        case (("sym", Reference(display=name)),) if row.is_call:
            return "_assert" in name
    return False


def _with_operand(row: DecodedInstruction, operand) -> DecodedInstruction:
    return replace(
        row,
        operands=(operand,),
        display=format_instruction(row.mnemonic, row.prefix, (operand,)),
    )


def assert_fixup(asm: AsmExcerpt):
    """Detect assert calls and replace the code filename and line number
    arguments with the macros (from assert.h)."""
    for i, row in enumerate(asm):
        if i >= 3 and _calls_assert(row):
            asm[i - 3] = _with_operand(asm[i - 3], _ASSERT_LINE)
            asm[i - 2] = _with_operand(asm[i - 2], _ASSERT_FILE)
