"""Resolve vtable slot targets through ILT thunks and raw mid-.text E9 jmp chains."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from reccmp.compare.asm.decode import get_detail_disassembler
from reccmp.formats import Image
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.formats.image import ImageSectionFlags
from reccmp.types import EntityType, ImageId

if TYPE_CHECKING:
    from reccmp.compare.db import EntityDb, ReccmpEntity

_MAX_HOPS = 8

# Entity kinds that are valid vtable slot targets when recorded in the DB.
_KNOWN_CODE_ENTITY_TYPES = frozenset(
    {
        EntityType.FUNCTION,
        EntityType.THUNK,
        EntityType.IMPORT,
        EntityType.IMPORT_THUNK,
        EntityType.VTORDISP,
    }
)


def read_e9_jmp_target(binfile: Image, addr: int) -> int | None:
    """If *addr* begins a 5-byte ``jmp rel32`` (0xE9), return the jump target."""
    try:
        data = binfile.read(addr, 5)
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return None
    if len(data) < 5 or data[0] != 0xE9:
        return None
    (operand,) = struct.unpack("<i", data[1:5])
    return addr + 5 + operand


def _in_executable_section(binfile: Image, addr: int) -> bool:
    """True when *addr* falls in a section marked executable.

    Returns False when the image has no section metadata — callers must not
    treat a missing map as "everything is code".
    """
    if not binfile.sections:
        return False
    for section in binfile.sections:
        if addr not in section.virtual_range:
            continue
        return ImageSectionFlags.EXECUTE in section.flags
    return False


def _decodes_as_instruction(binfile: Image, addr: int) -> bool:
    """True when Capstone decodes a non-padding instruction at *addr*."""
    try:
        view, remaining = binfile.seek(addr)
    except InvalidVirtualAddressError:
        return False
    if remaining <= 0:
        return False
    # x86 instructions are at most 15 bytes; take whatever is left in-range.
    size = min(15, remaining)
    data = bytes(view[:size]) if size <= len(view) else binfile.read(addr, size)
    if not data:
        return False
    # int3 / alignment padding is common in .text and is not a vtable target.
    if data[0] == 0xCC:
        return False
    disassembler = get_detail_disassembler(is_32=True)
    for insn in disassembler.disasm(data, addr, count=1):
        if insn.mnemonic == "int3":
            return False
        return True
    return False


def is_plausible_vtable_target(
    binfile: Image,
    addr: int,
    *,
    db: EntityDb | None = None,
    image_id: ImageId = ImageId.ORIG,
) -> bool:
    """Heuristic: vtable slots point at code or known thunks, not RTTI/strings.

    Accepts:
    - NULL / zero (empty reserved slot)
    - a known FUNCTION / THUNK / IMPORT / IMPORT_THUNK / VTORDISP entity
    - an address in an executable section where Capstone decodes an instruction

    Without section metadata the executable-section path is refused (no
    opcode whitelist fallback).
    """
    if addr == 0:
        return True
    if addr < 0x10000:
        return False

    if db is not None:
        entity = db.get(image_id, addr, exact=True)
        if entity is not None and entity.entity_type in _KNOWN_CODE_ENTITY_TYPES:
            return True

    if not _in_executable_section(binfile, addr):
        return False
    return _decodes_as_instruction(binfile, addr)


def effective_orig_vtable_size(
    binfile: Image,
    orig_addr: int,
    read_size: int,
    *,
    db: EntityDb | None = None,
    image_id: ImageId = ImageId.ORIG,
) -> int:
    """Trim comparison to the longest contiguous plausible orig prefix.

    Walk slots from the start of the table. Null slots are allowed (they do
    not end the prefix). The first non-null implausible target stops the walk.
    The returned size covers through the last plausible non-null slot in that
    prefix — trailing nulls after the last method are dropped.

    Recompiled vtables are often longer than the original (extra inherited
    tail). Reading ``recomp_size`` bytes from the orig address pulls in the
    next object in ``.rdata`` and tanks the match ratio (TMapMaker is the
    canonical case). Using a contiguous prefix (not "last plausible anywhere")
    also avoids extending past a mid-table hole of non-code data into a later
    accidental code pointer.
    """
    if read_size <= 0 or read_size % 4 != 0:
        return read_size

    try:
        table = binfile.read(orig_addr, read_size)
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return read_size

    last_nonzero_code = -1
    for i, (slot,) in enumerate(struct.iter_unpack("<L", table)):
        if slot == 0:
            continue
        if not is_plausible_vtable_target(binfile, slot, db=db, image_id=image_id):
            break
        last_nonzero_code = i

    if last_nonzero_code < 0:
        return read_size

    return (last_nonzero_code + 1) * 4


def resolve_vtable_slot(
    db: EntityDb,
    image_id: ImageId,
    binfile: Image,
    raw_addr: int,
    *,
    max_hops: int = _MAX_HOPS,
) -> ReccmpEntity | None:
    """Follow thunk DB entries and raw single-flow JMP stubs to a paired FUNCTION.

    MSVC incremental-link tables (ILT) at the start of ``.text`` are modeled as
    ``EntityType.THUNK`` with a single ``ref_*`` hop. Some vtable slots point at
    bad ILT aliases that forward through an unregistered mid-``.text`` ``E9`` stub
    before reaching a second ILT entry and the real body. Mirrors
    ``tools/ghidra/vtable_slots.py`` ``resolve()`` in the Imperialism decomp.
    """
    ref_key = "ref_orig" if image_id == ImageId.ORIG else "ref_recomp"
    addr = raw_addr
    last_entity: ReccmpEntity | None = None

    for _ in range(max_hops):
        entity = db.get(image_id, addr, exact=True)
        if entity is not None:
            last_entity = entity
            if entity.entity_type == EntityType.FUNCTION:
                return entity
            if entity.entity_type == EntityType.THUNK:
                ref_addr = entity.get(ref_key)
                if isinstance(ref_addr, int):
                    target = db.get(image_id, ref_addr, exact=True)
                    if target is not None and target.entity_type == EntityType.FUNCTION:
                        return target
                    addr = ref_addr
                    continue

        next_addr = read_e9_jmp_target(binfile, addr)
        if next_addr is None or next_addr == addr:
            break
        addr = next_addr

    return last_entity
