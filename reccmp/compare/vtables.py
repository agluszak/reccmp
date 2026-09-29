"""Slot-by-slot comparison of paired vtables."""

import enum
import logging
import struct
from dataclasses import dataclass
from itertools import zip_longest

from reccmp.formats import Image
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.types import EntityType, ImageId
from .db import EntityDb, ReccmpEntity, ReccmpMatch
from .thunk_resolve import effective_orig_vtable_size, resolve_vtable_slot

logger = logging.getLogger(__name__)


class SlotStatus(enum.Enum):
    MATCH = "match"
    # Both targets have the same complete return-only body. Their source
    # identities remain unresolved when retail folded several functions.
    CODE_EQUIVALENT = "code-equivalent"
    # Both slots name paired functions, and not the same pair.
    DIFFERENT = "different"
    # A slot names a function reccmp has not paired: the catalog cannot say
    # whether it is right. An original slot at a paired body next to an
    # unpaired recompiled function is typical of identical-code folding in
    # the original link: one retail body serves several source functions.
    UNPAIRED = "unpaired"


@dataclass(frozen=True)
class VtableSlot:
    """One pointer of a paired vtable, resolved on both sides."""

    offset: int
    orig_raw: int | None
    recomp_raw: int | None
    orig: ReccmpEntity | None
    recomp: ReccmpEntity | None
    status: SlotStatus

    @property
    def matches(self) -> bool:
        return self.status in (SlotStatus.MATCH, SlotStatus.CODE_EQUIVALENT)


@dataclass(frozen=True)
class VtableComparison:
    match: ReccmpMatch
    slots: tuple[VtableSlot, ...]

    @property
    def matches(self) -> bool:
        return all(slot.matches for slot in self.slots)


def _table_sizes(db: EntityDb, orig_bin: Image, match: ReccmpMatch) -> tuple[int, int]:
    recomp_size = match.any_size(ImageId.RECOMP)

    # The vtable size should always be a multiple of 4 because that
    # is the pointer size. If it is not (for whatever reason)
    # it would cause iter_unpack to blow up so let's just fix it.
    if recomp_size % 4 != 0:
        logger.warning(
            "Vtable for class %s has irregular size %d", match.name, recomp_size
        )
        recomp_size = 4 * (recomp_size // 4)

    # The PDB doesn't record a size for the vtable itself, so the recomp
    # size is an estimate: it's either the gap between this symbol and the
    # next, or the size listed in cvdump's SECTION CONTRIBUTIONS output.
    # Either estimate can include alignment padding after the table, and
    # reading the orig table with the padded size would run past the
    # actual end of the table. Just use the orig size if known.
    orig_size = match.size(ImageId.ORIG)
    if orig_size is None:
        orig_size = recomp_size
    elif orig_size % 4 != 0:
        logger.warning(
            "Vtable for class %s has irregular orig size %d", match.name, orig_size
        )
        orig_size = 4 * (orig_size // 4)

    orig_max = match.max_size(ImageId.ORIG)
    if orig_max is not None:
        orig_size = min(orig_size, orig_max)
    orig_size = effective_orig_vtable_size(
        orig_bin, match.orig_addr, orig_size, db=db, image_id=ImageId.ORIG
    )
    return orig_size, recomp_size


def _slot_status(
    db: EntityDb,
    orig_bin: Image,
    recomp_bin: Image,
    raw: tuple[int | None, int | None],
    orig: ReccmpEntity | None,
    recomp: ReccmpEntity | None,
) -> SlotStatus:
    raw_orig, raw_recomp = raw
    if raw_orig is None or raw_recomp is None:
        # One table is longer than the other.
        return SlotStatus.DIFFERENT
    if orig is not None and recomp is not None:
        if orig.recomp_addr is not None and orig.recomp_addr == recomp.recomp_addr:
            return SlotStatus.MATCH
        if (
            recomp.recomp_addr is not None
            and db.alias_canonical_orig(ImageId.RECOMP, recomp.recomp_addr) == raw_orig
        ):
            return SlotStatus.MATCH
    if orig is None or recomp is None or not orig.matched or not recomp.matched:
        if _same_return_only_body(orig_bin, recomp_bin, orig, recomp):
            return SlotStatus.CODE_EQUIVALENT
        return SlotStatus.UNPAIRED
    return SlotStatus.DIFFERENT


def _same_return_only_body(
    orig_bin: Image,
    recomp_bin: Image,
    orig: ReccmpEntity | None,
    recomp: ReccmpEntity | None,
) -> bool:
    if (
        orig is None
        or recomp is None
        or orig.get("type") != EntityType.FUNCTION
        or recomp.get("type") != EntityType.FUNCTION
    ):
        return False
    size = recomp.size(ImageId.RECOMP)
    if size not in (1, 3):
        return False
    orig_addr = orig.addr(ImageId.ORIG)
    recomp_addr = recomp.addr(ImageId.RECOMP)
    if orig_addr is None or recomp_addr is None:
        return False
    try:
        code = bytes(recomp_bin.read(recomp_addr, size))
        orig_code = bytes(orig_bin.read(orig_addr, size))
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return False
    if code != b"\xc3" and (size != 3 or code[:1] != b"\xc2"):
        return False
    return orig_code == code


def compare_vtable(
    db: EntityDb, orig_bin: Image, recomp_bin: Image, match: ReccmpMatch
) -> VtableComparison:
    orig_size, recomp_size = _table_sizes(db, orig_bin, match)
    orig_addrs = [
        t
        for (t,) in struct.iter_unpack("<L", orig_bin.read(match.orig_addr, orig_size))
    ]
    recomp_addrs = [
        t
        for (t,) in struct.iter_unpack(
            "<L", recomp_bin.read(match.recomp_addr, recomp_size)
        )
    ]

    # Trailing null entries on the recomp side are alignment padding, not
    # virtual functions missing from orig, so drop them. Non-null entries
    # past the end of the orig table are kept: these are virtual functions
    # that only exist in recomp.
    while len(recomp_addrs) > len(orig_addrs) and recomp_addrs[-1] == 0:
        recomp_addrs.pop()

    slots = []
    for i, (raw_orig, raw_recomp) in enumerate(zip_longest(orig_addrs, recomp_addrs)):
        recomp = (
            resolve_vtable_slot(db, ImageId.RECOMP, recomp_bin, raw_recomp)
            if raw_recomp is not None
            else None
        )
        # Orig binaries may contain literal NULL vtable slots (reserved gap).
        # MSVC cannot emit mid-table NULL entries, so accept any recomp slot.
        if raw_orig == 0:
            slots.append(
                VtableSlot(4 * i, raw_orig, raw_recomp, None, recomp, SlotStatus.MATCH)
            )
            continue

        orig = (
            resolve_vtable_slot(db, ImageId.ORIG, orig_bin, raw_orig)
            if raw_orig is not None
            else None
        )
        slots.append(
            VtableSlot(
                4 * i,
                raw_orig,
                raw_recomp,
                orig,
                recomp,
                _slot_status(
                    db, orig_bin, recomp_bin, (raw_orig, raw_recomp), orig, recomp
                ),
            )
        )

    return VtableComparison(match, tuple(slots))
