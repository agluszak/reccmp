"""Where a function's references land, in terms of the catalog.

The same location has a different address in each binary; what it means is
the paired object it belongs to, which this module finds for an address or
a loop bound, and how the function uses it, which Ghidra's p-code says.
"""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

import bisect
from dataclasses import dataclass
from typing import Any

from reccmp.compare.manifest import Manifest, NamedObject
from reccmp.types import EntityType, ImageId
from .results import ObjectOffset

STRING_TYPES = (EntityType.STRING, EntityType.WIDECHAR)
# How far before the nearest array end to look for an array a loop bound
# belongs to.
_BOUND_SEARCH = 0x1000


@dataclass(frozen=True)
class Located:
    """Where a referenced address falls in the catalog of one image."""

    start: int
    offset: int
    size: int | None
    entity_type: EntityType | None
    # The pair the location belongs to; None for unpaired catalog data.
    named: NamedObject | None


class Extents:
    """Catalog entities of one image, by the address range they occupy."""

    def is_data(self, address: int) -> bool:
        """Import slots and code extents contribute identity, not literal bytes."""
        located = self.containing(address)
        return located is None or located.entity_type not in (
            EntityType.IMPORT,
            EntityType.IMPORT_THUNK,
            EntityType.FUNCTION,
            EntityType.VTORDISP,
            EntityType.THUNK,
        )

    def __init__(self, manifest: Manifest, image_id: ImageId):
        spans = [
            Located(obj.addr(image_id), 0, obj.extent(image_id), obj.entity_type, obj)
            for obj in manifest.objects
        ] + [
            Located(entity.addr, 0, entity.size, entity.entity_type, None)
            for entity in manifest.unpaired
            if entity.image_id == image_id
        ]
        pairs = {obj.orig_addr: obj for obj in manifest.objects}
        spans += [
            Located(
                alias.addr,
                0,
                alias.size or pairs[alias.canonical_orig].extent(image_id),
                pairs[alias.canonical_orig].entity_type,
                pairs[alias.canonical_orig],
            )
            for alias in manifest.aliases
            if alias.image_id == image_id and alias.canonical_orig in pairs
        ]
        spans.sort(key=lambda located: located.start)
        self._starts = [located.start for located in spans]
        self._spans = spans
        # Paired objects of known extent, by the address just past them.
        sized = sorted(
            (span for span in spans if span.named is not None and span.size),
            key=lambda span: span.start + (span.size or 0),
        )
        self._sized = sized
        self._sized_ends = [span.start + (span.size or 0) for span in sized]

    def containing(self, addr: int) -> Located | None:
        i = bisect.bisect_right(self._starts, addr) - 1
        if i < 0:
            return None
        span = self._spans[i]
        offset = addr - span.start
        if offset == 0 or (span.size is not None and offset < span.size):
            return Located(span.start, offset, span.size, span.entity_type, span.named)
        return None

    def bound_at(self, addr: int) -> Located | None:
        """The paired object a loop bound at `addr` belongs to, located at
        the bound's offset from it.

        A loop over an array compares its pointer with the address just
        past the array, or, stepping through one field of each element,
        with that field's address in the element past the last: past the
        array's end by less than one element, so by less than its size.
        The exact end comes first; then a paired object starting at or
        holding `addr` (other than a string, whose middle nothing compares
        with), which the code compares with as itself; then the nearest
        array whose end is that close."""
        i = bisect.bisect_right(self._sized_ends, addr)
        exact = i > 0 and self._sized_ends[i - 1] == addr
        if not exact:
            inside = self.containing(addr)
            if inside is not None and inside.named is not None:
                if inside.offset == 0 or inside.entity_type not in STRING_TYPES:
                    return None
        for span in reversed(self._sized[:i]):
            assert span.size is not None
            offset = addr - span.start
            if offset < 2 * span.size:
                return Located(
                    span.start, offset, span.size, span.entity_type, span.named
                )
            if self._sized_ends[i - 1] - (span.start + span.size) > _BOUND_SEARCH:
                break
        return None


@dataclass(frozen=True)
class Use:
    """How a function refers to one data location."""

    addr: int
    # The array and offset, when the location is a loop bound.
    bound: ObjectOffset | None
    # The bytes an instruction loads or stores there; None when the
    # function only takes the address.
    access: int | None


def access_size(instruction: Any, addr: int) -> int | None:
    """How many bytes an instruction's p-code loads from or stores to a
    fixed address, if any."""
    from ghidra.program.model.pcode import PcodeOp

    sizes = []
    for op in instruction.getPcode():
        for varnode in (*op.getInputs(), op.getOutput()):
            if (
                varnode is not None
                and varnode.isAddress()
                and varnode.getOffset() == addr
            ):
                sizes.append(int(varnode.getSize()))
        if op.getOpcode() in (PcodeOp.LOAD, PcodeOp.STORE):
            pointer = op.getInput(1)
            if pointer.isConstant() and pointer.getOffset() == addr:
                accessed = (
                    op.getOutput() if op.getOpcode() == PcodeOp.LOAD else op.getInput(2)
                )
                sizes.append(int(accessed.getSize()))
    return max(sizes, default=None)


def only_compared(instruction: Any, value: int) -> bool:
    """Whether an instruction uses a constant only to compare with it, by
    its p-code: in comparisons, or in a difference kept only in a temporary
    for the flags it sets."""
    from ghidra.program.model.pcode import PcodeOp

    comparisons = {
        PcodeOp.INT_EQUAL,
        PcodeOp.INT_NOTEQUAL,
        PcodeOp.INT_LESS,
        PcodeOp.INT_SLESS,
        PcodeOp.INT_LESSEQUAL,
        PcodeOp.INT_SLESSEQUAL,
        PcodeOp.INT_CARRY,
        PcodeOp.INT_SCARRY,
        PcodeOp.INT_SBORROW,
    }
    used = False
    for op in instruction.getPcode():
        if not any(
            vn.isConstant() and vn.getOffset() == value for vn in op.getInputs()
        ):
            continue
        used = True
        if op.getOpcode() in comparisons:
            continue
        output = op.getOutput()
        if (
            op.getOpcode() == PcodeOp.INT_SUB
            and output is not None
            and output.isUnique()
        ):
            continue
        return False
    return used


def register_operand(instruction: Any, operand: int) -> bool:
    """Whether a reference hangs on an operand that is only a register.

    Ghidra's constant propagation attaches a reference to a register operand
    whose value it can compute, when that value happens to be an address;
    the instruction reads no memory there."""
    from ghidra.program.model.lang import OperandType

    if operand < 0:
        return False
    kind = instruction.getOperandType(operand)
    return OperandType.isRegister(kind) and not OperandType.isDynamic(kind)


def bitwise_scalar_operand(instruction: Any, operand: int, value: int) -> bool:
    """A scalar bit mask is not an address even if Ghidra resolves its value
    to a data location in the loaded image."""
    from ghidra.program.model.lang import OperandType

    if operand < 0 or instruction.getMnemonicString().upper() not in {
        "TEST",
        "AND",
        "OR",
        "XOR",
    }:
        return False
    if OperandType.isDynamic(instruction.getOperandType(operand)):
        return False
    scalar = instruction.getScalar(operand)
    return scalar is not None and scalar.getUnsignedValue() == value
