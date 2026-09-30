"""Read literal and initialized contents from Ghidra programs."""

# pylint: disable=import-outside-toplevel,import-error
from typing import TYPE_CHECKING, Any
from reccmp.types import EntityType
from .results import StringValue

if TYPE_CHECKING:
    from ghidra.program.model.listing import Program
    from ghidra.program.model.address import Address
_RAW_LIMIT = 64


def literal_data_type(entity_type: EntityType | None, size: int | None) -> Any:
    """The Ghidra data type for a literal the catalog found, if any."""
    from ghidra.program.model.data import (
        DoubleDataType,
        FloatDataType,
        TerminatedStringDataType,
        TerminatedUnicodeDataType,
    )

    match entity_type, size:
        case EntityType.STRING, _:
            return TerminatedStringDataType.dataType
        case EntityType.WIDECHAR, _:
            return TerminatedUnicodeDataType.dataType
        case EntityType.FLOAT, 4:
            return FloatDataType.dataType
        case EntityType.FLOAT, 8:
            return DoubleDataType.dataType
    return None


def decode_string(raw: bytes, entity_type: EntityType | None) -> StringValue:
    """A catalog string entity's text, up to its terminator."""
    wide = entity_type == EntityType.WIDECHAR
    text = raw.decode("utf-16-le" if wide else "latin1", errors="replace")
    return StringValue(text.split("\0", 1)[0])


def ghidra_string(program: "Program", address: "Address") -> StringValue | None:
    """The string Ghidra's analysis defined at (or around) an address."""
    from ghidra.program.model.data import StringDataInstance

    data = program.getListing().getDataContaining(address)
    if data is None or not StringDataInstance.isString(data):
        return None
    string = StringDataInstance.getStringDataInstance(data)
    offset = address.subtract(data.getMinAddress())
    if offset:
        string = string.getByteOffcut(offset)
    value = string.getStringValue()
    return StringValue(str(value)) if value is not None else None


def relocation_at(program: "Program", address: "Address") -> bool:
    return bool(program.getRelocationTable().hasRelocation(address))


def read_contents(
    program: "Program", address: "Address", length: int
) -> tuple[bytes, bool] | None:
    """Up to ``length`` initialized bytes and whether any is relocated."""
    import jpype
    from ghidra.program.model.address import AddressSet

    block = program.getMemory().getBlock(address)
    if block is None or not block.isInitialized():
        return None
    length = min(length, _RAW_LIMIT, block.getEnd().subtract(address) + 1)
    if length <= 0:
        return b"", False
    buffer = jpype.JArray(jpype.JByte)(length)
    program.getMemory().getBytes(address, buffer)
    relocated = (
        program.getRelocationTable()
        .getRelocations(AddressSet(address, address.add(length - 1)))
        .hasNext()
    )
    return bytes(b & 0xFF for b in buffer), relocated
