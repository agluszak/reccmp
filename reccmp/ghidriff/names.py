"""Canonical catalog names in Ghidra rendering."""

import re
from typing import TYPE_CHECKING, Any

# pylint: disable=import-outside-toplevel,import-error
from ghidriff import GhidraDiffEngine
from reccmp.compare.manifest import Manifest, NamedObject
from reccmp.types import EntityType, ImageId
from .results import DataReference, ObjectOffset

if TYPE_CHECKING:
    from ghidra.program.model.listing import Program
    from ghidra.program.model.address import Address

_LITERAL_TYPES = (EntityType.STRING, EntityType.WIDECHAR, EntityType.FLOAT)
_RAW_ADDRESS = re.compile(r"(?<![\w])0x[0-9a-fA-F]+(?![\w])")
_DEFAULT_PARAMETER = re.compile(r"\bparam_(\d+)\b")
_QUOTED = re.compile(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")


def canonical_parameter_names(code: list[str]) -> None:
    """Use zero-based parameter names without changing quoted literals."""
    for i, line in enumerate(code):
        parts = _QUOTED.split(line)
        code[i] = "".join(
            (
                part
                if j % 2
                else _DEFAULT_PARAMETER.sub(
                    lambda m: f"param{int(m.group(1)) - 1}", part
                )
            )
            for j, part in enumerate(parts)
        )


def unquoted_raw_addresses(code: str) -> set[int]:
    """Collect displayed addresses without treating strings/comments as code."""
    found: set[int] = set()
    for line in code.splitlines():
        if line.lstrip().startswith(("/*", "//", "*")):
            continue
        for index, part in enumerate(GhidraDiffEngine.QUOTED_LITERAL.split(line)):
            if index % 2 == 0:
                found.update(
                    int(match.group(), 16) for match in _RAW_ADDRESS.finditer(part)
                )
    return found


def unpaired_names(
    manifest: Manifest, shared: set[str]
) -> dict[tuple[ImageId, int], str]:
    """Own-image names for unpaired entities. A name that is also a shared
    name, or that several unpaired entities of one image carry, is
    qualified with the address, so it cannot suggest a correspondence."""
    named = [
        entity
        for entity in manifest.unpaired
        if entity.name is not None and entity.entity_type not in _LITERAL_TYPES
    ]
    counts: dict[tuple[ImageId, str], int] = {}
    for entity in named:
        assert entity.name is not None
        key = (entity.image_id, entity.name)
        counts[key] = counts.get(key, 0) + 1
    result = {}
    for entity in named:
        assert entity.name is not None
        unique = counts[(entity.image_id, entity.name)] == 1
        result[(entity.image_id, entity.addr)] = (
            entity.name
            if unique and entity.name not in shared
            else f"{entity.name}@{entity.addr:#x}"
        )
    return result


def replace_paired_raw_addresses(code: list[str], tokens: dict[int, str]) -> None:
    """Replace referenced raw addresses, leaving literals and comments intact."""
    if not tokens:
        return

    def rename(match: re.Match[str]) -> str:
        return tokens.get(int(match.group(), 16), match.group())

    for index, line in enumerate(code):
        if line.lstrip().startswith(("/*", "//", "*")):
            continue
        parts = GhidraDiffEngine.QUOTED_LITERAL.split(line)
        code[index] = "".join(
            part if part_index % 2 else _RAW_ADDRESS.sub(rename, part)
            for part_index, part in enumerate(parts)
        )


def paired_reference_tokens(
    orig_refs: tuple[DataReference, ...],
    recomp_refs: tuple[DataReference, ...],
    paired_recomp_addrs: dict[int, int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Name paired addresses confirmed by at least one side's reference.

    Ghidra sometimes prints the counterpart as a raw constant without making
    a reference. The manifest still supplies its exact paired address.
    """
    by_side = []
    for refs in (orig_refs, recomp_refs):
        locations: dict[ObjectOffset, set[int]] = {}
        for ref in refs:
            if ref.object is not None:
                locations.setdefault(ref.object, set()).add(ref.address)
        by_side.append(locations)
    orig, recomp = by_side[0], by_side[1]
    orig_tokens: dict[int, str] = {}
    recomp_tokens: dict[int, str] = {}
    for obj in orig.keys() | recomp.keys():
        canonical_recomp = (paired_recomp_addrs or {}).get(obj.orig_addr)
        if canonical_recomp is not None and (
            obj.orig_addr + obj.offset in orig.get(obj, ())
            or canonical_recomp + obj.offset in recomp.get(obj, ())
        ):
            orig_addr = obj.orig_addr + obj.offset
            recomp_addr = canonical_recomp + obj.offset
        elif obj in orig and obj in recomp and len(orig[obj]) == len(recomp[obj]) == 1:
            orig_addr = next(iter(orig[obj]))
            recomp_addr = next(iter(recomp[obj]))
        else:
            continue
        if orig_addr == recomp_addr:
            continue
        token = f"PAIRED_DATA_{obj.orig_addr:x}_{obj.offset:x}"
        orig_tokens[orig_addr] = token
        recomp_tokens[recomp_addr] = token
    return orig_tokens, recomp_tokens


def canonical_names(objects: tuple[NamedObject, ...]) -> dict[int, str]:
    """One Ghidra name per paired object, keyed by original address.

    A name shared by distinct objects is qualified with the original
    address, which identifies the pair on both sides."""
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.name] = counts.get(obj.name, 0) + 1
    return {
        obj.orig_addr: (
            obj.name if counts[obj.name] == 1 else f"{obj.name}@{obj.orig_addr:#x}"
        )
        for obj in objects
    }


def ghidra_name(name: str) -> str:
    from ghidra.program.model.symbol import SymbolUtilities

    return SymbolUtilities.replaceInvalidChars(name, True)


def rename_function(function: Any, name: str) -> None:
    from ghidra.program.model.symbol import SourceType

    rendered_name = ghidra_name(name)
    # The PE loader labels an export's entry with its decorated name; a
    # function cannot take a name another symbol already holds there.
    primary = function.getSymbol()
    table = function.getProgram().getSymbolTable()
    for symbol in table.getSymbols(function.getEntryPoint()):
        if symbol != primary and symbol.getName() == rendered_name:
            symbol.delete()
    function.setName(rendered_name, SourceType.USER_DEFINED)


def label(program: "Program", address: "Address", name: str) -> None:
    from ghidra.program.model.symbol import SourceType

    program.getSymbolTable().createLabel(
        address, ghidra_name(name), SourceType.USER_DEFINED
    ).setPrimary()


def name_end(program: "Program", instruction: Any, ref: Any, end: ObjectOffset) -> None:
    """Show the compared constant as its offset from its array. The
    decompiler shows an equate for the constant, also when it adjusts
    the constant by one to rewrite the comparison."""
    name = ghidra_name(f"{end.name}+{end.offset:#x}")
    value = ref.getToAddress().getOffset()
    equates = program.getEquateTable()
    equate = equates.getEquate(name) or equates.createEquate(name, value)
    if equate.getValue() == value:
        equate.addReference(instruction.getAddress(), ref.getOperandIndex())
