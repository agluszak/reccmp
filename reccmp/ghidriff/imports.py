"""Imports as each program's analysis sees them.

An import is an external location, and a function once Ghidra has made one
for it; its stack purge lives on the function.
"""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ghidra.program.model.listing import Program


def import_locations(program: "Program") -> dict[tuple[str, str], Any]:
    """A program's imports by library and imported name."""
    manager = program.getExternalManager()
    locations = {}
    for library in manager.getExternalLibraryNames():
        iterator = manager.getExternalLocations(library)
        while iterator.hasNext():
            location = iterator.next()
            name = location.getOriginalImportedName() or location.getLabel()
            locations[(str(library).upper(), str(name))] = location
    return locations


def known_purge(location: Any) -> int | None:
    """An import's stack purge, if its function has a known one."""
    from ghidra.program.model.listing import Function

    function = location.getFunction() if location is not None else None
    if function is None:
        return None
    purge = int(function.getStackPurgeSize())
    return None if purge == Function.UNKNOWN_STACK_DEPTH_CHANGE else purge
