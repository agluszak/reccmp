"""Normalize compile-database commands for the Clang source indexer."""

from __future__ import annotations

import shlex
from typing import Any


def _command_arguments(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if arguments:
        return [str(item) for item in arguments]
    return shlex.split(str(entry["command"]), posix=True)


def record_command(
    entry: dict[str, Any], indexer: str, clang: str | None = None
) -> list[str]:
    """Normalize a compile-database entry into an indexer driver command."""
    arguments = _command_arguments(entry)
    compiler = clang or arguments[0]
    filtered: list[str] = []
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-c", "/c", "-o", "-MF", "-MT", "-MQ"}:
            skip_next = argument in {"-o", "-MF", "-MT", "-MQ"}
            continue
        # codespell:ignore-begin
        if argument.startswith(("/Fo", "/Fd", "-o")):
            # codespell:ignore-end
            continue
        filtered.append(argument)
    try:
        separator = filtered.index("--")
    except ValueError:
        separator = len(filtered)
    return [
        indexer,
        compiler,
        *filtered[:separator],
        # clang-cl defers primary template bodies until instantiation by
        # default. The source-use index needs the original dependent
        # expressions too, so collect from Clang's eager AST while keeping
        # function ownership rules separate from body availability.
        "-fno-delayed-template-parsing",
        "-fsyntax-only",
        *filtered[separator:],
    ]
