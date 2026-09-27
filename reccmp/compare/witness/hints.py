"""Witness inputs suggested by the solver.

A candidate mismatch between two symbolic values (a return value, a stored
value, a branch predicate) often differs only for a few inputs: `x < 0x41`
against `x <= 0x41` only at `x == 0x41`. Deterministic seeds rarely hit
those. When Z3 finds leaf values under which the two differ, and every leaf
it constrains is one the witness sets at entry (a register, a stack
argument, or modelled memory read through an entry register), that
assignment becomes a run input tried before the seeds.

The run still has to reproduce the divergence on both machines: a solver
assignment is only a suggestion, never a refutation by itself.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from collections.abc import Hashable, Mapping
from typing import Any

from reccmp.compare.asm.verifier.addresses import (
    AddressTerm,
    CfgMemoryInit,
    Init,
    Load,
    MemoryAddress,
    SymbolValue,
)
from reccmp.compare.witness.machine import STACK_ARG_DWORDS, RunInput, lazily_mapped

_REGISTERS = {
    "a": "eax",
    "b": "ebx",
    "c": "ecx",
    "d": "edx",
    "si": "esi",
    "di": "edi",
    "bp": "ebp",
}
_ENTRY_MEMORY = (CfgMemoryInit(),)  # load tags of memory untouched since entry


def _stack_argument(term: Any) -> int | None:
    """The index of a dword stack argument read as it was at entry:
    `[esp + 4 * (index + 1)]`, the return address being `[esp]`."""
    match term:
        case Load(
            MemoryAddress("", (AddressTerm(Init("sp"), 1),), int() as displacement, ()),
            "dword",
            tag,
        ) if (
            tag in _ENTRY_MEMORY and displacement % 4 == 0
        ):
            index = displacement // 4 - 1
            return index if 0 <= index < STACK_ARG_DWORDS else None
    return None


_LOAD_SIZES = {"byte": 1, "word": 2, "dword": 4}


def _entry_load(term: Any) -> tuple[str, int, int] | None:
    """(register, displacement, size) of `[register + displacement]` read
    as it was at entry, through a register other than esp."""
    match term:
        case Load(
            MemoryAddress(
                "", (AddressTerm(Init(family), 1),), int() as displacement, ()
            ),
            size,
            tag,
        ) if (
            family in _REGISTERS and size in _LOAD_SIZES and tag in _ENTRY_MEMORY
        ):
            return _REGISTERS[family], displacement, _LOAD_SIZES[size]
    return None


@dataclass(frozen=True)
class Rejection:
    """Why a solver assignment cannot become a run input."""

    # uncontrollable_leaf: it constrains a term no run input sets (memory
    # reached otherwise, a call result); conflicting_memory: two assigned
    # loads overlap and disagree on a byte; invalid_destination: an assigned
    # load reads memory a run does not map on first touch (the image, the
    # stack), which an input cannot preset.
    reason: str
    term: str


def input_from_assignment(
    assignment: Mapping[Hashable, int], base: RunInput
) -> RunInput | Rejection:
    """``base`` with the assigned registers, stack arguments and memory read
    through entry registers (`this->field`). Symbol addresses are ignored:
    the images fix them."""
    registers = dict(base.registers)
    arguments = list(base.stack_args)
    loads: list[tuple[Hashable, str, int, int, int]] = []
    for term, value in assignment.items():
        match term:
            case SymbolValue():
                # Each image fixes its symbols' addresses; the solver's choice is
                # not an input. The run decides whether the rest reproduces.
                continue
            case Init(family) if family in _REGISTERS:
                registers[_REGISTERS[family]] = value & 0xFFFFFFFF
                continue
        index = _stack_argument(term)
        if index is not None:
            arguments[index] = value & 0xFFFFFFFF
            continue
        load = _entry_load(term)
        if load is None:
            return Rejection("uncontrollable_leaf", repr(term)[:200])
        loads.append((term, *load, value))
    memory: dict[int, int] = {}
    for term, register, displacement, size, value in loads:
        address = (registers[register] + displacement) & 0xFFFFFFFF
        for offset, byte in enumerate(value.to_bytes(8, "little")[:size]):
            at = (address + offset) & 0xFFFFFFFF
            if not lazily_mapped(at):
                return Rejection("invalid_destination", repr(term)[:200])
            if memory.setdefault(at, byte) != byte:
                return Rejection("conflicting_memory", repr(term)[:200])
    return dataclasses.replace(
        base,
        registers=registers,
        stack_args=tuple(arguments),
        memory=tuple(sorted(memory.items())),
    )
