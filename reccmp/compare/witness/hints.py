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
from collections.abc import Hashable, Mapping
from typing import Any

from reccmp.compare.witness.machine import STACK_ARG_DWORDS, RunInput

_REGISTERS = {
    "a": "eax",
    "b": "ebx",
    "c": "ecx",
    "d": "edx",
    "si": "esi",
    "di": "edi",
    "bp": "ebp",
}
_ENTRY_MEMORY = (0, ("cfg_mem_init",))  # load tags of memory untouched since entry


def _stack_argument(term: Any) -> int | None:
    """The index of a dword stack argument read as it was at entry:
    `[esp + 4 * (index + 1)]`, the return address being `[esp]`."""
    if not (isinstance(term, tuple) and len(term) == 4 and term[0] == "load"):
        return None
    _, address, size, tag = term
    if size != "dword" or tag not in _ENTRY_MEMORY:
        return None
    if not (isinstance(address, tuple) and len(address) == 5 and address[0] == "mem"):
        return None
    _, segment, terms, displacement, symbols = address
    if segment or symbols or terms != (((("init", "sp")), 1),):
        return None
    if not isinstance(displacement, int) or displacement % 4:
        return None
    index = displacement // 4 - 1
    return index if 0 <= index < STACK_ARG_DWORDS else None


_LOAD_SIZES = {"byte": 1, "word": 2, "dword": 4}


def _entry_load(term: Any) -> tuple[str, int, int] | None:
    """(register, displacement, size) of `[register + displacement]` read
    as it was at entry, through a register other than esp."""
    # pylint: disable=too-many-return-statements
    if not (isinstance(term, tuple) and len(term) == 4 and term[0] == "load"):
        return None
    _, address, size, tag = term
    if size not in _LOAD_SIZES or tag not in _ENTRY_MEMORY:
        return None
    if not (isinstance(address, tuple) and len(address) == 5 and address[0] == "mem"):
        return None
    _, segment, terms, displacement, symbols = address
    if segment or symbols or len(terms) != 1 or not isinstance(displacement, int):
        return None
    ((base, scale),) = terms
    if scale != 1 or not (isinstance(base, tuple) and base[:1] == ("init",)):
        return None
    if base[1] not in _REGISTERS:
        return None
    return _REGISTERS[base[1]], displacement, _LOAD_SIZES[size]


def input_from_assignment(
    assignment: Mapping[Hashable, int], base: RunInput
) -> RunInput | None:
    """``base`` with the assigned registers, stack arguments and memory read
    through entry registers (`this->field`); None when the assignment
    constrains anything else (memory reached otherwise, call results), which
    a run input cannot set. Symbol addresses are ignored: the images fix
    them."""
    registers = dict(base.registers)
    arguments = list(base.stack_args)
    loads: list[tuple[str, int, int, int]] = []
    for term, value in assignment.items():
        if isinstance(term, tuple) and term[:1] == ("sym",):
            # Each image fixes its symbols' addresses; the solver's choice is
            # not an input. The run decides whether the rest reproduces.
            continue
        if isinstance(term, tuple) and term[:1] == ("init",) and term[1] in _REGISTERS:
            registers[_REGISTERS[term[1]]] = value & 0xFFFFFFFF
            continue
        index = _stack_argument(term)
        if index is not None:
            arguments[index] = value & 0xFFFFFFFF
            continue
        load = _entry_load(term)
        if load is None:
            return None
        loads.append((*load, value))
    memory = tuple(
        ((registers[register] + displacement) & 0xFFFFFFFF, size, value)
        for register, displacement, size, value in loads
    )
    if len({address for address, _, _ in memory}) != len(memory):
        return None  # overlapping reads the assignment may not agree on
    return dataclasses.replace(
        base, registers=registers, stack_args=tuple(arguments), memory=memory
    )
