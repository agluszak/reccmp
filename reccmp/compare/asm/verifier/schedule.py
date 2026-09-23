"""Undo instruction scheduling inside a pair of matched blocks.

The compiler may emit independent instructions of a block in a different
order (stores to different fields, loads hoisted above unrelated stores).
Before aligning a block pair, the recompiled block is reordered towards the
original's order, moving an instruction only past instructions it provably
does not depend on (see relocation.effects_conflict). Each move is a legal
swap of independent instructions, so the reordered block computes the same
thing as the emitted one; the lockstep checks then decide equivalence.
"""

from __future__ import annotations

from reccmp.compare.asm.ir import ResolvedAsm
from reccmp.compare.asm.verifier.block_align import DpLine, dp_line
from reccmp.compare.asm.verifier.relocation import (
    LineEffects,
    effects_conflict,
    sequence_effects,
)

# Beyond this many instructions per block, scheduling is not attempted.
_MAX_BLOCK = 256


def _block_effects(stream: ResolvedAsm, indices: list[int]) -> list[LineEffects]:
    """Effects of a block's instructions, executed from a fresh state at
    the block entry so addresses are relative to the entry registers."""
    block = ResolvedAsm(
        [stream.displays[i] for i in indices],
        [stream.instructions[i] for i in indices],
        [stream.roles[i] for i in indices],
        from_ir=stream.from_ir,
    )
    effects = sequence_effects(block)
    return (
        effects if effects is not None else [LineEffects(barrier=True)] * len(indices)
    )


def _independent(moved: LineEffects, crossed: LineEffects) -> bool:
    if effects_conflict(moved, crossed) or effects_conflict(crossed, moved):
        return False
    # Swapping two flag writers changes which one a later reader sees.
    return not (moved.writes_flags and crossed.writes_flags)


def _same_instruction(a: DpLine, b: DpLine) -> bool:
    return a.display == b.display or (
        a.skeleton is not None and a.skeleton == b.skeleton
    )


def schedule_like(
    orig_stream: ResolvedAsm,
    recomp_stream: ResolvedAsm,
    indices_o: list[int],
    indices_r: list[int],
) -> list[int]:
    """The recompiled block's indices, reordered towards the original order
    where that only swaps independent instructions."""
    if len(indices_r) > _MAX_BLOCK or len(indices_r) < 2:
        return indices_r
    lines_o = [dp_line(orig_stream, i) for i in indices_o]
    lines_r = {i: dp_line(recomp_stream, i) for i in indices_r}
    effects = dict(zip(indices_r, _block_effects(recomp_stream, indices_r)))

    placed: list[int] = []
    remaining = list(indices_r)
    for line_o in lines_o:
        for position, candidate in enumerate(remaining):
            if not _same_instruction(line_o, lines_r[candidate]):
                continue
            moved = effects[candidate]
            if all(
                _independent(moved, effects[other]) for other in remaining[:position]
            ):
                placed.append(candidate)
                del remaining[position]
            break  # only the first matching instruction is considered
    return placed + remaining
