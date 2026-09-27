"""Cost-minimizing alignment of two paired blocks' instructions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from reccmp.compare.asm.ir import DecodedInstruction, instruction_match_key
from reccmp.compare.asm.model import REGISTERS
from reccmp.compare.asm.operand import Mem, Operand, Reg, ScaledReg
from reccmp.compare.asm.verifier.state import (
    JCC_MNEMONICS,
    STRING_OPS,
)
from reccmp.compare.asm.verifier.obligations import may_be_one_sided

# Line-alignment costs (integers to keep DP ties exact): prefer exact text,
# then identical instruction shape (registers anonymized), then a shared
# mnemonic, then anything in the same observable class. A gap (one-sided
# line) costs more than a good pairing but less than two forced bad ones.
_SUB_EXACT = 0
_SUB_SKELETON = 1
_SUB_MNEMONIC = 4
_SUB_CLASS = 7
_GAP = 5

_X87_MEM_WRITERS_DP = frozenset({"fst", "fstp", "fist", "fistp", "fnstcw", "fbstp"})


class LineClass(Enum):
    NONE = "none"
    PUSH = "push"
    STORE = "store"
    CALL = "call"


@dataclass(frozen=True, slots=True)
class AlignedPair:
    orig: int | None
    recomp: int | None


@dataclass(frozen=True, slots=True)
class BlockAlignment:
    pairs: tuple[AlignedPair, ...]
    cost: int


def _base_line_class(ins: DecodedInstruction) -> LineClass:
    # pylint: disable=too-many-return-statements
    mnemonic = ins.mnemonic
    if ins.is_call:
        return LineClass.CALL
    if mnemonic == "push":
        # Pushes pair with pushes, but may also go one-sided (a scratch
        # spill on one side only); the verifier gates the soundness.
        return LineClass.PUSH
    if mnemonic in STRING_OPS or ins.prefix:
        return LineClass.STORE
    if mnemonic in _X87_MEM_WRITERS_DP:
        return (
            LineClass.STORE
            if any(isinstance(op, Mem) for op in ins.operands)
            else LineClass.NONE
        )
    if mnemonic in ("cmp", "test"):
        return LineClass.NONE
    if ins.operands and isinstance(ins.operands[0], Mem) and not ins.is_jump:
        return LineClass.STORE
    return LineClass.NONE


def line_class(ins: DecodedInstruction, *, promote: bool = False) -> LineClass:
    """The instruction's observable alignment class. With the frame promoted,
    a store to a frame slot is no observable: it may pair with any instruction,
    or with none."""
    classification = _base_line_class(ins)
    if not promote or classification is not LineClass.STORE:
        return classification
    match ins.operands:
        case (
            Mem(segment="", terms=(ScaledReg("esp" | "ebp", 1),), symbols=()),
            *_,
        ) if (
            not ins.prefix and ins.mnemonic not in STRING_OPS
        ):
            return LineClass.NONE
    return classification


def instruction_skeleton(ins: DecodedInstruction) -> tuple:
    """Mnemonic plus operands with register identities erased: a register
    keeps its width, a memory operand the multiset of its scales."""
    shape: list[Operand] = []
    for op in ins.operands:
        match op:
            case Reg(name) if name in REGISTERS:
                shape.append(Reg(REGISTERS[name][1]))
            case Mem(terms=terms):
                shape.append(
                    replace(
                        op,
                        terms=tuple(
                            sorted(ScaledReg("", term.scale) for term in terms)
                        ),
                    )
                )
            case _:
                shape.append(op)
    return (ins.prefix, ins.mnemonic, tuple(shape))


def _sub_cost(
    ins_o: DecodedInstruction,
    ins_r: DecodedInstruction,
    *,
    promote: bool,
) -> int | None:
    if instruction_match_key(ins_o) == instruction_match_key(ins_r):
        return _SUB_EXACT
    class_o = line_class(ins_o, promote=promote)
    class_r = line_class(ins_r, promote=promote)
    if class_o is not class_r:
        return None
    if instruction_skeleton(ins_o) == instruction_skeleton(ins_r):
        return _SUB_SKELETON
    head_o = ins_o.prefix or ins_o.mnemonic
    head_r = ins_r.prefix or ins_r.mnemonic
    if head_o == head_r or (head_o in JCC_MNEMONICS and head_r in JCC_MNEMONICS):
        return _SUB_MNEMONIC
    return _SUB_CLASS


def _gap_cost(ins: DecodedInstruction, *, promote: bool) -> int | None:
    classification = line_class(ins, promote=promote)
    if classification in (LineClass.NONE, LineClass.PUSH) and may_be_one_sided(ins):
        return _GAP
    return None


def align_block_lines(
    lines_o: Sequence[DecodedInstruction],
    lines_r: Sequence[DecodedInstruction],
    *,
    promote: bool = False,
) -> BlockAlignment | None:
    """Pair up two blocks' instructions with a cost-minimizing alignment.
    Returns block-local index pairs and their cost; None when the blocks
    cannot be aligned (observable-class counts differ, or the blocks are
    absurdly large)."""
    n, m = len(lines_o), len(lines_r)
    if n * m > 1_000_000:
        return None
    inf = float("inf")
    # cost[i][j]: best cost aligning lines_o[:i] with lines_r[:j].
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    gap_o = [_gap_cost(line, promote=promote) for line in lines_o]
    gap_r = [_gap_cost(line, promote=promote) for line in lines_r]
    for i in range(1, n + 1):
        gap = gap_o[i - 1]
        if gap is not None:
            cost[i][0] = cost[i - 1][0] + gap
    for j in range(1, m + 1):
        gap = gap_r[j - 1]
        if gap is not None:
            cost[0][j] = cost[0][j - 1] + gap
    for i in range(1, n + 1):
        row = cost[i]
        prev = cost[i - 1]
        line_o = lines_o[i - 1]
        for j in range(1, m + 1):
            best = inf
            gap_i = gap_o[i - 1]
            gap_j = gap_r[j - 1]
            if gap_i is not None:
                best = prev[j] + gap_i
            if gap_j is not None:
                best = min(best, row[j - 1] + gap_j)
            sub = _sub_cost(line_o, lines_r[j - 1], promote=promote)
            if sub is not None:
                best = min(best, prev[j - 1] + sub)
            row[j] = best
    if cost[n][m] == inf:
        return None
    # Reconstruct.
    result: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = _sub_cost(lines_o[i - 1], lines_r[j - 1], promote=promote)
            if sub is not None and cost[i][j] == cost[i - 1][j - 1] + sub:
                result.append(AlignedPair(i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        gap = gap_o[i - 1] if i > 0 else None
        if gap is not None and cost[i][j] == cost[i - 1][j] + gap:
            result.append(AlignedPair(i - 1, None))
            i -= 1
            continue
        result.append(AlignedPair(None, j - 1))
        j -= 1
    result.reverse()
    return BlockAlignment(tuple(result), int(cost[n][m]))
