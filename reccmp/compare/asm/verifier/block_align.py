"""Cost-minimizing alignment of two paired blocks' instructions."""

from __future__ import annotations

from dataclasses import dataclass

from reccmp.compare.asm.ir import (
    ResolvedAsm,
    instruction_at,
)
from reccmp.compare.asm.model import (
    REGISTERS,
    Instruction,
    Reject,
)
from reccmp.compare.asm.verifier.state import (
    JCC_MNEMONICS,
    STRING_OPS,
)

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


@dataclass(frozen=True)
class DpLine:
    """Alignment view of one instruction, derived once from its structure."""

    display: str
    # Leading token of the instruction text: the prefix when present
    # ("rep", "lock"), else the mnemonic.
    head: str
    # Coarse observable class: lines with an observable effect only pair
    # within their class and never go one-sided (the verifier would reject
    # that anyway).
    line_class: str
    # Instruction shape with register identities erased; None if unknown.
    skeleton: tuple | None


def _dp_line_class(ins: Instruction) -> str:
    # pylint: disable=too-many-return-statements
    mnemonic = ins.mnemonic
    if mnemonic == "call":
        return "call"
    if mnemonic == "push":
        # Pushes pair with pushes, but may also go one-sided (a scratch
        # spill on one side only); the verifier gates the soundness.
        return "push"
    if mnemonic in STRING_OPS or ins.prefix:
        return "store"
    if mnemonic in _X87_MEM_WRITERS_DP:
        return "store" if any(op[0] == "mem" for op in ins.operands) else "none"
    if mnemonic in ("cmp", "test"):
        return "none"
    if ins.operands and ins.operands[0][0] == "mem" and not mnemonic.startswith("j"):
        return "store"
    return "none"


def _dp_skeleton(ins: Instruction) -> tuple:
    """Mnemonic plus operand kinds, keeping immediates, symbols, widths,
    displacements and scale multisets."""
    shape: list[tuple] = []
    for op in ins.operands:
        kind = op[0]
        if kind == "reg":
            shape.append(("reg", REGISTERS[op[1]][1]))
        elif kind == "mem":
            _, size, seg, reg_terms, disp, syms = op
            shape.append(
                (
                    "mem",
                    size,
                    seg,
                    tuple(sorted(scale for _, scale in reg_terms)),
                    disp,
                    syms,
                )
            )
        else:
            shape.append(op)
    return (ins.prefix, ins.mnemonic, tuple(shape))


def dp_line(stream: ResolvedAsm, index: int) -> DpLine:
    display = stream.displays[index]
    try:
        ins = instruction_at(stream, index)
    except (Reject, IndexError, KeyError, ValueError, TypeError):
        # Only reachable for text streams the model cannot parse; decoded
        # IR rows always carry an instruction. Calls and pushes keep their
        # class so an unmodeled operand does not change the pairing.
        head = display.partition(" ")[0]
        if head in ("call", "push"):
            line_class = head
        elif head in STRING_OPS or head.startswith("rep"):
            line_class = "store"
        else:
            line_class = "opaque"
        return DpLine(display, head, line_class, None)
    head = ins.prefix or ins.mnemonic
    return DpLine(display, head, _dp_line_class(ins), _dp_skeleton(ins))


def _dp_sub_cost(line_o: DpLine, line_r: DpLine) -> float | None:
    if line_o.display == line_r.display:
        return _SUB_EXACT
    if line_o.line_class != line_r.line_class or line_o.line_class == "opaque":
        return None
    if line_o.skeleton is not None and line_o.skeleton == line_r.skeleton:
        return _SUB_SKELETON
    if line_o.head == line_r.head or (
        line_o.head in JCC_MNEMONICS and line_r.head in JCC_MNEMONICS
    ):
        return _SUB_MNEMONIC
    return _SUB_CLASS


def align_block_lines(
    lines_o: list[DpLine], lines_r: list[DpLine]
) -> list[tuple[int | None, int | None]] | None:
    """Pair up two blocks' instructions with a cost-minimizing alignment.
    Returns block-local index pairs; None when the blocks cannot be aligned
    (observable-class counts differ, or the blocks are absurdly large)."""
    n, m = len(lines_o), len(lines_r)
    if n * m > 1_000_000:
        return None
    inf = float("inf")
    # cost[i][j]: best cost aligning lines_o[:i] with lines_r[:j].
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    gap_classes = ("none", "push")
    gap_o = [_GAP if line.line_class in gap_classes else inf for line in lines_o]
    gap_r = [_GAP if line.line_class in gap_classes else inf for line in lines_r]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + gap_o[i - 1]
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + gap_r[j - 1]
    for i in range(1, n + 1):
        row = cost[i]
        prev = cost[i - 1]
        line_o = lines_o[i - 1]
        for j in range(1, m + 1):
            best = min(prev[j] + gap_o[i - 1], row[j - 1] + gap_r[j - 1])
            sub = _dp_sub_cost(line_o, lines_r[j - 1])
            if sub is not None:
                best = min(best, prev[j - 1] + sub)
            row[j] = best
    if cost[n][m] == inf:
        return None
    # Reconstruct.
    result: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = _dp_sub_cost(lines_o[i - 1], lines_r[j - 1])
            if sub is not None and cost[i][j] == cost[i - 1][j - 1] + sub:
                result.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + gap_o[i - 1]:
            result.append((i - 1, None))
            i -= 1
            continue
        result.append((None, j - 1))
        j -= 1
    result.reverse()
    return result
