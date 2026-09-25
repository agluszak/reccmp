"""Bit-vector equivalence of symbolic values (optional, needs z3-solver).

The verifier compares the values two instruction streams observe
structurally: the same computation spelled differently (`and al, 1` on the
loaded byte versus `and eax, 1` then reading `al`) looks like a difference.
Where both values are built only from operations modelled here with their
exact x86 semantics, Z3 decides whether they are equal for every input;
everything else becomes an unconstrained variable shared by identical terms
on both sides. A proof is only accepted when Z3 says `unsat` for their
difference within a small time budget: unknown, a timeout or any term this
module cannot lower keeps the structural answer.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from functools import lru_cache
from typing import Any

try:
    import z3  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when z3 is absent
    z3 = None

from reccmp.compare.asm.verifier.state import WIDTHS

_TIMEOUT_MS = 200
_PART_BITS = {"l8": 8, "h8": 8, "r16": 16}
_EXTEND_BITS = {"byte": 8, "l8": 8, "h8": 8, "word": 16, "r16": 16}
_LOAD_BITS = {"byte": 8, "word": 16, "dword": 32}
_PREDICATES = {"eq", "ne", "lt_u", "le_u", "lt_s", "le_s"}


def available() -> bool:
    return z3 is not None


class _Unsupported(Exception):
    pass


class _Lowering:
    """One comparison's translation: identical opaque terms on either side
    share one variable."""

    def __init__(self) -> None:
        self.variables: dict[Hashable, Any] = {}

    def opaque(self, value: Hashable, bits: int = 32):
        key = (value, bits)
        if key not in self.variables:
            self.variables[key] = z3.BitVec(f"v{len(self.variables)}", bits)
        return self.variables[key]

    def opaque_bool(self, value: Hashable):
        key = (value, "bool")
        if key not in self.variables:
            self.variables[key] = z3.Bool(f"b{len(self.variables)}")
        return self.variables[key]

    # A value is an expression of a known width or an integer constant
    # whose width its context decides.
    def value(self, value: Any):
        # pylint: disable=too-many-return-statements,too-many-branches
        if not isinstance(value, tuple) or not value:
            raise _Unsupported
        tag = value[0]
        if tag == "imm" and isinstance(value[1], int):
            return value[1]
        if tag == "load":
            bits = _LOAD_BITS.get(value[2]) if len(value) > 2 else None
            if bits is None:
                raise _Unsupported
            return self.opaque(value, bits)
        if tag in _PART_BITS and len(value) == 2:
            whole = self.sized(value[1], 32)
            low = 8 if tag == "h8" else 0
            return z3.Extract(low + _PART_BITS[tag] - 1, low, whole)
        if tag in ("ins_l8", "ins_h8", "ins_r16") and len(value) == 3:
            return self.insert(tag[4:], value[1], value[2])
        if tag == "add" and len(value) >= 3:
            return self.fold(value[1:], lambda a, b: a + b)
        if tag in ("and", "or", "xor", "imul", "sub", "imul3") and len(value) == 3:
            operation = {
                "and": lambda a, b: a & b,
                "or": lambda a, b: a | b,
                "xor": lambda a, b: a ^ b,
                "imul": lambda a, b: a * b,
                "imul3": lambda a, b: a * b,
                "sub": lambda a, b: a - b,
            }[tag]
            return self.fold(value[1:], operation)
        if tag in ("shl", "shr", "sar") and len(value) == 3:
            return self.shift(tag, value[1], value[2])
        if tag in ("inc", "dec", "neg", "not") and len(value) == 2:
            operand = self.value(value[1])
            if isinstance(operand, int):
                raise _Unsupported
            return {
                "inc": lambda x: x + 1,
                "dec": lambda x: x - 1,
                "neg": lambda x: -x,
                "not": lambda x: ~x,
            }[tag](operand)
        if tag in ("movzx", "movsx") and len(value) == 3:
            bits = _EXTEND_BITS.get(value[1])
            if bits is None:
                raise _Unsupported
            source = self.sized(value[2], bits)
            extend = z3.ZeroExt if tag == "movzx" else z3.SignExt
            return extend(32 - bits, source)
        if tag == "setcc" and len(value) == 2:
            return z3.If(
                self.predicate(value[1]), z3.BitVecVal(1, 8), z3.BitVecVal(0, 8)
            )
        if tag == "cdq" and len(value) == 2:
            eax = self.sized(value[1], 32)
            return z3.If(eax < 0, z3.BitVecVal(-1, 32), z3.BitVecVal(0, 32))
        if tag == "cwde" and len(value) == 2:
            return z3.SignExt(16, self.sized(value[1], 16))
        return self.opaque(value)

    def sized(self, value: Any, bits: int):
        """``value`` at exactly ``bits``: constants take the width; a wider
        expression contributes its low bits (a partial register or a
        narrower store); a narrower one cannot."""
        lowered = self.value(value)
        if isinstance(lowered, int):
            return z3.BitVecVal(lowered, bits)
        width = lowered.size()
        if width == bits:
            return lowered
        if width > bits:
            return z3.Extract(bits - 1, 0, lowered)
        raise _Unsupported

    def fold(self, operands: Sequence[Any], operation):
        lowered = [self.value(item) for item in operands]
        widths = {item.size() for item in lowered if not isinstance(item, int)}
        if len(widths) != 1:
            raise _Unsupported  # all constants, or mixed widths
        (bits,) = widths
        terms = [
            z3.BitVecVal(item, bits) if isinstance(item, int) else item
            for item in lowered
        ]
        result = terms[0]
        for term in terms[1:]:
            result = operation(result, term)
        return result

    def shift(self, tag: str, value: Any, count: Any):
        operand = self.value(value)
        if isinstance(operand, int):
            raise _Unsupported
        bits = operand.size()
        # x86 masks the count to five bits whatever the operand width; a
        # count past the width then shifts everything out, as in Z3.
        amount = self.value(count)
        if isinstance(amount, int):
            shift = z3.BitVecVal(amount & 31, bits)
        else:
            masked = z3.Extract(4, 0, amount) if amount.size() > 5 else amount
            shift = z3.ZeroExt(bits - masked.size(), masked)
        if tag == "shl":
            return operand << shift
        if tag == "shr":
            return z3.LShR(operand, shift)
        return operand >> shift

    def insert(self, part: str, old: Any, new: Any):
        whole = self.sized(old, 32)
        if part == "l8":
            return z3.Concat(z3.Extract(31, 8, whole), self.sized(new, 8))
        if part == "h8":
            return z3.Concat(
                z3.Extract(31, 16, whole), self.sized(new, 8), z3.Extract(7, 0, whole)
            )
        return z3.Concat(z3.Extract(31, 16, whole), self.sized(new, 16))

    def predicate(self, predicate: Any):
        if not isinstance(predicate, tuple) or not predicate:
            raise _Unsupported
        tag = predicate[0]
        if tag not in _PREDICATES:
            return self.opaque_bool(predicate)
        if tag in ("eq", "ne"):
            if len(predicate) < 2 or len(predicate[1]) != 2:
                raise _Unsupported
            a, b = predicate[1]
            width = predicate[2] if len(predicate) > 2 else None
        else:
            if len(predicate) < 3:
                raise _Unsupported
            a, b = predicate[1], predicate[2]
            width = predicate[3] if len(predicate) > 3 else None
        left, right = self.operands(a, b, width)
        return {
            "eq": lambda: left == right,
            "ne": lambda: left != right,
            "lt_u": lambda: z3.ULT(left, right),
            "le_u": lambda: z3.ULE(left, right),
            "lt_s": lambda: left < right,
            "le_s": lambda: left <= right,
        }[tag]()

    def operands(self, a: Any, b: Any, width: Any):
        if isinstance(width, int):
            bits = 8 * width
            return self.sized(a, bits), self.sized(b, bits)
        left, right = self.value(a), self.value(b)
        if isinstance(left, int) and isinstance(right, int):
            raise _Unsupported
        if isinstance(left, int):
            return z3.BitVecVal(left, right.size()), right
        if isinstance(right, int):
            return left, z3.BitVecVal(right, left.size())
        if left.size() != right.size():
            raise _Unsupported
        return left, right


def _proved(claim) -> bool:
    solver = z3.Solver()
    solver.set("timeout", _TIMEOUT_MS)
    solver.add(z3.Not(claim))
    return solver.check() == z3.unsat


@lru_cache(maxsize=4096)
def values_equal(a: Any, b: Any, bits: int | None = None) -> bool:
    """Whether two symbolic values are equal for every input (at ``bits``,
    their natural width when None). False unless proven."""
    if a == b:
        return True
    if z3 is None:
        return False
    lowering = _Lowering()
    try:
        if bits is None:
            left, right = lowering.operands(a, b, None)
        else:
            left, right = lowering.sized(a, bits), lowering.sized(b, bits)
    except _Unsupported:
        return False
    return _proved(left == right)


@lru_cache(maxsize=4096)
def predicates_equal(a: Any, b: Any) -> bool:
    """Whether two branch predicates decide the same way for every input."""
    if a == b:
        return True
    if z3 is None:
        return False
    lowering = _Lowering()
    try:
        left, right = lowering.predicate(a), lowering.predicate(b)
    except _Unsupported:
        return False
    return _proved(left == right)


def entries_equal(entry_o: Any, entry_r: Any) -> bool:
    """Whether two observations agree, up to bit-vector equivalence of the
    values they carry: return values, stored values and branch predicates.
    Every other part (tags, addresses, widths, destinations) must be equal."""
    if entry_o == entry_r:
        return True
    if z3 is None or not isinstance(entry_o, tuple) or not isinstance(entry_r, tuple):
        return False
    if not entry_o or len(entry_o) != len(entry_r) or entry_o[0] != entry_r[0]:
        return False
    tag = entry_o[0]
    if tag == "retval":
        return all(values_equal(o, r) for o, r in zip(entry_o[1:], entry_r[1:]))
    bits = WIDTHS.get(entry_o[2]) if tag == "store" and len(entry_o) == 4 else None
    if bits is not None:
        return entry_o[1:3] == entry_r[1:3] and values_equal(
            entry_o[3], entry_r[3], 8 * bits
        )
    return (
        tag == "branch"
        and len(entry_o) == 3
        and entry_o[2] == entry_r[2]
        and predicates_equal(entry_o[1], entry_r[1])
    )


def observations_equal(obs_o: Sequence[Any], obs_r: Sequence[Any]) -> bool:
    return len(obs_o) == len(obs_r) and all(
        entries_equal(o, r) for o, r in zip(obs_o, obs_r)
    )
