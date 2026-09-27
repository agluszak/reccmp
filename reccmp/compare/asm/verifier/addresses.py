"""Shape of symbolic memory addresses: flattening, stack roots, and whether two accesses overlap."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Init:
    family: str

    @property
    def is_scratch(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class CallResult:
    site: int
    register: str

    @property
    def is_scratch(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class StringResult:
    site: int
    family: str

    @property
    def is_scratch(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class X87Slot:
    index: int


@dataclass(frozen=True, slots=True)
class Resync:
    site: int
    location: Hashable

    @property
    def is_scratch(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Phi:
    block: int
    class_id: int
    settled: bool = False

    @property
    def is_scratch(self) -> bool:
        return self.settled


@dataclass(frozen=True, slots=True)
class Constant:
    value: int


@dataclass(frozen=True, slots=True)
class SymbolValue:
    identity: Hashable


@dataclass(frozen=True, slots=True)
class AddressTerm:
    value: Value
    scale: int


@dataclass(frozen=True, slots=True)
class MemoryAddress:
    segment: str
    terms: tuple[AddressTerm, ...]
    displacement: int | Value
    symbols: tuple


@dataclass(frozen=True, slots=True)
class AddressValue:
    address: Value


@dataclass(frozen=True, slots=True)
class Load:
    address: Value
    width: str
    generation: MemoryGeneration


@dataclass(frozen=True, slots=True)
class StackOffset:
    base: Value
    offset: int


@dataclass(frozen=True, slots=True)
class Slot:
    index: int


class OperationKind(Enum):
    ADD = "add"
    AND = "and"
    OR = "or"
    XOR = "xor"
    IMUL = "imul"
    IMUL3 = "imul3"
    MUL = "mul"
    SUB = "sub"
    SHL = "shl"
    SHR = "shr"
    SAR = "sar"
    ROL = "rol"
    ROR = "ror"
    ADC = "adc"
    SBB = "sbb"
    INC = "inc"
    DEC = "dec"
    NEG = "neg"
    NOT = "not"
    LOOP_DECREMENT = "loopdec"
    CDQ = "cdq"
    CWDE = "cwde"


@dataclass(frozen=True, slots=True)
class Operation:
    kind: OperationKind
    operands: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class UnaryOperation:
    kind: OperationKind
    operand: Value


class RegisterPart(Enum):
    LOW8 = "l8"
    HIGH8 = "h8"
    LOW16 = "r16"


@dataclass(frozen=True, slots=True)
class Extract:
    part: RegisterPart
    whole: Value


@dataclass(frozen=True, slots=True)
class Insert:
    part: RegisterPart
    old: Value
    new: Value


class ExtendKind(Enum):
    MOVZX = "movzx"
    MOVSX = "movsx"


@dataclass(frozen=True, slots=True)
class Extend:
    kind: ExtendKind
    width: str
    source: Value


class CompareKind(Enum):
    EQ = "eq"
    NE = "ne"
    LT_U = "lt_u"
    LE_U = "le_u"
    LT_S = "lt_s"
    LE_S = "le_s"


@dataclass(frozen=True, slots=True)
class Compare:
    kind: CompareKind
    left: Value
    right: Value
    width: int | str | None = None


@dataclass(frozen=True, slots=True)
class ConditionCode:
    condition: str
    flags: Value
    carry: Value | None = None


@dataclass(frozen=True, slots=True)
class SetCondition:
    predicate: Value


@dataclass(frozen=True, slots=True)
class Select:
    predicate: Value
    fallthrough: Value
    taken: Value


class ProductPart(Enum):
    LOW = "lo"
    HIGH = "hi"
    QUOTIENT = "quot"
    REMAINDER = "rem"


@dataclass(frozen=True, slots=True)
class MultiplyResult:
    signed: bool
    part: ProductPart
    operands: tuple[Value, Value]


@dataclass(frozen=True, slots=True)
class DivideResult:
    signed: bool
    part: ProductPart
    high: Value
    low: Value
    divisor: Value


@dataclass(frozen=True, slots=True)
class FlagsResult:
    operation: Value


@dataclass(frozen=True, slots=True)
class CompareFlags:
    left: Value
    right: Value
    width: int | str | None = None


@dataclass(frozen=True, slots=True)
class TestFlags:
    left: Value
    right: Value
    width: int | str | None = None


@dataclass(frozen=True, slots=True)
class CarryResult:
    operation: Value


@dataclass(frozen=True, slots=True)
class CarryCleared:
    pass


@dataclass(frozen=True, slots=True)
class SahfFlags:
    value: Value
    previous: Value


@dataclass(frozen=True, slots=True)
class SahfCarry:
    value: Value


class OpaqueKind(Enum):
    DIVISION_FLAGS = "undef_flags"
    DIVISION_CARRY = "undef_cf"
    CALL_FLAGS = "callflags"
    CALL_CARRY = "callcf"
    STRING_FLAGS = "strflags"
    STRING_CARRY = "strcf"
    META_RESULT = "metastep"
    META_FLAGS = "metastep_flags"
    META_CARRY = "metastep_cf"
    HAVOC_REGISTER = "havoc"
    HAVOC_FLAGS = "havoc_flags"
    HAVOC_CARRY = "havoc_cf"
    HAVOC_FPU_FLAGS = "havoc_fpuflags"


@dataclass(frozen=True, slots=True)
class OpaqueValue:
    kind: OpaqueKind
    site: int
    location: Hashable | None = None


@dataclass(frozen=True, slots=True)
class CallStack:
    """The stack pointer after a call, still rooted at its incoming value."""

    site: int
    incoming: Value


@dataclass(frozen=True, slots=True)
class ReceiverLoad:
    """The canonical identity of a receiver read whose value cannot forward."""

    address: Value
    width: str


@dataclass(frozen=True, slots=True)
class VirtualCall:
    receiver: Value
    displacement: int


@dataclass(frozen=True, slots=True)
class CallThrough:
    slot: Hashable


@dataclass(frozen=True, slots=True)
class CallArgument:
    """A promoted frame slot rewritten by a callee."""

    site: int
    offset: int
    width: int


@dataclass(frozen=True, slots=True)
class CfgMemoryInit:
    """The memory state at the start of a CFG verification scope."""


@dataclass(frozen=True, slots=True)
class CfgMemoryPhi:
    """A joined memory state at a CFG block."""

    block: int


@dataclass(frozen=True, slots=True)
class MemoryStore:
    """The memory generation after one committed store."""

    site: Hashable
    ordinal: int


@dataclass(frozen=True, slots=True)
class ScratchStore:
    """The memory generation after a private one-sided push."""

    site: int


@dataclass(frozen=True, slots=True)
class MemoryClobber:
    """The memory generation after an opaque write."""

    site: Hashable
    ordinal: Hashable | None = None


MemoryGeneration: TypeAlias = (
    CfgMemoryInit | CfgMemoryPhi | MemoryStore | ScratchStore | MemoryClobber
)


class FloatConstantKind(Enum):
    ONE = "fld1"
    ZERO = "fldz"
    PI = "fldpi"
    LOG2E = "fldl2e"
    LOG2T = "fldl2t"
    LOG10_2 = "fldlg2"
    LN2 = "fldln2"


class FloatOperationKind(Enum):
    ADD = "fadd"
    MULTIPLY = "fmul"
    SUBTRACT = "fsub"
    DIVIDE = "fdiv"
    TO_INTEGER = "fist"
    CHANGE_SIGN = "fchs"
    ABSOLUTE = "fabs"
    SQUARE_ROOT = "fsqrt"
    ROUND_INTEGER = "frndint"
    COSINE = "fcos"
    SINE = "fsin"
    TANGENT = "ftan"
    TWO_X_MINUS_ONE = "f2xm1"
    PARTIAL_REMAINDER = "fprem"
    SCALE = "fscale"
    PARTIAL_ARCTANGENT = "fpatan"
    Y_LOG2_X = "fyl2x"


@dataclass(frozen=True, slots=True)
class FloatConstant:
    kind: FloatConstantKind


@dataclass(frozen=True, slots=True)
class FloatOperation:
    kind: FloatOperationKind
    operands: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class X87EpochJoin:
    block: int


@dataclass(frozen=True, slots=True)
class DeepFloat:
    epoch: int | X87EpochJoin
    index: int


@dataclass(frozen=True, slots=True)
class FloatCompare:
    left: Value
    right: Value


@dataclass(frozen=True, slots=True)
class FloatStatusWord:
    flags: Value


@dataclass(frozen=True, slots=True)
class FloatControlWord:
    pass


Value: TypeAlias = (
    Init
    | CallResult
    | StringResult
    | Resync
    | Phi
    | Constant
    | SymbolValue
    | MemoryAddress
    | AddressValue
    | Load
    | StackOffset
    | Slot
    | Operation
    | UnaryOperation
    | Extract
    | Insert
    | Extend
    | Compare
    | ConditionCode
    | SetCondition
    | Select
    | MultiplyResult
    | DivideResult
    | FlagsResult
    | CompareFlags
    | TestFlags
    | CarryResult
    | CarryCleared
    | SahfFlags
    | SahfCarry
    | OpaqueValue
    | FloatConstant
    | FloatOperation
    | DeepFloat
    | FloatCompare
    | FloatStatusWord
    | FloatControlWord
    | CallStack
    | ReceiverLoad
    | VirtualCall
    | CallThrough
    | CallArgument
)

_VALUE_TYPES = (
    Init,
    CallResult,
    StringResult,
    Resync,
    Phi,
    Constant,
    SymbolValue,
    MemoryAddress,
    AddressValue,
    Load,
    StackOffset,
    Slot,
    Operation,
    UnaryOperation,
    Extract,
    Insert,
    Extend,
    Compare,
    ConditionCode,
    SetCondition,
    Select,
    MultiplyResult,
    DivideResult,
    FlagsResult,
    CompareFlags,
    TestFlags,
    CarryResult,
    CarryCleared,
    SahfFlags,
    SahfCarry,
    OpaqueValue,
    FloatConstant,
    FloatOperation,
    DeepFloat,
    FloatCompare,
    FloatStatusWord,
    FloatControlWord,
    CallStack,
    ReceiverLoad,
    VirtualCall,
    CallThrough,
    CallArgument,
)


def is_value(value: object) -> bool:
    return isinstance(value, _VALUE_TYPES)


def value_children(value: Value) -> tuple[Value, ...]:
    # pylint: disable=too-many-return-statements
    match value:
        case MemoryAddress(terms=terms, displacement=displacement):
            children = tuple(term.value for term in terms)
            return (
                children if isinstance(displacement, int) else (*children, displacement)
            )
        case AddressValue(address):
            return (address,)
        case Load(address):
            return (address,)
        case CallStack(incoming=incoming) | ReceiverLoad(address=incoming):
            return (incoming,)
        case VirtualCall(receiver):
            return (receiver,)
        case (
            StackOffset(base)
            | Extract(_, base)
            | UnaryOperation(_, base)
            | Extend(_, _, base)
            | SetCondition(base)
        ):
            return (base,)
        case Operation(operands=operands) | MultiplyResult(operands=operands):
            return operands
        case Insert(old=old, new=new) | Compare(left=old, right=new):
            return (old, new)
        case ConditionCode(flags=flags, carry=carry):
            return (flags,) if carry is None else (flags, carry)
        case Select(predicate, fallthrough, taken):
            return (predicate, fallthrough, taken)
        case DivideResult(high=high, low=low, divisor=divisor):
            return (high, low, divisor)
        case FlagsResult(operation) | CarryResult(operation):
            return (operation,)
        case CompareFlags(left, right) | TestFlags(left, right):
            return (left, right)
        case SahfFlags(value, previous):
            return (value, previous)
        case SahfCarry(value) | FloatStatusWord(value):
            return (value,)
        case FloatOperation(operands=operands):
            return operands
        case FloatCompare(left, right):
            return (left, right)
    return ()


def _foldable_address_value(value: Value, scale: int, segment: str) -> bool:
    if scale != 1:
        return False
    if isinstance(value, AddressValue) and isinstance(value.address, MemoryAddress):
        return value.address.segment in ("", segment)
    return isinstance(value, Operation) and value.kind is OperationKind.ADD


def flatten_mem(addr: Value) -> MemoryAddress:
    """Fold scale-1 base registers that hold a computed address (from lea)
    into the memory expression itself, so `[esi]` with esi = &[ebx + 0x1c6]
    compares as `[ebx + 0x1c6]`."""
    mem = (
        addr
        if isinstance(addr, MemoryAddress)
        else MemoryAddress("", (AddressTerm(addr, 1),), 0, ())
    )
    seg, terms, disp, syms = (
        mem.segment,
        mem.terms,
        mem.displacement,
        mem.symbols,
    )
    for _ in range(8):
        folded = next(
            (
                term
                for term in terms
                if _foldable_address_value(term.value, term.scale, seg)
            ),
            None,
        )
        if folded is None:
            break
        value = folded.value
        terms = tuple(term for term in terms if term is not folded)
        if isinstance(value, AddressValue):
            assert isinstance(value.address, MemoryAddress)
            inner = value.address
            terms += inner.terms
            if isinstance(disp, int) and isinstance(inner.displacement, int):
                disp += inner.displacement
            else:
                break
            syms = tuple(sorted(set(syms) | set(inner.symbols), key=repr))
            seg = seg or inner.segment
        else:
            assert isinstance(value, Operation) and value.kind is OperationKind.ADD
            for leaf in value.operands:
                if isinstance(leaf, Constant):
                    assert isinstance(disp, int)
                    disp += leaf.value
                else:
                    terms += (AddressTerm(leaf, 1),)
    return MemoryAddress(seg, tuple(sorted(terms, key=repr)), disp, syms)


def stack_rooted(value: Value) -> bool:
    # pylint: disable=too-many-return-statements
    """Is the value derived from the stack pointer or frame pointer?"""
    if isinstance(value, Init):
        return value.family in ("sp", "bp")
    if isinstance(value, StackOffset):
        return stack_rooted(value.base)
    if isinstance(value, AddressValue):
        if isinstance(value.address, MemoryAddress):
            return any(stack_rooted(term.value) for term in value.address.terms)
        return stack_rooted(value.address)
    match value:
        case Operation(OperationKind.ADD, operands):
            return any(stack_rooted(child) for child in operands)
        case Insert(RegisterPart.LOW16, old, new):
            return stack_rooted(old) or stack_rooted(new)
    return False


def _is_pure_global(mem: MemoryAddress) -> bool:
    return not mem.terms and bool(mem.symbols)


def unwind_spadd(value: Value, offset: int = 0) -> tuple[Value, int]:
    while isinstance(value, StackOffset):
        offset += value.offset
        value = value.base
    return (value, offset)


def _signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >> 31 else value


def constant_offset(value: Value) -> tuple[Value, int]:
    """(root, offset) of a value that is a root plus a constant, whether
    built by push/pop (``spadd``) or by ``add``/``sub`` of an immediate."""
    offset = 0
    while True:
        if isinstance(value, StackOffset):
            offset += value.offset
            value = value.base
        elif (
            isinstance(value, Operation)
            and value.kind is OperationKind.SUB
            and len(value.operands) == 2
            and isinstance(value.operands[1], Constant)
        ):
            offset -= _signed32(value.operands[1].value)
            value = value.operands[0]
        elif isinstance(value, Operation) and value.kind is OperationKind.ADD:
            terms = [term for term in value.operands if not isinstance(term, Constant)]
            if len(terms) != 1:
                break
            offset += sum(
                _signed32(term.value)
                for term in value.operands
                if isinstance(term, Constant)
            )
            value = terms[0]
        else:
            break
    return (value, offset)


def abs_stack_offset(addr: Value, is_slot) -> tuple[Value, int] | None:
    """Resolve an access to (root value, byte offset) when its address is a
    plain chain of constant adjustments over one root — a push/pop slot, or
    a single-register memory operand like [ebp - 8] or [esp + 4]."""
    if is_slot:
        return unwind_spadd(addr)
    mem = flatten_mem(addr)
    if len(mem.terms) == 1 and not mem.symbols and isinstance(mem.displacement, int):
        term = mem.terms[0]
        if term.scale == 1:
            root, offset = unwind_spadd(term.value)
            return (root, offset + mem.displacement)
    return None


def _ranges_disjoint(a_disp, a_width, b_disp, b_width) -> bool:
    if isinstance(a_disp, int) and isinstance(b_disp, int):
        if a_width is None or b_width is None:
            return False
        return a_disp + a_width <= b_disp or b_disp + b_width <= a_disp
    if a_disp == b_disp:
        return False
    # Alpha-renamed frame slots: distinct slot ids are distinct locals
    # (their non-overlap is validated by _slots_consistent).
    return isinstance(a_disp, Slot) and isinstance(b_disp, Slot)


def mem_disjoint(a: tuple, b: tuple) -> bool:
    """Can the two memory accesses be proven non-overlapping?
    Accesses are (address value, width, stack_kind) where stack_kind is
    False for ordinary operands, "push" for a fresh slot below the stack
    pointer, "pop" for a read of the top of the stack."""
    # pylint: disable=too-many-return-statements
    a_addr, a_width, a_stack = a
    b_addr, b_width, b_stack = b

    if a_stack or b_stack:
        a_res = abs_stack_offset(a_addr, a_stack)
        b_res = abs_stack_offset(b_addr, b_stack)
        if (
            a_res is not None
            and b_res is not None
            and a_res[0] == b_res[0]
            and _ranges_disjoint(a_res[1], a_width, b_res[1], b_width)
        ):
            return True
        if a_stack and b_stack:
            return False
        other = flatten_mem(b_addr if a_stack else a_addr)
        # A stack slot never overlaps a named global. An access through an
        # unknown pointer, however, must be assumed to alias the stack:
        # nothing proves an incoming pointer cannot equal the slot address.
        return _is_pure_global(other)

    a_mem = flatten_mem(a_addr)
    b_mem = flatten_mem(b_addr)

    if a_mem.segment != b_mem.segment:
        # Different segment prefixes: assume they can alias.
        return False

    # Same base values (symbolically identical registers/symbols): the two
    # accesses differ only by constant displacement.
    if a_mem.terms == b_mem.terms and a_mem.symbols == b_mem.symbols:
        return _ranges_disjoint(
            a_mem.displacement, a_width, b_mem.displacement, b_width
        )

    global_a = _is_pure_global(a_mem)
    global_b = _is_pure_global(b_mem)

    if global_a and global_b and a_mem.symbols != b_mem.symbols:
        # Two different named globals do not overlap.
        return True

    # Stack/frame memory never overlaps a named global.
    stack_a = any(stack_rooted(term.value) for term in a_mem.terms)
    stack_b = any(stack_rooted(term.value) for term in b_mem.terms)
    if (global_a and stack_b) or (global_b and stack_a):
        return True

    return False
