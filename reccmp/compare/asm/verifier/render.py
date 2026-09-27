"""A C-like spelling of the verifier's symbolic values, for people."""

from __future__ import annotations

from typing import Any

from reccmp.compare.asm.model import FAMILY_REGISTER
from reccmp.compare.asm.verifier.addresses import (
    AddressTerm,
    AddressValue,
    CallArgument,
    CallResult,
    CallStack,
    CallThrough,
    CarryCleared,
    CarryResult,
    Compare,
    CompareFlags,
    CompareKind,
    ConditionCode,
    Constant,
    DeepFloat,
    DivideResult,
    Extend,
    ExtendKind,
    Extract,
    FlagsResult,
    FloatCompare,
    FloatConstant,
    FloatControlWord,
    FloatOperation,
    FloatStatusWord,
    Init,
    Insert,
    Load,
    MemoryAddress,
    MultiplyResult,
    OpaqueValue,
    Operation,
    OperationKind,
    Phi,
    ReceiverLoad,
    RegisterPart,
    Resync,
    SahfCarry,
    SahfFlags,
    Select,
    SetCondition,
    StackOffset,
    StringResult,
    SymbolValue,
    TestFlags,
    UnaryOperation,
    VirtualCall,
    X87Slot,
)

_BINARY = {
    OperationKind.AND: "&",
    OperationKind.OR: "|",
    OperationKind.XOR: "^",
    OperationKind.SUB: "-",
    OperationKind.IMUL: "*",
    OperationKind.IMUL3: "*",
    OperationKind.SHL: "<<",
    OperationKind.SHR: ">>u",
    OperationKind.SAR: ">>s",
    OperationKind.ROL: "rol",
    OperationKind.ROR: "ror",
    OperationKind.ADC: "+carry",
    OperationKind.SBB: "-carry",
}
_PREDICATE = {
    CompareKind.EQ: "==",
    CompareKind.NE: "!=",
    CompareKind.LT_U: "<u",
    CompareKind.LE_U: "<=u",
    CompareKind.LT_S: "<s",
    CompareKind.LE_S: "<=s",
}
_UNARY = {
    OperationKind.INC: "{} + 1",
    OperationKind.DEC: "{} - 1",
    OperationKind.NEG: "-{}",
    OperationKind.NOT: "~{}",
}
_PART = {
    RegisterPart.LOW8: "low8",
    RegisterPart.HIGH8: "bits8_15",
    RegisterPart.LOW16: "low16",
}
_INSERT = {
    RegisterPart.LOW8: "al",
    RegisterPart.HIGH8: "ah",
    RegisterPart.LOW16: "ax",
}


def render_number(value: int) -> str:
    return str(value) if -10 < value < 10 else hex(value)


_ENTRY_SP = AddressTerm(Init("sp"), 1)


def render(value: Any, depth: int = 0) -> str:
    """A C-like spelling of a verifier symbolic value."""
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
    if depth > 8:
        return "…"

    def inner(item: Any) -> str:
        return render(item, depth + 1)

    match value:
        case Constant(constant):
            return render_number(constant)
        case Init(family):
            return f"{FAMILY_REGISTER.get(family, family)}@entry"
        case Load(
            MemoryAddress("", (entry,), int() as displacement, ()), "dword", _
        ) if (entry == _ENTRY_SP and displacement > 0 and displacement % 4 == 0):
            return f"arg{displacement // 4}"
        case Load(address, size, _):
            return f"{size}[{_address(address, depth + 1)}]"
        case MemoryAddress():
            return _address(value, depth)
        case AddressValue(address):
            return f"&[{_address(address, depth + 1)}]"
        case SymbolValue(identity):
            return _symbol(identity)
        case StackOffset(base, offset):
            return f"{inner(base)} {'+' if offset >= 0 else '-'} {render_number(abs(offset))}"
        case Extract(part, whole):
            return f"{_PART[part]}({inner(whole)})"
        case Insert(part, old, new):
            return f"({inner(old)} with {_INSERT[part]} = {inner(new)})"
        case Operation(OperationKind.ADD, operands) if len(operands) >= 2:
            return "(" + " + ".join(inner(term) for term in operands) + ")"
        case Operation(kind, (left, right)) if kind in _BINARY:
            return f"({inner(left)} {_BINARY[kind]} {inner(right)})"
        case Operation(kind, operands):
            return f"{kind.value}(" + ", ".join(inner(term) for term in operands) + ")"
        case UnaryOperation(kind, operand) if kind in _UNARY:
            return "(" + _UNARY[kind].format(inner(operand)) + ")"
        case UnaryOperation(kind, operand):
            return f"{kind.value}({inner(operand)})"
        case Extend(ExtendKind.MOVZX, _, operand):
            return f"zext({inner(operand)})"
        case Extend(ExtendKind.MOVSX, _, operand):
            return f"sext({inner(operand)})"
        case CallResult(call, family):
            return f"{FAMILY_REGISTER.get(family, family)} after call@{call}"
        case StringResult(site, family):
            return f"{FAMILY_REGISTER.get(family, family)} after string@{site}"
        case Resync(site, X87Slot(index)):
            return f"st({index}) after resync@{site}"
        case Resync(site, location):
            return f"{location} after resync@{site}"
        case Phi(block, class_id):
            return f"join{block}#{class_id}"
        case Compare(kind, left, right, width):
            return _comparison(kind, left, right, width, depth)
        case ConditionCode(condition, flags, carry):
            text = f"{condition} of {inner(flags)}"
            return text if carry is None else f"{text}; cf {inner(carry)}"
        case SetCondition(predicate):
            return f"setcc({inner(predicate)})"
        case Select(predicate, fallthrough, taken):
            return f"({inner(taken)} if {inner(predicate)} else {inner(fallthrough)})"
        case MultiplyResult(signed, part, operands):
            mnemonic = "imul" if signed else "mul"
            return (
                f"{mnemonic}.{part.value}({inner(operands[0])}, {inner(operands[1])})"
            )
        case DivideResult(signed, part, high, low, divisor):
            mnemonic = "idiv" if signed else "div"
            return (
                f"{mnemonic}.{part.value}(({inner(high)}, {inner(low)}), "
                f"{inner(divisor)})"
            )
        case FlagsResult(operation) | CarryResult(operation):
            prefix = "flags" if isinstance(value, FlagsResult) else "carry"
            return f"{prefix}({inner(operation)})"
        case CompareFlags(left, right, width):
            bits = f" ({8 * width}-bit)" if isinstance(width, int) else ""
            return f"flags(cmp {inner(left)}, {inner(right)}{bits})"
        case TestFlags(left, right, width):
            bits = f" ({8 * width}-bit)" if isinstance(width, int) else ""
            return f"flags(test {inner(left)}, {inner(right)}{bits})"
        case CarryCleared():
            return "carry=0"
        case SahfFlags(source, previous):
            return f"sahf({inner(source)}; prior {inner(previous)})"
        case SahfCarry(source):
            return f"sahf-carry({inner(source)})"
        case OpaqueValue(kind, site, location):
            suffix = f".{location}" if location is not None else ""
            return f"{kind.value}{suffix}@{site}"
        case FloatConstant(kind):
            return kind.value
        case FloatOperation(kind, operands):
            return f"{kind.value}(" + ", ".join(inner(item) for item in operands) + ")"
        case DeepFloat(epoch, index):
            return f"st({index})@epoch{epoch}"
        case FloatCompare(left, right):
            return f"fcom({inner(left)}, {inner(right)})"
        case FloatStatusWord(flags):
            return f"fsw({inner(flags)})"
        case FloatControlWord():
            return "fcw"
        case CallStack(site, incoming):
            return f"esp after call@{site}({inner(incoming)})"
        case ReceiverLoad(address, width):
            return f"receiver {width}[{_address(address, depth + 1)}]"
        case VirtualCall(receiver, displacement):
            return f"virtual {inner(receiver)}+{render_number(displacement)}"
        case CallThrough(slot):
            return f"call through {_symbol(slot)}"
        case CallArgument(site, offset, width):
            return f"callarg{width}@{site}[{render_number(offset)}]"
        case _:
            return repr(value)


def _comparison(
    kind: CompareKind, left: Any, right: Any, width: int | str | None, depth: int
) -> str:
    bits = f" ({8 * width}-bit)" if isinstance(width, int) else ""
    return (
        f"{render(left, depth + 1)} {_PREDICATE[kind]} "
        f"{render(right, depth + 1)}{bits}"
    )


def _symbol(identity: Any) -> str:
    match identity:
        case ("entity", int() as address, int() as offset):
            return f"entity@0x{address:x}" + (
                f"+{render_number(offset)}" if offset else ""
            )
        case ("import", name):
            return str(name)
        case (*parts,) if parts:
            return "/".join(str(part) for part in parts)
        case _:
            return str(identity)


def _address(mem: Any, depth: int) -> str:
    match mem:
        case MemoryAddress(segment, terms, displacement, symbols):
            pass
        case _:
            return render(mem, depth)
    parts = [
        (
            render(term.value, depth + 1)
            if term.scale == 1
            else f"{render(term.value, depth + 1)}*{term.scale}"
        )
        for term in terms
    ]
    parts += [term.ref.display for term in symbols]
    match displacement:
        case 0 if parts:
            pass
        case int() as constant:
            parts.append(render_number(constant))
        case other:
            parts.append(str(other))
    text = " + ".join(parts).replace("+ -", "- ")
    return f"{segment}:{text}" if segment else text
