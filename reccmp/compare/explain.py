"""Explain a mismatch as a logic divergence a person can act on: where it
is, what each side computes there, an input under which they differ, and
the source-level cause that pattern usually has.

The values are the verifier's symbolic values at the first difference
(ComparisonDifference.values). The counterexample comes from Z3 over the
verifier's abstraction, where unknown memory and call results are free: it
shows how the two expressions differ, it does not prove the functions do.
A refutation (``analysis.witness``) is that proof, from running both.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.operand import (
    Imm,
    Mem,
    Operand,
    SignedSymbol,
    Sym,
    format_operand,
)
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import (
    AddressTerm,
    AddressValue,
    CallStack,
    CarryResult,
    Compare,
    CompareFlags,
    CompareKind,
    ConditionCode,
    Constant,
    DivideResult,
    Extend,
    ExtendKind,
    Extract,
    FlagsResult,
    FloatCompare,
    FloatOperation,
    FloatStatusWord,
    Init,
    Insert,
    Load,
    MemoryAddress,
    MultiplyResult,
    Operation,
    OperationKind,
    Phi,
    ReceiverLoad,
    SahfCarry,
    SahfFlags,
    Select,
    SetCondition,
    StackOffset,
    SymbolValue,
    TestFlags,
    UnaryOperation,
    VirtualCall,
    is_value,
    value_children,
)
from reccmp.compare.asm.verifier.render import render, render_number
from reccmp.source.records import SourceComparison, SourceComparisonOperand
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DifferenceKind,
    DifferenceSide,
    SolverResult,
)

_SIGN_SWAP = {
    CompareKind.LT_U: CompareKind.LT_S,
    CompareKind.LE_U: CompareKind.LE_S,
    CompareKind.LT_S: CompareKind.LT_U,
    CompareKind.LE_S: CompareKind.LE_U,
}


def _where(side: DifferenceSide) -> str:
    text = f"0x{side.address:x}" if side.address is not None else "function exit"
    if side.source is not None:
        text += f" ({side.source.path}:{side.source.line})"
    return text


def _symbol_ref(operand: Operand | None) -> Reference | None:
    """The reference an operand names: a symbol, or the one symbol of an
    absolute memory operand."""
    match operand:
        case Sym(ref) | Mem(symbols=(SignedSymbol(1, ref),)):
            return ref
    return None


def _identity_kind(operand: Operand | None) -> str | None:
    match _symbol_ref(operand):
        case Reference(identity=(str() as kind, *_)):
            return kind
    return None


def _retail_address(operand: Operand | None) -> int | None:
    """The original address an entity reference reaches (both sides'
    entity identities are in the original's address space)."""
    match _symbol_ref(operand):
        case Reference(identity=("entity", int() as entity, int() as offset)):
            return entity + offset
    return None


def _base_register(operand: Operand | None) -> str | None:
    match operand:
        case Mem(terms=terms):
            return next((term.register for term in terms if term.scale == 1), None)
    return None


def _describe_operand(operand: SourceComparisonOperand) -> str:
    if operand.constant is not None:
        return str(operand.constant)
    return f"{operand.type} field" if operand.field else operand.type


def _describe_comparison(comparison: SourceComparison) -> str:
    """`short field < 65 as int, signed`"""
    left, right = (_describe_operand(item) for item in comparison.operands)
    if comparison.signed is not None:
        how = "signed" if comparison.signed else "unsigned"
    else:
        how = "floating" if comparison.floating else "other"
    return f"{left} {comparison.operator} {right} as {comparison.type}, {how}"


def observed_text(side: DifferenceSide) -> str:
    """What one side has at a difference, for people."""
    observed = side.observed
    parts = []
    if observed.operand is not None:
        parts.append(format_operand(observed.operand))
    if observed.register is not None:
        parts.append(f"{observed.register} =")
    if observed.value is not None:
        parts.append(observed.value)
    if side.field is not None:
        parts.append(f"({side.field.class_name}::{'.'.join(side.field.path)})")
    if side.source_comparisons:
        parts.append(
            "(source: "
            + "; ".join(_describe_comparison(item) for item in side.source_comparisons)
            + ")"
        )
    return " ".join(parts)


def _equal(values: tuple) -> bool:
    return bitvector.compare(values).result is SolverResult.PROVED


def _swap_signedness(value: Any) -> Any:
    """``value`` with every comparison, extension and right shift of the
    other signedness."""
    # pylint: disable=too-many-return-statements
    match value:
        case Compare(kind, left, right, width):
            return Compare(
                _SIGN_SWAP.get(kind, kind),
                _swap_signedness(left),
                _swap_signedness(right),
                width,
            )
        case Extend(ExtendKind.MOVZX, width, operand):
            return Extend(ExtendKind.MOVSX, width, _swap_signedness(operand))
        case Extend(ExtendKind.MOVSX, width, operand):
            return Extend(ExtendKind.MOVZX, width, _swap_signedness(operand))
        case Operation(OperationKind.SHR, (operand, count)):
            return Operation(
                OperationKind.SAR,
                (_swap_signedness(operand), count),
            )
        case Operation(OperationKind.SAR, (operand, count)):
            return Operation(
                OperationKind.SHR,
                (_swap_signedness(operand), count),
            )
        case Operation(kind, operands):
            return Operation(kind, tuple(_swap_signedness(item) for item in operands))
        case UnaryOperation(kind, operand):
            return UnaryOperation(kind, _swap_signedness(operand))
        case Extract(part, whole):
            return Extract(part, _swap_signedness(whole))
        case Insert(part, old, new):
            return Insert(part, _swap_signedness(old), _swap_signedness(new))
        case SetCondition(predicate):
            return SetCondition(_swap_signedness(predicate))
        case Select(predicate, fallthrough, taken):
            return Select(
                _swap_signedness(predicate),
                _swap_signedness(fallthrough),
                _swap_signedness(taken),
            )
        case ConditionCode(condition, flags, carry):
            return ConditionCode(
                condition,
                _swap_signedness(flags),
                None if carry is None else _swap_signedness(carry),
            )
        case (*items,):
            return tuple(_swap_signedness(item) for item in items)
        case _:
            return value


# not (a < b) is b <= a, and not (a <= b) is b < a.
_NEGATED_ORDER = {
    CompareKind.LT_U: CompareKind.LE_U,
    CompareKind.LE_U: CompareKind.LT_U,
    CompareKind.LT_S: CompareKind.LE_S,
    CompareKind.LE_S: CompareKind.LT_S,
}


def _negate(predicate: Any) -> Any:
    """The predicate taken exactly when ``predicate`` is not."""
    match predicate:
        case Compare(CompareKind.EQ, left, right, width):
            return Compare(CompareKind.NE, left, right, width)
        case Compare(CompareKind.NE, left, right, width):
            return Compare(CompareKind.EQ, left, right, width)
        case Compare(kind, left, right, width) if kind in _NEGATED_ORDER:
            return Compare(_NEGATED_ORDER[kind], right, left, width)
        case _:
            return None


class Group(Enum):
    """Who has to act on a divergence."""

    LOGIC = "logic"  # the recovered source
    ANNOTATION = "annotation"  # the matching annotations
    TOOLING = "tooling"  # nobody: the comparison itself cannot tell
    UNEXPLAINED = "unexplained"


class CauseCode(Enum):
    FIELD_OFFSET = "field_offset"
    CONSTANT = "constant"
    OTHER_INPUT = "other_input"
    SYMBOL_OFFSET = "symbol_offset"
    OTHER_ENTITY = "other_entity"
    UNIDENTIFIED_ADDRESS = "unidentified_address"
    UNPAIRED_ENTITY = "unpaired_entity"
    SIGNEDNESS = "signedness"
    INVERTED = "inverted"
    WIDTH = "width"
    JOIN_VALUE = "join_value"
    OVERLAPPING_GLOBAL = "overlapping_global"
    OTHER_GLOBAL = "other_global"
    UNPAIRED_CALLEE = "unpaired_callee"
    OTHER_CALLEE = "other_callee"


@dataclass(frozen=True)
class Cause:
    code: CauseCode
    group: Group
    text: str


def _identity_kinds(value: Any) -> set[str]:
    """The kinds of every reference inside a symbolic value."""
    kinds: set[str] = set()
    seen: set[int] = set()
    stack = [value]
    while stack:
        match node := stack.pop():
            case Reference(identity=(kind, *_)) | SignedSymbol(
                ref=Reference(identity=(kind, *_))
            ):
                kinds.add(str(kind))
            case SymbolValue((kind, *_)):
                kinds.add(str(kind))
            case Load(address):
                stack.append(address)
            case AddressValue(address):
                stack.append(address)
            case MemoryAddress(terms=terms, displacement=displacement):
                stack.extend(term.value for term in terms)
                stack.append(displacement)
            case StackOffset(base):
                stack.append(base)
            case tuple() if id(node) not in seen:
                seen.add(id(node))
                stack.extend(node)
            case _ if is_value(node) and id(node) not in seen:
                seen.add(id(node))
                stack.extend(value_children(node))
    return kinds


def _without_joins(value: Any) -> Any:
    """``value`` with every join's identity erased."""
    # pylint: disable=too-many-return-statements,too-many-locals
    match value:
        case Phi():
            return Phi(0, 0)
        case Load(address, width, generation):
            return Load(_without_joins(address), width, generation)
        case AddressValue(address):
            return AddressValue(_without_joins(address))
        case MemoryAddress(segment, terms, displacement, symbols):
            return MemoryAddress(
                segment,
                tuple(
                    AddressTerm(_without_joins(term.value), term.scale)
                    for term in terms
                ),
                (
                    displacement
                    if isinstance(displacement, int)
                    else _without_joins(displacement)
                ),
                symbols,
            )
        case StackOffset(base, offset):
            return StackOffset(_without_joins(base), offset)
        case Operation(kind, operands):
            return Operation(kind, tuple(_without_joins(item) for item in operands))
        case UnaryOperation(kind, operand):
            return UnaryOperation(kind, _without_joins(operand))
        case Extract(part, whole):
            return Extract(part, _without_joins(whole))
        case Insert(part, old, new):
            return Insert(part, _without_joins(old), _without_joins(new))
        case Extend(kind, width, source):
            return Extend(kind, width, _without_joins(source))
        case Compare(kind, left, right, width):
            return Compare(kind, _without_joins(left), _without_joins(right), width)
        case ConditionCode(condition, flags, carry):
            return ConditionCode(
                condition,
                _without_joins(flags),
                None if carry is None else _without_joins(carry),
            )
        case SetCondition(predicate):
            return SetCondition(_without_joins(predicate))
        case Select(predicate, fallthrough, taken):
            return Select(
                _without_joins(predicate),
                _without_joins(fallthrough),
                _without_joins(taken),
            )
        case MultiplyResult(signed, part, operands):
            return MultiplyResult(
                signed, part, tuple(_without_joins(item) for item in operands)
            )
        case DivideResult(signed, part, high, low, divisor):
            return DivideResult(
                signed,
                part,
                _without_joins(high),
                _without_joins(low),
                _without_joins(divisor),
            )
        case FlagsResult(operation) | CarryResult(operation):
            return type(value)(_without_joins(operation))
        case CompareFlags(left, right, width) | TestFlags(left, right, width):
            return type(value)(_without_joins(left), _without_joins(right), width)
        case SahfFlags(source, previous):
            return SahfFlags(_without_joins(source), _without_joins(previous))
        case SahfCarry(source) | FloatStatusWord(source):
            return type(value)(_without_joins(source))
        case FloatOperation(kind, operands):
            return FloatOperation(
                kind, tuple(_without_joins(item) for item in operands)
            )
        case FloatCompare(left, right):
            return FloatCompare(_without_joins(left), _without_joins(right))
        case CallStack(site, incoming):
            return CallStack(site, _without_joins(incoming))
        case ReceiverLoad(address, width):
            return ReceiverLoad(_without_joins(address), width)
        case VirtualCall(receiver, displacement):
            return VirtualCall(_without_joins(receiver), displacement)
        case (*items,):
            return tuple(_without_joins(item) for item in items)
        case _:
            return value


def _difference_in_children(
    parent: Any, children_o: tuple[Any, ...], children_r: tuple[Any, ...]
) -> tuple[Any, Any, Any] | None:
    differing = [
        (child_o, child_r)
        for child_o, child_r in zip(children_o, children_r)
        if child_o != child_r
    ]
    if len(differing) != 1:
        return None
    inner = first_difference(*differing[0])
    return inner if inner is not None else (parent, *differing[0])


def first_difference(orig: Any, recomp: Any) -> tuple[Any, Any, Any] | None:
    # pylint: disable=too-many-locals
    """The smallest subterms where two values differ, with the node that
    holds them: ``(parent, orig part, recomp part)``. None when the two
    differ in shape (another operation) above any single part."""
    result: tuple[Any, Any, Any] | None = None
    match orig, recomp:
        case _ if orig == recomp:
            pass
        case (SymbolValue(), SymbolValue()) | (Constant(), Constant()):
            result = (None, orig, recomp)
        case AddressValue(address_o), AddressValue(address_r):
            result = first_difference(address_o, address_r)
        case Load(address_o, width_o, generation_o), Load(
            address_r, width_r, generation_r
        ) if (width_o, generation_o) == (width_r, generation_r):
            result = first_difference(address_o, address_r)
        case StackOffset(base_o, offset_o), StackOffset(base_r, offset_r) if (
            offset_o == offset_r
        ):
            result = first_difference(base_o, base_r)
        case MemoryAddress(
            segment_o, terms_o, displacement_o, symbols_o
        ), MemoryAddress(segment_r, terms_r, displacement_r, symbols_r) if (
            segment_o,
            terms_o,
            symbols_o,
        ) == (
            segment_r,
            terms_r,
            symbols_r,
        ):
            result = (orig, displacement_o, displacement_r)
        case Operation(kind_o, operands_o), Operation(
            kind_r, operands_r
        ) if kind_o == kind_r and len(operands_o) == len(operands_r):
            result = _difference_in_children(orig, operands_o, operands_r)
        case UnaryOperation(kind_o, operand_o), UnaryOperation(kind_r, operand_r) if (
            kind_o == kind_r
        ):
            result = _difference_in_children(orig, (operand_o,), (operand_r,))
        case Extract(part_o, whole_o), Extract(part_r, whole_r) if part_o == part_r:
            result = _difference_in_children(orig, (whole_o,), (whole_r,))
        case Insert(part_o, old_o, new_o), Insert(part_r, old_r, new_r) if (
            part_o == part_r
        ):
            result = _difference_in_children(orig, (old_o, new_o), (old_r, new_r))
        case Extend(kind_o, width_o, source_o), Extend(kind_r, width_r, source_r) if (
            kind_o,
            width_o,
        ) == (kind_r, width_r):
            result = _difference_in_children(orig, (source_o,), (source_r,))
        case Compare(kind_o, left_o, right_o, width_o), Compare(
            kind_r, left_r, right_r, width_r
        ) if (kind_o, width_o) == (kind_r, width_r):
            result = _difference_in_children(orig, (left_o, right_o), (left_r, right_r))
        case ConditionCode(condition_o, flags_o, carry_o), ConditionCode(
            condition_r, flags_r, carry_r
        ) if condition_o == condition_r and (carry_o is None) == (carry_r is None):
            children_o = (flags_o,) if carry_o is None else (flags_o, carry_o)
            children_r = (flags_r,) if carry_r is None else (flags_r, carry_r)
            result = _difference_in_children(orig, children_o, children_r)
        case SetCondition(predicate_o), SetCondition(predicate_r):
            result = _difference_in_children(orig, (predicate_o,), (predicate_r,))
        case Select(predicate_o, old_o, new_o), Select(predicate_r, old_r, new_r):
            result = _difference_in_children(
                orig,
                (predicate_o, old_o, new_o),
                (predicate_r, old_r, new_r),
            )
        case MultiplyResult(signed_o, part_o, operands_o), MultiplyResult(
            signed_r, part_r, operands_r
        ) if (signed_o, part_o) == (signed_r, part_r):
            result = _difference_in_children(orig, operands_o, operands_r)
        case DivideResult(signed_o, part_o, high_o, low_o, divisor_o), DivideResult(
            signed_r, part_r, high_r, low_r, divisor_r
        ) if (signed_o, part_o) == (signed_r, part_r):
            result = _difference_in_children(
                orig,
                (high_o, low_o, divisor_o),
                (high_r, low_r, divisor_r),
            )
        case (FlagsResult(operation_o) | CarryResult(operation_o)), (
            FlagsResult(operation_r) | CarryResult(operation_r)
        ) if type(orig) is type(recomp):
            result = _difference_in_children(orig, (operation_o,), (operation_r,))
        case (
            CompareFlags(left_o, right_o, width_o) | TestFlags(left_o, right_o, width_o)
        ), (
            CompareFlags(left_r, right_r, width_r) | TestFlags(left_r, right_r, width_r)
        ) if (
            type(orig) is type(recomp) and width_o == width_r
        ):
            result = _difference_in_children(orig, (left_o, right_o), (left_r, right_r))
        case SahfFlags(source_o, previous_o), SahfFlags(source_r, previous_r):
            result = _difference_in_children(
                orig, (source_o, previous_o), (source_r, previous_r)
            )
        case (SahfCarry(source_o) | FloatStatusWord(source_o)), (
            SahfCarry(source_r) | FloatStatusWord(source_r)
        ) if type(orig) is type(recomp):
            result = _difference_in_children(orig, (source_o,), (source_r,))
        case FloatOperation(kind_o, operands_o), FloatOperation(
            kind_r, operands_r
        ) if kind_o == kind_r and len(operands_o) == len(operands_r):
            result = _difference_in_children(orig, operands_o, operands_r)
        case FloatCompare(left_o, right_o), FloatCompare(left_r, right_r):
            result = _difference_in_children(orig, (left_o, right_o), (left_r, right_r))
        case CallStack(site_o, incoming_o), CallStack(site_r, incoming_r) if (
            site_o == site_r
        ):
            result = _difference_in_children(orig, (incoming_o,), (incoming_r,))
        case ReceiverLoad(address_o, width_o), ReceiverLoad(address_r, width_r) if (
            width_o == width_r
        ):
            result = _difference_in_children(orig, (address_o,), (address_r,))
        case VirtualCall(receiver_o, displacement_o), VirtualCall(
            receiver_r, displacement_r
        ):
            if displacement_o == displacement_r:
                result = _difference_in_children(orig, (receiver_o,), (receiver_r,))
        case (tag_o, *items_o), (tag_r, *items_r) if tag_o == tag_r and len(
            items_o
        ) == len(items_r):
            differing = [
                (item_o, item_r)
                for item_o, item_r in zip(items_o, items_r)
                if item_o != item_r
            ]
            if len(differing) == 1:
                inner = first_difference(*differing[0])
                result = inner if inner is not None else (orig, *differing[0])
    return result


def _identity(part: Any) -> Any:
    match part:
        case Reference(identity=identity) | SymbolValue(identity):
            return identity
        case _:
            return None


def _part_cause(parent: Any, part_o: Any, part_r: Any) -> Cause | None:
    """What the one differing part of two otherwise equal values means."""
    # pylint: disable=too-many-return-statements
    match parent, part_o, part_r:
        case MemoryAddress(), int(), int():
            return Cause(
                CauseCode.FIELD_OFFSET,
                Group.LOGIC,
                f"the same address expression at offset {render_number(part_o)} vs "
                f"{render_number(part_r)}: the wrong field or element, or a layout "
                "that places it elsewhere",
            )
        case _, Constant(value_o), Constant(value_r):
            return Cause(
                CauseCode.CONSTANT,
                Group.LOGIC,
                f"only a constant differs ({render_number(value_o)} vs "
                f"{render_number(value_r)}): a wrong literal, bound, enum value or size",
            )
        case _, Init(), Init():
            return Cause(
                CauseCode.OTHER_INPUT,
                Group.LOGIC,
                f"it reads {render(part_o)} on one side and {render(part_r)} on "
                "the other: a different argument or register",
            )
    identity_o, identity_r = _identity(part_o), _identity(part_r)
    match identity_o, identity_r:
        case ("entity", base_o, int() as offset_o), (
            "entity",
            base_r,
            int() as offset_r,
        ) if (
            base_o == base_r
        ):
            return Cause(
                CauseCode.SYMBOL_OFFSET,
                Group.ANNOTATION,
                f"the same entity at offset {render_number(offset_o)} vs "
                f"{render_number(offset_r)}: its annotated address is off by "
                f"{render_number(offset_o - offset_r)} on one side (a vtable annotated "
                "at its RTTI slot, say), or the source takes another element",
            )
        case ("entity", *_), ("entity", *_):
            return Cause(
                CauseCode.OTHER_ENTITY,
                Group.LOGIC,
                f"a different global, vtable or function: {render(part_o)} vs "
                f"{render(part_r)} — the wrong variable or class",
            )
        case (("unresolved", *_), _) | (_, ("unresolved", *_)):
            return Cause(
                CauseCode.UNIDENTIFIED_ADDRESS, Group.ANNOTATION, _UNIDENTIFIED_TEXT
            )
        case (kind_o, *_), (kind_r, *_) if "entity" in (kind_o, kind_r):
            return Cause(
                CauseCode.UNPAIRED_ENTITY,
                Group.ANNOTATION,
                f"{render(part_o)} vs {render(part_r)}: one side's global has no "
                "counterpart; pair it (or a constant is pooled differently) "
                "before judging the logic",
            )
    return None


_UNIDENTIFIED_TEXT = (
    "one side uses an address nothing names (no symbol, no pairing): "
    "identify and annotate it before judging the logic"
)


def _value_causes(values: tuple) -> list[Cause]:
    value_o, value_r, bits, kind = values
    causes: list[Cause] = []
    difference = first_difference(value_o, value_r)
    if difference is not None:
        cause = _part_cause(*difference)
        if cause is not None:
            causes.append(cause)
    if kind == "predicate":
        if _equal((value_o, _swap_signedness(value_r), None, "predicate")):
            causes.append(
                Cause(
                    CauseCode.SIGNEDNESS,
                    Group.LOGIC,
                    "the same comparison with the other signedness: an operand is "
                    "signed on one side and unsigned on the other (the variable's "
                    "or field's type, or a cast)",
                )
            )
        negated = _negate(value_r)
        if negated is not None and _equal((value_o, negated, None, "predicate")):
            causes.append(
                Cause(
                    CauseCode.INVERTED,
                    Group.LOGIC,
                    "the condition is inverted: the recovered test is the negation "
                    "of retail's (an `if` sense or a `!` flipped)",
                )
            )
    else:
        width = bits if bits is not None else bitvector.value_width(value_o, value_r)
        for narrow in (8, 16):
            if (
                width is not None
                and width > narrow
                and _equal((value_o, value_r, narrow, "value"))
            ):
                causes.append(
                    Cause(
                        CauseCode.WIDTH,
                        Group.LOGIC,
                        f"the low {narrow} bits agree and only the upper bits "
                        f"differ: a {narrow}-bit type on one side and a {width}-bit "
                        "one on the other (a return, field or argument type such "
                        "as bool/char/short vs int/BOOL)",
                    )
                )
                break
        if _equal((_swap_signedness(value_o), value_r, bits, kind)):
            causes.append(
                Cause(
                    CauseCode.SIGNEDNESS,
                    Group.LOGIC,
                    "equal with the other signedness of an extension or shift: "
                    "a signed/unsigned type mismatch",
                )
            )
    if not causes and _without_joins(value_o) == _without_joins(value_r):
        causes.append(
            Cause(
                CauseCode.JOIN_VALUE,
                Group.TOOLING,
                "the same expression over values merged differently at an earlier "
                "join: the difference, if any, is upstream where paths meet",
            )
        )
    if "unresolved" in _identity_kinds(value_o) | _identity_kinds(value_r):
        causes.append(
            Cause(CauseCode.UNIDENTIFIED_ADDRESS, Group.ANNOTATION, _UNIDENTIFIED_TEXT)
        )
    return causes


def _fact_causes(difference: ComparisonDifference) -> list[Cause]:
    # pylint: disable=too-many-return-statements
    operand_o = difference.orig.observed.operand
    operand_r = difference.recomp.observed.operand
    kind = difference.kind
    kinds = {_identity_kind(operand_o), _identity_kind(operand_r)}
    if "unresolved" in kinds:
        return [
            Cause(CauseCode.UNIDENTIFIED_ADDRESS, Group.ANNOTATION, _UNIDENTIFIED_TEXT)
        ]
    if kind in (DifferenceKind.MEMORY_ADDRESS, DifferenceKind.SYMBOL_RESOLUTION):
        at_o, at_r = _retail_address(operand_o), _retail_address(operand_r)
        named_o, named_r = _symbol_ref(operand_o), _symbol_ref(operand_r)
        if at_o is not None and at_o == at_r and named_o and named_r:
            return [
                Cause(
                    CauseCode.OVERLAPPING_GLOBAL,
                    Group.ANNOTATION,
                    f"both reach retail 0x{at_o:x}, named `{named_o.display}` "
                    f"on one side and `{named_r.display}` on the other: two "
                    "annotated globals overlap",
                )
            ]
        if at_o is not None and at_r is not None and named_o and named_r:
            return [
                Cause(
                    CauseCode.OTHER_GLOBAL,
                    Group.LOGIC,
                    f"a different global: retail `{named_o.display}` "
                    f"(0x{at_o:x}), recompiled `{named_r.display}` "
                    f"(retail 0x{at_r:x}) — the wrong variable",
                )
            ]
        match operand_o, operand_r:
            case Mem(displacement=displacement_o), Mem(displacement=displacement_r) if (
                _base_register(operand_o) == _base_register(operand_r)
                and displacement_o != displacement_r
            ):
                field_at = difference.recomp.field
                where = (
                    f" (recompiled: {field_at.class_name}::{'.'.join(field_at.path)})"
                    if field_at is not None
                    else ""
                )
                return [
                    Cause(
                        CauseCode.FIELD_OFFSET,
                        Group.LOGIC,
                        f"the same base at offset {displacement_o} vs "
                        f"{displacement_r}{where}: the wrong field, or a "
                        "struct layout that places it elsewhere",
                    )
                ]
    if kind == DifferenceKind.CALL_TARGET:
        kind_o, kind_r = _identity_kind(operand_o), _identity_kind(operand_r)
        if "entity" in (kind_o, kind_r) and kind_o != kind_r:
            return [
                Cause(
                    CauseCode.UNPAIRED_CALLEE,
                    Group.ANNOTATION,
                    f"one callee has no counterpart on the other side ({kind_o} vs "
                    f"{kind_r}): pair it, or check which function the source calls",
                )
            ]
        return [
            Cause(
                CauseCode.OTHER_CALLEE,
                Group.LOGIC,
                "a different function is called: the wrong callee, overload or "
                "virtual slot",
            )
        ]
    match kind, operand_o, operand_r:
        case DifferenceKind.IMMEDIATE_VALUE, Imm(value_o), Imm(value_r):
            return [
                Cause(
                    CauseCode.CONSTANT,
                    Group.LOGIC,
                    f"a different constant ({value_o} vs {value_r}): a wrong "
                    "literal, enum value, size or offset",
                )
            ]
    return []


@dataclass(frozen=True)
class Explanation:
    """A mismatch's first divergence, made readable."""

    # pylint: disable=too-many-instance-attributes

    kind: DifferenceKind
    orig: str
    recomp: str
    orig_value: str | None = None
    recomp_value: str | None = None
    example: str | None = None
    causes: tuple[Cause, ...] = ()
    confirmed: str | None = None
    agreed_through: bool = False
    observed: tuple[tuple[str, str], ...] = field(default=())

    @property
    def group(self) -> Group:
        """Who has to act: a cause's group, else ``logic`` when running both
        confirmed it, else ``unexplained``."""
        if self.causes:
            return self.causes[0].group
        if self.confirmed is not None:
            return Group.LOGIC
        return Group.TOOLING if self.agreed_through else Group.UNEXPLAINED


def explanation(analysis: ComparisonAnalysis) -> Explanation | None:
    """The explanation of a mismatch; None for other statuses."""
    difference = analysis.difference
    if analysis.status != ComparisonStatus.MISMATCH or difference is None:
        return None
    orig_value = recomp_value = example = None
    causes: list[Cause] = []
    if difference.values is not None:
        value_o, value_r, _, _ = difference.values
        orig_value, recomp_value = render(value_o), render(value_r)
        found = bitvector.counterexample(difference.values)
        if found is not None:
            given = _rendered_assignment(found.assignment)
            example = (
                f"{given or 'always'}: orig {_shown(found.orig)}, "
                f"recomp {_shown(found.recomp)}"
            )
        causes = _value_causes(difference.values)
    if not causes:
        causes = _fact_causes(difference)
    witness = analysis.witness
    execution = analysis.execution
    return Explanation(
        kind=difference.kind,
        orig=_where(difference.orig),
        recomp=_where(difference.recomp),
        orig_value=orig_value,
        recomp_value=recomp_value,
        example=example,
        causes=tuple(causes),
        confirmed=(
            f"seed {witness.seed}: {witness.location}: orig {witness.orig_value}, "
            f"recomp {witness.recomp_value}"
            if witness is not None
            else None
        ),
        agreed_through=bool(
            witness is None and execution is not None and execution.reached_location
        ),
        observed=(
            ("orig", observed_text(difference.orig)),
            ("recomp", observed_text(difference.recomp)),
        ),
    )


def _rendered_assignment(assignment: tuple[tuple[Hashable, int], ...]) -> str:
    parts = [
        f"{render(term)} = {render_number(value)}" for term, value in assignment[:6]
    ]
    if len(assignment) > 6:
        parts.append("…")
    return ", ".join(parts)


def _shown(value: int | bool) -> str:
    match value:
        case bool() as truth:
            return str(truth).lower()
        case int():
            return render_number(value)


_GROUP_TEXT = {
    Group.LOGIC: "LIKELY A Group.LOGIC DIFFERENCE in the recovered source",
    Group.ANNOTATION: "LIKELY AN Group.ANNOTATION PROBLEM (symbols/globals), not the logic",
    Group.TOOLING: "LIKELY NOT A SOURCE PROBLEM: a comparison artefact",
    Group.UNEXPLAINED: "unexplained: read the marked lines",
}


def explain(analysis: ComparisonAnalysis) -> list[str]:
    """Lines explaining a mismatch's first divergence; empty for others."""
    found = explanation(analysis)
    if found is None:
        return []
    lines = [
        f"FIRST DIVERGENCE: {found.kind.value.replace('_', ' ')} — {_GROUP_TEXT[found.group]}",
        f"  orig   {found.orig}",
        f"  recomp {found.recomp}",
    ]
    if found.orig_value is not None:
        lines.append(f"  orig computes   {found.orig_value}")
        lines.append(f"  recomp computes {found.recomp_value}")
    else:
        lines += [f"  {name:<6} {text}" for name, text in found.observed if text]
    if found.example is not None:
        lines.append(f"  e.g. {found.example}")
    if found.confirmed is not None:
        lines.append(f"  CONFIRMED by running both ({found.confirmed})")
    elif found.agreed_through:
        lines.append("  not confirmed: running both through this point always agreed")
    lines += [f"  likely cause: {cause.text}" for cause in found.causes]
    return lines


def divergence_addresses(analysis: ComparisonAnalysis) -> frozenset[str]:
    """The first divergence's instruction addresses, as a diff prints them."""
    difference = analysis.difference
    if analysis.status != ComparisonStatus.MISMATCH or difference is None:
        return frozenset()
    return frozenset(
        f"{side.address:#x}"
        for side in (difference.orig, difference.recomp)
        if side.address is not None
    )
