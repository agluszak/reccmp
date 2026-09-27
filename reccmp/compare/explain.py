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
from typing import Any

from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.operand import SignedSymbol
from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DifferenceSide,
)

_REGISTER = {
    "a": "eax",
    "b": "ebx",
    "c": "ecx",
    "d": "edx",
    "si": "esi",
    "di": "edi",
    "bp": "ebp",
    "sp": "esp",
}
_BINARY = {
    "and": "&",
    "or": "|",
    "xor": "^",
    "sub": "-",
    "imul": "*",
    "imul3": "*",
    "shl": "<<",
    "shr": ">>u",
    "sar": ">>s",
}
_PREDICATE = {
    "eq": "==",
    "ne": "!=",
    "lt_u": "<u",
    "le_u": "<=u",
    "lt_s": "<s",
    "le_s": "<=s",
}
_SIGN_SWAP = {"lt_u": "lt_s", "le_u": "le_s", "lt_s": "lt_u", "le_s": "le_u"}
_UNARY = {"inc": "{} + 1", "dec": "{} - 1", "neg": "-{}", "not": "~{}"}
_PART = {"l8": "low8", "h8": "bits8_15", "r16": "low16"}
_INSERT = {"ins_l8": "al", "ins_h8": "ah", "ins_r16": "ax"}


def _number(value: int) -> str:
    return str(value) if -10 < value < 10 else hex(value)


_ENTRY_SP = (("init", "sp"), 1)


def render(value: Any, depth: int = 0) -> str:
    """A C-like spelling of a verifier symbolic value."""
    # pylint: disable=too-many-return-statements,too-many-branches
    if depth > 8:
        return "…"

    def inner(item: Any) -> str:
        return render(item, depth + 1)

    match value:
        case ("imm", int() as number):
            return _number(number)
        case ("init", family):
            return f"{_REGISTER.get(family, family)}@entry"
        case ("load", ("mem", "", (entry,), int() as displacement, ()), "dword", _) if (
            entry == _ENTRY_SP and displacement > 0 and displacement % 4 == 0
        ):
            return f"arg{displacement // 4}"
        case ("load", address, size, *_):
            return f"{size}[{_address(address, depth + 1)}]"
        case ("mem", *_):
            return _address(value, depth)
        case ("addr", address):
            return f"&[{_address(address, depth + 1)}]"
        case ("sym", identity):
            return _symbol(identity)
        case ("spadd", base, int() as offset):
            return f"{inner(base)} {'+' if offset >= 0 else '-'} {_number(abs(offset))}"
        case (tag, whole) if tag in _PART:
            return f"{_PART[tag]}({inner(whole)})"
        case (tag, old, new) if tag in _INSERT:
            return f"({inner(old)} with {_INSERT[tag]} = {inner(new)})"
        case ("add", *terms) if len(terms) >= 2:
            return "(" + " + ".join(inner(term) for term in terms) + ")"
        case (tag, left, right) if tag in _BINARY:
            return f"({inner(left)} {_BINARY[tag]} {inner(right)})"
        case (tag, operand) if tag in _UNARY:
            return "(" + _UNARY[tag].format(inner(operand)) + ")"
        case ("movzx", _, operand):
            return f"zext({inner(operand)})"
        case ("movsx", _, operand):
            return f"sext({inner(operand)})"
        case ("callret", call, family):
            return f"{_REGISTER.get(family, family)} after call@{call}"
        case ("phi" | "scratch_phi", block, klass):
            return f"join{block}#{klass}"
        case ("eq" | "ne" as tag, (left, right), *width):
            return _comparison(tag, left, right, width, depth)
        case (tag, left, right, *width) if tag in _PREDICATE:
            return _comparison(tag, left, right, width, depth)
        case ("cc", condition, flags, *_):
            return f"{condition} of {inner(flags)}"
        case (tag, *operands):
            return f"{tag}(" + ", ".join(inner(item) for item in operands[:2]) + ")"
        case _:
            return repr(value)


def _comparison(tag: str, left: Any, right: Any, width: list, depth: int) -> str:
    match width:
        case [int() as size]:
            bits = f" ({8 * size}-bit)"
        case _:
            bits = ""
    return (
        f"{render(left, depth + 1)} {_PREDICATE[tag]} {render(right, depth + 1)}{bits}"
    )


def _symbol(identity: Any) -> str:
    match identity:
        case ("entity", int() as address, int() as offset):
            return f"entity@0x{address:x}" + (f"+{_number(offset)}" if offset else "")
        case ("import", name):
            return str(name)
        case (*parts,) if parts:
            return "/".join(str(part) for part in parts)
        case _:
            return str(identity)


def _address(mem: Any, depth: int) -> str:
    match mem:
        case ("mem", segment, terms, displacement, symbols):
            pass
        case _:
            return render(mem, depth)
    parts = [
        render(term, depth + 1) if scale == 1 else f"{render(term, depth + 1)}*{scale}"
        for term, scale in terms
    ]
    parts += [term.ref.display for term in symbols]
    match displacement:
        case 0 if parts:
            pass
        case int() as number:
            parts.append(_number(number))
        case other:
            parts.append(str(other))
    text = " + ".join(parts).replace("+ -", "- ")
    return f"{segment}:{text}" if segment else text


def _where(side: DifferenceSide) -> str:
    text = f"0x{side.address:x}" if side.address is not None else "function exit"
    match side.facts.get("source_path"), side.facts.get("source_line"):
        case str() as path, int() as line:
            text += f" ({path}:{line})"
    return text


def _equal(values: tuple) -> bool:
    return bitvector.compare(values).result == "proved"


def _swap_signedness(value: Any) -> Any:
    """``value`` with every comparison, extension and right shift of the
    other signedness."""
    # pylint: disable=too-many-return-statements
    match value:
        case (tag, *rest) if tag in _SIGN_SWAP:
            return (_SIGN_SWAP[tag], *rest)
        case ("movzx", width, operand):
            return ("movsx", width, _swap_signedness(operand))
        case ("movsx", width, operand):
            return ("movzx", width, _swap_signedness(operand))
        case ("shr", operand, count):
            return ("sar", _swap_signedness(operand), count)
        case ("sar", operand, count):
            return ("shr", _swap_signedness(operand), count)
        case (*items,):
            return tuple(_swap_signedness(item) for item in items)
        case _:
            return value


# not (a < b) is b <= a, and not (a <= b) is b < a.
_NEGATED_ORDER = {"lt_u": "le_u", "le_u": "lt_u", "lt_s": "le_s", "le_s": "lt_s"}


def _negate(predicate: Any) -> Any:
    """The predicate taken exactly when ``predicate`` is not."""
    match predicate:
        case ("eq", *rest):
            return ("ne", *rest)
        case ("ne", *rest):
            return ("eq", *rest)
        case (tag, left, right, *rest) if tag in _NEGATED_ORDER:
            return (_NEGATED_ORDER[tag], right, left, *rest)
        case _:
            return None


# Who has to act on a cause: the recovered source ("logic"), the matching
# annotations ("annotation"), or nobody, because the comparison itself
# cannot tell ("tooling").
LOGIC, ANNOTATION, TOOLING = "logic", "annotation", "tooling"


@dataclass(frozen=True)
class Cause:
    code: str
    group: str
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
            case ("sym", (kind, *_)):
                kinds.add(str(kind))
            case tuple() if id(node) not in seen:
                seen.add(id(node))
                stack.extend(node)
    return kinds


def _without_joins(value: Any) -> Any:
    """``value`` with every join's identity erased."""
    match value:
        case ("phi" | "scratch_phi", *_):
            return ("phi",)
        case (*items,):
            return tuple(_without_joins(item) for item in items)
        case _:
            return value


def first_difference(orig: Any, recomp: Any) -> tuple[Any, Any, Any] | None:
    """The smallest subterms where two values differ, with the node that
    holds them: ``(parent, orig part, recomp part)``. None when the two
    differ in shape (another operation) above any single part."""
    match orig, recomp:
        case _ if orig == recomp:
            return None
        case ("sym", _), ("sym", _):
            # A reference is compared whole, not by its identity's parts.
            return (None, orig, recomp)
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
                return inner if inner is not None else (orig, *differing[0])
            return None
        case _:
            return None


def _identity(part: Any) -> Any:
    match part:
        case Reference(identity=identity) | ("sym", identity):
            return identity
        case _:
            return None


def _part_cause(parent: Any, part_o: Any, part_r: Any) -> Cause | None:
    """What the one differing part of two otherwise equal values means."""
    # pylint: disable=too-many-return-statements
    match parent, part_o, part_r:
        case ("mem", *_), int(), int():
            return Cause(
                "field_offset",
                LOGIC,
                f"the same address expression at offset {_number(part_o)} vs "
                f"{_number(part_r)}: the wrong field or element, or a layout "
                "that places it elsewhere",
            )
        case _, ("imm", int() as value_o), ("imm", int() as value_r):
            return Cause(
                "constant",
                LOGIC,
                f"only a constant differs ({_number(value_o)} vs "
                f"{_number(value_r)}): a wrong literal, bound, enum value or size",
            )
        case _, ("init", _), ("init", _):
            return Cause(
                "other_input",
                LOGIC,
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
                "symbol_offset",
                ANNOTATION,
                f"the same entity at offset {_number(offset_o)} vs "
                f"{_number(offset_r)}: its annotated address is off by "
                f"{_number(offset_o - offset_r)} on one side (a vtable annotated "
                "at its RTTI slot, say), or the source takes another element",
            )
        case ("entity", *_), ("entity", *_):
            return Cause(
                "other_entity",
                LOGIC,
                f"a different global, vtable or function: {render(part_o)} vs "
                f"{render(part_r)} — the wrong variable or class",
            )
        case (("unresolved", *_), _) | (_, ("unresolved", *_)):
            return Cause("unidentified_address", ANNOTATION, _UNIDENTIFIED_TEXT)
        case (kind_o, *_), (kind_r, *_) if "entity" in (kind_o, kind_r):
            return Cause(
                "unpaired_entity",
                ANNOTATION,
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
                    "signedness",
                    LOGIC,
                    "the same comparison with the other signedness: an operand is "
                    "signed on one side and unsigned on the other (the variable's "
                    "or field's type, or a cast)",
                )
            )
        negated = _negate(value_r)
        if negated is not None and _equal((value_o, negated, None, "predicate")):
            causes.append(
                Cause(
                    "inverted",
                    LOGIC,
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
                        "width",
                        LOGIC,
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
                    "signedness",
                    LOGIC,
                    "equal with the other signedness of an extension or shift: "
                    "a signed/unsigned type mismatch",
                )
            )
    if not causes and _without_joins(value_o) == _without_joins(value_r):
        causes.append(
            Cause(
                "join_value",
                TOOLING,
                "the same expression over values merged differently at an earlier "
                "join: the difference, if any, is upstream where paths meet",
            )
        )
    if "unresolved" in _identity_kinds(value_o) | _identity_kinds(value_r):
        causes.append(Cause("unidentified_address", ANNOTATION, _UNIDENTIFIED_TEXT))
    return causes


def _retail_address(facts: dict) -> int | None:
    """The original address an entity reference reaches (both sides'
    entity identities are in the original's address space)."""
    match facts.get("symbol_entity"), facts.get("symbol_offset"):
        case int() as entity, int() as offset:
            return entity + offset
        case int() as entity, _:
            return entity
        case _:
            return None


def _fact_causes(difference: ComparisonDifference) -> list[Cause]:
    # pylint: disable=too-many-return-statements
    facts_o, facts_r = difference.orig.facts, difference.recomp.facts
    kind = difference.kind
    kinds = {
        facts.get(key)
        for facts in (facts_o, facts_r)
        for key in ("symbol_kind", "target_kind")
    }
    if "unresolved" in kinds:
        return [Cause("unidentified_address", ANNOTATION, _UNIDENTIFIED_TEXT)]
    if kind in ("memory_address", "symbol_resolution"):
        at_o, at_r = _retail_address(facts_o), _retail_address(facts_r)
        if at_o is not None and at_o == at_r:
            return [
                Cause(
                    "overlapping_global",
                    ANNOTATION,
                    f"both reach retail 0x{at_o:x}, named `{facts_o.get('symbol')}` "
                    f"on one side and `{facts_r.get('symbol')}` on the other: two "
                    "annotated globals overlap",
                )
            ]
        if at_o is not None and at_r is not None:
            return [
                Cause(
                    "other_global",
                    LOGIC,
                    f"a different global: retail `{facts_o.get('symbol')}` "
                    f"(0x{at_o:x}), recompiled `{facts_r.get('symbol')}` "
                    f"(retail 0x{at_r:x}) — the wrong variable",
                )
            ]
        if facts_o.get("base_register") == facts_r.get("base_register") and (
            facts_o.get("displacement") != facts_r.get("displacement")
        ):
            field_name = facts_r.get("field_path") or facts_r.get("field_name")
            where = (
                f" (recompiled: {facts_r.get('class_name')}::{field_name})"
                if field_name
                else ""
            )
            return [
                Cause(
                    "field_offset",
                    LOGIC,
                    f"the same base at offset {facts_o.get('displacement')} vs "
                    f"{facts_r.get('displacement')}{where}: the wrong field, or a "
                    "struct layout that places it elsewhere",
                )
            ]
    if kind == "call_target":
        kind_o, kind_r = facts_o.get("target_kind"), facts_r.get("target_kind")
        if "entity" in (kind_o, kind_r) and kind_o != kind_r:
            return [
                Cause(
                    "unpaired_callee",
                    ANNOTATION,
                    f"one callee has no counterpart on the other side ({kind_o} vs "
                    f"{kind_r}): pair it, or check which function the source calls",
                )
            ]
        return [
            Cause(
                "other_callee",
                LOGIC,
                "a different function is called: the wrong callee, overload or "
                "virtual slot",
            )
        ]
    if kind == "immediate_value":
        return [
            Cause(
                "constant",
                LOGIC,
                f"a different constant ({facts_o.get('value')} vs "
                f"{facts_r.get('value')}): a wrong literal, enum value, size or offset",
            )
        ]
    return []


@dataclass(frozen=True)
class Explanation:
    """A mismatch's first divergence, made readable."""

    # pylint: disable=too-many-instance-attributes

    kind: str
    orig: str
    recomp: str
    orig_value: str | None = None
    recomp_value: str | None = None
    example: str | None = None
    causes: tuple[Cause, ...] = ()
    confirmed: str | None = None
    agreed_through: bool = False
    facts: tuple[tuple[str, str], ...] = field(default=())

    @property
    def group(self) -> str:
        """Who has to act: a cause's group, else ``logic`` when running both
        confirmed it, else ``unexplained``."""
        if self.causes:
            return self.causes[0].group
        if self.confirmed is not None:
            return LOGIC
        return TOOLING if self.agreed_through else "unexplained"

    def json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "group": self.group,
            "causes": [cause.code for cause in self.causes],
        }
        for key in ("orig_value", "recomp_value", "example", "confirmed"):
            if getattr(self, key) is not None:
                value[key] = getattr(self, key)
        return value


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
        facts=tuple(
            (
                name,
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(side.facts.items())
                    if k not in ("source_path", "source_line") and v not in (None, "")
                ),
            )
            for name, side in (("orig", difference.orig), ("recomp", difference.recomp))
        ),
    )


def _rendered_assignment(assignment: tuple[tuple[Hashable, int], ...]) -> str:
    parts = [f"{render(term)} = {_number(value)}" for term, value in assignment[:6]]
    if len(assignment) > 6:
        parts.append("…")
    return ", ".join(parts)


def _shown(value: int | bool) -> str:
    match value:
        case bool() as truth:
            return str(truth).lower()
        case number:
            return _number(number)


_GROUP_TEXT = {
    LOGIC: "LIKELY A LOGIC DIFFERENCE in the recovered source",
    ANNOTATION: "LIKELY AN ANNOTATION PROBLEM (symbols/globals), not the logic",
    TOOLING: "LIKELY NOT A SOURCE PROBLEM: a comparison artefact",
    "unexplained": "unexplained: read the marked lines",
}


def explain(analysis: ComparisonAnalysis) -> list[str]:
    """Lines explaining a mismatch's first divergence; empty for others."""
    found = explanation(analysis)
    if found is None:
        return []
    lines = [
        f"FIRST DIVERGENCE: {found.kind.replace('_', ' ')} — {_GROUP_TEXT[found.group]}",
        f"  orig   {found.orig}",
        f"  recomp {found.recomp}",
    ]
    if found.orig_value is not None:
        lines.append(f"  orig computes   {found.orig_value}")
        lines.append(f"  recomp computes {found.recomp_value}")
    else:
        lines += [f"  {name:<6} {text}" for name, text in found.facts if text]
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
