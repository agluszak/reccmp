"""Bit-vector equivalence of the verifier's symbolic values (z3)."""

from reccmp.compare.asm.verifier import bitvector
from reccmp.compare.asm.verifier.addresses import (
    AddressTerm,
    CallResult,
    CfgMemoryInit,
    Compare,
    CompareKind,
    ConditionCode,
    Constant,
    Extend,
    ExtendKind,
    Extract,
    FlagsResult,
    Init,
    Insert,
    Load,
    MemoryAddress,
    MemoryStore,
    Operation,
    OperationKind,
    RegisterPart,
    SymbolValue,
    UnaryOperation,
)
from reccmp.compare.asm.verifier.state import Call, FunctionMetadata, Store
from tests.asm_rows import verify_effective_match

EAX = Init("a")
ECX = Init("c")
ENTRY_MEMORY = CfgMemoryInit()
LOAD = Load(MemoryAddress("", (AddressTerm(ECX, 1),), 4, ()), "dword", ENTRY_MEMORY)
BYTE = Load(MemoryAddress("", (AddressTerm(ECX, 1),), 8, ()), "byte", ENTRY_MEMORY)


def imm(value: int):
    return Constant(value)


def op(kind: str, *operands):
    return Operation(OperationKind(kind), tuple(operands))


def unary(kind: str, operand):
    return UnaryOperation(OperationKind(kind), operand)


def extract(part: str, whole):
    return Extract(RegisterPart(part), whole)


def insert(part: str, old, new):
    return Insert(RegisterPart(part), old, new)


def extend(kind: str, width: str, source):
    return Extend(ExtendKind(kind), width, source)


def pred(kind: str, left, right, width=None):
    return Compare(CompareKind(kind), left, right, width)


def test_the_same_low_byte_computed_at_different_widths():
    # and al, 1 on the low byte of x  ==  low byte of (x and 1)
    assert bitvector.values_equal(
        op("and", imm(1), extract("l8", LOAD)),
        extract("l8", op("and", imm(1), LOAD)),
    )
    # reading al of a zeroed register is the constant 0
    assert bitvector.values_equal(extract("l8", imm(0)), imm(0))
    # but the whole register differs in its upper bits
    assert not bitvector.values_equal(op("and", imm(0xFF), LOAD), LOAD)
    assert bitvector.values_equal(op("and", imm(0xFF), LOAD), LOAD, 8)


def test_algebra_the_structural_comparison_misses():
    assert bitvector.values_equal(op("sub", LOAD, imm(1)), unary("dec", LOAD))
    assert bitvector.values_equal(unary("inc", unary("not", LOAD)), unary("neg", LOAD))
    assert bitvector.values_equal(op("shl", LOAD, imm(1)), op("add", LOAD, LOAD))
    assert not bitvector.values_equal(op("shr", LOAD, imm(1)), op("sar", LOAD, imm(1)))


def test_x86_semantics_at_the_edges():
    # the count is masked to five bits: shl by 33 is shl by 1
    assert bitvector.values_equal(op("shl", LOAD, imm(33)), op("shl", LOAD, imm(1)))
    # a byte shifted by 8..31 is gone; by 32 it is masked to 0, untouched
    assert bitvector.values_equal(op("shl", extract("l8", LOAD), imm(8)), imm(0))
    assert bitvector.values_equal(
        op("shl", extract("l8", LOAD), imm(32)), extract("l8", LOAD)
    )
    # sign versus zero extension of a byte
    assert not bitvector.values_equal(
        extend("movzx", "byte", BYTE), extend("movsx", "byte", BYTE)
    )
    assert bitvector.values_equal(
        op("and", imm(0xFF), extend("movsx", "byte", BYTE)),
        extend("movzx", "byte", BYTE),
    )
    # inserting into al keeps the upper 24 bits of the register
    assert not bitvector.values_equal(insert("l8", EAX, extract("l8", LOAD)), LOAD)
    assert bitvector.values_equal(insert("l8", LOAD, extract("l8", LOAD)), LOAD)


def test_predicates():
    # x < 0x41 is x <= 0x40, unsigned
    assert bitvector.predicates_equal(
        pred("lt_u", LOAD, imm(0x41), 4), pred("le_u", LOAD, imm(0x40), 4)
    )
    # but not signed against unsigned
    assert not bitvector.predicates_equal(
        pred("lt_u", LOAD, imm(0x41), 4), pred("lt_s", LOAD, imm(0x41), 4)
    )
    # a byte comparison looks only at the low byte
    assert bitvector.predicates_equal(
        pred("eq", extract("l8", LOAD), imm(0), 1),
        pred("eq", op("and", imm(0xFF), LOAD), imm(0), 4),
    )
    # flag states it cannot lower stay distinct unless identical
    flags = FlagsResult(op("add", LOAD, imm(1)))
    opaque = ConditionCode("o", flags)
    assert bitvector.predicates_equal(opaque, opaque)
    assert not bitvector.predicates_equal(opaque, ConditionCode("no", flags))


def test_terms_it_cannot_lower_are_not_proven():
    qword = Load(
        MemoryAddress("", (AddressTerm(ECX, 1),), 4, ()), "qword", ENTRY_MEMORY
    )
    assert not bitvector.values_equal(qword, op("add", qword, imm(0)))
    # an opaque term is an unconstrained value: equal only to itself
    call = CallResult(3, "eax")
    assert bitvector.values_equal(op("add", call, imm(0)), call)
    assert not bitvector.values_equal(call, CallResult(4, "eax"))


def test_observations():
    address = MemoryAddress("", (AddressTerm(ECX, 1),), 4, ())
    assert bitvector.entries_equal(
        Store(address, "byte", op("and", imm(1), extract("l8", LOAD))),
        Store(address, "byte", extract("l8", op("and", imm(1), LOAD))),
    )
    # the address and width must be the same, not merely equivalent
    other = MemoryAddress("", (AddressTerm(ECX, 1),), 8, ())
    assert not bitvector.entries_equal(
        Store(address, "byte", extract("l8", LOAD)),
        Store(other, "byte", extract("l8", LOAD)),
    )
    assert not bitvector.entries_equal(
        Call(SymbolValue("f"), (EAX,)),
        Call(SymbolValue("f"), (op("add", EAX, imm(0)),)),
    )


def test_the_verifier_accepts_an_algebraic_identity():
    """`bool f(int* p) { return *p & 1; }` spelled at two widths."""
    orig = ["mov eax, dword ptr [ecx + 4]", "and al, 1", "ret"]
    recomp = ["mov eax, dword ptr [ecx + 4]", "and eax, 1", "ret"]
    byte_return = FunctionMetadata(return_kind="i8")
    assert verify_effective_match(orig, recomp, metadata=byte_return)
    # a project may turn algebraic identities off
    no_algebra = FunctionMetadata(return_kind="i8", algebraic_identities=False)
    assert not verify_effective_match(orig, recomp, metadata=no_algebra)
    # the whole of eax is returned: the upper bits differ
    assert not verify_effective_match(
        orig, recomp, metadata=FunctionMetadata(return_kind="i32")
    )


def _at(displacement: int, size: str, generation=ENTRY_MEMORY):
    return Load(
        MemoryAddress("", (AddressTerm(ECX, 1),), displacement, ()),
        size,
        generation,
    )


def test_a_narrower_load_is_the_bytes_of_a_wider_one():
    """The low byte of the dword at p is the byte at p (little-endian)."""
    dword, byte = _at(4, "dword"), _at(4, "byte")
    assert bitvector.values_equal(extract("l8", dword), byte)
    assert bitvector.values_equal(extract("h8", dword), _at(5, "byte"))
    assert bitvector.values_equal(extract("r16", dword), _at(4, "word"))
    # Bit 4 of the low byte, read either way (GetLevelDataFlag4).
    assert bitvector.values_equal(
        op("and", imm(1), op("shr", extract("l8", dword), imm(4))),
        extract(
            "l8",
            op("and", imm(1), op("shr", insert("l8", EAX, byte), imm(4))),
        ),
    )


def test_bytes_outside_the_wider_load_stay_unknown():
    dword = _at(4, "dword")
    assert not bitvector.values_equal(extract("l8", dword), _at(5, "byte"))
    assert not bitvector.values_equal(extract("h8", dword), _at(8, "byte"))
    # Another memory generation is another read.
    assert not bitvector.values_equal(
        extract("l8", dword), _at(4, "byte", generation=MemoryStore(1, 0))
    )
