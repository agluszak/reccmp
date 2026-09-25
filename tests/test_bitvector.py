"""Bit-vector equivalence of the verifier's symbolic values (z3)."""

import pytest

from reccmp.compare.asm.verifier import bitvector, verify_effective_match
from reccmp.compare.asm.verifier.state import FunctionMetadata

pytestmark = pytest.mark.skipif(not bitvector.available(), reason="needs z3-solver")

EAX = ("init", "a")
ECX = ("init", "c")
LOAD = ("load", ("mem", "", ((ECX, 1),), 4, ()), "dword", 0)
BYTE = ("load", ("mem", "", ((ECX, 1),), 8, ()), "byte", 0)


def imm(value: int):
    return ("imm", value)


def test_the_same_low_byte_computed_at_different_widths():
    # and al, 1 on the low byte of x  ==  low byte of (x and 1)
    assert bitvector.values_equal(
        ("and", imm(1), ("l8", LOAD)), ("l8", ("and", imm(1), LOAD))
    )
    # reading al of a zeroed register is the constant 0
    assert bitvector.values_equal(("l8", imm(0)), imm(0))
    # but the whole register differs in its upper bits
    assert not bitvector.values_equal(("and", imm(0xFF), LOAD), LOAD)
    assert bitvector.values_equal(("and", imm(0xFF), LOAD), LOAD, 8)


def test_algebra_the_structural_comparison_misses():
    assert bitvector.values_equal(("sub", LOAD, imm(1)), ("dec", LOAD))
    assert bitvector.values_equal(("inc", ("not", LOAD)), ("neg", LOAD))
    assert bitvector.values_equal(("shl", LOAD, imm(1)), ("add", LOAD, LOAD))
    assert not bitvector.values_equal(("shr", LOAD, imm(1)), ("sar", LOAD, imm(1)))


def test_x86_semantics_at_the_edges():
    # the count is masked to five bits: shl by 33 is shl by 1
    assert bitvector.values_equal(("shl", LOAD, imm(33)), ("shl", LOAD, imm(1)))
    # a byte shifted by 8..31 is gone; by 32 it is masked to 0, untouched
    assert bitvector.values_equal(("shl", ("l8", LOAD), imm(8)), imm(0))
    assert bitvector.values_equal(("shl", ("l8", LOAD), imm(32)), ("l8", LOAD))
    # sign versus zero extension of a byte
    assert not bitvector.values_equal(("movzx", "byte", BYTE), ("movsx", "byte", BYTE))
    assert bitvector.values_equal(
        ("and", imm(0xFF), ("movsx", "byte", BYTE)), ("movzx", "byte", BYTE)
    )
    # inserting into al keeps the upper 24 bits of the register
    assert not bitvector.values_equal(("ins_l8", EAX, ("l8", LOAD)), LOAD)
    assert bitvector.values_equal(("ins_l8", LOAD, ("l8", LOAD)), LOAD)


def test_predicates():
    # x < 0x41 is x <= 0x40, unsigned
    assert bitvector.predicates_equal(
        ("lt_u", LOAD, imm(0x41), 4), ("le_u", LOAD, imm(0x40), 4)
    )
    # but not signed against unsigned
    assert not bitvector.predicates_equal(
        ("lt_u", LOAD, imm(0x41), 4), ("lt_s", LOAD, imm(0x41), 4)
    )
    # a byte comparison looks only at the low byte
    assert bitvector.predicates_equal(
        ("eq", (("l8", LOAD), imm(0)), 1), ("eq", (("and", imm(0xFF), LOAD), imm(0)), 4)
    )
    # flag states it cannot lower stay distinct unless identical
    opaque = ("cc", "o", ("flags", "add", LOAD, imm(1)))
    assert bitvector.predicates_equal(opaque, opaque)
    assert not bitvector.predicates_equal(
        opaque, ("cc", "no", ("flags", "add", LOAD, imm(1)))
    )


def test_terms_it_cannot_lower_are_not_proven():
    qword = ("load", ("mem", "", ((ECX, 1),), 4, ()), "qword", 0)
    assert not bitvector.values_equal(qword, ("add", qword, imm(0)))
    # an opaque term is an unconstrained value: equal only to itself
    call = ("callret", 3, "eax")
    assert bitvector.values_equal(("add", call, imm(0)), call)
    assert not bitvector.values_equal(call, ("callret", 4, "eax"))


def test_observations():
    store = ("store", ("mem", "", ((ECX, 1),), 4, ()), "byte")
    assert bitvector.entries_equal(
        (*store, ("and", imm(1), ("l8", LOAD))), (*store, ("l8", ("and", imm(1), LOAD)))
    )
    # the address and width must be the same, not merely equivalent
    other = ("store", ("mem", "", ((ECX, 1),), 8, ()), "byte")
    assert not bitvector.entries_equal((*store, ("l8", LOAD)), (*other, ("l8", LOAD)))
    assert not bitvector.entries_equal(
        ("call", "f", EAX), ("call", "f", ("add", EAX, imm(0)))
    )


def test_the_verifier_accepts_an_algebraic_identity():
    """`bool f(int* p) { return *p & 1; }` spelled at two widths."""
    orig = ["mov eax, dword ptr [ecx + 4]", "and al, 1", "ret"]
    recomp = ["mov eax, dword ptr [ecx + 4]", "and eax, 1", "ret"]
    byte_return = FunctionMetadata(return_kind="i8")
    assert verify_effective_match(orig, recomp, metadata=byte_return)
    # the whole of eax is returned: the upper bits differ
    assert not verify_effective_match(
        orig, recomp, metadata=FunctionMetadata(return_kind="i32")
    )
