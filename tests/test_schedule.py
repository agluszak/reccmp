"""Undoing instruction scheduling inside matched blocks (verifier.schedule).

Each case gives the original block and a recompiled block holding the same
instructions in a different order; the scheduler may move a recompiled
instruction towards the original order only past instructions it provably
does not depend on.
"""

from reccmp.compare.asm.ir import resolve_asm_stream
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.asm.verifier.schedule import schedule_like

# Hand-assembled 32-bit instructions.
STORE_THIS_4_EAX = "894104"  # mov dword ptr [ecx + 4], eax
STORE_THIS_8_EDX = "895108"  # mov dword ptr [ecx + 8], edx
STORE_THIS_4_EDX = "895104"  # mov dword ptr [ecx + 4], edx
STORE_ESI_EAX = "8906"  # mov dword ptr [esi], eax
LOAD_EAX_THIS_4 = "8b4104"  # mov eax, dword ptr [ecx + 4]
MOV_EAX_1 = "b801000000"  # mov eax, 1
MOV_EBX_2 = "bb02000000"  # mov ebx, 2
MOV_EDX_EAX = "8bd0"  # mov edx, eax
MOV_EAX_EDX = "8bc2"  # mov eax, edx
CMP_EAX_1 = "83f801"  # cmp eax, 1
TEST_EDX_EDX = "85d2"  # test edx, edx
SETE_BL = "0f94c3"  # sete bl
CALL = "e800000000"  # call $+5
PUSH_EAX = "50"  # push eax
POP_EDX = "5a"  # pop edx
REP_STOSD = "f3ab"  # rep stosd (not modelled)


def _stream(*instructions: str):
    code = bytes.fromhex("".join(instructions))
    return resolve_asm_stream(ParseAsm().parse_asm(code, 0x1000))


def _schedule(orig: tuple[str, ...], recomp: tuple[str, ...]) -> list[int]:
    """The recompiled block's instruction order after scheduling."""
    return schedule_like(
        _stream(*orig),
        _stream(*recomp),
        list(range(len(orig))),
        list(range(len(recomp))),
    )


def test_stores_to_distinct_fields_cross():
    assert _schedule(
        (STORE_THIS_4_EAX, STORE_THIS_8_EDX), (STORE_THIS_8_EDX, STORE_THIS_4_EAX)
    ) == [1, 0]


def test_stores_to_the_same_field_do_not_cross():
    # Both write [ecx + 4]: the last one wins, so their order matters.
    assert _schedule(
        (STORE_THIS_4_EAX, STORE_THIS_4_EDX), (STORE_THIS_4_EDX, STORE_THIS_4_EAX)
    ) == [0, 1]


def test_a_store_through_an_unknown_pointer_is_a_wall():
    # [esi] may alias [ecx + 4].
    assert _schedule(
        (STORE_THIS_4_EAX, STORE_ESI_EAX), (STORE_ESI_EAX, STORE_THIS_4_EAX)
    ) == [0, 1]


def test_a_load_does_not_cross_a_store_to_its_field():
    assert _schedule(
        (LOAD_EAX_THIS_4, STORE_THIS_4_EDX), (STORE_THIS_4_EDX, LOAD_EAX_THIS_4)
    ) == [0, 1]


def test_register_dependencies_hold():
    # read after write: mov edx, eax needs mov eax, 1 first
    assert _schedule((MOV_EDX_EAX, MOV_EAX_1), (MOV_EAX_1, MOV_EDX_EAX)) == [0, 1]
    # write after read: mov eax, 1 must stay after mov edx, eax
    assert _schedule((MOV_EAX_1, MOV_EDX_EAX), (MOV_EDX_EAX, MOV_EAX_1)) == [0, 1]
    # write after write: two writers of eax
    assert _schedule((MOV_EAX_EDX, MOV_EAX_1), (MOV_EAX_1, MOV_EAX_EDX)) == [0, 1]
    # independent registers cross
    assert _schedule((MOV_EBX_2, MOV_EAX_1), (MOV_EAX_1, MOV_EBX_2)) == [1, 0]


def test_flag_producer_and_consumer_keep_their_order():
    assert _schedule((SETE_BL, CMP_EAX_1), (CMP_EAX_1, SETE_BL)) == [0, 1]


def test_two_flag_producers_before_a_reader_do_not_swap():
    # sete reads the flags of whichever comparison ran last.
    orig = (CMP_EAX_1, TEST_EDX_EDX, SETE_BL)
    recomp = (TEST_EDX_EDX, CMP_EAX_1, SETE_BL)
    assert _schedule(orig, recomp) == [0, 1, 2]


def test_calls_and_unmodelled_instructions_are_barriers():
    for barrier in (CALL, REP_STOSD):
        assert _schedule((MOV_EBX_2, barrier), (barrier, MOV_EBX_2)) == [0, 1], barrier


def test_stack_operations_are_modelled_not_barriers():
    # push eax touches esp and [esp]; mov ebx, 2 touches neither.
    assert _schedule((MOV_EBX_2, PUSH_EAX), (PUSH_EAX, MOV_EBX_2)) == [1, 0]
    # push and pop both move esp: never reordered.
    assert _schedule((PUSH_EAX, POP_EDX), (POP_EDX, PUSH_EAX)) == [0, 1]


def test_repeated_identical_instructions_are_matched_in_order():
    # Two identical stores to [ecx + 8] around a store to [ecx + 4]: the
    # first copy is taken for the first original line and the block comes
    # out in the original order, each copy used once.
    orig = (STORE_THIS_8_EDX, STORE_THIS_4_EAX, STORE_THIS_8_EDX)
    recomp = (STORE_THIS_4_EAX, STORE_THIS_8_EDX, STORE_THIS_8_EDX)
    assert _schedule(orig, recomp) == [1, 0, 2]


def test_chains_of_legal_moves():
    # Three independent writes in reverse order are fully restored.
    orig = (MOV_EBX_2, STORE_THIS_4_EAX, STORE_THIS_8_EDX)
    recomp = (STORE_THIS_8_EDX, STORE_THIS_4_EAX, MOV_EBX_2)
    assert _schedule(orig, recomp) == [2, 1, 0]
    # A dependent instruction keeps its predecessor even when others move.
    orig = (MOV_EBX_2, MOV_EAX_1, MOV_EDX_EAX)
    recomp = (MOV_EAX_1, MOV_EDX_EAX, MOV_EBX_2)
    assert _schedule(orig, recomp) == [2, 0, 1]
