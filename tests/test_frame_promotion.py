"""A local kept in a stack slot on one side and in a register on the other
(reccmp.compare.asm.verifier.frame): the product verifies it with each
side's private frame held apart, and must not where the frame could be
reached, or a callee could see a difference."""

from reccmp.compare.asm.verifier.state import FunctionMetadata
from reccmp.compare.callee_cleanup import CallStackEffect
from reccmp.compare.asm.verifier.iso_cfg import ProductResult
from reccmp.compare.diagnosis import EffectiveReason
from tests.asm_rows import run_product


def _run(orig, recomp, *, effects=None) -> ProductResult:
    """Straight-line code: no local branch targets. ``effects`` maps a line
    index to the stack effect of the call there (the same on both sides)."""
    metadata = None
    if effects is not None:
        metadata = FunctionMetadata(
            stack_effects=(effects.get, effects.get), algebraic_identities=False
        )
    return run_product(
        orig,
        recomp,
        [None] * len(orig),
        [None] * len(recomp),
        metadata=metadata,
        orig_addrs=list(range(len(orig))),
        recomp_addrs=list(range(len(recomp))),
    )


def _verify(orig, recomp, *, effects=None) -> bool:
    return _run(orig, recomp, effects=effects).proved


def test_a_local_in_a_slot_against_one_in_a_register():
    orig = [
        "sub esp, 4",
        "mov dword ptr [esp], 0",
        "mov eax, dword ptr [esp + 8]",
        "add dword ptr [esp], eax",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "xor ecx, ecx",
        "mov eax, dword ptr [esp + 4]",
        "add ecx, eax",
        "mov eax, ecx",
        "ret",
    ]
    result = _run(orig, recomp)
    assert result.proved
    assert EffectiveReason.FRAME_SLOT_PROMOTION in result.recorder.reasons


def test_the_slot_must_hold_what_the_register_does():
    orig = [
        "sub esp, 4",
        "mov dword ptr [esp], 1",
        "mov eax, dword ptr [esp + 8]",
        "add dword ptr [esp], eax",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "xor ecx, ecx",
        "mov eax, dword ptr [esp + 4]",
        "add ecx, eax",
        "mov eax, ecx",
        "ret",
    ]
    assert not _verify(orig, recomp)


def test_an_uninitialized_slot_is_not_a_value():
    """Two reads of never-written locals are two unknown values."""
    orig = ["sub esp, 4", "mov eax, dword ptr [esp]", "add esp, 4", "ret"]
    recomp = ["sub esp, 8", "mov eax, dword ptr [esp + 4]", "add esp, 8", "ret"]
    assert not _verify(orig, recomp)


def test_a_spill_across_a_call():
    """The value survives the call in a slot on one side and in a
    callee-saved register on the other; the callee pops its argument."""
    orig = [
        "sub esp, 4",
        "mov eax, dword ptr [esp + 8]",
        "mov dword ptr [esp], eax",
        "push 1",
        "call f",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov esi, dword ptr [esp + 8]",
        "push 1",
        "call f",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    effects = {4: CallStackEffect(4, 4), 3: CallStackEffect(4, 4)}
    assert _verify(orig, recomp, effects=effects)


def test_the_callee_sees_each_sides_arguments():
    orig = [
        "sub esp, 4",
        "mov eax, dword ptr [esp + 8]",
        "mov dword ptr [esp], eax",
        "push 1",
        "call f",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov esi, dword ptr [esp + 8]",
        "push 2",
        "call f",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    effects = {4: CallStackEffect(4, 4), 3: CallStackEffect(4, 4)}
    assert not _verify(orig, recomp, effects=effects)


def test_a_slot_is_not_read_across_a_call_of_unknown_cleanup():
    """Without the callee's ``ret N`` the stack pointer after the call is
    unknown, so the slot cannot be placed."""
    orig = [
        "sub esp, 4",
        "mov eax, dword ptr [esp + 8]",
        "mov dword ptr [esp], eax",
        "push 1",
        "call f",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov esi, dword ptr [esp + 8]",
        "push 1",
        "call f",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    assert not _verify(orig, recomp, effects={})


def test_a_callee_may_write_its_arguments():
    """A slot the callee received as an argument holds what the callee left
    there, not what was passed."""
    orig = [
        "mov eax, dword ptr [esp + 4]",
        "push eax",
        "call f",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov esi, dword ptr [esp + 8]",
        "push esi",
        "call f",
        "add esp, 4",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    effects = {2: CallStackEffect(0, 4), 3: CallStackEffect(0, 4)}
    assert not _verify(orig, recomp, effects=effects)


def test_a_frame_whose_address_leaves_is_not_promoted():
    """A callee given a pointer into the frame may read any slot: locals
    only it can see still differ."""
    orig = [
        "sub esp, 4",
        "mov dword ptr [esp], 0",
        "lea eax, [esp]",
        "push eax",
        "call f",
        "xor eax, eax",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "sub esp, 4",
        "mov dword ptr [esp], 1",
        "lea eax, [esp]",
        "push eax",
        "call f",
        "xor eax, eax",
        "add esp, 4",
        "ret",
    ]
    effects = {4: CallStackEffect(4, 4)}
    assert not _verify(orig, recomp, effects=effects)


def test_a_callee_given_a_register_sees_what_it_holds():
    """The same, with the pointer's local identical on both sides but for
    where the function keeps a copy: still a proof."""
    orig = [
        "sub esp, 4",
        "mov dword ptr [esp], 0",
        "push 7",
        "call f",
        "mov eax, dword ptr [esp]",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "push 7",
        "call f",
        "xor eax, eax",
        "ret",
    ]
    effects = {3: CallStackEffect(4, 4), 1: CallStackEffect(4, 4)}
    assert _verify(orig, recomp, effects=effects)


def test_an_indexed_access_may_reach_a_promoted_slot():
    orig = [
        "sub esp, 8",
        "mov dword ptr [esp], 0",
        "mov ecx, dword ptr [esp + 0xc]",
        "mov dword ptr [esp + ecx*4], 5",
        "mov eax, dword ptr [esp]",
        "add esp, 8",
        "ret",
    ]
    recomp = [
        "sub esp, 8",
        "mov dword ptr [esp], 0",
        "mov ecx, dword ptr [esp + 0xc]",
        "mov dword ptr [esp + ecx*4], 5",
        "xor eax, eax",
        "add esp, 8",
        "ret",
    ]
    assert not _verify(orig, recomp)


def _escape(through: list[str], value: int) -> list[str]:
    return [
        "sub esp, 4",
        f"mov dword ptr [esp], {value}",
        *through,
        "call f",
        "xor eax, eax",
        "add esp, 4",
        "ret",
    ]


def test_a_frame_pointer_stored_to_memory_escapes():
    through = ["lea eax, [esp]", "mov dword ptr [g], eax"]
    effects = {4: CallStackEffect(0, 0)}
    assert not _verify(_escape(through, 0), _escape(through, 1), effects=effects)


def test_a_frame_pointer_passed_in_a_register_escapes():
    through = ["lea ecx, [esp]"]
    effects = {3: CallStackEffect(0, 0)}
    assert not _verify(_escape(through, 0), _escape(through, 1), effects=effects)
