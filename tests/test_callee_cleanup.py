"""The stack effect of a call, read from a binary without running it
(reccmp.compare.callee_cleanup.StaticCode), and how a paired call's two
effects are reconciled (iso_cfg._agreed_effects)."""

from types import SimpleNamespace

from reccmp.compare.asm.verifier.iso_cfg import _agreed_effects
from reccmp.compare.callee_cleanup import CallStackEffect, StaticCode
from reccmp.compare.extent import EntityExtent
from reccmp.formats.image import ImageSection, ImageSectionFlags

CODE = 0x401000
CALLER, CALLEE = CODE, CODE + 0x100


def _rel32(at: int, target: int) -> bytes:
    return ((target - at) & 0xFFFFFFFF).to_bytes(4, "little")


def _call(at: int, target: int) -> bytes:
    return b"\xe8" + _rel32(at + 5, target)


def _code(caller: bytes, callee: bytes, *, window: int | None = None) -> StaticCode:
    page = bytearray(0x1000)
    page[: len(caller)] = caller
    page[CALLEE - CODE : CALLEE - CODE + len(callee)] = callee
    image = SimpleNamespace(
        sections=[
            ImageSection(
                virtual_range=range(CODE, CODE + len(page)),
                physical_range=range(0, len(page)),
                view=memoryview(bytes(page)),
                flags=ImageSectionFlags.EXECUTE,
            )
        ]
    )
    extent = EntityExtent(window) if window is not None else None
    return StaticCode(
        image,  # type: ignore[arg-type]
        function_window=lambda address: extent if address == CALLEE else None,
    )


def test_a_callee_that_pops_its_arguments():
    # push 1; push 2; call callee | callee: ret 8
    caller = bytes.fromhex("6a016a02") + _call(CALLER + 4, CALLEE)
    code = _code(caller, bytes.fromhex("c20800"))
    assert code.call_effect(CALLER + 4) == CallStackEffect(8, 8, True)


def test_a_caller_that_removes_the_arguments():
    # push 1; call callee; add esp, 4 | callee: ret
    caller = bytes.fromhex("6a01") + _call(CALLER + 2, CALLEE) + bytes.fromhex("83c404")
    code = _code(caller, bytes.fromhex("c3"))
    assert code.call_effect(CALLER + 2) == CallStackEffect(0, 4, True)


def test_a_return_past_a_call_is_uncertain():
    # callee: call elsewhere; ret 4 (the bytes after a call may be the next
    # function's, unless the callee's size is recorded)
    callee = _call(CALLEE, CODE + 0x800) + bytes.fromhex("c20400")
    caller = _call(CALLER, CALLEE)
    assert _code(caller, callee).call_effect(CALLER) == CallStackEffect(4, 4, False)
    recorded = _code(caller, callee, window=len(callee))
    assert recorded.call_effect(CALLER) == CallStackEffect(4, 4, True)


def test_a_computed_callee_has_no_known_effect():
    # call eax
    assert _code(bytes.fromhex("ffd0"), b"").call_effect(CALLER) is None
    # ... unless the caller removes the arguments itself: call eax; add esp, 8
    code = _code(bytes.fromhex("ffd083c408"), b"")
    assert code.call_effect(CALLER) == CallStackEffect(0, 8, True)


def test_an_uncertain_effect_needs_the_other_sides_certain_one():
    certain = CallStackEffect(4, 4, True)
    uncertain = CallStackEffect(4, 4, False)
    assert _agreed_effects(uncertain, certain) == (uncertain, certain)
    assert _agreed_effects(certain, uncertain) == (certain, uncertain)
    assert _agreed_effects(uncertain, CallStackEffect(8, 8, True))[0] is None
    assert _agreed_effects(uncertain, uncertain) == (None, None)
    assert _agreed_effects(uncertain, None) == (None, None)
