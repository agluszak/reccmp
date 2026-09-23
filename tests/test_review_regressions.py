"""Stage 0 architectural-review regression witnesses.

These tests encode the *correct* post-fix outcomes for known false-approval
bugs. They fail on the buggy baseline and pass once Stage 0 repairs land.
"""

from __future__ import annotations

import difflib
from unittest.mock import Mock

from reccmp.compare.asm.decode import disasm_detail
from reccmp.compare.asm.verifier import (
    verify_effective_match,
)
from reccmp.compare.asm.fixes import analyze_effective_match
from reccmp.compare.asm.instgen import InstructGen, InstructionMeta, SectionType
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.db import EntityDb
from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.cvdump.types import CvdumpTypesParser

# --- A1: reachable code after jmp-over-int3 must not vanish -----------------

_A1_ORIG = bytes.fromhex("EB01CCB801000000C3")  # jmp +1; int3; mov eax,1; ret
_A1_RECOMP = bytes.fromhex("EB01CCB802000000C3")  # jmp +1; int3; mov eax,2; ret


def _code_mnemonics(blob: bytes, base: int = 0x1000) -> list[tuple[str, str]]:
    """Collect (mnemonic, op_str) from InstructGen CODE sections."""
    rows: list[tuple[str, str]] = []
    for section in InstructGen(blob, base).sections:
        if section.type != SectionType.CODE:
            continue
        for _addr, _size, mnemonic, op_str in section.contents:
            rows.append((mnemonic, op_str))
    return rows


def test_a1_jmp_over_int3_extracts_divergent_mov_immediates():
    """Both sides jump over int3 then return different constants.

    Extraction must visit the pending branch target so the mov immediates
    survive; identical jmp-only lists must not silently become EXACT.
    """
    orig_rows = _code_mnemonics(_A1_ORIG)
    recomp_rows = _code_mnemonics(_A1_RECOMP)

    assert any(m == "mov" for m, _ in orig_rows)
    assert any(m == "mov" for m, _ in recomp_rows)
    assert orig_rows != recomp_rows

    orig_asm = [row.display for row in ParseAsm().parse_asm(_A1_ORIG, 0x1000)]
    recomp_asm = [row.display for row in ParseAsm().parse_asm(_A1_RECOMP, 0x1000)]
    assert any("mov" in line and "1" in line for line in orig_asm)
    assert any("mov" in line and "2" in line for line in recomp_asm)
    assert orig_asm != recomp_asm

    codes = difflib.SequenceMatcher(None, orig_asm, recomp_asm).get_opcodes()
    analysis = analyze_effective_match(codes, orig_asm, recomp_asm)
    assert analysis.status != ComparisonStatus.EXACT
    assert analysis.is_effective is False


def test_a1_disasm_detail_or_instructgen_sees_mov_after_jump():
    """Prefer detail/InstructGen coverage of the mov after the forward jmp."""
    # Linear Capstone stop-at-int3 may still truncate; InstructGen must not.
    for blob, imm in ((_A1_ORIG, "1"), (_A1_RECOMP, "2")):
        ig_movs = [op for m, op in _code_mnemonics(blob) if m == "mov"]
        assert ig_movs, "InstructGen must decode the mov after jmp-over-int3"
        assert any(imm in op for op in ig_movs)

        detail = disasm_detail(blob, 0x1000)
        detail_movs = [insn for insn in detail if insn.mnemonic == "mov"]
        if not detail_movs:
            # Accept InstructGen-only coverage until the shared detail walker
            # also drains pending targets; still require unequal ParseAsm lists.
            continue
        assert any(imm in insn.display for insn in detail_movs)


# --- A2: matched-node membership is not live-out proof at a branch ----------


def test_a2_edx_scratch_does_not_justify_live_eax_divergence():
    """edx was matched earlier, but eax differs on the taken jne path."""
    orig = [
        "mov ecx, 1",
        "mov edx, 1",
        "mov edx, 2",
        "mov eax, 1",
        "test ecx, ecx",
        "jne 0x2",
        "mov eax, 0",
        "ret",
    ]
    recomp = [
        "mov ecx, 1",
        "mov edx, 1",
        "mov edx, 2",
        "mov eax, 2",
        "test ecx, ecx",
        "jne 0x2",
        "mov eax, 0",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


# --- A3: one-sided self-jmp island cycle is not an alias proof --------------


def _alias_image(bodies: dict[int, bytes]) -> Mock:
    image = Mock(spec=[])
    image.read = Mock(side_effect=lambda addr, size: bodies[addr][:size])
    image.imagebase = 0
    image.is_relocated_addr = Mock(return_value=False)
    image.is_debug = Mock(return_value=False)
    return image


def test_a3_self_jmp_island_vs_return_one_is_not_alias_equivalent():
    """``E9 FB FF FF FF`` jumps to itself; recomp returns 1. Extent 6."""
    orig_addr = 0x200
    recomp_addr = 0x400
    island = bytes.fromhex("E9FBFFFFFF90")  # jmp $-5 ; nop
    ret1 = bytes.fromhex("B801000000C3")  # mov eax,1 ; ret
    assert len(island) == 6 and len(ret1) == 6

    comparator = FunctionComparator(
        EntityDb(),
        LinesDb(),
        _alias_image({orig_addr: island}),
        _alias_image({recomp_addr: ret1}),
        Mock(spec=ReccmpReportProtocol),
        CvdumpTypesParser(),
    )
    assert (
        comparator.raw_pair_alias_equivalent(orig_addr, recomp_addr, len(island))
        is False
    )


# --- A4: virtual call must observe the actual receiver ----------------------


def test_a4_virtual_call_different_lea_ecx_offsets_not_effective():
    """Same vtable slot, different ``this`` after lea ecx."""
    orig = [
        "mov eax, dword ptr [ecx]",
        "lea ecx, [ecx]",
        "call dword ptr [eax + 4]",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [ecx]",
        "lea ecx, [ecx + 4]",
        "call dword ptr [eax + 4]",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


def test_a4_pre_call_stack_discrepancy_survives_callesp():
    """Post-call ESP tokens must retain dependence on incoming ESP."""
    orig = [
        "sub esp, 4",
        "call 0x1000",
        "add esp, 4",
        "ret",
    ]
    recomp = [
        "sub esp, 8",
        "call 0x1000",
        "add esp, 4",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


# --- A5a: comparison width must distinguish cmp al vs cmp ax ----------------


def test_a5a_cmp_al_vs_cmp_ax_with_jg_not_effective():
    """Signed jg over 0x80 differs for 8-bit vs 16-bit compare width."""
    orig = [
        "mov eax, 0",
        "mov al, 0x80",
        "cmp al, 0",
        "mov eax, 0",
        "jg 0x2",
        "mov eax, 1",
        "ret",
        "mov eax, 2",
        "ret",
    ]
    recomp = [
        "mov eax, 0",
        "mov ax, 0x80",
        "cmp ax, 0",
        "mov eax, 0",
        "jg 0x2",
        "mov eax, 1",
        "ret",
        "mov eax, 2",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


# --- A5b: SAHF preserves OF -------------------------------------------------


def test_a5b_sahf_preserves_overflow_flag_difference():
    """Prior OF from add must survive SAHF and affect jo."""
    orig = [
        "mov eax, 0x7fffffff",
        "add eax, 1",
        "mov eax, 0",
        "sahf",
        "jo 0x2",
        "mov eax, 1",
        "ret",
        "mov eax, 2",
        "ret",
    ]
    recomp = [
        "mov eax, 0",
        "add eax, 1",
        "mov eax, 0",
        "sahf",
        "jo 0x2",
        "mov eax, 1",
        "ret",
        "mov eax, 2",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


# --- A6: unsupported path requires compatible meta on both sides ------------


def _bswap_meta(reads: tuple[str, ...], writes: tuple[str, ...]) -> InstructionMeta:
    return InstructionMeta(
        address=4,
        size=2,
        mnemonic="bswap",
        regs_read=reads,
        regs_written=writes,
        reads_flags=False,
        writes_flags=False,
        accesses_memory=False,
        is_jump=False,
        is_call=False,
        is_ret=False,
        branch_target=None,
    )


_BSWAP_ORIG = [
    "mov eax, dword ptr [esi]",
    "mov ecx, dword ptr [edi]",
    "bswap ecx",
    "mov dword ptr [ebx], ecx",
]
_BSWAP_RECOMP = [
    "mov edx, dword ptr [esi]",
    "mov ecx, dword ptr [edi]",
    "bswap ecx",
    "mov dword ptr [ebx], ecx",
]


def test_a6_one_sided_meta_cannot_step_divergent_unsupported_instruction():
    """``meta_o or meta_r`` must not invent agreement when only one side has meta."""
    orig, recomp = _BSWAP_ORIG, _BSWAP_RECOMP
    one_sided = [None, None, _bswap_meta(("ecx",), ("ecx",)), None]
    assert verify_effective_match(orig, recomp, orig_meta=one_sided) is False
    assert verify_effective_match(orig, recomp, recomp_meta=one_sided) is False


def test_meta_step_over_unmodeled_instruction():
    """With capstone metadata, an unmodeled register-only instruction
    (bswap) can be stepped over even while a rename is in flight — its
    reads must agree, its writes become fresh paired values."""
    orig, recomp = _BSWAP_ORIG, _BSWAP_RECOMP
    # Without metadata: bswap requires full sync, but eax/edx diverge.
    assert verify_effective_match(orig, recomp) is False
    both = [None, None, _bswap_meta(("ecx",), ("ecx",)), None]
    assert (
        verify_effective_match(orig, recomp, orig_meta=both, recomp_meta=both) is True
    )
    # If the bswap reads a diverged register, it must still reject.
    bad = [None, None, _bswap_meta(("eax",), ("eax",)), None]
    orig2 = [orig[0], orig[1], "bswap eax", "mov dword ptr [ebx], ecx"]
    recomp2 = [recomp[0], recomp[1], "bswap eax", "mov dword ptr [ebx], ecx"]
    assert (
        verify_effective_match(orig2, recomp2, orig_meta=bad, recomp_meta=bad) is False
    )
