"""Tests for CanonicalStackRef heuristics and helper effect summaries."""

from reccmp.compare.asm.ir import DecodedInstruction, AsmRole
from reccmp.compare.inlines import (
    HelperCatalogEntry,
    StoreEffect,
    analyze_inline_layout,
    asm_fingerprint_from_ir,
    asm_fingerprint_from_lines,
    register_normalized_fingerprint,
    summarize_helper_effects,
)
from reccmp.compare.stack_layout import canonical_stack_ref


def test_canonical_stack_ref_heuristics():
    assert canonical_stack_ref("ebp", 8).label() == "arg[0]"
    assert canonical_stack_ref("ebp", 0xC).label() == "arg[1]"
    assert canonical_stack_ref("ebp", -0x18).label() == "local[-24]"
    assert canonical_stack_ref("ebp", 0).kind == "saved"
    # Unaligned / packed positive ebp offsets are not dword args.
    assert canonical_stack_ref("ebp", 0xA).kind == "unknown"
    assert canonical_stack_ref("esp", 4).kind == "unknown"
    assert canonical_stack_ref("esp", 4, known_spills={4}).kind == "spill"


def test_summarize_helper_effects_detects_this_stores():
    fingerprint = (
        ("mov", "dword ptr [ecx + 0x4], eax"),
        ("mov", "dword ptr [ecx + 0x8], edx"),
        ("ret", ""),
    )
    summary = summarize_helper_effects(fingerprint)
    assert summary is not None
    assert summary.inputs == ("ecx", "edx")
    assert summary.stores == (
        StoreEffect("this", 4),
        StoreEffect("this", 8),
    )


def test_literal_inline_match_is_not_marked_semantic():
    """Exact fingerprint hits are never semantic — only non-literal matches are."""
    body = (
        "mov eax, dword ptr [ecx]",
        "mov dword ptr [ecx + 0x4], eax",
        "xor edx, edx",
        "ret",
    )
    fingerprint = tuple(
        (line.partition(" ")[0], line.partition(" ")[2]) for line in body[:-1]
    )
    summary = summarize_helper_effects(fingerprint)
    assert summary is not None
    helper = HelperCatalogEntry(
        orig_addr=0x100,
        recomp_addr=0x200,
        name="Foo::setX",
        fingerprint=fingerprint,
        byte_size=16,
        uniqueness=1.0,
        effect_summary=summary,
    )
    orig = [
        "push ebx",
        "mov eax, dword ptr [ecx]",
        "mov dword ptr [ecx + 0x4], eax",
        "xor edx, edx",
        "pop ebx",
    ]
    recomp = [
        "push ebx",
        "call Foo::setX",
        "pop ebx",
    ]
    result = analyze_inline_layout(orig, recomp, [helper])
    assert len(result.expansions) == 1
    assert result.expansions[0].semantic is False
    assert result.expansions[0].confidence >= 0.25


def test_register_normalized_inline_match_is_semantic():
    """Differing registers that share shape match via normalized fingerprint."""
    helper_fp = (
        ("mov", "eax, dword ptr [ecx]"),
        ("mov", "dword ptr [ecx + 0x4], eax"),
        ("xor", "edx, edx"),
    )
    # Host uses ebx instead of eax for the temporary — literal miss, register hit.
    orig = [
        "push esi",
        "mov ebx, dword ptr [ecx]",
        "mov dword ptr [ecx + 0x4], ebx",
        "xor edx, edx",
        "pop esi",
    ]
    recomp = [
        "push esi",
        "call Foo::setX",
        "pop esi",
    ]
    helper = HelperCatalogEntry(
        orig_addr=0x100,
        recomp_addr=0x200,
        name="Foo::setX",
        fingerprint=helper_fp,
        byte_size=16,
        uniqueness=1.0,
        effect_summary=summarize_helper_effects(helper_fp),
    )
    assert register_normalized_fingerprint(
        helper_fp
    ) == register_normalized_fingerprint(
        asm_fingerprint_from_lines(
            [
                "mov ebx, dword ptr [ecx]",
                "mov dword ptr [ecx + 0x4], ebx",
                "xor edx, edx",
            ]
        )
    )
    result = analyze_inline_layout(orig, recomp, [helper])
    assert len(result.expansions) == 1
    assert result.expansions[0].semantic is True


def test_asm_fingerprint_from_ir_uses_structured_operands():
    rows = [
        DecodedInstruction(
            address=0x1000,
            size=3,
            mnemonic="mov",
            prefix="",
            operands=(("reg", "eax"), ("mem", "dword", "", (("ecx", 1),), 4, ())),
            raw_operands=("eax", "dword ptr [ecx + 0x4]"),
            display="mov eax, dword ptr [ecx + 0x4]",
            role=AsmRole.CODE,
        ),
        DecodedInstruction(
            address=None,
            size=0,
            mnemonic="",
            prefix="",
            operands=(),
            raw_operands=(),
            display="Jump table:",
            role=AsmRole.JUMP_TABLE_HEADER,
        ),
    ]
    fp = asm_fingerprint_from_ir(rows)
    assert fp == (("mov", "eax, dword ptr [ecx + 4]"),)
