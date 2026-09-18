"""Tests for CanonicalStackRef heuristics and helper effect summaries."""

from reccmp.compare.inlines import (
    HelperCatalogEntry,
    StoreEffect,
    analyze_inline_layout,
    summarize_helper_effects,
)
from reccmp.compare.stack_layout import canonical_stack_ref


def test_canonical_stack_ref_heuristics():
    assert canonical_stack_ref("ebp", 8).label() == "arg[0]"
    assert canonical_stack_ref("ebp", 0xC).label() == "arg[1]"
    assert canonical_stack_ref("ebp", -0x18).label() == "local[-24]"
    assert canonical_stack_ref("ebp", 0).kind == "saved"
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


def test_semantic_inline_raises_confidence_when_summaries_match():
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
    assert StoreEffect("this", 4) in summary.stores
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
    assert result.expansions[0].semantic is True
    assert result.expansions[0].confidence >= 0.25
