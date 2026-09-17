"""Tests for structured IR match keys and stack-normalized scoring."""

from reccmp.compare.asm.ir import (
    instruction_match_key,
    rewrite_stack_displacements,
    stack_normalized_key,
)
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    EquivalenceLevel,
    derive_equivalence_level,
)
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins
from reccmp.compare.stack_layout import (
    StackPair,
    StackRegisterOffset,
    accuracy_after_stack_map,
    build_slot_bijection,
    extract_stack_offset_from_instruction,
)
from reccmp.compare.inlines import find_inline_expansions


def test_instruction_match_key_ignores_display_whitespace_equivalence():
    a = instruction_match_key("mov eax, dword ptr [ebp - 0x18]")
    b = instruction_match_key("mov eax, dword ptr [ebp - 0x18]")
    assert a == b
    assert a != instruction_match_key("mov ecx, dword ptr [ebp - 0x18]")


def test_stack_normalized_key_collapses_frame_displacements():
    a = stack_normalized_key("mov eax, dword ptr [ebp - 0x18]")
    b = stack_normalized_key("mov eax, dword ptr [ebp - 0x24]")
    assert a == b
    assert instruction_match_key("mov eax, dword ptr [ebp - 0x18]") != (
        instruction_match_key("mov eax, dword ptr [ebp - 0x24]")
    )


def test_ir_keyed_sequence_matcher_matches_string_ratio_for_identical_streams():
    orig = ["mov eax, ecx", "add eax, 1", "ret"]
    recomp = list(orig)
    string_ratio = SequenceMatcherWithPins(orig, recomp, []).ratio()
    key_ratio = SequenceMatcherWithPins(
        [instruction_match_key(x) for x in orig],
        [instruction_match_key(x) for x in recomp],
        [],
    ).ratio()
    assert string_ratio == key_ratio == 1.0


def test_extract_and_bijection_for_swapped_locals():
    pairs = {
        StackPair(
            StackRegisterOffset("ebp", -0x24),
            StackRegisterOffset("ebp", -0x18),
        ),
        StackPair(
            StackRegisterOffset("ebp", -0x18),
            StackRegisterOffset("ebp", -0x24),
        ),
    }
    mapping, bijective = build_slot_bijection(pairs)
    assert bijective
    assert mapping[("ebp", -0x24)] == ("ebp", -0x18)
    assert mapping[("ebp", -0x18)] == ("ebp", -0x24)


def test_accuracy_after_stack_map_reaches_one():
    orig = [
        "mov eax, dword ptr [ebp - 0x24]",
        "mov ecx, dword ptr [ebp - 0x18]",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [ebp - 0x18]",
        "mov ecx, dword ptr [ebp - 0x24]",
        "ret",
    ]
    mapping = {
        ("ebp", -0x24): ("ebp", -0x18),
        ("ebp", -0x18): ("ebp", -0x24),
    }
    assert accuracy_after_stack_map(orig, recomp, mapping) == 1.0


def test_rewrite_stack_displacements():
    line = "mov eax, dword ptr [ebp - 0x24]"
    rewritten = rewrite_stack_displacements(
        line, {("ebp", -0x24): ("ebp", -0x18)}
    )
    assert "ebp - 0x18" in rewritten
    assert extract_stack_offset_from_instruction(rewritten) == StackRegisterOffset(
        "ebp", -0x18
    )


def test_derive_equivalence_level_lattice():
    assert (
        derive_equivalence_level(ComparisonAnalysis.exact())
        == EquivalenceLevel.EXACT_INSTRUCTIONS
    )
    assert (
        derive_equivalence_level(
            ComparisonAnalysis.effective({"register_allocation"})
        )
        == EquivalenceLevel.REGISTER_ALLOCATION_EQUIVALENT
    )
    assert (
        derive_equivalence_level(
            ComparisonAnalysis.effective({"frame_slot_layout"})
        )
        == EquivalenceLevel.STACK_LAYOUT_EQUIVALENT
    )
    assert (
        derive_equivalence_level(
            ComparisonAnalysis.inconclusive("analysis_limit"),
            accuracy_modulo_stack=1.0,
        )
        == EquivalenceLevel.STACK_LAYOUT_EQUIVALENT
    )
    assert (
        derive_equivalence_level(
            ComparisonAnalysis.inconclusive("analysis_limit"),
            accuracy_modulo_inline=1.0,
        )
        == EquivalenceLevel.KNOWN_INLINE_EQUIVALENT
    )
    assert (
        derive_equivalence_level(
            ComparisonAnalysis.inconclusive("analysis_limit"),
            accuracy_modulo_stack=1.0,
            accuracy_modulo_inline=1.0,
        )
        == EquivalenceLevel.KNOWN_INLINE_EQUIVALENT
    )
    assert (
        derive_equivalence_level(ComparisonAnalysis.inconclusive("analysis_limit"))
        == EquivalenceLevel.UNKNOWN_DIFFERENCE
    )


def test_strip_helper_epilog_drops_trailing_ret():
    from reccmp.compare.inlines import strip_helper_epilog

    body = (("mov", "eax, ecx"), ("add", "eax, 1"), ("imul", "eax, 2"), ("ret", ""))
    assert strip_helper_epilog(body) == body[:-1]


def test_find_inline_expansions_detects_subsequence():
    helper = (
        ("mov", "eax, ecx"),
        ("add", "eax, 1"),
        ("imul", "eax, 2"),
        ("ret", ""),
    )
    host = (
        ("push", "ebp"),
        ("mov", "eax, ecx"),
        ("add", "eax, 1"),
        ("imul", "eax, 2"),
        ("pop", "ebp"),
    )

    def fingerprint_of(addr: int, size: int):
        assert addr == 0x200
        return host

    hits = find_inline_expansions(
        helper,
        [(0x200, "Host", 20)],
        fingerprint_of,
        min_helper_ops=3,
    )
    assert len(hits) == 1
    assert hits[0].host_addr == 0x200
    assert hits[0].match_offset == 1
    assert hits[0].match_length == 3


def test_accuracy_after_inline_elision_call_vs_body():
    from reccmp.compare.inlines import accuracy_after_inline_elision

    helper_body = [
        ("mov", "eax, ecx"),
        ("add", "eax, 1"),
        ("imul", "eax, 2"),
    ]
    orig = [("push", "ebx"), *helper_body, ("pop", "ebx")]
    recomp = [("push", "ebx"), ("call", "Foo::setX"), ("pop", "ebx")]
    placeholder = ("inline", 0x100)
    assert (
        accuracy_after_inline_elision(
            orig,
            recomp,
            orig_elide=[(1, 3, placeholder)],
            recomp_collapse=[(1, placeholder)],
        )
        == 1.0
    )


def test_analyze_inline_layout_call_vs_inline():
    from reccmp.compare.inlines import HelperCatalogEntry, analyze_inline_layout

    helper_lines = [
        "mov eax, ecx",
        "add eax, 1",
        "imul eax, 2",
        "ret",
    ]
    orig = [
        "push ebx",
        "mov eax, ecx",
        "add eax, 1",
        "imul eax, 2",
        "pop ebx",
    ]
    recomp = [
        "push ebx",
        "call Foo::setX",
        "pop ebx",
    ]
    helpers = [
        HelperCatalogEntry(
            orig_addr=0x100,
            recomp_addr=0x200,
            name="Foo::setX",
            fingerprint=(
                ("mov", "eax, ecx"),
                ("add", "eax, 1"),
                ("imul", "eax, 2"),
            ),
            byte_size=16,
        )
    ]
    result = analyze_inline_layout(orig, recomp, helpers)
    assert result.accuracy_modulo_inline == 1.0
    assert len(result.expansions) == 1
    assert result.expansions[0].helper_name == "Foo::setX"
    assert result.expansions[0].side == "orig"
    assert result.expansions[0].counterpart == "call"


def test_enrich_mismatch_side_preserves_kind():
    from reccmp.compare.diagnosis import ComparisonDifference, DifferenceSide
    from reccmp.compare.functions import FunctionComparator
    from unittest.mock import MagicMock

    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=MagicMock(),
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
    )
    comparator.lines_db.find_line_of_recomp_address.return_value = (
        __import__("pathlib").PurePath("foo.cpp"),
        183,
    )
    analysis = ComparisonAnalysis.mismatch(
        ComparisonDifference(
            "call_target",
            DifferenceSide(0, 0x401000, {}),
            DifferenceSide(1, 0x501000, {"target": "bar"}),
        )
    )
    enriched = comparator._enrich_analysis_with_source(analysis)
    assert enriched.difference is not None
    assert enriched.difference.recomp.facts["source_path"] == "foo.cpp"
    assert enriched.difference.recomp.facts["source_line"] == 183
    assert enriched.difference.recomp.facts["target"] == "bar"
