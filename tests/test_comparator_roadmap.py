"""Tests for structured IR match keys and stack-normalized scoring."""

import os
import subprocess
import sys
from pathlib import PurePath
from types import SimpleNamespace
from unittest.mock import MagicMock

from reccmp.compare.asm.ir import (
    instruction_match_key,
    rewrite_stack_displacements,
    stack_normalized_key,
)
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DiagnosticNormalization,
    DifferenceSide,
    derive_diagnostic_normalizations,
)
from reccmp.compare.functions import FunctionComparator, _longest_increasing_by_recomp
from reccmp.compare.inlines import (
    HelperCatalogEntry,
    accuracy_after_inline_elision,
    analyze_inline_layout,
    find_inline_expansions,
    strip_helper_epilog,
)
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins
from reccmp.compare.stack_layout import (
    StackPair,
    StackRegisterOffset,
    accuracy_after_stack_map,
    build_slot_bijection,
    extract_stack_offset_from_instruction,
)


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
    rewritten = rewrite_stack_displacements(line, {("ebp", -0x24): ("ebp", -0x18)})
    assert "ebp - 0x18" in rewritten
    assert extract_stack_offset_from_instruction(rewritten) == StackRegisterOffset(
        "ebp", -0x18
    )


def test_derive_diagnostic_normalizations():
    assert not derive_diagnostic_normalizations(ComparisonAnalysis.exact())
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.effective({"register_allocation"})
    ) == (DiagnosticNormalization.REGISTER_ALLOCATION,)
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.effective({"frame_slot_layout"})
    ) == (DiagnosticNormalization.STACK_LAYOUT,)
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.effective({"folded_symbol_alias"})
    ) == (DiagnosticNormalization.FOLDED_SYMBOL_ALIAS,)
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.inconclusive("analysis_limit"),
        accuracy_modulo_stack=1.0,
    ) == (DiagnosticNormalization.STACK_LAYOUT,)
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.inconclusive("analysis_limit"),
        accuracy_modulo_inline=1.0,
    ) == (DiagnosticNormalization.KNOWN_INLINE,)
    assert derive_diagnostic_normalizations(
        ComparisonAnalysis.inconclusive("analysis_limit"),
        accuracy_modulo_stack=1.0,
        accuracy_modulo_inline=1.0,
    ) == (
        DiagnosticNormalization.STACK_LAYOUT,
        DiagnosticNormalization.KNOWN_INLINE,
    )
    assert not derive_diagnostic_normalizations(
        ComparisonAnalysis.inconclusive("analysis_limit")
    )
    # Non-proof tags must not use the word "equivalent" in their values.
    for tag in DiagnosticNormalization:
        assert "equivalent" not in tag.value


def test_strip_helper_epilog_drops_trailing_ret():

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

    def fingerprint_of(addr: int, _size: int):
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
            uniqueness=1.0,
        )
    ]
    result = analyze_inline_layout(orig, recomp, helpers)
    assert result.accuracy_modulo_inline == 1.0
    assert len(result.expansions) == 1
    assert result.expansions[0].helper_name == "Foo::setX"
    assert result.expansions[0].side == "orig"
    assert result.expansions[0].counterpart == "call"


def test_analyze_inline_layout_repeated_calls():

    body = [
        "mov eax, ecx",
        "add eax, 1",
        "imul eax, 2",
    ]
    orig = ["push ebx", *body, "nop", *body, "pop ebx"]
    recomp = [
        "push ebx",
        "call Foo::setX",
        "nop",
        "call Foo::setX",
        "pop ebx",
    ]
    helpers = [
        HelperCatalogEntry(
            orig_addr=0x100,
            recomp_addr=0x200,
            name="Foo::setX",
            fingerprint=tuple(
                (line.partition(" ")[0], line.partition(" ")[2]) for line in body
            ),
            byte_size=16,
            uniqueness=1.0,
        )
    ]
    result = analyze_inline_layout(orig, recomp, helpers)
    assert result.accuracy_modulo_inline == 1.0
    assert len(result.expansions) == 2
    assert {e.counterpart_offset for e in result.expansions} == {1, 3}


def test_enrich_mismatch_side_preserves_kind():

    lines_db = MagicMock()
    lines_db.find_line_of_recomp_address.return_value = (PurePath("foo.cpp"), 183)
    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=lines_db,
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
    )
    analysis = ComparisonAnalysis.mismatch(
        ComparisonDifference(
            "call_target",
            DifferenceSide(0, 0x401000, {}),
            DifferenceSide(1, 0x501000, {"target": "bar"}),
        )
    )
    enriched = (
        comparator._enrich_analysis_with_source(  # pylint: disable=protected-access
            analysis
        )
    )
    assert enriched.difference is not None
    assert enriched.difference.recomp.facts["source_path"] == "foo.cpp"
    assert enriched.difference.recomp.facts["source_line"] == 183
    assert enriched.difference.recomp.facts["target"] == "bar"


def test_enrich_inconclusive_orig_location_uses_recomp_counterpart():
    """An orig address must never be looked up in the recomp PDB."""
    lines_db = MagicMock()
    lines_db.find_line_of_recomp_address.return_value = (PurePath("foo.cpp"), 7)
    comparator = FunctionComparator(
        db=MagicMock(),
        lines_db=lines_db,
        orig_bin=MagicMock(),
        recomp_bin=MagicMock(),
        report=MagicMock(),
        types=MagicMock(),
    )
    enrich = comparator._enrich_analysis_with_source  # pylint: disable=protected-access

    unpaired = enrich(
        ComparisonAnalysis.inconclusive(
            "non_isomorphic_cfg", DifferenceSide(3, 0x401000, {}, "orig")
        )
    )
    assert unpaired.inconclusive_location is not None
    assert "source_path" not in unpaired.inconclusive_location.facts
    lines_db.find_line_of_recomp_address.assert_not_called()

    paired = enrich(
        ComparisonAnalysis.inconclusive(
            "non_isomorphic_cfg",
            DifferenceSide(3, 0x401000, {"recomp_address": 0x501010}, "orig"),
        )
    )
    assert paired.inconclusive_location is not None
    assert paired.inconclusive_location.facts["source_line"] == 7
    lines_db.find_line_of_recomp_address.assert_called_once_with(0x501010)


def test_longest_increasing_source_pins_beats_greedy():
    """Crossing early pin should not discard a longer later chain."""

    # Greedy keeps only the first (recomp=100). LIS keeps the length-3 chain.
    annotations = [
        SimpleNamespace(recomp_addr=100, orig_addr=1, name="a"),
        SimpleNamespace(recomp_addr=10, orig_addr=2, name="b"),
        SimpleNamespace(recomp_addr=20, orig_addr=3, name="c"),
        SimpleNamespace(recomp_addr=30, orig_addr=4, name="d"),
    ]
    kept = _longest_increasing_by_recomp(annotations)  # type: ignore[arg-type]
    assert [a.name for a in kept] == ["b", "c", "d"]


def _permutation_order(hash_seed: str) -> str:
    code = (
        "from reccmp.compare.stack_layout import *\n"
        "pairs = {StackPair(StackRegisterOffset(r, 0x10), StackRegisterOffset(r, 0x10))"
        " for r in ('esp', 'ebp', 'ebx', 'esi')}\n"
        "print([e.orig for e in permutation_entries(pairs)])\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_permutation_entries_order_is_independent_of_hash_seed():
    """Slots sharing an offset on different base registers must be ordered the
    same in every process; string hashing (and so set order) is randomized."""
    assert len({_permutation_order(seed) for seed in ("1", "2", "3", "4")}) == 1
