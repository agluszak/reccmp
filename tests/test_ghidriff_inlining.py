"""Inline retries use catalog identity and preserve ordinary comparison evidence."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import BinaryInput, FunctionEntry, Manifest, NamedObject
from reccmp.ghidriff.inlining import (
    InlineCandidates,
    decompiled_lines,
    temporary_inline,
)
from reccmp.ghidriff.report import result_json
from reccmp.ghidriff.results import InlineCode, Outcome, classify, classify_inline
from reccmp.types import EntityType, ImageId


def fixture_model(size=3, entity_type=EntityType.FUNCTION, branching=False):
    obj = NamedObject(0x1100, 0x2100, "helper", entity_type, 1, 1, PairBasis.ANNOTATION)
    entry = FunctionEntry(0x1000, 0x2000, "caller", PairBasis.ANNOTATION, None, False)
    binary = BinaryInput(Path("unused"), "a")
    manifest = Manifest("T", binary, binary, (entry,), (obj,), ())
    functions = {}
    programs = {}
    for image in ImageId:
        fn = NS(
            getBody=lambda: None,
            isExternal=lambda: False,
            isThunk=lambda: False,
            inline=False,
        )
        fn.setInline = lambda value, fn=fn: setattr(fn, "inline", value)
        functions[image] = fn
        fn.saved = []

        def start(_name, fn=fn):
            fn.saved.append(fn.inline)
            return len(fn.saved) - 1

        def end(transaction, commit, fn=fn):
            assert not commit
            fn.inline = fn.saved[transaction]

        programs[image] = NS(
            getAddressFactory=lambda: NS(
                getDefaultAddressSpace=lambda: NS(getAddress=lambda a: a)
            ),
            getFunctionManager=lambda fn=fn: NS(getFunctionAt=lambda _a: fn),
            getListing=lambda: NS(
                getInstructions=lambda _body, _forward: [
                    NS(
                        getFlowType=lambda: NS(
                            isCall=lambda: False, isJump=lambda: branching
                        )
                    )
                    for _ in range(size)
                ]
            ),
            startTransaction=start,
            endTransaction=end,
        )
    candidates = InlineCandidates(manifest, programs)
    candidates.calls = {
        (ImageId.ORIG, 0x1000): [],
        (ImageId.RECOMP, 0x2000): [
            {"identity": "pair:0x1100", "paired": True, "target": "0x2100"}
        ],
        (ImageId.ORIG, 0x1100): [],
        (ImageId.RECOMP, 0x2100): [],
    }
    candidates.tails = {
        (image, address): []
        for image, address in (
            (ImageId.ORIG, 0x1000),
            (ImageId.RECOMP, 0x2000),
            (ImageId.ORIG, 0x1100),
            (ImageId.RECOMP, 0x2100),
        )
    }
    return entry, obj, candidates, functions


def test_small_asymmetric_internal_pair_is_candidate():
    entry, obj, candidates, _ = fixture_model()
    assert candidates.for_pair(entry) == (obj,)


def test_common_callee_is_not_candidate():
    entry, _, candidates, _ = fixture_model()
    candidates.calls[ImageId.ORIG, 0x1000] = candidates.calls[ImageId.RECOMP, 0x2000]
    assert not candidates.for_pair(entry)


@pytest.mark.parametrize("orig_count,recomp_count", [(1, 2), (2, 1), (2, 3)])
def test_partial_inlining_of_repeated_calls_is_candidate(orig_count, recomp_count):
    entry, obj, candidates, _ = fixture_model()
    candidates.calls[ImageId.ORIG, 0x1000] = [
        {"identity": "pair:0x1100", "paired": True, "target": "0x1100"}
    ] * orig_count
    candidates.calls[ImageId.RECOMP, 0x2000] *= recomp_count
    assert candidates.for_pair(entry) == (obj,)


def test_equal_repeated_call_counts_are_not_candidates():
    entry, _, candidates, _ = fixture_model()
    candidates.calls[ImageId.ORIG, 0x1000] = [
        {"identity": "pair:0x1100", "paired": True, "target": "0x1100"}
    ] * 2
    candidates.calls[ImageId.RECOMP, 0x2000] *= 2
    assert not candidates.for_pair(entry)


def test_repeated_branching_callee_stays_as_a_call():
    entry, _, candidates, _ = fixture_model(branching=True)
    candidates.calls[ImageId.RECOMP, 0x2000] *= 2
    assert not candidates.for_pair(entry)


def test_repeated_unreachable_body_does_not_prevent_inline():
    entry, obj, candidates, _ = fixture_model(branching=True)
    candidates.calls[ImageId.RECOMP, 0x3000] = (
        candidates.calls[ImageId.RECOMP, 0x2000] * 2
    )
    assert candidates.for_pair(entry) == (obj,)


@pytest.mark.parametrize("size", [100, 101])
def test_large_callee_is_rejected(size):
    entry, _, candidates, _ = fixture_model(size)
    assert not candidates.for_pair(entry)


def test_99_instructions_is_allowed():
    entry, obj, candidates, _ = fixture_model(99)
    assert candidates.for_pair(entry) == (obj,)


@pytest.mark.parametrize("entity_type", [EntityType.IMPORT, EntityType.IMPORT_THUNK])
def test_imports_are_never_candidates(entity_type):
    entry, _, candidates, _ = fixture_model(entity_type=entity_type)
    # The census does not classify imports as paired internal call identities.
    candidates.calls[ImageId.RECOMP, 0x2000][0]["paired"] = False
    assert not candidates.for_pair(entry)


def test_cycle_through_a_function_left_as_a_call_is_candidate():
    # Only inline-marked functions expand, so the cycle stays one call deep.
    entry, obj, candidates, _ = fixture_model()
    candidates.calls[ImageId.RECOMP, 0x2100] = [
        {"identity": "recomp:0x3000", "paired": False, "target": "0x3000"}
    ]
    candidates.calls[ImageId.RECOMP, 0x3000] = [
        {"identity": "pair:0x1100", "paired": True, "target": "0x2100"}
    ]
    assert candidates.for_pair(entry) == (obj,)


def test_self_recursive_candidate_is_rejected():
    entry, _, candidates, _ = fixture_model()
    candidates.calls[ImageId.ORIG, 0x1100] = [
        {"identity": "pair:0x1100", "paired": True, "target": "0x1100"}
    ]
    assert not candidates.for_pair(entry)


def test_cycle_among_selected_candidates_is_rejected():
    entry, _, _, candidates = nested_model()
    candidates.calls[ImageId.RECOMP, 0x2200] = [
        {"identity": "pair:0x1100", "paired": True, "target": "0x2100"}
    ]
    assert not candidates.for_pair(entry)


def test_incomplete_candidate_body_is_rejected():
    entry, _, candidates, _ = fixture_model()
    candidates.calls[ImageId.ORIG, 0x1100] = None
    assert not candidates.for_pair(entry)


def test_incomplete_body_beyond_the_candidate_is_irrelevant():
    entry, obj, candidates, _ = fixture_model()
    candidates.calls[ImageId.RECOMP, 0x2100] = [
        {"identity": "recomp:0x3000", "paired": False, "target": "0x3000"}
    ]
    candidates.calls[ImageId.RECOMP, 0x3000] = None
    assert candidates.for_pair(entry) == (obj,)


def nested_model():
    """Caller -> helper, which tail-jumps to nested; retail expands both."""
    entry, obj, candidates, _ = fixture_model()
    nested = NamedObject(
        0x1200, 0x2200, "nested", EntityType.FUNCTION, 1, 1, PairBasis.ANNOTATION
    )
    candidates.pairs[0x1200] = nested
    for image in ImageId:
        candidates.identities[image][nested.addr(image)] = nested
    candidates.tails[ImageId.RECOMP, 0x2100] = [
        {"identity": "pair:0x1200", "paired": True, "target": "0x2200"}
    ]
    for image, address in (
        (ImageId.ORIG, 0x1100),
        (ImageId.ORIG, 0x1200),
        (ImageId.RECOMP, 0x2200),
    ):
        candidates.tails[image, address] = []
    candidates.calls[ImageId.ORIG, 0x1200] = []
    candidates.calls[ImageId.RECOMP, 0x2200] = []
    return entry, obj, nested, candidates


def test_tail_call_continuation_of_candidate_is_candidate():
    entry, obj, nested, candidates = nested_model()
    assert candidates.for_pair(entry) == (obj, nested)
    assert candidates.reached(ImageId.RECOMP, 0x2000, (obj, nested)) == {
        0x1100,
        0x1200,
    }
    assert not candidates.reached(ImageId.ORIG, 0x1000, (obj, nested))


def test_entry_tail_call_is_candidate():
    entry, obj, candidates, _ = fixture_model()
    candidates.tails[ImageId.RECOMP, 0x2000] = candidates.calls[ImageId.RECOMP, 0x2000]
    candidates.calls[ImageId.RECOMP, 0x2000] = []
    assert candidates.for_pair(entry) == (obj,)
    assert candidates.reached(ImageId.RECOMP, 0x2000, (obj,)) == {0x1100}


def test_nested_direct_call_of_candidate_is_not_candidate():
    entry, obj, _, candidates = nested_model()
    candidates.calls[ImageId.RECOMP, 0x2100] = candidates.tails[ImageId.RECOMP, 0x2100]
    candidates.tails[ImageId.RECOMP, 0x2100] = []
    assert candidates.for_pair(entry) == (obj,)


def test_selected_nested_direct_calls_are_reached():
    entry, obj, nested, candidates = nested_model()
    candidates.calls[ImageId.RECOMP, 0x2100] = candidates.tails[ImageId.RECOMP, 0x2100]
    candidates.tails[ImageId.RECOMP, 0x2100] = []
    assert candidates.reached(ImageId.RECOMP, entry.recomp_addr, (obj, nested)) == {
        obj.orig_addr,
        nested.orig_addr,
    }


def test_unpaired_or_non_function_identity_is_not_candidate():
    entry, _, candidates, _ = fixture_model()
    candidates.calls[ImageId.RECOMP, 0x2000] = [
        {"identity": "pair:0x9999", "paired": True, "target": "0x2999"}
    ]
    assert not candidates.for_pair(entry)


def test_inline_flags_restore_original_values_after_exception():
    _, obj, candidates, functions = fixture_model()
    functions[ImageId.ORIG].inline = True
    with pytest.raises(ValueError), temporary_inline(candidates.programs, [obj]):
        assert all(fn.inline for fn in functions.values())
        raise ValueError("retry failed")
    assert functions[ImageId.ORIG].inline
    assert not functions[ImageId.RECOMP].inline


def result(entry, inline_code=None):
    normal = classify(
        entry,
        failures=(),
        orig_code=["return a->foo + 1;\n"],
        recomp_code=["return GetFoo(a) + 1;\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    return (
        classify_inline(
            normal, InlineCode(inline_code[0], inline_code[1], (0x1100,)), (), ()
        )
        if inline_code is not None
        else normal
    )


def test_retry_controls_outcome_and_both_diffs_are_reported():
    entry, _, _, _ = fixture_model()
    code = ["return a->foo + 1;\n"]
    normalized = result(entry, (code, code))
    assert normalized.outcome == Outcome.NO_DIFFERENCES
    assert normalized.normal_diff
    assert normalized.inline_normalized_diff == normalized.code_diff == ()
    row = result_json(normalized)
    assert row["normal_diff"] and row["inline_normalized_diff"] == []
    assert row["inline_callees"] == ["0x1100"]


def test_changed_value_survives_retry():
    entry, _, _, _ = fixture_model()
    normalized = result(entry, (["return a->foo + 1;\n"], ["return a->foo + 2;\n"]))
    assert normalized.outcome == Outcome.DIFFERENCES
    assert normalized.code_diff == normalized.inline_normalized_diff


def test_failed_retry_is_analysis_failure_with_normal_diff_retained():
    entry, _, _, _ = fixture_model()
    normalized = result(entry, (["return 1;\n"], None))
    assert normalized.outcome == Outcome.ANALYSIS_FAILED
    assert normalized.normal_diff and normalized.failures


def test_no_retry_is_distinct_from_empty_retry_diff():
    entry, _, _, _ = fixture_model()
    assert result(entry).inline_normalized_diff is None


def test_only_successful_inline_notice_is_removed():
    code = (
        "\n/* WARNING: Inlined function: Foo */\n"
        "/* RVA 1335: Ghidra metadata */\n"
        "\n/* WARNING: Could not inline here */\nreturn 1;\n"
    )
    assert decompiled_lines(code) == [
        "/* WARNING: Could not inline here */\n",
        "return 1;\n",
    ]
