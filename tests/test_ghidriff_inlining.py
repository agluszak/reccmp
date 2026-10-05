"""Inline retries use catalog identity and preserve ordinary comparison evidence."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import BinaryInput, FunctionEntry, Manifest, NamedObject
from reccmp.ghidriff.inlining import (
    Decompiled,
    InlineCandidates,
    InlineNormalizationMixin,
    decompiled_lines,
    temporary_inline,
)
from reccmp.ghidriff.report import result_json
from reccmp.ghidriff.results import (
    DataReference,
    StringValue,
    FunctionResult,
    Outcome,
    classify,
    classify_pass,
)
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
        FunctionResult(
            entry,
            normal.ordinary,
            classify_pass(
                entry,
                failures=(),
                orig_code=inline_code[0],
                recomp_code=inline_code[1],
                orig_refs=(),
                recomp_refs=(),
            ),
            (0x1100,),
        )
        if inline_code is not None
        else normal
    )


def test_retry_controls_outcome_and_both_diffs_are_reported():
    entry, _, _, _ = fixture_model()
    code = ["return a->foo + 1;\n"]
    normalized = result(entry, (code, code))
    assert normalized.outcome == Outcome.NO_DIFFERENCES
    assert normalized.ordinary.text.body_diff
    assert normalized.inline.text.body_diff == normalized.selected.text.body_diff == ()
    row = result_json(normalized)
    assert (
        row["passes"]["ordinary"]["body_diff"]
        and row["passes"]["inline"]["body_diff"] == []
    )
    assert row["inline_callees"] == ["0x1100"]


def test_changed_value_survives_retry():
    entry, _, _, _ = fixture_model()
    normalized = result(entry, (["return a->foo + 1;\n"], ["return a->foo + 2;\n"]))
    assert normalized.outcome == Outcome.DIFFERENCES
    assert normalized.selected.text.body_diff == normalized.inline.text.body_diff


def test_failed_retry_gates_analysis_and_retains_ordinary_evidence():
    entry, _, _, _ = fixture_model()
    normalized = result(entry, (["return 1;\n"], None))
    assert normalized.outcome == Outcome.ANALYSIS_FAILED
    assert normalized.selected_pass == "inline"
    assert normalized.ordinary.text.body_diff and normalized.inline.failures


def test_no_retry_is_distinct_from_empty_retry_diff():
    entry, _, _, _ = fixture_model()
    assert result(entry).inline is None


def test_only_successful_inline_notice_is_removed():
    code = (
        "\n/* WARNING: Inlined function: Foo */\n"
        "/* RVA 1335: Ghidra metadata */\n"
        "                    /* RVA  1335  ?getWorldSpaceMatrix@srNode@@ */\n"
        "\n/* WARNING: Could not inline here */\nreturn 1;\n"
    )
    assert decompiled_lines(code) == [
        "/* WARNING: Could not inline here */\n",
        "return 1;\n",
    ]


def test_export_metadata_is_removed_after_address_normalization():
    code = "                    /* 10054d10  1335  symbol */\nreturn 1;\n"

    class Engine(InlineNormalizationMixin):
        _inline_decompiled = {(ImageId.ORIG, 0x10054D10): Decompiled(code, None)}

        def normalize_ghidra_decomp_for_side(
            self, lines, _is_old, _address, _stack_setup, *, inline
        ):
            assert inline
            lines[:] = [line.replace("10054d10", "RVA") for line in lines]

        def normalized(self):
            return self._normalized(ImageId.ORIG, 0x10054D10, inline=True)

    assert Engine().normalized() == ["return 1;\n"]


def test_retry_does_not_override_a_clean_ordinary_result():
    entry, _, _, _ = fixture_model()
    code = ["return a->foo + 1;\n"]
    normal = classify(
        entry,
        failures=(),
        orig_code=code,
        recomp_code=code,
        orig_refs=(),
        recomp_refs=(),
    )
    retried = FunctionResult(
        entry,
        normal.ordinary,
        classify_pass(
            entry,
            failures=(),
            orig_code=code,
            recomp_code=["return 2;\n"],
            orig_refs=(),
            recomp_refs=(),
        ),
        (0x1100,),
    )
    assert retried.selected == normal.ordinary
    assert retried.selected_pass == "ordinary"


def test_pass_score_and_declaration_follow_retry():
    entry, _, _, _ = fixture_model()
    ordinary = classify(
        entry,
        failures=(),
        orig_code=["uint f()\n", "{\n", "return 1;\n", "}\n"],
        recomp_code=["int f()\n", "{\n", "return helper();\n", "}\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    code = ["int f()\n", "{\n", "return 1;\n", "}\n"]
    retry = classify_pass(
        entry,
        failures=(),
        orig_code=code,
        recomp_code=code,
        orig_refs=(),
        recomp_refs=(),
    )
    row = result_json(FunctionResult(entry, ordinary.ordinary, retry, (0x1100,)))
    assert row["selected_pass"] == "inline"
    assert row["passes"]["ordinary"]["signature_diff"]
    assert row["passes"]["ordinary"]["similarity"] < 1
    assert row["passes"]["inline"]["signature_diff"] == []
    assert row["passes"]["inline"]["similarity"] == 1


def test_inline_transaction_flushes_before_retry_and_after_rollback():
    _, obj, candidates, functions = fixture_model()
    observations = []
    cache = NS(
        flushCache=lambda: observations.append(
            tuple(fn.inline for fn in functions.values())
        )
    )
    with (
        pytest.raises(ValueError),
        temporary_inline(candidates.programs, [obj], [cache]),
    ):
        raise ValueError("decompile failed")
    assert observations == [(True, True), (False, False)]


def test_native_and_embedded_warnings_are_preserved():
    raw = Decompiled.from_native(
        NS(
            completed=True,
            error=None,
            warnings=("native warning",),
            code="/* WARNING: Could not recover jumptable */\nvoid f() { return; }",
        )
    )
    assert raw.warnings == ("native warning", "WARNING: Could not recover jumptable")
    assert "WARNING" in raw.code


def test_retry_keeps_ordinary_and_inline_reference_findings_separate():
    entry, _, _, _ = fixture_model()

    class Engine(InlineNormalizationMixin):
        manifest = NS(functions=(entry,))
        _failures = {}
        _inline_callees = {entry.orig_addr: (0x1100,)}
        _decompiled = {
            (ImageId.ORIG, entry.orig_addr): Decompiled("void f() { return; }", None),
            (ImageId.RECOMP, entry.recomp_addr): Decompiled(
                "void f() { return; }", None
            ),
        }
        _inline_decompiled = dict(_decompiled)
        _references = {
            (ImageId.ORIG, entry.orig_addr): (
                DataReference(0x3000, None, StringValue("old")),
            ),
            (ImageId.RECOMP, entry.orig_addr): (
                DataReference(0x4000, None, StringValue("new")),
            ),
        }
        _inline_references = {
            (image, entry.orig_addr): (
                DataReference(0x5000, None, StringValue("helper")),
            )
            for image in ImageId
        }

        def _entry_addr(self, selected, image):
            return selected.orig_addr if image == ImageId.ORIG else selected.recomp_addr

        def normalize_ghidra_decomp_for_side(self, *_args, **_kwargs):
            pass

    engine = Engine()
    comparison = engine.results()[0]
    assert comparison.inline is not None
    assert comparison.selected.text is not None
    assert comparison.ordinary.outcome == Outcome.DIFFERENCES
    assert comparison.ordinary.data_findings
    assert comparison.inline.outcome == Outcome.NO_DIFFERENCES
    assert not comparison.inline.data_findings
    assert comparison.selected.text.similarity == 1
    assert comparison.ordinary.data_findings[0].orig == (StringValue("old"),)


def test_clean_ordinary_pass_is_not_sent_to_candidate_discovery(monkeypatch):
    entry, _, _, _ = fixture_model()
    calls = []

    def discover(_entry):
        raise AssertionError("Clean ordinary results do not need candidate discovery")

    monkeypatch.setattr(
        "reccmp.ghidriff.inlining.InlineCandidates", lambda *_: NS(for_pair=discover)
    )

    class Engine(InlineNormalizationMixin):
        manifest = None
        _failures = {}
        _decompiled = {
            (ImageId.ORIG, entry.orig_addr): Decompiled("return;", None),
            (ImageId.RECOMP, entry.recomp_addr): Decompiled("return;", None),
        }

        def _entry_addr(self, selected, image):
            return selected.orig_addr if image == ImageId.ORIG else selected.recomp_addr

        def _comparable_entries(self):
            return (entry,)

        def _ordinary_result(self, entry):
            return classify(
                entry,
                failures=(),
                orig_code=["return;"],
                recomp_code=["return;"],
                orig_refs=(),
                recomp_refs=(),
            )

        def setup_decompliers(self, *_args, **_kwargs):
            calls.append("setup")

        def shutdown_decompilers(self, *_args):
            calls.append("shutdown")

    Engine().normalize_inlining(dict.fromkeys(ImageId))
    assert calls == ["setup", "shutdown"]
