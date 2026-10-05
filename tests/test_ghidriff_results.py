"""Outcomes and data findings of a code comparison."""

from pathlib import PurePath
import sys
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import FunctionEntry, SourceLocation
from reccmp.ghidriff.results import (
    AnalysisFailure,
    DataFindingKind,
    DataReference,
    FailureKind,
    ObjectOffset,
    Outcome,
    PastEnd,
    PointerValue,
    RawBytes,
    StringValue,
    Uninitialized,
    UnknownExtent,
    classify,
    compare_references,
    consistent,
)
from reccmp.types import ImageId
from reccmp.ghidriff.preparation import decompile_fresh, recover_requested_switches

if TYPE_CHECKING:
    from ghidriff import DecompileResult
    from ghidra.program.model.listing import Function, Program


def _entry(recomp_addr: int | None = 0x2000) -> FunctionEntry:
    return FunctionEntry(
        orig_addr=0x1000,
        recomp_addr=recomp_addr,
        name="Function",
        basis=PairBasis.ANNOTATION if recomp_addr is not None else None,
        source=SourceLocation(PurePath("a.cpp"), 3),
        library=False,
    )


def _ref(contents, obj: ObjectOffset | None = None, address: int = 0) -> DataReference:
    return DataReference(address, obj, contents)


def test_unpaired_function_is_reported_not_dropped():
    result = classify(
        _entry(None),
        failures=(),
        orig_code=None,
        recomp_code=None,
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.UNPAIRED


def test_missing_decompilation_is_an_analysis_failure():
    result = classify(
        _entry(),
        failures=(),
        orig_code=["int f(void)\n"],
        recomp_code=None,
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.ANALYSIS_FAILED
    assert result.failures == (
        AnalysisFailure(FailureKind.NOT_DECOMPILED, ImageId.RECOMP),
    )


def test_switch_decompilation_failure_is_returned_per_function(monkeypatch):
    visited: list[tuple[str, int]] = []
    lifecycle = []
    result = NS(
        decompileCompleted=lambda: False,
        getErrorMessage=lambda: "process timeout",
    )

    def decompile(function, timeout, _monitor):
        visited.append((function.getEntryPoint(), timeout))
        return result

    decompiler = NS(
        openProgram=lambda _program: True,
        decompileFunction=decompile,
        resetDecompiler=lambda: lifecycle.append("reset"),
        dispose=lambda: lifecycle.append("dispose"),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.app.decompiler",
        NS(DecompInterface=lambda: decompiler),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.app.plugin.core.analysis",
        NS(
            SwitchAnalysisDecompileConfigurer=lambda _program: NS(
                configure=lambda _d: None
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.app.cmd.function",
        NS(DecompilerSwitchAnalysisCmd=lambda _result: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.util.task",
        NS(TaskMonitor=NS(DUMMY=object())),
    )
    program = NS(
        getFunctionManager=lambda: NS(
            getFunctionAt=lambda address: NS(getEntryPoint=lambda: f"{address:08x}")
        ),
        getAddressFactory=lambda: NS(
            getDefaultAddressSpace=lambda: NS(getAddress=lambda address: address)
        ),
    )

    assert recover_requested_switches(
        cast("Program", program), [0x1000, 0x2000], 90
    ) == {
        0x1000: "Switch analysis failed at 00001000: process timeout",
        0x2000: "Switch analysis failed at 00002000: process timeout",
    }
    assert visited == [("00001000", 90), ("00002000", 90)]
    assert lifecycle == ["reset", "reset", "dispose"]


def test_fresh_decompilation_uses_program_options_timeout_and_disposes(monkeypatch):
    calls: list[tuple[object, ...]] = []
    native_result = object()
    converted = cast("DecompileResult", NS(code="int f(void) {}"))
    program = cast("Program", object())
    function = cast("Function", object())
    options = NS(
        grabFromProgram=lambda value: calls.append(("program", value)),
        setMaxPayloadMBytes=lambda value: calls.append(("payload", value)),
    )

    def decompile(func, timeout, _monitor):
        calls.append(("decompile", func, timeout))
        return native_result

    def convert(value):
        assert value is native_result
        return converted

    decompiler = NS(
        setOptions=lambda value: calls.append(("options", value)),
        openProgram=lambda _program: True,
        decompileFunction=decompile,
        dispose=lambda: calls.append(("dispose",)),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.app.decompiler",
        NS(DecompInterface=lambda: decompiler, DecompileOptions=lambda: options),
    )
    monkeypatch.setitem(
        sys.modules, "ghidra.util.task", NS(TaskMonitor=NS(DUMMY=object()))
    )
    result = decompile_fresh(program, function, 90, convert)
    assert result is converted
    assert calls == [
        ("program", program),
        ("payload", 100),
        ("options", options),
        ("decompile", function, 90),
        ("dispose",),
    ]


def test_entry_conflict_is_kept_as_the_reason():
    conflict = AnalysisFailure(
        FailureKind.ENTRY_CONFLICT, ImageId.ORIG, other_function=0xF00
    )
    result = classify(
        _entry(),
        failures=(conflict,),
        orig_code=None,
        recomp_code=["int f(void)\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.ANALYSIS_FAILED
    assert result.failures == (conflict,)


def test_one_sided_failure_is_the_whole_reason():
    conflict = AnalysisFailure(FailureKind.ENTRY_CONFLICT, ImageId.ORIG, 0xF00)
    result = classify(
        _entry(),
        failures=(conflict,),
        orig_code=None,
        recomp_code=None,
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.failures == (conflict,)


def test_equal_decompilation_has_no_differences():
    code = ["int f(void)\n", "{\n", "  return 1;\n", "}\n"]
    result = classify(
        _entry(),
        failures=(),
        orig_code=code,
        recomp_code=list(code),
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.NO_DIFFERENCES
    assert not result.code_diff


def test_code_difference_carries_the_diff():
    result = classify(
        _entry(),
        failures=(),
        orig_code=["  return a <= b;\n"],
        recomp_code=["  return a > b;\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.DIFFERENCES
    assert "-  return a <= b;\n" in result.code_diff
    assert "+  return a > b;\n" in result.code_diff


def test_equal_code_with_different_literal_is_a_difference():
    """The decompiler may print both literals under equal labels."""
    code = ["  wcscpy(out, &DAT_0);\n"]
    result = classify(
        _entry(),
        failures=(),
        orig_code=code,
        recomp_code=list(code),
        orig_refs=(_ref(StringValue("?")),),
        recomp_refs=(_ref(StringValue("")),),
    )
    assert result.outcome == Outcome.DIFFERENCES
    [finding] = result.data_findings
    assert finding.kind == DataFindingKind.REFERENCED_CONTENTS
    assert finding.orig == (StringValue("?"),)
    assert finding.recomp == (StringValue(""),)


def test_paired_object_contents_are_compared_by_identity():
    obj = ObjectOffset(0x5000, "g_table", 0)
    findings = compare_references(
        (_ref(RawBytes(b"\x01\x02", False, extent_known=True), obj),),
        (_ref(RawBytes(b"\x01\x03", False, extent_known=True), obj),),
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == DataFindingKind.OBJECT_CONTENTS
    assert finding.object == obj


def test_same_contents_through_different_objects_is_not_a_data_difference():
    """A reference to another object is visible in the code by its name."""
    findings = compare_references(
        (_ref(StringValue("%d"), ObjectOffset(0x5000, "g_format_d", 0)),),
        (_ref(StringValue("%d")),),
    )
    assert not findings


def test_different_paired_objects_are_not_compared_by_contents():
    """Different table references already have distinct identities in code."""
    findings = compare_references(
        (_ref(RawBytes(b"\x01", False, True), ObjectOffset(0x5000, "g_a", 0)),),
        (_ref(RawBytes(b"\x02", False, True), ObjectOffset(0x6000, "g_b", 0)),),
    )
    assert not findings


def test_an_array_end_is_not_the_object_the_linker_placed_after_it():
    """A loop bound names the end of its array on both sides, whatever
    follows the array in either binary."""
    end = ObjectOffset(0x5000, "g_buttons", 0x24)
    findings = compare_references(
        (_ref(PastEnd(), end, address=0x69B8D4),),
        (_ref(PastEnd(), end, address=0x695134),),
    )
    assert not findings


def test_unknown_contents_are_not_compared():
    findings = compare_references(
        (
            _ref(UnknownExtent()),
            _ref(RawBytes(b"\x00\x10\x40\x00", True, extent_known=True)),
            _ref(PointerValue(None)),
        ),
        (),
    )
    assert not findings


def test_string_and_raw_rendering_of_the_same_bytes_are_consistent():
    assert consistent(
        StringValue("%d"), RawBytes(b"%\x00d\x00\x00\x00", False, extent_known=True)
    )
    assert consistent(StringValue(""), RawBytes(b"\x00", False, extent_known=False))
    assert not consistent(StringValue("?"), RawBytes(b"\x00\x00", False, True))


def test_raw_bytes_without_catalog_extent_compare_as_prefixes():
    short = RawBytes(b"\x00", False, extent_known=False)
    long = RawBytes(b"\x00\x00\x00\x00", False, extent_known=True)
    assert consistent(short, long)
    assert not consistent(
        RawBytes(b"\x01", False, extent_known=True),
        RawBytes(b"\x00\x00", False, extent_known=True),
    )


def test_contents_on_one_side_only_is_not_a_contents_difference():
    """The code shows a missing reference; unknown contents say nothing."""
    findings = compare_references(
        (_ref(RawBytes(b"\x1b\x00\x00\x00", False, extent_known=True)),),
        (_ref(UnknownExtent()),),
    )
    assert not findings


def test_zero_filled_regions_are_consistent_whatever_their_extent():
    assert consistent(
        RawBytes(b"\x00\x00", False, extent_known=True),
        RawBytes(b"\x00" * 23, False, extent_known=True),
    )
    assert consistent(Uninitialized(), RawBytes(b"\x00" * 4, False, True))
    assert not consistent(
        RawBytes(b"\x00\x00", False, extent_known=True),
        RawBytes(b"\x00\x01", False, extent_known=True),
    )


def test_source_path_strings_are_consistent_build_context():
    assert consistent(
        StringValue(r"C:\Projects\SGP\DirectDraw Calls.c"),
        StringValue(r"Z:\repo\src\sgp\DirectDraw Calls.c"),
    )
    assert not consistent(StringValue(r"C:\a.c"), StringValue("a.c"))


def test_signature_only_changes_keep_evidence_without_body_regression():
    result = classify(
        _entry(),
        failures=(),
        orig_code=["void constructor(uint param0)\n", "{\n", "  use(param0);\n", "}\n"],
        recomp_code=[
            "void constructor(int param0)\n",
            "{\n",
            "  use(param0);\n",
            "}\n",
        ],
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.NO_DIFFERENCES
    assert not result.code_diff
    assert result.signature_diff
    assert result.code_change_kind is None


def test_signature_only_findings_survive_serialization_and_cli(capsys):
    from reccmp.ghidriff.report import print_summary, result_json

    result = classify(
        _entry(),
        failures=(),
        orig_code=["void f(uint x)\n", "{\n", "  use(x);\n", "}\n"],
        recomp_code=["void f(int x)\n", "{\n", "  use(x);\n", "}\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    row = result_json(result)
    assert row["outcome"] == "no-differences"
    assert row["signature_diff"] == list(result.signature_diff)
    assert row["code_diff"] == []
    print_summary([result], details=True)
    output = capsys.readouterr().out
    assert "unchanged bodies/data" in output
    assert "-void f(uint x)" in output
    assert "+void f(int x)" in output


def test_unsigned_condition_comparison_remains_a_difference():
    result = classify(
        _entry(),
        failures=(),
        orig_code=["int f(void)\n", "{\n", "  return *(uint *)p < 18;\n", "}\n"],
        recomp_code=["int f(void)\n", "{\n", "  return *(int *)p < 18;\n", "}\n"],
        orig_refs=(),
        recomp_refs=(),
    )
    assert result.outcome == Outcome.DIFFERENCES
    assert result.code_change_kind == "scalar-signedness"
    assert result.code_diff
    assert not result.signature_diff


def test_signature_only_change_does_not_hide_referenced_data():
    result = classify(
        _entry(),
        failures=(),
        orig_code=["void f(uint x)\n", "{\n", "  use(x);\n", "}\n"],
        recomp_code=["void f(int x)\n", "{\n", "  use(x);\n", "}\n"],
        orig_refs=(_ref(StringValue("old")),),
        recomp_refs=(_ref(StringValue("new")),),
    )
    assert result.outcome == Outcome.DIFFERENCES
    assert not result.code_diff
    assert result.data_findings
    assert result.signature_diff
