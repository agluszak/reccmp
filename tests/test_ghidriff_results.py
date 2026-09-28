"""Outcomes and data findings of a code comparison."""

from pathlib import PurePath

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import FunctionEntry, SourceLocation
from reccmp.ghidriff.results import (
    AnalysisFailure,
    DataFindingKind,
    DataReference,
    FailureKind,
    ObjectOffset,
    Outcome,
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
