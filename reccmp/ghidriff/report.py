"""Serialized and printed forms of a comparison run."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from reccmp.compare.manifest import FunctionEntry, Manifest
from reccmp.utils import format_address
from .results import (
    AnalysisFailure,
    Contents,
    DataFinding,
    FunctionResult,
    ObjectOffset,
    Outcome,
    PastEnd,
    PointerValue,
    RawBytes,
    StringValue,
    Uninitialized,
    UnknownExtent,
)


@dataclass(frozen=True)
class RunInputs:
    """Everything that decides a run's output besides the source tree."""

    manifest_sha256: str
    reccmp_version: str
    ghidra_version: str
    ghidriff_version: str
    ghidra_project: str


def _address(addr: int | None) -> str | None:
    return format_address(addr) if addr is not None else None


def _object_json(obj: ObjectOffset) -> dict[str, Any]:
    return {
        "orig": format_address(obj.orig_addr),
        "name": obj.name,
        "offset": obj.offset,
    }


def contents_json(contents: Contents) -> dict[str, Any]:
    # pylint: disable=too-many-return-statements
    match contents:
        case StringValue(text):
            return {"string": text}
        case PointerValue(target=ObjectOffset() as obj):
            return {"pointer": _object_json(obj)}
        case PointerValue(target=StringValue(text)):
            return {"pointer": {"string": text}}
        case PointerValue(target=None):
            return {"pointer": None}
        case RawBytes(data, relocated):
            return {"bytes": data.hex(" "), "relocated": relocated}
        case Uninitialized():
            return {"uninitialized": True}
        case UnknownExtent():
            return {"unknown_extent": True}
        case PastEnd():
            return {"past_end": True}
    raise TypeError(contents)


def contents_text(contents: Contents) -> str:
    # pylint: disable=too-many-return-statements
    match contents:
        case StringValue(text):
            return repr(text)
        case PointerValue(target=ObjectOffset(name=name, offset=0)):
            return f"&{name}"
        case PointerValue(target=ObjectOffset(name=name, offset=offset)):
            return f"&{name}+{offset:#x}"
        case PointerValue(target=StringValue(text)):
            return f"&{text!r}"
        case PointerValue(target=None):
            return "pointer to unidentified location"
        case RawBytes(data, relocated):
            return data.hex(" ") + (" (holds an address)" if relocated else "")
        case Uninitialized():
            return "uninitialized"
        case UnknownExtent():
            return "extent unknown"
        case PastEnd():
            return "end of the object"
    raise TypeError(contents)


def _finding_json(finding: DataFinding) -> dict[str, Any]:
    return {
        "kind": finding.kind.value,
        "object": _object_json(finding.object) if finding.object else None,
        "orig": [contents_json(c) for c in finding.orig],
        "recomp": [contents_json(c) for c in finding.recomp],
    }


def _failure_json(failure: AnalysisFailure) -> dict[str, Any]:
    return {
        "kind": failure.kind.value,
        "image": failure.image.name.lower(),
        "other_function": _address(failure.other_function),
        "message": failure.message,
    }


def _entry_json(entry: FunctionEntry) -> dict[str, Any]:
    return {
        "orig": format_address(entry.orig_addr),
        "recomp": _address(entry.recomp_addr),
        "name": entry.name,
        "basis": entry.basis.value if entry.basis is not None else None,
        "source": (
            {"path": str(entry.source.path), "line": entry.source.line}
            if entry.source is not None
            else None
        ),
        "library": entry.library,
    }


def result_json(result: FunctionResult) -> dict[str, Any]:
    return {
        **_entry_json(result.entry),
        "outcome": result.outcome.value,
        "code_diff": list(result.code_diff),
        "normal_diff": list(result.normal_diff),
        "signature_diff": list(result.signature_diff),
        "code_change_kind": result.code_change_kind,
        "inline_normalized_diff": (
            list(result.inline_normalized_diff)
            if result.inline_normalized_diff is not None
            else None
        ),
        "inline_callees": [_address(addr) for addr in result.inline_callees],
        "data": [_finding_json(f) for f in result.data_findings],
        "failures": [_failure_json(f) for f in result.failures],
        "unidentified_references": result.unidentified_references,
    }


def outcome_counts(results: list[FunctionResult]) -> dict[str, int]:
    counts = Counter(result.outcome for result in results)
    return {outcome.value: counts[outcome] for outcome in Outcome}


def summary_json(
    manifest: Manifest, inputs: RunInputs, results: list[FunctionResult]
) -> dict[str, Any]:
    return {
        "target": manifest.target_id,
        "inputs": {
            "orig": {"path": str(manifest.orig.path), "sha256": manifest.orig.sha256},
            "recomp": {
                "path": str(manifest.recomp.path),
                "sha256": manifest.recomp.sha256,
            },
            "manifest_sha256": inputs.manifest_sha256,
            "reccmp": inputs.reccmp_version,
            "ghidra": inputs.ghidra_version,
            "ghidriff": inputs.ghidriff_version,
            "ghidra_project": inputs.ghidra_project,
        },
        "requested": len(results),
        "counts": outcome_counts(results),
        "functions": [result_json(result) for result in results],
    }


def comparison_changes(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[int, str]:
    """Changes at original-address identities; absent selections are not fixes."""
    if current["target"] != previous["target"]:
        raise ValueError("comparison reports have different targets")
    before = {int(row["orig"], 16): row for row in previous["functions"]}
    after = {int(row["orig"], 16): row for row in current["functions"]}
    changes: dict[int, str] = {}
    for address in sorted(before.keys() | after.keys()):
        old, new = before.get(address), after.get(address)
        if old is None:
            changes[address] = "only-current"
        elif new is None:
            changes[address] = "only-previous"
        elif any(
            old.get(key) != new.get(key)
            for key in (
                "outcome",
                "code_diff",
                "signature_diff",
                "data",
                "failures",
                "unidentified_references",
                "name",
                "basis",
            )
        ):
            if new["outcome"] == "no-differences" and old["outcome"] == "differences":
                changes[address] = "resolved"
            elif new["outcome"] == "differences" and old["outcome"] != "differences":
                changes[address] = "newly-different"
            elif new["outcome"] in ("analysis-failed", "unpaired"):
                changes[address] = new["outcome"]
            else:
                changes[address] = "changed"
    return changes


def _location(entry: FunctionEntry) -> str:
    if entry.source is None:
        return ""
    return f"  {entry.source.path}:{entry.source.line}"


def _heading(result: FunctionResult) -> str:
    entry = result.entry
    return (
        f"{format_address(entry.orig_addr)} / {_address(entry.recomp_addr) or '-'}"
        f"  {entry.name}{_location(entry)}"
    )


def print_result(result: FunctionResult, *, details: bool) -> None:
    print(_heading(result))
    for failure in result.failures:
        where = failure.image.name.lower()
        match failure.other_function:
            case int() as other:
                print(
                    f"    {where}: {failure.kind.value} (inside {format_address(other)})"
                )
            case None:
                print(
                    f"    {where}: {failure.kind.value} {failure.message or ''}".rstrip()
                )
    if not details:
        return
    for finding in result.data_findings:
        subject = finding.object.name if finding.object else "other referenced data"
        print(f"    data ({finding.kind.value}): {subject}")
        for contents in finding.orig:
            print(f"      - {contents_text(contents)}")
        for contents in finding.recomp:
            print(f"      + {contents_text(contents)}")
    for line in result.code_diff:
        print("    " + line.rstrip("\n"))
    if result.signature_diff:
        print("    Inferred declaration difference:")
        for line in result.signature_diff:
            print("    " + line.rstrip("\n"))


def print_summary(results: list[FunctionResult], *, details: bool) -> None:
    by_outcome: dict[Outcome, list[FunctionResult]] = {o: [] for o in Outcome}
    for result in results:
        by_outcome[result.outcome].append(result)

    for outcome, title in (
        (Outcome.ANALYSIS_FAILED, "Analysis failed or incomplete"),
        (Outcome.DIFFERENCES, "Differences found"),
    ):
        if by_outcome[outcome]:
            print(f"{title}:")
            for result in by_outcome[outcome]:
                print_result(result, details=details)
            print()
    if details and by_outcome[Outcome.UNPAIRED]:
        print("Unpaired:")
        for result in by_outcome[Outcome.UNPAIRED]:
            print(_heading(result))
        print()

    declarations_only = [
        result for result in by_outcome[Outcome.NO_DIFFERENCES] if result.signature_diff
    ]
    if details and declarations_only:
        print("Inferred declaration differences with unchanged bodies/data:")
        for result in declarations_only:
            print_result(result, details=True)
        print()

    counts = outcome_counts(results)
    print(f"Requested functions: {len(results)}")
    print(f"  differences found:       {counts[Outcome.DIFFERENCES.value]}")
    print(f"  no differences found:    {counts[Outcome.NO_DIFFERENCES.value]}")
    print(f"  unpaired:                {counts[Outcome.UNPAIRED.value]}")
    print(f"  analysis failed:         {counts[Outcome.ANALYSIS_FAILED.value]}")
    print(f"  declaration differences: {sum(bool(r.signature_diff) for r in results)}")
