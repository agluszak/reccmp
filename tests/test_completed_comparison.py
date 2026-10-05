"""Completed comparisons are reused only under identical analysis inputs."""

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any


import ghidriff
import pytest

import reccmp
from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import BinaryInput, FunctionEntry, Manifest
from reccmp.ghidriff import engine
from reccmp.ghidriff.report import comparison_changes
from reccmp.ghidriff.results import classify
from reccmp.tools import compare
from reccmp.tools.compare import _run_engine


@pytest.fixture(name="prepared_comparison")
def fixture_prepared_comparison(tmp_path, monkeypatch):
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    module = implementation / "__init__.py"
    module.write_text("# initial implementation")
    monkeypatch.setattr(reccmp, "__file__", str(module))
    monkeypatch.setattr(ghidriff, "__file__", str(module))
    state: dict[str, Any] = {
        "version": "12.1.4",
        "native": "release-decompiler",
        "reviewed": {},
        "runs": 0,
        "failed": False,
    }

    monkeypatch.setattr(
        "reccmp.ghidriff.inputs.native_decompiler_digest",
        lambda _install_dir: state["native"],
    )

    class Engine:
        focused_switch_analysis = False
        launcher = SimpleNamespace(install_dir=tmp_path)

        def __init__(self, *_args, **_kwargs):
            pass

        def get_ghidra_version(self):
            return state["version"]

        def dump_pdiff_to_path(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(engine, "ReccmpDiffEngine", Engine)
    monkeypatch.setattr(
        compare, "_reviewed_signatures", lambda *_: ({}, state["reviewed"])
    )

    def analyze(_engine, _args, **_options):
        state["runs"] += 1
        result = classify(
            manifest.functions[0],
            failures=(),
            orig_code=["return 1;\n"],
            recomp_code=None if state["failed"] else ["return 2;\n"],
            orig_refs=(),
            recomp_refs=(),
        )
        return {"functions": {"modified": []}}, [result], {"functions": []}, []

    monkeypatch.setattr(compare, "_compare_programs", analyze)
    args = SimpleNamespace(
        output=tmp_path / "report",
        ghidra_projects=tmp_path / "projects",
        no_cache=False,
        threaded=True,
        max_ram_percent=60,
        decompiler_timeout=60,
        side_by_side=False,
        details=False,
        orig_address=[],
    )
    args.output.mkdir()
    manifest = Manifest(
        "TEST",
        BinaryInput(tmp_path / "orig.exe", "a" * 64),
        BinaryInput(tmp_path / "recomp.exe", "b" * 64),
        (FunctionEntry(0x1000, 0x2000, "f", PairBasis.ANNOTATION, None, False),),
        (),
        (),
    )
    return args, manifest, state, module


def test_completed_comparison_reuses_results_and_regenerates_report(
    prepared_comparison,
):
    args, manifest, state, _ = prepared_comparison
    _run_engine(args, manifest)
    first = (args.output / "summary.json").read_text()
    (args.output / "summary.json").unlink()
    _run_engine(args, manifest)
    assert state["runs"] == 1
    assert '"reused": false' in first
    assert '"reused": true' in (args.output / "summary.json").read_text()


@pytest.mark.parametrize(
    "change",
    [
        "binary",
        "identity",
        "timeout",
        "reviewed",
        "ghidra",
        "native",
        "implementation",
        "no-cache",
    ],
)
def test_changed_comparison_inputs_invalidate_completed_result(
    prepared_comparison, change
):
    args, manifest, state, module = prepared_comparison
    _run_engine(args, manifest)
    initial = json.loads((args.output / "summary.json").read_text())["inputs"]
    if change == "binary":
        manifest = replace(manifest, recomp=replace(manifest.recomp, sha256="c" * 64))
    elif change == "identity":
        manifest = replace(
            manifest, functions=(replace(manifest.functions[0], name="another"),)
        )
    elif change == "timeout":
        args.decompiler_timeout = 120
    elif change == "reviewed":
        state["reviewed"] = {0x1000: "unsigned long"}
    elif change == "ghidra":
        state["version"] = "12.2"
    elif change == "native":
        state["native"] = "patched-decompiler"
    elif change == "implementation":
        module.write_text("# changed implementation")
    elif change == "no-cache":
        args.no_cache = True
    _run_engine(args, manifest)
    assert state["runs"] == 2
    if change == "native":
        changed = json.loads((args.output / "summary.json").read_text())["inputs"]
        assert changed["analysis_key"] != initial["analysis_key"]
        assert changed["comparison_key"] != initial["comparison_key"]


def test_analysis_failure_is_retried(prepared_comparison):
    args, manifest, state, _ = prepared_comparison
    state["failed"] = True
    _run_engine(args, manifest)
    state["failed"] = False
    _run_engine(args, manifest)
    assert state["runs"] == 2


def test_report_delta_distinguishes_resolved_and_unselected_functions():
    def row(address, outcome, diff=()):
        return {
            "orig": hex(address),
            "outcome": outcome,
            "selected_pass": "ordinary",
            "passes": {"ordinary": {"body_diff": list(diff), "outcome": outcome}},
            "name": "f",
            "basis": "annotation",
        }

    before = {
        "target": "TEST",
        "functions": [
            row(1, "differences"),
            row(2, "differences"),
            row(3, "no-differences"),
            row(4, "differences"),
        ],
    }
    after = {
        "target": "TEST",
        "functions": [
            row(1, "no-differences"),
            row(3, "differences"),
            row(4, "differences", ["+ different"]),
            row(5, "unpaired"),
        ],
    }
    assert comparison_changes(after, before) == {
        1: "resolved",
        2: "only-previous",
        3: "newly-different",
        4: "changed",
        5: "only-current",
    }
    with pytest.raises(ValueError, match="different targets"):
        comparison_changes(dict(after, target="OTHER"), before)


def test_report_records_exact_cache_keys_and_configuration(prepared_comparison):
    args, manifest, state, _ = prepared_comparison
    _run_engine(args, manifest)
    first = json.loads((args.output / "summary.json").read_text())
    inputs = first["inputs"]
    assert inputs["manifest_sha256"] == manifest.digest()
    assert inputs["analysis_key"] in inputs["ghidra_project"]
    assert manifest.preparation_digest() in inputs["preparation_key"]
    assert inputs["normalization_key"]
    assert inputs["selection_sha256"]
    assert inputs["decompiler_timeout"] == args.decompiler_timeout
    _run_engine(args, manifest)
    second = json.loads((args.output / "summary.json").read_text())
    assert first["inputs"] == second["inputs"]
    assert state["runs"] == 1
