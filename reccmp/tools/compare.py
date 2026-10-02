#!/usr/bin/env python3
"""reccmp-reccmp: decompile and diff reconstructed functions with Ghidriff."""

import argparse
import contextlib
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import colorama

import reccmp
from reccmp.analysis_cache import AnalysisCache, fingerprint_files
from reccmp.compare import Compare
from reccmp.compare.db import ReccmpEntity
from reccmp.compare.manifest import Manifest, build_manifest
from reccmp.project.detect import (
    RecCmpProjectException,
    RecCmpTarget,
    argparse_add_project_target_args,
    argparse_parse_project_target,
)
from reccmp.project.logging import argparse_add_logging_args, argparse_parse_logging
from reccmp.source.index import SourceIndexError
from reccmp.types import EntityType, ImageId

logger = logging.getLogger(__name__)
colorama.just_fix_windows_console()


def parse_args() -> argparse.Namespace:
    def virtual_address(value: str) -> int:
        return int(value, 16)

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Recompilation Compare: decompile each reconstructed function and "
            "its original with Ghidra and report the differences."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    argparse_add_project_target_args(parser)
    parser.add_argument(
        "--orig-address",
        metavar="<offset>",
        type=virtual_address,
        action="append",
        default=[],
        help="Compare only the function at this original address (repeatable)",
    )
    parser.add_argument(
        "--filter",
        metavar="<substring>",
        help="Compare only functions whose name contains this substring",
    )
    parser.add_argument(
        "--nolib", action="store_true", help="Do not compare library functions"
    )
    parser.add_argument(
        "--output",
        metavar="<dir>",
        type=Path,
        help="Report directory (default: reccmp-<target> in the working directory)",
    )
    parser.add_argument(
        "--ghidra-projects",
        metavar="<dir>",
        type=Path,
        help=(
            "Where analyzed Ghidra projects are kept between runs "
            "(default: .reccmp-cache/ghidra next to the recompiled PDB)"
        ),
    )
    parser.add_argument(
        "--orig-ghidra-project",
        type=Path,
        help="Existing reviewed Ghidra project (.gpr) for original signatures",
    )
    parser.add_argument(
        "--orig-ghidra-program",
        help="Program path in --orig-ghidra-project",
    )
    parser.add_argument(
        "--sxs",
        dest="side_by_side",
        action="store_true",
        help="Also write Ghidriff's side-by-side HTML diffs",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print code diffs and data findings (default with --orig-address)",
    )
    parser.add_argument(
        "--decompiler-timeout",
        type=int,
        default=60,
        help="Decompiler timeout in seconds per function",
    )
    parser.add_argument(
        "--max-ram-percent",
        type=float,
        default=60.0,
        help="JVM maximum heap as a percentage of host RAM",
    )
    parser.add_argument(
        "--no-threaded",
        dest="threaded",
        action="store_false",
        help="Analyze and decompile on one thread",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not reuse cached catalog, prepared programs or completed comparisons",
    )
    argparse_add_logging_args(parser)
    args = parser.parse_args()
    argparse_parse_logging(args)
    return args


def _selection(args: argparse.Namespace, target: RecCmpTarget):
    wanted = frozenset(args.orig_address)
    name_filter = args.filter.lower() if args.filter else None
    ignored = frozenset(target.report_config.ignore_functions)

    def select(entity: ReccmpEntity) -> bool:
        name = entity.best_name() or ""
        if wanted and entity.orig_addr not in wanted:
            return False
        if name_filter is not None and name_filter not in name.lower():
            return False
        if args.nolib and entity.get("library", False):
            return False
        return name not in ignored

    return select


@contextlib.contextmanager
def _stage(name: str):
    """Report how long a pipeline stage took; the whole run is minutes long."""
    start = time.monotonic()
    yield
    print(
        f"[STAGE] {name}: {time.monotonic() - start:.0f}s", file=sys.stderr, flush=True
    )


def _reviewed_signatures(
    args: argparse.Namespace, manifest: Manifest
) -> tuple[dict[int, int], dict[int, str]]:
    """Read independently reviewed retail facts that agree with PDB decoration."""
    if not args.orig_ghidra_project and not args.orig_ghidra_program:
        return {}, {}
    if not args.orig_ghidra_project or not args.orig_ghidra_program:
        raise ValueError("both reviewed Ghidra project and program are required")
    # Ghidra imports require the engine-started JVM.
    # pylint: disable=import-outside-toplevel
    import pyghidra
    from ghidra.app.util.demangler import DemangledFunction, DemanglerUtil
    from reccmp.ghidra.signature_provenance import (
        independently_reviewed_return,
        independently_reviewed_signature,
    )

    project_file = args.orig_ghidra_project
    if not project_file.is_file():
        raise FileNotFoundError(project_file)
    project = pyghidra.open_project(
        project_file.parent, project_file.stem, create=False
    )
    try:
        with pyghidra.program_context(project, args.orig_ghidra_program) as program:
            original_md5 = hashlib.md5(manifest.orig.path.read_bytes()).hexdigest()
            if program.getExecutableMD5().lower() != original_md5:
                raise ValueError("reviewed Ghidra program is not the original binary")
            functions = program.getFunctionManager()
            space = program.getAddressFactory().getDefaultAddressSpace()
            signatures: dict[int, int] = {}
            scalar_returns: dict[int, str] = {}
            scalar_spellings = {
                "bool": ("bool",),
                "undefined1": ("bool", "char", "signed char", "unsigned char"),
                "undefined4": ("int", "unsigned int"),
                "uint": ("unsigned int",),
            }
            for obj in manifest.objects:
                if obj.entity_type != EntityType.FUNCTION or not obj.recomp_symbol:
                    continue
                function = functions.getFunctionAt(space.getAddress(obj.orig_addr))
                if function is None or not independently_reviewed_return(
                    program, function
                ):
                    continue
                demangled = DemanglerUtil.demangle(obj.recomp_symbol)
                if not isinstance(demangled, DemangledFunction):
                    continue
                return_name = str(function.getReturnType().getName())
                if str(demangled.getReturnType()) in scalar_spellings.get(
                    return_name, ()
                ):
                    scalar_returns[obj.orig_addr] = return_name
                if not independently_reviewed_signature(program, function):
                    continue
                if function.getCallingConventionName() != "__cdecl":
                    continue
                if demangled.getCallingConvention() != "__cdecl":
                    continue
                parameters = list(demangled.getParameters())
                if any(parameter.getType().isVarArgs() for parameter in parameters):
                    continue
                count = (
                    0
                    if len(parameters) == 1 and str(parameters[0]) == "void"
                    else len(parameters)
                )
                if function.getParameterCount() == count:
                    signatures[obj.orig_addr] = count
            return signatures, scalar_returns
    finally:
        project.close()


def _run_engine(args: argparse.Namespace, target: RecCmpTarget, manifest: Manifest):
    # pylint: disable=import-outside-toplevel
    # Importing the engine does not start the JVM, but it does need ghidriff.
    import ghidriff
    from reccmp.ghidriff.engine import (
        ANALYSIS_REVISION,
        PREPARATION_REVISION,
        ReccmpDiffEngine,
    )
    from reccmp.ghidriff.report import RunInputs, print_summary, summary_json
    from reccmp.ghidriff.results import Outcome

    output: Path = args.output
    projects = args.ghidra_projects or (
        target.recompiled_pdb.parent / ".reccmp-cache" / "ghidra"
    )
    engine = ReccmpDiffEngine(
        manifest,
        args=args,
        verbose=False,
        threaded=args.threaded,
        max_ram_percent=args.max_ram_percent,
        no_symbols=True,
        engine_log_path=output / "ghidriff.log",
        engine_log_level=logging.WARNING,
        min_func_len=1,
        bsim=False,
        decompiler_timeout=args.decompiler_timeout,
    )
    (
        engine.reviewed_cdecl_signatures,
        engine.reviewed_scalar_returns,
    ) = _reviewed_signatures(args, manifest)
    reviewed_digest = hashlib.sha256(
        json.dumps(
            [
                sorted(engine.reviewed_cdecl_signatures.items()),
                sorted(engine.reviewed_scalar_returns.items()),
            ]
        ).encode()
    ).hexdigest()
    ghidra_version = str(engine.get_ghidra_version())
    # One project per original binary and analyzer: the original's analysis
    # is reused across recompiled builds, whose programs replace each other.
    project_name = (
        f"{manifest.target_id}-{manifest.orig.sha256[:12]}"
        f"-ghidra{ghidra_version}-ghidriff{ghidriff.__version__}"
        f"-reccmp{ANALYSIS_REVISION}"
        f"{'-switchfocus1' if engine.focused_switch_analysis else ''}"
    )
    prepared_key = (
        f"v{PREPARATION_REVISION}:{manifest.preparation_digest()}:{reviewed_digest}"
    )
    # Hash the installed Python implementations as well as version strings:
    # local edits in either fork must invalidate completed results too.
    completed_key = fingerprint_files(
        [
            *Path(reccmp.__file__).parent.rglob("*.py"),
            *Path(ghidriff.__file__).parent.rglob("*.py"),
        ],
        context=json.dumps(
            [
                manifest.digest(),
                prepared_key,
                project_name,
                args.decompiler_timeout,
                args.threaded,
                args.max_ram_percent,
                reccmp.VERSION,
                ghidriff.__version__,
            ]
        ),
    )
    cache = AnalysisCache(projects / "completed", enabled=not args.no_cache)
    cached: tuple[Any, Any, Any] | None = cache.load(
        "comparison-" + completed_key, completed_key
    )
    if cached is not None:
        pdiff, results, calls = cached
        print("[CACHE] Reusing completed comparison", file=sys.stderr)
    else:
        pdiff, results, calls = _compare_programs(
            engine,
            args,
            project_name=project_name,
            prepared_key=prepared_key,
        )
        if all(result.outcome != Outcome.ANALYSIS_FAILED for result in results):
            cache.store(
                "comparison-" + completed_key, completed_key, (pdiff, results, calls)
            )

    (output / "direct-calls.json").write_text(
        json.dumps(calls, indent=1) + "\n", encoding="utf-8"
    )
    # The report covers the requested functions only; Ghidriff's global
    # symbol and string inventories describe whole binaries.
    differing = {
        result.entry.orig_addr
        for result in results
        if result.outcome == Outcome.DIFFERENCES
    }
    pdiff["symbols"] = {"added": [], "deleted": []}
    pdiff["strings"] = {"added": [], "deleted": []}
    pdiff["functions"]["modified"] = [
        func
        for func in pdiff["functions"]["modified"]
        if int(func["old"]["address"], 16) in differing
    ]
    with _stage("write Ghidriff report"):
        engine.dump_pdiff_to_path(
            f"{manifest.target_id}.ghidriff",
            pdiff,
            output,
            side_by_side=args.side_by_side,
            max_section_funcs=len(differing) or 1,
            md_title=f"{manifest.target_id}: reconstructed functions with differences",
        )

    inputs = RunInputs(
        manifest_sha256=manifest.digest(),
        reccmp_version=reccmp.VERSION,
        ghidra_version=ghidra_version,
        ghidriff_version=ghidriff.__version__,
        ghidra_project=str(projects / project_name),
    )
    summary = summary_json(manifest, inputs, results)
    summary["inputs"]["comparison_key"] = completed_key
    summary["cache"] = {"reused": cached is not None}
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=1) + "\n",
        encoding="utf-8",
    )
    print_summary(results, details=args.details or bool(args.orig_address))
    print(f"Report: {summary_path}")


def _compare_programs(engine, args, *, project_name, prepared_key):
    """Analyze and compare only on a completed-result cache miss."""
    # pylint: disable=import-outside-toplevel
    from reccmp.ghidriff.project_cache import (
        reset_programs,
        restore_prepared,
        save_prepared,
    )

    output = args.output
    projects = args.ghidra_projects or (
        engine.manifest.recomp.path.parent / ".reccmp-cache" / "ghidra"
    )
    prepared_stamp = projects / project_name / "prepared-key.txt"
    orig, recomp = engine.manifest.orig.path, engine.manifest.recomp.path
    try:
        with _stage("set up project"):
            engine.setup_project(
                [orig, recomp], projects, project_name, output / "symbols"
            )
            engine.prune_programs([orig, recomp])
            engine.align_import_purges(orig, recomp)
        with _stage("analyze programs"):
            engine.analyze_project()
        with _stage("restore prepared programs"):
            restored = not args.no_cache and restore_prepared(
                engine.project, prepared_stamp, prepared_key
            )
        if restored:
            with _stage("collect cached references"):
                engine.collect_prepared_references(orig, ImageId.ORIG)
                engine.collect_prepared_references(recomp, ImageId.RECOMP)
        else:
            with _stage("reset and align programs"):
                reset_programs(engine.project)
                engine.align_import_purges(orig, recomp)
                engine.align_memory_permissions(orig, recomp)
            with _stage("prepare original"):
                engine.prepare_program(orig, ImageId.ORIG)
            with _stage("prepare recompiled"):
                engine.prepare_program(recomp, ImageId.RECOMP)
            if not args.no_cache and not engine.preparation_failed:
                with _stage("save prepared programs"):
                    save_prepared(engine.project, prepared_stamp, prepared_key)
        with _stage("decompile and diff"):
            pdiff = engine.diff_pairs(
                orig, recomp, engine.function_matches(), force_diff=True
            )
        from reccmp.compare.call_census import direct_call_census

        programs = {}
        try:
            for image, path in ((ImageId.ORIG, orig), (ImageId.RECOMP, recomp)):
                programs[image] = engine.project.openProgram(
                    "/", engine.gen_proj_bin_name_from_path(path), False
                )
            calls = direct_call_census(engine.manifest, programs)
            with _stage("inline-normalized retry"):
                engine.normalize_inlining(programs)
            with _stage("collect results"):
                results = engine.results()
        finally:
            for program in programs.values():
                engine.project.close(program)
    finally:
        engine.project.close()
    return pdiff, results, calls


def main() -> int:
    args = parse_args()
    try:
        target = argparse_parse_project_target(args)
    except RecCmpProjectException as e:
        logger.error(e.args[0])
        return 1

    try:
        catalog = Compare.from_target(
            target, orig_addrs=args.orig_address, use_cache=not args.no_cache
        )
    except SourceIndexError as e:
        logger.error("%s", e)
        return 1

    args.output = args.output or Path.cwd() / f"reccmp-{target.target_id}"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        catalog,
        target_id=target.target_id,
        orig_path=target.original_path,
        recomp_path=target.recompiled_path,
        select=_selection(args, target),
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest.to_json(), indent=1) + "\n", encoding="utf-8"
    )
    if args.orig_address and not manifest.functions:
        logger.error("No function at the requested original addresses")
        return 1

    _run_engine(args, target, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
