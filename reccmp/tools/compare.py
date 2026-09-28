#!/usr/bin/env python3
"""reccmp-reccmp: decompile and diff reconstructed functions with Ghidriff."""

import argparse
import json
import logging
from pathlib import Path

import colorama

import reccmp
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
from reccmp.types import ImageId

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
        help="Do not reuse cached PDB and catalog preparation",
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


def _run_engine(args: argparse.Namespace, target: RecCmpTarget, manifest: Manifest):
    # pylint: disable=import-outside-toplevel
    # Importing the engine does not start the JVM, but it does need ghidriff.
    import ghidriff
    from reccmp.ghidriff.engine import ReccmpDiffEngine
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
    ghidra_version = str(engine.get_ghidra_version())
    # One project per original binary and analyzer: the original's analysis
    # is reused across recompiled builds, whose programs replace each other.
    project_name = (
        f"{manifest.target_id}-{manifest.orig.sha256[:12]}"
        f"-ghidra{ghidra_version}-ghidriff{ghidriff.__version__}"
    )
    orig, recomp = manifest.orig.path, manifest.recomp.path
    try:
        engine.setup_project([orig, recomp], projects, project_name, output / "symbols")
        engine.prune_programs([orig, recomp])
        engine.analyze_project()
        engine.reset_programs()
        engine.align_memory_permissions(orig, recomp)
        engine.prepare_program(orig, ImageId.ORIG)
        engine.prepare_program(recomp, ImageId.RECOMP)
        pdiff = engine.diff_bins(orig, recomp, force_diff=True)
        results = engine.results()
    finally:
        engine.project.close()

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
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary_json(manifest, inputs, results), indent=1) + "\n",
        encoding="utf-8",
    )
    print_summary(results, details=args.details or bool(args.orig_address))
    print(f"Report: {summary_path}")


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
