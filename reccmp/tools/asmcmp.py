#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import logging
import os

import colorama
import reccmp
from reccmp.utils import (
    gen_svg,
    print_combined_diff,
    diff_json,
    percent_string,
    safe_denominator,
    write_html_report,
)

from reccmp.compare import Compare
from reccmp.compare.exact import compare_object_to_original
from reccmp.formats.coff import parse_coff_object
from reccmp.formats.detect import detect_image
from reccmp.formats.pe import PEImage
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DiagnosticNormalization,
    EquivalenceLevel,
)
from reccmp.compare.db import ReccmpEntity
from reccmp.compare.diff import raw_diff_to_udiff
from reccmp.compare.report import (
    ReccmpStatusReport,
    ReccmpComparedEntity,
    deserialize_reccmp_report,
    serialize_reccmp_report,
    report_function_alignment,
    report_function_accuracy,
    format_address,
)
from reccmp.types import EntityType
from reccmp.project.logging import (
    argparse_add_logging_args,
    argparse_parse_logging,
)
from reccmp.project.detect import (
    RecCmpProjectException,
    argparse_add_project_target_args,
    argparse_parse_project_target,
    RecCmpProject,
)

logger = logging.getLogger()
colorama.just_fix_windows_console()


def gen_json(json_file: str, json_str: str):
    """Convert the status report to JSON and write to a file."""

    with open(json_file, "w", encoding="utf-8") as f:
        f.write(json_str)


def triage_status_note(analysis: ComparisonAnalysis) -> str | None:
    """A one-line reminder of what a triage status means, so the semantics
    travel with the output and a reader does not misread the result. Returned
    for the two statuses that are routinely misread; None for `exact` (needs no
    gloss) and `mismatch` (the actionable case, whose diff speaks for itself)."""
    if analysis.status == ComparisonStatus.EFFECTIVE:
        return "effective: proved semantically harmless — no action needed"
    if analysis.status == ComparisonStatus.INCONCLUSIVE:
        return (
            "inconclusive: verifier could not prove either outcome — "
            "NOT evidence of a source defect; investigate verifier/metadata/alignment"
        )
    return None


def semantic_similarity_text(match: ReccmpComparedEntity) -> str | None:
    """Render the optional repair-oriented score without implying proof."""
    if (
        match.analysis.status != ComparisonStatus.MISMATCH
        or match.semantic_similarity is None
    ):
        return None
    semantic = percent_string(match.semantic_similarity)
    raw = percent_string(match.accuracy)
    return f"{semantic} semantic similarity (diagnostic; {raw} raw)"


def inconclusive_diagnostic_text(analysis: ComparisonAnalysis) -> str | None:
    """Render the structured reason, location, and facts for an inconclusive result."""
    if analysis.status != ComparisonStatus.INCONCLUSIVE:
        return None
    reason = (analysis.inconclusive_reason or "analysis_limit").replace("_", " ")
    lines = [f"semantic analysis inconclusive: {reason}"]
    location = analysis.inconclusive_location
    if location is not None:
        if location.address is not None:
            lines.append(f"  location: {format_address(location.address)}")
        elif location.instruction_index is not None:
            lines.append(f"  instruction index: {location.instruction_index}")
        for key, value in sorted(location.facts.items()):
            lines.append(f"  {key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def mismatch_source_pin_text(match: ReccmpComparedEntity) -> str | None:
    """First recomp source line attached to a structured mismatch, if any."""
    difference = match.analysis.difference
    if difference is None:
        return None
    facts = difference.recomp.facts
    path = facts.get("source_path")
    line = facts.get("source_line")
    if isinstance(path, str) and isinstance(line, int):
        return f"probable first source-level discrepancy: {path}:{line}"
    return None


def stack_layout_text(match: ReccmpComparedEntity) -> str | None:
    """Human-readable stack permutation / modulo-stack score."""
    if not match.stack_permutation and match.accuracy_modulo_stack is None:
        return None
    lines: list[str] = []
    if match.accuracy_modulo_stack is not None:
        raw = percent_string(match.accuracy)
        modulo = percent_string(match.accuracy_modulo_stack)
        lines.append(f"{raw} raw / {modulo} modulo stack allocation")
    if match.stack_permutation:
        lines.append("stack permutation:")
        for entry in match.stack_permutation:
            if entry.orig == entry.recomp:
                continue
            symbol = f"  {entry.symbol}" if entry.symbol else ""
            lines.append(f"    {entry.orig} -> {entry.recomp}{symbol}")
    return "\n".join(lines) if lines else None


def inline_layout_text(match: ReccmpComparedEntity) -> str | None:
    """Human-readable known-inline expansions / modulo-inline score."""
    if not match.inline_expansions and match.accuracy_modulo_inline is None:
        return None
    lines: list[str] = []
    if match.accuracy_modulo_inline is not None:
        raw = percent_string(match.accuracy)
        modulo = percent_string(match.accuracy_modulo_inline)
        lines.append(f"{raw} raw / {modulo} modulo known inline expansion")
    if match.inline_expansions:
        lines.append("known inline expansions:")
        for entry in match.inline_expansions:
            where = entry.side
            counterpart = entry.counterpart
            detail = f"insn@{entry.match_offset}+{entry.match_length}"
            if entry.counterpart_offset is not None:
                detail += f" ↔ {counterpart}@{entry.counterpart_offset}"
            else:
                detail += f" ({counterpart})"
            lines.append(
                f"    {format_address(entry.helper_orig_addr)}  {entry.helper_name}  "
                f"on {where}: {detail}"
            )
    return "\n".join(lines) if lines else None


def equivalence_level_text(match: ReccmpComparedEntity) -> str | None:
    """Deprecated: prefer diagnostic_normalizations_text."""
    return diagnostic_normalizations_text(match)


def diagnostic_normalizations_text(match: ReccmpComparedEntity) -> str | None:
    """Render non-proof diagnostic tags without claiming equivalence."""
    tags = match.diagnostic_normalizations
    if not tags and match.equivalence_level not in (
        EquivalenceLevel.EXACT_INSTRUCTIONS,
        EquivalenceLevel.UNKNOWN_DIFFERENCE,
    ):
        # Legacy reports may only have equivalence_level.
        legacy = match.equivalence_level.value.replace("_equivalent", "")
        if legacy in {tag.value for tag in DiagnosticNormalization}:
            tags = (DiagnosticNormalization(legacy),)
    if not tags:
        return None
    joined = ", ".join(tag.value.replace("_", " ") for tag in tags)
    return f"diagnostic normalizations: {joined}"


def print_match_verbose(match: ReccmpComparedEntity, show_both_addrs: bool = False):
    percenttext = percent_string(match.effective_accuracy, match.is_effective_match)

    if show_both_addrs and match.recomp_addr is not None:
        addrs = (
            f"{format_address(match.orig_addr)} / {format_address(match.recomp_addr)}"
        )
    else:
        addrs = format_address(match.orig_addr)

    grouped_diff = match.type != EntityType.VTABLE
    assert match.rdiff is not None
    udiff = raw_diff_to_udiff(match.rdiff, grouped=grouped_diff)

    note = triage_status_note(match.analysis)

    if match.effective_accuracy == 1.0:
        ok_text = reccmp.color.Fore.GREEN + "✨ OK! ✨" + reccmp.color.Style.RESET_ALL
        if match.accuracy == 1.0:
            print(f"{addrs}: {match.name} 100% match.\n\n{ok_text}\n\n")
        else:
            print_combined_diff(udiff, show_both_addrs)

            print(
                f"\n{addrs}: {match.name} 100% effective match (differs, but only in ways that don't affect behavior)."
                f"\n{note}\n\n{ok_text}\n\n"
            )
            stack = stack_layout_text(match)
            if stack is not None:
                print(stack)
            inline = inline_layout_text(match)
            if inline is not None:
                print(inline)
            level = equivalence_level_text(match)
            if level is not None:
                print(level)

    else:
        print_combined_diff(udiff, show_both_addrs)
        semantic = semantic_similarity_text(match)
        if semantic is not None:
            print(f"\n{match.name} has {semantic}; diff above")
        else:
            print(
                f"\n{match.name} is only {percenttext} similar to the original, diff above"
            )
        stack = stack_layout_text(match)
        if stack is not None:
            print(stack)
        inline = inline_layout_text(match)
        if inline is not None:
            print(inline)
        level = equivalence_level_text(match)
        if level is not None:
            print(level)
        source_pin = mismatch_source_pin_text(match)
        if source_pin is not None:
            print(source_pin)
        diagnostic = inconclusive_diagnostic_text(match.analysis)
        if diagnostic is not None:
            print(diagnostic)
        if note is not None:
            print(note)


def print_match_oneline(match: ReccmpComparedEntity, show_both_addrs: bool = False):
    percenttext = percent_string(match.effective_accuracy, match.is_effective_match)

    if show_both_addrs and match.recomp_addr is not None:
        addrs = (
            f"{format_address(match.orig_addr)} / {format_address(match.recomp_addr)}"
        )
    else:
        addrs = format_address(match.orig_addr)

    if match.is_stub:
        print(f"  {match.name} ({addrs}) is a stub.")
    else:
        semantic = semantic_similarity_text(match)
        if semantic is not None:
            print(f"  {match.name} ({addrs}) has {semantic}")
        elif (
            match.accuracy_modulo_inline is not None
            and match.accuracy_modulo_inline > match.accuracy
        ):
            raw = percent_string(match.accuracy)
            modulo = percent_string(match.accuracy_modulo_inline)
            print(
                f"  {match.name} ({addrs}) is {raw} raw / {modulo} modulo known inline"
            )
        elif (
            match.accuracy_modulo_stack is not None
            and match.accuracy_modulo_stack > match.accuracy
        ):
            raw = percent_string(match.accuracy)
            modulo = percent_string(match.accuracy_modulo_stack)
            print(f"  {match.name} ({addrs}) is {raw} raw / {modulo} modulo stack")
        else:
            print(f"  {match.name} ({addrs}) is {percenttext} similar to the original")
        level = equivalence_level_text(match)
        if level is not None and match.effective_accuracy < 1.0:
            print(f"    {level}")


def parse_args() -> argparse.Namespace:
    def virtual_address(value) -> int:
        """Helper method for argparse, verbose parameter"""
        return int(value, 16)

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Recompilation Compare: compare an original EXE with a recompiled EXE + PDB.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    argparse_add_project_target_args(parser)
    parser.add_argument(
        "--object",
        type=Path,
        help="Compare a COFF contribution without a recompiled PE or PDB",
    )
    parser.add_argument(
        "--symbol", help="Exact COFF linker symbol for --object (including decoration)"
    )
    parser.add_argument(
        "--size",
        type=lambda value: int(value, 0),
        help="Independently known original extent for --object",
    )
    parser.add_argument(
        "--total",
        "-T",
        metavar="<count>",
        help="Total number of expected functions (improves total accuracy statistic)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        metavar="<offset>",
        type=virtual_address,
        help="Print assembly diff for specific function (original file's offset)",
    )
    parser.add_argument(
        "--orig-address",
        metavar="<offset>",
        type=virtual_address,
        action="append",
        default=[],
        help="Compare only this original address (repeatable).",
    )
    parser.add_argument(
        "--recomp-address",
        metavar="<offset>",
        type=virtual_address,
        action="append",
        default=[],
        help="Compare only this recompiled address (repeatable).",
    )
    parser.add_argument(
        "--json",
        metavar="<file>",
        help="Generate JSON file with match summary",
    )
    parser.add_argument(
        "--json-diet",
        action="store_true",
        help="Exclude diff from JSON report.",
    )
    parser.add_argument(
        "--diff",
        metavar="<file>",
        help="Diff against summary in JSON file",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Write decompiled assembly to debug files.",
    )
    parser.add_argument(
        "--html",
        "-H",
        metavar="<file>",
        help="Generate searchable HTML summary of status and diffs",
    )
    parser.add_argument(
        "--no-color", "-n", action="store_true", help="Do not color the output"
    )
    parser.add_argument(
        "--svg", "-S", metavar="<file>", help="Generate SVG graphic of progress"
    )
    parser.add_argument(
        "--svg-icon", metavar="icon", type=Path, help="Icon to use in SVG (PNG)"
    )
    parser.add_argument(
        "--print-rec-addr",
        action="store_true",
        help="Print addresses of recompiled functions too",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Don't display text summary of matches",
    )
    parser.add_argument(
        "--nolib",
        action="store_true",
        help="Exclude LIBRARY annotations from the analysis",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the local parsed-analysis cache.",
    )
    argparse_add_logging_args(parser)

    args = parser.parse_args()
    if args.object is not None:
        if (
            not args.symbol
            or args.size is None
            or args.size <= 0
            or len(args.orig_address) != 1
        ):
            parser.error(
                "--object requires --symbol, positive --size and one --orig-address"
            )
        if any(
            (
                args.verbose is not None,
                args.recomp_address,
                args.html,
                args.svg,
                args.diff,
                args.dump,
            )
        ):
            parser.error("--object does not use executable diff/report options")
    elif args.symbol is not None or args.size is not None:
        parser.error("--symbol and --size require --object")
    argparse_parse_logging(args)

    return args


def dump_all_matched_functions(report: ReccmpStatusReport):
    logger.info("Creating assembly dump files.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Extract instructions from each compared entity in both address spaces.
    orig_items = [
        (entity.orig_addr, entity.name, entity.rdiff.orig_inst)
        for entity in report.entities.values()
        if entity.recomp_addr is not None and entity.rdiff is not None
    ]

    # mypy: recomp_addr can be None, but not for the matched entities we are reviewing.
    recomp_items = [
        (entity.recomp_addr, entity.name, entity.rdiff.recomp_inst)
        for entity in report.entities.values()
        if entity.recomp_addr is not None and entity.rdiff is not None
    ]

    # Sort by each binary's address order
    orig_items.sort(key=lambda v: v[0])
    recomp_items.sort(key=lambda v: v[0])

    orig_filename = f"reccmp-{timestamp}-orig.txt"
    recomp_filename = f"reccmp-{timestamp}-recomp.txt"

    for filename, vitals in (
        (orig_filename, orig_items),
        (recomp_filename, recomp_items),
    ):
        with open(filename, "w+", encoding="utf-8") as f:
            for _, name, instructions in vitals:
                f.write(f"; {name}\n")
                for addr, line in instructions:
                    if addr:
                        f.write(f"{addr:10}: {line}\n")
                    else:
                        f.write(f"        : {line}\n")


def compare_object(args: argparse.Namespace) -> int:
    """Run the object view without requiring any recompiled linker output."""
    original_path = (
        RecCmpProject.from_directory(Path.cwd()).targets[args.target].original_path
        if args.target
        else args.paths_target.original_path
    )
    if original_path is None:
        raise ValueError("The selected target has no original binary")
    original = detect_image(original_path)
    if not isinstance(original, PEImage):
        raise ValueError("Object comparison currently supports i386 PE originals")
    result = compare_object_to_original(
        original,
        parse_coff_object(args.object),
        args.symbol,
        args.orig_address[0],
        args.size,
    )
    output = json.dumps(
        {
            "object": str(args.object),
            "symbol": args.symbol,
            "original_address": args.orig_address[0],
            **asdict(result),
        },
        indent=2,
    )
    if args.json:
        gen_json(args.json, output)
    if not args.silent:
        print(output)
    return 0 if result.exact else 1


def main() -> int:
    args = parse_args()

    if args.object is not None:
        return compare_object(args)

    try:
        target = argparse_parse_project_target(args)
    except RecCmpProjectException as e:
        logger.error("%s", e.args[0])
        return 1

    logging.basicConfig(level=args.loglevel, format="[%(levelname)s] %(message)s")

    selected = bool(args.orig_address or args.recomp_address)
    setup_orig_addresses = list(args.orig_address)
    if args.verbose is not None:
        setup_orig_addresses.append(args.verbose)
    compare = Compare.from_target(
        target,
        orig_addrs=setup_orig_addresses,
        recomp_addrs=args.recomp_address,
        use_cache=not args.no_cache,
    )

    print()

    ### Compare one or none.

    if args.verbose is not None:
        match = compare.compare_address(args.verbose)
        if match is None:
            logger.error("Failed to find a match at address 0x%x", args.verbose)
            return 1

        print_match_verbose(match, show_both_addrs=args.print_rec_addr)
        return 0

    ### Compare selected entities or everything.

    def entity_filter(entity: ReccmpEntity) -> bool:
        if (
            entity.entity_type == EntityType.FUNCTION
            and entity.name in target.report_config.ignore_functions
        ):
            return False

        if args.nolib and entity.get("library"):
            return False

        return True

    include_diff = bool(
        args.dump
        or args.html is not None
        or (args.json is not None and not args.json_diet)
    )
    if selected:
        report = ReccmpStatusReport(filename=target.original_path.name)
        for entity in compare.compare_addresses(
            args.orig_address,
            args.recomp_address,
            include_diff=include_diff,
            include_exact_diff=bool(args.dump),
        ):
            report.add_match(entity)
        report.asmcmp_filtering(args.nolib, target.report_config.ignore_functions)
    else:
        report = compare.to_report(
            filename=target.original_path.name,
            filter_fn=entity_filter,
            include_diff=include_diff,
            include_exact_diff=bool(args.dump),
        )

    if args.dump:
        dump_all_matched_functions(report)

    # If we know how many functions are in the file (via analysis with Ghidra or other tools)
    # we can substitute an alternate value to use when calculating the percentages below.
    if args.total:
        # Use the alternate value if it exceeds the number of known functions
        report.function_count = max(report.function_count, int(args.total))

    # Count how many functions have the same virtual address in orig and recomp.
    functions_aligned_count = report_function_alignment(report)

    # Number of functions compared (i.e. excluding stubs)
    implemented_funcs, _, total_effective_accuracy = report_function_accuracy(report)

    # Print diff summary to terminal
    if not args.silent and args.diff is None:
        for entity in report.entities.values():
            if entity.is_matched():
                print_match_oneline(entity, show_both_addrs=args.print_rec_addr)

    # Compare with saved diff report.
    if args.diff is not None:
        try:
            with open(args.diff, "r", encoding="utf-8") as f:
                saved_data = deserialize_reccmp_report(f.read())

            saved_data.asmcmp_filtering(
                args.nolib, target.report_config.ignore_functions
            )

            diff_json(
                saved_data,
                report,
                show_both_addrs=args.print_rec_addr,
            )
        except FileNotFoundError:
            # In a CI workflow, the JSON file might not exist on the first run in a new branch.
            # Continue without a fatal error so users don't have to bother handling this situation.
            logger.error("Could not open JSON report file '%s' for diff", args.diff)

    ## Generate files and show summary.

    if args.json is not None:
        # If we're on a diet, hold the diff.
        diff_included = not bool(args.json_diet)
        gen_json(
            args.json, serialize_reccmp_report(report, diff_included=diff_included)
        )

    target_icon = args.svg_icon or target.report_config.icon

    if args.html is not None:
        write_html_report(args.html, report, target_icon)

    if selected:
        return 0

    report.update_function_count()
    function_count = report.function_count

    implemented = implemented_funcs / safe_denominator(function_count) * 100

    effective_accuracy = (
        total_effective_accuracy / safe_denominator(implemented_funcs) * 100
    )
    progress = total_effective_accuracy / safe_denominator(function_count) * 100
    alignment_percentage = (
        functions_aligned_count / safe_denominator(function_count) * 100
    )

    print(
        f"\nImplemented:  {implemented:.2f}%  ({implemented_funcs} / {function_count})"
    )
    print(f"Accuracy:     {effective_accuracy:.2f}%")
    print(f"Progress:     {progress:.2f}%")

    if functions_aligned_count > 0:
        print(
            f"{functions_aligned_count} functions are aligned ({alignment_percentage:.2f}%)"
        )

    if args.svg is not None:
        gen_svg(
            args.svg,
            os.path.basename(target.original_path),
            target_icon,
            implemented_funcs,
            function_count,
            total_effective_accuracy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
