import logging
import argparse
from typing import Sequence

import colorama
import reccmp
import reccmp.color
from reccmp.compare import Compare
from reccmp.compare.diff import (
    CombinedDiffOutput,
    MatchingOrMismatchingBlock,
    raw_diff_to_udiff,
)
from reccmp.compare.stack_layout import (
    StackPair,
    StackPairs,
    StackRegisterOffset,
    StackSymbol,
    Warnings,
    analyze_diff_block,
    annotate_canonical_refs,
    annotate_recomp_symbols,
    collect_stack_pairs,
    extract_stack_offset_from_instruction,
)
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.project.detect import (
    argparse_add_project_target_args,
    argparse_parse_project_target,
    RecCmpProjectException,
)
from reccmp.project.logging import (
    argparse_add_logging_args,
    argparse_parse_logging,
)

# pylint: disable=duplicate-code # misdetects a code duplication with reccmp

logger = logging.getLogger(__name__)

colorama.just_fix_windows_console()

CHECK_ICON = f"{reccmp.color.Fore.GREEN}✓{reccmp.color.Style.RESET_ALL}"
SWAP_ICON = f"{reccmp.color.Fore.YELLOW}⇄{reccmp.color.Style.RESET_ALL}"
ERROR_ICON = f"{reccmp.color.Fore.RED}✗{reccmp.color.Style.RESET_ALL}"
UNCLEAR_ICON = f"{reccmp.color.Fore.BLUE}?{reccmp.color.Style.RESET_ALL}"

# Re-exports for tests / callers that imported these from the tool module.
__all__ = [
    "StackSymbol",
    "StackRegisterOffset",
    "StackPair",
    "StackPairs",
    "Warnings",
    "extract_stack_offset_from_instruction",
    "analyze_diff",
    "compare_function_stacks",
    "main",
]


def analyze_diff(diff: MatchingOrMismatchingBlock, warnings: Warnings) -> StackPairs:
    return analyze_diff_block(diff, warnings)


def print_bijective_match(left: str, right: str, exact: bool):
    icon = CHECK_ICON if exact else SWAP_ICON
    print(f"{icon}{reccmp.color.Style.RESET_ALL}  {left}: {right}")


def print_non_bijective_match(left: str, right: str):
    print(f"{ERROR_ICON}  {left}: {right}")


def print_structural_mismatch(
    orig: Sequence[tuple[str, ...]], recomp: Sequence[tuple[str, ...]]
) -> str:
    orig_str = "\n".join(f"-{x[1]}" for x in orig) if orig else "-"
    recomp_str = "\n".join(f"+{x[1]}" for x in recomp) if recomp else "+"
    return f"{reccmp.color.Fore.RED}{orig_str}\n{reccmp.color.Fore.GREEN}{recomp_str}\n{reccmp.color.Style.RESET_ALL}"


def format_list_of_offsets(offsets: list[StackRegisterOffset]) -> str:
    return str([str(x) for x in offsets])


def compare_function_stacks(udiff: CombinedDiffOutput, fn_symbol: SymbolsEntry):
    warnings = Warnings()
    stack_pairs, collected_warnings = collect_stack_pairs(udiff)
    warnings.structural_mismatches_present = (
        collected_warnings.structural_mismatches_present
    )

    # Preserve prior logging for structural mismatches in mismatch blocks.
    for block in udiff:
        for diff in block[1]:
            if "both" in diff:
                continue
            assert "orig" in diff and "recomp" in diff
            orig = diff["orig"]
            recomp = diff["recomp"]
            if len(orig) != len(recomp):
                if orig:
                    mismatch_location = f"orig={orig[0][0]}"
                else:
                    mismatch_location = f"recomp={recomp[0][0]}"
                logging.error(
                    "Structural mismatch at %s:\n%s",
                    mismatch_location,
                    print_structural_mismatch(orig, recomp),
                )

    stack_symbols = annotate_recomp_symbols(stack_pairs, fn_symbol)
    annotate_canonical_refs(stack_pairs)

    print_by_original_stack(stack_pairs, warnings)
    print_by_recomp_stack(stack_pairs, stack_symbols, warnings)
    print_footer(warnings)


def print_by_original_stack(stack_pairs: set[StackPair], warnings: Warnings):
    print("\nOrdered by original stack (left=orig, right=recomp):")

    all_orig_offsets = set(x.orig.offset for x in stack_pairs)

    for orig_offset in sorted(all_orig_offsets):
        orig = next(x.orig for x in stack_pairs if x.orig.offset == orig_offset)
        recomps = [x.recomp for x in stack_pairs if x.orig == orig]

        if len(recomps) == 1:
            recomp = recomps[0]
            print_bijective_match(str(orig), str(recomp), exact=orig == recomp)
        else:
            print_non_bijective_match(str(orig), format_list_of_offsets(recomps))
            warnings.error_map_not_bijective = True


def print_by_recomp_stack(
    stack_pairs: set[StackPair],
    stack_symbols: dict[int, StackSymbol],
    warnings: Warnings,
):
    all_recomp_offsets = set(x.recomp.offset for x in stack_pairs).union(
        stack_symbols.keys()
    )

    print("\nOrdered by recomp stack (left=orig, right=recomp):")
    for recomp_offset in sorted(all_recomp_offsets):
        recomp = next(
            (x.recomp for x in stack_pairs if x.recomp.offset == recomp_offset), None
        )

        if recomp is None:
            stack_offset = StackRegisterOffset(
                "ebp", recomp_offset, stack_symbols[recomp_offset]
            )
            print(f"{UNCLEAR_ICON}  not seen:   {stack_offset}")
            continue

        origs = [x.orig for x in stack_pairs if x.recomp == recomp]

        if len(origs) == 1:
            print_bijective_match(str(origs[0]), str(recomp), origs[0] == recomp)
        else:
            print_non_bijective_match(format_list_of_offsets(origs), str(recomp))
            warnings.error_map_not_bijective = True


def print_footer(warnings: Warnings):
    print(
        "\nLegend:\n"
        + f"{SWAP_ICON} : This stack variable matches 1:1, but the order of variables is not correct.\n"
        + f"{ERROR_ICON} : This stack variable matches multiple variables in the other binary.\n"
        + f"{UNCLEAR_ICON} : This stack variable did not appear in the diff. It either matches or only appears in structural mismatches.\n"
    )

    if warnings.error_map_not_bijective:
        print(
            "ERROR: The stack variables of original and recomp are not in a 1:1 correspondence, "
            + "suggesting that the logic in the recomp is incorrect."
        )
    elif warnings.structural_mismatches_present:
        print(
            "WARNING: Original and recomp have at least one structural discrepancy, "
            + "so the comparison of stack variables might be incomplete. "
            + "The structural mismatches above need to be checked manually."
        )


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
        "address",
        metavar="<offset>",
        type=virtual_address,
        help="The original file's offset of the function to be analyzed",
    )
    argparse_add_logging_args(parser)

    args = parser.parse_args()

    argparse_parse_logging(args=args)

    return args


def main() -> int:
    args = parse_args()

    try:
        target = argparse_parse_project_target(args=args)
    except RecCmpProjectException as e:
        logger.error(e.args[0])
        return 1

    compare = Compare.from_target(target)

    print()

    match = compare.compare_address(args.address)
    if match is None:
        print(f"Failed to find a match at address 0x{args.address:x}")
        return 1

    assert match.rdiff is not None
    # Analyze the entire function, including long sections that already match.
    # This comment explains why this is necessary:
    # https://github.com/isledecomp/reccmp/pull/307#issuecomment-3796146436
    udiff = raw_diff_to_udiff(match.rdiff, grouped=False)

    function_data = next(
        (y for y in compare.cvdump_analysis.nodes if y.addr == match.recomp_addr),
        None,
    )
    assert function_data is not None
    assert function_data.symbol_entry is not None

    compare_function_stacks(udiff, function_data.symbol_entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
