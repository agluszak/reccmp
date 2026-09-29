#!/usr/bin/env python3

import argparse
import logging
import colorama
import reccmp
from reccmp.compare import Compare
from reccmp.compare.db import ReccmpEntity
from reccmp.compare.vtables import SlotStatus, VtableComparison, compare_vtable
from reccmp.project.logging import (
    argparse_add_logging_args,
    argparse_parse_logging,
)
from reccmp.project.detect import (
    argparse_add_project_target_args,
    argparse_parse_project_target,
    RecCmpProjectException,
)
from reccmp.utils import format_address

logger = logging.getLogger(__name__)

# Ignore all compare-db messages.
logging.getLogger("compare").addHandler(logging.NullHandler())

colorama.just_fix_windows_console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparing vtables.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    argparse_add_project_target_args(parser)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show more detailed information"
    )
    parser.add_argument(
        "--no-color", "-n", action="store_true", help="Do not color the output"
    )
    parser.add_argument(
        "--filter",
        "-f",
        metavar="<substring>",
        default=None,
        help="Only compare vtables whose name contains this (case-insensitive) substring",
    )
    argparse_add_logging_args(parser)

    args = parser.parse_args()

    argparse_parse_logging(args)

    return args


def slot_text(entity: ReccmpEntity | None, raw_addr: int | None) -> str:
    """The function reference in one vtable slot."""
    if entity is not None:
        orig = (
            format_address(entity.orig_addr)
            if entity.orig_addr is not None
            else "no orig"
        )
        recomp = (
            format_address(entity.recomp_addr)
            if entity.recomp_addr is not None
            else "no recomp"
        )
        return f"({orig} / {recomp})  :  {entity.best_name()}"
    if raw_addr == 0:
        return "0x0 (null slot)"
    if raw_addr is not None:
        return f"{format_address(raw_addr)} not annotated"
    return "(no slot)"


def show_vtable(comparison: VtableComparison, plain: bool):
    for slot in comparison.slots:
        orig = slot_text(slot.orig, slot.orig_raw)
        recomp = slot_text(slot.recomp, slot.recomp_raw)
        index = f"vtable0x{slot.offset:02x}"
        if slot.status == SlotStatus.UNPAIRED:
            recomp += "  (not paired)"
        if slot.matches:
            print(f"  {index}  {recomp}")
        elif plain:
            print(f"- {index}  {orig}")
            print(f"+ {index}  {recomp}")
        else:
            print(f"{colorama.Fore.RED}- {index}  {orig}{colorama.Style.RESET_ALL}")
            print(f"{colorama.Fore.GREEN}+ {index}  {recomp}{colorama.Style.RESET_ALL}")


def print_summary(
    vtable_count: int, different: int, unpaired: int, code_equivalent: int
):
    if different == 0 and unpaired == 0:
        print(f"Vtables found: {vtable_count}.")
        if code_equivalent:
            print("No differing or unpaired slots.")
            print(f"Vtables with return-only code-equivalent slots: {code_equivalent}.")
        else:
            print("100% match.")
        return

    print(f"Vtables found: {vtable_count}.")
    print(f"Vtables with a different slot: {different}.")
    print(f"Vtables with only unpaired slots otherwise matching: {unpaired}.")
    if code_equivalent:
        print(f"Vtables with return-only code-equivalent slots: {code_equivalent}.")


def main():
    args = parse_args()
    vtable_count = 0
    different = 0
    unpaired = 0
    code_equivalent = 0

    try:
        target = argparse_parse_project_target(args)
    except RecCmpProjectException as e:
        logger.error(e.args[0])
        return 1

    catalog = Compare.from_target(target)

    name_filter = args.filter.lower() if args.filter else None
    catalog.report_vtable_size_warnings(name_filter)

    for tbl_match in catalog.get_vtables():
        if (
            name_filter is not None
            and name_filter not in (tbl_match.name or "").lower()
        ):
            continue
        vtable_count += 1
        comparison = compare_vtable(
            catalog.db, catalog.orig_bin, catalog.recomp_bin, tbl_match
        )
        if any(slot.status == SlotStatus.CODE_EQUIVALENT for slot in comparison.slots):
            code_equivalent += 1
        if comparison.matches:
            continue
        if any(slot.status == SlotStatus.DIFFERENT for slot in comparison.slots):
            different += 1
        else:
            unpaired += 1
        print(
            tbl_match.name,
            f": orig {format_address(tbl_match.orig_addr)}, recomp {format_address(tbl_match.recomp_addr)}",
        )
        show_vtable(comparison, args.no_color)
        print()

    if vtable_count == 0:
        if args.filter:
            logger.error("No vtables matched filter %r", args.filter)
        else:
            logger.error("No vtables found")
        return 1

    print_summary(vtable_count, different, unpaired, code_equivalent)
    return 1 if different or unpaired else 0


if __name__ == "__main__":
    raise SystemExit(main())
