"""Search original binaries for probable inline expansions of a helper."""

from __future__ import annotations

import argparse
import logging
import sys

import reccmp
from reccmp.compare import Compare
from reccmp.compare.report import format_address
from reccmp.project.detect import (
    RecCmpProjectException,
    argparse_add_project_target_args,
    argparse_parse_project_target,
)
from reccmp.project.logging import argparse_add_logging_args, argparse_parse_logging
from reccmp.types import EntityType

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Find probable inline expansions of a helper function inside larger "
            "original bodies."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    argparse_add_project_target_args(parser)
    parser.add_argument(
        "helper",
        help="Helper name (substring match) or original address (0x…)",
    )
    argparse_add_logging_args(parser)
    args = parser.parse_args()
    argparse_parse_logging(args=args)
    return args


def _resolve_helper(compare: Compare, helper: str):
    if helper.lower().startswith("0x"):
        addr = int(helper, 16)
        match = compare.get_match(addr)
        if match is None or match.entity_type != EntityType.FUNCTION:
            return None
        return match

    needle = helper.lower()
    candidates = []
    for entity in compare.get_functions():
        name = entity.best_name() or ""
        if needle in name.lower():
            candidates.append(entity)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    logger.error(
        "Ambiguous helper %r matches %d functions; pass an exact 0x address",
        helper,
        len(candidates),
    )
    for entity in candidates[:20]:
        name = entity.best_name() or "?"
        print(f"  {format_address(entity.orig_addr)}  {name}", file=sys.stderr)
    return None


def main() -> int:
    args = parse_args()
    try:
        target = argparse_parse_project_target(args=args)
    except RecCmpProjectException as exc:
        logger.error("%s", exc.args[0])
        return 1

    compare = Compare.from_target(target)
    helper = _resolve_helper(compare, args.helper)
    if helper is None:
        logger.error("Could not resolve helper %r", args.helper)
        return 1

    hits = compare.function_comparator.find_inlines(helper)
    helper_name = helper.best_name() or format_address(helper.orig_addr)
    if not hits:
        print(f"No probable inline expansions of {helper_name}")
        return 0

    print(f"Probable inline expansions of {helper_name}:")
    for hit in hits:
        print(
            f"  {format_address(hit.host_addr)}  {hit.host_name}  "
            f"insn@{hit.match_offset}+{hit.match_length}  "
            f"confidence={hit.confidence:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
