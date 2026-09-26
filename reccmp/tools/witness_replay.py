#!/usr/bin/env python3
"""Replay the witnesses of a comparison report from their records alone."""

import argparse
import importlib.util
import logging
from pathlib import Path

import reccmp
from reccmp.compare import Compare
from reccmp.compare.report import deserialize_reccmp_report
from reccmp.project.detect import (
    RecCmpProjectException,
    argparse_add_project_target_args,
    argparse_parse_project_target,
)
from reccmp.project.logging import argparse_add_logging_args, argparse_parse_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run each witness of a `reccmp-reccmp --witness --json` report again "
            "from the input it recorded, and say whether the same difference "
            "shows. Needs no solver and no search."
        )
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    parser.add_argument("report", type=Path, help="JSON report of reccmp-reccmp")
    parser.add_argument(
        "addresses",
        nargs="*",
        type=lambda text: int(text, 16),
        help="Original function addresses (hex); default: every witness",
    )
    argparse_add_project_target_args(parser)
    argparse_add_logging_args(parser)
    args = parser.parse_args()
    argparse_parse_logging(args)
    return args


def main() -> int:
    args = parse_args()
    if importlib.util.find_spec("unicorn") is None:
        logger.error("replay needs unicorn: pip install 'reccmp[witness]'")
        return 1
    # pylint: disable-next=import-outside-toplevel
    from reccmp.compare.witness.replay import replay

    try:
        target = argparse_parse_project_target(args)
    except RecCmpProjectException as error:
        logger.error("%s", error.args[0])
        return 1
    report = deserialize_reccmp_report(args.report.read_text(encoding="utf-8"))
    wanted = set(args.addresses)
    entities = [
        entity
        for address, entity in sorted(report.entities.items())
        if entity.analysis.witness is not None and (not wanted or address in wanted)
    ]
    if not entities:
        logger.error("no witnesses to replay")
        return 1
    compare = Compare.from_target(target)
    translator = compare.function_comparator.witness_translator()
    failed = 0
    for entity in entities:
        witness = entity.analysis.witness
        assert witness is not None
        result = replay(translator, witness)
        verdict = "reproduced" if result.reproduced else "NOT reproduced"
        if result.witness is not None and result.problems:
            verdict += f" (shows {result.witness.kind} at {result.witness.location})"
        problems = f": {', '.join(result.problems)}" if result.problems else ""
        print(f"{entity.orig_addr:#x} {entity.name}: {verdict}{problems}")
        failed += not result.reproduced
    print(f"{len(entities) - failed}/{len(entities)} reproduced")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
