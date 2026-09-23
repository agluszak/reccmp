#!/usr/bin/env python3
"""Cluster a comparison report's verdicts by the verifier's shortcomings."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import reccmp
from reccmp.compare.triage import bucket_counts, instruction_shape, triage, triage_text
from reccmp.formats import detect_image
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
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
            "Cluster the verdicts of a `reccmp-reccmp --json` report (run with "
            "--witness for the execution buckets). Candidate mismatches the "
            "witness executed through without diverging come first: each large "
            "cluster there is usually one missing verifier rule."
        )
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {reccmp.VERSION}"
    )
    parser.add_argument("report", type=Path, help="JSON report of reccmp-reccmp")
    argparse_add_project_target_args(parser)
    parser.add_argument(
        "--no-binaries",
        action="store_true",
        help="Do not decode instruction shapes from the target's binaries",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Clusters shown per bucket"
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="Examples shown per cluster"
    )
    parser.add_argument("--json", type=Path, help="Also write the clusters as JSON")
    argparse_add_logging_args(parser)
    args = parser.parse_args()
    argparse_parse_logging(args)
    return args


def _shapes(args: argparse.Namespace):
    if args.no_binaries:
        return None
    try:
        target = argparse_parse_project_target(args)
    except RecCmpProjectException as error:
        logger.warning("no instruction shapes: %s", error.args[0])
        return None
    images = {
        "orig": detect_image(target.original_path),
        "recomp": detect_image(target.recompiled_path),
    }

    def shape(image: str, address: int) -> str | None:
        try:
            code = bytes(images[image].read(address, 16))
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            return None
        return instruction_shape(code, address)

    return shape


def main() -> int:
    args = parse_args()
    entities = json.loads(args.report.read_text(encoding="utf-8"))["data"]
    clusters = triage(entities, _shapes(args))
    print(triage_text(clusters, limit=args.limit, samples=args.samples))
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "buckets": bucket_counts(clusters),
                    "clusters": [
                        {
                            **dataclasses.asdict(cluster.key),
                            "count": cluster.count,
                            "samples": cluster.samples,
                        }
                        for cluster in clusters
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
