"""Clang source-index loading for comparison sessions.

Does not run Clang during compare. Loads an explicitly supplied,
environment-selected, or build-configured (``source-index`` in
reccmp-build.yml) ``source-index.json`` and scopes it to one target.
Nearby unvalidated collector output is not auto-discovered: stale evidence
must be opted in, and the index is checked against the current sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from reccmp.project.detect import RecCmpTarget
from reccmp.source.index import SourceAbi, SourceIndex, SourceIndexError

logger = logging.getLogger(__name__)

# This backend compares 32-bit Microsoft-layout images. Indexes that claim
# another ABI must not feed pointer/layout comparison.
_MSVC32_POINTER_WIDTHS = frozenset({4, 32})


def resolve_source_index_path(
    target: RecCmpTarget, *, explicit: Path | None = None
) -> Path | None:
    """Locate an opted-in source index without collecting anew.

    ``explicit`` wins, then ``RECCMP_SOURCE_INDEX``, then the build config.
    Unvalidated files next to the binary are ignored.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("RECCMP_SOURCE_INDEX")
    if env:
        return Path(env)
    return target.source_index


def source_index_abi_compatible(abi: SourceAbi | None) -> bool:
    """True when the index may feed this 32-bit MSVC compare backend."""
    if abi is None:
        return True
    if abi.pointer_width not in _MSVC32_POINTER_WIDTHS:
        return False
    return bool(abi.ms_abi)


def _records_have_targets(index: SourceIndex) -> bool:
    for group in (
        index.classes,
        index.variables,
        index.declarations,
        index.markers,
    ):
        if any(item.target is not None for item in group):
            return True
    return False


def _scoped_is_empty(index: SourceIndex) -> bool:
    return not (index.classes or index.variables or index.declarations or index.markers)


def require_source_index(
    target: RecCmpTarget, *, explicit: Path | None = None
) -> SourceIndex:
    """Load the target-scoped source index, or say why it is unusable."""
    path = resolve_source_index_path(target, explicit=explicit)
    if path is None:
        raise SourceIndexError(
            f"no source index for target {target.target_id}: set `source-index` in "
            "reccmp-build.yml or RECCMP_SOURCE_INDEX"
        )
    if not path.is_file():
        raise SourceIndexError(f"source index path is not a file: {path}")
    index = SourceIndex.read(path)
    scoped = index.for_target(target.target_id)
    if _scoped_is_empty(scoped):
        if _records_have_targets(index):
            raise SourceIndexError(
                f"source index {path} has no records for target {target.target_id}"
            )
        scoped = index
    abi = scoped.abi
    if not source_index_abi_compatible(abi):
        raise SourceIndexError(
            "source index ABI is incompatible with 32-bit MSVC compare "
            f"(pointer_width={None if abi is None else abi.pointer_width} "
            f"ms_abi={None if abi is None else abi.ms_abi}): {path}"
        )
    return scoped


def load_source_index_for_target(
    target: RecCmpTarget, *, explicit: Path | None = None
) -> SourceIndex | None:
    """Load a target-scoped source index, or ``None`` when unavailable.

    Reports absence via debug log; parse/schema/ABI failures are logged as
    warnings and treated as absent capability.
    """
    if resolve_source_index_path(target, explicit=explicit) is None:
        logger.debug(
            "source index unavailable for target %s (set RECCMP_SOURCE_INDEX)",
            target.target_id,
        )
        return None
    try:
        return require_source_index(target, explicit=explicit)
    except SourceIndexError as exc:
        logger.warning("%s", exc)
        return None
