"""Optional Clang source-index loading for comparison sessions.

Does not run Clang during compare. Loads an explicitly supplied or
previously collected ``source-index.json`` and scopes it to one target.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from reccmp.project.detect import RecCmpTarget
from reccmp.source.index import SourceIndex, SourceIndexError

logger = logging.getLogger(__name__)


def resolve_source_index_path(
    target: RecCmpTarget, *, explicit: Path | None = None
) -> Path | None:
    """Locate a validated-or-supplied source index without collecting anew."""
    if explicit is not None:
        return explicit
    env = os.environ.get("RECCMP_SOURCE_INDEX")
    if env:
        return Path(env)
    # Common collector output next to the recompiled binary / source trees.
    candidates: list[Path] = [
        target.recompiled_path.parent / "reccmp-source" / "source-index.json",
        target.recompiled_path.parent / "build" / "reccmp-source" / "source-index.json",
    ]
    for source_path in target.source_paths:
        root = source_path if source_path.is_dir() else source_path.parent
        candidates.append(root / "build" / "reccmp-source" / "source-index.json")
        candidates.append(root / "reccmp-source" / "source-index.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_source_index_for_target(
    target: RecCmpTarget, *, explicit: Path | None = None
) -> SourceIndex | None:
    """Load a target-scoped source index, or ``None`` when unavailable.

    Reports absence via debug log; parse/schema failures are logged as
    warnings and treated as absent capability.
    """
    path = resolve_source_index_path(target, explicit=explicit)
    if path is None:
        logger.debug(
            "source index unavailable for target %s (set RECCMP_SOURCE_INDEX)",
            target.target_id,
        )
        return None
    if not path.is_file():
        logger.warning("source index path is not a file: %s", path)
        return None
    try:
        document = path.read_text(encoding="utf-8")
        import json

        index = SourceIndex.from_dict(json.loads(document))
    except (OSError, ValueError, SourceIndexError, TypeError) as exc:
        logger.warning("source index at %s is unusable: %s", path, exc)
        return None
    scoped = index.for_target(target.target_id)
    if not scoped.classes and not scoped.variables and not scoped.declarations:
        # Index may be single-target without ``target`` fields (legacy); keep it.
        if any(item.target is not None for item in index.classes):
            logger.debug(
                "source index has no records for target %s", target.target_id
            )
            return None
        return index
    return scoped
