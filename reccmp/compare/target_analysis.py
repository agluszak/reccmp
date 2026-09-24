"""Target loading, module-scoped PDB analysis, and validated setup caching."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterable

from reccmp.analysis_cache import (
    AnalysisCache,
    fingerprint_files,
    fingerprint_text_files,
)
from reccmp.cvdump import Cvdump, CvdumpAnalysis, CvdumpParser, CvdumpTypesParser
from reccmp.cvdump.targeted import (
    SymbolModuleHint,
    merge_module_symbols,
    select_modules,
)
from reccmp.dir import source_code_search
from reccmp.formats import Image, TextFile, detect_image
from reccmp.parser import DecompCodebase
from reccmp.parser.marker import ProjectAliases, normalize_project_aliases
from reccmp.project.detect import RecCmpTarget
from reccmp.source.index import SourceIndex, SourceIndexError

from .db import EntityDb
from .lines import LinesDb
from .source_capability import require_source_index

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_CVDUMP_CACHE_INPUTS = (
    _PACKAGE_ROOT / "cvdump" / "analysis.py",
    _PACKAGE_ROOT / "cvdump" / "parser.py",
    _PACKAGE_ROOT / "cvdump" / "runner.py",
    _PACKAGE_ROOT / "cvdump" / "symbols.py",
    _PACKAGE_ROOT / "cvdump" / "targeted.py",
    _PACKAGE_ROOT / "cvdump" / "types.py",
)
_MARKER_CACHE_INPUTS = (
    _PACKAGE_ROOT / "parser" / "codebase.py",
    _PACKAGE_ROOT / "parser" / "marker.py",
    _PACKAGE_ROOT / "parser" / "node.py",
    _PACKAGE_ROOT / "parser" / "reader.py",
)
_COMPARE_CACHE_INPUTS = tuple(
    sorted((_PACKAGE_ROOT / "compare").rglob("*.py"), key=str)
) + (
    _PACKAGE_ROOT / "analysis_cache.py",
    _PACKAGE_ROOT / "formats" / "image.py",
    _PACKAGE_ROOT / "formats" / "pe.py",
    _PACKAGE_ROOT / "types.py",
)


@dataclass
class PreparedAnalysis:
    db: EntityDb
    lines_db: LinesDb
    types: CvdumpTypesParser


@dataclass
class LoadedTargetAnalysis:
    # pylint: disable=too-many-instance-attributes
    orig_bin: Image
    recomp_bin: Image
    pdb_file: CvdumpAnalysis
    code_files: list[TextFile]
    data_sources: list[TextFile]
    equivalence_sources: list[TextFile]
    project_aliases: ProjectAliases
    codebase: DecompCodebase
    source_index: SourceIndex
    cache: AnalysisCache
    prepared_cache_name: str
    prepared_fingerprint: str

    def load_prepared(self) -> PreparedAnalysis | None:
        return self.cache.load(self.prepared_cache_name, self.prepared_fingerprint)

    def store_prepared(self, prepared: PreparedAnalysis) -> None:
        self.cache.store(self.prepared_cache_name, self.prepared_fingerprint, prepared)


def _load_full_cvdump(
    pdb_path: Path, cache: AnalysisCache, fingerprint: str
) -> CvdumpParser:
    cached: CvdumpParser | None = cache.load("cvdump-full", fingerprint)
    if cached is not None:
        return cached
    parser = (
        Cvdump(str(pdb_path))
        .lines()
        .globals()
        .publics()
        .symbols()
        .section_contributions()
        .types()
        .run()
    )
    cache.store("cvdump-full", fingerprint, parser)
    return parser


def _load_base_cvdump(
    pdb_path: Path, cache: AnalysisCache, fingerprint: str
) -> CvdumpParser:
    cached: CvdumpParser | None = cache.load("cvdump-base", fingerprint)
    if cached is not None:
        return cached
    parser = (
        Cvdump(str(pdb_path))
        .lines()
        .globals()
        .publics()
        .section_contributions()
        .types()
        .modules()
        .run()
    )
    cache.store("cvdump-base", fingerprint, parser)
    return parser


def _load_module_cvdump(
    pdb_path: Path,
    module_id: int,
    cache: AnalysisCache,
    fingerprint: str,
) -> CvdumpParser:
    cache_name = f"cvdump-module-{module_id}"
    cached: CvdumpParser | None = cache.load(cache_name, fingerprint)
    if cached is not None:
        return cached
    parser = Cvdump(str(pdb_path)).symbols().module(module_id).run()
    cache.store(cache_name, fingerprint, parser)
    return parser


def _load_source_markers(
    target: RecCmpTarget,
    code_files: list[TextFile],
    source_index: SourceIndex,
    *,
    use_cache: bool,
) -> tuple[DecompCodebase, str]:
    """The target's markers as the compiler saw them in the current sources."""
    stale = source_index.stale_sources(file.path for file in code_files)
    if stale:
        listed = ", ".join(str(path) for path in stale[:5])
        more = f" and {len(stale) - 5} more" if len(stale) > 5 else ""
        raise SourceIndexError(
            f"the source index is older than {listed}{more}; collect it again"
        )
    codebase = DecompCodebase.from_source_index(
        source_index,
        target.target_id,
        (file.path for file in code_files),
        aliases=normalize_project_aliases({target.target_id: target.marker_aliases}),
        encoding=target.encoding or "latin1",
    )
    marker_fingerprint = ""
    if use_cache:
        marker_fingerprint = fingerprint_files(
            _MARKER_CACHE_INPUTS,
            context=json.dumps(
                {
                    "target": target.target_id,
                    "aliases": target.marker_aliases,
                    "sources": source_index.source_digests,
                    "blocks": [block.to_dict() for block in source_index.marker_blocks],
                },
                sort_keys=True,
            ),
        )
    return codebase, marker_fingerprint


def _load_cvdump(
    target: RecCmpTarget,
    recomp_bin: Image,
    codebase: DecompCodebase,
    orig_addrs: tuple[int, ...],
    recomp_addrs: tuple[int, ...],
    cache: AnalysisCache,
    *,
    use_cache: bool,
) -> tuple[CvdumpParser, str, str]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pdb_fingerprint = (
        fingerprint_files(
            (target.recompiled_pdb, *_CVDUMP_CACHE_INPUTS),
            context="cvdump-analysis-v1",
        )
        if use_cache
        else ""
    )
    if not orig_addrs and not recomp_addrs:
        return (
            _load_full_cvdump(target.recompiled_pdb, cache, pdb_fingerprint),
            pdb_fingerprint,
            "full",
        )

    cached_full: CvdumpParser | None = cache.load("cvdump-full", pdb_fingerprint)
    if cached_full is not None:
        logger.debug("Reusing cached full cvdump analysis for targeted comparison")
        return cached_full, pdb_fingerprint, "full"

    cvdump = _load_base_cvdump(target.recompiled_pdb, cache, pdb_fingerprint)
    symbols_by_address = codebase.symbols_for_offsets(orig_addrs)
    if any(not symbols_by_address.get(addr) for addr in orig_addrs):
        # An address without an annotation can only be paired by discovery
        # (body equivalence, unique call sites), which needs every symbol.
        logger.debug("Targeted address has no annotation; using full symbols")
        return (
            _load_full_cvdump(target.recompiled_pdb, cache, pdb_fingerprint),
            pdb_fingerprint,
            "full",
        )
    symbol_hints = {
        SymbolModuleHint(
            source_file=symbol.filename,
            lookup_name=symbol.name if symbol.is_nameref() else None,
        )
        for symbols in symbols_by_address.values()
        for symbol in symbols
    }
    selection = select_modules(cvdump, recomp_bin, symbol_hints, recomp_addrs)
    if selection.requires_full_symbols:
        logger.debug(
            "Targeted cvdump module selection was ambiguous; using full symbols"
        )
        return (
            _load_full_cvdump(target.recompiled_pdb, cache, pdb_fingerprint),
            pdb_fingerprint,
            "full",
        )

    logger.debug(
        "Targeted cvdump modules: %s",
        ", ".join(str(value) for value in sorted(selection.module_ids)) or "none",
    )
    for module_id in sorted(selection.module_ids):
        merge_module_symbols(
            cvdump,
            _load_module_cvdump(
                target.recompiled_pdb, module_id, cache, pdb_fingerprint
            ),
        )
    symbol_scope = "modules:" + ",".join(
        str(value) for value in sorted(selection.module_ids)
    )
    return cvdump, pdb_fingerprint, symbol_scope


def load_target_analysis(
    target: RecCmpTarget,
    *,
    orig_addrs: Iterable[int] = (),
    recomp_addrs: Iterable[int] = (),
    use_cache: bool = True,
    source_index: SourceIndex | None = None,
) -> LoadedTargetAnalysis:
    """Load fresh binaries plus cached deterministic analysis inputs."""
    if source_index is None:
        source_index = require_source_index(target)
    orig_addrs = tuple(orig_addrs)
    recomp_addrs = tuple(recomp_addrs)
    orig_bin = detect_image(filepath=target.original_path)
    recomp_bin = detect_image(filepath=target.recompiled_path)

    code_files = list(
        TextFile.from_files(
            source_code_search(target.source_paths),
            allow_error=True,
            encoding=target.encoding or "utf-8",
        )
    )
    cache = AnalysisCache(
        target.recompiled_pdb.parent / ".reccmp-cache", enabled=use_cache
    )
    codebase, marker_fingerprint = _load_source_markers(
        target, code_files, source_index, use_cache=use_cache
    )

    logger.info("Parsing %s ...", target.recompiled_pdb)
    cvdump, pdb_fingerprint, symbol_scope = _load_cvdump(
        target,
        recomp_bin,
        codebase,
        orig_addrs,
        recomp_addrs,
        cache,
        use_cache=use_cache,
    )
    pdb_file = CvdumpAnalysis(cvdump)

    data_sources = list(
        TextFile.from_files(
            target.data_sources,
            allow_error=True,
            encoding=target.encoding or "utf-8",
        )
    )
    equivalence_sources = list(
        TextFile.from_files(
            target.equivalence_groups,
            allow_error=True,
            encoding=target.encoding or "utf-8",
        )
    )
    prepared_fingerprint = ""
    if use_cache:
        data_fingerprint = fingerprint_text_files(
            data_sources, context="data-sources-v1"
        )
        equivalence_fingerprint = fingerprint_text_files(
            equivalence_sources, context="equivalence-groups-v1"
        )
        prepared_context = json.dumps(
            {
                "target": target.target_id,
                "encoding": target.encoding,
                "aliases": target.marker_aliases,
                "markers": marker_fingerprint,
                "pdb": pdb_fingerprint,
                "data": data_fingerprint,
                "data_order": [str(source.path) for source in data_sources],
                "equivalence": equivalence_fingerprint,
                "equivalence_order": [
                    str(source.path) for source in equivalence_sources
                ],
                "symbols": symbol_scope,
            },
            sort_keys=True,
        )
        prepared_fingerprint = fingerprint_files(
            (target.original_path, target.recompiled_path, *_COMPARE_CACHE_INPUTS),
            context=prepared_context,
        )

    return LoadedTargetAnalysis(
        orig_bin=orig_bin,
        recomp_bin=recomp_bin,
        pdb_file=pdb_file,
        code_files=code_files,
        data_sources=data_sources,
        equivalence_sources=equivalence_sources,
        project_aliases={target.target_id: target.marker_aliases},
        codebase=codebase,
        source_index=source_index,
        cache=cache,
        prepared_cache_name=(
            "prepared-full" if symbol_scope == "full" else "prepared-targeted"
        ),
        prepared_fingerprint=prepared_fingerprint,
    )
