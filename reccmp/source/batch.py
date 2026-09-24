"""Native per-TU source collection against a pinned LLVM 19 indexer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

from reccmp.parser.marker import ProjectAliases
from .index import (
    SourceAbi,
    SourceClass,
    SourceDeclaration,
    SourceIndex,
    SourceIndexError,
    _member_use_key,
    TranslationUnitRecords,
    record_command,
    relative_unit_id,
    source_digest,
)
from .variables import SourceConflict, SourceVariable

_SOURCE = Path(__file__).with_name("indexer.cpp")
_LLVM_VERSION = "19"
_COMPILE = (
    "clang++ -O2 -std=c++17 -fno-rtti -fno-exceptions"
    " -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS"
    " -I{include} {source} -o {output}"
    " {clang_cpp} {llvm}"
)


def _run(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, **kwargs
    )
    if result.returncode:
        raise SourceIndexError(
            f"source index command failed ({result.returncode}): "
            f"{shlex.join(command)}\n{result.stderr}"
        )
    return result


def _lock_exclusive(lock_file) -> None:
    try:
        import fcntl  # pylint: disable=import-outside-toplevel  # POSIX only
    except ImportError:
        return
    fcntl.flock(lock_file, fcntl.LOCK_EX)


def _pick_library(candidates: list[str]) -> str | None:
    trees = sorted(path for path in candidates if f"/llvm-{_LLVM_VERSION}/" in path)
    if trees:
        return trees[-1]
    versioned = sorted(
        path
        for path in candidates
        if f".so.{_LLVM_VERSION}" in path or f"-{_LLVM_VERSION}." in path
    )
    if versioned:
        return versioned[-1]
    return sorted(candidates)[-1] if candidates else None


def _build_indexer(binary: Path) -> None:
    """Compile the collector once when no prebuilt binary is configured."""
    config = shutil.which(f"llvm-config-{_LLVM_VERSION}")
    include = f"/usr/lib/llvm-{_LLVM_VERSION}/include"
    if config:
        probed = subprocess.run(
            [config, "--includedir"], capture_output=True, text=True, check=False
        )
        if probed.returncode == 0 and probed.stdout.strip():
            include = probed.stdout.strip()
    patterns = (
        f"/usr/lib/llvm-{_LLVM_VERSION}/lib/libclang-cpp.so.*",
        "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.*",
        f"/usr/lib/llvm-{_LLVM_VERSION}/lib/libLLVM*.so*",
        "/usr/lib/x86_64-linux-gnu/libLLVM*.so*",
    )
    hits = [
        match
        for pattern in patterns
        for match in glob.glob(pattern)
        if os.path.isfile(match)
    ]
    clang_cpp = _pick_library([hit for hit in hits if "libclang-cpp" in hit])
    llvm = _pick_library(
        [hit for hit in hits if "libclang-cpp" not in hit and "libLLVM" in hit]
    )
    if clang_cpp is None or llvm is None:
        raise SourceIndexError(
            "no LLVM 19 development libraries found; set RECCMP_SOURCE_INDEXER "
            "to a prebuilt reccmp-source-indexer, or install libclang-19-dev"
        )
    _run(
        shlex.split(
            _COMPILE.format(
                include=include,
                clang_cpp=clang_cpp,
                llvm=llvm,
                source=shlex.quote(str(_SOURCE)),
                output=shlex.quote(str(binary)),
            )
        )
    )


def resolve_indexer(cache: Path) -> Path:
    """Prefer a prebuilt indexer; otherwise build one into the cache."""
    configured = os.environ.get("RECCMP_SOURCE_INDEXER")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise SourceIndexError(f"RECCMP_SOURCE_INDEXER is not a file: {path}")
        return path
    which = shutil.which("reccmp-source-indexer")
    if which:
        return Path(which)
    binary = cache / "indexer"
    stamp = cache / "indexer.sha256"
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    if not binary.is_file() or not stamp.is_file() or stamp.read_text() != digest:
        _build_indexer(binary)
        stamp.write_text(digest, encoding="utf-8")
    return binary


def _file_digest_factory():
    digests: dict[Path, bytes] = {}

    def file_digest(path: Path) -> bytes:
        cached = digests.get(path)
        if cached is not None:
            return cached
        digest = hashlib.sha256(path.read_bytes()).digest()
        digests[path] = digest
        return digest

    return file_digest


# pylint: disable=too-many-arguments,too-many-locals
def collect_compile_database(
    repository: Path,
    compilation_database: Path,
    targets: Mapping[str, Sequence[Path]],
    *,
    clang: str | None,
    jobs: int | None,
    cache_dir: Path | None,
    force: bool,
    aliases: ProjectAliases | None,
) -> SourceIndex:
    # pylint: disable=too-many-statements
    """Index wanted TUs natively and derive the multi-target SourceIndex.

    Only the expensive Clang NDJSON artifacts are cached. Python merge code,
    aliases, and target membership do not invalidate those artifacts.
    """
    repository = repository.resolve()
    root = str(repository)
    cache = (cache_dir or repository / "build/reccmp-source").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    tu_cache = cache / "tu"
    tu_cache.mkdir(parents=True, exist_ok=True)

    with (cache / "lock").open("a+b") as lock:
        _lock_exclusive(lock)
        database = json.loads(compilation_database.read_text(encoding="utf-8"))
        if jobs is not None and jobs < 1:
            raise ValueError("source index jobs must be positive")

        indexer = resolve_indexer(cache)
        compiler_identity = _run(["clang++", "--version"]).stdout
        indexer_digest = hashlib.sha256(
            indexer.read_bytes() + compiler_identity.encode()
        ).hexdigest()

        owned_paths = {
            relative_unit_id(repository, path)
            for paths in targets.values()
            for path in paths
        }
        # Only compile-database entries whose main file is owned by some target.
        wanted = [
            entry
            for entry in database
            if relative_unit_id(repository, entry["file"]) in owned_paths
        ]
        if not wanted:
            raise SourceIndexError(
                "no compile-database entries match the requested target sources"
            )

        parallelism = min(len(wanted), jobs or os.cpu_count() or 1) or 1
        file_digest = _file_digest_factory()

        def deps_digest(deps: Sequence[str]) -> str | None:
            digest = hashlib.sha256()
            for raw in sorted(deps):
                path = Path(raw)
                if not path.is_file():
                    # Toolchain / missing paths are covered by indexer identity.
                    continue
                digest.update(str(path).encode() + b"\0" + file_digest(path) + b"\0")
            return digest.hexdigest()

        def identity_of(entry: dict) -> str:
            source = Path(entry["file"])
            digest = hashlib.sha256()
            digest.update(indexer_digest.encode() + b"\0")
            digest.update(
                shlex.join(record_command(entry, str(indexer), clang)).encode() + b"\0"
            )
            digest.update(file_digest(source))
            return digest.hexdigest()

        def cached_path(identity: str) -> Path | None:
            if force:
                return None
            meta_path = tu_cache / f"{identity}.json"
            ndjson_path = tu_cache / f"{identity}.ndjson"
            if not meta_path.is_file() or not ndjson_path.is_file():
                return None
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                return None
            if deps_digest(list(meta.get("deps") or [])) != meta.get("deps_digest"):
                return None
            return ndjson_path

        units_by_index: dict[int, TranslationUnitRecords] = {}
        fresh: list[tuple[int, dict, str, Path]] = []
        for index, entry in enumerate(wanted):
            identity = identity_of(entry)
            unit_id = relative_unit_id(repository, entry["file"])
            hit = cached_path(identity)
            if hit is not None:
                units_by_index[index] = TranslationUnitRecords.load(hit, unit_id)
                continue
            output = tu_cache / f".tmp-{identity}.ndjson"
            fresh.append((index, entry, identity, output))

        if fresh:
            # Chunk misses across long-lived indexer workers so LLVM init runs
            # once per worker rather than once per translation unit.
            chunks = [fresh[offset::parallelism] for offset in range(parallelism)]
            chunks = [chunk for chunk in chunks if chunk]

            def run_chunk(chunk: list[tuple[int, dict, str, Path]]) -> None:
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".jsonl",
                    dir=cache,
                    delete=False,
                    encoding="utf-8",
                ) as manifest:
                    for _, entry, _, output in chunk:
                        if output.is_file():
                            output.unlink()
                        manifest.write(
                            json.dumps(
                                {
                                    "directory": entry["directory"],
                                    "output": str(output),
                                    "arguments": record_command(
                                        entry, str(indexer), clang
                                    )[1:],
                                }
                            )
                            + "\n"
                        )
                    manifest_path = Path(manifest.name)
                try:
                    result = subprocess.run(
                        [str(indexer), "--batch", str(manifest_path)],
                        capture_output=True,
                        text=True,
                        env={**os.environ, "RECCMP_SOURCE_ROOT": root},
                        check=False,
                    )
                finally:
                    manifest_path.unlink(missing_ok=True)
                if result.returncode:
                    failed = [
                        entry["file"]
                        for _, entry, _, output in chunk
                        if not output.is_file()
                    ]
                    detail = result.stderr.strip() or "indexer batch failed"
                    raise SourceIndexError(
                        "the source indexer failed on "
                        f"{', '.join(failed) or 'unknown units'}: {detail}"
                    )

            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                list(executor.map(run_chunk, chunks))

            for index, entry, identity, output in fresh:
                if not output.is_file():
                    raise SourceIndexError(
                        f"the source indexer produced no output for {entry['file']}"
                    )
                unit = TranslationUnitRecords.load(
                    output, relative_unit_id(repository, entry["file"])
                )
                digest = deps_digest(unit.dependencies)
                final_ndjson = tu_cache / f"{identity}.ndjson"
                final_meta = tu_cache / f"{identity}.json"
                output.replace(final_ndjson)
                final_meta.write_text(
                    json.dumps({"deps": unit.dependencies, "deps_digest": digest}),
                    encoding="utf-8",
                )
                units_by_index[index] = unit

        units = [units_by_index[index] for index in range(len(wanted))]

        target_units = {
            target: {
                relative_unit_id(repository, path)
                for path in paths
                if relative_unit_id(repository, path)
                in {unit.unit_id for unit in units}
            }
            for target, paths in targets.items()
        }
        # Include every owned path so marker-only headers still join; TU
        # membership for derivation uses the intersection above plus any
        # compile-entry main files listed in the target.
        for target, paths in targets.items():
            target_units[target] |= {
                relative_unit_id(repository, path) for path in paths
            }

        indexes = [
            SourceIndex.from_units(
                target,
                units,
                unit_ids={
                    unit.unit_id
                    for unit in units
                    if unit.unit_id in target_units[target]
                },
                aliases=aliases,
            )
            for target, paths in targets.items()
        ]

        variables: dict[tuple[str | None, str], SourceVariable] = {}
        for part in indexes:
            for variable in part.variables:
                variables.setdefault((variable.target, variable.semantic_id), variable)
        conflicts: dict[tuple[str | None, str, str], SourceConflict] = {}
        for part in indexes:
            for conflict in part.conflicts:
                conflicts.setdefault(
                    (
                        conflict.target,
                        conflict.record_kind,
                        conflict.semantic_id,
                    ),
                    conflict,
                )
        classes: dict[tuple[str | None, str], SourceClass] = {}
        for part in indexes:
            for source_class in part.classes:
                classes.setdefault(
                    (source_class.target, source_class.semantic_id), source_class
                )
        declarations: dict[tuple[str | None, tuple[str, ...]], SourceDeclaration] = {}
        for part in indexes:
            for declaration in part.declarations:
                declarations.setdefault(
                    (declaration.target, declaration.merge_key), declaration
                )
        member_uses = {
            _member_use_key(item): item for part in indexes for item in part.member_uses
        }

        result = SourceIndex(
            declarations=declarations.values(),
            classes=classes.values(),
            markers=(item for part in indexes for item in part.markers),
            variables=variables.values(),
            member_uses=member_uses.values(),
            conflicts=conflicts.values(),
            abi=_merged_abi(indexes),
            target_abis={
                target: part.abi
                for (target, _paths), part in zip(targets.items(), indexes)
                if part.abi is not None
            },
            marker_blocks=(block for unit in units for block in unit.marker_blocks),
            source_digests={
                relative_unit_id(repository, path): source_digest(path)
                for paths in targets.values()
                for path in paths
            },
        )
        result.write(cache / "source-index.json")
        return result


def _merged_abi(indexes: Sequence[SourceIndex]) -> SourceAbi | None:
    """Shared ABI only when every target that recorded one agrees."""
    abis = [part.abi for part in indexes if part.abi is not None]
    if not abis:
        return None
    first = abis[0]
    if all(abi == first for abi in abis[1:]):
        return first
    return None
