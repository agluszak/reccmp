"""Native per-TU source collection against a pinned LLVM 19 indexer."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Iterator, Mapping, Sequence
import uuid

from reccmp.parser.marker import ProjectAliases
from .index import (
    RecordPool,
    SourceIndex,
    SourceIndexError,
    TranslationUnitRecords,
    record_command,
    relative_unit_id,
)

logger = logging.getLogger(__name__)

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

    def current() -> bool:
        return binary.is_file() and stamp.is_file() and stamp.read_text() == digest

    if not current():
        # The only lock in collection: concurrent collectors wait for one
        # build instead of each compiling the collector.
        with _locked(cache / "indexer.lock"):
            if not current():
                building = cache / f".indexer.{os.getpid()}"
                _build_indexer(building)
                os.replace(building, binary)
                _publish(stamp, digest)
    return binary


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on ``path`` for the duration of the block."""
    with path.open("a+b") as handle:
        _lock_exclusive(handle)
        yield


def _publish(path: Path, content: str) -> None:
    """Write ``path`` atomically: readers see the old file or the new one."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class DigestCache:
    """sha256 of files, reused across runs while a file's size, mtime,
    inode and ctime are unchanged. ``paranoid`` rehashes every file, for
    tools that restore timestamps after editing."""

    def __init__(self, path: Path, *, paranoid: bool = False):
        self.path = path
        self.paranoid = paranoid
        self._entries: dict[str, list] = {}
        self._dirty = False
        try:
            self._entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def digest(self, path: Path) -> str | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(path)
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns]
        entry = self._entries.get(key)
        if not self.paranoid and entry is not None and entry[:4] == signature:
            return entry[4]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._entries[key] = [*signature, digest]
        self._dirty = True
        return digest

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            merged = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            merged = {}
        merged.update(self._entries)
        _publish(self.path, json.dumps(merged, separators=(",", ":")))
        self._dirty = False


def indexer_identity(indexer: Path, digests: DigestCache) -> str:
    """The collector binary plus the Clang libraries it runs against."""
    version = _run([str(indexer), "--version"]).stdout
    return hashlib.sha256(f"{digests.digest(indexer)}\0{version}".encode()).hexdigest()


@dataclass
class _Unit:
    """One wanted compile-database entry and what the cache knows of it."""

    entry: dict
    unit_id: str
    identity: str
    command_digest: str
    main_digest: str
    miss: str | None = None  # why it must be indexed, or None on a hit
    records: TranslationUnitRecords | None = None


@dataclass
class CollectionProfile:
    """Where one collection's time went; written to ``profile.json``."""

    phases: dict[str, float] = field(default_factory=dict)
    misses: dict[str, str] = field(default_factory=dict)  # unit id -> reason
    units: dict[str, dict] = field(default_factory=dict)  # fresh unit id -> profile
    hits: int = 0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - start

    def to_dict(self) -> dict:
        totals: dict[str, float] = {}
        records: Counter[str] = Counter()
        sizes: Counter[str] = Counter()
        for unit in self.units.values():
            for key, value in unit.items():
                if key.endswith("_ms"):
                    totals[key] = totals.get(key, 0.0) + value
            records.update(unit.get("records", {}))
            sizes.update(unit.get("bytes", {}))
        slowest = sorted(
            self.units.items(),
            key=lambda item: item[1].get("frontend_ms", 0.0),
            reverse=True,
        )[:20]
        return {
            "phases_s": {key: round(value, 3) for key, value in self.phases.items()},
            "units": {"hits": self.hits, "misses": len(self.misses)},
            "miss_reasons": dict(
                Counter(reason.split(":", 1)[0] for reason in self.misses.values())
            ),
            "misses": self.misses,
            "indexer_totals_ms": {
                key: round(value, 1) for key, value in totals.items()
            },
            "records": dict(records),
            "bytes": dict(sizes),
            "slowest_units": [
                {
                    "unit": unit_id,
                    **{k: v for k, v in data.items() if k.endswith("_ms")},
                }
                for unit_id, data in slowest
            ],
        }

    def summary(self) -> str:
        phases = ", ".join(f"{key} {value:.1f}s" for key, value in self.phases.items())
        return f"source index: {self.hits} cached, {len(self.misses)} indexed; {phases}"


class _Workers:
    """Persistent ``indexer --serve`` processes fed one job at a time, so
    LLVM starts once per worker and no worker idles while jobs remain."""

    def __init__(self, indexer: Path, count: int, environment: dict[str, str]):
        self.indexer = indexer
        self.count = count
        self.environment = environment

    def run(self, jobs: Sequence[dict]) -> dict[str, str]:
        """Run every job; returns diagnostics for each output that failed."""
        pending: queue.Queue[dict] = queue.Queue()
        for job in jobs:
            pending.put(job)
        failures: dict[str, str] = {}
        lock = threading.Lock()

        def work() -> None:
            process: subprocess.Popen | None = None
            try:
                while True:
                    try:
                        job = pending.get_nowait()
                    except queue.Empty:
                        return
                    if process is None:
                        process = self._start()
                    reply = self._ask(process, job)
                    if reply is None:
                        # The worker died on this unit: report it, start afresh.
                        detail = self._death(process)
                        process = None
                        reply = {
                            "output": job["output"],
                            "ok": False,
                            "diagnostics": detail,
                        }
                    if not reply.get("ok"):
                        with lock:
                            failures[job["output"]] = str(reply.get("diagnostics", ""))
            finally:
                if process is not None:
                    self._stop(process)

        threads = [threading.Thread(target=work) for _ in range(max(1, self.count))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return failures

    def _start(self) -> subprocess.Popen:
        return subprocess.Popen(
            [str(self.indexer), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=tempfile.TemporaryFile(),
            text=True,
            env=self.environment,
        )

    @staticmethod
    def _ask(process: subprocess.Popen, job: dict) -> dict | None:
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(job) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        line = process.stdout.readline()
        return json.loads(line) if line else None

    @staticmethod
    def _death(process: subprocess.Popen) -> str:
        code = process.wait()
        stderr = process.stderr
        text = ""
        if stderr is not None and hasattr(stderr, "seek"):
            stderr.seek(0)  # type: ignore[union-attr]
            text = stderr.read().decode(errors="replace")  # type: ignore[union-attr]
        return f"indexer exited with status {code}\n{text[-4000:]}"

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        assert process.stdin is not None
        try:
            process.stdin.close()
        except OSError:
            pass
        process.wait()


def _command_digest(entry: dict, clang: str | None) -> str:
    """The compile command, without the indexer's own path (a host path and
    a container path for the same collector must share artifacts)."""
    return hashlib.sha256(
        shlex.join(record_command(entry, "indexer", clang)[1:]).encode()
    ).hexdigest()


class _TuCache:
    """Content-addressed per-TU artifacts. Several processes may use one
    cache: artifacts are published atomically and never locked."""

    def __init__(self, root: Path, digests: DigestCache, force: bool):
        self.tu = root / "tu"
        self.units = root / "units"
        self.tu.mkdir(parents=True, exist_ok=True)
        self.units.mkdir(parents=True, exist_ok=True)
        self.digests = digests
        self.force = force

    def paths(self, identity: str) -> tuple[Path, Path]:
        return self.tu / f"{identity}.ndjson", self.tu / f"{identity}.json"

    def _pointer(self, unit_id: str) -> Path:
        return self.units / f"{hashlib.sha256(unit_id.encode()).hexdigest()}.json"

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def miss_reason(self, unit: _Unit, indexer_digest: str) -> str | None:
        """Why ``unit`` must be indexed again, or None when its artifact is
        current."""
        # pylint: disable=too-many-return-statements
        if self.force:
            return "forced"
        ndjson, meta_path = self.paths(unit.identity)
        meta = self._read_json(meta_path)
        if meta is not None and ndjson.is_file():
            for dependency, digest in meta.get("deps", {}).items():
                if self.digests.digest(Path(dependency)) != digest:
                    return f"dependency_changed:{dependency}"
            return None
        pointer = self._read_json(self._pointer(unit.unit_id))
        previous = (
            self._read_json(self.paths(pointer["identity"])[1])
            if pointer is not None
            else None
        )
        if previous is None:
            return "new"
        if previous.get("indexer_digest") != indexer_digest:
            return "indexer_changed"
        if previous.get("command_digest") != unit.command_digest:
            return "command_changed"
        return "main_file_changed"

    def publish(
        self, unit: _Unit, output: Path, indexer_digest: str, pool: RecordPool
    ) -> TranslationUnitRecords:
        records = TranslationUnitRecords.load(output, unit.unit_id, pool)
        deps = {
            dependency: digest
            for dependency in sorted(set(records.dependencies))
            if (digest := self.digests.digest(Path(dependency))) is not None
        }
        ndjson, meta_path = self.paths(unit.identity)
        os.replace(output, ndjson)
        _publish(
            meta_path,
            json.dumps(
                {
                    "unit_id": unit.unit_id,
                    "indexer_digest": indexer_digest,
                    "command_digest": unit.command_digest,
                    "main_digest": unit.main_digest,
                    "deps": deps,
                }
            ),
        )
        _publish(self._pointer(unit.unit_id), json.dumps({"identity": unit.identity}))
        return records


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
    paranoid: bool = False,
) -> SourceIndex:
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    """Index wanted TUs natively and derive the multi-target SourceIndex.

    Only the expensive Clang NDJSON artifacts are cached. Python merge code,
    aliases, and target membership do not invalidate those artifacts. Where
    the time went is written to ``profile.json`` in the cache directory.
    """
    profile = CollectionProfile()
    repository = repository.resolve()
    cache = (cache_dir or repository / "build/reccmp-source").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if jobs is not None and jobs < 1:
        raise ValueError("source index jobs must be positive")
    database = json.loads(compilation_database.read_text(encoding="utf-8"))
    digests = DigestCache(cache / "digests.json", paranoid=paranoid)

    with profile.phase("resolve_indexer"):
        indexer = resolve_indexer(cache)
        indexer_digest = indexer_identity(indexer, digests)

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

    tu_cache = _TuCache(cache, digests, force)
    units: list[_Unit] = []
    with profile.phase("validate_cache"):
        for entry in wanted:
            main_digest = digests.digest(Path(entry["file"])) or ""
            command_digest = _command_digest(entry, clang)
            identity = hashlib.sha256(
                f"{indexer_digest}\0{command_digest}\0{main_digest}".encode()
            ).hexdigest()
            unit = _Unit(
                entry,
                relative_unit_id(repository, entry["file"]),
                identity,
                command_digest,
                main_digest,
            )
            unit.miss = tu_cache.miss_reason(unit, indexer_digest)
            units.append(unit)

    fresh = [unit for unit in units if unit.miss is not None]
    profile.hits = len(units) - len(fresh)
    profile.misses = {unit.unit_id: unit.miss for unit in fresh if unit.miss}

    with profile.phase("index"):
        outputs = {
            unit.identity: tu_cache.tu
            / f".tmp-{unit.identity}.{os.getpid()}.{uuid.uuid4().hex}.ndjson"
            for unit in fresh
        }
        failures = _Workers(
            indexer,
            min(len(fresh), jobs or os.cpu_count() or 1),
            {**os.environ, "RECCMP_SOURCE_ROOT": str(repository)},
        ).run(
            [
                {
                    "directory": unit.entry["directory"],
                    "output": str(outputs[unit.identity]),
                    "arguments": record_command(unit.entry, str(indexer), clang)[1:],
                }
                for unit in fresh
            ]
        )
        if failures:
            for output in outputs.values():
                output.unlink(missing_ok=True)
            failed = [
                f"{unit.entry['file']}: {failures[str(outputs[unit.identity])]}".rstrip()
                for unit in fresh
                if str(outputs[unit.identity]) in failures
            ]
            raise SourceIndexError("the source indexer failed on " + "\n".join(failed))

    pool = RecordPool()
    with profile.phase("load"):
        for unit in fresh:
            unit.records = tu_cache.publish(
                unit, outputs[unit.identity], indexer_digest, pool
            )
            if unit.records.profile:
                profile.units[unit.unit_id] = unit.records.profile
        for unit in units:
            if unit.records is None:
                unit.records = TranslationUnitRecords.load(
                    tu_cache.paths(unit.identity)[0], unit.unit_id, pool
                )
    digests.save()

    records = [unit.records for unit in units if unit.records is not None]
    present = {unit.unit_id for unit in records}
    with profile.phase("derive"):
        result = SourceIndex.from_units(
            records,
            {
                target: {relative_unit_id(repository, path) for path in paths} & present
                for target, paths in targets.items()
            },
            aliases=aliases,
            source_digests={
                relative_unit_id(repository, path): digest
                for paths in targets.values()
                for path in paths
                if (digest := digests.digest(path)) is not None
            },
            repository=repository,
        )
    with profile.phase("write"):
        result.write(cache / "source-index.json")
    digests.save()
    _publish(cache / "profile.json", json.dumps(profile.to_dict(), indent=1))
    logger.info("%s", profile.summary())
    return result
