"""Parallel direct-record collection for native and container compile databases."""

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
from .index import SourceCollector, SourceIndex, SourceIndexError, ast_command
from .variables import SourceConflict, SourceVariable

_SOURCE = Path(__file__).with_name("indexer.cpp")
# The LLVM development tree in the collector image. Debian renames the shared
# objects across releases (bookworm's `libclang-cpp.so.14` became trixie's
# `libclang-cpp.so.19.1`), so the build resolves the actual files below
# instead of guessing SONAMEs. The default tracks Debian stable;
# RECCMP_LLVM_VERSION selects another `/usr/lib/llvm-<version>` tree.
_LLVM_VERSION = os.environ.get("RECCMP_LLVM_VERSION", "19")
_COMPILE = (
    "clang++ -O2 -std=c++17 -fno-rtti -fno-exceptions"
    " -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS"
    " -I{include} {source} -o {output}"
    " {clang_cpp} {llvm}"
)


def _pick_library(candidates: list[str]) -> str | None:
    """Prefer the versioned LLVM tree; otherwise the matching multiarch SONAME."""
    trees = sorted(path for path in candidates if f"/llvm-{_LLVM_VERSION}/" in path)
    if trees:
        return trees[-1]
    versioned = sorted(
        path for path in candidates if f".so.{_LLVM_VERSION}" in path or f"-{_LLVM_VERSION}." in path
    )
    if versioned:
        return versioned[-1]
    multiarch = sorted(candidates)
    return multiarch[-1] if multiarch else None


def _collector_libraries(container_image: str | None) -> tuple[str, str, str]:
    """Locate the headers and shared objects the collector builds against."""
    default_include = f"/usr/lib/llvm-{_LLVM_VERSION}/include"
    patterns = (
        f"/usr/lib/llvm-{_LLVM_VERSION}/lib/libclang-cpp.so.*",
        "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.*",
        f"/usr/lib/llvm-{_LLVM_VERSION}/lib/libLLVM*.so*",
        "/usr/lib/x86_64-linux-gnu/libLLVM*.so*",
    )
    if container_image is None:
        include = default_include
        config = shutil.which(f"llvm-config-{_LLVM_VERSION}")
        if config:
            probed = subprocess.run(
                [config, "--includedir"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probed.returncode == 0 and probed.stdout.strip():
                include = probed.stdout.strip()
        hits = [
            match
            for pattern in patterns
            for match in glob.glob(pattern)
            if os.path.isfile(match)
        ]
    else:
        # Expand each glob independently so a missing pattern does not make
        # `ls` fail before the others are inspected. Prefer llvm-config when
        # the image ships it for this LLVM version.
        probe = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/bin/sh",
                container_image,
                "-c",
                (
                    f'include=$(llvm-config-{_LLVM_VERSION} --includedir 2>/dev/null || true); '
                    f'libdir=$(llvm-config-{_LLVM_VERSION} --libdir 2>/dev/null || true); '
                    f'printf "INCLUDE:%s\\n" "${{include:-{default_include}}}"; '
                    f'if [ -n "$libdir" ]; then '
                    f'ls -d "$libdir"/libclang-cpp.so.* "$libdir"/libLLVM*.so* 2>/dev/null || true; '
                    f"fi; "
                    + "".join(
                        f"ls -d {pattern} 2>/dev/null || true; " for pattern in patterns
                    )
                ),
            ]
        )
        include = default_include
        hits = []
        for line in probe.stdout.splitlines():
            if line.startswith("INCLUDE:"):
                include = line.partition(":")[2] or default_include
            else:
                hits.extend(part for part in line.split() if ".so" in part)
    clang_cpp = _pick_library([hit for hit in hits if "libclang-cpp" in hit])
    llvm = _pick_library(
        [hit for hit in hits if "libclang-cpp" not in hit and "libLLVM" in hit]
    )
    if clang_cpp is None or llvm is None:
        raise SourceIndexError(
            f"the collector image has no LLVM {_LLVM_VERSION} development "
            f"libraries (probed {', '.join(patterns)})"
        )
    return include, clang_cpp, llvm


def record_command(entry: dict, indexer: str, clang: str | None = None) -> list[str]:
    """Replay build arguments, replacing the whole-AST dump with direct records."""
    arguments = ast_command(entry, clang, ())
    position = arguments.index("-ast-dump=json")
    del arguments[position - 1 : position + 1]
    return [indexer, *arguments]


def _run(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, **kwargs
    )
    if result.returncode:
        raise SourceIndexError(
            f"source index command failed ({result.returncode}): {shlex.join(command)}\n{result.stderr}"
        )
    return result


def _lock_exclusive(lock_file) -> None:
    """Advisory exclusive lock; no-op where fcntl is unavailable (Windows)."""
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(lock_file, fcntl.LOCK_EX)


# Execution options describe the compile environment directly, without a plugin layer.
# pylint: disable=too-many-arguments,too-many-locals
def collect_compile_database(
    repository: Path,
    compilation_database: Path,
    targets: Mapping[str, Sequence[Path]],
    *,
    clang: str | None,
    jobs: int | None,
    container_image: str | None,
    mounts: Mapping[Path, str] | None,
    compilation_root: Path | None,
    cache_dir: Path | None,
    force: bool,
    aliases: ProjectAliases | None,
) -> SourceIndex:
    repository = repository.resolve()
    mounts = mounts or {}
    root = str(compilation_root or repository)
    cache = (cache_dir or repository / "build/reccmp-source").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Serialize builders sharing a cache so a cancelled or concurrent rebuild
    # cannot expose a partially-written executable or JSON projection.
    with (cache / "lock").open("a+b") as lock:
        _lock_exclusive(lock)
        database = json.loads(compilation_database.read_text(encoding="utf-8"))
        if jobs is not None and jobs < 1:
            raise ValueError("source index jobs must be positive")
        parallelism = min(len(database), jobs or os.cpu_count() or 1) or 1
        if container_image:
            compiler_identity = _run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", container_image]
            ).stdout.strip()
        else:
            compiler_identity = _run(["clang++", "--version"]).stdout
        binary_digest = hashlib.sha256(
            (_COMPILE + _LLVM_VERSION + compiler_identity).encode()
            + _SOURCE.read_bytes()
        ).hexdigest()

        binary = cache / "indexer"
        binary_stamp = cache / "indexer.sha256"
        container = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--env",
            "TMPDIR=/tmp",
        ]
        if (
            not binary.is_file()
            or not binary_stamp.is_file()
            or binary_stamp.read_text() != binary_digest
        ):
            include, clang_cpp, llvm = _collector_libraries(container_image)
            compile_command = _COMPILE.format(
                include=include,
                clang_cpp=clang_cpp,
                llvm=llvm,
                source="{source}",
                output="{output}",
            )
            if container_image:
                _run(
                    [
                        *container,
                        "--volume",
                        f"{_SOURCE.parent}:/reccmp-source:ro",
                        "--volume",
                        f"{cache}:/reccmp-cache",
                        "--entrypoint",
                        "/bin/sh",
                        container_image,
                        "-c",
                        compile_command.format(
                            source="/reccmp-source/indexer.cpp",
                            output="/reccmp-cache/indexer",
                        ),
                    ]
                )
            else:
                _run(
                    shlex.split(
                        compile_command.format(
                            source=shlex.quote(str(_SOURCE)),
                            output=shlex.quote(str(binary)),
                        )
                    )
                )
            binary_stamp.write_text(binary_digest, encoding="utf-8")

        collector = SourceCollector(repository, Path(root))
        # Per-translation-unit cache: command + source + container/compiler
        # identity + compiler-reported dependency hashes. Validated TU
        # artifacts are aggregated into the final index; there is no separate
        # whole-tree content fingerprint for agents to maintain.
        implementation = [
            *_SOURCE.parent.glob("*.py"),
            *(_SOURCE.parents[1] / "parser").glob("*.py"),
        ]
        environment = hashlib.sha256()
        environment.update(
            json.dumps(
                (
                    root,
                    clang,
                    binary_digest,
                    {str(path): value for path, value in mounts.items()},
                    aliases,
                    {key: sorted(str(path) for path in paths) for key, paths in targets.items()},
                ),
                sort_keys=True,
            ).encode()
        )
        for path in sorted(implementation):
            environment.update(path.read_bytes())
        environment_digest = environment.hexdigest()

        def host_path(guest: str) -> Path | None:
            if not mounts:
                path = Path(guest)
                return path if path.is_file() else None
            for host, mount_guest in sorted(
                mounts.items(), key=lambda item: len(item[1]), reverse=True
            ):
                trimmed = mount_guest.rstrip("/")
                if guest == trimmed:
                    return host
                if guest.startswith(trimmed + "/"):
                    return host / guest[len(trimmed) + 1 :]
            return None

        def dependency_host_path(raw: str) -> Path | None:
            """Map a compiler-reported dependency onto the host filesystem.

            Dependencies from the container/toolchain itself are covered by
            container image identity, so they are not hashed on the host.
            """
            mapped = host_path(raw)
            if mapped is not None:
                return mapped
            return None

        file_digests: dict[Path, bytes] = {}

        def file_digest(path: Path) -> bytes:
            cached = file_digests.get(path)
            if cached is not None:
                return cached
            digest = path.read_bytes()
            file_digests[path] = digest
            return digest

        tu_cache = cache / "tu"

        def deps_digest(deps: list[str]) -> str | None:
            digest = hashlib.sha256()
            for raw in sorted(deps):
                path = dependency_host_path(raw)
                if path is None:
                    continue
                if not path.is_file():
                    return None
                digest.update(
                    str(path).encode() + b"\0" + file_digest(path) + b"\0"
                )
            return digest.hexdigest()

        def identity_of(entry: dict) -> str | None:
            host = host_path(str(entry.get("file", "")))
            if host is None or not host.is_file():
                return None
            digest = hashlib.sha256()
            digest.update(environment_digest.encode() + b"\0")
            digest.update(
                shlex.join(
                    record_command(entry, "/reccmp-cache/indexer", clang)
                ).encode()
                + b"\0"
            )
            digest.update(file_digest(host))
            return digest.hexdigest()

        def cached_records(identity: str | None) -> str | None:
            if not identity:
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
            return ndjson_path.read_text(encoding="utf-8")

        # Which translation units belong to each link namespace: the TU's main
        # file is listed among that target's source paths (marker ownership).
        target_units: dict[str, set[str]] = {}
        for target, paths in targets.items():
            target_units[target] = {
                collector.unit_id_for(path)
                for path in paths
                if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".c++"}
                or path.name.endswith((".C", ".CC", ".CPP"))
            }
            # Also accept any listed path as a potential main file — projects
            # may mark ownership with exact compile-entry paths.
            target_units[target].update(collector.unit_id_for(path) for path in paths)

        with tempfile.TemporaryDirectory(prefix="batch-", dir=cache) as raw:
            scratch = Path(raw)
            identities: dict[int, str | None] = {}
            fresh: list[int] = []
            unit_ids_by_index: dict[int, str] = {}
            for index, entry in enumerate(database):
                unit_ids_by_index[index] = collector.unit_id_for(entry["file"])
                identity = None if force else identity_of(entry)
                identities[index] = identity
                records = cached_records(identity)
                if records is None:
                    fresh.append(index)
                    continue
                (scratch / f"{index:05d}.ndjson").write_text(records, encoding="utf-8")
                (scratch / f"{index:05d}.status").write_text("0", encoding="utf-8")
            if container_image and fresh:
                for index in fresh:
                    entry = database[index]
                    command = shlex.join(
                        record_command(entry, "/reccmp-cache/indexer", clang)
                    )
                    (scratch / f"{index:05d}.sh").write_text(
                        f"cd {shlex.quote(entry['directory'])} || exit\n"
                        f"{command} > /reccmp-batch/{index:05d}.ndjson 2> /reccmp-batch/{index:05d}.err\n"
                        f"echo $? > /reccmp-batch/{index:05d}.status\n",
                        encoding="utf-8",
                    )
                volumes = [
                    arg
                    for host, guest in mounts.items()
                    for arg in ("--volume", f"{host}:{guest}:ro")
                ]
                _run(
                    [
                        *container,
                        *volumes,
                        "--env",
                        f"RECCMP_SOURCE_ROOT={root}",
                        "--volume",
                        f"{cache}:/reccmp-cache:ro",
                        "--volume",
                        f"{scratch}:/reccmp-batch",
                        "--entrypoint",
                        "/bin/sh",
                        container_image,
                        "-c",
                        f"printf '%s\\n' /reccmp-batch/*.sh | xargs -P {parallelism} -n 1 /bin/sh",
                    ]
                )
            elif fresh:

                def emit(index: int) -> None:
                    entry = database[index]
                    with (
                        (scratch / f"{index:05d}.ndjson").open("w") as out,
                        (scratch / f"{index:05d}.err").open("w") as err,
                    ):
                        result = subprocess.run(
                            record_command(entry, str(binary), clang),
                            cwd=entry["directory"],
                            stdout=out,
                            stderr=err,
                            env={**os.environ, "RECCMP_SOURCE_ROOT": root},
                            check=False,
                        )
                    (scratch / f"{index:05d}.status").write_text(str(result.returncode))

                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    list(executor.map(emit, fresh))
            for index, entry in enumerate(database):
                status = scratch / f"{index:05d}.status"
                if not status.is_file() or status.read_text().strip() != "0":
                    error = scratch / f"{index:05d}.err"
                    detail = (
                        error.read_text(errors="replace")
                        if error.is_file()
                        else "no compiler result"
                    )
                    raise SourceIndexError(
                        f"the source indexer failed on {entry['file']}: {detail}"
                    )
                text = (scratch / f"{index:05d}.ndjson").read_text(encoding="utf-8")
                if index in fresh and identities.get(index):
                    deps: list[str] = []
                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if record.get("record") == "dependency":
                            deps = [str(path) for path in record.get("files") or []]
                            break
                    digest = deps_digest(deps)
                    if digest is not None:
                        tu_cache.mkdir(parents=True, exist_ok=True)
                        (tu_cache / f"{identities[index]}.ndjson").write_text(
                            text, encoding="utf-8"
                        )
                        (tu_cache / f"{identities[index]}.json").write_text(
                            json.dumps({"deps": deps, "deps_digest": digest}),
                            encoding="utf-8",
                        )
                unit_id = unit_ids_by_index[index]
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("record") == "dependency":
                        continue
                    collector.collect_record(record, unit_id=unit_id)
        indexes = [
            SourceIndex.from_collector(
                repository,
                target,
                paths,
                collector,
                unit_ids=target_units[target],
                aliases=aliases,
            )
            for target, paths in targets.items()
        ]
        variables: dict[tuple[str | None, str], SourceVariable] = {}
        for part in indexes:
            for item in part.variables:
                variables.setdefault((item.target, item.semantic_id), item)
        conflicts: dict[tuple[str | None, str, str], SourceConflict] = {}
        for part in indexes:
            for item in part.conflicts:
                conflicts.setdefault(
                    (item.target, item.record_kind, item.semantic_id), item
                )
        # Classes are namespaced by target: the same header under different
        # macros can produce distinct layouts per binary.
        classes: dict[tuple[str | None, str], object] = {}
        for part in indexes:
            for item in part.classes:
                classes.setdefault((item.target, item.semantic_id), item)
        declarations: dict[tuple[str | None, tuple[str, ...]], object] = {}
        for part in indexes:
            for item in part.declarations:
                declarations.setdefault(
                    (item.target, item.merge_key), item
                )
        result_index = SourceIndex(
            declarations=declarations.values(),
            classes=classes.values(),
            markers=(item for part in indexes for item in part.markers),
            variables=variables.values(),
            conflicts=conflicts.values(),
        )
        result_index.write(cache / "source-index.json")
        return result_index
