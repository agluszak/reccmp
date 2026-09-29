"""Join reccmp markers to semantic declarations from the Clang AST.

The marker grammar (``reccmp.parser``) owns annotation syntax and addresses.
Clang owns C++ names, function and variable kinds, types, linkage, class
membership, inheritance, virtual declarations, and which declaration each
marker block annotates. This module assembles the index; ``source.derive``
selects link-namespace facts and ``source.markers`` binds marker blocks to
declarations and projects them for JSON output.

Compiler records arrive as per-TU observations. Link-namespace partitioning,
winner selection, and conflict derivation happen after collection — never by
globally collapsing bare ``semantic_id`` values first.
"""

from __future__ import annotations

# The optional execution backend imports this record model when first used.
# pylint: disable=cyclic-import

import hashlib
import json
from copy import copy
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

from reccmp.parser.marker import ProjectAliases
from reccmp.parser.reader import MarkerBlock, local_paths
from .variables import SourceConflict, SourceVariable
from .derive import _NamespaceRecords, derive_namespace
from .markers import merge_marker_blocks, _marker_projection, _join_markers
from .layout import SourceLayoutQueries
from .observations import (
    SourceIndexError,
    TranslationUnitRecords,
    _declaration_from_dict,
    _variable_from_dict,
    _conflict_from_dict,
    _class_from_dict,
)
from .records import (
    DeclarationKey,
    SourceDeclaration,
    SourceAbi,
    SourceClass,
    SourceMarker,
)


def relative_unit_id(
    repository: Path, main_file: str | Path, compilation_root: Path | None = None
) -> str:
    """Repo-relative identity of a translation unit's main file."""
    path = Path(main_file)
    if compilation_root is not None:
        try:
            path = repository / path.relative_to(compilation_root)
        except ValueError:
            pass
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


class _RepositoryPaths:
    """Repository-relative spellings of absolute paths, each resolved once:
    every unit lists the same headers."""

    def __init__(self, repository: Path):
        self.root = repository.resolve()
        self._known: dict[str, str | None] = {}

    def relative(self, raw: str) -> str | None:
        if raw not in self._known:
            try:
                self._known[raw] = Path(raw).resolve().relative_to(self.root).as_posix()
            except ValueError:
                self._known[raw] = None
        return self._known[raw]

    def files(self, paths: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted({path for raw in paths if (path := self.relative(raw)) is not None})
        )


def _plain(value: Any) -> Any:
    """JSON-ready form of a record (``dataclasses.asdict`` without its deep
    copies, which dominate writing the index)."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: _plain(item) for key, item in value.__dict__.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Test/fixture helper: accumulate observations across units without merging.
class SourceCollector:
    """Fixture helper that gathers ``TranslationUnitRecords`` by unit id."""

    def __init__(self, repository: Path, compilation_root: Path | None = None) -> None:
        self.repository = repository.resolve()
        self.compilation_root = compilation_root
        self.units: dict[str, TranslationUnitRecords] = {}

    def unit_id_for(self, main_file: str | Path) -> str:
        return relative_unit_id(self.repository, main_file, self.compilation_root)

    def collect_record(self, record: Mapping[str, Any], *, unit_id: str = "") -> None:
        self.units.setdefault(unit_id, TranslationUnitRecords(unit_id)).add(record)

    def collect_records(self, records: str, *, unit_id: str = "") -> None:
        for line in records.splitlines():
            if line.strip():
                self.collect_record(json.loads(line), unit_id=unit_id)

    @property
    def variables(self) -> list[SourceVariable]:
        return [item for unit in self.units.values() for item in unit.variables]

    def derive(
        self, *, target: str | None = None, unit_ids: set[str] | None = None
    ) -> _NamespaceRecords:
        return derive_namespace(
            tuple(self.units.values()), target=target, unit_ids=unit_ids
        )


def keyed(
    records: Iterable[Any], target: str | None = None
) -> dict[DeclarationKey, Any]:
    """Records of external entities (or classes) keyed in ``target``: for
    building an index directly."""
    return {DeclarationKey(target, item.semantic_id): item for item in records}


def _sorted_by_key(records: Mapping[DeclarationKey, Any]) -> dict[DeclarationKey, Any]:
    return {key: records[key] for key in sorted(records, key=DeclarationKey.sort_key)}


def _flattened(records: Mapping[DeclarationKey, Any]) -> list[dict[str, Any]]:
    """JSON rows: each record with its key's target and unit."""
    return [
        {**_plain(item), "target": key.target, "unit_id": key.unit_id}
        for key, item in records.items()
    ]


def _keyed(
    rows: Iterable[Mapping[str, Any]], parse, *, id_field: str = "semantic_id"
) -> dict[DeclarationKey, Any]:
    """Inverse of ``_flattened``."""
    records: dict[DeclarationKey, Any] = {}
    for row in rows:
        values = dict(row)
        target, unit_id = values.pop("target"), values.pop("unit_id")
        records[DeclarationKey(target, values[id_field], unit_id)] = parse(values)
    return records


def _unique_class_map(
    classes: Mapping[DeclarationKey, SourceClass],
    *,
    key,
) -> dict[str, SourceClass]:
    """Build an unscoped lookup that drops names colliding across targets."""
    by_key: dict[str, tuple[str | None, SourceClass]] = {}
    ambiguous: set[str] = set()
    for class_key, item in classes.items():
        map_key = key(item)
        if map_key in ambiguous:
            continue
        previous = by_key.get(map_key)
        if previous is None:
            by_key[map_key] = (class_key.target, item)
        elif previous[0] != class_key.target:
            del by_key[map_key]
            ambiguous.add(map_key)
    return {name: item for name, (_, item) in by_key.items()}


class SourceIndex(SourceLayoutQueries):
    """Canonical marker plus Clang semantic source index."""

    # pylint: disable=too-many-public-methods,too-many-instance-attributes

    def __init__(
        self,
        *,
        declarations: Mapping[DeclarationKey, SourceDeclaration],
        classes: Mapping[DeclarationKey, SourceClass],
        markers: Iterable[SourceMarker],
        variables: Mapping[DeclarationKey, SourceVariable] | None = None,
        conflicts: Iterable[SourceConflict] = (),
        abi: SourceAbi | None = None,
        target_abis: Mapping[str, SourceAbi] | None = None,
        marker_blocks: Iterable[MarkerBlock] = (),
        source_digests: Mapping[str, str] | None = None,
        unit_dependencies: Mapping[str, Iterable[str]] | None = None,
        document_digest: str | None = None,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-locals
        # Every marker block the compiler saw, for all targets: the marker
        # grammar picks out each target's markers when reading them.
        self.marker_blocks = merge_marker_blocks(marker_blocks)
        # sha256 of every target source file when the index was collected.
        self.source_digests: dict[str, str] = dict(
            sorted((source_digests or {}).items())
        )
        # Repository files each translation unit includes, by unit id.
        self.unit_dependencies: dict[str, tuple[str, ...]] = {
            unit: tuple(sorted(paths))
            for unit, paths in sorted((unit_dependencies or {}).items())
        }
        self.declarations = _sorted_by_key(declarations)
        self.classes = _sorted_by_key(classes)
        self.markers = tuple(
            sorted(markers, key=lambda item: (item.address, item.source_file))
        )
        self.variables = _sorted_by_key(variables or {})
        # The digest of the document this index was read from (see identity).
        self._document_digest = document_digest
        self.conflicts = tuple(sorted(conflicts, key=lambda item: item.semantic_id))
        self.abi = abi
        self.target_abis: dict[str, SourceAbi] = dict(target_abis or {})
        # Unscoped name/id maps only retain unambiguous entries. Cross-target
        # collisions must not silently pick a last-wins layout.
        self._classes_by_name = _unique_class_map(
            self.classes, key=lambda item: item.qualified_name
        )
        self._classes_by_semantic_id = _unique_class_map(
            self.classes, key=lambda item: item.semantic_id
        )
        self._classes_by_target_name: dict[tuple[str | None, str], SourceClass] = {
            (key.target, item.qualified_name): item
            for key, item in self.classes.items()
        }
        self._classes_by_target_semantic_id: dict[
            tuple[str | None, str], SourceClass
        ] = {(key.target, key.semantic_id): item for key, item in self.classes.items()}

    def targets(self) -> set[str]:
        """The targets any record belongs to."""
        keys = (*self.declarations, *self.classes, *self.variables)
        found = {key.target for key in keys} | {item.target for item in self.markers}
        return {target for target in found if target is not None}

    def for_target(self, target: str) -> "SourceIndex":
        """Return a view restricted to one link-namespace / reccmp target."""
        abi = self.target_abis.get(target)
        if abi is None and self.abi is not None:
            # An index built directly (not per target) may only carry ``abi``.
            if self.targets() <= {target}:
                abi = self.abi

        def scoped(records: Mapping[DeclarationKey, Any]) -> dict[DeclarationKey, Any]:
            return {key: item for key, item in records.items() if key.target == target}

        return SourceIndex(
            declarations=scoped(self.declarations),
            classes=scoped(self.classes),
            markers=(item for item in self.markers if item.target == target),
            variables=scoped(self.variables),
            conflicts=(item for item in self.conflicts if item.target == target),
            abi=abi,
            target_abis={target: abi} if abi is not None else {},
            marker_blocks=self.marker_blocks,
            source_digests=self.source_digests,
            unit_dependencies=self.unit_dependencies,
            document_digest=(
                f"{self._document_digest}:{target}"
                if self._document_digest is not None
                else None
            ),
        )

    def identity(self) -> str:
        """A digest of everything this index states: cheap for an index read
        from a file (the file's digest, recorded when it was read), a hash of
        its JSON projection otherwise. Any change to what Clang reported —
        declaration keys, marker ownership, ABI facts — changes it, also
        when no source file changed."""
        if self._document_digest is None:
            projection = json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":")
            )
            self._document_digest = hashlib.sha256(
                projection.encode("utf-8")
            ).hexdigest()
        return self._document_digest

    def stale_sources(self, paths: Iterable[PurePath]) -> list[PurePath]:
        """Source files that changed, or appeared, since the index was collected."""
        paths = list(paths)
        known = {
            path: relative
            for relative, path in local_paths(self.source_digests, paths).items()
        }
        return [
            path
            for path in paths
            if path not in known
            or source_digest(Path(path)) != self.source_digests[known[path]]
        ]

    @classmethod
    def from_units(
        cls,
        units: Sequence[TranslationUnitRecords],
        targets: Mapping[str, set[str] | None],
        *,
        target_files: Mapping[str, set[str]] | None = None,
        aliases: ProjectAliases | None = None,
        source_digests: Mapping[str, str] | None = None,
        repository: Path | None = None,
    ) -> "SourceIndex":
        """Derive every target's link namespace from TU observations in one
        pass, then join markers. ``targets`` maps each target to the units
        compiled into it (None: all of them). ``target_files`` gives each
        target's source files: its markers are read only from those, as a
        target's markers always have been (None: from every file).

        Marker blocks come from every unit and are merged once: a header's
        markers for one target may only be compiled by another target's
        translation units."""
        blocks = merge_marker_blocks(
            block for unit in units for block in unit.marker_blocks
        )
        declarations: dict[DeclarationKey, SourceDeclaration] = {}
        classes: dict[DeclarationKey, SourceClass] = {}
        markers: list[SourceMarker] = []
        variables: dict[DeclarationKey, SourceVariable] = {}
        conflicts: list[SourceConflict] = []
        abis: dict[str, SourceAbi] = {}
        for target, unit_ids in targets.items():
            namespace = derive_namespace(units, target=target, unit_ids=unit_ids)
            files = target_files.get(target) if target_files is not None else None
            target_classes, target_markers = _join_markers(
                target,
                namespace.declarations,
                namespace.classes,
                (
                    blocks
                    if files is None
                    else [block for block in blocks if block.source_file in files]
                ),
                aliases=aliases,
            )
            declarations.update(namespace.declarations)
            classes.update(target_classes)
            markers.extend(target_markers)
            variables.update(namespace.variables)
            conflicts.extend(namespace.conflicts)
            if namespace.abi is not None:
                abis[target] = namespace.abi
        distinct = set(abis.values())
        dependencies = None
        if repository is not None:
            paths = _RepositoryPaths(repository)
            dependencies = {
                unit.unit_id: paths.files(unit.dependencies) for unit in units
            }
        return cls(
            declarations=declarations,
            classes=classes,
            markers=markers,
            variables=variables,
            conflicts=conflicts,
            abi=distinct.pop() if len(distinct) == 1 else None,
            target_abis=abis,
            marker_blocks=blocks,
            source_digests=source_digests,
            unit_dependencies=dependencies,
        )

    @classmethod
    def from_collector(
        cls,
        target: str,
        collector: SourceCollector,
        *,
        unit_ids: set[str] | None = None,
        aliases: ProjectAliases | None = None,
    ) -> "SourceIndex":
        """Derive one target from a fixture ``SourceCollector`` (tests)."""
        return cls.from_units(
            tuple(collector.units.values()), {target: unit_ids}, aliases=aliases
        )

    @classmethod
    def from_dict(
        cls, document: Mapping[str, Any], *, document_digest: str | None = None
    ) -> "SourceIndex":
        """Read the public JSON projection back into its canonical records.
        A document of another shape raises (usually KeyError); collect the
        index again."""
        declarations = _keyed(document["declarations"], _declaration_from_dict)
        markers: list[SourceMarker] = []
        for item in document["markers"]:
            values = dict(item)
            encoded = values.pop("declaration_key")
            key = DeclarationKey.from_json(encoded) if encoded else None
            markers.append(
                SourceMarker(
                    **values,
                    declaration=declarations[key] if key is not None else None,
                    declaration_key=key,
                )
            )
        return cls(
            declarations=declarations,
            classes=_keyed(document["classes"], _class_from_dict),
            markers=markers,
            variables=_keyed(document["variables"], _variable_from_dict),
            conflicts=(_conflict_from_dict(item) for item in document["conflicts"]),
            abi=SourceAbi(**document["abi"]) if document["abi"] is not None else None,
            target_abis={
                target: SourceAbi(**values)
                for target, values in document["target_abis"].items()
            },
            marker_blocks=(
                MarkerBlock.from_dict(item) for item in document["marker_blocks"]
            ),
            source_digests=document["source_digests"],
            unit_dependencies=document["unit_dependencies"],
            document_digest=document_digest,
        )

    def functions_by_address(
        self, *, target: str | None = None
    ) -> dict[int, SourceMarker]:
        """Return one owner per address, preferring an unfolded body over aliases."""
        functions: dict[int, SourceMarker] = {}
        for marker in self.markers:
            if target is not None and marker.target != target:
                continue
            if not marker.name:
                raise SourceIndexError(
                    f"{marker.source_file}:{marker.line}: marker has no semantic identity"
                )
            previous = functions.get(marker.address)
            if previous is not None:
                if marker.folded and not previous.folded:
                    continue
                if marker.folded == previous.folded:
                    raise SourceIndexError(
                        f"0x{marker.address:08x} has more than one source owner"
                    )
            functions[marker.address] = marker
        return functions

    @classmethod
    def from_compile_database(
        cls,
        repository: Path,
        compilation_database: Path,
        targets: Mapping[str, Sequence[Path]],
        *,
        clang: str | None = None,
        jobs: int | None = None,
        cache_dir: Path | None = None,
        force: bool = False,
        aliases: ProjectAliases | None = None,
        paranoid: bool = False,
    ) -> "SourceIndex":
        # pylint: disable=too-many-arguments
        """Collect direct AST records natively, once for all marker targets.

        Expects to run in the same filesystem as the compile database (typically
        inside the pinned analysis image). ``RECCMP_SOURCE_INDEXER`` or
        ``reccmp-source-indexer`` on ``PATH`` supplies a prebuilt collector;
        otherwise the collector is built once into ``cache_dir`` against LLVM 21.
        """
        # pylint: disable=import-outside-toplevel
        from .batch import collect_compile_database

        return collect_compile_database(
            repository,
            compilation_database,
            targets,
            clang=clang,
            jobs=jobs,
            cache_dir=cache_dir,
            force=force,
            aliases=aliases,
            paranoid=paranoid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "markers": [_marker_projection(item) for item in self.markers],
            "declarations": _flattened(self.declarations),
            "classes": _flattened(self.classes),
            "variables": _flattened(self.variables),
            "conflicts": [_plain(item) for item in self.conflicts],
            "marker_blocks": [item.to_dict() for item in self.marker_blocks],
            "source_digests": self.source_digests,
            "unit_dependencies": {
                unit: list(paths) for unit, paths in self.unit_dependencies.items()
            },
            "abi": _plain(self.abi),
            "target_abis": {
                target: _plain(abi) for target, abi in sorted(self.target_abis.items())
            },
        }

    @classmethod
    def read(cls, path: Path) -> "SourceIndex":
        try:
            content = path.read_bytes()
            parsed = (
                _parsed_document(content)
                if cls is SourceIndex
                else cls.from_dict(
                    json.loads(content),
                    document_digest=hashlib.sha256(content).hexdigest(),
                )
            )
            # Records are immutable; give each reader independent lookup maps
            # so a caller cannot alter another reader's cached projection.
            index = copy(parsed)
            for name, value in vars(parsed).items():
                if isinstance(value, dict):
                    setattr(index, name, value.copy())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, SourceIndexError):
                raise
            raise SourceIndexError(
                f"source index at {path} is unusable: {exc}"
            ) from exc
        return index

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), separators=(",", ":")) + "\n"
        encoded = content.encode("utf-8")
        if not path.is_file() or path.read_bytes() != encoded:
            path.write_bytes(encoded)


@lru_cache(maxsize=2)
def _parsed_document(content: bytes) -> SourceIndex:
    """Reuse parsing for identical bytes; every read still observes the file."""
    return SourceIndex.from_dict(
        json.loads(content), document_digest=hashlib.sha256(content).hexdigest()
    )
