"""Join reccmp markers to semantic declarations from the Clang AST.

The marker parser owns annotation syntax and addresses. Clang owns C++ names,
function and variable kinds, types, linkage, class membership, inheritance,
and virtual declarations. This module only joins those two models by source
location and writes disposable JSON projections for downstream tools.

Compiler records arrive as per-TU observations. Link-namespace partitioning,
winner selection, and conflict derivation happen after collection — never by
globally collapsing bare ``semantic_id`` values first.
"""

from __future__ import annotations

# The optional execution backend imports this record model when first used.
# pylint: disable=cyclic-import

import json
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reccmp.formats import TextFile
from reccmp.parser.codebase import DecompCodebase
from reccmp.parser.marker import MarkerType, ProjectAliases
from .variables import SourceConflict, SourceConflictVariant, SourceVariable

_VARIABLE_RANK = {"declaration": 0, "tentative": 1, "definition": 2}


class SourceIndexError(ValueError):
    """The source markers and compiler model cannot be joined unambiguously."""


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceDeclaration:
    """One semantic function declaration emitted by Clang."""

    semantic_id: str
    qualified_name: str
    semantic_kind: str
    calling_convention: str
    return_type: str
    parameter_types: tuple[str, ...]
    owning_class: str | None
    has_this: bool
    is_virtual: bool
    source_file: str
    line: int
    end_line: int
    is_definition: bool
    source_signature: str | None = None
    parameter_references: tuple[bool, ...] = ()
    parameter_reference_forms: tuple[str, ...] = ()
    linkage: str = ""
    storage_class: str = ""
    is_variadic: bool = False
    # Compilation unit that observed this declaration (repo-relative main file).
    # External entities share one identity across units; non-external ones are
    # distinct per unit even when their unmangled spelling collides.
    unit_id: str = ""
    # Link namespace (reccmp target) assigned when observations are partitioned.
    target: str | None = None

    @property
    def prototype(self) -> str:
        """Render compiler-owned types for display, not ABI synchronization."""
        parameters = ", ".join(self.parameter_types) or "void"
        if self.is_variadic:
            parameters = f"{parameters}, ..." if self.parameter_types else "..."
        prefix = f"{self.return_type} " if self.return_type else ""
        return f"{prefix}{self.qualified_name}({parameters})"

    @property
    def is_external(self) -> bool:
        """Genuinely cross-TU linkage. Internal, unique-external (anonymous
        namespace) and unlinked declarations never join across units."""
        return self.linkage == "external"

    @property
    def signature(self) -> tuple[str, ...]:
        """The type identity a cross-TU consistency gate compares."""
        return (
            self.semantic_kind,
            self.calling_convention,
            self.return_type,
            *self.parameter_types,
            self.linkage,
            "..." if self.is_variadic else "",
        )

    @property
    def merge_key(self) -> tuple[str, ...]:
        """Identity used when grouping observations inside one link namespace."""
        if self.is_external:
            return (self.semantic_id,)
        return (self.unit_id, self.semantic_id)


@dataclass(frozen=True)
class SourceField:
    """One direct non-static source field emitted by Clang."""

    name: str
    type: str
    source_file: str
    line: int

    pointer_depth: int | None = None


@dataclass(frozen=True)
class SourceBaseVtable:
    """One vtable installed for a polymorphic base subobject."""

    address: int
    base_class: str


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class SourceClass:
    """One complete C++ record definition emitted by Clang."""

    semantic_id: str
    qualified_name: str
    bases: tuple[str, ...]
    fields: tuple[SourceField, ...]
    virtual_declarations: tuple[str, ...]
    source_file: str
    line: int
    end_line: int
    asserted_size: int | None = None
    vtable_address: int | None = None
    base_vtables: tuple[SourceBaseVtable, ...] = ()
    unit_id: str = ""
    target: str | None = None


@dataclass(frozen=True)
class SourceMarker:
    """A reccmp marker and its compiler-owned declaration, when applicable."""

    address: int
    marker_kind: str
    source_file: str
    line: int
    declaration: SourceDeclaration | None
    marker_name: str | None = None
    folded: bool = False
    target: str | None = None

    @property
    def name(self) -> str:
        """Compiler identity, or the name attached to a non-body marker."""
        return (
            self.declaration.qualified_name
            if self.declaration
            else self.marker_name or ""
        )


@dataclass(frozen=True)
class _SizeAssertion:
    unit_id: str
    qualified_name: str
    asserted_size: int


@dataclass(frozen=True)
class _NamespaceRecords:
    """Winners and conflicts derived inside one link namespace."""

    declarations: tuple[SourceDeclaration, ...]
    variables: tuple[SourceVariable, ...]
    classes: tuple[SourceClass, ...]
    conflicts: tuple[SourceConflict, ...]
    size_assertions: dict[str, int]


class SourceCollector:
    """Accumulate per-TU compiler observations without merging them.

    Call ``derive()`` (or ``SourceIndex.from_collector``) to partition by link
    namespace, group entities, and produce winners plus conflicts.
    """

    def __init__(self, repository: Path, compilation_root: Path | None = None) -> None:
        self.repository = repository.resolve()
        self.compilation_root = compilation_root
        self.declarations: list[SourceDeclaration] = []
        self.variables: list[SourceVariable] = []
        self.classes: list[SourceClass] = []
        self.size_assertions: list[_SizeAssertion] = []

    def collect_records(self, records: str, *, unit_id: str = "") -> None:
        """Consume a compiler's newline-delimited JSON output."""
        for line in records.splitlines():
            if line.strip():
                self.collect_record(json.loads(line), unit_id=unit_id)

    def collect_record(self, record: Mapping[str, Any], *, unit_id: str = "") -> None:
        """Store one compiler observation without collapsing identities.

        Non-external variables are dropped: the compiler already checks them
        inside their translation unit, and they have no legitimate cross-TU
        writer/reader disagreement to report. Conflicting size assertions are
        checked only after partitioning by link namespace.
        """
        values = dict(record)
        kind = values.pop("record")
        if kind == "dependency":
            return
        if kind == "declaration":
            declaration = _declaration_from_dict({**values, "unit_id": unit_id})
            self.declarations.append(declaration)
        elif kind == "variable":
            variable = _variable_from_dict({**values, "unit_id": unit_id})
            if not variable.is_external:
                return
            self.variables.append(variable)
        elif kind == "class":
            self.classes.append(_class_from_dict({**values, "unit_id": unit_id}))
        elif kind == "size-assertion":
            self.size_assertions.append(
                _SizeAssertion(
                    unit_id=unit_id,
                    qualified_name=str(values["qualified_name"]),
                    asserted_size=int(values["asserted_size"]),
                )
            )
        else:
            raise SourceIndexError(
                f"the source indexer emitted an unknown record: {kind!r}"
            )

    def _relative(self, source_file: str) -> str:
        path = Path(source_file)
        if self.compilation_root is not None:
            try:
                suffix = path.relative_to(self.compilation_root)
            except ValueError:
                pass
            else:
                path = self.repository / suffix
        try:
            return path.resolve().relative_to(self.repository).as_posix()
        except ValueError:
            return path.as_posix()

    def unit_id_for(self, main_file: str | Path) -> str:
        """Repo-relative identity of a translation unit's main file."""
        return self._relative(str(main_file))

    def derive(
        self,
        *,
        target: str | None = None,
        unit_ids: set[str] | None = None,
    ) -> _NamespaceRecords:
        """Partition observations, then derive winners and conflicts.

        An observation belongs to the namespace when its compilation unit is in
        ``unit_ids``. When ``unit_ids`` is omitted, every observation is kept
        (single-namespace fixtures and tests).
        """

        def belongs(unit_id: str) -> bool:
            return unit_ids is None or unit_id in unit_ids

        declarations = [item for item in self.declarations if belongs(item.unit_id)]
        variables = [item for item in self.variables if belongs(item.unit_id)]
        classes = [item for item in self.classes if belongs(item.unit_id)]
        assertions = [item for item in self.size_assertions if belongs(item.unit_id)]

        derived_declarations, declaration_conflicts = _derive_entities(
            declarations,
            key=lambda item: item.merge_key,
            rank=lambda item: 1 if item.is_definition else 0,
            record_kind="declaration",
            target=target,
        )
        derived_variables, variable_conflicts = _derive_entities(
            variables,
            key=lambda item: (item.semantic_id,),
            rank=lambda item: _VARIABLE_RANK.get(item.definition_kind, 0),
            record_kind="variable",
            target=target,
        )
        derived_classes = _derive_classes(classes, target=target)
        size_assertions = _derive_size_assertions(assertions)

        return _NamespaceRecords(
            declarations=derived_declarations,
            variables=derived_variables,
            classes=tuple(
                replace(
                    item,
                    asserted_size=size_assertions.get(item.qualified_name),
                )
                for item in derived_classes
            ),
            conflicts=declaration_conflicts + variable_conflicts,
            size_assertions=size_assertions,
        )


def _derive_entities(
    observations: Sequence[Any],
    *,
    key,
    rank,
    record_kind: str,
    target: str | None,
) -> tuple[tuple[Any, ...], tuple[SourceConflict, ...]]:
    """Group observations by merge key, pick a winner, retain type conflicts."""
    groups: dict[tuple[str, ...], list[Any]] = {}
    for item in observations:
        groups.setdefault(key(item), []).append(item)

    winners: list[Any] = []
    conflicts: list[SourceConflict] = []
    for group in groups.values():
        winner = group[0]
        for item in group[1:]:
            if rank(item) > rank(winner):
                winner = item
        winners.append(replace(winner, target=target) if target is not None else winner)

        variants: dict[tuple[str, ...], list[str]] = {}
        for item in group:
            location = f"{item.source_file}:{item.line}"
            variants.setdefault(item.signature, [])
            if location not in variants[item.signature]:
                variants[item.signature].append(location)
        if len(variants) > 1:
            sample = group[0]
            conflicts.append(
                SourceConflict(
                    semantic_id=sample.semantic_id,
                    qualified_name=sample.qualified_name,
                    record_kind=record_kind,
                    variants=tuple(
                        SourceConflictVariant(
                            signature=signature,
                            locations=tuple(locations),
                        )
                        for signature, locations in variants.items()
                    ),
                    target=target,
                )
            )
    return tuple(winners), tuple(conflicts)


def _derive_classes(
    observations: Sequence[SourceClass], *, target: str | None
) -> tuple[SourceClass, ...]:
    best: dict[str, SourceClass] = {}
    for item in observations:
        previous = best.get(item.semantic_id)
        if previous is None or (not previous.line and item.line):
            best[item.semantic_id] = (
                replace(item, target=target) if target is not None else item
            )
    return tuple(best.values())


def _derive_size_assertions(assertions: Sequence[_SizeAssertion]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in assertions:
        previous = sizes.get(item.qualified_name)
        if previous is not None and previous != item.asserted_size:
            raise SourceIndexError(
                f"{item.qualified_name} has conflicting size assertions: "
                f"{previous:#x} and {item.asserted_size:#x}"
            )
        sizes[item.qualified_name] = item.asserted_size
    return sizes


def _command_arguments(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if arguments:
        return [str(item) for item in arguments]
    return shlex.split(str(entry["command"]), posix=True)


def ast_command(
    entry: dict[str, Any], clang: str | None, command_prefix: Sequence[str]
) -> list[str]:
    arguments = _command_arguments(entry)
    compiler = [clang or arguments[0]] if not command_prefix else list(command_prefix)
    filtered: list[str] = []
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-c", "/c", "-o", "-MF", "-MT", "-MQ"}:
            skip_next = argument in {"-o", "-MF", "-MT", "-MQ"}
            continue
        # codespell:ignore-begin
        if argument.startswith(("/Fo", "/Fd", "-o")):
            # codespell:ignore-end
            continue
        filtered.append(argument)
    try:
        separator = filtered.index("--")
    except ValueError:
        separator = len(filtered)
    ast_options = ["-fsyntax-only", "-Xclang", "-ast-dump=json"]
    return [*compiler, *filtered[:separator], *ast_options, *filtered[separator:]]


def _declaration_from_dict(values: Mapping[str, Any]) -> SourceDeclaration:
    data = dict(values)
    for key in ("parameter_types", "parameter_references", "parameter_reference_forms"):
        data[key] = tuple(data.get(key) or ())
    return SourceDeclaration(**data)


def _variable_from_dict(values: Mapping[str, Any]) -> SourceVariable:
    return SourceVariable(**dict(values))


def _conflict_from_dict(values: Mapping[str, Any]) -> SourceConflict:
    return SourceConflict(
        semantic_id=str(values["semantic_id"]),
        qualified_name=str(values["qualified_name"]),
        record_kind=str(values["record_kind"]),
        target=values.get("target"),
        variants=tuple(
            SourceConflictVariant(
                signature=tuple(variant.get("signature") or ()),
                locations=tuple(variant.get("locations") or ()),
            )
            for variant in values.get("variants") or ()
        ),
    )


def _class_from_dict(values: Mapping[str, Any]) -> SourceClass:
    return SourceClass(
        **{
            **values,
            "bases": tuple(values["bases"]),
            "fields": tuple(SourceField(**field) for field in values["fields"]),
            "virtual_declarations": tuple(values["virtual_declarations"]),
            "base_vtables": tuple(
                SourceBaseVtable(**item) for item in values.get("base_vtables", ())
            ),
        }
    )


class SourceIndex:
    """Canonical marker plus Clang semantic source index."""

    def __init__(
        self,
        *,
        declarations: Iterable[SourceDeclaration],
        classes: Iterable[SourceClass],
        markers: Iterable[SourceMarker],
        variables: Iterable[SourceVariable] = (),
        conflicts: Iterable[SourceConflict] = (),
    ) -> None:
        self.declarations = tuple(
            sorted(declarations, key=lambda item: item.semantic_id)
        )
        self.classes = tuple(sorted(classes, key=lambda item: item.semantic_id))
        self.markers = tuple(
            sorted(markers, key=lambda item: (item.address, item.source_file))
        )
        self.variables = tuple(sorted(variables, key=lambda item: item.semantic_id))
        self.conflicts = tuple(sorted(conflicts, key=lambda item: item.semantic_id))

    @classmethod
    def from_collector(
        cls,
        repository: Path,
        target: str,
        source_paths: Sequence[Path],
        collector: SourceCollector,
        *,
        unit_ids: set[str] | None = None,
        aliases: ProjectAliases | None = None,
    ) -> "SourceIndex":
        """Derive one link namespace from collected observations, then join markers.

        ``unit_ids`` names the translation units that belong to this namespace
        (repo-relative main files). When omitted, every observation is treated
        as belonging to ``target`` — appropriate for fixtures that never ran a
        multi-target batch.
        """
        files = tuple(TextFile.from_files(source_paths))
        codebase = DecompCodebase(files, target, aliases=aliases)
        namespace = collector.derive(target=target, unit_ids=unit_ids)
        declarations = namespace.declarations
        variables = namespace.variables
        conflicts = namespace.conflicts

        by_location: dict[tuple[str, int], list[SourceDeclaration]] = {}
        for declaration in declarations:
            if declaration.is_definition:
                by_location.setdefault(
                    (declaration.source_file, declaration.line), []
                ).append(declaration)

        markers: list[SourceMarker] = []
        for method_symbol in (
            *codebase.iter_line_functions(),
            *codebase.iter_name_functions(),
        ):
            relative = (
                Path(method_symbol.filename)
                .resolve()
                .relative_to(repository.resolve())
                .as_posix()
            )
            candidates = by_location.get((relative, method_symbol.line_number), [])
            marker_declaration: SourceDeclaration | None = None
            if method_symbol.type in {MarkerType.FUNCTION, MarkerType.STUB}:
                if len(candidates) != 1:
                    raise SourceIndexError(
                        f"{relative}:{method_symbol.line_number}: {method_symbol.type.name} "
                        f"0x{method_symbol.offset:08x} "
                        f"binds to {len(candidates)} function definitions"
                    )
                marker_declaration = candidates[0]
            markers.append(
                SourceMarker(
                    address=method_symbol.offset,
                    marker_kind=method_symbol.type.name,
                    source_file=relative,
                    line=method_symbol.line_number,
                    declaration=marker_declaration,
                    folded=method_symbol.is_folded,
                    target=target,
                    marker_name=(
                        method_symbol.name if marker_declaration is None else None
                    ),
                )
            )

        classes = list(namespace.classes)
        class_by_location = {
            (item.source_file, item.line): index for index, item in enumerate(classes)
        }
        class_by_name = {
            item.qualified_name: index for index, item in enumerate(classes)
        }
        for vtable_symbol in codebase.iter_vtables():
            relative = (
                Path(vtable_symbol.filename)
                .resolve()
                .relative_to(repository.resolve())
                .as_posix()
            )
            key = (relative, vtable_symbol.line_number)
            index = class_by_location.get(key)
            if index is None:
                # Template-specialization vtables are commonly accounted for by
                # a standalone class comment instead of a repeated source
                # declaration:
                #
                #   // VTABLE: TARGET 0x1234
                #   // class Vector<Element *>
                #
                # The marker parser already recovers that qualified class name.
                # Bind it to Clang's canonical specialization rather than
                # attaching it positionally to the next class definition.
                index = class_by_name.get(vtable_symbol.name)
            if index is None:
                # A standalone specialization annotation need not be named by
                # any explicit source declaration. Preserve that reviewed
                # identity as an annotation-owned record; if Clang did emit the
                # specialization, the name lookup above retains its bases,
                # fields, virtual declarations, and source extent instead.
                source_class = SourceClass(
                    semantic_id=f"record:{vtable_symbol.name}",
                    qualified_name=vtable_symbol.name,
                    bases=(),
                    fields=(),
                    virtual_declarations=(),
                    source_file=relative,
                    line=vtable_symbol.line_number,
                    end_line=vtable_symbol.line_number,
                    vtable_address=vtable_symbol.offset,
                    target=target,
                )
                classes.append(source_class)
                class_by_name[source_class.qualified_name] = len(classes) - 1
                continue
            source_class = classes[index]
            base_class = vtable_symbol.base_class
            class_names = {
                source_class.qualified_name,
                source_class.qualified_name.rsplit("::", 1)[-1],
            }
            if base_class is not None and base_class not in class_names:
                base_vtable = SourceBaseVtable(vtable_symbol.offset, base_class)
                if base_vtable in source_class.base_vtables:
                    raise SourceIndexError(
                        f"{relative}:{vtable_symbol.line_number}: duplicate VTABLE marker "
                        f"for base {base_class}"
                    )
                classes[index] = replace(
                    source_class,
                    base_vtables=(*source_class.base_vtables, base_vtable),
                )
                continue
            if source_class.vtable_address is not None:
                raise SourceIndexError(
                    f"{relative}:{vtable_symbol.line_number}: class has more than one "
                    "primary VTABLE marker"
                )
            classes[index] = replace(source_class, vtable_address=vtable_symbol.offset)

        return cls(
            declarations=declarations,
            classes=classes,
            markers=markers,
            variables=variables,
            conflicts=conflicts,
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "SourceIndex":
        """Read the public JSON projection back into its canonical records."""
        if document.get("schema") != "reccmp-source-index-v2":
            raise SourceIndexError("unsupported source-index schema")
        return cls(
            declarations=(
                _declaration_from_dict(item) for item in document["declarations"]
            ),
            classes=(_class_from_dict(item) for item in document["classes"]),
            markers=(
                SourceMarker(
                    **{
                        **item,
                        "declaration": (
                            _declaration_from_dict(item["declaration"])
                            if item.get("declaration")
                            else None
                        ),
                    }
                )
                for item in document["markers"]
            ),
            variables=(
                _variable_from_dict(item) for item in document.get("variables", ())
            ),
            conflicts=(
                _conflict_from_dict(item) for item in document.get("conflicts", ())
            ),
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
    # pylint: disable=too-many-arguments
    def from_compile_database(
        cls,
        repository: Path,
        compilation_database: Path,
        targets: Mapping[str, Sequence[Path]],
        *,
        clang: str | None = None,
        jobs: int | None = None,
        container_image: str | None = None,
        mounts: Mapping[Path, str] | None = None,
        compilation_root: Path | None = None,
        cache_dir: Path | None = None,
        force: bool = False,
        aliases: ProjectAliases | None = None,
    ) -> "SourceIndex":
        """Collect direct AST records in parallel, once for all marker targets.

        A container image runs the whole batch in one container; mounts describe
        the paths already used by its compile database. Native collection uses
        each entry's working directory. The image/host needs Clang and LLVM 19
        development libraries. Per-TU dependency digests invalidate the cache.
        """
        # Delay the execution backend until collection is requested. The record
        # model remains importable without a Linux compiler environment.
        # pylint: disable=import-outside-toplevel
        from .batch import collect_compile_database

        return collect_compile_database(
            repository,
            compilation_database,
            targets,
            clang=clang,
            jobs=jobs,
            container_image=container_image,
            mounts=mounts,
            compilation_root=compilation_root,
            cache_dir=cache_dir,
            force=force,
            aliases=aliases,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "reccmp-source-index-v2",
            "markers": [asdict(item) for item in self.markers],
            "declarations": [asdict(item) for item in self.declarations],
            "classes": [asdict(item) for item in self.classes],
            "variables": [asdict(item) for item in self.variables],
            "conflicts": [asdict(item) for item in self.conflicts],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2) + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
