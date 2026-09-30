"""The prepared pairs and names a code comparison consumes.

A manifest records what reccmp asks an external differ to compare, under
which names, and from which inputs. It is small on purpose: it carries
correspondence and project context, not a model of either program.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Iterable

from reccmp.parser.node import ParserFunction
from reccmp.types import EntityType, ImageId
from .core import Compare
from .db import PairBasis, ReccmpEntity


@dataclass(frozen=True)
class SourceLocation:
    path: PurePath
    line: int


@dataclass(frozen=True)
class FunctionEntry:
    """One function the comparison is asked about.

    ``recomp_addr`` is ``None`` when the catalog has no counterpart: the
    function is reported as unpaired rather than dropped.
    """

    orig_addr: int
    recomp_addr: int | None
    name: str
    basis: PairBasis | None
    source: SourceLocation | None
    library: bool


@dataclass(frozen=True)
class NamedObject:  # pylint: disable=too-many-instance-attributes
    """A paired entity whose canonical name both programs receive."""

    orig_addr: int
    recomp_addr: int
    name: str
    entity_type: EntityType | None
    orig_size: int | None
    recomp_size: int | None
    basis: PairBasis
    recomp_symbol: str | None = None

    def addr(self, image_id: ImageId) -> int:
        return self.orig_addr if image_id == ImageId.ORIG else self.recomp_addr

    def extent(self, image_id: ImageId) -> int | None:
        """The object's size in one image. One object has one extent: when
        only the other image records a size (typically the recompiled PDB),
        that size stands for both."""
        own, other = (
            (self.orig_size, self.recomp_size)
            if image_id == ImageId.ORIG
            else (self.recomp_size, self.orig_size)
        )
        return own if own is not None else other


@dataclass(frozen=True)
class UnpairedEntity:
    """A catalog entity with no counterpart in the other image.

    Its name is its own image's (annotation or PDB), never shared: it
    identifies nothing on the other side. Its extent says how much of the
    image its contents occupy."""

    image_id: ImageId
    addr: int
    size: int | None
    entity_type: EntityType
    name: str | None


@dataclass(frozen=True)
class Alias:
    """A side-local duplicate of a pair (an identical
    duplicate emission): it has the pair's identity, at another address."""

    image_id: ImageId
    addr: int
    canonical_orig: int
    size: int | None


@dataclass(frozen=True)
class BinaryInput:
    path: Path
    sha256: str


@dataclass(frozen=True)
class Manifest:
    target_id: str
    orig: BinaryInput
    recomp: BinaryInput
    functions: tuple[FunctionEntry, ...]
    objects: tuple[NamedObject, ...]
    unpaired: tuple[UnpairedEntity, ...]
    aliases: tuple[Alias, ...] = ()

    def to_json(self) -> dict:
        return {
            "target": self.target_id,
            "orig": {"path": str(self.orig.path), "sha256": self.orig.sha256},
            "recomp": {"path": str(self.recomp.path), "sha256": self.recomp.sha256},
            "functions": [
                {
                    "orig": f"{entry.orig_addr:#x}",
                    "recomp": (
                        f"{entry.recomp_addr:#x}"
                        if entry.recomp_addr is not None
                        else None
                    ),
                    "name": entry.name,
                    "basis": entry.basis.value if entry.basis is not None else None,
                    "source": (
                        {"path": str(entry.source.path), "line": entry.source.line}
                        if entry.source is not None
                        else None
                    ),
                    "library": entry.library,
                }
                for entry in self.functions
            ],
            "objects": [
                {
                    "orig": f"{obj.orig_addr:#x}",
                    "recomp": f"{obj.recomp_addr:#x}",
                    "name": obj.name,
                    "type": obj.entity_type.name if obj.entity_type else None,
                    "orig_size": obj.orig_size,
                    "recomp_size": obj.recomp_size,
                    "basis": obj.basis.value,
                    **(
                        {"recomp_symbol": obj.recomp_symbol}
                        if obj.recomp_symbol
                        else {}
                    ),
                }
                for obj in self.objects
            ],
            "unpaired": [
                {
                    "image": entity.image_id.name.lower(),
                    "addr": f"{entity.addr:#x}",
                    "size": entity.size,
                    "type": entity.entity_type.name,
                    "name": entity.name,
                }
                for entity in self.unpaired
            ],
            "aliases": [
                {
                    "image": alias.image_id.name.lower(),
                    "addr": f"{alias.addr:#x}",
                    "canonical": f"{alias.canonical_orig:#x}",
                }
                for alias in self.aliases
            ],
        }

    def digest(self) -> str:
        """Identity of the metadata a comparison ran with."""
        text = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def preparation_digest(self) -> str:
        """Identity of metadata that changes the prepared Ghidra programs.

        Source locations explain the report but do not enter program
        preparation. A line-number-only source edit can reuse the prepared
        programs when the binaries and catalog facts are otherwise unchanged.
        """
        document = self.to_json()
        for entry in document["functions"]:
            del entry["source"]
        text = json.dumps(document, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_COMPARED_TYPES = (EntityType.FUNCTION, EntityType.VTORDISP)
_UNPAIRED_TYPES = (
    # Unpaired loader slots are still imports, not anonymous literal data.
    EntityType.IMPORT,
    EntityType.FUNCTION,
    EntityType.VTORDISP,
    EntityType.DATA,
    EntityType.POINTER,
    EntityType.STRING,
    EntityType.WIDECHAR,
    EntityType.FLOAT,
    EntityType.VTABLE,
)


def _source_locations(catalog: Compare) -> dict[int, SourceLocation]:
    locations: dict[int, SourceLocation] = {}
    for function in (
        *catalog.codebase.iter_line_functions(),
        *catalog.codebase.iter_name_functions(),
    ):
        assert isinstance(function, ParserFunction)
        locations.setdefault(
            function.offset, SourceLocation(function.filename, function.line_number)
        )
    return locations


def _entity_type(entity: ReccmpEntity) -> EntityType | None:
    value = entity.entity_type
    return EntityType(value) if value is not None else None


def build_manifest(
    catalog: Compare,
    *,
    target_id: str,
    orig_path: Path,
    recomp_path: Path,
    select: Callable[[ReccmpEntity], bool] = lambda _: True,
) -> Manifest:
    """Every selected original function, paired or not, plus the named
    objects that give both programs the same vocabulary."""
    locations = _source_locations(catalog)
    functions: list[FunctionEntry] = []
    objects: list[NamedObject] = []

    for entity in catalog.db.all(ImageId.ORIG):
        entity_type = _entity_type(entity)
        orig_addr = entity.orig_addr
        assert orig_addr is not None
        name = entity.best_name()

        if entity.matched and name is not None:
            recomp_addr = entity.recomp_addr
            basis = catalog.pair_basis(orig_addr)
            assert recomp_addr is not None and basis is not None
            objects.append(
                NamedObject(
                    orig_addr=orig_addr,
                    recomp_addr=recomp_addr,
                    name=name,
                    entity_type=entity_type,
                    orig_size=entity.size(ImageId.ORIG),
                    recomp_size=entity.size(ImageId.RECOMP),
                    basis=basis,
                    recomp_symbol=(
                        entity.fact(ImageId.RECOMP, "symbol")
                        if entity_type == EntityType.FUNCTION
                        else None
                    ),
                )
            )

        if entity_type not in _COMPARED_TYPES:
            continue
        # Stubs are declared as not yet implemented: there is nothing to
        # compare. Thunks never reach here (their own entity type).
        if entity.get("stub", False) or entity.get("skip", False):
            continue
        if not select(entity):
            continue
        functions.append(
            FunctionEntry(
                orig_addr=orig_addr,
                recomp_addr=entity.recomp_addr,
                name=name or f"FUNCTION_{orig_addr:x}",
                basis=catalog.pair_basis(orig_addr) if entity.matched else None,
                source=locations.get(orig_addr),
                library=bool(entity.get("library", False)),
            )
        )

    aliases = tuple(
        Alias(image_id, addr, canonical.orig_addr, alias.size(image_id))
        for image_id in (ImageId.ORIG, ImageId.RECOMP)
        for alias, canonical in catalog.get_aliases(image_id)
        if (addr := alias.addr(image_id)) is not None
    )
    aliased = {(alias.image_id, alias.addr) for alias in aliases}
    unpaired = tuple(
        UnpairedEntity(
            image_id,
            addr,
            entity.size(image_id),
            entity_type,
            entity.best_name(),
        )
        for image_id in (ImageId.ORIG, ImageId.RECOMP)
        for entity in catalog.db.unmatched(image_id)
        if (entity_type := _entity_type(entity)) in _UNPAIRED_TYPES
        and (addr := entity.addr(image_id)) is not None
        and (image_id, addr) not in aliased
    )

    return Manifest(
        target_id=target_id,
        orig=BinaryInput(orig_path, file_sha256(orig_path)),
        recomp=BinaryInput(recomp_path, file_sha256(recomp_path)),
        functions=tuple(functions),
        objects=tuple(objects),
        unpaired=unpaired,
        aliases=aliases,
    )


def select_addresses(addresses: Iterable[int]) -> Callable[[ReccmpEntity], bool]:
    wanted = frozenset(addresses)
    return lambda entity: entity.orig_addr in wanted
