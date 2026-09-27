"""Wrapper for database (here an in-memory sqlite database) that collects the
addresses/symbols that we want to compare between the original and recompiled binaries.
"""

import bisect
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping
from reccmp.types import EntityType, ImageId


def entity_name_from_string(text: str, wide: bool = False) -> str:
    """Create an entity name for the given string by escaping
    control characters and double quotes, then wrapping in double quotes."""
    escaped = text.encode("unicode_escape").decode("utf-8").replace('"', '\\"')
    return f'{"L" if wide else ""}"{escaped}"'


EntityTypeLookup: dict[int, str] = {
    value: name for name, value in EntityType.__members__.items()
}


class FrozenEntityDbError(RuntimeError):
    """Raised when a frozen entity catalog is mutated."""


@dataclass(frozen=True)
class SideEntity:
    """Facts from one binary. Matching links two records without merging them."""

    address: int
    facts: Mapping[str, Any] = field(default_factory=dict)

    def freeze(self) -> None:
        """Seal the side facts shared by all views of a matched entity."""
        if isinstance(self.facts, dict):
            object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))

    def __getstate__(self) -> tuple[int, dict[str, Any], bool]:
        """Keep frozen prepared analyses usable in the local pickle cache."""
        return self.address, dict(self.facts), not isinstance(self.facts, dict)

    def __setstate__(self, state: tuple[int, dict[str, Any], bool]) -> None:
        address, facts, frozen = state
        object.__setattr__(self, "address", address)
        object.__setattr__(self, "facts", MappingProxyType(facts) if frozen else facts)


class ReccmpEntity:
    """One or two side records joined by a match."""

    orig: SideEntity | None
    recomp: SideEntity | None

    def __init__(
        self,
        orig: int | None,
        recomp: int | None,
        kvstore: dict[str, Any] | None = None,
    ) -> None:
        """Requires one or both of the addresses to be defined"""
        assert orig is not None or recomp is not None
        # Direct construction is also used by fixtures. Route the few
        # side-qualified input fields to their owner; ordinary facts stay
        # with the original side when both addresses are present.
        orig_facts = {
            key: value
            for key, value in (kvstore or {}).items()
            if key not in {"recomp_size", "recomp_max_size", "ref_recomp"}
        }
        recomp_facts = {
            key: value
            for key, value in (kvstore or {}).items()
            if orig is None or key in {"recomp_size", "recomp_max_size", "ref_recomp"}
        }
        self.orig = SideEntity(orig, orig_facts) if orig is not None else None
        self.recomp = SideEntity(recomp, recomp_facts) if recomp is not None else None

    def side(self, image_id: ImageId) -> SideEntity | None:
        if image_id == ImageId.ORIG:
            return self.orig
        if image_id == ImageId.RECOMP:
            return self.recomp
        raise ValueError("Invalid image id")

    def fact(self, image_id: ImageId, key: str, default: Any = None) -> Any:
        """Read a fact only from the specified binary."""
        side = self.side(image_id)
        return side.facts.get(key, default) if side is not None else default

    def addr(self, image_id: ImageId) -> int | None:
        if image_id == ImageId.ORIG:
            return self.orig_addr

        if image_id == ImageId.RECOMP:
            return self.recomp_addr

        assert False, "Invalid image id"

    @property
    def orig_addr(self) -> int | None:
        return self.orig.address if self.orig is not None else None

    @property
    def recomp_addr(self) -> int | None:
        return self.recomp.address if self.recomp is not None else None

    @property
    def entity_type(self) -> int | None:
        return self.get("type")

    @property
    def name(self) -> str | None:
        return self.get("name")

    def max_size(self, image_id: ImageId) -> int | None:
        if image_id == ImageId.RECOMP:
            return self.fact(ImageId.RECOMP, "recomp_max_size")

        if image_id == ImageId.ORIG:
            return self.fact(ImageId.ORIG, "orig_max_size")

        assert False, "Invalid image id"

    def any_size(self, image_id: ImageId = ImageId.RECOMP) -> int:
        """Returns any size for this entity: the returned value cannot be null.
        Prefer to return the size attribute for the provided ImageId if it exists.
        With no ImageId, prefer recomp_size first, then orig_size, default to zero.
        (This matches the previous behavior.)"""
        if image_id == ImageId.RECOMP:
            return self.size(ImageId.RECOMP) or self.size(ImageId.ORIG) or 0

        if image_id == ImageId.ORIG:
            return self.size(ImageId.ORIG) or self.size(ImageId.RECOMP) or 0

        return 0

    def size(self, image_id: ImageId) -> int | None:
        """Return the size attribute for the provided ImageId."""
        if image_id == ImageId.ORIG:
            return self.fact(ImageId.ORIG, "orig_size")

        if image_id == ImageId.RECOMP:
            return self.fact(ImageId.RECOMP, "recomp_size")

        assert False, "Invalid image id"

    @property
    def matched(self) -> bool:
        return self.orig is not None and self.recomp is not None

    def get(self, key: str, default: Any = None) -> Any:
        recomp = self.fact(ImageId.RECOMP, key)
        return recomp if recomp is not None else self.fact(ImageId.ORIG, key, default)

    def best_name(self) -> str | None:
        """Return the first name that exists from our
        priority list of name attributes for this entity."""
        for key in ("computed_name", "name"):
            if (value := self.get(key)) is not None:
                return str(value)

        return None

    def match_name(self, suffix: str = "") -> str | None:
        """Combination of the name and compare type.
        Intended for name substitution in the diff. If there is a diff,
        it will be more obvious what this symbol indicates."""
        best_name = self.best_name()
        if best_name is None:
            return None

        if suffix:
            return f"{best_name}{suffix} (OFFSET)"

        ctype = EntityTypeLookup.get(self.entity_type or -1, "UNK")
        return f"{best_name} ({ctype})"


class ReccmpMatch(ReccmpEntity):
    """To simplify type checking, use this object when a "match" is
    required or expected. Meaning: both orig and recomp addresses are set."""

    def __init__(
        self, orig: int, recomp: int, kvstore: dict[str, Any] | None = None
    ) -> None:
        assert orig is not None and recomp is not None
        super().__init__(orig, recomp, kvstore)

    @classmethod
    def link(cls, orig: SideEntity, recomp: SideEntity) -> "ReccmpMatch":
        """Link existing side records without copying or merging their facts."""
        match = cls.__new__(cls)
        match.orig = orig
        match.recomp = recomp
        return match

    @property
    def orig_addr(self) -> int:
        assert self.orig is not None
        return self.orig.address

    @property
    def recomp_addr(self) -> int:
        assert self.recomp is not None
        return self.recomp.address


logger = logging.getLogger(__name__)


class EntityBatch:
    base: "EntityDb"

    _orig: dict[int, dict[str, Any]]
    _recomp: dict[int, dict[str, Any]]
    _matches: list[tuple[int, int]]

    def __init__(self, backref: "EntityDb") -> None:
        self.base = backref
        self._orig = {}
        self._recomp = {}
        self._matches = []

    def reset(self):
        """Clear all pending changes"""
        self._orig.clear()
        self._recomp.clear()
        self._matches.clear()

    # pylint: disable=too-many-positional-arguments
    def set(
        self,
        img: ImageId,
        addr: int,
        ref: int | None = None,
        size: int | None = None,
        max_size: int | None = None,
        **kwargs,
    ):
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        if ref is not None:
            kwargs["ref_orig" if img == ImageId.ORIG else "ref_recomp"] = ref

        if size is not None:
            kwargs["orig_size" if img == ImageId.ORIG else "recomp_size"] = size

        if max_size is not None:
            kwargs["orig_max_size" if img == ImageId.ORIG else "recomp_max_size"] = (
                max_size
            )

        if img == ImageId.ORIG:
            self._orig.setdefault(addr, {}).update(kwargs)

        elif img == ImageId.RECOMP:
            self._recomp.setdefault(addr, {}).update(kwargs)

    def set_ref(
        self,
        img: ImageId,
        addr: int,
        *,
        ref: int,
        displacement: tuple[int, int] = (0, 0),
    ):
        self.set(img, addr, ref=ref, displacement=displacement)

    def match(self, orig: int, recomp: int):
        self._matches.append((orig, recomp))

    def set_recomp_addr(self, orig: int, recomp: int):
        self.match(orig, recomp)

    def _finalized_matches(self) -> Iterator[tuple[int, int]]:
        """Reduce the list of matches so that each orig and recomp addr appears once.
        If an address is repeated, retain the first pair where it is used and ignore any others.
        """
        used_orig = set()
        used_recomp = set()

        # This should have the same effect as the original implementation
        # that used two dicts to check uniqueness during each call to match().
        for orig, recomp in self._matches:
            if orig not in used_orig and recomp not in used_recomp:
                used_orig.add(orig)
                used_recomp.add(recomp)
                yield ((orig, recomp))
            else:
                logger.warning(
                    "Match (%x, %x) collides with previous staged match", orig, recomp
                )

    def commit(self):
        if self._orig:
            self.base.bulk_insert(ImageId.ORIG, self._orig.items())

        if self._recomp:
            self.base.bulk_insert(ImageId.RECOMP, self._recomp.items())

        if self._matches:
            self.base.bulk_match(self._finalized_matches())

        self.reset()

    def __enter__(self) -> "EntityBatch":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            self.reset()
        else:
            self.commit()


class EntityDb:
    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-instance-attributes
    _entities: dict[ImageId, dict[int, ReccmpEntity]]
    _matches: dict[ImageId, dict[int, int]]
    _addr_set: dict[ImageId, set[int]]
    _addr_order: dict[ImageId, list[int]]
    _sections: dict[ImageId, list[range]]

    def __init__(self):
        self._entities = {ImageId.ORIG: {}, ImageId.RECOMP: {}}
        self._matches = {ImageId.ORIG: {}, ImageId.RECOMP: {}}
        # Side-local duplicate bodies that have been proven equivalent to a
        # real matched pair.  The value is always the canonical original
        # address; aliases deliberately do not occupy the one-to-one match map.
        self._aliases: dict[ImageId, dict[int, int]] = {
            ImageId.ORIG: {},
            ImageId.RECOMP: {},
        }

        self._addr_set = {ImageId.ORIG: set(), ImageId.RECOMP: set()}
        self._addr_order = {ImageId.ORIG: [], ImageId.RECOMP: []}

        self._sections = {ImageId.ORIG: [], ImageId.RECOMP: []}
        self._equivalence_groups: dict[int, int] = {}
        self._frozen = False
        self._generation = 0

    @property
    def frozen(self) -> bool:
        return getattr(self, "_frozen", False)

    @property
    def generation(self) -> int:
        """Identity-cache generation; increments when pairings or aliases change."""
        return getattr(self, "_generation", 0)

    def _bump_generation(self) -> None:
        self._generation = getattr(self, "_generation", 0) + 1

    def freeze(self) -> None:
        """Seal facts and pairing/identity after ingest. Resolver caches may follow."""
        if self._frozen:
            return
        for image_id in (ImageId.ORIG, ImageId.RECOMP):
            for entity in self._entities[image_id].values():
                side = entity.side(image_id)
                if side is not None:
                    side.freeze()
        self._frozen = True

    def set_equivalence_groups(self, groups: Mapping[int, int]) -> None:
        """Install project-declared canonical original addresses during ingest."""
        if dict(groups) == self._equivalence_groups:
            return
        self._require_mutable()
        self._equivalence_groups = dict(groups)
        self._bump_generation()

    def canonical_orig(
        self,
        image_id: ImageId,
        addr: int,
        groups: Mapping[int, int] | None = None,
        entity: ReccmpEntity | None = None,
    ) -> int | None:
        """Proven original identity of a paired, folded, or declared alias."""
        canonical = self.alias_canonical_orig(image_id, addr)
        if canonical is None and entity is not None and entity.matched:
            canonical = entity.orig_addr
        if (
            canonical is None
            and image_id == ImageId.ORIG
            and addr in self._entities[image_id]
        ):
            canonical = addr
        if canonical is None:
            return None
        chosen_groups = groups if groups is not None else self._equivalence_groups
        mapped = chosen_groups.get(canonical)
        if mapped is not None:
            return mapped
        if (
            image_id == ImageId.ORIG
            and addr not in self._matches[image_id]
            and addr not in self._aliases[image_id]
            and not (entity is not None and entity.matched)
        ):
            return None
        return canonical

    def _require_mutable(self) -> None:
        if self.frozen:
            raise FrozenEntityDbError("entity catalog is frozen")

    def batch(self) -> EntityBatch:
        return EntityBatch(self)

    def count(self) -> int:
        return len(list(self.get_all()))

    def _update_addr_index(self, img: ImageId, addrs: set[int]):
        """Update the ordered list of addresses in this address space."""
        extent = self._addr_set[img]
        order = self._addr_order[img]
        order.extend(addrs - extent)
        order.sort()
        extent |= addrs

    def bulk_insert(self, image: ImageId, rows: Iterable[tuple[int, dict[str, Any]]]):
        self._require_mutable()
        assert image in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        new_addrs = set()
        entities = self._entities[image]

        for addr, values in rows:
            new_addrs.add(addr)

            if addr not in entities:
                if image == ImageId.ORIG:
                    entities[addr] = ReccmpEntity(addr, None, values)
                else:
                    entities[addr] = ReccmpEntity(None, addr, values)
            else:
                side = entities[addr].side(image)
                assert side is not None
                assert isinstance(side.facts, dict)
                side.facts.update(values)

        self._update_addr_index(image, new_addrs)
        self._bump_generation()

    def bulk_match(self, pairs: Iterable[tuple[int, int]]):
        """Expects iterable of `(orig_addr, recomp_addr)`."""
        self._require_mutable()

        orig_entities = self._entities[ImageId.ORIG]
        recomp_entities = self._entities[ImageId.RECOMP]

        new_x = set()
        new_y = set()

        for x, y in pairs:
            # Cannot replace existing match.
            if x in self._matches[ImageId.ORIG] or y in self._matches[ImageId.RECOMP]:
                continue

            new_x.add(x)
            new_y.add(y)

            self._matches[ImageId.ORIG][x] = y
            self._matches[ImageId.RECOMP][y] = x
            self._aliases[ImageId.ORIG].pop(x, None)
            self._aliases[ImageId.RECOMP].pop(y, None)

            orig_side = orig_entities[x].orig if x in orig_entities else SideEntity(x)
            recomp_side = (
                recomp_entities[y].recomp if y in recomp_entities else SideEntity(y)
            )
            assert orig_side is not None and recomp_side is not None
            match = ReccmpMatch.link(orig_side, recomp_side)

            orig_entities[x] = match
            recomp_entities[y] = match

        self._update_addr_index(ImageId.ORIG, new_x)
        self._update_addr_index(ImageId.RECOMP, new_y)
        self._bump_generation()

    def add_section(self, img: ImageId, range_: range):
        self._require_mutable()
        self._sections[img].append(range_)

    def sections(self, img: ImageId) -> Iterator[range]:
        yield from self._sections[img]

    def all(self, img: ImageId) -> Iterator[ReccmpEntity]:
        """Iterate entities in order for the given the address space."""
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        entities = self._entities[img]

        for addr in self._addr_order[img]:
            if addr in entities:
                yield entities[addr]

    def all_in_range(self, img: ImageId, range_: range) -> Iterator[ReccmpEntity]:
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        addrs = self._addr_order[img]
        entities = self._entities[img]

        i = bisect.bisect_left(addrs, range_.start)
        j = bisect.bisect_left(addrs, range_.stop)

        for addr in addrs[i:j]:
            if addr in entities:
                yield entities[addr]

    def unmatched(self, img: ImageId) -> Iterator[ReccmpEntity]:
        """Iterate unmatched entities only in order for the given the address space."""
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        entities = self._entities[img]
        matches = self._matches[img]

        for addr in self._addr_order[img]:
            if addr in entities and addr not in matches:
                yield entities[addr]

    def get_all(self) -> Iterator[ReccmpEntity]:
        orig_entities = self._entities[ImageId.ORIG]
        recomp_entities = self._entities[ImageId.RECOMP]

        for orig_addr in self._addr_order[ImageId.ORIG]:
            yield orig_entities[orig_addr]

        for recomp_addr in self._addr_order[ImageId.RECOMP]:
            if recomp_addr not in self._matches[ImageId.RECOMP]:
                yield recomp_entities[recomp_addr]

    def set_alias(self, image_id: ImageId, addr: int, canonical_orig: int) -> bool:
        """Record a proven side-local duplicate of a real matched function.

        Aliases are accounting/identity edges, not matches: several addresses
        on either image may name the same canonical pair.  Return ``False``
        when either endpoint is unsuitable rather than inventing an entity or
        replacing a one-to-one pair.
        """
        self._require_mutable()
        assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        if addr in self._matches[image_id]:
            return False
        canonical = self.get_one_match(canonical_orig)
        if canonical is None or addr not in self._entities[image_id]:
            return False
        existing = self._aliases[image_id].get(addr)
        if existing is not None:
            return False
        self._aliases[image_id][addr] = canonical_orig
        self._bump_generation()
        return True

    def alias_canonical_orig(self, image_id: ImageId, addr: int) -> int | None:
        """Canonical original address for an alias, or for a real pair."""
        assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        if image_id == ImageId.ORIG and addr in self._matches[ImageId.ORIG]:
            return addr
        if image_id == ImageId.RECOMP:
            paired_orig = self._matches[ImageId.RECOMP].get(addr)
            if paired_orig is not None:
                return paired_orig
        return self._aliases[image_id].get(addr)

    def get_aliases(
        self, image_id: ImageId
    ) -> Iterator[tuple[ReccmpEntity, ReccmpMatch]]:
        """Yield side-local alias entities and their canonical real pairs."""
        assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        for addr in self._addr_order[image_id]:
            canonical_orig = self._aliases[image_id].get(addr)
            if canonical_orig is None:
                continue
            canonical = self.get_one_match(canonical_orig)
            if canonical is not None:
                yield self._entities[image_id][addr], canonical

    def unexplained(self, image_id: ImageId) -> Iterator[ReccmpEntity]:
        """Unmatched entities excluding proven aliases.

        ``unmatched`` remains the raw inventory API and intentionally includes
        aliases, so callers can report both headline and raw counts.
        """
        aliases = self._aliases[image_id]
        for entity in self.unmatched(image_id):
            addr = entity.addr(image_id)
            if addr is not None and addr not in aliases:
                yield entity

    def get_matches(self) -> Iterator[ReccmpMatch]:
        matches = self._matches[ImageId.ORIG]
        entities = self._entities[ImageId.ORIG]
        for orig_addr in self._addr_order[ImageId.ORIG]:
            if orig_addr in matches:
                ent = entities[orig_addr]
                assert isinstance(ent, ReccmpMatch)
                yield ent

    def get_one_match(self, orig_addr: int) -> ReccmpMatch | None:
        if orig_addr not in self._entities[ImageId.ORIG]:
            return None

        ent = self._entities[ImageId.ORIG][orig_addr]
        if ent.recomp_addr is None:
            return None

        assert isinstance(ent, ReccmpMatch)
        return ent

    def nearest(self, img: ImageId, addr: int) -> int | None:
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        addrs = self._addr_order[img]

        i = bisect.bisect_right(addrs, addr)
        if i == 0:
            return None

        return addrs[i - 1]

    def get(
        self, img: ImageId, addr: int, *, exact: bool = True
    ) -> ReccmpEntity | None:
        """Return the ReccmpEntity at the given address and address space (ImageId).
        If there is no entry for the address and exact=True (default), return None.
        Otherwise, return the preceding (by address, in this image) entity if it exists.
        The caller should check the entity's size to make sure it covers the address."""
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        if not exact and addr not in self._entities[img]:
            prev_addr = self.nearest(img, addr)
            if prev_addr is None:
                return None

            addr = prev_addr

        return self._entities[img].get(addr)

    def callee_names(self, img: ImageId, addr: int) -> set[str]:
        """The symbol, name and import name of the entity a call to ``addr``
        reaches, following thunk references to their target."""
        ref_key = "ref_orig" if img == ImageId.ORIG else "ref_recomp"
        seen: set[int] = set()
        while addr not in seen and (entity := self.get(img, addr)) is not None:
            seen.add(addr)
            ref = entity.get(ref_key)
            if not isinstance(ref, int):
                names = (entity.get("symbol"), entity.name, entity.get("import_name"))
                return {name for name in names if isinstance(name, str)}
            addr = ref
        return set()

    def get_functions(self) -> Iterator[ReccmpMatch]:
        """Return all function-like matched entities. Previously, all functions
        had type=FUNCTION but there are now THUNK and VTORDISP types."""
        for ent in self.get_matches():
            if ent.get("type") in (
                EntityType.FUNCTION,
                EntityType.THUNK,
                EntityType.VTORDISP,
            ):
                yield ent

    def get_matches_by_type(self, entity_type: EntityType) -> Iterator[ReccmpMatch]:
        for ent in self.get_matches():
            if ent.get("type") == entity_type:
                yield ent

    def get_lines_in_recomp_range(
        self, start_recomp_addr: int, end_recomp_addr: int
    ) -> Iterator[ReccmpMatch]:
        """Fetches all matched annotations of the form `// LINE: TARGET 0x1234` in the given recomp address range."""
        addrs = self._addr_order[ImageId.RECOMP]
        i = bisect.bisect_left(addrs, start_recomp_addr)
        j = bisect.bisect_right(addrs, end_recomp_addr)

        recomp_matches = self._matches[ImageId.RECOMP]
        candidates = addrs[i:j]
        orig_addrs = [
            recomp_matches[addr] for addr in candidates if addr in recomp_matches
        ]

        for orig_addr in sorted(orig_addrs):
            match = self._entities[ImageId.ORIG][orig_addr]
            if match.get("type") == EntityType.LINE:
                assert isinstance(match, ReccmpMatch)
                yield match

    def exists(self, img: ImageId, addr: int) -> bool:
        """Is there an entity at this address?"""
        assert img in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"
        return addr in self._addr_set[img]

    def intersects(self, img: ImageId, addr: int) -> bool:
        """Is there an entity with a size that covers this address?"""
        if self.exists(img, addr):
            return True

        entity = self.get(img, addr, exact=False)
        if entity is None:
            return False

        base_addr = entity.addr(img)
        assert isinstance(base_addr, int)

        size = entity.any_size(img)
        return addr - base_addr < size

    def is_match(self, orig_addr: int, recomp_addr: int) -> bool:
        return self._matches[ImageId.ORIG].get(orig_addr) == recomp_addr

    def get_max_size(self, image_id: ImageId, addr: int) -> int | None:
        """Get the distance between this entity and the next "solid" entity or
        the end of the section/image.
        Returns None if no estimation is possible."""
        assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

        # Stop when we reach the first entity that takes up space.
        solid_types = EntityType.solid_types()

        for range_ in self.sections(image_id):
            # Find the image section that contains the input address.
            if addr in range_:
                # For all entities after the input address:
                # (Note that this does not require the starting entity to exist.)
                from_addr_on = range(addr + 1, range_.stop)
                for ent in self.all_in_range(image_id, from_addr_on):
                    this_type = ent.get("type")
                    if this_type not in solid_types:
                        continue

                    this_addr = ent.addr(image_id)
                    assert this_addr is not None
                    return this_addr - addr

                return range_.stop - addr

        return None
