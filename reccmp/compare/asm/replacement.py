import bisect
from functools import cache
from collections.abc import Hashable
from typing import Callable, Protocol
from reccmp.compare.db import EntityDb, EntityTypeLookup, ReccmpEntity
from reccmp.cvdump.types import CvdumpTypeKey
from reccmp.types import EntityType, ImageId


class AddrTestProtocol(Protocol):
    def __call__(self, addr: int, /) -> bool: ...


class NameReplacementProtocol(Protocol):
    def __call__(
        self, addr: int, exact: bool = False, indirect: bool = False
    ) -> str | None: ...


_CALLABLE_TYPES = {
    EntityType.FUNCTION,
    EntityType.THUNK,
    EntityType.VTORDISP,
    EntityType.IMPORT,
}


def canonical_callee_name(
    db: EntityDb,
    image_id: ImageId,
    entity: ReccmpEntity,
    equivalence_groups: dict[int, int] | None = None,
) -> str | None:
    """Display name plus a stable identity for a callable entity.

    Names remain useful diagnostics, but are not identities: FID guesses and
    local wrapper names can disagree, while unrelated functions can share a
    display name.  Prefer an explicitly configured original-address alias,
    then a matched pair, a decorated symbol/import, and finally a side-local
    opaque identity that cannot accidentally compare equal across images.
    """
    if entity.entity_type not in _CALLABLE_TYPES:
        return entity.match_name()

    canonical_entity = entity
    entity_addr = entity.addr(image_id)
    discovered_orig = (
        db.alias_canonical_orig(image_id, entity_addr)
        if entity_addr is not None
        else None
    )
    canonical_orig = (
        discovered_orig if discovered_orig is not None else entity.orig_addr
    )
    if discovered_orig is not None:
        discovered_entity = db.get(ImageId.ORIG, discovered_orig, exact=True)
        if discovered_entity is not None:
            canonical_entity = discovered_entity
    configured_alias = False
    if canonical_orig is not None and equivalence_groups:
        configured_orig = equivalence_groups.get(canonical_orig)
        configured_alias = configured_orig is not None
        canonical_orig = (
            configured_orig if configured_orig is not None else canonical_orig
        )
        configured_entity = db.get(ImageId.ORIG, canonical_orig, exact=True)
        if configured_entity is not None:
            canonical_entity = configured_entity

    symbol = canonical_entity.get("symbol") or entity.get("symbol")
    if canonical_orig is not None and (
        configured_alias or entity.matched or canonical_entity.matched
    ):
        # A proven pair (or a configured equivalence-group alias) is a stronger
        # identity than either side's local decorated symbol: the two images can
        # spell the same callee differently while a match proves they are one
        # entity. Resolve display through the canonical original entity too.
        original_entity = db.get(ImageId.ORIG, canonical_orig, exact=True)
        if original_entity is not None:
            canonical_entity = original_entity
        identity = f"orig:{canonical_orig:x}"
        display = canonical_entity.match_name() or entity.match_name()
    elif symbol:
        identity = f"symbol:{symbol}"
        display = f"{symbol} ({EntityTypeLookup.get(entity.entity_type or -1, 'UNK')})"
    elif entity.entity_type == EntityType.IMPORT:
        identity = f"import:{canonical_entity.best_name() or entity.best_name()}"
        display = canonical_entity.match_name() or entity.match_name()
    elif canonical_orig is not None and entity.addr(image_id) is None:
        identity = f"orig:{canonical_orig:x}"
        display = canonical_entity.match_name() or entity.match_name()
    else:
        addr = entity.addr(image_id)
        assert addr is not None
        identity = f"{image_id.name.lower()}:{addr:x}"
        display = canonical_entity.match_name() or entity.match_name()
    if display is None:
        return None
    return f"{display} [CALLEE {identity}]"


def entity_proof_identity(
    db: EntityDb,
    image_id: ImageId,
    entity: ReccmpEntity,
    offset: int = 0,
    equivalence_groups: dict[int, int] | None = None,
) -> Hashable:
    """Structured proof identity for a resolved entity.

    Display names are diagnostics. Matched or aliased entities share the
    canonical original address. Unmatched data, strings, and locals stay
    side-local so identical names cannot prove correspondence.
    """
    # pylint: disable=too-many-return-statements
    if entity.entity_type == EntityType.IMPORT_THUNK:
        # `jmp [__imp_X]`: calling it is calling through that slot.
        ref = entity.get("ref_orig" if image_id == ImageId.ORIG else "ref_recomp")
        slot = db.get(image_id, ref) if ref is not None else None
        if slot is not None and slot.entity_type == EntityType.IMPORT:
            return (
                "jmp_through",
                entity_proof_identity(db, image_id, slot, 0, equivalence_groups),
            )
    if entity.entity_type in _CALLABLE_TYPES:
        entity_addr = entity.addr(image_id)
        discovered_orig = (
            db.alias_canonical_orig(image_id, entity_addr)
            if entity_addr is not None
            else None
        )
        canonical_orig = (
            discovered_orig if discovered_orig is not None else entity.orig_addr
        )
        configured_alias = False
        if canonical_orig is not None and equivalence_groups:
            configured_orig = equivalence_groups.get(canonical_orig)
            configured_alias = configured_orig is not None
            canonical_orig = (
                configured_orig if configured_orig is not None else canonical_orig
            )
        symbol = entity.get("symbol")
        # alias_canonical_orig covers real pairs and proven aliases alike.
        if canonical_orig is not None and (
            configured_alias or entity.matched or discovered_orig is not None
        ):
            return ("entity", canonical_orig, offset)
        if entity.entity_type == EntityType.IMPORT:
            return ("import", entity.best_name() or entity.get("symbol"))
        if symbol:
            return ("symbol", symbol, offset)
        addr = entity.addr(image_id)
        assert addr is not None
        return ("unmatched", image_id.name.lower(), addr, offset)

    entity_addr = entity.addr(image_id)
    discovered_orig = (
        db.alias_canonical_orig(image_id, entity_addr)
        if entity_addr is not None
        else None
    )
    canonical_orig = (
        discovered_orig if discovered_orig is not None else entity.orig_addr
    )
    configured_alias = False
    if canonical_orig is not None and equivalence_groups:
        configured_orig = equivalence_groups.get(canonical_orig)
        configured_alias = configured_orig is not None
        canonical_orig = (
            configured_orig if configured_orig is not None else canonical_orig
        )
    if canonical_orig is not None and (
        configured_alias or entity.matched or discovered_orig is not None
    ):
        return ("entity", canonical_orig, offset)
    addr = entity.addr(image_id)
    assert addr is not None
    return ("unmatched", image_id.name.lower(), addr, offset)


def _resolve_raw_jump_entity(
    db: EntityDb,
    image_id: ImageId,
    addr: int,
    jump_target: Callable[[int], int | None] | None,
) -> ReccmpEntity | None:
    """Follow raw E9 chains whose intermediate islands have no entity row."""
    if jump_target is None:
        return None
    seen: set[int] = set()
    current = addr
    for _ in range(8):
        if current in seen:
            return None
        seen.add(current)
        target = jump_target(current)
        if target is None:
            break
        current = target
    if current == addr:
        return None
    entity = db.get(image_id, current, exact=True)
    if entity is None or entity.entity_type not in _CALLABLE_TYPES:
        return None
    return entity


def create_name_lookup(
    db: EntityDb,
    image_id: ImageId,
    bin_read: Callable[[int], int | None],
    offset_name: Callable[[CvdumpTypeKey, int], str],
    equivalence_groups: dict[int, int] | None = None,
    *,
    jump_target: Callable[[int], int | None] | None = None,
) -> NameReplacementProtocol:
    # pylint: disable=too-many-statements
    """Function generator for name replacement"""
    assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

    def follow_indirect(pointer: int) -> ReccmpEntity | None:
        """Read the pointer address and open the entity (if it exists) at the indirect location."""
        addr = bin_read(pointer)
        if addr is not None:
            return db.get(image_id, addr, exact=True)

        return None

    ref_key = "ref_orig" if image_id == ImageId.ORIG else "ref_recomp"

    # Build the point-query index lazily: FunctionComparator is constructed
    # before entity ingestion populates the shared database. A paired
    # containing object is stronger evidence than a nearer unpaired interior
    # annotation. Prefix maximum ends let each lookup stop as soon as no
    # earlier interval can contain the requested address.
    @cache
    def paired_index(
        _gen: int = 0,
    ) -> tuple[list[tuple[int, int, ReccmpEntity]], list[int], list[int]]:
        del _gen
        paired_ranges: list[tuple[int, int, ReccmpEntity]] = []
        for candidate in db.all(image_id):
            base = candidate.addr(image_id)
            size = candidate.any_size(image_id)
            if (
                candidate.matched
                and candidate.entity_type in (EntityType.DATA, EntityType.OFFSET)
                and base is not None
                and size > 0
            ):
                paired_ranges.append((base, base + size, candidate))
        paired_ranges.sort(key=lambda item: item[0])
        paired_starts = [item[0] for item in paired_ranges]
        paired_max_ends: list[int] = []
        maximum = 0
        for _, end, _ in paired_ranges:
            maximum = max(maximum, end)
            paired_max_ends.append(maximum)
        return paired_ranges, paired_starts, paired_max_ends

    def paired_containing(addr: int) -> ReccmpEntity | None:
        paired_ranges, paired_starts, paired_max_ends = paired_index(db.generation)
        candidates: list[tuple[int, int, ReccmpEntity]] = []
        index = bisect.bisect_right(paired_starts, addr) - 1
        while index >= 0 and paired_max_ends[index] > addr:
            start, end, candidate = paired_ranges[index]
            if start <= addr < end:
                other_image = (
                    ImageId.RECOMP if image_id == ImageId.ORIG else ImageId.ORIG
                )
                other_size = candidate.size(other_image)
                if other_size is not None and addr - start < other_size:
                    candidates.append((start, end, candidate))
            index -= 1
        if not candidates:
            return None

        # The smallest paired object is the most specific ownership claim;
        # original address makes the choice deterministic across both images.
        def specificity(item: tuple[int, int, ReccmpEntity]) -> tuple[int, int]:
            entity = item[2]
            paired_size = max(
                entity.size(ImageId.ORIG) or 0,
                entity.size(ImageId.RECOMP) or 0,
            )
            return (paired_size, entity.orig_addr or 0)

        return min(candidates, key=specificity)[2]

    def equivalence_canonical_name(entity: ReccmpEntity) -> str | None:
        """The canonical group member's name, when the entity's original
        address belongs to a configured equivalence group (fold islands,
        per-TU duplicate COMDATs). Both images canonicalize through the same
        orig-address groups, so a reference to any group member on either
        side emits the identical name and compares equal."""
        if not equivalence_groups:
            return None
        orig_addr = entity.orig_addr
        if orig_addr is None:
            return None
        canonical = equivalence_groups.get(orig_addr)
        if canonical is None:
            return None
        canonical_entity = db.get(ImageId.ORIG, canonical, exact=True)
        if canonical_entity is None:
            return None
        return canonical_callee_name(db, image_id, canonical_entity, equivalence_groups)

    def get_name(entity: ReccmpEntity, offset: int = 0) -> str | None:
        """The offset is the difference between the input search address and the entity's
        starting address. Decide whether to return the base name (match_name) or
        a string with the base name plus the offset.
        Returns None if there is no suitable name."""
        if offset == 0:
            # Resolve jmp thunks (e.g. incremental-link table entries) to their
            # target so a call through the thunk compares equal to a direct call.
            if entity.entity_type == EntityType.THUNK:
                ref_addr = entity.get(ref_key)
                if isinstance(ref_addr, int):
                    target = db.get(image_id, ref_addr, exact=True)
                    if target is not None and target.entity_type == EntityType.FUNCTION:
                        return equivalence_canonical_name(
                            target
                        ) or canonical_callee_name(
                            db, image_id, target, equivalence_groups
                        )
            return equivalence_canonical_name(entity) or canonical_callee_name(
                db, image_id, entity, equivalence_groups
            )

        # We will not return an offset name if this is not a variable
        # or if the offset is outside the range of the entity.
        if entity.entity_type not in (
            EntityType.DATA,
            EntityType.OFFSET,
        ) or offset >= entity.any_size(image_id):
            return None

        type_key = entity.get("data_type")
        if type_key:
            suffix = offset_name(CvdumpTypeKey(type_key), offset)
            return entity.match_name(suffix)

        return entity.match_name(f"+{offset}")

    def follow_thunk(entity: ReccmpEntity) -> ReccmpEntity:
        if entity.entity_type != EntityType.THUNK:
            return entity
        ref_addr = entity.get(ref_key)
        if isinstance(ref_addr, int):
            target = db.get(image_id, ref_addr, exact=True)
            if target is not None and target.entity_type == EntityType.FUNCTION:
                return target
        return entity

    def resolve_entity(
        addr: int, exact: bool = False, indirect: bool = False
    ) -> tuple[ReccmpEntity, int] | None:
        # pylint: disable=too-many-return-statements
        if indirect:
            entity = db.get(image_id, addr, exact=True)
            if entity is not None:
                if entity.entity_type == EntityType.DATA:
                    return entity, 0
                if entity.entity_type == EntityType.IMPORT:
                    return entity, 0
            entity = follow_indirect(addr)
            if entity is None:
                return None
            return entity, 0

        entity = db.get(image_id, addr, exact=True)
        if entity is None or entity.entity_type == EntityType.THUNK:
            raw_entity = _resolve_raw_jump_entity(db, image_id, addr, jump_target)
            if raw_entity is not None:
                return raw_entity, 0

        if not exact:
            entity = (
                paired_containing(addr) or entity or db.get(image_id, addr, exact=False)
            )

        if entity is None:
            return None

        base_addr = entity.addr(image_id)
        assert base_addr is not None
        return entity, addr - base_addr

    @cache
    def lookup_cached(_gen: int, addr: int, exact: bool, indirect: bool) -> str | None:
        del _gen
        resolved = resolve_entity(addr, exact=exact, indirect=indirect)
        if resolved is None:
            return None
        entity, offset = resolved
        return get_name(entity, offset)

    def lookup(addr: int, exact: bool = False, indirect: bool = False) -> str | None:
        """Returns the name that represents the entity at the given address.
        If there is no suitable name, return None and let the caller choose one (i.e. placeholder).
        * exact:    If the addr is an offset of an entity (e.g. struct/array) we may return
                    a name like 'variable+8'. If exact is True, return a name only if the entity's addr
                    matches the addr parameter.
        * indirect: If True, the given addr is a pointer so we have the option to read the address
                    from the binary to find the name."""
        return lookup_cached(db.generation, addr, exact, indirect)

    @cache
    def identity_cached(
        _gen: int, addr: int, exact: bool, indirect: bool
    ) -> Hashable | None:
        del _gen
        resolved = resolve_entity(addr, exact=exact, indirect=indirect)
        if resolved is None:
            return None
        entity, offset = resolved
        if offset == 0:
            entity = follow_thunk(entity)
        return entity_proof_identity(db, image_id, entity, offset, equivalence_groups)

    def identity(
        addr: int, exact: bool = False, indirect: bool = False
    ) -> Hashable | None:
        return identity_cached(db.generation, addr, exact, indirect)

    lookup.identity = identity  # type: ignore[attr-defined]
    return lookup
