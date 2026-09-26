import re

from reccmp.types import EntityType
from reccmp.compare.db import EntityDb, ReccmpEntity
from reccmp.compare.lines import LinesDb
from reccmp.compare.event import (
    ReccmpEvent,
    ReccmpReportProtocol,
    reccmp_report_nop,
)
from reccmp.compare.equivalence import canonical_orig_addr
from reccmp.compare.queries import get_referencing_entity_matches
from reccmp.types import ImageId


class EntityIndex:
    """One-to-many index. Maps string value to address."""

    _dict: dict[str, list[int]]

    def __init__(self) -> None:
        self._dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def add(self, key: str, value: int):
        self._dict.setdefault(key, []).append(value)

    def get(self, key: str) -> list[int]:
        return self._dict.get(key, [])

    def count(self, key: str) -> int:
        return len(self._dict.get(key, []))

    def pop(self, key: str) -> int:
        value = self._dict[key].pop(0)
        if len(self._dict[key]) == 0:
            del self._dict[key]

        return value


def match_symbols(
    db: EntityDb,
    report: ReccmpReportProtocol = reccmp_report_nop,
    *,
    truncate: bool = False,
):
    """Match all entities with the 'symbol' attribute set. We expect this value to be unique."""

    symbol_index = EntityIndex()

    for ent in db.unmatched(ImageId.RECOMP):
        symbol = ent.get("symbol")
        if not symbol:
            continue

        # Truncate symbol to 255 chars for older MSVC. See also: Warning C4786.
        if truncate:
            symbol = symbol[:255]

        assert ent.recomp_addr is not None
        symbol_index.add(symbol, ent.recomp_addr)

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.ORIG):
            assert ent.orig_addr is not None
            symbol = ent.get("symbol")

            if not symbol:
                continue

            # Repeat the truncate for our match search
            if truncate:
                symbol = symbol[:255]

            if symbol in symbol_index:
                recomp_addr = symbol_index.pop(symbol)

                # If match was not unique:
                if symbol in symbol_index:
                    report(
                        ReccmpEvent.NON_UNIQUE_SYMBOL,
                        ent.orig_addr,
                        msg=f"Matched 0x{ent.orig_addr:x} using non-unique symbol '{symbol}'",
                    )

                batch.match(ent.orig_addr, recomp_addr)

            else:
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    msg=f"Failed to match at 0x{ent.orig_addr:x} with symbol '{symbol}'",
                )


_ELABORATED_ARGUMENT = re.compile(r"([<,])\s*(?:class|struct|union|enum)\s+")
_ARGUMENT_SPACE = re.compile(r"\s*,\s*")
_CLOSING_SPACE = re.compile(r">\s+(?=>)")


def match_name(name: str) -> str:
    """One comparable spelling for demangled-name matching.

    MSVC's demangler spells template arguments loosely: a space before a
    pointer or reference sigil (``T *>``), the elaborated keyword of a class
    argument (``<T, class U, 1>``), a space after each comma and between
    closing brackets (``> >``). Annotation-side names spell them tight
    (``<T,U,1>``, ``T*>``, ``>>``). None of it carries identity, so names
    match on the tight form."""
    name = name.replace(" *", "*").replace(" &", "&")
    name = _ELABORATED_ARGUMENT.sub(r"\1", name)
    name = _ARGUMENT_SPACE.sub(",", name)
    return _CLOSING_SPACE.sub(">", name)


def match_functions(
    db: EntityDb,
    report: ReccmpReportProtocol = reccmp_report_nop,
    *,
    truncate: bool = False,
    equivalence_groups: dict[int, int] | None = None,
):
    """Match functions by name only when the identity is unique on both sides.

    Multiple original addresses may carry the same name when the original
    binary emitted the same body more than once (e.g. a per-TU COMDAT copy).
    If the project declares those addresses equivalent in an
    ``equivalence-groups`` file, they count as a single identity here: the
    canonical member takes the real match and the other members become
    original-side aliases of it. Distinct (non-equivalent) bodies that merely
    share a name are still reported as ambiguous."""
    groups = equivalence_groups or {}

    recomp_symbols: dict[int, str] = {}
    name_index = EntityIndex()

    for ent in db.unmatched(ImageId.RECOMP):
        symbol = ent.get("symbol")
        name = ent.get("name")
        if ent.get("type") and ent.get("type") != EntityType.FUNCTION:
            continue
        if not name:
            continue
        if truncate:
            name = name[:255]
        name = match_name(name)
        assert ent.recomp_addr is not None
        name_index.add(name, ent.recomp_addr)
        if symbol is not None:
            recomp_symbols[ent.recomp_addr] = symbol

    orig_entities = [
        ent
        for ent in db.unmatched(ImageId.ORIG)
        if ent.get("type") == EntityType.FUNCTION and ent.get("name")
    ]
    orig_by_addr: dict[int, ReccmpEntity] = {}
    orig_name_identities: dict[str, set[int]] = {}
    normalized_names: dict[int, str] = {}
    for ent in orig_entities:
        assert ent.orig_addr is not None
        name = ent.get("name")
        assert isinstance(name, str)
        if truncate:
            name = name[:255]
        name = match_name(name)
        normalized_names[ent.orig_addr] = name
        orig_by_addr[ent.orig_addr] = ent
        orig_name_identities.setdefault(name, set()).add(
            canonical_orig_addr(groups, ent.orig_addr)
        )

    # For names that resolve to a single original identity, decide which
    # entity owns the real match. Normally that is the entity itself; for an
    # equivalence group it is the canonical member. If the canonical is not
    # an unmatched entity under this name (already matched elsewhere or not
    # an entity at all), the lowest-addressed member takes the match instead.
    name_owners: dict[str, int] = {}
    for name, identities in orig_name_identities.items():
        if len(identities) != 1:
            continue
        canonical = next(iter(identities))
        if db.get_one_match(canonical) is not None:
            name_owners[name] = canonical
            continue
        owner = orig_by_addr.get(canonical)
        if owner is None or normalized_names.get(canonical) != name:
            owner = min(
                (
                    e
                    for e in orig_entities
                    if normalized_names.get(e.orig_addr or 0) == name
                ),
                key=lambda e: e.orig_addr or 0,
            )
        name_owners[name] = owner.orig_addr or 0

    pending_aliases: list[tuple[int, int]] = []

    with db.batch() as batch:
        for ent in orig_entities:
            assert ent.orig_addr is not None
            name = normalized_names[ent.orig_addr]
            identities = orig_name_identities[name]

            if len(identities) == 1 and ent.orig_addr != name_owners[name]:
                # Equivalent duplicate: the owner takes the real match and
                # this address becomes an original-side alias of it.
                pending_aliases.append((ent.orig_addr, name_owners[name]))
                continue

            candidates = name_index.get(name)
            if not candidates:
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    msg=f"Failed to match function at 0x{ent.orig_addr:x} with name '{name}'",
                )
                continue

            if len(identities) != 1 or len(candidates) != 1:
                symbols = [recomp_symbols.get(addr, "None") for addr in candidates]
                report(
                    ReccmpEvent.AMBIGUOUS_MATCH,
                    ent.orig_addr,
                    msg=f"Ambiguous function name '{name}' has "
                    f"{len(identities)} original and {len(candidates)} recomp candidates:\n"
                    + ",\n".join(f"'{symbol}'" for symbol in symbols),
                )
                continue

            batch.match(ent.orig_addr, name_index.pop(name))

    for member_addr, owner_addr in pending_aliases:
        # Aliases require a real canonical match. If the owner could not be
        # matched its own report already covers the group.
        if db.get_one_match(owner_addr) is None:
            continue
        if not db.set_alias(ImageId.ORIG, member_addr, owner_addr):
            report(
                ReccmpEvent.NO_MATCH,
                member_addr,
                msg=(
                    f"Could not alias equivalent original 0x{member_addr:x} to "
                    f"0x{owner_addr:x}"
                ),
            )


def _find_vtable_match(
    class_name: str, base_class: str | None, vtable_name_index: EntityIndex
) -> int | None:
    """Try to resolve a single class_name/base_class candidate against the
    recomp vtable name index."""

    # Most classes will not use multiple inheritance, so try the regular vtable
    # first, unless a base class is provided.
    if base_class is None or base_class == class_name:
        bare_vftable = match_name(f"{class_name}::`vftable'")

        if bare_vftable in vtable_name_index:
            return vtable_name_index.pop(bare_vftable)

    # If we didn't find a match above, search for the multiple inheritance vtable.
    for_name = base_class if base_class is not None else class_name
    for_vftable = match_name(f"{class_name}::`vftable'{{for `{for_name}'}}")

    if for_vftable in vtable_name_index:
        return vtable_name_index.pop(for_vftable)

    return None


def match_vtables(db: EntityDb, report: ReccmpReportProtocol = reccmp_report_nop):
    """The requirements for matching are:
    1.  Recomp entity has name attribute in this format: "Pizza::`vftable'"
        This is derived from the symbol: "??_7Pizza@@6B@"
    2.  Orig entity has name attribute with class name only. (e.g. "Pizza")
    3.  If multiple inheritance is used, the orig entity has the base_class attribute set.

    For multiple inheritance, the vtable name references the base class like this:

        - X::`vftable'{for `Y'}

    The vtable for the derived class will take one of these forms:

        - X::`vftable'{for `X'}
        - X::`vftable'

    We assume only one of the above will appear for a given class."""

    vtable_name_index = EntityIndex()

    for ent in db.unmatched(ImageId.RECOMP):
        name = ent.get("name")
        if not name or ent.get("type") != EntityType.VTABLE:
            continue

        assert ent.recomp_addr is not None
        vtable_name_index.add(match_name(name), ent.recomp_addr)

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.ORIG):
            class_name = ent.get("name")
            if (
                not class_name
                or ent.get("type") != EntityType.VTABLE
                or ent.get("inferred_vtable")
            ):
                continue

            assert ent.orig_addr is not None

            base_class = ent.get("base_class")
            candidates = ent.get("folded_vtables") or [(class_name, base_class)]

            for candidate_name, candidate_base_class in candidates:
                recomp_addr = _find_vtable_match(
                    candidate_name, candidate_base_class, vtable_name_index
                )
                if recomp_addr is not None:
                    batch.match(ent.orig_addr, recomp_addr)
                    break
            else:
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    msg=f"Failed to match vtable at 0x{ent.orig_addr:x} for class '{class_name}' (base={base_class or 'None'})",
                )


def match_static_variables(
    db: EntityDb, report: ReccmpReportProtocol = reccmp_report_nop
):
    """To match a static variable, we need the following:
    1. Orig entity function with symbol
    2. Orig entity variable with:
        - name = name of variable
        - static_var = True
        - parent_function = orig address of function
    3. Recomp entity for the static variable with symbol

    Requirement #1 is most likely to be met by matching the entity with recomp data.
    Therefore, this function should be called after match_symbols or match_functions."""
    symbols = {}

    for recomp_ent in db.unmatched(ImageId.RECOMP):
        if recomp_ent.get("type") and recomp_ent.get("type") != EntityType.DATA:
            continue

        recomp_sym = recomp_ent.get("symbol")
        if not recomp_sym:
            continue

        assert recomp_ent.recomp_addr is not None
        symbols[recomp_ent.recomp_addr] = recomp_sym

    with db.batch() as batch:
        for variable_ent in db.unmatched(ImageId.ORIG):
            variable_addr = variable_ent.orig_addr
            assert variable_addr is not None

            if not variable_ent.get("static_var"):
                continue

            variable_name = variable_ent.get("name")
            if not variable_name:
                continue

            function_name = None
            function_symbol = None

            parent_addr = variable_ent.get("parent_function")
            if parent_addr:
                parent_ent = db.get(ImageId.ORIG, parent_addr)
                if parent_ent is not None:
                    function_name = parent_ent.get("name")
                    function_symbol = parent_ent.get("symbol")

            # If we could not find the parent function, or if it has no symbol:
            if function_symbol is None:
                report(
                    ReccmpEvent.NO_MATCH,
                    variable_addr,
                    msg=f"No function for static variable '{variable_name}'",
                )
                continue

            for recomp_addr, recomp_sym in symbols.items():
                if function_symbol in recomp_sym and variable_name in recomp_sym:
                    batch.match(variable_addr, recomp_addr)
                    del symbols[recomp_addr]
                    break
            else:
                report(
                    ReccmpEvent.NO_MATCH,
                    variable_addr,
                    msg=f"Failed to match static variable {variable_name} from function {function_name} annotated with 0x{variable_addr:x}",
                )


def match_variables(db: EntityDb, report: ReccmpReportProtocol = reccmp_report_nop):
    var_name_index = EntityIndex()

    # TODO: We allow a match if entity_type is null.
    # This can be removed if we can more confidently declare a symbol is a variable
    # when adding from the PDB.
    for ent in db.unmatched(ImageId.RECOMP):
        if ent.get("type") and ent.get("type") != EntityType.DATA:
            continue

        name = ent.get("name")
        if not name:
            continue

        assert ent.recomp_addr is not None
        var_name_index.add(name, ent.recomp_addr)

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.ORIG):
            if ent.get("type") != EntityType.DATA:
                continue

            name = ent.get("name")
            if not name:
                continue

            if ent.get("static_var"):
                continue

            assert ent.orig_addr is not None

            if name in var_name_index:
                recomp_addr = var_name_index.pop(name)
                batch.match(ent.orig_addr, recomp_addr)
            else:
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    msg=f"Failed to match variable {name} at 0x{ent.orig_addr:x}",
                )


def match_strings(db: EntityDb, report: ReccmpReportProtocol = reccmp_report_nop):
    string_index = EntityIndex()

    for ent in db.unmatched(ImageId.RECOMP):
        if ent.get("type") not in (EntityType.STRING, EntityType.WIDECHAR):
            continue

        text = ent.get("name")
        if not text:
            continue

        assert ent.recomp_addr is not None
        string_index.add(text, ent.recomp_addr)

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.ORIG):
            if ent.get("type") not in (EntityType.STRING, EntityType.WIDECHAR):
                continue

            text = ent.get("name")
            if not text:
                continue

            verified = ent.get("verified", False)
            assert ent.orig_addr is not None

            if text in string_index:
                recomp_addr = string_index.pop(text)
                batch.match(ent.orig_addr, recomp_addr)
            elif verified:
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    msg=f"Failed to match string {text} at 0x{ent.orig_addr:x}",
                )


def classify_exact_string_aliases(db: EntityDb) -> None:
    """Record side-local duplicate strings once a unique canonical pair exists."""
    canonical: dict[tuple[EntityType, str], set[int]] = {}
    for canonical_entity in db.get_matches():
        entity_type = canonical_entity.get("type")
        text = canonical_entity.get("name")
        if entity_type in (EntityType.STRING, EntityType.WIDECHAR) and text:
            canonical.setdefault((entity_type, text), set()).add(
                canonical_entity.orig_addr
            )

    for image_id in (ImageId.ORIG, ImageId.RECOMP):
        for candidate in tuple(db.unexplained(image_id)):
            entity_type = candidate.get("type")
            text = candidate.get("name")
            identities = canonical.get((entity_type, text), set())
            addr = candidate.addr(image_id)
            if addr is not None and len(identities) == 1:
                db.set_alias(image_id, addr, next(iter(identities)))


def match_lines(
    db: EntityDb,
    lines: LinesDb,
    report: ReccmpReportProtocol = reccmp_report_nop,
):
    """
    This function requires access to `cv` and `recomp_bin` because most lines will not have an annotation.
    It would therefore be quite inefficient to load all recomp lines into the `entities` table
    and only match a tiny fraction of them to symbols.
    """

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.ORIG):
            if ent.get("type") != EntityType.LINE:
                continue

            assert ent.orig_addr is not None
            filename = ent.get("filename")
            line = ent.get("line")

            #
            # We only match the line directly below the annotation since not all lines of code result in a debug line, especially if optimizations are turned on.
            # However, this does cause false positives in cases like
            # ```
            # // LINE: TARGET 0x1234
            # // OTHER_ANNOTATION: ...
            # actual_code();
            # ```
            # or
            # ```
            # // LINE: TARGET 0x1234
            #
            # actual_code();
            # ```
            # but it is significantly more effort to detect these false positives.
            #

            # We match `line + 1` since `line` is the comment itself
            for recomp_addr in lines.search_line(filename, line + 1):
                batch.match(ent.orig_addr, recomp_addr)
                break
            else:
                # No results
                report(
                    ReccmpEvent.NO_MATCH,
                    ent.orig_addr,
                    f"Found no matching debug symbol for {filename}:{line}",
                )


def match_ref(
    db: EntityDb,
    report: ReccmpReportProtocol = reccmp_report_nop,
):
    """Matches child entities that refer to the same parent entity.
    Repeats until there are no new matches."""
    new_matches = False

    for _ in range(10):
        new_matches = False
        with db.batch() as batch:
            for orig_addr, recomp_addr in get_referencing_entity_matches(db):
                new_matches = True
                batch.match(orig_addr, recomp_addr)

            if not new_matches:
                break

    # If we did not break out of the loop:
    if new_matches:
        report(
            ReccmpEvent.GENERAL_WARNING,
            -1,
            "Reached maximum iteration depth while matching referencing entities.",
        )


def match_imports(db: EntityDb):
    orig_imports = {}

    # n.b. Case insensitive match here to preserve previous behavior.
    # The final entity will use the name from the recomp side.
    for ent in db.unmatched(ImageId.ORIG):
        if ent.get("type") != EntityType.IMPORT:
            continue

        name = ent.get("name")
        if not name:
            continue

        assert isinstance(ent.orig_addr, int)
        orig_imports[name.upper()] = ent.orig_addr

    with db.batch() as batch:
        for ent in db.unmatched(ImageId.RECOMP):
            if ent.get("type") != EntityType.IMPORT:
                continue

            name = ent.get("name")
            if not name:
                continue

            orig_addr = orig_imports.get(name.upper())
            if orig_addr is not None:
                assert isinstance(ent.recomp_addr, int)
                batch.match(orig_addr, ent.recomp_addr)
