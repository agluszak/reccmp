"""Known-inline accounting: fingerprint paired helpers and find their
expansions at call sites."""

from reccmp.compare.asm.parse import AsmExcerpt
from reccmp.compare.body_equivalence import BodyEquivalenceMixin
from reccmp.compare.db import ReccmpMatch
from reccmp.compare.inlines import (
    Fingerprint,
    HelperCatalogEntry,
    InlineHit,
    InlineLayoutResult,
    analyze_inline_layout,
    asm_fingerprint_from_ir,
    find_call_sites,
    find_inline_expansions,
    fingerprint_from_asm,
    strip_helper_epilog,
    summarize_helper_effects,
)
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.types import ImageId


class InlineAccountingMixin(BodyEquivalenceMixin):
    """Part of FunctionComparator; relies on its attributes."""

    def _analyze_inline_expansions(
        self,
        match: ReccmpMatch,
        orig_asm: AsmExcerpt,
        recomp_asm: AsmExcerpt,
    ) -> InlineLayoutResult | None:
        """Call-driven inline accounting: only fingerprint helpers named by CALLs."""
        orig_fp = fingerprint_from_asm(orig_asm)
        recomp_fp = fingerprint_from_asm(recomp_asm)
        helpers_by_orig: dict[int, HelperCatalogEntry] = {}
        for fingerprint in (orig_fp, recomp_fp):
            for _index, identities in find_call_sites(fingerprint):
                helper = self._resolve_helper_from_call_identities(identities)
                if helper is None or helper.orig_addr == match.orig_addr:
                    continue
                helpers_by_orig[helper.orig_addr] = helper
        if not helpers_by_orig:
            return None

        # Uniqueness among the call-referenced helpers only.
        freq: dict[Fingerprint, int] = {}
        for entry in helpers_by_orig.values():
            freq[entry.fingerprint] = freq.get(entry.fingerprint, 0) + 1
        helpers = [
            HelperCatalogEntry(
                orig_addr=entry.orig_addr,
                recomp_addr=entry.recomp_addr,
                name=entry.name,
                fingerprint=entry.fingerprint,
                byte_size=entry.byte_size,
                uniqueness=1.0 / freq[entry.fingerprint],
                effect_summary=entry.effect_summary,
            )
            for entry in helpers_by_orig.values()
        ]
        result = analyze_inline_layout(
            orig_asm,
            recomp_asm,
            helpers,
            exclude_orig_addrs=(match.orig_addr,),
        )
        if not result.expansions:
            return None
        return result

    def _helper_entry_for_match(self, entity: ReccmpMatch) -> HelperCatalogEntry | None:
        """Lazily fingerprint one paired helper (memoized in the catalog map)."""
        memo = getattr(self, "_helper_by_orig", None)
        if memo is None:
            self._helper_by_orig = {}
            memo = self._helper_by_orig
        if entity.orig_addr in memo:
            return memo[entity.orig_addr]

        recomp_size = entity.size(ImageId.RECOMP)
        if recomp_size is None or recomp_size <= 0:
            memo[entity.orig_addr] = None
            return None
        try:
            raw = self.recomp_bin.read(entity.recomp_addr, recomp_size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            memo[entity.orig_addr] = None
            return None
        excerpt = self.recomp_sanitize.parse_asm(raw, entity.recomp_addr)
        fingerprint = asm_fingerprint_from_ir(excerpt)
        needle = strip_helper_epilog(fingerprint)
        if len(needle) < 3:
            memo[entity.orig_addr] = None
            return None
        name = entity.best_name() or f"sub_{entity.orig_addr:x}"
        entry = HelperCatalogEntry(
            orig_addr=entity.orig_addr,
            recomp_addr=entity.recomp_addr,
            name=name,
            fingerprint=needle,
            byte_size=recomp_size,
            effect_summary=summarize_helper_effects(needle),
        )
        memo[entity.orig_addr] = entry
        return entry

    def _ensure_helper_identity_index(self) -> None:
        """Build exact-identity maps for call-driven helper resolution.

        Ambiguous names (overloads / collisions) are recorded but never chosen.
        Substring matching is not used for modulo-inline accounting.
        """
        if self._helper_identity_index is not None:
            return
        index: dict[str, int] = {}
        ambiguous: set[str] = set()

        def add_key(key: str, orig_addr: int) -> None:
            if not key or key in ambiguous:
                return
            existing = index.get(key)
            if existing is None:
                index[key] = orig_addr
            elif existing != orig_addr:
                del index[key]
                ambiguous.add(key)

        for entity in self.db.get_functions():
            add_key(f"{entity.orig_addr:#x}", entity.orig_addr)
            add_key(f"{entity.recomp_addr:#x}", entity.orig_addr)
            add_key(f"{entity.orig_addr:x}", entity.orig_addr)
            add_key(f"{entity.recomp_addr:x}", entity.orig_addr)
            name = entity.best_name() or ""
            if name:
                add_key(name, entity.orig_addr)

        self._helper_identity_index = index
        self._helper_identity_ambiguous = ambiguous

    def _ensure_helper_catalog(self) -> list[HelperCatalogEntry]:
        """Full catalog for ``find-inlines`` only — not used on the hot compare path."""
        if self._helper_catalog is not None:
            return self._helper_catalog

        catalog: list[HelperCatalogEntry] = []
        freq: dict[Fingerprint, int] = {}
        for entity in self.db.get_functions():
            entry = self._helper_entry_for_match(entity)
            if entry is None:
                continue
            catalog.append(entry)
            freq[entry.fingerprint] = freq.get(entry.fingerprint, 0) + 1
        # Attach inverse-frequency uniqueness.
        catalog = [
            HelperCatalogEntry(
                orig_addr=entry.orig_addr,
                recomp_addr=entry.recomp_addr,
                name=entry.name,
                fingerprint=entry.fingerprint,
                byte_size=entry.byte_size,
                uniqueness=1.0 / freq[entry.fingerprint],
                effect_summary=entry.effect_summary,
            )
            for entry in catalog
        ]
        catalog.sort(key=lambda entry: len(entry.fingerprint), reverse=True)
        self._helper_catalog = catalog
        return catalog

    def _resolve_helper_from_call_identities(
        self, identities: set[str]
    ) -> HelperCatalogEntry | None:
        """Map sanitized call-operand identities to a paired helper."""
        self._ensure_helper_identity_index()
        assert self._helper_identity_index is not None
        assert self._helper_identity_ambiguous is not None

        matched_orig: set[int] = set()
        for identity in identities:
            if identity in self._helper_identity_ambiguous:
                continue
            orig_addr = self._helper_identity_index.get(identity)
            if orig_addr is not None:
                matched_orig.add(orig_addr)
        if len(matched_orig) != 1:
            return None
        entity = self.db.get_one_match(next(iter(matched_orig)))
        if entity is None:
            return None
        return self._helper_entry_for_match(entity)

    def find_inlines(self, helper: ReccmpMatch) -> list[InlineHit]:
        """Search original functions for probable expansions of ``helper``'s body."""
        helper_size = helper.size(ImageId.RECOMP)
        if helper_size is None or helper_size <= 0:
            return []
        helper_fp = self._alias_fingerprint(
            ImageId.RECOMP, helper.recomp_addr, helper_size
        )
        if helper_fp is None:
            return []

        hosts: list[tuple[int, str, int]] = []
        for entity in self.db.get_functions():
            if entity.orig_addr == helper.orig_addr:
                continue
            size = entity.size(ImageId.ORIG)
            if size is None or size <= helper_size:
                continue
            name = entity.best_name() or f"sub_{entity.orig_addr:x}"
            hosts.append((entity.orig_addr, name, size))

        return find_inline_expansions(
            helper_fp,
            hosts,
            lambda addr, size: self._alias_fingerprint(ImageId.ORIG, addr, size),
        )
