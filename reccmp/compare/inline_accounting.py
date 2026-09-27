"""Known-inline accounting: fingerprint paired helpers and find their
expansions at call sites."""

from collections.abc import Hashable

from reccmp.compare.asm.ir import DecodedInstruction, ExtentKind
from reccmp.compare.asm.replacement import entity_proof_identity
from reccmp.compare.body_equivalence import BodyEquivalenceMixin
from reccmp.compare.db import ReccmpMatch
from reccmp.compare.inlines import (
    Fingerprint,
    HelperCatalogEntry,
    InlineHit,
    InlineLayoutResult,
    analyze_inline_layout,
    find_call_sites,
    find_inline_expansions,
    fingerprint_of,
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
        orig_asm: list[DecodedInstruction],
        recomp_asm: list[DecodedInstruction],
    ) -> InlineLayoutResult | None:
        """Call-driven inline accounting: only fingerprint helpers named by CALLs."""
        helpers_by_orig: dict[int, HelperCatalogEntry] = {}
        for rows in (orig_asm, recomp_asm):
            for _index, callee in find_call_sites(fingerprint_of(rows)):
                helper = self._helper_called(callee)
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
                identity=entry.identity,
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
        image = self._load_function_image(
            ImageId.RECOMP, raw, entity.recomp_addr, ExtentKind.KNOWN
        )
        needle = strip_helper_epilog(fingerprint_of(image.instructions))
        if len(needle) < 3:
            memo[entity.orig_addr] = None
            return None
        name = entity.best_name() or f"sub_{entity.orig_addr:x}"
        entry = HelperCatalogEntry(
            orig_addr=entity.orig_addr,
            recomp_addr=entity.recomp_addr,
            name=name,
            fingerprint=needle,
            identity=entity_proof_identity(self.db, ImageId.RECOMP, entity),
            byte_size=recomp_size,
            effect_summary=summarize_helper_effects(needle),
        )
        memo[entity.orig_addr] = entry
        return entry

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
                identity=entry.identity,
                byte_size=entry.byte_size,
                uniqueness=1.0 / freq[entry.fingerprint],
                effect_summary=entry.effect_summary,
            )
            for entry in catalog
        ]
        catalog.sort(key=lambda entry: len(entry.fingerprint), reverse=True)
        self._helper_catalog = catalog
        return catalog

    def _helper_called(self, callee: Hashable) -> HelperCatalogEntry | None:
        """The paired helper a call's callee identity names, if any."""
        match callee:
            case ("entity", int() as orig_addr, 0):
                entity = self.db.get_one_match(orig_addr)
                return self._helper_entry_for_match(entity) if entity else None
        return None

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
