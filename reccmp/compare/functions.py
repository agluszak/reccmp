import dataclasses
from dataclasses import dataclass, field
from functools import cache
import struct
import re
from itertools import pairwise
from typing import Callable, Iterator
from reccmp.compare.lines import LinesDb
from reccmp.compare.thunk_resolve import read_e9_jmp_target
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins
from reccmp.compare.asm.effective import CallAbi, FunctionMetadata
from reccmp.compare.asm.fixes import analyze_effective_match, assert_fixup
from reccmp.compare.asm.const import JUMP_MNEMONICS
from reccmp.compare.asm.instgen import InstructGen, InstructionMeta, SectionType
from reccmp.compare.asm.parse import AsmExcerpt, ParseAsm
from reccmp.compare.asm.replacement import (
    canonical_callee_name,
    create_name_lookup,
)
from reccmp.compare.db import EntityDb, ReccmpMatch
from reccmp.compare.diff import EntityCompareResult, RawDiffOutput
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    ComparisonStatus,
    DifferenceSide,
    FactValue,
)
from reccmp.compare.asm.ir import instruction_match_key
from reccmp.compare.stack_layout import analyze_stack_layout
from reccmp.compare.inlines import (
    Fingerprint,
    HelperCatalogEntry,
    InlineHit,
    InlineLayoutResult,
    analyze_inline_layout,
    asm_fingerprint_from_lines,
    find_inline_expansions,
    strip_helper_epilog,
)
from reccmp.compare.event import ReccmpEvent, ReccmpReportProtocol
from reccmp.compare.equivalence import canonical_orig_addr
from reccmp.cvdump.analysis import CvdumpNode
from reccmp.cvdump.cvinfo import CvdumpTypeKey, CvdumpTypeMap
from reccmp.cvdump.demangler import parse_function_signature
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.cvdump.types import CvdumpKeyError, CvdumpTypesParser
from reccmp.types import EntityType
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.formats import Image, PEImage
from reccmp.types import ImageId

_ISLAND_PADDING = (0x90, 0xCC)  # nop / int3


def _code_instructions(
    raw: bytes, addr: int, is_32bit: bool
) -> list[tuple[int, int, str, str]] | None:
    """The body's raw instruction tuples, or None when the body carries
    non-code sections (jump-table data) that would break the positional
    pairing with the sanitized excerpt."""
    instructions: list[tuple[int, int, str, str]] = []
    for section in InstructGen(bytes(raw), addr, is_32bit).sections:
        if section.type != SectionType.CODE:
            return None
        instructions.extend(section.contents)
    return instructions


def _is_bare_jmp_island(raw: bytes) -> bool:
    """True when the function body is a lone `jmp rel32` followed only by
    padding: the shape an incremental link leaves at a moved/folded symbol's
    old address."""
    if len(raw) < 5 or raw[0] != 0xE9:
        return False
    return all(b in _ISLAND_PADDING for b in raw[5:])


# Register-argument usage by PDB calling convention. cdecl and stdcall
# take all arguments on the stack; thiscall reads the receiver from ecx;
# fastcall reads its first two register-sized arguments from ecx and edx.
_CALL_ABI_BY_CONVENTION = {
    "C Near": CallAbi(uses_ecx=False, uses_edx=False),
    "STD Near": CallAbi(uses_ecx=False, uses_edx=False),
    "ThisCall": CallAbi(uses_ecx=True, uses_edx=False),
    "Fast Near": CallAbi(uses_ecx=True, uses_edx=True),
    # Conventions as recovered from decorated names.
    "cdecl": CallAbi(uses_ecx=False, uses_edx=False),
    "stdcall": CallAbi(uses_ecx=False, uses_edx=False),
    "thiscall": CallAbi(uses_ecx=True, uses_edx=False),
    "fastcall": CallAbi(uses_ecx=True, uses_edx=True),
}

_RETURN_KIND_BY_SIZE = {1: "i8", 2: "i16", 4: "i32", 8: "i64"}


def has_asserts(image: Image) -> bool:
    if isinstance(image, PEImage):
        return image.is_debug

    return False


def create_valid_addr_lookup(
    db: EntityDb,
    image_id: ImageId,
    bin_file: Image,
) -> Callable[[int], bool]:
    """
    Function generator for a lookup whether an address from a call is valid
    (either a relocation or pointing to something else we know, like a global variable)
    """
    assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

    @cache
    def lookup(addr: int) -> bool:
        # Check if in relocation table
        if addr > bin_file.imagebase and bin_file.is_relocated_addr(addr):
            return True

        return db.intersects(image_id, addr)

    return lookup


def create_bin_lookup(bin_file: Image) -> Callable[[int], int | None]:
    """Function generator to read a pointer from the bin file"""

    def lookup(addr: int) -> int | None:
        try:
            (ptr,) = struct.unpack("<L", bin_file.read(addr, 4))
            return ptr
        except (struct.error, InvalidVirtualAddressError, InvalidVirtualReadError):
            return None

    return lookup


@dataclass
class FunctionComparator:
    # pylint: disable=too-many-instance-attributes
    db: EntityDb
    lines_db: LinesDb
    orig_bin: Image
    recomp_bin: Image
    report: ReccmpReportProtocol
    types: CvdumpTypesParser
    is_32bit: bool = True
    # PDB function nodes keyed by recomp address, used to derive return
    # kinds and callee register-argument conventions for the effective-match
    # verifier. Optional: without it the verifier stays fully conservative.
    func_nodes: dict[int, CvdumpNode] = field(default_factory=dict)
    # Proven-equivalent original addresses (member -> canonical): references to
    # any group member sanitize to the canonical name on both sides.
    equivalence_groups: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self._call_abi_cache: dict[str, CallAbi | None] | None = None
        self._fp_cache: dict[tuple[ImageId, int, int], tuple[tuple[str, str], ...] | None] = (
            {}
        )
        self._helper_catalog: list[HelperCatalogEntry] | None = None
        self._helper_by_orig: dict[int, HelperCatalogEntry | None] = {}
        self.orig_sanitize = ParseAsm(
            addr_test=create_valid_addr_lookup(self.db, ImageId.ORIG, self.orig_bin),
            name_lookup=create_name_lookup(
                self.db,
                ImageId.ORIG,
                create_bin_lookup(self.orig_bin),
                self.types.get_name_for_offset,
                self.equivalence_groups,
                jump_target=lambda addr: read_e9_jmp_target(self.orig_bin, addr),
            ),
            is_32bit=self.is_32bit,
            collect_meta=False,
        )
        self.recomp_sanitize = ParseAsm(
            addr_test=create_valid_addr_lookup(
                self.db, ImageId.RECOMP, self.recomp_bin
            ),
            name_lookup=create_name_lookup(
                self.db,
                ImageId.RECOMP,
                create_bin_lookup(self.recomp_bin),
                self.types.get_name_for_offset,
                self.equivalence_groups,
                jump_target=lambda addr: read_e9_jmp_target(self.recomp_bin, addr),
            ),
            is_32bit=self.is_32bit,
            collect_meta=False,
        )

    def _source_ref_of_recomp_addr(self, recomp_addr: int | None) -> str | None:
        if recomp_addr is None:
            return None
        path_line_pair = self.lines_db.find_line_of_recomp_address(recomp_addr)
        if path_line_pair is None:
            return None
        return f"{path_line_pair[0].name}:{path_line_pair[1]}"

    def _source_facts_of_recomp_addr(
        self, recomp_addr: int | None
    ) -> dict[str, FactValue]:
        if recomp_addr is None:
            return {}
        path_line_pair = self.lines_db.find_line_of_recomp_address(recomp_addr)
        if path_line_pair is None:
            return {}
        return {
            "source_path": path_line_pair[0].name,
            "source_line": path_line_pair[1],
        }

    def _enrich_side_with_source(self, side: DifferenceSide) -> DifferenceSide:
        """Attach PDB line info when the side address is a recomp VA."""
        extra = self._source_facts_of_recomp_addr(side.address)
        if not extra:
            return side
        facts = {**side.facts, **extra}
        return DifferenceSide(side.instruction_index, side.address, facts)

    def _enrich_analysis_with_source(
        self, analysis: ComparisonAnalysis
    ) -> ComparisonAnalysis:
        """Pin mismatch / inconclusive locations to recomp source lines."""
        if analysis.status == ComparisonStatus.MISMATCH and analysis.difference is not None:
            diff = analysis.difference
            enriched = ComparisonDifference(
                diff.kind,
                diff.orig,
                self._enrich_side_with_source(diff.recomp),
            )
            return ComparisonAnalysis.mismatch(
                enriched, semantic_similarity=analysis.semantic_similarity
            )
        if (
            analysis.status == ComparisonStatus.INCONCLUSIVE
            and analysis.inconclusive_location is not None
        ):
            return ComparisonAnalysis.inconclusive(
                analysis.inconclusive_reason or "analysis_limit",
                self._enrich_side_with_source(analysis.inconclusive_location),
            )
        return analysis

    def _fn_symbol_entry(self, match: ReccmpMatch | None) -> SymbolsEntry | None:
        if match is None:
            return None
        node = self.func_nodes.get(match.recomp_addr)
        if node is None:
            return None
        return node.symbol_entry

    def compare_function(
        self,
        match: ReccmpMatch,
        *,
        include_diff: bool = True,
        include_exact_diff: bool = True,
    ) -> EntityCompareResult:
        # Detect when the recomp function size would cause us to read
        # enough bytes from the original function that we cross into
        # the next annotated function.
        orig_size = match.size(ImageId.ORIG)
        recomp_size = match.size(ImageId.RECOMP)

        if orig_size is None:
            assert recomp_size is not None
            orig_max = match.max_size(ImageId.ORIG)
            if orig_max is not None:
                orig_size = min(orig_max, recomp_size)
            else:
                orig_size = recomp_size

        assert orig_size is not None and recomp_size is not None

        orig_raw = self.orig_bin.read(match.orig_addr, orig_size)
        recomp_raw = self.recomp_bin.read(match.recomp_addr, recomp_size)

        # It's unlikely that a function other than an adjuster thunk would
        # start with a SUB instruction, so alert to a possible wrong
        # annotation here.
        # There's probably a better place to do this, but we're reading
        # the function bytes here already.
        try:
            if orig_raw[0] == 0x2B and recomp_raw[0] != 0x2B:
                self.report(
                    ReccmpEvent.GENERAL_WARNING,
                    match.orig_addr,
                    f"Possible thunk ({match.name})",
                )
        except IndexError:
            pass

        orig_combined = self.orig_sanitize.parse_asm(orig_raw, match.orig_addr)
        recomp_combined = self.recomp_sanitize.parse_asm(recomp_raw, match.recomp_addr)

        # Check for assert calls only if we expect to find them
        if has_asserts(self.orig_bin):
            assert_fixup(orig_combined)

        if has_asserts(self.recomp_bin):
            assert_fixup(recomp_combined)

        line_annotations = self._collect_line_annotations(recomp_combined)

        split_points = self._compute_split_points(
            orig_combined, recomp_combined, line_annotations
        )

        result = self._compare_function_assembly(
            orig_combined,
            recomp_combined,
            split_points,
            match=match,
            orig_raw=orig_raw,
            recomp_raw=recomp_raw,
            include_diff=include_diff,
            include_exact_diff=include_exact_diff,
        )

        # Folded-symbol island: the original address is a group member whose
        # entire original body is a stale `jmp rel32` island (plus padding)
        # left by an incremental link, while the recomp emits the real body.
        # The equivalence-groups metadata (validated project-side) proves the
        # fold chain lands on an equivalent shared body, so the pair is an
        # effective match, not a source defect.
        if (
            not result.analysis.is_effective
            and match.orig_addr in self.equivalence_groups
            and _is_bare_jmp_island(orig_raw)
        ):
            result = dataclasses.replace(
                result,
                analysis=ComparisonAnalysis.effective(("folded_symbol_alias",)),
            )
            result.refresh_equivalence_level()

        return result

    def _alias_fingerprint(
        self, image_id: ImageId, addr: int, size: int
    ) -> tuple[tuple[str, str], ...] | None:
        """Cheap body shape used only to limit alias-equivalence proofs.

        Address operands are erased only when the image identifies them as a
        relocation or an existing entity.  Ordinary immediates remain in the
        key, so equal-sized functions with different constants are rejected
        before the more expensive proof.  Relocation position and instruction
        shape remain part of the key.
        """
        cache_key = (image_id, addr, size)
        cache = getattr(self, "_fp_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        image = self.orig_bin if image_id == ImageId.ORIG else self.recomp_bin
        valid_addr = create_valid_addr_lookup(self.db, image_id, image)
        try:
            raw = image.read(addr, size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            if cache is not None:
                cache[cache_key] = None
            return None
        instructions = _code_instructions(raw, addr, self.is_32bit)
        if instructions is None:
            if cache is not None:
                cache[cache_key] = None
            return None

        def normalize_operand(operand: str) -> str:
            def replace(match: re.Match[str]) -> str:
                value = int(match.group(0), 16)
                return "<ADDR>" if valid_addr(value) else match.group(0)

            return re.sub(r"0x[0-9a-fA-F]+", replace, operand)

        fingerprint = tuple(
            (mnemonic, normalize_operand(operand))
            for _, _, mnemonic, operand in instructions
        )
        if cache is not None:
            cache[cache_key] = fingerprint
        return fingerprint

    def _helper_entry_for_match(
        self, entity: ReccmpMatch
    ) -> HelperCatalogEntry | None:
        """Lazily fingerprint one paired helper (memoized in the catalog map)."""
        cache = getattr(self, "_helper_by_orig", None)
        if cache is None:
            self._helper_by_orig = {}
            cache = self._helper_by_orig
        if entity.orig_addr in cache:
            return cache[entity.orig_addr]

        recomp_size = entity.size(ImageId.RECOMP)
        if recomp_size is None or recomp_size <= 0:
            cache[entity.orig_addr] = None
            return None
        try:
            raw = self.recomp_bin.read(entity.recomp_addr, recomp_size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            cache[entity.orig_addr] = None
            return None
        excerpt = self.recomp_sanitize.parse_asm(raw, entity.recomp_addr)
        fingerprint = asm_fingerprint_from_lines([line for _, line in excerpt])
        needle = strip_helper_epilog(fingerprint)
        if len(needle) < 3:
            cache[entity.orig_addr] = None
            return None
        name = entity.best_name() or f"sub_{entity.orig_addr:x}"
        entry = HelperCatalogEntry(
            orig_addr=entity.orig_addr,
            recomp_addr=entity.recomp_addr,
            name=name,
            fingerprint=needle,
            byte_size=recomp_size,
        )
        cache[entity.orig_addr] = entry
        return entry

    def _resolve_helper_from_call_identities(
        self, identities: set[str]
    ) -> HelperCatalogEntry | None:
        """Map sanitized call-operand identities to a paired helper."""
        for entity in self.db.get_functions():
            name = entity.best_name() or ""
            candidates = {
                name,
                f"{entity.orig_addr:#x}",
                f"{entity.recomp_addr:#x}",
                f"{entity.orig_addr:x}",
                f"{entity.recomp_addr:x}",
            }
            if not (identities & candidates) and not any(
                name and name in identity for identity in identities
            ):
                continue
            return self._helper_entry_for_match(entity)
        return None

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
            )
            for entry in catalog
        ]
        catalog.sort(key=lambda entry: len(entry.fingerprint), reverse=True)
        self._helper_catalog = catalog
        return catalog

    def _analyze_inline_expansions(
        self,
        match: ReccmpMatch,
        orig_asm: list[str],
        recomp_asm: list[str],
    ) -> InlineLayoutResult | None:
        """Call-driven inline accounting: only fingerprint helpers named by CALLs."""
        from reccmp.compare.inlines import find_call_sites

        orig_fp = asm_fingerprint_from_lines(orig_asm)
        recomp_fp = asm_fingerprint_from_lines(recomp_asm)
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

    def _unpaired_function_candidates(
        self, image_id: ImageId
    ) -> dict[tuple[int, tuple[tuple[str, str], ...]], list[int]]:
        groups: dict[tuple[int, tuple[tuple[str, str], ...]], list[int]] = {}
        for entity in self.db.unexplained(image_id):
            if entity.entity_type != EntityType.FUNCTION:
                continue
            addr = entity.addr(image_id)
            size = entity.size(image_id)
            if addr is None or size is None or size <= 0:
                continue
            fingerprint = self._alias_fingerprint(image_id, addr, size)
            if fingerprint is not None:
                groups.setdefault((size, fingerprint), []).append(addr)
        return groups

    def _classify_function_aliases(
        self, image_id: ImageId, opposite_id: ImageId, matches: list[ReccmpMatch]
    ) -> bool:
        """Classify one image's remaining bodies against canonical pairs."""
        added = False
        canonical_groups: dict[
            tuple[int, tuple[tuple[str, str], ...]], list[ReccmpMatch]
        ] = {}
        for canonical in matches:
            opposite_addr = canonical.addr(opposite_id)
            opposite_size = canonical.size(opposite_id)
            if opposite_addr is None or opposite_size is None or opposite_size <= 0:
                continue
            fingerprint = self._alias_fingerprint(
                opposite_id, opposite_addr, opposite_size
            )
            if fingerprint is not None:
                canonical_groups.setdefault((opposite_size, fingerprint), []).append(
                    canonical
                )

        groups = self._unpaired_function_candidates(image_id)
        for (size, fingerprint), addrs in groups.items():
            for addr in addrs:
                identities: set[int] = set()
                for canonical in canonical_groups.get((size, fingerprint), []):
                    opposite_addr = canonical.addr(opposite_id)
                    assert opposite_addr is not None
                    orig_addr = addr if image_id == ImageId.ORIG else opposite_addr
                    recomp_addr = opposite_addr if image_id == ImageId.ORIG else addr
                    if self.raw_pair_alias_equivalent(orig_addr, recomp_addr, size):
                        identities.add(canonical.orig_addr)
                if len(identities) == 1:
                    added |= self.db.set_alias(image_id, addr, identities.pop())
        return added

    def _paired_caller_identity_edges(self) -> set[tuple[int, int]]:
        """Direct-call identities at corresponding instructions in paired owners."""
        orig_candidates = {
            entity.orig_addr
            for entity in self.db.unexplained(ImageId.ORIG)
            if entity.entity_type == EntityType.FUNCTION
            and entity.orig_addr is not None
        }
        recomp_candidates = {
            entity.recomp_addr
            for entity in self.db.unexplained(ImageId.RECOMP)
            if entity.entity_type == EntityType.FUNCTION
            and entity.recomp_addr is not None
        }
        edges: set[tuple[int, int]] = set()
        for caller in self.db.get_functions():
            orig_size = caller.size(ImageId.ORIG)
            recomp_size = caller.size(ImageId.RECOMP)
            if orig_size is None or recomp_size is None:
                continue
            try:
                orig_raw = self.orig_bin.read(caller.orig_addr, orig_size)
                recomp_raw = self.recomp_bin.read(caller.recomp_addr, recomp_size)
            except (InvalidVirtualAddressError, InvalidVirtualReadError):
                continue
            orig_ins = _code_instructions(orig_raw, caller.orig_addr, self.is_32bit)
            recomp_ins = _code_instructions(
                recomp_raw, caller.recomp_addr, self.is_32bit
            )
            if (
                orig_ins is None
                or recomp_ins is None
                or len(orig_ins) != len(recomp_ins)
                or any(a[2] != b[2] for a, b in zip(orig_ins, recomp_ins))
            ):
                continue
            for orig_instruction, recomp_instruction in zip(orig_ins, recomp_ins):
                if orig_instruction[2] != "call":
                    continue
                orig_operand = re.fullmatch(r"0x([0-9a-fA-F]+)", orig_instruction[3])
                recomp_operand = re.fullmatch(
                    r"0x([0-9a-fA-F]+)", recomp_instruction[3]
                )
                if orig_operand is None or recomp_operand is None:
                    continue
                orig_target = int(orig_operand.group(1), 16)
                recomp_target = int(recomp_operand.group(1), 16)
                if (
                    orig_target in orig_candidates
                    and recomp_target in recomp_candidates
                ):
                    edges.add((orig_target, recomp_target))
        return edges

    def discover_unique_called_functions(self) -> list[tuple[int, int]]:
        """Pair callees only when paired-callsite evidence is mutually unique."""
        discovered: list[tuple[int, int]] = []
        while True:
            edges = self._paired_caller_identity_edges()
            orig_degree: dict[int, int] = {}
            recomp_degree: dict[int, int] = {}
            for orig_addr, recomp_addr in edges:
                orig_degree[orig_addr] = orig_degree.get(orig_addr, 0) + 1
                recomp_degree[recomp_addr] = recomp_degree.get(recomp_addr, 0) + 1
            pairs = sorted(
                (orig_addr, recomp_addr)
                for orig_addr, recomp_addr in edges
                if orig_degree[orig_addr] == 1 and recomp_degree[recomp_addr] == 1
            )
            if not pairs:
                return discovered
            self.db.bulk_match(pairs)
            discovered.extend(pairs)

    def discover_unpaired_function_bodies(self) -> list[tuple[int, int]]:
        """Discover differently named function pairs to a conservative fixed point.

        Candidate edges require equal size, equal relocation-aware shape and a
        full ``raw_pair_alias_equivalent`` proof.  A pair is committed only
        when its edge has degree one at *both* endpoints.  All newly committed
        pairs become operand identities for the next pass.  Ambiguous graphs
        are left untouched instead of falling back to name/FIFO order.

        Once real pairs stop growing, remaining bodies may be recorded as
        side-local aliases of an existing canonical pair.  This is symmetric:
        original folded addresses and recomp duplicate emissions use the same
        canonical-original identity and never create fake one-to-one pairs.
        """
        discovered: list[tuple[int, int]] = []
        while True:
            orig_groups = self._unpaired_function_candidates(ImageId.ORIG)
            recomp_groups = self._unpaired_function_candidates(ImageId.RECOMP)
            edges: set[tuple[int, int]] = set()
            for key in orig_groups.keys() & recomp_groups.keys():
                size = key[0]
                for orig_addr in orig_groups[key]:
                    for recomp_addr in recomp_groups[key]:
                        if self.raw_pair_alias_equivalent(orig_addr, recomp_addr, size):
                            edges.add((orig_addr, recomp_addr))
            orig_degree: dict[int, int] = {}
            recomp_degree: dict[int, int] = {}
            for orig_addr, recomp_addr in edges:
                orig_degree[orig_addr] = orig_degree.get(orig_addr, 0) + 1
                recomp_degree[recomp_addr] = recomp_degree.get(recomp_addr, 0) + 1
            unique_pairs = {
                (orig_addr, recomp_addr)
                for orig_addr, recomp_addr in edges
                if orig_degree[orig_addr] == 1 and recomp_degree[recomp_addr] == 1
            }

            # A curated equivalence group supplies the otherwise-missing identity
            # for a many-originals-to-one-recomp folded body. Pair only its declared
            # canonical member; the remaining members become side-local aliases on
            # the next fixed-point iteration. Without that evidence, ambiguity stays.
            origs_by_recomp: dict[int, set[int]] = {}
            for orig_addr, recomp_addr in edges:
                origs_by_recomp.setdefault(recomp_addr, set()).add(orig_addr)
            canonical_pairs: set[tuple[int, int]] = set()
            for recomp_addr, orig_addrs in origs_by_recomp.items():
                if len(orig_addrs) < 2 or any(
                    orig_degree[orig_addr] != 1 for orig_addr in orig_addrs
                ):
                    continue
                canonical_addrs = {
                    canonical_orig_addr(self.equivalence_groups, orig_addr)
                    for orig_addr in orig_addrs
                }
                if len(canonical_addrs) != 1:
                    continue
                (canonical_addr,) = canonical_addrs
                if canonical_addr in orig_addrs:
                    canonical_pairs.add((canonical_addr, recomp_addr))

            pairs = sorted(unique_pairs | canonical_pairs)
            if pairs:
                self.db.bulk_match(pairs)
                discovered.extend(pairs)
                continue

            # Alias identities can themselves unlock mutually unique callers,
            # so interleave symmetric classification with pair discovery.
            matches = list(self.db.get_functions())
            orig_added = self._classify_function_aliases(
                ImageId.ORIG, ImageId.RECOMP, matches
            )
            recomp_added = self._classify_function_aliases(
                ImageId.RECOMP, ImageId.ORIG, matches
            )
            if not orig_added and not recomp_added:
                break
        return discovered

    def raw_pair_alias_equivalent(
        self,
        orig_addr: int,
        recomp_addr: int,
        size: int,
        *,
        _depth: int = 0,
        _seen: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Recomputed, conservative compiler-alias equivalence for one
        (orig, recomp) body pair that is not an annotated match.

        MSVC folds identical COMDAT bodies (template deleting destructors
        above all), so one original address may serve slots that the rebuild
        gives distinct functions. This proves the equivalence from the bodies
        themselves instead of a declared alias ledger: both sides must
        disassemble to the same sanitized instruction sequence. Where one
        instruction pair diverges (or resolves only to a positional
        placeholder), it may still be accepted when it is a direct call or
        jump whose two targets are themselves alias-equivalent — folding is
        transitive: the linker folded a derived deleting destructor onto its
        base's only because their destructor chains folded too. The
        recursion is depth-bounded and cycle-guarded, and any operand that
        cannot be proven fails the check rather than matching positionally.
        A bare original ``jmp rel32`` island (a stale incremental-link
        forwarder) is followed to the shared body it lands on.
        """
        # pylint: disable=too-many-return-statements

        if size <= 0 or _depth > 3:
            return False
        seen = _seen if _seen is not None else set()
        if (orig_addr, recomp_addr) in seen:
            return True
        seen.add((orig_addr, recomp_addr))
        try:
            orig_raw = self.orig_bin.read(orig_addr, size)
            recomp_raw = self.recomp_bin.read(recomp_addr, size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            return False
        if len(orig_raw) != size or len(recomp_raw) != size:
            return False
        if _is_bare_jmp_island(orig_raw):
            island_target = (
                orig_addr + 5 + int.from_bytes(orig_raw[1:5], "little", signed=True)
            )
            return self.raw_pair_alias_equivalent(
                island_target, recomp_addr, size, _depth=_depth + 1, _seen=seen
            )
        orig_asm = self.orig_sanitize.parse_asm(orig_raw, orig_addr)
        recomp_asm = self.recomp_sanitize.parse_asm(recomp_raw, recomp_addr)
        if not orig_asm or len(orig_asm) != len(recomp_asm):
            return False
        orig_insts = _code_instructions(orig_raw, orig_addr, self.is_32bit)
        recomp_insts = _code_instructions(recomp_raw, recomp_addr, self.is_32bit)
        if (
            orig_insts is None
            or recomp_insts is None
            or len(orig_insts) != len(orig_asm)
            or len(recomp_insts) != len(orig_asm)
        ):
            return False
        for index, ((_, orig_line), (_, recomp_line)) in enumerate(
            zip(orig_asm, recomp_asm)
        ):
            if orig_line == recomp_line and "<OFFSET" not in orig_line:
                continue
            if not self._transfer_targets_alias_equivalent(
                orig_insts[index], recomp_insts[index], _depth, seen
            ):
                return False
        return True

    def _transfer_targets_alias_equivalent(
        self,
        orig_inst: tuple[int, int, str, str],
        recomp_inst: tuple[int, int, str, str],
        depth: int,
        seen: set[tuple[int, int]],
    ) -> bool:
        """Whether a diverging instruction pair is a direct transfer whose
        two targets are themselves alias-equivalent bodies. The target size
        comes from the annotated recomp entity; an unknown target is not
        evidence."""

        _, _, orig_mnemonic, orig_op = orig_inst
        _, _, recomp_mnemonic, recomp_op = recomp_inst
        if orig_mnemonic != recomp_mnemonic:
            return False
        if orig_mnemonic != "call" and orig_mnemonic not in JUMP_MNEMONICS:
            return False
        try:
            orig_target = int(orig_op, 16)
            recomp_target = int(recomp_op, 16)
        except ValueError:
            return False  # indirect or composite operand
        entity = self.db.get(ImageId.RECOMP, recomp_target)
        target_size = entity.size(ImageId.RECOMP) if entity is not None else None
        if target_size is None or target_size <= 0:
            return False
        return self.raw_pair_alias_equivalent(
            orig_target, recomp_target, target_size, _depth=depth + 1, _seen=seen
        )

    # ------------------------------------------------------------------
    # PDB-derived metadata for the effective-match verifier

    def _return_kind_of_type(self, type_key: CvdumpTypeKey) -> str:
        # pylint: disable=too-many-return-statements
        """Reduce a PDB return type to the register footprint of the
        returned value. Unknown or by-value aggregate returns stay
        "unknown", which makes the verifier compare eax exactly."""
        for _ in range(8):
            if type_key.is_scalar():
                scalar = CvdumpTypeMap.get(type_key)
                if scalar is None:
                    return "unknown"
                if scalar.name == "T_VOID":
                    return "void"
                if scalar.pointer is not None:
                    return "i32"
                if scalar.name.startswith("T_REAL"):
                    return "float"
                return _RETURN_KIND_BY_SIZE.get(scalar.size, "unknown")
            try:
                obj = self.types.from_key(type_key)
            except CvdumpKeyError:
                return "unknown"
            leaf = obj.get("type")
            if leaf == "LF_POINTER":
                return "i32"
            if leaf == "LF_ENUM" and "underlying_type" in obj:
                type_key = obj["underlying_type"]
                continue
            if leaf == "LF_MODIFIER" and "modifies" in obj:
                type_key = obj["modifies"]
                continue
            return "unknown"
        return "unknown"

    def _signature_of_node(self, node: CvdumpNode) -> tuple[str, CallAbi | None]:
        """(return kind, register-argument ABI) for one function node.
        Prefers the PDB TYPES record; falls back to the decorated name,
        which encodes the convention and return type even when the PDB
        (like Imperialism's) carries no type records at all."""
        return_kind = "unknown"
        abi = None
        if node.symbol_entry is not None:
            try:
                func = self.types.from_key(node.symbol_entry.func_type)
                abi = _CALL_ABI_BY_CONVENTION.get(func.get("call_type", ""))
                if "return_type" in func:
                    return_kind = self._return_kind_of_type(func["return_type"])
            except CvdumpKeyError:
                pass
        if (return_kind == "unknown" or abi is None) and node.decorated_name:
            mangled = parse_function_signature(node.decorated_name)
            if return_kind == "unknown":
                return_kind = mangled.return_kind
            if abi is None and mangled.convention is not None:
                abi = _CALL_ABI_BY_CONVENTION.get(mangled.convention)
        return (return_kind, abi)

    def _call_abi_map(self) -> dict[str, CallAbi | None]:
        """Map from a sanitized call-target name (as the diff displays it)
        to the callee's register-argument usage. A name shared by several
        functions with conflicting conventions resolves to None (unknown)."""
        if self._call_abi_cache is not None:
            return self._call_abi_cache
        result: dict[str, CallAbi | None] = {}
        for entity in self.db.get_all():
            if entity.entity_type != EntityType.FUNCTION:
                continue
            recomp_addr = entity.recomp_addr
            if recomp_addr is None:
                continue
            node = self.func_nodes.get(recomp_addr)
            if node is None:
                continue
            name = canonical_callee_name(
                self.db,
                ImageId.RECOMP,
                entity,
                self.equivalence_groups,
            )
            if name is None:
                continue
            _, abi = self._signature_of_node(node)
            if name in result and result[name] != abi:
                result[name] = None
            else:
                result[name] = abi
        self._call_abi_cache = result
        return result

    def _function_metadata(self, match: ReccmpMatch) -> FunctionMetadata | None:
        if not self.func_nodes:
            return None
        return_kind = "unknown"
        node = self.func_nodes.get(match.recomp_addr)
        if node is not None:
            return_kind, _ = self._signature_of_node(node)
        abi_map = self._call_abi_map()
        return FunctionMetadata(
            return_kind=return_kind,
            call_abi=abi_map.get,
        )

    @staticmethod
    def _print_recomp_instruction(
        instruction: str, *, source_ref: str | None, is_pinned: bool
    ) -> str:
        match source_ref, is_pinned:
            case None, _:
                # cannot be pinned if it has no source reference
                return instruction
            case source_ref_str, False:
                return f"{instruction} \t({source_ref_str})"
            case source_ref_str, True:
                return f"{instruction} \t({source_ref_str}, pinned)"
            case _:
                # Unreachable, but mypy doesn't understand
                assert False

    def _compare_function_assembly(
        self,
        orig: AsmExcerpt,
        recomp: AsmExcerpt,
        split_points: list[tuple[int, int]],
        *,
        match: ReccmpMatch | None = None,
        orig_raw: bytes | None = None,
        recomp_raw: bytes | None = None,
        include_diff: bool = True,
        include_exact_diff: bool = True,
        metadata: FunctionMetadata | None = None,
        orig_meta: list[InstructionMeta | None] | None = None,
        recomp_meta: list[InstructionMeta | None] | None = None,
    ) -> EntityCompareResult:
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        # Detach addresses from asm lines for the text diff.
        orig_asm = [x[1] for x in orig]
        recomp_asm = [x[1] for x in recomp]

        # Align on structured IR keys; display strings stay for printing/scoring UI.
        orig_keys = [instruction_match_key(line) for line in orig_asm]
        recomp_keys = [instruction_match_key(line) for line in recomp_asm]
        diff = SequenceMatcherWithPins(orig_keys, recomp_keys, split_points)

        ratio = diff.ratio()
        opcodes = diff.get_opcodes()
        if ratio == 1.0:
            analysis = ComparisonAnalysis.exact()
        else:
            if metadata is None and match is not None:
                metadata = self._function_metadata(match)
            if orig_meta is None and orig_raw is not None:
                orig_meta_by_addr = self.orig_sanitize.collect_instruction_meta(
                    orig_raw,
                    (
                        match.orig_addr
                        if match is not None
                        else (orig[0][0] if orig and orig[0][0] is not None else 0)
                    ),
                )
                orig_meta = [
                    orig_meta_by_addr.get(addr) if addr is not None else None
                    for addr, _ in orig
                ]
            if recomp_meta is None and recomp_raw is not None:
                recomp_meta_by_addr = self.recomp_sanitize.collect_instruction_meta(
                    recomp_raw,
                    (
                        match.recomp_addr
                        if match is not None
                        else (
                            recomp[0][0] if recomp and recomp[0][0] is not None else 0
                        )
                    ),
                )
                recomp_meta = [
                    recomp_meta_by_addr.get(addr) if addr is not None else None
                    for addr, _ in recomp
                ]
            analysis = analyze_effective_match(
                opcodes,
                orig_asm,
                recomp_asm,
                orig_addrs=[x[0] for x in orig],
                metadata=metadata,
                orig_meta=orig_meta,
                recomp_addrs=[x[0] for x in recomp],
                recomp_meta=recomp_meta,
            )

        analysis = self._enrich_analysis_with_source(analysis)

        inline_layout = None
        if (
            ratio < 1.0
            and match is not None
            and not analysis.is_effective
        ):
            inline_layout = self._analyze_inline_expansions(
                match, orig_asm, recomp_asm
            )

        if not include_diff or (ratio == 1.0 and not include_exact_diff):
            result = EntityCompareResult(
                match_ratio=ratio,
                analysis=analysis,
                inline_expansions=(
                    inline_layout.expansions if inline_layout is not None else ()
                ),
                accuracy_modulo_inline=(
                    inline_layout.accuracy_modulo_inline
                    if inline_layout is not None
                    else None
                ),
            )
            result.refresh_equivalence_level()
            return result

        # Convert the addresses to hex string for the diff output
        orig_for_printing = [
            (hex(addr) if addr is not None else "", instr) for addr, instr in orig
        ]

        recomp_for_printing = [
            (
                hex(addr) if addr is not None else "",
                self._print_recomp_instruction(
                    instruction,
                    source_ref=self._source_ref_of_recomp_addr(addr),
                    is_pinned=any(
                        recomp_addr == line_index for _, recomp_addr in split_points
                    ),
                ),
            )
            for line_index, (addr, instruction) in enumerate(recomp)
        ]

        rdiff = RawDiffOutput(
            codes=opcodes,
            orig_inst=orig_for_printing,
            recomp_inst=recomp_for_printing,
        )

        stack_layout = None
        if ratio < 1.0:
            stack_layout = analyze_stack_layout(
                rdiff,
                orig_asm,
                recomp_asm,
                fn_symbol=self._fn_symbol_entry(match),
            )

        result = EntityCompareResult(
            diff=rdiff,
            match_ratio=ratio,
            analysis=analysis,
            stack_permutation=(
                stack_layout.permutation if stack_layout is not None else ()
            ),
            accuracy_modulo_stack=(
                stack_layout.accuracy_modulo_stack if stack_layout is not None else None
            ),
            inline_expansions=(
                inline_layout.expansions if inline_layout is not None else ()
            ),
            accuracy_modulo_inline=(
                inline_layout.accuracy_modulo_inline
                if inline_layout is not None
                else None
            ),
        )
        result.refresh_equivalence_level()
        return result

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

    def _collect_line_annotations(self, recomp: AsmExcerpt) -> list[ReccmpMatch]:
        """
        Finds all `// LINE:` annotations within the given function
        and drops any whose order is not consistent between original and recomp.
        """
        if len(recomp) == 0:
            return []

        recomp_start_addr = recomp[0][0]
        recomp_end_addr = recomp[-1][0]
        assert recomp_start_addr is not None and recomp_end_addr is not None
        line_annotations = self.db.get_lines_in_recomp_range(
            recomp_start_addr, recomp_end_addr
        )

        # This is a naive/greedy algorithm to remove the non-monotonous entries.
        # There likely is a "better" way to do this, in the sense that the smallest number
        # of entries is removed.
        line_annotations_monotonous: list[ReccmpMatch] = []
        last_address = 0
        for sync_point in line_annotations:
            if sync_point.recomp_addr > last_address:
                line_annotations_monotonous.append(sync_point)
                last_address = sync_point.recomp_addr
            else:
                self.report(
                    ReccmpEvent.WRONG_ORDER,
                    sync_point.orig_addr,
                    f"Line annotation '{sync_point.name}' is out of order relative to other line annotations.",
                )

        return line_annotations_monotonous

    def _split_code_on_line_annotations(
        self,
        orig_combined: AsmExcerpt,
        recomp_combined: AsmExcerpt,
        line_annotations: list[ReccmpMatch],
    ) -> Iterator[tuple[AsmExcerpt, AsmExcerpt]]:
        """
        For each given `// LINE:` annotation, splits the code into the part before,
        the annotated line, and the part after it.
        """
        split_points = self._compute_split_points(
            orig_combined, recomp_combined, line_annotations
        )

        for (orig_start, recomp_start), (orig_end, recomp_end) in pairwise(
            split_points
        ):
            yield (
                orig_combined[orig_start:orig_end],
                recomp_combined[recomp_start:recomp_end],
            )

    def _compute_split_points(
        self, orig: AsmExcerpt, recomp: AsmExcerpt, line_annotations: list[ReccmpMatch]
    ) -> list[tuple[int, int]]:
        """
        Computes the index pairs into `orig` and `recomp`
        that correspond to the line annotations given in `line_annotations`.
        """
        split_points: list[tuple[int, int]] = []

        for line_annotation in line_annotations:
            orig_split_index = next(
                (
                    i
                    for i, entry in enumerate(orig)
                    if entry[0] == line_annotation.orig_addr
                ),
                None,
            )
            if orig_split_index is None:
                self.report(
                    ReccmpEvent.NO_MATCH,
                    line_annotation.orig_addr,
                    "Found no code line corresponding to this original address",
                )
                continue

            recomp_split_index = next(
                (
                    i
                    for i, entry in enumerate(recomp)
                    if entry[0] == line_annotation.recomp_addr
                ),
                None,
            )
            if recomp_split_index is None:
                self.report(
                    ReccmpEvent.NO_MATCH,
                    line_annotation.orig_addr,
                    f"Found no code line corresponding to recomp address {hex(line_annotation.recomp_addr)}. Recompilation may fix this problem.",
                )
                continue

            split_points.append((orig_split_index, recomp_split_index))
            split_points.append((orig_split_index + 1, recomp_split_index + 1))

        return split_points
