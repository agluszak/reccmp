# pylint: disable=too-many-lines
import dataclasses
from dataclasses import dataclass, field
from functools import cache
import struct
from itertools import pairwise
from typing import Callable, Iterator
from reccmp.compare.lines import LinesDb
from reccmp.compare.thunk_resolve import read_e9_jmp_target
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins
from reccmp.compare.asm.verifier import FunctionMetadata
from reccmp.compare.call_facts import CallFacts
from reccmp.compare.asm.verifier import analyze_effective_match
from reccmp.compare.asm.parse import assert_fixup
from reccmp.compare.asm.instgen import (
    InstructionMeta,
    meta_from_decoded,
)
from reccmp.compare.asm.ir import (
    ExtentKind,
    FunctionImage,
    compute_extent_closed,
    control_flow_topology_keys,
    excerpt_addrs,
    excerpt_displays,
    instruction_match_key,
    instruction_semantic_key,
    rebind_local_identities,
    resolve_asm_stream,
)
from reccmp.compare.asm.parse import AsmExcerpt, ParseAsm
from reccmp.compare.asm.replacement import (
    create_name_lookup,
)
from reccmp.compare.db import EntityDb, ReccmpMatch
from reccmp.compare.diff import EntityCompareResult, RawDiffOutput
from reccmp.compare.verification import (
    admit_effective,
    admit_exact_analysis,
    admit_proof,
)
from reccmp.compare.stack_layout import analyze_stack_layout
from reccmp.compare.inlines import (
    HelperCatalogEntry,
)
from reccmp.compare.event import ReccmpEvent, ReccmpReportProtocol
from reccmp.source import SourceIndex
from reccmp.cvdump.analysis import CvdumpNode
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.formats import Image, PEImage
from reccmp.types import ImageId

from reccmp.compare.body_equivalence import _is_bare_jmp_island
from reccmp.compare.extent import discover_extent
from reccmp.compare.refutation import RefutationMixin
from reccmp.compare.inline_accounting import InlineAccountingMixin


def _longest_increasing_by_recomp(
    annotations: list[ReccmpMatch],
) -> list[ReccmpMatch]:
    """Keep the longest subsequence with strictly increasing recomp addresses."""
    n = len(annotations)
    if n == 0:
        return []

    # tails[k] = index of the smallest-recomp-addr end of an IS of length k+1
    tails: list[int] = []
    predecessor: list[int | None] = [None] * n

    for i, ann in enumerate(annotations):
        addr = ann.recomp_addr
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if annotations[tails[mid]].recomp_addr < addr:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            predecessor[i] = tails[lo - 1]
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i

    result: list[ReccmpMatch] = []
    idx: int | None = tails[-1]
    while idx is not None:
        result.append(annotations[idx])
        idx = predecessor[idx]
    result.reverse()
    return result


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


def _stamp_instruction_ids(excerpt: AsmExcerpt) -> tuple:
    """Assign stable program-point ids when a test excerpt has none."""
    rows = []
    for index, row in enumerate(excerpt):
        if row.instruction_id is None:
            rows.append(dataclasses.replace(row, instruction_id=index))
        else:
            rows.append(row)
    return tuple(rows)


@dataclass
class FunctionComparator(InlineAccountingMixin, RefutationMixin):
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
    # Optional Clang-backed layout/ownership index for mismatch enrichment.
    source_index: SourceIndex | None = None
    # Try to refute unproven results by differential execution (needs unicorn).
    witness_search: bool = False

    def __post_init__(self):
        self._call_facts_cache: dict[str, CallFacts | None] | None = None
        self._fp_cache: dict[
            tuple[ImageId, int, int], tuple[tuple[str, str], ...] | None
        ] = {}
        self._helper_catalog: list[HelperCatalogEntry] | None = None
        self._helper_by_orig: dict[int, HelperCatalogEntry | None] = {}
        # Exact call-identity → unique orig_addr (ambiguous keys omitted).
        self._helper_identity_index: dict[str, int] | None = None
        self._helper_identity_ambiguous: set[str] | None = None
        self._witness_translator = None
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
            image_id=ImageId.ORIG,
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
            image_id=ImageId.RECOMP,
        )

    def rebuild_lookups(self) -> None:
        """Rebuild name-lookup closures after the entity catalog is frozen."""
        self.__post_init__()

    def _source_ref_of_recomp_addr(self, recomp_addr: int | None) -> str | None:
        if recomp_addr is None:
            return None
        path_line_pair = self.lines_db.find_line_of_recomp_address(recomp_addr)
        if path_line_pair is None:
            return None
        return f"{path_line_pair[0].name}:{path_line_pair[1]}"

    def _load_function_image(
        self,
        sanitizer: ParseAsm,
        raw: bytes,
        start_addr: int,
        extent: int,
        extent_kind: ExtentKind,
    ) -> FunctionImage:
        """Decode one side into an owned function image (excerpt + tables)."""
        excerpt = sanitizer.parse_asm(raw, start_addr)
        stamped = tuple(
            dataclasses.replace(row, instruction_id=index)
            for index, row in enumerate(excerpt)
        )
        tables = tuple(sanitizer.jump_tables)
        stamped = rebind_local_identities(
            stamped,
            start_addr=start_addr,
            extent=extent,
            jump_tables=tables,
            image_id=(
                sanitizer.image_id.name.lower()
                if sanitizer.image_id is not None
                else "unknown"
            ),
        )
        return FunctionImage(
            start_addr=start_addr,
            extent=extent,
            extent_kind=extent_kind,
            excerpt=stamped,
            jump_tables=tables,
            coverage_incomplete=sanitizer.coverage_incomplete,
            extent_closed=compute_extent_closed(
                stamped,
                start_addr=start_addr,
                extent=extent,
                coverage_incomplete=sanitizer.coverage_incomplete,
                jump_tables=tables,
                extent_kind=extent_kind,
            ),
            raw=raw,
        )

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
        annotated_orig_size = match.size(ImageId.ORIG)
        recomp_size = match.size(ImageId.RECOMP)

        orig_extent_kind = ExtentKind.KNOWN
        if annotated_orig_size is None:
            orig_extent_kind = ExtentKind.ESTIMATED
            assert recomp_size is not None
            orig_max = match.max_size(ImageId.ORIG)
            if orig_max is not None:
                orig_size = min(orig_max, recomp_size)
            else:
                orig_size = recomp_size
            discovered = discover_extent(
                self.orig_bin, match.orig_addr, orig_max, is_32bit=self.is_32bit
            )
            # Much larger than the recompilation usually means the walk ran
            # past a call that does not return into the next function.
            if discovered is not None and discovered <= 2 * recomp_size + 64:
                orig_size = discovered
        else:
            orig_size = annotated_orig_size

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

        orig_image = self._load_function_image(
            self.orig_sanitize,
            orig_raw,
            match.orig_addr,
            orig_size,
            orig_extent_kind,
        )
        recomp_image = self._load_function_image(
            self.recomp_sanitize,
            recomp_raw,
            match.recomp_addr,
            recomp_size,
            ExtentKind.KNOWN,
        )
        orig_rows = list(orig_image.excerpt)
        recomp_rows = list(recomp_image.excerpt)

        # Check for assert calls only if we expect to find them
        if has_asserts(self.orig_bin):
            assert_fixup(orig_rows)
            orig_image = orig_image.with_excerpt(orig_rows)

        if has_asserts(self.recomp_bin):
            assert_fixup(recomp_rows)
            recomp_image = recomp_image.with_excerpt(recomp_rows)

        line_annotations = self._collect_line_annotations(list(recomp_image.excerpt))

        split_points = self._compute_split_points(
            list(orig_image.excerpt), list(recomp_image.excerpt), line_annotations
        )

        result = self.compare_function_images(
            orig_image,
            recomp_image,
            split_points,
            match=match,
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
            # The island is a modeled thunk to a grouped body; its guessed
            # byte window is not the function extent this proof depends on.
            alias = admit_effective(
                ("folded_symbol_alias",),
                coverage_incomplete=(
                    orig_image.coverage_incomplete or recomp_image.coverage_incomplete
                ),
                extent_closed=True,
            )
            if alias is not None:
                return dataclasses.replace(result, analysis=alias.analysis)

        analysis = admit_proof(
            result.analysis,
            coverage_incomplete=(
                orig_image.coverage_incomplete or recomp_image.coverage_incomplete
            ),
            extent_closed=orig_image.extent_closed and recomp_image.extent_closed,
        )
        if orig_image.extent_closed and recomp_image.extent_closed:
            # Emulation needs the real extents: with an open one it may run
            # a different stretch of code than the comparison looked at.
            analysis = self._refute(match, analysis, orig_image, recomp_image)
        if analysis is not result.analysis:
            result = dataclasses.replace(result, analysis=analysis)
        return result

    # ------------------------------------------------------------------
    # PDB-derived metadata for the effective-match verifier

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

    def compare_function_images(
        self,
        orig: FunctionImage,
        recomp: FunctionImage,
        split_points: list[tuple[int, int]],
        *,
        match: ReccmpMatch | None = None,
        include_diff: bool = True,
        include_exact_diff: bool = True,
        metadata: FunctionMetadata | None = None,
        orig_meta: list[InstructionMeta | None] | None = None,
        recomp_meta: list[InstructionMeta | None] | None = None,
    ) -> EntityCompareResult:
        """Compare two owned function images; they are the source of excerpt,
        tables, addresses, and coverage."""
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        orig_rows = list(orig.excerpt)
        recomp_rows = list(recomp.excerpt)
        coverage_incomplete = orig.coverage_incomplete or recomp.coverage_incomplete
        extent_closed = orig.extent_closed and recomp.extent_closed
        orig_asm = excerpt_displays(orig_rows)
        recomp_asm = excerpt_displays(recomp_rows)

        orig_keys = [instruction_match_key(row) for row in orig_rows]
        recomp_keys = [instruction_match_key(row) for row in recomp_rows]
        orig_sem = [instruction_semantic_key(row) for row in orig_rows]
        recomp_sem = [instruction_semantic_key(row) for row in recomp_rows]
        display_diff = SequenceMatcherWithPins(orig_keys, recomp_keys, split_points)
        semantic_diff = SequenceMatcherWithPins(orig_sem, recomp_sem, split_points)

        ratio = semantic_diff.ratio()
        display_similarity = display_diff.ratio()
        opcodes = display_diff.get_opcodes()
        operands_complete = all(
            row.operand_model_complete for row in (*orig_rows, *recomp_rows)
        )
        control_flow_complete = all(
            (not row.is_code) or row.control_flow_known
            for row in (*orig_rows, *recomp_rows)
        )
        displays_match = orig_asm == recomp_asm
        orig_topology = control_flow_topology_keys(orig_rows, orig.jump_tables)
        recomp_topology = control_flow_topology_keys(recomp_rows, recomp.jump_tables)
        exact = admit_exact_analysis(
            displays_equal=displays_match,
            topology_equal=(
                orig_topology is not None and orig_topology == recomp_topology
            ),
            keys_equal=orig_sem == recomp_sem,
            operands_complete=operands_complete,
            control_flow_complete=control_flow_complete,
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        )
        if exact is not None:
            analysis = exact
        else:
            if metadata is None and match is not None:
                metadata = self._function_metadata(match)
            if orig_meta is None:
                orig_meta = [
                    meta_from_decoded(row) if row.is_code else None for row in orig_rows
                ]
            if recomp_meta is None:
                recomp_meta = [
                    meta_from_decoded(row) if row.is_code else None
                    for row in recomp_rows
                ]
            analysis = analyze_effective_match(
                opcodes,
                resolve_asm_stream(orig_rows, jump_tables=orig.jump_tables),
                resolve_asm_stream(recomp_rows, jump_tables=recomp.jump_tables),
                orig_addrs=excerpt_addrs(orig_rows),
                metadata=metadata,
                orig_meta=orig_meta,
                recomp_addrs=excerpt_addrs(recomp_rows),
                recomp_meta=recomp_meta,
                coverage_incomplete=coverage_incomplete,
                extent_closed=extent_closed,
            )
        analysis = admit_proof(
            analysis,
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        )

        analysis = self._enrich_analysis_with_source(analysis, match=match)

        inline_layout = None
        if ratio < 1.0 and match is not None and not analysis.is_effective:
            inline_layout = self._analyze_inline_expansions(
                match, orig_rows, recomp_rows
            )

        stack_layout = None
        if ratio < 1.0:
            stack_rdiff = RawDiffOutput(
                codes=opcodes,
                orig_inst=[
                    (
                        hex(row.address) if row.address is not None else "",
                        row.display,
                    )
                    for row in orig_rows
                ],
                recomp_inst=[
                    (
                        hex(row.address) if row.address is not None else "",
                        row.display,
                    )
                    for row in recomp_rows
                ],
            )
            stack_layout = analyze_stack_layout(
                stack_rdiff,
                orig_asm,
                recomp_asm,
                fn_symbol=self._fn_symbol_entry(match),
                types=self.types,
            )

        if not include_diff or (ratio == 1.0 and not include_exact_diff):
            result = EntityCompareResult(
                match_ratio=ratio,
                display_similarity=display_similarity,
                analysis=analysis,
                stack_permutation=(
                    stack_layout.permutation if stack_layout is not None else ()
                ),
                accuracy_modulo_stack=(
                    stack_layout.accuracy_modulo_stack
                    if stack_layout is not None
                    else None
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
            return result

        # Convert the addresses to hex string for the diff output
        orig_for_printing = [
            (hex(row.address) if row.address is not None else "", row.display)
            for row in orig_rows
        ]

        recomp_for_printing = [
            (
                hex(row.address) if row.address is not None else "",
                self._print_recomp_instruction(
                    row.display,
                    source_ref=self._source_ref_of_recomp_addr(row.address),
                    is_pinned=any(
                        recomp_addr == line_index for _, recomp_addr in split_points
                    ),
                ),
            )
            for line_index, row in enumerate(recomp_rows)
        ]

        rdiff = RawDiffOutput(
            codes=opcodes,
            orig_inst=orig_for_printing,
            recomp_inst=recomp_for_printing,
        )

        result = EntityCompareResult(
            diff=rdiff,
            match_ratio=ratio,
            display_similarity=display_similarity,
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
        return result

    def _compare_function_assembly(
        self,
        orig: AsmExcerpt,
        recomp: AsmExcerpt,
        split_points: list[tuple[int, int]],
        *,
        match: ReccmpMatch | None = None,
        include_diff: bool = True,
        include_exact_diff: bool = True,
        metadata: FunctionMetadata | None = None,
        orig_meta: list[InstructionMeta | None] | None = None,
        recomp_meta: list[InstructionMeta | None] | None = None,
        coverage_incomplete: bool = False,
    ) -> EntityCompareResult:
        # pylint: disable=too-many-arguments
        """Test/legacy wrapper that lifts excerpts into ephemeral function images."""
        orig_image = FunctionImage(
            start_addr=0,
            extent=0,
            extent_kind=ExtentKind.KNOWN,
            excerpt=_stamp_instruction_ids(orig),
            coverage_incomplete=coverage_incomplete,
        )
        recomp_image = FunctionImage(
            start_addr=0,
            extent=0,
            extent_kind=ExtentKind.KNOWN,
            excerpt=_stamp_instruction_ids(recomp),
            coverage_incomplete=coverage_incomplete,
        )
        return self.compare_function_images(
            orig_image,
            recomp_image,
            split_points,
            match=match,
            include_diff=include_diff,
            include_exact_diff=include_exact_diff,
            metadata=metadata,
            orig_meta=orig_meta,
            recomp_meta=recomp_meta,
        )

    def _collect_line_annotations(self, recomp: AsmExcerpt) -> list[ReccmpMatch]:
        """
        Finds all `// LINE:` annotations within the given function
        and drops any whose order is not consistent between original and recomp.
        """
        if len(recomp) == 0:
            return []

        recomp_start_addr = recomp[0].address
        recomp_end_addr = recomp[-1].address
        assert recomp_start_addr is not None and recomp_end_addr is not None
        line_annotations = list(
            self.db.get_lines_in_recomp_range(recomp_start_addr, recomp_end_addr)
        )

        # Longest increasing subsequence of recomp addresses keeps the maximum
        # set of monotonic source pins (O(n log n)).
        line_annotations_monotonous = _longest_increasing_by_recomp(line_annotations)
        dropped = len(line_annotations) - len(line_annotations_monotonous)
        if dropped:
            kept = {id(ann) for ann in line_annotations_monotonous}
            for sync_point in line_annotations:
                if id(sync_point) in kept:
                    continue
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
                    if entry.address == line_annotation.orig_addr
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
                    if entry.address == line_annotation.recomp_addr
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
