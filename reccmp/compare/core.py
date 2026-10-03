"""Target metadata preparation: binaries, PDB, annotations and pairs.

``Compare`` builds the entity catalog that every retained check consumes. It
pairs original and recompiled entities from annotations, debug information
and binary structure, and records why each pair exists
(:class:`~reccmp.compare.db.PairBasis`). It does not compare code: a pair
means "compare these implementations", not "these are equivalent".
"""

import logging
from typing import Iterable, Iterator
from typing_extensions import Self
from reccmp.project.detect import RecCmpTarget
from reccmp.parser import DecompCodebase
from reccmp.parser.marker import ProjectAliases, normalize_project_aliases
from reccmp.compare.variables import VariableComparator
from reccmp.formats import (
    Image,
    PEImage,
    TextFile,
)
from reccmp.cvdump import CvdumpTypesParser, CvdumpAnalysis
from reccmp.types import EntityType, ImageId
from reccmp.compare.event import (
    ReccmpReportProtocol,
    create_logging_wrapper,
)
from reccmp.source.index import SourceIndex
from .match_msvc import (
    match_lines,
    match_symbols,
    match_annotation_selectors,
    match_functions,
    match_vtables,
    match_static_variables,
    match_variables,
    match_strings,
    classify_exact_string_aliases,
    match_ref,
    match_imports,
)
from .match_folded import match_folded_function_aliases, match_seh
from .db import EntityDb, PairBasis, ReccmpEntity, ReccmpMatch
from .lines import LinesDb
from .target_analysis import PreparedAnalysis, load_target_analysis
from .analyze import (
    create_imports,
    create_import_thunks,
    create_thunks,
    create_analysis_floats,
    create_analysis_strings,
    create_analysis_widechars,
    create_analysis_vtordisps,
    create_crt_functions,
    create_seh_entities,
    complete_partial_floats,
    complete_partial_strings,
    match_entry,
    match_exports,
    import_sections,
    normalize_original_zero_size_data,
    classify_exact_vtable_aliases,
    classify_folded_function_aliases,
    classify_synthetic_jump_aliases,
    match_unpaired_direct_callees,
    classify_folded_vtable_aliases,
    match_inferred_vtables_by_slots,
)
from .ingest import (
    load_cvdump,
    load_cvdump_types,
    load_cvdump_lines,
    load_markers,
    load_data_sources,
)
from .mutate import (
    name_thunks,
    unique_names_for_overloaded_functions,
    match_crt_startup,
    set_max_size,
)
from .verify import (
    check_vtables,
)

logger = logging.getLogger(__name__)


class Compare:
    # pylint: disable=too-many-instance-attributes
    _db: EntityDb
    _lines_db: LinesDb
    code_files: list[TextFile]
    cvdump_analysis: CvdumpAnalysis
    orig_bin: Image
    recomp_bin: Image
    report: ReccmpReportProtocol
    target_id: str
    src_encoding: str
    bin_encoding: str
    types: CvdumpTypesParser
    variable_comparator: VariableComparator
    data_sources: list[TextFile]
    project_aliases: ProjectAliases
    codebase: DecompCodebase
    source_index: SourceIndex | None

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        orig_bin: Image,
        recomp_bin: Image,
        pdb_file: CvdumpAnalysis,
        target_id: str,
        encoding: str | None = None,
        code_files: list[TextFile] | None = None,
        data_sources: list[TextFile] | None = None,
        project_aliases: ProjectAliases | None = None,
        codebase: DecompCodebase | None = None,
        source_index: SourceIndex | None = None,
    ):
        self.orig_bin = orig_bin
        self.recomp_bin = recomp_bin
        self.cvdump_analysis = pdb_file
        self.target_id = target_id
        self.src_encoding = encoding or "utf-8"
        self.bin_encoding = encoding or "latin1"
        self.project_aliases = normalize_project_aliases(project_aliases or {})
        self.codebase = codebase or DecompCodebase([], target_id)

        if isinstance(code_files, list):
            self.code_files = code_files
        else:
            self.code_files = []

        if isinstance(data_sources, list):
            self.data_sources = data_sources
        else:
            self.data_sources = []

        self._lines_db = LinesDb()
        self._db = EntityDb()

        # For now, just redirect match alerts to the logger.
        self.report = create_logging_wrapper(logger)

        self.types = CvdumpTypesParser()

        self.source_index = source_index
        self.variable_comparator = VariableComparator(
            db=self._db,
            types=self.types,
            orig_bin=self.orig_bin,
            recomp_bin=self.recomp_bin,
            source_index=source_index,
        )

    def _seal_catalog(self) -> None:
        """No catalog mutation happens after preparation."""
        if not self._db.frozen:
            self._db.freeze()
        self._locate_pdb_nodes()

    def _locate_pdb_nodes(self) -> None:
        """Give PDB nodes their recompiled addresses, also when the catalog
        came from the cache and ingestion did not run (the Ghidra importer
        finds nodes by address)."""
        if not isinstance(self.recomp_bin, PEImage):
            return
        for node in self.cvdump_analysis.nodes:
            if self.recomp_bin.is_valid_section(node.section):
                node.addr = self.recomp_bin.get_abs_addr(node.section, node.offset)

    def _prepared_analysis(self) -> PreparedAnalysis:
        return PreparedAnalysis(self._db, self._lines_db, self.types)

    def _restore_prepared_analysis(self, analysis: PreparedAnalysis) -> None:
        self._db = analysis.db
        self._lines_db = analysis.lines_db
        self.types = analysis.types
        self.variable_comparator = VariableComparator(
            db=self._db,
            types=self.types,
            orig_bin=self.orig_bin,
            recomp_bin=self.recomp_bin,
            source_index=self.source_index,
        )
        self._seal_catalog()

    def run(self):
        if not isinstance(self.orig_bin, PEImage) or not isinstance(
            self.recomp_bin, PEImage
        ):
            return

        # Each task creates new entities or overwrites existing data.
        # The tasks are ordered roughly according to the principle
        # of highest-to-lowest confidence of data validity.
        load_cvdump_types(self.cvdump_analysis, self.types)
        load_cvdump(self.cvdump_analysis, self._db, self.recomp_bin)
        load_cvdump_lines(self.cvdump_analysis, self._lines_db, self.recomp_bin)

        match_entry(self._db, self.orig_bin, self.recomp_bin)

        # Data-source labels supplement source markers; they must not replace
        # a marker's established identity when the two disagree.
        load_data_sources(self._db, self.data_sources)
        load_markers(
            self.code_files,
            self._lines_db,
            self.orig_bin,
            self.codebase,
            self._db,
            self.bin_encoding,
            self.report,
        )

        normalize_original_zero_size_data(self._db, self.orig_bin)

        # Match using PDB and annotation data
        truncate = self.cvdump_analysis.truncate_symbols
        match_annotation_selectors(self._db, self.report, truncate=truncate)
        match_symbols(self._db, self.report, truncate=truncate)
        match_functions(self._db, self.report, truncate=truncate)
        match_folded_function_aliases(
            self._db,
            self.codebase,
            self._lines_db,
            self.report,
            truncate=truncate,
        )
        match_vtables(self._db, self.report)
        classify_exact_vtable_aliases(self._db, self.orig_bin, self.recomp_bin)
        match_static_variables(self._db, self.report)
        match_variables(self._db, self.report)
        match_lines(self._db, self._lines_db, self.report)

        # Detect floats first to eliminate potential overlap with string data
        for img_id, binfile in (
            (ImageId.ORIG, self.orig_bin),
            (ImageId.RECOMP, self.recomp_bin),
        ):
            create_imports(self._db, img_id, binfile)
            create_import_thunks(self._db, img_id, binfile)
            create_seh_entities(self._db, img_id, binfile)
            create_thunks(self._db, img_id, binfile)
            create_analysis_vtordisps(self._db, img_id, binfile)
            create_crt_functions(self._db, img_id, binfile)
            import_sections(self._db, img_id, binfile)

        match_imports(self._db)
        match_exports(self._db, self.orig_bin, self.recomp_bin)

        for img_id in (ImageId.ORIG, ImageId.RECOMP):
            set_max_size(self._db, img_id)

        match_crt_startup(self._db, self.orig_bin, self.recomp_bin)
        check_vtables(self._db)
        match_seh(self._db)

        match_ref(self._db, self.report)
        match_inferred_vtables_by_slots(self._db, self.orig_bin, self.recomp_bin)
        classify_exact_vtable_aliases(self._db, self.orig_bin, self.recomp_bin)
        classify_folded_function_aliases(self._db, self.orig_bin, self.recomp_bin)
        classify_synthetic_jump_aliases(self._db, self.orig_bin, self.codebase)
        match_unpaired_direct_callees(
            self._db, self.orig_bin, self.recomp_bin, self.codebase
        )
        classify_folded_vtable_aliases(self._db, self.orig_bin, self.recomp_bin)
        unique_names_for_overloaded_functions(self._db)
        name_thunks(self._db)

        # Search for const data values and read bytes from the binaries.
        # This happens last because establishing all other entities first
        # will reduce false positives. For each address presumed to be a
        # float or string, skip if there is an existing entity at the address.
        # A packed or protected original can leave its read-only sections
        # writable. The recompiled image is the linker's own output: its
        # sections say which data is constant in both.
        write_permissions = self.recomp_bin.section_write_permissions()
        for img_id, binfile in (
            (ImageId.ORIG, self.orig_bin),
            (ImageId.RECOMP, self.recomp_bin),
        ):
            # Some float consts may appear to be strings.
            # Detect floats first because we can identify them with more confidence
            # and this eliminates them from consideration as strings.
            create_analysis_floats(self._db, img_id, binfile, write_permissions)
            # Wide before Latin1: otherwise L"F1" is misread as the short string "F".
            create_analysis_widechars(self._db, img_id, binfile)
            create_analysis_strings(self._db, img_id, binfile, self.bin_encoding)
            complete_partial_floats(self._db, img_id, binfile)
            complete_partial_strings(self._db, img_id, binfile, self.bin_encoding)

        match_strings(self._db, self.report)
        classify_exact_string_aliases(self._db)
        self._seal_catalog()

    @classmethod
    def from_target(
        cls,
        target: RecCmpTarget,
        *,
        orig_addrs: Iterable[int] = (),
        recomp_addrs: Iterable[int] = (),
        use_cache: bool = True,
        source_index: SourceIndex | None = None,
    ) -> Self:
        loaded = load_target_analysis(
            target,
            orig_addrs=orig_addrs,
            recomp_addrs=recomp_addrs,
            use_cache=use_cache,
            source_index=source_index,
        )
        compare = cls(
            loaded.orig_bin,
            loaded.recomp_bin,
            loaded.pdb_file,
            target_id=target.target_id,
            encoding=target.encoding,
            data_sources=loaded.data_sources,
            code_files=loaded.code_files,
            project_aliases=loaded.project_aliases,
            codebase=loaded.codebase,
            source_index=loaded.source_index,
        )
        prepared = loaded.load_prepared()
        if prepared is not None:
            compare._restore_prepared_analysis(prepared)
        else:
            compare.run()
            loaded.store_prepared(compare._prepared_analysis())
        return compare

    def report_vtable_size_warnings(self, name_filter: str | None = None) -> None:
        """Log oversized-vtable evidence, optionally limited by name."""
        check_vtables(self._db, self.orig_bin, name_filter)

    @property
    def db(self) -> EntityDb:
        """The prepared, frozen entity catalog."""
        return self._db

    ## Public API

    def get_all(self) -> Iterator[ReccmpEntity]:
        return self._db.get_all()

    def get_match(self, orig_addr: int) -> ReccmpMatch | None:
        """Public lookup for a paired original address."""
        return self._db.get_one_match(orig_addr)

    def get(self, image_id: ImageId, addr: int) -> ReccmpEntity | None:
        """The entity starting at this address of one image."""
        return self._db.get(image_id, addr)

    def pair_basis(self, orig_addr: int) -> PairBasis | None:
        """How the pair at this original address was established."""
        return self._db.pair_basis(orig_addr)

    def get_functions(self) -> Iterator[ReccmpMatch]:
        return self._db.get_functions()

    def get_aliases(
        self, image_id: ImageId
    ) -> Iterator[tuple[ReccmpEntity, ReccmpMatch]]:
        """Side-local duplicates and their canonical pairs."""
        return self._db.get_aliases(image_id)

    def get_vtables(self) -> Iterator[ReccmpMatch]:
        return self._db.get_matches_by_type(EntityType.VTABLE)

    def get_variables(self) -> Iterator[ReccmpMatch]:
        return self._db.get_matches_by_type(EntityType.DATA)
