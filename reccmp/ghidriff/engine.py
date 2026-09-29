"""Ghidriff engine driven by reccmp's pairs and names.

Ghidra analyzes and decompiles both programs; ghidriff diffs and reports.
reccmp contributes what only the reconstruction knows: which functions
correspond, under which names, and where their source is. Both programs are
analyzed without imported debug types. A recompiled PDB symbol may correct
an inferred cdecl arity when retail independently agrees.
"""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from ghidriff import DecompileResult, FunctionMatch, GhidraDiffEngine

from reccmp.compare.manifest import FunctionEntry, Manifest, NamedObject
from reccmp.types import EntityType, ImageId

from .imports import import_locations, known_purge
from .locations import (
    STRING_TYPES,
    Extents,
    Located,
    Use,
    access_size,
    bitwise_scalar_operand,
    only_compared,
    register_operand,
)
from .preparation import (
    apply_stack_probe_call_fixups,
    correct_recompiled_signatures,
    correct_import_purges,
    infer_requested_callee_parameters,
    recover_requested_switches,
)
from .project_cache import _PRISTINE_FOLDER
from .results import (
    AnalysisFailure,
    Contents,
    DataReference,
    FailureKind,
    FunctionResult,
    ObjectOffset,
    PastEnd,
    PointerValue,
    RawBytes,
    StringValue,
    Uninitialized,
    UnknownExtent,
    classify,
)

if TYPE_CHECKING:
    from ghidra.program.model.address import Address
    from ghidra.program.model.listing import Function, Program

# Ghidra names imports itself, the same way in both programs.
_UNNAMED_TYPES = (EntityType.IMPORT, EntityType.IMPORT_THUNK)
_FUNCTION_TYPES = (EntityType.FUNCTION, EntityType.VTORDISP, EntityType.THUNK)
# Code the catalog knows starts a function. An import thunk is one too: left
# alone, Ghidra reads a jump to it as part of the function that jumps.
_ENTRY_TYPES = (*_FUNCTION_TYPES, EntityType.IMPORT_THUNK)
# Named by their contents in the catalog; shown by their contents instead.
_LITERAL_TYPES = (EntityType.STRING, EntityType.WIDECHAR, EntityType.FLOAT)
_STACK_PROBE_NAMES = frozenset(
    {"__chkstk", "__alloca_probe", "__alloca_probe_8", "__alloca_probe_16"}
)
# Upper bound on the bytes shown for one referenced location.
_RAW_LIMIT = 64
_RAW_ADDRESS = re.compile(r"(?<![\w])0x[0-9a-fA-F]+(?![\w])")
# Changes whenever what reccmp does to a program before Ghidra's analysis
# changes, so that analyses cached before the change are not reused.
ANALYSIS_REVISION = 3
# Bump when prepared-program mutations change; the key includes the manifest.
PREPARATION_REVISION = 13


@dataclass(frozen=True)
class _Decompiled:
    code: str | None
    error: str | None


def _literal_data_type(entity_type: EntityType | None, size: int | None) -> Any:
    """The Ghidra data type for a literal the catalog found, if any."""
    from ghidra.program.model.data import (
        DoubleDataType,
        FloatDataType,
        TerminatedStringDataType,
        TerminatedUnicodeDataType,
    )

    match entity_type, size:
        case EntityType.STRING, _:
            return TerminatedStringDataType.dataType
        case EntityType.WIDECHAR, _:
            return TerminatedUnicodeDataType.dataType
        case EntityType.FLOAT, 4:
            return FloatDataType.dataType
        case EntityType.FLOAT, 8:
            return DoubleDataType.dataType
    return None


def _decode_string(raw: bytes, entity_type: EntityType | None) -> StringValue:
    """A catalog string entity's text, up to its terminator."""
    wide = entity_type == EntityType.WIDECHAR
    text = raw.decode("utf-16-le" if wide else "latin1", errors="replace")
    return StringValue(text.split("\0", 1)[0])


def unpaired_names(
    manifest: Manifest, shared: set[str]
) -> dict[tuple[ImageId, int], str]:
    """Own-image names for unpaired entities. A name that is also a shared
    name, or that several unpaired entities of one image carry, is
    qualified with the address, so it cannot suggest a correspondence."""
    named = [
        entity
        for entity in manifest.unpaired
        if entity.name is not None and entity.entity_type not in _LITERAL_TYPES
    ]
    counts: dict[tuple[ImageId, str], int] = {}
    for entity in named:
        assert entity.name is not None
        key = (entity.image_id, entity.name)
        counts[key] = counts.get(key, 0) + 1
    result = {}
    for entity in named:
        assert entity.name is not None
        unique = counts[(entity.image_id, entity.name)] == 1
        result[(entity.image_id, entity.addr)] = (
            entity.name
            if unique and entity.name not in shared
            else f"{entity.name}@{entity.addr:#x}"
        )
    return result


def canonical_names(objects: tuple[NamedObject, ...]) -> dict[int, str]:
    """One Ghidra name per paired object, keyed by original address.

    A name shared by distinct objects is qualified with the original
    address, which identifies the pair on both sides."""
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.name] = counts.get(obj.name, 0) + 1
    return {
        obj.orig_addr: (
            obj.name if counts[obj.name] == 1 else f"{obj.name}@{obj.orig_addr:#x}"
        )
        for obj in objects
    }


def paired_reference_tokens(
    orig_refs: tuple[DataReference, ...],
    recomp_refs: tuple[DataReference, ...],
    paired_recomp_addrs: dict[int, int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Name paired addresses confirmed by at least one side's reference.

    Ghidra sometimes prints the counterpart as a raw constant without making
    a reference. The manifest still supplies its exact paired address.
    """
    by_side = []
    for refs in (orig_refs, recomp_refs):
        locations: dict[ObjectOffset, set[int]] = {}
        for ref in refs:
            if ref.object is not None:
                locations.setdefault(ref.object, set()).add(ref.address)
        by_side.append(locations)
    orig, recomp = by_side
    orig_tokens: dict[int, str] = {}
    recomp_tokens: dict[int, str] = {}
    for obj in orig.keys() | recomp.keys():
        canonical_recomp = (paired_recomp_addrs or {}).get(obj.orig_addr)
        if canonical_recomp is not None and (
            obj.orig_addr + obj.offset in orig.get(obj, ())
            or canonical_recomp + obj.offset in recomp.get(obj, ())
        ):
            orig_addr = obj.orig_addr + obj.offset
            recomp_addr = canonical_recomp + obj.offset
        elif obj in orig and obj in recomp and len(orig[obj]) == len(recomp[obj]) == 1:
            orig_addr = next(iter(orig[obj]))
            recomp_addr = next(iter(recomp[obj]))
        else:
            continue
        if orig_addr == recomp_addr:
            continue
        token = f"PAIRED_DATA_{obj.orig_addr:x}_{obj.offset:x}"
        orig_tokens[orig_addr] = token
        recomp_tokens[recomp_addr] = token
    return orig_tokens, recomp_tokens


def replace_paired_raw_addresses(code: list[str], tokens: dict[int, str]) -> None:
    """Replace referenced raw addresses, leaving literals and comments intact."""
    if not tokens:
        return

    def rename(match: re.Match[str]) -> str:
        return tokens.get(int(match.group(), 16), match.group())

    for index, line in enumerate(code):
        if line.lstrip().startswith(("/*", "//", "*")):
            continue
        parts = GhidraDiffEngine.QUOTED_LITERAL.split(line)
        code[index] = "".join(
            part if part_index % 2 else _RAW_ADDRESS.sub(rename, part)
            for part_index, part in enumerate(parts)
        )


def unquoted_raw_addresses(code: str) -> set[int]:
    """Collect displayed addresses without treating strings/comments as code."""
    found: set[int] = set()
    for line in code.splitlines():
        if line.lstrip().startswith(("/*", "//", "*")):
            continue
        for index, part in enumerate(GhidraDiffEngine.QUOTED_LITERAL.split(line)):
            if index % 2 == 0:
                found.update(
                    int(match.group(), 16) for match in _RAW_ADDRESS.finditer(part)
                )
    return found


# Matches come from the manifest through `diff_pairs` only; Ghidriff's
# unused matcher raises NotImplementedError, which pylint reads as abstract.
# pylint: disable-next=abstract-method
class ReccmpDiffEngine(GhidraDiffEngine):
    """A GhidraDiffEngine whose function matches come from a manifest."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, manifest: Manifest, *args: Any, **kwargs: Any) -> None:
        self.manifest = manifest
        # A focused comparison needs switch recovery only for its requested
        # functions. Ghidra's whole-image switch pass dominates cold analysis.
        self.focused_switch_analysis = len(manifest.functions) <= 32
        self._names = canonical_names(manifest.objects)
        self._unpaired_names = unpaired_names(manifest, set(self._names.values()))
        self._pair_types = {obj.orig_addr: obj.entity_type for obj in manifest.objects}
        self._paired_recomp_addrs = {
            obj.orig_addr: obj.recomp_addr for obj in manifest.objects
        }
        self._paired_data_addrs = {
            obj.orig_addr: obj.recomp_addr
            for obj in manifest.objects
            if obj.entity_type not in (*_FUNCTION_TYPES, *_UNNAMED_TYPES)
        }
        self._extents = {
            image_id: Extents(manifest, image_id)
            for image_id in (ImageId.ORIG, ImageId.RECOMP)
        }
        self._failures: dict[int, list[AnalysisFailure]] = {}
        self._references: dict[tuple[ImageId, int], tuple[DataReference, ...]] = {}
        self._decompiled: dict[tuple[ImageId, int], _Decompiled] = {}
        self._function_entries: dict[tuple[ImageId, int], int] = {}
        self._recomp_entries: dict[int, int] = {}
        for entry in manifest.functions:
            if entry.recomp_addr is None:
                continue
            self._recomp_entries[entry.orig_addr] = entry.recomp_addr
            for side, address in (
                (ImageId.ORIG, entry.orig_addr),
                (ImageId.RECOMP, entry.recomp_addr),
            ):
                self._function_entries[(side, address)] = entry.orig_addr
        self._sides: dict[Any, ImageId] = {}
        self._retail_signatures: dict[int, tuple[int, str]] = {}
        super().__init__(*args, **kwargs)

    # --- ghidriff hooks ---------------------------------------------------

    def get_pdb(self, prog: "Program", allow_remote: bool = True) -> None:
        """Neither program gets debug information: the comparison is of the
        binaries as Ghidra sees them, under reccmp's names only."""
        return None

    def function_matches(self) -> list[FunctionMatch]:
        """Every requested pair that has a function on both sides."""
        return [
            FunctionMatch(entry.orig_addr, entry.recomp_addr, ("reccmp",))
            for entry in self._comparable_entries()
        ]

    def normalize_ghidra_decomp_for_side(
        self,
        code: list[str],
        is_old: bool,
        entry_address: int | None = None,
        stack_setup: bool = False,
    ) -> None:
        super().normalize_ghidra_decomp(code, entry_address, stack_setup)
        side = ImageId.ORIG if is_old else ImageId.RECOMP
        if (
            entry_address is None
            or (orig_addr := self._function_entries.get((side, entry_address))) is None
        ):
            return
        orig_tokens, recomp_tokens = paired_reference_tokens(
            self._references.get((ImageId.ORIG, orig_addr), ()),
            self._references.get((ImageId.RECOMP, orig_addr), ()),
            self._paired_recomp_addrs,
        )
        original = self._decompiled.get((ImageId.ORIG, orig_addr))
        recomp_addr = self._recomp_entries.get(orig_addr)
        recompiled = (
            self._decompiled.get((ImageId.RECOMP, recomp_addr))
            if recomp_addr is not None
            else None
        )
        if (
            original is not None
            and recompiled is not None
            and original.code
            and recompiled.code
        ):
            orig_raw = unquoted_raw_addresses(original.code)
            recomp_raw = unquoted_raw_addresses(recompiled.code)
            for address in orig_raw & self._paired_data_addrs.keys():
                paired = self._paired_data_addrs[address]
                if paired in recomp_raw and address != paired:
                    token = f"PAIRED_DATA_{address:x}_0"
                    orig_tokens[address] = token
                    recomp_tokens[paired] = token
        replace_paired_raw_addresses(
            code, orig_tokens if side == ImageId.ORIG else recomp_tokens
        )

    def diff_nf_symbols(self, p1: Any, p2: Any) -> list[list[Any]]:
        """Skip Ghidriff's unused whole-image symbol inventory."""
        return [[], []]

    def decompile_func(
        self, prog: "Program", func: Any, timeout: int = 15
    ) -> DecompileResult:
        result = super().decompile_func(prog, func, timeout)
        side = self._sides.get(self._program_key(prog))
        if side is not None:
            self._decompiled[(side, func.getEntryPoint().getOffset())] = _Decompiled(
                code=result.code if result.completed else None,
                error=result.error,
            )
        return result

    def analyze_program(
        self,
        df_or_prog: Any,
        require_symbols: bool,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
    ) -> Any:
        """Correct the imports' stack purge before Ghidra's first analysis."""
        from ghidra.program.util import GhidraProgramUtilities

        # ghidriff closes the program it is handed.
        program = self.project.openProgram("/", df_or_prog.getName(), False)
        if GhidraProgramUtilities.shouldAskToAnalyze(program):
            transaction = program.startTransaction("reccmp import purges")
            try:
                correct_import_purges(program)
                if self.focused_switch_analysis:
                    self.set_analysis_option(
                        program, "Decompiler Switch Analysis", False
                    )
            finally:
                program.endTransaction(transaction, True)
        return super().analyze_program(
            program, require_symbols, force_analysis, verbose_analysis
        )

    # --- program preparation ----------------------------------------------

    def _pair_type(self, orig_addr: int) -> EntityType | None:
        return self._pair_types.get(orig_addr)

    def _comparable_entries(self) -> list[FunctionEntry]:
        return [
            entry
            for entry in self.manifest.functions
            if entry.recomp_addr is not None and entry.orig_addr not in self._failures
        ]

    def _entry_addr(self, entry: FunctionEntry, image_id: ImageId) -> int:
        addr = entry.orig_addr if image_id == ImageId.ORIG else entry.recomp_addr
        assert addr is not None
        return addr

    def prune_programs(self, paths: list[Path]) -> None:
        """Delete programs of binaries this run does not compare, such as an
        earlier recompiled build, with their pristine copies."""
        keep = {self.gen_proj_bin_name_from_path(path) for path in paths}
        root = self.project.getRootFolder()
        pristine = root.getFolder(_PRISTINE_FOLDER)
        for folder in (root, pristine):
            if folder is None:
                continue
            for domain_file in folder.getFiles():
                if domain_file.getName() not in keep:
                    domain_file.delete()

    def collect_prepared_references(self, path: Path, image_id: ImageId) -> None:
        """Refresh selection-dependent signatures and references in a cached image."""
        program = self.project.openProgram(
            "/", self.gen_proj_bin_name_from_path(path), False
        )
        try:
            self._require_functions(program, image_id)
            transaction = program.startTransaction("reccmp references")
            try:
                self._infer_requested_prototypes(program, image_id)
                self._collect_references(program, image_id)
            finally:
                program.endTransaction(transaction, True)
            self._sides[self._program_key(program)] = image_id
            self.project.save(program)
        finally:
            self.project.close(program)

    def _infer_requested_prototypes(
        self, program: "Program", image_id: ImageId
    ) -> None:
        """Infer callees for this selection, including after a prepared-cache hit."""
        from ghidra.program.model.symbol import SourceType

        known_callees = {
            obj.addr(image_id)
            for obj in self.manifest.objects
            if obj.entity_type in _FUNCTION_TYPES
        }
        known_callees.update(
            entity.addr
            for entity in self.manifest.unpaired
            if entity.image_id == image_id and entity.entity_type in _FUNCTION_TYPES
        )
        known_callees.update(
            alias.addr
            for alias in self.manifest.aliases
            if alias.image_id == image_id
            and self._pair_type(alias.canonical_orig) in _FUNCTION_TYPES
        )
        requested = [
            self._entry_addr(entry, image_id) for entry in self._comparable_entries()
        ]
        infer_requested_callee_parameters(program, requested, known_callees)
        if image_id == ImageId.ORIG:
            functions = program.getFunctionManager()
            space = program.getAddressFactory().getDefaultAddressSpace()
            self._retail_signatures = {
                obj.orig_addr: (
                    function.getParameterCount(),
                    function.getCallingConventionName(),
                )
                for obj in self.manifest.objects
                if obj.entity_type == EntityType.FUNCTION
                and (
                    function := functions.getFunctionAt(space.getAddress(obj.orig_addr))
                )
                is not None
                and function.getSignatureSource() == SourceType.ANALYSIS
            }
        else:
            correct_recompiled_signatures(
                program,
                {
                    obj.recomp_addr: (
                        obj.recomp_symbol,
                        *self._retail_signatures[obj.orig_addr],
                    )
                    for obj in self.manifest.objects
                    if obj.entity_type == EntityType.FUNCTION
                    and obj.orig_addr in self._retail_signatures
                    and obj.recomp_symbol
                },
            )

    def align_import_purges(self, orig: Path, recomp: Path) -> None:
        """Give an import whose stack purge one program does not know the
        purge the other program has for it.

        Ghidra takes an import's purge from the imported library when it
        finds the library beside the binary, and leaves it unknown when it
        does not. An unknown purge leaves the stack depth after every call
        unknown, so stack variables and parameters go missing on one side
        only. An imported function pops the same arguments whichever binary
        calls it. Runs before analysis, which the purges shape, and again
        after the pristine-project reset, which can restore an unknown purge."""
        programs = [
            self.project.openProgram("/", self.gen_proj_bin_name_from_path(path), False)
            for path in (orig, recomp)
        ]
        try:
            imports = [import_locations(program) for program in programs]
            for program, own, other in (
                (programs[0], imports[0], imports[1]),
                (programs[1], imports[1], imports[0]),
            ):
                transaction = program.startTransaction("reccmp import purges")
                try:
                    for key, location in own.items():
                        counterpart = other.get(key)
                        purge = known_purge(counterpart)
                        if purge is None or known_purge(location) is not None:
                            continue
                        # Before analysis an import is only a location; its
                        # function is what carries the purge.
                        function = location.getFunction() or location.createFunction()
                        function.setStackPurgeSize(purge)
                finally:
                    program.endTransaction(transaction, True)
                self.project.save(program)
        finally:
            for program in programs:
                self.project.close(program)

    def align_memory_permissions(self, orig: Path, recomp: Path) -> None:
        """Give the original's memory blocks the write permission of the
        recompiled blocks with the same name.

        A packer or copy protection may leave the original's read-only
        sections writable. The decompiler folds a read of read-only memory
        into its value, so the same instruction would show a constant in one
        program and a name in the other. The recompiled image is the
        linker's own output, so its permissions are the ones both get."""
        recomp_program = self.project.openProgram(
            "/", self.gen_proj_bin_name_from_path(recomp), True
        )
        try:
            writable = {
                block.getName(): block.isWrite()
                for block in recomp_program.getMemory().getBlocks()
            }
        finally:
            self.project.close(recomp_program)

        program = self.project.openProgram(
            "/", self.gen_proj_bin_name_from_path(orig), False
        )
        try:
            transaction = program.startTransaction("reccmp permissions")
            try:
                for block in program.getMemory().getBlocks():
                    write = writable.get(block.getName())
                    if write is not None and write != block.isWrite():
                        block.setWrite(write)
            finally:
                program.endTransaction(transaction, True)
            self.project.save(program)
        finally:
            self.project.close(program)

    def prepare_program(self, path: Path, image_id: ImageId) -> None:
        """Give one analyzed program reccmp's functions and names, and record
        the data each requested function refers to."""
        from ghidra.app.script import GhidraScriptUtil
        from ghidra.program.flatapi import FlatProgramAPI

        program = self.project.openProgram(
            "/", self.gen_proj_bin_name_from_path(path), False
        )
        requested = [
            self._entry_addr(entry, image_id) for entry in self._comparable_entries()
        ]
        try:
            transaction = program.startTransaction("reccmp functions")
            try:
                self._create_functions(program, image_id)
            finally:
                program.endTransaction(transaction, True)
            # Analyzers that run scripts need the script bundle host, as
            # in ghidriff's own analysis.
            GhidraScriptUtil.acquireBundleHostReference()
            try:
                FlatProgramAPI(program).analyzeChanges(program)
                if self.focused_switch_analysis:
                    recover_requested_switches(program, requested)
                    FlatProgramAPI(program).analyzeChanges(program)
            finally:
                GhidraScriptUtil.releaseBundleHostReference()
            self._require_functions(program, image_id)

            transaction = program.startTransaction("reccmp names")
            try:
                self._align_data_types(program, image_id)
                probes = {
                    obj.addr(image_id)
                    for obj in self.manifest.objects
                    if obj.name in _STACK_PROBE_NAMES
                    and obj.entity_type in _FUNCTION_TYPES
                }
                probes.update(
                    entity.addr
                    for entity in self.manifest.unpaired
                    if entity.image_id == image_id
                    and entity.name in _STACK_PROBE_NAMES
                    and entity.entity_type in _FUNCTION_TYPES
                )
                apply_stack_probe_call_fixups(program, probes)
                self._infer_requested_prototypes(program, image_id)
                self._apply_names(program, image_id)
                self._collect_references(program, image_id)
            finally:
                program.endTransaction(transaction, True)
            self._sides[self._program_key(program)] = image_id
            self.project.save(program)
        finally:
            self.project.close(program)

    def _create_functions(self, program: "Program", image_id: ImageId) -> None:
        """Make functions at the entries the catalog knows, through Ghidra's
        own commands.

        Ghidra's analysis may absorb a tail-called function into its caller
        as a separate piece of the caller's body. An entry at such a piece
        is split off into its own function. An entry inside the piece that
        holds the containing function's own entry is a conflict to report,
        not a reason to compare the containing function."""
        from ghidra.app.cmd.disassemble import DisassembleCommand
        from ghidra.app.cmd.function import CreateFunctionCmd
        from ghidra.util.task import TaskMonitor

        functions = program.getFunctionManager()
        space = program.getAddressFactory().getDefaultAddressSpace()
        requested = {
            self._entry_addr(entry, image_id): entry
            for entry in self.manifest.functions
            if entry.recomp_addr is not None
        }
        known = (
            {
                obj.addr(image_id)
                for obj in self.manifest.objects
                if obj.entity_type in _ENTRY_TYPES
            }
            | {
                entity.addr
                for entity in self.manifest.unpaired
                if entity.image_id == image_id and entity.entity_type in _ENTRY_TYPES
            }
            | {
                alias.addr
                for alias in self.manifest.aliases
                if alias.image_id == image_id
                and self._pair_type(alias.canonical_orig) in _FUNCTION_TYPES
            }
        )
        split = []
        for addr in sorted(known | requested.keys()):
            address = space.getAddress(addr)
            if functions.getFunctionAt(address) is not None:
                continue
            # Ghidra returns null when no function contains the address.
            containing: "Function | None" = functions.getFunctionContaining(address)
            if containing is not None and self._split_absorbed_piece(
                containing, address
            ):
                split.append(containing)
                containing = None
            if containing is not None:
                if addr in requested:
                    self._fail(
                        requested[addr],
                        AnalysisFailure(
                            FailureKind.ENTRY_CONFLICT,
                            image_id,
                            other_function=containing.getEntryPoint().getOffset(),
                        ),
                    )
                continue
            self._clear_data(program, address)
            # A null restricted set leaves disassembly unrestricted.
            DisassembleCommand(address, None, True).applyTo(  # type: ignore[call-overload]
                program, TaskMonitor.DUMMY
            )
            CreateFunctionCmd(address).applyTo(program, TaskMonitor.DUMMY)
        for function in split:
            CreateFunctionCmd.fixupFunctionBody(program, function, TaskMonitor.DUMMY)

    def _require_functions(self, program: "Program", image_id: ImageId) -> None:
        """Report requested entries still without a function once analysis
        is done; ghidriff diffs supplied pairs only when both resolve."""
        functions = program.getFunctionManager()
        space = program.getAddressFactory().getDefaultAddressSpace()
        for entry in self._comparable_entries():
            address = space.getAddress(self._entry_addr(entry, image_id))
            if functions.getFunctionAt(address) is None:
                self._fail(entry, AnalysisFailure(FailureKind.NO_FUNCTION, image_id))

    @staticmethod
    def _clear_data(program: "Program", address: "Address") -> None:
        """Remove data Ghidra's analysis defined over a known function entry,
        such as a string it guessed in the instruction bytes; it would keep
        the entry from being disassembled."""
        listing = program.getListing()
        data = listing.getDataContaining(address)
        if data is not None and data.isDefined():
            listing.clearCodeUnits(data.getMinAddress(), data.getMaxAddress(), False)

    @staticmethod
    def _split_absorbed_piece(function: Any, address: "Address") -> bool:
        """Remove the piece of `function`'s body that starts at `address`,
        when that piece does not hold the function's entry. Returns whether
        it was removed."""
        from ghidra.program.model.address import AddressSet

        body = function.getBody()
        piece = body.getRangeContaining(address)
        if piece is None or piece.contains(function.getEntryPoint()):
            return False
        if piece.getMinAddress() != address:
            return False
        function.setBody(body.subtract(AddressSet(piece)))
        return True

    def _fail(self, entry: FunctionEntry, failure: AnalysisFailure) -> None:
        self._failures.setdefault(entry.orig_addr, []).append(failure)

    def _align_data_types(self, program: "Program", image_id: ImageId) -> None:
        """Give paired objects the same representation in both programs.

        Ghidra's analysis may type an object as a string or a float in one
        program and leave it undefined in the other; the decompiler then
        shows a literal on one side and a name on the other. Literals the
        catalog found in both binaries get the same data type on both
        sides; every other paired object loses such one-sided typing, so
        both sides show its shared name. Pointers are left alone.

        Ghidra may also type a run of bytes that holds two catalog objects
        as one scalar, so a paired object reads as an offcut of the item
        before it. That item is cleared: the catalog says they are two.
        Arrays and structures Ghidra typed are left alone; their elements
        are how the code indexes them. Undefined items of some width inside
        a paired object are only guesses from single accesses, which differ
        between the programs, and are cleared too."""
        from ghidra.program.model.data import (
            AbstractFloatDataType,
            StringDataInstance,
            Undefined,
        )
        from ghidra.program.model.address import AddressSet
        from ghidra.program.model.util import CodeUnitInsertionException

        listing = program.getListing()
        space = program.getAddressFactory().getDefaultAddressSpace()
        for obj in self.manifest.objects:
            if obj.entity_type in _FUNCTION_TYPES or obj.entity_type in _UNNAMED_TYPES:
                continue
            address = space.getAddress(obj.addr(image_id))
            size = obj.extent(image_id)
            data_type = _literal_data_type(obj.entity_type, size)
            containing = listing.getDataContaining(address)
            if (
                containing is not None
                and containing.isDefined()
                and containing.getNumComponents() == 0
                and containing.getMinAddress() != address
            ):
                listing.clearCodeUnits(
                    containing.getMinAddress(), containing.getMaxAddress(), False
                )
            existing = listing.getDataAt(address)
            if data_type is not None and size:
                end = address.add(size - 1)
                if listing.getInstructions(AddressSet(address, end), True).hasNext():
                    # The catalog's extent overlaps code: leave Ghidra's
                    # listing alone rather than trust the extent.
                    continue
                listing.clearCodeUnits(address, end, False)
                try:
                    listing.createData(address, data_type)
                except CodeUnitInsertionException:
                    pass
            elif existing is not None and (
                StringDataInstance.isString(existing)
                or isinstance(existing.getDataType(), AbstractFloatDataType)
            ):
                listing.clearCodeUnits(address, existing.getMaxAddress(), False)
            if data_type is None and size:
                # Undefined items of some width inside the object are
                # guesses from single accesses, which differ between the
                # programs.
                guessed = [
                    item
                    for item in listing.getDefinedData(
                        AddressSet(address, address.add(size - 1)), True
                    )
                    if Undefined.isUndefined(item.getDataType())
                ]
                for item in guessed:
                    listing.clearCodeUnits(
                        item.getMinAddress(), item.getMaxAddress(), False
                    )

    def _apply_names(self, program: "Program", image_id: ImageId) -> None:
        functions = program.getFunctionManager()
        space = program.getAddressFactory().getDefaultAddressSpace()
        named = (
            [
                (obj.addr(image_id), self._names[obj.orig_addr])
                for obj in self.manifest.objects
                if obj.entity_type not in _UNNAMED_TYPES
            ]
            + [
                (addr, name)
                for (side, addr), name in self._unpaired_names.items()
                if side == image_id
            ]
            + [
                # A duplicate has the identity of its pair, so its name too.
                (alias.addr, self._names[alias.canonical_orig])
                for alias in self.manifest.aliases
                if alias.image_id == image_id and alias.canonical_orig in self._names
            ]
        )
        for addr, name in named:
            address = space.getAddress(addr)
            function = functions.getFunctionAt(address)
            if function is not None:
                self._rename_function(function, name)
            else:
                self._label(program, address, name)

    @staticmethod
    def _ghidra_name(name: str) -> str:
        from ghidra.program.model.symbol import SymbolUtilities

        return SymbolUtilities.replaceInvalidChars(name, True)

    def _rename_function(self, function: Any, name: str) -> None:
        from ghidra.program.model.symbol import SourceType

        ghidra_name = self._ghidra_name(name)
        # The PE loader labels an export's entry with its decorated name; a
        # function cannot take a name another symbol already holds there.
        primary = function.getSymbol()
        table = function.getProgram().getSymbolTable()
        for symbol in table.getSymbols(function.getEntryPoint()):
            if symbol != primary and symbol.getName() == ghidra_name:
                symbol.delete()
        function.setName(ghidra_name, SourceType.USER_DEFINED)

    def _label(self, program: "Program", address: "Address", name: str) -> None:
        from ghidra.program.model.symbol import SourceType

        program.getSymbolTable().createLabel(
            address, self._ghidra_name(name), SourceType.USER_DEFINED
        ).setPrimary()

    # --- referenced data --------------------------------------------------

    def _collect_references(self, program: "Program", image_id: ImageId) -> None:
        functions = program.getFunctionManager()
        references = program.getReferenceManager()
        space = program.getAddressFactory().getDefaultAddressSpace()
        for entry in self._comparable_entries():
            function = functions.getFunctionAt(
                space.getAddress(self._entry_addr(entry, image_id))
            )
            if function is None:
                continue
            targets: dict[Use, "Address"] = {}
            for instruction in program.getListing().getInstructions(
                function.getBody(), True
            ):
                for ref in references.getReferencesFrom(instruction.getAddress()):
                    to = ref.getToAddress()
                    if ref.getReferenceType().isFlow() or not to.isMemoryAddress():
                        continue
                    if functions.getFunctionContaining(
                        to
                    ) is not None or not self._is_data(image_id, to.getOffset()):
                        continue
                    operand = ref.getOperandIndex()
                    if register_operand(instruction, operand) or bitwise_scalar_operand(
                        instruction, operand, to.getOffset()
                    ):
                        continue
                    bound = self._compared_end(image_id, instruction, operand, to)
                    access = access_size(instruction, to.getOffset())
                    if bound is not None:
                        self._name_end(program, instruction, ref, bound)
                    else:
                        bound = self._iterated_past_end(
                            image_id, instruction, operand, to
                        )
                    if bound is None and access is None:
                        to = self._string_start(program, image_id, to)
                    targets.setdefault(Use(to.getOffset(), bound, access), to)
            self._references[(image_id, entry.orig_addr)] = tuple(
                self._reference(program, image_id, to, use)
                for use, to in sorted(
                    targets.items(),
                    key=lambda item: (
                        item[0].addr,
                        item[0].bound is None,
                        item[0].access or 0,
                    ),
                )
            )

    def _string_start(
        self, program: "Program", image_id: ImageId, address: "Address"
    ) -> "Address":
        """The string an address inside a string belongs to.

        Scanning a string, as an inlined strlen does, leaves Ghidra's
        constant propagation with a reference a byte or so into it; the
        function refers to the string, not to its tail."""
        from ghidra.program.model.data import StringDataInstance

        located = self._extents[image_id].containing(address.getOffset())
        if located is not None:
            if located.entity_type in STRING_TYPES:
                return address.subtract(located.offset)
            return address
        data = program.getListing().getDataContaining(address)
        if data is not None and StringDataInstance.isString(data):
            return data.getMinAddress()
        return address

    def _compared_end(
        self, image_id: ImageId, instruction: Any, operand: int, to: "Address"
    ) -> ObjectOffset | None:
        """The paired array a constant an instruction only compares with is
        a loop bound over (see `Extents.bound_at`), with the bound's offset.

        A bound past an array's end lies at or in whatever the linker placed
        next, which differs between the binaries; Ghidra names it after
        that."""
        ended = self._extents[image_id].bound_at(to.getOffset())
        if ended is None or ended.named is None:
            return None
        scalar = instruction.getScalar(operand)
        if scalar is None or scalar.getUnsignedValue() != to.getOffset():
            return None
        if not only_compared(instruction, to.getOffset()):
            return None
        orig_addr = ended.named.orig_addr
        return ObjectOffset(orig_addr, self._names[orig_addr], ended.offset)

    def _iterated_past_end(
        self, image_id: ImageId, instruction: Any, operand: int, to: "Address"
    ) -> ObjectOffset | None:
        """The paired array a register-relative access runs past, when
        Ghidra's constant propagation follows a loop over it one iteration
        further than the loop bound allows.

        The address lies past the array's end by less than the array's
        size, in whatever the linker placed next, and differs between the
        binaries. It is the array's, at that offset, with no contents."""
        from ghidra.program.model.lang import OperandType
        from ghidra.program.model.scalar import Scalar

        if not OperandType.isDynamic(instruction.getOperandType(operand)):
            return None
        if any(
            isinstance(part, Scalar) and part.getUnsignedValue() == to.getOffset()
            for part in instruction.getOpObjects(operand)
        ):
            # The operand names the address itself: a table access.
            return None
        past = self._extents[image_id].bound_at(to.getOffset())
        if past is None or past.named is None or past.size is None:
            return None
        if past.offset < past.size:
            return None
        orig_addr = past.named.orig_addr
        return ObjectOffset(orig_addr, self._names[orig_addr], past.offset)

    def _name_end(
        self, program: "Program", instruction: Any, ref: Any, end: ObjectOffset
    ) -> None:
        """Show the compared constant as its offset from its array. The
        decompiler shows an equate for the constant, also when it adjusts
        the constant by one to rewrite the comparison."""
        name = self._ghidra_name(f"{end.name}+{end.offset:#x}")
        value = ref.getToAddress().getOffset()
        equates = program.getEquateTable()
        equate = equates.getEquate(name) or equates.createEquate(name, value)
        if equate.getValue() == value:
            equate.addReference(instruction.getAddress(), ref.getOperandIndex())

    def _is_data(self, image_id: ImageId, addr: int) -> bool:
        """Import slots are compared by the import name Ghidra shows in the
        code; jump tables inside a function's extent are part of its code."""
        located = self._extents[image_id].containing(addr)
        return located is None or (
            located.entity_type not in _UNNAMED_TYPES
            and located.entity_type not in _FUNCTION_TYPES
        )

    def _reference(
        self,
        program: "Program",
        image_id: ImageId,
        address: "Address",
        use: Use,
    ) -> DataReference:
        if use.bound is not None:
            return DataReference(address.getOffset(), use.bound, PastEnd())
        located = self._extents[image_id].containing(address.getOffset())
        if located is None:
            contents = (
                self._accessed_contents(program, image_id, address, use.access)
                if use.access is not None
                else self._ghidra_contents(program, image_id, address)
            )
            return DataReference(address.getOffset(), None, contents)
        obj = None
        if located.named is not None:
            obj = ObjectOffset(
                located.named.orig_addr,
                self._names[located.named.orig_addr],
                located.offset,
            )
            if located.offset:
                # Name a location inside a paired object by the object and
                # offset, so `array + 4` cannot read as `array`.
                self._label(program, address, f"{obj.name}+{obj.offset:#x}")
        return DataReference(
            address.getOffset(),
            obj,
            self._catalog_contents(program, image_id, address, located),
        )

    def _catalog_contents(
        self,
        program: "Program",
        image_id: ImageId,
        address: "Address",
        located: Located,
    ) -> Contents:
        """Contents over the catalog's extent for the entity, the same way
        in both images, whatever Ghidra's data typing on either side."""
        if located.size is None:
            return UnknownExtent()
        read = self._read(program, address, located.size - located.offset)
        if read is None:
            return Uninitialized()
        raw, relocated = read
        if relocated and self._relocation_at(program, address) and len(raw) >= 4:
            return self._pointer(program, image_id, raw)
        if located.entity_type in STRING_TYPES and not relocated:
            return _decode_string(raw, located.entity_type)
        return RawBytes(raw, relocated, extent_known=True)

    def _accessed_contents(
        self, program: "Program", image_id: ImageId, address: "Address", size: int
    ) -> Contents:
        """The bytes an instruction loads or stores at a location the
        catalog does not know: what the function uses, whatever Ghidra's
        data typing there (two float constants typed as one string)."""
        read = self._read(program, address, size)
        if read is None:
            return Uninitialized()
        raw, relocated = read
        if relocated and self._relocation_at(program, address) and len(raw) >= 4:
            return self._pointer(program, image_id, raw)
        return RawBytes(raw, relocated, extent_known=True)

    def _ghidra_contents(
        self, program: "Program", image_id: ImageId, address: "Address"
    ) -> Contents:
        """Contents of a location the catalog does not know, as far as
        Ghidra's own data typing reaches."""
        data = program.getListing().getDataContaining(address)
        if data is None or not data.isDefined():
            return UnknownExtent()
        string = self._ghidra_string(program, address)
        if string is not None:
            return string
        offset = address.subtract(data.getMinAddress())
        read = self._read(program, address, data.getLength() - offset)
        if read is None:
            return Uninitialized()
        raw, relocated = read
        if relocated and self._relocation_at(program, address) and len(raw) >= 4:
            return self._pointer(program, image_id, raw)
        return RawBytes(raw, relocated, extent_known=False)

    @staticmethod
    def _ghidra_string(program: "Program", address: "Address") -> StringValue | None:
        """The string Ghidra's analysis defined at (or around) an address."""
        from ghidra.program.model.data import StringDataInstance

        data = program.getListing().getDataContaining(address)
        if data is None or not StringDataInstance.isString(data):
            return None
        string = StringDataInstance.getStringDataInstance(data)
        offset = address.subtract(data.getMinAddress())
        if offset:
            string = string.getByteOffcut(offset)
        value = string.getStringValue()
        return StringValue(str(value)) if value is not None else None

    @staticmethod
    def _relocation_at(program: "Program", address: "Address") -> bool:
        return bool(program.getRelocationTable().hasRelocation(address))

    @staticmethod
    def _read(
        program: "Program", address: "Address", length: int
    ) -> tuple[bytes, bool] | None:
        """Up to ``length`` initialized bytes and whether any is relocated."""
        import jpype
        from ghidra.program.model.address import AddressSet

        block = program.getMemory().getBlock(address)
        if block is None or not block.isInitialized():
            return None
        length = min(length, _RAW_LIMIT, block.getEnd().subtract(address) + 1)
        if length <= 0:
            return b"", False
        buffer = jpype.JArray(jpype.JByte)(length)
        program.getMemory().getBytes(address, buffer)
        relocated = (
            program.getRelocationTable()
            .getRelocations(AddressSet(address, address.add(length - 1)))
            .hasNext()
        )
        return bytes(b & 0xFF for b in buffer), relocated

    def _pointer(self, program: "Program", image_id: ImageId, raw: bytes) -> Contents:
        space = program.getAddressFactory().getDefaultAddressSpace()
        target = space.getAddress(int.from_bytes(raw[:4], "little"))
        return PointerValue(self._target(program, image_id, target))

    def _target(
        self, program: "Program", image_id: ImageId, target: "Address"
    ) -> ObjectOffset | StringValue | None:
        """What a pointer points at, in terms that mean the same on both
        sides: a paired object, or a string's contents."""
        located = self._extents[image_id].containing(target.getOffset())
        if (
            located is not None
            and located.entity_type in _FUNCTION_TYPES
            and located.offset
        ):
            # A code address inside a function: its offset depends on code
            # layout, not on what is referenced.
            return None
        if located is not None and located.named is not None:
            return ObjectOffset(
                located.named.orig_addr,
                self._names[located.named.orig_addr],
                located.offset,
            )
        if (
            located is not None
            and located.entity_type in STRING_TYPES
            and located.size is not None
        ):
            read = self._read(program, target, located.size - located.offset)
            if read is not None and not read[1]:
                return _decode_string(read[0], located.entity_type)
        return self._ghidra_string(program, target)

    # --- results ----------------------------------------------------------

    def _normalized(self, image_id: ImageId, addr: int) -> list[str] | None:
        decompiled = self._decompiled.get((image_id, addr))
        if decompiled is None or decompiled.code is None:
            return None
        lines = decompiled.code.splitlines(True)
        stack_setup = (
            "replaced with injection: alloca_probe" in decompiled.code
            or "ExceptionList" in decompiled.code
        )
        self.normalize_ghidra_decomp_for_side(
            lines, image_id == ImageId.ORIG, addr, stack_setup
        )
        return lines

    def results(self) -> list[FunctionResult]:
        """One result for every function the manifest requested."""
        results = []
        for entry in self.manifest.functions:
            failures = list(self._failures.get(entry.orig_addr, ()))
            if entry.recomp_addr is not None and not failures:
                for image_id in (ImageId.ORIG, ImageId.RECOMP):
                    decompiled = self._decompiled.get(
                        (image_id, self._entry_addr(entry, image_id))
                    )
                    if decompiled is not None and decompiled.error is not None:
                        failures.append(
                            AnalysisFailure(
                                FailureKind.DECOMPILE_ERROR,
                                image_id,
                                message=decompiled.error,
                            )
                        )
            results.append(
                classify(
                    entry,
                    failures=tuple(failures),
                    orig_code=(
                        self._normalized(ImageId.ORIG, entry.orig_addr)
                        if entry.recomp_addr is not None
                        else None
                    ),
                    recomp_code=(
                        self._normalized(ImageId.RECOMP, entry.recomp_addr)
                        if entry.recomp_addr is not None
                        else None
                    ),
                    orig_refs=self._references.get((ImageId.ORIG, entry.orig_addr), ()),
                    recomp_refs=self._references.get(
                        (ImageId.RECOMP, entry.orig_addr), ()
                    ),
                )
            )
        return results
