"""Ghidriff engine driven by reccmp's pairs and names.

Ghidra analyzes and decompiles both programs; ghidriff diffs and reports.
reccmp contributes what only the reconstruction knows: which functions
correspond, under which names, and where their source is. Both programs are
analyzed the same way, without debug information, so neither decompilation
is shaped by the reconstruction's own types.
"""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ghidriff import DecompileResult, FunctionMatch, GhidraDiffEngine

from reccmp.compare.manifest import FunctionEntry, Manifest, NamedObject
from reccmp.types import EntityType, ImageId
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
# Named by their contents in the catalog; shown by their contents instead.
_LITERAL_TYPES = (EntityType.STRING, EntityType.WIDECHAR, EntityType.FLOAT)
# Upper bound on the bytes shown for one referenced location.
_RAW_LIMIT = 64
_STRING_TYPES = (EntityType.STRING, EntityType.WIDECHAR)
_PRISTINE_FOLDER = "pristine"
# How far before the nearest array end to look for an array a loop bound
# belongs to.
_BOUND_SEARCH = 0x1000
# Changes whenever what reccmp does to a program before Ghidra's analysis
# changes, so that analyses cached before the change are not reused.
ANALYSIS_REVISION = 1


@dataclass(frozen=True)
class _Decompiled:
    code: str | None
    error: str | None


@dataclass(frozen=True)
class _Located:
    """Where a referenced address falls in the catalog of one image."""

    start: int
    offset: int
    size: int | None
    entity_type: EntityType | None
    # The pair the location belongs to; None for unpaired catalog data.
    named: NamedObject | None


class _Extents:
    """Catalog entities of one image, by the address range they occupy."""

    def __init__(self, manifest: Manifest, image_id: ImageId):
        spans = [
            _Located(obj.addr(image_id), 0, obj.extent(image_id), obj.entity_type, obj)
            for obj in manifest.objects
        ] + [
            _Located(entity.addr, 0, entity.size, entity.entity_type, None)
            for entity in manifest.unpaired
            if entity.image_id == image_id
        ]
        pairs = {obj.orig_addr: obj for obj in manifest.objects}
        spans += [
            _Located(
                alias.addr,
                0,
                alias.size or pairs[alias.canonical_orig].extent(image_id),
                pairs[alias.canonical_orig].entity_type,
                pairs[alias.canonical_orig],
            )
            for alias in manifest.aliases
            if alias.image_id == image_id and alias.canonical_orig in pairs
        ]
        spans.sort(key=lambda located: located.start)
        self._starts = [located.start for located in spans]
        self._spans = spans
        # Paired objects of known extent, by the address just past them.
        sized = sorted(
            (span for span in spans if span.named is not None and span.size),
            key=lambda span: span.start + (span.size or 0),
        )
        self._sized = sized
        self._sized_ends = [span.start + (span.size or 0) for span in sized]

    def containing(self, addr: int) -> _Located | None:
        i = bisect.bisect_right(self._starts, addr) - 1
        if i < 0:
            return None
        span = self._spans[i]
        offset = addr - span.start
        if offset == 0 or (span.size is not None and offset < span.size):
            return _Located(span.start, offset, span.size, span.entity_type, span.named)
        return None

    def bound_at(self, addr: int) -> _Located | None:
        """The paired object a loop bound at `addr` belongs to, located at
        the bound's offset from it.

        A loop over an array compares its pointer with the address just
        past the array, or, stepping through one field of each element,
        with that field's address in the element past the last: past the
        array's end by less than one element, so by less than its size.
        The exact end comes first; then a paired object starting at or
        holding `addr` (other than a string, whose middle nothing compares
        with), which the code compares with as itself; then the nearest
        array whose end is that close."""
        i = bisect.bisect_right(self._sized_ends, addr)
        exact = i > 0 and self._sized_ends[i - 1] == addr
        if not exact:
            inside = self.containing(addr)
            if inside is not None and inside.named is not None:
                if inside.offset == 0 or inside.entity_type not in _STRING_TYPES:
                    return None
        for span in reversed(self._sized[:i]):
            assert span.size is not None
            offset = addr - span.start
            if offset < 2 * span.size:
                return _Located(
                    span.start, offset, span.size, span.entity_type, span.named
                )
            if self._sized_ends[i - 1] - (span.start + span.size) > _BOUND_SEARCH:
                break
        return None


def _only_compared(instruction: Any, value: int) -> bool:
    """Whether an instruction uses a constant only to compare with it, by
    its p-code: in comparisons, or in a difference kept only in a temporary
    for the flags it sets."""
    from ghidra.program.model.pcode import PcodeOp

    comparisons = {
        PcodeOp.INT_EQUAL,
        PcodeOp.INT_NOTEQUAL,
        PcodeOp.INT_LESS,
        PcodeOp.INT_SLESS,
        PcodeOp.INT_LESSEQUAL,
        PcodeOp.INT_SLESSEQUAL,
        PcodeOp.INT_CARRY,
        PcodeOp.INT_SCARRY,
        PcodeOp.INT_SBORROW,
    }
    used = False
    for op in instruction.getPcode():
        if not any(
            vn.isConstant() and vn.getOffset() == value for vn in op.getInputs()
        ):
            continue
        used = True
        if op.getOpcode() in comparisons:
            continue
        output = op.getOutput()
        if (
            op.getOpcode() == PcodeOp.INT_SUB
            and output is not None
            and output.isUnique()
        ):
            continue
        return False
    return used


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


class ReccmpDiffEngine(GhidraDiffEngine):
    """A GhidraDiffEngine whose function matches come from a manifest."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, manifest: Manifest, *args: Any, **kwargs: Any) -> None:
        self.manifest = manifest
        self._names = canonical_names(manifest.objects)
        self._unpaired_names = unpaired_names(manifest, set(self._names.values()))
        self._pair_types = {obj.orig_addr: obj.entity_type for obj in manifest.objects}
        self._extents = {
            image_id: _Extents(manifest, image_id)
            for image_id in (ImageId.ORIG, ImageId.RECOMP)
        }
        self._failures: dict[int, list[AnalysisFailure]] = {}
        self._references: dict[tuple[ImageId, int], tuple[DataReference, ...]] = {}
        self._decompiled: dict[tuple[ImageId, int], _Decompiled] = {}
        self._sides: dict[Any, ImageId] = {}
        super().__init__(*args, **kwargs)

    # --- ghidriff hooks ---------------------------------------------------

    def get_pdb(self, prog: "Program", allow_remote: bool = True) -> None:
        """Neither program gets debug information: the comparison is of the
        binaries as Ghidra sees them, under reccmp's names only."""
        return None

    def find_matches(self, p1: "Program", p2: "Program") -> list:
        """Pairs come from the catalog only, through `diff_pairs`."""
        raise NotImplementedError("reccmp supplies its pairs to diff_pairs")

    def function_matches(self) -> list[FunctionMatch]:
        """Every requested pair that has a function on both sides."""
        return [
            FunctionMatch(entry.orig_addr, entry.recomp_addr, ("reccmp",))
            for entry in self._comparable_entries()
        ]

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
                self._correct_import_purges(program)
            finally:
                program.endTransaction(transaction, True)
        return super().analyze_program(
            program, require_symbols, force_analysis, verbose_analysis
        )

    # --- program preparation ----------------------------------------------

    @staticmethod
    def _correct_import_purges(program: "Program") -> None:
        """Give imports the caller cleans up after a stack purge of zero.

        With the imported library beside the binary, Ghidra takes each
        import's stack purge from its own analysis of the library. That
        analysis may count arguments pushed on a path that never returns,
        such as a failed assertion's call to exit, as the function's own
        purge. A caller that cleans up after the call then leaves the
        decompiler's stack pointer off by that amount for the rest of the
        function, and the parameter analysis gives its callers parameters
        they do not have. A variadic or `__cdecl` function never pops its
        arguments; the signature comes from the import's mangled name,
        before analysis has applied it."""
        from ghidra.app.util.demangler import DemangledFunction, DemanglerUtil
        from ghidra.program.model.lang import CompilerSpec

        for function in program.getFunctionManager().getExternalFunctions():
            if function.getStackPurgeSize() == 0:
                continue
            imported = function.getExternalLocation().getOriginalImportedName()
            demangled = DemanglerUtil.demangle(imported or function.getName())
            if not isinstance(demangled, DemangledFunction):
                continue
            if (
                demangled.getCallingConvention()
                == CompilerSpec.CALLING_CONVENTION_cdecl
                or any(
                    parameter.getType().isVarArgs()
                    for parameter in demangled.getParameters()
                )
            ):
                function.setStackPurgeSize(0)

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

    def reset_programs(self) -> None:
        """Start from the analyzed programs, without names or functions a
        previous run added. The analysis itself is kept and reused."""
        from ghidra.util.task import TaskMonitor

        root = self.project.getRootFolder()
        pristine = root.getFolder(_PRISTINE_FOLDER) or root.createFolder(
            _PRISTINE_FOLDER
        )
        for domain_file in root.getFiles():
            saved = pristine.getFile(domain_file.getName())
            if saved is None:
                domain_file.copyTo(pristine, TaskMonitor.DUMMY)
            else:
                domain_file.delete()
                saved.copyTo(root, TaskMonitor.DUMMY)

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
            finally:
                GhidraScriptUtil.releaseBundleHostReference()
            self._require_functions(program, image_id)

            transaction = program.startTransaction("reccmp names")
            try:
                self._align_data_types(program, image_id)
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
                if obj.entity_type in _FUNCTION_TYPES
            }
            | {
                entity.addr
                for entity in self.manifest.unpaired
                if entity.image_id == image_id and entity.entity_type in _FUNCTION_TYPES
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
        both sides show its shared name. Pointers are left alone."""
        from ghidra.program.model.data import (
            AbstractFloatDataType,
            StringDataInstance,
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

        function.setName(self._ghidra_name(name), SourceType.USER_DEFINED)

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
            targets: dict[tuple[int, ObjectOffset | None], "Address"] = {}
            for instruction in program.getListing().getInstructions(
                function.getBody(), True
            ):
                for ref in references.getReferencesFrom(instruction.getAddress()):
                    to = ref.getToAddress()
                    if (
                        not ref.getReferenceType().isFlow()
                        and to.isMemoryAddress()
                        and functions.getFunctionContaining(to) is None
                        and self._is_data(image_id, to.getOffset())
                    ):
                        end = self._compared_end(
                            image_id, instruction, ref.getOperandIndex(), to
                        )
                        if end is not None:
                            self._name_end(program, instruction, ref, end)
                        targets.setdefault((to.getOffset(), end), to)
            self._references[(image_id, entry.orig_addr)] = tuple(
                self._reference(program, image_id, to, end)
                for (_, end), to in sorted(
                    targets.items(), key=lambda item: (item[0][0], item[0][1] is None)
                )
            )

    def _compared_end(
        self, image_id: ImageId, instruction: Any, operand: int, to: "Address"
    ) -> ObjectOffset | None:
        """The paired array a constant an instruction only compares with is
        a loop bound over (see `_Extents.bound_at`), with the bound's offset.

        A bound past an array's end lies at or in whatever the linker placed
        next, which differs between the binaries; Ghidra names it after
        that."""
        ended = self._extents[image_id].bound_at(to.getOffset())
        if ended is None or ended.named is None:
            return None
        scalar = instruction.getScalar(operand)
        if scalar is None or scalar.getUnsignedValue() != to.getOffset():
            return None
        if not _only_compared(instruction, to.getOffset()):
            return None
        orig_addr = ended.named.orig_addr
        return ObjectOffset(orig_addr, self._names[orig_addr], ended.offset)

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
        end: ObjectOffset | None = None,
    ) -> DataReference:
        if end is not None:
            return DataReference(address.getOffset(), end, PastEnd())
        located = self._extents[image_id].containing(address.getOffset())
        if located is None:
            return DataReference(
                address.getOffset(),
                None,
                self._ghidra_contents(program, image_id, address),
            )
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
        located: _Located,
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
        if located.entity_type in _STRING_TYPES and not relocated:
            return _decode_string(raw, located.entity_type)
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
            and located.entity_type in _STRING_TYPES
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
        self.normalize_ghidra_decomp(lines)
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
