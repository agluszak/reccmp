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

from ghidriff import GhidraDiffEngine

from reccmp.compare.manifest import FunctionEntry, Manifest, NamedObject
from reccmp.types import EntityType, ImageId
from .results import (
    AnalysisFailure,
    Contents,
    DataReference,
    FailureKind,
    FunctionResult,
    ObjectOffset,
    PointerValue,
    RawBytes,
    StringValue,
    Uninitialized,
    UnknownExtent,
    classify,
)

if TYPE_CHECKING:
    from ghidra.program.model.address import Address
    from ghidra.program.model.listing import Program

# Ghidra names imports itself, the same way in both programs.
_UNNAMED_TYPES = (EntityType.IMPORT, EntityType.IMPORT_THUNK)
_FUNCTION_TYPES = (EntityType.FUNCTION, EntityType.VTORDISP, EntityType.THUNK)
# Named by their contents in the catalog; shown by their contents instead.
_LITERAL_TYPES = (EntityType.STRING, EntityType.WIDECHAR, EntityType.FLOAT)
# Upper bound on the bytes shown for one referenced location.
_RAW_LIMIT = 64
_STRING_TYPES = (EntityType.STRING, EntityType.WIDECHAR)
_PRISTINE_FOLDER = "pristine"


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

    def containing(self, addr: int) -> _Located | None:
        i = bisect.bisect_right(self._starts, addr) - 1
        if i < 0:
            return None
        span = self._spans[i]
        offset = addr - span.start
        if offset == 0 or (span.size is not None and offset < span.size):
            return _Located(span.start, offset, span.size, span.entity_type, span.named)
        return None


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
        """Every requested pair that has a function on both sides."""
        self._sides = {
            self._program_key(p1): ImageId.ORIG,
            self._program_key(p2): ImageId.RECOMP,
        }
        orig_functions = p1.getFunctionManager()
        recomp_functions = p2.getFunctionManager()
        orig_space = p1.getAddressFactory().getDefaultAddressSpace()
        recomp_space = p2.getAddressFactory().getDefaultAddressSpace()

        matched = []
        for entry in self._comparable_entries():
            assert entry.recomp_addr is not None
            orig = orig_functions.getFunctionAt(orig_space.getAddress(entry.orig_addr))
            recomp = recomp_functions.getFunctionAt(
                recomp_space.getAddress(entry.recomp_addr)
            )
            if orig is not None and recomp is not None:
                matched.append([orig.getSymbol(), recomp.getSymbol(), ["reccmp"]])
        return [[], matched, []]

    def syms_need_diff(self, *_: Any, **__: Any) -> bool:
        """Decompile every requested pair. Equal size and reference counts
        are exactly where subtle differences hide."""
        return True

    def decompile_func(self, prog: "Program", func: Any, timeout: int = 15) -> Any:
        error, code = super().decompile_func(prog, func, timeout)
        side = self._sides.get(self._program_key(prog))
        if side is not None:
            self._decompiled[(side, func.getEntryPoint().getOffset())] = _Decompiled(
                code=str(code) if not error else None,
                error=str(error) if error else None,
            )
        return error, code

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

            transaction = program.startTransaction("reccmp names")
            try:
                self._align_data_types(program, image_id)
                self._apply_names(program, image_id)
                self._collect_references(program, image_id)
            finally:
                program.endTransaction(transaction, True)
            self.project.save(program)
        finally:
            self.project.close(program)

    def _create_functions(self, program: "Program", image_id: ImageId) -> None:
        """Make functions at the entries the catalog knows, through Ghidra's
        own commands. An entry inside another function is a conflict to
        report, not a reason to compare the containing function."""
        from ghidra.app.cmd.disassemble import DisassembleCommand
        from ghidra.app.cmd.function import CreateFunctionCmd

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
        for addr in sorted(known | requested.keys()):
            address = space.getAddress(addr)
            if functions.getFunctionAt(address) is not None:
                continue
            containing = functions.getFunctionContaining(address)
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
            DisassembleCommand(address, None, True).applyTo(program)
            CreateFunctionCmd(address).applyTo(program)
            if functions.getFunctionAt(address) is None and addr in requested:
                self._fail(
                    requested[addr], AnalysisFailure(FailureKind.NO_FUNCTION, image_id)
                )

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
            targets: dict[int, "Address"] = {}
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
                        targets.setdefault(to.getOffset(), to)
            self._references[(image_id, entry.orig_addr)] = tuple(
                self._reference(program, image_id, targets[addr])
                for addr in sorted(targets)
            )

    def _is_data(self, image_id: ImageId, addr: int) -> bool:
        """Import slots are compared by the import name Ghidra shows in the
        code; jump tables inside a function's extent are part of its code."""
        located = self._extents[image_id].containing(addr)
        return located is None or (
            located.entity_type not in _UNNAMED_TYPES
            and located.entity_type not in _FUNCTION_TYPES
        )

    def _reference(
        self, program: "Program", image_id: ImageId, address: "Address"
    ) -> DataReference:
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
