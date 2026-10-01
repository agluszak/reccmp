"""Native x86 call/body pairs exercise the actual adapter and Ghidra retry."""

from pathlib import Path

import pytest
from pyghidra import HeadlessPyGhidraLauncher

from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import BinaryInput, FunctionEntry, Manifest, NamedObject
from reccmp.ghidriff.engine import ReccmpDiffEngine
from reccmp.ghidriff.results import Outcome
from reccmp.types import EntityType, ImageId

# Java packages become importable after the launcher starts.
# pylint: disable=import-outside-toplevel,import-error


class NativeFixtureEngine(ReccmpDiffEngine):
    """Register in-memory programs normally registered during preparation."""

    def register_side(self, program, image):
        """Use the same program identity as production decompilation pools."""
        self._sides[self._program_key(program)] = image

    def find_matches(self, *_args, **_kwargs):
        """Fixture pairs are authoritative; the matcher must never run."""
        raise AssertionError("Native fixture uses manifest pairs")


def _create_functions(prog, image, changed):
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.model.data import IntegerDataType, PointerDataType
    from ghidra.program.model.listing import ParameterImpl, Function
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import ConsoleTaskMonitor
    from java.io import ByteArrayInputStream  # type: ignore[import-not-found]

    api = FlatProgramAPI(prog)
    transaction = prog.startTransaction("synthetic fixture")
    try:
        space = prog.getAddressFactory().getDefaultAddressSpace()
        # cdecl Caller(int *a): retail loads *a directly; rebuild calls
        # GetFoo(a). Native bytes include a real call and stack cleanup.
        code = (
            bytes.fromhex("8b4424048b0040c3")
            if image == ImageId.ORIG
            else bytes.fromhex("ff742404e8f700000083c40440c3")
        )
        if changed and image == ImageId.RECOMP:
            code = code[:-2] + bytes.fromhex("83c002c3")
        blob = bytearray(b"\x90" * 0x200)
        blob[: len(code)] = code
        blob[0x100:0x107] = bytes.fromhex("8b4424048b00c3")
        prog.getMemory().createInitializedBlock(
            "text",
            space.getAddress(0x1000),
            ByteArrayInputStream(bytes(blob)),
            len(blob),
            ConsoleTaskMonitor(),
            False,
        )
        for address, name in [(0x1000, "Caller"), (0x1100, "GetFoo")]:
            api.disassemble(space.getAddress(address))
            function = api.createFunction(space.getAddress(address), name)
            function.setCallingConvention("__cdecl")
            function.setReturnType(IntegerDataType.dataType, SourceType.USER_DEFINED)
            function.replaceParameters(
                Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
                True,
                SourceType.USER_DEFINED,
                ParameterImpl("a", PointerDataType(IntegerDataType.dataType), prog),
            )
    finally:
        prog.endTransaction(transaction, True)
    return function


@pytest.mark.parametrize("changed", [False, True])
def test_native_inline_retry(pytestconfig, changed):
    if not pytestconfig.getoption("--require-ghidra"):
        pytest.skip("Native decompiler test requires --require-ghidra")
    HeadlessPyGhidraLauncher().start()
    from ghidra.program.database import ProgramDB
    from ghidra.program.util import DefaultLanguageService
    from ghidra.program.model.lang import LanguageID, CompilerSpecID
    from java.lang import Object  # type: ignore[import-not-found]

    lang = DefaultLanguageService.getLanguageService().getLanguage(
        LanguageID("x86:LE:32:default")
    )
    consumer = Object()
    programs = {}
    helpers = {}
    entry = FunctionEntry(0x1000, 0x1000, "Caller", PairBasis.ANNOTATION, None, False)
    binary = BinaryInput(Path("synthetic-x86"), "fixture")
    manifest = Manifest(
        "T",
        binary,
        binary,
        (entry,),
        (
            NamedObject(
                0x1100,
                0x1100,
                "GetFoo",
                EntityType.FUNCTION,
                7,
                7,
                PairBasis.ANNOTATION,
            ),
        ),
        (),
    )
    engine = NativeFixtureEngine(manifest, threaded=False)
    try:
        for image in ImageId:
            prog = ProgramDB(
                image.name,
                lang,
                lang.getCompilerSpecByID(CompilerSpecID("windows")),
                consumer,
            )
            programs[image] = prog
            helpers[image] = _create_functions(prog, image, changed)
            engine.register_side(prog, image)
        old, new = programs[ImageId.ORIG], programs[ImageId.RECOMP]
        engine.setup_decompliers(old, new, pair_count=1)
        for image, program in programs.items():
            caller = program.getFunctionManager().getFunctionAt(
                program.getAddressFactory().getDefaultAddressSpace().getAddress(0x1000)
            )
            assert engine.decompile_func(program, caller).completed
        engine.shutdown_decompilers(old, new)
        assert engine.results()[0].outcome == Outcome.DIFFERENCES
        engine.normalize_inlining(programs)
        result = engine.results()[0]
        assert result.normal_diff
        assert result.inline_callees == (0x1100,)
        assert result.inline_normalized_diff is not None
        assert result.outcome == (
            Outcome.DIFFERENCES if changed else Outcome.NO_DIFFERENCES
        )
        assert all(not function.isInline() for function in helpers.values())
        # A later ordinary pass must still see the original call, proving both
        # rollback and decompiler-cache isolation beyond just the flag value.
        engine.setup_decompliers(old, new, pair_count=1)
        caller = new.getFunctionManager().getFunctionAt(
            new.getAddressFactory().getDefaultAddressSpace().getAddress(0x1000)
        )
        assert "GetFoo(a)" in engine.decompile_func(new, caller).code
        engine.shutdown_decompilers(old, new)
    finally:
        for program in programs.values():
            program.release(consumer)


def _create_tail_call_functions(prog, image):
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.model.data import IntegerDataType, PointerDataType
    from ghidra.program.model.listing import ParameterImpl, Function
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import ConsoleTaskMonitor
    from java.io import ByteArrayInputStream  # type: ignore[import-not-found]

    api = FlatProgramAPI(prog)
    transaction = prog.startTransaction("synthetic fixture")
    try:
        space = prog.getAddressFactory().getDefaultAddressSpace()
        blob = bytearray(b"\x90" * 0x300)
        # Caller(int *a): retail stores a[0] = 2 and a[1] = 3 directly; rebuild
        # calls Pair(a), which stores a[0] and tail-jumps to Overlay(a).
        code = (
            bytes.fromhex("8b442404c70002000000c7400403000000c3")
            if image == ImageId.ORIG
            else bytes.fromhex("ff742404e8f700000083c404c3")
        )
        blob[: len(code)] = code
        blob[0x100:0x10F] = bytes.fromhex("8b442404c70002000000e9f1000000")
        blob[0x200:0x20C] = bytes.fromhex("8b442404c7400403000000c3")
        prog.getMemory().createInitializedBlock(
            "text",
            space.getAddress(0x1000),
            ByteArrayInputStream(bytes(blob)),
            len(blob),
            ConsoleTaskMonitor(),
            False,
        )
        functions = {}
        for address, name in [
            (0x1000, "Caller"),
            (0x1100, "Pair"),
            (0x1200, "Overlay"),
        ]:
            api.disassemble(space.getAddress(address))
            function = api.createFunction(space.getAddress(address), name)
            function.setCallingConvention("__cdecl")
            function.setReturnType(IntegerDataType.dataType, SourceType.USER_DEFINED)
            function.replaceParameters(
                Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
                True,
                SourceType.USER_DEFINED,
                ParameterImpl("a", PointerDataType(IntegerDataType.dataType), prog),
            )
            functions[name] = function
    finally:
        prog.endTransaction(transaction, True)
    return functions


def test_native_inline_retry_follows_tail_call(pytestconfig):
    if not pytestconfig.getoption("--require-ghidra"):
        pytest.skip("Native decompiler test requires --require-ghidra")
    HeadlessPyGhidraLauncher().start()
    from ghidra.program.database import ProgramDB
    from ghidra.program.util import DefaultLanguageService
    from ghidra.program.model.lang import LanguageID, CompilerSpecID
    from java.lang import Object  # type: ignore[import-not-found]

    lang = DefaultLanguageService.getLanguageService().getLanguage(
        LanguageID("x86:LE:32:default")
    )
    consumer = Object()
    programs = {}
    entry = FunctionEntry(0x1000, 0x1000, "Caller", PairBasis.ANNOTATION, None, False)
    binary = BinaryInput(Path("synthetic-x86"), "fixture")
    manifest = Manifest(
        "T",
        binary,
        binary,
        (entry,),
        tuple(
            NamedObject(
                address,
                address,
                name,
                EntityType.FUNCTION,
                size,
                size,
                PairBasis.ANNOTATION,
            )
            for address, name, size in [(0x1100, "Pair", 15), (0x1200, "Overlay", 12)]
        ),
        (),
    )
    engine = NativeFixtureEngine(manifest, threaded=False)
    try:
        for image in ImageId:
            prog = ProgramDB(
                image.name,
                lang,
                lang.getCompilerSpecByID(CompilerSpecID("windows")),
                consumer,
            )
            programs[image] = prog
            _create_tail_call_functions(prog, image)
            engine.register_side(prog, image)
        old, new = programs[ImageId.ORIG], programs[ImageId.RECOMP]
        engine.setup_decompliers(old, new, pair_count=1)
        for program in programs.values():
            caller = program.getFunctionManager().getFunctionAt(
                program.getAddressFactory().getDefaultAddressSpace().getAddress(0x1000)
            )
            assert engine.decompile_func(program, caller).completed
        engine.shutdown_decompilers(old, new)
        assert engine.results()[0].outcome == Outcome.DIFFERENCES
        engine.normalize_inlining(programs)
        result = engine.results()[0]
        assert set(result.inline_callees) == {0x1100, 0x1200}
        assert result.outcome == Outcome.NO_DIFFERENCES, result.inline_normalized_diff
    finally:
        for program in programs.values():
            program.release(consumer)
