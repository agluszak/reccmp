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

    def inline_code(self, image, address):
        """Read native retry text for the synthetic fixture."""
        return self._inline_decompiled[(image, address)].code

    def register_side(self, program, image):
        """Use the same program identity as production decompilation pools."""
        self._sides[self._program_key(program)] = image

    def find_matches(self, *_args, **_kwargs):
        """Fixture pairs are authoritative; the matcher must never run."""
        raise AssertionError("Native fixture uses manifest pairs")


def _create_shared_callee_functions(prog, image):
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.model.data import IntegerDataType, PointerDataType
    from ghidra.program.model.listing import ParameterImpl, Function
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import ConsoleTaskMonitor
    from java.io import ByteArrayInputStream  # type: ignore[import-not-found]

    api = FlatProgramAPI(prog)
    transaction = prog.startTransaction("shared inline callee fixture")
    try:
        space = prog.getAddressFactory().getDefaultAddressSpace()
        blob = bytearray(b"\x90" * 0x500)

        def call(site, target):
            return b"\xe8" + (target - site - 5).to_bytes(4, "little", signed=True)

        def wrapper(address):
            return (
                bytes.fromhex("568b74240856")
                + call(address + 6, 0x1200)
                + bytes.fromhex("83c40456")
                + call(address + 15, 0x1300)
                + bytes.fromhex("83c4048b065ec3")
            )

        caller = (
            wrapper(0x1000)
            if image == ImageId.ORIG
            else bytes.fromhex("ff742404")
            + call(0x1004, 0x1100)
            + bytes.fromhex("83c404c3")
        )
        code = {
            0x1000: caller,
            0x1100: wrapper(0x1100),
            0x1200: (
                bytes.fromhex("568b742408833e00740956")
                + call(0x120B, 0x1300)
                + bytes.fromhex("83c4048b065ec3")
            ),
            0x1300: (
                bytes.fromhex("8b442404833800750650")
                + call(0x130A, 0x1400)
                + bytes.fromhex("c3")
            ),
            0x1400: bytes.fromhex("cc"),
        }
        for address, body in code.items():
            blob[address - 0x1000 : address - 0x1000 + len(body)] = body
        prog.getMemory().createInitializedBlock(
            "text",
            space.getAddress(0x1000),
            ByteArrayInputStream(bytes(blob)),
            len(blob),
            ConsoleTaskMonitor(),
            False,
        )
        for address, name in [
            (0x1400, "Abort"),
            (0x1300, "SetState"),
            (0x1200, "Read"),
            (0x1100, "Wrapper"),
            (0x1000, "Caller"),
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
            if address == 0x1400:
                function.setNoReturn(True)
    finally:
        prog.endTransaction(transaction, True)


def test_nested_inline_callees_with_shared_branching_body(pytestconfig):
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
    binary = BinaryInput(Path("synthetic-x86"), "shared-callee")
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
                32,
                32,
                PairBasis.ANNOTATION,
            )
            for address, name in [
                (0x1100, "Wrapper"),
                (0x1200, "Read"),
                (0x1300, "SetState"),
            ]
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
            _create_shared_callee_functions(prog, image)
            engine.register_side(prog, image)
        old, new = programs[ImageId.ORIG], programs[ImageId.RECOMP]
        engine.setup_decompliers(old, new, pair_count=1)
        for program in programs.values():
            caller = program.getFunctionManager().getFunctionAt(
                program.getAddressFactory().getDefaultAddressSpace().getAddress(0x1000)
            )
            assert engine.decompile_func(program, caller).completed
        engine.shutdown_decompilers(old, new)
        engine.normalize_inlining(programs)
        result = engine.results()[0]
        assert result.ordinary.text is not None
        assert result.ordinary.text.body_diff
        assert result.inline_callees == (0x1100, 0x1200)
        assert result.inline is not None
        assert result.inline.text is not None
        assert result.inline.text.body_diff is not None
        assert not result.selected.failures
        for program in programs.values():
            space = program.getAddressFactory().getDefaultAddressSpace()
            assert all(
                not program.getFunctionManager()
                .getFunctionAt(space.getAddress(address))
                .isInline()
                for address in (0x1100, 0x1200, 0x1300)
            )
    finally:
        for program in programs.values():
            program.release(consumer)


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
        assert result.ordinary.text is not None
        assert result.ordinary.text.body_diff
        assert result.inline_callees == (0x1100,)
        assert result.inline is not None
        assert result.inline.text is not None
        assert result.inline.text.body_diff is not None
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


def _create_tail_call_functions(prog, image, branching=False):
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
            bytes.fromhex(
                "8b442404c74008020000008338007408c7400403000000c3c7400404000000c3"
                if branching
                else "8b442404c70002000000c7400403000000c3"
            )
            if image == ImageId.ORIG
            else bytes.fromhex("ff742404e8f700000083c404c3")
        )
        blob[: len(code)] = code
        pair = bytes.fromhex(
            "8b442404c7400802000000e9f0000000"
            if branching
            else "8b442404c70002000000e9f1000000"
        )
        blob[0x100 : 0x100 + len(pair)] = pair
        overlay = bytes.fromhex(
            "8b4424048338007408c7400403000000c3c7400404000000c3"
            if branching
            else "8b442404c7400403000000c3"
        )
        blob[0x200 : 0x200 + len(overlay)] = overlay
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


@pytest.mark.parametrize("branching", [False, True])
def test_native_inline_retry_follows_tail_call(pytestconfig, branching):
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
            for address, name, size in [
                (0x1100, "Pair", 16 if branching else 15),
                (0x1200, "Overlay", 25 if branching else 12),
            ]
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
            _create_tail_call_functions(prog, image, branching)
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
        assert result.inline is not None
        assert result.inline.text is not None
        if branching:
            # Hard inline expansion can choose a different if/return layout.
            # Check the native body, rather than treating that layout as equal.
            code = engine.inline_code(ImageId.RECOMP, 0x1000)
            assert code is not None
            assert "Overlay(" not in code and "Pair(" not in code, code
            assert "if  {" not in code
            assert "a[2] = 2" in code
            assert "a[1] = 3" in code and "a[1] = 4" in code
            assert "*a == 0" in code
            assert not any(
                "prevents inlining" in w.message for w in result.inline.warnings
            )
        else:
            assert result.outcome == Outcome.NO_DIFFERENCES, "".join(
                result.inline.text.body_diff
            )
        from ghidra.program.model.listing import FlowOverride

        instruction = new.getListing().getInstructionAt(
            new.getAddressFactory()
            .getDefaultAddressSpace()
            .getAddress(0x110B if branching else 0x110A)
        )
        assert instruction.getFlowOverride() == FlowOverride.NONE
    finally:
        for program in programs.values():
            program.release(consumer)


def _create_cycle_functions(prog, image):
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
        blob = bytearray(b"\x90" * 0x200)
        # Caller(int *a) returns GetFoo(a) + 1. GetFoo calls the unpaired Back
        # and then loads *a; Back calls GetFoo again. Retail expands GetFoo, so
        # its Caller calls Back directly.
        code = (
            bytes.fromhex("ff742404e87701000083c4048b4424048b0040c3")
            if image == ImageId.ORIG
            else bytes.fromhex("ff742404e8f700000083c40440c3")
        )
        blob[: len(code)] = code
        blob[0x100:0x113] = bytes.fromhex("ff742404e87700000083c4048b4424048b00c3")
        blob[0x180:0x18D] = bytes.fromhex("ff742404e877ffffff83c404c3")
        prog.getMemory().createInitializedBlock(
            "text",
            space.getAddress(0x1000),
            ByteArrayInputStream(bytes(blob)),
            len(blob),
            ConsoleTaskMonitor(),
            False,
        )
        for address, name in [(0x1000, "Caller"), (0x1100, "GetFoo"), (0x1180, "Back")]:
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


def test_native_inline_retry_through_cycle_left_as_call(pytestconfig):
    """A cycle through a function the retry does not mark inline stays one
    ordinary call deep, so the callee on that cycle is still substituted."""
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
    helper = NamedObject(
        0x1100, 0x1100, "GetFoo", EntityType.FUNCTION, 19, 19, PairBasis.ANNOTATION
    )
    manifest = Manifest("T", binary, binary, (entry,), (helper,), ())
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
            _create_cycle_functions(prog, image)
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
        assert result.inline_callees == (0x1100,)
        assert not result.selected.failures
        assert result.outcome == Outcome.NO_DIFFERENCES
    finally:
        for program in programs.values():
            program.release(consumer)


def test_native_tail_only_callee_parameter_inference(pytestconfig):
    """A focused tail caller infers ECX just as a larger selection does."""
    if not pytestconfig.getoption("--require-ghidra"):
        pytest.skip("Native decompiler test requires --require-ghidra")
    HeadlessPyGhidraLauncher().start()
    from ghidra.program.database import ProgramDB
    from ghidra.program.util import DefaultLanguageService
    from ghidra.program.model.lang import LanguageID, CompilerSpecID
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import TaskMonitor
    from java.io import ByteArrayInputStream  # type: ignore[import-not-found]
    from java.lang import Object  # type: ignore[import-not-found]
    from reccmp.ghidriff.preparation import infer_requested_callee_parameters

    lang = DefaultLanguageService.getLanguageService().getLanguage(
        LanguageID("x86:LE:32:default")
    )
    consumer = Object()
    for requested in ([0x1000], [0x1000, 0x1100]):
        program = ProgramDB(
            "tail-only",
            lang,
            lang.getCompilerSpecByID(CompilerSpecID("windows")),
            consumer,
        )
        transaction = program.startTransaction("authored tail caller")
        try:
            api = FlatProgramAPI(program)
            blob = bytearray(b"\x90" * 0x200)
            # Load the object into ECX, then jump to a field-reading callee.
            blob[:11] = bytes.fromhex("8b0d00200000e9f5000000")
            blob[0x100:0x104] = bytes.fromhex("8b412cc3")
            space = program.getAddressFactory().getDefaultAddressSpace()
            program.getMemory().createInitializedBlock(
                "text",
                space.getAddress(0x1000),
                ByteArrayInputStream(bytes(blob)),
                len(blob),
                TaskMonitor.DUMMY,
                False,
            )
            for address in (0x1000, 0x1100):
                api.disassemble(space.getAddress(address))
                api.createFunction(space.getAddress(address), f"f{address:x}")
            callee = program.getFunctionManager().getFunctionAt(
                space.getAddress(0x1100)
            )
            assert callee.getSignatureSource() == SourceType.DEFAULT
            assert callee.getParameterCount() == 0
            for _ in range(
                2
            ):  # Reuse prepared analysis without changing inferred parameters.
                infer_requested_callee_parameters(program, requested, {0x1100})
                assert callee.getSignatureSource() == SourceType.ANALYSIS
                assert callee.getParameterCount() == 1
                assert callee.getParameter(0).getRegister().getName() == "ECX"
            from reccmp.ghidriff.preparation import (
                apply_reviewed_scalar_returns,
                preparation_corrections,
            )

            apply_reviewed_scalar_returns(program, {0x1100: "bool"})
            [correction] = preparation_corrections(program)
            assert correction.evidence == "reviewed-retail-return"
            apply_reviewed_scalar_returns(program, {0x1100: "bool"})
            callee.setName("catalogued_callee", SourceType.USER_DEFINED)
            assert preparation_corrections(program) == (correction,)
        finally:
            program.endTransaction(transaction, False)
            program.release(consumer)


def test_legacy_msvcrt_swprintf_signature(pytestconfig):
    if not pytestconfig.getoption("--require-ghidra"):
        pytest.skip("Native prototype test requires --require-ghidra")
    HeadlessPyGhidraLauncher().start()
    from ghidra.program.database import ProgramDB
    from ghidra.program.util import DefaultLanguageService
    from ghidra.program.model.lang import LanguageID, CompilerSpecID
    from ghidra.program.model.data import IntegerDataType
    from ghidra.program.model.listing import ParameterImpl, Function
    from ghidra.program.model.symbol import SourceType
    from java.lang import Object  # type: ignore[import-not-found]
    from reccmp.ghidriff.preparation import (
        correct_legacy_crt_signatures,
        preparation_corrections,
    )

    lang = DefaultLanguageService.getLanguageService().getLanguage(
        LanguageID("x86:LE:32:default")
    )
    consumer = Object()
    program = ProgramDB(
        "legacy-crt",
        lang,
        lang.getCompilerSpecByID(CompilerSpecID("windows")),
        consumer,
    )
    transaction = program.startTransaction("import prototype fixture")
    try:
        functions = {}
        for library in ("MSVCRT.DLL", "ucrtbase.dll"):
            location = program.getExternalManager().addExtFunction(  # type: ignore[call-overload]
                library, "swprintf", None, SourceType.IMPORTED
            )
            function = location.getFunction()
            function.replaceParameters(
                Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
                True,
                SourceType.IMPORTED,
                *(
                    ParameterImpl(name, IntegerDataType.dataType, program)
                    for name in ("buffer", "size", "format")
                ),
            )
            functions[library] = function
        correct_legacy_crt_signatures(program)
        legacy = functions["MSVCRT.DLL"]
        assert legacy.getParameterCount() == 2
        assert legacy.getCallingConventionName() == "__cdecl"
        assert legacy.hasVarArgs()
        assert legacy.getStackPurgeSize() == 0
        assert all(
            p.getDataType().getDataType().getLength() == 2
            for p in legacy.getParameters()
        )
        assert functions["ucrtbase.dll"].getParameterCount() == 3
        corrections = preparation_corrections(program)
        assert len(corrections) == 1
        correct_legacy_crt_signatures(program)
        assert preparation_corrections(program) == corrections
    finally:
        program.endTransaction(transaction, False)
        program.release(consumer)
