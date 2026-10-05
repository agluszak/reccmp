"""Recomp primitive constraints use only the recomp's own symbol and ABI."""

import pytest
from pyghidra import HeadlessPyGhidraLauncher

from reccmp.ghidriff.preparation import apply_recompiled_scalar_parameters

# pylint: disable=import-outside-toplevel,import-error


@pytest.mark.parametrize(
    "symbol,convention,reviewed,expected",
    [
        ("?Grow@@YAXK@Z", "__cdecl", False, "ulong"),
        ("?Grow@Array@@QAEXK@Z", "__thiscall", False, "ulong"),
        ("?Grow@@YAXPAK@Z", "__cdecl", False, "uint *"),
        ("?Grow@@YAXG@Z", "__cdecl", False, "uint *"),
        ("?Grow@@YAXKK@Z", "__cdecl", False, "uint *"),
        ("?Grow@@YGXK@Z", "__cdecl", False, "uint *"),
        ("?Grow@@YAXK@Z", "__cdecl", True, "uint *"),
        (("?Grow@@YAXK@Z", "?Other@@YAXH@Z"), "__cdecl", False, "uint *"),
        ("?Grow@@YAXW4Capacity@@@Z", "__cdecl", False, "uint *"),
    ],
)
def test_own_symbol_repairs_inferred_scalar_only(
    pytestconfig, symbol, convention, reviewed, expected
):
    if not pytestconfig.getoption("--require-ghidra"):
        pytest.skip("Native signature fixture requires --require-ghidra")
    HeadlessPyGhidraLauncher().start()
    from ghidra.program.database import ProgramDB
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.util import DefaultLanguageService
    from ghidra.program.model.lang import LanguageID, CompilerSpecID
    from ghidra.program.model.data import PointerDataType, UnsignedIntegerDataType
    from ghidra.program.model.listing import Function, ParameterImpl
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import TaskMonitor
    from java.lang import Object  # type: ignore[import-not-found]
    from java.io import ByteArrayInputStream  # type: ignore[import-not-found]

    lang = DefaultLanguageService.getLanguageService().getLanguage(
        LanguageID("x86:LE:32:default")
    )
    consumer = Object()
    program = ProgramDB(
        "own scalar",
        lang,
        lang.getCompilerSpecByID(CompilerSpecID("windows")),
        consumer,
    )
    try:
        transaction = program.startTransaction("fixture")
        try:
            address = (
                program.getAddressFactory().getDefaultAddressSpace().getAddress(0x1000)
            )
            api = FlatProgramAPI(program)
            code = bytes.fromhex("8b442404c1e002c3")
            program.getMemory().createInitializedBlock(
                "text",
                address,
                ByteArrayInputStream(code),
                len(code),
                TaskMonitor.DUMMY,
                False,
            )
            api.disassemble(address)
            function = api.createFunction(address, "Grow")
            function.setCallingConvention(convention)
            source = SourceType.USER_DEFINED if reviewed else SourceType.ANALYSIS
            function.replaceParameters(
                Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
                True,
                source,
                ParameterImpl(
                    "capacity",
                    PointerDataType(UnsignedIntegerDataType.dataType),
                    program,
                ),
            )
            function.setSignatureSource(source)
            apply_recompiled_scalar_parameters(
                program,
                [
                    (0x1000, name)
                    for name in ((symbol,) if isinstance(symbol, str) else symbol)
                ],
            )
            explicit = [p for p in function.getParameters() if not p.isAutoParameter()]
            assert str(explicit[0].getDataType().getName()) == expected
            assert str(function.getSignatureSource()) == (
                "IMPORTED" if expected == "ulong" else str(source)
            )
        finally:
            program.endTransaction(transaction, False)
    finally:
        program.release(consumer)
