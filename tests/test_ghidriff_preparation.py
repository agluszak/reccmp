"""ABI preparation guards without starting Ghidra."""

import sys
from typing import TYPE_CHECKING, cast

# These fakes implement Ghidra's Java method names.
# pylint: disable=invalid-name

from types import SimpleNamespace as NS

import pytest

from reccmp.ghidriff.preparation import (
    align_retail_receivers,
    correct_recompiled_signatures,
)
from reccmp.types import EntityType

if TYPE_CHECKING:
    from ghidra.program.model.listing import Program


@pytest.mark.parametrize(
    "retail_convention,symbol_convention,source,expected",
    [
        ("__cdecl", "__cdecl", "ANALYSIS", "__cdecl"),
        ("__stdcall", "__cdecl", "ANALYSIS", "__fastcall"),
        ("__cdecl", "__stdcall", "ANALYSIS", "__fastcall"),
        ("__cdecl", "__cdecl", "USER_DEFINED", "__fastcall"),
    ],
)
def test_equal_arity_does_not_skip_independently_confirmed_convention(
    monkeypatch, retail_convention, symbol_convention, source, expected
):
    class Demangled:
        def getParameters(self):
            return [NS(getType=lambda: NS(isVarArgs=lambda: False))]

        def getCallingConvention(self):
            return symbol_convention

    class Function:
        convention = "__fastcall"

        def getParameterCount(self):
            return 1

        def getReturnType(self):
            return NS(getName=lambda: "void")

        def getParameters(self):
            return [NS(getDataType=lambda: NS(getName=lambda: "int"))]

        def getCallingConventionName(self):
            return self.convention

        def getSignatureSource(self):
            return source

        def setCallingConvention(self, convention):
            self.convention = convention

        def setSignatureSource(self, value):
            pass

    function = Function()
    modules = {
        "ghidra.app.util.demangler": NS(
            DemangledFunction=Demangled,
            DemanglerUtil=NS(demangle=lambda _: Demangled()),
        ),
        "ghidra.program.model.data": NS(Undefined4DataType=NS(dataType=object())),
        "ghidra.program.model.listing": NS(ParameterImpl=object),
        "ghidra.program.model.symbol": NS(
            SourceType=NS(
                DEFAULT="DEFAULT", ANALYSIS="ANALYSIS", USER_DEFINED="USER_DEFINED"
            )
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "reccmp.ghidriff.preparation._record_correction", lambda *_: None
    )
    program = NS(
        getFunctionManager=lambda: NS(getFunctionAt=lambda _: function),
        getAddressFactory=lambda: NS(
            getDefaultAddressSpace=lambda: NS(getAddress=lambda x: x)
        ),
    )
    correct_recompiled_signatures(
        cast("Program", program), {0x1000: ("decorated", 1, retail_convention)}
    )
    assert function.convention == expected


class _Register:
    def __init__(self, name):
        self.name = name

    def getBaseRegister(self):
        return self

    def getName(self):
        return self.name


def _instruction(mnemonic, inputs=(), results=(), call=False):
    return NS(
        getMnemonicString=lambda: mnemonic,
        getInputObjects=lambda: list(inputs),
        getResultObjects=lambda: list(results),
        getFlowType=lambda: NS(isCall=lambda: call),
    )


def _receiver_fixture(monkeypatch, instructions, convention="__thiscall"):
    class Demangled:
        def getCallingConvention(self):
            return convention

    class Function:
        convention = "unknown"
        source = "DEFAULT"

        def isThunk(self):
            return False

        def isExternal(self):
            return False

        def getSignatureSource(self):
            return self.source

        def setSignatureSource(self, value):
            self.source = value

        def setCallingConvention(self, value):
            self.convention = value

        def getCallingConventionName(self):
            return self.convention

        def getReturnType(self):
            return NS(getName=lambda: "void")

        def getParameters(self):
            return []

        def getBody(self):
            return None

    function = Function()
    modules = {
        "ghidra.app.util.demangler": NS(
            DemangledFunction=Demangled,
            DemanglerUtil=NS(demangle=lambda _: Demangled()),
        ),
        "ghidra.program.model.lang": NS(Register=_Register),
        "ghidra.program.model.listing": NS(
            Function=NS(FunctionUpdateType=NS()), ParameterImpl=object
        ),
        "ghidra.program.model.symbol": NS(
            SourceType=NS(
                DEFAULT="DEFAULT", ANALYSIS="ANALYSIS", USER_DEFINED="USER_DEFINED"
            )
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "reccmp.ghidriff.preparation._record_correction", lambda *_: None
    )
    program = NS(
        getFunctionManager=lambda: NS(getFunctionAt=lambda _: function),
        getAddressFactory=lambda: NS(
            getDefaultAddressSpace=lambda: NS(getAddress=lambda x: x)
        ),
        getListing=lambda: NS(getInstructions=lambda _body, _forward: instructions),
    )
    obj = NS(
        entity_type=EntityType.FUNCTION, recomp_symbol="decorated", orig_addr=0x1000
    )
    align_retail_receivers(cast("Program", program), [obj])
    return function


def test_retail_receiver_read_aligns_thiscall(monkeypatch):
    ecx = _Register("ECX")
    function = _receiver_fixture(
        monkeypatch,
        [
            _instruction("PUSH", inputs=[ecx]),  # local slot, not an input
            _instruction("MOV", inputs=[ecx], results=[_Register("ESI")]),
        ],
    )
    assert function.convention == "__thiscall"
    assert function.source == "DEFAULT"


@pytest.mark.parametrize(
    "instructions",
    [
        [_instruction("MOV", results=[_Register("ECX")])],
        [
            _instruction("CALL", call=True),
            _instruction("MOV", inputs=[_Register("ECX")]),
        ],
        [_instruction("PUSH", inputs=[_Register("ECX")])],
    ],
)
def test_retail_without_receiver_input_keeps_its_convention(monkeypatch, instructions):
    assert _receiver_fixture(monkeypatch, instructions).convention == "unknown"


def test_non_member_recomp_symbol_keeps_retail_convention(monkeypatch):
    function = _receiver_fixture(
        monkeypatch, [_instruction("MOV", inputs=[_Register("ECX")])], "__cdecl"
    )
    assert function.convention == "unknown"
