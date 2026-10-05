"""ABI preparation guards without starting Ghidra."""

import sys
from typing import TYPE_CHECKING, cast

# These fakes implement Ghidra's Java method names.
# pylint: disable=invalid-name

from types import SimpleNamespace as NS

import pytest

from reccmp.ghidriff.preparation import correct_recompiled_signatures

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
