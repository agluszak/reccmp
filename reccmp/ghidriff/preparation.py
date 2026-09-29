"""Binary-derived Ghidra preparation before decompiling paired functions."""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghidra.program.model.listing import Program


def apply_stack_probe_call_fixups(program: "Program", addresses: set[int]) -> None:
    """Apply Ghidra's compiler fixup to catalogued stack-probe functions.

    The probe changes ESP by the requested allocation size. Without its
    call fixup, Ghidra loses the caller's stack frame and parameters after
    a large local allocation, including parameters of calls in that frame.
    """
    fixups = program.getCompilerSpec().getPcodeInjectLibrary().getCallFixupNames()
    if "alloca_probe" not in fixups:
        return
    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    for address in addresses:
        function = functions.getFunctionAt(space.getAddress(address))
        if function is not None and function.getCallFixup() is None:
            function.setCallFixup("alloca_probe")


def apply_reviewed_cdecl_signatures(
    program: "Program", signatures: dict[int, int]
) -> None:
    """Constrain private retail analysis with independently reviewed signatures."""
    from ghidra.program.model.data import Undefined4DataType
    from ghidra.program.model.listing import ParameterImpl
    from ghidra.program.model.symbol import SourceType

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    for address, count in signatures.items():
        function = functions.getFunctionAt(space.getAddress(address))
        if function is None:
            continue
        existing = function.getParameterCount()
        # An inferred extra argument may be an auto-parameter. The reviewed
        # arity alone does not establish which private parameter to discard.
        if existing >= count:
            continue
        function.setCallingConvention("__cdecl")
        if function.getParameterCount() > count:
            continue
        for index in range(function.getParameterCount(), count):
            function.addParameter(
                ParameterImpl(f"param{index}", Undefined4DataType.dataType, program),
                SourceType.USER_DEFINED,
            )
        function.setSignatureSource(SourceType.USER_DEFINED)


def apply_reviewed_bool_returns(program: "Program", addresses: set[int]) -> None:
    """Use a return type independently established by retail and PDB."""
    from ghidra.program.model.data import BooleanDataType
    from ghidra.program.model.symbol import SourceType

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    for address in addresses:
        function = functions.getFunctionAt(space.getAddress(address))
        if function is not None and function.getReturnType().getName() != "bool":
            function.setReturnType(BooleanDataType.dataType, SourceType.USER_DEFINED)


def correct_recompiled_signatures(
    program: "Program", symbols: dict[int, tuple[str, int, str]]
) -> None:
    """Correct inferred ABI when retail and the recomp PDB agree.

    Parameter ID can mistake live registers for arguments, or miss a trailing
    argument unused by the callee. Retail must independently infer the same
    count as the recompilation's decorated symbol. Neither side imports the
    other image's parameter types.
    """
    from ghidra.app.util.demangler import DemangledFunction, DemanglerUtil
    from ghidra.program.model.data import Undefined4DataType
    from ghidra.program.model.listing import ParameterImpl
    from ghidra.program.model.symbol import SourceType

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    for address, (symbol, retail_count, retail_convention) in symbols.items():
        function = functions.getFunctionAt(space.getAddress(address))
        if function is None or function.getParameterCount() == retail_count:
            continue
        if function.getSignatureSource() not in (
            SourceType.DEFAULT,
            SourceType.ANALYSIS,
        ):
            continue
        demangled = DemanglerUtil.demangle(symbol)
        if not isinstance(demangled, DemangledFunction):
            continue
        parameters = list(demangled.getParameters())
        if len(parameters) == 1 and str(parameters[0]) == "void":
            parameters = []
        if (
            demangled.getCallingConvention() == "__thiscall"
            and retail_convention == "__thiscall"
            and retail_count == len(parameters) + 1
            and function.getParameterCount() == len(parameters)
        ):
            function.setCallingConvention("__thiscall")
            function.setSignatureSource(SourceType.USER_DEFINED)
            continue
        if (
            demangled.getCallingConvention() != "__cdecl"
            or any(parameter.getType().isVarArgs() for parameter in parameters)
            or len(parameters) != retail_count
        ):
            continue
        missing_types = [
            parameter.getType().getDataType(program.getDataTypeManager())
            for parameter in parameters[function.getParameterCount() :]
        ]
        if any(
            data_type is None or not 0 < data_type.getLength() <= 4
            for data_type in missing_types
        ):
            continue
        function.setCallingConvention("__cdecl")
        while function.getParameterCount() > retail_count:
            function.removeParameter(function.getParameterCount() - 1)
        for index in range(function.getParameterCount(), retail_count):
            function.addParameter(
                ParameterImpl(
                    f"param{index}",
                    Undefined4DataType.dataType,
                    program,
                ),
                SourceType.USER_DEFINED,
            )
        function.setSignatureSource(SourceType.USER_DEFINED)


def correct_import_purges(program: "Program") -> None:
    """Give cdecl and variadic imports a stack purge of zero.

    With the imported library beside the binary, Ghidra may count arguments
    pushed on a path that never returns as the import's own stack purge. That
    corrupts the caller's stack model and its inferred parameters. The
    calling convention comes from the import's mangled name.
    """
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
            demangled.getCallingConvention() == CompilerSpec.CALLING_CONVENTION_cdecl
            or any(
                parameter.getType().isVarArgs()
                for parameter in demangled.getParameters()
            )
        ):
            function.setStackPurgeSize(0)


def infer_requested_callee_parameters(
    program: "Program", requested: list[int], known_callees: set[int]
) -> None:
    """Infer catalogued direct callees' ABI from this image's own code.

    Ghidra can assign an outer call's early-pushed argument to an untyped
    inner callee. Parameter ID resolves the callee's arity from its body.
    Include one-sided template emissions: they can be called by many paired
    functions even though the other image inlined the helper.
    """
    from ghidra.app.cmd.function import DecompilerParameterIdCmd
    from ghidra.program.model.address import AddressSet
    from ghidra.program.model.symbol import SourceType
    from ghidra.util.task import TaskMonitor

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    entries = AddressSet()
    for addr in requested:
        root = functions.getFunctionAt(space.getAddress(addr))
        if root is None:
            continue
        for callee in root.getCalledFunctions(TaskMonitor.DUMMY):
            if (
                callee.getEntryPoint().getOffset() in known_callees
                and callee.getSignatureSource() == SourceType.DEFAULT
            ):
                entries.add(callee.getEntryPoint())
    if entries.isEmpty():
        return
    command = DecompilerParameterIdCmd(
        "reccmp catalogued callees", entries, SourceType.ANALYSIS, False, False, 15
    )
    if not command.applyTo(program, TaskMonitor.DUMMY):
        raise RuntimeError(f"Ghidra Parameter ID failed: {command.getStatusMsg()}")


def recover_requested_switches(program: "Program", requested: list[int]) -> None:
    """Run Ghidra's switch command on the functions this run compares."""
    from ghidra.app.cmd.function import DecompilerSwitchAnalysisCmd
    from ghidra.app.decompiler import DecompInterface
    from ghidra.app.plugin.core.analysis import SwitchAnalysisDecompileConfigurer
    from ghidra.util.task import TaskMonitor

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    decompiler = DecompInterface()
    try:
        SwitchAnalysisDecompileConfigurer(program).configure(decompiler)
        if not decompiler.openProgram(program):
            raise RuntimeError("Could not open program for switch analysis")
        for addr in requested:
            function = functions.getFunctionAt(space.getAddress(addr))
            if function is None:
                continue
            results = decompiler.decompileFunction(function, 60, TaskMonitor.DUMMY)
            if not results.decompileCompleted():
                raise RuntimeError(
                    f"Switch analysis failed at {function.getEntryPoint()}"
                )
            if not DecompilerSwitchAnalysisCmd(results).applyTo(
                program, TaskMonitor.DUMMY
            ):
                raise RuntimeError(
                    f"Switch recovery failed at {function.getEntryPoint()}"
                )
    finally:
        decompiler.dispose()
