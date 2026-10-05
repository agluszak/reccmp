"""Binary-derived Ghidra preparation before decompiling paired functions."""

# pylint: disable=import-outside-toplevel,import-error
# Ghidra's Java packages exist only after the engine starts the JVM.

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ghidriff import DecompileResult
    from ghidra.app.decompiler import DecompileResults
    from ghidra.program.model.address import Address
    from ghidra.program.model.listing import Function, Program


_CORRECTIONS = "reccmp.preparation-corrections"


@dataclass(frozen=True)
class PreparationCorrection:
    address: int
    evidence: str
    before: str
    after: str


def _signature(function) -> str:
    # Names are presentation: later catalog labelling must not erase diagnostics.
    return json.dumps(
        [
            str(function.getCallingConventionName()),
            str(function.getReturnType().getName()),
            [
                str(parameter.getDataType().getName())
                for parameter in function.getParameters()
            ],
        ]
    )


def _record_correction(program, function, before: str, evidence: str) -> None:
    after = _signature(function)
    if before == after:
        return
    properties = program.getUsrPropertyManager()
    records = properties.getStringPropertyMap(_CORRECTIONS)
    if records is None:
        records = properties.createStringPropertyMap(_CORRECTIONS)
    address = function.getEntryPoint()
    existing = records.getString(address)
    history = json.loads(str(existing)) if existing is not None else []
    history.append({"evidence": evidence, "before": before, "after": after})
    records.add(address, json.dumps(history))


def preparation_corrections(program) -> tuple[PreparationCorrection, ...]:
    """Read persisted preparation diagnostics, including after a cache restore."""
    records = program.getUsrPropertyManager().getStringPropertyMap(_CORRECTIONS)
    if records is None:
        return ()
    corrections: list[PreparationCorrection] = []
    addresses = records.getPropertyIterator()
    while addresses.hasNext():
        address = addresses.next()
        function = program.getFunctionManager().getFunctionAt(address)
        history = json.loads(str(records.getString(address)))
        # A later signature change makes this record stale, not new evidence.
        if function is None or history[-1]["after"] != _signature(function):
            continue
        corrections.extend(
            PreparationCorrection(int(address.getOffset()), **record)
            for record in history
        )
    return tuple(corrections)


def correct_legacy_crt_signatures(program: "Program") -> None:
    """MSVCRT's swprintf predates the size-taking ISO C signature.

    The PE import identifies this ABI; UCRT and non-Windows swprintf are
    deliberately excluded. On x86 it is cdecl (buffer, format, ...), with
    two UTF-16 pointer parameters. A size-taking prototype consumes caller
    stack state as a third argument, including an inline call's return address.
    """
    from ghidra.program.model.data import (
        IntegerDataType,
        PointerDataType,
        WideChar16DataType,
    )
    from ghidra.program.model.listing import Function, ParameterImpl
    from ghidra.program.model.symbol import SourceType
    from .imports import import_locations

    if (
        program.getDefaultPointerSize() != 4
        or str(program.getLanguage().getProcessor()) != "x86"
    ):
        return
    for (library, name), location in import_locations(program).items():
        if library != "MSVCRT.DLL" or name != "swprintf":
            continue
        function = location.getFunction() or location.createFunction()
        before = _signature(function)
        wide_pointer = PointerDataType(WideChar16DataType.dataType)
        function.setCallingConvention("__cdecl")
        function.setReturnType(IntegerDataType.dataType, SourceType.IMPORTED)
        function.replaceParameters(
            Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
            True,
            SourceType.IMPORTED,
            ParameterImpl("buffer", wide_pointer, program),
            ParameterImpl("format", wide_pointer, program),
        )
        function.setVarArgs(True)
        function.setStackPurgeSize(0)
        _record_correction(
            program,
            function,
            before,
            "MSVCRT.DLL legacy swprintf ABI: buffer, format, ...",
        )


def decompile_fresh(
    program: "Program",
    function: "Function",
    timeout: int,
    read_results: Callable[["DecompileResults"], "DecompileResult"],
    debug_path: Path | None = None,
) -> "DecompileResult | None":
    """Retry with a native process that has no cached earlier decompilations."""
    from ghidra.app.decompiler import DecompInterface, DecompileOptions
    from ghidra.util.task import TaskMonitor

    decompiler = DecompInterface()
    try:
        options = DecompileOptions()
        options.grabFromProgram(program)
        options.setMaxPayloadMBytes(100)
        decompiler.setOptions(options)
        if not decompiler.openProgram(program):
            return None
        if debug_path is not None:
            import jpype

            debug_path.parent.mkdir(parents=True, exist_ok=True)
            decompiler.enableDebug(jpype.JClass("java.io.File")(str(debug_path)))
        return read_results(
            decompiler.decompileFunction(function, timeout, TaskMonitor.DUMMY)
        )
    finally:
        decompiler.dispose()


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
        before = _signature(function)
        function.setCallingConvention("__cdecl")
        if function.getParameterCount() > count:
            continue
        for index in range(function.getParameterCount(), count):
            function.addParameter(
                ParameterImpl(f"param{index}", Undefined4DataType.dataType, program),
                SourceType.USER_DEFINED,
            )
        function.setSignatureSource(SourceType.USER_DEFINED)
        _record_correction(program, function, before, "reviewed-retail-cdecl")


def apply_reviewed_scalar_returns(
    program: "Program", returns: dict[int, str], *, recomp: bool = False
) -> None:
    """Restore lost returns without retyping already inferred integer calls."""
    from ghidra.program.model.data import (
        BooleanDataType,
        Undefined1DataType,
        Undefined4DataType,
        UnsignedIntegerDataType,
    )
    from ghidra.program.model.symbol import SourceType

    data_types = {
        "bool": BooleanDataType.dataType,
        "undefined1": Undefined1DataType.dataType,
        "undefined4": Undefined4DataType.dataType,
        "uint": UnsignedIntegerDataType.dataType,
    }
    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    for address, return_name in returns.items():
        function = functions.getFunctionAt(space.getAddress(address))
        if function is None:
            continue
        current = function.getReturnType().getName()
        if return_name == "uint" and (not recomp or current != "void"):
            continue
        if return_name == "undefined4" and (
            recomp
            or function.getReturnType().getLength()
            >= Undefined4DataType.dataType.getLength()
        ):
            continue
        if return_name == "undefined1" and (
            recomp or function.getReturnType().getLength() not in (0, 2, 4)
        ):
            continue
        if current != return_name:
            before = _signature(function)
            function.setReturnType(data_types[return_name], SourceType.USER_DEFINED)
            _record_correction(program, function, before, "reviewed-retail-return")


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
        if function is None or (
            function.getParameterCount() == retail_count
            and function.getCallingConventionName() == retail_convention
        ):
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
        before = _signature(function)
        if (
            demangled.getCallingConvention() == "__thiscall"
            and retail_convention == "__thiscall"
            and retail_count == len(parameters) + 1
            and function.getParameterCount() == len(parameters)
        ):
            function.setCallingConvention("__thiscall")
            function.setSignatureSource(SourceType.USER_DEFINED)
            _record_correction(
                program, function, before, "retail-analysis-and-recomp-symbol"
            )
            continue
        if (
            demangled.getCallingConvention() != "__cdecl"
            or retail_convention != "__cdecl"
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
        _record_correction(
            program, function, before, "retail-analysis-and-recomp-symbol"
        )


def apply_recompiled_scalar_parameters(
    program: "Program", symbols: list[tuple[int, str]]
) -> None:
    """Constrain inferred parameters using this binary's own decorated symbols.

    Reused argument slots can make Parameter ID infer a count as a pointer.
    Only explicit primitive parameters with matching convention, arity and
    width are imported. Retail analysis and aggregate/template types stay separate.
    """
    from ghidra.app.util.demangler import DemangledFunction, DemanglerUtil
    from ghidra.program.model.data import (
        AbstractIntegerDataType,
        AbstractFloatDataType,
        BooleanDataType,
    )
    from ghidra.program.model.symbol import SourceType

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    counts = Counter(address for address, _ in symbols)
    for address, symbol in symbols:
        if counts[address] != 1:
            continue
        function = functions.getFunctionAt(space.getAddress(address))
        if function is None or function.getSignatureSource() not in (
            SourceType.DEFAULT,
            SourceType.ANALYSIS,
        ):
            continue
        demangled = DemanglerUtil.demangle(symbol)
        if not isinstance(demangled, DemangledFunction) or (
            demangled.getCallingConvention() != function.getCallingConventionName()
        ):
            continue
        expected = list(demangled.getParameters())
        actual = [p for p in function.getParameters() if not p.isAutoParameter()]
        if len(expected) != len(actual):
            continue
        before = _signature(function)
        changed = False
        for parameter, original in zip(actual, expected):
            declared = original.getType()
            if any(
                (
                    declared.isPointer(),
                    declared.isReference(),
                    declared.isArray(),
                    declared.isClass(),
                    declared.isStruct(),
                    declared.isUnion(),
                    declared.isEnum(),
                )
            ):
                continue
            data_type = declared.getDataType(program.getDataTypeManager())
            if (
                not isinstance(
                    data_type,
                    (AbstractIntegerDataType, AbstractFloatDataType, BooleanDataType),
                )
                or data_type.getLength() != parameter.getDataType().getLength()
            ):
                continue
            if not parameter.getDataType().isEquivalent(data_type):
                parameter.setDataType(data_type, SourceType.IMPORTED)
                changed = True
        if changed:
            function.setSignatureSource(SourceType.IMPORTED)
            _record_correction(
                program, function, before, "recomp-decorated-scalar-parameter"
            )


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
        callees = set(root.getCalledFunctions(TaskMonitor.DUMMY))
        # A direct tail jump continues another function's body. Ghidra's
        # getCalledFunctions only includes call-flow references, so otherwise
        # a tail-only callee keeps its default signature in focused selections.
        for instruction in program.getListing().getInstructions(root.getBody(), True):
            flow = instruction.getFlowType()
            if not flow.isJump() or not flow.isUnConditional() or flow.isComputed():
                continue
            for target in instruction.getFlows():
                if root.getBody().contains(target):
                    continue
                callee = functions.getFunctionAt(target)
                if callee is not None:
                    callees.add(callee)
        for callee in callees:
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


def recover_requested_switches(
    program: "Program", requested: list[int], timeout: int
) -> dict[int, str]:
    """Run Ghidra's switch command on the functions this run compares."""
    from ghidra.app.cmd.function import DecompilerSwitchAnalysisCmd
    from ghidra.app.decompiler import DecompInterface
    from ghidra.app.plugin.core.analysis import SwitchAnalysisDecompileConfigurer
    from ghidra.util.task import TaskMonitor

    functions = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()
    decompiler = DecompInterface()
    failures = {}
    try:
        SwitchAnalysisDecompileConfigurer(program).configure(decompiler)
        if not decompiler.openProgram(program):
            raise RuntimeError("Could not open program for switch analysis")
        for addr in requested:
            function = functions.getFunctionAt(space.getAddress(addr))
            if function is None:
                continue
            results = decompiler.decompileFunction(function, timeout, TaskMonitor.DUMMY)
            if not results.decompileCompleted():
                failures[addr] = (
                    f"Switch analysis failed at {function.getEntryPoint()}: "
                    f"{results.getErrorMessage()}"
                )
                decompiler.resetDecompiler()
                continue
            if not DecompilerSwitchAnalysisCmd(results).applyTo(
                program, TaskMonitor.DUMMY
            ):
                failures[addr] = f"Switch recovery failed at {function.getEntryPoint()}"
    finally:
        decompiler.dispose()
    return failures


def clear_data(program: "Program", address: "Address") -> None:
    """Remove data Ghidra's analysis defined over a known function entry,
    such as a string it guessed in the instruction bytes; it would keep
    the entry from being disassembled."""
    listing = program.getListing()
    data = listing.getDataContaining(address)
    if data is not None and data.isDefined():
        listing.clearCodeUnits(data.getMinAddress(), data.getMaxAddress(), False)


def split_absorbed_piece(function: Any, address: "Address") -> bool:
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
