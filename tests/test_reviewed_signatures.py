"""A synchronized source prototype is not an independent retail observation."""

import contextlib
import hashlib
import json
import sys
from types import SimpleNamespace
from typing import Any

from reccmp.ghidra.signature_provenance import record_signature_origin
from reccmp.tools.compare import _reviewed_signatures
from reccmp.ghidriff.preparation import apply_reviewed_scalar_returns
from reccmp.types import EntityType


def test_projected_wrong_signature_cannot_confirm_itself(monkeypatch, tmp_path):
    binary = tmp_path / "retail.exe"
    binary.write_bytes(b"retail")
    project_file = tmp_path / "reviewed.gpr"
    project_file.touch()
    stamps: dict[int, str] = {}
    properties = SimpleNamespace(
        getStringPropertyMap=lambda _: SimpleNamespace(
            add=stamps.__setitem__,
            getString=stamps.get,
        )
    )
    # Retail is actually void(int); the recovered source incorrectly says bool().
    # Both source sync (IMPORTED) and the old PDB importer (USER_DEFINED) can
    # store that same wrong prototype. Decoration agreement is circular.
    function = SimpleNamespace(
        getEntryPoint=lambda: 0x401000,
        getSignature=lambda: "bool wrong()",
        getSignatureSource=lambda: "USER_DEFINED",
        getReturnType=lambda: SimpleNamespace(getName=lambda: "bool"),
        getCallingConventionName=lambda: "__cdecl",
        getParameterCount=lambda: 0,
        hasVarArgs=lambda: False,
    )
    program: Any = SimpleNamespace(
        getUsrPropertyManager=lambda: properties,
        getExecutableMD5=lambda: hashlib.md5(binary.read_bytes()).hexdigest(),
        getFunctionManager=lambda: SimpleNamespace(getFunctionAt=lambda _: function),
        getAddressFactory=lambda: SimpleNamespace(
            getDefaultAddressSpace=lambda: SimpleNamespace(getAddress=lambda a: a)
        ),
    )

    class DemangledFunction:  # pylint: disable=invalid-name
        def getReturnType(self):
            return "bool"

        def getCallingConvention(self):
            return "__cdecl"

        def getParameters(self):
            return []

    monkeypatch.setitem(
        sys.modules,
        "pyghidra",
        SimpleNamespace(
            open_project=lambda *a, **kw: SimpleNamespace(close=lambda: None),
            program_context=lambda *a: contextlib.nullcontext(program),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.app.util.demangler",
        SimpleNamespace(
            DemangledFunction=DemangledFunction,
            DemanglerUtil=SimpleNamespace(demangle=lambda _: DemangledFunction()),
        ),
    )
    args: Any = SimpleNamespace(
        orig_ghidra_project=project_file, orig_ghidra_program="retail"
    )
    manifest: Any = SimpleNamespace(
        orig=SimpleNamespace(path=binary),
        objects=[
            SimpleNamespace(
                entity_type=EntityType.FUNCTION,
                recomp_symbol="wrong",
                orig_addr=0x401000,
            )
        ],
    )
    # Legacy untagged USER_DEFINED prototypes are unknown, too.
    assert _reviewed_signatures(args, manifest) == ({}, {})
    for origin in ("source-projection", "pdb-projection", "analysis-inference"):
        record_signature_origin(program, function, origin)
        assert _reviewed_signatures(args, manifest) == ({}, {})
    # Positive control: only an explicit independent binary review admits facts.
    record_signature_origin(program, function, "retail-reviewed")
    assert _reviewed_signatures(args, manifest) == ({0x401000: 0}, {0x401000: "bool"})
    function.getSignature = lambda: "bool wrong(int)"
    assert _reviewed_signatures(args, manifest) == ({}, {})

    # A native return review must not certify projected/unknown parameters.
    function.getSignature = lambda: "undefined4 wrong()"
    function.getReturnType = lambda: SimpleNamespace(getName=lambda: "undefined4")
    monkeypatch.setattr(DemangledFunction, "getReturnType", lambda self: "int")
    for origin in ("source-projection", "pdb-projection", "analysis-inference"):
        record_signature_origin(program, function, origin)
        assert _reviewed_signatures(args, manifest) == ({}, {})
    record_signature_origin(program, function, "retail-return-reviewed")
    assert _reviewed_signatures(args, manifest) == ({}, {0x401000: "undefined4"})
    # Sync invalidates even an unchanged prototype's earlier retail review.
    record_signature_origin(program, function, "source-projection")
    assert _reviewed_signatures(args, manifest) == ({}, {})
    record_signature_origin(program, function, "retail-return-reviewed")
    function.getSignature = lambda: "undefined4 wrong(int)"
    assert _reviewed_signatures(args, manifest) == ({}, {})

    function.getSignature = lambda: "undefined1 wrong()"
    function.getReturnType = lambda: SimpleNamespace(getName=lambda: "undefined1")
    monkeypatch.setattr(DemangledFunction, "getReturnType", lambda self: "bool")
    record_signature_origin(program, function, "retail-return-reviewed")
    assert _reviewed_signatures(args, manifest) == ({}, {0x401000: "undefined1"})
    record_signature_origin(program, function, "pdb-projection")
    assert _reviewed_signatures(args, manifest) == ({}, {})


def test_reviewed_integer_width_repairs_only_narrow_retail(monkeypatch):
    int_type = SimpleNamespace(getName=lambda: "undefined4", getLength=lambda: 4)
    byte_type = SimpleNamespace(getName=lambda: "undefined1", getLength=lambda: 1)
    monkeypatch.setitem(
        sys.modules,
        "ghidra.program.model.data",
        SimpleNamespace(
            BooleanDataType=SimpleNamespace(dataType=None),
            Undefined1DataType=SimpleNamespace(dataType=byte_type),
            Undefined4DataType=SimpleNamespace(dataType=int_type),
            UnsignedIntegerDataType=SimpleNamespace(dataType=None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ghidra.program.model.symbol",
        SimpleNamespace(SourceType=SimpleNamespace(USER_DEFINED="USER_DEFINED")),
    )
    changed = []
    functions = {}
    return_types = {}
    records: dict[int, str] = {}

    def set_return_type(address, dtype):
        return_types[address] = dtype
        changed.append((address, dtype))

    for address, width in ((1, 1), (2, 4), (3, 4), (4, 1)):
        dtype = SimpleNamespace(
            getName=lambda: "undefined", getLength=lambda w=width: w
        )
        return_types[address] = dtype
        functions[address] = SimpleNamespace(
            getReturnType=lambda a=address: return_types[a],
            setReturnType=lambda dt, source, a=address: set_return_type(a, dt),
            getCallingConventionName=lambda: "__cdecl",
            getParameters=lambda: [],
            getEntryPoint=lambda a=address: a,
        )
    program: Any = SimpleNamespace(
        getUsrPropertyManager=lambda: SimpleNamespace(
            getStringPropertyMap=lambda _: SimpleNamespace(
                add=records.__setitem__,
                getString=records.get,
            )
        ),
        getFunctionManager=lambda: SimpleNamespace(getFunctionAt=functions.get),
        getAddressFactory=lambda: SimpleNamespace(
            getDefaultAddressSpace=lambda: SimpleNamespace(getAddress=lambda a: a)
        ),
    )
    returns = {1: "undefined4", 2: "undefined4", 3: "undefined1", 4: "undefined1"}
    apply_reviewed_scalar_returns(program, returns, recomp=True)
    assert not changed
    apply_reviewed_scalar_returns(program, returns)
    assert changed == [(1, int_type), (3, byte_type)]
    assert records.keys() == {1, 3}
    assert all(
        json.loads(record)[0]["before"] != json.loads(record)[0]["after"]
        for record in records.values()
    )
