"""A synchronized source prototype is not an independent retail observation."""

import contextlib
import hashlib
import sys
from types import SimpleNamespace

from reccmp.ghidra.signature_provenance import record_signature_origin
from reccmp.tools.compare import _reviewed_signatures
from reccmp.types import EntityType


def test_projected_wrong_signature_cannot_confirm_itself(monkeypatch, tmp_path):
    binary = tmp_path / "retail.exe"
    binary.write_bytes(b"retail")
    project_file = tmp_path / "reviewed.gpr"
    project_file.touch()
    stamps = {}
    properties = SimpleNamespace(
        getStringPropertyMap=lambda _: SimpleNamespace(
            add=lambda address, value: stamps.__setitem__(address, value),
            getString=lambda address: stamps.get(address),
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
    program = SimpleNamespace(
        getUsrPropertyManager=lambda: properties,
        getExecutableMD5=lambda: hashlib.md5(binary.read_bytes()).hexdigest(),
        getFunctionManager=lambda: SimpleNamespace(getFunctionAt=lambda _: function),
        getAddressFactory=lambda: SimpleNamespace(
            getDefaultAddressSpace=lambda: SimpleNamespace(getAddress=lambda a: a)
        ),
    )

    class DemangledFunction:
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
    args = SimpleNamespace(orig_ghidra_project=project_file, orig_ghidra_program="retail")
    manifest = SimpleNamespace(
        orig=SimpleNamespace(path=binary),
        objects=[
            SimpleNamespace(
                entity_type=EntityType.FUNCTION, recomp_symbol="wrong", orig_addr=0x401000
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
