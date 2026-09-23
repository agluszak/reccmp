"""Return kind and callee calling conventions for the verifier, from PDB
types or decorated names."""

from reccmp.compare.asm.replacement import canonical_callee_name
from reccmp.compare.asm.verifier import CallAbi, FunctionMetadata
from reccmp.compare.comparator_state import ComparatorState
from reccmp.compare.db import ReccmpMatch
from reccmp.cvdump.analysis import CvdumpNode
from reccmp.cvdump.cvinfo import CvdumpTypeKey, CvdumpTypeMap
from reccmp.cvdump.demangler import parse_function_signature
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.cvdump.types import CvdumpKeyError
from reccmp.types import EntityType, ImageId

# Register-argument usage by PDB calling convention. cdecl and stdcall
# take all arguments on the stack; thiscall reads the receiver from ecx;
# fastcall reads its first two register-sized arguments from ecx and edx.
_CALL_ABI_BY_CONVENTION = {
    "C Near": CallAbi(uses_ecx=False, uses_edx=False),
    "STD Near": CallAbi(uses_ecx=False, uses_edx=False),
    "ThisCall": CallAbi(uses_ecx=True, uses_edx=False),
    "Fast Near": CallAbi(uses_ecx=True, uses_edx=True),
    # Conventions as recovered from decorated names.
    "cdecl": CallAbi(uses_ecx=False, uses_edx=False),
    "stdcall": CallAbi(uses_ecx=False, uses_edx=False),
    "thiscall": CallAbi(uses_ecx=True, uses_edx=False),
    "fastcall": CallAbi(uses_ecx=True, uses_edx=True),
}

_RETURN_KIND_BY_SIZE = {1: "i8", 2: "i16", 4: "i32", 8: "i64"}


class FunctionMetadataMixin(ComparatorState):
    """Part of FunctionComparator; relies on its attributes."""

    def _return_kind_of_type(self, type_key: CvdumpTypeKey) -> str:
        # pylint: disable=too-many-return-statements
        """Reduce a PDB return type to the register footprint of the
        returned value. Unknown or by-value aggregate returns stay
        "unknown", which makes the verifier compare eax exactly."""
        for _ in range(8):
            if type_key.is_scalar():
                scalar = CvdumpTypeMap.get(type_key)
                if scalar is None:
                    return "unknown"
                if scalar.name == "T_VOID":
                    return "void"
                if scalar.pointer is not None:
                    return "i32"
                if scalar.name.startswith("T_REAL"):
                    return "float"
                return _RETURN_KIND_BY_SIZE.get(scalar.size, "unknown")
            try:
                obj = self.types.from_key(type_key)
            except CvdumpKeyError:
                return "unknown"
            leaf = obj.get("type")
            if leaf == "LF_POINTER":
                return "i32"
            if leaf == "LF_ENUM" and "underlying_type" in obj:
                type_key = obj["underlying_type"]
                continue
            if leaf == "LF_MODIFIER" and "modifies" in obj:
                type_key = obj["modifies"]
                continue
            return "unknown"
        return "unknown"

    def _call_abi_map(self) -> dict[str, CallAbi | None]:
        """Map from a sanitized call-target name (as the diff displays it)
        to the callee's register-argument usage. A name shared by several
        functions with conflicting conventions resolves to None (unknown)."""
        if self._call_abi_cache is not None:
            return self._call_abi_cache
        result: dict[str, CallAbi | None] = {}
        for entity in self.db.get_all():
            if entity.entity_type != EntityType.FUNCTION:
                continue
            recomp_addr = entity.recomp_addr
            if recomp_addr is None:
                continue
            node = self.func_nodes.get(recomp_addr)
            if node is None:
                continue
            name = canonical_callee_name(
                self.db,
                ImageId.RECOMP,
                entity,
                self.equivalence_groups,
            )
            if name is None:
                continue
            _, abi = self._signature_of_node(node)
            if name in result and result[name] != abi:
                result[name] = None
            else:
                result[name] = abi
        self._call_abi_cache = result
        return result

    def _signature_of_node(self, node: CvdumpNode) -> tuple[str, CallAbi | None]:
        """(return kind, register-argument ABI) for one function node.
        Prefers the PDB TYPES record; falls back to the decorated name,
        which encodes the convention and return type even when the PDB
        (like Imperialism's) carries no type records at all."""
        return_kind = "unknown"
        abi = None
        if node.symbol_entry is not None:
            try:
                func = self.types.from_key(node.symbol_entry.func_type)
                abi = _CALL_ABI_BY_CONVENTION.get(func.get("call_type", ""))
                if "return_type" in func:
                    return_kind = self._return_kind_of_type(func["return_type"])
            except CvdumpKeyError:
                pass
        if (return_kind == "unknown" or abi is None) and node.decorated_name:
            mangled = parse_function_signature(node.decorated_name)
            if return_kind == "unknown":
                return_kind = mangled.return_kind
            if abi is None and mangled.convention is not None:
                abi = _CALL_ABI_BY_CONVENTION.get(mangled.convention)
        return (return_kind, abi)

    def _function_metadata(self, match: ReccmpMatch) -> FunctionMetadata | None:
        if not self.func_nodes:
            return None
        return_kind = "unknown"
        node = self.func_nodes.get(match.recomp_addr)
        if node is not None:
            return_kind, _ = self._signature_of_node(node)
        abi_map = self._call_abi_map()
        return FunctionMetadata(
            return_kind=return_kind,
            call_abi=abi_map.get,
        )

    def _fn_symbol_entry(self, match: ReccmpMatch | None) -> SymbolsEntry | None:
        if match is None:
            return None
        node = self.func_nodes.get(match.recomp_addr)
        if node is None:
            return None
        return node.symbol_entry
