"""Call facts (return kind, register arguments, stack cleanup) of the
recompiled functions, from PDB types first and decorated names second."""

from dataclasses import replace

from reccmp.compare.asm.replacement import canonical_callee_name
from reccmp.compare.asm.verifier import FunctionMetadata
from reccmp.compare.call_facts import CallFacts, convention_facts, mangled_facts
from reccmp.compare.comparator_state import ComparatorState
from reccmp.compare.db import ReccmpMatch
from reccmp.cvdump.analysis import CvdumpNode
from reccmp.cvdump.cvinfo import CvdumpTypeKey, CvdumpTypeMap
from reccmp.cvdump.symbols import SymbolsEntry
from reccmp.cvdump.types import CvdumpKeyError
from reccmp.types import EntityType, ImageId

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

    def _call_facts_map(self) -> dict[str, CallFacts | None]:
        """Map from a sanitized call-target name (as the diff displays it)
        to the callee's call facts. For a name shared by several functions,
        only the facts they agree on are known."""
        if self._call_facts_cache is not None:
            return self._call_facts_cache
        result: dict[str, CallFacts | None] = {}
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
            facts = self._call_facts_of_node(node)
            previous = result.get(name, facts)
            result[name] = None if previous is None else previous.agreed(facts)
        self._call_facts_cache = result
        return result

    def _call_facts_of_node(self, node: CvdumpNode) -> CallFacts:
        """Call facts of one function node. Each field prefers the PDB TYPES
        record and falls back to the decorated name, which encodes the
        convention, return type and parameters even when the PDB (like
        Imperialism's) carries no type records at all."""
        facts = CallFacts()
        if node.symbol_entry is not None:
            try:
                func = self.types.from_key(node.symbol_entry.func_type)
                facts = convention_facts(func.get("call_type"))
                if "return_type" in func:
                    facts = replace(
                        facts,
                        return_kind=self._return_kind_of_type(func["return_type"]),
                    )
            except CvdumpKeyError:
                pass
        if node.decorated_name:
            facts = facts.merged(mangled_facts(node.decorated_name))
        return facts

    def _call_facts_at(self, recomp_addr: int) -> CallFacts | None:
        node = self.func_nodes.get(recomp_addr)
        return self._call_facts_of_node(node) if node is not None else None

    def _function_metadata(self, match: ReccmpMatch) -> FunctionMetadata | None:
        if not self.func_nodes:
            return None
        facts = self._call_facts_at(match.recomp_addr)
        return FunctionMetadata(
            return_kind=facts.return_kind if facts is not None else "unknown",
            call_facts=self._call_facts_map().get,
        )

    def _fn_symbol_entry(self, match: ReccmpMatch | None) -> SymbolsEntry | None:
        if match is None:
            return None
        node = self.func_nodes.get(match.recomp_addr)
        if node is None:
            return None
        return node.symbol_entry
