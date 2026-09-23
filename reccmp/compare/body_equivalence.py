"""Body-equivalence proofs for bodies that are not annotated pairs: folded
COMDAT aliases, stale jmp islands and uniquely discoverable pairs."""

import re

from reccmp.compare.asm.const import JUMP_MNEMONICS
from reccmp.compare.asm.instgen import InstructGen, SectionType
from reccmp.compare.asm.ir import local_destination_keys
from reccmp.compare.comparator_state import ComparatorState
from reccmp.compare.db import ReccmpMatch
from reccmp.compare.equivalence import canonical_orig_addr
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
)
from reccmp.types import EntityType, ImageId

_ISLAND_PADDING = (0x90, 0xCC)  # nop / int3


def _code_instructions(
    raw: bytes, addr: int, is_32bit: bool
) -> list[tuple[int, int, str, str]] | None:
    """The body's raw instruction tuples, or None when the body carries
    non-code sections (jump-table data) that would break the positional
    pairing with the sanitized excerpt."""
    instructions: list[tuple[int, int, str, str]] = []
    for section in InstructGen(bytes(raw), addr, is_32bit).sections:
        if section.type != SectionType.CODE:
            return None
        instructions.extend(section.contents)
    return instructions


def _is_bare_jmp_island(raw: bytes) -> bool:
    """True when the function body is a lone `jmp rel32` followed only by
    padding: the shape an incremental link leaves at a moved/folded symbol's
    old address."""
    if len(raw) < 5 or raw[0] != 0xE9:
        return False
    return all(b in _ISLAND_PADDING for b in raw[5:])


class BodyEquivalenceMixin(ComparatorState):
    """Part of FunctionComparator; relies on its attributes."""

    def raw_pair_alias_equivalent(
        self,
        orig_addr: int,
        recomp_addr: int,
        size: int,
        *,
        _depth: int = 0,
        _active: set[tuple[int, int]] | None = None,
        _proved: dict[tuple[int, int], bool] | None = None,
    ) -> bool:
        """Recomputed, conservative compiler-alias equivalence for one
        (orig, recomp) body pair that is not an annotated match.

        MSVC folds identical COMDAT bodies (template deleting destructors
        above all), so one original address may serve slots that the rebuild
        gives distinct functions. This proves the equivalence from the bodies
        themselves instead of a declared alias ledger: both sides must
        disassemble to the same sanitized instruction sequence. Where one
        instruction pair diverges (or resolves only to a positional
        placeholder), it may still be accepted when it is a direct call or
        jump whose two targets are themselves alias-equivalent — folding is
        transitive: the linker folded a derived deleting destructor onto its
        base's only because their destructor chains folded too. The
        recursion is depth-bounded and cycle-guarded, and any operand that
        cannot be proven fails the check rather than matching positionally.
        A bare original ``jmp rel32`` island (a stale incremental-link
        forwarder) is followed to the shared body it lands on.
        """
        # pylint: disable=too-many-return-statements

        if size <= 0 or _depth > 3:
            return False
        pair = (orig_addr, recomp_addr)
        active = _active if _active is not None else set()
        proved = _proved if _proved is not None else {}
        cached = proved.get(pair)
        if cached is not None:
            return cached
        if pair in active:
            # Re-entering an unresolved pair is a cycle, not a completed proof.
            return False
        active.add(pair)
        try:
            result = self._raw_pair_alias_equivalent_body(
                orig_addr, recomp_addr, size, _depth, active, proved
            )
        finally:
            active.discard(pair)
        proved[pair] = result
        return result

    def _raw_pair_alias_equivalent_body(
        self,
        orig_addr: int,
        recomp_addr: int,
        size: int,
        depth: int,
        active: set[tuple[int, int]],
        proved: dict[tuple[int, int], bool],
    ) -> bool:
        # pylint: disable=too-many-positional-arguments,too-many-return-statements
        try:
            orig_raw = self.orig_bin.read(orig_addr, size)
            recomp_raw = self.recomp_bin.read(recomp_addr, size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            return False
        if len(orig_raw) != size or len(recomp_raw) != size:
            return False
        if _is_bare_jmp_island(orig_raw):
            island_target = (
                orig_addr + 5 + int.from_bytes(orig_raw[1:5], "little", signed=True)
            )
            if island_target == orig_addr:
                return False
            return self.raw_pair_alias_equivalent(
                island_target,
                recomp_addr,
                size,
                _depth=depth + 1,
                _active=active,
                _proved=proved,
            )
        # Symmetric recomp-side forwarder: an incremental link records the
        # PDB symbol on the `jmp rel32` thunk, so the entity's own body is the
        # island. Follow it to the real body and take that entity's size —
        # keeping the thunk's size would truncate the landing body mid-insn.
        if _is_bare_jmp_island(recomp_raw):
            island_target = (
                recomp_addr + 5 + int.from_bytes(recomp_raw[1:5], "little", signed=True)
            )
            target = self.db.get(ImageId.RECOMP, island_target)
            target_size = target.size(ImageId.RECOMP) if target is not None else None
            return self.raw_pair_alias_equivalent(
                orig_addr,
                island_target,
                target_size if target_size and target_size > 0 else size,
                _depth=depth + 1,
                _active=active,
                _proved=proved,
            )
        orig_asm = self.orig_sanitize.parse_asm(orig_raw, orig_addr)
        orig_topology = local_destination_keys(
            orig_asm,
            self.orig_sanitize.jump_tables,
            start_addr=orig_addr,
            extent=size,
        )
        recomp_asm = self.recomp_sanitize.parse_asm(recomp_raw, recomp_addr)
        recomp_topology = local_destination_keys(
            recomp_asm,
            self.recomp_sanitize.jump_tables,
            start_addr=recomp_addr,
            extent=size,
        )
        if not orig_asm or len(orig_asm) != len(recomp_asm):
            return False
        if orig_topology is None or recomp_topology is None:
            return False
        orig_insts = _code_instructions(orig_raw, orig_addr, self.is_32bit)
        recomp_insts = _code_instructions(recomp_raw, recomp_addr, self.is_32bit)
        if (
            orig_insts is None
            or recomp_insts is None
            or len(orig_insts) != len(orig_asm)
            or len(recomp_insts) != len(orig_asm)
        ):
            return False
        for index, (orig_row, recomp_row) in enumerate(zip(orig_asm, recomp_asm)):
            orig_line = orig_row.display
            recomp_line = recomp_row.display
            # Local branches must reach the same instruction id: equal
            # displacement text does not imply that when encodings differ.
            if (
                orig_line == recomp_line
                and "<OFFSET" not in orig_line
                and orig_topology[index] == recomp_topology[index]
            ):
                continue
            if not self._transfer_targets_alias_equivalent(
                orig_insts[index], recomp_insts[index], depth, active, proved
            ):
                return False
        return True

    def _transfer_targets_alias_equivalent(
        self,
        orig_inst: tuple[int, int, str, str],
        recomp_inst: tuple[int, int, str, str],
        depth: int,
        active: set[tuple[int, int]],
        proved: dict[tuple[int, int], bool],
    ) -> bool:
        """Whether a diverging instruction pair is a direct transfer whose
        two targets are themselves alias-equivalent bodies. The target size
        comes from the annotated recomp entity; an unknown target is not
        evidence."""

        _, _, orig_mnemonic, orig_op = orig_inst
        _, _, recomp_mnemonic, recomp_op = recomp_inst
        if orig_mnemonic != recomp_mnemonic:
            return False
        if orig_mnemonic != "call" and orig_mnemonic not in JUMP_MNEMONICS:
            return False
        try:
            orig_target = int(orig_op, 16)
            recomp_target = int(recomp_op, 16)
        except ValueError:
            return False  # indirect or composite operand
        entity = self.db.get(ImageId.RECOMP, recomp_target)
        target_size = entity.size(ImageId.RECOMP) if entity is not None else None
        if target_size is None or target_size <= 0:
            return False
        return self.raw_pair_alias_equivalent(
            orig_target,
            recomp_target,
            target_size,
            _depth=depth + 1,
            _active=active,
            _proved=proved,
        )

    def discover_unpaired_function_bodies(self) -> list[tuple[int, int]]:
        """Discover differently named function pairs to a conservative fixed point.

        Candidate edges require equal size, equal relocation-aware shape and a
        full ``raw_pair_alias_equivalent`` proof.  A pair is committed only
        when its edge has degree one at *both* endpoints.  All newly committed
        pairs become operand identities for the next pass.  Ambiguous graphs
        are left untouched instead of falling back to name/FIFO order.

        Once real pairs stop growing, remaining bodies may be recorded as
        side-local aliases of an existing canonical pair.  This is symmetric:
        original folded addresses and recomp duplicate emissions use the same
        canonical-original identity and never create fake one-to-one pairs.
        """
        discovered: list[tuple[int, int]] = []
        while True:
            orig_groups = self._unpaired_function_candidates(ImageId.ORIG)
            recomp_groups = self._unpaired_function_candidates(ImageId.RECOMP)
            edges: set[tuple[int, int]] = set()
            for key in orig_groups.keys() & recomp_groups.keys():
                size = key[0]
                for orig_addr in orig_groups[key]:
                    for recomp_addr in recomp_groups[key]:
                        if self.raw_pair_alias_equivalent(orig_addr, recomp_addr, size):
                            edges.add((orig_addr, recomp_addr))
            orig_degree: dict[int, int] = {}
            recomp_degree: dict[int, int] = {}
            for orig_addr, recomp_addr in edges:
                orig_degree[orig_addr] = orig_degree.get(orig_addr, 0) + 1
                recomp_degree[recomp_addr] = recomp_degree.get(recomp_addr, 0) + 1
            unique_pairs = {
                (orig_addr, recomp_addr)
                for orig_addr, recomp_addr in edges
                if orig_degree[orig_addr] == 1 and recomp_degree[recomp_addr] == 1
            }

            # A curated equivalence group supplies the otherwise-missing identity
            # for a many-originals-to-one-recomp folded body. Pair only its declared
            # canonical member; the remaining members become side-local aliases on
            # the next fixed-point iteration. Without that evidence, ambiguity stays.
            origs_by_recomp: dict[int, set[int]] = {}
            for orig_addr, recomp_addr in edges:
                origs_by_recomp.setdefault(recomp_addr, set()).add(orig_addr)
            canonical_pairs: set[tuple[int, int]] = set()
            for recomp_addr, orig_addrs in origs_by_recomp.items():
                if len(orig_addrs) < 2 or any(
                    orig_degree[orig_addr] != 1 for orig_addr in orig_addrs
                ):
                    continue
                canonical_addrs = {
                    canonical_orig_addr(self.equivalence_groups, orig_addr)
                    for orig_addr in orig_addrs
                }
                if len(canonical_addrs) != 1:
                    continue
                (canonical_addr,) = canonical_addrs
                if canonical_addr in orig_addrs:
                    canonical_pairs.add((canonical_addr, recomp_addr))

            pairs = sorted(unique_pairs | canonical_pairs)
            if pairs:
                self.db.bulk_match(pairs)
                discovered.extend(pairs)
                self.rebuild_lookups()
                continue

            # Alias identities can themselves unlock mutually unique callers,
            # so interleave symmetric classification with pair discovery.
            matches = list(self.db.get_functions())
            orig_added = self._classify_function_aliases(
                ImageId.ORIG, ImageId.RECOMP, matches
            )
            recomp_added = self._classify_function_aliases(
                ImageId.RECOMP, ImageId.ORIG, matches
            )
            if not orig_added and not recomp_added:
                break
            self.rebuild_lookups()
        return discovered

    def _paired_caller_identity_edges(self) -> set[tuple[int, int]]:
        """Direct-call identities at corresponding instructions in paired owners."""
        orig_candidates = {
            entity.orig_addr
            for entity in self.db.unexplained(ImageId.ORIG)
            if entity.entity_type == EntityType.FUNCTION
            and entity.orig_addr is not None
        }
        recomp_candidates = {
            entity.recomp_addr
            for entity in self.db.unexplained(ImageId.RECOMP)
            if entity.entity_type == EntityType.FUNCTION
            and entity.recomp_addr is not None
        }
        edges: set[tuple[int, int]] = set()
        for caller in self.db.get_functions():
            orig_size = caller.size(ImageId.ORIG)
            recomp_size = caller.size(ImageId.RECOMP)
            if orig_size is None or recomp_size is None:
                continue
            try:
                orig_raw = self.orig_bin.read(caller.orig_addr, orig_size)
                recomp_raw = self.recomp_bin.read(caller.recomp_addr, recomp_size)
            except (InvalidVirtualAddressError, InvalidVirtualReadError):
                continue
            orig_ins = _code_instructions(orig_raw, caller.orig_addr, self.is_32bit)
            recomp_ins = _code_instructions(
                recomp_raw, caller.recomp_addr, self.is_32bit
            )
            if (
                orig_ins is None
                or recomp_ins is None
                or len(orig_ins) != len(recomp_ins)
                or any(a[2] != b[2] for a, b in zip(orig_ins, recomp_ins))
            ):
                continue
            for orig_instruction, recomp_instruction in zip(orig_ins, recomp_ins):
                if orig_instruction[2] != "call":
                    continue
                orig_operand = re.fullmatch(r"0x([0-9a-fA-F]+)", orig_instruction[3])
                recomp_operand = re.fullmatch(
                    r"0x([0-9a-fA-F]+)", recomp_instruction[3]
                )
                if orig_operand is None or recomp_operand is None:
                    continue
                orig_target = int(orig_operand.group(1), 16)
                recomp_target = int(recomp_operand.group(1), 16)
                if (
                    orig_target in orig_candidates
                    and recomp_target in recomp_candidates
                ):
                    edges.add((orig_target, recomp_target))
        return edges

    def discover_unique_called_functions(self) -> list[tuple[int, int]]:
        """Pair callees only when paired-callsite evidence is mutually unique."""
        discovered: list[tuple[int, int]] = []
        while True:
            edges = self._paired_caller_identity_edges()
            orig_degree: dict[int, int] = {}
            recomp_degree: dict[int, int] = {}
            for orig_addr, recomp_addr in edges:
                orig_degree[orig_addr] = orig_degree.get(orig_addr, 0) + 1
                recomp_degree[recomp_addr] = recomp_degree.get(recomp_addr, 0) + 1
            pairs = sorted(
                (orig_addr, recomp_addr)
                for orig_addr, recomp_addr in edges
                if orig_degree[orig_addr] == 1 and recomp_degree[recomp_addr] == 1
            )
            if not pairs:
                return discovered
            self.db.bulk_match(pairs)
            discovered.extend(pairs)
            self.rebuild_lookups()

    def _alias_fingerprint(
        self, image_id: ImageId, addr: int, size: int
    ) -> tuple[tuple[str, str], ...] | None:
        """Cheap body shape used only to limit alias-equivalence proofs.

        Address operands are erased only when the image identifies them as a
        relocation or an existing entity.  Ordinary immediates remain in the
        key, so equal-sized functions with different constants are rejected
        before the more expensive proof.  Relocation position and instruction
        shape remain part of the key.
        """
        cache_key = (image_id, addr, size)
        memo = getattr(self, "_fp_cache", None)
        if memo is not None and cache_key in memo:
            return memo[cache_key]

        image = self.orig_bin if image_id == ImageId.ORIG else self.recomp_bin
        # functions.py imports this module, so import its helper lazily.
        from reccmp.compare.functions import (  # pylint: disable=import-outside-toplevel,cyclic-import
            create_valid_addr_lookup,
        )

        valid_addr = create_valid_addr_lookup(self.db, image_id, image)
        try:
            raw = image.read(addr, size)
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            if memo is not None:
                memo[cache_key] = None
            return None
        instructions = _code_instructions(raw, addr, self.is_32bit)
        if instructions is None:
            if memo is not None:
                memo[cache_key] = None
            return None

        def normalize_operand(operand: str) -> str:
            def replace(match: re.Match[str]) -> str:
                value = int(match.group(0), 16)
                return "<ADDR>" if valid_addr(value) else match.group(0)

            return re.sub(r"0x[0-9a-fA-F]+", replace, operand)

        fingerprint = tuple(
            (mnemonic, normalize_operand(operand))
            for _, _, mnemonic, operand in instructions
        )
        if memo is not None:
            memo[cache_key] = fingerprint
        return fingerprint

    def _classify_function_aliases(
        self, image_id: ImageId, opposite_id: ImageId, matches: list[ReccmpMatch]
    ) -> bool:
        """Classify one image's remaining bodies against canonical pairs."""
        added = False
        canonical_groups: dict[
            tuple[int, tuple[tuple[str, str], ...]], list[ReccmpMatch]
        ] = {}
        for canonical in matches:
            opposite_addr = canonical.addr(opposite_id)
            # An original entity from symbol data often has no size; the pair
            # is compared over the recompiled extent, so its bodies are too.
            opposite_size = canonical.size(opposite_id) or canonical.size(image_id)
            if opposite_addr is None or opposite_size is None or opposite_size <= 0:
                continue
            fingerprint = self._alias_fingerprint(
                opposite_id, opposite_addr, opposite_size
            )
            if fingerprint is not None:
                canonical_groups.setdefault((opposite_size, fingerprint), []).append(
                    canonical
                )

        groups = self._unpaired_function_candidates(image_id)
        for (size, fingerprint), addrs in groups.items():
            for addr in addrs:
                identities: set[int] = set()
                for canonical in canonical_groups.get((size, fingerprint), []):
                    opposite_addr = canonical.addr(opposite_id)
                    assert opposite_addr is not None
                    orig_addr = addr if image_id == ImageId.ORIG else opposite_addr
                    recomp_addr = opposite_addr if image_id == ImageId.ORIG else addr
                    if self.raw_pair_alias_equivalent(orig_addr, recomp_addr, size):
                        identities.add(canonical.orig_addr)
                if len(identities) == 1:
                    added |= self.db.set_alias(image_id, addr, identities.pop())
        return added

    def _unpaired_function_candidates(
        self, image_id: ImageId
    ) -> dict[tuple[int, tuple[tuple[str, str], ...]], list[int]]:
        groups: dict[tuple[int, tuple[tuple[str, str], ...]], list[int]] = {}
        for entity in self.db.unexplained(image_id):
            if entity.entity_type != EntityType.FUNCTION:
                continue
            addr = entity.addr(image_id)
            size = entity.size(image_id)
            if addr is None or size is None or size <= 0:
                continue
            fingerprint = self._alias_fingerprint(image_id, addr, size)
            if fingerprint is not None:
                groups.setdefault((size, fingerprint), []).append(addr)
        return groups
