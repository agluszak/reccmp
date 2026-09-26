"""A callee's stack cleanup, from the returns its control flow reaches.

Neither the witness nor the verifier runs callees; after a call they remove
the argument bytes the callee's own ``ret N`` would. See
CalleeCleanupMixin._callee_pop_bytes for what counts as evidence of that
``N``. StaticCode reads that evidence from an image without running it.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Callable, Mapping

from capstone import CS_GRP_CALL, CsInsn  # type: ignore[import-untyped]
from capstone.x86 import (  # type: ignore
    X86_INS_ADD,
    X86_INS_CALL,
    X86_INS_JMP,
    X86_INS_RET,
    X86_OP_IMM,
    X86_OP_MEM,
    X86_OP_REG,
    X86_REG_ESP,
)

from reccmp.call_facts import CallFacts
from reccmp.compare.asm.decode import decode_one, direct_branch_target
from reccmp.compare.extent import EntityExtent
from reccmp.formats import Image

# Bounds of the search for a callee's returns.
_MAX_CALLEE_WINDOW = 0x10000
_MAX_CALLEE_INSTRUCTIONS = 4096
# Instructions after which a path does not continue.
_TERMINAL_MNEMONICS = frozenset({"int3", "hlt", "ud2", "iretd", "retf"})


@dataclass(frozen=True)
class Cleanup:
    """The bytes a callee's ``ret N`` removes, and whether the returns it
    was read from are certainly the callee's own code."""

    pop: int
    certain: bool


class CalleeCleanupMixin:
    """Part of SideMachine; relies on its attributes."""

    image_range: range
    _pristine: bytes
    insn_at: Callable[[int], CsInsn | None]
    function_window: Callable[[int], EntityExtent | None]

    def _callee_pop_bytes(self, target: int, depth: int = 0) -> Cleanup | None:
        """The ``ret N`` of a callee in the image, from the returns its
        control flow reaches, and whether that is certain. None when
        unknown; an unknown cleanup is guessed for exploration and makes
        the run give no verdict.

        Direct jumps, conditional branches and ordinary fallthrough lead
        certainly through the callee's own code. Past a call, which may not
        return, the bytes that follow are the callee's own only inside a
        recorded size; without one they may be the next function. Through a
        jump table (whose end is guessed) any case may be foreign. Every
        return found must agree, and the answer is certain when one of them
        was certainly reached. An indirect jump that is not a table gives
        no answer when certainly reached; a direct jump out of the window is
        a tail call, whose own cleanup the path has. Every instruction lies
        whole inside ``function_window``."""
        # pylint: disable=too-many-branches,too-many-return-statements
        if depth > 4 or target not in self.image_range:
            return None
        window = self.function_window(target)
        recorded = window is not None and window.recorded
        size = window.size if window is not None and window.size else _MAX_CALLEE_WINDOW
        stop = min(target + min(size, _MAX_CALLEE_WINDOW), self.image_range.stop)
        pops: set[int] = set()
        certain_return = False
        # Address -> reached certainly; a certain arrival re-walks the path.
        seen: dict[int, bool] = {}
        pending = [(target, True)]
        while pending:
            addr, certain = pending.pop()
            # Walk on unless this path adds nothing: the address was already
            # reached as certainly as now.
            while seen.get(addr) not in (True, certain):
                seen[addr] = certain
                if len(seen) > _MAX_CALLEE_INSTRUCTIONS:
                    return None
                insn = self.insn_at(addr)
                if insn is None or not target <= addr or addr + insn.size > stop:
                    if certain:
                        return None  # the callee's own code runs off its window
                    break  # a guessed path left it: not evidence
                if insn.id == X86_INS_RET:
                    pops.add(insn.operands[0].imm if insn.operands else 0)
                    certain_return |= certain
                    break
                if insn.mnemonic in _TERMINAL_MNEMONICS:
                    break
                if insn.id == X86_INS_JMP:
                    step = self._jump_step(insn, certain, range(target, stop), depth)
                    if step is None:
                        if certain:
                            return None  # an unresolved transfer
                        break
                    successors, tail = step
                    pending.extend(successors)
                    if tail is not None:
                        pops.add(tail.pop)
                        certain_return |= certain and tail.certain
                    break
                if insn.group(CS_GRP_CALL):
                    certain = certain and recorded
                elif (branch := direct_branch_target(insn)) is not None:
                    pending.append((branch, certain))
                addr += insn.size
        if len(pops) != 1:
            return None
        return Cleanup(pops.pop(), certain_return)

    def _jump_step(
        self, insn: CsInsn, certain: bool, window: range, depth: int
    ) -> tuple[list[tuple[int, bool]], Cleanup | None] | None:
        """Where a ``jmp`` in a callee's walk leads: the (address, certain)
        paths it continues on, and the cleanup of a tail call it makes.
        None when it cannot be resolved."""
        branch = direct_branch_target(insn)
        if branch is None:
            cases = self._jump_table_targets(insn, window.start, window.stop)
            if not cases:
                return None
            return [(case, False) for case in cases], None
        if branch in window:
            return [(branch, certain)], None
        tail = self._callee_pop_bytes(branch, depth + 1)
        return None if tail is None else ([], tail)

    def _jump_table_targets(self, insn: CsInsn, start: int, stop: int) -> list[int]:
        """Targets of ``jmp dword ptr [reg*4 + table]`` inside ``[start,
        stop)``, read until the first entry outside it; none when ``insn``
        is not one."""
        op = insn.operands[0] if insn.operands else None
        if (
            op is None
            or op.type != X86_OP_MEM
            or op.mem.scale != 4
            or not op.mem.index
            or op.mem.base
        ):
            return []
        table = op.mem.disp & 0xFFFFFFFF
        targets = []
        for entry in range(256):
            offset = table + 4 * entry - self.image_range.start
            if not 0 <= offset <= len(self._pristine) - 4:
                break
            case = int.from_bytes(self._pristine[offset : offset + 4], "little")
            if not start <= case < stop:
                break
            targets.append(case)
        return targets


def import_key(module: str, name: str) -> str:
    """One import's identity: module names are case-insensitive."""
    return f"{module.lower()}!{name}"


def image_imports(image: Image) -> list[tuple[str, int]]:
    """(key, import table slot) of each import of an image."""
    return [
        (import_key(imp.module, imp.name or f"#{imp.ordinal}"), imp.addr)
        for imp in getattr(image, "get_imports", lambda: ())()
    ]


@dataclass(frozen=True)
class CallStackEffect:
    """What one call does to the caller's stack, from evidence in the
    caller's own binary.

    ``callee_pops``: argument bytes the callee's ``ret N`` removes.
    ``arguments``: argument bytes the callee may read (and write): what it
    pops, or what the caller removes right after the call when the callee
    pops nothing. None when unknown.
    ``certain``: whether ``callee_pops`` was read from returns certainly
    the callee's own (see Cleanup); an uncertain one needs the paired
    call's certain effect to agree."""

    callee_pops: int | None
    arguments: int | None
    certain: bool = True


class StaticCode(CalleeCleanupMixin):
    # pylint: disable=too-many-instance-attributes
    """One binary's code, read without running it: the stack effect of each
    call, by the rules the witness uses (see SideMachine.stack_cleanup)."""

    def __init__(
        self,
        image: Image,
        import_facts: Mapping[str, CallFacts] | None = None,
        function_window: Callable[[int], EntityExtent | None] = lambda _address: None,
    ):
        self.import_facts = import_facts or {}
        self.function_window = function_window
        lo = min(s.virtual_address for s in image.sections)
        hi = max(s.virtual_address + s.extent for s in image.sections)
        self.image_range = range(lo, hi)
        memory = bytearray(hi - lo)
        for section in image.sections:
            data = bytes(section.view[: section.size_of_raw_data])
            offset = section.virtual_address - lo
            memory[offset : offset + len(data)] = data
        self._pristine = bytes(memory)
        # Import table slot -> import name (without its module).
        self.import_slots = {
            slot: key.split("!", 1)[1] for key, slot in image_imports(image)
        }
        self.insn_at = functools.cache(self._decode)
        self.callee_pop_bytes = functools.cache(self._callee_pop_bytes)
        self.call_effect = functools.cache(self._call_effect)

    def _decode(self, addr: int) -> CsInsn | None:
        if addr not in self.image_range:
            return None
        offset = addr - self.image_range.start
        return decode_one(self._pristine[offset : offset + 16], addr)

    def _slot_import(self, op) -> str | None:
        """The import a ``[slot]`` operand reads, if it is an import slot."""
        if op.type != X86_OP_MEM or op.mem.base or op.mem.index:
            return None
        return self.import_slots.get(op.mem.disp & 0xFFFFFFFF)

    def _callee(self, call: CsInsn) -> int | str | None:
        """What a call runs: a code address, through ``jmp`` thunks, or the
        name of an import; None when it is computed."""
        op = call.operands[0] if call.operands else None
        if op is None:
            return None
        if op.type == X86_OP_MEM:
            return self._slot_import(op)
        if op.type != X86_OP_IMM:
            return None
        target = op.imm
        for _ in range(4):
            insn = self.insn_at(target)
            if insn is None or insn.id != X86_INS_JMP or not insn.operands:
                break
            jump = insn.operands[0]
            if jump.type == X86_OP_IMM:
                target = jump.imm
            elif (name := self._slot_import(jump)) is not None:
                return name
            else:
                break
        return target

    def _caller_cleanup(self, after_call: int) -> int | None:
        """Bytes the caller removes right after the call (cdecl)."""
        insn = self.insn_at(after_call)
        if (  # pylint: disable=too-many-boolean-expressions
            insn is not None
            and insn.id == X86_INS_ADD
            and len(insn.operands) == 2
            and insn.operands[0].type == X86_OP_REG
            and insn.operands[0].reg == X86_REG_ESP
            and insn.operands[1].type == X86_OP_IMM
        ):
            return insn.operands[1].imm
        return None

    def _call_effect(self, call_addr: int) -> CallStackEffect | None:
        """The stack effect of the call instruction at ``call_addr``."""
        call = self.insn_at(call_addr)
        if call is None or call.id != X86_INS_CALL:
            return None
        callee = self._callee(call)
        pops: int | None = None
        certain = True
        if isinstance(callee, str):
            facts = self.import_facts.get(callee)
            pops = facts.stack_cleanup if facts is not None else None
        elif callee is not None:
            cleanup = self.callee_pop_bytes(callee)
            if cleanup is not None:
                pops, certain = cleanup.pop, cleanup.certain
        caller = self._caller_cleanup(call_addr + call.size)
        if (pops is None or not certain) and caller is not None and not pops:
            # A caller that removes the arguments itself calls a callee
            # that pops none.
            pops, certain = 0, True
        if pops is None:
            return None
        if pops:
            return CallStackEffect(pops, None if caller else pops, certain)
        return CallStackEffect(0, caller, certain)
