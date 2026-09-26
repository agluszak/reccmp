"""A callee's stack cleanup, from the returns its control flow reaches.

The witness does not run callees; after a call it removes the argument bytes
the callee's own ``ret N`` would. See CalleeCleanupMixin._callee_pop_bytes
for what counts as evidence of that ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from capstone import CS_GRP_CALL, CsInsn  # type: ignore[import-untyped]
from capstone.x86 import (  # type: ignore
    X86_INS_JMP,
    X86_INS_RET,
    X86_OP_MEM,
)

from reccmp.compare.asm.decode import direct_branch_target
from reccmp.compare.extent import EntityExtent

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
