"""Search for inputs under which two function bodies behave differently.

A witness is a seed whose run shows a different observable effect on the
two sides: a different final value in some non-stack memory location, a
different argument (on the stack, or in ecx/edx when the callee's call
facts say it reads them) to the same call, a different return value, or a
different stack cleanup. It is reported only when both runs made the same
sequence of calls (by paired identity) up to that point, so callee
behaviour is modelled identically on both sides, and only while every
callee's stack cleanup came from evidence rather than a guess.

A witness refutes equivalence under the model described in ``machine``:
callees return seeded values and do not touch memory, reads never fault,
and the input may not be reachable from the program's real callers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from reccmp.compare.asm.ir import FunctionImage
from reccmp.call_facts import CallFacts
from reccmp.compare.db import EntityDb
from reccmp.compare.diagnosis import RefutationWitness as Witness
from reccmp.types import ImageId

from .machine import (
    LOW_PAGES,
    NEAR_IMAGE,
    SEH_CHAIN,
    STACK_TOP,
    AccessSpan,
    CallEvent,
    RunInput,
    SideMachine,
    Trace,
)

Identity = tuple
UNRESOLVED = ("unresolved",)
STACK = ("stack",)

_RETURN_MASKS = {"i8": 0xFF, "i16": 0xFFFF, "i32": 0xFFFFFFFF}


class Translator:
    """Map side addresses to identities that are equal across the sides."""

    def __init__(
        self,
        db: EntityDb,
        orig: SideMachine,
        recomp: SideMachine,
        call_facts: Callable[[Identity], CallFacts | None] | None = None,
    ):
        """``call_facts`` gives what is known about calling a paired callee
        (``("entity", ...)`` or ``("import", key)`` identity)."""
        self.db = db
        self.machines = {ImageId.ORIG: orig, ImageId.RECOMP: recomp}
        self.call_facts = call_facts or (lambda _identity: None)

    def identity(self, side: ImageId, addr: int) -> Identity:
        # pylint: disable=too-many-return-statements
        machine = self.machines[side]
        if machine.in_stack(addr):
            return STACK
        if addr not in machine.image_range:
            return ("abs", addr)
        entity = self.db.get(side, addr, exact=False)
        if entity is None:
            return UNRESOLVED
        base = entity.addr(side)
        size = entity.size(side)
        if base is None or size is None or not base <= addr < base + max(size, 1):
            return UNRESOLVED
        canonical = self.db.alias_canonical_orig(side, base)
        if canonical is None:
            return UNRESOLVED
        if addr != base and machine.in_code(addr):
            # Code is only referenced at function starts; an address inside a
            # function comes from arithmetic or an over-estimated extent.
            return UNRESOLVED
        return ("entity", canonical, addr - base)

    def identity_span(self, side: ImageId, span: AccessSpan) -> Identity:
        """Identity of an access, when all its bytes belong to one object
        (or all lie outside the image): the identity of its first byte."""
        first = self.identity(side, span.address)
        if span.size <= 1:
            return first
        last = self.identity(side, span.last)
        if first == STACK and last == STACK:
            return STACK
        if first[0] == "abs" and last[0] == "abs":
            return first
        if (
            first[0] == "entity"
            and last[0] == "entity"
            and first[1] == last[1]
            and last[2] - first[2] == span.last - span.address
        ):
            return first
        return UNRESOLVED

    def address(self, side: ImageId, ident: Identity) -> int | None:
        """Inverse of ``identity`` for memory locations."""
        if not isinstance(ident, tuple) or not ident:
            return None
        if ident[0] == "abs":
            return ident[1]
        if ident[0] != "entity":
            return None
        _, canonical, offset = ident
        if side == ImageId.ORIG:
            return canonical + offset
        match = self.db.get_one_match(canonical)
        if match is None or match.recomp_addr is None:
            return None
        return match.recomp_addr + offset

    def value(self, side: ImageId, value: int) -> Identity:
        """A stored or returned dword, compared as a pointer when it is one."""
        machine = self.machines[side]
        if machine.in_stack(value):
            return STACK
        if value in machine.image_range:
            return self.identity(side, value)
        lo, hi = machine.image_range.start, machine.image_range.stop
        if max(lo - NEAR_IMAGE, LOW_PAGES) <= value < hi + NEAR_IMAGE:
            return UNRESOLVED
        return ("abs", value)

    def call_target(self, side: ImageId, target: int | None, slot: int | None):
        machine = self.machines[side]
        if target is not None:
            target = machine.resolve_code(target)
            if target in machine.image_range:
                return self.identity(side, target)
            if target in machine.imports:
                return ("import", machine.imports[target].lower())
        if slot is not None:
            # A pointer loaded from modelled memory (e.g. a vtable slot):
            # identify the call by where the pointer came from.
            return ("via", self.identity(side, slot))
        return UNRESOLVED


@dataclass
class SearchResult:
    witness: Witness | None = None
    # Why seeds produced no verdict, e.g. {"call_structure": 3, "limit": 1}.
    skipped: Counter[str] = field(default_factory=Counter)
    agreeing_seeds: int = 0
    runs: int = 0
    # Instructions executed by the seeds on which both sides agreed.
    agreeing_orig: list[frozenset[int]] = field(default_factory=list)
    agreeing_recomp: list[frozenset[int]] = field(default_factory=list)

    def agreeing_runs_through(self, orig: int | None, recomp: int | None) -> int:
        """Agreeing seeds that executed both given instructions."""
        return sum(
            (orig is None or orig in executed_o)
            and (recomp is None or recomp in executed_r)
            for executed_o, executed_r in zip(self.agreeing_orig, self.agreeing_recomp)
        )


def _fmt(ident: Identity) -> str:
    if isinstance(ident, tuple) and ident and ident[0] == "abs":
        return f"{ident[1]:#x}"
    if isinstance(ident, tuple) and ident and ident[0] == "entity":
        return f"<{ident[1]:#x}+{ident[2]:#x}>"
    return str(ident)


def _argument_witness(
    translator: Translator,
    index: int,
    call_o: CallEvent,
    call_r: CallEvent,
    target: Identity,
    *,
    seed: int,
) -> Witness | None:
    """A difference in what the index-th call receives: its stack arguments,
    and ecx/edx when the callee's facts say it reads them."""
    orig, recomp = ImageId.ORIG, ImageId.RECOMP
    facts = translator.call_facts(target)
    arguments = [
        (f"argument {arg}", a_o, a_r)
        for arg, (a_o, a_r) in enumerate(zip(call_o.stack_args, call_r.stack_args))
    ]
    if facts is not None and facts.uses_ecx:
        arguments.append(("ecx", call_o.ecx, call_r.ecx))
    if facts is not None and facts.uses_edx:
        arguments.append(("edx", call_o.edx, call_r.edx))
    for name, a_o, a_r in arguments:
        v_o, v_r = translator.value(orig, a_o), translator.value(recomp, a_r)
        if UNRESOLVED not in (v_o, v_r) and v_o != v_r:
            return Witness(
                seed,
                "call_argument",
                f"call #{index} {name}",
                _fmt(v_o),
                _fmt(v_r),
                call_o.call_site,
                call_r.call_site,
            )
    return None


def _compare(
    t_o: Trace,
    t_r: Trace,
    translator: Translator,
    return_kind: str,
    seed: int,
) -> tuple[Witness | None, str | None]:
    """Return a witness, or the reason this seed gives no verdict."""
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
    ends = {t_o.end, t_r.end}
    if not ends <= {"return", "truncated"}:
        return None, sorted(ends - {"return", "truncated"})[0]
    orig, recomp = ImageId.ORIG, ImageId.RECOMP
    for side, trace in ((orig, t_o), (recomp, t_r)):
        machine = translator.machines[side]
        for span in trace.image_reads:
            if machine.in_import_slot(span):
                continue
            if translator.identity_span(side, span) == UNRESOLVED:
                return None, "unknown_image_read"
    targets: list[Identity] = []
    for call_o, call_r in zip(t_o.calls, t_r.calls):
        target_o = translator.call_target(orig, call_o.target, call_o.target_slot)
        target_r = translator.call_target(recomp, call_r.target, call_r.target_slot)
        if UNRESOLVED in (target_o, target_r) or ("via", UNRESOLVED) in (
            target_o,
            target_r,
        ):
            return None, "unresolved_call"
        if target_o != target_r:
            return None, "call_structure"
        targets.append(target_o)
    if len(t_o.calls) != len(t_r.calls) and "truncated" not in ends:
        return None, "call_structure"
    if t_o.end != t_r.end:
        return None, "truncated_one_side"
    if t_o.end == "truncated" and len(t_o.calls) != len(t_r.calls):
        return None, "call_structure"

    for index, (call_o, call_r) in enumerate(zip(t_o.calls, t_r.calls)):
        if call_o.assumed_cleanup or call_r.assumed_cleanup:
            # The guessed cleanup may be wrong: the pushes it counted may not
            # be arguments, and esp (with everything read through it) may be
            # off from here on. Nothing after this point is evidence.
            return None, "assumed_call_cleanup"
        if len(call_o.stack_args) != len(call_r.stack_args):
            return None, "call_arity"
        witness = _argument_witness(
            translator, index, call_o, call_r, targets[index], seed=seed
        )
        if witness is not None:
            return witness, None

    locations: dict[Identity, int] = {}
    for side, trace in ((orig, t_o), (recomp, t_r)):
        for addr, size in trace.writes.items():
            span = AccessSpan(addr, size)
            if span.within(SEH_CHAIN):
                continue
            ident = translator.identity_span(side, span)
            if ident in (UNRESOLVED, STACK):
                continue
            locations[ident] = max(size, locations.get(ident, 0))

    def settled(trace: Trace, addr: int, size: int) -> bool:
        """Whether the final value is the function's own: a call after the
        last write (or any call, if it never wrote) could have changed it,
        and callees are modelled as leaving memory alone."""
        for byte in range(addr, addr + size):
            writer = trace.last_writer.get(byte)
            calls_after = len(trace.calls) - (writer[3] if writer else 0)
            if calls_after:
                return False
        return True

    for ident in sorted(locations, key=repr):
        size = locations[ident]
        loc_o = translator.address(orig, ident)
        loc_r = translator.address(recomp, ident)
        if loc_o is None or loc_r is None:
            continue
        if t_o.mixes_pointer_bytes(loc_o, size) or t_r.mixes_pointer_bytes(loc_r, size):
            continue
        if not (settled(t_o, loc_o, size) and settled(t_r, loc_r, size)):
            continue
        raw_o = translator.machines[orig].read(loc_o, size)
        raw_r = translator.machines[recomp].read(loc_r, size)
        if size == 4:
            v_o, v_r = translator.value(orig, raw_o), translator.value(recomp, raw_r)
        else:
            v_o, v_r = ("abs", raw_o), ("abs", raw_r)
        if UNRESOLVED in (v_o, v_r):
            continue
        if v_o != v_r:
            return (
                Witness(seed, "memory_value", _fmt(ident), _fmt(v_o), _fmt(v_r)),
                None,
            )

    if t_o.end != "return":
        # Both stopped at the same call: whatever follows was not observed,
        # so this seed agrees on nothing beyond what it already compared.
        return None, "truncated"
    if t_o.esp_after_return != t_r.esp_after_return:
        return (
            Witness(
                seed,
                "stack_cleanup",
                "esp after return",
                f"ret {t_o.esp_after_return - STACK_TOP - 4:#x}",
                f"ret {t_r.esp_after_return - STACK_TOP - 4:#x}",
            ),
            None,
        )
    mask = _RETURN_MASKS.get(return_kind)
    if mask is not None:
        if mask == 0xFFFFFFFF:
            v_o = translator.value(orig, t_o.eax)
            v_r = translator.value(recomp, t_r.eax)
        else:
            v_o, v_r = ("abs", t_o.eax & mask), ("abs", t_r.eax & mask)
        if UNRESOLVED not in (v_o, v_r) and v_o != v_r:
            differing = t_o.eax ^ t_r.eax
            both_plain = all(isinstance(v, tuple) and v[0] == "abs" for v in (v_o, v_r))
            if mask != 0xFF and both_plain and not differing & 0xFF:
                # Only the bits above al differ. The return width comes from
                # the reconstruction's declaration; retail may return a bool
                # in al with garbage above it.
                return None, "return_upper_bits"
            return Witness(seed, "return_value", "eax", _fmt(v_o), _fmt(v_r)), None
    if return_kind == "i64" and (t_o.eax, t_o.edx) != (t_r.eax, t_r.edx):
        return (
            Witness(
                seed,
                "return_value",
                "edx:eax",
                f"{t_o.edx:#x}:{t_o.eax:#x}",
                f"{t_r.edx:#x}:{t_r.eax:#x}",
            ),
            None,
        )
    return None, None


def _excerpt_constants(image: FunctionImage, machine: SideMachine) -> set[int]:
    """Immediates of the canonical decode that are not addresses. Branch
    targets and ``ret N`` are not data; table rows are not instructions."""
    return {
        value & 0xFFFFFFFF
        for row in image.excerpt
        if row.is_code and not (row.is_call or row.is_jump or row.is_ret)
        for operand in row.operands
        if isinstance(operand, tuple)
        and len(operand) == 2
        and operand[0] == "imm"
        and isinstance(value := operand[1], int)
        and value & 0xFFFFFFFF not in machine.image_range
    }


# Seeds of solver-suggested inputs (witness.hints): far above the plans'.
HINT_SEED = 10_000


def find_witness(
    translator: Translator,
    orig_image: FunctionImage,
    recomp_image: FunctionImage,
    *,
    return_kind: str = "unknown",
    seeds: int = 8,
    focused_seeds: int = 24,
    hints: Sequence[RunInput] = (),
) -> SearchResult:
    # pylint: disable=too-many-arguments
    """Run both functions on the same inputs until their observables
    diverge: ``hints`` first (inputs a solver suggested; their seeds are
    HINT_SEED and up), then plain seeds, then seeds focused on each code
    constant."""
    result = SearchResult()
    orig = translator.machines[ImageId.ORIG]
    recomp = translator.machines[ImageId.RECOMP]
    orig_range = range(orig_image.start_addr, orig_image.start_addr + orig_image.extent)
    recomp_range = range(
        recomp_image.start_addr, recomp_image.start_addr + recomp_image.extent
    )
    constants = sorted(
        _excerpt_constants(orig_image, orig) | _excerpt_constants(recomp_image, recomp)
    )
    # Plain seeds first, then one seed focused on each code constant, so a
    # comparison against it is exercised on both sides of the boundary.
    plans: list[tuple[int, tuple[int, ...]]] = [(seed, ()) for seed in range(seeds)]
    plans += [
        (seeds + i, tuple((c + d) & 0xFFFFFFFF for d in (-1, 0, 1)))
        for i, c in enumerate(constants[:focused_seeds])
    ]
    inputs = [*hints, *(RunInput.from_seed(seed, pool) for seed, pool in plans)]
    for run_input in inputs:
        seed = run_input.seed
        t_o = orig.run(orig_range, run_input)
        t_r = recomp.run(recomp_range, run_input)
        witness, reason = _compare(t_o, t_r, translator, return_kind, seed)
        result.runs += 1
        if witness is not None:
            result.witness = witness
            return result
        if reason is not None:
            result.skipped[reason] += 1
        else:
            result.agreeing_seeds += 1
            result.agreeing_orig.append(frozenset(t_o.executed))
            result.agreeing_recomp.append(frozenset(t_r.executed))
    return result
