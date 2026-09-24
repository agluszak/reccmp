"""Search for inputs under which two function bodies behave differently.

A witness is a seed whose run shows a different observable effect on the
two sides: a different final value in some non-stack memory location, a
different argument to the same call, a different return value, or a
different stack cleanup. It is reported only when both runs made the same
sequence of calls (by paired identity) up to that point, so callee
behaviour is modelled identically on both sides.

A witness refutes equivalence under the model described in ``machine``:
callees return seeded values and do not touch memory, reads never fault,
and the input may not be reachable from the program's real callers.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from reccmp.compare.db import EntityDb
from reccmp.compare.diagnosis import RefutationWitness as Witness
from reccmp.types import ImageId

from .machine import (
    LOW_PAGES,
    NEAR_IMAGE,
    SEH_CHAIN,
    STACK_TOP,
    RunInput,
    SideMachine,
    Trace,
    model_dword,
)

Identity = Hashable
UNRESOLVED = ("unresolved",)
STACK = ("stack",)

_RETURN_MASKS = {"i8": 0xFF, "i16": 0xFFFF, "i32": 0xFFFFFFFF}


class Translator:
    """Map side addresses to identities that are equal across the sides."""

    def __init__(self, db: EntityDb, orig: SideMachine, recomp: SideMachine):
        self.db = db
        self.machines = {ImageId.ORIG: orig, ImageId.RECOMP: recomp}

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
    skipped: dict[str, int] = field(default_factory=dict)
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

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _fmt(ident: Identity) -> str:
    if isinstance(ident, tuple) and ident and ident[0] == "abs":
        return f"{ident[1]:#x}"
    if isinstance(ident, tuple) and ident and ident[0] == "entity":
        return f"<{ident[1]:#x}+{ident[2]:#x}>"
    return str(ident)


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
        for addr in trace.image_reads:
            if addr in machine.import_slots:
                continue
            if translator.identity(side, addr) == UNRESOLVED:
                return None, "unknown_image_read"
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
    if len(t_o.calls) != len(t_r.calls) and "truncated" not in ends:
        return None, "call_structure"
    if t_o.end != t_r.end:
        return None, "truncated_one_side"
    if t_o.end == "truncated" and len(t_o.calls) != len(t_r.calls):
        return None, "call_structure"

    for index, (call_o, call_r) in enumerate(zip(t_o.calls, t_r.calls)):
        if len(call_o.stack_args) != len(call_r.stack_args):
            return None, "call_arity"
        for arg, (a_o, a_r) in enumerate(zip(call_o.stack_args, call_r.stack_args)):
            v_o, v_r = translator.value(orig, a_o), translator.value(recomp, a_r)
            if UNRESOLVED in (v_o, v_r):
                continue
            if v_o != v_r:
                return (
                    Witness(
                        seed,
                        "call_argument",
                        f"call #{index} argument {arg}",
                        _fmt(v_o),
                        _fmt(v_r),
                        call_o.call_site,
                        call_r.call_site,
                    ),
                    None,
                )

    locations: dict[Identity, int] = {}
    for side, trace in ((orig, t_o), (recomp, t_r)):
        for addr, size in trace.writes.items():
            if addr in SEH_CHAIN:
                continue
            ident = translator.identity(side, addr)
            if ident in (UNRESOLVED, STACK):
                continue
            locations[ident] = max(size, locations.get(ident, 0))

    def mixes_pointer_bytes(trace: Trace, addr: int, size: int) -> bool:
        """Bytes of a stored pointer, not read back as that same store, have
        a layout-dependent value."""
        writers = {
            w[:3] if w else None
            for w in (trace.last_writer.get(b) for b in range(addr, addr + size))
        }
        if writers in ({(addr, size, True)}, {(addr, size, False)}):
            return False
        return any(w is not None and w[2] for w in writers)

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
        if mixes_pointer_bytes(t_o, loc_o, size) or mixes_pointer_bytes(
            t_r, loc_r, size
        ):
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
        return None, None
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
            if mask != 0xFF and v_o[0] == v_r[0] == "abs" and not differing & 0xFF:
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


def find_witness(
    translator: Translator,
    orig_range: range,
    recomp_range: range,
    *,
    return_kind: str = "unknown",
    seeds: int = 8,
    focused_seeds: int = 24,
) -> SearchResult:
    result = SearchResult()
    orig = translator.machines[ImageId.ORIG]
    recomp = translator.machines[ImageId.RECOMP]
    constants = sorted(
        orig.code_constants(orig_range) | recomp.code_constants(recomp_range)
    )
    # Plain seeds first, then one seed focused on each code constant, so a
    # comparison against it is exercised on both sides of the boundary.
    plans: list[tuple[int, tuple[int, ...]]] = [(seed, ()) for seed in range(seeds)]
    plans += [
        (seeds + i, tuple((c + d) & 0xFFFFFFFF for d in (-1, 0, 1)))
        for i, c in enumerate(constants[:focused_seeds])
    ]
    for seed, seed_pool in plans:
        run_input = RunInput.from_seed(seed, seed_pool)

        def call_result(
            index: int, seed: int = seed, seed_pool: tuple[int, ...] = seed_pool
        ) -> tuple[int, int]:
            return (
                model_dword(seed, 3, index, pool=seed_pool),
                model_dword(seed, 4, index, pool=seed_pool),
            )

        t_o = orig.run(orig_range, run_input, call_result)
        t_r = recomp.run(recomp_range, run_input, call_result)
        witness, reason = _compare(t_o, t_r, translator, return_kind, seed)
        result.runs += 1
        if witness is not None:
            result.witness = witness
            return result
        if reason is not None:
            result.skip(reason)
        else:
            result.agreeing_seeds += 1
            result.agreeing_orig.append(frozenset(t_o.executed))
            result.agreeing_recomp.append(frozenset(t_r.executed))
    return result
