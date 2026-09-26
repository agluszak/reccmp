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

import dataclasses
import functools
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from capstone.x86 import X86_OP_REG  # type: ignore

from reccmp.compare.asm.ir import FunctionImage
from reccmp.call_facts import CallFacts
from reccmp.compare.db import EntityDb, EntityTypeLookup, ReccmpEntity
from reccmp.compare.diagnosis import RefutationWitness as Witness
from reccmp.compare.diagnosis import WitnessReplay
from reccmp.compare.extent import EntityExtent
from reccmp.formats.image import image_digest
from reccmp.types import ImageId

from .machine import (
    WITNESS_MODEL,
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
        extent: Callable[[ImageId, ReccmpEntity], EntityExtent | None] | None = None,
    ):
        """``call_facts`` gives what is known about calling a paired callee
        (``("entity", ...)`` or ``("import", key)`` identity). ``extent``
        gives an entity's size on one side, by default the recorded one."""
        self.db = db
        self.machines = {ImageId.ORIG: orig, ImageId.RECOMP: recomp}
        self.call_facts = call_facts or (lambda _identity: None)
        self.extent = extent or _recorded_extent
        # Whether an estimated extent may say an address belongs to an
        # entity: only while exploring, never for a verdict.
        self.admit_estimates = False
        for side, machine in self.machines.items():
            machine.counterpart_cleanup = functools.partial(
                self._counterpart_cleanup, side
            )
        self._image_digests: tuple[str | None, str | None] | None = None

    @property
    def image_digests(self) -> tuple[str | None, str | None]:
        """SHA-256 of the original and recompiled images, when known."""
        if self._image_digests is None:
            self._image_digests = (
                image_digest(self.machines[ImageId.ORIG].image),
                image_digest(self.machines[ImageId.RECOMP].image),
            )
        return self._image_digests

    @contextmanager
    def estimates_admitted(self) -> Iterator[None]:
        self.admit_estimates = True
        try:
            yield
        finally:
            self.admit_estimates = False

    def _counterpart_cleanup(self, side: ImageId, target: int) -> int | None:
        """The certain cleanup of the callee paired with ``target``."""
        ident = self.identity(side, target)
        if ident[0] != "entity" or ident[2] != 0:
            return None
        other = ImageId.RECOMP if side == ImageId.ORIG else ImageId.ORIG
        address = self.address(other, ident)
        if address is None:
            return None
        cleanup = self.machines[other].callee_pop_bytes(address)
        return cleanup.pop if cleanup is not None and cleanup.certain else None

    def identity(self, side: ImageId, addr: int) -> Identity:
        return self.classify(side, addr)[0]

    def classify(self, side: ImageId, addr: int) -> tuple[Identity, str]:
        """``identity`` and why it is what it is: ``stack``, ``outside_image``,
        ``entity``, or for UNRESOLVED ``no_entity``, ``unknown_extent``,
        ``outside_extent``, ``estimated_extent``, ``unpaired`` or
        ``inside_code``."""
        # pylint: disable=too-many-return-statements
        machine = self.machines[side]
        if machine.in_stack(addr):
            return STACK, "stack"
        if addr not in machine.image_range:
            return ("abs", addr), "outside_image"
        entity = self.db.get(side, addr, exact=False)
        if entity is None:
            return UNRESOLVED, "no_entity"
        base = entity.addr(side)
        extent = self.extent(side, entity) if base is not None else None
        if base is None or (extent is None and addr != base):
            # Only an address past the start needs the extent to say that
            # it still belongs to the entity.
            return UNRESOLVED, "unknown_extent"
        if extent is not None and not base <= addr < base + max(extent.size, 1):
            return UNRESOLVED, "outside_extent"
        if (
            addr != base
            and extent is not None
            and not extent.recorded
            and not self.admit_estimates
        ):
            return UNRESOLVED, "estimated_extent"
        canonical = self.db.alias_canonical_orig(side, base)
        if canonical is None:
            return UNRESOLVED, "unpaired"
        if addr != base and machine.in_code(addr):
            # Code is only referenced at function starts; an address inside a
            # function comes from arithmetic or an over-estimated extent.
            return UNRESOLVED, "inside_code"
        return ("entity", canonical, addr - base), "entity"

    def explain(self, side: ImageId, addr: int) -> dict[str, object]:
        """What the database knows about ``addr``, for no-verdict
        diagnostics: its section, why it has its identity, and the entity at
        or before it."""
        machine = self.machines[side]
        ident, why = self.classify(side, addr)
        info: dict[str, object] = {
            "address": f"{addr:#x}",
            "section": machine.section_name(addr),
            "why": why,
        }
        if ident != UNRESOLVED:
            info["identity"] = _fmt(ident)
        entity = (
            self.db.get(side, addr, exact=False)
            if addr in machine.image_range
            else None
        )
        if entity is not None and (base := entity.addr(side)) is not None:
            extent = self.extent(side, entity)
            info["entity"] = {
                "address": f"{base:#x}",
                "size": extent.size if extent is not None else None,
                "size_recorded": extent is not None and extent.recorded,
                "type": EntityTypeLookup.get(entity.entity_type or -1, "UNK"),
                "name": entity.best_name(),
                "matched": entity.matched,
                # The paired original address, for a pair or an alias.
                "canonical": (
                    None
                    if (canonical := self.db.alias_canonical_orig(side, base)) is None
                    else f"{canonical:#x}"
                ),
            }
        return info

    def object_at(self, side: ImageId, addr: int) -> tuple[int, int] | None:
        """(base, size) of the known entity whose extent, recorded or
        estimated, covers ``addr``."""
        entity = self.db.get(side, addr, exact=False)
        if entity is None or (base := entity.addr(side)) is None:
            return None
        extent = self.extent(side, entity)
        if extent is None or not base <= addr < base + max(extent.size, 1):
            return None
        return base, extent.size

    def constant_span(
        self, side: ImageId, span: AccessSpan, anchor: int | None
    ) -> bool:
        """Whether an access reads bytes whose values do not depend on the
        layout although no pair identifies them: all inside one unpaired
        object with a recorded extent in read-only data, the object its
        instruction addresses (``anchor``), and none of them part of a
        relocated pointer. Such a read sees a constant (a float, a string),
        the same wherever the binary put it."""
        first, why = self.classify(side, span.address)
        if first != UNRESOLVED or why != "unpaired" or anchor is None:
            return False
        machine = self.machines[side]
        if not machine.read_only(span):
            return False
        entity = self.db.get(side, span.address, exact=False)
        assert entity is not None
        base = entity.addr(side)
        extent = self.extent(side, entity)
        if base is None or extent is None or not extent.recorded:
            return False
        if span.last >= base + extent.size:
            return False  # runs past the object's end
        if not base <= anchor < base + extent.size:
            return False  # the index left the object the instruction meant
        image = machine.image
        return not any(
            image.is_relocated_addr(address)
            for address in range(span.address - 3, span.last + 1)
        )

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


def _recorded_extent(side: ImageId, entity: ReccmpEntity) -> EntityExtent | None:
    size = entity.size(side)
    return EntityExtent(size) if size is not None else None


@dataclass
class SearchResult:
    # pylint: disable=too-many-instance-attributes
    witness: Witness | None = None
    # Why seeds produced no verdict, e.g. {"call_structure": 3, "limit": 1}.
    skipped: Counter[str] = field(default_factory=Counter)
    # For some of those reasons, what the first such run could not identify.
    skipped_details: dict[str, dict[str, object]] = field(default_factory=dict)
    agreeing_seeds: int = 0
    runs: int = 0
    # Instructions executed by the seeds on which both sides agreed.
    agreeing_orig: list[frozenset[int]] = field(default_factory=list)
    agreeing_recomp: list[frozenset[int]] = field(default_factory=list)
    # For each hint run that refuted nothing: why it gave no verdict
    # ("agreed" if none), and the instructions it executed on each side.
    hint_runs: list[tuple[str, frozenset[int], frozenset[int]]] = field(
        default_factory=list
    )

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


def _resolved(ident: Identity) -> bool:
    return ident not in (UNRESOLVED, ("via", UNRESOLVED))


def _image_read_detail(
    translator: Translator,
    side: ImageId,
    span: AccessSpan,
    trace: Trace,
    counterpart: Trace,
) -> dict[str, object]:
    """An image read without an object identity, and what the other side
    read from its image in the same run."""
    other = ImageId.RECOMP if side == ImageId.ORIG else ImageId.ORIG
    image = translator.machines[side].image
    detail: dict[str, object] = {
        "side": side.name.lower(),
        "size": span.size,
        "instruction": f"{trace.image_reads[span]:#x}",
        # A relocated dword overlaps the bytes read: the value is a pointer.
        "relocated": any(
            image.is_relocated_addr(a) for a in range(span.address - 3, span.last + 1)
        ),
        "first": translator.explain(side, span.address),
    }
    if span.size > 1 and translator.classify(side, span.last)[0] != (
        translator.classify(side, span.address)[0]
    ):
        detail["last"] = translator.explain(side, span.last)
    other_machine = translator.machines[other]
    detail["counterpart_reads"] = [
        {
            "address": f"{read.address:#x}",
            "size": read.size,
            "instruction": f"{site:#x}",
            "identity": _fmt(translator.identity_span(other, read)),
        }
        for read, site in sorted(counterpart.image_reads.items(), key=lambda r: r[1])
        if not other_machine.in_import_slot(read)
    ][:8]
    return detail


def _off_anchor(
    translator: Translator, side: ImageId, trace: Trace
) -> tuple[str, AccessSpan, Identity, Identity] | None:
    """An access that left the object its instruction addresses: an index
    past a table's end reaches whatever each binary placed after it, so
    what it reads or overwrites depends on the layout, not the code. The
    object is the known entity around the anchor, paired or not."""
    image = translator.machines[side].image_range
    for (kind, span), anchor in trace.anchors.items():
        inside = span.within(image)
        if inside:
            extent = translator.object_at(side, anchor)
            if extent is None:
                continue  # no known object to leave
            base, size = extent
            if base <= span.address and span.last < base + max(size, 1):
                continue
        return (
            kind,
            span,
            translator.identity(side, anchor),
            translator.identity_span(side, span),
        )
    return None


def _call_detail(
    translator: Translator, side: ImageId, call: CallEvent, target: Identity
) -> dict[str, object]:
    """How one side's call was made and what its target and slot are."""
    machine = translator.machines[side]
    insn = machine.insn_at(call.call_site)
    if call.target == call.call_site:
        kind = "left_function"  # control left the function other than by call
    elif call.target_slot is not None:
        kind = "memory"
    elif insn is not None and insn.operands and insn.operands[0].type == X86_OP_REG:
        kind = "register"
    else:
        kind = "direct"
    detail: dict[str, object] = {
        "resolved": _resolved(target),
        "kind": kind,
        "call_site": f"{call.call_site:#x}",
        "identity": _fmt(target),
    }
    if call.target is not None:
        resolved = machine.resolve_code(call.target)
        detail["target"] = translator.explain(side, call.target)
        if resolved != call.target:
            detail["thunk_destination"] = translator.explain(side, resolved)
        if resolved in machine.imports:
            detail["import"] = machine.imports[resolved]
    if call.target_slot is not None:
        detail["slot"] = translator.explain(side, call.target_slot)
    return detail


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


def _layout_dependent_access(
    translator: Translator,
    t_o: Trace,
    t_r: Trace,
    details: dict[str, dict[str, object]] | None,
) -> str | None:
    """Why what a run read or wrote may depend on each binary's layout:
    ``unknown_image_read`` (image bytes no object identifies) or
    ``out_of_object_access`` (an access left the object its instruction
    addresses); None when it does not. ``details`` gets the first case."""
    orig, recomp = ImageId.ORIG, ImageId.RECOMP
    for side, trace in ((orig, t_o), (recomp, t_r)):
        machine = translator.machines[side]
        for span in trace.image_reads:
            anchor = trace.anchors.get(("read", span))
            if machine.in_import_slot(span) or translator.constant_span(
                side, span, anchor
            ):
                continue
            if translator.identity_span(side, span) == UNRESOLVED:
                if details is not None and "unknown_image_read" not in details:
                    details["unknown_image_read"] = _image_read_detail(
                        translator, side, span, trace, t_r if side == orig else t_o
                    )
                return "unknown_image_read"
    for side, trace in ((orig, t_o), (recomp, t_r)):
        stray = _off_anchor(translator, side, trace)
        if stray is not None:
            if details is not None and "out_of_object_access" not in details:
                kind, span, meant, got = stray
                details["out_of_object_access"] = {
                    "side": side.name.lower(),
                    "access": kind,
                    "address": f"{span.address:#x}",
                    "size": span.size,
                    "meant": _fmt(meant),
                    "reached": _fmt(got),
                }
            return "out_of_object_access"
    return None


def _compare(
    t_o: Trace,
    t_r: Trace,
    translator: Translator,
    return_kind: str,
    seed: int,
    details: dict[str, dict[str, object]] | None = None,
) -> tuple[Witness | None, str | None]:
    """Return a witness, or the reason this seed gives no verdict. For
    ``unknown_image_read`` and ``unresolved_call``, ``details`` gets what
    was not identified (first occurrence per reason). A location is
    compared only where both sides' addresses for it have its identity."""
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ends = {t_o.end, t_r.end}
    if not ends <= {"return", "truncated"}:
        return None, sorted(ends - {"return", "truncated"})[0]
    orig, recomp = ImageId.ORIG, ImageId.RECOMP
    unsettled = _layout_dependent_access(translator, t_o, t_r, details)
    if unsettled is not None:
        return None, unsettled
    targets: list[Identity] = []
    for index, (call_o, call_r) in enumerate(zip(t_o.calls, t_r.calls)):
        target_o = translator.call_target(orig, call_o.target, call_o.target_slot)
        target_r = translator.call_target(recomp, call_r.target, call_r.target_slot)
        if UNRESOLVED in (target_o, target_r) or ("via", UNRESOLVED) in (
            target_o,
            target_r,
        ):
            if details is not None and "unresolved_call" not in details:
                details["unresolved_call"] = {
                    "index": index,
                    "orig": _call_detail(translator, orig, call_o, target_o),
                    "recomp": _call_detail(translator, recomp, call_r, target_r),
                }
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
        if any(
            translator.identity_span(side, AccessSpan(loc, size)) != ident
            for side, loc in ((orig, loc_o), (recomp, loc_r))
        ):
            # Only one side's store showed where the location is; on the
            # other, nothing shows those bytes belong to the same object.
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
    for index, run_input in enumerate(inputs):
        seed = run_input.seed
        t_o = orig.run(orig_range, run_input)
        t_r = recomp.run(recomp_range, run_input)
        witness, reason = _compare(
            t_o, t_r, translator, return_kind, seed, result.skipped_details
        )
        if witness is None:
            # Would the run refute the pair if estimated extents counted?
            # Then it depends on bytes no record says belong to the object.
            with translator.estimates_admitted():
                estimated, _ = _compare(t_o, t_r, translator, return_kind, seed)
            if estimated is not None:
                reason = "estimated_extent"
                result.skipped_details.setdefault(
                    "estimated_extent",
                    {
                        "kind": estimated.kind,
                        "location": estimated.location,
                        "orig": estimated.orig_value,
                        "recomp": estimated.recomp_value,
                    },
                )
        result.runs += 1
        if witness is not None:
            result.witness = dataclasses.replace(
                witness,
                replay=WitnessReplay(
                    run_input.record(),
                    (orig_range.start, len(orig_range)),
                    (recomp_range.start, len(recomp_range)),
                    return_kind,
                    WITNESS_MODEL,
                    translator.image_digests,
                ),
            )
            return result
        if index < len(hints):
            result.hint_runs.append(
                (reason or "agreed", frozenset(t_o.executed), frozenset(t_r.executed))
            )
        if reason is not None:
            result.skipped[reason] += 1
        else:
            result.agreeing_seeds += 1
            result.agreeing_orig.append(frozenset(t_o.executed))
            result.agreeing_recomp.append(frozenset(t_r.executed))
    return result
