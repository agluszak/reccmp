"""Per-pair Ghidra inline hints, never source or decompiled-text substitution."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
import re

from ghidriff.code import code_tokens

from capstone import x86_const  # type: ignore

from reccmp.analysis.x86 import decode_one
from reccmp.compare.call_census import _body_calls
from reccmp.types import EntityType, ImageId

from .results import (
    AnalysisFailure,
    AnalysisWarning,
    FailureKind,
    FunctionResult,
    classify_pass,
    Outcome,
)

# Bounds nested substitution; a longer chain of inlined helpers is unusual.
_MAX_INLINE_DEPTH = 4
_INLINE_WARNING = re.compile(r" \[inline callee=(0x[0-9a-fA-F]+) reason=([a-z-]+)\]")


def native_warning(image: ImageId, message: str) -> AnalysisWarning:
    """Keep native diagnostics and expose the inliner's own failure identity."""
    match = _INLINE_WARNING.search(message)
    return AnalysisWarning(
        image,
        message,
        match[2] if match else None,
        int(match[1], 16) if match else None,
    )


def decompiled_lines(code):
    """Remove inline metadata while retaining code and analysis warning text."""
    tokens = []
    for kind, token in code_tokens(code):
        if kind == "comment":
            if token.startswith("/* WARNING: Inlined function: "):
                continue
            token = _INLINE_WARNING.sub("", token)
        tokens.append(token)
    return "".join(tokens).lstrip("\n").splitlines(True)


class InlineCandidates:
    """Canonical call and tail-call asymmetry with small, acyclic callees."""

    def __init__(self, manifest, programs):
        self.programs = programs
        self.pairs = {
            obj.orig_addr: obj
            for obj in manifest.objects
            if obj.entity_type == EntityType.FUNCTION
        }
        self.identities = {
            image: {obj.addr(image): obj for obj in self.pairs.values()}
            for image in programs
        }
        for alias in manifest.aliases:
            if alias.canonical_orig in self.pairs:
                self.identities[alias.image_id][alias.addr] = self.pairs[
                    alias.canonical_orig
                ]
        self.calls = {}
        self.tails = {}

    def function(self, image, address):
        program = self.programs[image]
        space = program.getAddressFactory().getDefaultAddressSpace()
        return program.getFunctionManager().getFunctionAt(space.getAddress(address))

    def observation(self, image, address):
        key = image, address
        if key not in self.calls:
            self.calls[key] = _body_calls(
                self.programs[image], image, address, self.identities[image], {}
            )["calls"]
        return self.calls[key]

    def continued(self, image, address):
        """Direct calls and tail-call continuations, or None without evidence."""
        calls = self.observation(image, address)
        return None if calls is None else calls + self.tail_calls(image, address)

    def tail_calls(self, image, address):
        """``jmp rel32`` from the body to another function's entry."""
        key = image, address
        if key in self.tails:
            return self.tails[key]
        self.tails[key] = []
        function = self.function(image, address)
        if function is None:
            return []
        program = self.programs[image]
        space = program.getAddressFactory().getDefaultAddressSpace()
        body = function.getBody()
        result = []
        for instruction in program.getListing().getInstructions(body, True):
            site = int(instruction.getAddress().getOffset())
            decoded = decode_one(
                bytes(int(b) & 255 for b in instruction.getBytes()), site
            )
            if (
                decoded is None
                or decoded.mnemonic != "jmp"
                or len(decoded.operands) != 1
                or decoded.operands[0].type != x86_const.X86_OP_IMM
            ):
                continue
            target = decoded.operands[0].imm & 0xFFFFFFFF
            if body.contains(space.getAddress(target)) or (
                self.function(image, target) is None
            ):
                continue
            obj = self.identities[image].get(target)
            result.append(
                {
                    "site": f"{site:#x}",
                    "target": f"{target:#x}",
                    "identity": (
                        f"pair:{obj.orig_addr:#x}"
                        if obj
                        else f"{image.name.lower()}:{target:#x}"
                    ),
                    "paired": obj is not None,
                }
            )
        self.tails[key] = result
        return result

    def recursive(self, selected):
        """Selected identities on a cycle of selected callees in either image.

        Ghidra expands only the functions this retry marks inline, so only a
        cycle made entirely of selected callees can expand without end. A
        cycle through any other function stays an ordinary call, and that
        function's body evidence is irrelevant to the substitution."""
        result = set()
        for image in self.programs:
            edges = {
                identity: {
                    call["identity"]
                    for call in self.continued(image, obj.addr(image)) or ()
                    if call["identity"] in selected
                }
                for identity, obj in selected.items()
            }
            for identity in selected:
                pending = list(edges[identity])
                seen = set()
                while pending:
                    current = pending.pop()
                    if current == identity:
                        result.add(identity)
                        break
                    if current not in seen:
                        seen.add(current)
                        pending.extend(edges[current])
        return result

    def eligible(self, obj):
        functions = [self.function(image, obj.addr(image)) for image in self.programs]
        if any(
            function is None or function.isExternal() or function.isThunk()
            for function in functions
        ):
            return False
        if any(
            sum(
                1
                for _ in self.programs[image]
                .getListing()
                .getInstructions(function.getBody(), True)
            )
            >= 100
            for image, function in zip(self.programs, functions)
        ):
            return False
        # The substituted body itself must be completely observed.
        return not any(
            self.observation(image, obj.addr(image)) is None for image in self.programs
        )

    def for_pair(self, entry):
        """Asymmetric callees, closed over their tail-call continuations.

        A tail jump continues the callee's body in another function, so
        substituting the callee alone leaves that continuation as a call."""
        sides = [
            self.continued(
                image, entry.orig_addr if image == ImageId.ORIG else entry.recomp_addr
            )
            for image in (ImageId.ORIG, ImageId.RECOMP)
        ]
        if any(calls is None for calls in sides):
            return ()
        identities = [
            Counter(call["identity"] for call in calls if call["paired"])
            for calls in sides
        ]
        selected: dict[str, Any] = {}
        for identity in sorted(identities[0].keys() | identities[1].keys()):
            if identities[0][identity] == identities[1][identity]:
                continue
            obj = self.pairs.get(int(identity.removeprefix("pair:"), 16))
            if obj is not None and self.eligible(obj):
                selected[identity] = obj
        self._close_tails(selected)
        # Retail may have inlined a callee that had itself inlined another.
        # Substituting the outer callee exposes the inner call on one side
        # only, so close the selection over the expanded bodies.
        addresses = (entry.orig_addr, entry.recomp_addr)
        for _ in range(_MAX_INLINE_DEPTH):
            expanded = [
                self.expanded(image, address, selected)
                for image, address in zip((ImageId.ORIG, ImageId.RECOMP), addresses)
            ]
            added = False
            for identity in sorted(expanded[0].keys() | expanded[1].keys()):
                if (
                    identity in selected
                    or expanded[0][identity] == expanded[1][identity]
                ):
                    continue
                obj = self.pairs.get(int(identity.removeprefix("pair:"), 16))
                if obj is not None and self.eligible(obj):
                    selected[identity] = obj
                    added = True
            if not added:
                break
            self._close_tails(selected)
        cyclic = self.recursive(selected)
        selected = {
            identity: obj
            for identity, obj in selected.items()
            if identity not in cyclic
        }
        return tuple(selected.values())

    def _close_tails(self, selected):
        """Add the tail-call continuations of selected callees."""
        pending = list(selected.values())
        while pending:
            obj = pending.pop()
            for image in self.programs:
                for tail in self.tail_calls(image, obj.addr(image)):
                    identity = tail["identity"]
                    if not tail["paired"] or identity in selected:
                        continue
                    target = self.pairs.get(int(identity.removeprefix("pair:"), 16))
                    if target is not None and self.eligible(target):
                        selected[identity] = target
                        pending.append(target)

    def expanded(self, image, address, selected, depth=0):
        """Paired calls of a body after substituting the selected callees."""
        result: Counter = Counter()
        for call in self.continued(image, address) or ():
            if not call["paired"]:
                continue
            identity = call["identity"]
            obj = selected.get(identity)
            if obj is None or depth >= _MAX_INLINE_DEPTH:
                result[identity] += 1
            else:
                result.update(
                    self.expanded(image, obj.addr(image), selected, depth + 1)
                )
        return result

    def resolves(self, entry, callees):
        """Whether substituting these callees leaves both sides the same calls."""
        selected = {f"pair:{obj.orig_addr:#x}": obj for obj in callees}
        return self.expanded(ImageId.ORIG, entry.orig_addr, selected) == self.expanded(
            ImageId.RECOMP, entry.recomp_addr, selected
        )

    def reached(self, image, address, callees):
        """Selected callees whose bodies this side's substitution includes."""
        selected = {f"pair:{obj.orig_addr:#x}": obj for obj in callees}
        reached: set[int] = set()
        pending = [
            call["identity"]
            for call in self.continued(image, address) or ()
            if call["identity"] in selected
        ]
        while pending:
            obj = selected[pending.pop()]
            if obj.orig_addr in reached:
                continue
            reached.add(obj.orig_addr)
            pending.extend(
                call["identity"]
                for call in self.continued(image, obj.addr(image)) or ()
                if call["identity"] in selected
            )
        return reached


@contextmanager
def temporary_inline(programs, candidates, decompilers=()):
    """Rollback every program transaction, including after a failed retry."""
    transactions = []
    try:
        for image, program in programs.items():
            transactions.append(
                (program, program.startTransaction("inline comparison"))
            )
            space = program.getAddressFactory().getDefaultAddressSpace()
            for obj in candidates:
                function = program.getFunctionManager().getFunctionAt(
                    space.getAddress(obj.addr(image))
                )
                function.setInline(True)
            # A direct jump into an inline callee is a tail call: retain its
            # existing machine stack and terminate the caller after expansion.
            # CALL_RETURN tells Ghidra this explicitly instead of leaving an
            # inferred CALL without a continuation. Transaction rollback also
            # restores every instruction override.
            listing = program.getListing()
            references = program.getReferenceManager()
            for obj in candidates:
                target = space.getAddress(obj.addr(image))
                for reference in references.getReferencesTo(target):
                    instruction = listing.getInstructionAt(reference.getFromAddress())
                    if instruction is None:
                        continue
                    # JVM packages become available only after startup.
                    # pylint: disable=import-outside-toplevel,import-error
                    from ghidra.program.model.listing import FlowOverride

                    if instruction.getFlowOverride() not in (
                        FlowOverride.NONE,
                        FlowOverride.CALL,
                    ):
                        continue
                    flow = instruction.getPrototype().getFlowType(
                        instruction.getInstructionContext()
                    )
                    if not flow.isJump() or flow.isConditional():
                        continue
                    flows = instruction.getFlows()
                    if len(flows) != 1 or flows[0] != target:
                        continue
                    owner = program.getFunctionManager().getFunctionContaining(
                        instruction.getAddress()
                    )
                    if owner is not None and not owner.getBody().contains(target):
                        instruction.setFlowOverride(FlowOverride.CALL_RETURN)
        for decompiler in decompilers:
            decompiler.flushCache()
        yield
    finally:
        for program, transaction in reversed(transactions):
            program.endTransaction(transaction, False)
        for decompiler in decompilers:
            decompiler.flushCache()


@dataclass(frozen=True)
class Decompiled:
    """Raw decompilation evidence, kept separately for ordinary and inline passes."""

    code: str | None
    error: str | None
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_native(cls, result):
        comments = tuple(
            token.strip()[3:-2].strip()
            for kind, token in code_tokens(result.code or "")
            if kind == "comment" and token.lstrip().startswith("/* WARNING:")
        )
        return cls(
            result.code if result.completed else None,
            result.error,
            tuple(dict.fromkeys((*result.warnings, *comments))),
        )


class InlineNormalizationMixin:
    """Serial retry lifecycle for the reccmp-driven Ghidriff engine."""

    def normalize_inlining(self: Any, programs):
        """Retry pairs serially; flush caches before and after rollback."""
        self._inline_resolved = {}
        candidates = InlineCandidates(self.manifest, programs)
        old, new = programs[ImageId.ORIG], programs[ImageId.RECOMP]
        self.setup_decompliers(old, new, pair_count=1)
        try:
            for entry in self._comparable_entries():
                # A retry must not conceal an unsuccessful ordinary analysis.
                if self._failures.get(entry.orig_addr) or any(
                    self._decompiled.get((image, self._entry_addr(entry, image)))
                    is None
                    or self._decompiled[(image, self._entry_addr(entry, image))].code
                    is None
                    for image in programs
                ):
                    continue
                if self._ordinary_result(entry).outcome == Outcome.NO_DIFFERENCES:
                    continue
                callees = candidates.for_pair(entry)
                if not callees:
                    continue
                self._inline_callees[entry.orig_addr] = tuple(
                    obj.orig_addr for obj in callees
                )
                self._inline_resolved[entry.orig_addr] = candidates.resolves(
                    entry, callees
                )
                decompilers = tuple(
                    self.decompilers[self._program_key(program)][0]
                    for program in programs.values()
                )
                with temporary_inline(programs, callees, decompilers):
                    for image, program in programs.items():
                        address = self._entry_addr(entry, image)
                        function = candidates.function(image, address)
                        # Include data reached through the substituted bodies;
                        # normalization must not hide changed helper literals.
                        self._inline_references.update(
                            self._collect_references(
                                program,
                                image,
                                {
                                    entry.orig_addr: [function]
                                    + [
                                        candidates.function(image, obj.addr(image))
                                        for obj in callees
                                        if obj.orig_addr
                                        in candidates.reached(image, address, callees)
                                    ]
                                },
                            )
                        )
                        result = self._decompile_native(
                            program, function, self.decompiler_timeout
                        )
                        self._inline_decompiled[(image, address)] = (
                            Decompiled.from_native(result)
                        )
        finally:
            self.shutdown_decompilers(old, new)

    # --- results ----------------------------------------------------------

    def _normalized(
        self: Any, image_id: ImageId, addr: int | None, *, inline: bool = False
    ) -> list[str] | None:
        if addr is None:
            return None
        decompiled = (self._inline_decompiled if inline else self._decompiled).get(
            (image_id, addr)
        )
        if decompiled is None or decompiled.code is None:
            return None
        lines = decompiled_lines(decompiled.code)
        stack_setup = (
            "replaced with injection: alloca_probe" in decompiled.code
            or "ExceptionList" in decompiled.code
        )
        self.normalize_ghidra_decomp_for_side(
            lines, image_id == ImageId.ORIG, addr, stack_setup, inline=inline
        )
        return decompiled_lines("".join(lines))

    def _ordinary_result(self: Any, entry) -> FunctionResult:
        return FunctionResult(
            entry, classify_pass(entry, **self._pass_evidence(entry, inline=False))
        )

    def _pass_evidence(self: Any, entry, *, inline: bool):
        raw_results = self._inline_decompiled if inline else self._decompiled
        references = self._inline_references if inline else self._references
        failures = [] if inline else list(self._failures.get(entry.orig_addr, ()))
        if entry.recomp_addr is not None and not failures:
            for image in (ImageId.ORIG, ImageId.RECOMP):
                raw = raw_results.get((image, self._entry_addr(entry, image)))
                if raw is not None and raw.error is not None:
                    failures.append(
                        AnalysisFailure(
                            FailureKind.DECOMPILE_ERROR, image, message=raw.error
                        )
                    )
        return {
            "failures": tuple(failures),
            "warnings": tuple(
                native_warning(image, warning)
                for image, address in (
                    (ImageId.ORIG, entry.orig_addr),
                    (ImageId.RECOMP, entry.recomp_addr),
                )
                if address is not None
                and (raw := raw_results.get((image, address))) is not None
                for warning in raw.warnings
            ),
            "orig_code": self._normalized(ImageId.ORIG, entry.orig_addr, inline=inline),
            "recomp_code": self._normalized(
                ImageId.RECOMP, entry.recomp_addr, inline=inline
            ),
            "orig_refs": references.get((ImageId.ORIG, entry.orig_addr), ()),
            "recomp_refs": references.get((ImageId.RECOMP, entry.orig_addr), ()),
        }

    def results(self: Any) -> list[FunctionResult]:
        """One result per requested function, retaining each pass's own evidence."""
        results = []
        for entry in self.manifest.functions:
            ordinary = self._ordinary_result(entry)
            callees = self._inline_callees.get(entry.orig_addr, ())
            retry = (
                classify_pass(entry, **self._pass_evidence(entry, inline=True))
                if callees
                else None
            )
            results.append(
                FunctionResult(
                    entry,
                    ordinary.ordinary,
                    retry,
                    callees,
                    getattr(self, "_inline_resolved", {}).get(entry.orig_addr, True),
                )
            )
        return results
