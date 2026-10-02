"""Per-pair Ghidra inline hints, never source or decompiled-text substitution."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from capstone import x86_const  # type: ignore

from reccmp.analysis.x86 import decode_one
from reccmp.compare.call_census import _body_calls
from reccmp.types import EntityType, ImageId

from .results import (
    AnalysisFailure,
    FailureKind,
    FunctionResult,
    InlineCode,
    classify,
    classify_inline,
)


def decompiled_lines(code):
    """Discard only Ghidra's inline-expansion notice, not analysis warnings."""
    return (
        "".join(
            line
            for line in code.splitlines(True)
            if not (
                line.startswith("/* WARNING: Inlined function: ")
                and line.rstrip().endswith(" */")
            )
        )
        .lstrip("\n")
        .splitlines(True)
    )


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

    def repeated_nonleaf(self, entry, callees):
        """Ghidra's hard inline model requires one expansion per body."""
        selected = {f"pair:{obj.orig_addr:#x}": obj for obj in callees}
        rejected = set()
        for image, program in self.programs.items():
            address = entry.orig_addr if image == ImageId.ORIG else entry.recomp_addr
            reached = self.reached(image, address, callees)
            owners = [address] + [
                obj.addr(image) for obj in callees if obj.orig_addr in reached
            ]
            counts = Counter(
                call["identity"]
                for owner in owners
                for call in self.continued(image, owner) or ()
                if call["identity"] in selected
            )
            for identity, count in counts.items():
                if count < 2:
                    continue
                function = self.function(image, selected[identity].addr(image))
                if any(
                    instruction.getFlowType().isCall()
                    or instruction.getFlowType().isJump()
                    for instruction in program.getListing().getInstructions(
                        function.getBody(), True
                    )
                ):
                    rejected.add(identity)
        return rejected

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
        cyclic = self.recursive(selected)
        selected = {
            identity: obj
            for identity, obj in selected.items()
            if identity not in cyclic
        }
        repeated = self.repeated_nonleaf(entry, tuple(selected.values()))
        return tuple(
            obj for identity, obj in selected.items() if identity not in repeated
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
def temporary_inline(programs, candidates):
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
        yield
    finally:
        for program, transaction in reversed(transactions):
            program.endTransaction(transaction, False)


@dataclass(frozen=True)
class Decompiled:
    """Raw decompilation evidence, kept separately for ordinary and inline passes."""

    code: str | None
    error: str | None


class InlineNormalizationMixin:
    """Serial retry lifecycle for the reccmp-driven Ghidriff engine."""

    def normalize_inlining(self: Any, programs):
        """Retry pairs serially; flush caches before and after rollback."""
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
                callees = candidates.for_pair(entry)
                if not callees:
                    continue
                self._inline_callees[entry.orig_addr] = tuple(
                    obj.orig_addr for obj in callees
                )
                try:
                    with temporary_inline(programs, callees):
                        for image, program in programs.items():
                            address = self._entry_addr(entry, image)
                            function = candidates.function(image, address)
                            # Include data reached through the substituted bodies;
                            # normalization must not hide changed helper literals.
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
                            self.decompilers[self._program_key(program)][0].flushCache()
                            result = cast(Any, super()).decompile_func(
                                program, function, self.decompiler_timeout
                            )
                            self._inline_decompiled[(image, address)] = Decompiled(
                                result.code if result.completed else None, result.error
                            )
                finally:
                    # The next pair must observe the restored program flags.
                    for program in programs.values():
                        self.decompilers[self._program_key(program)][0].flushCache()
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
        return lines

    def results(self: Any) -> list[FunctionResult]:
        """One result for every function the manifest requested."""
        results = []
        for entry in self.manifest.functions:
            failures = list(self._failures.get(entry.orig_addr, ()))
            if entry.recomp_addr is not None and not failures:
                for image_id in (ImageId.ORIG, ImageId.RECOMP):
                    decompiled = self._decompiled.get(
                        (image_id, self._entry_addr(entry, image_id))
                    )
                    if decompiled is not None and decompiled.error is not None:
                        failures.append(
                            AnalysisFailure(
                                FailureKind.DECOMPILE_ERROR,
                                image_id,
                                message=decompiled.error,
                            )
                        )
            orig_refs = self._references.get((ImageId.ORIG, entry.orig_addr), ())
            recomp_refs = self._references.get((ImageId.RECOMP, entry.orig_addr), ())
            result = classify(
                entry,
                failures=tuple(failures),
                orig_code=self._normalized(ImageId.ORIG, entry.orig_addr),
                recomp_code=self._normalized(ImageId.RECOMP, entry.recomp_addr),
                orig_refs=orig_refs,
                recomp_refs=recomp_refs,
            )
            if entry.orig_addr in self._inline_callees:
                retry_failures = tuple(
                    AnalysisFailure(
                        FailureKind.DECOMPILE_ERROR, image, message=raw.error
                    )
                    for image in (ImageId.ORIG, ImageId.RECOMP)
                    if (
                        raw := self._inline_decompiled.get(
                            (image, self._entry_addr(entry, image))
                        )
                    )
                    is not None
                    and raw.error is not None
                )
                result = classify_inline(
                    result,
                    InlineCode(
                        self._normalized(ImageId.ORIG, entry.orig_addr, inline=True),
                        self._normalized(
                            ImageId.RECOMP, entry.recomp_addr, inline=True
                        ),
                        self._inline_callees[entry.orig_addr],
                        retry_failures,
                    ),
                    orig_refs,
                    recomp_refs,
                )
            results.append(result)
        return results
