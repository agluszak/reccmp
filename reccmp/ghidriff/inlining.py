"""Per-pair Ghidra inline hints, never source or decompiled-text substitution."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

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
    """Canonical direct-call asymmetry with small, nonrecursive internal callees."""

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
        self.recursive = {}

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

    def is_recursive(self, image, address):
        # Reachability back to the entry is precisely membership in a recursive
        # direct-call SCC. Walk through large and unpaired functions too.
        key = image, address
        if key not in self.recursive:
            pending = [address]
            seen = set()
            recursive = False
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                calls = self.observation(image, current)
                if calls is None:
                    # Incomplete body evidence cannot establish nonrecursion.
                    recursive = True
                    break
                targets = {int(call["target"], 16) for call in calls}
                if address in targets:
                    recursive = True
                    break
                pending.extend(
                    target
                    for target in targets
                    if target not in seen
                    and self.function(image, target) is not None
                    and not self.function(image, target).isExternal()
                )
            self.recursive[key] = recursive
        return self.recursive[key]

    def for_pair(self, entry):
        sides = [
            self.observation(
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
        result = []
        for identity in sorted(identities[0].keys() | identities[1].keys()):
            if identities[0][identity] == identities[1][identity]:
                continue
            obj = self.pairs[int(identity.removeprefix("pair:"), 16)]
            functions = [
                self.function(image, obj.addr(image)) for image in self.programs
            ]
            if any(
                function is None or function.isExternal() or function.isThunk()
                for function in functions
            ):
                continue
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
                continue
            if any(
                self.is_recursive(image, obj.addr(image)) for image in self.programs
            ):
                continue
            result.append(obj)
        return tuple(result)


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
                                        if any(
                                            call["identity"]
                                            == f"pair:{obj.orig_addr:#x}"
                                            for call in candidates.observation(
                                                image, address
                                            )
                                        )
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
