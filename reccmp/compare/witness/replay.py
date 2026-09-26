"""Reproduce a recorded witness from its replay record alone.

A witness records the run input (seed, registers, stack arguments, memory
presets), both function ranges and the return kind; the seed determines
everything the model generates. Replaying runs that input again on both
machines and compares the traces as the search did: no solver, no search.
"""

from __future__ import annotations

from dataclasses import dataclass

from reccmp.compare.diagnosis import RefutationWitness
from reccmp.types import ImageId

from .machine import WITNESS_MODEL, RunInput
from .search import Translator, _compare


@dataclass(frozen=True)
class ReplayResult:
    # The witness the recorded input shows now, if any.
    witness: RefutationWitness | None
    # Why it is not the recorded one: no_record, model, orig_image,
    # recomp_image (the images differ from those it was found on; the run
    # still happens), or the reason the run gave no verdict.
    problems: tuple[str, ...] = ()

    @property
    def reproduced(self) -> bool:
        return self.witness is not None and not self.problems


def _same_difference(a: RefutationWitness, b: RefutationWitness) -> bool:
    return (a.kind, a.location, a.orig_value, a.recomp_value) == (
        b.kind,
        b.location,
        b.orig_value,
        b.recomp_value,
    )


def replay(translator: Translator, witness: RefutationWitness) -> ReplayResult:
    record = witness.replay
    if record is None:
        return ReplayResult(None, ("no_record",))
    if record.model != WITNESS_MODEL:
        return ReplayResult(None, ("model",))
    problems = [
        name
        for name, recorded, current in zip(
            ("orig_image", "recomp_image"), record.images, translator.image_digests
        )
        if recorded is not None and recorded != current
    ]
    run_input = RunInput.from_record(record.input)
    traces = [
        translator.machines[side].run(range(start, start + extent), run_input)
        for side, (start, extent) in (
            (ImageId.ORIG, record.orig_function),
            (ImageId.RECOMP, record.recomp_function),
        )
    ]
    shown, reason = _compare(
        traces[0], traces[1], translator, record.return_kind, run_input.seed
    )
    if shown is None:
        problems.append(reason or "agreed")
    elif not _same_difference(shown, witness):
        problems.append("different_difference")
    return ReplayResult(shown, tuple(problems))
