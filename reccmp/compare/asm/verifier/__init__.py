"""Relational effective-match verifier.

Decides whether two sequences of sanitized assembly lines are
semantically equivalent modulo compiler entropy: register allocation,
swapped commutative operands, and inverted-condition compare/jump pairs.

The two instruction streams are executed in lockstep with a symbolic value
for every register, flag state and x87 stack slot on each side. Registers
never appear inside values; a value records *how* it was computed
(loads, arithmetic, call results...). Renaming a register therefore has no
effect on the values that flow through the function. Equivalence is judged
on the observable effects of each instruction pair:

  * memory stores (address, width and stored value must agree),
  * call targets,
  * branch conditions (canonicalized, so `cmp a, b; jg` equals
    `cmp b, a; jl`) and branch displacements,
  * the returned value in eax (or st(0) for x87 returns).

Commutative operations (add, and, or, xor, imul, test, fadd, fmul) sort
their operand values, so operand-order entropy cancels out.

Anything the model does not understand is handled conservatively: an
unsupported instruction is only allowed when its text is identical on both
sides *and* the two symbolic states are fully synchronized; otherwise the
whole function is rejected (not an effective match).

Layers, each importing only from the ones above it: addresses, state,
evidence, semantics, obligations; then the strategies lockstep, cfg and
iso_cfg, with relocation, cfg_build and block_align supporting them.
"""

from .state import CallAbi, FunctionMetadata, JCC_MNEMONICS
from .relocation import LineEffects, effects_conflict, flags_dead_at, sequence_effects
from .lockstep import verify_effective_match
from .cfg import verify_cfg_effective_match
from .iso_cfg import verify_isomorphic_cfg_effective_match

__all__ = [
    "CallAbi",
    "FunctionMetadata",
    "JCC_MNEMONICS",
    "LineEffects",
    "effects_conflict",
    "flags_dead_at",
    "sequence_effects",
    "verify_cfg_effective_match",
    "verify_effective_match",
    "verify_isomorphic_cfg_effective_match",
]
