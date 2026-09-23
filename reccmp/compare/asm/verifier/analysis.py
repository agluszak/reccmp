"""Strategy orchestration: EXACT admission, then each verifier strategy in
turn, and the choice of the reported difference or blocker when none proves
equivalence."""

import dataclasses
import logging
from typing import Sequence

from reccmp.compare.asm.const import JUMP_MNEMONICS
from reccmp.compare.asm.instgen import InstructionMeta
from reccmp.compare.asm.ir import (
    AsmStream,
    ResolvedAsm,
    instruction_at,
    instruction_semantic_key,
    resolve_asm_stream,
)
from reccmp.compare.asm.model import Reject
from reccmp.compare.asm.verifier.cfg import verify_cfg_effective_match
from reccmp.compare.asm.verifier.iso_cfg import verify_isomorphic_cfg_effective_match
from reccmp.compare.asm.verifier.lockstep import verify_effective_match
from reccmp.compare.asm.verifier.relocation import undo_relocations
from reccmp.compare.asm.verifier.state import FunctionMetadata
from reccmp.compare.diagnosis import (
    AnalysisRecorder,
    ComparisonAnalysis,
    ComparisonStatus,
    StrategyAttempt,
)
from reccmp.compare.pinned_sequences import DiffOpcode
from reccmp.compare.verification import admit_effective, admit_exact_analysis

logger = logging.getLogger(__name__)


# Alignment padding emitted between functions. int3 traps if executed, so
# it is only trimmed as trailing padding behind an instruction that does
# not fall through — never excused as a one-sided instruction.
_PADDING = ("nop", "int3")


def _trim_padding(stream: ResolvedAsm) -> ResolvedAsm:
    """Strip trailing nop/int3 alignment padding, but only behind an
    instruction that does not fall through into it."""
    displays = stream.displays
    end = len(displays)
    while end > 0 and displays[end - 1] in _PADDING:
        end -= 1
    if 0 < end < len(displays):
        try:
            mnemonic = instruction_at(stream, end - 1).mnemonic
        except (Reject, IndexError, KeyError, ValueError, TypeError):
            mnemonic = displays[end - 1].partition(" ")[0]
        if mnemonic in ("ret", "jmp"):
            return stream.slice(end)
    return stream


def analyze_effective_match(  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-locals
    codes: Sequence[DiffOpcode],
    orig_asm: AsmStream,
    recomp_asm: AsmStream,
    orig_addrs: Sequence[int | None] | None = None,
    metadata: FunctionMetadata | None = None,
    orig_meta: list[InstructionMeta | None] | None = None,
    recomp_addrs: Sequence[int | None] | None = None,
    recomp_meta: list[InstructionMeta | None] | None = None,
    *,
    coverage_incomplete: bool = False,
    extent_closed: bool = True,
) -> ComparisonAnalysis:
    """Canonical semantic analysis of two sanitized instruction streams.

    Prefer ``DecodedInstruction`` excerpts so the verifier uses Capstone
    operands directly. Legacy ``list[str]`` still works via text parse.

    The relational verifier (see the verifier package) proves equivalence modulo
    register allocation, commutative-operand order and inverted compare/jump
    conditions. Instruction-scheduling differences are handled by undoing
    relocations that are proven independent of everything they cross, then
    running the verifier on the reordered sequence — so relocations compose
    with register renames and operand swaps.

    `orig_addrs` (optional) provides the virtual address of each orig line;
    with it, a relocation may cross a forward conditional jump whose target
    lies within the crossed region. `metadata` (optional) provides
    PDB-derived return-type and callee-convention facts that widen what
    the verifier can prove."""
    orig = resolve_asm_stream(orig_asm)
    recomp = resolve_asm_stream(recomp_asm)
    orig_addr_list = list(orig_addrs) if orig_addrs is not None else None
    recomp_addr_list = list(recomp_addrs) if recomp_addrs is not None else None
    orig_sem = tuple(
        instruction_semantic_key(ins) if ins is not None else ("raw", display)
        for ins, display in zip(orig.instructions, orig.displays)
    )
    recomp_sem = tuple(
        instruction_semantic_key(ins) if ins is not None else ("raw", display)
        for ins, display in zip(recomp.instructions, recomp.displays)
    )
    exact = admit_exact_analysis(
        displays_equal=orig.displays == recomp.displays,
        topology_equal=_display_topology_equal(
            orig,
            recomp,
            orig_addr_list,
            orig_meta,
            recomp_addr_list,
            recomp_meta,
        ),
        keys_equal=orig_sem == recomp_sem,
        operands_complete=False,
        control_flow_complete=False,
        coverage_incomplete=coverage_incomplete,
        extent_closed=extent_closed,
    )
    if exact is not None:
        return exact

    def finish_effective(reasons) -> ComparisonAnalysis:
        reason_set = set(reasons)
        if not reason_set:
            reason_set.add("instruction_reorder")
        admitted = admit_effective(
            reason_set,
            coverage_incomplete=coverage_incomplete,
            extent_closed=extent_closed,
        )
        if admitted is None:
            return ComparisonAnalysis.inconclusive(
                "incomplete_coverage" if coverage_incomplete else "open_extent"
            )
        return admitted.analysis

    def new_recorder() -> AnalysisRecorder:
        return AnalysisRecorder(orig_addr_list, recomp_addr_list)

    # Plain lockstep pairing first (with trailing alignment padding
    # trimmed): for equal-length sequences the diff's insert/delete blocks
    # can misalign lines that pair up fine positionally.
    trimmed_orig = _trim_padding(orig)
    trimmed_recomp = _trim_padding(recomp)
    padding = len(trimmed_orig) != len(orig) or len(trimmed_recomp) != len(recomp)
    trimmed_meta = orig_meta[: len(trimmed_orig)] if orig_meta is not None else None
    trimmed_recomp_meta = (
        recomp_meta[: len(trimmed_recomp)] if recomp_meta is not None else None
    )
    relocation_normalized = undo_relocations(codes, orig, recomp, orig_addrs)
    lockstep = new_recorder()
    if verify_effective_match(
        trimmed_orig,
        trimmed_recomp,
        metadata=metadata,
        orig_meta=trimmed_meta,
        recomp_meta=trimmed_recomp_meta,
        recorder=lockstep,
    ):
        if padding:
            logger.debug("effective match: lockstep (padding trimmed)")
        else:
            logger.debug("effective match: lockstep")
        extra_reasons = {"padding"} if padding else set()
        if relocation_normalized is not None:
            extra_reasons.add("instruction_reorder")
        return finish_effective(lockstep.effective_reasons(extra_reasons))

    # Diff-aligned pairing: handles length differences (one-sided entries
    # for whitelisted unobservable instructions, e.g. a redundant
    # register copy) and transposed independent lines.
    diff_aligned = new_recorder()
    if verify_effective_match(
        orig,
        recomp,
        codes,
        metadata=metadata,
        orig_meta=orig_meta,
        recomp_meta=recomp_meta,
        recorder=diff_aligned,
    ):
        logger.debug("effective match: diff-aligned")
        return finish_effective(diff_aligned.effective_reasons())

    relocation = new_recorder()
    if relocation_normalized is not None and verify_effective_match(
        orig, relocation_normalized, metadata=metadata, recorder=relocation
    ):
        logger.debug("effective match: instruction relocation")
        return finish_effective(relocation.effective_reasons({"instruction_reorder"}))

    # CFG-aware verification: needs branch targets for both sides.
    orig_targets = _branch_targets(trimmed_orig, orig_addrs, orig_meta)
    recomp_targets = _branch_targets(trimmed_recomp, recomp_addrs, recomp_meta)
    cfg = new_recorder()
    cfg_attempted = orig_targets is not None and recomp_targets is not None
    if orig_targets is not None and recomp_targets is not None:
        cfg_effective = verify_cfg_effective_match(
            trimmed_orig,
            trimmed_recomp,
            orig_targets,
            recomp_targets,
            metadata=metadata,
            orig_meta=trimmed_meta,
            recomp_meta=trimmed_recomp_meta,
            recorder=cfg,
        )
    else:
        cfg_effective = False
    if cfg_effective:
        logger.debug("effective match: cfg")
        return finish_effective(cfg.effective_reasons({"padding"} if padding else ()))

    # Isomorphic-CFG verification: per-side block graphs matched by
    # structure. Tolerates different instruction counts (folded loads,
    # elided copies) and the shifted branch displacements they cause.
    full_orig_targets = _branch_targets(orig, orig_addr_list, orig_meta)
    full_recomp_targets = _branch_targets(recomp, recomp_addr_list, recomp_meta)
    iso = new_recorder()
    iso_attempted = full_orig_targets is not None and full_recomp_targets is not None
    if full_orig_targets is not None and full_recomp_targets is not None:
        iso_effective = verify_isomorphic_cfg_effective_match(
            orig,
            recomp,
            full_orig_targets,
            full_recomp_targets,
            metadata=metadata,
            orig_meta=orig_meta,
            recomp_meta=recomp_meta,
            recorder=iso,
            orig_addrs=orig_addr_list,
            recomp_addrs=recomp_addr_list,
        )
    else:
        iso_effective = False
    if iso_effective:
        logger.debug("effective match: isomorphic cfg")
        return finish_effective(iso.effective_reasons())

    if not cfg_attempted:
        cfg.mark_inconclusive("missing_metadata")
    attempts = [lockstep.attempt("lockstep"), diff_aligned.attempt("diff_aligned")]
    if relocation_normalized is not None:
        attempts.append(relocation.attempt("relocation"))
    attempts.append(cfg.attempt("cfg"))
    attempts.append(
        iso.attempt("isomorphic_cfg")
        if iso_attempted
        else StrategyAttempt("isomorphic_cfg", blocker="missing_metadata")
    )

    def failed(recorder: AnalysisRecorder) -> ComparisonAnalysis:
        return dataclasses.replace(
            recorder.failure_analysis(), attempts=tuple(attempts)
        )

    # Only positional lockstep and the two CFG strategies establish trusted
    # program points. Diff alignment and relocation are proof-only.
    if cfg_attempted and cfg.best_difference is not None:
        return failed(cfg)
    if lockstep.best_difference is not None:
        return failed(lockstep)
    if iso_attempted and iso.best_difference is not None:
        return failed(iso)
    for candidate in (iso, cfg, lockstep):
        if candidate.inconclusive_reason is not None:
            inconclusive = candidate
            break
    else:
        inconclusive = cfg
    analysis = failed(inconclusive)
    assert analysis.status == ComparisonStatus.INCONCLUSIVE
    return analysis


def _stream_has_local_jumps(stream: ResolvedAsm) -> bool:
    return any(
        display.partition(" ")[0] in JUMP_MNEMONICS for display in stream.displays
    )


def _display_topology_equal(
    orig: ResolvedAsm,
    recomp: ResolvedAsm,
    orig_addrs: Sequence[int | None] | None,
    orig_meta: Sequence[InstructionMeta | None] | None,
    recomp_addrs: Sequence[int | None] | None,
    recomp_meta: Sequence[InstructionMeta | None] | None,
) -> bool:
    # pylint: disable=too-many-positional-arguments
    """True when local branch destinations (instruction ids) are known and agree.

    Jump-free streams are vacuously equal. Jump-bearing streams without
    address/metadata cannot prove topology from displacement text alone.
    """
    if not _stream_has_local_jumps(orig) and not _stream_has_local_jumps(recomp):
        return True
    orig_targets = _branch_targets(orig, orig_addrs, orig_meta)
    recomp_targets = _branch_targets(recomp, recomp_addrs, recomp_meta)
    return orig_targets is not None and orig_targets == recomp_targets


def _branch_targets(
    asm: ResolvedAsm | Sequence[str],
    addrs: Sequence[int | None] | None,
    metas: Sequence[InstructionMeta | None] | None,
) -> list[int | None] | None:
    """Line-index branch targets, resolved through the capstone metadata.
    Targets outside the excerpt resolve to None (external)."""
    if addrs is None or metas is None:
        return None
    length = len(asm)
    index_of = {addr: i for i, addr in enumerate(addrs) if addr is not None}
    result: list[int | None] = []
    for i in range(length):
        meta = metas[i] if i < len(metas) else None
        target = meta.branch_target if meta is not None else None
        # Calls are not local control flow.
        if meta is not None and meta.is_call:
            target = None
        result.append(index_of.get(target) if target is not None else None)
    return result
