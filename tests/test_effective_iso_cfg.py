"""Tests for the isomorphic-CFG effective-match verifier
(reccmp.compare.asm.effective.verify_isomorphic_cfg_effective_match) and the
flag/one-sided generalizations that support it.

The centerpiece sample is a real MSVC 5.0 register-allocation wobble:
TNewsMgr::CreateNewspaper from the Imperialism decompilation, compiled twice
from identical source with a different global-declaration stream. VC5's
allocator swapped esi<->ebx, folded one load into a cmp, elided a register
copy and mirrored one compare region — shifting the length of the function
and every crossing branch displacement.
"""

import difflib
from pathlib import Path

import pytest

from reccmp.compare.asm.effective import (
    FunctionMetadata,
    estimate_isomorphic_cfg_semantic_similarity,
    verify_cfg_effective_match,
    verify_effective_match,
    verify_isomorphic_cfg_effective_match,
)
from reccmp.compare.asm.fixes import analyze_effective_match
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.diagnosis import AnalysisRecorder, ComparisonStatus
from reccmp.compare.pinned_sequences import SequenceMatcherWithPins

SAMPLES = Path(__file__).parent / "samples"


def require_inconclusive_location(analysis):
    location = analysis.inconclusive_location
    assert location is not None
    return location


# --- The real thing ---------------------------------------------------------


@pytest.fixture(name="wobble_analysis")
def fixture_wobble_analysis():
    """Run the full strategy stack on the real VC5 wobble function pair."""
    base = 0x10000000
    orig_raw = (SAMPLES / "msvc5_regalloc_wobble_orig.bin").read_bytes()
    recomp_raw = (SAMPLES / "msvc5_regalloc_wobble_recomp.bin").read_bytes()

    orig_parser = ParseAsm()
    recomp_parser = ParseAsm()
    orig = orig_parser.parse_asm(orig_raw, base)
    recomp = recomp_parser.parse_asm(recomp_raw, base)
    orig_meta_by_addr = orig_parser.collect_instruction_meta(orig_raw, base)
    recomp_meta_by_addr = recomp_parser.collect_instruction_meta(recomp_raw, base)

    orig_asm = [x.display for x in orig]
    recomp_asm = [x.display for x in recomp]
    codes = SequenceMatcherWithPins(orig_asm, recomp_asm, []).get_opcodes()
    return analyze_effective_match(
        codes,
        orig_asm,
        recomp_asm,
        orig_addrs=[x.address for x in orig],
        orig_meta=[
            orig_meta_by_addr.get(row.address) if row.address is not None else None
            for row in orig
        ],
        recomp_addrs=[x.address for x in recomp],
        recomp_meta=[
            recomp_meta_by_addr.get(row.address) if row.address is not None else None
            for row in recomp
        ],
    )


def test_real_vc5_register_wobble_proves_effective(wobble_analysis):
    """The register-allocation wobble class in its full generality: an
    esi<->ebx rename composed with a folded load, an elided copy, a mirrored
    compare region and shifted branch displacements."""
    assert wobble_analysis.status == ComparisonStatus.EFFECTIVE
    assert "register_allocation" in wobble_analysis.effective_reasons
    assert "load_folding" in wobble_analysis.effective_reasons


# --- Distilled positives ----------------------------------------------------


def test_folded_load_with_shifted_branch():
    """orig materializes a load that recomp folds into the cmp: the branch
    displacements differ (the streams have different lengths), so only the
    structural CFG pairing can prove it."""
    orig = [
        "xor eax, eax",
        "mov word ptr [ecx], ax",  # loop body (block 1)
        "mov edx, dword ptr [edi + 8]",
        "inc eax",
        "add ecx, 2",
        "cmp eax, edx",
        "jl -0xe",
        "ret",
    ]
    recomp = [
        "xor eax, eax",
        "mov word ptr [ecx], ax",
        "inc eax",
        "add ecx, 2",
        "cmp eax, dword ptr [edi + 8]",
        "jl -0xc",
        "ret",
    ]
    orig_targets = [None, None, None, None, None, None, 1, None]
    recomp_targets = [None, None, None, None, None, 1, None]
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is True
    )


def test_folded_load_requires_matching_other_side_read():
    """Trap parity: a one-sided load is only harmless when the other side
    provably reads the same address. Here the recomp reads a different
    address, so the extra load could fault on its own."""
    orig = [
        "mov edx, dword ptr [edi + 8]",
        "mov eax, dword ptr [edi + 4]",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [edi + 4]",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    orig_targets: list[int | None] = [None] * 4
    recomp_targets: list[int | None] = [None] * 3
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is False
    )


def test_elided_register_copy():
    """recomp skips a register-to-register copy and uses the source register
    directly; the store and return value still prove correspondence."""
    orig = [
        "mov ecx, ebx",
        "mov dword ptr [esp + 0x1c], ecx",
        "mov eax, ecx",
        "ret",
    ]
    recomp = [
        "mov dword ptr [esp + 0x1c], ebx",
        "mov eax, ebx",
        "ret",
    ]
    orig_targets: list[int | None] = [None] * 4
    recomp_targets: list[int | None] = [None] * 3
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is True
    )


def test_back_edge_into_one_sided_line():
    """A loop back-edge lands on the extra (one-sided) instruction on one
    side and on its paired consumer on the other: the per-side block cuts
    differ, which only per-side CFG construction can represent."""
    orig = [
        "xor eax, eax",
        # loop header (orig target: the extra load)
        "mov edx, dword ptr [edi + 8]",
        "inc eax",
        "cmp eax, edx",
        "jl -0xa",
        "ret",
    ]
    recomp = [
        "xor eax, eax",
        "inc eax",
        "cmp eax, dword ptr [edi + 8]",
        "jl -0x8",
        "ret",
    ]
    orig_targets = [None, None, None, None, 1, None]
    recomp_targets = [None, None, None, 1, None]
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is True
    )


def test_renamed_live_range_across_join():
    """orig keeps a value in ebx where recomp keeps it in edx, and the live
    range crosses a control-flow merge: the join must retain the cross-
    register association for the store after the merge."""
    orig = [
        "mov edx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "mov dword ptr [edi], eax",
        "mov dword ptr [esi], edx",
        "ret",
    ]
    recomp = [
        "mov ecx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "mov dword ptr [edi], eax",
        "mov dword ptr [esi], ecx",
        "ret",
    ]
    orig_targets = [None, None, 4, None, None, None]
    recomp_targets = [None, None, 4, None, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is True
    )


def test_join_through_different_call_sites():
    """Two arms each contain a call; the merged block returns. The x87
    epochs differ across the incoming edges, which must join instead of
    rejecting the function."""
    orig = [
        "test eax, eax",
        "je 0x7",
        "call <OFFSET1>",
        "jmp 0x5",
        "call <OFFSET2>",
        "ret",
    ]
    # `je` jumps to the second call; `jmp` jumps to the shared ret.
    orig_targets: list[int | None] = [None, 4, None, 5, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, list(orig), orig_targets, list(orig_targets)
        )
        is True
    )


# --- Distilled negatives ----------------------------------------------------


def test_reject_one_sided_store_iso():
    """An unmatched store stays a real difference under the CFG pairing."""
    orig = [
        "mov eax, dword ptr [esi]",
        "mov dword ptr [edi], 0",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [esi]",
        "ret",
    ]
    no_targets_orig: list[int | None] = [None] * 3
    no_targets_recomp: list[int | None] = [None] * 2
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, no_targets_orig, no_targets_recomp
        )
        is False
    )


def test_reject_different_store_value_after_join():
    """Genuinely different values reaching the merged store must fail even
    though both sides' registers were renamed."""
    orig = [
        "mov ebx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "inc ebx",
        "mov dword ptr [esi], ebx",
        "ret",
    ]
    recomp = [
        "mov edx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "dec edx",
        "mov dword ptr [esi], edx",
        "ret",
    ]
    targets = [None, None, 4, None, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(orig, recomp, targets, list(targets))
        is False
    )


def test_semantic_similarity_counts_one_local_edit_without_cascade():
    """One wrong constant is one repair unit even though register allocation
    makes every raw line in the computation look different."""
    orig = [
        "mov eax, dword ptr [ebp - 4]",
        "add eax, 5",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    recomp = [
        "mov ecx, dword ptr [ebp - 4]",
        "add ecx, 6",
        "mov dword ptr [esi], ecx",
        "ret",
    ]
    targets: list[int | None] = [None] * 4
    score = estimate_isomorphic_cfg_semantic_similarity(
        orig,
        recomp,
        targets,
        list(targets),
        metadata=FunctionMetadata(return_kind="void"),
    )
    assert score == pytest.approx(0.75)


def test_semantic_similarity_decreases_for_an_independent_edit():
    orig = [
        "mov eax, dword ptr [ebp - 4]",
        "add eax, 5",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    recomp = [
        "mov ecx, dword ptr [ebp - 4]",
        "add ecx, 6",
        "mov dword ptr [edi], ecx",
        "ret",
    ]
    targets: list[int | None] = [None] * 4
    score = estimate_isomorphic_cfg_semantic_similarity(
        orig,
        recomp,
        targets,
        list(targets),
        metadata=FunctionMetadata(return_kind="void"),
    )
    assert score == pytest.approx(0.5)


def test_semantic_similarity_is_unavailable_for_different_cfgs():
    orig = ["test eax, eax", "je 0x2", "inc ebx", "ret"]
    recomp = ["test eax, eax", "inc ebx", "je 0x1", "ret"]
    assert (
        estimate_isomorphic_cfg_semantic_similarity(
            orig,
            recomp,
            [None, 3, None, None],
            [None, None, 3, None],
        )
        is None
    )


def test_semantic_similarity_preserves_one_sided_load_obligations():
    """Recovering an unrelated edit must not invent the load needed to prove
    that an earlier one-sided potentially faulting read is harmless."""
    orig = [
        "mov eax, dword ptr [ebp - 4]",
        "mov ecx, 1",
        "mov edx, 10",
        "ret",
    ]
    recomp = [
        "mov ecx, 2",
        "mov edx, 10",
        "ret",
    ]
    assert (
        estimate_isomorphic_cfg_semantic_similarity(
            orig,
            recomp,
            [None] * len(orig),
            [None] * len(recomp),
        )
        is None
    )


def test_reject_divergent_branch_structure():
    """Different reachable block graphs are not comparable."""
    orig = [
        "test eax, eax",
        "je 0x2",
        "inc ebx",
        "ret",
    ]
    recomp = [
        "test eax, eax",
        "inc ebx",
        "je 0x1",
        "ret",
    ]
    recorder = AnalysisRecorder(
        orig_addrs=[0x1000, 0x1002, 0x1004, 0x1005],
        recomp_addrs=[0x2000, 0x2002, 0x2003, 0x2005],
    )
    assert not verify_isomorphic_cfg_effective_match(
        orig,
        recomp,
        [None, 3, None, None],
        [None, None, 3, None],
        recorder=recorder,
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "non_isomorphic_cfg"
    location = require_inconclusive_location(analysis)
    assert location.address == 0x1004
    assert location.facts["failure"] == "edge_roles"
    assert location.facts["orig_block_count"] == 3
    assert location.facts["recomp_block_count"] == 2


def test_jump_table_data_reports_side_location_and_count():
    orig = ["jmp dword ptr [eax * 4]", "Jump table:", "0x1000"]
    recomp = ["jmp dword ptr [ecx * 4]", "Jump table:", "0x2000"]
    recorder = AnalysisRecorder(
        orig_addrs=[0x1000, 0x1002, 0x1006],
        recomp_addrs=[0x2000, 0x2002, 0x2006],
    )
    assert not verify_isomorphic_cfg_effective_match(
        orig,
        recomp,
        [None, None, None],
        [None, None, None],
        recorder=recorder,
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "jump_table_data"
    location = require_inconclusive_location(analysis)
    assert location.address == 0x1002
    assert location.facts == {
        "side": "orig",
        "data_line_count": 2,
        "data_line": "Jump table:",
    }


def test_empty_control_flow_identifies_the_empty_side():
    recorder = AnalysisRecorder(orig_addrs=[], recomp_addrs=[0x2000])
    assert not verify_isomorphic_cfg_effective_match(
        [], ["ret"], [], [None], recorder=recorder
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "empty_control_flow"
    location = require_inconclusive_location(analysis)
    assert location.instruction_index is None
    assert location.facts == {
        "side": "orig",
        "instruction_count": 0,
    }


def test_invalid_control_flow_target_is_located():
    recorder = AnalysisRecorder(orig_addrs=[0x1000], recomp_addrs=[0x2000])
    assert not verify_isomorphic_cfg_effective_match(
        ["jmp 0x10"], ["jmp 0x10"], [2], [2], recorder=recorder
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "invalid_control_flow_target"
    location = require_inconclusive_location(analysis)
    assert location.address == 0x1000
    assert location.facts["target_instruction_index"] == 2


def test_indirect_jump_is_located():
    recorder = AnalysisRecorder(orig_addrs=[0x1000], recomp_addrs=[0x2000])
    assert not verify_isomorphic_cfg_effective_match(
        ["jmp dword ptr [eax]"],
        ["jmp dword ptr [eax]"],
        [None],
        [None],
        recorder=recorder,
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "indirect_jump"
    assert require_inconclusive_location(analysis).address == 0x1000


def test_external_edge_with_divergent_state_is_located():
    recorder = AnalysisRecorder(
        orig_addrs=[0x1000, 0x1005], recomp_addrs=[0x2000, 0x2005]
    )
    assert not verify_isomorphic_cfg_effective_match(
        ["mov eax, 1", "jmp 0x10"],
        ["mov eax, 2", "jmp 0x10"],
        [None, None],
        [None, None],
        recorder=recorder,
    )
    assert recorder.inconclusive_reason == "external_control_flow_state"
    location = recorder.inconclusive_location
    assert location is not None
    assert location.address == 0x1005
    assert location.facts["edge_kind"] == "jmp"


def test_function_fallthrough_is_located():
    recorder = AnalysisRecorder(orig_addrs=[0x1000], recomp_addrs=[0x2000])
    assert not verify_isomorphic_cfg_effective_match(
        ["mov eax, ecx"], ["mov eax, ecx"], [None], [None], recorder=recorder
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "function_fallthrough"
    assert require_inconclusive_location(analysis).address == 0x1000


def test_incompatible_x87_paths_report_state_join_shape():
    asm = [
        "test eax, eax",
        "je 0x4",
        "fld1",
        "jmp 0x5",
        "nop",
        "ret",
    ]
    targets = [None, 4, None, 5, None, None]
    recorder = AnalysisRecorder(
        orig_addrs=[0x1000, 0x1002, 0x1004, 0x1006, 0x1008, 0x1009],
        recomp_addrs=[0x2000, 0x2002, 0x2004, 0x2006, 0x2008, 0x2009],
    )
    assert not verify_isomorphic_cfg_effective_match(
        asm, list(asm), targets, list(targets), recorder=recorder
    )
    analysis = recorder.failure_analysis()
    assert analysis.inconclusive_reason == "state_join_failure"
    location = require_inconclusive_location(analysis)
    assert location.address == 0x1009
    facts = location.facts
    assert facts["entry_orig_x87_depth"] != facts["incoming_orig_x87_depth"]


def test_reject_mirrored_condition_with_live_difference():
    """A swapped compare that actually changes the taken branch (operands
    are distinct values) must not be excused by target canonicalization."""
    orig = [
        "mov eax, dword ptr [esi]",
        "mov ecx, dword ptr [edi]",
        "cmp eax, ecx",
        "jl 0x2",
        "inc ebx",
        "mov dword ptr [ebp + 8], 1",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [esi]",
        "mov ecx, dword ptr [edi]",
        "cmp ecx, eax",
        "jl 0x2",
        "inc ebx",
        "mov dword ptr [ebp + 8], 1",
        "ret",
    ]
    targets = [None, None, None, 5, None, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(orig, recomp, targets, list(targets))
        is False
    )


def test_reject_folded_load_across_intervening_store():
    """The one-sided load reads [edi + 8] before a store that may alias it;
    the recomp reads it after. The memory generations differ, so the loads
    are not the same read and the values may genuinely differ."""
    orig = [
        "mov edx, dword ptr [edi + 8]",
        "mov dword ptr [ecx], 5",
        "mov dword ptr [esi], edx",
        "ret",
    ]
    recomp = [
        "mov dword ptr [ecx], 5",
        "mov edx, dword ptr [edi + 8]",
        "mov dword ptr [esi], edx",
        "ret",
    ]
    targets: list[int | None] = [None] * 4
    assert (
        verify_isomorphic_cfg_effective_match(orig, recomp, targets, list(targets))
        is False
    )


def test_reject_renamed_callee_saved_across_join():
    """A live range renamed into a callee-saved register without a matching
    save/restore is externally observable at ret."""
    orig = [
        "mov ebx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "mov dword ptr [edi], eax",
        "mov dword ptr [esi], ebx",
        "ret",
    ]
    recomp = [
        "mov edx, dword ptr [ecx]",
        "test eax, eax",
        "je 0x2",
        "mov dword ptr [edi], eax",
        "mov dword ptr [esi], edx",
        "ret",
    ]
    targets = [None, None, 4, None, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(orig, recomp, targets, list(targets))
        is False
    )


# --- Flag canonicalization (lockstep-level) ---------------------------------


def test_test_self_equals_cmp_zero():
    """`test r, r` and `cmp r, 0` produce identical flag states, composed
    with a register rename (void return: eax is dead at ret)."""
    orig = [
        "mov eax, dword ptr [esi]",
        "test eax, eax",
        "jle 0x5",
        "mov dword ptr [edi], eax",
        "ret",
    ]
    recomp = [
        "mov ecx, dword ptr [esi]",
        "cmp ecx, 0",
        "jle 0x5",
        "mov dword ptr [edi], ecx",
        "ret",
    ]
    metadata = FunctionMetadata(return_kind="void")
    assert verify_effective_match(orig, recomp, metadata=metadata) is False
    targets = [None, None, 4, None, None]
    assert (
        verify_cfg_effective_match(orig, recomp, targets, targets, metadata)
        is True
    )


def test_test_self_equals_cmp_zeroed_register():
    """`test r, r` vs `cmp r, z` where z was zeroed by xor: the zero idiom
    must evaluate to the immediate for the flag states to unify."""
    orig = [
        "xor ebx, ebx",
        "mov eax, dword ptr [esi]",
        "test eax, eax",
        "jle 0x5",
        "ret",
    ]
    recomp = [
        "xor ebx, ebx",
        "mov eax, dword ptr [esi]",
        "cmp eax, ebx",
        "jle 0x5",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is True


def test_or_self_preserves_value():
    """`or r, r` does not change the register's value and sets the same
    flags as `test r, r`."""
    orig = [
        "mov eax, dword ptr [esi]",
        "or eax, eax",
        "je 0x3",
        "mov dword ptr [edi], eax",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [esi]",
        "test eax, eax",
        "je 0x3",
        "mov dword ptr [edi], eax",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is True


def test_test_self_unsigned_conditions_stay_sound():
    """After `test r, r`, CF is zero, so `jb` is never taken; after
    `cmp r, 0` likewise. Both canonicalize without asserting a bogus
    equality with a genuine unsigned comparison."""
    orig = [
        "mov eax, dword ptr [esi]",
        "test eax, eax",
        "jb 0x5",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [esi]",
        "cmp eax, 0",
        "jb 0x5",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is True


def test_zero_idioms_unify():
    """xor r,r and sub r,r leave identical flag states."""
    orig = ["xor eax, eax", "je 0x2", "inc ebx", "ret"]
    recomp = ["sub eax, eax", "je 0x2", "inc ebx", "ret"]
    assert verify_effective_match(orig, recomp) is True


def test_test_different_operands_not_conflated_with_cmp():
    """`test a, b` with distinct operands is a bitwise AND, not a compare:
    it must not unify with `cmp a, b`."""
    orig = [
        "test eax, ecx",
        "je 0x2",
        "inc ebx",
        "ret",
    ]
    recomp = [
        "cmp eax, ecx",
        "je 0x2",
        "inc ebx",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False


# --- Scheduling across the stack (per-load alias tags) -----------------------


def test_load_scheduled_across_push():
    """The same argument load before vs after a push: [esp + 8] and
    [esp + 0xc] denote the same slot once the stack adjustment is folded,
    and the push's write to private scratch does not invalidate the load."""
    orig = [
        "mov eax, dword ptr [esp + 8]",
        "push esi",
        "cdq ",
        "mov esi, 0x1d",
        "idiv esi",
        "mov eax, edx",
        "pop esi",
        "ret 8",
    ]
    recomp = [
        "push esi",
        "mov esi, 0x1d",
        "mov eax, dword ptr [esp + 0xc]",
        "cdq ",
        "idiv esi",
        "mov eax, edx",
        "pop esi",
        "ret 8",
    ]
    assert verify_effective_match(orig, recomp) is False  # lengths equal, but
    # positional pairing misaligns; the CFG pairing proves it:
    t_o: list[int | None] = [None] * 8
    t_r: list[int | None] = [None] * 8
    assert verify_isomorphic_cfg_effective_match(orig, recomp, t_o, t_r) is True


def test_unknown_pointer_load_across_push():
    """A load through an incoming pointer is not aliased by a push: while
    no frame pointer has escaped, callee scratch is private."""
    orig = [
        "mov eax, dword ptr [ecx + 0x14]",
        "push esi",
        "mov esi, eax",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov eax, dword ptr [ecx + 0x14]",
        "mov esi, eax",
        "mov eax, esi",
        "pop esi",
        "ret",
    ]
    t: list[int | None] = [None] * 6
    assert verify_isomorphic_cfg_effective_match(orig, recomp, t, list(t)) is True


def test_pushed_then_popped_values_stay_distinct():
    """Two successive push/pop round-trips at the same slot must not be
    conflated: the second pop reads the second pushed value."""
    orig = [
        "push ecx",
        "pop eax",
        "push edx",
        "pop eax",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    recomp = [
        "push ecx",
        "pop eax",
        "push edx",
        "pop ecx",
        "mov dword ptr [esi], ecx",
        "mov eax, ecx",
        "ret",
    ]
    codes = difflib.SequenceMatcher(None, orig, recomp).get_opcodes()
    assert verify_effective_match(orig, recomp, codes) is True
    bad = [
        "push ecx",
        "pop eax",
        "push edx",
        "pop ecx",
        "mov dword ptr [esi], eax",  # stores the FIRST pushed value
        "ret",
    ]
    t: list[int | None] = [None] * 6
    assert verify_isomorphic_cfg_effective_match(orig, bad, t, list(t)) is False


# --- One-sided scratch spills -------------------------------------------------


def test_one_sided_register_spill():
    """One side needs an extra temp register and wraps the region in a
    push/pop the other side never emits; the callee-saved register is
    restored, so the round-trip is externally invisible."""
    orig = [
        "mov ax, word ptr [ecx + 8]",
        "shl ax, 1",
        "add ax, word ptr [ecx + 6]",
        "ret",
    ]
    recomp = [
        "push esi",
        "mov si, word ptr [ecx + 8]",
        "mov ax, word ptr [ecx + 6]",
        "shl si, 1",
        "add ax, si",
        "pop esi",
        "ret",
    ]
    t_o: list[int | None] = [None] * 4
    t_r: list[int | None] = [None] * 7
    assert verify_isomorphic_cfg_effective_match(orig, recomp, t_o, t_r) is True


def test_one_sided_push_live_at_call_rejected():
    """A one-sided push still on the stack at a call site is an extra
    argument, not a spill."""
    orig = [
        "call <OFFSET1>",
        "ret",
    ]
    recomp = [
        "push eax",
        "call <OFFSET1>",
        "add esp, 4",
        "ret",
    ]
    t_o: list[int | None] = [None] * 2
    t_r: list[int | None] = [None] * 4
    assert verify_isomorphic_cfg_effective_match(orig, recomp, t_o, t_r) is False


# --- Recognized switch jump tables -----------------------------------------


def _switch_fixture(index_reg: str = "eax", scratch: str = "ebx"):
    """Minimal cmp/ja + jmp-[idx*4+table] switch with two cases and a default."""
    asm = [
        # Shared stack argument so index-reg renames stay equivalent.
        f"mov {index_reg}, dword ptr [ebp + 8]",
        f"cmp {index_reg}, 1",
        "ja 0x24",
        f"jmp dword ptr [{index_reg}*4 + <OFFSET1>]",
        "Jump table:",
        "start + 0x14",
        "start + 0x18",
        f"mov {scratch}, 1",
        "ret",
        f"mov {scratch}, 2",
        "ret",
        f"xor {scratch}, {scratch}",
        "ret",
    ]
    # Function start 0x1000; case0 at +0x14, case1 at +0x18, default at +0x24.
    addrs: list[int | None] = [
        0x1000,
        0x1003,
        0x1006,
        0x1008,
        None,
        0x100C,  # table slot 0
        0x1010,  # table slot 1
        0x1014,  # case 0
        0x1016,
        0x1018,  # case 1
        0x101A,
        0x1024,  # default
        0x1026,
    ]
    # ja -> default at index 11; table fills the rest.
    targets: list[int | None] = [
        None,
        None,
        11,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    return asm, addrs, targets


def test_recognized_switch_table_iso_cfg_match_under_register_rename():
    orig, orig_addrs, targets = _switch_fixture("eax", "edx")
    recomp, recomp_addrs, _ = _switch_fixture("ecx", "edx")
    # Shift recomp into a different VA range; relative start+ offsets unchanged.
    recomp_addrs = [None if a is None else a + 0x1000 for a in recomp_addrs]
    metadata = FunctionMetadata(return_kind="void")
    assert (
        verify_isomorphic_cfg_effective_match(
            orig,
            recomp,
            targets,
            list(targets),
            metadata=metadata,
            orig_addrs=orig_addrs,
            recomp_addrs=recomp_addrs,
        )
        is True
    )


def test_recognized_switch_wrong_case_order_is_non_isomorphic():
    orig, orig_addrs, targets = _switch_fixture()
    recomp, recomp_addrs, _ = _switch_fixture()
    # Swap case destinations in the recomp table.
    recomp = list(recomp)
    recomp[5], recomp[6] = recomp[6], recomp[5]
    recomp_addrs = [None if a is None else a + 0x1000 for a in recomp_addrs]
    recorder = AnalysisRecorder(orig_addrs=orig_addrs, recomp_addrs=recomp_addrs)
    assert not verify_isomorphic_cfg_effective_match(
        orig,
        recomp,
        targets,
        list(targets),
        metadata=FunctionMetadata(return_kind="void"),
        recorder=recorder,
        orig_addrs=orig_addrs,
        recomp_addrs=recomp_addrs,
    )
    analysis = recorder.failure_analysis()
    assert analysis.status != ComparisonStatus.EFFECTIVE
    assert (
        analysis.inconclusive_reason == "non_isomorphic_cfg"
        or analysis.difference is not None
    )


def test_unresolved_switch_without_addrs_stays_jump_table_data():
    orig, _addrs, targets = _switch_fixture()
    recorder = AnalysisRecorder(
        orig_addrs=[0x1000] * len(orig),
        recomp_addrs=[0x2000] * len(orig),
    )
    assert not verify_isomorphic_cfg_effective_match(
        orig,
        list(orig),
        targets,
        list(targets),
        recorder=recorder,
        # No addrs → cannot resolve start+ entries.
    )
    assert recorder.failure_analysis().inconclusive_reason == "jump_table_data"


# --- CFG canonicalization ----------------------------------------------------


def test_empty_jump_only_block_canonicalizes_to_direct_edge():
    """A → empty jmp → B is the same CFG as A → B after canonicalization.

    Compilers sometimes emit a trampoline basic block that is only ``jmp``;
    isomorphic matching should not fail solely because one side has it.
    """
    # orig: entry --jmp--> empty-jmp --jmp--> ret
    orig = [
        "xor eax, eax",
        "jmp 0x2",
        "jmp 0x3",
        "ret",
    ]
    orig_targets: list[int | None] = [None, 2, 3, None]
    # recomp: entry --jmp--> ret (no trampoline)
    recomp = [
        "xor eax, eax",
        "jmp 0x2",
        "ret",
    ]
    recomp_targets: list[int | None] = [None, 2, None]
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is True
    )


def test_empty_jump_only_block_both_sides_still_match():
    """Trampolines on both sides collapse to the same shape."""
    side = [
        "mov eax, ecx",
        "jmp 0x2",
        "jmp 0x3",
        "mov dword ptr [esi], eax",
        "ret",
    ]
    targets: list[int | None] = [None, 2, 3, None, None]
    assert (
        verify_isomorphic_cfg_effective_match(side, list(side), targets, list(targets))
        is True
    )


def test_ret_and_ret_imm_are_not_collapsed_together():
    """stdcall ``ret 4`` must not be equated with bare ``ret`` by CFG canon."""
    # Two exit paths: bare ret vs ret 4 — different stack cleanup.
    orig = [
        "test eax, eax",
        "je 0x4",
        "ret",
        "ret 4",
    ]
    orig_targets: list[int | None] = [None, 3, None, None]
    recomp = [
        "test eax, eax",
        "je 0x4",
        "ret 4",
        "ret",
    ]
    recomp_targets: list[int | None] = [None, 3, None, None]
    # If canon collapsed all rets into one, both sides would look isomorphic
    # with swapped exits and could wrongly prove. Require divergence or at
    # least no false EFFECTIVE via identical structure after bad collapse.
    # With text-aware collapse, each side keeps two distinct ret blocks →
    # edge roles still match (both have two ret exits), but bodies differ.
    assert (
        verify_isomorphic_cfg_effective_match(
            orig, recomp, orig_targets, recomp_targets
        )
        is False
    )
