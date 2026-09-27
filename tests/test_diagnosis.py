"""Structured semantic diagnosis and strategy-selection tests."""

from difflib import SequenceMatcher
from dataclasses import replace

import pytest

from reccmp.compare.asm.ir import DecodedInstruction, FlowKind
from reccmp.compare.asm.operand import Imm, Mem, Sym
from reccmp.compare.asm.verifier import CallFacts, FunctionMetadata
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonStatus,
    DifferenceKind,
    EffectiveReason,
    InconclusiveReason,
    Strategy,
    StrategyAttempt,
)
from tests.asm_rows import analyze_effective_match, rows


def analyze(orig, recomp, **kwargs):
    codes = SequenceMatcher(None, orig, recomp).get_opcodes()
    return analyze_effective_match(codes, orig, recomp, **kwargs)


def test_exact_result():
    result = analyze(["mov eax, ecx", "ret"], ["mov eax, ecx", "ret"])
    assert result.status == ComparisonStatus.EXACT


def test_register_allocation_reason():
    result = analyze(
        ["mov eax, dword ptr [ebp - 4]", "mov dword ptr [esi], eax"],
        ["mov ecx, dword ptr [ebp - 4]", "mov dword ptr [esi], ecx"],
    )
    assert result.effective_reasons == (EffectiveReason.REGISTER_ALLOCATION,)


def test_frame_slot_layout_reason():
    result = analyze(
        ["mov dword ptr [ebp - 4], eax", "mov ecx, dword ptr [ebp - 4]"],
        ["mov dword ptr [ebp - 8], eax", "mov ecx, dword ptr [ebp - 8]"],
    )
    assert result.effective_reasons == (EffectiveReason.FRAME_SLOT_LAYOUT,)


def test_callee_save_substitution_reason():
    result = analyze(
        [
            "push esi",
            "mov esi, ecx",
            "mov eax, dword ptr [esi + 4]",
            "pop esi",
            "ret",
        ],
        [
            "push edi",
            "mov edi, ecx",
            "mov eax, dword ptr [edi + 4]",
            "pop edi",
            "ret",
        ],
    )
    assert EffectiveReason.CALLEE_SAVE_SUBSTITUTION in result.effective_reasons


def test_instruction_reorder_reason():
    result = analyze(
        [
            "mov eax, dword ptr [ebp - 4]",
            "mov ecx, dword ptr [ebp - 8]",
            "push ecx",
            "push eax",
        ],
        [
            "mov ecx, dword ptr [ebp - 8]",
            "mov eax, dword ptr [ebp - 4]",
            "push ecx",
            "push eax",
        ],
    )
    assert EffectiveReason.INSTRUCTION_REORDER in result.effective_reasons


def test_commutative_order_reason():
    result = analyze(
        [
            "mov eax, dword ptr [g_a (DATA)]",
            "add eax, dword ptr [g_b (DATA)]",
            "mov dword ptr [esi], eax",
        ],
        [
            "mov eax, dword ptr [g_b (DATA)]",
            "add eax, dword ptr [g_a (DATA)]",
            "mov dword ptr [esi], eax",
        ],
    )
    assert result.effective_reasons == (EffectiveReason.COMMUTATIVE_ORDER,)


def test_associative_add_order_reason():
    result = analyze(
        [
            "mov eax, dword ptr [g_a (DATA)]",
            "add eax, dword ptr [g_b (DATA)]",
            "add eax, dword ptr [g_c (DATA)]",
            "ret",
        ],
        [
            "mov eax, dword ptr [g_a (DATA)]",
            "add eax, dword ptr [g_c (DATA)]",
            "add eax, dword ptr [g_b (DATA)]",
            "ret",
        ],
        metadata=FunctionMetadata(return_kind="i32"),
    )
    assert result.status == ComparisonStatus.EFFECTIVE
    assert result.effective_reasons == (EffectiveReason.COMMUTATIVE_ORDER,)


def test_commutative_address_term_order_reason():
    result = analyze(
        ["mov eax, dword ptr [eax + edx]", "ret"],
        ["mov eax, dword ptr [edx + eax]", "ret"],
    )
    assert result.effective_reasons == (EffectiveReason.COMMUTATIVE_ORDER,)


def test_condition_inversion_reason():
    result = analyze(
        ["cmp eax, ebx", "jg 0x2", "ret"],
        ["cmp ebx, eax", "jl 0x2", "ret"],
    )
    assert result.effective_reasons == (EffectiveReason.CONDITION_INVERSION,)


def test_dead_operation_reason_and_effective_precedence():
    result = analyze(
        ["mov eax, dword ptr [esi]", "mov dword ptr [edi], eax"],
        [
            "mov ecx, dword ptr [esi]",
            "mov eax, ecx",
            "mov dword ptr [edi], eax",
        ],
    )
    assert result.status == ComparisonStatus.EFFECTIVE
    assert EffectiveReason.DEAD_OPERATION in result.effective_reasons


def test_padding_reason():
    result = analyze(["ret"], ["ret", "int3"])
    assert result.status == ComparisonStatus.EFFECTIVE
    assert result.effective_reasons == (EffectiveReason.PADDING,)


def test_call_target_difference():
    result = analyze(["call TView::Refresh"], ["call TView::Update"])
    assert result.difference.kind == DifferenceKind.CALL_TARGET
    operand = result.difference.orig.observed.operand
    assert isinstance(operand, Sym)
    assert operand.ref.display == "TView::Refresh"


def test_thiscall_argument_difference():
    metadata = FunctionMetadata(
        return_kind="void",
        call_facts=lambda key: (
            CallFacts(True, False) if key == "TView::Refresh" else None
        ),
    )
    result = analyze(
        [
            "mov ecx, dword ptr [g_pMainView (DATA)]",
            "call TView::Refresh",
            "ret",
        ],
        [
            "mov ecx, dword ptr [g_pTitleView (DATA)]",
            "call TView::Refresh",
            "ret",
        ],
        metadata=metadata,
    )
    assert result.difference.kind == DifferenceKind.CALL_ARGUMENT
    assert result.difference.orig.observed.register == "ecx"
    assert result.difference.orig.observed.value == "dword[g_pMainView (DATA)]"


def test_memory_address_difference_has_components():
    result = analyze(
        ["mov eax, dword ptr [esi + 0x98]", "ret"],
        ["mov eax, dword ptr [esi + 0x9c]", "ret"],
    )
    assert result.difference.kind == DifferenceKind.MEMORY_ADDRESS
    operand = result.difference.orig.observed.operand
    assert isinstance(operand, Mem)
    assert [(term.register, term.scale) for term in operand.terms] == [("esi", 1)]
    assert operand.displacement == 0x98
    assert not operand.symbols


def test_memory_value_difference():
    result = analyze(
        ["mov dword ptr [esi], 1"],
        ["mov dword ptr [esi], 2"],
    )
    assert result.difference.kind == DifferenceKind.MEMORY_VALUE


def test_immediate_value_difference():
    result = analyze(
        ["mov eax, 4", "mov dword ptr [esi], eax"],
        ["mov eax, 5", "mov dword ptr [esi], eax"],
    )
    assert result.difference.kind == DifferenceKind.IMMEDIATE_VALUE
    operand = result.difference.orig.observed.operand
    assert isinstance(operand, Imm)
    assert operand.value == 4


def test_register_renamed_constant_difference_is_a_mismatch():
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
    result = analyze(
        orig,
        recomp,
        orig_addrs=[0x1000, 0x1002, 0x1005, 0x1007],
        recomp_addrs=[0x2000, 0x2002, 0x2005, 0x2007],
        orig_meta=[None] * 4,
        recomp_meta=[None] * 4,
        metadata=FunctionMetadata(return_kind="void"),
    )
    assert result.status == ComparisonStatus.MISMATCH
    assert result.difference.kind == DifferenceKind.IMMEDIATE_VALUE


def test_branch_condition_difference():
    result = analyze(
        ["cmp eax, ebx", "je 0x2", "ret"],
        ["cmp eax, ebx", "jne 0x2", "ret"],
    )
    assert result.difference.kind == DifferenceKind.BRANCH_CONDITION


def _jump_meta(address: int, target: int) -> DecodedInstruction:
    return replace(
        rows(["je 0x2"])[0],
        address=address,
        size=2,
        regs_read=("eflags",),
        regs_written=(),
        reads_flags=True,
        writes_flags=False,
        accesses_memory=False,
        flow=FlowKind.CONDITIONAL,
        branch_target=target,
    )


def test_branch_target_difference_uses_canonical_indices():
    orig = ["cmp eax, ebx", "je 0x2", "inc eax", "ret"]
    recomp = ["cmp eax, ebx", "je 0x4", "inc eax", "ret"]
    result = analyze(
        orig,
        recomp,
        orig_addrs=[0x1000, 0x1002, 0x1004, 0x1005],
        recomp_addrs=[0x2000, 0x2002, 0x2004, 0x2005],
        orig_meta=[None, _jump_meta(0x1002, 0x1005), None, None],
        recomp_meta=[None, _jump_meta(0x2002, 0x2004), None, None],
    )
    assert result.difference.kind == DifferenceKind.BRANCH_TARGET
    assert result.difference.orig.observed.target_index == 3
    assert result.difference.recomp.observed.target_index == 2


def test_typed_return_value_difference():
    result = analyze(
        ["mov eax, 1", "ret"],
        ["mov eax, 2", "ret"],
        metadata=FunctionMetadata(return_kind="i32"),
    )
    assert result.difference.kind == DifferenceKind.RETURN_VALUE


def test_preserved_state_difference():
    result = analyze(
        ["mov ebx, 1"],
        ["mov ebx, 2"],
        metadata=FunctionMetadata(return_kind="void"),
    )
    assert result.difference.kind == DifferenceKind.PRESERVED_STATE
    assert result.difference.orig.observed.register == "ebx"


def test_symbol_resolution_difference():
    result = analyze(
        ["mov eax, g_a (DATA)", "ret"],
        ["mov eax, g_b (DATA)", "ret"],
    )
    assert result.difference.kind == DifferenceKind.SYMBOL_RESOLUTION


def test_unsupported_instruction_is_inconclusive():
    result = analyze(["bswap eax", "ret"], ["bswap ecx", "ret"])
    assert result.status == ComparisonStatus.INCONCLUSIVE
    assert result.inconclusive_reason == InconclusiveReason.UNSUPPORTED_INSTRUCTION
    by_strategy = {attempt.strategy: attempt for attempt in result.attempts}
    assert (
        by_strategy[Strategy.LOCKSTEP].blocker
        == InconclusiveReason.UNSUPPORTED_INSTRUCTION
    )
    assert by_strategy[Strategy.LOCKSTEP].location.instruction_index == 0
    assert (
        by_strategy[Strategy.ISOMORPHIC_CFG].blocker
        == InconclusiveReason.UNSUPPORTED_INSTRUCTION
    )


def test_mismatch_keeps_every_strategy_attempt():
    result = analyze(["mov eax, 1", "ret"], ["mov eax, 2", "ret"])
    assert result.status == ComparisonStatus.MISMATCH
    assert [attempt.strategy for attempt in result.attempts] == [
        Strategy.LOCKSTEP,
        Strategy.DIFF_ALIGNED,
        Strategy.ISOMORPHIC_CFG,
    ]
    lockstep = result.attempts[0]
    assert lockstep.strategy.trusted_alignment
    assert lockstep.difference == result.difference
    assert not result.attempts[1].strategy.trusted_alignment


def test_proven_results_carry_no_attempts():
    assert not analyze(["mov eax, 1", "ret"], ["mov eax, 1", "ret"]).attempts
    with pytest.raises(ValueError):
        ComparisonAnalysis(
            ComparisonStatus.EXACT,
            attempts=(
                StrategyAttempt(
                    Strategy.LOCKSTEP, blocker=InconclusiveReason.MISSING_METADATA
                ),
            ),
        )


def test_attempt_has_exactly_one_outcome():
    with pytest.raises(ValueError):
        StrategyAttempt(Strategy.LOCKSTEP)
    with pytest.raises(ValueError):
        StrategyAttempt("unknown", blocker=InconclusiveReason.ANALYSIS_LIMIT)


def test_reason_order_is_deterministic():
    result = ComparisonAnalysis.effective(
        {EffectiveReason.PADDING, EffectiveReason.DEAD_OPERATION}
    )
    assert result.effective_reasons == (
        EffectiveReason.DEAD_OPERATION,
        EffectiveReason.PADDING,
    )
