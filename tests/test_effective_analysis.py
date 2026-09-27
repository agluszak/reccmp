import difflib
from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.asm.ir import instruction_match_key
from reccmp.compare.asm.model import ResolvedAddress
from reccmp.compare.asm.parse import decode_function
from reccmp.compare.asm.verifier import verify_effective_match
from reccmp.compare.asm.verifier.relocation import undo_relocations
from tests.asm_rows import analyze_effective_match


def is_effective_match(*args, **kwargs) -> bool:
    """Boolean assertion helper for the semantic regression corpus."""
    return analyze_effective_match(*args, **kwargs).is_effective


def test_unreachable_textual_matches_do_not_hide_a_mismatch():
    """A near-perfect display ratio from unreachable code is still a mismatch."""
    orig_asm = ["mov eax, 1", "ret"] + ["mov ecx, ecx"] * 100
    recomp_asm = ["mov eax, 2", "ret"] + ["mov ecx, ecx"] * 100
    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)

    analysis = analyze_effective_match(
        diff.get_opcodes(),
        orig_asm,
        recomp_asm,
        orig_addrs=list(range(0x1000, 0x1000 + len(orig_asm))),
        orig_meta=[None] * len(orig_asm),
        recomp_addrs=list(range(0x2000, 0x2000 + len(recomp_asm))),
        recomp_meta=[None] * len(recomp_asm),
    )

    assert diff.ratio() > 0.99
    assert analysis.status == ComparisonStatus.MISMATCH


def test_fix_cmp_jmp():
    orig_asm = ["mov eax, 1", "mov ebx, 2", "cmp eax, ebx", "jg 0x1"]
    recomp_asm = ["mov eax, 1", "mov ebx, 2", "cmp ebx, eax", "jl 0x1"]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_test_jmp():
    """`test` is commutative: swapping its operands produces identical flags.
    An identical jump is therefore an effective match..."""
    orig_asm = ["mov eax, 1", "mov ebx, 2", "test eax, ebx", "jg 0x1"]
    recomp_asm = ["mov eax, 1", "mov ebx, 2", "test ebx, eax", "jg 0x1"]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_test_jmp_inverted_jump_invalid():
    """...but an inverted jump is not. Because the flags are the same either
    way, jg and jl react differently to them. (This was previously accepted:
    a false positive of the swapped-cmp patch.)"""
    orig_asm = ["mov eax, 1", "mov ebx, 2", "test eax, ebx", "jg 0x1"]
    recomp_asm = ["mov eax, 1", "mov ebx, 2", "test ebx, eax", "jl 0x1"]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_cmp_jmp_mem_with_different_operands():
    """This should not be fixed up, since the operands are different"""
    orig_asm = [
        "mov eax, dword ptr [ebp-4]",
        "cmp dword ptr [global_var_1 (DATA)], eax",
        "jne 0x1",
    ]
    recomp_asm = [
        "mov eax, dword ptr [global_var_2 (DATA)]",
        "cmp dword ptr [ebp-4], eax",
        "jne 0x1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_cmp_jmp_mem_with_non_matching_jmp():

    orig_asm = [
        "mov eax, dword ptr [ebp-4]",
        "cmp dword ptr [gCurrent_key (DATA)], eax",
        "jl 0x1",
    ]
    recomp_asm = [
        "mov eax, [gCurrent_key (DATA)]",
        "cmp dword ptr [ebp-4], eax",
        "jl 0x1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_cmp_jmp_mem_with_non_matching_jmp_2():

    orig_asm = [
        "mov eax, dword ptr [ebp-4]",
        "cmp dword ptr [gCurrent_key (DATA)], eax",
        "jg 0x1",
    ]
    recomp_asm = [
        "mov eax, [gCurrent_key (DATA)]",
        "cmp dword ptr [ebp-4], eax",
        "jle 0x1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_cmp_jmp_mem_valid():
    """Swapped cmp memories with a live EAX at the jcc.

    Linear verification does not treat predicate membership as live-out
    justification. Without CFG addresses this pair stays unproven.
    """

    orig_asm = [
        "mov eax, dword ptr [ebp-4]",
        "cmp dword ptr [gCurrent_key (DATA)], eax",
        "jne 0x1",
    ]
    recomp_asm = [
        "mov eax, dword ptr [gCurrent_key (DATA)]",
        "cmp dword ptr [ebp-4], eax",
        "jne 0x1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_test_jmp_mem_valid():
    """Same as ``test_fix_mov_cmp_jmp_mem_valid`` for TEST."""

    orig_asm = [
        "mov eax, dword ptr [ebp-4]",
        "test dword ptr [gCurrent_key (DATA)], eax",
        "jne 0x1",
    ]
    recomp_asm = [
        "mov eax, dword ptr [gCurrent_key (DATA)]",
        "test dword ptr [ebp-4], eax",
        "jne 0x1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_fld_fmul_valid():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "fmul dword ptr [ebp - 8]",
        "faddp st(1)",
        "fld dword ptr [ebp - 4]",
        "fadd dword ptr [ebp - 0x14]",
    ]
    recomp_asm = [
        "fld dword ptr [ebp - 8]",
        "fmul dword ptr [ebp - 0x18]",
        "faddp st(1)",
        "fld dword ptr [ebp - 0x14]",
        "fadd dword ptr [ebp - 4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_fld_fadd_fsub():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "fadd dword ptr [ebp - 8]",
    ]
    recomp_asm = ["fld dword ptr [ebp - 8]", "fsub dword ptr [ebp - 0x18]"]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_fld_fadd_with_instruction_between():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "mov eax, 1",
        "fadd dword ptr [ebp - 8]",
    ]
    recomp_asm = ["fld dword ptr [ebp - 8]", "fadd dword ptr [ebp - 0x18]"]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False

    # fadd is commutative, so swapped operands with an intervening
    # non-x87 instruction are an effective match
    # (via is_commutative_x87_chain_swap).
    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "mov eax, 1",
        "fadd dword ptr [ebp - 8]",
    ]
    recomp_asm = [
        "fld dword ptr [ebp - 8]",
        "mov eax, 1",
        "fadd dword ptr [ebp - 0x18]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_fld_fmul_invalid_duplication():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "fmul dword ptr [ebp - 8]",
        "fld dword ptr [ebp - 0x18]",
        "fmul dword ptr [ebp - 8]",
    ]
    recomp_asm = [
        "fld dword ptr [ebp - 8]",
        "fmul dword ptr [ebp - 0x18]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_fld_fmul_invalid_diff_operands():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "fmul dword ptr [ebp - 9]",
    ]
    recomp_asm = [
        "fld dword ptr [ebp - 8]",
        "fmul dword ptr [ebp - 0x18]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_fld_fsub_invalid():

    orig_asm = [
        "fld dword ptr [ebp - 0x18]",
        "fsub dword ptr [ebp - 8]",
    ]
    recomp_asm = [
        "fld dword ptr [ebp - 8]",
        "fsub dword ptr [ebp - 0x18]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_imul_swap_valid():

    orig_asm = [
        "mov eax, dword ptr [ebp - 0x4]",
        "imul eax, dword ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 0x8]",
        "imul eax, dword ptr [ebp - 0x4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_mov_imul_single_operand_imul():
    """Single-operand IMUL multiplies into AX (for a word operand), and
    multiplication is commutative, so loading the other factor first is an
    effective match."""

    orig_asm = [
        "mov ax, word ptr [ebp - 0x4]",
        "imul word ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov ax, word ptr [ebp - 0x8]",
        "imul word ptr [ebp - 0x4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_mov_add_swap_valid():

    orig_asm = [
        "mov eax, dword ptr [ebp - 0x4]",
        "add eax, dword ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 0x8]",
        "add eax, dword ptr [ebp - 0x4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_mov_add_swap_with_literal_valid():

    orig_asm = [
        "mov eax, 1",
        "add eax, dword ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 0x8]",
        "add eax, 1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is True


def test_fix_mov_add_swap_on_stack_invalid():

    orig_asm = [
        "mov dword ptr [ebp - 0x4], 1",
        "add dword ptr [ebp - 0x4], 2",
    ]
    recomp_asm = [
        "mov dword ptr [ebp - 0x4], 2",
        "add dword ptr [ebp - 0x4], 1",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    # Pretty sure this is actually safe, but not implemented
    assert is_effective is False


def test_fix_mov_sub_swap_invalid():

    orig_asm = [
        "mov eax, dword ptr [ebp - 0x4]",
        "sub eax, dword ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 0x8]",
        "sub eax, dword ptr [ebp - 0x4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    # Like the add/imul tests except subtraction is NOT commutative
    assert is_effective is False


def test_fix_mov_add_invalid_dest():

    orig_asm = [
        "mov eax, dword ptr [ebp - 0x4]",
        "add eax, dword ptr [ebp - 0x8]",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 0x8]",
        "add ebx, dword ptr [ebp - 0x4]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_this_should_not_be_marked_as_effective():
    """The instructions `mov eax, 0` and `mov ecx, 1` cannot have their registers swapped."""

    orig_asm = [
        "mov eax, dword ptr [esi + 0x100]",
        "mov ecx, dword ptr [eax + 0x74]",
        "add eax, 0x74",
        "sub ecx, 3",
        "cmp ecx, 0xc",
        "ja 0x0",
        "mov eax, 0",
        "mov ecx, 1",
        "mov dword ptr [eax], 2",
    ]
    recomp_asm = [
        "mov ecx, dword ptr [esi + 0x100]",
        "mov eax, dword ptr [ecx + 0x74]",
        "add ecx, 0x74",
        "sub eax, 3",
        "cmp eax, 0xc",
        "ja 0x0",
        "mov eax, 0",
        "mov ecx, 1",
        "mov dword ptr [ecx], 2",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_fix_mov_cmp_jmp_unsafe_intermediate_reuse():
    # These are NOT equivalent since eax is used after the jmp
    orig_asm = [
        "mov eax, dword ptr [ebp - 8]",
        "cmp eax, dword ptr [ebp - 4]",
        "jl 0x2",
        "mov dword ptr [ebp - 0xc], eax",
    ]
    recomp_asm = [
        "mov eax, dword ptr [ebp - 4]",
        "cmp eax, dword ptr [ebp - 8]",
        "jg 0x2",
        "mov dword ptr [ebp - 0xc], eax",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_and_swap_not_allowed():
    """Cannot move the `and eax, 0xff` instruction for an effective match.
    `eax` is modified by the intermediate instructions. (GH #322)"""

    orig_asm = [
        "mov eax, dword ptr [ebp - 4]",
        "and eax, 0xff",  # Move this
        "mov ecx, dword ptr [gReal_render_palette (DATA)]",
        "mov eax, dword ptr [ecx + eax*4]",
        # To here
        "mov ecx, dword ptr [gRender_palette (DATA)]",
    ]

    recomp_asm = [
        "mov eax, dword ptr [ebp - 4]",
        "mov ecx, dword ptr [gReal_render_palette (DATA)]",
        "mov eax, dword ptr [ecx + eax*4]",
        "and eax, 0xff",
        "mov ecx, dword ptr [gRender_palette (DATA)]",
    ]

    diff = difflib.SequenceMatcher(None, orig_asm, recomp_asm)
    is_effective = is_effective_match(diff.get_opcodes(), orig_asm, recomp_asm)

    assert is_effective is False


def test_recomp_shorter_than_orig():
    """Sequences of different length never crash and are not effective."""
    orig = ["mov eax, 1", "mov ebx, 2", "cmp eax, ebx", "jg 0x1"]
    recomp = ["mov eax, 1", "mov ebx, 2"]
    diff = difflib.SequenceMatcher(None, orig, recomp)
    assert is_effective_match(diff.get_opcodes(), orig, recomp) is False


# The following tests cover the commutative x87 operand-chain swap: MSVC
# nondeterministically swaps the operand chains of commutative x87
# computations (e.g. tableA[i] + tableB[j]) between recompiles.
# The asm below is the real shape of Imperialism 0x4e0590 after sanitization.

X87_CHAIN_ORIG = [
    "mov eax, dword ptr [ecx + 0x94]",
    "movsx edx, word ptr [eax + 0xc]",
    "mov eax, dword ptr [ecx + 0x9c]",
    "fld dword ptr [edx*4 + g_skillTableA (DATA)]",
    "movsx ecx, word ptr [eax + 0xc]",
    "fadd dword ptr [ecx*4 + g_skillTableB (DATA)]",
    "ret",
]

X87_CHAIN_RECOMP = [
    "mov eax, dword ptr [ecx + 0x9c]",
    "movsx edx, word ptr [eax + 0xc]",
    "mov eax, dword ptr [ecx + 0x94]",
    "fld dword ptr [edx*4 + g_skillTableB (DATA)]",
    "movsx ecx, word ptr [eax + 0xc]",
    "fadd dword ptr [ecx*4 + g_skillTableA (DATA)]",
    "ret",
]


def test_commutative_x87_chain_swap_valid():
    """The fld/fadd displacements are cross-swapped and the two
    address-load movs transpose: an effective match."""
    diff = difflib.SequenceMatcher(None, X87_CHAIN_ORIG, X87_CHAIN_RECOMP)
    assert (
        is_effective_match(diff.get_opcodes(), X87_CHAIN_ORIG, X87_CHAIN_RECOMP) is True
    )


def test_commutative_x87_chain_swap_fsub_invalid():
    """fsub is not commutative: must not be an effective match."""
    recomp = list(X87_CHAIN_RECOMP)
    recomp[5] = "fsub dword ptr [ecx*4 + g_skillTableA (DATA)]"

    diff = difflib.SequenceMatcher(None, X87_CHAIN_ORIG, recomp)
    assert is_effective_match(diff.get_opcodes(), X87_CHAIN_ORIG, recomp) is False


def test_commutative_x87_chain_swap_mov_not_transposed():
    """A mov that differs without a transposed partner is a real diff."""
    recomp = list(X87_CHAIN_RECOMP)
    recomp[0] = "mov eax, dword ptr [ecx + 0xa0]"

    diff = difflib.SequenceMatcher(None, X87_CHAIN_ORIG, recomp)
    assert is_effective_match(diff.get_opcodes(), X87_CHAIN_ORIG, recomp) is False


def test_commutative_x87_chain_swap_x87_instruction_between():
    """An x87 instruction between the fld and the fadd modifies st(0),
    so the operand order matters: must not be an effective match."""
    orig = [
        "fld dword ptr [g_floatA (FLOAT)]",
        "fsqrt",
        "fadd dword ptr [g_floatB (FLOAT)]",
        "ret",
    ]
    recomp = [
        "fld dword ptr [g_floatB (FLOAT)]",
        "fsqrt",
        "fadd dword ptr [g_floatA (FLOAT)]",
        "ret",
    ]

    diff = difflib.SequenceMatcher(None, orig, recomp)
    assert is_effective_match(diff.get_opcodes(), orig, recomp) is False


def test_commutative_x87_chain_swap_index_registers_stay():
    """Only the displacements swap; if the index registers differ too,
    the skeletons don't match and this is a real diff."""
    recomp = list(X87_CHAIN_RECOMP)
    recomp[3] = "fld dword ptr [eax*4 + g_skillTableB (DATA)]"

    diff = difflib.SequenceMatcher(None, X87_CHAIN_ORIG, recomp)
    assert is_effective_match(diff.get_opcodes(), X87_CHAIN_ORIG, recomp) is False


def test_commutative_x87_chain_swap_mov_after_fld_invalid():
    """A differing mov after the fld is not operand-chain setup."""
    orig = [
        "mov eax, dword ptr [ecx + 0x94]",
        "fld dword ptr [g_floatA (FLOAT)]",
        "mov edx, dword ptr [ecx + 0x9c]",
        "fadd dword ptr [g_floatB (FLOAT)]",
        "ret",
    ]
    recomp = [
        "mov eax, dword ptr [ecx + 0x9c]",
        "fld dword ptr [g_floatB (FLOAT)]",
        "mov edx, dword ptr [ecx + 0x94]",
        "fadd dword ptr [g_floatA (FLOAT)]",
        "ret",
    ]

    diff = difflib.SequenceMatcher(None, orig, recomp)
    assert is_effective_match(diff.get_opcodes(), orig, recomp) is False


# The following tests cover the dependency-aware relocate_instructions:
# an instruction may only move across instructions it does not depend on.
# They are machine code: register and flag effects come from Capstone.


def _resolve_global(addr: int, exact: bool = False, indirect: bool = False):
    del exact, indirect
    return ResolvedAddress(f"g_{addr:x}", ("entity", addr, 0))


def _relocation_match(orig_hex: str, recomp_hex: str) -> bool:
    """Whether undoing the relocations in two byte sequences makes them
    verify in lockstep; addresses outside the bodies name the same globals."""
    orig, recomp = (
        decode_function(bytes.fromhex(code), start, resolver=_resolve_global)
        for code, start in ((orig_hex, 0x1000), (recomp_hex, 0x2000))
    )
    codes = difflib.SequenceMatcher(
        None,
        [instruction_match_key(row) for row in orig.instructions],
        [instruction_match_key(row) for row in recomp.instructions],
    ).get_opcodes()
    relocated = undo_relocations(codes, orig.instructions, recomp.instructions)
    return relocated is not None and verify_effective_match(
        orig.instructions, relocated
    )


_CALL_GLOBAL = "ff1500005000"  # call dword ptr [0x500000]


def test_relocate_independent_load():
    """Two independent loads scheduled in opposite order."""
    load_eax, load_ecx = "8b45fc", "8b4df8"  # mov eax/ecx, [ebp - 4/8]
    tail = "51" + "50" + _CALL_GLOBAL + "c3"  # push ecx; push eax; call; ret
    assert _relocation_match(load_eax + load_ecx + tail, load_ecx + load_eax + tail)


def test_relocate_rejects_store_across_aliasing_load():
    """A store may not move across a load of the same address."""
    store, load = "8945fc", "8b4dfc"  # mov [ebp - 4], eax; mov ecx, [ebp - 4]
    tail = "51" + "56" + "c3"  # push ecx; push esi; ret
    assert not _relocation_match(store + load + tail, load + store + tail)


def test_relocate_store_across_disjoint_frame_slot():
    """Stores to provably distinct ebp frame slots may reorder."""
    first, second = "8945fc", "894df8"  # mov [ebp - 4], eax; mov [ebp - 8], ecx
    tail = "56" + "57" + "c3"  # push esi; push edi; ret
    assert _relocation_match(first + second + tail, second + first + tail)


def test_relocate_rejects_move_across_call():
    """A call is a barrier: memory and registers may change."""
    load = "a100004000"  # mov eax, [0x400000]
    tail = "50" + "56" + "c3"  # push eax; push esi; ret
    assert not _relocation_match(load + _CALL_GLOBAL + tail, _CALL_GLOBAL + load + tail)


def test_relocate_rejects_flags_consumed_after_move():
    """Both the moved instruction and a crossed instruction write flags,
    and a jump reads them afterward: the move changes the branch."""
    cmp, add = "83f801", "83c102"  # cmp eax, 1; add ecx, 2
    tail = "7401" + "56" + "c3"  # je +1; push esi; ret
    assert not _relocation_match(cmp + add + tail, add + cmp + tail)


def test_relocate_rejects_x87_reorder():
    """x87 instructions depend on the fp stack order: fadd and fmul on
    st(0) do not commute with each other."""
    fadd, fmul = "d80500004000", "d80d04004000"  # fadd/fmul dword ptr [global]
    tail = "5e" + "5f" + "c3"  # pop esi; pop edi; ret
    assert not _relocation_match(fadd + fmul + tail, fmul + fadd + tail)


def test_relocate_across_forward_jcc():
    """A store may cross a forward conditional jump whose target lies
    within the crossed region (both placements execute it on both paths),
    plus an inc that is provably disjoint through the lea-resolved base.
    (Imperialism 0x4dd1b0 shape)"""
    # lea esi, [ebx + 0x1c6]; mov ax, [esi + 0x8a]; cmp ax, -1
    head = "8db3c6010000" + "668b868a000000" + "6683f8ff"
    store = "668906"  # mov word ptr [esi], ax
    # jne +7 (over the inc); inc word ptr [ebx + 0xb0]; mov ecx, ebx
    crossed = "7507" + "66ff83b0000000" + "89d9"
    assert _relocation_match(
        head + store + crossed + "c3", head + crossed + store + "c3"
    )


def test_relocate_rejects_backward_jcc():
    """A backward jump is a loop edge: never cross it."""
    # mov dword ptr [esi], 1; jne -0x10; mov dword ptr [edi], 2; push eax; ret
    orig = "c70601000000" + "75f0" + "c70702000000" + "50" + "c3"
    recomp = "75f0" + "c70702000000" + "c70601000000" + "50" + "c3"
    assert not _relocation_match(orig, recomp)


def test_relocate_rejects_jcc_target_beyond_move():
    """A forward jump whose target lies beyond the moved instruction's new
    position would skip the instruction on the taken path."""
    # mov dword ptr [esi], 1; jne ret; mov dword ptr [edi], 2; push eax; ret
    orig = "c70601000000" + "7507" + "c70702000000" + "50" + "c3"
    recomp = "750d" + "c70702000000" + "c70601000000" + "50" + "c3"
    assert not _relocation_match(orig, recomp)


def test_relocate_store_across_push():
    """A store through an unknown pointer must not cross a push: nothing
    proves the pointer cannot equal the pushed slot's address."""
    store, pushes = "894608", "51" + "52"  # mov [esi + 8], eax; push ecx; push edx
    tail = _CALL_GLOBAL + "c3"
    assert not _relocation_match(store + pushes + tail, pushes + store + tail)


def test_relocate_rejects_esp_read_across_push():
    """An esp-relative read may not cross a push that writes the same slot."""
    load, push = "8b4424fc", "51"  # mov eax, [esp - 4]; push ecx
    tail = "50" + _CALL_GLOBAL + "c3"  # push eax; call; ret
    assert not _relocation_match(load + push + tail, push + load + tail)
