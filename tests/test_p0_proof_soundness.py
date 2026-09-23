"""P0 proof-soundness witnesses: exact topology and linear branch live-out."""

from __future__ import annotations

import struct
from unittest.mock import Mock

import pytest

from reccmp.compare.asm.effective import verify_effective_match
from reccmp.compare.asm.ir import ExtentKind, compute_extent_closed
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.db import EntityDb, ReccmpMatch
from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.report import ReccmpComparedEntity
from reccmp.compare.verification import admit_exact_analysis
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.tools.asmcmp import print_match_verbose
from reccmp.types import EntityType, ImageId

# cmp ecx,0; je +5; push 0 (imm32); add eax,1; ret
_TOPOLOGY_ORIG = bytes.fromhex("83F9007405680000000083C001C3")
# cmp ecx,0; je +5; push 0 (imm8); add eax,1; ret — je lands on ret, not add
_TOPOLOGY_RECOMP = bytes.fromhex("83F90074056A0083C001C3")

# xchg eax,eax; cmp eax,ecx; jl done; xor eax,eax; done: ret
_LIVEOUT_ORIG = bytes.fromhex("87C03BC17C0233C0C3")
# xchg eax,ecx; cmp eax,ecx; jg done; xor eax,eax; done: ret
_LIVEOUT_RECOMP = bytes.fromhex("913BC17F0233C0C3")


def _compare_bytes(orig: bytes, recomp: bytes, db: EntityDb | None = None):
    orig_bin = Mock(spec=[])
    orig_bin.read = Mock(return_value=orig)
    orig_bin.imagebase = 0
    orig_bin.is_relocated_addr = Mock(return_value=False)
    orig_bin.is_debug = Mock(return_value=False)
    recomp_bin = Mock(spec=[])
    recomp_bin.read = Mock(return_value=recomp)
    recomp_bin.imagebase = 0
    recomp_bin.is_relocated_addr = Mock(return_value=False)
    recomp_bin.is_debug = Mock(return_value=False)
    comparator = FunctionComparator(
        db if db is not None else EntityDb(),
        LinesDb(),
        orig_bin,
        recomp_bin,
        Mock(spec=ReccmpReportProtocol),
        CvdumpTypesParser(),
    )
    return comparator.compare_function(
        ReccmpMatch(
            0x200,
            0x400,
            {
                "type": EntityType.FUNCTION,
                "stub": False,
                "name": "witness",
                "symbol": "?Witness",
                "orig_size": len(orig),
                "recomp_size": len(recomp),
            },
        )
    )


def test_admit_exact_requires_topology_not_just_displays():
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=True,
            extent_closed=True,
        )
        is not None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=False,
            keys_equal=True,
            extent_closed=True,
        )
        is None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=True,
            coverage_incomplete=True,
            extent_closed=True,
        )
        is None
    )
    admitted = admit_exact_analysis(
        displays_equal=False,
        topology_equal=True,
        keys_equal=True,
        operands_complete=True,
        control_flow_complete=True,
        extent_closed=True,
    )
    assert admitted is not None
    assert admitted.status == ComparisonStatus.EXACT
    assert (
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            keys_equal=False,
            extent_closed=True,
        )
        is None
    )


def test_admit_exact_requires_keys_equal_argument():
    with pytest.raises(TypeError):
        admit_exact_analysis(
            displays_equal=True,
            topology_equal=True,
            extent_closed=True,
        )


def test_encoding_length_shift_is_not_exact_or_effective():
    """Identical ``je +5`` text can still jump to different instructions."""
    orig_asm = [row.display for row in ParseAsm().parse_asm(_TOPOLOGY_ORIG, 0x200)]
    recomp_asm = [row.display for row in ParseAsm().parse_asm(_TOPOLOGY_RECOMP, 0x400)]
    assert [line.rstrip() for line in orig_asm] == [
        "cmp ecx, 0",
        "je 0x5",
        "push 0",
        "add eax, 1",
        "ret",
    ]
    assert orig_asm == recomp_asm

    result = _compare_bytes(_TOPOLOGY_ORIG, _TOPOLOGY_RECOMP)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False


def test_matching_encodings_with_same_branch_target_remain_exact():
    result = _compare_bytes(_TOPOLOGY_ORIG, _TOPOLOGY_ORIG)
    assert result.analysis.status == ComparisonStatus.EXACT


def test_linear_verifier_does_not_excuse_divergent_live_out_via_predicate():
    """Taken path returns EAX=a vs EAX=c; fallthrough xor must not hide that."""
    orig = [
        "xchg eax, eax",
        "cmp eax, ecx",
        "jl 0x2",
        "xor eax, eax",
        "ret",
    ]
    recomp = [
        "xchg eax, ecx",
        "cmp eax, ecx",
        "jg 0x2",
        "xor eax, eax",
        "ret",
    ]
    assert verify_effective_match(orig, recomp) is False

    result = _compare_bytes(_LIVEOUT_ORIG, _LIVEOUT_RECOMP)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False


def _call_ret(func_addr: int, target: int) -> bytes:
    rel = (target - (func_addr + 5)) & 0xFFFFFFFF
    return b"\xe8" + struct.pack("<I", rel) + b"\xc3"


def _mov_abs_ret(target: int) -> bytes:
    return b"\xa1" + struct.pack("<I", target) + b"\xc3"


def test_unresolved_call_offsets_are_not_exact_or_effective():
    """Same ``<OFFSET1>`` display is not semantic identity across images."""
    orig = _call_ret(0x200, 0x401000)
    recomp = _call_ret(0x400, 0x527000)
    orig_rows = ParseAsm(image_id=ImageId.ORIG).parse_asm(orig, 0x200)
    recomp_rows = ParseAsm(image_id=ImageId.RECOMP).parse_asm(recomp, 0x400)
    assert orig_rows[0].display == recomp_rows[0].display
    assert "<OFFSET" in orig_rows[0].display
    orig_id = orig_rows[0].operands[0][1].identity
    recomp_id = recomp_rows[0].operands[0][1].identity
    assert orig_id != recomp_id

    result = _compare_bytes(orig, recomp)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False
    assert result.display_similarity == 1.0
    assert result.match_ratio < 1.0


def test_unresolved_data_offsets_are_not_exact_or_effective():
    orig = _mov_abs_ret(0x401000)
    recomp = _mov_abs_ret(0x527000)
    orig_rows = ParseAsm(image_id=ImageId.ORIG).parse_asm(orig, 0x200)
    recomp_rows = ParseAsm(image_id=ImageId.RECOMP).parse_asm(recomp, 0x400)
    assert orig_rows[0].display == recomp_rows[0].display
    assert "<OFFSET" in orig_rows[0].display

    result = _compare_bytes(orig, recomp)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False
    assert result.display_similarity == 1.0
    assert result.match_ratio < 1.0


def test_external_jcc_displacement_is_not_proof_identity():
    """Same ``je +0x20`` text can land on unrelated absolute destinations."""
    body = bytes.fromhex("7420C3")  # je +0x20; ret
    orig_rows = ParseAsm(image_id=ImageId.ORIG).parse_asm(body, 0x200)
    recomp_rows = ParseAsm(image_id=ImageId.RECOMP).parse_asm(body, 0x400)
    assert orig_rows[0].display == recomp_rows[0].display
    assert orig_rows[0].display.startswith("je ")
    assert orig_rows[0].control_target != recomp_rows[0].control_target

    result = _compare_bytes(body, body)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False
    assert result.display_similarity == 1.0
    assert result.match_ratio < 1.0


def test_estimated_extent_jmp_past_window_is_open():
    blob = bytes.fromhex("EB05")  # jmp +5, destination is start+7
    excerpt = tuple(ParseAsm().parse_asm(blob, 0x1000))
    assert excerpt[0].branch_target == 0x1007
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=len(blob),
            extent_kind=ExtentKind.ESTIMATED,
        )
        is False
    )
    assert (
        compute_extent_closed(
            excerpt,
            start_addr=0x1000,
            extent=len(blob),
            extent_kind=ExtentKind.KNOWN,
        )
        is True
    )


def test_unmatched_data_display_names_are_not_proof_identity():
    orig = _mov_abs_ret(0x401000)
    recomp = _mov_abs_ret(0x527000)
    db = EntityDb()
    with db.batch() as batch:
        batch.set(
            ImageId.ORIG,
            0x401000,
            type=EntityType.DATA,
            name="g_state",
            size=4,
        )
        batch.set(
            ImageId.RECOMP,
            0x527000,
            type=EntityType.DATA,
            name="g_state",
            size=4,
        )
    orig_rows = ParseAsm(
        image_id=ImageId.ORIG,
        name_lookup=db_lookup(db, ImageId.ORIG),
        addr_test=lambda addr: addr in {0x401000},
    ).parse_asm(orig, 0x200)
    recomp_rows = ParseAsm(
        image_id=ImageId.RECOMP,
        name_lookup=db_lookup(db, ImageId.RECOMP),
        addr_test=lambda addr: addr in {0x527000},
    ).parse_asm(recomp, 0x400)
    assert "g_state" in orig_rows[0].display
    assert orig_rows[0].display == recomp_rows[0].display
    orig_mem = orig_rows[0].operands[1]
    recomp_mem = recomp_rows[0].operands[1]
    orig_ref = orig_mem[5][0][1]
    recomp_ref = recomp_mem[5][0][1]
    assert orig_ref.identity != recomp_ref.identity

    result = _compare_bytes(orig, recomp, db=db)
    assert result.analysis.status != ComparisonStatus.EXACT
    assert result.analysis.is_effective is False


def test_unproven_display_match_is_not_printed_ok(capsys):
    orig = _call_ret(0x200, 0x401000)
    recomp = _call_ret(0x400, 0x527000)
    result = _compare_bytes(orig, recomp)
    assert result.analysis.status not in {
        ComparisonStatus.EXACT,
        ComparisonStatus.EFFECTIVE,
    }
    match = ReccmpComparedEntity(
        orig_addr=0x200,
        name="witness",
        accuracy=result.match_ratio,
        analysis=result.analysis,
        rdiff=result.diff,
        display_similarity=result.display_similarity,
    )
    print_match_verbose(match)
    out = capsys.readouterr().out
    assert "OK!" not in out
    assert "100% match" not in out


def db_lookup(db: EntityDb, image_id: ImageId):
    from reccmp.compare.asm.replacement import create_name_lookup

    return create_name_lookup(
        db,
        image_id,
        lambda _addr: None,
        lambda _key, _off: "",
    )
