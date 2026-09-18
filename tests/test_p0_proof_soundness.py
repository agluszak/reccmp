"""P0 proof-soundness witnesses: exact topology and linear branch live-out."""

from __future__ import annotations

from unittest.mock import Mock

from reccmp.compare.asm.effective import verify_effective_match
from reccmp.compare.asm.parse import ParseAsm
from reccmp.compare.db import EntityDb, ReccmpMatch
from reccmp.compare.diagnosis import ComparisonStatus
from reccmp.compare.verification import admit_exact_analysis
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.types import EntityType

# cmp ecx,0; je +5; push 0 (imm32); add eax,1; ret
_TOPOLOGY_ORIG = bytes.fromhex("83F9007405680000000083C001C3")
# cmp ecx,0; je +5; push 0 (imm8); add eax,1; ret — je lands on ret, not add
_TOPOLOGY_RECOMP = bytes.fromhex("83F90074056A0083C001C3")

# xchg eax,eax; cmp eax,ecx; jl done; xor eax,eax; done: ret
_LIVEOUT_ORIG = bytes.fromhex("87C03BC17C0233C0C3")
# xchg eax,ecx; cmp eax,ecx; jg done; xor eax,eax; done: ret
_LIVEOUT_RECOMP = bytes.fromhex("913BC17F0233C0C3")


def _compare_bytes(orig: bytes, recomp: bytes):
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
        EntityDb(),
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
    assert admit_exact_analysis(displays_equal=True, topology_equal=True) is not None
    assert (
        admit_exact_analysis(displays_equal=True, topology_equal=False) is None
    )
    assert (
        admit_exact_analysis(
            displays_equal=True, topology_equal=True, coverage_incomplete=True
        )
        is None
    )
    admitted = admit_exact_analysis(
        displays_equal=False,
        topology_equal=True,
        keys_equal=True,
        operands_complete=True,
        control_flow_complete=True,
    )
    assert admitted is not None
    assert admitted.status == ComparisonStatus.EXACT


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
