"""Command-line selection for reccmp-reccmp reports."""

from unittest.mock import patch

from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceKind,
    DifferenceSide,
    EffectiveReason,
    InconclusiveReason,
    Observed,
    StopDetail,
    StopLocation,
)
from reccmp.types import ImageId
from reccmp.tools.asmcmp import (
    inconclusive_diagnostic_text,
    parse_args,
    triage_status_note,
)


def test_parse_repeated_report_address_filters():
    argv = [
        "reccmp-reccmp",
        "--target",
        "TEST",
        "--orig-address",
        "0x401000",
        "--orig-address",
        "0x402000",
        "--recomp-address",
        "0x501000",
        "--no-cache",
    ]

    with patch("sys.argv", argv):
        args = parse_args()

    assert args.orig_address == [0x401000, 0x402000]
    assert args.recomp_address == [0x501000]
    assert args.no_cache


def test_triage_note_inconclusive_disclaims_source_defect():
    note = triage_status_note(
        ComparisonAnalysis.inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
    )
    assert note is not None
    assert note.startswith("inconclusive:")
    assert "could not prove either outcome" in note
    assert "NOT evidence of a source defect" in note
    assert "verifier/metadata/alignment" in note


def test_triage_note_effective_says_no_action_needed():
    note = triage_status_note(
        ComparisonAnalysis.effective({EffectiveReason.REGISTER_ALLOCATION})
    )
    assert note == "effective: proved semantically harmless — no action needed"


def test_triage_note_exact_and_mismatch_have_no_gloss():
    assert triage_status_note(ComparisonAnalysis.exact()) is None

    difference = ComparisonDifference(
        DifferenceKind.MEMORY_ADDRESS,
        DifferenceSide(ImageId.ORIG, 0, 0x401000, Observed()),
        DifferenceSide(ImageId.RECOMP, 0, 0x501000, Observed()),
    )
    assert triage_status_note(ComparisonAnalysis.mismatch(difference)) is None


def test_inconclusive_diagnostic_renders_reason_location_and_detail():
    analysis = ComparisonAnalysis.inconclusive(
        InconclusiveReason.NON_ISOMORPHIC_CFG,
        StopLocation(
            ImageId.ORIG,
            4,
            0x401020,
            detail=StopDetail.EDGE_ROLES,
        ),
    )
    text = inconclusive_diagnostic_text(analysis)
    assert text is not None
    assert "non_isomorphic_cfg" in text
    assert "0x401020" in text
    assert "stage: edge_roles" in text
