"""Clustering comparison verdicts (reccmp.compare.triage)."""

import dataclasses

from reccmp.compare import triage
from reccmp.compare.asm.model import Reference
from reccmp.compare.asm.operand import Imm, Mem, Reg, SignedSymbol, Sym
from reccmp.compare.diagnosis import (
    ComparisonAnalysis,
    ComparisonDifference,
    DifferenceKind,
    DifferenceSide,
    ExecutionEvidence,
    InconclusiveReason,
    Observed,
    RefutationWitness,
    StopDetail,
    StopLocation,
    Strategy,
    StrategyAttempt,
    WitnessKind,
)
from reccmp.compare.report import ReccmpComparedEntity
from reccmp.types import ImageId


def _entity(address: int, analysis: ComparisonAnalysis) -> ReccmpComparedEntity:
    return ReccmpComparedEntity(address, f"f{address:#x}", 0.5, analysis=analysis)


def _mismatch(
    address: int,
    kind: DifferenceKind,
    orig: Observed,
    recomp: Observed,
    execution: ExecutionEvidence | None = None,
    *,
    witness: RefutationWitness | None = None,
) -> ReccmpComparedEntity:
    difference = ComparisonDifference(
        kind,
        DifferenceSide(ImageId.ORIG, address=0x401000, observed=orig),
        DifferenceSide(ImageId.RECOMP, address=0x501000, observed=recomp),
    )
    analysis = ComparisonAnalysis(
        status=ComparisonAnalysis.mismatch(difference).status,
        difference=difference,
        attempts=(StrategyAttempt(Strategy.LOCKSTEP, difference=difference),),
        witness=witness,
        execution=execution,
    )
    return _entity(address, analysis)


AGREED = ExecutionEvidence(16, 16, reached_location=12)
NOT_REACHED = ExecutionEvidence(16, 16, reached_location=0)


def test_buckets():
    difference = ComparisonDifference(
        DifferenceKind.RETURN_VALUE,
        DifferenceSide(ImageId.ORIG),
        DifferenceSide(ImageId.RECOMP),
    )
    assert triage.bucket_of(ComparisonAnalysis.exact()) is None
    assert (
        triage.bucket_of(ComparisonAnalysis.mismatch(difference))
        == triage.Bucket.NOT_EXECUTED
    )
    assert (
        triage.bucket_of(
            dataclasses.replace(
                ComparisonAnalysis.mismatch(difference), execution=NOT_REACHED
            )
        )
        == triage.Bucket.NEVER_REACHED
    )
    assert (
        triage.bucket_of(
            dataclasses.replace(
                ComparisonAnalysis.mismatch(difference), execution=AGREED
            )
        )
        == triage.Bucket.AGREED_THROUGH_DIFFERENCE
    )
    witness = RefutationWitness(0, WitnessKind.RETURN_VALUE, "eax", "1", "2")
    assert (
        triage.bucket_of(
            dataclasses.replace(
                ComparisonAnalysis.mismatch(difference),
                execution=AGREED,
                witness=witness,
            )
        )
        == triage.Bucket.REFUTED
    )
    assert (
        triage.bucket_of(
            dataclasses.replace(
                ComparisonAnalysis.inconclusive(InconclusiveReason.ANALYSIS_LIMIT),
                execution=AGREED,
            )
        )
        == triage.Bucket.AGREED_THROUGH_BLOCKER
    )
    assert (
        triage.bucket_of(
            ComparisonAnalysis.inconclusive(InconclusiveReason.ANALYSIS_LIMIT)
        )
        == triage.Bucket.INCONCLUSIVE
    )


def test_clusters_group_the_same_shape_and_rank_the_useful_bucket_first():
    strict = Observed(value="lt_u:load:initial:sp+4,65")
    loose = Observed(value="le_u:load:initial:sp+4,64")
    import_ref = Reference("arbitrary text", ("import", "a"), "IMPORT")
    thunk_ref = Reference("arbitrary text", ("import", "a"), "IMPORT_THUNK")
    entities = [
        _mismatch(1, DifferenceKind.BRANCH_CONDITION, strict, loose, AGREED),
        _mismatch(2, DifferenceKind.BRANCH_CONDITION, strict, loose, AGREED),
        _mismatch(3, DifferenceKind.BRANCH_CONDITION, strict, loose),
        _mismatch(
            4,
            DifferenceKind.CALL_TARGET,
            Observed(operand=Mem("dword", "", (), 0, (SignedSymbol(1, import_ref),))),
            Observed(operand=Sym(thunk_ref)),
            AGREED,
        ),
        _entity(5, ComparisonAnalysis.exact()),
    ]

    def shape(image: ImageId, _address: int) -> triage.InstructionShape:
        return triage.InstructionShape("jb" if image is ImageId.ORIG else "jbe", (Imm,))

    clusters = triage.triage(entities, shape)

    first, second, third = clusters
    assert first.key.bucket == triage.Bucket.AGREED_THROUGH_DIFFERENCE
    assert first.count == 2
    assert (triage.side_text(first.key.orig), triage.side_text(first.key.recomp)) == (
        "jb imm",
        "jbe imm",
    )
    assert first.key.strategy == Strategy.LOCKSTEP
    assert "callee=IMPORT | indirect" in triage.side_text(second.key.orig)
    assert "callee=IMPORT_THUNK" in triage.side_text(second.key.recomp)
    assert third.key.bucket == triage.Bucket.NOT_EXECUTED
    assert triage.bucket_counts(clusters) == {
        triage.Bucket.AGREED_THROUGH_DIFFERENCE: 3,
        triage.Bucket.NOT_EXECUTED: 1,
    }
    assert "e.g. 0x1 f0x1" in triage.triage_text(clusters)


def test_non_isomorphic_graphs_cluster_by_where_the_product_stopped():
    location = StopLocation(ImageId.ORIG, address=0x401000)

    def blocked(address: int, product: StrategyAttempt) -> ReccmpComparedEntity:
        analysis = dataclasses.replace(
            ComparisonAnalysis.inconclusive(
                InconclusiveReason.NON_ISOMORPHIC_CFG, location
            ),
            attempts=(
                StrategyAttempt(
                    Strategy.ISOMORPHIC_CFG,
                    blocker=InconclusiveReason.NON_ISOMORPHIC_CFG,
                    location=location,
                ),
                product,
            ),
        )
        return _entity(address, analysis)

    alignment = StrategyAttempt(
        Strategy.UNANCHORED_PRODUCT,
        blocker=InconclusiveReason.ALIGNMENT_FAILURE,
        location=StopLocation(ImageId.ORIG, detail=StopDetail.BLOCK_ALIGNMENT),
    )
    difference = ComparisonDifference(
        DifferenceKind.MEMORY_ADDRESS,
        DifferenceSide(ImageId.ORIG),
        DifferenceSide(ImageId.RECOMP),
    )
    clusters = triage.triage(
        [
            blocked(1, alignment),
            blocked(2, alignment),
            blocked(
                3,
                StrategyAttempt(Strategy.UNANCHORED_PRODUCT, difference=difference),
            ),
        ]
    )
    assert [
        (triage.details_text(cluster.key), cluster.count) for cluster in clusters
    ] == [
        ("product: alignment_failure at block_alignment", 2),
        ("product: memory_address", 1),
    ]


def test_instruction_shape():
    assert triage.instruction_shape(
        bytes.fromhex("83f841"), 0x1000
    ) == triage.InstructionShape("cmp", (Reg, Imm))
    assert triage.instruction_shape(
        bytes.fromhex("894104"), 0x1000
    ) == triage.InstructionShape("mov", (Mem, Reg))
