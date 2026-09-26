"""Clustering comparison verdicts (reccmp.compare.triage)."""

from reccmp.compare import triage


def _mismatch(address, kind, orig_facts, recomp_facts, execution=None, *, witness=None):
    difference = {
        "kind": kind,
        "orig": {"address": 0x401000, "facts": orig_facts, "image": "orig"},
        "recomp": {"address": 0x501000, "facts": recomp_facts, "image": "recomp"},
    }
    comparison = {
        "status": "mismatch",
        "difference": difference,
        "attempts": [{"strategy": "lockstep", "difference": difference}],
    }
    if execution is not None:
        comparison["execution"] = execution
    if witness is not None:
        comparison["witness"] = witness
    return {"address": address, "name": f"f{address}", "comparison": comparison}


AGREED = {"runs": 16, "agreeing": 16, "reached_location": 12}
NOT_REACHED = {"runs": 16, "agreeing": 16, "reached_location": 0}


def test_buckets():
    assert triage.bucket_of({"status": "exact"}) is None
    assert triage.bucket_of({"status": "mismatch"}) == triage.NOT_EXECUTED
    assert triage.bucket_of({"status": "mismatch", "execution": NOT_REACHED}) == (
        triage.NEVER_REACHED
    )
    assert triage.bucket_of({"status": "mismatch", "execution": AGREED}) == (
        triage.AGREED_THROUGH_DIFFERENCE
    )
    refuted = {"status": "mismatch", "execution": AGREED, "witness": {"kind": "x"}}
    assert triage.bucket_of(refuted) == triage.REFUTED
    assert triage.bucket_of({"status": "inconclusive", "execution": AGREED}) == (
        triage.AGREED_THROUGH_BLOCKER
    )
    assert triage.bucket_of({"status": "inconclusive"}) == triage.INCONCLUSIVE


def test_clusters_group_the_same_shape_and_rank_the_useful_bucket_first():
    strict = {"predicate": "lt_u:load:initial:sp+4,65"}
    loose = {"predicate": "le_u:load:initial:sp+4,64"}
    entities = [
        _mismatch("0x1", "branch_condition", strict, loose, AGREED),
        _mismatch("0x2", "branch_condition", strict, loose, AGREED),
        _mismatch("0x3", "branch_condition", strict, loose),
        _mismatch(
            "0x4",
            "call_target",
            {"target_name": "dword ptr [X (IMPORT) [CALLEE orig:5eb7c0]]"},
            {"target_name": "X (IMPORT_THUNK)"},
            AGREED,
        ),
        {"address": "0x5", "name": "g", "comparison": {"status": "exact"}},
    ]

    def shape(image: str, _address: int) -> str:
        return "jb imm" if image == "orig" else "jbe imm"

    clusters = triage.triage(entities, shape)

    first, second, third = clusters
    assert first.key.bucket == triage.AGREED_THROUGH_DIFFERENCE and first.count == 2
    assert (first.key.orig, first.key.recomp) == (
        "jb imm | predicate=lt_u",
        "jbe imm | predicate=le_u",
    )
    assert first.key.strategy == "lockstep"
    assert "callee=IMPORT indirect" in second.key.orig
    assert "callee=IMPORT_THUNK" in second.key.recomp
    assert third.key.bucket == triage.NOT_EXECUTED
    assert triage.bucket_counts(clusters) == {
        triage.AGREED_THROUGH_DIFFERENCE: 3,
        triage.NOT_EXECUTED: 1,
    }
    assert "e.g. 0x1 f0x1" in triage.triage_text(clusters)


def test_instruction_shape():
    assert triage.instruction_shape(bytes.fromhex("83f841"), 0x1000) == "cmp reg, imm"
    assert triage.instruction_shape(bytes.fromhex("894104"), 0x1000) == "mov mem, reg"
