#!/usr/bin/env python3
"""Export a fixed corpus of reccmp function pairs for the ghidriff POC.

Reads a reccmp report (`reccmp-reccmp --target WIZ8 --json report.json
--json-diet`) and emits `selected-pairs.json`: orig/recomp addresses plus
reccmp's own verdict per pair, so the ghidriff result can later be scored
against what reccmp already knows.

The corpus is deliberately adversarial:

  exact        - byte-identical pairs; ghidriff should report nothing
  effective    - reccmp "effective": known compiler noise (register alloc,
                 scheduling, branch inversion); the false-positive test
  known-bad    - recoveries with a confirmed or strongly suspected logic
                 divergence (see `note`); ghidriff should make it visible
  ugly         - switches, x87, big CFGs; decompiler-instability probe

Usage:
    python export_pairs.py REPORT_JSON [OUT_JSON]
"""

import json
import sys
from pathlib import Path

# name substring -> (category, note). Resolved against the report by name.
SELECTED = {
    # --- exact (10): sanity floor -------------------------------------------
    "FileExists": ("exact", ""),
    "DrawGenericButton": ("exact", ""),
    "W8Octree::GetPathSurfaceNormal": ("exact", ""),
    "WorldGetPropAt": ("exact", ""),
    "W8PathingService::PlanMovementToPosition": ("exact", "sibling control for PlanMovement"),
    "MonsterFleeAction": ("exact", ""),
    "TickSpellEffects": ("exact", ""),
    "DrawDialog": ("exact", ""),
    "GetCharActionRange": ("exact", ""),
    "ReleaseReadMeshScratch": ("exact", ""),
    # --- effective / compiler noise (18): must stay CLEAN --------------------
    "W8Monster::AdvanceAnimationFrame": ("effective", ""),
    "W8Monster::HandleAnimationFrame": ("effective", ""),
    "MonGen::Load": ("effective", ""),
    "InsertionSort": ("effective", ""),
    "QuickSort": ("effective", ""),
    "ReadParticleSystemFile": ("effective", ""),
    "ProjectPointThroughCamera": ("effective", ""),
    "ChooseDifferentMonsterDirection": ("effective", ""),
    "StartCombat": ("effective", ""),
    "EndMonsterTurn": ("effective", ""),
    "GetMonsterCombatMoveRange": ("effective", ""),
    "CalcCharacterLevelBand": ("effective", ""),
    "ApplyModifierBlock": ("effective", ""),
    "RefreshMonsterGroupAndAllies": ("effective", ""),
    "CreateItemIntoHandOrPool": ("effective", ""),
    "IsSpellBlockedForMonster": ("effective", ""),
    "GetSpellFailureChance": ("effective", ""),
    "W8PathingService::WritePathNodes": ("effective", ""),
    # --- known-bad (10): real logic divergence, must be ACTIONABLE -----------
    "W8Monster::Query": ("known-bad", "missing result=0 init (fixed in 498cf7dfa)"),
    "W8PathingService::PlanMovement": ("known-bad", "pathing loop bound/parent mixup (498cf7dfa)"),
    "StepMonsterCombatAction": ("known-bad", "ternary <= vs > inversion (498cf7dfa)"),
    "FormatMonsterHealth": ("known-bad", 'L"" vs L"?" HP display (498cf7dfa)'),
    "RepickActionTarget": ("known-bad", "arg vs resolved targeting context (498cf7dfa)"),
    "RefreshAllPartyTargets": ("known-bad", "inverted weapon-swap condition (498cf7dfa)"),
    "W8Monster::IsCycleInterruptable": ("known-bad", "recomp resolves string ref to empty literal"),
    "ElevationToTargetCPP": ("known-bad", "immediate 16 vs 24 offset suspect"),
    "MonsterSetStateA0": ("known-bad", "orig loads arg; recomp booleanizes (setcc)"),
    "WorldCursorNodeApplyItemEffect004D9560": ("known-bad", "orig loads global; recomp uses literal 27"),
    # --- ugly (12): x87 / switches / big CFGs --------------------------------
    "W8Monster::Update": ("ugly", "large switch"),
    "W8Monster::ProcessScript": ("ugly", "large switch"),
    "ExecuteMonsterAction": ("ugly", "large switch"),
    "ExecuteCharacterAction": ("ugly", "large switch"),
    "Trigger::Run": ("ugly", "switch"),
    "CastSpellFromSource": ("ugly", "switch"),
    "EvaluateFact": ("ugly", "switch"),
    "GetItemUseDifficulty": ("ugly", "switch"),
    "srMatrix3T<float>::MultiplyBy": ("ugly", "x87"),
    "W8GameData::ProcessCrossedSurface": ("ugly", "x87 compare"),
    "RotateNodeInDegrees": ("ugly", "x87"),
    "W8Octree::BuildCellWalk(class srVector3T<float> const *, class srVector3T<float> const *, struct W8OctreeWalk *)": (
        "ugly",
        "float-heavy traversal, big CFG",
    ),
}


def main() -> int:
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "wiz8-report.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "selected-pairs.json")

    report = json.loads(report_path.read_text())
    entries = report["data"]
    by_name = {}
    for e in entries:
        name = e.get("name")
        if name:
            by_name.setdefault(name, []).append(e)

    pairs = []
    missing = []
    for want, (category, note) in SELECTED.items():
        hits = by_name.get(want) or []
        if len(hits) != 1:
            missing.append((want, len(hits)))
            continue
        e = hits[0]
        name = want
        if e.get("recomp") is None:
            missing.append((want, [f"{name}: no recomp addr"]))
            continue
        pairs.append(
            {
                "orig": int(e["address"], 16),
                "recomp": int(e["recomp"], 16),
                "name": name,
                "category": category,
                "reccmp_status": e["comparison"]["status"],
                "note": note,
            }
        )

    for want, names in missing:
        print(f"SKIP {want}: matches={names}", file=sys.stderr)

    pairs.sort(key=lambda p: (p["category"], p["orig"]))
    out_path.write_text(json.dumps(pairs, indent=1) + "\n")
    print(f"wrote {len(pairs)} pairs to {out_path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
