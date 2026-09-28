# Ghidriff evaluation (proof of concept)

The evaluation that led to comparing code with Ghidriff (`reccmp-reccmp`,
see `code-comparison.md`). It ran a disposable engine outside reccmp on 50
Wizardry 8 pairs; the production adapter replaced its content-derived data
names with catalog correspondence and its string scanner with Ghidra's
`StringDataInstance`, and reports every requested pair.


Run: `Wiz8-retail.exe` vs `Wiz8-recomp.exe` (repin pre-fix build), 50 supplied
reccmp pairs, 48 matched, 2 skipped (no Ghidra function at one side:

- `WorldCursorNodeApplyItemEffect004D9560` (orig 0x4d9560)
- `W8Monster::Update` (orig 0x4c2100)

## Critical upstream bug found (worked around)

Ghidriff's decompiler pool is keyed by `prog.name`:

```python
self.decompilers[prog.name] = decompilers  # setup_decompliers
... = self.decompilers[prog.name]          # decompile_func
```

When both inputs are named `Wiz8.exe`, the second pool overwrites the first and
**every "old-side" decompile is actually performed against the NEW program's
bytes at the old VA**. Verified byte-for-byte: the bogus "retail" decompile of
`FormatMonsterHealth` matched recomp bytes at the retail VA (`call esi` →
`(*unaff_ESI)(0x10003)`).

Workaround used here: copy binaries to distinct names (`bins/Wiz8-retail.exe`,
`bins/Wiz8-recomp.exe`). A real fix is to key the pool by program or index, not
name — worth an upstream patch.

## Classification (48 matched pairs)

CLEAN = compiler noise disappears or is trivially identifiable.
ACTIONABLE = diff points at the real logic difference.
NOISY = decompiler churn dominates; the real difference is buried.
MISLEADING = diff suggests equivalence/divergence opposite to reality.

### Exact byte matches (reccmp verdict: exact)

| pair | ratio | +/- | verdict | notes |
|---|---|---|---|---|
| FileExists | .95 | 4 | CLEAN | |
| DrawGenericButton | .85 | 50 | CLEAN | all symbol-address churn |
| GetPathSurfaceNormal | .91 | 4 | CLEAN | |
| PlanMovementToPosition | .90 | 4 | CLEAN | |
| WorldGetPropAt | .80 | 4 | CLEAN | |
| ReleaseReadMeshScratch | 1.00 | 2 | CLEAN | |
| MonsterFleeAction | .74 | 20 | CLEAN | all symbol-address churn |
| TickSpellEffects | .89 | 12 | CLEAN | |
| GetCharActionRange | .89 | 8 | CLEAN | |
| DrawDialog | .88 | 4 | CLEAN | |

10/10 CLEAN. Byte-identical code still diffs textually (FUN_/DAT_/s_ labels,
occasional signature/type drift) but the churn is uniform and recognizable.

### Effective / compiler-noise

| pair | ratio | +/- | verdict | notes |
|---|---|---|---|---|
| QuickSort | .91 | 14 | CLEAN | var-role swap only |
| WritePathNodes | .66 | 42 | CLEAN | addr churn |
| MonGen::Load | .56 | 37 | CLEAN | `joined_r0x` label move, bool hoist |
| ProjectPointThroughCamera | 1.00 | 2 | CLEAN | |
| ChooseDifferentMonsterDirection | .97 | 4 | CLEAN | |
| AdvanceAnimationFrame | .83 | 10 | CLEAN | |
| HandleAnimationFrame | .82 | 12 | CLEAN | |
| InsertionSort | .84 | 12 | CLEAN | |
| ReadParticleSystemFile | .75 | 30 | CLEAN | *500.0 statement block moved |
| GetMonsterCombatMoveRange | .83 | 14 | CLEAN | |
| StartCombat | .67 | 124 | CLEAN | large body; noise is all addr churn |
| EndMonsterTurn | .82 | 12 | CLEAN | |
| CalcCharacterLevelBand | 1.00 | 2 | CLEAN | |
| IsSpellBlockedForMonster | .94 | 4 | CLEAN | |
| GetSpellFailureChance | 1.00 | 2 | CLEAN | |
| ApplyModifierBlock | 1.00 | 2 | CLEAN | |
| RefreshMonsterGroupAndAllies | .71 | 15 | CLEAN | |
| CreateItemIntoHandOrPool | .75 | 10 | CLEAN | |

18/18 CLEAN. This is the headline result: the false-positive class that costs
reccmp the most verifier effort is essentially absent at decompiled level.

### Known-bad recoveries (pre-fix repin build)

| pair | ratio | +/- | verdict | notes |
|---|---|---|---|---|
| W8PathingService::PlanMovement | .06 | 1696 | NOISY | 5 KB body; bug buried |
| ElevationToTargetCPP | .27 | 36 | CLEAN | mislabeled: only stack-layout noise (`local_c/8/4` vs `local_18/14/10`); `_DAT_1` = 1.5707963 |
| W8Monster::IsCycleInterruptable | .87 | 14 | CLEAN | mislabeled: pure `PTR_s_BIRTH_` renames |
| W8Monster::Query | .88 | 17 | NOISY | CFG restructure; `result=0` init diff not plainly pointed at |
| MonsterSetStateA0 | .86 | 4 | ACTIONABLE | `*(undefined1*)x = param_2` vs `*(bool*)x = param_2 != '\0'` — the entire diff is the bug |
| FormatMonsterHealth | .81 | 38 | MISLEADING | logic looks identical; `L"?"` vs `L""` hidden behind identical `&DAT_2` labels |
| StepMonsterCombatAction | .65 | 66 | ACTIONABLE | `*(float*)(x+0x130) <= _DAT_1` vs `0.0 < *(float*)(x+0x130)` — inverted comparison plainly visible |
| RepickActionTarget | .37 | 224 | NOISY | unreachable-block warnings + stack restructure bury the arg-passing bug |
| RefreshAllPartyTargets | .56 | 72 | ACTIONABLE | `cVar1 == '\0'` vs `cVar1 != '\0'` polarity flip visible amid rename churn |

ACTIONABLE: 3/7 real bugs. NOISY: 2/7 (big CFG-heavy functions). CLEAN: 2
mislabeled entries (not actually buggy). MISLEADING: 1 (FormatMonsterHealth —
the only case where Ghidra output would actively deceive a reviewer; string
contents live behind equal `DAT_N` labels, though `added/deleted_strings` in the
JSON does flag `u"?"`).

### Ugly functions

| pair | ratio | +/- | verdict |
|---|---|---|---|
| ProcessCrossedSurface | .55 | 62 | NOISY |
| srMatrix3T<float>::MultiplyBy | 1.00 | 2 | CLEAN |
| RotateNodeInDegrees | .73 | 8 | CLEAN |
| BuildCellWalk | .39 | 95 | NOISY |
| Trigger::Run | .18 | 3252 | NOISY |
| W8Monster::ProcessScript | .11 | 1816 | NOISY |
| ExecuteCharacterAction | .27 | 346 | NOISY |
| ExecuteMonsterAction | .58 | 238 | NOISY |
| CastSpellFromSource | .20 | 1826 | NOISY |
| EvaluateFact | .04 | 701 | NOISY |
| GetItemUseDifficulty | .91 | 8 | CLEAN |

NOISY: 7/11. Expected — multi-KB functions with switches/x87 produce decompiler
instability that diffing cannot rescue.

## Totals

- CLEAN: 30 (10 exact + 18 effective + 2 ugly + 2 mislabeled "bad")
- ACTIONABLE: 3
- NOISY: 14
- MISLEADING: 1
- Skipped by Ghidra (no function): 2

## Interpretation vs success criteria

- "large majority of compiler-noise cases CLEAN" — **met** (28/28 intended).
- "large majority of known bad recoveries ACTIONABLE" — **not met** (3/7;
  big CFG-heavy functions still bury the signal, and FormatMonsterHealth is
  actively hidden).
- "very few MISLEADING" — met in count (1), but that 1 is a real correctness bug
  invisible in the function diff.

Noise source ranking (what a POC-2 would attack):
1. Global/function symbol renames — the dominant line-churn everywhere. Feeding
   reccmp's known names (retail symbols + recomp PDB) into Ghidra before
   decompiling would collapse most of this.
2. `DAT_N` global labels hide string/constant contents — string-valued diffs
   need inlining or a strings sub-report.
3. Big functions (>2 KB) — decompiler instability; no cheap fix.
4. Calling-convention/signature inference differences between programs —
   occurred in the first (collided) run; not observed after the rename fix.

## A/B: unified markdown vs `--sxs` side-by-side HTML

`out-sxs/sxs_html/` holds one `.md`-wrapped HTML table per code-diffing pair
plus `combined.html`. Findings:

- SxS rows carry **intra-line character highlighting** (`diff_sub`/`diff_chg`
  spans). Symbol renames like `FUN_004e5720` → `FUN_0049e820` highlight only
  the changed digits, so address churn is visually cheaper than in the unified
  `-`/`+` block form.
- SxS does **not** normalize `DAT_` names the way the unified path did — e.g.
  the StepMonsterCombatAction inversion renders as
  `*(float *)(iVar4 + 0x130) <= _DAT_005ebb34` vs `0.0 < *(float *)(iVar4 + 0x130)`
  with the changed spans highlighted — more informative than the unified view.
- Verdict: prefer SxS HTML for manual triage; unified md is fine for grep.

## Verdict so far

Worth a POC-2: inject reccmp/recovered names + recomp PDB into Ghidra before
decompile, re-run same corpus. That tests whether removing noise class (1)
promotes the NOISY known-bad cases to ACTIONABLE. It cannot fix class (2)
without a small output tweak.

---

# POC-2 — canonical reccmp names + string contents

Changes since POC-1:

1. **Upstream fix landed in fork** (`agluszak/ghidriff` branch
   `fix-same-name-decompile-corruption`, two commits `aeb7b03` + `c4aba02`):
   decompiler pool / `esym_memo` / `program_options` now keyed by program
   identity, not `prog.name`; regression test with same-named binaries added;
   `FUN_`/`OFF_` normalization fixed (trailing-underscore patterns never
   matched). POC-2 was then re-run on the **same-named** `Wiz8.exe` inputs —
   the fix holds in the real pipeline.
2. `reccmp_pairs.py --names <reccmp-report.json>` injects canonical names into
   both analyzed programs before diffing: functions become `rc_<reccmp name>`
   (same entity, same name both sides); data referenced by corpus functions
   gets content-derived names (`stra_/strw_<content>`, `rcp_<pointee>`,
   `rcd_<len>b_<hash>`). No PDB types, no prototypes, no calling conventions.
   Data symbols renamed: 154 (retail) / 144 (recomp).

## Results (same 48 pairs)

| metric | POC-1 | POC-2 |
|---|---|---|
| total +/- diff lines | 10981 | 10393 |
| pairs with empty diff | ~1 | **19** |
| pairs ≤10 lines | ~20 | **33** |
| name/fullname/sig diff-types | 48 each | **0** |
| calling/called diff-types | 46 / 36 | 6 / 6 |

- **Exact matches: 10/10 → 0–4 lines.** Byte-identical functions are now
  essentially empty diffs (FileExists, DrawDialog, WorldGetPropAt, … = 0).
- **Effective/compiler-noise: all ≤11 lines**, 12 of 18 at 0 — diff text is
  now almost purely semantic residue (e.g. QuickSort 14→10, StartCombat
  124→100).
- **Actionable bugs stayed actionable**: `MonsterSetStateA0` still the same
  clean 4-line diff; `StepMonsterCombatAction` 66→38, inversion still plain;
  `RefreshAllPartyTargets` 72→53, polarity flip still visible.
- **FormatMonsterHealth is no longer misleading**: the `L""`/`L"?"` operands
  now differ (`rcd_1b_5bab61eb` vs `DAT_2`) instead of identical `&DAT_2` on
  both sides, and surrounding string reads show real content names
  (`stra_pmonsterinfo_null_hc5339c27`, `strw_d_d_h65e19fe7`). The recomp-side
  `L'?'` item stayed `DAT_2` (lone wide char: content-detection path needs
  follow-up) — asymmetric naming still flags the divergence.
- **>2 KB functions barely moved** — Trigger::Run 3252→3138,
  ProcessScript 1816→1850, PlanMovement 1696→1660, CastSpellFromSource
  1826→1841. Confirmed: their noise is decompiler/CFG structural instability,
  not symbols. Recorded as a limitation; not attacked here.

## Against the POC-2 success bar

- 28 exact/effective stay CLEAN with dramatically less churn — **met**
  (median effective pair went from ~12 lines to ~2; 19 pairs total now empty).
- Byte-exact nearly empty — **met**.
- 3 ACTIONABLE stay ACTIONABLE — **met**.
- FormatMonsterHealth stops being misleading — **met** (the differing operand
  is now literally on screen; prettier `strw_qmark`/`strw_empty` names are a
  nice-to-have, not required for visibility).
- ≥1–2 of Query/RepickActionTarget/PlanMovement NOISY→ACTIONABLE — **not
  met** (17→15, 224→224, 1696→1660). Canonical naming does not rescue big
  functions; their instability is structural.

## Conclusion

The dominant POC-1 noise source *was* the integration problem (cross-binary
symbol identity), not a decompiler limitation: naming alone removed ~half of
all small-pair churn and emptied 19 diffs entirely. What remains is bimodal:
small/medium functions are effectively solved by decompile+canonical-names;
multi-KB functions need a different strategy (they fail for CFG reasons, and
no naming scheme fixes that). That is the natural split for deciding which
reccmp comparison machinery decompilation can replace.
