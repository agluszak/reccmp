# Structured IR migration

Branch: `refactor/structured-identities`. The starting commit was `6256ed08`;
the work below continues it. The target is that reccmp parses text at input
boundaries and never parses a string it rendered itself.

## Current state

The verifier accepts `DecodedInstruction` rows and `FunctionImage`s only.
`DisasmLiteTuple`, `ResolvedAsm`, `AsmStream`, `Instruction`, `InstructionMeta`,
`with_display`, the text sanitizer, and the production text fallbacks are gone.
The only `parse_instruction()` caller is the test fixture boundary. Direct
control targets, relocations, helper calls, stack offsets, jump tables, and
call facts travel as typed operands or identities. `RawDiffOutput` is used for
reports, not as an analysis input. The `.display` reads in production are for
reporting and diagnostics. Unsupported operands are keyed by instruction
bytes, never Capstone's operand text.

Static locals retain their PDB parent function key through `CvdumpAnalysis`
and `load_cvdump`; matching uses the matched parent address and exact local
name, rejecting ambiguity. The marker's variable semantic ID is retained.
The Clang index emits variable storage kind and base semantic IDs, takes
function return types from `getReturnType()`, and requires structural field,
variable, and base metadata in Python. The old source-index schema is not
accepted. Physical layout conflict checks use structural facts rather than
type spelling. PDB frame offsets are parsed as signed integers at the cvdump
boundary.

The resolver returns display and proof identity together. Call facts use the
identity. Triage uses typed value tags and entity types, and the explanation
path uses symbolic values and identities. `mangled_facts` fills only missing
PDB facts. Importing `reccmp.analysis.crt_startup` before `reccmp.compare`
works; the package's `Compare` export loads lazily.

Exact admission requires equal semantic keys and branch topology, plus either
equal raw bytes or complete operand and control-flow models. An indirect call's
modeled operand completes its control target. A recognized jump table completes
its indirect jump through `FunctionImage.control_flow_complete`. Opaque operands
never gain exact proof from equal display text.

## Validation

- Full local suite: 1,408 passed, 221 skipped, 4 expected failures.
- Updated C++ indexer built with LLVM 21 in the pinned analysis image; source
  collector, source index, and source record integration tests: 30 passed.
- The finished WIZ8/SURRENDER corpus run `refactor-final/run5` had zero
  function status or reason changes against `refactor-3`. WIZ8 took 122.94 s
  and SURRENDER 16.26 s. The previous `refactor-3` WIZ8 run took 126.48 s.
- The corpus's original 150 MB source index uses the old schema. For validation,
  `source-index.json` was regenerated from frozen sources with the updated
  collector. The frozen snapshot lacks its original compile database; a
  rewritten database from the matching Wizardry checkout omitted two frozen
  translation units. The separate validation artifact
  `source-index-merged.json` adds those two units' original marker blocks and
  declarations. This is a corpus-only input artifact, not a compatibility path
  in reccmp.

## Publication

The structured-identities work is on reccmp PR #42 at `71a79dec`. Wizardry
agent6 pins that revision with `[witness]` in PR #693. Keep later function-image
work separate until its corpus evidence and review are complete.

## Function-image and graph continuation (2026-09-27)

The continuation is on local branch
`refactor/function-image-graph`, based on `71a79dec`.
It is published as draft PR #43, stacked on PR #42.

Commits on this branch:

- `33207099`: `FunctionImage.instructions` contains only instructions;
  jump tables and embedded data are separate and the renderer interleaves them.
- `c70c24eb`: build one `FunctionGraph` from decoded code and tables; extent
  closure and the product verifier read this graph.
- `84adc0b2`: accept only current structured report format 2; remove the
  format-1 importer, `udiff`, and legacy effective reconstruction.
- `7ee6d0d3`: matched entities link original and recompiled side records
  without merging their dictionaries.
- `b50d1435`: entity catalog owns canonical original identities and project
  equivalence groups.
- `844e318b`: production assembly entry points use pure `decode_function()`;
  the old mutable `ParseAsm.parse_asm()` API is gone.
- `6faafa55`: external conditional graph edges keep their taken label, and
  embedded-data differences block effective proof without hiding earlier
  mismatch diagnostics.
- `c0e595a4`: CRT startup and body-equivalence analysis use the same function
  decoder instead of accessing `InstructGen` directly or decoding a body again.
- `558be1d6`: compiler fact dataclasses live in `source/records.py` rather
  than sharing a file with index orchestration and layout queries.
- `6c947933`: section discovery is private inside `decode_function()`;
  `instgen.py` and direct section-shaped callers are gone.
- `a40e5789`: delete `AsmRole` and all fake table instructions. Test switches
  pass explicit `JumpTable` records. Alias proofs check table case destinations.
- `bb586099`: make the report and graph tests pass CI Pylint.
- `630941a7`: per-TU Clang observation parsing and JSON fact construction
  moved to `source/observations.py`; `source/index.py` retains namespace
  derivation, marker binding, layout queries, and orchestration.

Full Python suite: 1,387 passed, 221 skipped, 3 expected failures. Mypy and
Pylint pass for the latest changed production modules. The Wizardry WIZ8
corpus retained all 2,684 exact and 284 effective proofs against the pinned
`71a79dec` baseline. The decoder, role-removal, and alias-guard runs had
exactly the same status and reason for all 6,098 entries. Against
the pinned baseline, 14 inconclusive entries became mismatch and 8 mismatches
became inconclusive; those diagnosis changes remain under review. The web UI
lints and builds; Playwright Chromium
could not start because the browser executable is not installed locally.

The positional CFG verifier cannot yet be removed: a product-only probe
retained its successful proofs but lost branch-target diagnoses on 63 WIZ8
functions when the product graph alignment failed. Lockstep, diff-aligned,
and relocation strategies also have successful proofs the product strategy
does not yet cover. Port those proof and diagnosis capabilities before
deleting their implementations.

Still open: replace the separate extent and callee-cleanup walks with
queries over the canonical graph where their evidence permits it; unify
machine and semantic instruction effects; make the entity catalog genuinely
immutable before constructing comparison context; finish separating source
index marker binding, layout queries, and orchestration; finish typed
diagnosis payloads.
Keep cvdump and Unicorn as separate input and
execution engines.

Approved disk cleanup removed about 32 GB of Imperialism Rust incremental
artifacts and 124 MB of Wizardry temporary probe artifacts, without touching
source checkouts or retail evidence.
